#!/usr/bin/env python3
"""
Interactive CLI for spread predictions.

Saves as: models/interactive_predict_spread.py

Run:
  python models/interactive_predict_spread.py --games data/games_summary_merged.csv --model spread_model.pkl

Then follow the prompts. Commands:
  - Type a game as: HOME AWAY DATE SPREAD
    e.g. LAL BOS 2024-10-25 -3.5
  - Or type 'batch path/to/upcoming.csv' to predict all rows in a CSV (expects columns home_team,away_team,date,spread)
  - Type 'help' to show instructions, 'quit' or 'exit' to leave

This script mirrors the feature logic used by the training pipeline (5- and 10-game rolling stats)
and estimates probability using the residual std from an eval CSV if present.
"""

import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from math import erf, sqrt

# ---------------------- utility functions ----------------------

def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def estimate_residual_std(eval_report_path: Path):
    if eval_report_path.exists():
        try:
            er = pd.read_csv(eval_report_path)
            if ("margin" in er.columns) and ("pred_margin" in er.columns):
                resid = er["margin"] - er["pred_margin"]
                return float(resid.std(ddof=0))
        except Exception:
            pass
    return None


# Reusable rolling feature builder (same logic as predict_spread.py)

def compute_team_rolls(games_df, date_col="date", windows=(5,10)):
    rows = []
    for idx, r in games_df.iterrows():
        rows.append({
            "game_id": idx,
            "date": pd.to_datetime(r[date_col]),
            "team": r["home_team"],
            "is_home": 1,
            "opp": r["away_team"],
            "team_pts": r["home_pts"],
            "opp_pts": r["away_pts"],
        })
        rows.append({
            "game_id": idx,
            "date": pd.to_datetime(r[date_col]),
            "team": r["away_team"],
            "is_home": 0,
            "opp": r["home_team"],
            "team_pts": r["away_pts"],
            "opp_pts": r["home_pts"],
        })
    tg = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    tg["margin"] = tg["team_pts"] - tg["opp_pts"]
    tg["win"] = (tg["margin"] > 0).astype(int)
    for w in windows:
        tg[f"margin_roll_{w}"] = tg.groupby("team")["margin"].shift(1).rolling(window=w, min_periods=1).mean()
        tg[f"winrate_roll_{w}"] = tg.groupby("team")["win"].shift(1).rolling(window=w, min_periods=1).mean()
        tg[f"pts_roll_{w}"] = tg.groupby("team")["team_pts"].shift(1).rolling(window=w, min_periods=1).mean()
        tg[f"opp_pts_roll_{w}"] = tg.groupby("team")["opp_pts"].shift(1).rolling(window=w, min_periods=1).mean()
    return tg


def get_last_team_feats(team_games, team, as_of_date, windows=(5,10)):
    df = team_games[(team_games["team"] == team) & (team_games["date"] < as_of_date)].sort_values("date")
    if df.empty:
        return None
    last = df.iloc[-1]
    feats = {}
    for w in windows:
        feats[f"margin_roll_{w}"] = last.get(f"margin_roll_{w}", np.nan)
        feats[f"winrate_roll_{w}"] = last.get(f"winrate_roll_{w}", np.nan)
        feats[f"pts_roll_{w}"] = last.get(f"pts_roll_{w}", np.nan)
        feats[f"opp_pts_roll_{w}"] = last.get(f"opp_pts_roll_{w}", np.nan)
    feats["last_game_date"] = last["date"]
    return feats


def build_feature_vector(home_feats, away_feats, as_of_date, windows=(5,10)):
    vec = {}
    for w in windows:
        vec[f"home_minus_away_margin_roll_{w}"] = home_feats[f"margin_roll_{w}"] - away_feats[f"margin_roll_{w}"]
        vec[f"home_minus_away_winrate_{w}"] = home_feats[f"winrate_roll_{w}"] - away_feats[f"winrate_roll_{w}"]
        vec[f"home_minus_away_pts_{w}"] = home_feats[f"pts_roll_{w}"] - away_feats[f"pts_roll_{w}"]
    # rest days
    home_last = home_feats.get("last_game_date", None)
    away_last = away_feats.get("last_game_date", None)
    if pd.isna(home_last) or home_last is None:
        home_rest = np.nan
    else:
        home_rest = (as_of_date - pd.to_datetime(home_last)).days
    if pd.isna(away_last) or away_last is None:
        away_rest = np.nan
    else:
        away_rest = (as_of_date - pd.to_datetime(away_last)).days
    vec["home_rest"] = home_rest
    vec["away_rest"] = away_rest
    vec["rest_diff"] = home_rest - away_rest if (home_rest is not None and away_rest is not None) else np.nan
    return vec


# ---------------------- CLI logic ----------------------

def prompt_help():
    print("Commands:")
    print("  HOME AWAY DATE SPREAD   e.g. LAL BOS 2024-10-25 -3.5")
    print("  batch PATH_TO_CSV       CSV must contain columns: home_team,away_team,date,spread")
    print("  help                   show this message")
    print("  quit / exit            exit")
    print("")


def predict_single(model, team_games, home, away, date_str, spread, eval_std, windows=(5,10)):
    as_of_date = pd.to_datetime(date_str)
    home_feats = get_last_team_feats(team_games, home, as_of_date, windows=windows)
    away_feats = get_last_team_feats(team_games, away, as_of_date, windows=windows)
    if home_feats is None:
        print(f"No historical data for home team '{home}' before {as_of_date.date()}')")
        return
    if away_feats is None:
        print(f"No historical data for away team '{away}' before {as_of_date.date()}')")
        return
    feat_vec = build_feature_vector(home_feats, away_feats, as_of_date, windows=windows)
    X = pd.DataFrame([feat_vec])
    # align features if model expects more
    try:
        pred = float(model.predict(X)[0])
    except Exception:
        try:
            fns = model.feature_name()
        except Exception:
            fns = None
        if fns:
            for fn in fns:
                if fn not in X.columns:
                    X[fn] = np.nan
            X = X[fns]
            pred = float(model.predict(X)[0])
        else:
            raise
    book_line = float(spread)
    edge = pred - book_line
    pick = "HOME" if edge > 0 else "AWAY"
    z = (book_line - pred) / eval_std
    p_home = 1.0 - normal_cdf(z)
    p_away = 1.0 - p_home
    print("-----------------------------------------------------------")
    print(f"{home} vs {away} on {as_of_date.date()}")
    print(f"book spread (home): {book_line:+.2f}")
    print(f"predicted margin (home - away): {pred:+.2f}")
    print(f"edge = pred_margin - book_line = {edge:+.2f}")
    if pick == "HOME":
        print(f"PICK: HOME to cover (p_home ≈ {p_home*100:.1f}%)")
    else:
        print(f"PICK: AWAY to cover (p_away ≈ {p_away*100:.1f}%)")
    print("-----------------------------------------------------------")


def batch_predict(model, team_games, csv_path, eval_std, windows=(5,10)):
    p = Path(csv_path)
    if not p.exists():
        print("Batch file not found:", csv_path)
        return
    df = pd.read_csv(p, low_memory=False)
    required = [c for c in ["home_team","away_team","date","spread"] if c not in df.columns]
    if required:
        print("Batch CSV missing columns. Required: home_team,away_team,date,spread")
        return
    outs = []
    for _, r in df.iterrows():
        try:
            home = r["home_team"]
            away = r["away_team"]
            date = r["date"]
            spread = r["spread"]
            # compute
            as_of = pd.to_datetime(date)
            home_feats = get_last_team_feats(team_games, home, as_of, windows=windows)
            away_feats = get_last_team_feats(team_games, away, as_of, windows=windows)
            if home_feats is None or away_feats is None:
                outs.append({"home":home,"away":away,"date":date,"pred_margin":np.nan,"edge":np.nan,"pick":"MISSING_DATA"})
                continue
            feat_vec = build_feature_vector(home_feats, away_feats, as_of, windows=windows)
            X = pd.DataFrame([feat_vec])
            try:
                pred = float(model.predict(X)[0])
            except Exception:
                fns = None
                try:
                    fns = model.feature_name()
                except Exception:
                    pass
                if fns:
                    for fn in fns:
                        if fn not in X.columns:
                            X[fn] = np.nan
                    X = X[fns]
                    pred = float(model.predict(X)[0])
                else:
                    pred = np.nan
            edge = pred - float(spread)
            pick = "HOME" if edge > 0 else "AWAY"
            outs.append({"home":home,"away":away,"date":date,"pred_margin":pred,"edge":edge,"pick":pick})
        except Exception as e:
            outs.append({"home":None,"away":None,"date":None,"pred_margin":np.nan,"edge":np.nan,"pick":"ERROR"})
    outdf = pd.DataFrame(outs)
    outp = p.parent / (p.stem + "_predictions.csv")
    outdf.to_csv(outp, index=False)
    print("Wrote batch predictions to:", outp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", required=True, help="historical games CSV (one row per game with date,home_team,away_team,home_pts,away_pts)")
    parser.add_argument("--model", required=True, help="trained model (.pkl)")
    parser.add_argument("--eval", default="models/eval_report.csv", help="optional eval report to estimate residual std")
    args = parser.parse_args()

    games_path = Path(args.games)
    if not games_path.exists():
        print("Games CSV not found:", games_path)
        sys.exit(2)
    df_games = pd.read_csv(games_path, low_memory=False)
    # normalize columns
    if "date" not in df_games.columns:
        date_cands = [c for c in df_games.columns if "date" in c.lower()]
        if date_cands:
            df_games = df_games.rename(columns={date_cands[0]: "date"})
    if "home_team" not in df_games.columns or "away_team" not in df_games.columns:
        mapping = {}
        for c in df_games.columns:
            low = c.lower()
            if "home" in low and "team" in low:
                mapping[c] = "home_team"
            if ("visitor" in low or "away" in low) and "team" in low:
                mapping[c] = "away_team"
        if mapping:
            df_games = df_games.rename(columns=mapping)
    if "home_pts" not in df_games.columns or "away_pts" not in df_games.columns:
        # attempt heuristics
        for c in df_games.columns:
            low = c.lower()
            if "home" in low and ("pts" in low or "points" in low or "score" in low):
                df_games = df_games.rename(columns={c: "home_pts"})
            if ("away" in low or "visitor" in low) and ("pts" in low or "points" in low or "score" in low):
                df_games = df_games.rename(columns={c: "away_pts"})

    df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce")

    # build team_game rolls once
    print("Building rolling team features from historical games... (may take a few seconds)")
    team_games = compute_team_rolls(df_games, date_col="date", windows=(5,10))

    # load model
    model_path = Path(args.model)
    if not model_path.exists():
        print("Model file not found:", model_path)
        sys.exit(2)
    model = joblib.load(model_path)

    # estimate residual std
    eval_std = estimate_residual_std(Path(args.eval))
    if eval_std is None:
        print("No eval report found or could not read it; using fallback residual std = 12.0")
        eval_std = 12.0
    else:
        print(f"Using residual std from eval report: {eval_std:.3f}")

    print("-----------------------------------------------------------")
    print("Instructions:")
    print(" Type a game as: HOME AWAY DATE SPREAD")
    print("\nInteractive CLI started. Type 'help' for usage. 'quit' to exit.\n")
    while True:
        try:
            line = input("predict> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not line:
            continue
        if line.lower() in ("quit","exit"):
            print("bye")
            break
        if line.lower() == "help":
            prompt_help()
            continue
        if line.lower().startswith("batch "):
            _, path = line.split(None, 1)
            batch_predict(model, team_games, path.strip(), eval_std)
            continue
        # otherwise parse tokens: HOME AWAY DATE SPREAD
        parts = line.split()
        if len(parts) < 4:
            print("Unrecognized input. Type 'help' for examples.")
            continue
        home, away, date_str, spread = parts[0], parts[1], parts[2], parts[3]
        try:
            predict_single(model, team_games, home, away, date_str, spread, eval_std)
        except Exception as e:
            print("Error during prediction:", e)


if __name__ == "__main__":
    main()
