#!/usr/bin/env python3
"""
Interactive CLI (market-aware)

Usage:
  python models/interactive_predict_spread_market.py \
    --games data/games_summary_merged.csv \
    --market-betting data/"NBA Betting Data 2008-2025.csv" \
    --market-model models/market_spread_model.pkl \
    --eval models/eval_report.csv

At the prompt you can:
  - Type: HOME AWAY DATE [spread]
    e.g. SAS IND 2025-10-18 -5.5
    If spread omitted the CLI will try to look up the market spread from betting CSV.
  - Type: batch path/to/upcoming.csv   (CSV must contain home_team,away_team,date,spread [optional])
  - Type: help, quit, exit
"""
import argparse
from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
from math import erf, sqrt

def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def estimate_residual_std(eval_report_path: Path):
    if eval_report_path.exists():
        try:
            er = pd.read_csv(eval_report_path, low_memory=False)
            if ("margin" in er.columns) and ("pred_margin" in er.columns):
                resid = er["margin"] - er["pred_margin"]
                return float(resid.std(ddof=0))
        except Exception:
            pass
    return None

# team rolling features (same as before)
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

def prompt_help():
    print("Commands:")
    print("  HOME AWAY DATE [SPREAD]   e.g. LAL BOS 2024-10-25 -3.5")
    print("  batch PATH_TO_CSV         CSV must contain columns: home_team,away_team,date,spread (spread optional if betting CSV available)")
    print("  help                      show this message")
    print("  quit / exit               exit")
    print("")

def find_market_spread(betting_df, home, away, as_of_date):
    """Try exact match by date + teams (case-insensitive abbreviations)."""
    if betting_df is None:
        return None
    # normalize
    b = betting_df.copy()
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    # columns in your betting CSV were home and away abbreviations like 'home'/'away'
    # try matching using those fields
    cand = b[
        (b["date"] == as_of_date) &
        ((b["home"].str.lower() == home.lower()) & (b["away"].str.lower() == away.lower()))
    ]
    if not cand.empty:
        # use the 'spread' column (closing spread) and 'whos_favored' to get signed market spread
        row = cand.iloc[0]
        s = float(row["spread"])
        if str(row.get("whos_favored", "home")).lower() == "home":
            return s
        else:
            return -s
    return None

def predict_single_market(model_market, model_margin, team_games, betting_df, home, away, date_str, input_spread, eval_std, resid_epsilon=0.5):
    """
    Improved single-game predict function:
    - Uses margin model (if available) as primary prediction of margin.
    - Uses market model residual only when |pred_resid| > resid_epsilon (i.e., meaningful).
    - Computes a consistent 'combined_edge' = predicted_margin - market_spread
      and uses that for pick/probability.
    """
    as_of_date = pd.to_datetime(date_str)

    # find market spread (signed - positive means home favored)
    market_spread = None
    if input_spread is None:
        market_spread = find_market_spread(betting_df, home, away, as_of_date)
    if market_spread is None and input_spread is not None:
        market_spread = float(input_spread)
    if market_spread is None:
        print("No market spread found for this game and none provided. Please provide spread or ensure betting CSV contains the game.")
        return

    # rolling feats
    home_feats = get_last_team_feats(team_games, home, as_of_date)
    away_feats = get_last_team_feats(team_games, away, as_of_date)
    if home_feats is None or away_feats is None:
        print(f"No historical data for one of the teams before {as_of_date.date()}.")
        return

    feat_vec = build_feature_vector(home_feats, away_feats, as_of_date)
    X = pd.DataFrame([feat_vec])

    # Predict residual (market model): predicted_resid = E[actual_margin - market_spread]
    pred_resid = 0.0
    used_market_model = False
    if model_market is not None:
        try:
            pred_resid = float(model_market.predict(X)[0])
            used_market_model = True
        except Exception:
            try:
                fns = model_market.feature_name()
            except Exception:
                fns = None
            if fns:
                for fn in fns:
                    if fn not in X.columns:
                        X[fn] = np.nan
                Xm = X[fns]
                pred_resid = float(model_market.predict(Xm)[0])
                used_market_model = True
            else:
                pred_resid = 0.0

    # Predict absolute margin (margin model) if available
    pred_margin = None
    if model_margin is not None:
        Xm = pd.DataFrame([feat_vec])
        try:
            pred_margin = float(model_margin.predict(Xm)[0])
        except Exception:
            try:
                fns = model_margin.feature_name()
            except Exception:
                fns = None
            if fns:
                for fn in fns:
                    if fn not in Xm.columns:
                        Xm[fn] = np.nan
                Xm = Xm[fns]
                pred_margin = float(model_margin.predict(Xm)[0])
            else:
                pred_margin = None

    # Decide which prediction to use for the final "predicted margin"
    # Strategy:
    #  - If pred_margin exists, use it (primary).
    #  - If no pred_margin but market model predicts a meaningful residual, compute pred_margin = market_spread + pred_resid.
    #  - If neither available, fall back to market_spread (no edge).
    if pred_margin is None:
        # fallback to market_model if it gave a meaningful residual
        if used_market_model and abs(pred_resid) >= resid_epsilon:
            pred_margin = market_spread + pred_resid
            source = "market_model"
        else:
            pred_margin = market_spread  # no info
            source = "market_only"
    else:
        source = "margin_model"

    # combined edge (how many points our predicted margin is above market)
    combined_edge = pred_margin - market_spread

    # pick
    pick = "HOME" if combined_edge > 0 else "AWAY"

    # probability estimate: P(home covers) = 1 - Phi((spread - pred_margin)/sigma)
    if eval_std is None or eval_std <= 0:
        eval_std = 12.0
    z = (market_spread - pred_margin) / eval_std
    p_home_covers = 1.0 - normal_cdf(z)
    p_away_covers = 1.0 - p_home_covers

    # Print consistent results
    print("-----------------------------------------------------------")
    print(f"{home} vs {away} on {as_of_date.date()}")
    print(f"Market spread (home): {market_spread:+.2f}")
    print(f"Predicted margin (home - away) [{source}]: {pred_margin:+.2f}")
    print(f"Model line (home) [= market + pred_resid]: {(market_spread + pred_resid):+.2f}  (pred_resid = {pred_resid:+.2f})")
    print(f"Combined edge = pred_margin - market_spread = {combined_edge:+.2f}")
    if pick == "HOME":
        print(f"PICK: HOME to cover (p_home ≈ {p_home_covers*100:.1f}%)")
    else:
        print(f"PICK: AWAY to cover (p_away ≈ {p_away_covers*100:.1f}%)")
    print("-----------------------------------------------------------")


def batch_predict_market(model_market, model_margin, team_games, betting_df, csv_path, eval_std):
    p = Path(csv_path)
    if not p.exists():
        print("Batch file not found:", csv_path)
        return
    df = pd.read_csv(p, low_memory=False)
    if "home_team" not in df.columns or "away_team" not in df.columns or "date" not in df.columns:
        print("Batch CSV must have columns: home_team,away_team,date (optional spread column named 'spread').")
        return
    outs = []
    for _, r in df.iterrows():
        home = r["home_team"]; away = r["away_team"]; date = r["date"]
        spread = r.get("spread", None)
        # attempt to find market if spread empty
        market_spread = None
        if pd.isna(spread) or spread is None:
            market_spread = find_market_spread(betting_df, home, away, pd.to_datetime(date))
            if market_spread is None:
                # skip or use NaN
                outs.append({"home":home,"away":away,"date":date,"pred_margin":np.nan,"model_line":np.nan,"edge":np.nan,"pick":"MISSING_MARKET"})
                continue
        try:
            # reuse single predict
            as_of = pd.to_datetime(date)
            home_feats = get_last_team_feats(team_games, home, as_of)
            away_feats = get_last_team_feats(team_games, away, as_of)
            if home_feats is None or away_feats is None:
                outs.append({"home":home,"away":away,"date":date,"pred_margin":np.nan,"model_line":np.nan,"edge":np.nan,"pick":"MISSING_HISTORY"})
                continue
            feat_vec = build_feature_vector(home_feats, away_feats, as_of)
            X = pd.DataFrame([feat_vec])
            pred_resid = float(model_market.predict(X)[0]) if model_market is not None else 0.0
            market_spread = market_spread if market_spread is not None else float(spread)
            model_line = market_spread + pred_resid
            pred_margin = float(model_margin.predict(X)[0]) if model_margin is not None else np.nan
            edge = model_line - market_spread
            pick = "HOME" if edge > 0 else "AWAY"
            outs.append({"home":home,"away":away,"date":date,"pred_margin":pred_margin,"model_line":model_line,"edge":edge,"pick":pick})
        except Exception as e:
            outs.append({"home":home,"away":away,"date":date,"pred_margin":np.nan,"model_line":np.nan,"edge":np.nan,"pick":"ERROR"})
    outdf = pd.DataFrame(outs)
    outp = p.parent / (p.stem + "_predictions.csv")
    outdf.to_csv(outp, index=False)
    print("Wrote batch predictions to:", outp)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="historical games CSV (one row per game with date,home_team,away_team,home_pts,away_pts)")
    p.add_argument("--market-betting", required=False, help="betting CSV (e.g. 'NBA Betting Data 2008-2025.csv')")
    p.add_argument("--market-model", required=False, help="market model (predicting residual) .pkl")
    p.add_argument("--model", required=False, help="optional margin model .pkl (predicts absolute margin)")
    p.add_argument("--eval", default="models/eval_report.csv", help="eval report to estimate residual std")
    args = p.parse_args()

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
        for c in df_games.columns:
            low = c.lower()
            if "home" in low and ("pts" in low or "points" in low or "score" in low):
                df_games = df_games.rename(columns={c: "home_pts"})
            if ("away" in low or "visitor" in low) and ("pts" in low or "points" in low or "score" in low):
                df_games = df_games.rename(columns={c: "away_pts"})

    df_games["date"] = pd.to_datetime(df_games["date"], errors="coerce")
    print("Building rolling team features from historical games... (may take a few seconds)")
    team_games = compute_team_rolls(df_games, date_col="date", windows=(5,10))

    # load market betting CSV if provided
    betting_df = None
    if args.market_betting:
        bet_path = Path(args.market_betting)
        if bet_path.exists():
            betting_df = pd.read_csv(bet_path, low_memory=False)
            # normalize betting df column names if needed: expecting columns 'date','home','away','spread','whos_favored'
            if "date" not in betting_df.columns:
                dc = next((c for c in betting_df.columns if "date" in c.lower()), None)
                if dc:
                    betting_df = betting_df.rename(columns={dc: "date"})
        else:
            print("Warning: betting CSV not found:", bet_path)

    # load models
    model_market = None
    if args.market_model:
        mmp = Path(args.market_model)
        if mmp.exists():
            model_market = joblib.load(mmp)
            print("Loaded market model:", mmp)
        else:
            print("Warning: market model not found:", mmp)

    model_margin = None
    if args.model:
        mm = Path(args.model)
        if mm.exists():
            model_margin = joblib.load(mm)
            print("Loaded margin model:", mm)

    eval_std = estimate_residual_std(Path(args.eval))
    if eval_std is None:
        print("No eval report found or could not read it; using fallback residual std = 12.0")
        eval_std = 12.0
    else:
        print(f"Using residual std from eval report: {eval_std:.3f}")

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
            batch_predict_market(model_market, model_margin, team_games, betting_df, path.strip(), eval_std)
            continue
        parts = line.split()
        if len(parts) < 3:
            print("Unrecognized input. Type 'help' for examples.")
            continue
        # parse flexible input: HOME AWAY DATE [SPREAD]
        home, away, date_str = parts[0], parts[1], parts[2]
        spread = parts[3] if len(parts) >= 4 else None
        try:
            predict_single_market(model_market, model_margin, team_games, betting_df, home, away, date_str, spread, eval_std)
        except Exception as e:
            print("Error during prediction:", e)

if __name__ == "__main__":
    main()
