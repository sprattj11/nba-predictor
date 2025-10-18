#!/usr/bin/env python3
"""
predict_spread.py

Usage (example):
python models/predict_spread.py \
  --games data/games_summary_merged.csv \
  --model models/spread_model.pkl \
  --home "LAL" --away "BOS" --date "2024-10-25" --spread -3.5

Meaning of spread: "home spread" (positive means home is favored by that many points).
If spread = -3.5, that means the bookmaker shows Home -3.5 (home favored).
Outputs predicted margin (home - away), edge, pick, and probability that the chosen side covers.

Notes:
 - This script expects the games CSV to contain one-row-per-game with columns:
   GAME_ID (optional), date (datetime or parseable), home_team (abbr), away_team (abbr),
   home_pts, away_pts. If your column names differ, adjust the `colmap` logic below.
 - It estimates probability using a Gaussian approximation of model residuals. If you have
   eval_report.csv (predictions on test), place it in same folder and the script will use its residual std.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from math import erf, sqrt

# import helper functions from your project if available
try:
    from features_and_prep import load_games, infer_columns, build_team_game_table
except Exception:
    # minimal fallback if import fails: we'll load CSV directly and assume standard columns
    load_games = None
    build_team_game_table = None

def normal_cdf(x):
    # standard normal CDF via error function
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

def compute_team_rolls(games_df, date_col="date", windows=(5,10)):
    """
    Return a team-game table with rolling features for each team.
    """
    # expect games_df has columns: date, home_team, away_team, home_pts, away_pts
    # build per-team rows
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
    """
    Return a dict of rolling features for `team` right before as_of_date.
    """
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
    # last game date
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="Path to historical game-level CSV (one row per game with date/home/away/home_pts/away_pts).")
    p.add_argument("--model", required=True, help="Path to trained model (joblib .pkl)")
    p.add_argument("--eval", required=False, default="models/eval_report.csv", help="Optional eval report to estimate residual std")
    p.add_argument("--home", required=True, help="Home team abbreviation (match your games CSV)")
    p.add_argument("--away", required=True, help="Away team abbreviation")
    p.add_argument("--date", required=True, help="Game date (YYYY-MM-DD) - prediction uses history strictly before this date")
    p.add_argument("--spread", required=True, type=float, help="Bookmaker home spread (positive => home favored). Example: -3.5 means home -3.5")
    args = p.parse_args()

    games_path = Path(args.games)
    if not games_path.exists():
        raise SystemExit(f"Games file not found: {games_path}")

    # load games CSV
    df = pd.read_csv(games_path, low_memory=False)
    # try to find date column if not exactly 'date'
    if "date" not in df.columns:
        date_candidates = [c for c in df.columns if "date" in c.lower()]
        if date_candidates:
            df = df.rename(columns={date_candidates[0]: "date"})
        else:
            raise SystemExit("Couldn't find a date column in games CSV; ensure there's a date column.")

    # ensure home/away column names exist and are 'home_team' and 'away_team'
    # common variants handled:
    if "home_team" not in df.columns or "away_team" not in df.columns:
        # try common names
        mapping = {}
        if "home" in df.columns and "away" in df.columns:
            mapping["home"] = "home_team"
            mapping["away"] = "away_team"
        else:
            # look for columns with 'home' and 'away' substrings
            for c in df.columns:
                low = c.lower()
                if "home" in low and "team" in low:
                    mapping[c] = "home_team"
                if ("visitor" in low or "away" in low) and "team" in low:
                    mapping[c] = "away_team"
        if mapping:
            df = df.rename(columns=mapping)
    if "home_team" not in df.columns or "away_team" not in df.columns:
        raise SystemExit("Couldn't find home_team/away_team columns in games CSV. Rename them or use a file with those columns.")

    # ensure scores
    if "home_pts" not in df.columns or "away_pts" not in df.columns:
        # try common alternatives
        alt_h = next((c for c in df.columns if ("home" in c.lower() and ("pts" in c.lower() or "points" in c.lower() or "score" in c.lower()))), None)
        alt_a = next((c for c in df.columns if (("away" in c.lower() or "visitor" in c.lower()) and ("pts" in c.lower() or "points" in c.lower() or "score" in c.lower()))), None)
        if alt_h and alt_a:
            df = df.rename(columns={alt_h: "home_pts", alt_a: "away_pts"})
        else:
            # other layout: some files store team totals as HOME_PTS and VISITOR_PTS
            if "TEAM_PTS_HOME" in df.columns and "TEAM_PTS_AWAY" in df.columns:
                df = df.rename(columns={"TEAM_PTS_HOME":"home_pts","TEAM_PTS_AWAY":"away_pts"})
            else:
                raise SystemExit("Couldn't find home_pts/away_pts in games CSV. Ensure the file contains final scores per team.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    as_of_date = pd.to_datetime(args.date)

    # compute team-game rolling features
    team_games = compute_team_rolls(df, date_col="date", windows=(5,10))

    # get last features for each team
    home_feats = get_last_team_feats(team_games, args.home, as_of_date, windows=(5,10))
    away_feats = get_last_team_feats(team_games, args.away, as_of_date, windows=(5,10))
    if home_feats is None:
        raise SystemExit(f"No historical games found for home team {args.home} before {as_of_date.date()}")
    if away_feats is None:
        raise SystemExit(f"No historical games found for away team {args.away} before {as_of_date.date()}")

    feat_vec = build_feature_vector(home_feats, away_feats, as_of_date, windows=(5,10))
    # create pandas DataFrame with single row for model
    X = pd.DataFrame([feat_vec])

    # load model
    model = joblib.load(args.model)

    # If model expects additional features (like elo_diff), we try to add zeros or NaNs
    # but we'll attempt to align X to model feature names if possible
    try:
        # predict
        pred_margin = float(model.predict(X)[0])
    except Exception as e:
        # try to align features: get model feature names if available
        try:
            fns = model.feature_name()
        except Exception:
            fns = None
        if fns:
            # add missing columns with NaN
            for fn in fns:
                if fn not in X.columns:
                    X[fn] = np.nan
            X = X[fns]
            pred_margin = float(model.predict(X)[0])
        else:
            raise

    # compute residual std from eval report if available
    eval_std = estimate_residual_std(Path(args.eval))
    if eval_std is None:
        # fallback reasonable std
        eval_std = 12.0

    # compute edge and pick
    book_line = float(args.spread)
    edge = pred_margin - book_line
    pick = "HOME" if edge > 0 else "AWAY"
    # probability home covers (P(margin - spread > 0)) = 1 - Phi((spread - pred_margin)/sigma)
    z = (book_line - pred_margin) / eval_std
    p_home_covers = 1.0 - normal_cdf(z)
    p_away_covers = 1.0 - p_home_covers

    # pretty print
    print("=== Spread prediction ===")
    print(f"Date: {as_of_date.date()}")
    print(f"Home: {args.home}  vs  Away: {args.away}")
    print(f"Book spread (home): {book_line:+.2f}")
    print(f"Predicted margin (home - away): {pred_margin:+.2f}")
    print(f"Edge = pred_margin - book_line = {edge:+.2f}")
    if pick == "HOME":
        print(f"Model PICK: HOME to cover (prob home covers ≈ {p_home_covers*100:.1f}%)")
    else:
        print(f"Model PICK: AWAY to cover (prob away covers ≈ {p_away_covers*100:.1f}%)")
    print("\nNotes:")
    print(" - Probability is estimated with a Gaussian approx using eval residual std.")
    print(" - Adjust the threshold you require (edge) before actually placing money.")
    print(" - This script uses only rolling-team stats (5/10 game windows). For better results add more features.")

if __name__ == "__main__":
    main()
