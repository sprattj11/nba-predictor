# predict_by_abbrev_with_nbaapi.py
"""
Predict an upcoming game by team abbreviation using a saved model and your historical CSV.

Usage:
    python3 predict_by_abbrev_with_nbaapi.py

Prompts:
 - Home team abbrev (e.g., LAL)
 - Away team abbrev (e.g., BOS)
 - Game date (YYYY-MM-DD)

Expectations:
 - Trained model: nba_win_predictor.pkl
 - Historical CSV: nba_games.csv (has GAME_DATE_EST, HOME_TEAM_ID, VISITOR_TEAM_ID, plus the _home/_away stat columns)
"""

import sys
import joblib
import pandas as pd
from datetime import datetime
from nba_api.stats.static import teams as nba_teams

MODEL_PATH = "nba_win_predictor.pkl"
GAMES_CSV = "nba_games.csv"

FEATURES = [
    "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"
]


def build_abbrev_map():
    """Use nba_api to build abbrev -> team_id mapping."""
    try:
        all_teams = nba_teams.get_teams()
    except Exception as e:
        print("ERROR: nba_api call failed. Make sure nba_api is installed (`pip install nba_api`).")
        print("Exception:", e)
        return {}
    mapping = {}
    for t in all_teams:
        # `t` is a dict with keys like 'id', 'full_name', 'abbreviation'
        abb = str(t.get("abbreviation", "")).strip().upper()
        tid = int(t.get("id"))
        if abb:
            mapping[abb] = tid
    return mapping


def team_averages_before(df, team_id, date, home_or_away="home"):
    """Compute mean stats for a team before `date` using HOME_TEAM_ID / VISITOR_TEAM_ID filter."""
    if df["GAME_DATE_EST"].dtype == object:
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"], errors="coerce")
    mask = (df["GAME_DATE_EST"] < date)
    if home_or_away == "home":
        if "HOME_TEAM_ID" in df.columns:
            mask &= (df["HOME_TEAM_ID"] == team_id)
        cols = ["PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home"]
    else:
        if "VISITOR_TEAM_ID" in df.columns:
            mask &= (df["VISITOR_TEAM_ID"] == team_id)
        cols = ["PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"]
    subset = df.loc[mask, cols]
    if subset.empty:
        # fallback: league-average for those columns (safer than zeros)
        league_means = df[cols].mean().fillna(0.0)
        return list(league_means)
    return list(subset.mean())


def build_feature_row(df, home_team_id, away_team_id, game_date_str):
    game_date = pd.to_datetime(game_date_str)
    home_stats = team_averages_before(df, home_team_id, game_date, home_or_away="home")
    away_stats = team_averages_before(df, away_team_id, game_date, home_or_away="away")
    row = dict(zip(FEATURES, home_stats + away_stats))
    return pd.DataFrame([row], columns=FEATURES)


def main():
    # Load historical data
    try:
        df = pd.read_csv(GAMES_CSV)
    except FileNotFoundError:
        print(f"ERROR: Could not find {GAMES_CSV} in working directory.")
        sys.exit(1)

    # Build mapping via nba_api
    mapping = build_abbrev_map()
    if not mapping:
        print("Could not build abbreviation -> team_id map via nba_api.")
        print("If you prefer, provide numeric team IDs instead of abbreviations.")
        sys.exit(1)

    print("Detected team abbreviations (sample):", list(mapping.keys())[:20])

    home_abbrev = input("Home team abbreviation (e.g., LAL): ").strip().upper()
    away_abbrev = input("Away team abbreviation (e.g., BOS): ").strip().upper()
    game_date = input("Game date (YYYY-MM-DD): ").strip()

    if home_abbrev not in mapping:
        print(f"Home abbreviation '{home_abbrev}' not found. Available examples: {list(mapping.keys())[:30]}")
        sys.exit(1)
    if away_abbrev not in mapping:
        print(f"Away abbreviation '{away_abbrev}' not found. Available examples: {list(mapping.keys())[:30]}")
        sys.exit(1)

    home_team_id = mapping[home_abbrev]
    away_team_id = mapping[away_abbrev]

    feat_row = build_feature_row(df, home_team_id, away_team_id, game_date)
    print("\nConstructed feature row (inputs to model):")
    print(feat_row.to_string(index=False))

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"ERROR loading model {MODEL_PATH}: {e}")
        sys.exit(1)

    # Optional scaler
    scaler = None
    try:
        scaler = joblib.load("scaler.pkl")
        print("Loaded scaler 'scaler.pkl' and will apply it to features.")
    except Exception:
        scaler = None

    X = feat_row.values
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print("WARNING: scaler.transform failed:", e)
            print("Proceeding without scaling.")
            X = feat_row.values

    # Predict probabilities
    if hasattr(model, "predict_proba"):
        proba_home = float(model.predict_proba(X)[0][1])
        proba_away = 1.0 - proba_home
    else:
        pred = int(model.predict(X)[0])
        proba_home = 1.0 if pred == 1 else 0.0
        proba_away = 1.0 - proba_home

    hard_pred = "HOME" if proba_home >= 0.5 else "AWAY"
    print(f"\nPredicted winner (hard): {hard_pred}")
    print(f"Probability HOME wins: {proba_home:.3f}")
    print(f"Probability AWAY wins: {proba_away:.3f}")


if __name__ == "__main__":
    main()
