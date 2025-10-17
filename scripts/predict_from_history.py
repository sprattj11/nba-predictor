# predict_from_history.py
import joblib
import pandas as pd
from datetime import datetime

MODEL_PATH = "nba_win_predictor.pkl"
GAMES_CSV = "nba_games.csv"
FEATURES = [
    "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"
]

def team_averages_before(df, team_id, date, home_or_away="home"):
    """
    Compute mean stats for a team before `date`.
    `home_or_away` is 'home' or 'away' to select columns like 'PTS_home' or 'PTS_away' from historical rows.
    """
    # choose the correct team id column to filter
    mask = (df["GAME_DATE_EST"] < date)
    # For home stats, we need rows where team was home; for away stats, team was away.
    if home_or_away == "home":
        mask &= (df["HOME_TEAM_ID"] == team_id)
        cols = ["PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home"]
    else:
        mask &= (df["VISITOR_TEAM_ID"] == team_id)
        cols = ["PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"]
    subset = df.loc[mask, cols]
    if subset.empty:
        # fallback: if no prior games in dataset for this team before date, return zeros or NaNs
        return [0.0]*len(cols)
    return list(subset.mean())

def build_feature_row(df, home_team_id, away_team_id, game_date_str):
    # ensure GAME_DATE_EST is datetime
    if df["GAME_DATE_EST"].dtype == object:
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    game_date = pd.to_datetime(game_date_str)

    home_stats = team_averages_before(df, home_team_id, game_date, home_or_away="home")
    away_stats = team_averages_before(df, away_team_id, game_date, home_or_away="away")

    # Concatenate in the same order as FEATURES
    row = dict(zip(FEATURES, home_stats + away_stats))
    return pd.DataFrame([row], columns=FEATURES)

def main():
    # Example usage variables - replace with real ids/date
    # Example: home_team_id = 1610612747, away_team_id = 1610612739, date = "2023-03-10"
    home_team_id = int(input("Home team ID: ").strip())
    away_team_id = int(input("Away team ID: ").strip())
    game_date = input("Game date (YYYY-MM-DD): ").strip()

    df = pd.read_csv(GAMES_CSV)
    model = joblib.load(MODEL_PATH)

    feat_row = build_feature_row(df, home_team_id, away_team_id, game_date)
    print("Feature row constructed:\n", feat_row.to_string(index=False))

    prob = model.predict_proba(feat_row)[0][1]
    pred = model.predict(feat_row)[0]
    print(f"\nPredicted: {'HOME wins' if pred==1 else 'AWAY wins'}")
    print(f"Probability HOME wins: {prob:.3f}")

if __name__ == "__main__":
    main()
