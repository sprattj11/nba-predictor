# src/features.py
import pandas as pd

def add_rolling_features(df, window=10):
    """
    Adds rolling features for both home and away teams, including:
      - Rolling averages for box score stats (PTS, FG_PCT, FT_PCT, FG3_PCT, AST, REB)
      - Rolling win percentage over the last `window` games
    Returns the dataframe with new columns added.
    """

    # Ensure date and sort by date
    if "GAME_DATE_EST" not in df.columns:
        raise ValueError("DataFrame must contain GAME_DATE_EST column")
    if df["GAME_DATE_EST"].dtype == object or not pd.api.types.is_datetime64_any_dtype(df["GAME_DATE_EST"]):
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"], errors="coerce")
    df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

    # Choose team ID columns (supports your current column names)
    home_id_col = "HOME_TEAM_ID" if "HOME_TEAM_ID" in df.columns else "TEAM_ID_home"
    away_id_col = "VISITOR_TEAM_ID" if "VISITOR_TEAM_ID" in df.columns else "TEAM_ID_away"

    # Create away team wins (1 - home win)
    df["AWAY_TEAM_WINS"] = 1 - df["HOME_TEAM_WINS"]

    # List of basic boxscore stats
    stats = ["PTS", "FG_PCT", "FT_PCT", "FG3_PCT", "AST", "REB"]

    # Helper: compute shifted rolling mean for each team
    def _rolling_group_mean(series, team_ids):
        temp = pd.DataFrame({"team_id": team_ids, "val": series})
        return (
            temp.groupby("team_id")["val"]
            .apply(lambda x: x.shift().rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # ---- Rolling averages for boxscore stats ----
    for stat in stats:
        home_col = f"{stat}_home"
        away_col = f"{stat}_away"

        home_out = f"home_{stat}_last{window}"
        away_out = f"away_{stat}_last{window}"

        if home_col in df.columns:
            df[home_out] = _rolling_group_mean(df[home_col], df[home_id_col])
        else:
            df[home_out] = pd.NA

        if away_col in df.columns:
            df[away_out] = _rolling_group_mean(df[away_col], df[away_id_col])
        else:
            df[away_out] = pd.NA

    # ---- Rolling win % ----
    df["home_win_pct_last10"] = (
        df.groupby(home_id_col)["HOME_TEAM_WINS"]
        .apply(lambda x: x.shift().rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df["away_win_pct_last10"] = (
        df.groupby(away_id_col)["AWAY_TEAM_WINS"]
        .apply(lambda x: x.shift().rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    return df
