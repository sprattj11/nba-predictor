"""
features_and_prep.py

Helpers to load games_details.csv, infer column names, build chronological
(pre-game) features for spread prediction.

Exposes:
- load_games(csv_path)
- build_features(games_df, min_games_required=5)

Designed to be imported by train_spread.py
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict

COMMON_COLUMN_ALIASES = {
    "date": ["date", "game_date", "gamedate", "game_date_utc"],
    "season": ["season", "season_id", "season_year"],
    "home_team": ["home_team", "home", "home_team_name", "home_team_id"],
    "away_team": ["away_team", "away", "away_team_name", "away_team_id"],
    "home_pts": ["home_pts", "home_points", "home_score", "pts_home", "score_home"],
    "away_pts": ["away_pts", "away_points", "away_score", "pts_away", "score_away"],
    "book_line": ["bookmaker_line", "line", "spread", "home_spread", "book_line", "line_home"],
    # optional: elo or ratings if present
    "home_elo": ["home_elo", "elo_home", "home_rating"],
    "away_elo": ["away_elo", "elo_away", "away_rating"],
}

def _find_column(df: pd.DataFrame, aliases: list):
    for a in aliases:
        if a in df.columns:
            return a
    return None

def infer_columns(df: pd.DataFrame) -> Dict[str,str]:
    """Return mapping from canonical name -> actual column name in df."""
    mapping = {}
    for canonical, aliases in COMMON_COLUMN_ALIASES.items():
        col = _find_column(df, aliases)
        if col:
            mapping[canonical] = col
    return mapping

def _looks_like_date_series(s: pd.Series) -> bool:
    """Heuristic: check whether a series contains many date-like strings or datetimes."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    # sample a few non-null values and see if they parse to datetimes
    sample = s.dropna().astype(str).head(20).tolist()
    if not sample:
        return False
    parsed = 0
    for v in sample:
        # quick heuristics: common separators or year-like prefix
        if "-" in v or "/" in v or (len(v) >= 4 and v[:4].isdigit()):
            parsed += 1
    return parsed >= max(1, len(sample)//4)  # if ~25% of sample looks like date, treat as date-like

def load_games(csv_path: str) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Load CSV, parse dates, infer columns, return df and mapping.

    More robust than earlier: uses low_memory=False, prefers actual date-like
    columns (excluding id columns), and avoids forcing the first column to be parsed.
    """
    # read with low_memory=False to avoid mixed-type chunking warnings
    df = pd.read_csv(csv_path, low_memory=False)

    # look for explicit date-like columns (prefer names containing 'date' but exclude "*id*")
    name_candidates = [c for c in df.columns if "date" in c.lower() and "id" not in c.lower()]
    date_col_chosen = None

    # prefer explicit names list first (existing behavior)
    for dcol in ["date", "game_date", "gamedate", "game_date_utc"]:
        if dcol in df.columns:
            date_col_chosen = dcol
            break

    # if not found, try name_candidates
    if date_col_chosen is None and name_candidates:
        date_col_chosen = name_candidates[0]

    # if still not found, check for any column that looks like a date (but skip obvious id columns)
    if date_col_chosen is None:
        for c in df.columns:
            if "id" in c.lower():
                continue
            if _looks_like_date_series(df[c]):
                date_col_chosen = c
                break

    # If still none, fall back to first column ONLY if it looks like a date
    if date_col_chosen is None:
        first_col = df.columns[0]
        if _looks_like_date_series(df[first_col]) and "id" not in first_col.lower():
            date_col_chosen = first_col

    # If we still couldn't find a date, raise a clear error (instead of silently picking GAME_ID)
    if date_col_chosen is None:
        raise ValueError(
            "Couldn't find a date column automatically. Candidate columns inspected: "
            f"{[c for c in df.columns if 'date' in c.lower() or 'game' in c.lower()][:20]}. "
            "Please ensure your CSV contains a valid date column (e.g., 'date' or 'game_date'), "
            "or modify COMMON_COLUMN_ALIASES accordingly."
        )

    # Parse the selected date column to datetime
    try:
        df[date_col_chosen] = pd.to_datetime(df[date_col_chosen], errors="coerce")
    except Exception:
        df[date_col_chosen] = pd.to_datetime(df[date_col_chosen].astype(str), errors="coerce")

    # warn if many values failed to parse
    n_failed = df[date_col_chosen].isna().sum()
    if n_failed > 0:
        pct = n_failed / max(1, len(df)) * 100
        print(f"Warning: {n_failed} / {len(df)} ({pct:.2f}%) rows failed to parse from '{date_col_chosen}' to datetime. "
              "Check CSV formatting or provide a clearer date column.")

    colmap = infer_columns(df)

    # Ensure canonical 'date' mapping exists; prefer the explicit chosen column
    colmap["date"] = date_col_chosen

    return df, colmap


def build_team_game_table(games: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """
    Build a per-team per-game table (one row per team per game) useful
    to compute rolling per-team stats.
    Assumes games index is the original game identifier (used as game_id).
    """
    date_col = colmap["date"]
    # preserve original index as game_id to make merges unambiguous
    games = games.copy()
    games = games.reset_index().rename(columns={"index":"game_id"})
    rows = []
    for _, r in games.iterrows():
        home = {
            "game_id": r["game_id"],
            "date": r[date_col],
            "team": r[colmap["home_team"]],
            "is_home": 1,
            "opp": r[colmap["away_team"]],
            "team_pts": r[colmap["home_pts"]],
            "opp_pts": r[colmap["away_pts"]],
        }
        away = {
            "game_id": r["game_id"],
            "date": r[date_col],
            "team": r[colmap["away_team"]],
            "is_home": 0,
            "opp": r[colmap["home_team"]],
            "team_pts": r[colmap["away_pts"]],
            "opp_pts": r[colmap["home_pts"]],
        }
        rows.append(home)
        rows.append(away)
    team_games = pd.DataFrame(rows).sort_values(["team","date"]).reset_index(drop=True)
    # basic derived stats
    team_games["margin"] = team_games["team_pts"] - team_games["opp_pts"]
    team_games["win"] = (team_games["margin"] > 0).astype(int)
    return team_games

def add_rolling_features(games: pd.DataFrame,
                         colmap: dict,
                         windows=(5,10),
                         min_games_required:int=5) -> pd.DataFrame:
    """
    Accepts raw games (one row per game), returns games augmented with features.
    Features are computed using only prior games (shifted) to avoid leakage.

    NOTE: This function computes robust rest features (days since previous game)
    for both home and away teams plus b2b flags. It also computes the rolling
    statistics used previously.
    """
    # Work on a copy, but keep original index to use as game_id
    g = games.copy()
    # make sure date col is datetime
    g[colmap["date"]] = pd.to_datetime(g[colmap["date"]], errors="coerce")
    # basic validation
    if colmap.get("home_pts") not in g.columns or colmap.get("away_pts") not in g.columns:
        raise ValueError("Couldn't find home/away score columns in CSV. Found mapping: {}".format(colmap))

    # Build per-team per-game table (this sets game_id = original index)
    team_games = build_team_game_table(g, colmap)

    # --- Rolling stats (margin/pts/winrate) computed using prior games only ---
    feat_cols = []
    for w in windows:
        mcol = f"margin_roll_{w}"
        wcol = f"winrate_roll_{w}"
        pcol = f"pts_roll_{w}"
        ocol = f"opp_pts_roll_{w}"
        # shift by 1 to ensure only prior games are used
        team_games[mcol] = team_games.groupby("team")["margin"].shift(1).rolling(window=w, min_periods=1).mean()
        team_games[wcol] = team_games.groupby("team")["win"].shift(1).rolling(window=w, min_periods=1).mean()
        team_games[pcol] = team_games.groupby("team")["team_pts"].shift(1).rolling(window=w, min_periods=1).mean()
        team_games[ocol] = team_games.groupby("team")["opp_pts"].shift(1).rolling(window=w, min_periods=1).mean()
        feat_cols += [mcol, wcol, pcol, ocol]

    # --- Rest features: days since previous game per team (robust vectorized approach) ---
    # for each team row in team_games compute prev_date and days_rest
    team_games["prev_date"] = team_games.groupby("team")["date"].shift(1)
    team_games["days_rest"] = (team_games["date"] - team_games["prev_date"]).dt.days
    # b2b flag: played previous day
    team_games["is_b2b"] = (team_games["days_rest"] == 1).astype(int)

    # --- Prepare lookups to attach rolling features and rest back to game-level rows ---
    # We'll pivot team_games to have per-game entries for home and away teams keyed by (game_id, team)
    # Select columns to bring back
    return_cols = ["game_id", "team"] + feat_cols + ["days_rest", "is_b2b"]
    tg_small = team_games[return_cols].copy()

    # split into home/away features by merging twice
    # first ensure g has game_id column matching team_games' game_id
    g = g.reset_index().rename(columns={"index":"game_id"})

    # prepare home features: merge on game_id and team == home_team
    home_tg = tg_small.rename(columns={
        "team": "home_team",
        "days_rest": "days_rest_home",
        "is_b2b": "is_b2b_home",
    })
    # rename rolling feature columns to prefix 'home_'
    home_tg = home_tg.rename(columns={c: f"home_{c}" for c in home_tg.columns if c not in ["game_id", "home_team"]})

    away_tg = tg_small.rename(columns={
        "team": "away_team",
        "days_rest": "days_rest_away",
        "is_b2b": "is_b2b_away",
    })
    away_tg = away_tg.rename(columns={c: f"away_{c}" for c in away_tg.columns if c not in ["game_id", "away_team"]})

    # Merge home & away features onto g using (game_id, team)
    merged = pd.merge(g, home_tg, left_on=["game_id", colmap["home_team"]], right_on=["game_id", "home_team"], how="left")
    merged = pd.merge(merged, away_tg, left_on=["game_id", colmap["away_team"]], right_on=["game_id", "away_team"], how="left")

    # Some seasons or first games will have NaNs for days_rest etc. Fill with reasonable defaults
    # For rest days, a common neutral default is 3 (offseason/preseason or first game)
    merged["days_rest_home"] = merged["home_days_rest"].fillna(3).astype(float)
    merged["days_rest_away"] = merged["away_days_rest"].fillna(3).astype(float)
    # b2b flags: if missing treat as 0
    home_b2b = merged["home_is_b2b"] if "home_is_b2b" in merged.columns else pd.Series(0, index=merged.index)
    away_b2b = merged["away_is_b2b"] if "away_is_b2b" in merged.columns else pd.Series(0, index=merged.index)

    merged["is_b2b_home"] = home_b2b.fillna(0).astype(int)
    merged["is_b2b_away"] = away_b2b.fillna(0).astype(int)

    # rename any rolling feature columns that used the original names (they are prefixed with home_/away_)
    # create derived differences for selected windows
    for w in windows:
        # names created earlier were like 'margin_roll_5' -> after merge they should be 'home_margin_roll_5'
        h_margin = f"home_margin_roll_{w}"
        a_margin = f"away_margin_roll_{w}"
        h_win = f"home_winrate_roll_{w}"
        a_win = f"away_winrate_roll_{w}"
        h_pts = f"home_pts_roll_{w}"
        a_pts = f"away_pts_roll_{w}"

        # derived differences (these will be NaN if underlying NaNs exist)
        merged[f"home_minus_away_margin_roll_{w}"] = merged.get(h_margin) - merged.get(a_margin)
        merged[f"home_minus_away_winrate_{w}"] = merged.get(h_win) - merged.get(a_win)
        merged[f"home_minus_away_pts_{w}"] = merged.get(h_pts) - merged.get(a_pts)

    # rest diff and normalized rest features
    merged["rest_diff"] = merged["days_rest_home"] - merged["days_rest_away"]
    merged["home_rest"] = merged["days_rest_home"]
    merged["away_rest"] = merged["days_rest_away"]

    # target: margin = home_pts - away_pts (keep canonical naming)
    merged["margin"] = merged[colmap["home_pts"]] - merged[colmap["away_pts"]]

    # keep original columns from g (including date/home/away) plus features
    # drop helper duplicate columns introduced by merges
    # identify columns to drop ('home_team'/'away_team' from tg merges)
    drop_cols = [c for c in merged.columns if c in ("home_team","away_team","home_days_rest","away_days_rest","home_is_b2b","away_is_b2b")]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # return merged which contains the original game-level data plus rolling & rest features
    return merged

def get_feature_list(windows=(5,10)):
    feats = []
    for w in windows:
        feats += [
            f"home_minus_away_margin_roll_{w}",
            f"home_minus_away_winrate_{w}",
            f"home_minus_away_pts_{w}",
        ]
    feats += ["rest_diff", "home_rest", "away_rest", "is_b2b_home", "is_b2b_away"]
    return feats

if __name__ == "__main__":
    # quick local smoke test if run directly
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to games_details.csv")
    args = p.parse_args()
    df, colmap = load_games(args.csv)
    print("Inferred columns:", colmap)
    feats = add_rolling_features(df, colmap)
    print("Built features, sample:")
    # show a stable subset of columns for quick inspection
    sample_cols = [colmap["date"], colmap["home_team"], colmap["away_team"],
                   "home_rest", "away_rest", "rest_diff"] + get_feature_list()
    # print only columns that exist
    sample_cols = [c for c in sample_cols if c in feats.columns]
    print(feats.head()[sample_cols])
