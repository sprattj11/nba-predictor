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
    "home_pts": ["home_pts", "home_points", "home_score", "pts_home"],
    "away_pts": ["away_pts", "away_points", "away_score", "pts_away"],
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

def load_games(csv_path: str) -> Tuple[pd.DataFrame, Dict[str,str]]:
    """Load CSV, parse dates, infer columns, return df and mapping."""
    df = pd.read_csv(csv_path)
    # try parse any date-like column automatically if present
    # Prefer explicit names
    for dcol in ["date", "game_date", "gamedate", "game_date_utc"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
            break
    # Fallback: parse first datetime-like column
    if not any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns):
        # attempt to parse a column that contains 'date' in its name
        date_candidates = [c for c in df.columns if "date" in c.lower()]
        if date_candidates:
            df[date_candidates[0]] = pd.to_datetime(df[date_candidates[0]], errors="coerce")
    # If still none, try to parse first column
    if not any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns):
        try:
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0], errors="coerce")
        except Exception:
            pass

    colmap = infer_columns(df)
    # Ensure canonical 'date' mapping exists: choose existing date-like column
    if "date" not in colmap:
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                colmap["date"] = c
                break
    # If still missing, raise an informative error
    if "date" not in colmap:
        raise ValueError("Couldn't find a date column. Please ensure your CSV has a date column.")
    # Standardize date column to datetime
    df[colmap["date"]] = pd.to_datetime(df[colmap["date"]], errors="coerce")
    if df[colmap["date"]].isna().any():
        print("Warning: some dates failed to parse to datetime; check CSV formatting.")

    return df, colmap

def build_team_game_table(games: pd.DataFrame, colmap: dict) -> pd.DataFrame:
    """
    Build a per-team per-game table (one row per team per game) useful
    to compute rolling per-team stats.
    """
    date_col = colmap["date"]
    rows = []
    for idx, r in games.iterrows():
        home = {
            "game_id": idx,
            "date": r[date_col],
            "team": r[colmap["home_team"]],
            "is_home": 1,
            "opp": r[colmap["away_team"]],
            "team_pts": r[colmap["home_pts"]],
            "opp_pts": r[colmap["away_pts"]],
        }
        away = {
            "game_id": idx,
            "date": r[date_col],
            "team": r[colmap["away_team"]],
            "is_home": 0,
            "opp": r[colmap["home_team"]],
            "team_pts": r[colmap["away_pts"]],
            "opp_pts": r[colmap["home_pts"]],
        }
        rows.append(home)
        rows.append(away)
    team_games = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
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
    """
    g = games.copy().sort_values(colmap["date"]).reset_index(drop=True)
    # ensure score columns exist
    if colmap.get("home_pts") not in g.columns or colmap.get("away_pts") not in g.columns:
        raise ValueError("Couldn't find home/away score columns in CSV. Found mapping: {}".format(colmap))
    # build team-level table
    team_games = build_team_game_table(g, colmap)
    # compute per-team rolling stats
    rolling_stats = []
    for w in windows:
        # margin rolling mean
        team_games[f"margin_roll_{w}"] = team_games.groupby("team")["margin"].shift(1).rolling(window=w, min_periods=1).mean()
        team_games[f"winrate_roll_{w}"] = team_games.groupby("team")["win"].shift(1).rolling(window=w, min_periods=1).mean()
        # scoring rate
        team_games[f"pts_roll_{w}"] = team_games.groupby("team")["team_pts"].shift(1).rolling(window=w, min_periods=1).mean()
        team_games[f"opp_pts_roll_{w}"] = team_games.groupby("team")["opp_pts"].shift(1).rolling(window=w, min_periods=1).mean()

    # merge rolling features back into original games for home and away
    # prepare a lookup table: keys (date, team) -> features (we use game_id for uniqueness)
    feat_cols = [c for c in team_games.columns if c.startswith("margin_roll_") or c.startswith("winrate_roll_") or c.startswith("pts_roll_") or c.startswith("opp_pts_roll_")]
    team_feat = team_games[["game_id","date","team"] + feat_cols].copy()
    # For merging we need the mapping from game index to game_id used above (we used idx before reset)
    # In build_team_game_table we used the original games' index as game_id (that's fine)
    # Build home features:
    home_feat = team_feat.rename(columns={"team":"home_team"})
    away_feat = team_feat.rename(columns={"team":"away_team"})
    # merge on date & team by left join (use game index mapping game_id to game index)
    # first add the original index as a column to g to map to game_id
    g = g.reset_index().rename(columns={"index":"orig_index"})
    # team_games has game_id equal to original index; so we can merge on game_id
    # split team_games into home and away by selecting rows where is_home==1/0
    # But simpler: create dictionaries keyed by (orig_index, team) to features
    tg = team_games.copy()
    tg_lookup = tg.set_index(["game_id","team"])[feat_cols].to_dict(orient="index")

    # helper to fetch features
    def _get_team_feats(row, team_col):
        key = (row.name, row[team_col])  # row.name is original index in g because of how build_team_game_table used idx
        # if missing, fallback by matching on date & team (slower but safe)
        if key in tg_lookup:
            return tg_lookup[key]
        # fallback:
        match = tg[(tg["date"]==row[colmap["date"]]) & (tg["team"]==row[team_col])]
        if not match.empty:
            return match.iloc[0][feat_cols].to_dict()
        # final fallback: return NaNs
        return {c: np.nan for c in feat_cols}

    # Now for each game in g, attach home/away rolling features
    home_feat_rows = []
    away_feat_rows = []
    for i, row in g.iterrows():
        hf = _get_team_feats(row, colmap["home_team"])
        af = _get_team_feats(row, colmap["away_team"])
        home_feat_rows.append({f"home_{k}": v for k,v in hf.items()})
        away_feat_rows.append({f"away_{k}": v for k,v in af.items()})

    home_df = pd.DataFrame(home_feat_rows, index=g.index)
    away_df = pd.DataFrame(away_feat_rows, index=g.index)
    out = pd.concat([g, home_df, away_df], axis=1)

    # create derived features
    for w in windows:
        out[f"home_minus_away_margin_roll_{w}"] = out[f"home_margin_roll_{w}"] - out[f"away_margin_roll_{w}"]
        out[f"home_minus_away_winrate_{w}"] = out[f"home_winrate_roll_{w}"] - out[f"away_winrate_roll_{w}"]
        out[f"home_minus_away_pts_{w}"] = out[f"home_pts_roll_{w}"] - out[f"away_pts_roll_{w}"]

    # rest days: compute days since previous game per team
    out[colmap["date"]] = pd.to_datetime(out[colmap["date"]])
    # compute per-team last game date
    last_game_date = {}
    home_rest = []
    away_rest = []
    for i, row in out.iterrows():
        ht = row[colmap["home_team"]]
        at = row[colmap["away_team"]]
        d = row[colmap["date"]]
        # home rest
        if ht in last_game_date:
            delta = (d - last_game_date[ht]).days
            home_rest.append(delta)
        else:
            home_rest.append(np.nan)
        # away rest
        if at in last_game_date:
            delta = (d - last_game_date[at]).days
            away_rest.append(delta)
        else:
            away_rest.append(np.nan)
        # update last_game_date for both teams to this date
        last_game_date[ht] = d
        last_game_date[at] = d
    out["home_rest"] = home_rest
    out["away_rest"] = away_rest
    out["rest_diff"] = out["home_rest"] - out["away_rest"]

    # target: margin = home_pts - away_pts
    out["margin"] = out[colmap["home_pts"]] - out[colmap["away_pts"]]

    # filter early-season rows where not enough prior games if desired
    # here we keep rows but user may drop NaNs later
    return out

def get_feature_list(windows=(5,10)):
    feats = []
    for w in windows:
        feats += [
            f"home_minus_away_margin_roll_{w}",
            f"home_minus_away_winrate_{w}",
            f"home_minus_away_pts_{w}",
        ]
    feats += ["rest_diff", "home_rest", "away_rest"]
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
    print(feats.head()[list(feats.columns[:40])])
