#!/usr/bin/env python3
"""
scripts/predict_by_abbrev_with_nbaapi.py

Interactive CLI to predict an upcoming NBA game by team abbreviations and date.

Behavior:
- Loads historical games from data/nba_games.csv (project-root relative)
- Loads a trained model from models/xgb_nba_model.joblib (or other common locations)
- Builds the exact feature vector the model expects by computing season-to-date means
  and shifted rolling statistics (no leakage) for the requested date
- Accepts home/away team abbreviations (e.g., LAL, BOS) and a date (YYYY-MM-DD)
- Prints the constructed feature row and predicted probabilities for HOME and AWAY

Usage:
    source .venv/bin/activate
    python3 scripts/predict_by_abbrev_with_nbaapi.py
"""

from pathlib import Path
import sys
import joblib
import pandas as pd
import numpy as np
import re

# Project root (one level up from scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data & model paths relative to repo root
GAMES_CSV = PROJECT_ROOT / "data" / "nba_games.csv"
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "xgb_nba_model.joblib",
    PROJECT_ROOT / "models" / "xgb_nba_model.pkl",
    PROJECT_ROOT / "models" / "xgb_nba_model.joblib",
    PROJECT_ROOT / "xgb_nba_model.joblib",
    PROJECT_ROOT / "xgb_nba_model.pkl",
    PROJECT_ROOT / "nba_win_predictor.pkl",  # legacy fallback
]

# try to build abbreviation -> team_id map using nba_api if available, else fallback to data/team files
def build_abbrev_map_from_nba_api():
    try:
        from nba_api.stats.static import teams as nba_teams
    except Exception:
        return {}
    try:
        all_teams = nba_teams.get_teams()
    except Exception:
        return {}
    mapping = {}
    for t in all_teams:
        abb = str(t.get("abbreviation", "")).strip().upper()
        tid = t.get("id")
        if abb and tid is not None:
            mapping[abb] = int(tid)
    return mapping


def build_abbrev_map_from_data(df: pd.DataFrame):
    """
    Attempt to infer an abbreviation -> team_id map from CSVs in data/.
    Looks for 'teams.csv' in data/ that may contain columns like 'abbreviation' and 'team_id' or 'id'.
    """
    mapping = {}
    teams_csv = PROJECT_ROOT / "data" / "teams.csv"
    if teams_csv.exists():
        try:
            tdf = pd.read_csv(teams_csv)
            # common column names
            possible_abbrev_cols = [c for c in tdf.columns if "abbrev" in c.lower() or "code" in c.lower()]
            possible_id_cols = [c for c in tdf.columns if "id" in c.lower()]
            if possible_abbrev_cols and possible_id_cols:
                for _, r in tdf.iterrows():
                    abb = str(r[possible_abbrev_cols[0]]).strip().upper()
                    tid = int(r[possible_id_cols[0]])
                    mapping[abb] = tid
        except Exception:
            pass

    # fallback: if df has HOME_TEAM_ID and some name-like columns, try to map names -> ids
    name_cols = [c for c in df.columns if ("TEAM_NAME" in c.upper() or "TEAM" in c.upper()) and c.lower().endswith("name")]
    if "HOME_TEAM_ID" in df.columns and name_cols:
        try:
            pairs = df[[name_cols[0], "HOME_TEAM_ID"]].dropna().drop_duplicates()
            for name, tid in pairs.values:
                abb = str(name).strip().upper()[:3]
                mapping[abb] = int(tid)
        except Exception:
            pass

    return mapping


# pick first existing model file
MODEL_PATH = None
for p in MODEL_CANDIDATES:
    if p.exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    print(f"ERROR: Could not find a trained model. Looked for: {', '.join(str(p) for p in MODEL_CANDIDATES)}")
    sys.exit(1)

if not GAMES_CSV.exists():
    print(f"ERROR: Could not find {GAMES_CSV}. Put your dataset at data/nba_games.csv (relative to repo root).")
    sys.exit(1)

# Load historical dataframe
df = pd.read_csv(GAMES_CSV, parse_dates=["GAME_DATE_EST"])
df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

# Build abbreviation -> team_id mapping
abbrev_map = build_abbrev_map_from_nba_api()
if not abbrev_map:
    abbrev_map = build_abbrev_map_from_data(df)

# As last fallback: create mapping from unique HOME_TEAM_ID/ VISITOR_TEAM_ID with synthetic abbreviations
if not abbrev_map:
    ids = sorted(pd.concat([df.get("HOME_TEAM_ID", pd.Series(dtype=int)), df.get("VISITOR_TEAM_ID", pd.Series(dtype=int))]).dropna().unique().tolist())
    for i, tid in enumerate(ids):
        # synthetic abb like T01, T02... only used if user can't provide true abbrev
        abb = f"T{i+1:02d}"
        abbrev_map[abb] = int(tid)

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"ERROR loading model {MODEL_PATH}: {e}")
    sys.exit(1)

# Helper: determine model's expected feature names (ordered)
def get_model_feature_names(model):
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None:
        return list(fn)
    try:
        booster = model.get_booster()
        fns = getattr(booster, "feature_names", None)
        if fns:
            return list(fns)
    except Exception:
        pass
    # fallback: try some reasonable default (will likely fail later if mismatch)
    fallback = [
        "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
        "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",
        "home_PTS_last10", "home_FG_PCT_last10", "home_FT_PCT_last10", "home_FG3_PCT_last10",
        "home_AST_last10", "home_REB_last10",
        "away_PTS_last10", "away_FG_PCT_last10", "away_FT_PCT_last10", "away_FG3_PCT_last10",
        "away_AST_last10", "away_REB_last10",
        "home_win_pct_last10", "away_win_pct_last10"
    ]
    return fallback

# helpers to compute stats for feature construction
def mean_stat_before(df, team_id, colname, date, team_role):
    if df["GAME_DATE_EST"].dtype == object:
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"], errors="coerce")
    mask = df["GAME_DATE_EST"] < date
    if team_role == "home":
        if "HOME_TEAM_ID" in df.columns:
            mask = mask & (df["HOME_TEAM_ID"] == team_id)
    else:
        if "VISITOR_TEAM_ID" in df.columns:
            mask = mask & (df["VISITOR_TEAM_ID"] == team_id)
    if colname not in df.columns:
        return np.nan
    subset = df.loc[mask, colname]
    if subset.empty:
        return np.nan
    return float(subset.mean())

def rolling_mean_before(df, team_id, colname, date, team_role, k):
    if df["GAME_DATE_EST"].dtype == object:
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"], errors="coerce")
    mask = df["GAME_DATE_EST"] < date
    if team_role == "home":
        mask = mask & (df.get("HOME_TEAM_ID") == team_id)
    else:
        mask = mask & (df.get("VISITOR_TEAM_ID") == team_id)
    # if the column doesn't exist, try some sensible alternatives before erroring out
    if colname not in df.columns:
        # common fallback for win pct style if someone wrote home_win_pct_lastK -> underlying HOME_TEAM_WINS
        if colname.lower().startswith("win_pct") or "win" in colname.lower():
            # use HOME_TEAM_WINS/AWAY_TEAM_WINS instead if present/derivable
            if team_role == "home" and "HOME_TEAM_WINS" in df.columns:
                subset = df.loc[mask, ["GAME_DATE_EST", "HOME_TEAM_WINS"]].sort_values("GAME_DATE_EST")
            elif team_role == "away":
                if "AWAY_TEAM_WINS" not in df.columns and "HOME_TEAM_WINS" in df.columns:
                    df["AWAY_TEAM_WINS"] = 1 - df["HOME_TEAM_WINS"]
                subset = df.loc[mask, ["GAME_DATE_EST", "AWAY_TEAM_WINS"]].sort_values("GAME_DATE_EST")
            else:
                return np.nan
        else:
            return np.nan
    else:
        subset = df.loc[mask, ["GAME_DATE_EST", colname]].sort_values("GAME_DATE_EST")
    if subset.empty:
        return np.nan
    lastk = subset.iloc[:, 1].dropna().tail(k)
    if lastk.empty:
        return np.nan
    return float(lastk.mean())

def rolling_winpct_before(df, team_id, date, team_role, k):
    if df["GAME_DATE_EST"].dtype == object:
        df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"], errors="coerce")
    mask = df["GAME_DATE_EST"] < date
    if team_role == "home":
        mask = mask & (df.get("HOME_TEAM_ID") == team_id)
        colname = "HOME_TEAM_WINS"
    else:
        mask = mask & (df.get("VISITOR_TEAM_ID") == team_id)
        if "AWAY_TEAM_WINS" not in df.columns:
            if "HOME_TEAM_WINS" in df.columns:
                df["AWAY_TEAM_WINS"] = 1 - df["HOME_TEAM_WINS"]
            else:
                # cannot compute away wins without HOME_TEAM_WINS
                return np.nan
        colname = "AWAY_TEAM_WINS"
    subset = df.loc[mask, ["GAME_DATE_EST", colname]].sort_values("GAME_DATE_EST")
    if subset.empty:
        return np.nan
    lastk = subset[colname].dropna().tail(k)
    if lastk.empty:
        return np.nan
    return float(lastk.mean())

# Build feature row function that matches model's expected features
def build_feature_row_for_model(df, model, home_team_id, away_team_id, game_date_str):
    feature_names = get_model_feature_names(model)
    game_date = pd.to_datetime(game_date_str)
    row = {}
    # check winpct patterns BEFORE generic rolling patterns (fixes the bug you hit)
    re_home_winpct = re.compile(r"^home_win_pct_last(\d+)$")
    re_away_winpct = re.compile(r"^away_win_pct_last(\d+)$")
    re_home_rolling = re.compile(r"^home_([A-Za-z0-9_]+)_last(\d+)$")
    re_away_rolling = re.compile(r"^away_([A-Za-z0-9_]+)_last(\d+)$")

    for fname in feature_names:
        # First: direct boxscore columns like 'PTS_home' or 'FG3_PCT_away'
        if fname.endswith("_home") or fname.endswith("_away"):
            role = "home" if fname.endswith("_home") else "away"
            team_id = home_team_id if role == "home" else away_team_id
            val = mean_stat_before(df, team_id, fname, game_date, role)
            row[fname] = val
            continue

        # Next: explicit win_pct patterns
        m = re_home_winpct.match(fname)
        if m:
            k = int(m.group(1))
            val = rolling_winpct_before(df, home_team_id, game_date, "home", k)
            row[fname] = val
            continue

        m = re_away_winpct.match(fname)
        if m:
            k = int(m.group(1))
            val = rolling_winpct_before(df, away_team_id, game_date, "away", k)
            row[fname] = val
            continue

        # Generic rolling home_..._lastK
        m = re_home_rolling.match(fname)
        if m:
            stat = m.group(1)
            k = int(m.group(2))
            # try common underlying column patterns in order: <STAT>_home, STAT_home uppercase preserved, TRY variations
            candidates = [
                f"{stat}_home",
                f"{stat.upper()}_home",
                f"{stat.lower()}_home"
            ]
            val = np.nan
            for col in candidates:
                if col in df.columns:
                    val = rolling_mean_before(df, home_team_id, col, game_date, "home", k)
                    break
            # last-resort: if stat looks like 'win_pct' use rolling_winpct_before
            if np.isnan(val) and ("win" in stat.lower()):
                val = rolling_winpct_before(df, home_team_id, game_date, "home", k)
            row[fname] = val
            continue

        # Generic rolling away_..._lastK
        m = re_away_rolling.match(fname)
        if m:
            stat = m.group(1)
            k = int(m.group(2))
            candidates = [
                f"{stat}_away",
                f"{stat.upper()}_away",
                f"{stat.lower()}_away"
            ]
            val = np.nan
            for col in candidates:
                if col in df.columns:
                    val = rolling_mean_before(df, away_team_id, col, game_date, "away", k)
                    break
            if np.isnan(val) and ("win" in stat.lower()):
                val = rolling_winpct_before(df, away_team_id, game_date, "away", k)
            row[fname] = val
            continue

        # Fallback: overall mean of that column before the date, or 0.0
        if fname in df.columns:
            mask = df["GAME_DATE_EST"] < game_date
            overall = df.loc[mask, fname].dropna()
            row[fname] = float(overall.mean()) if not overall.empty else 0.0
        else:
            row[fname] = 0.0

    feat_df = pd.DataFrame([[row.get(f, 0.0) for f in feature_names]], columns=feature_names)
    # final safety: fill any remaining NaNs with column-wise means from historical data or 0
    for c in feat_df.columns:
        if pd.isna(feat_df.at[0, c]):
            if c in df.columns:
                mval = df.loc[df["GAME_DATE_EST"] < game_date, c].mean()
                feat_df.at[0, c] = float(mval) if not np.isnan(mval) else 0.0
            else:
                feat_df.at[0, c] = 0.0
    return feat_df

# Interactive CLI
def main():
    print("Detected team abbreviations (sample):", list(sorted(abbrev_map.keys()))[:20])
    home_abbrev = input("Home team abbreviation (e.g., LAL): ").strip().upper()
    away_abbrev = input("Away team abbreviation (e.g., BOS): ").strip().upper()
    game_date = input("Game date (YYYY-MM-DD): ").strip()

    if home_abbrev not in abbrev_map:
        print(f"Home abbreviation '{home_abbrev}' not found in mapping. Available examples: {list(abbrev_map.keys())[:30]}")
        return
    if away_abbrev not in abbrev_map:
        print(f"Away abbreviation '{away_abbrev}' not found in mapping. Available examples: {list(abbrev_map.keys())[:30]}")
        return

    home_team_id = abbrev_map[home_abbrev]
    away_team_id = abbrev_map[away_abbrev]

    feat_row = build_feature_row_for_model(df, model, home_team_id, away_team_id, game_date)
    print("\nConstructed feature row (model order):")
    pd.set_option("display.width", 200)
    print(feat_row.to_string(index=False))

    # apply scaler if available in models/
    scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
    X_for_pred = feat_row.values
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
            X_for_pred = scaler.transform(feat_row)
            print("Applied scaler from models/scaler.pkl")
        except Exception as e:
            print("Warning: failed to apply scaler:", e)
            X_for_pred = feat_row.values

    # predict
    if hasattr(model, "predict_proba"):
        proba_home = float(model.predict_proba(X_for_pred)[0][1])
        proba_away = 1.0 - proba_home
    else:
        pred = int(model.predict(X_for_pred)[0])
        proba_home = 1.0 if pred == 1 else 0.0
        proba_away = 1.0 - proba_home

    hard = "HOME" if proba_home >= 0.5 else "AWAY"
    print(f"\nPredicted winner (hard): {hard}")
    print(f"Probability HOME wins: {proba_home:.3f}")
    print(f"Probability AWAY wins: {proba_away:.3f}")


if __name__ == "__main__":
    main()
