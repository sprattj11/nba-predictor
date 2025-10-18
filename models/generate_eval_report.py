#!/usr/bin/env python3
"""
Generate eval_report.csv by predicting a holdout set with an existing model.
Usage:
  python models/generate_eval_report.py --games data/games_summary_merged.csv --model models/spread_model.pkl --out eval_report.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Minimal feature builder that mirrors your training features.
# Adjust if your training used a more complex pipeline.
def build_basic_features(df):
    # expecting columns: date, home_team, away_team, home_pts, away_pts
    df = df.sort_values("date").reset_index(drop=True)
    df["margin"] = df["home_pts"] - df["away_pts"]
    # simple rolling last-5 margin for home and away
    # build team-game table
    rows = []
    for idx, r in df.iterrows():
        rows.append({"game_idx": idx, "date": r["date"], "team": r["home_team"], "is_home":1, "team_pts":r["home_pts"], "opp_pts":r["away_pts"]})
        rows.append({"game_idx": idx, "date": r["date"], "team": r["away_team"], "is_home":0, "team_pts":r["away_pts"], "opp_pts":r["home_pts"]})
    tg = pd.DataFrame(rows).sort_values("date")
    tg["margin"] = tg["team_pts"] - tg["opp_pts"]
    tg["win"] = (tg["margin"]>0).astype(int)
    tg = tg.reset_index(drop=True)
    # compute last-5 rolling mean per team
    for w in (5,10):
        tg[f"margin_roll_{w}"] = tg.groupby("team")["margin"].shift(1).rolling(window=w, min_periods=1).mean()
        tg[f"pts_roll_{w}"] = tg.groupby("team")["team_pts"].shift(1).rolling(window=w, min_periods=1).mean()
    # pivot back to games
    feat_cols = ["margin_roll_5","margin_roll_10","pts_roll_5","pts_roll_10"]
    # create dict (game_idx, team) -> feats
    feat_map = {}
    for _, r in tg.iterrows():
        feat_map[(r["game_idx"], r["team"])] = {c: r.get(c, np.nan) for c in feat_cols}
    # attach to game-level
    recs = []
    for idx, r in df.iterrows():
        home_feats = feat_map.get((idx, r["home_team"]), {c:np.nan for c in feat_cols})
        away_feats = feat_map.get((idx, r["away_team"]), {c:np.nan for c in feat_cols})
        rec = {"date": r["date"], "home_team": r["home_team"], "away_team": r["away_team"],
               "home_pts": r["home_pts"], "away_pts": r["away_pts"], "margin": r["home_pts"]-r["away_pts"]}
        # derived features similar to your pipeline
        rec["home_minus_away_margin_roll_5"] = home_feats["margin_roll_5"] - away_feats["margin_roll_5"]
        rec["home_minus_away_margin_roll_10"] = home_feats["margin_roll_10"] - away_feats["margin_roll_10"]
        rec["home_minus_away_pts_roll_5"] = home_feats["pts_roll_5"] - away_feats["pts_roll_5"]
        rec["home_minus_away_pts_roll_10"] = home_feats["pts_roll_10"] - away_feats["pts_roll_10"]
        recs.append(rec)
    feats_df = pd.DataFrame(recs)
    return feats_df

def time_holdout_split(df):
    # hold out the last season if season col present, else last 20% by date
    if "season" in df.columns and df["season"].notna().any():
        last_season = sorted(df["season"].unique())[-1]
        train = df[df["season"] < last_season]
        test = df[df["season"] == last_season]
        if test.empty:
            cutoff = df["date"].quantile(0.8)
            train = df[df["date"] <= cutoff]
            test = df[df["date"] > cutoff]
    else:
        cutoff = df["date"].quantile(0.8)
        train = df[df["date"] <= cutoff]
        test = df[df["date"] > cutoff]
    return train, test

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="eval_report.csv")
    args = p.parse_args()

    games_path = Path(args.games)
    model_path = Path(args.model)
    out_path = Path(args.out)

    df = pd.read_csv(games_path, low_memory=False)
    # normalize date and column names
    if "date" not in df.columns:
        dcands = [c for c in df.columns if "date" in c.lower()]
        if dcands:
            df = df.rename(columns={dcands[0]:"date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # ensure score columns exist
    if "home_pts" not in df.columns or "away_pts" not in df.columns:
        # try alternatives
        for c in df.columns:
            low = c.lower()
            if "home" in low and ("pts" in low or "score" in low or "points" in low):
                df = df.rename(columns={c:"home_pts"})
            if ("away" in low or "visitor" in low) and ("pts" in low or "score" in low or "points" in low):
                df = df.rename(columns={c:"away_pts"})
    # build features
    feats = build_basic_features(df)
    # do time split
    train, test = time_holdout_split(df)
    # choose test indices
    test_idx = test.index
    # load model
    model = joblib.load(model_path)
    # align features: model may expect different columns; try to pick intersection
    X_test = feats.loc[test_idx].copy()
    # try direct prediction; if it fails, attempt to align to model.feature_name()
    try:
        preds = model.predict(X_test)
    except Exception:
        try:
            fns = model.feature_name()
            for fn in fns:
                if fn not in X_test.columns:
                    X_test[fn] = np.nan
            X_test = X_test[fns]
            preds = model.predict(X_test)
        except Exception as e:
            raise RuntimeError("Model prediction failed; feature mismatch.") from e

    out = test.reset_index(drop=True).loc[:, ["date","home_team","away_team","home_pts","away_pts"]].copy()
    out["margin"] = out["home_pts"] - out["away_pts"]
    out["pred_margin"] = preds
    out.to_csv(out_path, index=False)
    print("Wrote eval report to", out_path)

if __name__ == "__main__":
    main()
