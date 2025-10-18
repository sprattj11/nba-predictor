#!/usr/bin/env python3
"""
Compare one-or-more saved models quickly on a test games file.

Usage:
  # compare single model on your eval_report (already contains pred_margin from earlier model)
  python models/compare_models.py --test eval_report.csv --model models/spread_model.pkl --name baseline

  # compare two models
  python models/compare_models.py --test data/games_summary_merged.csv \
    --model models/spread_model.pkl --name baseline \
    --model models/market_spread_model.pkl --name market_resid

Notes:
 - test file must contain: date, home_team, away_team, home_pts, away_pts (or margin)
 - if test file also has market fields (spread, whos_favored or book_line), cover metrics will be computed
 - models should accept the same feature columns the script builds (home_minus_away_margin_roll_5, ...).
 - Output: printed metrics + appended row to metrics_history.csv
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# ---- Feature builders (same logic as your interactive/ training scripts) ----
def compute_team_rolls(games_df, date_col="date", windows=(5,10)):
    rows = []
    for idx, r in games_df.iterrows():
        rows.append({
            "game_idx": idx,
            "date": pd.to_datetime(r[date_col]),
            "team": r["home_team"],
            "is_home": 1,
            "opp": r["away_team"],
            "team_pts": r["home_pts"],
            "opp_pts": r["away_pts"],
        })
        rows.append({
            "game_idx": idx,
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

def build_game_features(games_df, windows=(5,10)):
    # Build per-game features: home_minus_away_{...}
    games_df = games_df.reset_index(drop=True).copy()
    games_df["date"] = pd.to_datetime(games_df["date"], errors="coerce")
    team_games = compute_team_rolls(games_df, date_col="date", windows=windows)

    feat_rows = []
    feat_cols = []
    for w in windows:
        feat_cols += [f"home_minus_away_margin_roll_{w}", f"home_minus_away_winrate_{w}",
                      f"home_minus_away_pts_{w}", f"home_minus_away_opp_pts_{w}"]
    # compute for each game
    for idx, r in games_df.iterrows():
        as_of = pd.to_datetime(r["date"])
        home = r["home_team"]
        away = r["away_team"]
        # pick last rows for each team
        h_df = team_games[(team_games["team"]==home) & (team_games["date"] < as_of)].sort_values("date")
        a_df = team_games[(team_games["team"]==away) & (team_games["date"] < as_of)].sort_values("date")
        feats = {"date": as_of, "home_team": home, "away_team": away, "index": idx}
        if h_df.empty or a_df.empty:
            # missing history -> fill nans
            for c in feat_cols:
                feats[c] = np.nan
        else:
            h_last = h_df.iloc[-1]
            a_last = a_df.iloc[-1]
            for w in windows:
                feats[f"home_minus_away_margin_roll_{w}"] = h_last.get(f"margin_roll_{w}", np.nan) - a_last.get(f"margin_roll_{w}", np.nan)
                feats[f"home_minus_away_winrate_{w}"] = h_last.get(f"winrate_roll_{w}", np.nan) - a_last.get(f"winrate_roll_{w}", np.nan)
                feats[f"home_minus_away_pts_{w}"] = h_last.get(f"pts_roll_{w}", np.nan) - a_last.get(f"pts_roll_{w}", np.nan)
                feats[f"home_minus_away_opp_pts_{w}"] = h_last.get(f"opp_pts_roll_{w}", np.nan) - a_last.get(f"opp_pts_roll_{w}", np.nan)
        feat_rows.append(feats)
    feats_df = pd.DataFrame(feat_rows).set_index("index")
    # join with original scores if present
    out = pd.concat([games_df, feats_df], axis=1)
    return out

# ---- evaluation metrics ----
def compute_metrics(df, pred_col="pred_margin", label_mode="win", market_col=None):
    out = {}
    # ensure basic columns
    if "margin" not in df.columns and ("home_pts" in df.columns and "away_pts" in df.columns):
        df["margin"] = df["home_pts"] - df["away_pts"]
    if label_mode == "win":
        df["true"] = (df["margin"] > 0).astype(int)
        df["pred"] = (df[pred_col] > 0).astype(int)
    else:
        # cover
        if market_col is None:
            raise ValueError("market_col required for cover metrics")
        df["true"] = (df["margin"] > df[market_col]).astype(int)
        df["pred"] = (df[pred_col] > df[market_col]).astype(int)
    # basic metrics
    out["accuracy"] = accuracy_score(df["true"], df["pred"])
    out["precision"] = precision_score(df["true"], df["pred"], zero_division=0)
    out["recall"] = recall_score(df["true"], df["pred"], zero_division=0)
    try:
        out["auc"] = roc_auc_score(df["true"], df[pred_col])  # use raw pred margin as score
    except Exception:
        out["auc"] = None
    out["confusion"] = confusion_matrix(df["true"], df["pred"]).tolist()
    return out

# ---- main CLI ----
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True, help="Test/eval CSV (date,home_team,away_team,home_pts,away_pts or margin)")
    p.add_argument("--model", action="append", help="Model path (can be specified multiple times)", required=True)
    p.add_argument("--name", action="append", help="Name for corresponding model (append in same order)")
    p.add_argument("--out", default="metrics_history.csv", help="CSV to append results to")
    p.add_argument("--market-betting", help="Optional betting CSV (for market cover metrics)")
    p.add_argument("--windows", default="5,10", help="Rolling windows (comma sep)")
    args = p.parse_args()

    test_path = Path(args.test)
    if not test_path.exists():
        print("Test file not found:", test_path); sys.exit(2)
    df_test = pd.read_csv(test_path, low_memory=False)
    if "date" not in df_test.columns:
        date_cands = [c for c in df_test.columns if "date" in c.lower()]
        if date_cands:
            df_test = df_test.rename(columns={date_cands[0]: "date"})
    # ensure home/away columns
    if "home_team" not in df_test.columns or "away_team" not in df_test.columns:
        print("Test file must contain home_team and away_team columns."); sys.exit(2)
    # ensure scores
    if "home_pts" not in df_test.columns or "away_pts" not in df_test.columns:
        print("Test file missing home_pts/away_pts; cover metrics won't be available but we can still predict."); 
    df_test["date"] = pd.to_datetime(df_test["date"], errors="coerce")

    windows = tuple(int(x) for x in args.windows.split(","))
    feats_df = build_game_features(df_test, windows=windows)

    # load betting (optional)
    market_col = None
    betting_df = None
    if args.market_betting:
        bpath = Path(args.market_betting)
        if bpath.exists():
            betting_df = pd.read_csv(bpath, low_memory=False)
            # normalize name columns
            bet_date_col = next((c for c in betting_df.columns if "date" in c.lower()), None)
            betting_df[bet_date_col] = pd.to_datetime(betting_df[bet_date_col], errors="coerce")
            betting_df["home_abbrev"] = betting_df.get("home", betting_df.get("home_team", betting_df.get("HOME"))).astype(str).str.lower()
            betting_df["away_abbrev"] = betting_df.get("away", betting_df.get("away_team", betting_df.get("AWAY"))).astype(str).str.lower()
            # --- robustly create home/away abbrev columns (avoid pandas reindex alignment issues) ---
            def _col_to_safe_series(df, colname):
                """Return a 1-D iterable of values for column `colname` whether it's stored as
                a Series or a single-column DataFrame. Always returns a numpy array."""
                col = df[colname]
                # if someone accidentally stored a single-column DataFrame, pick the first column
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                # convert to string and lower; return raw numpy values to avoid index-alignment on setitem
                return col.astype(str).str.lower().to_numpy()

            # create columns
            feats_df["home_abbrev"] = _col_to_safe_series(feats_df, "home_team")
            feats_df["away_abbrev"] = _col_to_safe_series(feats_df, "away_team")
            # --------------------------------------------------------------------------------------

            merged = feats_df.merge(betting_df, left_on=["date","home_abbrev","away_abbrev"],
            right_on=[bet_date_col,"home_abbrev","away_abbrev"], how="left", suffixes=("","_bet"))
            # create market_expected_margin
            if "spread" in merged.columns and "whos_favored" in merged.columns:
                def mk(row):
                    s = row["spread"]
                    fav = str(row.get("whos_favored","")).lower()
                    return abs(s) if fav=="home" else -abs(s)
                merged["market_expected_margin"] = merged.apply(mk, axis=1)
                feats_df = merged
                market_col = "market_expected_margin"

    # for each model, predict & evaluate
    if not args.name:
        names = [Path(m).stem for m in args.model]
    else:
        names = args.name
    results = []
    for mpath, mname in zip(args.model, names):
        print(f"\n--- Evaluating model: {mname} ({mpath}) ---")
        mfile = Path(mpath)
        if not mfile.exists():
            print("Model file not found:", mfile); continue
        model = joblib.load(mfile)
        X = feats_df.copy()
        # try to predict directly, else align with model features
        try:
            preds = model.predict(X)
        except Exception:
            try:
                fns = model.feature_name()
                for fn in fns:
                    if fn not in X.columns:
                        X[fn] = np.nan
                X = X[fns]
                preds = model.predict(X)
            except Exception as e:
                print("Prediction failed for model:", e)
                continue
        feats_df[f"pred_margin_{mname}"] = preds
        # compute metrics: win prediction
        tmp = feats_df.copy()
        tmp["pred_margin"] = tmp[f"pred_margin_{mname}"]
        win_metrics = compute_metrics(tmp, pred_col="pred_margin", label_mode="win")
        print("Win metrics:", win_metrics)
        cover_metrics = None
        if market_col is not None:
            cover_metrics = compute_metrics(tmp, pred_col="pred_margin", label_mode="cover", market_col=market_col)
            print("Cover metrics:", cover_metrics)
        # append to results
        results.append({"time": time.time(), "model": mname,
                        "win_acc": win_metrics["accuracy"], "win_prec": win_metrics["precision"], "win_rec": win_metrics["recall"], "win_auc": win_metrics["auc"],
                        "cover_acc": cover_metrics["accuracy"] if cover_metrics else None,
                        "cover_prec": cover_metrics["precision"] if cover_metrics else None,
                        "cover_rec": cover_metrics["recall"] if cover_metrics else None,
                        "cover_auc": cover_metrics["auc"] if cover_metrics else None})
    # write metrics_history.csv
    outp = Path(args.out)
    rows = []
    for r in results:
        rows.append(r)
    if rows:
        dfh = pd.DataFrame(rows)
        header = not outp.exists()
        dfh.to_csv(outp, mode="a", index=False, header=header)
        print("\nAppended summary to", outp)
    else:
        print("No results to write.")

if __name__ == "__main__":
    main()
