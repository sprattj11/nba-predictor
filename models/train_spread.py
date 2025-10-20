#!/usr/bin/env python3
"""
train_spread.py

Train a margin (home - away) regression model using features from features_and_prep.py.

Outputs:
- models/spread_model.pkl
- eval/predictions.csv (predicted margins for test set)
- prints MAE / RMSE and ATS accuracy (if betting CSV with spreads is provided)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from features_and_prep import load_games, add_rolling_features, get_feature_list

# ATS helper (same logic we used before)
def compute_ats_results(df, spread_col="spread", whos_favored_col="whos_favored",
                        home_pts_col="home_pts", away_pts_col="away_pts"):
    """Compute signed spread and ats_winner on a dataframe that contains scores."""
    df = df.copy()
    df["actual_margin"] = df[home_pts_col] - df[away_pts_col]
    df["signed_spread"] = np.where(df[whos_favored_col].str.lower()=="home", -df[spread_col], df[spread_col])
    df["ats_margin"] = df["actual_margin"] + df["signed_spread"]
    df["ats_winner"] = np.where(df["ats_margin"]>0, "home", np.where(df["ats_margin"]<0, "away", "push"))
    return df

def ats_accuracy_from_predictions(merged, pred_col="pred_margin", signed_spread_col="signed_spread"):
    """Given merged predictions and a signed spread, compute model ATS picks and accuracy."""
    df = merged.copy()
    # model_ats_margin = pred_margin + signed_spread
    df["model_ats_margin"] = df[pred_col] + df[signed_spread_col]
    df["model_pick"] = np.where(df["model_ats_margin"]>0, "home", np.where(df["model_ats_margin"]<0, "away", "push"))
    df["correct_pick"] = df["model_pick"] == df["ats_winner"]
    incl = df["correct_pick"].mean() * 100
    excl = df[df["ats_winner"]!="push"]["correct_pick"].mean() * 100
    pushes = (df["ats_winner"]=="push").sum()
    return incl, excl, pushes

def main(args):
    # 1) Load games and build features
    print("Loading games:", args.games_csv)
    games_df, colmap = load_games(args.games_csv)
    print("Inferred columns:", colmap)
    feats_df = add_rolling_features(games_df, colmap, windows=tuple(args.windows), min_games_required=args.min_games)
    print("Built features. Rows:", len(feats_df))

    # 2) Choose feature columns
    base_feats = get_feature_list(windows=tuple(args.windows))
    # also include rest & b2b features added in the patch
    additional = ["home_rest", "away_rest", "rest_diff", "is_b2b_home", "is_b2b_away"]
    feature_columns = [f for f in base_feats + additional if f in feats_df.columns]
    print("Using features:", feature_columns)

    # 3) Drop rows missing target or features
    # defensive check before dropping NAs
    # --- defensive patch start ---
    # Ensure the required identifying columns exist in feats_df before dropping NAs.
    # Use colmap to determine original names, and copy from games_df if necessary.

    # names to look up (colmap maps logical -> actual CSV names)
    date_col = colmap.get("date", "date")
    home_col = colmap.get("home_team", "home_team")
    away_col = colmap.get("away_team", "away_team")
    home_pts_col = colmap.get("home_pts", "home_pts")
    away_pts_col = colmap.get("away_pts", "away_pts")
    margin_col = "margin"

    required_cols = [date_col, home_col, away_col, home_pts_col, away_pts_col, margin_col]

    # find which required columns are missing from feats_df
    missing = [c for c in required_cols if c not in feats_df.columns]

    if missing:
        # attempt to copy missing columns from the original games_df by index alignment
        # games_df is the pre-feature dataframe loaded earlier in this script
        print("Some required columns are missing from feats_df; attempting to copy from games_df:", missing)
        # If feats_df has a different index, try to reset and align by position.
        try:
            # best-effort: align by index - this works if add_rolling_features preserved row order/index
            for c in missing:
                if c in games_df.columns:
                    feats_df[c] = games_df[c].reindex(feats_df.index).values
                else:
                    # couldn't find the exact name in games_df; try common alternatives
                    # (for robustness -- not exhaustive)
                    alt_map = {
                        date_col: ["date", "game_date", "Date"],
                        home_col: ["home_team", "home", "home_abbrev"],
                        away_col: ["away_team", "away", "away_abbrev"],
                        home_pts_col: ["home_pts", "home_score", "home_points"],
                        away_pts_col: ["away_pts", "away_score", "away_points"],
                        margin_col: ["margin", "point_diff", "score_diff"]
                    }
                    found = False
                    for alt in alt_map.get(c, []):
                        if alt in games_df.columns:
                            feats_df[c] = games_df[alt].reindex(feats_df.index).values
                            found = True
                            break
                    if not found:
                        # fallback: create NaNs so dropna will still behave predictably
                        import numpy as _np
                        feats_df[c] = _np.nan
                        print(f"Warning: couldn't find column '{c}' or alternatives in games_df. Filled with NaN.")
        except Exception as e:
            # fallback: fail with a helpful message rather than a raw KeyError
            raise SystemExit("Failed while trying to copy missing columns from games_df into feats_df: " + str(e))

    # Re-evaluate missing columns
    still_missing = [c for c in required_cols if c not in feats_df.columns]
    if still_missing:
        raise SystemExit("Missing required columns in features dataframe after attempted copy: " + ", ".join(still_missing) +
                        ". Run the debug script to inspect colmap and feats_df.columns.")

    # Now it's safe to drop NA rows using the explicit required column names
    feats_df = feats_df.dropna(subset=required_cols)
    # --- defensive patch end ---


    # optionally drop rows where core features are NaN (early-season)
    train_df = feats_df.dropna(subset=feature_columns)

    # 4) Train/test split by date (train on everything before last season if seasons present)
    # Use index-aware selection and intersect with train_df.index to avoid KeyError when train_df is a subset.

    def _index_mask_to_indices(df_index, mask):
        """Return Index of df_index where mask is True (mask may be numpy array or Series)."""
        # ensure boolean numpy array aligned to df_index positions
        mvals = mask.values if hasattr(mask, "values") else mask
        return df_index[mvals]

    if "season" in colmap and colmap["season"] in games_df.columns:
        # use season if present: train on seasons < max, test on max season
        if colmap["season"] in feats_df.columns:
            last_season = feats_df[colmap["season"]].max()
            sel_idx = _index_mask_to_indices(feats_df.index, feats_df[colmap["season"]] < last_season)
            sel_idx = sel_idx.intersection(train_df.index)  # only keep indices present in train_df
            train = train_df.loc[sel_idx]
            test  = train_df.loc[train_df.index.difference(sel_idx)]
            if len(test) == 0:
                # fallback to date split: last 10% for test
                cutoff = feats_df[colmap["date"]].quantile(0.90)
                mask = feats_df[colmap["date"]] <= cutoff
                sel_idx = _index_mask_to_indices(feats_df.index, mask)
                sel_idx = sel_idx.intersection(train_df.index)
                train = train_df.loc[sel_idx]
                test  = train_df.loc[train_df.index.difference(sel_idx)]
        else:
            cutoff = feats_df[colmap["date"]].quantile(0.90)
            mask = feats_df[colmap["date"]] <= cutoff
            sel_idx = _index_mask_to_indices(feats_df.index, mask)
            sel_idx = sel_idx.intersection(train_df.index)
            train = train_df.loc[sel_idx]
            test  = train_df.loc[train_df.index.difference(sel_idx)]
    else:
        cutoff = feats_df[colmap["date"]].quantile(0.90)
        mask = feats_df[colmap["date"]] <= cutoff
        sel_idx = _index_mask_to_indices(feats_df.index, mask)
        sel_idx = sel_idx.intersection(train_df.index)
        train = train_df.loc[sel_idx]
        test  = train_df.loc[train_df.index.difference(sel_idx)]

    print("Train size:", len(train), "Test size:", len(test))



    # 5) Fit a model (RandomForest or Ridge as fallback)
    X_train = train[feature_columns]
    y_train = train["margin"]
    X_test = test[feature_columns]
    y_test = test["margin"]

    if args.model == "rf":
        model = RandomForestRegressor(n_estimators=args.n_estimators, n_jobs=args.n_jobs, random_state=42)
    else:
        model = Ridge(alpha=1.0)

    print("Training model:", model)
    model.fit(X_train, y_train)

    # 6) Predict and evaluate
    preds = model.predict(X_test)
    test = test.copy()
    test["pred_margin"] = preds

    mae = mean_absolute_error(y_test, preds)

    # portable RMSE computation to avoid sklearn kwarg compatibility issues
    import numpy as _np
    mse = mean_squared_error(y_test, preds)   # always supported
    rmse = float(_np.sqrt(mse))
    bias = (y_test - preds).mean()

    # optional extras: r2 (import earlier if you want)
    # from sklearn.metrics import r2_score
    # r2 = r2_score(y_test, preds)
    print(f"MAE: {mae:.4f}  RMSE: {rmse:.4f}  Bias(mean error): {bias:.4f}")


    # 7) Save model and predictions
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "spread_model.pkl")
    joblib.dump({"model": model, "features": feature_columns, "colmap": colmap}, model_path)
    print("Saved model to", model_path)

    os.makedirs("eval", exist_ok=True)
    pred_out = os.path.join("eval", "predictions.csv")
    test_out = test[[colmap["date"], colmap["home_team"], colmap["away_team"], colmap["home_pts"], colmap["away_pts"], "pred_margin"] + feature_columns]
    test_out.to_csv(pred_out, index=False)
    print("Wrote predictions to", pred_out)

    # 8) Optional: evaluate ATS accuracy if betting CSV provided and contains spread & whos_favored
    if args.betting_csv:
        print("Loading betting CSV for ATS evaluation:", args.betting_csv)
        bet = pd.read_csv(args.betting_csv, parse_dates=[colmap.get("date", "date")])
        # normalize keys for merging: lower-case team codes/names
        # The training outputs use the original team values; attempt to match on lowercase short codes
        pred_df = test_out.copy()
        pred_df = pred_df.rename(columns={
            colmap["date"]: "date",
            colmap["home_team"]: "home",
            colmap["away_team"]: "away",
            colmap["home_pts"]: "home_pts",
            colmap["away_pts"]: "away_pts"
        })
        # betting data often uses 'home'/'away' columns; try to standardize
        bet = bet.rename(columns={args.betting_home_col: "home", args.betting_away_col: "away",
                                  args.betting_home_pts_col: "home_pts", args.betting_away_pts_col: "away_pts"})
        # lowercase team codes for better matching
        pred_df["home_lower"] = pred_df["home"].astype(str).str.lower()
        pred_df["away_lower"] = pred_df["away"].astype(str).str.lower()
        bet["home_lower"] = bet["home"].astype(str).str.lower()
        bet["away_lower"] = bet["away"].astype(str).str.lower()
        bet["date"] = pd.to_datetime(bet["date"], errors="coerce")
        merged = pd.merge(pred_df, bet[["date","home_lower","away_lower","spread","whos_favored","home_pts","away_pts"]],
                          left_on=["date","home_lower","away_lower"], right_on=["date","home_lower","away_lower"], how="inner")

        # compute ats ground truth and signed_spread
        merged = compute_ats_results(merged, spread_col="spread", whos_favored_col="whos_favored",
                                     home_pts_col="home_pts", away_pts_col="away_pts")
        incl, excl, pushes = ats_accuracy_from_predictions(merged, pred_col="pred_margin", signed_spread_col="signed_spread")
        print(f"ATS accuracy (incl pushes): {incl:.2f}%  (excl pushes): {excl:.2f}%  pushes: {pushes}")

    print("Training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--games-csv", dest="games_csv", required=True, help="Path to games details CSV")
    p.add_argument("--betting-csv", dest="betting_csv", required=False, help="Optional: betting CSV for ATS eval")
    p.add_argument("--betting-home-col", dest="betting_home_col", default="home", help="col name for home team in betting CSV")
    p.add_argument("--betting-away-col", dest="betting_away_col", default="away", help="col name for away team in betting CSV")
    p.add_argument("--betting-home-pts-col", dest="betting_home_pts_col", default="score_home", help="home score col in betting CSV")
    p.add_argument("--betting-away-pts-col", dest="betting_away_pts_col", default="score_away", help="away score col in betting CSV")
    p.add_argument("--model-dir", dest="model_dir", default="models", help="Directory to save model")
    p.add_argument("--model", dest="model", choices=["rf","ridge"], default="rf", help="Model type")
    p.add_argument("--n-estimators", dest="n_estimators", type=int, default=200, help="RF n_estimators")
    p.add_argument("--n-jobs", dest="n_jobs", type=int, default=-1, help="n_jobs for RF")
    p.add_argument("--windows", dest="windows", nargs="+", type=int, default=[5,10], help="rolling windows to compute")
    p.add_argument("--min-games", dest="min_games", type=int, default=5, help="min games required for rolling windows")
    args = p.parse_args()
    main(args)
