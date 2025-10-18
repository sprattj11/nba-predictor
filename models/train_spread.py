"""
train_spread.py

Train a LightGBM regression to predict game margin (home - away) from games_details.csv.
Saves model artifact and evaluation CSV.

Usage:
python train_spread.py --csv path/to/games_details.csv --model-out models/spread_model.pkl --report-out models/eval_report.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

from features_and_prep import load_games, add_rolling_features, get_feature_list

def prepare_data(csv_path):
    df, colmap = load_games(csv_path)
    df_feats = add_rolling_features(df, colmap, windows=(5,10))
    # drop rows missing target or core features
    # identify feature list
    features = get_feature_list(windows=(5,10))
    # add any extra features if present (elo)
    if "home_elo" in colmap and "away_elo" in colmap:
        df_feats["elo_diff"] = df_feats[colmap["home_elo"]] - df_feats[colmap["away_elo"]]
        features.append("elo_diff")
    # add bookmaker_line if present
    if colmap.get("book_line"):
        # align book_line name
        df_feats["book_line"] = df_feats[colmap["book_line"]]
    else:
        df_feats["book_line"] = np.nan

    # drop extreme missing rows
    required_cols = ["margin"] + features
    df_feats = df_feats.dropna(subset=required_cols, how="any")
    return df_feats, features, colmap

def time_split_train_test(df, date_col, holdout_strategy="season_last", holdout_seasons=1):
    """
    Provide a simple time-based train/test split:
    - If 'season_last': hold out the last season in the file
    - Alternatively, can hold out most recent N days by specifying threshold date
    """
    if "season" in df.columns:
        # if season exists (the original CSV may have it), use that; else fallback
        if df["season"].notna().any():
            # pick most recent season value(s)
            seasons = sorted(df["season"].unique())
            last_season = seasons[-1]
            train = df[df["season"] < last_season].copy()
            test = df[df["season"] == last_season].copy()
            if train.empty or test.empty:
                # fallback to date split (80/20)
                cutoff = df["date"].quantile(0.8)
                train = df[df["date"] <= cutoff].copy()
                test = df[df["date"] > cutoff].copy()
        else:
            cutoff = df["date"].quantile(0.8)
            train = df[df["date"] <= cutoff].copy()
            test = df[df["date"] > cutoff].copy()
    else:
        cutoff = df["date"].quantile(0.8)
        train = df[df["date"] <= cutoff].copy()
        test = df[df["date"] > cutoff].copy()
    return train, test

def train_and_evaluate(df, features, colmap, model_out, report_out, random_seed=42):
    # train/test split
    train, test = time_split_train_test(df, date_col=colmap["date"])
    X_train = train[features]
    y_train = train["margin"]
    X_test = test[features]
    y_test = test["margin"]

    # LightGBM dataset
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "l1",       # MAE
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": random_seed,
    }

    print("Training LightGBM on {} rows, validating on {} rows".format(len(X_train), len(X_test)))
        # ---------- Train (backwards-compatible for different LightGBM versions) ----------
    try:
        # Newer API — accepts early_stopping_rounds & verbose_eval directly
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            num_boost_round=2000,
            early_stopping_rounds=100,
            verbose_eval=100,
        )
    except TypeError:
        # Fallback for LightGBM builds that require callbacks instead
        callbacks = [
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100),
        ]
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            num_boost_round=2000,
            callbacks=callbacks,
        )


    # predictions
    test["pred_margin"] = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, test["pred_margin"])
    rmse = np.sqrt(mean_squared_error(y_test, test["pred_margin"]))   # compatibility fix
    print(f"HOLDOUT MAE: {mae:.4f}, RMSE: {rmse:.4f}")


    # simple betting backtest vs book_line (if present)
    if "book_line" in test.columns and test["book_line"].notna().any():
        # assume book_line is "home spread" (positive means home favored by that many)
        test["edge"] = test["pred_margin"] - test["book_line"]
        # backtest thresholded flat bets with -110 odds
        stake = 100.0
        decimal_odds = 1.909  # -110
        bets = []
        for _, r in test.iterrows():
            e = r["edge"]
            # threshold: abs(edge) >= 2 (configurable)
            if abs(e) >= 2.0:
                pick_home = e > 0
                # did home cover? home_margin > book_line
                home_cover = (r["margin"] > r["book_line"])
                won = home_cover if pick_home else (not home_cover)
                if won:
                    profit = stake * (decimal_odds - 1)
                else:
                    profit = -stake
                bets.append(profit)
        total_profit = sum(bets) if bets else 0.0
        num_bets = len(bets)
        roi = total_profit / (stake * num_bets) if num_bets else 0.0
    else:
        total_profit = np.nan
        num_bets = 0
        roi = np.nan

    # save model
    joblib.dump(model, model_out)
    print(f"Saved model to {model_out}")

    # Save evaluation report (test rows with preds and edge)
    report_cols = ["date", colmap.get("home_team"), colmap.get("away_team"), "margin", "pred_margin", "book_line", "edge"] if colmap.get("home_team") else ["date","margin","pred_margin","book_line","edge"]
    # ensure report columns exist
    for c in report_cols:
        if c not in test.columns:
            test[c] = test.get(c, np.nan)
    test[report_cols].to_csv(report_out, index=False)
    print(f"Saved evaluation report to {report_out}")

    summary = {
        "mae": mae,
        "rmse": rmse,
        "num_test_games": len(test),
        "num_train_games": len(train),
        "num_bets": num_bets,
        "total_profit": float(total_profit),
        "roi": float(roi),
        "model_path": model_out,
        "report_path": report_out
    }
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to games_details.csv")
    p.add_argument("--model-out", default="spread_model.pkl", help="Path to save trained model (joblib .pkl)")
    p.add_argument("--report-out", default="eval_report.csv", help="Path to save CSV with test predictions + evaluation")
    args = p.parse_args()

    df, features, colmap = prepare_data(args.csv)
    summary = train_and_evaluate(df, features, colmap, args.model_out, args.report_out)
    print("Training summary:")
    for k,v in summary.items():
        print(f" - {k}: {v}")

if __name__ == "__main__":
    main()
