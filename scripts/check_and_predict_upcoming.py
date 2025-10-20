#!/usr/bin/env python3
"""
check_and_predict_upcoming.py

Computes rolling features for the appended upcoming game and runs your saved
spread model on it.

Usage:
    python scripts/check_and_predict_upcoming.py
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ✅ Import from the models/ folder
from models.features_and_prep import load_games, add_rolling_features, get_feature_list

# --- CONFIG ---
GAMES_CSV = "data/games_summary_with_upcoming.csv"
MODEL_PATH = "models/spread_model.pkl"
HOME = "OKC"
AWAY = "HOU"
DATE = "2025-10-21"
MARKET_SPREAD = -7.5  # spread from home team's perspective (negative = favorite)
# ----------------

print(f"Reading raw CSV: {GAMES_CSV}")
raw = pd.read_csv(GAMES_CSV, low_memory=False, parse_dates=["date"])
print("Rows total:", len(raw))

# Show appended row (sanity check)
upcoming = raw[(raw["date"] == pd.to_datetime(DATE)) &
               (raw["home_team"].str.upper() == HOME.upper()) &
               (raw["away_team"].str.upper() == AWAY.upper())]
print("\nUpcoming rows for", DATE, ":", len(upcoming))
print(upcoming[["home_team", "away_team", "date", "home_days_rest", "away_days_rest"]].tail(5))

# 2️⃣ Build rolling features
print("\nBuilding rolling features (this may take a few seconds)...")
games_df, colmap = load_games(GAMES_CSV)
feats_df = add_rolling_features(games_df, colmap, windows=(5, 10), min_games_required=5)
print("Built features. Rows:", len(feats_df))

# 3️⃣ Locate the feature row for the upcoming game
date_col = colmap.get("date", "date")
home_col = colmap.get("home_team", "home_team")
away_col = colmap.get("away_team", "away_team")

mask = (
    feats_df[home_col].astype(str).str.upper() == HOME.upper()
) & (
    feats_df[away_col].astype(str).str.upper() == AWAY.upper()
) & (
    pd.to_datetime(feats_df[date_col]).dt.strftime("%Y-%m-%d") == DATE
)

row = feats_df[mask]
print("Matched feature rows:", len(row))
if len(row) == 0:
    print("⚠️  No feature row found — check team codes or date formatting.")
    raise SystemExit(1)

r = row.iloc[0]
non_null = r.dropna()
print("\nFeature row (non-null values):")
for k, v in non_null.items():
    print(f"{k}: {v}")

# 4️⃣ Load model
print("\nLoading model:", MODEL_PATH)
obj = joblib.load(MODEL_PATH)
model = obj["model"]
model_features = obj["features"]
print("Model expects features:", len(model_features))

missing_feats = [f for f in model_features if f not in feats_df.columns]
if missing_feats:
    print("\n⚠️  WARNING: model expects features not present in feats_df:")
    for f in missing_feats:
        print("  -", f)
else:
    X = row[model_features].to_frame().T.fillna(0)
    pred_margin = float(model.predict(X)[0])

    print("\n✅ Prediction results:")
    print(f"Predicted margin (home - away): {pred_margin:+.2f}")
    edge = pred_margin - MARKET_SPREAD
    print(f"Market spread (home): {MARKET_SPREAD:+.2f}")
    print(f"Combined edge (pred - market): {edge:+.2f}")

    if edge > 0:
        print("PICK: HOME to cover")
    elif edge < 0:
        print("PICK: AWAY to cover")
    else:
        print("PICK: No edge (≈50/50)")
