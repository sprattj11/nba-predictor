# train_xgb.py (robust version)
"""
Robust training script for XGBoost that:
 - auto-selects train/val/test seasons from your CSV
 - fills missing feature values with column means to avoid empty splits
 - trains XGBClassifier in a version-tolerant way
 - prints diagnostics and saves the model
"""

import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

# change this if your features file is elsewhere
from src.features import add_rolling_features

GAMES_CSV = "nba_games.csv"
MODEL_OUT = "xgb_nba_model.joblib"
RANDOM_STATE = 42

# ---- Load and prepare data ----
df = pd.read_csv(GAMES_CSV, parse_dates=["GAME_DATE_EST"])
df = add_rolling_features(df, window=10)

# show available seasons
if "SEASON" not in df.columns:
    print("ERROR: SEASON column not found in CSV. Add a SEASON column (e.g., 2022) and retry.")
    sys.exit(1)

seasons = sorted(df["SEASON"].dropna().unique().tolist())
print("Seasons found in data:", seasons)

if len(seasons) < 2:
    print("Not enough seasons to create train/val/test splits. Need >= 2 seasons. Found:", seasons)
    # fallback: we can still do a time-based split by date, but for clarity, stop here
    sys.exit(1)

# choose automatic split:
# - test = max season (most recent)
# - val = previous season
# - train = all seasons before val
test_season = seasons[-1]
val_season = seasons[-2]
train_seasons = [s for s in seasons if s < val_season]

print(f"Using train seasons: {train_seasons}, val season: {val_season}, test season: {test_season}")

train_df = df[df["SEASON"].isin(train_seasons)].copy()
val_df   = df[df["SEASON"] == val_season].copy()
test_df  = df[df["SEASON"] == test_season].copy()

# --- Feature list (update if you added/removed columns) ---
features = [
    "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",
    "home_PTS_last10", "home_FG_PCT_last10", "home_FT_PCT_last10", "home_FG3_PCT_last10",
    "home_AST_last10", "home_REB_last10",
    "away_PTS_last10", "away_FG_PCT_last10", "away_FT_PCT_last10", "away_FG3_PCT_last10",
    "away_AST_last10", "away_REB_last10",
    "home_win_pct_last10", "away_win_pct_last10"
]

# sanity check: which features are missing from the df?
missing_features = [c for c in features if c not in df.columns]
if missing_features:
    print("WARNING: The following expected feature columns are missing from your dataframe:")
    print(missing_features)
    print("You can either remove them from the 'features' list or ensure they are created in add_rolling_features().")
    # we'll proceed but only use present columns
features = [c for c in features if c in df.columns]

if "HOME_TEAM_WINS" not in df.columns:
    print("ERROR: 'HOME_TEAM_WINS' column not found in CSV. This is required as the target.")
    sys.exit(1)

# Fill missing feature values with column means (avoids dropping too many rows)
def fill_with_mean(dframe, cols):
    for c in cols:
        if c in dframe.columns:
            mean_val = dframe[c].mean()
            dframe[c] = dframe[c].fillna(mean_val)
    return dframe

train_df = fill_with_mean(train_df, features)
val_df   = fill_with_mean(val_df, features)
test_df  = fill_with_mean(test_df, features)

# After filling, drop rows missing target only (should be rare)
train_df = train_df.dropna(subset=["HOME_TEAM_WINS"])
val_df   = val_df.dropna(subset=["HOME_TEAM_WINS"])
test_df  = test_df.dropna(subset=["HOME_TEAM_WINS"])

# Print sizes
print("Sizes after fill/dropna:")
print("  train:", train_df.shape)
print("  val:  ", val_df.shape)
print("  test: ", test_df.shape)

# if any split is empty, stop with actionable message
if train_df.shape[0] < 50:
    print("ERROR: Training set is very small (<50 rows). Check your SEASON split or data completeness.")
    sys.exit(1)
if val_df.shape[0] < 10:
    print("WARNING: Validation set is small (<10 rows). Early stopping may be unreliable.")
if test_df.shape[0] < 10:
    print("WARNING: Test set is small (<10 rows). Metrics may be unreliable.")

# build matrices
X_train = train_df[features].astype(float)
y_train = train_df["HOME_TEAM_WINS"].astype(int)
X_val   = val_df[features].astype(float)
y_val   = val_df["HOME_TEAM_WINS"].astype(int)
X_test  = test_df[features].astype(float)
y_test  = test_df["HOME_TEAM_WINS"].astype(int)

# --- XGBoost model ---
# remove deprecated use_label_encoder arg to avoid warning
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Try fitting with early_stopping_rounds if supported; otherwise plain fit
fit_success = False
try:
    # many xgboost versions accept early_stopping_rounds in sklearn API
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,  # safe to try
        verbose=True,
    )
    fit_success = True
except TypeError:
    print("Note: this xgboost version doesn't accept early_stopping_rounds in .fit(); training without early stopping.")
except Exception as e:
    print("Warning while attempting fit with early stopping:", e)

if not fit_success:
    model.fit(X_train, y_train)
    print("Trained model with plain fit (no early stopping).")

# --- Evaluate ---
try:
    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= 0.5).astype(int)

    print("Test accuracy:", accuracy_score(y_test, preds_test))
    print("Test log loss:", log_loss(y_test, probs_test))
except Exception as e:
    print("Error computing test metrics (likely empty arrays):", e)
    print("test_df shape:", X_test.shape)
    sys.exit(1)

# Save model
joblib.dump(model, MODEL_OUT)
print("Saved model to", MODEL_OUT)
