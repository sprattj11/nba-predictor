import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Load betting data
df = pd.read_csv("data/NBA Betting Data 2008-2025.csv", parse_dates=["date"])

# Compute actual game margin
df["margin"] = df["score_home"] - df["score_away"]

# Market target: difference between actual result and bookmaker line
# i.e. how much the closing line missed by
df["target"] = df["margin"] - np.where(df["whos_favored"] == "home", df["spread"], -df["spread"])

# Core predictive features
features = [
    "spread", "total",
    "moneyline_home", "moneyline_away",
    "h2_spread", "h2_total"
]

# Optional: implied probabilities
df["prob_home"] = 1 / (1 + np.exp(np.log(df["moneyline_home"] / -df["moneyline_away"])))
features.append("prob_home")

# Drop NaNs
df = df.dropna(subset=["target"])

# Train/test split by season (hold out most recent)
last_season = df["season"].max()
train = df[df["season"] < last_season]
test = df[df["season"] == last_season]

X_train, y_train = train[features], train["target"]
X_test, y_test = test[features], test["target"]

# Train LightGBM regression on residual vs line
params = dict(objective="regression", metric="l1", learning_rate=0.03, num_leaves=31, seed=42)
model = lgb.train(params, lgb.Dataset(X_train, label=y_train), num_boost_round=500)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"MAE vs market spread: {mae:.3f}")
