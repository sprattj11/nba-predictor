# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load your dataset
df = pd.read_csv("nba_games.csv")

# Step 2: Choose your features (inputs)
features = [
    "PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home",
    "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away"
]

X = df[features]
y = df["HOME_TEAM_WINS"]  # Target output (1 = win, 0 = loss)

# Step 3: Split into train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Step 4: Choose and train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions
preds = model.predict(X_test)

# Step 6: Evaluate how accurate it is
acc = accuracy_score(y_test, preds)
print(f"Model accuracy: {acc:.3f}")

# Step 7: Example prediction (make up some stats)
example_game = pd.DataFrame([{
    "PTS_home": 110, "FG_PCT_home": 0.47, "FT_PCT_home": 0.78, "FG3_PCT_home": 0.37,
    "AST_home": 25, "REB_home": 44,
    "PTS_away": 104, "FG_PCT_away": 0.45, "FT_PCT_away": 0.74, "FG3_PCT_away": 0.34,
    "AST_away": 22, "REB_away": 41
}])

predicted = model.predict(example_game)[0]
print("Predicted winner:", "Home team" if predicted == 1 else "Away team")

import joblib
joblib.dump(model, "nba_win_predictor.pkl")
