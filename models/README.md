# 🧠 NBA Game Predictor — Model Details

Model: xgb_nba_model.joblib
Trained: 2025-10-17
Training Data: data/nba_games.csv

---

1️⃣ Base Box Score Features (Season-to-Date Averages)
Home Team:
 - PTS_home
 - FG_PCT_home
 - FT_PCT_home
 - FG3_PCT_home
 - AST_home
 - REB_home

Away Team:
 - PTS_away
 - FG_PCT_away
 - FT_PCT_away
 - FG3_PCT_away
 - AST_away
 - REB_away

---

2️⃣ Rolling 10-Game Averages (Recent Form)
Home Team:
 - home_PTS_last10
 - home_FG_PCT_last10
 - home_FT_PCT_last10
 - home_FG3_PCT_last10
 - home_AST_last10
 - home_REB_last10

Away Team:
 - away_PTS_last10
 - away_FG_PCT_last10
 - away_FT_PCT_last10
 - away_FG3_PCT_last10
 - away_AST_last10
 - away_REB_last10

---

3️⃣ Rolling Win Percentages (Short-Term Success)
 - home_win_pct_last10
 - away_win_pct_last10

---

Feature Engineering Notes
 - Rolling features use only games prior to the matchup (no leakage).
 - Early-season missing values (fewer than 10 prior games) filled with shorter windows or league averages.
 - Combines long-term consistency and short-term momentum.

---

Metrics (Test):
 - Accuracy = 0.99
 - Log Loss = 0.02

Reproduce:
 source .venv/bin/activate && make train