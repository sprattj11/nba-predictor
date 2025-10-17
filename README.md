# 🏀 NBA Game Predictor

**Author:** Jason Spratt  
**Last Updated:** 2025-10-17  
**Language:** Python 3.13  
**Environment:** macOS / venv (`.venv`)

---

## 📘 Project Description
This project builds and trains a machine learning model to **predict NBA game outcomes** — specifically, the probability that the **home team wins**.

It uses historical NBA data and engineered features such as:
- Season-to-date averages (team boxscore stats)
- Rolling 10-game averages (momentum)
- Rolling win percentages (short-term form)

The model is based on **XGBoost (Gradient Boosted Trees)** and provides **probabilistic predictions** (e.g., _Home team wins: 68%_).  
An interactive command-line tool lets you input team abbreviations and a date to generate predictions for upcoming or past games.

---

## ⚙️ Key Features
- Fully reproducible data pipeline (via `pandas`)
- Automatic rolling feature generation (last 10 games)
- Gradient Boosted Trees (XGBoost) with time-aware splits
- Interactive prediction CLI using team abbreviations
- Organized modular code: `src/`, `scripts/`, `data/`, `models/`
- Automated workflows with `Makefile` commands

------------------------------------------------------
2) Project Directory Structure
------------------------------------------------------

nba-predictor/
│
├── data/
│   ├── nba_games.csv
│   ├── games_details.csv
│   ├── players.csv
│   ├── ranking.csv
│   └── teams.csv
│
├── models/
│   ├── xgb_nba_model.joblib
│   └── README.md
│
├── scripts/
│   ├── train_xgb.py
│   ├── predict_by_abbrev_with_nbaapi.py
│   └── predict_from_history.py
│
├── src/
│   └── features.py
│
├── Makefile
├── requirements.txt
└── README.md

------------------------------------------------------

## 🚀 Setup & Installation
Clone the repository:
git clone https://github.com/yourusername/nba-predictor.git
cd nba-predictor

Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

Or just run:
make env

---

## 🧠 Usage

Train the Model:
make train

Run an Interactive Prediction:
make predict
Example input:
Home team abbreviation (e.g., LAL): BOS
Away team abbreviation (e.g., BOS): TOR
Game date (YYYY-MM-DD): 2025-10-16

Example output:
Predicted winner: HOME
Probability HOME wins: 0.68
Probability AWAY wins: 0.32

Clean Temporary Files:
make clean

---

## 🧩 Model Details
Algorithm: XGBoost (Gradient Boosted Trees)
Model File: models/xgb_nba_model.joblib
Trained: 2025-10-17
Training Data: data/nba_games.csv
Features: 26 engineered features
Metrics (Test):
- Accuracy: 0.99
- Log Loss: 0.02

Reproduce training:
source .venv/bin/activate && make train

---

## 🧮 Feature Engineering Overview
- Season averages: PTS, FG%, FT%, FG3%, AST, REB for both teams
- Rolling 10-game averages: same stats, computed on last 10 games
- Rolling win percentages: win rate over last 10 games (home/away)
- All rolling windows use `.shift()` to avoid data leakage.
- Missing early-season data filled using shorter windows or league averages.

---

## 🔮 Future Enhancements
- Add rest-day and travel distance features
- Include player injuries / lineup availability
- Integrate Elo ratings and Vegas odds
- Deploy via Streamlit web app for public demo

---

## 🧰 Makefile Reference
make env      → Create virtual environment and install packages
make train    → Train XGBoost model
make predict  → Run interactive CLI prediction
make clean    → Remove cache files and compiled artifacts
make tidy     → Print repository cleanup suggestions

---

## 🧾 Dependencies
Python packages (see requirements.txt):
pandas
scikit-learn
xgboost
joblib
matplotlib
nba_api

---

## 🙏 Acknowledgments
- Data: NBA API and public Kaggle datasets
- Inspired by analytics work from FiveThirtyEight
- Built for research and educational use (not betting)

---

## 📅 License
MIT License © 2025 Jason Spratt

---