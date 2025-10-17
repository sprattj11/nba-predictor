========================================
NBA Game Predictor — Project Overview
========================================

Author: Jason Spratt
Last Updated: 2025-10-17
Language: Python 3.13
Environment: macOS / venv (.venv)
========================================

1) Project Description
----------------------
This project builds and trains a machine learning model to predict the outcome
of NBA games (win/loss for the home team).

It uses historical game-level data to engineer team features, such as:
- season-to-date boxscore averages,
- rolling 10-game performance stats,
- and recent win percentages.

The model is trained using Gradient Boosted Trees (XGBoost) and provides
probabilistic predictions for any matchup (e.g., “Home team wins: 68%”).
A simple command-line interface (CLI) lets you enter team abbreviations and
a date to generate predictions for upcoming or historical games.


2) Key Features
---------------
- Fully reproducible data pipeline using pandas
- Automatic rolling features (10-game averages + win percentages)
- Robust XGBoost training with time-aware train/validation/test split
- Interactive prediction via team abbreviations and date
- Modular code organization (src/, scripts/, data/, models/)
- Easy automation through a Makefile (`make train`, `make predict`, etc.)


3) Project Directory Structure
-------------------------------
nba-predictor/
├── data/
│   ├── nba_games.csv            ← main dataset
│   ├── teams.csv, players.csv   ← supporting datasets
│   └── ranking.csv, games_details.csv
├── models/
│   ├── xgb_nba_model.joblib     ← trained model artifact
│   └── README.md or features.txt
├── scripts/
│   ├── train_xgb.py             ← trains model on dataset
│   ├── predict_by_abbrev_with_nbaapi.py ← CLI prediction tool (team abbreviations)
│   └── predict_from_history.py  ← alternative predictor (team IDs)
├── src/
│   └── features.py              ← feature engineering (rolling stats, win%)
├── Makefile                     ← automation (env, train, predict, clean)
├── requirements.txt             ← package dependencies
└── README.txt (this file)


4) Setup & Installation
------------------------
Clone this repository:
    git clone https://github.com/yourusername/nba-predictor.git
    cd nba-predictor

Create and activate a virtual environment:
    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:
    pip install -r requirements.txt

(Or simply run `make env` which creates the environment and installs requirements.)


5) Usage
--------
Train the model:
    make train

Run an interactive prediction:
    make predict
    # Example prompts:
    # Home team abbreviation (e.g., LAL): BOS
    # Away team abbreviation (e.g., BOS): TOR
    # Game date (YYYY-MM-DD): 2025-10-16

This will display a feature summary for the matchup and output predicted
probabilities for the home and away teams.

Clean up temporary files:
    make clean


6) Model Details
----------------
Algorithm: XGBoost (Gradient Boosted Trees)
Model File: models/xgb_nba_model.joblib
Trained: 2025-10-17
Training Data: data/nba_games.csv
Features: 26 (boxscore averages, rolling stats, win% features)

Test Metrics:
    Accuracy  : 0.99
    Log Loss  : 0.02

Reproduce:
    source .venv/bin/activate && make train


7) Feature Engineering Summary
-------------------------------
- Season-to-date boxscore averages for home and away teams
- Rolling 10-game averages for key stats (PTS, FG%, FT%, FG3%, AST, REB)
- Rolling 10-game win percentages (home/away)
- All rolling windows computed with shift() to avoid data leakage
- Missing early-season values filled using shorter windows or league averages


8) Future Enhancements
----------------------
- Add rest-day and travel distance features
- Include player availability and injuries
- Integrate Elo ratings and Vegas odds as predictive inputs
- Deploy as a Streamlit web app for interactive demo


9) Makefile Quick Reference
----------------------------
make env      → create virtual environment and install requirements
make train    → train model (runs scripts/train_xgb.py)
make predict  → run interactive prediction CLI
make clean    → delete cache/__pycache__ files
make tidy     → show repo cleanup suggestions

(Ensure your virtual environment is activated before running these commands.)


10) Dependencies
----------------
Python packages (see requirements.txt):
    pandas
    scikit-learn
    xgboost
    joblib
    matplotlib
    nba_api


11) Acknowledgments
-------------------
- Data sourced via NBA API and public Kaggle datasets.
- Inspired by analytics work from FiveThirtyEight and NBA community models.
- Built for educational and research purposes — not for betting use.


========================================
End of File
========================================
