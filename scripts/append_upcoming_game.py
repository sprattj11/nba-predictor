#!/usr/bin/env python3
# scripts/append_upcoming_game.py
import pandas as pd
from pathlib import Path
from datetime import datetime

SRC = Path("data/games_summary_merged_with_rest.csv")
OUT = Path("data/games_summary_with_upcoming.csv")

# EDIT these values for the matchup you want to test:
HOME = "OKC"          # home team code as used in your CSV
AWAY = "HOU"          # away team code as used in your CSV
DATE = "2025-10-21"   # YYYY-MM-DD

df = pd.read_csv(SRC, low_memory=False, parse_dates=["date"])
# create an empty row with same columns
new_row = {c: pd.NA for c in df.columns}
new_row["date"] = pd.to_datetime(DATE)
# Columns names inferred from your dataset: home_team, away_team, home_pts, away_pts etc.
# set the teams
if "home_team" in df.columns:
    new_row["home_team"] = HOME
if "away_team" in df.columns:
    new_row["away_team"] = AWAY
# leave scores (home_pts/away_pts) as NaN
# other columns will be NaN and OK — add_rolling_features only needs date and team ids

# append and save
df2 = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
df2.to_csv(OUT, index=False)
print(f"Wrote {OUT} with appended game {HOME} vs {AWAY} on {DATE}")
print("Now run interactive_predict_spread_market.py with --games", OUT)
