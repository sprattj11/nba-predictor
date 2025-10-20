#!/usr/bin/env python3
"""
eval_ats_accuracy.py

Compute ATS (Against The Spread) accuracy for model predictions.

It compares your model's predicted margins in `eval/predictions.csv`
to the real game results and Vegas spreads in
`data/NBA Betting Data 2008-2025.csv`.

Outputs:
- Total games merged
- Push count
- ATS accuracy (including pushes)
- ATS accuracy (excluding pushes)
"""

import pandas as pd
import numpy as np

# --- Config ---
PREDICTIONS_CSV = "eval/predictions.csv"
BETTING_CSV = "data/NBA Betting Data 2008-2025.csv"
# --------------

print("Loading predictions:", PREDICTIONS_CSV)
preds = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"])
print("Loading betting data:", BETTING_CSV)
bet = pd.read_csv(BETTING_CSV, parse_dates=["date"])

# Normalize team names to lowercase for merging
preds["home_lower"] = preds["home_team"].str.lower()
preds["away_lower"] = preds["away_team"].str.lower()
bet["home_lower"] = bet["home"].str.lower()
bet["away_lower"] = bet["away"].str.lower()

# Merge on date + home + away
merged = pd.merge(
    preds,
    bet[["date","home_lower","away_lower","spread","whos_favored","score_home","score_away"]],
    on=["date","home_lower","away_lower"],
    how="inner"
)

print(f"Merged {len(merged):,} games.")

# Compute actual and predicted ATS results
merged["actual_margin"] = merged["score_home"] - merged["score_away"]

# Signed spread: negative for home favorite, positive for away favorite
merged["signed_spread"] = np.where(
    merged["whos_favored"].str.lower() == "home",
    -merged["spread"],
    merged["spread"]
)

# Who actually covered the spread
merged["ats_margin"] = merged["actual_margin"] + merged["signed_spread"]
merged["ats_winner"] = np.where(
    merged["ats_margin"] > 0, "home",
    np.where(merged["ats_margin"] < 0, "away", "push")
)

# Model’s predicted ATS pick
merged["model_ats_margin"] = merged["pred_margin"] + merged["signed_spread"]
merged["model_pick"] = np.where(
    merged["model_ats_margin"] > 0, "home",
    np.where(merged["model_ats_margin"] < 0, "away", "push")
)

# Compare and compute accuracy
merged["correct"] = merged["model_pick"] == merged["ats_winner"]
incl = merged["correct"].mean() * 100
excl = merged.loc[merged["ats_winner"] != "push", "correct"].mean() * 100
pushes = (merged["ats_winner"] == "push").sum()

print()
print(f"Pushes: {pushes:,}")
print(f"ATS accuracy (including pushes): {incl:.2f}%")
print(f"ATS accuracy (excluding pushes): {excl:.2f}%")

# Optional: save merged data for inspection
merged.to_csv("eval/merged_ats_results.csv", index=False)
print("\nSaved detailed merged results to eval/merged_ats_results.csv")
