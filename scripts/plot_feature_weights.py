#!/usr/bin/env python3
"""
scripts/plot_feature_weights.py

Generates and saves a PNG graph of XGBoost feature importances.

Usage:
    source .venv/bin/activate
    python3 scripts/plot_feature_weights.py
"""

import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_nba_model.joblib"
OUTPUT_PATH = PROJECT_ROOT / "models" / "feature_weights.png"

def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # Load model
    model = joblib.load(MODEL_PATH)

    # Try to get feature names
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        try:
            booster = model.get_booster()
            feature_names = booster.feature_names
        except Exception:
            feature_names = [f"f{i}" for i in range(len(model.feature_importances_))]

    # Get importance scores (from fitted model)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        try:
            importances = model.get_booster().get_score(importance_type="gain")
            # Convert dict to aligned arrays
            keys, vals = zip(*sorted(importances.items(), key=lambda x: x[1], reverse=True))
            feature_names, importances = keys, vals
        except Exception:
            print("Error: Could not extract feature importances.")
            return

    # Build DataFrame
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    # Normalize importance values (optional)
    df["Importance"] = df["Importance"] / df["Importance"].sum()

    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(df["Feature"], df["Importance"], color="royalblue")
    plt.gca().invert_yaxis()  # highest at top
    plt.xlabel("Normalized Importance")
    plt.title("XGBoost Feature Importance Weights")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.show()

    print(f"\n✅ Saved feature importance plot to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
