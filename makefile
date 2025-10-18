# ==========================
# NBA Predictor Makefile
# ==========================

PYTHON := python
DATA_DIR := data
MODEL_DIR := models

# default target
.DEFAULT_GOAL := help

# -------------------------------------
# 🧠  Model training and evaluation
# -------------------------------------
train:
	@echo "🚀 Training LightGBM spread model..."
	$(PYTHON) $(MODEL_DIR)/train_spread.py --csv $(DATA_DIR)/games_summary_merged.csv --model-out $(MODEL_DIR)/spread_model.pkl --report-out $(MODEL_DIR)/eval_report.csv

# -------------------------------------
# 🎯  Interactive prediction CLI
# -------------------------------------
spread:
	@echo "🏀 Starting interactive spread predictor..."
	$(PYTHON) $(MODEL_DIR)/interactive_predict_spread.py --games $(DATA_DIR)/games_summary_merged.csv --model $(MODEL_DIR)/spread_model.pkl

# -------------------------------------
# 🧹  Cleanup compiled files and caches
# -------------------------------------
clean:
	@echo "🧹 Cleaning __pycache__ and temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f *.pyc
	rm -f $(MODEL_DIR)/*.pyc

# -------------------------------------
# 🆘  Show help
# -------------------------------------
help:
	@echo "Available commands:"
	@echo "  make train     - Train LightGBM model using merged game summary data"
	@echo "  make spread    - Run interactive CLI to predict spreads"
	@echo "  make clean     - Remove caches and temp files"
	@echo ""
	@echo "Examples:"
	@echo "  make train"
	@echo "  make spread"
