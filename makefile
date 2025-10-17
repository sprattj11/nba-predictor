# Makefile for nba-predictor
# Usage: make <target>
SHELL := /bin/bash

# change these if you use different names/paths
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
REQ := requirements.txt

# scripts
TRAIN := scripts/train_xgb.py
PREDICT := scripts/predict_by_abbrev_with_nbaapi.py
EVAL := scripts/evaluate.py    # optional (create later)
FEATURES := src/features.py

# default
.PHONY: help
help:
	@echo "Makefile targets:"
	@echo "  make env        -> create virtualenv and install requirements"
	@echo "  make install    -> install requirements into active venv"
	@echo "  make train      -> train model (runs train_xgb.py)"
	@echo "  make predict    -> run CLI prediction (interactive)"
	@echo "  make fmt        -> run python formatter (black) if installed"
	@echo "  make test       -> run pytest (if you add tests/)"
	@echo "  make clean      -> remove pyc, __pycache__, and temp files"
	@echo "  make tidy       -> run basic repo tidying commands (gitignored outputs)"
	@echo ""
	@echo "Examples:"
	@echo "  make env"
	@echo "  source .venv/bin/activate && make train"

# create a venv and install requirements
.PHONY: env
env:
	@test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	@if [ -f $(REQ) ]; then $(PIP) install -r $(REQ); else echo "No requirements.txt found"; fi
	@echo "Virtualenv created and packages installed. Activate with: source $(VENV)/bin/activate"

# install into already-created venv (assumes venv active)
.PHONY: install
install:
	@if [ -f $(REQ) ]; then $(PIP) install -r $(REQ); else echo "No requirements.txt found"; fi

# run training
.PHONY: train
train:
	@echo "Running training script: $(TRAIN)"
	@PYTHONPATH=. $(PY) $(TRAIN)

# run interactive prediction script
.PHONY: predict
predict:
	@echo "Running prediction CLI: $(PREDICT)"
	@PYTHONPATH=. $(PY) $(PREDICT)


# format code (optional; requires black)
.PHONY: fmt
fmt:
	@$(PIP) show black >/dev/null 2>&1 || (echo "black not installed. Run 'make env' or 'make install' to install."; exit 1)
	@$(VENV)/bin/black src scripts

# run tests (optional)
.PHONY: test
test:
	@$(PIP) show pytest >/dev/null 2>&1 || (echo "pytest not installed. Run 'make env' or add pytest to requirements."; exit 1)
	@$(VENV)/bin/pytest -q

# remove temporary/python build files (safe)
.PHONY: clean
clean:
	@echo "Cleaning python cache and temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + || true
	@find . -type f -name "*.pyc" -delete || true
	@rm -f .coverage || true
	@echo "done."

# tidy: non-destructive (does not delete tracked files)
.PHONY: tidy
tidy:
	@echo "Suggested tidy actions (not executed):"
	@echo "  - Move old artifacts to backup/"
	@echo "  - Ensure large data files are in data/ and listed in .gitignore"
	@echo "To actually perform moves, use git mv or manual commands."

