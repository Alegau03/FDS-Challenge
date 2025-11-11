# Pokemon Battle Predictor — README

Brief professional overview and usage instructions for the repository.

## Overview

This repository contains a production-ready pipeline to predict the winner of competitive Pokemon battles using an ensemble of gradient-boosting models (LightGBM, CatBoost, XGBoost). The pipeline includes:

- Feature engineering that produces 339 features (team static features, dynamic battle-log features and interaction features).
- Per-model hyperparameter optimization via Optuna (separate optimizer scripts for LightGBM, CatBoost and XGBoost).
- A robust training pipeline with stratified K-fold CV, side-swap augmentation, seed bagging and isotonic calibration.
- An ensemble stage supporting weighted-average and stacking, plus automated selection between them.
- Prediction utilities to generate a final submission CSV.

This README explains how to run each step and where to find key outputs.

## Requirements

- Python 3.11+ (the project was developed with Python 3.13 in a venv). 
- Install project dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- CatBoost can run on CPU or GPU. GPU requires a CUDA-capable device and the GPU-enabled CatBoost build. The `optimizer_cat.py` is configured to use `task_type='GPU'` if CatBoost is installed with GPU support; otherwise it will fail and you should switch it to CPU or install GPU drivers/CatBoost GPU build.
- If you don't have a GPU, set `task_type: 'CPU'` in `optimizer_cat.py` (or install CatBoost CPU-only package).

## File structure (high-level)

- `feature_engineering.py` — generates the DataFrame used by training/prediction (includes `create_feature_df`).
- `optimizer_lightbgm.py` — LightGBM Optuna optimizer (writes `models/best_params.json`).
- `optimizer_cat.py` — CatBoost Optuna optimizer (writes `models/best_params_cat.json`).
- `optimizer_xgb.py` — XGBoost Optuna optimizer (writes `models/best_params_xgb.json`).
- `train_ensemble.py` — main training pipeline that trains models per-fold, performs calibration, and creates the ensemble.
- `predict_ensemble.py` — loads trained models and produces `submission/ensamble_submission.csv`.
- `config.py` — central configuration: paths, CV settings and model selection.
- `models/` — output folder for saved models, calibrators and best params JSON files.
- `submission/` — output folder for final submission CSV.

## Quickstart: Optimizers (find best params)

Purpose: the optimizer scripts search hyperparameter spaces (via Optuna) and save the best parameters as JSON. Each optimizer implements a CV-based objective (LogLoss) and returns the best configuration for the corresponding model.

Run optimizers (recommended: first ensure `config.py` points to the correct `TRAIN_FILE_PATH` and `MODEL_OUTPUT_DIR`):

```bash
# LightGBM (CPU)
python optimizer_lightbgm.py

# XGBoost (CPU)
python optimizer_xgb.py

# CatBoost (GPU recommended if available)
python optimizer_cat.py
```

Recommended defaults:
- `OPTUNA_TRIALS` in `config.py` (or environment) controls the number of trials. Typical values: 50–200.
- Outputs saved under `models/`:
  - `best_params.json` (LightGBM)
  - `best_params_cat.json` (CatBoost)
  - `best_params_xgb.json` (XGBoost)

What each optimizer does (short):
- `optimizer_lightbgm.py`: loads features once (for efficiency), runs stratified K-fold trials and minimizes CV LogLoss.
- `optimizer_cat.py`: runs CatBoost CV trials (now configured to use GPU if available), minimizing CV LogLoss.
- `optimizer_xgb.py`: runs XGBoost CV trials and minimizes CV LogLoss.

After completion, the best params files will be used automatically by `train_ensemble.py` if present.

## Feature engineering (how it works)

`feature_engineering.py` exposes `create_feature_df(file_path, max_turns=30)` that:

- Parses each battle JSONL entry and extracts static team features (base stats, type coverage, predicted unseen types using `predictor.py`) and dynamic battle-log features.
- Generates temporal-windowed statistics (e.g. damage/HP/KO counts across early/mid/late windows), switch and status tracking, momentum indicators, and 15 V7 interaction features.
- Returns a pandas DataFrame with `battle_id`, (optionally) `player_won` and 339 model-ready features.

Notes & tips:
- Feature generation runs reasonably fast; tests showed ~1295 battles/sec in the dev environment for the V7 pipeline. 
- `predictor.py` is used to load a global type distribution (`predict.csv`) to estimate unseen types; ensure `predict.csv` is present if using unseen-type features.

## Training (train_ensemble.py) — what it does

`train_ensemble.py` orchestrates the entire training pipeline:

1. Loads feature DataFrame from `feature_engineering.create_feature_df`.
2. Applies stratified K-fold CV (controlled by `config.N_SPLITS`) — default 10 folds.
3. Optionally applies side-swap augmentation to double the training set by swapping player perspectives (controlled in code).
4. Trains base models per-fold (LightGBM, CatBoost, XGBoost) using `BEST_PARAMS` if found in `models/`.
   - Each model uses seed bagging, early stopping and per-fold isotonic calibration.
   - Scale-pos-weight is computed to help with class imbalance.
5. Aggregates out-of-fold (OOF) probabilities from base models and evaluates metrics (Accuracy, AUC, LogLoss).
6. Tunes ensemble weights via a small grid search and optionally trains a stacking meta-learner (LogisticRegression) on OOF predictions.
7. Chooses final ensemble method (weighted-average or stacking) based on OOF performance, saves best threshold and ensemble configuration.

Outputs saved into `models/` (per fold): trained model files, calibrators, `ensemble_weights.json`, `best_threshold.json` and other artifacts referenced in `config.py`.

How to run final training (after optimizers finished):

```bash
# Ensure best params files exist in models/ (created by optimizers)
python train_ensemble.py
```

Expected runtime: depends on machine, number of folds and model complexity. Use GPU-enabled CatBoost to reduce time for CatBoost models.

## Prediction (predict_ensemble.py)

Once `train_ensemble.py` completes successfully, run:

```bash
python predict_ensemble.py
```

This does:
- Loads test features via `create_feature_df` for `config.TEST_FILE_PATH`.
- Loads per-fold models and calibrators, averages fold predictions and applies side-swap averaging.
- Combines model predictions by the saved ensemble method and weights.
- Applies ensemble-level isotonic calibration if available.
- Binarizes with the saved threshold and writes `submission/ensamble_submission.csv`.

## Configuration

Primary settings are in `config.py`. Key parameters:
- `TRAIN_FILE_PATH`, `TEST_FILE_PATH` — source data paths.
- `MODEL_OUTPUT_DIR`, `ENSEMBLE_WEIGHTS_PATH`, `THRESHOLD_PATH` — output artifact paths.
- `N_SPLITS` — number of CV folds (default: 10).
- `RANDOM_STATE` — seed for reproducibility.
- `MODEL_TYPES` — list of models to train/use in ensemble (default `['lgbm', 'cat', 'xgb']`).
- `ENSEMBLE_METHOD` — `'weighted_average'`, `'stacking'` or `'auto'`.

Edit `config.py` to adapt to your environment (paths, number of trials, CV settings). The training and prediction scripts consult this file.

## Practical tips and troubleshooting

- GPU for CatBoost: if you want the CatBoost optimizer and training to use GPU, ensure CatBoost GPU is installed and drivers are present. If not available, switch `task_type` to `'CPU'` inside `optimizer_cat.py` or install CatBoost GPU build.
- Memory: feature matrix with 339 features can be memory intensive for large datasets—monitor memory during CV training.
- Reproducibility: set `RANDOM_STATE` in `config.py`. The pipeline also uses seed bagging; results depend on seeds and CV splits.
- Faster experimentation: run optimizers with fewer trials (`config.OPTUNA_TRIALS`) to iterate quickly, then increase trials for final runs.

## Outputs to include in submission

For a professor or reproducible deliverable, include:
- `feature_engineering.py`, `train_ensemble.py`, `predict_ensemble.py`, `config.py` (clean documented code)
- `REPORT.md` (technical report)
- `models/` folder with final models and `*.json` best-params files (optional, large)
- `submission/ensamble_submission.csv` (final predictions)

## Contact & next steps

If you want, I can:
- Run the three optimizers in parallel (or sequentially) and collect the `best_params` files.
- Launch `train_ensemble.py` with those params and produce a final submission.
- Add a small `Makefile` or orchestration script to run the full pipeline with one command.

---

End of README — concise, professional project overview and run instructions.
