"""
Optimizer XGBoost V2 con Optuna per minimizzare il LogLoss via k-fold stratificato.

Flusso in breve:
- Carica e crea le feature dal TRAIN_FILE_PATH (vedi feature_engineering_v2.py);
- Definisce un objective Optuna che esegue CV con early stopping e ritorna la media dei LogLoss;
- Salva i migliori parametri in BEST_PARAMS_XGB_PATH per il training dell'ensemble.
"""

import optuna
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np

from feature_engineering_v2 import create_feature_df
import codice.config as config
try:
    import xgboost as xgb
except ImportError:
    xgb = None


def make_objective(X_train, y_train):
    """Crea l'objective Optuna chiudendo su X,y per evitare global."""
    def objective(trial):
        if xgb is None:
            raise RuntimeError("XGBoost non installato")

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': trial.suggest_float('eta', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'tree_method': 'hist'
        }

        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        losses = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
            dtr = xgb.DMatrix(X_tr, label=y_tr)
            dva = xgb.DMatrix(X_va, label=y_va)
            m = xgb.train(params, dtr, num_boost_round=6000, evals=[(dva, 'valid')], early_stopping_rounds=200, verbose_eval=False)
            p = m.predict(xgb.DMatrix(X_va))
            losses.append(log_loss(y_va, p))
        return float(np.mean(losses))

    return objective


if __name__ == '__main__':
    print("Caricamento dati per Optuna XGBoost...")
    df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)
    y_target = df['player_won'].astype(int)
    X_features = df.drop(columns=['battle_id', 'player_won'])

    print("Ottimizzazione XGBoost (minimize LogLoss)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(make_objective(X_features, y_target), n_trials=50)
    print("Best LogLoss:", study.best_value)
    print("Best params:", study.best_params)

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    with open(config.BEST_PARAMS_XGB_PATH, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
    print("Salvati in", config.BEST_PARAMS_XGB_PATH)
