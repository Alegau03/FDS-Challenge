"""
Optimizer CatBoost V2 con Optuna per minimizzare il LogLoss via k-fold stratificato.

Flusso:
- Carica e crea le feature dal TRAIN_FILE_PATH;
- Definisce un objective Optuna che fa CV e ritorna la media dei LogLoss;
- Salva i migliori parametri in BEST_PARAMS_CAT_PATH.
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
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def make_objective(X_train, y_train):
    """Crea l'objective Optuna per CatBoost chiudendo su X,y (niente global)."""
    def objective(trial):
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost non installato")

        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'random_seed': config.RANDOM_STATE,
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            'border_count': trial.suggest_int('border_count', 32, 254),
            'iterations': trial.suggest_int('iterations', 1500, 6000),
            'verbose': False
        }

        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        losses = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            X_va, y_va = X_train.iloc[va_idx], y_train.iloc[va_idx]
            m = CatBoostClassifier(**params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, use_best_model=True)
            p = m.predict_proba(X_va)[:, 1]
            losses.append(log_loss(y_va, p))
        return float(np.mean(losses))

    return objective


if __name__ == '__main__':
    print("Caricamento dati per Optuna CatBoost...")
    df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)
    y_target = df['player_won'].astype(int)
    X_features = df.drop(columns=['battle_id', 'player_won'])

    print("Ottimizzazione CatBoost (minimize LogLoss)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(make_objective(X_features, y_target), n_trials=50)
    print("Best LogLoss:", study.best_value)
    print("Best params:", study.best_params)

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    with open(config.BEST_PARAMS_CAT_PATH, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
    print("Salvati in", config.BEST_PARAMS_CAT_PATH)
