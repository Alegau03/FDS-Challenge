# optimizer_cat.py

import optuna
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd

import config as config
from feature_engineering_finale import create_feature_df

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


def _compute_scale_pos_weight(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(1.0, neg / max(1.0, pos))


def objective(trial: optuna.Trial):
    df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)
    y = df['player_won'].fillna(0).astype(int)
    X = df.drop(columns=['battle_id', 'player_won']).astype(np.float32)

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    params = {
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-6, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
        'iterations': trial.suggest_int('iterations', 1500, 6000),
        'random_seed': config.RANDOM_STATE,
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'verbose': False,
    }

    oof = np.zeros(len(y))
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        spw = _compute_scale_pos_weight(y_tr)
        params_fold = {**params, 'scale_pos_weight': spw}

        model = CatBoostClassifier(**params_fold)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, use_best_model=True)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]

    return log_loss(y, np.clip(oof, 1e-6, 1-1e-6))


def main():
    if CatBoostClassifier is None:
        print("CatBoost non disponibile")
        return

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=getattr(config, 'OPTUNA_TRIALS', 20))

    with open(config.BEST_PARAMS_CAT_PATH, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
    print("Migliori parametri CatBoost salvati.")


if __name__ == '__main__':
    np.random.seed(getattr(config, 'RANDOM_STATE', 42))
    main()
