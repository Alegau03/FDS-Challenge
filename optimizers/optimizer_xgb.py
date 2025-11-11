# optimizer_xgb.py

import optuna
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import numpy as np

from feature_engineering import create_feature_df
import config as config

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def _compute_scale_pos_weight(y):
    y = np.asarray(y)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos <= 0:
        return 1.0
    return max(1.0, neg / max(1.0, pos))


def objective(trial):
    if xgb is None:
        raise RuntimeError("XGBoost non installato")

    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': trial.suggest_float('eta', 0.01, 0.1, log=True),  # Learning rate più alto
        'max_depth': trial.suggest_int('max_depth', 6, 14),  # Più profondo (era 3-12)
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 5.0, log=True),  # Ridotto max
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),  # Aumentato min
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),  # Aumentato min
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),  # Meno L1
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),  # Meno L2
        'gamma': trial.suggest_float('gamma', 0.0, 2.0),  # Ridotto max (min_split_loss)
        'tree_method': 'hist',
        'verbosity': 0,
        'nthread': -1,
        'seed': config.RANDOM_STATE,
    }

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    losses = []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        spw = _compute_scale_pos_weight(y_tr)
        params_fold = {**params, 'scale_pos_weight': spw}
        dtr = xgb.DMatrix(X_tr, label=y_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        m = xgb.train(params_fold, dtr, num_boost_round=6000, evals=[(dva, 'valid')], early_stopping_rounds=200, verbose_eval=False)
        p = m.predict(xgb.DMatrix(X_va))
        losses.append(log_loss(y_va, p))
    return float(np.mean(losses))


if __name__ == '__main__':
    print("\nCaricamento dati per Optuna XGBoost\n")
    df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)
    y = df['player_won'].astype(int)
    X = df.drop(columns=['battle_id', 'player_won']).astype(np.float32)

    print("Ottimizzazione XGBoost (minimize LogLoss)\n")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=getattr(config, 'OPTUNA_TRIALS', 50))
    print("\n" + "=" * 80)
    print("Best LogLoss:", study.best_value)
    print("Best params:", study.best_params)
    print("=" * 80)

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    with open(config.BEST_PARAMS_XGB_PATH, 'w', encoding='utf-8') as f:
        json.dump(study.best_params, f, indent=2)
    print("\nSalvati in", config.BEST_PARAMS_XGB_PATH)
