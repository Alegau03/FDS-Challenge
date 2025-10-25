# optimizer_lightgbm_v3.py

import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

import config as config
import json
import os
from feature_engineering_finale import create_feature_df

# Carichiamo i dati una sola volta per efficienza
print("Caricamento dati v3 per l'ottimizzazione LightGBM...")
train_df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)
y = train_df['player_won'].astype(int)
X = train_df.drop(columns=['battle_id', 'player_won']).astype('float32')

def objective(trial):
    """Funzione obiettivo per Optuna: minimizziamo la LogLoss media in CV.
    Allineata al training v3 (stratified k-fold, early stopping sul fold).
    """
    # Definiamo lo spazio di ricerca degli iperparametri
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 500, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', -1, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'max_bin': trial.suggest_int('max_bin', 63, 511),
        'extra_trees': trial.suggest_categorical('extra_trees', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
    }

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_loglosses = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='logloss',
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        # Predizione sul fold di validazione corrente
        val_preds_proba = model.predict_proba(X_val)[:, 1]
        # Ottimizziamo direttamente la LogLoss (senza soglia)
        ll = log_loss(y_val, val_preds_proba)
        fold_loglosses.append(ll)

    # Restituiamo la media delle LogLoss dei fold (da minimizzare)
    return sum(fold_loglosses) / len(fold_loglosses)

if __name__ == "__main__":
    print("Inizio ottimizzazione LightGBM v3 con Optuna (metrica: LogLoss - minimize)...")
    # Creiamo uno studio, specificando che vogliamo minimizzare la LogLoss
    study = optuna.create_study(direction='minimize')

    # Avviamo l'ottimizzazione per 50 tentativi
    study.optimize(objective, n_trials=20)

    print("\n------------------ OTTIMIZZAZIONE COMPLETATA ------------------")
    print(f"Miglior LogLoss CV: {study.best_value:.5f}")
    print("Migliori iperparametri trovati:")
    for key, value in study.best_params.items():
        print(f"  '{key}': {value},")

    # Salva i migliori parametri in JSON per caricarli automaticamente nel training
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)
    best_params_payload = {
        **study.best_params,
        # Parametri fissi fondamentali
        'objective': 'binary',
        'metric': 'binary_logloss',
        'random_state': config.RANDOM_STATE,
        'n_jobs': -1,
    }
    with open(config.BEST_PARAMS_PATH, 'w') as f:
        json.dump(best_params_payload, f, indent=2)
    print(f"\nParametri migliori salvati in: {config.BEST_PARAMS_PATH}")
    print("Ora puoi lanciare il training v3 (ensemble) senza modifiche manuali.")
    print("---------------------------------------------------------------")