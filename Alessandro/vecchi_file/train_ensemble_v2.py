"""
Training ensemble V2: k-fold (o ripetuto) con side-swap augmentation SOLO sul training fold,
seed bagging, calibrazione per-fold (isotonic), ricerca pesi dell'ensemble e soglia su OOF,
più meta-calibrazione dell'ensemble.

In parole semplici:
- Creiamo le feature dal JSONL (vedi feature_engineering_v2.py) e prendiamo p1_* come player;
- facciamo K fold stratificati: per ciascun fold addestriamo i modelli su (train + train_swappato) e
    validiamo sull'originale (niente swap sul validation!);
- per ogni modello facciamo seed bagging (media di più semi) e poi calibratura isotonic sul fold;
- raccogliamo le predizioni OOF di ogni modello e cerchiamo i pesi dell'ensemble e la soglia
    che massimizza l'accuracy su OOF; stampiamo anche AUC e LogLoss;
- salviamo i modelli per-fold, i calibratori per-fold, i pesi dell'ensemble, la soglia ottima e
    un calibratore isotonic a livello ensemble (meta-calibration).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression


import codice.config as config
from feature_engineering_v2 import create_feature_df

# Optional imports for models
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
except (ImportError, ModuleNotFoundError):
    CatBoostClassifier = None
try:
    import xgboost as xgb
except (ImportError, ModuleNotFoundError):
    xgb = None

# --- Funzione per scambiare le colonne di un DataFrame ---
# Ad esempio, le feature relative al giocatore 1 (p1_) vengono sostituite
# con quelle del giocatore 2 (p2_) e viceversa. Le feature di differenza (diff_)
# vengono negate.

def swap_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Restituisce una copia con p1_ e p2_ scambiati e diff_ negati.

    Uso: per side-swap augmentation SOLO sul training fold (mai sulla validation/test)
    per evitare leakage. Le colonne p1_ diventano p2_ e viceversa; le colonne diff_
    vengono moltiplicate per -1.
    """
    swapped = df.copy()
    cols = df.columns
    for c in cols:
        if c.startswith('p1_'):
            alt = 'p2_' + c[3:]
            if alt in cols:
                swapped[c] = df[alt]
        elif c.startswith('p2_'):
            alt = 'p1_' + c[3:]
            if alt in cols:
                swapped[c] = df[alt]
        elif c.startswith('diff_'):
            swapped[c] = -df[c]
    return swapped

# --- Funzione per effettuare il training dell'ensemble ---

def train_ensemble():
    """Addestra i modelli base per-fold, calibra e costruisce l'ensemble.

    Passi chiave:
    1) Crea feature e target (player_won) dal file TRAIN_FILE_PATH.
    2) Definisce lo split (StratifiedKFold o RepeatedStratifiedKFold) secondo config.
    3) Per ogni fold e per ciascun modello in config.MODEL_TYPES:
       - Side-swap augmentation sul solo training fold; y_sw = 1 - y per gli esempi swappati.
       - Seed bagging: media delle predizioni di più semi.
       - Calibrazione isotonic per-fold su preds del validation fold.
       - Salvataggio del modello (ultimo seed) e del calibratore per il fold.
       - Accumulo delle OOF calibrate in oof[model].
    4) Ricerca su una griglia leggera dei pesi dell'ensemble e della soglia che massimizza accuracy OOF.
    5) Stampa metriche OOF (Accuracy, AUC, LogLoss) e salva threshold, pesi e meta-calibratore ensemble.
    """
    print("Training ensemble: ", config.MODEL_TYPES)
    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # 1) Carica dati e crea features (X) + target (y)
    df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)
    y = df['player_won'].fillna(0).astype(int)
    X = df.drop(columns=['battle_id', 'player_won'])

    # NOTA: l'augmentation side-swap va applicata SOLO sul training fold per evitare leakage.
    # L'OOF viene calcolato solo sugli esempi originali (senza duplicati swappati).

    # 2) Definisce lo split: KFold stratificato (eventualmente ripetuto)
    if config.N_REPEATS > 1:
        splitter = RepeatedStratifiedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS, random_state=config.RANDOM_STATE)
    else:
        splitter = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    # Probs OOF per ogni modello (solo su dataset originale, no swap)
    oof = {m: np.zeros(len(y)) for m in config.MODEL_TYPES}
    # Per riferimento/analisi: indici di train/validazione per fold
    fold_indices = []

    # 3) Carica best params salvati (o usa default robusti)
    lgb_params = None
    if os.path.exists(config.BEST_PARAMS_PATH):
        with open(config.BEST_PARAMS_PATH, 'r', encoding='utf-8') as f:
            lgb_params = json.load(f)

    # Default params o best salvati
    cat_params = None
    if os.path.exists(config.BEST_PARAMS_CAT_PATH):
        with open(config.BEST_PARAMS_CAT_PATH, 'r', encoding='utf-8') as f:
            cat_params = json.load(f)
    else:
        cat_params = dict(loss_function='Logloss', eval_metric='Logloss', random_seed=config.RANDOM_STATE,
                          depth=8, learning_rate=0.03, iterations=3000, verbose=False)

    xgb_params = None
    if os.path.exists(config.BEST_PARAMS_XGB_PATH):
        with open(config.BEST_PARAMS_XGB_PATH, 'r', encoding='utf-8') as f:
            xgb_params = json.load(f)
    else:
        xgb_params = dict(objective='binary:logistic', eval_metric='logloss',
                          eta=0.03, max_depth=8, subsample=0.8, colsample_bytree=0.8, reg_alpha=1e-6, reg_lambda=0.4,
                          tree_method='hist')

    fold_no = 0
    for tr_idx, va_idx in splitter.split(X, y.values):
        fold_no += 1
        print(f"Fold {fold_no}")
        X_tr_base, y_tr_base = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        fold_indices.append((tr_idx, va_idx))

        # Seed bagging: media delle predizioni su più semi diversi (stabilizza)
        seeds = [config.RANDOM_STATE + s for s in range(config.SEED_BAGGING)]

        # LIGHTGBM
        if 'lgbm' in config.MODEL_TYPES:
            # Side-swap augmentation SOLO sul training fold
            params_base = lgb_params or {
                'objective': 'binary', 'metric': 'logloss', 'random_state': config.RANDOM_STATE,
                'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 128,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 3,
            }
            X_tr_sw = swap_features_df(X_tr_base)
            y_tr_sw = 1 - y_tr_base
            X_tr = pd.concat([X_tr_base, X_tr_sw], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr_base, y_tr_sw], axis=0).reset_index(drop=True).astype(int)

            preds_bag = np.zeros(len(X_va))
            for sd in seeds:
                params = {**params_base, 'random_state': sd}
                m = lgb.LGBMClassifier(**params)
                m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='logloss',
                      callbacks=[lgb.early_stopping(200, verbose=False)])
                preds_bag += m.predict_proba(X_va)[:, 1]
            preds_bag /= max(1, len(seeds))
            # Calibrazione per-fold (isotonic) sulle predizioni del validation fold
            cal = IsotonicRegression(out_of_bounds='clip').fit(preds_bag, y_va)
            p_cal = cal.predict(preds_bag)
            oof['lgbm'][va_idx] = p_cal
            # Salviamo ultimo seed per inferenza (opzionale: salvare tutti e mediare in pred)
            joblib.dump(m, f"{config.MODEL_OUTPUT_DIR}lgbm_model_fold_{fold_no}.joblib")
            joblib.dump(cal, f"{config.MODEL_OUTPUT_DIR}lgbm_calibrator_fold_{fold_no}.joblib")

        # CATBOOST
        if 'cat' in config.MODEL_TYPES and CatBoostClassifier is not None:
            # Side-swap augmentation SOLO sul training fold
            X_tr_sw = swap_features_df(X_tr_base)
            y_tr_sw = 1 - y_tr_base
            X_tr = pd.concat([X_tr_base, X_tr_sw], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr_base, y_tr_sw], axis=0).reset_index(drop=True).astype(int)

            preds_bag = np.zeros(len(X_va))
            for sd in seeds:
                cm = CatBoostClassifier(**{**cat_params, 'random_seed': sd})
                cm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, use_best_model=True)
                preds_bag += cm.predict_proba(X_va)[:, 1]
            preds_bag /= max(1, len(seeds))
            # Calibrazione per-fold (isotonic)
            cal = IsotonicRegression(out_of_bounds='clip').fit(preds_bag, y_va)
            p_cal = cal.predict(preds_bag)
            oof['cat'][va_idx] = p_cal
            joblib.dump(cm, f"{config.MODEL_OUTPUT_DIR}cat_model_fold_{fold_no}.joblib")
            joblib.dump(cal, f"{config.MODEL_OUTPUT_DIR}cat_calibrator_fold_{fold_no}.joblib")
        elif 'cat' in config.MODEL_TYPES:
            print("[WARN] CatBoost non disponibile. Salto.")

        # XGBOOST
        if 'xgb' in config.MODEL_TYPES and xgb is not None:
            # Side-swap augmentation SOLO sul training fold
            X_tr_sw = swap_features_df(X_tr_base)
            y_tr_sw = 1 - y_tr_base
            X_tr = pd.concat([X_tr_base, X_tr_sw], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr_base, y_tr_sw], axis=0).reset_index(drop=True).astype(int)

            preds_bag = np.zeros(len(X_va))
            for sd in seeds:
                params = {**xgb_params, 'seed': sd}
                dtr = xgb.DMatrix(X_tr, label=y_tr)
                dva = xgb.DMatrix(X_va, label=y_va)
                xm = xgb.train(params, dtr, num_boost_round=6000, evals=[(dva, 'valid')],
                               early_stopping_rounds=200, verbose_eval=False)
                preds_bag += xm.predict(xgb.DMatrix(X_va))
            preds_bag /= max(1, len(seeds))
            # Calibrazione per-fold (isotonic)
            cal = IsotonicRegression(out_of_bounds='clip').fit(preds_bag, y_va)
            p_cal = cal.predict(preds_bag)
            oof['xgb'][va_idx] = p_cal
            xm.save_model(f"{config.MODEL_OUTPUT_DIR}xgb_model_fold_{fold_no}.json")
            joblib.dump(cal, f"{config.MODEL_OUTPUT_DIR}xgb_calibrator_fold_{fold_no}.joblib")
        elif 'xgb' in config.MODEL_TYPES:
            print("[WARN] XGBoost non disponibile. Salto.")

    # --- Ricerca pesi ensemble su OOF ---
    # 4) Cerchiamo pesi dell'ensemble su OOF con una griglia leggera e normalizzazione a somma=1
    def _normalize(d):
        s = sum(d[m] for m in config.MODEL_TYPES)
        return {m: (d[m] / s if s > 0 else 1.0/len(config.MODEL_TYPES)) for m in config.MODEL_TYPES}

    candidate_weights = []
    # Grid leggera
    grid = [0.2, 0.33, 0.5, 0.67, 0.8]
    for wl in grid:
        for wc in grid:
            for wx in grid:
                w = {'lgbm': wl, 'cat': wc, 'xgb': wx}
                w = _normalize(w)
                candidate_weights.append(w)
    # Aggiungi default
    candidate_weights.append(_normalize(config.ENSEMBLE_WEIGHTS.copy()))

    best_acc = -1.0
    best_thr = 0.5
    best_w = candidate_weights[0]
    best_probs = None
    for w in candidate_weights:
        probs = np.zeros(len(y))
        for m in config.MODEL_TYPES:
            probs += w.get(m, 0.0) * oof[m]
        # Cerchiamo la soglia che massimizza accuracy OOF in un range ragionevole
        thr_list = np.linspace(0.2, 0.8, 61)
        accs = [accuracy_score(y, (probs > t).astype(int)) for t in thr_list]
        idx = int(np.argmax(accs))
        if accs[idx] > best_acc:
            best_acc = float(accs[idx])
            best_thr = float(thr_list[idx])
            best_w = w
            best_probs = probs

    # 5) Metriche OOF con i pesi ottimi
    probs = best_probs
    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)
    print("\nEnsemble OOF:")
    print(f"Accuracy: {best_acc:.5f} @thr={best_thr:.3f}")
    print(f"AUC: {auc:.5f} | LogLoss: {ll:.5f}")

    # 6) Salva soglia e pesi dell'ensemble
    with open(config.THRESHOLD_PATH, 'w', encoding='utf-8') as f:
        json.dump({'threshold': best_thr, 'oof_accuracy': best_acc}, f, indent=2)
    with open(config.ENSEMBLE_WEIGHTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(best_w, f, indent=2)

    # 7) Meta-calibrazione dell'ensemble (isotonic su OOF)
    ens_cal = IsotonicRegression(out_of_bounds='clip').fit(probs, y)
    joblib.dump(ens_cal, config.ENSEMBLE_CALIBRATOR_PATH)
    print("Calibratore ensemble salvato.")


if __name__ == '__main__':
    train_ensemble()
