"""
Prediction ensemble V2: carica feature di test, applica TTA tramite side-swap,
carica modelli e calibratori per-fold, combina con pesi normalizzati, applica
meta-calibrazione (se presente) e soglia ottima OOF, infine salva la submission.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from feature_engineering_v2 import create_feature_df
import codice.config as config

try:
    import xgboost as xgb
except (ImportError, ModuleNotFoundError):
    xgb = None

def swap_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Scambia p1_ con p2_ e nega diff_ per TTA side-swap in inferenza.

    Usiamo questa trasformazione per ottenere una seconda vista del match dal
    lato avversario e poi mediare: p = 0.5 * (p + (1 - p_swapped)).
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

def predict_ensemble():
    """Esegue la predizione ensemble su test con TTA, calibrazione e soglia OOF."""
    print("Predict ensemble: ", config.MODEL_TYPES)
    test_df = create_feature_df(config.TEST_FILE_PATH, max_turns=config.MAX_TURNS)
    X_test = test_df.drop(columns=['battle_id'])
    X_test_sw = swap_features_df(X_test)

    # Pesi normalizzati (prova a caricare i pesi ottimizzati)
    w = None
    try:
        with open(config.ENSEMBLE_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            w = json.load(f)
            print("Pesi ensemble caricati da file.")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        w = config.ENSEMBLE_WEIGHTS.copy()
        print("Uso pesi ensemble da config.")
    s = sum(w.get(m, 0.0) for m in config.MODEL_TYPES)
    for k in w:
        w[k] = w[k] / s if s > 0 else 1.0/len(config.MODEL_TYPES)

    preds_accum = np.zeros(len(X_test))

    for fold in range(1, config.N_SPLITS + 1):
        # LGBM
        if 'lgbm' in config.MODEL_TYPES:
            m = joblib.load(f"{config.MODEL_OUTPUT_DIR}lgbm_model_fold_{fold}.joblib")
            cal = joblib.load(f"{config.MODEL_OUTPUT_DIR}lgbm_calibrator_fold_{fold}.joblib")
            p = cal.predict(m.predict_proba(X_test)[:, 1])
            ps = cal.predict(m.predict_proba(X_test_sw)[:, 1])
            preds_accum += w['lgbm'] * 0.5 * (p + (1.0 - ps))

        # CATBOOST
        if 'cat' in config.MODEL_TYPES:
            cat_path = f"{config.MODEL_OUTPUT_DIR}cat_model_fold_{fold}.joblib"
            cal_path = f"{config.MODEL_OUTPUT_DIR}cat_calibrator_fold_{fold}.joblib"
            if os.path.exists(cat_path) and os.path.exists(cal_path):
                cm = joblib.load(cat_path)
                cal = joblib.load(cal_path)
                p = cal.predict(cm.predict_proba(X_test)[:, 1])
                ps = cal.predict(cm.predict_proba(X_test_sw)[:, 1])
                preds_accum += w['cat'] * 0.5 * (p + (1.0 - ps))
            else:
                print("[WARN] CatBoost artifacts mancanti per fold", fold)

        # XGBOOST
        if 'xgb' in config.MODEL_TYPES and xgb is not None:
            xgb_path = f"{config.MODEL_OUTPUT_DIR}xgb_model_fold_{fold}.json"
            cal_path = f"{config.MODEL_OUTPUT_DIR}xgb_calibrator_fold_{fold}.joblib"
            if os.path.exists(xgb_path) and os.path.exists(cal_path):
                xm = xgb.Booster()
                xm.load_model(xgb_path)
                cal = joblib.load(cal_path)
                p = cal.predict(xm.predict(xgb.DMatrix(X_test)))
                ps = cal.predict(xm.predict(xgb.DMatrix(X_test_sw)))
                preds_accum += w['xgb'] * 0.5 * (p + (1.0 - ps))
            else:
                print("[WARN] XGBoost artifacts mancanti per fold", fold)

    avg_preds = preds_accum / config.N_SPLITS
    # Meta-calibrazione dell'ensemble (se disponibile)
    try:
        cal = joblib.load(config.ENSEMBLE_CALIBRATOR_PATH)
        avg_preds = cal.predict(avg_preds)
        print("Applicata calibrazione isotonic ensemble.")
    except (FileNotFoundError, OSError):
        pass
    avg_preds = np.clip(avg_preds, 1e-4, 1-1e-4)

    # Soglia
    thr = 0.5
    try:
        with open(config.THRESHOLD_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            thr = float(data.get('threshold', 0.5))
            print(f"Soglia caricata: {thr:.3f}")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
        print("Soglia non trovata, uso 0.5")

    preds_class = (avg_preds > thr).astype(int)
    submission_df = pd.DataFrame({'battle_id': test_df['battle_id'], 'player_won': preds_class})
    os.makedirs(os.path.dirname(config.ENSAMBLE_SUBMISSION_FILE_PATH), exist_ok=True)
    submission_df.to_csv(config.ENSAMBLE_SUBMISSION_FILE_PATH, index=False)
    print(f"Submission salvata in {config.ENSAMBLE_SUBMISSION_FILE_PATH}")


if __name__ == '__main__':
    predict_ensemble()
