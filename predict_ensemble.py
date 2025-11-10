"""
Pokemon Battle Predictor - Ensemble Prediction Module

Questo modulo genera predizioni su battaglie test usando l'ensemble di modelli trainati.

Funzionalità:
    1. PREDICTION PIPELINE:
       - Carica feature da test.jsonl (339 features)
       - Genera predizioni con tutti i modelli trainati (LGBM, Cat, XGB)
       - Media predizioni cross-validation (10 folds)
       - Side-swap averaging per robustezza
       - Calibrazione isotonica per-fold e finale
    
    2. ENSEMBLE METHODS:
       - Weighted Average: combina modelli con pesi ottimizzati
       - Stacking: usa meta-learner trainato su OOF predictions
       - Selezione automatica del metodo salvato durante training
    
    3. OUTPUT:
       - submission/ensemble_submission.csv con formato (battle_id, player_won)
       - Applica threshold ottimale trovato in training

Input:
    - config.TEST_FILE_PATH: battaglie test (JSONL)
    - models/*_fold_*.joblib: modelli trainati per ogni fold
    - models/ensemble_weights.json: configurazione ensemble
    - models/best_threshold.json: soglia ottimale

Output:
    - submission/ensemble_submission.csv: predizioni finali
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from feature_engineering import create_feature_df
import config as config

try:
    import xgboost as xgb
except Exception:
    xgb = None


def swap_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Scambia prospettiva P1 ↔ P2 per side-swap averaging.
    
    Usato durante predizione per generare predizioni da entrambe le prospettive
    e mediarle per maggiore robustezza.
    
    Args:
        df: DataFrame con feature delle battaglie
        
    Returns:
        DataFrame con prospettiva scambiata
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
    """Genera predizioni ensemble su dataset test.
    
    Pipeline completa:
    1. Carica configurazione ensemble (metodo, pesi, threshold)
    2. Genera feature da test.jsonl (339 features)
    3. Per ogni fold (1-10):
       - Carica modello + calibratore
       - Predice su X_test e X_test_swapped
       - Media le due predizioni (side-swap averaging)
    4. Media predizioni di tutti i folds
    5. Combina modelli con weighted average o stacking
    6. Applica meta-calibrazione isotonica finale
    7. Binarizza con threshold ottimale
    8. Salva submission CSV
    
    Output:
        submission/ensemble_submission.csv con predizioni finali
    """
    print("="*60)
    print("=== ENSEMBLE PREDICTION ===")
    print("="*60)
    
    # Carica configurazione ensemble (metodo e pesi salvati durante training)
    ensemble_config = None
    try:
        with open(config.ENSEMBLE_WEIGHTS_PATH, 'r', encoding='utf-8') as f:
            ensemble_config = json.load(f)
            ensemble_method = ensemble_config.get('method', 'weighted_average')
            print(f"Ensemble method: {ensemble_method}")
    except Exception as e:
        print(f"[WARN] Impossibile caricare configurazione ensemble: {e}")
        ensemble_method = 'weighted_average'
        ensemble_config = {}
    
    print("Model types:", config.MODEL_TYPES)
    
    # Genera feature da test.jsonl (339 features identiche al training)
    test_df = create_feature_df(config.TEST_FILE_PATH, max_turns=config.MAX_TURNS)
    X_test = test_df.drop(columns=['battle_id']).astype(np.float32)
    X_test_sw = swap_features_df(X_test)

    # Accumulatori per predizioni di ogni fold
    preds_per_model = {m: [] for m in config.MODEL_TYPES}

    # Carica e predice con ogni fold (cross-validation averaging)
    for fold in range(1, config.N_SPLITS + 1):
        # LightGBM
        if 'lgbm' in config.MODEL_TYPES:
            try:
                m = joblib.load(f"{config.MODEL_OUTPUT_DIR}lgbm_model_fold_{fold}.joblib")
                cal = joblib.load(f"{config.MODEL_OUTPUT_DIR}lgbm_calibrator_fold_{fold}.joblib")
                # Predici da entrambe le prospettive
                p = cal.predict(m.predict_proba(X_test)[:, 1])
                ps = cal.predict(m.predict_proba(X_test_sw)[:, 1])
                # Side-swap averaging: media(p, 1-ps)
                preds_per_model['lgbm'].append(0.5 * (p + (1.0 - ps)))
            except Exception as e:
                print(f"[WARN] LGBM fold {fold}: {e}")

        # CatBoost
        if 'cat' in config.MODEL_TYPES:
            cat_path = f"{config.MODEL_OUTPUT_DIR}cat_model_fold_{fold}.joblib"
            cal_path = f"{config.MODEL_OUTPUT_DIR}cat_calibrator_fold_{fold}.joblib"
            if os.path.exists(cat_path) and os.path.exists(cal_path):
                try:
                    cm = joblib.load(cat_path)
                    cal = joblib.load(cal_path)
                    p = cal.predict(cm.predict_proba(X_test)[:, 1])
                    ps = cal.predict(cm.predict_proba(X_test_sw)[:, 1])
                    preds_per_model['cat'].append(0.5 * (p + (1.0 - ps)))
                except Exception as e:
                    print(f"[WARN] CatBoost fold {fold}: {e}")
            else:
                print(f"[WARN] CatBoost artifacts mancanti per fold {fold}")

        # XGBoost
        if 'xgb' in config.MODEL_TYPES and xgb is not None:
            xgb_path = f"{config.MODEL_OUTPUT_DIR}xgb_model_fold_{fold}.json"
            cal_path = f"{config.MODEL_OUTPUT_DIR}xgb_calibrator_fold_{fold}.joblib"
            if os.path.exists(xgb_path) and os.path.exists(cal_path):
                try:
                    xm = xgb.Booster()
                    xm.load_model(xgb_path)
                    cal = joblib.load(cal_path)
                    p = cal.predict(xm.predict(xgb.DMatrix(X_test)))
                    ps = cal.predict(xm.predict(xgb.DMatrix(X_test_sw)))
                    preds_per_model['xgb'].append(0.5 * (p + (1.0 - ps)))
                except Exception as e:
                    print(f"[WARN] XGBoost fold {fold}: {e}")
            else:
                print(f"[WARN] XGBoost artifacts mancanti per fold {fold}")

        # Logistic Regression (se presente)
        if 'logreg' in config.MODEL_TYPES:
            lr_path = f"{config.MODEL_OUTPUT_DIR}logreg_model_fold_{fold}.joblib"
            cal_path = f"{config.MODEL_OUTPUT_DIR}logreg_calibrator_fold_{fold}.joblib"
            if os.path.exists(lr_path) and os.path.exists(cal_path):
                try:
                    lr = joblib.load(lr_path)
                    cal = joblib.load(cal_path)
                    p = cal.predict(lr.predict_proba(X_test)[:, 1])
                    ps = cal.predict(lr.predict_proba(X_test_sw)[:, 1])
                    preds_per_model['logreg'].append(0.5 * (p + (1.0 - ps)))
                except Exception as e:
                    print(f"[WARN] LogisticRegression fold {fold}: {e}")
            else:
                print(f"[WARN] LogisticRegression artifacts mancanti per fold {fold}")

    # Media predizioni cross-validation per ogni modello
    avg_preds_per_model = {}
    for m in config.MODEL_TYPES:
        if len(preds_per_model[m]) > 0:
            avg_preds_per_model[m] = np.mean(preds_per_model[m], axis=0)
            print(f"  {m.upper()}: {len(preds_per_model[m])} folds averaged")
        else:
            print(f"  [WARN] {m.upper()}: nessun fold disponibile")

    # Combina predizioni dei modelli: Stacking o Weighted Average
    if ensemble_method == 'stacking':
        print("\n🔥 Using STACKING ensemble...")
        try:
            # Carica meta-learner (LogisticRegression trainato su OOF)
            meta_learner = joblib.load(ensemble_config['meta_learner_path'])
            model_types_stack = ensemble_config.get('model_types', list(avg_preds_per_model.keys()))
            
            # Costruisci matrice input: [n_samples, n_models]
            X_stack = []
            for m in model_types_stack:
                if m in avg_preds_per_model:
                    X_stack.append(avg_preds_per_model[m])
            
            if len(X_stack) > 0:
                X_stack = np.column_stack(X_stack)
                avg_preds = meta_learner.predict_proba(X_stack)[:, 1]
                print(f"  Stacking applied with {len(model_types_stack)} models")
            else:
                print("[ERROR] Nessun modello disponibile per stacking!")
                avg_preds = np.zeros(len(X_test))
        except Exception as e:
            print(f"[ERROR] Stacking fallito: {e}")
            print("  Fallback a weighted average...")
            ensemble_method = 'weighted_average'
    
    if ensemble_method == 'weighted_average':
        print("\n📊 Using WEIGHTED AVERAGE ensemble...")
        # Carica pesi ottimizzati (salvati durante training)
        w = ensemble_config.get('weights', None)
        if w is None:
            try:
                w = config.ENSEMBLE_WEIGHTS.copy()
                print("  Uso pesi da config.")
            except Exception:
                w = {m: 1.0/len(config.MODEL_TYPES) for m in config.MODEL_TYPES}
                print("  Uso pesi uniformi.")
        else:
            print("  Uso pesi ottimizzati da file.")
        
        # Normalizza pesi per garantire somma=1
        s = sum(w.get(m, 0.0) for m in config.MODEL_TYPES if m in avg_preds_per_model)
        for k in w:
            w[k] = w[k] / s if s > 0 else 1.0/len(config.MODEL_TYPES)
        
        print(f"  Weights: {dict((k, round(v, 3)) for k, v in w.items() if k in config.MODEL_TYPES)}")
        
        # Combina predizioni con weighted average
        avg_preds = np.zeros(len(X_test))
        for m in config.MODEL_TYPES:
            if m in avg_preds_per_model:
                avg_preds += w.get(m, 0.0) * avg_preds_per_model[m]

    # Calibrazione isotonica finale dell'ensemble
    try:
        cal = joblib.load(config.ENSEMBLE_CALIBRATOR_PATH)
        avg_preds = cal.predict(avg_preds)
        print("✅ Applicata meta-calibrazione isotonic ensemble.")
    except Exception:
        print("[WARN] Meta-calibratore non trovato, salto calibrazione.")
    
    # Clip probabilità per stabilità numerica
    avg_preds = np.clip(avg_preds, 1e-4, 1-1e-4)

    # Carica threshold ottimale (trovato durante training)
    thr = 0.5
    try:
        with open(config.THRESHOLD_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            thr = float(data.get('threshold', 0.5))
            print(f"✅ Soglia caricata: {thr:.3f}")
    except Exception:
        print("[WARN] Soglia non trovata, uso 0.5")

    # Binarizza predizioni con threshold
    preds_class = (avg_preds > thr).astype(int)
    
    # Salva submission
    submission_df = pd.DataFrame({'battle_id': test_df['battle_id'], 'player_won': preds_class})
    os.makedirs(os.path.dirname(config.ENSAMBLE_SUBMISSION_FILE_PATH), exist_ok=True)
    submission_df.to_csv(config.ENSAMBLE_SUBMISSION_FILE_PATH, index=False)
    
    print("="*60)
    print(f"✅ Submission salvata: {config.ENSAMBLE_SUBMISSION_FILE_PATH}")
    print(f"   Predizioni: {len(preds_class)} battaglie")
    print(f"   Class 1: {preds_class.sum()} ({100*preds_class.mean():.1f}%)")
    print("="*60)


if __name__ == '__main__':
    predict_ensemble()
