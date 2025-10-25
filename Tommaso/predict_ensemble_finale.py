# predict_ensemble.py

import os
import json
import joblib
import numpy as np
import pandas as pd

from feature_engineering_finale import create_feature_df
import config as config

try:
    import xgboost as xgb
except Exception:
    xgb = None


def swap_features_df(df: pd.DataFrame) -> pd.DataFrame:
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
    """Predice usando l'ensemble con supporto sia per stacking che weighted average."""
    print("="*60)
    print("=== ENSEMBLE PREDICTION V4 ===")
    print("="*60)
    
    # Carica configurazione ensemble
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
    
    # Carica dati test e crea features (con interaction features V4)
    test_df = create_feature_df(config.TEST_FILE_PATH, max_turns=config.MAX_TURNS)
    X_test = test_df.drop(columns=['battle_id']).astype(np.float32)
    X_test_sw = swap_features_df(X_test)

    # Accumulatori per predizioni per-fold
    preds_per_model = {m: [] for m in config.MODEL_TYPES}  # Lista di predizioni per fold

    # Ciclo sui fold
    for fold in range(1, config.N_SPLITS + 1):
        # LGBM
        if 'lgbm' in config.MODEL_TYPES:
            try:
                m = joblib.load(f"{config.MODEL_OUTPUT_DIR}lgbm_model_fold_{fold}.joblib")
                cal = joblib.load(f"{config.MODEL_OUTPUT_DIR}lgbm_calibrator_fold_{fold}.joblib")
                p = cal.predict(m.predict_proba(X_test)[:, 1])
                ps = cal.predict(m.predict_proba(X_test_sw)[:, 1])
                preds_per_model['lgbm'].append(0.5 * (p + (1.0 - ps)))
            except Exception as e:
                print(f"[WARN] LGBM fold {fold}: {e}")

        # CATBOOST
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

        # XGBOOST
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

    # Media delle predizioni dei fold per ogni modello
    avg_preds_per_model = {}
    for m in config.MODEL_TYPES:
        if len(preds_per_model[m]) > 0:
            avg_preds_per_model[m] = np.mean(preds_per_model[m], axis=0)
            print(f"  {m.upper()}: {len(preds_per_model[m])} folds averaged")
        else:
            print(f"  [WARN] {m.upper()}: nessun fold disponibile")

    # --- Ensemble finale: Stacking o Weighted Average ---
    if ensemble_method == 'stacking':
        print("\n🔥 Using STACKING ensemble...")
        try:
            meta_learner = joblib.load(ensemble_config['meta_learner_path'])
            model_types_stack = ensemble_config.get('model_types', list(avg_preds_per_model.keys()))
            
            # Costruisci matrice di input per meta-learner
            X_stack = []
            for m in model_types_stack:
                if m in avg_preds_per_model:
                    X_stack.append(avg_preds_per_model[m])
            
            if len(X_stack) > 0:
                X_stack = np.column_stack(X_stack)  # (n_samples, n_models)
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
        # Carica pesi (prova da config, altrimenti usa default)
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
        
        # Normalizza pesi
        s = sum(w.get(m, 0.0) for m in config.MODEL_TYPES if m in avg_preds_per_model)
        for k in w:
            w[k] = w[k] / s if s > 0 else 1.0/len(config.MODEL_TYPES)
        
        print(f"  Weights: {dict((k, round(v, 3)) for k, v in w.items() if k in config.MODEL_TYPES)}")
        
        avg_preds = np.zeros(len(X_test))
        for m in config.MODEL_TYPES:
            if m in avg_preds_per_model:
                avg_preds += w.get(m, 0.0) * avg_preds_per_model[m]

    # Meta-calibrazione dell'ensemble (se disponibile)
    try:
        cal = joblib.load(config.ENSEMBLE_CALIBRATOR_PATH)
        avg_preds = cal.predict(avg_preds)
        print("✅ Applicata meta-calibrazione isotonic ensemble.")
    except Exception:
        print("[WARN] Meta-calibratore non trovato, salto calibrazione.")
    
    avg_preds = np.clip(avg_preds, 1e-4, 1-1e-4)

    # Soglia
    thr = 0.5
    try:
        with open(config.THRESHOLD_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            thr = float(data.get('threshold', 0.5))
            print(f"✅ Soglia caricata: {thr:.3f}")
    except Exception:
        print("[WARN] Soglia non trovata, uso 0.5")

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
