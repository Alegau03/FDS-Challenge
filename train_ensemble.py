"""
Pokemon Battle Predictor - Ensemble Training Module

Questo modulo addestra un ensemble di gradient boosting models (LGBM, CatBoost, XGBoost) 
per predire il vincitore di battaglie Pokemon competitive.

Architettura Ensemble:
    1. BASE MODELS:
       - LightGBM: Gradient boosting veloce con class balancing
       - CatBoost: Ottimizzato per feature categoriche
       - XGBoost: Gradient boosting robusto con regolarizzazione
    
    2. TRAINING STRATEGY:
       - Stratified K-Fold Cross-Validation (10 folds)
       - Side-swap data augmentation (raddoppia training set)
       - Seed bagging per ridurre varianza
       - Isotonic calibration delle probabilità
    
    3. ENSEMBLE METHODS:
       - Weighted Average: Combina modelli con pesi ottimizzati
       - Stacking: Meta-learner (LogisticRegression) su OOF predictions
       - Automatic selection basata su accuracy OOF
    
    4. OTTIMIZZAZIONE:
       - Ricerca threshold ottimale (0.2-0.8)
       - Grid search pesi ensemble
       - Calibrazione isotonica finale

Input:
    - config.TRAIN_FILE_PATH: battaglie di train (JSONL)
    - config.MODEL_TYPES: lista modelli da usare (['lgbm', 'cat', 'xgb'])
    - config.BEST_PARAMS_PATH: iperparametri ottimizzati (JSON)

Output:
    - models/: modelli trained per ogni fold (.joblib / .json)
    - models/best_threshold.json: soglia ottimale e configurazione
    - models/ensemble_weights.json: pesi ensemble e metodo selezionato
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import config as config
from feature_engineering import create_feature_df

# Optional imports for models

import lightgbm as lgb

try:

    from catboost import CatBoostClassifier

except Exception:

    CatBoostClassifier = None

try:

    import xgboost as xgb

except Exception:

    xgb = None


def swap_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Scambia prospettiva P1 ↔ P2 per data augmentation.
    
    Inverte tutte le feature con prefix 'p1_' e 'p2_', e nega le 'diff_' features.
    Usato per side-swap augmentation: raddoppia il training set generando il matchup 
    speculare di ogni battaglia.
    
    Args:
        df: DataFrame con feature delle battaglie
        
    Returns:
        DataFrame con prospettiva scambiata (P1 diventa P2 e viceversa)
        
    Example:
        >>> X_swapped = swap_features_df(X_train)
        >>> y_swapped = 1 - y_train
        >>> X_augmented = pd.concat([X_train, X_swapped])
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



def _compute_scale_pos_weight(y: pd.Series) -> float:
    """Calcola peso per class imbalancing (neg/pos ratio).
    
    Restituisce il rapporto tra classi negative e positive,
    utile per bilanciare dataset sbilanciati in modelli GBDT.
    
    Args:
        y: Serie pandas con label binarie (0/1)
        
    Returns:
        Float: rapporto neg/pos (min 1.0)
    """
    pos = float((y == 1).sum())

    neg = float((y == 0).sum())

    if pos <= 0:

        return 1.0

    return max(1.0, neg / max(1.0, pos))


def train_ensemble():
    """Addestra ensemble di modelli GBDT per predizione battaglie Pokemon.
    
    Pipeline completa di training:
    1. Carica dati e genera feature (339 feature totali)
    2. Stratified K-Fold Cross-Validation (10 folds)
    3. Side-swap augmentation per bilanciamento
    4. Train LightGBM, CatBoost, XGBoost con calibrazione isotonica
    5. Ottimizzazione pesi ensemble (grid search)
    6. Selezione metodo: Weighted Average vs Stacking
    7. Salvataggio modelli, pesi, threshold ottimale
    
    Ogni modello viene trainato con:
    - Seed bagging (media di più seed per ridurre varianza)
    - Early stopping su validation set
    - Isotonic calibration delle probabilità
    
    Output salvati:
        - models/lgbm_model_fold_*.joblib: modelli LightGBM
        - models/cat_model_fold_*.joblib: modelli CatBoost
        - models/xgb_model_fold_*.json: modelli XGBoost
        - models/*_calibrator_fold_*.joblib: calibratori isotonic
        - models/best_threshold.json: threshold ottimale e metodo
        - models/ensemble_weights.json: pesi ensemble
    """

    print("Training ensemble: ", config.MODEL_TYPES)

    os.makedirs(config.MODEL_OUTPUT_DIR, exist_ok=True)

    # Reproducibility

    np.random.seed(getattr(config, 'RANDOM_STATE', 42))

    # Carica dati e crea features

    df = create_feature_df(config.TRAIN_FILE_PATH, max_turns=config.MAX_TURNS)

    y = df['player_won'].fillna(0).astype(int)

    X = df.drop(columns=['battle_id', 'player_won']).astype(np.float32)
    # Repeated Stratified K-Fold

    if config.N_REPEATS > 1:

        splitter = RepeatedStratifiedKFold(n_splits=config.N_SPLITS, n_repeats=config.N_REPEATS, random_state=config.RANDOM_STATE)

    else:

        splitter = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)

    # Probs OOF per ogni modello (solo su dataset originale)

    oof = {m: np.zeros(len(y)) for m in config.MODEL_TYPES}

    # Per stacking/meta-learner e pesi

    fold_indices = []

    # Carica best params LGBM se esistono

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

                          tree_method='hist', verbosity=0, n_jobs=-1)



    fold_no = 0

    for tr_idx, va_idx in splitter.split(X, y.values):

        fold_no += 1

        print(f"Fold {fold_no}")

        X_tr_base, y_tr_base = X.iloc[tr_idx], y.iloc[tr_idx]

        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        fold_indices.append((tr_idx, va_idx))



        # Seed bagging: media predictions di più semi

        seeds = [config.RANDOM_STATE + s for s in range(max(1, config.SEED_BAGGING))]



        # Imbalance handling

        spw = _compute_scale_pos_weight(y_tr_base)

        # LIGHTGBM

        if 'lgbm' in config.MODEL_TYPES:

            params_base = lgb_params or {

                'objective': 'binary', 'metric': 'logloss', 'random_state': config.RANDOM_STATE,

                'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 128,

                'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 3,

            }

            # add class_weight balanced if not present

            if 'class_weight' not in params_base:

                params_base['class_weight'] = 'balanced'

            # Augment SOLO training fold con side-swap

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

            cal = IsotonicRegression(out_of_bounds='clip').fit(preds_bag, y_va)

            p_cal = cal.predict(preds_bag)

            oof['lgbm'][va_idx] = p_cal

            # Per-fold metrics (calibrated)

            print(f"  LGBM  fold{fold_no}: AUC={roc_auc_score(y_va, p_cal):.5f} | LogLoss={log_loss(y_va, p_cal):.5f}")

            # Salva solo l'ultimo seed per inferenza (opzionale: salvare tutti e mediare anche in pred)

            joblib.dump(m, f"{config.MODEL_OUTPUT_DIR}lgbm_model_fold_{fold_no}.joblib")

            joblib.dump(cal, f"{config.MODEL_OUTPUT_DIR}lgbm_calibrator_fold_{fold_no}.joblib")



        # CATBOOST

        if 'cat' in config.MODEL_TYPES and CatBoostClassifier is not None:

            # Augment SOLO training fold

            X_tr_sw = swap_features_df(X_tr_base)

            y_tr_sw = 1 - y_tr_base

            X_tr = pd.concat([X_tr_base, X_tr_sw], axis=0).reset_index(drop=True)

            y_tr = pd.concat([y_tr_base, y_tr_sw], axis=0).reset_index(drop=True).astype(int)



            preds_bag = np.zeros(len(X_va))

            for sd in seeds:

                cat_cfg = {**cat_params, 'random_seed': sd}

                # add scale_pos_weight if not present

                if 'scale_pos_weight' not in cat_cfg:

                    cat_cfg['scale_pos_weight'] = spw

                cm = CatBoostClassifier(**cat_cfg)

                cm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, use_best_model=True)

                preds_bag += cm.predict_proba(X_va)[:, 1]

            preds_bag /= max(1, len(seeds))

            cal = IsotonicRegression(out_of_bounds='clip').fit(preds_bag, y_va)

            p_cal = cal.predict(preds_bag)

            oof['cat'][va_idx] = p_cal

            print(f"  CAT   fold{fold_no}: AUC={roc_auc_score(y_va, p_cal):.5f} | LogLoss={log_loss(y_va, p_cal):.5f}")

            joblib.dump(cm, f"{config.MODEL_OUTPUT_DIR}cat_model_fold_{fold_no}.joblib")

            joblib.dump(cal, f"{config.MODEL_OUTPUT_DIR}cat_calibrator_fold_{fold_no}.joblib")

        elif 'cat' in config.MODEL_TYPES:

            print("[WARN] CatBoost non disponibile. Salto.")

        # XGBOOST

        if 'xgb' in config.MODEL_TYPES and xgb is not None:

            # Augment SOLO training fold
            X_tr_sw = swap_features_df(X_tr_base)
            y_tr_sw = 1 - y_tr_base
            X_tr = pd.concat([X_tr_base, X_tr_sw], axis=0).reset_index(drop=True)
            y_tr = pd.concat([y_tr_base, y_tr_sw], axis=0).reset_index(drop=True).astype(int)

            preds_bag = np.zeros(len(X_va))

            for sd in seeds:

                params = {**xgb_params, 'seed': sd}

                if 'scale_pos_weight' not in params:

                    params['scale_pos_weight'] = spw

                dtr = xgb.DMatrix(X_tr, label=y_tr)

                dva = xgb.DMatrix(X_va, label=y_va)

                xm = xgb.train(params, dtr, num_boost_round=6000, evals=[(dva, 'valid')],

                               early_stopping_rounds=200, verbose_eval=False)

                preds_bag += xm.predict(xgb.DMatrix(X_va))

            preds_bag /= max(1, len(seeds))

            cal = IsotonicRegression(out_of_bounds='clip').fit(preds_bag, y_va)

            p_cal = cal.predict(preds_bag)

            oof['xgb'][va_idx] = p_cal

            print(f"  XGB   fold{fold_no}: AUC={roc_auc_score(y_va, p_cal):.5f} | LogLoss={log_loss(y_va, p_cal):.5f}")

            xm.save_model(f"{config.MODEL_OUTPUT_DIR}xgb_model_fold_{fold_no}.json")

            joblib.dump(cal, f"{config.MODEL_OUTPUT_DIR}xgb_calibrator_fold_{fold_no}.joblib")

        elif 'xgb' in config.MODEL_TYPES:
            print("[WARN] XGBoost non disponibile ma richiesto.")

    print("\n" + "="*60)
    print("="*60)

    # Creiamo matrice OOF (n_samples, n_models) con predizioni di tutti i modelli
    oof_stack = []
    model_names_stack = []
    for m in config.MODEL_TYPES:
        if m in oof:
            oof_stack.append(oof[m])
            model_names_stack.append(m)


    use_stacking = len(oof_stack) > 0
    meta_learner = None
    meta_probs = None
    meta_best_thr = 0.5
    meta_best_acc = 0.0


    if use_stacking:
        X_stack = np.column_stack(oof_stack)  # Shape: (n_samples, n_models)

        # Meta-learner: LogisticRegression con L2 regularization
        meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=config.RANDOM_STATE,
            max_iter=1000,
            solver='lbfgs'
        )

        

        # Train su OOF predictions
        meta_learner.fit(X_stack, y)
        meta_probs = meta_learner.predict_proba(X_stack)[:, 1]

        

        # Metriche
        meta_auc = roc_auc_score(y, meta_probs)
        meta_ll = log_loss(y, meta_probs)

        

        # Trova soglia ottimale
        thr_list = np.linspace(0.2, 0.8, 61)
        accs = [accuracy_score(y, (meta_probs > t).astype(int)) for t in thr_list]
        best_thr_idx = int(np.argmax(accs))
        meta_best_thr = thr_list[best_thr_idx]
        meta_best_acc = accs[best_thr_idx]

        print(f"Meta-Learner OOF Results:")
        print(f"  Accuracy: {meta_best_acc:.5f} @ threshold={meta_best_thr:.3f}")
        print(f"  AUC: {meta_auc:.5f} | LogLoss: {meta_ll:.5f}")
        print(f"  Coefficients: {dict(zip(model_names_stack, np.round(meta_learner.coef_[0], 4)))}")
        print(f"  Intercept: {meta_learner.intercept_[0]:.4f}")

        # Salva meta-learner
        joblib.dump(meta_learner, f"{config.MODEL_OUTPUT_DIR}meta_learner_stacking.joblib")
        print(f"  Meta-learner salvato in: {config.MODEL_OUTPUT_DIR}meta_learner_stacking.joblib")
    else:
        print("[WARN] Nessun modello OOF disponibile per stacking.")

    print("="*60)

    

    # --- Ricerca pesi ensemble su OOF (weighted average baseline) ---
    print("\n" + "="*60)
    print("=== WEIGHTED AVERAGE ENSEMBLE ===")
    print("="*60)



    # --- Ricerca pesi ensemble su OOF ---
    def _normalize(d):
        s = sum(d[m] for m in config.MODEL_TYPES)
        return {m: (d[m] / s if s > 0 else 1.0/len(config.MODEL_TYPES)) for m in config.MODEL_TYPES}


    candidate_weights = []
    # Grid leggera
    grid = [0.2, 0.33, 0.5, 0.67, 0.8]

    for wl in grid:
        for wc in grid:
            wx = 1.0 - wl - wc
            if wx >= 0:
                candidate_weights.append(_normalize({'lgbm': wl, 'cat': wc, 'xgb': wx}))

            for wx in grid:
                w = {'lgbm': wl, 'cat': wc, 'xgb': wx}
                w = _normalize(w)
                candidate_weights.append(w)

    # Aggiungi default
    candidate_weights.append(_normalize(config.ENSEMBLE_WEIGHTS.copy()))

    # Aggiungi estremi mono-modello
    for m in config.MODEL_TYPES:
        w = {mm: (1.0 if mm == m else 0.0) for mm in ['lgbm', 'cat', 'xgb']}
        candidate_weights.append(_normalize(w))

    best_acc = -1.0
    best_thr = 0.5
    best_w = candidate_weights[0]
    best_probs = None

    for w in candidate_weights:
        probs = np.zeros(len(y))
        for m in config.MODEL_TYPES:
            probs += w.get(m, 0.0) * oof[m]
        thr_list = np.linspace(0.2, 0.8, 61)
        accs = [accuracy_score(y, (probs > t).astype(int)) for t in thr_list]
        idx = int(np.argmax(accs))

        if accs[idx] > best_acc:
            best_acc = float(accs[idx])
            best_thr = float(thr_list[idx])
            best_w = w
            best_probs = probs



    # Metriche con pesi ottimi
    probs = best_probs
    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)

    print(f"\nWeighted Average OOF:")
    print(f"  Accuracy: {best_acc:.5f} @ threshold={best_thr:.3f}")
    print(f"  AUC: {auc:.5f} | LogLoss: {ll:.5f}")
    print(f"  Optimal weights: {dict((k, round(v, 3)) for k, v in best_w.items() if k in config.MODEL_TYPES)}")
    print("="*60)

    # --- Scelta finale: Stacking vs Weighted Average ---
    print("\n" + "="*60)

    print("=== ENSEMBLE METHOD SELECTION ===")
    if use_stacking and meta_best_acc >= best_acc:
        print(f"USING STACKING (Acc: {meta_best_acc:.5f} vs {best_acc:.5f})")
        final_method = 'stacking'
        final_thr = meta_best_thr
        final_acc = meta_best_acc
        final_probs = meta_probs

        ensemble_config = {
            'method': 'stacking',
            'threshold': final_thr,
            'oof_accuracy': final_acc,
            'meta_learner_path': f"{config.MODEL_OUTPUT_DIR}meta_learner_stacking.joblib",
            'model_types': model_names_stack
        }

    else:
        print(f"USING WEIGHTED AVERAGE (Acc: {best_acc:.5f})")
        final_method = 'weighted_average'
        final_thr = best_thr
        final_acc = best_acc
        final_probs = best_probs

        ensemble_config = {
            'method': 'weighted_average',
            'threshold': final_thr,
            'oof_accuracy': final_acc,
            'weights': best_w
        }

    print("="*60)

    # Salva soglia e configurazione ensemble
    with open(config.THRESHOLD_PATH, 'w', encoding='utf-8') as f:
        json.dump({'threshold': final_thr, 'oof_accuracy': final_acc, 'method': final_method}, f, indent=2)
    with open(config.ENSEMBLE_WEIGHTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(ensemble_config, f, indent=2)
        
        
    print(f"\nConfigurazione ensemble salvata:")
    print(f"  - Threshold: {config.THRESHOLD_PATH}")
    print(f"  - Ensemble config: {config.ENSEMBLE_WEIGHTS_PATH}")
    # Meta-calibrazione dell'ensemble (isotonic su OOF del metodo scelto)
    ens_cal = IsotonicRegression(out_of_bounds='clip').fit(final_probs, y)
    joblib.dump(ens_cal, config.ENSEMBLE_CALIBRATOR_PATH)
    print(f"  - Ensemble calibrator: {config.ENSEMBLE_CALIBRATOR_PATH}")
    print("\n Training completato!")

if __name__ == '__main__':

    train_ensemble()
