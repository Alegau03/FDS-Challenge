"""
Pokemon Battle Predictor - Configuration Module

Configurazione centralizzata per training, validation e prediction.

Parametri principali:
    - Percorsi dati (train/test JSONL, submission CSV)
    - Cross-validation: 10-fold stratified
    - Feature engineering: max 30 turni per battaglia
    - Ensemble: LightGBM + CatBoost + XGBoost
    - Metodo: Weighted Average (ottimale per V7)

"""

# --- Percorsi dei file di dati ---
TRAIN_FILE_PATH = "/kaggle/input/fds-pokemon-battles-prediction-2025/train.jsonl"
TEST_FILE_PATH = "/kaggle/input/fds-pokemon-battles-prediction-2025/test.jsonl"

# --- Percorsi di output ---
# Assicurati che la cartella 'models/' esista
MODEL_OUTPUT_DIR = 'models/'
SUBMISSION_FILE_PATH = 'submission/submission_v1.csv'
ENSAMBLE_SUBMISSION_FILE_PATH = 'submission/ensamble_submission.csv'
BEST_PARAMS_PATH = 'models/best_params.json'
THRESHOLD_PATH = 'models/best_threshold.json'

# --- Impostazioni di training ---
N_SPLITS = 10  # Numero di fold per la Cross-Validation
RANDOM_STATE = 42 # Seed per la riproducibilità
MAX_TURNS = 30  
N_REPEATS = 1  
SEED_BAGGING = 3  

# --- Modelli ed Ensembling ---
# Scegli quali modelli allenare e usare in ensemble
MODEL_TYPES = ['lgbm', 'cat', 'xgb']  

# Metodo ensemble preferito: 'stacking' o 'weighted_average' o 'auto'
# 'stacking' -> Usa sempre stacking con LogisticRegression L2
# 'weighted_average' -> Usa sempre weighted average con grid search
# 'auto' -> Confronta entrambi e sceglie il migliore (comportamento originale)
ENSEMBLE_METHOD = 'weighted_average'  
ENSEMBLE_METHOD = 'weighted_average'  
MAX_TURNS = 30
# Pesi dell'ensemble (si normalizzano in predict/train)
ENSEMBLE_WEIGHTS = {
        'lgbm': 1.0,
        'cat': 1.0,
        'xgb': 1.0,
}# Artifacts aggiuntivi
BEST_PARAMS_CAT_PATH = 'models/best_params_cat.json'
BEST_PARAMS_XGB_PATH = 'models/best_params_xgb.json'
BEST_PARAMS_LOGREG_PATH = 'models/best_params_logreg.json'
ENSEMBLE_WEIGHTS_PATH = 'models/ensemble_weights.json'
ENSEMBLE_CALIBRATOR_PATH = 'models/ensemble_calibrator.joblib'

# --- Stacking meta-learner ---
# 'lr' = LogisticRegression (default, lineare e robusto)
# 'lgbm' = LightGBM (non-lineare, potenzialmente più espressivo; usare con cautela per overfitting)
STACKING_META = 'lr'