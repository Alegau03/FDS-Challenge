# config.py

# --- Percorsi dei file di dati ---
TRAIN_FILE_PATH = 'data/train.jsonl'
TEST_FILE_PATH = 'data/test.jsonl'

# --- Percorsi di output ---
# Assicurati che la cartella 'models/' esista
MODEL_OUTPUT_DIR = 'models/'
SUBMISSION_FILE_PATH = 'submission/submission_v1.csv'
ENSAMBLE_SUBMISSION_FILE_PATH = 'submission/ensamble_submission.csv'
BEST_PARAMS_PATH = 'models/best_params.json'
THRESHOLD_PATH = 'models/best_threshold.json'

# --- Impostazioni di training ---
N_SPLITS = 5  # Numero di fold per la Cross-Validation
RANDOM_STATE = 42 # Seed per la riproducibilità
MAX_TURNS = 30  # Usa solo i primi 30 turni del battle_log
N_REPEATS = 1  # Numero di ripetizioni della K-Fold (Repeated StratifiedKFold)
SEED_BAGGING = 1  # Numero di semi diversi per model bagging per ogni fold

# --- Modelli ed Ensembling ---

MODEL_TYPES = ['lgbm', 'cat', 'xgb']  
# Pesi dell'ensemble (si normalizzano in predict/train)
ENSEMBLE_WEIGHTS = {
	'lgbm': 1.0,
	'cat': 1.0,
	'xgb': 1.0,
}

# Artifacts aggiuntivi
BEST_PARAMS_CAT_PATH = 'models/best_params_cat.json'
BEST_PARAMS_XGB_PATH = 'models/best_params_xgb.json'
ENSEMBLE_WEIGHTS_PATH = 'models/ensemble_weights.json'
ENSEMBLE_CALIBRATOR_PATH = 'models/ensemble_calibrator.joblib'