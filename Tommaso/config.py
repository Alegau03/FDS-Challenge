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
N_REPEATS = 2  # V6.1: Aumentato a 2 per maggiore robustezza (10 fold totali)
SEED_BAGGING = 3  # V6.1: Aumentato a 3 per stabilità predizioni (era 1)

# --- Modelli ed Ensembling ---
# Scegli quali modelli allenare e usare in ensemble
MODEL_TYPES = ['lgbm', 'cat']  # V6.2: Rimosso xgb (contribuisce solo 12%, rallenta training)
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