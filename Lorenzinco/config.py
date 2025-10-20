from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dataset"          # ../dataset/
ARTIFACTS = BASE_DIR / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

TRAIN_JSONL = DATA_DIR / "train.jsonl"
TEST_JSONL  = DATA_DIR / "test.jsonl"

N_SPLITS = 5
RANDOM_STATE = 42

# Outputs
RF_MODEL_PATH     = ARTIFACTS / "rf_model.joblib"
RF_FEATURES_PATH  = ARTIFACTS / "rf_features.json"
SUBMISSION_PATH   = ARTIFACTS / "submission.csv"
PRIORS_PATH       = ARTIFACTS / "priors.json"
