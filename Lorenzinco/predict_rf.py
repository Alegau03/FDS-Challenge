# Predict with RF using priors-enriched features + TTA side-swap + PROGRESS.
import json, joblib, numpy as np, pandas as pd
from tqdm import tqdm
import config
from priors import load_priors
from features import create_feature_df

def swap_features_df(df: pd.DataFrame) -> pd.DataFrame:
    sw = df.copy(); cols = df.columns
    for c in cols:
        if c.startswith('p1_'):
            alt='p2_'+c[3:]; 
            if alt in cols: sw[c]=df[alt]
        elif c.startswith('p2_'):
            alt='p1_'+c[3:]; 
            if alt in cols: sw[c]=df[alt]
        elif c.startswith('diff_'):
            sw[c]=-df[c]
    return sw

if __name__ == "__main__":
    _ = load_priors()
    tqdm.write("Building test features...")
    test_df = create_feature_df(config.TEST_JSONL, show_progress=True)
    X = test_df.drop(columns=['battle_id'])

    cols_train = json.loads(config.RF_FEATURES_PATH.read_text())
    X = X.reindex(columns=cols_train, fill_value=0.0)
    X_sw = swap_features_df(X)

    tqdm.write("Loading model & predicting...")
    model = joblib.load(config.RF_MODEL_PATH)

    # Predict with small progress on batches
    batch = 2048
    p = np.zeros(len(X)); ps = np.zeros(len(X))
    for i in tqdm(range(0, len(X), batch), desc="Predict", unit="batch"):
        j = min(i+batch, len(X))
        p[i:j]  = model.predict_proba(X.iloc[i:j])[:,1]
        ps[i:j] = model.predict_proba(X_sw.iloc[i:j])[:,1]

    p_avg = np.clip(0.5*(p + (1.0 - ps)), 1e-4, 1-1e-4)
    yhat = (p_avg > 0.5).astype(int)

    out = config.SUBMISSION_PATH
    pd.DataFrame({'battle_id': test_df['battle_id'], 'player_won': yhat}).to_csv(out, index=False)
    tqdm.write(f"Saved submission: {out}")
