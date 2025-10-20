# Faster RF training with clear progress (no calibration during tuning; pruner on).
import json, time, joblib, optuna, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import optuna.logging as optlog

import config
from priors import load_priors, build_priors
from features import create_feature_df

N_TRIALS = 10
CV_FOLDS = config.N_SPLITS

def ensure_priors():
    if not config.PRIORS_PATH.exists():
        print("Priors not found. Building from train...")
        build_priors()
    else:
        _ = load_priors()
        print("Loaded priors:", config.PRIORS_PATH)

def objective_factory(X, y):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    def obj(trial):
        # Tighter, faster search space
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 300, 900),  # was up to 1500
            max_depth=trial.suggest_int('max_depth', 8, 100),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 12),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 6),
            max_features=trial.suggest_float('max_features', 0.4, 1.0),
            bootstrap=True,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )

        # No calibration here (much faster). We’ll calibrate once after best params found.
        # Add fold-level timing so you see movement inside a trial.
        losses = []
        t0 = time.time()
        for fold_idx, (tr, va) in enumerate(skf.split(X, y), 1):
            fstart = time.time()
            clf = RandomForestClassifier(**params)
            clf.fit(X.iloc[tr], y.iloc[tr])
            p = clf.predict_proba(X.iloc[va])[:, 1]
            loss = log_loss(y.iloc[va], np.clip(p, 1e-5, 1-1e-5))
            losses.append(loss)
            fsec = time.time() - fstart
            tqdm.write(f"[trial {trial.number}] fold {fold_idx}/{CV_FOLDS} logloss={loss:.5f} ({fsec:.1f}s)")

            # Let the pruner judge early
            trial.report(float(np.mean(losses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        mean_loss = float(np.mean(losses))
        sec = time.time() - t0
        tqdm.write(f"[trial {trial.number}] CV mean logloss={mean_loss:.6f} ({sec:.1f}s total)")
        return mean_loss

    return obj

if __name__ == "__main__":
    optlog.set_verbosity(optlog.INFO)
    print("== RF training with priors-enriched features (fast) ==")
    ensure_priors()

    print("Loading & building features...")
    df = create_feature_df(config.TRAIN_JSONL, show_progress=True)
    y = df['player_won'].astype(int)
    X = df.drop(columns=['battle_id', 'player_won'])
    print(f"Feature matrix: X={X.shape}, y={y.shape}")

    # Pruner: prune bad trials early so the outer progress jumps sooner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction='minimize', pruner=pruner)

    print(f"Starting Optuna with {N_TRIALS} trials...")
    study.optimize(objective_factory(X, y), n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    best.update(random_state=config.RANDOM_STATE, n_jobs=-1)
    print("Best params:", json.dumps(best, indent=2))

    # Refit on full data with calibration (isotonic; slower but only once)
    print("Refitting final calibrated model on all data...")
    base = RandomForestClassifier(**best)
    model = CalibratedClassifierCV(base, method='isotonic', cv=5)
    model.fit(X, y)

    joblib.dump(model, config.RF_MODEL_PATH)
    config.RF_FEATURES_PATH.write_text(json.dumps(list(X.columns), indent=2))
    print("Saved model:", config.RF_MODEL_PATH)
