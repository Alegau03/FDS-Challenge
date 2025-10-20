# Small CatBoost + Optuna optimizer (reads ../dataset/train.jsonl).
import json, os, numpy as np, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from features import create_feature_df
import config
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

def objective_factory(X, y):
    def obj(trial):
        if CatBoostClassifier is None: raise RuntimeError("CatBoost non installato")
        params = {
            'loss_function':'Logloss','eval_metric':'Logloss',
            'random_seed':config.RANDOM_STATE,'verbose':False,
            'depth': trial.suggest_int('depth',4,10),
            'learning_rate': trial.suggest_float('learning_rate',0.01,0.2,log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg',1e-3,10.0,log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature',0.0,5.0),
            'border_count': trial.suggest_int('border_count',32,254),
            'iterations': trial.suggest_int('iterations',1500,6000),
        }
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        losses=[]
        for tr,va in skf.split(X,y):
            m=CatBoostClassifier(**params); m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False, use_best_model=True)
            losses.append(log_loss(y.iloc[va], m.predict_proba(X.iloc[va])[:,1]))
        return float(np.mean(losses))
    return obj

if __name__ == "__main__":
    df = create_feature_df(config.TRAIN_JSONL)
    y = df['player_won'].astype(int); X = df.drop(columns=['battle_id','player_won'])
    study = optuna.create_study(direction='minimize'); study.optimize(objective_factory(X,y), n_trials=50)
    (config.ARTIFACTS / "best_cat_params.json").write_text(json.dumps(study.best_params, indent=2))
    print("Best LogLoss:", study.best_value)
