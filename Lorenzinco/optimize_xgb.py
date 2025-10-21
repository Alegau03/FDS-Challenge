# Small XGBoost + Optuna optimizer (reads ../dataset/train.jsonl).
import json, os, numpy as np, optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from features import create_feature_df
import config
try:
    import xgboost as xgb
except ImportError:
    xgb = None

def objective_factory(X, y):
    def obj(trial):
        if xgb is None: raise RuntimeError("XGBoost non installato")
        params = {
            'objective':'binary:logistic','eval_metric':'logloss','tree_method':'hist',
            'eta': trial.suggest_float('eta',0.01,0.2,log=True),
            'max_depth': trial.suggest_int('max_depth',3,12),
            'min_child_weight': trial.suggest_float('min_child_weight',1e-3,10.0,log=True),
            'subsample': trial.suggest_float('subsample',0.5,1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.5,1.0),
            'reg_alpha': trial.suggest_float('reg_alpha',1e-8,10.0,log=True),
            'reg_lambda': trial.suggest_float('reg_lambda',1e-8,10.0,log=True),
        }
        skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
        losses=[]
        for tr,va in skf.split(X,y):
            dtr=xgb.DMatrix(X.iloc[tr], label=y.iloc[tr]); dva=xgb.DMatrix(X.iloc[va], label=y.iloc[va])
            m=xgb.train(params, dtr, num_boost_round=6000, evals=[(dva,'v')], early_stopping_rounds=200, verbose_eval=False)
            p=m.predict(xgb.DMatrix(X.iloc[va]))
            losses.append(log_loss(y.iloc[va], p))
        return float(np.mean(losses))
    return obj

if __name__ == "__main__":
    df = create_feature_df(config.TRAIN_JSONL)
    y = df['player_won'].astype(int); X = df.drop(columns=['battle_id','player_won'])
    study = optuna.create_study(direction='minimize'); study.optimize(objective_factory(X,y), n_trials=50)
    (config.ARTIFACTS / "best_xgb_params.json").write_text(json.dumps(study.best_params, indent=2))
    print("Best LogLoss:", study.best_value)
