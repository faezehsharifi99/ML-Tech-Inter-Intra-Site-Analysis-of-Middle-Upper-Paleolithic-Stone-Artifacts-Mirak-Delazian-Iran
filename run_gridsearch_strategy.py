import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import METRIC_LABELS
from build_ensemble import build_ensemble
from dynamically_resample_and_fit import dynamically_resample_and_fit
from evaluate_fold_performance import evaluate_fold_performance

def run_gridsearch_strategy(X_df, y, preprocessor, inner_cv, smallest_size):
    
    # Define grid configurations 
    grids = {
        "DecisionTree": {
            "model_class": DecisionTreeClassifier,
            "grid": {"max_depth": [5, 15]}
        },
        "RandomForest": {
            "model_class": RandomForestClassifier,
            "grid": {"n_estimators": [50, 100], "max_depth": [10, 20]}
        },
        "XGBoost": {
            "model_class": XGBClassifier,
            "grid": {"learning_rate": [0.1, 0.3], "n_estimators": [50, 100]}
        },
        "Ensemble": {
            "builder": build_ensemble,
            "grid": {"rf_n_estimators": [50, 100], "xgb_n_estimators": [50, 100]}
        }
    }
    
    results_data = []
    best_score = -1.0
    best_config_params = None
    best_config_name = None

    for clf_name, config in grids.items():        
        if "builder" in config:
            param_grid = list(ParameterGrid(config["grid"]))
            for p_set in param_grid:
                clf = config["builder"](**p_set)
                f_mean_scores = []
                for train_idx, val_idx in inner_cv.split(X_df, y):
                    base_pipe = Pipeline([('preprocessor', preprocessor)])
                    try:
                        fitted_pipe = dynamically_resample_and_fit(base_pipe, X_df.iloc[train_idx], y[train_idx], X_df.iloc[val_idx], clf)
                        preds = fitted_pipe.predict(X_df.iloc[val_idx])
                        probs = fitted_pipe.predict_proba(X_df.iloc[val_idx])
                        f_score_m, *_ = evaluate_fold_performance(y[val_idx], preds, probs, METRIC_LABELS)
                        f_mean_scores.append(f_score_m)
                    except Exception:
                        f_mean_scores.append(0.0)
                mean_score = np.mean(f_mean_scores) if f_mean_scores else 0.0
                metrics = {"classifier": clf_name, "value_opt_metric": mean_score}
                metrics.update({f"param_{k}": v for k, v in p_set.items()})
                results_data.append(metrics)
                if mean_score > best_score:
                    best_score = mean_score
                    best_config_name = clf_name
                    best_config_params = p_set
        else:
            model_class = config["model_class"]
            param_grid = list(ParameterGrid(config["grid"]))
            for p_set in param_grid:
                clean_params = {k: v for k, v in p_set.items()}
                clf = model_class(**clean_params, random_state=42) if 'random_state' in model_class.__init__.__code__.co_varnames else model_class(**clean_params)
                f_mean_scores = []
                for train_idx, val_idx in inner_cv.split(X_df, y):
                    base_pipe = Pipeline([('preprocessor', preprocessor)])
                    try:
                        fitted_pipe = dynamically_resample_and_fit(base_pipe, X_df.iloc[train_idx], y[train_idx], X_df.iloc[val_idx], clf)
                        preds = fitted_pipe.predict(X_df.iloc[val_idx])
                        probs = fitted_pipe.predict_proba(X_df.iloc[val_idx])
                        f_score_m, *_ = evaluate_fold_performance(y[val_idx], preds, probs, METRIC_LABELS)
                        f_mean_scores.append(f_score_m)
                    except Exception:
                        f_mean_scores.append(0.0)
                mean_score = np.mean(f_mean_scores) if f_mean_scores else 0.0
                metrics = {"classifier": clf_name, "value_opt_metric": mean_score}
                metrics.update({f"param_{k}": v for k, v in clean_params.items()})
                results_data.append(metrics)
                if mean_score > best_score:
                    best_score = mean_score
                    best_config_name = clf_name
                    best_config_params = clean_params

    return pd.DataFrame(results_data), best_config_name, best_config_params