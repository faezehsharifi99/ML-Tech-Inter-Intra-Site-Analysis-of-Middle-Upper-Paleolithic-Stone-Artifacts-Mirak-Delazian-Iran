import numpy as np
import pandas as pd
import optuna
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from config import METRIC_LABELS
from build_ensemble import build_ensemble
from dynamically_resample_and_fit import dynamically_resample_and_fit
from evaluate_fold_performance import evaluate_fold_performance


def run_optuna_strategy(X_df, y, preprocessor, inner_cv, smallest_size, n_trials=15):
    
    def objective(trial):
        clf_name = trial.suggest_categorical("classifier", ["DecisionTree", "LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "SVM", "KNN", "Ensemble"])
        if clf_name == "DecisionTree":
            clf = DecisionTreeClassifier(criterion=trial.suggest_categorical("criterion", ["gini", "entropy"]), max_depth=trial.suggest_int("max_depth", 3, 100), random_state=42)
        elif clf_name == "LogisticRegression":
            clf = LogisticRegression(C=trial.suggest_float("lr_C", 1e-3, 10.0, log=True), solver=trial.suggest_categorical("solver", ["liblinear", "lbfgs"]), multi_class='ovr', max_iter=1500, random_state=42)
        elif clf_name == "RandomForest":
            clf = RandomForestClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 200), max_depth=trial.suggest_int("max_depth", 3, 25), class_weight="balanced", random_state=42)
        elif clf_name == "XGBoost":
            clf = XGBClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 200), max_depth=trial.suggest_int("max_depth", 3, 20), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3), eval_metric='mlogloss', random_state=42, use_label_encoder=False)
        elif clf_name == "LightGBM":
            clf = LGBMClassifier(n_estimators=trial.suggest_int("n_estimators", 50, 200), max_depth=trial.suggest_int("max_depth", 3, 22), learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3), class_weight="balanced", random_state=42, verbose=-1)
        elif clf_name == "SVM":
            clf = SVC(C=trial.suggest_float("svm_C", 0.1, 20.0), kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]), probability=True, random_state=42)
        elif clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=trial.suggest_int("n_neighbors", 3, min(15, smallest_size - 1)))
        elif clf_name == "Ensemble":
            rf_n = trial.suggest_int("ensemble_rf_n", 50, 150)
            xgb_n = trial.suggest_int("ensemble_xgb_n", 50, 150)
            clf = build_ensemble(rf_n, xgb_n)

        f_mean_list = []
        for train_idx, val_idx in inner_cv.split(X_df, y):
            base_pipe = Pipeline([('preprocessor', preprocessor)])
            try:
                fitted_pipe = dynamically_resample_and_fit(base_pipe, X_df.iloc[train_idx], y[train_idx], X_df.iloc[val_idx], clf)
                preds = fitted_pipe.predict(X_df.iloc[val_idx])
                probs = fitted_pipe.predict_proba(X_df.iloc[val_idx])
                f_score_m, *_ = evaluate_fold_performance(y[val_idx], preds, probs, METRIC_LABELS)
                f_mean_list.append(f_score_m)
            except Exception:
                f_mean_list.append(0.0)
        return np.mean(f_mean_list) if f_mean_list else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    results_data = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            metrics = {"classifier": trial.params.get("classifier"), "trial_number": trial.number, "value_opt_metric": trial.value}
            for k, v in trial.params.items():
                metrics[f"param_{k}"] = v
            results_data.append(metrics)
            
    return study, pd.DataFrame(results_data)