import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    BASE_DIR, DATA_DIR, RESULTS_DIR, 
    DEFAULT_CSV_PATH, DEFAULT_EXCEL_PATH,
    METRIC_LABELS, CLASS_NAMES
)

from load_and_preprocess_data import load_and_preprocess_data
from build_column_preprocessor import build_column_preprocessor
from run_optuna_strategy import run_optuna_strategy
from run_gridsearch_strategy import run_gridsearch_strategy
from build_ensemble import build_ensemble
from execute_final_model_pipeline import execute_final_model_pipeline


from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def main():

    parser.add_argument(
        '--input', 
        type=str, 
        default=str(DEFAULT_CSV_PATH),
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=str(DEFAULT_EXCEL_PATH),
    )
    parser.add_argument(
        '--trials', 
        type=int, 
        default=50,
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Ensure the output directory exists
    os.makedirs(output_path.parent, exist_ok=True)
    
    print(f" Reading data from: {input_path}")
    print(f" Results will be saved to: {output_path}")
    
    
    X_df, y = load_and_preprocess_data(CSV_FILE_PATH)
    
    preprocessor = build_column_preprocessor(X_df)
    outer_cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    min_samples = np.min(np.bincount(y))
    approx_inner_train_size = max(2, int(len(y) * (1 - 1/15) * (1 - 1/10) * (min_samples / len(y))))    
    study, optuna_perf_df = run_optuna_strategy(X_df, y, preprocessor, inner_cv, approx_inner_train_size, n_trials=10)   
    grid_perf_df, best_grid_name, best_grid_params = run_gridsearch_strategy(X_df, y, preprocessor, inner_cv, approx_inner_train_size)
    
    # -------- Best Model ------------
    if study.best_trial and study.best_trial.params.get("classifier") != "Ensemble":
        best_params = study.best_trial.params.copy()
        best_clf_name = best_params.pop("classifier")        
        
        if "lr_C" in best_params: best_params["C"] = best_params.pop("lr_C")
        if "svm_C" in best_params: best_params["C"] = best_params.pop("svm_C")        
        if best_clf_name == "DecisionTree": final_clf = DecisionTreeClassifier(**best_params, random_state=42)
        elif best_clf_name == "LogisticRegression": final_clf = LogisticRegression(**best_params, random_state=42)
        elif best_clf_name == "RandomForest": final_clf = RandomForestClassifier(**best_params, random_state=42)
        elif best_clf_name == "XGBoost": final_clf = XGBClassifier(**best_params, random_state=42)
        elif best_clf_name == "LightGBM": final_clf = LGBMClassifier(**best_params, random_state=42)
        elif best_clf_name == "SVM": final_clf = SVC(**best_params, probability=True, random_state=42)
        else: final_clf = KNeighborsClassifier(**best_params)
    elif study.best_trial and study.best_trial.params.get("classifier") == "Ensemble":
        best_params = study.best_trial.params
        rf_n = best_params.get("ensemble_rf_n", 100)
        xgb_n = best_params.get("ensemble_xgb_n", 100)
        final_clf = build_ensemble(rf_n, xgb_n)
        best_clf_name = "Ensemble"
    else:
        best_clf_name = "BackupRandomForest"
        final_clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        
    test_report_df, test_metrics_df, shap_importance_df = execute_final_model_pipeline(X_df, y, preprocessor, final_clf, best_clf_name)
    # -------- Report Results ------------
    try:
        with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine='openpyxl') as writer:
            if not optuna_perf_df.empty:
                optuna_perf_df.to_excel(writer, sheet_name='Optuna_All_Trials_Performance', index=False)
            if not grid_perf_df.empty:
                grid_perf_df.to_excel(writer, sheet_name='GridSearch_Configurations', index=False)
            if test_report_df is not None and not test_report_df.empty:
                test_report_df.to_excel(writer, sheet_name='Final_Model_Class_Report')
            if test_metrics_df is not None and not test_metrics_df.empty:
                test_metrics_df.to_excel(writer, sheet_name='Final_Model_Global_Metrics', index=False)
            if shap_importance_df is not None and not shap_importance_df.empty:
                shap_importance_df.to_excel(writer, sheet_name='SHAP_Feature_Importance', index=False)
        print("Pipeline reporting generation task terminated successfully.")
    except Exception as ex:
        print(f"Error persisting output")
        
if __name__ == "__main__":
    main()