import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Import config for paths and constants
from config import RESULTS_DIR, CLASS_NAMES, METRIC_LABELS
from dynamically_resample_and_fit import dynamically_resample_and_fit
from evaluate_fold_performance import evaluate_fold_performance


def execute_final_model_pipeline(X_df, y, preprocessor, final_clf, final_clf_name):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, stratify=y, random_state=42)
    
    base_pipe = Pipeline([('preprocessor', preprocessor)])
    final_pipeline = dynamically_resample_and_fit(base_pipe, X_train, y_train, X_test, final_clf)
    
    y_pred = final_pipeline.predict(X_test)
    y_proba = final_pipeline.predict_proba(X_test)
    
    print(f"\n--- Final Model ({final_clf_name}) Assessment Report on Validation Split ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    rep_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(rep_dict).transpose()
    
    f_score_m, precision_m, recall_m, acc, bal_acc, mcc, kappa, roc_auc, pr_auc, *_ = evaluate_fold_performance(y_test, y_pred, y_proba, METRIC_LABELS)
    
    metrics_summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Balanced Accuracy', 'MCC', 'Cohen Kappa', 'ROC AUC (Macro OVR)', 'PR AUC (Macro)'],
        'Value': [acc, bal_acc, mcc, kappa, roc_auc, pr_auc]
    })
    
    cm = confusion_matrix(y_test, y_pred, labels=METRIC_LABELS)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix - {final_clf_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
           
    return report_df, metrics_summary_df
