import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, roc_auc_score, average_precision_score
)

def evaluate_fold_performance(y_val_fold, y_pred_fold, y_proba_fold, metric_labels):
    precision_m, recall_m, f_score_m, _ = precision_recall_fscore_support(y_val_fold, y_pred_fold, average='macro', zero_division=0)
    acc = accuracy_score(y_val_fold, y_pred_fold)
    bal_acc = balanced_accuracy_score(y_val_fold, y_pred_fold)
    mcc = matthews_corrcoef(y_val_fold, y_pred_fold)
    kappa = cohen_kappa_score(y_val_fold, y_pred_fold)
    
    roc_auc = 0.5
    pr_auc = 0.0
    if y_proba_fold.shape[1] == len(metric_labels):
        try:
            roc_auc = roc_auc_score(y_val_fold, y_proba_fold, multi_class='ovr', average='macro', labels=metric_labels)
        except ValueError:
            roc_auc = 0.5
        y_val_fold_binarized = pd.get_dummies(y_val_fold, columns=metric_labels).reindex(columns=metric_labels, fill_value=0).values
        pr_auc_per_class = []
        for i_cls in range(y_proba_fold.shape[1]):
            if np.sum(y_val_fold_binarized[:, i_cls]) > 0:
                pr_auc_per_class.append(average_precision_score(y_val_fold_binarized[:, i_cls], y_proba_fold[:, i_cls]))
            else:
                pr_auc_per_class.append(0.0)
        pr_auc = np.mean(pr_auc_per_class) if pr_auc_per_class else 0.0

    p_class, r_class, f1_class, _ = precision_recall_fscore_support(y_val_fold, y_pred_fold, labels=metric_labels, average=None, zero_division=0)
    return f_score_m, precision_m, recall_m, acc, bal_acc, mcc, kappa, roc_auc, pr_auc, f1_class, p_class, r_class