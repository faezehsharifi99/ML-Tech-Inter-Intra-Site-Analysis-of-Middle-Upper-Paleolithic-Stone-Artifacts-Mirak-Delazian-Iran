from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

def build_ensemble(rf_n_estimators, xgb_n_estimators, random_state=42):
    rf = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=10,
                                class_weight='balanced', random_state=random_state)
    xgb = XGBClassifier(n_estimators=xgb_n_estimators, max_depth=5,
                        eval_metric='mlogloss', random_state=random_state)
    return VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')