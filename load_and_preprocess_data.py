import pandas as pd
import numpy as np
from config import NUM_EXPECTED_FEATURE_COLUMNS, USER_ASSIGNED_FEATURE_NAMES, DEFINED_NUMERIC_FEATURES, DEFINED_CATEGORICAL_FEATURES

def load_and_preprocess_data(file_path):
    df_raw = pd.read_csv(file_path, header=None, engine='python', on_bad_lines='warn')
        
    if df_raw.shape[1] < NUM_EXPECTED_FEATURE_COLUMNS + 1:
        raise ValueError(f"CSV file has only {df_raw.shape[1]} columns, but expected at least {NUM_EXPECTED_FEATURE_COLUMNS + 1}")
        
    X_df_temp_values = df_raw.iloc[:, :NUM_EXPECTED_FEATURE_COLUMNS].values
    X_df = pd.DataFrame(X_df_temp_values, columns=USER_ASSIGNED_FEATURE_NAMES)
    y_raw_values = df_raw.iloc[:, NUM_EXPECTED_FEATURE_COLUMNS].values
    y = y_raw_values.astype(int)        

    for col_name in X_df.columns:
        if col_name in DEFINED_NUMERIC_FEATURES:
            X_df[col_name] = pd.to_numeric(X_df[col_name], errors='coerce')
            if X_df[col_name].isnull().any():
                X_df[col_name] = X_df[col_name].fillna(X_df[col_name].median())
            elif col_name in DEFINED_CATEGORICAL_FEATURES:
                X_df[col_name] = X_df[col_name].astype(str).str.strip()
                if X_df[col_name].isnull().any() or (X_df[col_name] == '').any():
                    mode_val = X_df[col_name].mode()
                    X_df[col_name] = X_df[col_name].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
                    X_df[col_name] = X_df[col_name].replace('', mode_val[0] if not mode_val.empty else 'Unknown')        
    return X_df, y