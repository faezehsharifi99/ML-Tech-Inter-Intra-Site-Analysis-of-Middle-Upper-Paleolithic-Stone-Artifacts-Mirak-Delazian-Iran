import pandas as pd
import numpy as np
from config import NUM_EXPECTED_FEATURE_COLUMNS, USER_ASSIGNED_FEATURE_NAMES, DEFINED_NUMERIC_FEATURES, DEFINED_CATEGORICAL_FEATURES
from config import LABEL_MAPPING
from config import NUM_EXPECTED_FEATURE_COLUMNS, USER_ASSIGNED_FEATURE_NAMES, LABEL_MAPPING


def load_and_preprocess_data(file_path):
    df_raw = pd.read_csv(file_path, header=None, engine='python', on_bad_lines='warn')
    
    if df_raw.shape[1] < NUM_EXPECTED_FEATURE_COLUMNS + 1:
        raise ValueError(f"CSV has {df_raw.shape[1]} columns, expected {NUM_EXPECTED_FEATURE_COLUMNS + 1}")
    
    # Split features and target
    X_df_temp = df_raw.iloc[:, :NUM_EXPECTED_FEATURE_COLUMNS].values
    X_df = pd.DataFrame(X_df_temp, columns=USER_ASSIGNED_FEATURE_NAMES)
    y_raw = df_raw.iloc[:, NUM_EXPECTED_FEATURE_COLUMNS].values
    
    # Encode string labels to integers
    y = np.array([LABEL_MAPPING[val] for val in y_raw])
    
   
    return X_df, y
    
