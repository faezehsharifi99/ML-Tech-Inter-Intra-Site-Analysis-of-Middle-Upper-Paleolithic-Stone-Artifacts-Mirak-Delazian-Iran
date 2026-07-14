import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from config import DEFINED_NUMERIC_FEATURES, DEFINED_CATEGORICAL_FEATURES


def build_column_preprocessor(X_df):
    actual_numeric_features = [f for f in X_df.columns if f in DEFINED_NUMERIC_FEATURES]
    actual_categorical_features = [f for f in X_df.columns if f in DEFINED_CATEGORICAL_FEATURES]
    transformers_list = []
    if actual_numeric_features:
        transformers_list.append(('num', StandardScaler(), actual_numeric_features))
    if actual_categorical_features:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), actual_categorical_features))
    return ColumnTransformer(transformers=transformers_list, remainder='passthrough')