import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # Root of the repository
DATA_DIR = BASE_DIR / 'data'                        # Input data folder
RESULTS_DIR = BASE_DIR / 'results'                  # Output results folder

# Ensure the results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# File paths 
DEFAULT_CSV_PATH = DATA_DIR / 'All_Mirak_Delazian.csv'
DEFAULT_EXCEL_PATH = RESULTS_DIR / 'results.xlsx'

NUM_EXPECTED_FEATURE_COLUMNS = 10 
USER_ASSIGNED_FEATURE_NAMES = [f'feature_{i}' for i in range(NUM_EXPECTED_FEATURE_COLUMNS)]

DEFINED_NUMERIC_FEATURES = USER_ASSIGNED_FEATURE_NAMES[:5]  
DEFINED_CATEGORICAL_FEATURES = USER_ASSIGNED_FEATURE_NAMES[5:]

METRIC_LABELS = [0, 3]  
CLASS_NAMES = ['Delazian', 'Mirak_L1', 'Mirak_L2', 'Mirak_L3'] 