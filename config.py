import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)
DEFINED_CATEGORICAL_FEATURES = []


# ------------- File paths ---------------
DEFAULT_CSV_PATH = DATA_DIR / 'All_Mirak_Delazian.csv'
DEFAULT_EXCEL_PATH = RESULTS_DIR / 'results.xlsx'


NUM_EXPECTED_FEATURE_COLUMNS = 9

FEATURE_NAMES = [
    'Typology',            # column 0
    'Blank_Typology',      # column 1
    'Tool_Typology',       # column 2
    'Length',              # column 3
    'Width',               # column 4
    'Length_Per_Width',    # column 5
    'Thickness',           # column 6
    'Breakage_Type',       # column 7
    'Platform_Type',       # column 8   
]

DEFINED_NUMERIC_FEATURES = FEATURE_NAMES  
DEFINED_CATEGORICAL_FEATURES = []

LABEL_MAPPING = {
    'Delazian': 0,
    'Mirak 1': 1,
    'Mirak 2': 2,
    'Mirak 3': 3
}
CLASS_NAMES = list(LABEL_MAPPING.keys())
METRIC_LABELS = list(LABEL_MAPPING.values())   # [0,1,2,3]

