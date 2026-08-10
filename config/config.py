from pathlib import Path

RANDOM_STATE = 42
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
HIGH_MISSING_THRESHOLD = 60.0
LOW_CARDINALITY_THRESHOLD = 10
DOWNSAMPLE_RATIO = 0.10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# Registry of every raw table on disk. Task 5 reworks the loader to pull subsets per
# stage rather than reading all of these at once (defect D1), so listing a file here
# does not by itself mean every stage loads it.
DATA_FILES = {
    "application_train": "application_train.csv",
    # D6: the holdout applicants were previously absent from the registry, leaving no
    # scoring/submission path. The training pipeline splits application_train and does
    # not consume this table.
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
    "installments": "installments_payments.csv",
}
