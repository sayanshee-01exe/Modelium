from pathlib import Path

RANDOM_STATE = 42
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"
HIGH_MISSING_THRESHOLD = 60.0
LOW_CARDINALITY_THRESHOLD = 10
DOWNSAMPLE_RATIO = 0.10

# Three-way split sizes, as fractions of the full dataset (Step 1).
# Step 7 of the refactor plan moves these into params.yaml; they live here for now so
# Step 1 does not depend on a file that does not exist yet.
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

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

# Step 4 randomised-search budget. n_iter x cv_folds fits per tuned model, so 20x3 = 60.
# Kept modest so a full run stays feasible on a laptop; raise for a serious sweep.
TUNING_N_ITER = 20
TUNING_CV_FOLDS = 3
