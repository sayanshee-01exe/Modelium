"""Structural constants only — paths, filenames, and the two column names the schema
is built around.

Everything an *experiment* would change (split sizes, seeds, search spaces, gate
thresholds) lives in `params.yaml` and is read through `src/utils/config_loader.py`.
Splitting it this way gives DVC something meaningful to hash: editing a search space
invalidates the stages that read it, while moving a directory does not pretend to be a
new experiment.

`RANDOM_STATE`, `VALIDATION_SIZE` and `TEST_SIZE` used to live here and now come from
params.yaml. `HIGH_MISSING_THRESHOLD`, `LOW_CARDINALITY_THRESHOLD` and
`DOWNSAMPLE_RATIO` were also defined here and were read by nothing; they were removed
rather than migrated, since a parameter no code consults is a claim the pipeline does
not honour.
"""

from pathlib import Path

TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACT_DIR / "metrics"
PREDICTIONS_DIR = ARTIFACT_DIR / "predictions"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# DVC stage artifacts. Named here so dvc.yaml, the producing script and the consuming
# script cannot drift on a path.
TRAIN_FEATURES_FILE = PROCESSED_DIR / "train_features.parquet"
TEST_FEATURES_FILE = PROCESSED_DIR / "test_features.parquet"
PREPARE_REPORT_FILE = PROCESSED_DIR / "prepare_report.json"
VALIDATION_REPORT_FILE = METRICS_DIR / "raw_data_validation.json"
# Handoff from the train stage to the register stage: which tracked run produced the
# champion, and whether it was approved. Written on every training run, including one
# with tracking disabled, because DVC declares it as a stage output.
RUN_INFO_FILE = ARTIFACT_DIR / "run_information.json"
# What the register stage actually did: which version was created, under which alias,
# and whether it was approved. The registry itself is external state DVC cannot restore,
# so this is the pipeline's own record of the outcome.
REGISTRY_RECORD_FILE = ARTIFACT_DIR / "registry_record.json"

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
