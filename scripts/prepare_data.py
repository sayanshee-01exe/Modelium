"""DVC stage 2 — build the applicant-level feature tables for training and scoring.

Separated from training because it is the expensive, parameter-free half of the work:
aggregating ~57 M child rows takes minutes and does not depend on a single value in
`params.yaml`. Keeping it as its own stage means editing a search space re-runs the
search alone, not the aggregation — which is the entire point of a DVC cache.

Both tables are built by the *same two functions*, with only the applicant table
switched, so training and inference cannot drift apart:

    application_train -> relational aggregation -> domain features -> drop low-info
    application_test  -> relational aggregation -> domain features

`drop_low_information_columns` runs on the training table only, deliberately. It is a
training-time decision, and recomputing it from application_test would let the batch
being scored determine the production feature schema. The scoring table keeps every
column and the Predictor aligns it to the schema the champion was actually fitted on.

Orchestration only; the feature logic lives in `src/data/data_preparation.py` and
`src/features/feature_engineering.py`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_DIR, DATA_FILES, ID_COL, TARGET_COL,
    PREPARE_REPORT_FILE, TEST_FEATURES_FILE, TRAIN_FEATURES_FILE,
)
from src.data.data_cleaning import drop_low_information_columns, optimize_memory
from src.data.data_loader import load_home_credit_tables
from src.data.data_preparation import build_relational_feature_table
from src.data.data_validation import validate_inference_tables
from src.features.feature_engineering import add_domain_features
from src.utils.logger import get_logger

logger = get_logger("modelium.prepare")

TRAIN_APPLICATION_TABLE = "application_train"
TEST_APPLICATION_TABLE = "application_test"


def main():
    tables = load_home_credit_tables(DATA_DIR, DATA_FILES)

    logger.info("Starting memory optimization...")
    tables = {name: optimize_memory(df) for name, df in tables.items()}

    # --- training features -------------------------------------------------------
    logger.info("Building training feature table...")
    train_df = add_domain_features(
        build_relational_feature_table(tables, TRAIN_APPLICATION_TABLE))
    train_df, dropped = drop_low_information_columns(train_df, ID_COL, TARGET_COL)

    # --- scoring features --------------------------------------------------------
    # Same two functions, different applicant table. No low-information drop here.
    #
    # The inference contract is checked here rather than in the predict stage, because
    # this is where raw scoring data is last seen: predict consumes the parquet below
    # and never touches application_test. Its rules differ from the training contract —
    # no TARGET is required, and application_test must be one row per applicant or the
    # joins fan out and silently multiply predictions.
    logger.info("Validating inference tables...")
    validate_inference_tables(tables, application_table=TEST_APPLICATION_TABLE)

    logger.info("Building scoring feature table...")
    test_df = add_domain_features(
        build_relational_feature_table(tables, TEST_APPLICATION_TABLE))

    TRAIN_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_FEATURES_FILE, index=False)
    test_df.to_parquet(TEST_FEATURES_FILE, index=False)

    payload = {
        "prepared_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train": {"rows": int(len(train_df)), "columns": int(train_df.shape[1]),
                  "application_table": TRAIN_APPLICATION_TABLE},
        "test": {"rows": int(len(test_df)), "columns": int(test_df.shape[1]),
                 "application_table": TEST_APPLICATION_TABLE},
        # Recorded so training metadata can report what was dropped without recomputing
        # it, and so a reviewer can see the decision rather than infer it.
        "dropped_columns": dropped,
        "n_dropped": len(dropped),
    }
    with open(PREPARE_REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(
        f"Prepared features\n"
        f"  train {len(train_df):>7,} rows x {train_df.shape[1]:>4} cols -> {TRAIN_FEATURES_FILE}\n"
        f"  test  {len(test_df):>7,} rows x {test_df.shape[1]:>4} cols -> {TEST_FEATURES_FILE}\n"
        f"  dropped {len(dropped)} low-information column(s) from the training table"
    )


if __name__ == "__main__":
    main()
