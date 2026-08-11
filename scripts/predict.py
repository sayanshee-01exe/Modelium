"""Batch inference entry point.

Flow: validate files -> load inference tables -> validate tables -> aggregate ->
domain features -> align to the training schema -> champion pipeline -> CSV.

Orchestration only. Every step below is a call into `src/`, and the feature-building
calls are the *same functions* `scripts/train.py` uses — `build_relational_feature_table`
and `add_domain_features` — with the applicant table switched to application_test. A
second implementation for inference is how training/serving skew starts.

Nothing here trains, fits, or tunes. Preprocessing is transform-only inside the loaded
champion pipeline, and the decision threshold comes from metadata, frozen on validation
in Step 4.

`drop_low_information_columns` is deliberately **not** called: it is a training-time
decision, and recomputing it from application_test would let the batch being scored
determine the production feature schema. The Predictor aligns to the schema the
pipeline was actually fitted on instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_DIR, DATA_FILES, MODEL_DIR, ARTIFACT_DIR
from src.data.data_loader import load_home_credit_tables
from src.data.data_validation import INFERENCE_TABLES, validate_inference_tables, validate_raw_files
from src.data.data_cleaning import optimize_memory
from src.data.data_preparation import build_relational_feature_table
from src.features.feature_engineering import add_domain_features
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger("modelium.predict")

APPLICATION_TABLE = "application_test"
PREDICTIONS_DIR = ARTIFACT_DIR / "predictions"
PREDICTIONS_FILENAME = "test_predictions.csv"


def main():
    # Only the tables inference actually needs. application_train is excluded: scoring
    # must not depend on the training applicants, and loading them would cost a 158 MB
    # read for nothing.
    inference_files = {name: DATA_FILES[name] for name in INFERENCE_TABLES}

    # File existence is table-agnostic, so the training checker is reused as-is against
    # the narrower inference file map rather than reimplemented.
    logger.info("Validating inference dataset files...")
    validate_raw_files(DATA_DIR, inference_files)

    tables = load_home_credit_tables(DATA_DIR, inference_files)

    logger.info("Validating loaded inference dataframes...")
    validate_inference_tables(tables, application_table=APPLICATION_TABLE)

    logger.info("Starting memory optimization...")
    tables = {name: optimize_memory(df) for name, df in tables.items()}

    # The same two functions training calls, in the same order.
    df = build_relational_feature_table(tables, application_table=APPLICATION_TABLE)
    df = add_domain_features(df)
    print(f"Inference feature table: {len(df):,} applicants x {df.shape[1]:,} columns")

    predictor = Predictor.load(MODEL_DIR, ARTIFACT_DIR)
    print(
        f"Champion: {predictor.model_name} | frozen threshold {predictor.threshold:.4f} "
        f"| {len(predictor.expected_columns):,} expected raw features"
    )

    predictions = predictor.predict_dataframe(df)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTIONS_DIR / PREDICTIONS_FILENAME
    predictions.to_csv(output_path, index=False)

    flagged = int(predictions["PREDICTED_CLASS"].sum())
    print(
        f"\nWrote {len(predictions):,} predictions to {output_path}\n"
        f"  flagged as default: {flagged:,} ({flagged / len(predictions):.2%})\n"
        f"  probability range : {predictions['DEFAULT_PROBABILITY'].min():.4f} - "
        f"{predictions['DEFAULT_PROBABILITY'].max():.4f}"
    )
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
