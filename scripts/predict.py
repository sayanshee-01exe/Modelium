"""DVC stage 4 — batch inference entry point.

Flow: read prepared scoring features -> align to the training schema -> champion
pipeline -> CSV.

Orchestration only. The scoring features were built by the `prepare` stage using the
*same two functions* that built the training features — `build_relational_feature_table`
and `add_domain_features` — with only the applicant table switched to application_test.
A second implementation for inference is how training/serving skew starts.

Nothing here trains, fits, or tunes. Preprocessing is transform-only inside the loaded
champion pipeline, and the decision threshold comes from metadata, frozen on validation
in Step 4.

`drop_low_information_columns` is deliberately never applied to the scoring table: it is
a training-time decision, and recomputing it from application_test would let the batch
being scored determine the production feature schema. The Predictor aligns to the schema
the pipeline was actually fitted on instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.config import ARTIFACT_DIR, MODEL_DIR, PREDICTIONS_DIR, TEST_FEATURES_FILE
from src.inference.predictor import Predictor
from src.utils.logger import get_logger

logger = get_logger("modelium.predict")

PREDICTIONS_FILENAME = "test_predictions.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--allow-unpromoted",
        action="store_true",
        help="Score with a champion that failed its Step 4 quality gates. Debugging "
             "only — an unpromoted model was measured and found wanting, and its "
             "predictions must not reach a decision.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    df = pd.read_parquet(TEST_FEATURES_FILE)
    print(f"Inference feature table: {len(df):,} applicants x {df.shape[1]:,} columns")

    # Refuses an unpromoted or otherwise invalid artifact unless explicitly overridden.
    predictor = Predictor.load(MODEL_DIR, ARTIFACT_DIR, allow_unpromoted=args.allow_unpromoted)
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
