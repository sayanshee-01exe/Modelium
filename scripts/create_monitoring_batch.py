"""Build a deterministic **demonstration** batch for the monitoring stage.

This project has no live production traffic, so there is no genuine "current batch" to
monitor. Rather than pretend otherwise, this script manufactures one and marks it as
manufactured: every batch it writes carries `demonstration: true` alongside a record of
exactly what was done to it, and the monitoring report reproduces that notice.

The rows are real and held out. They come from the **test split** — the 15% the champion
never saw during fitting, selection or threshold tuning — so the feature values and the
labels are genuine, and the batch is a legitimate stand-in for scoring a fresh cohort.

What is *not* genuine is the drift. With `--drift` a controlled shift is applied to a few
named numeric features and one categorical, purely so the detectors have something to
find in a demonstration. The affected feature names are written into the batch metadata,
so a reader can always separate injected movement from anything real.

Labels ride along because the source split has them, which lets the labelled monitoring
path be exercised. They are marked `label_source: held_out_test_split` — these are not
observed production outcomes, and monitoring must not present them as such.

The prepared datasets are never modified; this only ever writes into `data/monitoring/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ID_COL, TARGET_COL, TRAIN_FEATURES_FILE
from src.data.data_preparation import split_train_val_test
from src.utils.config_loader import load_params
from src.utils.logger import get_logger

logger = get_logger("modelium.monitoring_batch")

MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
CURRENT_BATCH_FILE = MONITORING_DIR / "current_batch.parquet"
BATCH_METADATA_FILE = MONITORING_DIR / "current_batch_metadata.json"

# Features perturbed by --drift, chosen because they are high-importance in the SHAP
# ranking, so a shift is both detectable and meaningful rather than an invisible tweak
# to an unused column.
DRIFT_NUMERIC = {
    "AMT_INCOME_TOTAL": ("scale", 1.18),
    "EXT_SOURCE_MEAN": ("shift_std", -0.35),
    "ANNUITY_CREDIT_RATIO": ("scale", 1.12),
    "DAYS_BIRTH": ("shift_std", -0.25),
}
DRIFT_CATEGORICAL = "NAME_INCOME_TYPE"
DRIFT_MISSING_COLUMN = "EXT_SOURCE_1"
DRIFT_MISSING_EXTRA_RATE = 0.12


def build_batch(rows: int, seed: int, inject_drift: bool):
    """Return ``(batch, metadata)`` sampled from the held-out test split."""
    frame = pd.read_parquet(TRAIN_FEATURES_FILE)
    data_params = load_params()["data"]

    features = frame.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    target = frame[TARGET_COL].astype(int)
    _, _, X_test, _, _, y_test = split_train_val_test(
        features, target,
        validation_size=float(data_params["validation_size"]),
        test_size=float(data_params["test_size"]),
        random_state=int(data_params["random_state"]),
    )
    logger.info("Held-out test split: %d rows", len(X_test))

    take = min(rows, len(X_test))
    sampled = X_test.sample(n=take, random_state=seed).sort_index()
    batch = sampled.copy()
    batch.insert(0, ID_COL, frame.loc[sampled.index, ID_COL].to_numpy())
    batch[TARGET_COL] = y_test.loc[sampled.index].to_numpy()

    metadata = {
        "demonstration": True,
        "description": (
            "Rows are genuine and held out — sampled from the test split the champion "
            "never saw. Any drift below was injected by this script for demonstration "
            "and is not observed production behaviour."
        ),
        "source": "held-out test split of data/processed/train_features.parquet",
        "label_source": "held_out_test_split",
        "labels_are_observed_production_outcomes": False,
        "rows": int(len(batch)),
        "random_state": int(seed),
        "simulated_drift": bool(inject_drift),
        "simulated_drift_features": [],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if inject_drift:
        rng = np.random.default_rng(seed)
        applied: list[str] = []

        for column, (mode, amount) in DRIFT_NUMERIC.items():
            if column not in batch.columns:
                continue
            values = pd.to_numeric(batch[column], errors="coerce")
            if mode == "scale":
                batch[column] = values * amount
            else:
                spread = float(np.nanstd(values.to_numpy(dtype=float)))
                batch[column] = values + amount * spread
            applied.append(column)

        # Tilt one categorical mix by reassigning a slice of rows to the most common
        # value, which raises that category's share without inventing a new label.
        if DRIFT_CATEGORICAL in batch.columns:
            column = batch[DRIFT_CATEGORICAL]
            if column.notna().any():
                dominant = column.mode().iloc[0]
                flip = rng.random(len(batch)) < 0.15
                batch.loc[flip, DRIFT_CATEGORICAL] = dominant
                applied.append(DRIFT_CATEGORICAL)

        # Blank extra values in one column so the missing-rate detector has a signal —
        # a broken upstream feed is a common and very detectable production failure.
        if DRIFT_MISSING_COLUMN in batch.columns:
            blank = rng.random(len(batch)) < DRIFT_MISSING_EXTRA_RATE
            batch.loc[blank, DRIFT_MISSING_COLUMN] = np.nan
            applied.append(f"{DRIFT_MISSING_COLUMN} (missing rate)")

        metadata["simulated_drift_features"] = applied
        logger.info("Injected controlled drift into: %s", ", ".join(applied))

    return batch, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=15000,
                        help="rows to draw from the held-out test split")
    parser.add_argument("--seed", type=int, default=2026,
                        help="sampling seed; a different seed gives a different cohort")
    parser.add_argument("--drift", action="store_true",
                        help="inject controlled, clearly-recorded drift for demonstration")
    parser.add_argument("--no-drift", dest="drift", action="store_false",
                        help="write an undrifted batch (expect a stable monitoring run)")
    parser.set_defaults(drift=True)
    args = parser.parse_args()

    batch, metadata = build_batch(args.rows, args.seed, args.drift)
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    batch.to_parquet(CURRENT_BATCH_FILE, index=False)
    with open(BATCH_METADATA_FILE, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"\nWrote demonstration batch: {CURRENT_BATCH_FILE.relative_to(PROJECT_ROOT)}\n"
        f"  rows            : {len(batch):,}\n"
        f"  labels          : yes ({metadata['label_source']}, NOT observed outcomes)\n"
        f"  simulated drift : {metadata['simulated_drift']}"
        + (f" -> {', '.join(metadata['simulated_drift_features'])}"
           if metadata["simulated_drift_features"] else "")
        + f"\n  metadata        : {BATCH_METADATA_FILE.relative_to(PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
