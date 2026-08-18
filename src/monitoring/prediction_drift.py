"""Drift in what the model *outputs*, as opposed to what it is fed.

Feature drift and prediction drift answer different questions, and either can move
without the other. Features can shift in ways the model is insensitive to, leaving
scores unchanged; and the score distribution can shift from a small move in one
high-importance feature that barely registers across 407 columns. Both are monitored.

The decision threshold is read from the frozen deployment metadata, never defaulted to
0.5. The champion ships with a threshold tuned on validation — comparing positive rates
at 0.5 would describe a classifier this project does not operate.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.monitoring.data_drift import (
    CRITICAL, EPS, STABLE, WARNING, compute_psi, jensen_shannon_distance,
)
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Fixed bins over [0, 1]. Unlike feature PSI, which uses reference quantiles, the
# probability scale is known in advance, so fixed bins keep the number comparable
# between monitoring runs rather than relative to each run's reference.
PROBABILITY_BINS = np.linspace(0.0, 1.0, 11)


def describe_predictions(probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Summary statistics for one batch of predicted probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.size == 0:
        raise DataValidationError("Cannot describe an empty set of predictions")
    finite = probabilities[np.isfinite(probabilities)]
    if finite.size == 0:
        raise DataValidationError("All predicted probabilities are non-finite")
    if finite.min() < 0.0 or finite.max() > 1.0:
        raise DataValidationError(
            f"Predicted probabilities must lie in [0, 1], got "
            f"[{finite.min()}, {finite.max()}]"
        )
    return {
        "n": int(finite.size),
        "mean_probability": float(finite.mean()),
        "median_probability": float(np.median(finite)),
        "std_probability": float(finite.std()),
        "min_probability": float(finite.min()),
        "max_probability": float(finite.max()),
        "positive_rate": float((finite >= threshold).mean()),
        "threshold": float(threshold),
    }


def prediction_drift_report(
    reference_probabilities: np.ndarray,
    current_probabilities: np.ndarray,
    threshold: float,
    thresholds: dict[str, float],
    psi_warning: float,
    psi_critical: float,
) -> dict[str, Any]:
    """Compare two score distributions at the frozen threshold.

    Args:
        reference_probabilities: Champion scores on the reference batch.
        current_probabilities: Champion scores on the current batch.
        threshold: The frozen decision threshold from deployment metadata.
        thresholds: `monitoring.prediction` limits for the mean and positive-rate moves.
        psi_warning: PSI level at which the score distribution counts as warning.
        psi_critical: PSI level at which it counts as critical.

    Returns:
        A JSON-serialisable mapping carrying both batches' statistics, the changes
        between them, and an overall status with the reasons that produced it.
    """
    reference = describe_predictions(reference_probabilities, threshold)
    current = describe_predictions(current_probabilities, threshold)

    psi = compute_psi(reference_probabilities, current_probabilities, buckets=10)
    ref_hist = np.histogram(np.asarray(reference_probabilities, dtype=float),
                            bins=PROBABILITY_BINS)[0].astype(float)
    cur_hist = np.histogram(np.asarray(current_probabilities, dtype=float),
                            bins=PROBABILITY_BINS)[0].astype(float)
    distance = jensen_shannon_distance(ref_hist, cur_hist)

    mean_change = current["mean_probability"] - reference["mean_probability"]
    positive_rate_change = current["positive_rate"] - reference["positive_rate"]

    reasons: list[str] = []
    status = STABLE

    if np.isfinite(psi):
        if psi >= psi_critical:
            status = CRITICAL
            reasons.append(f"probability PSI {psi:.3f} >= critical {psi_critical}")
        elif psi >= psi_warning:
            status = WARNING
            reasons.append(f"probability PSI {psi:.3f} >= warning {psi_warning}")

    mean_limit = thresholds["mean_probability_change_threshold"]
    if abs(mean_change) >= mean_limit:
        reasons.append(f"mean probability moved {mean_change:+.4f} (limit {mean_limit})")
        status = CRITICAL if status == CRITICAL else WARNING

    rate_limit = thresholds["positive_rate_change_threshold"]
    if abs(positive_rate_change) >= rate_limit:
        reasons.append(
            f"positive rate moved {positive_rate_change:+.4f} (limit {rate_limit})")
        status = CRITICAL if status == CRITICAL else WARNING

    return {
        "reference": reference,
        "current": current,
        "threshold": float(threshold),
        "threshold_source": "frozen deployment metadata",
        "probability_psi": float(psi) if np.isfinite(psi) else None,
        "probability_distribution_distance": float(distance),
        "mean_probability_change": float(mean_change),
        "median_probability_change": float(
            current["median_probability"] - reference["median_probability"]),
        "std_probability_change": float(
            current["std_probability"] - reference["std_probability"]),
        "positive_rate_change": float(positive_rate_change),
        "status": status,
        "reasons": reasons or ["within thresholds"],
        "histogram_bins": PROBABILITY_BINS.tolist(),
        "reference_histogram": (ref_hist / max(ref_hist.sum(), EPS)).tolist(),
        "current_histogram": (cur_hist / max(cur_hist.sum(), EPS)).tolist(),
    }
