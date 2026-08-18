"""Model performance on a labelled production batch, and how far it has degraded.

Labels arrive late in credit risk — a default is only observed after the loan has had
time to go bad — so a monitoring batch usually has none. That shapes the contract here:
**labels are optional, and their absence is reported, never filled in.** A monitoring
report that prints an Average Precision computed from predictions alone would be
fabricating the one number a reviewer most wants to trust.

Degradation is measured against the champion's own recorded test metrics rather than
against a hard-coded expectation, so the comparison stays meaningful after a retrain
moves the baseline.

Nothing here retrains, and a failed gate raises no automatic action. Monitoring reports;
deciding what to do about a drop is a human decision this stage deliberately does not
make.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.monitoring.data_drift import CRITICAL, STABLE, WARNING
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

HEALTHY = "healthy"
NOT_AVAILABLE = "not_available"

# Metric keys as the training pipeline records them, so the baseline in
# deployment_meta.json can be compared without a translation table.
METRIC_KEYS = ("Average Precision", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1")

# params.yaml gate name -> metric key it constrains.
GATE_TO_METRIC = {
    "min_average_precision": "Average Precision",
    "min_roc_auc": "ROC-AUC",
    "min_recall": "Recall",
    "min_precision": "Precision",
    "min_f1": "F1",
}

# A metric may sit this far below its baseline before the drop is called a warning;
# twice this makes it critical. Expressed in absolute metric points.
DEGRADATION_WARNING = 0.02
DEGRADATION_CRITICAL = 0.05


def compute_performance(
    y_true, probabilities, threshold: float,
) -> dict[str, Any]:
    """Threshold-dependent and threshold-independent metrics for a labelled batch.

    Args:
        y_true: Observed binary outcomes.
        probabilities: Champion probabilities for the positive class.
        threshold: The frozen decision threshold.

    Raises:
        DataValidationError: on length mismatch, empty input, or a single-class batch —
            ROC-AUC and Average Precision are undefined with one class present, and
            returning a placeholder would put a meaningless number in the report.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix, f1_score,
        precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    if y_true.size == 0:
        raise DataValidationError("Cannot score performance on an empty batch")
    if y_true.size != probabilities.size:
        raise DataValidationError(
            f"Labels and predictions disagree in length: {y_true.size} vs "
            f"{probabilities.size}"
        )
    if np.unique(y_true).size < 2:
        raise DataValidationError(
            "The labelled batch contains a single class, so ROC-AUC and Average "
            "Precision are undefined. Monitor drift instead of fabricating a score."
        )

    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return {
        "Average Precision": float(average_precision_score(y_true, probabilities)),
        "ROC-AUC": float(roc_auc_score(y_true, probabilities)),
        "Accuracy": float(accuracy_score(y_true, predicted)),
        "Precision": float(precision_score(y_true, predicted, zero_division=0)),
        "Recall": float(recall_score(y_true, predicted, zero_division=0)),
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "predicted_positive_rate": float(predicted.mean()),
        "actual_positive_rate": float(y_true.mean()),
        "n": int(y_true.size),
    }


def check_gates(metrics: dict[str, Any], gates: dict[str, float]) -> tuple[list[str], list[str]]:
    """Compare metrics against the configured floors. Returns ``(passed, failed)``."""
    passed: list[str] = []
    failed: list[str] = []
    for gate, key in GATE_TO_METRIC.items():
        if gate not in gates or key not in metrics:
            continue
        floor = float(gates[gate])
        value = float(metrics[key])
        description = f"{key} {value:.4f} vs floor {floor:.4f}"
        (passed if value >= floor else failed).append(description)
    return passed, failed


def compare_to_baseline(metrics: dict[str, Any],
                        baseline: dict[str, Any] | None) -> dict[str, Any]:
    """Metric-by-metric change against the champion's recorded test performance."""
    if not baseline:
        return {"available": False,
                "reason": "no baseline metrics recorded for the registered champion"}

    changes: dict[str, Any] = {}
    for key in METRIC_KEYS:
        if key in metrics and key in baseline:
            try:
                changes[key] = {
                    "current": float(metrics[key]),
                    "baseline": float(baseline[key]),
                    "change": float(metrics[key]) - float(baseline[key]),
                }
            except (TypeError, ValueError):
                continue
    return {"available": True, "metrics": changes}


def performance_report(
    y_true, probabilities, threshold: float,
    gates: dict[str, float], baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full performance section, including degradation status.

    Returns a mapping that always carries ``labels_available``; when it is False the
    mapping contains no metrics at all rather than nulls that could be mistaken for
    measurements.
    """
    if y_true is None:
        logger.info("No labels in this batch; performance monitoring is skipped.")
        return {
            "labels_available": False,
            "status": NOT_AVAILABLE,
            "reason": (
                "the current batch carries no observed outcomes. In credit risk labels "
                "arrive months after scoring, so this is the normal case, not a fault."
            ),
        }

    metrics = compute_performance(y_true, probabilities, threshold)
    passed, failed = check_gates(metrics, gates)
    comparison = compare_to_baseline(metrics, baseline)

    reasons: list[str] = []
    status = HEALTHY
    if failed:
        status = CRITICAL
        reasons.extend(f"below floor: {item}" for item in failed)

    if comparison.get("available"):
        for key, change in comparison["metrics"].items():
            drop = -change["change"]
            if drop >= DEGRADATION_CRITICAL:
                status = CRITICAL
                reasons.append(
                    f"{key} fell {drop:.4f} below the champion's test baseline "
                    f"(critical at {DEGRADATION_CRITICAL})")
            elif drop >= DEGRADATION_WARNING:
                if status == HEALTHY:
                    status = WARNING
                reasons.append(
                    f"{key} fell {drop:.4f} below the champion's test baseline "
                    f"(warning at {DEGRADATION_WARNING})")

    return {
        "labels_available": True,
        "status": status,
        "metrics": metrics,
        "gates_passed": passed,
        "gates_failed": failed,
        "baseline_comparison": comparison,
        "reasons": reasons or ["all gates passed and no material degradation"],
        # Stated explicitly because the natural next thought on seeing a red status is
        # "should this retrain?" — and that is not this stage's call to make.
        "action_taken": "none — monitoring reports status and never retrains, "
                        "re-registers, or changes an alias",
    }


def normalise_status(status: str) -> str:
    """Map the performance vocabulary onto the drift vocabulary for the summary."""
    return {HEALTHY: STABLE, NOT_AVAILABLE: NOT_AVAILABLE}.get(status, status)
