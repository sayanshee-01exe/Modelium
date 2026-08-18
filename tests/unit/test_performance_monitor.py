"""Step 9 — performance monitoring, and the no-label mode.

The single most important behaviour here is what happens when labels are *absent*. In
credit risk they usually are — a default is observed months after scoring — so the
no-label path is the normal path, not an edge case. It must report
`labels_available: false` and carry no metrics at all, because a plausible-looking
Average Precision computed without outcomes is the most damaging thing this module could
produce.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.data_drift import CRITICAL, STABLE, WARNING
from src.monitoring.performance_monitor import (
    HEALTHY, NOT_AVAILABLE, check_gates, compare_to_baseline, compute_performance,
    normalise_status, performance_report,
)
from src.utils.exceptions import DataValidationError

GATES = {"min_average_precision": 0.15, "min_roc_auc": 0.65, "min_recall": 0.35,
         "min_precision": 0.20, "min_f1": 0.25}
THRESHOLD = 0.5


@pytest.fixture
def separable():
    """A batch the model scores well but not perfectly.

    Deliberately overlapping: perfectly separable scores put every metric at exactly
    1.0, which leaves no headroom to test a gate failing or a drop from baseline —
    `min(1.0 + 0.3, 1.0)` is still 1.0, so the "degradation" would be zero.
    """
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(800), np.ones(200)]).astype(int)
    probabilities = np.concatenate([
        rng.beta(2, 5, 800), rng.beta(5, 2, 200),
    ])
    return y, probabilities


# ---------------------------------------------------------------------------
# No-label mode
# ---------------------------------------------------------------------------

def test_absent_labels_report_not_available(separable) -> None:
    _, probabilities = separable
    report = performance_report(None, probabilities, THRESHOLD, GATES)
    assert report["labels_available"] is False
    assert report["status"] == NOT_AVAILABLE


def test_absent_labels_produce_no_metrics_at_all(separable) -> None:
    """Not nulls that could be mistaken for measurements — no metrics key at all."""
    _, probabilities = separable
    report = performance_report(None, probabilities, THRESHOLD, GATES)
    assert "metrics" not in report
    assert "reason" in report


def test_the_no_label_reason_explains_it_is_normal(separable) -> None:
    _, probabilities = separable
    report = performance_report(None, probabilities, THRESHOLD, GATES)
    assert "labels arrive" in report["reason"]


# ---------------------------------------------------------------------------
# Labelled mode
# ---------------------------------------------------------------------------

def test_every_documented_metric_is_computed(separable) -> None:
    y, probabilities = separable
    metrics = compute_performance(y, probabilities, THRESHOLD)
    for key in ("Average Precision", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1",
                "confusion_matrix", "false_positive_rate", "false_negative_rate",
                "predicted_positive_rate", "actual_positive_rate"):
        assert key in metrics, key


def test_the_confusion_matrix_sums_to_the_batch(separable) -> None:
    y, probabilities = separable
    matrix = compute_performance(y, probabilities, THRESHOLD)["confusion_matrix"]
    assert sum(matrix.values()) == len(y)


def test_a_separable_batch_scores_well(separable) -> None:
    y, probabilities = separable
    metrics = compute_performance(y, probabilities, THRESHOLD)
    assert 0.8 < metrics["ROC-AUC"] < 1.0
    assert metrics["Recall"] > 0.8


def test_the_threshold_changes_the_threshold_dependent_metrics(separable) -> None:
    y, probabilities = separable
    low = compute_performance(y, probabilities, 0.3)
    high = compute_performance(y, probabilities, 0.9)
    assert low["Recall"] > high["Recall"]
    assert low["predicted_positive_rate"] > high["predicted_positive_rate"]
    # Threshold-independent metrics must not move.
    assert low["ROC-AUC"] == pytest.approx(high["ROC-AUC"])


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(DataValidationError, match="length"):
        compute_performance(np.array([0, 1, 0]), np.array([0.1, 0.9]), THRESHOLD)


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        compute_performance(np.array([]), np.array([]), THRESHOLD)


def test_a_single_class_batch_is_refused_rather_than_faked() -> None:
    """ROC-AUC is undefined with one class; a placeholder would be a fabricated number."""
    with pytest.raises(DataValidationError, match="single class"):
        compute_performance(np.zeros(100, dtype=int), np.linspace(0, 1, 100), THRESHOLD)


def test_the_inputs_are_not_mutated(separable) -> None:
    y, probabilities = separable
    before_y, before_p = y.copy(), probabilities.copy()
    compute_performance(y, probabilities, THRESHOLD)
    np.testing.assert_array_equal(y, before_y)
    np.testing.assert_array_equal(probabilities, before_p)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def test_gates_pass_on_a_healthy_batch(separable) -> None:
    y, probabilities = separable
    passed, failed = check_gates(compute_performance(y, probabilities, THRESHOLD), GATES)
    assert failed == [] and len(passed) == len(GATES)


def test_a_failing_gate_is_named() -> None:
    metrics = {"Average Precision": 0.05, "ROC-AUC": 0.55, "Recall": 0.1,
               "Precision": 0.1, "F1": 0.1}
    passed, failed = check_gates(metrics, GATES)
    assert passed == [] and len(failed) == len(GATES)
    assert any("ROC-AUC" in item for item in failed)


def test_a_failing_gate_makes_the_status_critical(separable) -> None:
    y, probabilities = separable
    strict = {**GATES, "min_roc_auc": 0.999}
    report = performance_report(y, probabilities, THRESHOLD, strict)
    assert report["status"] == CRITICAL
    assert report["gates_failed"]


def test_a_healthy_batch_reports_healthy(separable) -> None:
    y, probabilities = separable
    assert performance_report(y, probabilities, THRESHOLD, GATES)["status"] == HEALTHY


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def test_baseline_comparison_reports_each_change(separable) -> None:
    y, probabilities = separable
    metrics = compute_performance(y, probabilities, THRESHOLD)
    baseline = {"Average Precision": 0.30, "ROC-AUC": 0.78}
    comparison = compare_to_baseline(metrics, baseline)
    assert comparison["available"] is True
    assert comparison["metrics"]["ROC-AUC"]["change"] == pytest.approx(
        metrics["ROC-AUC"] - 0.78)


def test_an_absent_baseline_is_reported_not_invented(separable) -> None:
    y, probabilities = separable
    comparison = compare_to_baseline(compute_performance(y, probabilities, THRESHOLD), None)
    assert comparison["available"] is False and "reason" in comparison


def test_a_large_drop_from_baseline_is_critical(separable) -> None:
    y, probabilities = separable
    metrics = compute_performance(y, probabilities, THRESHOLD)
    inflated = {key: min(value + 0.30, 1.0) for key, value in metrics.items()
                if isinstance(value, float) and key in
                ("Average Precision", "ROC-AUC", "Recall", "Precision", "F1", "Accuracy")}
    report = performance_report(y, probabilities, THRESHOLD, GATES, baseline=inflated)
    assert report["status"] == CRITICAL
    assert any("below the champion" in reason for reason in report["reasons"])


def test_a_small_drop_from_baseline_is_only_a_warning(separable) -> None:
    y, probabilities = separable
    metrics = compute_performance(y, probabilities, THRESHOLD)
    slightly_better = {key: min(metrics[key] + 0.03, 1.0)
                       for key in ("Average Precision", "ROC-AUC")}
    report = performance_report(y, probabilities, THRESHOLD, GATES,
                                baseline=slightly_better)
    assert report["status"] == WARNING


# ---------------------------------------------------------------------------
# Monitoring never acts
# ---------------------------------------------------------------------------

def test_the_report_states_that_no_action_was_taken(separable) -> None:
    """The natural next thought on a red status is "retrain" — and that is not this
    stage's call."""
    y, probabilities = separable
    report = performance_report(y, probabilities, THRESHOLD, GATES)
    assert "never retrains" in report["action_taken"]


def test_the_module_contains_no_training_call() -> None:
    source = (PROJECT_ROOT / "src" / "monitoring" / "performance_monitor.py").read_text(
        encoding="utf-8")
    for forbidden in (".fit(", ".fit_transform(", "RandomizedSearchCV"):
        assert forbidden not in source


def test_statuses_map_onto_the_shared_vocabulary() -> None:
    assert normalise_status(HEALTHY) == STABLE
    assert normalise_status(NOT_AVAILABLE) == NOT_AVAILABLE
    assert normalise_status(CRITICAL) == CRITICAL
