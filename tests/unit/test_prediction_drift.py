"""Step 9 — drift in the model's output distribution.

The contract that matters most here is the threshold. This project ships a decision
threshold tuned on validation and frozen; comparing positive rates at sklearn's default
0.5 would describe a classifier it does not operate, and would do so while producing
entirely plausible numbers.
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
from src.monitoring.prediction_drift import describe_predictions, prediction_drift_report
from src.utils.exceptions import DataValidationError

LIMITS = {"mean_probability_change_threshold": 0.05,
          "positive_rate_change_threshold": 0.05}
THRESHOLD = 0.6916


def _report(reference, current, threshold=THRESHOLD):
    return prediction_drift_report(reference, current, threshold, LIMITS,
                                   psi_warning=0.10, psi_critical=0.25)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Describing one batch
# ---------------------------------------------------------------------------

def test_description_reports_the_expected_statistics(rng) -> None:
    probabilities = rng.uniform(0, 1, 2000)
    described = describe_predictions(probabilities, THRESHOLD)
    for key in ("n", "mean_probability", "median_probability", "std_probability",
                "positive_rate", "threshold"):
        assert key in described
    assert described["n"] == 2000


def test_the_positive_rate_uses_the_supplied_threshold() -> None:
    probabilities = np.array([0.1, 0.4, 0.8, 0.95])
    assert describe_predictions(probabilities, 0.5)["positive_rate"] == pytest.approx(0.5)
    assert describe_predictions(probabilities, 0.9)["positive_rate"] == pytest.approx(0.25)


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        describe_predictions(np.array([]), THRESHOLD)


def test_probabilities_outside_the_unit_interval_are_refused() -> None:
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        describe_predictions(np.array([0.5, 1.4]), THRESHOLD)


def test_all_non_finite_probabilities_are_refused() -> None:
    with pytest.raises(DataValidationError, match="non-finite"):
        describe_predictions(np.array([np.nan, np.nan]), THRESHOLD)


# ---------------------------------------------------------------------------
# Comparing two batches
# ---------------------------------------------------------------------------

def test_identical_distributions_are_stable(rng) -> None:
    probabilities = rng.beta(2, 8, 4000)
    report = _report(probabilities, probabilities.copy())
    assert report["status"] == STABLE
    assert report["positive_rate_change"] == pytest.approx(0.0)
    assert report["probability_psi"] == pytest.approx(0.0, abs=1e-6)


def test_a_shifted_score_distribution_is_detected(rng) -> None:
    reference = rng.beta(2, 8, 4000)
    current = rng.beta(6, 4, 4000)
    report = _report(reference, current)
    assert report["status"] in (WARNING, CRITICAL)
    assert report["probability_psi"] > 0.25
    assert report["mean_probability_change"] > 0.05


def test_a_positive_rate_move_alone_raises_a_warning() -> None:
    """Scores can shift around the threshold without the mean moving much."""
    reference = np.concatenate([np.full(900, 0.60), np.full(100, 0.75)])
    current = np.concatenate([np.full(700, 0.60), np.full(300, 0.75)])
    report = _report(reference, current)
    assert report["positive_rate_change"] == pytest.approx(0.20, abs=0.01)
    assert report["status"] in (WARNING, CRITICAL)
    assert any("positive rate" in r for r in report["reasons"])


def test_the_frozen_threshold_is_recorded_and_used(rng) -> None:
    probabilities = rng.uniform(0, 1, 1000)
    report = _report(probabilities, probabilities.copy(), threshold=0.42)
    assert report["threshold"] == pytest.approx(0.42)
    assert report["reference"]["threshold"] == pytest.approx(0.42)
    assert "frozen" in report["threshold_source"]


def test_the_threshold_is_not_defaulted_to_one_half() -> None:
    """A different frozen threshold must give a different positive rate."""
    probabilities = np.linspace(0.01, 0.99, 1000)
    at_half = _report(probabilities, probabilities.copy(), threshold=0.5)
    at_frozen = _report(probabilities, probabilities.copy(), threshold=0.9)
    assert at_half["current"]["positive_rate"] != at_frozen["current"]["positive_rate"]


def test_the_report_carries_both_histograms(rng) -> None:
    reference, current = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
    report = _report(reference, current)
    assert len(report["reference_histogram"]) == len(report["histogram_bins"]) - 1
    assert sum(report["reference_histogram"]) == pytest.approx(1.0, abs=1e-6)
    assert sum(report["current_histogram"]) == pytest.approx(1.0, abs=1e-6)


def test_the_report_is_json_serialisable(rng) -> None:
    import json

    report = _report(rng.uniform(0, 1, 500), rng.uniform(0, 1, 500))
    assert json.loads(json.dumps(report, default=str))["status"] in (
        STABLE, WARNING, CRITICAL)


def test_stable_batches_state_that_explicitly(rng) -> None:
    probabilities = rng.beta(2, 8, 3000)
    report = _report(probabilities, probabilities.copy())
    assert report["reasons"] == ["within thresholds"]


def test_the_inputs_are_not_mutated(rng) -> None:
    reference, current = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500)
    before_ref, before_cur = reference.copy(), current.copy()
    _report(reference, current)
    np.testing.assert_array_equal(reference, before_ref)
    np.testing.assert_array_equal(current, before_cur)
