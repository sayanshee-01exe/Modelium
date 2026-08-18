"""Step 9 — group-wise disparity measurement.

The behaviours worth pinning are the ones that keep the output honest rather than
flattering:

*A group below the minimum size is reported, never merged.* Combining unrelated segments
to reach a sample-size threshold produces a confident number about a population that does
not exist.

*Label-dependent metrics require labels.* Recall and equal opportunity are undefined
without outcomes, and must stay NaN rather than being computed from predictions alone.

*The limitations travel with the numbers.* Every report carries the disclaimer, because
a disparity table separated from its caveats reads as a compliance finding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.data_drift import STABLE, WARNING
from src.monitoring.fairness_monitor import (
    DEFAULT_GROUP_COLUMNS, DISCLAIMER, INSUFFICIENT, NOT_AVAILABLE, disparity,
    fairness_report, group_metrics,
)

SETTINGS = {"enabled": True, "minimum_group_size": 100,
            "max_demographic_parity_difference": 0.10,
            "max_equal_opportunity_difference": 0.10}
THRESHOLD = 0.5


@pytest.fixture
def batch():
    """400 rows across three groups, one of which is deliberately tiny."""
    rng = np.random.default_rng(0)
    groups = ["A"] * 200 + ["B"] * 190 + ["TINY"] * 10
    frame = pd.DataFrame({
        "CODE_GENDER": groups,
        "NAME_INCOME_TYPE": rng.choice(["W", "P"], 400),
    })
    probabilities = np.concatenate([
        rng.uniform(0.5, 1.0, 200),   # group A scored high
        rng.uniform(0.0, 0.5, 190),   # group B scored low
        rng.uniform(0.0, 1.0, 10),
    ])
    labels = rng.integers(0, 2, 400)
    return frame, probabilities, pd.Series(labels)


# ---------------------------------------------------------------------------
# Per-group metrics
# ---------------------------------------------------------------------------

def test_metrics_are_computed_per_group(batch) -> None:
    frame, probabilities, labels = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD, y_true=labels)
    assert set(result["group_value"]) == {"A", "B", "TINY"}


def test_group_rates_differ_when_the_scores_differ(batch) -> None:
    frame, probabilities, labels = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD,
                           y_true=labels).set_index("group_value")
    assert result.loc["A", "positive_prediction_rate"] > 0.9
    assert result.loc["B", "positive_prediction_rate"] < 0.1


def test_a_small_group_is_marked_insufficient(batch) -> None:
    frame, probabilities, labels = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD,
                           y_true=labels).set_index("group_value")
    assert result.loc["TINY", "status"] == INSUFFICIENT
    assert result.loc["TINY", "n"] == 10


def test_a_small_group_reports_no_rates(batch) -> None:
    """An unreliable number is worse than an absent one."""
    frame, probabilities, labels = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD,
                           y_true=labels).set_index("group_value")
    assert np.isnan(result.loc["TINY", "positive_prediction_rate"])
    assert np.isnan(result.loc["TINY", "recall"])


def test_a_small_group_is_not_merged_into_another(batch) -> None:
    frame, probabilities, labels = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD, y_true=labels)
    # Every group survives as its own row and the counts still add up, which is what
    # "not merged" means structurally.
    assert len(result) == 3
    assert result["n"].sum() == 400
    tiny = result.set_index("group_value").loc["TINY"]
    assert tiny["n"] == 10
    assert "rather than merged" in tiny["note"]


def test_label_dependent_metrics_are_absent_without_labels(batch) -> None:
    frame, probabilities, _ = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD, y_true=None)
    measured = result[result["status"] == "measured"]
    assert measured["recall"].isna().all()
    assert measured["actual_default_rate"].isna().all()
    # Prediction-only metrics still carry information.
    assert measured["positive_prediction_rate"].notna().all()


def test_label_dependent_metrics_are_present_with_labels(batch) -> None:
    frame, probabilities, labels = batch
    result = group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD, y_true=labels)
    measured = result[result["status"] == "measured"]
    assert measured["actual_default_rate"].notna().all()
    assert measured["false_positive_rate"].notna().all()


def test_the_inputs_are_not_mutated(batch) -> None:
    frame, probabilities, labels = batch
    before_frame, before_probs = frame.copy(deep=True), probabilities.copy()
    group_metrics(frame["CODE_GENDER"], probabilities, THRESHOLD, y_true=labels)
    pd.testing.assert_frame_equal(frame, before_frame)
    np.testing.assert_array_equal(probabilities, before_probs)


# ---------------------------------------------------------------------------
# Disparity
# ---------------------------------------------------------------------------

def test_disparity_is_the_spread_across_measured_groups() -> None:
    frame = pd.DataFrame({
        "status": ["measured", "measured", INSUFFICIENT],
        "positive_prediction_rate": [0.10, 0.45, 0.99],
    })
    assert disparity(frame, "positive_prediction_rate") == pytest.approx(0.35)


def test_disparity_ignores_insufficient_groups() -> None:
    """A tiny group's extreme rate must not drive the headline disparity."""
    frame = pd.DataFrame({
        "status": ["measured", INSUFFICIENT],
        "positive_prediction_rate": [0.10, 0.99],
    })
    assert np.isnan(disparity(frame, "positive_prediction_rate"))


def test_disparity_needs_two_groups() -> None:
    frame = pd.DataFrame({"status": ["measured"], "positive_prediction_rate": [0.4]})
    assert np.isnan(disparity(frame, "positive_prediction_rate"))


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_a_large_disparity_raises_a_warning(batch) -> None:
    frame, probabilities, labels = batch
    report = fairness_report(frame, probabilities, THRESHOLD, SETTINGS, y_true=labels)
    assert report["status"] == WARNING
    assert any("CODE_GENDER" in reason for reason in report["reasons"])


def test_a_small_disparity_stays_stable() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"CODE_GENDER": ["A"] * 200 + ["B"] * 200})
    probabilities = rng.uniform(0, 1, 400)
    report = fairness_report(frame, probabilities, THRESHOLD, SETTINGS)
    assert report["status"] == STABLE


def test_the_report_carries_its_limitations(batch) -> None:
    frame, probabilities, labels = batch
    report = fairness_report(frame, probabilities, THRESHOLD, SETTINGS, y_true=labels)
    assert report["limitations"] == DISCLAIMER
    assert "not verified protected attributes" in report["limitations"]
    assert "legal compliance" in report["limitations"]


def test_the_report_states_no_action_was_taken(batch) -> None:
    frame, probabilities, labels = batch
    report = fairness_report(frame, probabilities, THRESHOLD, SETTINGS, y_true=labels)
    assert "changes no model" in report["action_taken"]


def test_absent_grouping_columns_are_reported_not_raised() -> None:
    frame = pd.DataFrame({"unrelated": [1, 2, 3]})
    report = fairness_report(frame, np.array([0.1, 0.2, 0.3]), THRESHOLD, SETTINGS)
    assert report["status"] == NOT_AVAILABLE
    assert "none of the configured grouping columns" in report["reason"]


def test_only_present_columns_are_used(batch) -> None:
    frame, probabilities, labels = batch
    report = fairness_report(frame, probabilities, THRESHOLD, SETTINGS, y_true=labels)
    assert set(report["group_columns_used"]) == {"CODE_GENDER", "NAME_INCOME_TYPE"}
    assert "NAME_EDUCATION_TYPE" in report["group_columns_absent"]


def test_disabling_fairness_skips_it(batch) -> None:
    frame, probabilities, labels = batch
    report = fairness_report(frame, probabilities, THRESHOLD,
                             {**SETTINGS, "enabled": False}, y_true=labels)
    assert report["status"] == NOT_AVAILABLE
    assert "disabled" in report["reason"]


def test_all_groups_below_the_minimum_is_insufficient() -> None:
    frame = pd.DataFrame({"CODE_GENDER": ["A"] * 5 + ["B"] * 5})
    report = fairness_report(frame, np.linspace(0, 1, 10), THRESHOLD, SETTINGS)
    assert report["status"] == INSUFFICIENT


def test_equal_opportunity_needs_labels(batch) -> None:
    """It compares true-positive rates, which do not exist without outcomes."""
    frame, probabilities, labels = batch
    unlabelled = fairness_report(frame, probabilities, THRESHOLD, SETTINGS)
    labelled = fairness_report(frame, probabilities, THRESHOLD, SETTINGS, y_true=labels)
    gender = next(s for s in unlabelled["summaries"] if s["group_column"] == "CODE_GENDER")
    assert gender["equal_opportunity_difference"] is None
    assert gender["demographic_parity_difference"] is not None
    gender_labelled = next(s for s in labelled["summaries"]
                           if s["group_column"] == "CODE_GENDER")
    assert gender_labelled["equal_opportunity_difference"] is not None


def test_the_table_is_returned_for_writing(batch) -> None:
    frame, probabilities, labels = batch
    report = fairness_report(frame, probabilities, THRESHOLD, SETTINGS, y_true=labels)
    table = report["table"]
    assert isinstance(table, pd.DataFrame) and not table.empty
    assert "group_column" in table.columns and "group_value" in table.columns


def test_the_default_group_columns_are_documented_proxies() -> None:
    assert "CODE_GENDER" in DEFAULT_GROUP_COLUMNS
    assert len(DEFAULT_GROUP_COLUMNS) == 4
