"""Step 9 — status aggregation and report generation.

The aggregation rule carries the weight here: **the overall status is the worst section
status.** A green headline that absorbs a red section is worse than no headline, because
it is the one line an operator reads. Several tests exist only to prove a single bad
section cannot be averaged away.

The report itself must be generated from computed results, not from a template that
would render the same prose whatever the numbers were.
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

from src.monitoring.data_drift import CRITICAL, INSUFFICIENT, STABLE, WARNING
from src.monitoring.performance_monitor import HEALTHY, NOT_AVAILABLE
from src.monitoring.reporting import (
    build_report, build_summary, overall_status, plot_prediction_distribution,
    plot_top_feature_drift, recommend_actions, write_json, write_text,
)


@pytest.fixture
def drift_frame():
    return pd.DataFrame({
        "feature": ["income", "age", "steady"],
        "feature_type": ["numeric"] * 3,
        "psi": [0.40, 0.15, 0.01],
        "ks_statistic": [0.3, 0.1, 0.01],
        "ks_pvalue": [1e-10, 1e-4, 0.8],
        "distribution_distance": [0.4, 0.2, 0.01],
        "reference_missing_rate": [0.0, 0.0, 0.0],
        "current_missing_rate": [0.1, 0.0, 0.0],
        "missing_rate_change": [0.1, 0.0, 0.0],
        "drift_status": [CRITICAL, WARNING, STABLE],
        "drift_reason": ["PSI high", "PSI moderate", "within thresholds"],
    })


@pytest.fixture
def drift_summary():
    return {"monitored_features": 3, "stable_features": 1, "warning_features": 1,
            "critical_features": 1, "insufficient_features": 0, "error_features": 0,
            "drifted_feature_ratio": 0.667, "max_drifted_feature_ratio": 0.20,
            "status": CRITICAL,
            "top_drifted": [{"feature": "income", "feature_type": "numeric", "psi": 0.4,
                             "drift_status": CRITICAL, "drift_reason": "PSI high"}]}


@pytest.fixture
def prediction_drift():
    return {
        "reference": {"mean_probability": 0.08, "median_probability": 0.05,
                      "std_probability": 0.1, "positive_rate": 0.03, "threshold": 0.69,
                      "n": 100},
        "current": {"mean_probability": 0.09, "median_probability": 0.06,
                    "std_probability": 0.11, "positive_rate": 0.04, "threshold": 0.69,
                    "n": 100},
        "threshold": 0.69, "threshold_source": "frozen deployment metadata",
        "probability_psi": 0.02, "probability_distribution_distance": 0.05,
        "mean_probability_change": 0.01, "median_probability_change": 0.01,
        "std_probability_change": 0.01, "positive_rate_change": 0.01,
        "status": STABLE, "reasons": ["within thresholds"],
        "histogram_bins": np.linspace(0, 1, 11).tolist(),
        "reference_histogram": [0.1] * 10, "current_histogram": [0.1] * 10,
    }


@pytest.fixture
def performance_absent():
    return {"labels_available": False, "status": NOT_AVAILABLE,
            "reason": "the current batch carries no observed outcomes"}


@pytest.fixture
def fairness_stable():
    return {"status": STABLE, "labels_available": False, "minimum_group_size": 100,
            "group_columns_used": ["CODE_GENDER"], "group_columns_absent": [],
            "summaries": [{"group_column": "CODE_GENDER", "groups_measured": 2,
                           "groups_insufficient": 1,
                           "demographic_parity_difference": 0.02,
                           "equal_opportunity_difference": None,
                           "recall_difference": None}],
            "table": pd.DataFrame(), "reasons": ["within limits"],
            "limitations": "demonstration only", "action_taken": "none"}


def _summary(drift_summary, prediction_drift, performance, fairness):
    return build_summary(
        model_info={"registered_model_name": "m", "model_version": "2",
                    "model_alias": "champion", "source_run_id": "abc"},
        reference_name="ref", current_name="cur",
        reference_rows=100, current_rows=100,
        drift_summary=drift_summary, prediction_drift=prediction_drift,
        performance=performance, fairness=fairness, artifacts={},
    )


# ---------------------------------------------------------------------------
# Status aggregation
# ---------------------------------------------------------------------------

def test_all_stable_is_stable() -> None:
    assert overall_status({"a": STABLE, "b": STABLE, "c": HEALTHY}) == STABLE


def test_a_single_critical_section_makes_the_whole_run_critical() -> None:
    assert overall_status({"a": STABLE, "b": STABLE, "c": CRITICAL}) == CRITICAL


def test_a_warning_beats_stable() -> None:
    assert overall_status({"a": STABLE, "b": WARNING}) == WARNING


def test_critical_beats_warning() -> None:
    assert overall_status({"a": WARNING, "b": CRITICAL}) == CRITICAL


def test_not_available_does_not_mask_a_warning() -> None:
    """An unmeasured section is not evidence of a problem, but must not hide one."""
    assert overall_status({"a": NOT_AVAILABLE, "b": WARNING}) == WARNING


def test_not_available_alone_is_not_a_failure() -> None:
    assert overall_status({"a": NOT_AVAILABLE, "b": STABLE}) == NOT_AVAILABLE


def test_an_empty_status_set_is_not_available() -> None:
    assert overall_status({}) == NOT_AVAILABLE


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

def test_summary_carries_every_documented_field(
    drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    for key in ("registered_model_name", "model_version", "model_alias", "source_run_id",
                "monitoring_timestamp", "reference_dataset", "current_dataset",
                "reference_rows", "current_rows", "monitored_features",
                "warning_features", "critical_features", "prediction_drift_status",
                "performance_status", "fairness_status", "overall_status",
                "labels_available", "artifacts"):
        assert key in summary, key


def test_summary_keeps_each_section_status_visible(
    drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    """The overall status says how bad; the sections say where."""
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    assert summary["overall_status"] == CRITICAL
    assert summary["section_statuses"] == {
        "data_drift": CRITICAL, "prediction_drift": STABLE,
        "performance": NOT_AVAILABLE, "fairness": STABLE,
    }


def test_summary_records_that_nothing_was_acted_on(
    drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    assert "never retrains" in summary["action_taken"]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_report_contains_every_required_section(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable)
    for heading in ("# Monitoring report", "## Model", "## Batches", "## Feature drift",
                    "## Prediction drift", "## Performance", "## Fairness",
                    "## Limitations", "## Recommended next actions"):
        assert heading in report, heading


def test_report_names_the_actually_drifted_features(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    """Computed results, not template prose."""
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable)
    assert "income" in report
    assert "0.400" in report


def test_report_states_it_is_offline_batch_monitoring(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable)
    assert "offline batch monitoring" in report.lower()
    assert "no live traffic" in report.lower() or "no traffic" in report.lower()


def test_report_explains_a_missing_performance_section(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable)
    assert "no observed outcomes" in report
    assert "not_available" in report


def test_report_marks_a_demonstration_batch(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    """Simulated drift must never read as observed production behaviour."""
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    metadata = {"demonstration": True, "description": "synthetic",
                "simulated_drift_features": ["income"]}
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable, metadata)
    assert "Demonstration batch" in report
    assert "simulated, not observed" in report


def test_report_carries_the_fairness_limitations(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable)
    assert "demonstration only" in report


def test_report_says_it_takes_no_action(
    drift_frame, drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance_absent, fairness_stable)
    assert "does not retrain" in report


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def test_critical_drift_produces_an_investigation_action(
    drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    actions = recommend_actions(summary, drift_summary, prediction_drift,
                                performance_absent, fairness_stable)
    assert any("critical feature" in action for action in actions)


def test_absent_labels_produce_a_re_run_action(
    drift_summary, prediction_drift, performance_absent, fairness_stable,
) -> None:
    summary = _summary(drift_summary, prediction_drift, performance_absent,
                       fairness_stable)
    actions = recommend_actions(summary, drift_summary, prediction_drift,
                               performance_absent, fairness_stable)
    assert any("once outcomes mature" in action for action in actions)


def test_a_clean_run_recommends_nothing(prediction_drift, fairness_stable) -> None:
    clean = {"monitored_features": 3, "stable_features": 3, "warning_features": 0,
             "critical_features": 0, "insufficient_features": 0, "error_features": 0,
             "drifted_feature_ratio": 0.0, "max_drifted_feature_ratio": 0.20,
             "status": STABLE, "top_drifted": []}
    performance = {"labels_available": True, "status": HEALTHY, "metrics": {},
                   "reasons": []}
    summary = _summary(clean, prediction_drift, performance, fairness_stable)
    actions = recommend_actions(summary, clean, prediction_drift, performance,
                                fairness_stable)
    assert actions == ["No action indicated — every section is within its thresholds."]


# ---------------------------------------------------------------------------
# Writers and plots
# ---------------------------------------------------------------------------

def test_json_and_text_writers_create_parents(tmp_path) -> None:
    import json

    json_path = write_json({"a": 1}, tmp_path / "nested" / "x.json")
    text_path = write_text("hello", tmp_path / "nested" / "x.md")
    assert json.loads(json_path.read_text())["a"] == 1
    assert text_path.read_text() == "hello"


def test_the_drift_plot_is_written(drift_frame, tmp_path) -> None:
    path = plot_top_feature_drift(drift_frame, tmp_path / "drift.png")
    assert path is not None and path.exists() and path.stat().st_size > 0


def test_the_drift_plot_is_skipped_when_nothing_is_scorable(tmp_path) -> None:
    """A blank plot is worse than an absent one."""
    empty = pd.DataFrame({"feature": ["a"], "psi": [np.nan], "drift_status": [INSUFFICIENT]})
    assert plot_top_feature_drift(empty, tmp_path / "drift.png") is None


def test_the_prediction_plot_is_written(prediction_drift, tmp_path) -> None:
    path = plot_prediction_distribution(prediction_drift, tmp_path / "pred.png")
    assert path.exists() and path.stat().st_size > 0
