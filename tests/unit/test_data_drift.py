"""Step 9 — feature drift detection.

Drift statistics are easy to compute and easy to get subtly wrong, and every mistake
produces a number rather than an error. The contracts here are the ones that would
otherwise fail silently:

*Identical distributions must read stable.* A detector that flags everything is as
useless as one that flags nothing, and at 10,000 rows per side a significance test alone
will flag almost everything — which is why a KS p-value cannot raise a warning without a
supporting effect size.

*A degenerate feature must not lose the run.* Constant, all-null and single-row columns
all exist among the 407 real features. Each is an expected input, not an exception.

*Reference owns the bins and the categories.* Deriving either from the current batch
would hide the change being looked for.

Small synthetic frames only.
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

from src.monitoring.data_drift import (
    CRITICAL, INSUFFICIENT, MIN_KS_EFFECT, MIN_UNSEEN_CATEGORY_RATE, STABLE, WARNING,
    categorical_feature_drift, classify_drift, compute_psi, feature_drift_report,
    jensen_shannon_distance, numeric_feature_drift, summarise_drift,
    total_variation_distance,
)

THRESHOLDS = {
    "psi_warning_threshold": 0.10,
    "psi_critical_threshold": 0.25,
    "ks_pvalue_threshold": 0.05,
    "max_drifted_feature_ratio": 0.20,
}


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------

def test_identical_distributions_have_near_zero_psi(rng) -> None:
    values = rng.normal(size=5000)
    assert compute_psi(values, values) == pytest.approx(0.0, abs=1e-6)


def test_a_shifted_distribution_has_a_large_psi(rng) -> None:
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(1.5, 1, 5000)
    assert compute_psi(reference, current) > 0.25


def test_psi_grows_with_the_size_of_the_shift(rng) -> None:
    reference = rng.normal(0, 1, 5000)
    small = compute_psi(reference, rng.normal(0.2, 1, 5000))
    large = compute_psi(reference, rng.normal(1.5, 1, 5000))
    assert large > small


def test_psi_of_a_constant_column_is_zero_not_an_error() -> None:
    """One distinct value means no bins to shift between — 0.0 is the truthful answer."""
    constant = np.full(500, 7.0)
    assert compute_psi(constant, constant) == 0.0
    assert compute_psi(constant, np.full(500, 9.0)) == 0.0


def test_psi_of_an_empty_side_is_nan() -> None:
    assert np.isnan(compute_psi(np.array([]), np.arange(100.0)))
    assert np.isnan(compute_psi(np.arange(100.0), np.array([])))


def test_psi_ignores_infinities(rng) -> None:
    """An inf from a divide-by-zero ratio would otherwise collapse every other value."""
    reference = rng.normal(size=1000)
    current = np.concatenate([rng.normal(size=1000), [np.inf, -np.inf]])
    assert np.isfinite(compute_psi(reference, current))


def test_psi_is_finite_when_a_bin_is_empty() -> None:
    """Without smoothing, an empty bin makes the log ratio infinite."""
    reference = np.concatenate([np.zeros(500), np.ones(500)])
    current = np.zeros(500)
    assert np.isfinite(compute_psi(reference, current))


def test_psi_counts_values_beyond_the_reference_range(rng) -> None:
    """Values outside the reference's span belong in the end bins, not dropped."""
    reference = rng.uniform(0, 1, 2000)
    current = rng.uniform(5, 6, 2000)
    assert compute_psi(reference, current) > 0.25


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def test_jensen_shannon_is_zero_for_identical_distributions() -> None:
    p = np.array([0.2, 0.3, 0.5])
    assert jensen_shannon_distance(p, p) == pytest.approx(0.0, abs=1e-6)


def test_jensen_shannon_is_bounded_and_symmetric() -> None:
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    forward, backward = jensen_shannon_distance(p, q), jensen_shannon_distance(q, p)
    assert forward == pytest.approx(backward)
    assert 0.0 <= forward <= 1.0


def test_total_variation_distance_bounds() -> None:
    assert total_variation_distance(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.0
    assert total_variation_distance(np.array([1.0, 0.0]),
                                    np.array([0.0, 1.0])) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Numeric features
# ---------------------------------------------------------------------------

def test_numeric_drift_reports_both_sides_statistics(rng) -> None:
    reference = pd.Series(rng.normal(10, 2, 1000))
    current = pd.Series(rng.normal(10, 2, 1000))
    row = numeric_feature_drift(reference, current)
    assert row["feature_type"] == "numeric"
    assert row["reference_mean"] == pytest.approx(10, abs=0.5)
    assert row["current_std"] == pytest.approx(2, abs=0.5)
    assert row["reference_n"] == 1000


def test_numeric_drift_detects_a_missing_rate_change() -> None:
    reference = pd.Series([1.0] * 100)
    current = pd.Series([1.0] * 50 + [np.nan] * 50)
    row = numeric_feature_drift(reference, current)
    assert row["missing_rate_change"] == pytest.approx(0.5)


def test_numeric_drift_on_an_all_null_column_is_insufficient_not_an_error() -> None:
    reference = pd.Series([np.nan] * 200)
    current = pd.Series([np.nan] * 200)
    row = numeric_feature_drift(reference, current)
    status, _ = classify_drift(row, THRESHOLDS)
    assert status == INSUFFICIENT


def test_numeric_drift_on_a_tiny_sample_is_insufficient() -> None:
    row = numeric_feature_drift(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0]))
    status, reason = classify_drift(row, THRESHOLDS)
    assert status == INSUFFICIENT and "usable rows" in reason


def test_ks_is_not_attempted_on_small_samples() -> None:
    row = numeric_feature_drift(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.0, 2.0]))
    assert np.isnan(row["ks_statistic"]) and np.isnan(row["ks_pvalue"])


# ---------------------------------------------------------------------------
# Categorical features
# ---------------------------------------------------------------------------

def test_categorical_drift_detects_unseen_categories() -> None:
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 400 + ["b"] * 400 + ["NEW"] * 200)
    row = categorical_feature_drift(reference, current)
    assert row["unseen_category_rate"] == pytest.approx(0.2)
    status, reason = classify_drift(row, THRESHOLDS)
    assert status in (WARNING, CRITICAL) and "absent from reference" in reason


def test_identical_categorical_distributions_are_stable() -> None:
    values = pd.Series(["a"] * 500 + ["b"] * 300 + ["c"] * 200)
    row = categorical_feature_drift(values, values.copy())
    status, _ = classify_drift(row, THRESHOLDS)
    assert status == STABLE
    assert row["unseen_category_rate"] == 0.0


def test_a_single_rare_unseen_row_does_not_raise_a_warning() -> None:
    """The reference is a sample, so a merely-rare category can be missing by chance."""
    reference = pd.Series(["a"] * 5000 + ["b"] * 5000)
    current = pd.Series(["a"] * 5000 + ["b"] * 4999 + ["rare"])
    row = categorical_feature_drift(reference, current)
    assert 0 < row["unseen_category_rate"] < MIN_UNSEEN_CATEGORY_RATE
    status, _ = classify_drift(row, THRESHOLDS)
    assert status == STABLE


def test_categorical_mix_change_moves_the_distance() -> None:
    reference = pd.Series(["a"] * 800 + ["b"] * 200)
    current = pd.Series(["a"] * 200 + ["b"] * 800)
    row = categorical_feature_drift(reference, current)
    assert row["distribution_distance"] > 0.3
    assert row["total_variation_distance"] == pytest.approx(0.6, abs=0.01)


def test_categorical_drift_on_an_empty_side_is_handled() -> None:
    row = categorical_feature_drift(pd.Series(["a", "b"]), pd.Series([np.nan, np.nan]))
    assert np.isnan(row["psi"])


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_psi_above_critical_is_critical() -> None:
    row = {"psi": 0.4, "reference_n": 500, "current_n": 500}
    status, reason = classify_drift(row, THRESHOLDS)
    assert status == CRITICAL and "critical" in reason


def test_psi_in_the_warning_band_is_a_warning() -> None:
    row = {"psi": 0.15, "reference_n": 500, "current_n": 500}
    status, _ = classify_drift(row, THRESHOLDS)
    assert status == WARNING


def test_a_significant_ks_with_a_negligible_effect_stays_stable() -> None:
    """At 10k rows a p-value alone flags differences far too small to act on."""
    row = {"psi": 0.01, "reference_n": 10_000, "current_n": 10_000,
           "ks_statistic": 0.02, "ks_pvalue": 1e-8}
    status, _ = classify_drift(row, THRESHOLDS)
    assert status == STABLE


def test_a_significant_ks_with_a_real_effect_is_a_warning() -> None:
    row = {"psi": 0.01, "reference_n": 10_000, "current_n": 10_000,
           "ks_statistic": MIN_KS_EFFECT + 0.01, "ks_pvalue": 1e-8}
    status, reason = classify_drift(row, THRESHOLDS)
    assert status == WARNING and "KS statistic" in reason


def test_a_large_missing_rate_move_raises_a_warning() -> None:
    row = {"psi": 0.01, "reference_n": 500, "current_n": 500,
           "missing_rate_change": 0.20}
    status, reason = classify_drift(row, THRESHOLDS)
    assert status == WARNING and "missing rate" in reason


def test_a_stable_feature_says_so() -> None:
    row = {"psi": 0.01, "reference_n": 500, "current_n": 500}
    status, reason = classify_drift(row, THRESHOLDS)
    assert status == STABLE and reason == "within thresholds"


# ---------------------------------------------------------------------------
# The whole report
# ---------------------------------------------------------------------------

def _frames(rng, n=600):
    reference = pd.DataFrame({
        "steady": rng.normal(0, 1, n),
        "shifted": rng.normal(0, 1, n),
        "constant": np.full(n, 3.0),
        "all_null": np.full(n, np.nan),
        "cat_steady": rng.choice(["x", "y"], n),
        "cat_new": rng.choice(["x", "y"], n),
    })
    current = pd.DataFrame({
        "steady": rng.normal(0, 1, n),
        "shifted": rng.normal(2.5, 1, n),
        "constant": np.full(n, 3.0),
        "all_null": np.full(n, np.nan),
        "cat_steady": rng.choice(["x", "y"], n),
        "cat_new": rng.choice(["x", "y", "z"], n),
    })
    return reference, current


def test_report_covers_every_requested_feature(rng) -> None:
    reference, current = _frames(rng)
    frame = feature_drift_report(
        reference, current, ["steady", "shifted", "constant", "all_null"],
        ["cat_steady", "cat_new"], THRESHOLDS)
    assert len(frame) == 6
    assert set(frame["feature"]) == set(reference.columns)


def test_report_has_the_documented_columns(rng) -> None:
    reference, current = _frames(rng)
    frame = feature_drift_report(reference, current, ["steady"], ["cat_steady"], THRESHOLDS)
    for column in ["feature", "feature_type", "psi", "ks_statistic", "ks_pvalue",
                   "distribution_distance", "reference_missing_rate",
                   "current_missing_rate", "drift_status", "drift_reason"]:
        assert column in frame.columns, column


def test_report_finds_the_shifted_feature_and_not_the_steady_one(rng) -> None:
    reference, current = _frames(rng)
    frame = feature_drift_report(
        reference, current, ["steady", "shifted"], [], THRESHOLDS).set_index("feature")
    assert frame.loc["shifted", "drift_status"] == CRITICAL
    assert frame.loc["steady", "drift_status"] == STABLE


def test_report_is_ordered_worst_first(rng) -> None:
    reference, current = _frames(rng)
    frame = feature_drift_report(
        reference, current, ["steady", "shifted", "constant"], [], THRESHOLDS)
    assert frame.iloc[0]["feature"] == "shifted"


def test_a_missing_column_is_reported_not_raised(rng) -> None:
    reference, current = _frames(rng)
    frame = feature_drift_report(
        reference, current.drop(columns=["shifted"]), ["shifted"], [], THRESHOLDS)
    assert frame.iloc[0]["drift_status"] == INSUFFICIENT
    assert "absent" in frame.iloc[0]["drift_reason"]


def test_one_broken_feature_does_not_lose_the_others(rng) -> None:
    """With 407 real features something always breaks; the rest must still be scored."""
    reference, current = _frames(rng)
    reference["explosive"] = pd.Series(["a"] * len(reference))
    current["explosive"] = pd.Series([object()] * len(current))
    frame = feature_drift_report(
        reference, current, ["steady", "shifted"], ["explosive"], THRESHOLDS)
    assert len(frame) == 3
    assert (frame["drift_status"] != "").all()
    scored = frame[frame["feature"].isin(["steady", "shifted"])]
    assert scored["psi"].notna().all()


def test_the_input_frames_are_not_mutated(rng) -> None:
    reference, current = _frames(rng)
    before_ref, before_cur = reference.copy(deep=True), current.copy(deep=True)
    feature_drift_report(reference, current, ["steady", "shifted"],
                         ["cat_steady"], THRESHOLDS)
    pd.testing.assert_frame_equal(reference, before_ref)
    pd.testing.assert_frame_equal(current, before_cur)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_summary_counts_each_status() -> None:
    frame = pd.DataFrame({
        "feature": list("abcde"),
        "feature_type": ["numeric"] * 5,
        "psi": [0.4, 0.15, 0.01, 0.01, np.nan],
        "drift_status": [CRITICAL, WARNING, STABLE, STABLE, INSUFFICIENT],
        "drift_reason": [""] * 5,
    })
    summary = summarise_drift(frame, 0.20)
    assert summary["critical_features"] == 1
    assert summary["warning_features"] == 1
    assert summary["stable_features"] == 2
    assert summary["monitored_features"] == 5


def test_any_critical_feature_makes_the_batch_critical() -> None:
    frame = pd.DataFrame({
        "feature": ["a"] + [f"f{i}" for i in range(99)],
        "feature_type": ["numeric"] * 100,
        "psi": [0.4] + [0.0] * 99,
        "drift_status": [CRITICAL] + [STABLE] * 99,
        "drift_reason": [""] * 100,
    })
    assert summarise_drift(frame, 0.20)["status"] == CRITICAL


def test_too_many_mild_warnings_also_make_the_batch_critical() -> None:
    """Many small moves are a change of regime even with no single critical feature."""
    frame = pd.DataFrame({
        "feature": [f"f{i}" for i in range(10)],
        "feature_type": ["numeric"] * 10,
        "psi": [0.15] * 10,
        "drift_status": [WARNING] * 10,
        "drift_reason": [""] * 10,
    })
    summary = summarise_drift(frame, 0.20)
    assert summary["drifted_feature_ratio"] == 1.0
    assert summary["status"] == CRITICAL


def test_all_stable_is_stable() -> None:
    frame = pd.DataFrame({
        "feature": list("abc"), "feature_type": ["numeric"] * 3, "psi": [0.01] * 3,
        "drift_status": [STABLE] * 3, "drift_reason": [""] * 3,
    })
    summary = summarise_drift(frame, 0.20)
    assert summary["status"] == STABLE and summary["drifted_feature_ratio"] == 0.0
