"""Step 4 — champion selection and two-stage quality gates.

Two contracts here. Ranking uses **Average Precision**, the same estimator the search
optimises — previously the search optimised average_precision while selection ranked by
a trapezoidal PR-AUC, so the model that won the search was not necessarily the model
that won selection.

And gates are split by when they become meaningful. Recall/Precision/F1 depend on a
decision threshold, so before threshold tuning they describe the arbitrary 0.5 default,
not the cut-off the pipeline will actually ship. Pre-threshold gates are therefore
restricted to threshold-independent metrics, with the operational gates applied after.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.selection import (
    DEFAULT_OPERATIONAL_GATES,
    DEFAULT_QUALITY_GATES,
    PRIMARY_METRIC,
    THRESHOLD_DEPENDENT_METRICS,
    build_leaderboard,
    check_operational_gates,
    check_quality_gates,
    select_champion,
)


def _metrics(name, ap, roc_auc=0.78, recall=0.62, precision=0.20, f1=0.30, accuracy=0.70):
    return {"Model": name, "Average Precision": ap, "ROC-AUC": roc_auc, "Recall": recall,
            "Precision": precision, "F1": f1, "Accuracy": accuracy}


@pytest.fixture
def results():
    """Not in AP order, and the ROC-AUC leader is deliberately NOT the AP leader."""
    return [
        _metrics("Logistic Regression", ap=0.18, roc_auc=0.82),
        _metrics("LightGBM (Tuned)", ap=0.27, roc_auc=0.79),
        _metrics("Random Forest (Tuned)", ap=0.21, roc_auc=0.76),
        _metrics("XGBoost (Tuned)", ap=0.25, roc_auc=0.80),
    ]


# --------------------------------------------------------------------- leaderboard

def test_primary_metric_is_average_precision() -> None:
    """Must match the search's `scoring="average_precision"` exactly."""
    assert PRIMARY_METRIC == "Average Precision"


def test_leaderboard_ranks_by_average_precision(results) -> None:
    board = build_leaderboard(results)
    assert list(board["Model"]) == [
        "LightGBM (Tuned)", "XGBoost (Tuned)", "Random Forest (Tuned)", "Logistic Regression",
    ]


def test_leaderboard_does_not_rank_by_roc_auc(results) -> None:
    board = build_leaderboard(results)
    assert board.iloc[0]["Model"] != "Logistic Regression"


def test_baseline_appears_in_leaderboard(results) -> None:
    board = build_leaderboard(results)
    assert "Logistic Regression" in list(board["Model"]) and len(board) == 4


def test_leaderboard_keeps_secondary_metrics(results) -> None:
    board = build_leaderboard(results)
    for column in ("Recall", "ROC-AUC", "F1", "Precision", "Accuracy"):
        assert column in board.columns


def test_leaderboard_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        build_leaderboard([])


def test_leaderboard_rejects_missing_primary_metric() -> None:
    with pytest.raises(ValueError, match="Average Precision"):
        build_leaderboard([{"Model": "X", "ROC-AUC": 0.8}])


# ----------------------------------------------- pre-threshold gates (§9 ordering)

def test_default_gates_are_threshold_independent_only() -> None:
    """The whole point of §9: no Recall/Precision/F1 gate before threshold tuning."""
    assert set(DEFAULT_QUALITY_GATES) & THRESHOLD_DEPENDENT_METRICS == set()
    assert set(DEFAULT_QUALITY_GATES) == {"Average Precision", "ROC-AUC"}


def test_pre_threshold_gate_rejects_a_threshold_dependent_metric() -> None:
    """Guard the ordering in code, not just in convention."""
    with pytest.raises(ValueError, match="threshold"):
        check_quality_gates(_metrics("m", ap=0.3), gates={"Recall": 0.5})


def test_quality_gate_passes_valid_metrics() -> None:
    passed, failures = check_quality_gates(_metrics("good", ap=0.30, roc_auc=0.80))
    assert passed and failures == []


def test_quality_gate_rejects_failing_metrics() -> None:
    passed, failures = check_quality_gates(_metrics("bad", ap=0.02, roc_auc=0.51))
    assert not passed and len(failures) == 2


def test_quality_gate_reports_which_metric_failed() -> None:
    passed, failures = check_quality_gates(_metrics("partial", ap=0.30, roc_auc=0.51))
    assert not passed
    assert len(failures) == 1 and "ROC-AUC" in failures[0]


def test_quality_gates_are_configurable() -> None:
    metrics = _metrics("m", ap=0.30, roc_auc=0.80)
    assert check_quality_gates(metrics, gates={"Average Precision": 0.99})[0] is False
    assert check_quality_gates(metrics, gates={"Average Precision": 0.10})[0] is True


def test_default_gates_are_conservative() -> None:
    """Floors for 'is this model broken', not targets. At an ~8% positive rate a random
    classifier scores AP ~0.08 and ROC-AUC 0.50."""
    assert 0.08 < DEFAULT_QUALITY_GATES["Average Precision"] < 0.30
    assert 0.50 < DEFAULT_QUALITY_GATES["ROC-AUC"] < 0.7692


def test_missing_metric_is_treated_as_a_gate_failure() -> None:
    passed, failures = check_quality_gates({"Model": "m", "Average Precision": 0.3})
    assert not passed and any("ROC-AUC" in f for f in failures)


# ------------------------------------------- post-threshold operational gates (§10)

def test_operational_gates_are_threshold_dependent() -> None:
    assert set(DEFAULT_OPERATIONAL_GATES) <= THRESHOLD_DEPENDENT_METRICS
    assert "Recall" in DEFAULT_OPERATIONAL_GATES


def test_operational_gate_passes_valid_metrics() -> None:
    passed, failures = check_operational_gates({"Recall": 0.65, "Precision": 0.2})
    assert passed and failures == []


def test_operational_gate_rejects_low_recall() -> None:
    """Applied to tuned-threshold metrics, where Recall finally means something."""
    passed, failures = check_operational_gates({"Recall": 0.05})
    assert not passed and any("Recall" in f for f in failures)


def test_operational_gates_are_configurable() -> None:
    assert check_operational_gates({"Recall": 0.5}, gates={"Recall": 0.9})[0] is False
    assert check_operational_gates({"Recall": 0.5}, gates={"Recall": 0.1})[0] is True


# ----------------------------------------------------------------------- champion

def test_champion_is_the_average_precision_leader(results) -> None:
    champion = select_champion(results)
    assert champion.name == "LightGBM (Tuned)"
    assert champion.metrics["Average Precision"] == 0.27


def test_champion_passing_gates_is_promoted(results) -> None:
    champion = select_champion(results, gates={"Average Precision": 0.10, "ROC-AUC": 0.60})
    assert champion.promoted is True and champion.gate_failures == ()


def test_champion_failing_gates_is_not_promoted(results) -> None:
    champion = select_champion(results, gates={"Average Precision": 0.90})
    assert champion.name == "LightGBM (Tuned)"
    assert champion.promoted is False
    assert any("Average Precision" in f for f in champion.gate_failures)


def test_champion_exposes_the_leaderboard(results) -> None:
    champion = select_champion(results)
    assert isinstance(champion.leaderboard, pd.DataFrame) and len(champion.leaderboard) == 4


def test_selection_does_not_mutate_input_results(results) -> None:
    snapshot = copy.deepcopy(results)
    select_champion(results)
    assert results == snapshot


def test_leaderboard_does_not_mutate_input_results(results) -> None:
    snapshot = copy.deepcopy(results)
    build_leaderboard(results)
    assert results == snapshot


def test_selection_works_with_baseline_only(results) -> None:
    assert select_champion([results[0]]).name == "Logistic Regression"


def test_recall_does_not_influence_champion_selection() -> None:
    """§9: the AP leader wins even with the *worst* recall in the field.

    Recall before threshold tuning is recall at 0.5, a cut-off the pipeline will never
    ship. Letting it sway selection is precisely the ordering bug this step removes.
    """
    champion = select_champion([
        _metrics("weak-AP high-recall", ap=0.11, roc_auc=0.80, recall=0.95),
        _metrics("strong-AP low-recall", ap=0.29, roc_auc=0.79, recall=0.03),
    ])
    assert champion.name == "strong-AP low-recall"


# ------------------------------------------- §9/§10/§12: the real end-to-end ordering

def test_threshold_dependent_checks_run_only_after_threshold_tuning(tmp_path) -> None:
    """Walk the actual Step 4 sequence on synthetic data, with the real modules.

    The unit tests above pin each stage in isolation; this one pins the *order* they
    compose in — champion chosen on threshold-independent metrics, threshold then tuned
    on validation alone, and only then Recall/Precision/F1 judged at that threshold.
    """
    import joblib
    from sklearn.pipeline import Pipeline

    from src.models.evaluation import evaluate_at_threshold, evaluate_model
    from src.models.serialization import CHAMPION_PIPELINE_FILENAME, save_champion_pipeline
    from src.models.threshold import find_f1_optimal_threshold
    from src.models.train import build_baseline_pipeline
    from src.models.tune import MODEL_STEP, PREPROCESSOR_STEP

    import numpy as np

    rng = np.random.default_rng(0)

    def make(n):
        signal = rng.normal(size=n)
        X = pd.DataFrame({"num_a": signal,
                          "num_b": rng.normal(size=n),
                          "cat_a": rng.choice(["x", "y"], size=n)})
        X.loc[X.sample(frac=0.1, random_state=1).index, "num_b"] = np.nan
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(2 * signal - 1)))).astype(int))
        return X, y

    X_train, y_train = make(400)
    X_val, y_val = make(200)
    X_test, y_test = make(200)

    # TRAIN — raw frames straight into the pipeline; no preprocessing fitted outside it.
    model = build_baseline_pipeline(X_train, random_state=42)
    model.fit(X_train, y_train)

    # VALIDATION — champion chosen on threshold-independent metrics only.
    champion = select_champion([evaluate_model(model, X_val, y_val, "Logistic Regression")])
    assert set(DEFAULT_QUALITY_GATES) & THRESHOLD_DEPENDENT_METRICS == set()

    # THRESHOLD — tuned on validation probabilities, never on test.
    val_probs = model.predict_proba(X_val)[:, 1]
    frozen = float(find_f1_optimal_threshold(y_val, val_probs)["threshold"])

    # POST-THRESHOLD — only now are Recall/Precision/F1 meaningful.
    at_threshold = evaluate_at_threshold(y_val, val_probs, frozen, champion.name)
    assert at_threshold["Threshold"] == pytest.approx(frozen)
    check_operational_gates(at_threshold)

    # The tuned threshold is a real choice, not the 0.5 default carried through.
    default_metrics = evaluate_at_threshold(y_val, val_probs, 0.5, champion.name)
    assert at_threshold["F1"] >= default_metrics["F1"]

    # TEST — read once, with model and threshold already frozen.
    test_metrics = evaluate_at_threshold(
        y_test, model.predict_proba(X_test)[:, 1], frozen, champion.name)
    assert test_metrics["Threshold"] == pytest.approx(frozen)
    assert PRIMARY_METRIC in test_metrics

    # SERIALIZE — the artifact carries preprocessing with it and scores raw frames.
    save_champion_pipeline(model, {"model_name": champion.name, "optimal_threshold": frozen},
                           tmp_path / "models", tmp_path / "artifacts")
    reloaded = joblib.load(tmp_path / "models" / CHAMPION_PIPELINE_FILENAME)
    assert isinstance(reloaded, Pipeline)
    assert list(reloaded.named_steps) == [PREPROCESSOR_STEP, MODEL_STEP]
    np.testing.assert_allclose(reloaded.predict_proba(X_test)[:, 1],
                               model.predict_proba(X_test)[:, 1])
