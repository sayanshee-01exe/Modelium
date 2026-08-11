"""Step 4 — champion selection and quality gates.

Selection reads validation metrics only. The gate mechanism exists so a model that
trains without error but predicts poorly cannot be promoted silently.
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
    DEFAULT_QUALITY_GATES,
    PRIMARY_METRIC,
    build_leaderboard,
    check_quality_gates,
    select_champion,
)


def _metrics(name, pr_auc, roc_auc=0.78, recall=0.62, precision=0.20, f1=0.30, accuracy=0.70):
    return {"Model": name, "PR-AUC": pr_auc, "ROC-AUC": roc_auc, "Recall": recall,
            "Precision": precision, "F1": f1, "Accuracy": accuracy}


@pytest.fixture
def results():
    """Deliberately not in PR-AUC order, and the ROC-AUC leader is NOT the PR-AUC leader."""
    return [
        _metrics("Logistic Regression", pr_auc=0.18, roc_auc=0.82),
        _metrics("LightGBM (Tuned)", pr_auc=0.27, roc_auc=0.79),
        _metrics("Random Forest (Tuned)", pr_auc=0.21, roc_auc=0.76),
        _metrics("XGBoost (Tuned)", pr_auc=0.25, roc_auc=0.80),
    ]


# --------------------------------------------------------------------- leaderboard

def test_primary_metric_is_pr_auc() -> None:
    assert PRIMARY_METRIC == "PR-AUC"


def test_leaderboard_ranks_by_pr_auc(results) -> None:
    board = build_leaderboard(results)
    assert list(board["Model"]) == [
        "LightGBM (Tuned)", "XGBoost (Tuned)", "Random Forest (Tuned)", "Logistic Regression",
    ]


def test_leaderboard_does_not_rank_by_roc_auc(results) -> None:
    """LogReg has the best ROC-AUC (0.82) but the worst PR-AUC — ranking must not follow it."""
    board = build_leaderboard(results)
    assert board.iloc[0]["Model"] != "Logistic Regression"


def test_baseline_appears_in_leaderboard(results) -> None:
    board = build_leaderboard(results)
    assert "Logistic Regression" in list(board["Model"])
    assert len(board) == 4


def test_leaderboard_keeps_secondary_metrics(results) -> None:
    board = build_leaderboard(results)
    for column in ("Recall", "ROC-AUC", "F1", "Precision"):
        assert column in board.columns


def test_leaderboard_rejects_empty_results() -> None:
    with pytest.raises(ValueError):
        build_leaderboard([])


def test_leaderboard_rejects_missing_primary_metric() -> None:
    with pytest.raises(ValueError, match="PR-AUC"):
        build_leaderboard([{"Model": "X", "ROC-AUC": 0.8}])


# -------------------------------------------------------------------- quality gates

def test_quality_gate_passes_valid_metrics() -> None:
    passed, failures = check_quality_gates(
        _metrics("good", pr_auc=0.30, roc_auc=0.80, recall=0.65)
    )
    assert passed and failures == []


def test_quality_gate_rejects_failing_metrics() -> None:
    passed, failures = check_quality_gates(
        _metrics("bad", pr_auc=0.02, roc_auc=0.51, recall=0.05)
    )
    assert not passed
    assert len(failures) == 3
    assert any("PR-AUC" in f for f in failures)


def test_quality_gate_reports_which_metric_failed() -> None:
    passed, failures = check_quality_gates(
        _metrics("partial", pr_auc=0.30, roc_auc=0.80, recall=0.01)
    )
    assert not passed
    assert len(failures) == 1 and "Recall" in failures[0]


def test_quality_gates_are_configurable() -> None:
    metrics = _metrics("m", pr_auc=0.30, roc_auc=0.80, recall=0.65)
    assert check_quality_gates(metrics, gates={"PR-AUC": 0.99})[0] is False
    assert check_quality_gates(metrics, gates={"PR-AUC": 0.10})[0] is True


def test_default_gates_are_conservative() -> None:
    """Floors for 'is this model broken', not performance targets. A ~8% positive rate
    means a random classifier scores PR-AUC ~0.08 and ROC-AUC 0.50."""
    assert DEFAULT_QUALITY_GATES["PR-AUC"] > 0.08
    assert DEFAULT_QUALITY_GATES["ROC-AUC"] > 0.50
    assert DEFAULT_QUALITY_GATES["PR-AUC"] < 0.30
    assert DEFAULT_QUALITY_GATES["ROC-AUC"] < 0.7692   # below the known-achievable baseline


def test_missing_metric_is_treated_as_a_gate_failure() -> None:
    passed, failures = check_quality_gates({"Model": "m", "PR-AUC": 0.3})
    assert not passed
    assert any("ROC-AUC" in f for f in failures)


# ----------------------------------------------------------------------- champion

def test_champion_is_the_pr_auc_leader(results) -> None:
    champion = select_champion(results)
    assert champion.name == "LightGBM (Tuned)"
    assert champion.metrics["PR-AUC"] == 0.27


def test_champion_passing_gates_is_promoted(results) -> None:
    champion = select_champion(results, gates={"PR-AUC": 0.10, "ROC-AUC": 0.60, "Recall": 0.20})
    assert champion.promoted is True
    assert champion.gate_failures == ()


def test_champion_failing_gates_is_not_promoted(results) -> None:
    """A model can top the leaderboard and still be too weak to ship."""
    champion = select_champion(results, gates={"PR-AUC": 0.90})
    assert champion.name == "LightGBM (Tuned)"
    assert champion.promoted is False
    assert any("PR-AUC" in f for f in champion.gate_failures)


def test_champion_exposes_the_leaderboard(results) -> None:
    champion = select_champion(results)
    assert isinstance(champion.leaderboard, pd.DataFrame)
    assert len(champion.leaderboard) == 4


def test_selection_does_not_mutate_input_results(results) -> None:
    snapshot = copy.deepcopy(results)
    select_champion(results)
    assert results == snapshot


def test_leaderboard_does_not_mutate_input_results(results) -> None:
    snapshot = copy.deepcopy(results)
    build_leaderboard(results)
    assert results == snapshot


def test_selection_works_with_baseline_only(results) -> None:
    champion = select_champion([results[0]])
    assert champion.name == "Logistic Regression"
