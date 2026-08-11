"""Champion selection from validation metrics, with promotion quality gates.

Everything here reads **validation** results. Test metrics must not reach this module:
choosing a model by test performance is the leak Step 1 removed, and re-selecting after
seeing the test score would reintroduce it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

PRIMARY_METRIC = "PR-AUC"
SECONDARY_METRICS = ("Recall", "ROC-AUC", "F1", "Precision")

# Promotion floors, not performance targets. Justification for each, given a ~8%
# default rate on Home Credit:
#   PR-AUC  0.15 — a random classifier scores ~0.08 (the positive rate), so this is
#                  roughly 2x random. Published solutions reach ~0.25-0.30.
#   ROC-AUC 0.65 — random is 0.50; the prior single-table champion on this dataset
#                  reached 0.7692, so 0.65 sits clearly below known-achievable while
#                  still rejecting a model that has learned nothing.
#   Recall  0.40 — at the F1-tuned threshold. Missing more than 60% of defaulters
#                  makes the model commercially pointless regardless of its AUC.
# These are conservative on purpose: they are meant to catch a broken pipeline, not to
# encode a business risk appetite, which nobody has specified. Override per call.
DEFAULT_QUALITY_GATES: dict[str, float] = {
    "PR-AUC": 0.15,
    "ROC-AUC": 0.65,
    "Recall": 0.40,
}


@dataclass(frozen=True)
class ChampionSelection:
    """Outcome of ranking candidates and gating the leader."""

    name: str
    metrics: dict
    promoted: bool
    gate_failures: tuple[str, ...] = ()
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)


def build_leaderboard(results: list[dict], primary_metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Rank candidate metric dicts by the primary metric, best first.

    Args:
        results: One dict per candidate, each with "Model" and the metric keys.
        primary_metric: Sort key. PR-AUC by default.

    Returns:
        A new DataFrame sorted descending by `primary_metric`. The input is not modified.

    Raises:
        ValueError: if `results` is empty or any row lacks the primary metric.
    """
    if not results:
        raise ValueError("Cannot build a leaderboard from an empty result set")

    missing = [r.get("Model", "<unnamed>") for r in results if primary_metric not in r]
    if missing:
        raise ValueError(
            f"Result(s) missing the primary metric '{primary_metric}': {missing}"
        )

    board = pd.DataFrame(copy.deepcopy(results))
    return board.sort_values(primary_metric, ascending=False).reset_index(drop=True)


def check_quality_gates(
    metrics: dict, gates: dict[str, float] | None = None
) -> tuple[bool, list[str]]:
    """Check a candidate's metrics against minimum promotion thresholds.

    A metric absent from `metrics` counts as a failure rather than a pass — a gate that
    silently skips what it cannot find is not a gate.

    Args:
        metrics: Validation metrics for one model.
        gates: Metric -> minimum value. `DEFAULT_QUALITY_GATES` if omitted.

    Returns:
        ``(passed, failures)`` where each failure is a human-readable explanation.
    """
    active = DEFAULT_QUALITY_GATES if gates is None else gates
    failures: list[str] = []

    for metric, minimum in active.items():
        value = metrics.get(metric)
        if value is None:
            failures.append(f"{metric} is missing from the metrics (gate requires >= {minimum})")
        elif float(value) < minimum:
            failures.append(f"{metric}={float(value):.4f} is below the required {minimum}")

    return (not failures), failures


def select_champion(
    results: list[dict],
    *,
    gates: dict[str, float] | None = None,
    primary_metric: str = PRIMARY_METRIC,
) -> ChampionSelection:
    """Rank candidates on validation metrics and gate the leader for promotion.

    The leader is always reported, even when it fails its gates — knowing the best
    available model is still 0.03 PR-AUC short is more useful than an empty result.
    `promoted` is what callers should branch on.

    Args:
        results: Validation metrics, one dict per candidate.
        gates: Promotion thresholds; `DEFAULT_QUALITY_GATES` if omitted.
        primary_metric: Ranking metric.

    Returns:
        A `ChampionSelection`. The input list is not modified.
    """
    leaderboard = build_leaderboard(results, primary_metric=primary_metric)
    top = leaderboard.iloc[0].to_dict()
    name = str(top["Model"])

    passed, failures = check_quality_gates(top, gates)
    if passed:
        logger.info(
            "Champion: %s (%s=%.4f) — passed all quality gates",
            name, primary_metric, float(top[primary_metric]),
        )
    else:
        logger.warning(
            "Champion candidate %s (%s=%.4f) FAILED %d quality gate(s): %s",
            name, primary_metric, float(top[primary_metric]), len(failures), "; ".join(failures),
        )

    return ChampionSelection(
        name=name,
        metrics=top,
        promoted=passed,
        gate_failures=tuple(failures),
        leaderboard=leaderboard,
    )
