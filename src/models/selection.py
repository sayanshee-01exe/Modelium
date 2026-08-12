"""Champion selection from validation metrics, with two-stage promotion gates.

Everything here reads **validation** results. Test metrics must not reach this module:
choosing a model by test performance is the leak Step 1 removed, and re-selecting after
seeing the test score would reintroduce it.

Ranking uses Average Precision — the same estimator `RandomizedSearchCV` optimises via
`scoring="average_precision"`. Ranking by a different PR-curve estimator would let the
model that won the search lose the selection.

Gates are split by when they become meaningful:

    pre-threshold   Average Precision, ROC-AUC   threshold-independent; safe to gate on
                                                 before a cut-off has been chosen
    post-threshold  Recall, Precision, F1        only meaningful once the threshold is
                                                 frozen, since before that they describe
                                                 the arbitrary 0.5 default
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from src.models.evaluation import PRIMARY_METRIC
from src.utils.config_loader import get_section
from src.utils.logger import get_logger

logger = get_logger(__name__)

SECONDARY_METRICS = ("ROC-AUC", "Recall", "Precision", "F1", "Accuracy")

# Metrics computed from hard predictions, and therefore a function of the decision
# threshold. Gating on these before threshold tuning would judge a model by the 0.5
# default it will never actually ship with.
THRESHOLD_DEPENDENT_METRICS = frozenset({"Recall", "Precision", "F1", "Accuracy"})

# Pre-threshold promotion floors — threshold-independent metrics only. Values come from
# params.yaml (`selection:`), which carries the justification: they are conservative
# floors that catch a broken pipeline, not a business risk appetite, which nobody has
# specified. Override per call.
_SELECTION = get_section("selection")
DEFAULT_QUALITY_GATES: dict[str, float] = {
    PRIMARY_METRIC: float(_SELECTION["min_average_precision"]),
    "ROC-AUC": float(_SELECTION["min_roc_auc"]),
}

# Post-threshold operational floor, applied to metrics recomputed at the frozen
# threshold. Missing more than 60% of defaulters makes the model commercially
# pointless regardless of how good its ranking metrics look.
DEFAULT_OPERATIONAL_GATES: dict[str, float] = {
    "Recall": float(get_section("threshold")["min_recall"]),
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
        primary_metric: Sort key. Average Precision by default.

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


def _apply_gates(metrics: dict, gates: dict[str, float]) -> tuple[bool, list[str]]:
    """Compare metrics against minimums; a missing metric counts as a failure.

    A gate that silently skips what it cannot find is not a gate.
    """
    failures: list[str] = []
    for metric, minimum in gates.items():
        value = metrics.get(metric)
        if value is None:
            failures.append(f"{metric} is missing from the metrics (gate requires >= {minimum})")
        elif float(value) < minimum:
            failures.append(f"{metric}={float(value):.4f} is below the required {minimum}")
    return (not failures), failures


def check_quality_gates(
    metrics: dict, gates: dict[str, float] | None = None
) -> tuple[bool, list[str]]:
    """Pre-threshold gates. Threshold-independent metrics only.

    Args:
        metrics: Validation metrics for one model.
        gates: Metric -> minimum. `DEFAULT_QUALITY_GATES` if omitted.

    Returns:
        ``(passed, failures)``.

    Raises:
        ValueError: if a threshold-dependent metric is used as a pre-threshold gate.
            Enforced in code rather than left to convention, because gating Recall at
            the 0.5 default while calling it "the model's recall" is exactly the
            mistake this split exists to prevent.
    """
    active = DEFAULT_QUALITY_GATES if gates is None else gates

    misplaced = set(active) & THRESHOLD_DEPENDENT_METRICS
    if misplaced:
        raise ValueError(
            f"{sorted(misplaced)} are threshold-dependent and cannot be used as "
            f"pre-threshold gates; pass them to check_operational_gates() after the "
            f"decision threshold has been tuned on validation"
        )

    return _apply_gates(metrics, active)


def check_operational_gates(
    metrics: dict, gates: dict[str, float] | None = None
) -> tuple[bool, list[str]]:
    """Post-threshold gates, applied to metrics recomputed at the frozen threshold.

    Args:
        metrics: Metrics from `evaluate_at_threshold`, not from the 0.5 default.
        gates: Metric -> minimum. `DEFAULT_OPERATIONAL_GATES` if omitted.

    Returns:
        ``(passed, failures)``.
    """
    return _apply_gates(metrics, DEFAULT_OPERATIONAL_GATES if gates is None else gates)


def select_champion(
    results: list[dict],
    *,
    gates: dict[str, float] | None = None,
    primary_metric: str = PRIMARY_METRIC,
) -> ChampionSelection:
    """Rank candidates on validation metrics and apply the pre-threshold gates.

    The leader is always reported, even when it fails its gates — knowing the best
    available model is still 0.03 Average Precision short is more useful than an empty
    result. `promoted` is what callers should branch on.

    Args:
        results: Validation metrics, one dict per candidate.
        gates: Pre-threshold thresholds; `DEFAULT_QUALITY_GATES` if omitted.
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
            "Champion: %s (%s=%.4f) — passed all pre-threshold quality gates",
            name, primary_metric, float(top[primary_metric]),
        )
    else:
        logger.warning(
            "Champion candidate %s (%s=%.4f) FAILED %d pre-threshold gate(s): %s",
            name, primary_metric, float(top[primary_metric]), len(failures), "; ".join(failures),
        )

    return ChampionSelection(
        name=name,
        metrics=top,
        promoted=passed,
        gate_failures=tuple(failures),
        leaderboard=leaderboard,
    )
