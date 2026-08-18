"""Group-wise outcome comparison across applicant segments.

**Scope and limits, stated first because they bound everything below.** This is a
*technical demonstration* of disparity measurement on a public research dataset. It is
not a legal compliance assessment, and passing these checks is not evidence that the
model is fair.

Three specific reasons:

*The grouping columns are not protected attributes.* `CODE_GENDER`,
`NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE` and `NAME_INCOME_TYPE` are self-reported
application fields in a Kaggle dataset. They are proxies at best, and treating them as
established protected-class membership would overstate what the numbers support.

*Disparity is not discrimination.* A rate difference between groups can arise from
genuine differences in the underlying population, from sampling, or from the model.
These metrics cannot separate those causes, and nothing here attempts to.

*The measured metrics are not the whole of fairness.* Demographic parity and equal
opportunity are two definitions among many, and they are mutually incompatible in
general — satisfying both simultaneously is impossible except in degenerate cases.

Small groups are reported as `insufficient_data`, never merged into a larger bucket to
reach the threshold: combining unrelated segments to manufacture significance would
produce a number about a population that does not exist.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.monitoring.data_drift import CRITICAL, STABLE, WARNING
from src.utils.logger import get_logger

logger = get_logger(__name__)

INSUFFICIENT = "insufficient_data"
NOT_AVAILABLE = "not_available"

# Application fields used as grouping proxies. See the module docstring: these are not
# verified protected attributes.
DEFAULT_GROUP_COLUMNS = (
    "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
)

DISCLAIMER = (
    "Technical demonstration on a public research dataset. The grouping columns are "
    "self-reported application fields, not verified protected attributes. These metrics "
    "do not establish legal compliance, and a disparity is not by itself evidence of "
    "discrimination."
)


def group_metrics(
    values: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    y_true=None,
    minimum_group_size: int = 100,
) -> pd.DataFrame:
    """Per-group outcome rates for one grouping column.

    Label-dependent columns (recall, precision, false-negative rate, actual default
    rate) are only populated when `y_true` is supplied; otherwise they are NaN and the
    prediction-only columns still carry information.

    Neither the series nor the arrays are modified.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    labels = None if y_true is None else np.asarray(y_true).astype(int)

    records: list[dict[str, Any]] = []
    for group in sorted(values.dropna().astype(str).unique()):
        mask = (values.astype(str) == group).to_numpy()
        size = int(mask.sum())
        row: dict[str, Any] = {
            "group_value": group,
            "n": size,
            "positive_prediction_rate": float("nan"),
            "mean_probability": float("nan"),
            "actual_default_rate": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "false_positive_rate": float("nan"),
            "false_negative_rate": float("nan"),
            "status": INSUFFICIENT,
        }
        if size < minimum_group_size:
            row["note"] = (
                f"{size} rows is below the minimum of {minimum_group_size}; reported "
                f"rather than merged into another group"
            )
            records.append(row)
            continue

        row["positive_prediction_rate"] = float(predicted[mask].mean())
        row["mean_probability"] = float(probabilities[mask].mean())
        row["status"] = "measured"
        row["note"] = ""

        if labels is not None:
            from sklearn.metrics import confusion_matrix

            group_true, group_pred = labels[mask], predicted[mask]
            tn, fp, fn, tp = confusion_matrix(
                group_true, group_pred, labels=[0, 1]).ravel()
            row["actual_default_rate"] = float(group_true.mean())
            row["precision"] = float(tp / (tp + fp)) if (tp + fp) else 0.0
            row["recall"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
            row["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) else 0.0
            row["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) else float("nan")
        records.append(row)

    return pd.DataFrame.from_records(records)


def disparity(frame: pd.DataFrame, column: str) -> float:
    """Max-minus-min of a column across measured groups, or NaN if fewer than two."""
    measured = frame[frame["status"] == "measured"][column].dropna()
    if measured.size < 2:
        return float("nan")
    return float(measured.max() - measured.min())


def fairness_report(
    features: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    settings: dict[str, Any],
    y_true=None,
    group_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Disparity metrics for every configured grouping column present in the batch.

    Returns:
        A mapping with a per-column table, the disparity figures, an overall status and
        the limitations that qualify all of it.
    """
    if not settings.get("enabled", True):
        return {"status": NOT_AVAILABLE, "reason": "fairness monitoring is disabled",
                "limitations": DISCLAIMER}

    columns = list(group_columns or DEFAULT_GROUP_COLUMNS)
    present = [c for c in columns if c in features.columns]
    missing = [c for c in columns if c not in features.columns]
    if missing:
        logger.info("Grouping columns absent from the batch, skipped: %s", missing)
    if not present:
        return {"status": NOT_AVAILABLE,
                "reason": f"none of the configured grouping columns {columns} are present",
                "limitations": DISCLAIMER}

    minimum = int(settings.get("minimum_group_size", 100))
    parity_limit = float(settings.get("max_demographic_parity_difference", 0.10))
    opportunity_limit = float(settings.get("max_equal_opportunity_difference", 0.10))

    tables: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    reasons: list[str] = []
    status = STABLE

    for column in present:
        frame = group_metrics(features[column], probabilities, threshold,
                              y_true=y_true, minimum_group_size=minimum)
        frame.insert(0, "group_column", column)
        tables.append(frame)

        parity = disparity(frame, "positive_prediction_rate")
        # Equal opportunity compares true-positive rates, i.e. recall, so it needs labels.
        opportunity = disparity(frame, "recall")
        summary = {
            "group_column": column,
            "groups_measured": int((frame["status"] == "measured").sum()),
            "groups_insufficient": int((frame["status"] == INSUFFICIENT).sum()),
            "demographic_parity_difference": None if np.isnan(parity) else parity,
            "equal_opportunity_difference": None if np.isnan(opportunity) else opportunity,
            "recall_difference": None if np.isnan(opportunity) else opportunity,
        }

        if np.isfinite(parity) and parity > parity_limit:
            status = WARNING if status == STABLE else status
            reasons.append(
                f"{column}: positive-prediction-rate spread {parity:.3f} exceeds "
                f"{parity_limit}")
        if np.isfinite(opportunity) and opportunity > opportunity_limit:
            status = WARNING if status == STABLE else status
            reasons.append(
                f"{column}: recall spread {opportunity:.3f} exceeds {opportunity_limit}")
        summaries.append(summary)

    combined = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    measured_any = any(s["groups_measured"] >= 2 for s in summaries)
    if not measured_any:
        status = INSUFFICIENT
        reasons.append("no grouping column had two or more groups above the minimum size")

    return {
        "status": status,
        "labels_available": y_true is not None,
        "minimum_group_size": minimum,
        "group_columns_used": present,
        "group_columns_absent": missing,
        "summaries": summaries,
        "table": combined,
        "reasons": reasons or ["all measured disparities within configured limits"],
        "limitations": DISCLAIMER,
        "action_taken": "none — monitoring reports disparities and changes no model",
    }


def escalate(status: str, parity_values: Sequence[float], limit: float) -> str:
    """Raise a warning to critical when a disparity is more than double its limit."""
    finite = [v for v in parity_values if v is not None and np.isfinite(v)]
    if finite and max(finite) > 2 * limit:
        return CRITICAL
    return status
