"""Aggregate the monitoring sections into a status, a written report, and plots.

The aggregation rule is deliberately blunt: **the overall status is the worst section
status, and every section status is carried alongside it.** A single green headline that
absorbs a red section is worse than no headline, because it is the one line an operator
will read. `overall_status` says how bad things are; the section fields say where.

The report is generated from the computed results only. There is no template prose that
would render identically whatever the numbers turned out to be.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.monitoring.data_drift import CRITICAL, ERROR, INSUFFICIENT, STABLE, WARNING
from src.monitoring.performance_monitor import HEALTHY, NOT_AVAILABLE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ranked worst-first. `not_available` and `insufficient_data` sit below `warning`: they
# mean "not measured", which is not the same as "measured and fine", but is also not
# evidence of a problem.
STATUS_RANK = {
    CRITICAL: 4, WARNING: 3, INSUFFICIENT: 2, ERROR: 2, NOT_AVAILABLE: 1,
    STABLE: 0, HEALTHY: 0,
}


def overall_status(section_statuses: dict[str, str]) -> str:
    """The worst section status. Never better than its worst input."""
    if not section_statuses:
        return NOT_AVAILABLE
    worst = max(section_statuses.values(), key=lambda s: STATUS_RANK.get(s, 0))
    return CRITICAL if STATUS_RANK.get(worst, 0) == 4 else worst


def build_summary(
    *,
    model_info: dict[str, Any],
    reference_name: str,
    current_name: str,
    reference_rows: int,
    current_rows: int,
    drift_summary: dict[str, Any],
    prediction_drift: dict[str, Any],
    performance: dict[str, Any],
    fairness: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """The machine-readable monitoring status."""
    sections = {
        "data_drift": drift_summary["status"],
        "prediction_drift": prediction_drift["status"],
        "performance": performance["status"],
        "fairness": fairness["status"],
    }
    return {
        "registered_model_name": model_info.get("registered_model_name"),
        "model_version": model_info.get("model_version"),
        "model_alias": model_info.get("model_alias"),
        "source_run_id": model_info.get("source_run_id"),
        "monitoring_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference_dataset": reference_name,
        "current_dataset": current_name,
        "reference_rows": int(reference_rows),
        "current_rows": int(current_rows),
        "monitored_features": drift_summary["monitored_features"],
        "warning_features": drift_summary["warning_features"],
        "critical_features": drift_summary["critical_features"],
        "drifted_feature_ratio": drift_summary["drifted_feature_ratio"],
        "data_drift_status": sections["data_drift"],
        "prediction_drift_status": sections["prediction_drift"],
        "performance_status": sections["performance"],
        "fairness_status": sections["fairness"],
        "labels_available": bool(performance.get("labels_available", False)),
        "overall_status": overall_status(sections),
        "section_statuses": sections,
        "artifacts": artifacts,
        "action_taken": (
            "none — this stage reports only. It never retrains, registers a version, "
            "moves an alias, promotes or rolls back."
        ),
    }


def _fmt(value: Any, spec: str = ".4f") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(number) else format(number, spec)


def build_report(
    summary: dict[str, Any],
    drift_frame: pd.DataFrame,
    drift_summary: dict[str, Any],
    prediction_drift: dict[str, Any],
    performance: dict[str, Any],
    fairness: dict[str, Any],
    batch_metadata: dict[str, Any] | None = None,
) -> str:
    """Render the markdown report from computed results only."""
    lines: list[str] = []
    add = lines.append

    add("# Monitoring report")
    add("")
    add(f"**Overall status: `{summary['overall_status']}`** — "
        f"generated {summary['monitoring_timestamp']}")
    add("")
    add("This is **offline batch monitoring**. It compares a stored current batch "
        "against a stored reference batch when the stage is run. There is no live "
        "traffic, no streaming, and no continuous evaluation.")
    add("")

    add("## Model")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Registered model | `{summary['registered_model_name']}` |")
    add(f"| Version / alias | {summary['model_version']} / `{summary['model_alias']}` |")
    add(f"| Source training run | `{summary['source_run_id']}` |")
    add(f"| Decision threshold | {_fmt(prediction_drift.get('threshold'))} "
        f"({prediction_drift.get('threshold_source')}) |")
    add("")

    add("## Batches")
    add("")
    add("| | Reference | Current |")
    add("| --- | --- | --- |")
    add(f"| Dataset | `{summary['reference_dataset']}` | `{summary['current_dataset']}` |")
    add(f"| Rows | {summary['reference_rows']:,} | {summary['current_rows']:,} |")
    add(f"| Labels | yes | {'yes' if summary['labels_available'] else 'no'} |")
    if batch_metadata:
        add("")
        if batch_metadata.get("demonstration"):
            add(f"> **Demonstration batch.** {batch_metadata.get('description', '')}")
            drifted = batch_metadata.get("simulated_drift_features") or []
            if drifted:
                add(f"> Controlled drift was injected into {len(drifted)} feature(s): "
                    f"{', '.join(drifted)}. This is simulated, not observed production "
                    f"behaviour.")
    add("")

    add("## Feature drift")
    add("")
    add(f"Status: `{drift_summary['status']}`. "
        f"{drift_summary['warning_features']} warning and "
        f"{drift_summary['critical_features']} critical of "
        f"{drift_summary['monitored_features']} features "
        f"({drift_summary['drifted_feature_ratio']:.1%} drifted; limit "
        f"{drift_summary['max_drifted_feature_ratio']:.0%}).")
    add("")
    top = [r for r in drift_summary["top_drifted"]
           if r.get("drift_status") in (WARNING, CRITICAL)][:10]
    if top:
        add("Most drifted features:")
        add("")
        add("| Feature | Type | PSI | Status | Reason |")
        add("| --- | --- | --- | --- | --- |")
        for row in top:
            add(f"| `{row['feature']}` | {row['feature_type']} | "
                f"{_fmt(row.get('psi'), '.3f')} | {row['drift_status']} | "
                f"{row.get('drift_reason', '')} |")
    else:
        add("No feature exceeded its drift thresholds.")
    add("")

    add("## Prediction drift")
    add("")
    add(f"Status: `{prediction_drift['status']}`.")
    add("")
    add("| Measure | Reference | Current | Change |")
    add("| --- | --- | --- | --- |")
    ref, cur = prediction_drift["reference"], prediction_drift["current"]
    add(f"| Mean probability | {_fmt(ref['mean_probability'])} | "
        f"{_fmt(cur['mean_probability'])} | "
        f"{_fmt(prediction_drift['mean_probability_change'], '+.4f')} |")
    add(f"| Median probability | {_fmt(ref['median_probability'])} | "
        f"{_fmt(cur['median_probability'])} | "
        f"{_fmt(prediction_drift['median_probability_change'], '+.4f')} |")
    add(f"| Std probability | {_fmt(ref['std_probability'])} | "
        f"{_fmt(cur['std_probability'])} | "
        f"{_fmt(prediction_drift['std_probability_change'], '+.4f')} |")
    add(f"| Positive rate | {_fmt(ref['positive_rate'])} | {_fmt(cur['positive_rate'])} | "
        f"{_fmt(prediction_drift['positive_rate_change'], '+.4f')} |")
    add("")
    add(f"Probability PSI {_fmt(prediction_drift.get('probability_psi'), '.4f')}; "
        f"distribution distance "
        f"{_fmt(prediction_drift.get('probability_distribution_distance'), '.4f')}.")
    for reason in prediction_drift.get("reasons", []):
        add(f"- {reason}")
    add("")

    add("## Performance")
    add("")
    if not performance.get("labels_available"):
        add(f"Status: `{performance['status']}` — {performance.get('reason', '')}")
    else:
        metrics = performance["metrics"]
        add(f"Status: `{performance['status']}` on {metrics['n']:,} labelled rows.")
        add("")
        add("| Metric | Current | Champion test baseline | Change |")
        add("| --- | --- | --- | --- |")
        comparison = performance.get("baseline_comparison", {})
        changes = comparison.get("metrics", {}) if comparison.get("available") else {}
        for key in ("Average Precision", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1"):
            entry = changes.get(key)
            if entry:
                add(f"| {key} | {_fmt(entry['current'])} | {_fmt(entry['baseline'])} | "
                    f"{_fmt(entry['change'], '+.4f')} |")
            else:
                add(f"| {key} | {_fmt(metrics.get(key))} | n/a | n/a |")
        matrix = metrics["confusion_matrix"]
        add("")
        add(f"Confusion matrix at threshold {_fmt(metrics['threshold'])}: "
            f"TN {matrix['TN']:,}, FP {matrix['FP']:,}, FN {matrix['FN']:,}, "
            f"TP {matrix['TP']:,}. "
            f"False-positive rate {_fmt(metrics['false_positive_rate'])}, "
            f"false-negative rate {_fmt(metrics['false_negative_rate'])}.")
        add("")
        for reason in performance.get("reasons", []):
            add(f"- {reason}")
    add("")

    add("## Fairness")
    add("")
    add(f"Status: `{fairness['status']}`.")
    add("")
    add(f"> {fairness.get('limitations', '')}")
    add("")
    if fairness.get("summaries"):
        add("| Grouping column | Groups measured | Below minimum size | "
            "Demographic parity diff | Equal opportunity diff |")
        add("| --- | --- | --- | --- | --- |")
        for item in fairness["summaries"]:
            add(f"| `{item['group_column']}` | {item['groups_measured']} | "
                f"{item['groups_insufficient']} | "
                f"{_fmt(item['demographic_parity_difference'], '.4f')} | "
                f"{_fmt(item['equal_opportunity_difference'], '.4f')} |")
        add("")
    for reason in fairness.get("reasons", []):
        add(f"- {reason}")
    add("")

    add("## Limitations")
    add("")
    add("- Offline batch monitoring only. Nothing here observes live traffic.")
    add("- The current batch is a stored file, so results describe that file and the "
        "moment it was built, not a rolling production window.")
    add("- Drift statistics detect *distributional change*. They cannot say whether a "
        "change harms the model; only labelled performance can.")
    if not summary["labels_available"]:
        add("- No labels in this batch, so no performance measurement was possible. In "
            "credit risk outcomes arrive months after scoring, so this is normal.")
    add("- Fairness figures are a technical demonstration on proxy columns and establish "
        "no legal compliance.")
    add("")

    add("## Recommended next actions")
    add("")
    for action in recommend_actions(summary, drift_summary, prediction_drift,
                                    performance, fairness):
        add(f"- {action}")
    add("")
    add("_This stage reports only. It does not retrain, register, promote, or roll back._")
    add("")
    return "\n".join(lines)


def recommend_actions(summary, drift_summary, prediction_drift, performance,
                      fairness) -> list[str]:
    """Concrete follow-ups implied by the computed statuses, not generic advice."""
    actions: list[str] = []

    if drift_summary["status"] == CRITICAL:
        actions.append(
            f"Investigate the {drift_summary['critical_features']} critical feature(s) "
            f"before trusting new scores — start with the table above, which is ordered "
            f"by severity then PSI.")
    elif drift_summary["status"] == WARNING:
        actions.append(
            f"Review the {drift_summary['warning_features']} warning feature(s); no "
            f"single feature crossed the critical PSI level.")

    if prediction_drift["status"] in (WARNING, CRITICAL):
        actions.append(
            f"The score distribution moved (positive rate "
            f"{prediction_drift['positive_rate_change']:+.4f}). Confirm downstream "
            f"approval volumes match what the frozen threshold is expected to produce.")

    if not performance.get("labels_available"):
        actions.append(
            "Re-run monitoring once outcomes mature for this cohort; drift alone cannot "
            "confirm whether accuracy has moved.")
    elif performance["status"] in (WARNING, CRITICAL):
        actions.append(
            "Performance is below its configured floor or materially below the champion "
            "baseline. Retraining is a human decision and this stage takes none.")

    if fairness["status"] in (WARNING, CRITICAL):
        actions.append(
            "Group disparities exceed the configured limits. Treat as a prompt for "
            "review, not as a compliance finding.")

    if not actions:
        actions.append("No action indicated — every section is within its thresholds.")
    return actions


def write_json(payload: Any, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote %s", path)
    return path


def write_text(text: str, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


# ---------------------------------------------------------------------------
# Plots — each answers a monitoring question that the tables answer less quickly
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_top_feature_drift(frame: pd.DataFrame, path, top_n: int = 20) -> Path | None:
    """The features carrying the most distributional change, by PSI."""
    plt = _plt()
    usable = frame[frame["psi"].notna()].head(top_n)
    if usable.empty:
        logger.info("No scorable PSI values; skipping the feature drift plot")
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = usable.iloc[::-1]
    colours = {CRITICAL: "#c0392b", WARNING: "#e08e0b"}
    plt.figure(figsize=(10, max(4, 0.34 * len(ordered))))
    plt.barh(ordered["feature"], ordered["psi"],
             color=[colours.get(s, "#3b7dd8") for s in ordered["drift_status"]])
    plt.xlabel("Population Stability Index")
    plt.title(f"Top {len(ordered)} features by PSI")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", path)
    return path


def plot_prediction_distribution(prediction_drift: dict[str, Any], path) -> Path:
    """Reference against current score distribution, with the frozen threshold marked."""
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.asarray(prediction_drift["histogram_bins"], dtype=float)
    centres = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4

    plt.figure(figsize=(9, 5))
    plt.bar(centres - width / 2, prediction_drift["reference_histogram"], width=width,
            label="reference", color="#3b7dd8")
    plt.bar(centres + width / 2, prediction_drift["current_histogram"], width=width,
            label="current", color="#e08e0b")
    plt.axvline(prediction_drift["threshold"], color="#c0392b", linestyle="--",
                label=f"frozen threshold {prediction_drift['threshold']:.4f}")
    plt.xlabel("predicted default probability")
    plt.ylabel("share of batch")
    plt.title("Predicted probability distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", path)
    return path


def plot_missing_rate_change(frame: pd.DataFrame, path, top_n: int = 20) -> Path | None:
    """Features whose missing rate moved most — often the first sign of a broken feed."""
    plt = _plt()
    if "missing_rate_change" not in frame.columns:
        return None
    usable = frame[frame["missing_rate_change"].notna()].copy()
    usable = usable[usable["missing_rate_change"].abs() > 0]
    if usable.empty:
        logger.info("No missing-rate movement; skipping that plot")
        return None
    usable = usable.reindex(
        usable["missing_rate_change"].abs().sort_values(ascending=False).index
    ).head(top_n).iloc[::-1]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, max(4, 0.34 * len(usable))))
    plt.barh(usable["feature"], usable["missing_rate_change"],
             color=["#c0392b" if v > 0 else "#2e86c1"
                    for v in usable["missing_rate_change"]])
    plt.axvline(0, color="#555", linewidth=0.8)
    plt.xlabel("current missing rate − reference missing rate")
    plt.title("Largest missing-rate changes")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", path)
    return path


def plot_group_comparison(fairness: dict[str, Any], path) -> Path | None:
    """Per-group rates, only where at least two groups cleared the minimum size."""
    table = fairness.get("table")
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return None
    measured = table[table["status"] == "measured"]
    if measured.empty:
        logger.info("No group cleared the minimum size; skipping the fairness plot")
        return None

    labelled = fairness.get("labels_available") and measured["recall"].notna().any()
    metric = "recall" if labelled else "positive_prediction_rate"
    usable = measured[measured[metric].notna()]
    if usable.empty or usable.groupby("group_column")[metric].count().max() < 2:
        logger.info("Fewer than two comparable groups; skipping the fairness plot")
        return None

    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r.group_column}={r.group_value}" for r in usable.itertuples()]
    plt.figure(figsize=(10, max(4, 0.34 * len(usable))))
    plt.barh(labels[::-1], usable[metric].tolist()[::-1], color="#3b7dd8")
    plt.xlabel(metric.replace("_", " "))
    plt.title(f"Per-group {metric.replace('_', ' ')} "
              f"({'labelled' if labelled else 'predictions only'})")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", path)
    return path
