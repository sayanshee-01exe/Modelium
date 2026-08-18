"""DVC stage 7 — offline batch monitoring for the registered champion.

Orchestration only: this file decides what runs in what order and where results land.
Every measurement lives in `src/monitoring/`.

**Offline batch monitoring**, not live monitoring. It compares one stored current batch
against one stored reference batch at the moment the stage is run. There is no traffic,
no stream and no continuous evaluation.

The model is the registry's approved champion, resolved through
`models:/<name>@<champion_alias>` — the same artifact inference and SHAP use. Monitoring
calls `predict_proba` and `transform` only; it never fits, retrains, registers a version,
moves an alias, promotes or rolls back. It reports, and stops there.

Labels are optional. Without them the stage still measures feature drift, prediction
drift and prediction-only fairness, and records `labels_available: false` rather than
inventing performance numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ARTIFACT_DIR, ID_COL, REPORT_DIR, TARGET_COL, TRAIN_FEATURES_FILE
from src.data.data_preparation import split_train_val_test
from src.explainability.shap_explainer import (
    load_champion_from_registry, verify_champion_pipeline, resolve_positive_class_index,
)
from src.monitoring.data_drift import feature_drift_report, summarise_drift
from src.monitoring.fairness_monitor import fairness_report
from src.monitoring.performance_monitor import normalise_status, performance_report
from src.monitoring.prediction_drift import prediction_drift_report
from src.monitoring.reporting import (
    build_report, build_summary, plot_group_comparison, plot_missing_rate_change,
    plot_prediction_distribution, plot_top_feature_drift, write_json, write_text,
)
from src.utils.config_loader import load_params
from src.utils.exceptions import ConfigurationError, DataValidationError
from src.utils.logger import get_logger

logger = get_logger("modelium.monitor")

# Declared here, not in config/config.py: that module is a train dependency, and adding
# a constant to it would invalidate a multi-hour training run to move a path only this
# stage uses.
MONITORING_DIR = ARTIFACT_DIR / "monitoring"
FEATURE_DRIFT_FILE = MONITORING_DIR / "feature_drift.csv"
PREDICTION_DRIFT_FILE = MONITORING_DIR / "prediction_drift.json"
PERFORMANCE_FILE = MONITORING_DIR / "performance_metrics.json"
FAIRNESS_FILE = MONITORING_DIR / "fairness_metrics.csv"
SUMMARY_FILE = MONITORING_DIR / "monitoring_summary.json"
REPORT_FILE = MONITORING_DIR / "monitoring_report.md"
FIGURES_DIR = REPORT_DIR / "figures" / "monitoring"

CURRENT_BATCH_FILE = PROJECT_ROOT / "data" / "monitoring" / "current_batch.parquet"
BATCH_METADATA_FILE = PROJECT_ROOT / "data" / "monitoring" / "current_batch_metadata.json"
DEPLOYMENT_META_FILE = ARTIFACT_DIR / "deployment_meta.json"

REQUIRED_SETTINGS = ("enabled", "reference_sample_size", "current_sample_size",
                     "random_state", "drift", "prediction", "performance", "fairness")


def validate_settings(settings) -> dict:
    """Check the `monitoring:` section before any model or data is loaded."""
    if not isinstance(settings, dict):
        raise ConfigurationError(
            f"params.yaml: 'monitoring' must be a mapping, got {type(settings).__name__}")
    missing = [k for k in REQUIRED_SETTINGS if k not in settings]
    if missing:
        raise ConfigurationError(f"params.yaml: 'monitoring' is missing {missing}")
    if not isinstance(settings["enabled"], bool):
        raise ConfigurationError(
            f"params.yaml: 'monitoring.enabled' must be a boolean, got "
            f"{settings['enabled']!r}; the string \"false\" is truthy")
    for key in ("reference_sample_size", "current_sample_size"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(
                f"params.yaml: 'monitoring.{key}' must be a positive integer, "
                f"got {value!r}")
    return settings


def feature_types(preprocessor) -> tuple[list[str], list[str]]:
    """Numeric and categorical column names, as the *champion* sees them.

    Read from the fitted preprocessor rather than inferred from dtypes, so monitoring
    types every column exactly the way the model does. Inferring separately is how the
    two drift apart.
    """
    numeric: list[str] = []
    categorical: list[str] = []
    for name, _transformer, columns in preprocessor.transformers_:
        if not hasattr(columns, "__len__") or isinstance(columns, str):
            continue
        (numeric if name == "num" else categorical).append(list(columns))
    flat = lambda groups: [c for group in groups for c in group]      # noqa: E731
    return flat(numeric), flat(categorical)


def load_reference(settings, data_params) -> pd.DataFrame:
    """A deterministic sample of the split the champion was fitted on."""
    frame = pd.read_parquet(TRAIN_FEATURES_FILE)
    features = frame.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    target = frame[TARGET_COL].astype(int)
    X_train, _, _, _, _, _ = split_train_val_test(
        features, target,
        validation_size=float(data_params["validation_size"]),
        test_size=float(data_params["test_size"]),
        random_state=int(data_params["random_state"]),
    )
    size = min(int(settings["reference_sample_size"]), len(X_train))
    sample = X_train.sample(n=size, random_state=int(settings["random_state"])).sort_index()
    logger.info("Reference: %d rows sampled from the %d-row training split",
                len(sample), len(X_train))
    return sample


def load_current(settings) -> tuple[pd.DataFrame, pd.Series | None, dict]:
    """The stored current batch, its labels if present, and its metadata."""
    if not CURRENT_BATCH_FILE.exists():
        raise DataValidationError(
            f"No monitoring batch at {CURRENT_BATCH_FILE}. Build one with "
            f"`python scripts/create_monitoring_batch.py` before running this stage."
        )
    batch = pd.read_parquet(CURRENT_BATCH_FILE)
    metadata = {}
    if BATCH_METADATA_FILE.exists():
        metadata = json.loads(BATCH_METADATA_FILE.read_text(encoding="utf-8"))

    size = min(int(settings["current_sample_size"]), len(batch))
    batch = batch.sample(n=size, random_state=int(settings["random_state"])).sort_index()

    labels = None
    if TARGET_COL in batch.columns:
        labels = batch[TARGET_COL].astype(int)
        logger.info("Current batch carries labels (%s)",
                    metadata.get("label_source", "source unrecorded"))
    else:
        logger.info("Current batch carries no labels; running in no-label mode.")

    features = batch.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    logger.info("Current: %d rows", len(features))
    return features, labels, metadata


def main() -> int:
    params = load_params()
    settings = validate_settings(params.get("monitoring"))
    if not settings["enabled"]:
        logger.info("monitoring.enabled is false; nothing to do.")
        return 0

    registry = params["mlflow"]

    # ---------------------------------------------------------------- champion
    logger.info("Loading the registered champion...")
    pipeline, model_info = load_champion_from_registry(
        registry["registered_model_name"], registry["champion_alias"],
        tracking_uri=registry["tracking_uri"],
    )
    preprocessor, estimator = verify_champion_pipeline(pipeline)
    positive_index = resolve_positive_class_index(estimator)
    numeric_columns, categorical_columns = feature_types(preprocessor)
    logger.info("Champion %s: %d numeric / %d categorical features",
                type(estimator).__name__, len(numeric_columns), len(categorical_columns))

    if not DEPLOYMENT_META_FILE.exists():
        raise DataValidationError(f"No deployment metadata at {DEPLOYMENT_META_FILE}")
    deployment = json.loads(DEPLOYMENT_META_FILE.read_text(encoding="utf-8"))
    threshold = float(deployment["optimal_threshold"])
    baseline = deployment.get("test_metrics")

    # -------------------------------------------------------------- the batches
    reference = load_reference(settings, params["data"])
    current, labels, batch_metadata = load_current(settings)
    if reference.equals(current):
        raise DataValidationError(
            "The reference and current batches are identical. Monitoring a dataset "
            "against itself reports no drift by construction."
        )

    # ------------------------------------------------------- scoring (no fitting)
    logger.info("Scoring both batches with the champion (predict_proba only)...")
    reference_probabilities = pipeline.predict_proba(reference)[:, positive_index]
    current_probabilities = pipeline.predict_proba(current)[:, positive_index]

    # ------------------------------------------------------------ feature drift
    logger.info("Measuring feature drift over %d features...",
                len(numeric_columns) + len(categorical_columns))
    drift_frame = feature_drift_report(
        reference, current, numeric_columns, categorical_columns, settings["drift"])
    drift_summary = summarise_drift(
        drift_frame, float(settings["drift"]["max_drifted_feature_ratio"]))
    logger.info("Feature drift: %s (%d critical, %d warning of %d)",
                drift_summary["status"], drift_summary["critical_features"],
                drift_summary["warning_features"], drift_summary["monitored_features"])

    # --------------------------------------------------------- prediction drift
    prediction_drift = prediction_drift_report(
        reference_probabilities, current_probabilities, threshold, settings["prediction"],
        psi_warning=float(settings["drift"]["psi_warning_threshold"]),
        psi_critical=float(settings["drift"]["psi_critical_threshold"]),
    )
    logger.info("Prediction drift: %s (positive rate %+.4f)",
                prediction_drift["status"], prediction_drift["positive_rate_change"])

    # -------------------------------------------------------------- performance
    performance = performance_report(
        labels, current_probabilities, threshold, settings["performance"], baseline)
    logger.info("Performance: %s (labels available: %s)",
                performance["status"], performance.get("labels_available"))

    # ----------------------------------------------------------------- fairness
    fairness = fairness_report(
        current, current_probabilities, threshold, settings["fairness"], y_true=labels)
    logger.info("Fairness: %s", fairness["status"])

    # ------------------------------------------------------------------ outputs
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    drift_frame.to_csv(FEATURE_DRIFT_FILE, index=False)
    logger.info("Wrote %s", FEATURE_DRIFT_FILE)
    write_json(prediction_drift, PREDICTION_DRIFT_FILE)
    write_json(performance, PERFORMANCE_FILE)

    fairness_table = fairness.get("table")
    if isinstance(fairness_table, pd.DataFrame) and not fairness_table.empty:
        fairness_table.to_csv(FAIRNESS_FILE, index=False)
    else:
        # The stage declares this as a DVC output, so it must exist either way; an empty
        # table with a reason is a truthful result, a missing file is a broken contract.
        pd.DataFrame([{"group_column": None, "status": fairness["status"],
                       "note": fairness.get("reason", "no measurable groups")}]
                     ).to_csv(FAIRNESS_FILE, index=False)
    logger.info("Wrote %s", FAIRNESS_FILE)

    plots = {
        "top_feature_drift": plot_top_feature_drift(
            drift_frame, FIGURES_DIR / "top_feature_drift.png"),
        "prediction_distribution": plot_prediction_distribution(
            prediction_drift, FIGURES_DIR / "prediction_distribution.png"),
        "missing_rate_change": plot_missing_rate_change(
            drift_frame, FIGURES_DIR / "missing_rate_change.png"),
        "group_recall_comparison": plot_group_comparison(
            fairness, FIGURES_DIR / "group_recall_comparison.png"),
    }
    relative = lambda p: None if p is None else str(Path(p).relative_to(PROJECT_ROOT))  # noqa: E731

    artifacts = {
        "feature_drift": str(FEATURE_DRIFT_FILE.relative_to(PROJECT_ROOT)),
        "prediction_drift": str(PREDICTION_DRIFT_FILE.relative_to(PROJECT_ROOT)),
        "performance_metrics": str(PERFORMANCE_FILE.relative_to(PROJECT_ROOT)),
        "fairness_metrics": str(FAIRNESS_FILE.relative_to(PROJECT_ROOT)),
        "monitoring_report": str(REPORT_FILE.relative_to(PROJECT_ROOT)),
        "plots": {name: relative(path) for name, path in plots.items()},
    }

    summary = build_summary(
        model_info=model_info,
        reference_name="held-out training split of data/processed/train_features.parquet",
        current_name=str(CURRENT_BATCH_FILE.relative_to(PROJECT_ROOT)),
        reference_rows=len(reference), current_rows=len(current),
        drift_summary=drift_summary,
        prediction_drift=prediction_drift,
        performance={**performance, "status": normalise_status(performance["status"])},
        fairness=fairness,
        artifacts=artifacts,
    )
    summary["current_batch_metadata"] = batch_metadata
    write_json(summary, SUMMARY_FILE)

    report = build_report(summary, drift_frame, drift_summary, prediction_drift,
                          performance, fairness, batch_metadata)
    write_text(report, REPORT_FILE)

    log_to_mlflow(summary, drift_summary, prediction_drift, performance, fairness,
                  registry, batch_metadata,
                  [FEATURE_DRIFT_FILE, PREDICTION_DRIFT_FILE, PERFORMANCE_FILE,
                   FAIRNESS_FILE, SUMMARY_FILE, REPORT_FILE]
                  + [p for p in plots.values() if p is not None])

    print(
        f"\nMonitoring complete — overall status: {summary['overall_status'].upper()}\n"
        f"  model      : {summary['registered_model_name']} v{summary['model_version']} "
        f"(@{summary['model_alias']})\n"
        f"  batches    : {summary['reference_rows']:,} reference vs "
        f"{summary['current_rows']:,} current"
        + (" [demonstration batch]" if batch_metadata.get("demonstration") else "") + "\n"
        f"  data drift : {summary['data_drift_status']} "
        f"({summary['critical_features']} critical, {summary['warning_features']} warning "
        f"of {summary['monitored_features']})\n"
        f"  pred drift : {summary['prediction_drift_status']} "
        f"(positive rate {prediction_drift['positive_rate_change']:+.4f})\n"
        f"  performance: {summary['performance_status']} "
        f"(labels {'available' if summary['labels_available'] else 'absent'})\n"
        f"  fairness   : {summary['fairness_status']}\n"
        f"  report     : {REPORT_FILE.relative_to(PROJECT_ROOT)}"
    )
    return 0


def log_to_mlflow(summary, drift_summary, prediction_drift, performance, fairness,
                  registry, batch_metadata, artifacts) -> None:
    """Record the monitoring run as its own MLflow run, tagged back to the champion.

    A separate run, never an append to the training run: monitoring must not touch the
    metrics of the run the registered version points at. Failure is warned about and
    never fatal — the artifacts on disk are the deliverable.
    """
    if not registry.get("enabled"):
        logger.info("MLflow tracking is disabled; monitoring artifacts stay on disk only.")
        return
    try:
        from datetime import datetime, timezone

        import mlflow

        mlflow.set_tracking_uri(registry["tracking_uri"])
        mlflow.set_experiment(registry["experiment_name"])
        stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        with mlflow.start_run(run_name=f"monitor-{stamp}"):
            mlflow.set_tags({
                "monitoring_run": "true",
                "stage": "monitor",
                "source_training_run_id": summary.get("source_run_id"),
                "registered_model_name": summary.get("registered_model_name"),
                "model_version": summary.get("model_version"),
                "model_alias": summary.get("model_alias"),
                "batch_name": summary.get("current_dataset"),
                "labels_available": str(summary.get("labels_available")).lower(),
                "overall_status": summary.get("overall_status"),
                "demonstration_batch": str(bool(batch_metadata.get("demonstration"))).lower(),
            })
            metrics = {
                "drifted_feature_ratio": drift_summary["drifted_feature_ratio"],
                "critical_feature_count": drift_summary["critical_features"],
                "warning_feature_count": drift_summary["warning_features"],
                "monitored_feature_count": drift_summary["monitored_features"],
                "positive_rate_change": prediction_drift["positive_rate_change"],
                "mean_probability_change": prediction_drift["mean_probability_change"],
            }
            if prediction_drift.get("probability_psi") is not None:
                metrics["prediction_probability_psi"] = prediction_drift["probability_psi"]
            if performance.get("labels_available"):
                for key, name in (("Average Precision", "current_average_precision"),
                                  ("ROC-AUC", "current_roc_auc"),
                                  ("Recall", "current_recall"),
                                  ("Precision", "current_precision"),
                                  ("F1", "current_f1")):
                    metrics[name] = performance["metrics"][key]
            for item in fairness.get("summaries", []):
                column = item["group_column"].lower()
                if item.get("demographic_parity_difference") is not None:
                    metrics[f"demographic_parity_{column}"] = \
                        item["demographic_parity_difference"]
                if item.get("equal_opportunity_difference") is not None:
                    metrics[f"equal_opportunity_{column}"] = \
                        item["equal_opportunity_difference"]
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()
                                if v is not None and np.isfinite(float(v))})

            for path in artifacts:
                mlflow.log_artifact(str(path), artifact_path="monitoring")
        logger.info("Logged the monitoring run to MLflow.")
    except Exception as err:
        logger.warning("Could not log monitoring to MLflow: %s. Artifacts on disk are "
                       "unaffected.", err)


if __name__ == "__main__":
    raise SystemExit(main())
