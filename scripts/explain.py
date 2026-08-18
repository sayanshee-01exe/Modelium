"""DVC stage 6 — SHAP explanations for the registered champion.

Orchestration only: this file decides *what happens in what order*, and every SHAP
detail lives in `src/explainability/shap_explainer.py`.

It explains the model the registry says is approved, resolved through
``models:/<name>@<champion_alias>`` rather than read from a local file, so the report
cannot describe a model nobody is serving. Nothing here fits, tunes or selects — the
stage is inference-only, and re-running it can never change the champion.

The sample comes from the **test** split, reconstructed with the same sizes, seed and
stratification the train stage used, so the explanations describe held-out behaviour
rather than rows the model was fitted on.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ARTIFACT_DIR, ID_COL, REPORT_DIR, TARGET_COL, TRAIN_FEATURES_FILE
from src.data.data_preparation import split_train_val_test
from src.explainability.shap_explainer import (  # noqa: E402
    build_local_explanation, choose_local_examples, compute_shap_values,
    global_feature_importance, load_champion_from_registry, resolve_positive_class_index,
    save_bar_plot, save_local_plot, save_summary_plot, select_explanation_sample,
    transform_for_explanation, validate_explainability_settings, validate_probabilities,
    verify_champion_pipeline, write_json,
)
from src.utils.config_loader import load_params
from src.utils.logger import get_logger

logger = get_logger("modelium.explain")

# Declared here rather than in config/config.py: that module is a dependency of the train
# stage, and adding a constant to it would invalidate a multi-hour training run to move a
# path that only this stage uses.
EXPLAINABILITY_DIR = ARTIFACT_DIR / "explainability"
GLOBAL_IMPORTANCE_FILE = EXPLAINABILITY_DIR / "global_feature_importance.csv"
LOCAL_EXPLANATIONS_FILE = EXPLAINABILITY_DIR / "local_explanations.json"
EXPLANATION_REPORT_FILE = EXPLAINABILITY_DIR / "explanation_report.json"
FIGURES_DIR = REPORT_DIR / "figures"
# Per-applicant plots live in their own DVC-owned directory; see dvc.yaml for why.
LOCAL_FIGURES_DIR = FIGURES_DIR / "shap_local"

SUMMARY_PLOT = "shap_summary.png"
BAR_PLOT = "shap_bar.png"


def main() -> int:
    params = load_params()
    settings = validate_explainability_settings(params.get("explainability"))
    if not settings["enabled"]:
        logger.info("explainability.enabled is false; nothing to do.")
        return 0

    registry = params["mlflow"]
    data_params = params["data"]

    # ---------------------------------------------------------------- champion
    logger.info("Loading the registered champion...")
    pipeline, model_info = load_champion_from_registry(
        registry["registered_model_name"], registry["champion_alias"],
        tracking_uri=registry["tracking_uri"],
    )
    preprocessor, estimator = verify_champion_pipeline(pipeline)
    positive_index = resolve_positive_class_index(estimator)
    logger.info("Champion: %s (version %s), positive class at column %d",
                type(estimator).__name__, model_info["model_version"], positive_index)

    # ------------------------------------------------------------------- data
    # The test split is rebuilt, not re-split: same sizes, seed and stratification as the
    # train stage, so these are the rows the model never saw.
    logger.info("Reconstructing the held-out test split...")
    frame = pd.read_parquet(TRAIN_FEATURES_FILE)
    features = frame.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    target = frame[TARGET_COL].astype(int)
    _, _, X_test, _, _, y_test = split_train_val_test(
        features, target,
        validation_size=float(data_params["validation_size"]),
        test_size=float(data_params["test_size"]),
        random_state=int(data_params["random_state"]),
    )
    # SK_ID_CURR identifies a local explanation and is never a model input; it is carried
    # alongside by index rather than inside the feature frame.
    identifiers = frame.loc[X_test.index, ID_COL]
    logger.info("Test split: %d rows (%.3f%% positive)", len(X_test), 100 * y_test.mean())

    X_sample, y_sample, id_sample = select_explanation_sample(
        X_test, int(settings["sample_size"]), int(settings["random_state"]),
        y=y_test, ids=identifiers,
    )

    # -------------------------------------------------------------- transform
    logger.info("Transforming with the fitted preprocessor (transform-only)...")
    transformed, feature_names = transform_for_explanation(preprocessor, X_sample)
    logger.info("Transformed matrix: %d x %d", *transformed.shape)

    background = None
    background_size = min(int(settings["background_size"]), len(transformed))
    if background_size > 0:
        rng = np.random.default_rng(int(settings["random_state"]))
        background = transformed[
            rng.choice(len(transformed), size=background_size, replace=False)
        ]

    # ------------------------------------------------------------------- SHAP
    result = compute_shap_values(
        estimator, transformed, feature_names,
        positive_index=positive_index, background=background,
    )
    logger.info("SHAP: %s over %d features, %s space (additivity: %s)",
                result.explainer_type, len(feature_names), result.output_space,
                result.additivity)

    probabilities = pipeline.predict_proba(X_sample)[:, positive_index]
    validate_probabilities(probabilities)
    threshold = float(load_threshold(params))

    # ----------------------------------------------------------------- global
    EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    max_display = int(settings["max_display"])

    importance = global_feature_importance(result)
    importance.to_csv(GLOBAL_IMPORTANCE_FILE, index=False)
    logger.info("Wrote %s", GLOBAL_IMPORTANCE_FILE)

    summary_path = save_summary_plot(result, FIGURES_DIR / SUMMARY_PLOT, max_display)
    bar_path = save_bar_plot(importance, FIGURES_DIR / BAR_PLOT, max_display)

    # ------------------------------------------------------------------ local
    picks = choose_local_examples(
        probabilities, threshold, int(settings["local_examples"]), y=y_sample,
    )
    logger.info("Explaining %d applicant(s): %s", len(picks), ", ".join(picks))

    local, local_plots = [], []
    labels = None if y_sample is None else np.asarray(y_sample)
    for reason, position in picks.items():
        applicant_id = None if id_sample is None else id_sample.iloc[position]
        local.append(build_local_explanation(
            result, position, applicant_id=applicant_id,
            probability=float(probabilities[position]), threshold=threshold,
            reason=reason, actual=None if labels is None else labels[position],
        ))
        path = save_local_plot(
            result, position, LOCAL_FIGURES_DIR / f"shap_local_{applicant_id}.png",
            max_display,
        )
        local_plots.append(str(path.relative_to(PROJECT_ROOT)))

    write_json(local, LOCAL_EXPLANATIONS_FILE)

    # ----------------------------------------------------------------- report
    report = {
        "champion_model": type(estimator).__name__,
        **model_info,
        "explained_split": "test",
        "sample_size": int(len(X_sample)),
        "background_size": int(background_size),
        "random_state": int(settings["random_state"]),
        "n_transformed_features": int(len(feature_names)),
        "shap_explainer": result.explainer_type,
        "output_space": result.output_space,
        "base_value": float(result.base_value),
        "additivity_check": result.additivity,
        "positive_class_index": int(positive_index),
        "frozen_threshold": threshold,
        "top_global_features": importance.head(max_display).to_dict(orient="records"),
        "local_examples": [
            {"SK_ID_CURR": item["SK_ID_CURR"], "reason": item["selection_reason"],
             "predicted_probability": item["predicted_probability"]}
            for item in local
        ],
        "artifacts": {
            "global_feature_importance": str(GLOBAL_IMPORTANCE_FILE.relative_to(PROJECT_ROOT)),
            "local_explanations": str(LOCAL_EXPLANATIONS_FILE.relative_to(PROJECT_ROOT)),
            "summary_plot": str(summary_path.relative_to(PROJECT_ROOT)),
            "bar_plot": str(bar_path.relative_to(PROJECT_ROOT)),
            "local_plots": local_plots,
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(report, EXPLANATION_REPORT_FILE)

    log_to_mlflow(report, registry, [
        summary_path, bar_path, GLOBAL_IMPORTANCE_FILE, LOCAL_EXPLANATIONS_FILE,
        EXPLANATION_REPORT_FILE,
    ])

    print(
        f"\nExplained {model_info['model_uri']} (version {model_info['model_version']})\n"
        f"  estimator : {type(estimator).__name__} via {result.explainer_type}\n"
        f"  sample    : {len(X_sample)} rows from the test split x "
        f"{len(feature_names)} transformed features\n"
        f"  top feature: {importance.iloc[0]['feature']} "
        f"(mean |SHAP| {importance.iloc[0]['mean_abs_shap']:.4f})\n"
        f"  local     : {len(local)} applicant(s)\n"
        f"  report    : {EXPLANATION_REPORT_FILE.relative_to(PROJECT_ROOT)}"
    )
    return 0


def load_threshold(params: dict) -> float:
    """The frozen decision threshold from deployment metadata.

    Read from the artifact rather than recomputed: the threshold was tuned on validation
    during training and frozen, and re-deriving it here could disagree with the one the
    pipeline actually ships.
    """
    import json

    meta_path = ARTIFACT_DIR / "deployment_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No deployment metadata at {meta_path}; run the train stage first."
        )
    return float(json.loads(meta_path.read_text(encoding="utf-8"))["optimal_threshold"])


def log_to_mlflow(report: dict, registry: dict, artifacts) -> None:
    """Attach the explanation to MLflow as its own run, tagged back to the model.

    A separate run rather than an append to the training run: reopening a finished run to
    add files risks disturbing the record that the registered version points at, and the
    tags below make the link explicit in both directions. Failure here is warned about,
    never fatal — the artifacts on disk are the deliverable.
    """
    if not registry.get("enabled"):
        logger.info("MLflow tracking is disabled; explanation artifacts stay on disk only.")
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(registry["tracking_uri"])
        mlflow.set_experiment(registry["experiment_name"])
        with mlflow.start_run(run_name=f"explain-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
            mlflow.set_tags({
                "stage": "explain",
                "source_run_id": report.get("source_run_id"),
                "registered_model_name": report.get("registered_model_name"),
                "model_alias": report.get("model_alias"),
                "model_version": report.get("model_version"),
                "shap_explainer": report.get("shap_explainer"),
            })
            mlflow.log_params({
                "sample_size": report["sample_size"],
                "background_size": report["background_size"],
                "random_state": report["random_state"],
                "n_transformed_features": report["n_transformed_features"],
                "explained_split": report["explained_split"],
            })
            for path in artifacts:
                mlflow.log_artifact(str(path), artifact_path="explainability")
        logger.info("Logged explanation artifacts to a dedicated MLflow run.")
    except Exception as err:
        logger.warning("Could not log explanations to MLflow: %s. Artifacts on disk are "
                       "unaffected.", err)


if __name__ == "__main__":
    raise SystemExit(main())
