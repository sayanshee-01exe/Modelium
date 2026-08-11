"""Persist the champion as a complete, self-sufficient inference artifact.

The saved object is the fitted ``Pipeline([("preprocessor", ...), ("model", ...)])``,
not the bare estimator, so a consumer can do::

    champion = joblib.load("models/champion_pipeline.pkl")
    champion.predict_proba(raw_dataframe)

Saving only the estimator would force every consumer to rebuild imputation, IQR
clipping, scaling and encoding by hand — two implementations of the same transformations
that drift apart, which is how training/serving skew starts.

The decision threshold is *not* part of the pipeline: sklearn's `predict` hard-codes
0.5, and the threshold this project ships was tuned separately on validation. It lives
in the metadata sidecar, and callers must apply it to `predict_proba` themselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.utils.exceptions import ModelArtifactError
from src.utils.logger import get_logger

logger = get_logger(__name__)

CHAMPION_PIPELINE_FILENAME = "champion_pipeline.pkl"
METADATA_FILENAME = "deployment_meta.json"
PREPROCESSOR_STEP = "preprocessor"


def save_champion_pipeline(pipeline, metadata: dict, model_dir, artifact_dir) -> dict[str, Path]:
    """Persist the fitted champion pipeline and its metadata sidecar.

    Args:
        pipeline: Fitted `Pipeline` containing a preprocessing step and a model step.
        metadata: Deployment metadata; must carry the frozen decision threshold, since
            the pipeline itself cannot.
        model_dir: Destination for the pipeline artifact; created if absent.
        artifact_dir: Destination for the metadata sidecar; created if absent.

    Returns:
        ``{"pipeline": Path, "metadata": Path}``.

    Raises:
        ModelArtifactError: if `pipeline` is not a `Pipeline`, or has no preprocessing
            step. Failing here is far cheaper than discovering at serving time that the
            artifact cannot accept raw data.
    """
    if not isinstance(pipeline, Pipeline):
        raise ModelArtifactError(
            f"Champion artifact must be a sklearn Pipeline carrying its own "
            f"preprocessing, got {type(pipeline).__name__}. A bare estimator cannot "
            f"score raw data."
        )
    if PREPROCESSOR_STEP not in pipeline.named_steps:
        raise ModelArtifactError(
            f"Champion pipeline has no '{PREPROCESSOR_STEP}' step (found: "
            f"{list(pipeline.named_steps)}); the artifact would not be able to "
            f"transform raw input."
        )

    model_dir, artifact_dir = Path(model_dir), Path(artifact_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = model_dir / CHAMPION_PIPELINE_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    joblib.dump(pipeline, pipeline_path, compress=3)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    logger.info(
        "Saved champion pipeline (%s) to %s and metadata to %s",
        " -> ".join(pipeline.named_steps), pipeline_path, metadata_path,
    )
    return {"pipeline": pipeline_path, "metadata": metadata_path}


def save_production_bundle(model, preprocessor, metadata: dict, model_dir, artifact_dir):
    """Deprecated: kept so existing callers keep working.

    Superseded by `save_champion_pipeline`, which persists preprocessing and model as
    one artifact instead of two files a consumer has to reassemble in the right order.
    """
    logger.warning(
        "save_production_bundle is deprecated; use save_champion_pipeline to persist "
        "preprocessing and model as a single inference artifact"
    )
    model_dir, artifact_dir = Path(model_dir), Path(artifact_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.pkl", compress=3)
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl", compress=3)
    with open(artifact_dir / METADATA_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
