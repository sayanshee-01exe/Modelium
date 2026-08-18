"""SHAP explanations for the registered champion, computed without refitting anything.

The model this explains is the one the registry says is approved — loaded through
``models:/<name>@<alias>``, not from an arbitrary local file. That matters: a joblib on
disk is whatever the last run happened to leave there, while the alias is the artifact
the pipeline actually promoted. Explaining the wrong one produces a plausible report
about a model nobody is serving.

Three decisions here are deliberate:

*Everything is transform-only.* The champion arrives as a fitted
``Pipeline([("preprocessor", ...), ("model", ...)])``. SHAP needs the transformed matrix
and the estimator separately, so the preprocessor is used via ``transform`` and the
estimator is explained directly. Nothing calls ``fit`` or ``fit_transform`` — refitting
preprocessing on the explanation sample would describe a model that was never trained.

*The positive class is located, never assumed.* SHAP returns different shapes across
estimators and library versions — a 2-D array, a list of per-class arrays, or a 3-D
stack. Each is normalised to the contributions for class ``1`` using the estimator's own
``classes_``. Taking index 1 on faith would silently invert the sign of every
explanation while producing perfectly well-formed output.

*Explanations are computed in margin space, and the report says so.* Tree and linear
explainers attribute to the model's raw score (log-odds), not its probability, so
``base_value + sum(shap) == raw_margin`` holds while the same identity against
``predict_proba`` does not. The additivity check records which space it verified rather
than failing a valid explanation for disagreeing with the wrong reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.features.data_preprocessing import (
    align_to_training_schema, get_input_feature_names, get_output_feature_names,
)
from src.utils.exceptions import DataValidationError, ModelArtifactError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREPROCESSOR_STEP = "preprocessor"
MODEL_STEP = "model"

# This project scores a binary 0/1 default target. Anything else is a different problem.
POSITIVE_CLASS = 1
EXPECTED_CLASSES = {0, 1}

# Explainer kinds, recorded in the report so a reader knows how the numbers were made.
TREE_EXPLAINER = "TreeExplainer"
LINEAR_EXPLAINER = "LinearExplainer"
FALLBACK_EXPLAINER = "Explainer"

# Estimator module prefixes that TreeExplainer supports natively.
_TREE_MODULES = ("lightgbm", "xgboost", "catboost", "sklearn.ensemble", "sklearn.tree")
_LINEAR_MODULES = ("sklearn.linear_model", "sklearn.svm")

# Keys the explainability section must declare, with the minimum each must meet.
# Validated here rather than in src/utils/config_loader.py on purpose: that module is a
# dependency of the train stage, so editing it would invalidate a multi-hour training run
# to add a check that only this stage performs.
EXPLAINABILITY_INT_KEYS: tuple[str, ...] = (
    "sample_size", "background_size", "max_display", "local_examples",
)

# Additivity tolerance. The identity is exact in exact arithmetic; this allows for
# float32 accumulation across ~537 features.
ADDITIVITY_ATOL = 1e-3


@dataclass
class ShapResult:
    """Positive-class SHAP values for one explanation sample, plus their provenance."""

    values: np.ndarray                 # (n_rows, n_features), positive-class
    base_value: float                  # scalar expected value in the explained space
    feature_names: list[str]
    transformed: np.ndarray            # (n_rows, n_features), the explained matrix
    explainer_type: str
    output_space: str                  # "margin" or "probability"
    additivity: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_shap_result(self)


def validate_explainability_settings(settings: Any) -> dict:
    """Check the `explainability:` section before any model or data is touched.

    Raises:
        ConfigurationError: naming the offending field. A typo caught here costs
            milliseconds; the same typo caught after the champion has been loaded and
            2,000 rows transformed costs minutes.
    """
    from src.utils.exceptions import ConfigurationError

    if not isinstance(settings, dict):
        raise ConfigurationError(
            f"params.yaml: 'explainability' must be a mapping, got "
            f"{type(settings).__name__}"
        )
    enabled = settings.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigurationError(
            f"params.yaml: 'explainability.enabled' must be a boolean, got "
            f"{type(enabled).__name__} ({enabled!r}); the string \"false\" is truthy."
        )
    for key in EXPLAINABILITY_INT_KEYS:
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(
                f"params.yaml: 'explainability.{key}' must be an integer, got "
                f"{type(value).__name__} ({value!r})"
            )
        if value < 1:
            raise ConfigurationError(
                f"params.yaml: 'explainability.{key}' must be >= 1, got {value}"
            )
    seed = settings.get("random_state")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError(
            f"params.yaml: 'explainability.random_state' must be a non-negative "
            f"integer, got {seed!r}"
        )
    return settings


# ---------------------------------------------------------------------------
# Loading the registered champion
# ---------------------------------------------------------------------------

def build_model_uri(registered_model_name: str, alias: str) -> str:
    """``models:/<name>@<alias>`` — the URI a consumer resolves, not a file path."""
    if not registered_model_name or not alias:
        raise ModelArtifactError(
            f"Cannot build a model URI from name={registered_model_name!r} "
            f"alias={alias!r}; both are required."
        )
    return f"models:/{registered_model_name}@{alias}"


def load_champion_from_registry(
    registered_model_name: str,
    alias: str,
    tracking_uri: str | None = None,
):
    """Load the aliased champion from the MLflow Model Registry.

    Args:
        registered_model_name: Registered name from params.yaml.
        alias: Alias that marks the approved version.
        tracking_uri: Store to resolve against. None uses the ambient MLflow setting.

    Returns:
        ``(pipeline, info)`` where *info* records the version and source run the alias
        resolved to, so the report can state exactly what was explained.

    Raises:
        ModelArtifactError: if the model is not registered, the alias does not exist, or
            the artifact cannot be loaded. Each is reported as itself rather than as an
            MLflow stack trace, because they need different fixes.
    """
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        versions = client.search_model_versions(f"name='{registered_model_name}'")
    except MlflowException as err:
        raise ModelArtifactError(
            f"No registered model named {registered_model_name!r} in {tracking_uri!r}: "
            f"{err}. Run the register stage before explaining."
        ) from err
    if not versions:
        raise ModelArtifactError(
            f"No versions registered under {registered_model_name!r}. Run the register "
            f"stage before explaining."
        )

    try:
        version = client.get_model_version_by_alias(registered_model_name, alias)
    except MlflowException as err:
        raise ModelArtifactError(
            f"Alias {alias!r} does not resolve for {registered_model_name!r} ({err}). "
            f"{len(versions)} version(s) exist but none holds that alias — the champion "
            f"failed its quality gates, or registration has not run since training."
        ) from err

    uri = build_model_uri(registered_model_name, alias)
    try:
        pipeline = mlflow.sklearn.load_model(uri)
    except Exception as err:
        raise ModelArtifactError(f"Could not load {uri}: {err}") from err

    info = {
        "model_uri": uri,
        "registered_model_name": registered_model_name,
        "model_alias": alias,
        "model_version": str(version.version),
        "source_run_id": version.run_id,
        "validation_status": version.tags.get("validation_status"),
    }
    logger.info("Loaded %s (version %s, run %s)", uri, version.version, version.run_id)
    return pipeline, info


def verify_champion_pipeline(pipeline) -> tuple[Any, Any]:
    """Check the loaded object is a fitted preprocessing+estimator pipeline.

    Returns:
        ``(preprocessor, estimator)``.

    Raises:
        ModelArtifactError: naming the specific missing capability. Failing here beats
            failing inside SHAP, where the message would be about array shapes.
    """
    from sklearn.exceptions import NotFittedError
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import check_is_fitted

    if not isinstance(pipeline, Pipeline):
        raise ModelArtifactError(
            f"Champion must be a sklearn Pipeline carrying its own preprocessing, got "
            f"{type(pipeline).__name__}."
        )
    for step in (PREPROCESSOR_STEP, MODEL_STEP):
        if step not in pipeline.named_steps:
            raise ModelArtifactError(
                f"Champion pipeline has no {step!r} step (found "
                f"{list(pipeline.named_steps)}); SHAP needs the transform and the "
                f"estimator separately."
            )

    preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
    estimator = pipeline.named_steps[MODEL_STEP]

    if not hasattr(estimator, "predict_proba"):
        raise ModelArtifactError(
            f"Champion estimator ({type(estimator).__name__}) has no predict_proba, so "
            f"there is no default probability to explain."
        )
    try:
        check_is_fitted(preprocessor)
    except NotFittedError as err:
        raise ModelArtifactError(
            f"Champion preprocessor is not fitted: {err}. Explanation is transform-only "
            f"and will not fit it."
        ) from err
    if not hasattr(estimator, "classes_"):
        raise ModelArtifactError(
            f"Champion estimator ({type(estimator).__name__}) exposes no classes_, so it "
            f"is either unfitted or cannot say which column is the positive class."
        )
    return preprocessor, estimator


def resolve_positive_class_index(estimator) -> int:
    """Index of class ``1`` in the estimator's own ``classes_``.

    Deliberately no fallback to 1. The position depends on label ordering, and guessing
    it would flip the sign of every explanation while the output still looked valid.
    """
    classes = getattr(estimator, "classes_", None)
    if classes is None:
        raise ModelArtifactError(
            f"{type(estimator).__name__} exposes no classes_; the positive class cannot "
            f"be identified and must not be guessed."
        )
    observed = list(classes.tolist() if hasattr(classes, "tolist") else classes)
    if set(observed) != EXPECTED_CLASSES:
        raise ModelArtifactError(
            f"Champion was fitted on classes {observed}, but this project explains a "
            f"binary 0/1 default target."
        )
    return observed.index(POSITIVE_CLASS)


# ---------------------------------------------------------------------------
# Deterministic explanation sample
# ---------------------------------------------------------------------------

def select_explanation_sample(
    X: pd.DataFrame,
    sample_size: int,
    random_state: int,
    y: pd.Series | None = None,
    ids: pd.Series | None = None,
):
    """Take a reproducible subsample, preserving index alignment across X, y and ids.

    Sampling rather than explaining every row is a memory decision: the holdout is ~46k
    rows by 537 transformed features, and SHAP allocates a value per cell.

    The input frame is never modified — the returned objects are copies.
    """
    if len(X) == 0:
        raise DataValidationError("Cannot explain an empty sample (0 rows)")
    if sample_size <= 0:
        raise DataValidationError(f"sample_size must be positive, got {sample_size}")

    take = min(int(sample_size), len(X))
    if take < len(X):
        # sample() with a fixed seed is reproducible and keeps the original index, which
        # is what maps a row back to its SK_ID_CURR.
        sampled = X.sample(n=take, random_state=random_state).sort_index()
    else:
        logger.info("Sample size %d >= available rows %d; explaining all", sample_size, len(X))
        sampled = X.copy()

    out_y = None if y is None else y.loc[sampled.index].copy()
    out_ids = None if ids is None else ids.loc[sampled.index].copy()
    logger.info("Explanation sample: %d rows (seed %d)", len(sampled), random_state)
    return sampled, out_y, out_ids


def transform_for_explanation(preprocessor, X_raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Align to the training schema and apply the *fitted* preprocessor.

    Transform-only by construction: this calls ``preprocessor.transform``, never ``fit``
    or ``fit_transform``. The caller's frame is not mutated.
    """
    expected = get_input_feature_names(preprocessor)
    aligned = align_to_training_schema(X_raw, expected)
    transformed = preprocessor.transform(aligned)
    if hasattr(transformed, "toarray"):           # a sparse block would break SHAP
        transformed = transformed.toarray()
    transformed = np.asarray(transformed)

    names = get_output_feature_names(preprocessor)
    if transformed.shape[1] != len(names):
        raise ModelArtifactError(
            f"Transformed matrix has {transformed.shape[1]} columns but the preprocessor "
            f"reports {len(names)} feature names. Labelling the SHAP matrix with "
            f"misaligned names would attribute each value to the wrong feature."
        )
    return transformed, names


# ---------------------------------------------------------------------------
# Explainer selection and output normalisation
# ---------------------------------------------------------------------------

def select_explainer(estimator, background: np.ndarray | None = None):
    """Pick the explainer that matches the estimator, returning ``(explainer, kind)``.

    Tree ensembles get ``TreeExplainer`` and linear models ``LinearExplainer`` — both
    exact and cheap for their family. Anything else falls back to the general
    ``shap.Explainer``, which is slower but correct, rather than being refused.
    """
    import shap

    module = type(estimator).__module__

    if any(module.startswith(prefix) for prefix in _TREE_MODULES):
        logger.info("Using TreeExplainer for %s", type(estimator).__name__)
        return shap.TreeExplainer(estimator), TREE_EXPLAINER

    if any(module.startswith(prefix) for prefix in _LINEAR_MODULES):
        if background is None:
            raise ModelArtifactError(
                f"LinearExplainer needs background data for {type(estimator).__name__}; "
                f"none was supplied."
            )
        logger.info("Using LinearExplainer for %s", type(estimator).__name__)
        return shap.LinearExplainer(estimator, background), LINEAR_EXPLAINER

    logger.info("Using the general shap.Explainer fallback for %s", type(estimator).__name__)
    if background is None:
        raise ModelArtifactError(
            f"The fallback explainer needs background data for "
            f"{type(estimator).__name__}; none was supplied."
        )
    return shap.Explainer(estimator, background), FALLBACK_EXPLAINER


def to_positive_class_values(
    raw_values, expected_value, positive_index: int, n_rows: int, n_features: int,
) -> tuple[np.ndarray, float]:
    """Normalise any supported SHAP output to positive-class values and a scalar base.

    SHAP returns three shapes depending on estimator and library version, and all three
    occur across the four models this project trains:

      * ``(n_rows, n_features)``               already collapsed to the positive class
      * ``list[(n_rows, n_features)]``         one array per class
      * ``(n_rows, n_features, n_classes)``    a stacked class dimension

    Raises:
        ModelArtifactError: on any other shape, rather than reshaping until something
            fits — a wrong guess here silently mislabels every contribution.
    """
    if isinstance(raw_values, list):
        if not raw_values:
            raise ModelArtifactError("SHAP returned an empty list of class arrays")
        if positive_index >= len(raw_values):
            raise ModelArtifactError(
                f"SHAP returned {len(raw_values)} class array(s) but the positive class "
                f"is at index {positive_index}."
            )
        values = np.asarray(raw_values[positive_index])
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3:
            if values.shape[2] <= positive_index:
                raise ModelArtifactError(
                    f"SHAP returned a class dimension of {values.shape[2]} but the "
                    f"positive class is at index {positive_index}."
                )
            values = values[:, :, positive_index]
        elif values.ndim != 2:
            raise ModelArtifactError(
                f"Unsupported SHAP output with {values.ndim} dimension(s) and shape "
                f"{values.shape}; expected 2-D, 3-D, or a list of per-class arrays."
            )

    if values.shape != (n_rows, n_features):
        raise ModelArtifactError(
            f"SHAP values have shape {values.shape}, expected ({n_rows}, {n_features}) "
            f"to match the explained sample."
        )

    base = np.asarray(expected_value)
    if base.ndim == 0:
        base_value = float(base)
    elif base.ndim == 1:
        if base.shape[0] == 1:
            base_value = float(base[0])
        elif positive_index < base.shape[0]:
            base_value = float(base[positive_index])
        else:
            raise ModelArtifactError(
                f"expected_value has {base.shape[0]} entries but the positive class is "
                f"at index {positive_index}."
            )
    else:
        raise ModelArtifactError(
            f"Unsupported expected_value with shape {base.shape}; expected a scalar or "
            f"one value per class."
        )
    return values.astype(np.float64), base_value


def raw_margin(estimator, transformed: np.ndarray, positive_index: int):
    """The model's raw score, if the estimator exposes one. ``(values, space)`` or None.

    This is what tree and linear SHAP attribute to. Falling back to a logit of the
    probability would compare against a *reconstruction* of the margin rather than the
    margin itself, so it is reported as a different output space.
    """
    module = type(estimator).__module__
    try:
        if module.startswith("lightgbm"):
            return np.asarray(estimator.predict(transformed, raw_score=True)), "margin"
        if module.startswith("xgboost"):
            booster = estimator.get_booster()
            import xgboost as xgb
            margin = booster.predict(xgb.DMatrix(transformed), output_margin=True)
            return np.asarray(margin), "margin"
        if hasattr(estimator, "decision_function"):
            return np.asarray(estimator.decision_function(transformed)), "margin"
    except Exception as err:                                    # pragma: no cover
        logger.warning("Could not read a raw margin from %s: %s",
                       type(estimator).__name__, err)
    return None, None


def check_additivity(values: np.ndarray, base_value: float, reference) -> dict[str, Any]:
    """Verify ``base + sum(shap) ≈ model output`` where a reference is available.

    Recorded rather than enforced. A mismatch is worth surfacing, but refusing to emit an
    otherwise-valid explanation because the comparison was made in the wrong output space
    would be the more damaging failure.
    """
    if reference is None:
        return {"performed": False,
                "reason": "estimator exposes no raw margin to compare against"}
    reconstruction = base_value + values.sum(axis=1)
    reference = np.asarray(reference).reshape(-1)
    if reference.shape != reconstruction.shape:
        return {"performed": False,
                "reason": f"reference shape {reference.shape} != {reconstruction.shape}"}
    max_error = float(np.abs(reconstruction - reference).max())
    passed = bool(max_error <= ADDITIVITY_ATOL)
    if not passed:
        logger.warning(
            "SHAP additivity is looser than expected: max |base+sum(shap) - margin| = "
            "%.3g (tolerance %.3g)", max_error, ADDITIVITY_ATOL,
        )
    return {"performed": True, "passed": passed, "max_abs_error": max_error,
            "tolerance": ADDITIVITY_ATOL}


def compute_shap_values(
    estimator, transformed: np.ndarray, feature_names: Sequence[str],
    *, positive_index: int, background: np.ndarray | None = None,
) -> ShapResult:
    """Explain the fitted estimator on an already-transformed matrix."""
    explainer, kind = select_explainer(estimator, background)
    logger.info("Computing SHAP values for %d rows x %d features...",
                transformed.shape[0], transformed.shape[1])

    if hasattr(explainer, "shap_values"):
        raw = explainer.shap_values(transformed)
        expected = explainer.expected_value
    else:                                                        # pragma: no cover
        explanation = explainer(transformed)
        raw, expected = explanation.values, explanation.base_values

    values, base_value = to_positive_class_values(
        raw, expected, positive_index, transformed.shape[0], transformed.shape[1],
    )
    reference, space = raw_margin(estimator, transformed, positive_index)
    additivity = check_additivity(values, base_value, reference)

    return ShapResult(
        values=values, base_value=base_value, feature_names=list(feature_names),
        transformed=transformed, explainer_type=kind,
        output_space=space or "unknown", additivity=additivity,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_shap_result(result: ShapResult) -> None:
    """Structural checks run on every ShapResult at construction."""
    if result.values.ndim != 2:
        raise ModelArtifactError(f"SHAP values must be 2-D, got {result.values.shape}")
    rows, features = result.values.shape
    if rows == 0:
        raise DataValidationError("SHAP values contain no rows")
    if features != len(result.feature_names):
        raise ModelArtifactError(
            f"{features} SHAP columns but {len(result.feature_names)} feature names; the "
            f"report would attribute every value to the wrong feature."
        )
    if result.transformed.shape != result.values.shape:
        raise ModelArtifactError(
            f"Explained matrix {result.transformed.shape} does not match SHAP values "
            f"{result.values.shape}"
        )
    if not np.isfinite(result.base_value):
        raise ModelArtifactError(f"base_value is not finite: {result.base_value}")
    if not np.isfinite(result.values).all():
        raise ModelArtifactError("SHAP values contain non-finite entries")


def validate_probabilities(proba: np.ndarray) -> None:
    proba = np.asarray(proba)
    if proba.size == 0:
        raise DataValidationError("No probabilities were produced")
    if not np.isfinite(proba).all() or proba.min() < 0.0 or proba.max() > 1.0:
        raise DataValidationError(
            f"Probabilities must lie in [0, 1], got [{proba.min()}, {proba.max()}]"
        )


# ---------------------------------------------------------------------------
# Global explanation
# ---------------------------------------------------------------------------

def global_feature_importance(result: ShapResult) -> pd.DataFrame:
    """Mean absolute SHAP per feature, ranked. Columns: feature, mean_abs_shap, rank."""
    importance = np.abs(result.values).mean(axis=0)
    frame = pd.DataFrame({
        "feature": result.feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False, kind="mergesort").reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


# ---------------------------------------------------------------------------
# Local explanations
# ---------------------------------------------------------------------------

def choose_local_examples(
    proba: np.ndarray, threshold: float, n: int, y: pd.Series | None = None,
) -> dict[str, int]:
    """Pick positional indices worth explaining, labelled by why they were chosen.

    Deliberately not the top-N riskiest: a report of five near-identical high scores
    teaches nothing. The selection spans the decision — the extremes, the boundary where
    the threshold actually decides, and where labels are available a true positive and a
    false negative, which are the two cases a credit reviewer asks about.
    """
    proba = np.asarray(proba)
    picks: dict[str, int] = {}

    order = np.argsort(proba)
    picks["highest_risk"] = int(order[-1])
    picks["lowest_risk"] = int(order[0])
    picks["nearest_threshold"] = int(np.argmin(np.abs(proba - threshold)))

    if y is not None:
        labels = np.asarray(y)
        predicted = (proba >= threshold).astype(int)
        true_positive = np.flatnonzero((labels == 1) & (predicted == 1))
        false_negative = np.flatnonzero((labels == 1) & (predicted == 0))
        if true_positive.size:
            picks["true_positive"] = int(true_positive[np.argmax(proba[true_positive])])
        if false_negative.size:
            picks["false_negative"] = int(false_negative[np.argmax(proba[false_negative])])

    # De-duplicate while keeping the first (most descriptive) reason for each row.
    unique: dict[str, int] = {}
    seen: set[int] = set()
    for reason, index in picks.items():
        if index not in seen:
            unique[reason] = index
            seen.add(index)
        if len(unique) >= n:
            break

    # Top up from the highest-risk end if the labelled categories were unavailable.
    for index in reversed(order.tolist()):
        if len(unique) >= n:
            break
        if index not in seen:
            unique[f"high_risk_{len(unique)}"] = int(index)
            seen.add(int(index))
    return unique


def build_local_explanation(
    result: ShapResult, position: int, *, applicant_id, probability: float,
    threshold: float, reason: str, actual: int | None = None, top_n: int = 10,
) -> dict[str, Any]:
    """One applicant's contributions, split into what pushed risk up and what pushed it down."""
    row = result.values[position]
    feature_values = result.transformed[position]

    order = np.argsort(row)
    positive = [i for i in reversed(order.tolist()) if row[i] > 0][:top_n]
    negative = [i for i in order.tolist() if row[i] < 0][:top_n]

    def contributions(indices):
        return [
            {
                "feature": result.feature_names[i],
                "feature_value": float(feature_values[i]),
                "shap_value": float(row[i]),
            }
            for i in indices
        ]

    return {
        "SK_ID_CURR": None if applicant_id is None else int(applicant_id),
        "selection_reason": reason,
        "predicted_probability": float(probability),
        "frozen_threshold": float(threshold),
        "predicted_class": int(probability >= threshold),
        "actual_class": None if actual is None else int(actual),
        "base_value": float(result.base_value),
        "output_space": result.output_space,
        "sum_shap_values": float(row.sum()),
        "top_positive_contributors": contributions(positive),
        "top_negative_contributors": contributions(negative),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _figure_axes():
    """Matplotlib with a non-interactive backend, selected before pyplot is imported."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_summary_plot(result: ShapResult, path, max_display: int) -> Path:
    """Beeswarm: per-row contributions, so spread and direction are both visible."""
    import shap

    plt = _figure_axes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        result.values, features=result.transformed, feature_names=result.feature_names,
        max_display=max_display, show=False,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", path)
    return path


def save_bar_plot(importance: pd.DataFrame, path, max_display: int) -> Path:
    """Global ranking by mean |SHAP| — the one number per feature, plotted directly."""
    plt = _figure_axes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = importance.head(max_display).iloc[::-1]
    plt.figure(figsize=(10, max(4, 0.32 * len(top))))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#3b7dd8")
    plt.xlabel("mean |SHAP value|")
    plt.title(f"Global feature importance (top {len(top)})")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    logger.info("Wrote %s", path)
    return path


def save_local_plot(result: ShapResult, position: int, path, max_display: int) -> Path:
    """Waterfall for one applicant, falling back to a bar chart if SHAP declines."""
    import shap

    plt = _figure_axes()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    try:
        explanation = shap.Explanation(
            values=result.values[position],
            base_values=result.base_value,
            data=result.transformed[position],
            feature_names=result.feature_names,
        )
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
    except Exception as err:
        # Waterfall is version-sensitive; a missing local plot is worse than a plainer one.
        logger.warning("Waterfall plot unavailable (%s); drawing a bar chart instead", err)
        plt.close("all")
        row = result.values[position]
        top = np.argsort(np.abs(row))[::-1][:max_display][::-1]
        plt.figure(figsize=(10, max(4, 0.32 * len(top))))
        plt.barh([result.feature_names[i] for i in top], row[top],
                 color=["#c0392b" if row[i] > 0 else "#2e86c1" for i in top])
        plt.xlabel("SHAP value")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    return path


def write_json(payload, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote %s", path)
    return path
