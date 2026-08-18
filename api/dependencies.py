"""Model state: loaded once at startup, injected into handlers.

Two properties matter here.

*The champion is loaded once.* Resolving a registry alias and unpickling a fitted
pipeline takes seconds; doing it per request would make latency dominated by loading and
would hammer the tracking store. The load happens in the lifespan hook and the result is
held on `app.state`, not in a module-level global that any import could rebind.

*A failed load is a served 503, not a crashed process.* If the alias is missing or the
artifact is unloadable, the service still comes up and reports itself unready with the
reason. A container that exits immediately tells an operator far less than one answering
`/ready` with "champion alias does not resolve".

Nothing here re-implements inference. `Predictor` already owns preprocessing reuse,
schema alignment, the frozen threshold, the positive-class column and the promotion
gate; this module hands it a pipeline and its metadata and gets those guarantees for
free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Request

from api.config import PROJECT_ROOT, ApiSettings
from api.exceptions import ApplicantNotFoundError, ModelUnavailableError
from src.utils.logger import get_logger

logger = get_logger("modelium.api.model")

DEPLOYMENT_META_FILE = PROJECT_ROOT / "artifacts" / "deployment_meta.json"
ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


@dataclass
class ModelState:
    """Everything one process needs to serve, plus why it cannot if it cannot."""

    settings: ApiSettings
    predictor: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    registry_info: dict[str, Any] = field(default_factory=dict)
    feature_store: pd.DataFrame | None = None
    explainer: Any | None = None
    explainer_kind: str | None = None
    load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.predictor is not None

    def readiness(self) -> list[tuple[str, bool, str | None]]:
        """Named checks, each reported separately.

        A single boolean would say "not ready" without saying which of five things is
        wrong, which is the question an operator actually has.
        """
        checks: list[tuple[str, bool, str | None]] = []
        checks.append(("model_loaded", self.is_loaded, self.load_error))

        has_metadata = bool(self.metadata)
        checks.append(("metadata_loaded", has_metadata,
                       None if has_metadata else "deployment metadata was not read"))

        threshold = self.metadata.get("optimal_threshold")
        valid_threshold = isinstance(threshold, (int, float)) and 0.0 < float(threshold) < 1.0
        checks.append(("threshold_valid", valid_threshold,
                       None if valid_threshold else "threshold is absent or outside (0, 1)"))

        columns = self.metadata.get("input_feature_columns") or []
        checks.append(("schema_available", bool(columns),
                       None if columns else "no input feature schema in metadata"))

        alias_resolved = bool(self.registry_info.get("model_version"))
        checks.append(("champion_alias_resolved", alias_resolved,
                       None if alias_resolved
                       else f"alias {self.settings.champion_alias!r} did not resolve"))
        return checks

    @property
    def is_ready(self) -> bool:
        return all(passed for _, passed, _ in self.readiness())

    @property
    def model_version(self) -> str | None:
        return self.registry_info.get("model_version")


def load_model_state(settings: ApiSettings) -> ModelState:
    """Resolve the champion alias and build a Predictor around it.

    Never raises. Any failure is recorded on the returned state so the service can start
    and explain itself through `/ready`.
    """
    state = ModelState(settings=settings)

    try:
        from src.explainability.shap_explainer import (
            load_champion_from_registry, verify_champion_pipeline,
        )
        from src.inference.predictor import Predictor

        if not DEPLOYMENT_META_FILE.exists():
            raise FileNotFoundError(
                "deployment metadata is missing; run the train stage")
        metadata = json.loads(DEPLOYMENT_META_FILE.read_text(encoding="utf-8"))

        pipeline, registry_info = load_champion_from_registry(
            settings.registered_model_name, settings.champion_alias,
            tracking_uri=settings.tracking_uri,
        )
        verify_champion_pipeline(pipeline)

        # The pipeline comes from the registry and the metadata from disk, so they are
        # only the same champion if they agree. Serving a model with another run's
        # threshold and schema would be silently wrong in the worst way.
        registered_name = registry_info.get("champion_model") or ""
        recorded_name = str(metadata.get("model_name", ""))
        if registered_name and recorded_name and registered_name != recorded_name:
            raise ValueError(
                f"the registered champion is {registered_name!r} but the deployment "
                f"metadata on disk describes {recorded_name!r}; refusing to serve a "
                f"model with another run's threshold and schema"
            )

        # Predictor performs the promotion check. allow_unpromoted stays False: a model
        # is unpromoted precisely because it was measured and found wanting.
        state.predictor = Predictor(pipeline, metadata)
        state.metadata = metadata
        state.registry_info = registry_info
        logger.info(
            "Serving %s version %s (%s), threshold %.4f, %d expected features",
            settings.registered_model_name, registry_info.get("model_version"),
            metadata.get("model_name"), state.predictor.threshold,
            len(state.predictor.expected_columns),
        )
    except Exception as err:
        state.load_error = f"{type(err).__name__}: {err}"
        logger.error("Champion could not be loaded: %s", state.load_error)
        return state

    state.feature_store = _load_feature_store(
        settings.feature_store_path, state.predictor.expected_columns)
    state.explainer, state.explainer_kind = _build_explainer(state)
    return state


def _load_feature_store(path: Path, expected_columns: list[str]) -> pd.DataFrame | None:
    """Precomputed engineered features, indexed by applicant id for O(1) lookup.

    Trimmed to exactly the columns the champion was fitted on. The stored table is wider
    than the model's schema, and leaving the extras in would make `align_to_training_
    schema` drop them — correctly, but with a WARNING naming all 28 on *every request*,
    which floods the log and puts feature names in it. Dropping them once at startup is
    also less to copy per lookup.
    """
    if not path.exists():
        logger.warning("No feature store at the configured path; id lookups will 404.")
        return None
    try:
        frame = pd.read_parquet(path)
        if ID_COLUMN not in frame.columns:
            logger.warning("Feature store has no %s column; id lookups disabled.",
                           ID_COLUMN)
            return None
        frame = frame.set_index(ID_COLUMN, drop=True)

        available = [c for c in expected_columns if c in frame.columns]
        missing = len(expected_columns) - len(available)
        if missing:
            # Left as-is: Predictor raises a precise InferenceSchemaError per request,
            # which is a better error than a startup guess about why.
            logger.warning("Feature store is missing %d column(s) the champion expects; "
                           "id lookups will fail schema validation.", missing)
        frame = frame.loc[:, available]
        logger.info("Feature store loaded: %d applicants x %d columns (trimmed to the "
                    "champion's schema)", len(frame), frame.shape[1])
        return frame
    except Exception as err:
        logger.warning("Feature store could not be read: %s", err)
        return None


def _build_explainer(state: ModelState):
    """Create the SHAP explainer once, so /explain does not rebuild it per request."""
    try:
        from src.explainability.shap_explainer import select_explainer

        estimator = state.predictor.pipeline.named_steps["model"]
        explainer, kind = select_explainer(estimator)
        logger.info("Explanation enabled using %s", kind)
        return explainer, kind
    except Exception as err:
        # Linear and fallback explainers need background data the API does not hold.
        # /explain reports this cleanly rather than the service failing to start.
        logger.info("Online explanation unavailable: %s", err)
        return None, None


# ---------------------------------------------------------------------------
# Feature resolution
# ---------------------------------------------------------------------------

def resolve_features(state: ModelState, sk_id_curr: int,
                     features: dict[str, Any] | None) -> pd.DataFrame:
    """One applicant as a single-row frame, from the payload or the feature store.

    The returned frame carries the identifier column so `Predictor.predict_dataframe`
    can echo it back; `prepare_features` drops it before the model sees anything, since
    training built its matrix without it.

    Raises:
        ApplicantNotFoundError: if the id is absent from the feature store.
    """
    if features:
        frame = pd.DataFrame([features])
        frame[ID_COLUMN] = sk_id_curr
        return frame.drop(columns=[TARGET_COLUMN], errors="ignore")

    if state.feature_store is None:
        raise ApplicantNotFoundError(
            "No feature store is available on this instance, so applicants cannot be "
            "looked up by id. Send an engineered feature row instead.",
            {"sk_id_curr": sk_id_curr},
        )
    if sk_id_curr not in state.feature_store.index:
        raise ApplicantNotFoundError(
            "No applicant with that identifier is present in the feature store.",
            {"sk_id_curr": sk_id_curr},
        )

    # .loc[[id]] keeps the result a frame even for a single match.
    frame = state.feature_store.loc[[sk_id_curr]].copy()
    frame.insert(0, ID_COLUMN, sk_id_curr)
    return frame.drop(columns=[TARGET_COLUMN], errors="ignore").reset_index(drop=True)


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def get_state(request: Request) -> ModelState:
    """The process-wide model state, from app.state rather than a module global."""
    state = getattr(request.app.state, "model_state", None)
    if state is None:
        raise ModelUnavailableError("The service has not finished starting up.")
    return state


def get_ready_state(request: Request) -> ModelState:
    """As `get_state`, but refuses when the instance cannot actually serve."""
    state = get_state(request)
    if not state.is_loaded:
        raise ModelUnavailableError(
            "The champion model is not loaded, so no prediction can be served. "
            "See /ready for which check failed.",
        )
    return state
