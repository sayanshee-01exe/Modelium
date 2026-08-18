"""Endpoint handlers.

Each handler does three things and no more: resolve the applicant's features, call into
`Predictor` or the SHAP module, and shape the result. Every decision about what a
prediction *means* — which column is the positive class, what threshold separates the
classes, whether this model may be served at all — was made when the `Predictor` was
constructed, and is not revisited per request.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Request, Response, status

from api.dependencies import ModelState, get_ready_state, get_state, resolve_features
from api.exceptions import BadRequestError, ExplanationError, PredictionError
from api.schemas import (
    BatchPredictionRequest, BatchPredictionResponse, ErrorResponse, ExplanationRequest,
    ExplanationResponse, HealthResponse, ModelInfoResponse, PredictionRequest,
    PredictionResponse, ReadinessCheck, ReadinessResponse,
)
from src.utils.logger import get_logger

logger = get_logger("modelium.api.routes")

router = APIRouter()

# Applied to every documented endpoint so the error shape is discoverable from /docs
# rather than only from a failing call.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorResponse, "description": "The request cannot be served as asked."},
    404: {"model": ErrorResponse, "description": "Applicant not in the feature store."},
    422: {"model": ErrorResponse, "description": "Payload failed validation."},
    500: {"model": ErrorResponse, "description": "Internal error."},
    503: {"model": ErrorResponse, "description": "Model unavailable."},
}


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


@router.get(
    "/health", response_model=HealthResponse, tags=["operations"],
    summary="Liveness probe",
    description="Reports that the process is up. It does **not** confirm the service can "
                "serve a prediction — use `/ready` for that.",
)
def health(request: Request, state: ModelState = Depends(get_state)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=state.is_loaded,
        registered_model=state.settings.registered_model_name,
        alias=state.settings.champion_alias,
    )


@router.get(
    "/ready", response_model=ReadinessResponse, tags=["operations"],
    summary="Readiness probe",
    description="Verifies the model, metadata, threshold, schema and champion alias. "
                "Returns 200 when this instance can serve, 503 when it cannot, and "
                "names the failing check either way. Runs no prediction.",
    responses={503: {"model": ReadinessResponse,
                     "description": "One or more readiness checks failed."}},
)
def ready(request: Request, response: Response,
          state: ModelState = Depends(get_state)) -> ReadinessResponse:
    checks = [ReadinessCheck(name=name, passed=passed, detail=detail)
              for name, passed, detail in state.readiness()]
    is_ready = all(check.passed for check in checks)
    response.status_code = (status.HTTP_200_OK if is_ready
                            else status.HTTP_503_SERVICE_UNAVAILABLE)
    return ReadinessResponse(ready=is_ready, checks=checks,
                             request_id=_request_id(request))


@router.get(
    "/model/info", response_model=ModelInfoResponse, tags=["operations"],
    summary="Provenance of the served model",
    description="Which registered version is answering, the frozen threshold it applies, "
                "and the metrics it was promoted on. Carries no filesystem path or "
                "tracking URI.",
    responses={503: ERROR_RESPONSES[503]},
)
def model_info(request: Request,
               state: ModelState = Depends(get_ready_state)) -> ModelInfoResponse:
    metadata = state.metadata
    estimator = state.predictor.pipeline.named_steps["model"]
    test_metrics = {
        key: float(value)
        for key, value in (metadata.get("test_metrics") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return ModelInfoResponse(
        registered_model=state.settings.registered_model_name,
        alias=state.settings.champion_alias,
        model_version=state.model_version,
        model_type=type(estimator).__name__,
        champion_model=metadata.get("model_name"),
        source_run_id=state.registry_info.get("source_run_id"),
        threshold=float(state.predictor.threshold),
        expected_feature_count=len(state.predictor.expected_columns),
        trained_at=metadata.get("trained_at"),
        promoted=bool(metadata.get("promoted", False)),
        primary_metric=metadata.get("primary_metric"),
        test_metrics=test_metrics,
    )


def _score_frame(state: ModelState, frame: pd.DataFrame) -> pd.DataFrame:
    """Delegate to Predictor, converting any failure into a safe API error."""
    try:
        return state.predictor.predict_dataframe(frame)
    except Exception as err:
        # The Predictor's own messages are safe and specific (missing columns, wrong
        # shape); anything else is summarised rather than echoed.
        from src.utils.exceptions import InferenceSchemaError

        if isinstance(err, InferenceSchemaError):
            raise BadRequestError(str(err)) from err
        raise PredictionError("The model could not score this request.") from err


@router.post(
    "/predict", response_model=PredictionResponse, tags=["scoring"],
    summary="Score one applicant",
    description="Send an applicant id to score the precomputed feature row, or supply an "
                "engineered `features` row directly. The class is decided by the frozen "
                "threshold from deployment metadata, never sklearn's 0.5.",
    responses=ERROR_RESPONSES,
)
def predict(request: Request, payload: PredictionRequest,
            state: ModelState = Depends(get_ready_state)) -> PredictionResponse:
    frame = resolve_features(state, payload.sk_id_curr, payload.features)
    scored = _score_frame(state, frame)
    row = scored.iloc[0]
    return PredictionResponse(
        sk_id_curr=int(payload.sk_id_curr),
        default_probability=float(row["DEFAULT_PROBABILITY"]),
        predicted_class=int(row["PREDICTED_CLASS"]),
        threshold=float(state.predictor.threshold),
        model_version=state.model_version,
        request_id=_request_id(request),
    )


@router.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"],
    summary="Score a bounded list of applicants",
    description="One result per applicant, in request order. The batch size is capped by "
                "configuration and duplicate identifiers are rejected, since a duplicate "
                "makes the response ambiguous to join back.",
    responses=ERROR_RESPONSES,
)
def predict_batch(request: Request, payload: BatchPredictionRequest,
                  state: ModelState = Depends(get_ready_state)) -> BatchPredictionResponse:
    limit = state.settings.max_batch_size
    if len(payload.applicants) > limit:
        raise BadRequestError(
            f"The batch contains {len(payload.applicants)} applicants, above the "
            f"configured maximum of {limit}.",
            {"max_batch_size": limit, "received": len(payload.applicants)},
        )

    # Resolved one at a time so a missing applicant is reported with its own id rather
    # than failing the batch with an unattributable error.
    frames = [resolve_features(state, item.sk_id_curr, item.features)
              for item in payload.applicants]
    scored = _score_frame(state, pd.concat(frames, ignore_index=True))

    predictions = [
        PredictionResponse(
            sk_id_curr=int(item.sk_id_curr),
            default_probability=float(row["DEFAULT_PROBABILITY"]),
            predicted_class=int(row["PREDICTED_CLASS"]),
            threshold=float(state.predictor.threshold),
            model_version=state.model_version,
        )
        for item, (_, row) in zip(payload.applicants, scored.iterrows())
    ]
    return BatchPredictionResponse(
        predictions=predictions, count=len(predictions),
        threshold=float(state.predictor.threshold),
        model_version=state.model_version, request_id=_request_id(request),
    )


@router.post(
    "/explain", response_model=ExplanationResponse, tags=["scoring"],
    summary="Explain one applicant's score",
    description="Per-applicant SHAP contributions for a single row, using the explainer "
                "built at startup. Contributions are in **margin (log-odds) space**, so "
                "they sum to the raw score rather than to the probability. Global SHAP "
                "artifacts are produced by the `explain` pipeline stage, not here.",
    responses=ERROR_RESPONSES,
)
def explain(request: Request, payload: ExplanationRequest,
            state: ModelState = Depends(get_ready_state)) -> ExplanationResponse:
    if state.explainer is None:
        raise ExplanationError(
            "Online explanation is not available for the model currently served. The "
            "pipeline's explain stage produces explanations offline.",
        )

    from src.explainability.shap_explainer import (
        build_local_explanation, compute_shap_values, resolve_positive_class_index,
        transform_for_explanation,
    )

    frame = resolve_features(state, payload.sk_id_curr, payload.features)
    if len(frame) > state.settings.max_explain_rows:
        raise BadRequestError(
            f"Explanation is limited to {state.settings.max_explain_rows} applicant(s) "
            f"per request.")

    scored = _score_frame(state, frame)
    probability = float(scored.iloc[0]["DEFAULT_PROBABILITY"])

    try:
        pipeline = state.predictor.pipeline
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]

        # Transform-only, using the champion's own fitted preprocessor.
        aligned = state.predictor.prepare_features(frame)
        transformed, names = transform_for_explanation(preprocessor, aligned)

        result = compute_shap_values(
            estimator, transformed, names,
            positive_index=resolve_positive_class_index(estimator),
            explainer=state.explainer, explainer_kind=state.explainer_kind,
        )
        local = build_local_explanation(
            result, 0, applicant_id=payload.sk_id_curr, probability=probability,
            threshold=float(state.predictor.threshold), reason="api_request",
            top_n=payload.top_n,
        )
    except Exception as err:
        raise ExplanationError("The explanation could not be computed.") from err

    return ExplanationResponse(
        sk_id_curr=int(payload.sk_id_curr),
        default_probability=probability,
        predicted_class=int(local["predicted_class"]),
        threshold=float(state.predictor.threshold),
        base_value=float(local["base_value"]),
        output_space=str(local["output_space"]),
        top_positive_contributors=local["top_positive_contributors"],
        top_negative_contributors=local["top_negative_contributors"],
        model_version=state.model_version,
        request_id=_request_id(request),
    )
