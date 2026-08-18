"""Pydantic request and response models.

Two decisions shape the request shape:

*An applicant is identified, not described.* The champion expects 407 engineered
features built by a relational aggregation over ~57 M child rows. Requiring a caller to
supply those by hand would be unusable, and rebuilding them inside a request would be a
second implementation of the training feature pipeline — which is how training/serving
skew starts. So the default contract is an id, resolved against the precomputed feature
store. A caller that genuinely holds an engineered row may pass it instead.

*`SK_ID_CURR` is an identifier, never a feature.* It is carried through to the response
so a caller can join results back, and dropped before the matrix reaches the model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Home Credit ids are positive integers. Bounded rather than unbounded so a nonsense
# value fails validation instead of becoming a lookup miss.
MIN_ID = 1
MAX_ID = 1_000_000_000


class ErrorDetail(BaseModel):
    category: str = Field(..., description="Stable machine-readable error slug.")
    message: str = Field(..., description="Safe, client-facing description.")
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """The single error shape every failing endpoint returns."""

    error: ErrorDetail
    request_id: str | None = Field(
        None, description="Correlates this response with the service log.")


class HealthResponse(BaseModel):
    """Liveness. Answers 'is the process up', not 'can it serve'."""

    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    registered_model: str
    alias: str


class ReadinessCheck(BaseModel):
    name: str
    passed: bool
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """Readiness. Answers 'can this instance serve a prediction right now'."""

    ready: bool
    checks: list[ReadinessCheck]
    request_id: str | None = None


class ModelInfoResponse(BaseModel):
    """Provenance of the model currently being served.

    Deliberately carries no filesystem path and no tracking URI: a caller needs to know
    *which* model answered, not where this host keeps its files.
    """

    registered_model: str
    alias: str
    model_version: str | None = None
    model_type: str = Field(..., description="Estimator class serving requests.")
    champion_model: str | None = Field(
        None, description="Display name recorded when the champion was selected.")
    source_run_id: str | None = None
    threshold: float = Field(..., description="Frozen decision threshold, tuned on "
                                              "validation and never 0.5 by default.")
    expected_feature_count: int
    trained_at: str | None = None
    promoted: bool
    primary_metric: str | None = None
    test_metrics: dict[str, float] = Field(default_factory=dict)


class PredictionRequest(BaseModel):
    """One applicant, by id or by an already-engineered feature row."""

    model_config = ConfigDict(extra="forbid")

    sk_id_curr: int = Field(
        ..., ge=MIN_ID, le=MAX_ID,
        description="Applicant identifier. Used to look the row up in the feature "
                    "store, and echoed back in the response.",
        examples=[100001],
    )
    features: dict[str, Any] | None = Field(
        None,
        description="Optional pre-engineered feature row. When supplied it is used "
                    "instead of the feature-store lookup. Must match the training "
                    "schema; TARGET is rejected.",
    )

    @field_validator("features")
    @classmethod
    def reject_target_and_empty(cls, value):
        """`TARGET` is the label. Accepting it would invite scoring on leaked data."""
        if value is None:
            return value
        if not value:
            raise ValueError("features was supplied but empty; omit it to use the "
                             "feature store")
        for forbidden in ("TARGET", "target"):
            if forbidden in value:
                raise ValueError("TARGET is the label and must not be sent as a feature")
        return value


class PredictionResponse(BaseModel):
    sk_id_curr: int
    default_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_class: int = Field(..., ge=0, le=1)
    threshold: float
    model_version: str | None = None
    request_id: str | None = None


class BatchPredictionRequest(BaseModel):
    """A bounded list of applicants.

    Bounded because an unbounded list is a denial-of-service shape: the cap is enforced
    by validation, so an oversized payload is rejected before any scoring begins.
    """

    model_config = ConfigDict(extra="forbid")

    applicants: list[PredictionRequest] = Field(..., min_length=1)

    @model_validator(mode="after")
    def reject_duplicate_ids(self):
        """Duplicates make the response ambiguous to join back against."""
        ids = [item.sk_id_curr for item in self.applicants]
        duplicates = sorted({value for value in ids if ids.count(value) > 1})
        if duplicates:
            raise ValueError(
                f"duplicate sk_id_curr values in the batch: {duplicates[:5]}")
        return self


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    threshold: float
    model_version: str | None = None
    request_id: str | None = None


class Contribution(BaseModel):
    feature: str
    feature_value: float
    shap_value: float


class ExplanationRequest(PredictionRequest):
    """Same addressing as a prediction; explanation is a view of the same scoring."""

    top_n: int = Field(
        10, ge=1, le=50,
        description="Contributors returned per direction. Bounded to keep a single "
                    "request cheap.",
    )


class ExplanationResponse(BaseModel):
    sk_id_curr: int
    default_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_class: int = Field(..., ge=0, le=1)
    threshold: float
    base_value: float = Field(
        ..., description="Model output for an average applicant, in the explained space.")
    output_space: str = Field(
        ..., description="'margin' means contributions sum to the raw log-odds score, "
                         "not to the probability.")
    top_positive_contributors: list[Contribution]
    top_negative_contributors: list[Contribution]
    model_version: str | None = None
    request_id: str | None = None
