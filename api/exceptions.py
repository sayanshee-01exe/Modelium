"""API error types and the handlers that turn them into safe JSON.

One rule shapes this module: **a client never sees a stack trace.** An exception from
deep inside sklearn, MLflow or pandas carries file paths, library versions and sometimes
data values, none of which belong in a response body. Every error leaves here as a
structured payload with a category a caller can branch on, while the detail goes to the
log where the request id ties it back.

Status codes follow what the caller can do about it:

    400  the request was understood but cannot be served as asked
    404  the applicant is not in the feature store
    422  the payload failed validation (FastAPI's own default)
    500  something broke on our side
    503  the model is not available, so no prediction is possible right now
"""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger("modelium.api.errors")


class ApiError(Exception):
    """Base for errors this service raises deliberately.

    Attributes:
        message: Safe, client-facing text. Must never embed a path or a stack trace.
        category: Stable machine-readable slug, so a caller can branch without parsing
            prose that may be reworded later.
        status_code: HTTP status to return.
        details: Optional structured extras, already vetted as safe to disclose.
    """

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    category = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ModelUnavailableError(ApiError):
    """The champion could not be loaded, so nothing can be scored.

    503 rather than 500: the service is working, the model is not, and a caller should
    retry later rather than treat the request as malformed.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    category = "model_unavailable"


class ApplicantNotFoundError(ApiError):
    status_code = status.HTTP_404_NOT_FOUND
    category = "applicant_not_found"


class BadRequestError(ApiError):
    status_code = status.HTTP_400_BAD_REQUEST
    category = "bad_request"


class PredictionError(ApiError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    category = "prediction_failed"


class ExplanationError(ApiError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    category = "explanation_failed"


def _payload(request: Request, category: str, message: str,
             details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "category": category,
            "message": message,
            "details": details or {},
        },
        "request_id": getattr(request.state, "request_id", None),
    }


async def api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    logger.warning(
        "request_id=%s endpoint=%s category=%s status=%d message=%s",
        getattr(request.state, "request_id", None), request.url.path,
        exc.category, exc.status_code, exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(request, exc.category, exc.message, exc.details),
    )


async def validation_error_handler(
    request: Request, exc: RequestValidationError,
) -> JSONResponse:
    """Pydantic's report, trimmed to field/message pairs.

    The raw `errors()` output embeds the offending input value, which for this service
    is applicant data. Only the location and the rule are echoed back.
    """
    fields = [
        {"field": ".".join(str(part) for part in error.get("loc", [])),
         "message": error.get("msg", "invalid value")}
        for error in exc.errors()
    ]
    logger.info(
        "request_id=%s endpoint=%s category=validation_error status=422 fields=%d",
        getattr(request.state, "request_id", None), request.url.path, len(fields),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=_payload(request, "validation_error",
                         "The request payload failed validation.", {"fields": fields}),
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. The traceback is logged; the client gets a generic sentence.

    Deliberately says nothing about what broke: an unexpected exception is exactly the
    case where the message is most likely to contain a path or a value.
    """
    logger.error(
        "request_id=%s endpoint=%s category=internal_error status=500 error=%s",
        getattr(request.state, "request_id", None), request.url.path,
        type(exc).__name__, exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_payload(request, "internal_error",
                         "An internal error occurred. The request id identifies it in "
                         "the service log."),
    )
