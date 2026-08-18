"""FastAPI application: lifespan, middleware, error handlers.

Run locally with::

    uvicorn api.main:app --host 127.0.0.1 --port 8000

**This is local serving.** There is no authentication, no rate limiting and no transport
security in this layer; it is not safe to expose publicly as it stands, and the README
says so rather than implying otherwise.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError

from api.config import load_settings
from api.dependencies import load_model_state
from api.exceptions import (
    ApiError, api_error_handler, unhandled_error_handler, validation_error_handler,
)
from api.routes import router
from src.utils.logger import get_logger

logger = get_logger("modelium.api")

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-ms"

DESCRIPTION = """
Serves the **registered champion** credit-default model.

The model is resolved through the MLflow Model Registry alias
`models:/modelium-credit-risk-champion@champion`, never from a local filename, so this
service always answers with the version the pipeline actually promoted. A model that
failed its quality gates is refused rather than served.

Classification uses the **frozen decision threshold** tuned on validation during
training — not sklearn's default 0.5.

*Local serving only: no authentication, no rate limiting.*
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion once, before the first request.

    A failed load is recorded rather than raised: the service starts and reports itself
    unready with the reason, which tells an operator far more than a process that exits.
    """
    settings = load_settings()
    logger.info("Starting API — resolving %s", settings.model_uri)
    app.state.settings = settings
    app.state.model_state = load_model_state(settings)
    if app.state.model_state.is_loaded:
        logger.info("Champion loaded; service is ready.")
    else:
        logger.error("Service started WITHOUT a model: %s",
                     app.state.model_state.load_error)
    yield
    logger.info("Shutting down API.")


app = FastAPI(
    title="Modelium credit-risk API",
    description=DESCRIPTION,
    version="0.10.0",
    lifespan=lifespan,
    # No debug mode, and no docs suppression: the schema is the contract, and exposing
    # it locally is the point. CORS middleware is deliberately absent — no browser
    # origin is permitted by default, so a page cannot call this service cross-origin.
)

app.include_router(router)

app.add_exception_handler(ApiError, api_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id, time the call, and log an entry per request.

    The log line carries the id, endpoint, status and latency — enough to trace and
    profile a call. It deliberately carries no payload: applicant records and feature
    vectors are personal data and have no place in a service log.
    """
    request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
    request.state.request_id = request_id

    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.error("request_id=%s method=%s endpoint=%s status=500 latency_ms=%.1f",
                     request_id, request.method, request.url.path, elapsed_ms)
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers[RESPONSE_TIME_HEADER] = f"{elapsed_ms:.1f}"

    state = getattr(request.app.state, "model_state", None)
    logger.info(
        "request_id=%s method=%s endpoint=%s status=%d latency_ms=%.1f model_version=%s",
        request_id, request.method, request.url.path, response.status_code, elapsed_ms,
        getattr(state, "model_version", None) if state else None,
    )
    return response
