# Serving image for the Modelium credit-risk API.
#
# Multi-stage: the build stage compiles wheels for packages with native extensions
# (lightgbm, xgboost, scipy, shap), and the runtime stage installs those wheels without
# a compiler toolchain. That keeps gcc/g++ out of the shipped image, which is both
# smaller and a smaller attack surface.
#
# The image contains code only. Every piece of mutable state — the MLflow database, the
# model artifacts, the deployment metadata, the feature store — is mounted at run time.
# Baking a database into an image makes the image a snapshot of a moment, and every
# rebuild silently changes what the service serves.

# --- build stage --------------------------------------------------------------------
# Pinned to 3.13 to match the interpreter the champion pipeline was pickled under;
# unpickling a fitted sklearn/LightGBM object across minor versions is not guaranteed.
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Dependency manifest first, so the wheel layer is cached until requirements change.
# requirements-api.txt is the serving subset: the full development manifest carries the
# training stack (DVC, Optuna, CatBoost) and test tooling that the API never imports.
COPY requirements-api.txt ./
RUN pip install --upgrade pip wheel \
    && pip wheel --wheel-dir=/wheels -r requirements-api.txt

# --- runtime stage ------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MODELIUM_API_HOST=0.0.0.0 \
    MODELIUM_API_PORT=8000

# libgomp1 is the OpenMP runtime LightGBM and XGBoost link against. curl is installed
# deliberately: the compose healthcheck uses it, and a healthcheck that cannot run is
# worse than none.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root from here on. A serving process has no reason to be able to write to its own
# code, and root inside a container is root on a shared kernel.
RUN groupadd --system --gid 1001 modelium \
    && useradd --system --uid 1001 --gid modelium --create-home modelium

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements-api.txt ./
RUN pip install --no-index --find-links=/wheels -r requirements-api.txt \
    && rm -rf /wheels

# Source last: it changes most often, so everything above stays cached.
COPY --chown=modelium:modelium api/ ./api/
COPY --chown=modelium:modelium src/ ./src/
COPY --chown=modelium:modelium config/ ./config/
COPY --chown=modelium:modelium scripts/prepare_container_registry.py ./scripts/
COPY --chown=modelium:modelium docker/entrypoint.sh ./docker/entrypoint.sh
COPY --chown=modelium:modelium params.yaml ./params.yaml

# Mount points and the writable location for the rebased registry copy. Created with
# ownership up front so a read-only mount over them does not leave an unwritable parent.
RUN mkdir -p /app/runtime/state /app/logs \
    && chown -R modelium:modelium /app \
    && chmod +x /app/docker/entrypoint.sh

USER modelium

EXPOSE 8000

# Hits /health, which reports process liveness. Never /predict: a health probe that
# scores an applicant turns monitoring into load, and a slow model into an outage.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
# No --reload: the reloader watches the filesystem and forks workers, neither of which
# belongs in a served image.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
