#!/usr/bin/env bash
# Container entrypoint: make the mounted MLflow store readable from inside the container,
# then hand off to the API.
#
# The rebasing step is not optional decoration. MLflow records artifact locations as
# absolute paths on the machine that wrote them, so a store mounted from a developer's
# laptop points at directories that do not exist here. prepare_container_registry.py
# rewrites those onto the container's mounts, reading the mounted database read-only and
# writing a rebased copy into /app/runtime/state.
#
# It fails loudly. A container that starts with an unresolvable champion would answer
# /health with 200 and every prediction with 503, which is a far more confusing failure
# than refusing to start with the missing mount named.
set -euo pipefail

echo "[entrypoint] preparing the container's view of the MLflow registry..."
python /app/scripts/prepare_container_registry.py

echo "[entrypoint] starting: $*"
exec "$@"
