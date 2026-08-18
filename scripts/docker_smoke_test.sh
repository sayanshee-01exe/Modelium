#!/usr/bin/env bash
# End-to-end check that the container actually serves the registered champion.
#
# This is the test the static config tests cannot be: it builds the image, starts the
# stack, and drives the real HTTP contract. It is the only way to catch the failure that
# matters most here — a container that starts cleanly but cannot resolve the champion
# because a mount is missing or an artifact path was not rebased.
#
# Cleans up on any exit path, so a failure never leaves a stack running.
set -euo pipefail

API="${MODELIUM_SMOKE_URL:-http://127.0.0.1:8000}"
APPLICANT="${MODELIUM_SMOKE_ID:-100001}"
TIMEOUT="${MODELIUM_SMOKE_TIMEOUT:-180}"
COMPOSE="docker compose"

cd "$(dirname "$0")/.."

cleanup() {
    local code=$?
    echo
    echo "--- cleaning up ---"
    if [ $code -ne 0 ]; then
        echo "container log (last 40 lines):"
        $COMPOSE logs --tail 40 api || true
    fi
    $COMPOSE down --remove-orphans >/dev/null 2>&1 || true
    exit $code
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "=== 1/7 build ==="
$COMPOSE build

echo "=== 2/7 start ==="
$COMPOSE up -d

echo "=== 3/7 wait for /health (up to ${TIMEOUT}s) ==="
deadline=$((SECONDS + TIMEOUT))
until curl -fsS "$API/health" >/dev/null 2>&1; do
    [ $SECONDS -lt $deadline ] || fail "/health did not respond within ${TIMEOUT}s"
    sleep 3
done
echo "healthy after ${SECONDS}s"

echo "=== 4/7 readiness ==="
ready_code=$(curl -sS -o /tmp/modelium_ready.json -w '%{http_code}' "$API/ready")
[ "$ready_code" = "200" ] || { cat /tmp/modelium_ready.json; fail "/ready returned $ready_code"; }
echo "/ready 200"

echo "=== 5/7 model info ==="
curl -fsS "$API/model/info" -o /tmp/modelium_info.json || fail "/model/info failed"
python3 - <<'PY' || fail "/model/info did not carry the expected champion"
import json
info = json.load(open("/tmp/modelium_info.json"))
assert info["alias"] == "champion", info["alias"]
assert info["promoted"] is True, "a model that failed its gates must never be served"
assert 0.0 < info["threshold"] < 1.0, info["threshold"]
assert info["threshold"] != 0.5, "the frozen threshold must not be sklearn's default"
assert info["expected_feature_count"] > 0
print(f"   serving {info['registered_model']} v{info['model_version']} "
      f"({info['model_type']}) threshold={info['threshold']:.4f}")
PY

echo "=== 6/7 prediction ==="
curl -fsS -X POST "$API/predict" \
    -H 'Content-Type: application/json' \
    -d "{\"sk_id_curr\": ${APPLICANT}}" -o /tmp/modelium_pred.json \
    || fail "/predict failed"
APPLICANT="$APPLICANT" python3 - <<'PY' || fail "/predict response was not well formed"
import json, os
body = json.load(open("/tmp/modelium_pred.json"))
for field in ("sk_id_curr", "default_probability", "predicted_class", "threshold",
              "model_version", "request_id"):
    assert field in body, f"missing field: {field}"
assert body["sk_id_curr"] == int(os.environ["APPLICANT"]), "identifier was not preserved"
assert 0.0 <= body["default_probability"] <= 1.0, body["default_probability"]
assert body["predicted_class"] in (0, 1)
# The class must follow the frozen threshold, not 0.5. No exact probability is asserted:
# that would pin the test to one trained artifact.
assert body["predicted_class"] == int(body["default_probability"] >= body["threshold"])
print(f"   applicant {body['sk_id_curr']}: p={body['default_probability']:.4f} "
      f"class={body['predicted_class']} at threshold {body['threshold']:.4f}")
PY

echo "=== 7/7 error contract ==="
missing=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$API/predict" \
    -H 'Content-Type: application/json' -d '{"sk_id_curr": 999999999}')
[ "$missing" = "404" ] || fail "unknown applicant returned $missing, expected 404"
invalid=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$API/predict" \
    -H 'Content-Type: application/json' -d '{"sk_id_curr": "abc"}')
[ "$invalid" = "422" ] || fail "invalid payload returned $invalid, expected 422"
echo "404 and 422 behave as documented"

echo
echo "SMOKE TEST PASSED"
