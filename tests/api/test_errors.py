"""Error handling.

One rule dominates: **a client never sees a stack trace.** An exception from inside
sklearn, MLflow or pandas carries file paths, library versions and sometimes data
values. Every failure leaves as a structured payload with a category a caller can branch
on, and the detail goes to the log where the request id ties it back.
"""

from __future__ import annotations

import warnings

from tests.api.conftest import APPLICANT_IDS, build_metadata


def _error(response):
    return response.json()["error"]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_every_error_uses_the_same_envelope(client) -> None:
    response = client.post("/predict", json={"sk_id_curr": 424242})
    body = response.json()
    assert set(body) == {"error", "request_id"}
    assert set(_error(response)) == {"category", "message", "details"}


def test_errors_carry_the_request_id(client) -> None:
    response = client.post("/predict", json={"sk_id_curr": 424242})
    assert response.json()["request_id"]


# ---------------------------------------------------------------------------
# Status codes
# ---------------------------------------------------------------------------

def test_a_missing_applicant_is_404(client) -> None:
    response = client.post("/predict", json={"sk_id_curr": 424242})
    assert response.status_code == 404
    assert _error(response)["category"] == "applicant_not_found"


def test_an_invalid_payload_is_422(client) -> None:
    response = client.post("/predict", json={"sk_id_curr": "not-an-integer"})
    assert response.status_code == 422
    assert _error(response)["category"] == "validation_error"


def test_a_missing_required_field_is_422(client) -> None:
    assert client.post("/predict", json={}).status_code == 422


def test_an_unknown_field_is_rejected(client) -> None:
    """extra="forbid" — a typo'd field name must fail rather than be ignored."""
    response = client.post("/predict",
                           json={"sk_id_curr": APPLICANT_IDS[0], "sk_id": 1})
    assert response.status_code == 422


def test_a_negative_identifier_is_422(client) -> None:
    assert client.post("/predict", json={"sk_id_curr": -5}).status_code == 422


def test_target_in_features_is_rejected(client) -> None:
    """TARGET is the label; accepting it would invite scoring on leaked data."""
    response = client.post("/predict", json={
        "sk_id_curr": 1, "features": {"AMT_INCOME_TOTAL": 1.0, "TARGET": 1}})
    assert response.status_code == 422


def test_an_empty_feature_map_is_rejected(client) -> None:
    response = client.post("/predict", json={"sk_id_curr": 1, "features": {}})
    assert response.status_code == 422


def test_an_incomplete_feature_row_is_400_with_the_missing_columns(client) -> None:
    """The Predictor's schema message is specific and safe to pass through."""
    response = client.post("/predict", json={
        "sk_id_curr": 1, "features": {"AMT_INCOME_TOTAL": 150000.0}})
    assert response.status_code == 400
    assert _error(response)["category"] == "bad_request"
    assert "missing" in _error(response)["message"].lower()


# ---------------------------------------------------------------------------
# Model availability
# ---------------------------------------------------------------------------

def test_a_missing_champion_alias_is_503(registry_stub) -> None:
    from fastapi.testclient import TestClient

    from mlflow.exceptions import MlflowException

    registry_stub["raise_on_load"] = MlflowException("alias champion does not resolve")
    from api.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]})
    assert response.status_code == 503
    assert _error(response)["category"] == "model_unavailable"


def test_an_unpromoted_model_is_blocked(registry_stub) -> None:
    """A model is unpromoted precisely because it was measured and found wanting."""
    from fastapi.testclient import TestClient

    registry_stub["metadata"] = build_metadata(promoted=False)
    registry_stub["write_metadata"]()
    from api.main import app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as client:
            ready = client.get("/ready")
            predicted = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]})
    assert ready.status_code == 503
    assert predicted.status_code == 503
    assert _error(predicted)["category"] == "model_unavailable"


def test_a_mismatched_metadata_and_registry_pair_is_refused(registry_stub) -> None:
    """Serving one run's pipeline with another run's threshold is silently wrong."""
    from fastapi.testclient import TestClient

    registry_stub["registry_info"] = {**registry_stub["registry_info"],
                                      "champion_model": "XGBoost (Tuned)"}
    from api.main import app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as client:
            response = client.get("/ready")
    assert response.status_code == 503
    checks = {c["name"]: c for c in response.json()["checks"]}
    assert "refusing to serve" in checks["model_loaded"]["detail"]


# ---------------------------------------------------------------------------
# Nothing leaks
# ---------------------------------------------------------------------------

def test_no_error_body_contains_a_stack_trace(client) -> None:
    responses = [
        client.post("/predict", json={"sk_id_curr": 424242}),
        client.post("/predict", json={"sk_id_curr": "bad"}),
        client.post("/predict/batch", json={"applicants": []}),
    ]
    for response in responses:
        text = response.text
        for leak in ("Traceback", "File \"", ".py\", line", "site-packages"):
            assert leak not in text, f"{leak} leaked in {response.url}"


def test_validation_errors_do_not_echo_the_submitted_value(client) -> None:
    """Pydantic's raw errors() embeds the input, which here is applicant data."""
    response = client.post("/predict", json={"sk_id_curr": 987654321012345})
    assert "987654321012345" not in response.text


def test_an_unexpected_error_is_a_generic_500(serving_client, monkeypatch) -> None:
    """Uses the fixture that does not re-raise, so this exercises what a deployed
    service returns rather than what TestClient does for debugging."""
    import api.routes as routes

    def explode(*_args, **_kwargs):
        raise RuntimeError("secret detail /Users/someone/model.joblib")

    monkeypatch.setattr(routes, "resolve_features", explode)
    response = serving_client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]})
    assert response.status_code == 500
    assert "secret detail" not in response.text
    assert "/Users/" not in response.text
    assert _error(response)["category"] == "internal_error"
