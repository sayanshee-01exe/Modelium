"""`/health` — liveness only.

It must answer even when the model failed to load: an orchestrator uses liveness to
decide whether to restart the process, and a health check that fails because the model
is missing would trigger a restart loop that cannot fix the problem. `/ready` is the
endpoint that reports serviceability.
"""

from __future__ import annotations


def test_health_returns_200(client) -> None:
    assert client.get("/health").status_code == 200


def test_health_reports_the_loaded_model(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["registered_model"] == "test-champion"
    assert body["alias"] == "champion"


def test_health_answers_even_without_a_model(registry_stub, monkeypatch) -> None:
    """Liveness must not depend on the model, or a restart loop follows."""
    from fastapi.testclient import TestClient

    registry_stub["raise_on_load"] = RuntimeError("alias missing")
    from api.main import app

    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_health_leaks_no_paths(client) -> None:
    """An API has no reason to disclose its own directory layout."""
    body = client.get("/health").text
    for leak in ("/Users/", "sqlite:", ".parquet", "artifacts/"):
        assert leak not in body


def test_health_carries_a_request_id_header(client) -> None:
    assert client.get("/health").headers.get("X-Request-ID")


def test_an_incoming_request_id_is_preserved(client) -> None:
    """A caller's correlation id must survive, so traces join across services."""
    supplied = "caller-supplied-id-123"
    response = client.get("/health", headers={"X-Request-ID": supplied})
    assert response.headers["X-Request-ID"] == supplied


def test_latency_is_reported(client) -> None:
    value = client.get("/health").headers.get("X-Response-Time-ms")
    assert value is not None and float(value) >= 0.0
