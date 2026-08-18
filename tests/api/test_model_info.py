"""`/model/info` — provenance of whatever is answering requests.

The endpoint exists so a caller can tell *which* model produced a score. It must
therefore carry the version and the threshold, and must not carry the host's filesystem
layout or tracking URI, which tell a client nothing and an attacker something.
"""

from __future__ import annotations

from tests.api.conftest import THRESHOLD


def test_model_info_returns_200(client) -> None:
    assert client.get("/model/info").status_code == 200


def test_model_info_identifies_the_served_version(client) -> None:
    body = client.get("/model/info").json()
    assert body["registered_model"] == "test-champion"
    assert body["alias"] == "champion"
    assert body["model_version"] == "2"
    assert body["source_run_id"] == "abc123"


def test_model_info_reports_the_frozen_threshold(client) -> None:
    assert client.get("/model/info").json()["threshold"] == THRESHOLD


def test_model_info_reports_the_schema_size(client) -> None:
    assert client.get("/model/info").json()["expected_feature_count"] == 4


def test_model_info_reports_promotion_and_metrics(client) -> None:
    body = client.get("/model/info").json()
    assert body["promoted"] is True
    assert body["primary_metric"] == "Average Precision"
    assert body["test_metrics"]["ROC-AUC"] == 0.7821


def test_model_info_names_the_estimator_not_a_hard_coded_family(client) -> None:
    assert client.get("/model/info").json()["model_type"] == "LGBMClassifier"


def test_model_info_exposes_no_paths_or_uris(client) -> None:
    body = client.get("/model/info").text
    for leak in ("/Users/", "sqlite:", ".parquet", ".joblib", "mlflow.db", "/var/folders"):
        assert leak not in body, f"{leak} leaked in /model/info"


def test_model_info_is_unavailable_without_a_model(registry_stub) -> None:
    from fastapi.testclient import TestClient

    registry_stub["raise_on_load"] = RuntimeError("no alias")
    from api.main import app

    with TestClient(app) as client:
        response = client.get("/model/info")
    assert response.status_code == 503
    assert response.json()["error"]["category"] == "model_unavailable"
