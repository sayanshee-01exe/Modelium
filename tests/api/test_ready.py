"""`/ready` — can this instance actually serve?

Readiness reports each check by name. A single boolean would say "not ready" without
saying which of five things is wrong, which is the only question an operator has.
"""

from __future__ import annotations

import warnings

CHECKS = {"model_loaded", "metadata_loaded", "threshold_valid", "schema_available",
          "champion_alias_resolved"}


def test_ready_returns_200_when_loaded(client) -> None:
    assert client.get("/ready").status_code == 200


def test_ready_reports_every_check(client) -> None:
    body = client.get("/ready").json()
    assert body["ready"] is True
    assert {check["name"] for check in body["checks"]} == CHECKS
    assert all(check["passed"] for check in body["checks"])


def test_ready_returns_503_when_the_model_is_missing(registry_stub) -> None:
    from fastapi.testclient import TestClient

    registry_stub["raise_on_load"] = RuntimeError("champion alias does not resolve")
    from api.main import app

    with TestClient(app) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["ready"] is False


def test_a_failed_check_names_itself(registry_stub) -> None:
    """"Not ready" without a reason is not actionable."""
    from fastapi.testclient import TestClient

    registry_stub["raise_on_load"] = RuntimeError("champion alias does not resolve")
    from api.main import app

    with TestClient(app) as client:
        checks = {c["name"]: c for c in client.get("/ready").json()["checks"]}
    assert checks["model_loaded"]["passed"] is False
    assert "alias" in checks["model_loaded"]["detail"]


def test_an_invalid_threshold_blocks_readiness(registry_stub, client_factory) -> None:
    """A threshold outside (0, 1) means the artifact cannot decide a class."""
    from tests.api.conftest import build_metadata

    registry_stub["metadata"] = build_metadata(threshold=1.5)
    registry_stub["write_metadata"]()

    with client_factory() as client:
        response = client.get("/ready")
    # The Predictor refuses the artifact outright, so the model never loads.
    assert response.status_code == 503
    checks = {c["name"]: c for c in response.json()["checks"]}
    assert checks["threshold_valid"]["passed"] is False


def test_readiness_runs_no_prediction(client) -> None:
    """It is a probe, called frequently; scoring inside it would be a cost per poll."""
    import api.routes as routes

    source = routes.__file__
    with open(source, encoding="utf-8") as handle:
        body = handle.read()
    ready_block = body.split("def ready(")[1].split("def model_info(")[0]
    assert "predict" not in ready_block
