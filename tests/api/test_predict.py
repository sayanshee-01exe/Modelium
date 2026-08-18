"""`/predict` — one applicant.

The behaviours worth pinning: the frozen threshold decides the class, the identifier
survives the round trip without reaching the model, and nothing is fitted.
"""

from __future__ import annotations

import numpy as np

from tests.api.conftest import APPLICANT_IDS, THRESHOLD


def test_predict_returns_200(client) -> None:
    assert client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).status_code == 200


def test_predict_returns_the_documented_shape(client) -> None:
    body = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    for key in ("sk_id_curr", "default_probability", "predicted_class", "threshold",
                "model_version"):
        assert key in body, key


def test_the_applicant_id_is_preserved(client) -> None:
    for applicant in APPLICANT_IDS:
        body = client.post("/predict", json={"sk_id_curr": applicant}).json()
        assert body["sk_id_curr"] == applicant


def test_the_probability_is_a_valid_probability(client) -> None:
    value = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert 0.0 <= value["default_probability"] <= 1.0


def test_the_frozen_threshold_is_returned_not_one_half(client) -> None:
    body = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert body["threshold"] == THRESHOLD
    assert body["threshold"] != 0.5


def test_the_class_is_decided_by_the_frozen_threshold(client) -> None:
    """Not sklearn's 0.5 — the two disagree for any probability between them."""
    for applicant in APPLICANT_IDS:
        body = client.post("/predict", json={"sk_id_curr": applicant}).json()
        expected = int(body["default_probability"] >= THRESHOLD)
        assert body["predicted_class"] == expected


def test_an_inline_feature_row_is_accepted(client) -> None:
    payload = {
        "sk_id_curr": 999999,
        "features": {"AMT_INCOME_TOTAL": 150000.0, "EXT_SOURCE_1": 0.5,
                     "DAYS_BIRTH": 15000, "CODE_GENDER": "M"},
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sk_id_curr"] == 999999


def test_the_identifier_never_reaches_the_model(client) -> None:
    """Two applicants with identical features but different ids must score identically."""
    features = {"AMT_INCOME_TOTAL": 150000.0, "EXT_SOURCE_1": 0.5,
                "DAYS_BIRTH": 15000, "CODE_GENDER": "M"}
    first = client.post("/predict", json={"sk_id_curr": 111111, "features": features})
    second = client.post("/predict", json={"sk_id_curr": 222222, "features": features})
    assert first.json()["default_probability"] == second.json()["default_probability"]


def test_the_feature_store_is_not_mutated_by_scoring(client) -> None:
    import api.main as main

    store = main.app.state.model_state.feature_store
    before = store.copy(deep=True)
    client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]})
    import pandas as pd

    pd.testing.assert_frame_equal(store, before)


def test_repeated_requests_are_deterministic(client) -> None:
    first = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    second = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert first["default_probability"] == second["default_probability"]


def test_the_model_is_loaded_once(client) -> None:
    """Reloading per request would make latency dominated by unpickling."""
    import api.main as main

    predictor = main.app.state.model_state.predictor
    for applicant in APPLICANT_IDS:
        client.post("/predict", json={"sk_id_curr": applicant})
    assert main.app.state.model_state.predictor is predictor


def test_no_fit_call_exists_in_the_api_package() -> None:
    from pathlib import Path

    package = Path(__file__).resolve().parents[2] / "api"
    for module in package.glob("*.py"):
        source = module.read_text(encoding="utf-8")
        for forbidden in (".fit(", ".fit_transform(", "RandomizedSearchCV"):
            assert forbidden not in source, f"{module.name} must not call {forbidden}"


def test_a_request_id_is_returned_in_the_body(client) -> None:
    body = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert body["request_id"]
