"""`/predict/batch` — a bounded list.

Bounded is the point. An unbounded list is a denial-of-service shape, and the cap must
be enforced *before* any scoring begins rather than discovered part-way through.
"""

from __future__ import annotations

from tests.api.conftest import APPLICANT_IDS, THRESHOLD


def _payload(ids):
    return {"applicants": [{"sk_id_curr": int(i)} for i in ids]}


def test_batch_returns_200(client) -> None:
    assert client.post("/predict/batch", json=_payload(APPLICANT_IDS)).status_code == 200


def test_one_result_per_applicant(client) -> None:
    body = client.post("/predict/batch", json=_payload(APPLICANT_IDS)).json()
    assert body["count"] == len(APPLICANT_IDS)
    assert len(body["predictions"]) == len(APPLICANT_IDS)


def test_results_keep_request_order_and_ids(client) -> None:
    """A caller joins on the id, so a reordered response would silently mis-attribute."""
    body = client.post("/predict/batch", json=_payload(APPLICANT_IDS)).json()
    assert [p["sk_id_curr"] for p in body["predictions"]] == APPLICANT_IDS


def test_every_result_uses_the_frozen_threshold(client) -> None:
    body = client.post("/predict/batch", json=_payload(APPLICANT_IDS)).json()
    assert body["threshold"] == THRESHOLD
    for prediction in body["predictions"]:
        assert prediction["threshold"] == THRESHOLD
        assert prediction["predicted_class"] == int(
            prediction["default_probability"] >= THRESHOLD)


def test_batch_matches_single_prediction(client) -> None:
    """Two code paths that disagree would make the batch endpoint quietly wrong."""
    single = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    batch = client.post("/predict/batch", json=_payload(APPLICANT_IDS)).json()
    first = batch["predictions"][0]
    assert first["default_probability"] == single["default_probability"]


def test_the_batch_size_limit_is_enforced(client) -> None:
    """The fixture caps the batch at 5."""
    response = client.post("/predict/batch", json=_payload(list(range(100001, 100010))))
    assert response.status_code == 400
    assert response.json()["error"]["category"] == "bad_request"
    assert response.json()["error"]["details"]["max_batch_size"] == 5


def test_an_empty_batch_is_rejected(client) -> None:
    assert client.post("/predict/batch", json={"applicants": []}).status_code == 422


def test_duplicate_identifiers_are_rejected(client) -> None:
    """A duplicate makes the response ambiguous to join back against."""
    duplicated = _payload([APPLICANT_IDS[0], APPLICANT_IDS[0]])
    response = client.post("/predict/batch", json=duplicated)
    assert response.status_code == 422


def test_a_missing_applicant_fails_the_batch_with_its_id(client) -> None:
    response = client.post("/predict/batch", json=_payload([APPLICANT_IDS[0], 424242]))
    assert response.status_code == 404
    assert response.json()["error"]["details"]["sk_id_curr"] == 424242


def test_the_batch_carries_a_request_id(client) -> None:
    body = client.post("/predict/batch", json=_payload(APPLICANT_IDS)).json()
    assert body["request_id"]
