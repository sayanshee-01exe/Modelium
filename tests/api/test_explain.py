"""`/explain` — per-applicant SHAP for one row.

Online explanation reuses the pipeline's SHAP module rather than reimplementing it, and
is bounded: one applicant per call, a capped contributor count, and an explainer built
once at startup rather than per request.
"""

from __future__ import annotations

from tests.api.conftest import APPLICANT_IDS, THRESHOLD


def test_explain_returns_200(client) -> None:
    assert client.post("/explain", json={"sk_id_curr": APPLICANT_IDS[0]}).status_code == 200


def test_explanation_has_every_required_field(client) -> None:
    body = client.post("/explain", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    for key in ("sk_id_curr", "default_probability", "predicted_class", "threshold",
                "base_value", "output_space", "top_positive_contributors",
                "top_negative_contributors"):
        assert key in body, key


def test_contributions_carry_feature_value_and_shap(client) -> None:
    body = client.post("/explain", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    for group in ("top_positive_contributors", "top_negative_contributors"):
        for entry in body[group]:
            assert set(entry) == {"feature", "feature_value", "shap_value"}


def test_contributor_signs_match_their_direction(client) -> None:
    body = client.post("/explain", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert all(c["shap_value"] > 0 for c in body["top_positive_contributors"])
    assert all(c["shap_value"] < 0 for c in body["top_negative_contributors"])


def test_the_explained_output_space_is_stated(client) -> None:
    """Contributions sum to the raw margin, not the probability; saying so avoids a
    reader checking them against predict_proba and concluding they are broken."""
    body = client.post("/explain", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert body["output_space"] == "margin"


def test_the_explanation_agrees_with_the_prediction(client) -> None:
    predicted = client.post("/predict", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    explained = client.post("/explain", json={"sk_id_curr": APPLICANT_IDS[0]}).json()
    assert explained["default_probability"] == predicted["default_probability"]
    assert explained["predicted_class"] == predicted["predicted_class"]
    assert explained["threshold"] == THRESHOLD


def test_top_n_bounds_the_response(client) -> None:
    body = client.post("/explain",
                       json={"sk_id_curr": APPLICANT_IDS[0], "top_n": 2}).json()
    assert len(body["top_positive_contributors"]) <= 2
    assert len(body["top_negative_contributors"]) <= 2


def test_an_excessive_top_n_is_rejected(client) -> None:
    response = client.post("/explain",
                           json={"sk_id_curr": APPLICANT_IDS[0], "top_n": 5000})
    assert response.status_code == 422


def test_the_explainer_is_built_once(client) -> None:
    """Rebuilding a TreeExplainer per request would dominate the response time."""
    import api.main as main

    explainer = main.app.state.model_state.explainer
    assert explainer is not None
    for applicant in APPLICANT_IDS:
        client.post("/explain", json={"sk_id_curr": applicant})
    assert main.app.state.model_state.explainer is explainer


def test_a_missing_applicant_returns_404(client) -> None:
    response = client.post("/explain", json={"sk_id_curr": 424242})
    assert response.status_code == 404


def test_explain_reuses_the_pipeline_shap_module() -> None:
    """Duplicated SHAP logic would drift from the offline explanations."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "api" / "routes.py").read_text(
        encoding="utf-8")
    assert "from src.explainability.shap_explainer import" in source
    assert "shap.TreeExplainer" not in source
