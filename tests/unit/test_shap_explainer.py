"""Step 8 — SHAP explanations for the registered champion.

The contracts worth pinning are about *correctness of attribution*, because every way
this can go wrong produces output that still looks like a valid explanation:

*The positive class must be located, not assumed.* SHAP returns a 2-D array, a list of
per-class arrays, or a 3-D stack depending on estimator and library version. Taking
index 1 on faith inverts the sign of every contribution while the report still renders.

*Feature names must line up with the SHAP matrix.* An off-by-one here attributes each
value to its neighbour — the numbers stay plausible and the conclusion is wrong.

*Nothing may be fitted.* Refitting preprocessing on the explanation sample would describe
a transformation the champion was never trained with. A stub that raises on `fit` proves
the code path never reaches one.

Small synthetic pipelines throughout — the Home Credit data is not needed to test any of
this, and a unit suite that needs 2.5 GB is a suite nobody runs.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.shap_explainer import (
    ADDITIVITY_ATOL, FALLBACK_EXPLAINER, LINEAR_EXPLAINER, TREE_EXPLAINER,
    build_local_explanation, build_model_uri, check_additivity, choose_local_examples,
    compute_shap_values, global_feature_importance, load_champion_from_registry,
    resolve_positive_class_index, select_explainer, select_explanation_sample,
    to_positive_class_values, transform_for_explanation, validate_probabilities,
    verify_champion_pipeline,
)
from src.utils.exceptions import DataValidationError, ModelArtifactError


# ---------------------------------------------------------------------------
# Fixtures — small, fitted, and shaped like the real champion
# ---------------------------------------------------------------------------

def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(150_000, 40_000, n),
        "EXT_SOURCE_1": rng.random(n),
        "DAYS_BIRTH": rng.integers(7000, 25000, n),
        "CODE_GENDER": rng.choice(["M", "F"], n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
    })


def _fit_pipeline(estimator, n=120, seed=0):
    from sklearn.pipeline import Pipeline

    from src.features.data_preprocessing import build_preprocessor

    X = _frame(n, seed)
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.7).astype(int)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X, iqr_factor=1.5)),
        ("model", estimator),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y)
    return pipeline, X, pd.Series(y)


@pytest.fixture
def tree_pipeline():
    from lightgbm import LGBMClassifier

    return _fit_pipeline(LGBMClassifier(n_estimators=15, num_leaves=7, verbose=-1))


@pytest.fixture
def linear_pipeline():
    from sklearn.linear_model import LogisticRegression

    return _fit_pipeline(LogisticRegression(max_iter=500))


@pytest.fixture
def shap_result(tree_pipeline):
    pipeline, X, _ = tree_pipeline
    pre, est = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    transformed, names = transform_for_explanation(pre, X)
    return compute_shap_values(est, transformed, names,
                               positive_index=resolve_positive_class_index(est))


# ---------------------------------------------------------------------------
# Champion loading and verification
# ---------------------------------------------------------------------------

def test_model_uri_is_the_alias_form() -> None:
    assert build_model_uri("m", "champion") == "models:/m@champion"


@pytest.mark.parametrize("name,alias", [("", "champion"), ("m", ""), ("m", None)])
def test_incomplete_uri_inputs_fail_clearly(name, alias) -> None:
    with pytest.raises(ModelArtifactError, match="model URI"):
        build_model_uri(name, alias)


def test_champion_pipeline_verification_returns_both_steps(tree_pipeline) -> None:
    pipeline, _, _ = tree_pipeline
    pre, est = verify_champion_pipeline(pipeline)
    assert hasattr(pre, "transform") and hasattr(est, "predict_proba")


def test_a_bare_estimator_is_refused() -> None:
    """SHAP needs the transform and the estimator separately."""
    from lightgbm import LGBMClassifier

    with pytest.raises(ModelArtifactError, match="Pipeline"):
        verify_champion_pipeline(LGBMClassifier())


def test_a_pipeline_without_a_preprocessor_is_refused() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    pipe = Pipeline([("model", LogisticRegression())]).fit(X, [0, 1, 0, 1])
    with pytest.raises(ModelArtifactError, match="preprocessor"):
        verify_champion_pipeline(pipe)


def test_an_unfitted_preprocessor_is_refused() -> None:
    """Explanation is transform-only and will not fit anything for you."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([("preprocessor", StandardScaler()), ("model", LogisticRegression())])
    with pytest.raises(ModelArtifactError, match="not fitted"):
        verify_champion_pipeline(pipe)


def test_missing_registered_model_fails_clearly(tmp_path) -> None:
    with pytest.raises(ModelArtifactError, match="No registered model|No versions"):
        load_champion_from_registry("absent-model", "champion",
                                    tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")


def test_a_registered_model_without_the_alias_fails_clearly(tmp_path, tree_pipeline) -> None:
    """A version existing is not the same as a version being approved."""
    import mlflow

    pipeline, _, _ = tree_pipeline
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment("e")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(
                pipeline, name="model", serialization_format="cloudpickle")
        mlflow.register_model(model_uri=info.model_uri, name="toy")

    with pytest.raises(ModelArtifactError, match="does not resolve"):
        load_champion_from_registry("toy", "champion", tracking_uri=uri)


def test_champion_loads_through_the_alias(tmp_path, tree_pipeline) -> None:
    import mlflow
    from mlflow import MlflowClient

    pipeline, X, _ = tree_pipeline
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("e", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment("e")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(
                pipeline, name="model", serialization_format="cloudpickle")
        version = mlflow.register_model(model_uri=info.model_uri, name="toy")
        MlflowClient(tracking_uri=uri).set_registered_model_alias(
            "toy", "champion", str(version.version))
        loaded, meta = load_champion_from_registry("toy", "champion", tracking_uri=uri)

    assert meta["model_uri"] == "models:/toy@champion"
    assert meta["model_version"] == str(version.version)
    assert loaded.predict_proba(X).shape == (len(X), 2)


# ---------------------------------------------------------------------------
# Positive class
# ---------------------------------------------------------------------------

def test_positive_class_index_is_read_from_the_estimator(tree_pipeline) -> None:
    pipeline, _, _ = tree_pipeline
    est = pipeline.named_steps["model"]
    assert resolve_positive_class_index(est) == list(est.classes_).index(1)


def test_non_binary_classes_are_refused() -> None:
    class _Multi:
        classes_ = np.array([0, 1, 2])

    with pytest.raises(ModelArtifactError, match="binary"):
        resolve_positive_class_index(_Multi())


def test_an_estimator_without_classes_is_refused() -> None:
    class _NoClasses:
        pass

    with pytest.raises(ModelArtifactError, match="classes_"):
        resolve_positive_class_index(_NoClasses())


def test_reversed_class_order_selects_the_right_column() -> None:
    """classes_=[1,0] puts the positive class at index 0; assuming 1 would invert signs."""
    class _Reversed:
        classes_ = np.array([1, 0])

    assert resolve_positive_class_index(_Reversed()) == 0


# ---------------------------------------------------------------------------
# SHAP output normalisation — the three shapes that occur in practice
# ---------------------------------------------------------------------------

def test_2d_output_passes_through() -> None:
    values = np.arange(12, dtype=float).reshape(3, 4)
    out, base = to_positive_class_values(values, 0.5, 1, 3, 4)
    assert out.shape == (3, 4) and base == 0.5


def test_list_output_selects_the_positive_class() -> None:
    negative, positive = np.zeros((3, 4)), np.ones((3, 4))
    out, base = to_positive_class_values([negative, positive], [0.1, 0.9], 1, 3, 4)
    assert (out == 1).all() and base == pytest.approx(0.9)


def test_3d_output_selects_the_positive_class() -> None:
    stacked = np.stack([np.zeros((3, 4)), np.ones((3, 4))], axis=2)
    out, _ = to_positive_class_values(stacked, np.array([0.1, 0.9]), 1, 3, 4)
    assert out.shape == (3, 4) and (out == 1).all()


def test_reversed_class_order_picks_the_other_array() -> None:
    negative, positive = np.zeros((3, 4)), np.ones((3, 4))
    out, base = to_positive_class_values([positive, negative], [0.9, 0.1], 0, 3, 4)
    assert (out == 1).all() and base == pytest.approx(0.9)


@pytest.mark.parametrize("bad", [
    np.zeros((3, 4, 5, 6)),
    np.zeros(7),
])
def test_unsupported_shap_shape_fails_clearly(bad) -> None:
    with pytest.raises(ModelArtifactError, match="Unsupported SHAP output|shape"):
        to_positive_class_values(bad, 0.0, 1, 3, 4)


def test_shap_values_must_match_the_sample(tree_pipeline) -> None:
    with pytest.raises(ModelArtifactError, match="expected"):
        to_positive_class_values(np.zeros((9, 4)), 0.0, 1, 3, 4)


def test_empty_class_list_fails_clearly() -> None:
    with pytest.raises(ModelArtifactError, match="empty list"):
        to_positive_class_values([], 0.0, 1, 3, 4)


def test_a_single_expected_value_is_the_positive_class_base() -> None:
    """Binary tree explainers report one base value, and it is already the positive
    class's. Rejecting it for not having two entries would refuse a valid explanation."""
    _, base = to_positive_class_values(np.zeros((3, 4)), np.array([0.1]), 1, 3, 4)
    assert base == pytest.approx(0.1)


def test_expected_value_shorter_than_the_class_index_fails() -> None:
    """Two entries but an index past the end means the class order was misread."""
    with pytest.raises(ModelArtifactError, match="expected_value"):
        to_positive_class_values(np.zeros((3, 4)), np.array([0.1, 0.2]), 5, 3, 4)


# ---------------------------------------------------------------------------
# Explainer selection
# ---------------------------------------------------------------------------

def test_tree_models_get_the_tree_explainer(tree_pipeline) -> None:
    pipeline, _, _ = tree_pipeline
    _, kind = select_explainer(pipeline.named_steps["model"])
    assert kind == TREE_EXPLAINER


def test_random_forest_gets_the_tree_explainer() -> None:
    from sklearn.ensemble import RandomForestClassifier

    pipeline, _, _ = _fit_pipeline(RandomForestClassifier(n_estimators=5, max_depth=3))
    _, kind = select_explainer(pipeline.named_steps["model"])
    assert kind == TREE_EXPLAINER


def test_linear_models_get_the_linear_explainer(linear_pipeline) -> None:
    pipeline, X, _ = linear_pipeline
    pre, est = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    transformed, _ = transform_for_explanation(pre, X)
    _, kind = select_explainer(est, background=transformed[:20])
    assert kind == LINEAR_EXPLAINER


def test_a_linear_model_without_background_fails_clearly(linear_pipeline) -> None:
    pipeline, _, _ = linear_pipeline
    with pytest.raises(ModelArtifactError, match="background"):
        select_explainer(pipeline.named_steps["model"], background=None)


def test_the_explainer_is_not_hard_coded_to_lightgbm(linear_pipeline, tree_pipeline) -> None:
    """Two different families must not resolve to the same explainer."""
    lin, X, _ = linear_pipeline
    tree, _, _ = tree_pipeline
    pre = lin.named_steps["preprocessor"]
    transformed, _ = transform_for_explanation(pre, X)
    _, linear_kind = select_explainer(lin.named_steps["model"], background=transformed[:20])
    _, tree_kind = select_explainer(tree.named_steps["model"])
    assert linear_kind != tree_kind


# ---------------------------------------------------------------------------
# Transform-only guarantees
# ---------------------------------------------------------------------------

def test_transform_never_fits(tree_pipeline) -> None:
    """A preprocessor that raises on fit must survive the whole transform path."""
    pipeline, X, _ = tree_pipeline
    pre = pipeline.named_steps["preprocessor"]

    def _explode(*_a, **_k):
        raise AssertionError("explanation must never fit the preprocessor")

    pre.fit, pre.fit_transform = _explode, _explode
    transformed, names = transform_for_explanation(pre, X)
    assert transformed.shape[0] == len(X) and len(names) == transformed.shape[1]


def test_the_input_frame_is_not_mutated(tree_pipeline) -> None:
    pipeline, X, _ = tree_pipeline
    before = X.copy(deep=True)
    transform_for_explanation(pipeline.named_steps["preprocessor"], X)
    pd.testing.assert_frame_equal(X, before)


def test_schema_alignment_handles_extra_and_missing_columns(tree_pipeline) -> None:
    pipeline, X, _ = tree_pipeline
    drifted = X.drop(columns=["EXT_SOURCE_1"]).assign(UNEXPECTED=1.0)
    transformed, names = transform_for_explanation(
        pipeline.named_steps["preprocessor"], drifted)
    assert transformed.shape == (len(X), len(names))


def test_feature_names_match_the_transformed_width(shap_result) -> None:
    assert len(shap_result.feature_names) == shap_result.values.shape[1]
    assert shap_result.transformed.shape == shap_result.values.shape


# ---------------------------------------------------------------------------
# Deterministic sampling and identifiers
# ---------------------------------------------------------------------------

def test_sampling_is_deterministic() -> None:
    X = _frame(200)
    a, _, _ = select_explanation_sample(X, 50, 42)
    b, _, _ = select_explanation_sample(X, 50, 42)
    pd.testing.assert_frame_equal(a, b)


def test_a_different_seed_gives_a_different_sample() -> None:
    X = _frame(200)
    a, _, _ = select_explanation_sample(X, 50, 42)
    b, _, _ = select_explanation_sample(X, 50, 7)
    assert list(a.index) != list(b.index)


def test_ids_and_labels_stay_aligned_with_the_sample() -> None:
    X = _frame(200)
    ids = pd.Series(range(9000, 9200), index=X.index)
    y = pd.Series((np.arange(200) % 3 == 0).astype(int), index=X.index)
    sample, y_out, id_out = select_explanation_sample(X, 40, 42, y=y, ids=ids)
    assert list(id_out.index) == list(sample.index) == list(y_out.index)
    assert list(id_out) == [ids.loc[i] for i in sample.index]


def test_sampling_does_not_mutate_the_source() -> None:
    X = _frame(100)
    before = X.copy(deep=True)
    select_explanation_sample(X, 20, 42)
    pd.testing.assert_frame_equal(X, before)


def test_a_sample_larger_than_the_data_uses_everything() -> None:
    X = _frame(30)
    sample, _, _ = select_explanation_sample(X, 500, 42)
    assert len(sample) == 30


def test_an_empty_frame_fails_clearly() -> None:
    with pytest.raises(DataValidationError, match="empty"):
        select_explanation_sample(pd.DataFrame(), 10, 42)


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------

def test_global_importance_is_sorted_and_ranked(shap_result) -> None:
    frame = global_feature_importance(shap_result)
    assert list(frame.columns) == ["feature", "mean_abs_shap", "rank"]
    assert frame["mean_abs_shap"].is_monotonic_decreasing
    assert list(frame["rank"]) == list(range(1, len(frame) + 1))


def test_global_importance_covers_every_feature(shap_result) -> None:
    frame = global_feature_importance(shap_result)
    assert len(frame) == len(shap_result.feature_names)
    assert set(frame["feature"]) == set(shap_result.feature_names)


def test_global_importance_is_non_negative(shap_result) -> None:
    """It is a mean of absolute values; a negative entry means a sign error upstream."""
    assert (global_feature_importance(shap_result)["mean_abs_shap"] >= 0).all()


# ---------------------------------------------------------------------------
# Local explanations
# ---------------------------------------------------------------------------

def test_local_explanation_has_every_required_field(shap_result) -> None:
    item = build_local_explanation(
        shap_result, 0, applicant_id=12345, probability=0.61, threshold=0.5,
        reason="highest_risk", actual=1,
    )
    for key in ["SK_ID_CURR", "predicted_probability", "frozen_threshold",
                "predicted_class", "actual_class", "base_value",
                "top_positive_contributors", "top_negative_contributors"]:
        assert key in item, key
    assert item["SK_ID_CURR"] == 12345
    assert item["predicted_class"] == 1


def test_local_contributions_carry_feature_value_and_shap(shap_result) -> None:
    item = build_local_explanation(
        shap_result, 0, applicant_id=1, probability=0.2, threshold=0.5, reason="x")
    for group in ("top_positive_contributors", "top_negative_contributors"):
        for entry in item[group]:
            assert set(entry) == {"feature", "feature_value", "shap_value"}


def test_positive_and_negative_contributors_have_the_right_sign(shap_result) -> None:
    item = build_local_explanation(
        shap_result, 0, applicant_id=1, probability=0.2, threshold=0.5, reason="x")
    assert all(e["shap_value"] > 0 for e in item["top_positive_contributors"])
    assert all(e["shap_value"] < 0 for e in item["top_negative_contributors"])


def test_local_explanation_without_a_label_is_allowed(shap_result) -> None:
    item = build_local_explanation(
        shap_result, 0, applicant_id=1, probability=0.2, threshold=0.5, reason="x")
    assert item["actual_class"] is None


def test_local_examples_span_the_decision() -> None:
    """Extremes, the boundary, and the two label-dependent cases a reviewer asks about."""
    proba = np.array([0.02, 0.44, 0.52, 0.60, 0.71, 0.99])
    y = pd.Series([0, 1, 0, 1, 0, 0])
    picks = choose_local_examples(proba, 0.5, 6, y=y)
    assert picks["highest_risk"] == 5
    assert picks["lowest_risk"] == 0
    assert picks["nearest_threshold"] == 2
    # index 3 is labelled 1 and scored above the threshold -> a true positive;
    # index 1 is labelled 1 and scored below it -> a false negative.
    assert picks["true_positive"] == 3
    assert picks["false_negative"] == 1


def test_colliding_local_examples_are_deduplicated() -> None:
    """When the riskiest applicant is also the true positive, one row is not explained
    twice under two headings; the remaining slots are topped up by risk."""
    proba = np.array([0.01, 0.2, 0.49, 0.51, 0.95])
    y = pd.Series([0, 0, 1, 1, 1])
    picks = choose_local_examples(proba, 0.5, 5, y=y)
    assert len(set(picks.values())) == len(picks) == 5
    assert picks["highest_risk"] == 4


def test_local_examples_are_unique_positions() -> None:
    proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4])
    picks = choose_local_examples(proba, 0.5, 5)
    assert len(set(picks.values())) == len(picks) == 5


def test_local_examples_respect_the_requested_count() -> None:
    proba = np.linspace(0.01, 0.99, 20)
    assert len(choose_local_examples(proba, 0.5, 3)) == 3


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def test_probabilities_outside_the_unit_interval_are_refused() -> None:
    with pytest.raises(DataValidationError, match=r"\[0, 1\]"):
        validate_probabilities(np.array([0.5, 1.7]))


def test_empty_probabilities_are_refused() -> None:
    with pytest.raises(DataValidationError, match="No probabilities"):
        validate_probabilities(np.array([]))


def test_valid_probabilities_pass(shap_result, tree_pipeline) -> None:
    pipeline, X, _ = tree_pipeline
    validate_probabilities(pipeline.predict_proba(X)[:, 1])


# ---------------------------------------------------------------------------
# Additivity
# ---------------------------------------------------------------------------

def test_additivity_holds_for_a_tree_model(shap_result) -> None:
    """base + sum(shap) reconstructs the raw margin, which is what trees attribute to."""
    assert shap_result.additivity["performed"] is True
    assert shap_result.additivity["passed"] is True
    assert shap_result.additivity["max_abs_error"] < ADDITIVITY_ATOL


def test_the_explained_output_space_is_recorded(shap_result) -> None:
    """Probability-space comparison would fail against margin-space values."""
    assert shap_result.output_space == "margin"


def test_additivity_is_skipped_rather_than_failed_without_a_reference() -> None:
    outcome = check_additivity(np.zeros((3, 4)), 0.0, None)
    assert outcome["performed"] is False and "reason" in outcome
