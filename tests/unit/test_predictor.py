"""Step 5 — batch inference against the frozen champion pipeline.

The contracts these pin:

*No refitting.* Inference is transform-only. `test_predictor_never_refits_the_pipeline`
proves it by hand: a pipeline whose `fit` raises still scores fine, and the
preprocessor's learned statistics are byte-identical before and after a batch.

*No 0.5 fallback.* `predict` uses the threshold frozen on validation in Step 4. The
threshold tests move it either side of a known probability and check the class flips.

*Training owns the schema.* A missing column fails loudly rather than being imputed;
extras are dropped and order is corrected.

Small synthetic fits only — no Home Credit data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_preparation import build_relational_feature_table
from src.features.feature_engineering import add_domain_features
from src.inference.predictor import (
    CLASS_COLUMN,
    ID_OUTPUT_COLUMN,
    OUTPUT_COLUMNS,
    PROBABILITY_COLUMN,
    Predictor,
)
from src.models.serialization import save_champion_pipeline
from src.models.train import build_baseline_pipeline
from src.utils.exceptions import InferenceSchemaError, ModelArtifactError

FEATURES = ["num_a", "num_b", "cat_a"]


def _raw_frame(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "num_a": signal,
        "num_b": rng.normal(size=n),
        "cat_a": rng.choice(["x", "y"], size=n),
    })
    X.loc[X.sample(frac=0.1, random_state=1).index, "num_b"] = np.nan
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(2 * signal - 1)))).astype(int))
    return X, y


@pytest.fixture
def fitted_pipeline():
    X, y = _raw_frame()
    pipeline = build_baseline_pipeline(X, random_state=42)
    pipeline.fit(X, y)
    return pipeline


@pytest.fixture
def metadata():
    return {"model_name": "Logistic Regression", "optimal_threshold": 0.35,
            "id_column": "SK_ID_CURR", "target": "TARGET",
            "primary_metric": "Average Precision"}


@pytest.fixture
def predictor(fitted_pipeline, metadata):
    return Predictor(fitted_pipeline, metadata)


@pytest.fixture
def scoring_frame():
    """Raw applicants to score: model features plus the identifier, no TARGET."""
    X, _ = _raw_frame(n=25, seed=7)
    X.insert(0, "SK_ID_CURR", range(100001, 100026))
    return X


# ------------------------------------------------------------------------ loading

def test_predictor_loads_pipeline_and_threshold_from_disk(fitted_pipeline, metadata, tmp_path):
    save_champion_pipeline(fitted_pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")
    loaded = Predictor.load(tmp_path / "models", tmp_path / "artifacts")

    assert loaded.threshold == pytest.approx(0.35)
    assert loaded.model_name == "Logistic Regression"
    assert loaded.expected_columns == FEATURES


def test_loaded_predictor_matches_the_in_memory_one(fitted_pipeline, metadata, scoring_frame, tmp_path):
    save_champion_pipeline(fitted_pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")
    loaded = Predictor.load(tmp_path / "models", tmp_path / "artifacts")

    np.testing.assert_allclose(loaded.predict_proba(scoring_frame),
                               Predictor(fitted_pipeline, metadata).predict_proba(scoring_frame))


def test_missing_artifact_names_the_path(tmp_path):
    with pytest.raises(ModelArtifactError, match="champion pipeline"):
        Predictor.load(tmp_path / "models", tmp_path / "artifacts")


def test_missing_metadata_fails_clearly(fitted_pipeline, metadata, tmp_path):
    save_champion_pipeline(fitted_pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")
    (tmp_path / "artifacts" / "deployment_meta.json").unlink()
    with pytest.raises(ModelArtifactError, match="metadata"):
        Predictor.load(tmp_path / "models", tmp_path / "artifacts")


def test_rejects_a_bare_estimator(metadata):
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(ModelArtifactError, match="Pipeline"):
        Predictor(LogisticRegression(), metadata)


def test_rejects_an_unfitted_pipeline(metadata):
    """Inference is transform-only; it will not fit a preprocessor to rescue a bad load."""
    X, _ = _raw_frame()
    with pytest.raises(ModelArtifactError, match="not fitted"):
        Predictor(build_baseline_pipeline(X, random_state=42), metadata)


def test_missing_threshold_is_fatal_rather_than_defaulting_to_half(fitted_pipeline):
    """§20: never silently score at 0.5 when the frozen threshold is unavailable."""
    with pytest.raises(ModelArtifactError, match="optimal_threshold"):
        Predictor(fitted_pipeline, {"model_name": "m"})


# -------------------------------------------------------------------- probabilities

def test_predict_proba_returns_probabilities(predictor, scoring_frame):
    proba = predictor.predict_proba(scoring_frame)
    assert proba.shape == (len(scoring_frame),)
    assert np.isfinite(proba).all()
    assert ((proba >= 0.0) & (proba <= 1.0)).all()


def test_predict_proba_reports_the_positive_class(fitted_pipeline, metadata):
    """Column located via classes_, not hard-coded — reading the wrong one inverts scores."""
    X, y = _raw_frame()
    predictor = Predictor(fitted_pipeline, metadata)
    proba = predictor.predict_proba(X)
    # Positives must score higher on average than negatives, or the columns are swapped.
    assert proba[y == 1].mean() > proba[y == 0].mean()


# ----------------------------------------------------------------------- threshold

def test_predict_uses_the_stored_threshold(fitted_pipeline, metadata, scoring_frame):
    predictor = Predictor(fitted_pipeline, metadata)
    proba = predictor.predict_proba(scoring_frame)
    np.testing.assert_array_equal(predictor.predict(scoring_frame),
                                  (proba >= 0.35).astype(int))


def test_a_threshold_other_than_half_changes_the_classes(fitted_pipeline, metadata, scoring_frame):
    """The point of §2: a frozen threshold must actually move the decision boundary."""
    proba = Predictor(fitted_pipeline, metadata).predict_proba(scoring_frame)
    # A probability strictly between the two thresholds must classify differently.
    assert ((proba >= 0.2) & (proba < 0.8)).any(), "fixture cannot distinguish thresholds"

    low = Predictor(fitted_pipeline, {**metadata, "optimal_threshold": 0.2})
    high = Predictor(fitted_pipeline, {**metadata, "optimal_threshold": 0.8})
    assert low.predict(scoring_frame).sum() > high.predict(scoring_frame).sum()


def test_threshold_is_inclusive_at_the_boundary(fitted_pipeline, metadata, scoring_frame):
    """`>=`, matching evaluate_at_threshold, so training and serving agree on ties."""
    proba = Predictor(fitted_pipeline, metadata).predict_proba(scoring_frame)
    exact = Predictor(fitted_pipeline, {**metadata, "optimal_threshold": float(proba[0])})
    assert exact.predict(scoring_frame)[0] == 1


# -------------------------------------------------------------------------- schema

def test_extra_columns_are_dropped(predictor, scoring_frame):
    """SK_ID_CURR and any stray column must not reach the model."""
    noisy = scoring_frame.assign(UNSEEN_COLUMN="junk", ANOTHER=1.0)
    np.testing.assert_allclose(predictor.predict_proba(noisy),
                               predictor.predict_proba(scoring_frame))


def test_target_column_in_the_input_is_ignored(predictor, scoring_frame):
    """§7: TARGET is training-only. If a caller supplies one, it must not be a feature."""
    with_target = scoring_frame.assign(TARGET=1)
    np.testing.assert_allclose(predictor.predict_proba(with_target),
                               predictor.predict_proba(scoring_frame))


def test_missing_required_column_fails_clearly(predictor, scoring_frame):
    with pytest.raises(InferenceSchemaError, match="num_a"):
        predictor.predict_proba(scoring_frame.drop(columns=["num_a"]))


def test_missing_column_error_names_the_count(predictor, scoring_frame):
    with pytest.raises(InferenceSchemaError, match="missing 2 column"):
        predictor.predict_proba(scoring_frame.drop(columns=["num_a", "cat_a"]))


def test_column_order_is_corrected(predictor, scoring_frame):
    shuffled = scoring_frame[list(reversed(scoring_frame.columns))]
    np.testing.assert_allclose(predictor.predict_proba(shuffled),
                               predictor.predict_proba(scoring_frame))


def test_prepare_features_returns_the_training_layout(predictor, scoring_frame):
    prepared = predictor.prepare_features(scoring_frame)
    assert list(prepared.columns) == FEATURES
    assert len(prepared) == len(scoring_frame)


def test_empty_input_fails_clearly(predictor, scoring_frame):
    with pytest.raises(InferenceSchemaError, match="empty"):
        predictor.predict_proba(scoring_frame.iloc[0:0])


def test_non_dataframe_input_fails_clearly(predictor):
    with pytest.raises(InferenceSchemaError, match="DataFrame"):
        predictor.predict_proba(np.zeros((5, 3)))


# ------------------------------------------------------------------ output contract

def test_prediction_frame_has_the_expected_columns(predictor, scoring_frame):
    out = predictor.predict_dataframe(scoring_frame)
    assert list(out.columns) == list(OUTPUT_COLUMNS)


def test_prediction_frame_preserves_sk_id_curr(predictor, scoring_frame):
    out = predictor.predict_dataframe(scoring_frame)
    assert list(out[ID_OUTPUT_COLUMN]) == list(scoring_frame["SK_ID_CURR"])
    assert len(out) == len(scoring_frame)


def test_prediction_dtypes_are_float_and_int(predictor, scoring_frame):
    out = predictor.predict_dataframe(scoring_frame)
    assert out[PROBABILITY_COLUMN].dtype.kind == "f"
    assert out[CLASS_COLUMN].dtype.kind == "i"
    assert set(out[CLASS_COLUMN].unique()) <= {0, 1}


def test_prediction_classes_agree_with_the_threshold(predictor, scoring_frame):
    out = predictor.predict_dataframe(scoring_frame)
    expected = (out[PROBABILITY_COLUMN] >= predictor.threshold).astype(int)
    np.testing.assert_array_equal(out[CLASS_COLUMN], expected)


def test_missing_identifier_fails_clearly(predictor, scoring_frame):
    with pytest.raises(InferenceSchemaError, match="SK_ID_CURR"):
        predictor.predict_dataframe(scoring_frame.drop(columns=["SK_ID_CURR"]))


# ---------------------------------------------------------- no retraining, no mutation

def test_predictor_never_refits_the_pipeline(fitted_pipeline, metadata, scoring_frame):
    """§20's hard rule, enforced two ways at once.

    The pipeline's `fit` is replaced with a landmine, and the preprocessor's learned
    statistics are compared before and after a full batch. Either check alone could be
    passed by accident; together they pin that scoring is transform-only.
    """
    def _explode(*args, **kwargs):
        raise AssertionError("inference must never call fit()")

    fitted_pipeline.fit = _explode
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    before = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_.copy()

    predictor = Predictor(fitted_pipeline, metadata)
    predictor.predict_dataframe(scoring_frame)

    after = preprocessor.named_transformers_["num"].named_steps["scaler"].mean_
    np.testing.assert_allclose(before, after)


def test_input_dataframe_is_not_mutated(predictor, scoring_frame):
    snapshot = scoring_frame.copy(deep=True)
    predictor.predict_dataframe(scoring_frame)
    pd.testing.assert_frame_equal(scoring_frame, snapshot)


def test_repeated_scoring_is_stable(predictor, scoring_frame):
    first = predictor.predict_dataframe(scoring_frame)
    second = predictor.predict_dataframe(scoring_frame)
    pd.testing.assert_frame_equal(first, second)


def test_expected_columns_property_is_a_copy(predictor):
    predictor.expected_columns.append("MUTATED")
    assert "MUTATED" not in predictor.expected_columns


# ------------------------------- §17: load -> features -> predictor, end to end small

def _relational_tables(n: int = 30) -> dict[str, pd.DataFrame]:
    """Minimal stand-ins for the Home Credit relations, same keys and shapes."""
    ids = np.arange(100001, 100001 + n)
    rng = np.random.default_rng(3)
    app = pd.DataFrame({
        "SK_ID_CURR": ids,
        "AMT_INCOME_TOTAL": rng.uniform(50_000, 300_000, n),
        "AMT_CREDIT": rng.uniform(100_000, 900_000, n),
        "AMT_ANNUITY": rng.uniform(5_000, 50_000, n),
        "AMT_GOODS_PRICE": rng.uniform(90_000, 850_000, n),
        "DAYS_BIRTH": rng.integers(-25_000, -8_000, n),
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
    })
    bureau = pd.DataFrame({
        "SK_ID_CURR": np.repeat(ids, 2),
        "SK_ID_BUREAU": np.arange(2 * n),
        "AMT_CREDIT_SUM": rng.uniform(1_000, 90_000, 2 * n),
    })
    child = lambda col: pd.DataFrame({                                  # noqa: E731
        "SK_ID_CURR": np.repeat(ids, 2),
        "SK_ID_PREV": np.arange(2 * n),
        col: rng.uniform(0, 1000, 2 * n),
    })
    return {
        "application_train": app.assign(TARGET=rng.integers(0, 2, n)),
        "application_test": app,
        "bureau": bureau,
        "bureau_balance": pd.DataFrame({"SK_ID_BUREAU": np.arange(2 * n),
                                        "MONTHS_BALANCE": rng.integers(-60, 0, 2 * n)}),
        "previous_application": child("AMT_APPLICATION"),
        "pos_cash": child("CNT_INSTALMENT"),
        "credit_card": child("AMT_BALANCE"),
        "installments": child("AMT_PAYMENT"),
    }


def test_end_to_end_small_batch_inference(tmp_path):
    """load -> relational features -> domain features -> Predictor -> prediction frame.

    Trains on the synthetic *train* applicants and scores the *test* applicants through
    the identical feature path, which is the parity §12 requires.
    """
    tables = _relational_tables()

    # Train on application_train, exactly as scripts/train.py does.
    train_df = add_domain_features(build_relational_feature_table(tables, "application_train"))
    y = train_df["TARGET"].astype(int)
    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    pipeline = build_baseline_pipeline(X, random_state=42)
    pipeline.fit(X, y)

    meta = {"model_name": "Logistic Regression", "optimal_threshold": 0.42,
            "id_column": "SK_ID_CURR"}
    save_champion_pipeline(pipeline, meta, tmp_path / "models", tmp_path / "artifacts")

    # Score application_test through the SAME two feature functions.
    test_df = add_domain_features(build_relational_feature_table(tables, "application_test"))
    assert "TARGET" not in test_df.columns

    predictions = Predictor.load(tmp_path / "models", tmp_path / "artifacts").predict_dataframe(test_df)

    assert list(predictions.columns) == list(OUTPUT_COLUMNS)
    assert list(predictions[ID_OUTPUT_COLUMN]) == list(tables["application_test"]["SK_ID_CURR"])
    assert predictions[PROBABILITY_COLUMN].between(0, 1).all()
    assert set(predictions[CLASS_COLUMN].unique()) <= {0, 1}


def test_end_to_end_output_round_trips_through_csv(tmp_path):
    """The deliverable is a CSV; it must survive the write/read with usable dtypes."""
    tables = _relational_tables()
    train_df = add_domain_features(build_relational_feature_table(tables, "application_train"))
    X = train_df.drop(columns=["TARGET", "SK_ID_CURR"])
    pipeline = build_baseline_pipeline(X, random_state=42)
    pipeline.fit(X, train_df["TARGET"].astype(int))

    predictor = Predictor(pipeline, {"model_name": "m", "optimal_threshold": 0.42})
    test_df = add_domain_features(build_relational_feature_table(tables, "application_test"))
    predictions = predictor.predict_dataframe(test_df)

    path = tmp_path / "predictions" / "test_predictions.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(path, index=False)

    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == list(OUTPUT_COLUMNS)
    assert reloaded[CLASS_COLUMN].dtype.kind == "i"
    np.testing.assert_allclose(reloaded[PROBABILITY_COLUMN], predictions[PROBABILITY_COLUMN])
