"""Step 4 §12 — the saved champion must be the complete inference pipeline.

Saving only the estimator would force every future consumer to rebuild preprocessing
by hand, which is how training/serving skew starts: two implementations of the same
transformations that drift apart. The artifact must accept a raw DataFrame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.serialization import (
    CHAMPION_PIPELINE_FILENAME,
    METADATA_FILENAME,
    save_champion_pipeline,
)
from src.models.train import build_baseline_pipeline
from src.models.tune import MODEL_STEP, PREPROCESSOR_STEP
from src.utils.exceptions import ModelArtifactError


@pytest.fixture
def fitted_pipeline():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "num_a": signal,
        "num_b": rng.normal(size=n),
        "cat_a": rng.choice(["x", "y"], size=n),
    })
    X.loc[X.sample(frac=0.1, random_state=1).index, "num_b"] = np.nan
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(signal * 2 - 1)))).astype(int))
    pipeline = build_baseline_pipeline(X, random_state=42)
    pipeline.fit(X, y)
    return pipeline, X


@pytest.fixture
def metadata():
    return {"model_name": "Logistic Regression", "optimal_threshold": 0.4321,
            "threshold_selected_on": "validation", "primary_metric": "Average Precision"}


def test_saves_a_pipeline_containing_preprocessing_and_model(fitted_pipeline, metadata, tmp_path):
    pipeline, _ = fitted_pipeline
    save_champion_pipeline(pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")

    reloaded = joblib.load(tmp_path / "models" / CHAMPION_PIPELINE_FILENAME)
    assert isinstance(reloaded, Pipeline)
    assert PREPROCESSOR_STEP in reloaded.named_steps
    assert MODEL_STEP in reloaded.named_steps


def test_reloaded_artifact_predicts_from_a_raw_dataframe(fitted_pipeline, metadata, tmp_path):
    """The §12 acceptance test: champion_pipeline.predict_proba(raw_dataframe)."""
    pipeline, X = fitted_pipeline
    save_champion_pipeline(pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")

    reloaded = joblib.load(tmp_path / "models" / CHAMPION_PIPELINE_FILENAME)
    proba = reloaded.predict_proba(X.head(5))[:, 1]        # raw frame, NaNs and strings
    assert proba.shape == (5,) and np.isfinite(proba).all()


def test_reloaded_predictions_match_the_original(fitted_pipeline, metadata, tmp_path):
    pipeline, X = fitted_pipeline
    before = pipeline.predict_proba(X)[:, 1]
    save_champion_pipeline(pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")
    after = joblib.load(tmp_path / "models" / CHAMPION_PIPELINE_FILENAME).predict_proba(X)[:, 1]
    np.testing.assert_allclose(before, after)


def test_frozen_threshold_is_preserved_in_metadata(fitted_pipeline, metadata, tmp_path):
    pipeline, _ = fitted_pipeline
    save_champion_pipeline(pipeline, metadata, tmp_path / "models", tmp_path / "artifacts")

    saved = json.loads((tmp_path / "artifacts" / METADATA_FILENAME).read_text())
    assert saved["optimal_threshold"] == pytest.approx(0.4321)
    assert saved["threshold_selected_on"] == "validation"


def test_directories_are_created_if_absent(fitted_pipeline, metadata, tmp_path):
    pipeline, _ = fitted_pipeline
    paths = save_champion_pipeline(pipeline, metadata,
                                   tmp_path / "deep" / "models", tmp_path / "deep" / "artifacts")
    assert all(p.exists() for p in paths.values())


def test_rejects_a_bare_estimator(metadata, tmp_path):
    """A bare estimator would silently produce an artifact nobody can serve raw data to."""
    from sklearn.linear_model import LogisticRegression

    with pytest.raises(ModelArtifactError, match="Pipeline"):
        save_champion_pipeline(LogisticRegression(), metadata,
                               tmp_path / "models", tmp_path / "artifacts")


def test_rejects_a_pipeline_without_preprocessing(metadata, tmp_path):
    from sklearn.linear_model import LogisticRegression

    bare = Pipeline([(MODEL_STEP, LogisticRegression())])
    with pytest.raises(ModelArtifactError, match=PREPROCESSOR_STEP):
        save_champion_pipeline(bare, metadata, tmp_path / "models", tmp_path / "artifacts")
