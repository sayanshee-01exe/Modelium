"""Shared fixtures: a tiny fitted pipeline standing in for the champion.

The API is exercised against a real `Predictor` wrapping a real fitted `Pipeline`, so
the promotion gate, threshold logic and schema alignment under test are the production
ones. Only the *registry load* is replaced — resolving an MLflow alias is the one part
that needs infrastructure rather than logic.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURES = ["AMT_INCOME_TOTAL", "EXT_SOURCE_1", "DAYS_BIRTH", "CODE_GENDER"]
THRESHOLD = 0.42          # deliberately not 0.5, so a default would be visible
APPLICANT_IDS = [100001, 100005, 100013]


def build_pipeline():
    """A fitted preprocessing+model Pipeline with the champion's step names."""
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline

    from src.features.data_preprocessing import build_preprocessor

    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "AMT_INCOME_TOTAL": rng.normal(150_000, 40_000, n),
        "EXT_SOURCE_1": rng.random(n),
        "DAYS_BIRTH": rng.integers(7000, 25000, n),
        "CODE_GENDER": rng.choice(["M", "F"], n),
    })
    y = (rng.random(n) > 0.7).astype(int)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X, iqr_factor=1.5)),
        ("model", LGBMClassifier(n_estimators=15, num_leaves=7, verbose=-1)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X, y)
    return pipeline


def build_metadata(promoted: bool = True, threshold: float = THRESHOLD) -> dict:
    return {
        "model_name": "LightGBM (Tuned)",
        "optimal_threshold": threshold,
        "promoted": promoted,
        "input_feature_columns": FEATURES,
        "id_column": "SK_ID_CURR",
        "trained_at": "2026-08-17T20:54:20+00:00",
        "primary_metric": "Average Precision",
        "test_metrics": {"Average Precision": 0.2764, "ROC-AUC": 0.7821},
    }


def build_feature_store() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({
        "SK_ID_CURR": APPLICANT_IDS,
        "AMT_INCOME_TOTAL": rng.normal(150_000, 40_000, len(APPLICANT_IDS)),
        "EXT_SOURCE_1": rng.random(len(APPLICANT_IDS)),
        "DAYS_BIRTH": rng.integers(7000, 25000, len(APPLICANT_IDS)),
        "CODE_GENDER": ["M", "F", "M"],
        # A column the model was not fitted on, so schema trimming is exercised.
        "UNUSED_EXTRA": [1.0, 2.0, 3.0],
    })
    return frame


@pytest.fixture
def registry_stub(monkeypatch, tmp_path):
    """Replace the registry load and the metadata file; leave everything else real.

    Returns a mutable dict the caller can adjust before building the client, so a test
    can simulate a missing alias or an unpromoted model without touching MLflow.
    """
    import json

    import api.dependencies as dependencies

    config = {
        "pipeline": build_pipeline(),
        "metadata": build_metadata(),
        "registry_info": {
            "model_uri": "models:/test-champion@champion",
            "registered_model_name": "test-champion",
            "model_alias": "champion",
            "model_version": "2",
            "source_run_id": "abc123",
            "validation_status": "approved",
        },
        "raise_on_load": None,
    }

    def fake_load(name, alias, tracking_uri=None):
        if config["raise_on_load"] is not None:
            raise config["raise_on_load"]
        return config["pipeline"], config["registry_info"]

    monkeypatch.setattr(dependencies, "load_champion_from_registry", fake_load,
                        raising=False)
    monkeypatch.setattr(
        "src.explainability.shap_explainer.load_champion_from_registry", fake_load)

    meta_file = tmp_path / "deployment_meta.json"

    def write_metadata():
        meta_file.write_text(json.dumps(config["metadata"]), encoding="utf-8")

    write_metadata()
    config["write_metadata"] = write_metadata
    config["metadata_path"] = meta_file
    monkeypatch.setattr(dependencies, "DEPLOYMENT_META_FILE", meta_file)

    store = tmp_path / "features.parquet"
    build_feature_store().to_parquet(store, index=False)
    config["feature_store_path"] = store
    return config


def _patch_settings(registry_stub, monkeypatch):
    from api.config import load_settings

    def patched_settings():
        base = load_settings()
        return type(base)(
            **{**base.__dict__,
               "feature_store_path": registry_stub["feature_store_path"],
               # Settings now carry the metadata path, and they take precedence over the
               # module constant — without this the tests would load the *real* champion's
               # deployment metadata and assert against the wrong threshold.
               "deployment_metadata_path": registry_stub["metadata_path"],
               "registered_model_name": "test-champion",
               "max_batch_size": 5})

    monkeypatch.setattr("api.main.load_settings", patched_settings)


@pytest.fixture
def client(registry_stub, monkeypatch):
    """A TestClient whose lifespan has run, so the model is loaded exactly once."""
    from fastapi.testclient import TestClient

    _patch_settings(registry_stub, monkeypatch)
    from api.main import app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app) as test_client:
            yield test_client


@pytest.fixture
def client_factory(registry_stub, monkeypatch):
    """Build a client *after* the test has adjusted the stub.

    Tests that simulate a broken artifact — an unpromoted model, an invalid threshold —
    must rewrite the metadata before the lifespan runs. They also need the same settings
    patch the `client` fixture applies, or they would read the real champion's metadata
    and assert against the wrong threshold.
    """
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    @contextmanager
    def _make():
        _patch_settings(registry_stub, monkeypatch)
        from api.main import app

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with TestClient(app) as test_client:
                yield test_client

    return _make


@pytest.fixture
def serving_client(registry_stub, monkeypatch):
    """As `client`, but returning the 500 a real deployment would return.

    TestClient re-raises server exceptions by default so a failing test shows the
    traceback. That is the opposite of what a handler test needs: under uvicorn the
    registered handler turns the exception into a JSON 500 and the client never sees the
    traceback, and this fixture reproduces that path.
    """
    from fastapi.testclient import TestClient

    _patch_settings(registry_stub, monkeypatch)
    from api.main import app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client
