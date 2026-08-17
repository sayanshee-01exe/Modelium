"""Step 8 — MLflow Model Registry.

Training already produces an artifact on disk and a tracked run. The registry adds the
missing question: *which* recorded run is the one a consumer should serve, and has it
been approved?

The contracts worth pinning here are about promotion, not about MLflow:

*A rejected model must never reach the production alias.* The pipeline already refuses
to score with an unpromoted champion; the registry must not hand one out under a name
that reads as approved. Registration itself still happens — the rejected version is
recorded, aliased ``candidate`` and tagged ``rejected``, because a refused run is part
of the history.

*Re-running must not fan out versions.* DVC re-runs a stage whenever any dependency
changes, so registering the same run twice must reuse the version it already created.

*Tracking disabled must stay inert.* With ``mlflow.enabled: false`` there is no run to
register, and the stage must say so and succeed rather than fail the pipeline.

Registry behaviour is exercised against a real SQLite-backed MLflow store under
``tmp_path`` — the same backend `params.yaml` configures — so these are integration
tests, not assertions about a mock.
"""

from __future__ import annotations

import copy
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

from src.models.register_model import (
    CANDIDATE_STATUS,
    PROMOTED_STATUS,
    RUN_INFO_REQUIRED_KEYS,
    build_registry_record,
    build_run_information,
    load_run_information,
    register_champion,
    write_registry_record,
    write_run_information,
)
from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.config_loader import load_params, validate_params
from src.utils.exceptions import ConfigurationError, ModelArtifactError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_pipeline():
    """A minimal fitted Pipeline with the same step names as the real champion."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((30, 3)), columns=["a", "b", "c"])
    y = (rng.random(30) > 0.5).astype(int)
    pipeline = Pipeline([
        ("preprocessor", StandardScaler()),
        ("model", LogisticRegression()),
    ])
    pipeline.fit(X, y)
    return pipeline, X


@pytest.fixture
def local_tracker(tmp_path):
    """A real, enabled tracker writing to a throwaway SQLite store."""
    return MLflowTracker(
        enabled=True,
        experiment_name="register-test",
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
    )


@pytest.fixture
def logged_run(local_tracker, toy_pipeline, tmp_path):
    """Log a model to a real run and return its run information dict."""
    pipeline, X = toy_pipeline
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with local_tracker.start_run(run_name="probe"):
            model_uri = local_tracker.log_model(pipeline)
            info = build_run_information(
                local_tracker,
                model_uri=model_uri,
                registered_model_name="test-champion",
                champion_model="Logistic Regression",
                promoted=True,
                optimal_threshold=0.42,
                test_metrics={"Average Precision": 0.31, "ROC-AUC": 0.72},
            )
    return info


# ---------------------------------------------------------------------------
# params.yaml declares the registry
# ---------------------------------------------------------------------------

def test_params_yaml_declares_a_registry_section() -> None:
    registry = load_params()["mlflow"]["registry"]
    assert registry["registered_model_name"]
    assert registry["production_alias"]
    assert registry["candidate_alias"]


@pytest.mark.parametrize("bad", [
    {"registered_model_name": ""},
    {"registered_model_name": None},
    {"production_alias": ""},
    {"candidate_alias": ""},
    {"production_alias": 42},
])
def test_invalid_registry_config_is_rejected(bad) -> None:
    params = copy.deepcopy(load_params())
    params["mlflow"]["registry"].update(bad)
    with pytest.raises(ConfigurationError, match="registry"):
        validate_params(params)


def test_the_two_aliases_must_differ() -> None:
    """One alias for both states would make an approved model indistinguishable."""
    params = copy.deepcopy(load_params())
    alias = params["mlflow"]["registry"]["production_alias"]
    params["mlflow"]["registry"]["candidate_alias"] = alias
    with pytest.raises(ConfigurationError, match="alias"):
        validate_params(params)


def test_missing_registry_section_is_rejected() -> None:
    params = copy.deepcopy(load_params())
    del params["mlflow"]["registry"]
    with pytest.raises(ConfigurationError, match="registry"):
        validate_params(params)


# ---------------------------------------------------------------------------
# The tracker can expose a run and log a model
# ---------------------------------------------------------------------------

def test_disabled_tracker_has_no_run_id() -> None:
    tracker = MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")
    with tracker.start_run():
        assert tracker.active_run_id is None


def test_disabled_tracker_logs_no_model(toy_pipeline) -> None:
    """With tracking off there is nothing to register, and that must not raise."""
    pipeline, _ = toy_pipeline
    tracker = MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")
    with tracker.start_run():
        assert tracker.log_model(pipeline) is None


def test_enabled_tracker_exposes_the_active_run_id(local_tracker) -> None:
    assert local_tracker.active_run_id is None
    with local_tracker.start_run(run_name="probe"):
        run_id = local_tracker.active_run_id
        assert isinstance(run_id, str) and len(run_id) == 32
    assert local_tracker.active_run_id is None


def test_log_model_returns_a_resolvable_uri(logged_run) -> None:
    assert logged_run["model_uri"]
    assert logged_run["run_id"]


def test_logged_model_can_be_loaded_back(local_tracker, logged_run, toy_pipeline) -> None:
    """A model URI that cannot be reloaded is not a deployable artifact."""
    import mlflow

    _, X = toy_pipeline
    mlflow.set_tracking_uri(local_tracker.resolved_uri)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = mlflow.sklearn.load_model(logged_run["model_uri"])
    assert loaded.predict_proba(X).shape == (len(X), 2)


def test_model_logging_failure_degrades_rather_than_crashes(
    local_tracker, toy_pipeline, caplog,
) -> None:
    """Losing a completed model to a logging hiccup is the worse trade."""
    import mlflow.sklearn

    pipeline, _ = toy_pipeline

    def _explode(*_args, **_kwargs):
        raise RuntimeError("artifact store unavailable")

    with local_tracker.start_run(run_name="probe"):
        original = mlflow.sklearn.log_model
        mlflow.sklearn.log_model = _explode
        try:
            assert local_tracker.log_model(pipeline) is None
        finally:
            mlflow.sklearn.log_model = original
    assert local_tracker.degraded is True


# ---------------------------------------------------------------------------
# run_information.json — the handoff between train and register
# ---------------------------------------------------------------------------

def test_run_information_carries_every_required_key(logged_run) -> None:
    assert set(RUN_INFO_REQUIRED_KEYS).issubset(logged_run)


def test_run_information_records_the_promotion_decision(logged_run) -> None:
    """The register stage must not have to re-derive whether the model passed."""
    assert logged_run["promoted"] is True
    assert logged_run["champion_model"] == "Logistic Regression"
    assert logged_run["optimal_threshold"] == pytest.approx(0.42)


def test_run_information_round_trips_through_disk(logged_run, tmp_path) -> None:
    path = tmp_path / "run_information.json"
    write_run_information(logged_run, path)
    assert load_run_information(path) == logged_run


def test_disabled_tracking_still_writes_run_information(tmp_path) -> None:
    """The train stage declares this file as a DVC output; it must always exist."""
    tracker = MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")
    with tracker.start_run():
        info = build_run_information(
            tracker, model_uri=None, registered_model_name="test-champion",
            champion_model="XGBoost", promoted=True, optimal_threshold=0.5,
            test_metrics={},
        )
    assert info["tracking_enabled"] is False
    assert info["model_uri"] is None
    assert set(RUN_INFO_REQUIRED_KEYS).issubset(info)


def test_missing_run_information_is_a_clear_error(tmp_path) -> None:
    with pytest.raises(ModelArtifactError, match="run_information"):
        load_run_information(tmp_path / "absent.json")


def test_unparseable_run_information_is_a_clear_error(tmp_path) -> None:
    path = tmp_path / "run_information.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="JSON"):
        load_run_information(path)


def test_incomplete_run_information_is_rejected(tmp_path, logged_run) -> None:
    """A half-written handoff file must fail here, not inside the registry call."""
    partial = {k: v for k, v in logged_run.items() if k != "model_uri"}
    path = tmp_path / "run_information.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="model_uri"):
        load_run_information(path)


# ---------------------------------------------------------------------------
# Registration and promotion
# ---------------------------------------------------------------------------

def _client(tracker):
    from mlflow import MlflowClient

    return MlflowClient(tracking_uri=tracker.resolved_uri)


def test_a_promoted_champion_is_registered(local_tracker, logged_run) -> None:
    result = register_champion(logged_run, production_alias="champion",
                               candidate_alias="candidate")
    assert result is not None
    versions = _client(local_tracker).search_model_versions("name='test-champion'")
    assert [v.version for v in versions] == [result.version]


def test_a_promoted_champion_takes_the_production_alias(local_tracker, logged_run) -> None:
    result = register_champion(logged_run, production_alias="champion",
                               candidate_alias="candidate")
    client = _client(local_tracker)
    promoted = client.get_model_version_by_alias("test-champion", "champion")
    assert promoted.version == result.version
    assert promoted.tags["validation_status"] == PROMOTED_STATUS


def test_the_alias_resolves_to_a_loadable_model(local_tracker, logged_run, toy_pipeline) -> None:
    """`models:/<name>@<alias>` is the URI a serving layer would use."""
    import mlflow

    _, X = toy_pipeline
    register_champion(logged_run, production_alias="champion", candidate_alias="candidate")
    mlflow.set_tracking_uri(local_tracker.resolved_uri)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        served = mlflow.sklearn.load_model("models:/test-champion@champion")
    assert served.predict_proba(X).shape == (len(X), 2)


def test_a_rejected_champion_never_takes_the_production_alias(
    local_tracker, logged_run,
) -> None:
    """The gate that blocks batch scoring must also block the registry."""
    from mlflow.exceptions import MlflowException

    rejected = {**logged_run, "promoted": False}
    result = register_champion(rejected, production_alias="champion",
                               candidate_alias="candidate")
    client = _client(local_tracker)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("test-champion", "champion")
    assert client.get_model_version_by_alias("test-champion", "candidate").version == result.version


def test_a_rejected_champion_is_still_recorded(local_tracker, logged_run) -> None:
    """A refused run is part of the history, not something to erase."""
    rejected = {**logged_run, "promoted": False}
    result = register_champion(rejected, production_alias="champion",
                               candidate_alias="candidate")
    version = _client(local_tracker).get_model_version("test-champion", result.version)
    assert version.tags["validation_status"] == CANDIDATE_STATUS


def test_registration_of_the_same_run_is_idempotent(local_tracker, logged_run) -> None:
    """DVC re-runs a stage on any dependency change; that must not fan out versions."""
    first = register_champion(logged_run, production_alias="champion",
                              candidate_alias="candidate")
    second = register_champion(logged_run, production_alias="champion",
                               candidate_alias="candidate")
    assert first.version == second.version
    versions = _client(local_tracker).search_model_versions("name='test-champion'")
    assert len(versions) == 1


def test_a_promotion_reversal_moves_the_alias_off_the_old_version(
    local_tracker, logged_run,
) -> None:
    """Re-registering the same run as rejected must not leave it aliased approved."""
    from mlflow.exceptions import MlflowException

    register_champion(logged_run, production_alias="champion", candidate_alias="candidate")
    register_champion({**logged_run, "promoted": False},
                      production_alias="champion", candidate_alias="candidate")
    client = _client(local_tracker)
    with pytest.raises(MlflowException):
        client.get_model_version_by_alias("test-champion", "champion")


def test_registration_is_skipped_when_tracking_was_disabled(tmp_path) -> None:
    """No run means nothing to register — and that is not a pipeline failure."""
    tracker = MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")
    with tracker.start_run():
        info = build_run_information(
            tracker, model_uri=None, registered_model_name="test-champion",
            champion_model="XGBoost", promoted=True, optimal_threshold=0.5,
            test_metrics={},
        )
    assert register_champion(info, production_alias="champion",
                             candidate_alias="candidate") is None


def test_registration_records_run_provenance(local_tracker, logged_run) -> None:
    """A registry entry that cannot be traced to its run is not auditable."""
    result = register_champion(logged_run, production_alias="champion",
                               candidate_alias="candidate")
    version = _client(local_tracker).get_model_version("test-champion", result.version)
    assert version.run_id == logged_run["run_id"]
    assert version.tags["champion_model"] == "Logistic Regression"
    assert float(version.tags["optimal_threshold"]) == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# registry_record.json — the stage's own record of what it did
# ---------------------------------------------------------------------------

def test_registry_record_names_the_servable_uri(local_tracker, logged_run) -> None:
    """The record must answer "what is serving" without a working MLflow install."""
    version = register_champion(logged_run, production_alias="champion",
                                candidate_alias="candidate")
    record = build_registry_record(logged_run, version, production_alias="champion",
                                   candidate_alias="candidate")
    assert record["registered"] is True
    assert record["model_uri"] == "models:/test-champion@champion"
    assert record["version"] == str(version.version)
    assert record["validation_status"] == PROMOTED_STATUS


def test_registry_record_points_a_rejected_model_at_the_candidate_alias(
    local_tracker, logged_run,
) -> None:
    rejected = {**logged_run, "promoted": False}
    version = register_champion(rejected, production_alias="champion",
                                candidate_alias="candidate")
    record = build_registry_record(rejected, version, production_alias="champion",
                                   candidate_alias="candidate")
    assert record["model_uri"] == "models:/test-champion@candidate"
    assert record["validation_status"] == CANDIDATE_STATUS
    assert record["promoted"] is False


def test_a_skip_is_recorded_with_its_reason(tmp_path) -> None:
    """DVC declares this file as the stage output, so a skip must still write one."""
    tracker = MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")
    with tracker.start_run():
        info = build_run_information(
            tracker, model_uri=None, registered_model_name="test-champion",
            champion_model="XGBoost", promoted=True, optimal_threshold=0.5,
            test_metrics={},
        )
    record = build_registry_record(info, None, production_alias="champion",
                                   candidate_alias="candidate")
    assert record["registered"] is False
    assert "tracking was disabled" in record["skipped_reason"]


def test_registry_record_round_trips_through_disk(tmp_path, local_tracker, logged_run) -> None:
    version = register_champion(logged_run, production_alias="champion",
                                candidate_alias="candidate")
    record = build_registry_record(logged_run, version, production_alias="champion",
                                   candidate_alias="candidate")
    path = tmp_path / "nested" / "registry_record.json"
    write_registry_record(record, path)
    assert json.loads(path.read_text(encoding="utf-8")) == record


# ---------------------------------------------------------------------------
# Wiring: the stage exists, is declared, and does not read the model back in
# ---------------------------------------------------------------------------

def test_register_stage_entry_point_exists() -> None:
    assert (PROJECT_ROOT / "scripts" / "register_model.py").exists()


def test_train_stage_declares_run_information_as_an_output() -> None:
    import yaml

    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as handle:
        stages = yaml.safe_load(handle)["stages"]
    outs = [next(iter(o)) if isinstance(o, dict) else o for o in stages["train"]["outs"]]
    assert "artifacts/run_information.json" in outs


def test_register_stage_is_declared_and_depends_on_the_handoff() -> None:
    import yaml

    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as handle:
        stages = yaml.safe_load(handle)["stages"]
    assert "register" in stages
    deps = stages["register"]["deps"]
    assert "artifacts/run_information.json" in deps
    assert "scripts/register_model.py" in deps
    assert "src/models/register_model.py" in deps

    outs = [next(iter(o)) if isinstance(o, dict) else o for o in stages["register"]["outs"]]
    assert outs == ["artifacts/registry_record.json"]


def test_register_stage_does_not_gate_batch_inference() -> None:
    """Scoring reads the local champion artifact; the registry must not sit in that path."""
    import yaml

    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as handle:
        stages = yaml.safe_load(handle)["stages"]
    predict_deps = " ".join(stages["predict"]["deps"])
    assert "register_model" not in predict_deps


def test_registry_config_is_not_a_dvc_train_dependency() -> None:
    """Renaming a registered model must not invalidate a multi-hour training run."""
    import yaml

    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as handle:
        stages = yaml.safe_load(handle)["stages"]
    assert "mlflow" not in stages["train"].get("params", [])
