"""Step 7 — MLflow experiment tracking.

Tracking is observability, so the contracts worth pinning are about what it must *not*
do to training:

*Disabled must be genuinely inert.* Every method is a no-op and the run context managers
still yield, so `scripts/train.py` needs no `if tracker:` guards.

*Enabled must actually work.* Initialisation failure raises — a run that silently
discards four hours of metrics is worse than one that refuses to start.

*A logging failure must not destroy a completed model.* Per-call failures warn loudly and
set `degraded`; nothing is swallowed by a bare `except`.

No model is fitted here. A real local `mlruns` directory under `tmp_path` covers the
integration path; a stub captures calls where the assertion is about *what* gets logged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.mlflow_tracker import (
    PROJECT_NAME,
    MLflowTracker,
    MLflowTrackingError,
    get_git_commit,
    normalise_metric_name,
)
from src.utils.config_loader import TRACKING_SECTIONS, load_params
from src.utils.exceptions import ConfigurationError, ModeliumError


class _StubMlflow:
    """Captures what would have been logged, without an MLflow backend."""

    def __init__(self):
        self.params, self.metrics, self.tags, self.artifacts = {}, {}, {}, []
        self.runs, self.open_runs = [], 0

    # --- run management -----------------------------------------------------
    def start_run(self, run_name=None, nested=False):
        self.runs.append({"run_name": run_name, "nested": nested})
        stub = self

        class _Run:
            def __enter__(self_inner):
                stub.open_runs += 1
                return self_inner

            def __exit__(self_inner, *exc):
                stub.open_runs -= 1
                return False

        return _Run()

    # --- logging ------------------------------------------------------------
    def log_params(self, payload):
        self.params.update(payload)

    def log_metrics(self, payload):
        self.metrics.update(payload)

    def set_tags(self, payload):
        self.tags.update(payload)

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


@pytest.fixture
def stub_tracker():
    """An enabled tracker whose MLflow calls are captured rather than executed."""
    tracker = MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")
    tracker.enabled = True                       # bypass real initialisation
    tracker._mlflow = _StubMlflow()
    return tracker, tracker._mlflow


@pytest.fixture
def disabled():
    return MLflowTracker(enabled=False, experiment_name="x", tracking_uri="mlruns")


# --------------------------------------------------------------- configuration

def test_params_yaml_carries_an_mlflow_section() -> None:
    section = load_params()["mlflow"]
    assert set(section) >= {"enabled", "experiment_name", "tracking_uri"}


def test_experiment_name_is_configured() -> None:
    assert load_params()["mlflow"]["experiment_name"] == "modelium-credit-risk"


def test_tracker_builds_from_params(tmp_path) -> None:
    params = {"mlflow": {"enabled": True, "experiment_name": "from-params",
                         "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"}}
    tracker = MLflowTracker.from_params(params)
    assert tracker.enabled and tracker.experiment_name == "from-params"


def test_from_params_defaults_to_disabled_when_absent() -> None:
    """Tracking is opt-in; a params file without the section must not switch it on."""
    assert MLflowTracker.from_params({}).enabled is False


@pytest.mark.parametrize("bad", [{"enabled": "false"}, {"enabled": 1},
                                 {"experiment_name": ""}, {"tracking_uri": ""},
                                 {"experiment_name": None}])
def test_invalid_mlflow_config_is_rejected(bad) -> None:
    """`"false"` is truthy in Python — coercing it would switch tracking on silently."""
    import copy

    from src.utils.config_loader import validate_params

    params = copy.deepcopy(load_params())
    params["mlflow"].update(bad)
    with pytest.raises(ConfigurationError, match="mlflow"):
        validate_params(params)


def test_tracking_config_is_not_a_dvc_train_dependency() -> None:
    """Renaming an experiment must not invalidate a multi-hour training run."""
    import yaml

    with open(PROJECT_ROOT / "dvc.yaml", encoding="utf-8") as handle:
        stages = yaml.safe_load(handle)["stages"]
    assert set(TRACKING_SECTIONS).isdisjoint(stages["train"].get("params", []))


def test_relative_sqlite_uri_resolves_against_the_repo_root() -> None:
    """Otherwise the database lands wherever the process started, splitting history."""
    resolved = MLflowTracker._resolve_uri("sqlite:///mlflow.db")
    assert resolved == f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def test_absolute_sqlite_uri_is_left_alone() -> None:
    assert MLflowTracker._resolve_uri("sqlite:////tmp/x.db") == "sqlite:////tmp/x.db"


def test_relative_path_uri_resolves_against_the_repo_root() -> None:
    resolved = MLflowTracker._resolve_uri("mlruns")
    assert Path(resolved) == PROJECT_ROOT / "mlruns"


def test_remote_tracking_uri_is_left_alone() -> None:
    assert MLflowTracker._resolve_uri("http://localhost:5000") == "http://localhost:5000"


def test_configured_backend_is_a_database_not_the_retired_file_store() -> None:
    """MLflow 3 refuses "./mlruns" by default; configuring it would fail every run."""
    assert load_params()["mlflow"]["tracking_uri"].startswith("sqlite:///")


# ------------------------------------------------------------------- disabled

def test_disabled_tracker_logs_nothing(disabled) -> None:
    assert disabled.enabled is False
    disabled.log_params({"a": 1})
    disabled.log_metrics({"b": 2.0})
    disabled.set_tags({"c": "d"})
    disabled.log_artifact(PROJECT_ROOT / "params.yaml")
    assert disabled.degraded is False


def test_disabled_run_context_still_yields(disabled) -> None:
    """Training must not need `if tracker:` guards around its own body."""
    ran = False
    with disabled.start_run("run") as tracker:
        with tracker.child_run("candidate"):
            ran = True
    assert ran


def test_disabled_tracker_never_imports_a_backend(disabled) -> None:
    assert disabled._mlflow is None


# ---------------------------------------------------------------- log content

def test_parameters_are_logged(stub_tracker) -> None:
    tracker, stub = stub_tracker
    tracker.log_params({"n_iter": 20, "scoring": "average_precision"})
    assert stub.params == {"n_iter": "20", "scoring": "average_precision"}


def test_metrics_are_logged_with_normalised_names(stub_tracker) -> None:
    tracker, stub = stub_tracker
    tracker.log_metrics({"Average Precision": 0.28, "ROC-AUC": 0.78})
    assert stub.metrics == {"average_precision": 0.28, "roc_auc": 0.78}


def test_metric_prefixes_separate_the_splits(stub_tracker) -> None:
    tracker, stub = stub_tracker
    tracker.log_metrics({"Average Precision": 0.28}, prefix="val_")
    tracker.log_metrics({"Average Precision": 0.27}, prefix="test_")
    assert stub.metrics == {"val_average_precision": 0.28, "test_average_precision": 0.27}


def test_non_numeric_metrics_are_skipped_not_coerced(stub_tracker) -> None:
    """Evaluation dicts carry a "Model" name; it is not a metric."""
    tracker, stub = stub_tracker
    tracker.log_metrics({"Model": "XGBoost (Tuned)", "F1": 0.33, "promoted": True})
    assert stub.metrics == {"f1": 0.33}


def test_tags_are_logged(stub_tracker) -> None:
    tracker, stub = stub_tracker
    tracker.set_tags({"project_name": PROJECT_NAME, "promoted": "true"})
    assert stub.tags == {"project_name": "Modelium", "promoted": "true"}


def test_artifacts_are_logged(stub_tracker, tmp_path) -> None:
    tracker, stub = stub_tracker
    artifact = tmp_path / "leaderboard.csv"
    artifact.write_text("Model,Average Precision\nXGBoost,0.28\n")
    tracker.log_artifact(artifact, artifact_path="metrics")
    assert stub.artifacts == [(str(artifact), "metrics")]


def test_missing_artifact_is_warned_not_raised(stub_tracker, tmp_path, caplog) -> None:
    """An absent artifact must not fail a completed training run."""
    tracker, stub = stub_tracker
    tracker.log_artifact(tmp_path / "absent.csv")
    assert stub.artifacts == []


def test_log_artifacts_skips_absent_files(stub_tracker, tmp_path) -> None:
    tracker, stub = stub_tracker
    present = tmp_path / "there.json"
    present.write_text("{}")
    tracker.log_artifacts([present, tmp_path / "gone.json"])
    assert len(stub.artifacts) == 1


def test_long_param_values_are_truncated(stub_tracker) -> None:
    """MLflow rejects oversized param values; a 407-column list is not a param."""
    tracker, stub = stub_tracker
    tracker.log_params({"columns": "col," * 500})
    assert len(stub.params["columns"]) <= 250


@pytest.mark.parametrize("raw,expected", [
    ("Average Precision", "average_precision"), ("ROC-AUC", "roc_auc"),
    ("F1", "f1"), ("TN", "tn"), ("Threshold", "threshold"),
])
def test_metric_name_normalisation(raw, expected) -> None:
    assert normalise_metric_name(raw) == expected


# ------------------------------------------------------------- no mutation

def test_input_metric_dict_is_not_mutated(stub_tracker) -> None:
    tracker, _ = stub_tracker
    metrics = {"Model": "XGBoost", "Average Precision": 0.28}
    snapshot = dict(metrics)
    tracker.log_metrics(metrics, prefix="val_")
    assert metrics == snapshot


def test_input_param_dict_is_not_mutated(stub_tracker) -> None:
    tracker, _ = stub_tracker
    params = {"n_iter": 20}
    snapshot = dict(params)
    tracker.log_params(params, prefix="cfg_")
    assert params == snapshot


def test_input_tag_dict_is_not_mutated(stub_tracker) -> None:
    tracker, _ = stub_tracker
    tags = {"promoted": True}
    snapshot = dict(tags)
    tracker.set_tags(tags)
    assert tags == snapshot


# ---------------------------------------------------------------- nested runs

def test_parent_and_nested_runs_open_and_close(stub_tracker) -> None:
    tracker, stub = stub_tracker
    with tracker.start_run("parent"):
        with tracker.child_run("XGBoost (Tuned)"):
            pass
        with tracker.child_run("LightGBM (Tuned)"):
            pass

    assert stub.open_runs == 0, "a run was left open"
    assert stub.runs[0] == {"run_name": "parent", "nested": False}
    assert [r["run_name"] for r in stub.runs[1:]] == ["XGBoost (Tuned)", "LightGBM (Tuned)"]
    assert all(r["nested"] for r in stub.runs[1:])


def test_runs_close_even_when_the_body_raises(stub_tracker) -> None:
    """A failed training run must not leave a dangling MLflow run behind."""
    tracker, stub = stub_tracker
    with pytest.raises(ValueError):
        with tracker.start_run("parent"):
            raise ValueError("training blew up")
    assert stub.open_runs == 0


# ------------------------------------------------------------- error handling

def test_enabled_tracker_raises_when_initialisation_fails(monkeypatch) -> None:
    """Enabled-but-broken must be loud: silently dropping metrics is the worse failure."""
    import mlflow

    def _explode(*_a, **_k):
        raise RuntimeError("cannot reach tracking store")

    monkeypatch.setattr(mlflow, "set_experiment", _explode)
    with pytest.raises(MLflowTrackingError, match="could not be initialised"):
        MLflowTracker(enabled=True, experiment_name="x", tracking_uri="mlruns")


def test_initialisation_error_is_a_modelium_error() -> None:
    assert issubclass(MLflowTrackingError, ModeliumError)


def test_a_logging_failure_degrades_rather_than_crashes(stub_tracker, caplog) -> None:
    """Losing a completed model to a metric-logging hiccup is the worse trade — but it
    is surfaced, not swallowed."""
    tracker, stub = stub_tracker

    def _explode(_payload):
        raise RuntimeError("backend unavailable")

    stub.log_metrics = _explode
    with caplog.at_level("WARNING"):
        tracker.log_metrics({"Average Precision": 0.28})

    assert tracker.degraded is True
    assert any("failed to log" in r.message for r in caplog.records)


def test_a_healthy_tracker_is_not_degraded(stub_tracker) -> None:
    tracker, _ = stub_tracker
    tracker.log_metrics({"F1": 0.3})
    assert tracker.degraded is False


# ----------------------------------------------------------------- provenance

def test_git_commit_is_captured() -> None:
    provenance = get_git_commit()
    assert set(provenance) == {"git_commit", "git_dirty"}
    assert len(provenance["git_commit"]) == 40
    assert provenance["git_dirty"] in {"true", "false"}


def test_git_commit_failure_is_not_fatal(monkeypatch) -> None:
    """A run outside version control is still a run worth recording."""
    import subprocess

    def _explode(*_a, **_k):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", _explode)
    assert get_git_commit() == {}


# ------------------------------------------- integration against a real local store

def test_real_local_tracking_round_trip(tmp_path) -> None:
    """Exercise the genuine MLflow path once — no model, just a run with a metric."""
    import mlflow

    tracker = MLflowTracker(enabled=True, experiment_name="modelium-test",
                            tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    with tracker.start_run("unit-test-run", tags={"project_name": PROJECT_NAME}):
        tracker.log_params({"n_iter": 3})
        tracker.log_metrics({"Average Precision": 0.2803}, prefix="val_")
        with tracker.child_run("XGBoost (Tuned)", tags={"is_champion": "true"}):
            tracker.log_metrics({"cv_average_precision": 0.2642})

    assert not tracker.degraded
    runs = mlflow.search_runs(experiment_names=["modelium-test"])
    assert len(runs) == 2                                   # parent + one nested
    parent = runs[runs["tags.mlflow.runName"] == "unit-test-run"].iloc[0]
    assert parent["metrics.val_average_precision"] == pytest.approx(0.2803)
    assert parent["params.n_iter"] == "3"
    assert parent["tags.project_name"] == PROJECT_NAME


def test_training_script_imports_the_tracker() -> None:
    """Guard against a refactor quietly dropping tracking from the training stage."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mlflow_train_script", PROJECT_ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "MLflowTracker")
