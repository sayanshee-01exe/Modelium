"""Step 9 — the monitor stage's wiring, configuration and DVC declaration.

`scripts/monitor.py` is orchestration; the measurements are tested in the four module
suites. What matters here is that the stage cannot quietly do something it must not:

*It must never fit.* Monitoring exists to observe the champion, and refitting anything
would make the report describe a model that was never deployed.

*It must never act.* No retraining, no registration, no alias move, no promotion or
rollback. A monitoring stage that can change what is served turns an alert into an
outage.

*It must not invalidate training.* Adding a constant to `config/config.py` once started a
three-hour retrain, so the monitoring paths and settings validation live outside every
file the train stage depends on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import load_params
from src.utils.exceptions import ConfigurationError

DVC_FILE = PROJECT_ROOT / "dvc.yaml"
MONITOR_SCRIPT = PROJECT_ROOT / "scripts" / "monitor.py"
MONITORING_PACKAGE = PROJECT_ROOT / "src" / "monitoring"


@pytest.fixture(scope="module")
def stages() -> dict:
    with open(DVC_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["stages"]


@pytest.fixture(scope="module")
def monitor_module():
    spec = importlib.util.spec_from_file_location("dvc_stage_monitor", MONITOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(entries) -> set[str]:
    return {next(iter(e)) if isinstance(e, dict) else e for e in entries or []}


# ---------------------------------------------------------------------------
# The script imports and stays orchestration-only
# ---------------------------------------------------------------------------

def test_the_stage_script_imports_without_side_effects(monitor_module) -> None:
    """Importing must not load a model, read a batch, or compute anything."""
    assert callable(monitor_module.main)


def test_the_batch_builder_imports_without_side_effects() -> None:
    spec = importlib.util.spec_from_file_location(
        "probe_batch", PROJECT_ROOT / "scripts" / "create_monitoring_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_the_script_delegates_measurement_to_the_package() -> None:
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "from src.monitoring.data_drift import" in source
    assert "from src.monitoring.fairness_monitor import" in source
    # Statistics belong in the modules, not the orchestrator.
    assert "ks_2samp" not in source
    assert "def compute_psi" not in source


# ---------------------------------------------------------------------------
# Inference-only, and never acts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "scripts/monitor.py",
    "src/monitoring/data_drift.py",
    "src/monitoring/prediction_drift.py",
    "src/monitoring/performance_monitor.py",
    "src/monitoring/fairness_monitor.py",
    "src/monitoring/reporting.py",
])
def test_monitoring_never_fits(path) -> None:
    source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    for forbidden in (".fit(", ".fit_transform(", "RandomizedSearchCV"):
        assert forbidden not in source, f"{path} must not call {forbidden}"


@pytest.mark.parametrize("forbidden", [
    "register_model(", "set_registered_model_alias", "delete_registered_model_alias",
    "transition_model_version_stage",
])
def test_monitoring_never_changes_the_registry(forbidden) -> None:
    """An alert must never be able to change what is served."""
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert forbidden not in source


def test_monitoring_does_not_invoke_training() -> None:
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "scripts/train.py" not in source
    assert "train.py" not in source


def test_the_script_resolves_the_champion_through_the_registry() -> None:
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "load_champion_from_registry" in source
    assert "champion_alias" in source
    assert "champion_pipeline.joblib" not in source


def test_the_script_uses_the_frozen_threshold_not_a_default() -> None:
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert 'deployment["optimal_threshold"]' in source
    assert "threshold = 0.5" not in source


def test_the_script_refuses_to_compare_a_dataset_with_itself() -> None:
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "reference.equals(current)" in source


def test_the_identifier_is_not_a_monitored_feature() -> None:
    source = MONITOR_SCRIPT.read_text(encoding="utf-8")
    assert "drop(columns=[TARGET_COL, ID_COL]" in source


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------

def test_the_real_params_file_validates(monitor_module) -> None:
    settings = monitor_module.validate_settings(load_params()["monitoring"])
    assert settings["enabled"] is True


@pytest.mark.parametrize("key", [
    "enabled", "reference_sample_size", "current_sample_size", "random_state",
    "drift", "prediction", "performance", "fairness",
])
def test_a_missing_setting_is_rejected(monitor_module, key) -> None:
    settings = {k: v for k, v in load_params()["monitoring"].items() if k != key}
    with pytest.raises(ConfigurationError, match="monitoring"):
        monitor_module.validate_settings(settings)


def test_a_string_enabled_flag_is_rejected(monitor_module) -> None:
    settings = {**load_params()["monitoring"], "enabled": "false"}
    with pytest.raises(ConfigurationError, match="enabled"):
        monitor_module.validate_settings(settings)


@pytest.mark.parametrize("key", ["reference_sample_size", "current_sample_size"])
def test_a_non_positive_sample_size_is_rejected(monitor_module, key) -> None:
    settings = {**load_params()["monitoring"], key: 0}
    with pytest.raises(ConfigurationError, match=key):
        monitor_module.validate_settings(settings)


def test_a_non_mapping_section_is_rejected(monitor_module) -> None:
    with pytest.raises(ConfigurationError, match="monitoring"):
        monitor_module.validate_settings(None)


def test_params_declares_every_documented_threshold() -> None:
    monitoring = load_params()["monitoring"]
    assert set(monitoring["drift"]) == {
        "psi_warning_threshold", "psi_critical_threshold", "ks_pvalue_threshold",
        "max_drifted_feature_ratio"}
    assert set(monitoring["prediction"]) == {
        "mean_probability_change_threshold", "positive_rate_change_threshold"}
    assert set(monitoring["performance"]) == {
        "min_average_precision", "min_roc_auc", "min_recall", "min_precision", "min_f1"}
    assert set(monitoring["fairness"]) == {
        "enabled", "minimum_group_size", "max_demographic_parity_difference",
        "max_equal_opportunity_difference"}


def test_the_performance_floors_match_the_promotion_gates() -> None:
    """Monitoring must not invent stricter production limits than promotion applied."""
    params = load_params()
    assert params["monitoring"]["performance"]["min_average_precision"] == \
        params["selection"]["min_average_precision"]
    assert params["monitoring"]["performance"]["min_roc_auc"] == \
        params["selection"]["min_roc_auc"]


# ---------------------------------------------------------------------------
# DVC stage declaration
# ---------------------------------------------------------------------------

def test_the_monitor_stage_is_declared(stages) -> None:
    assert "monitor" in stages
    assert "scripts/monitor.py" in stages["monitor"]["cmd"]


def test_the_stage_depends_on_the_registry_record_and_both_batches(stages) -> None:
    deps = _paths(stages["monitor"]["deps"])
    assert "artifacts/registry_record.json" in deps
    assert "data/processed/train_features.parquet" in deps
    assert "data/monitoring/current_batch.parquet" in deps
    assert "artifacts/deployment_meta.json" in deps
    assert not any(dep.endswith(".db") for dep in deps)


def test_the_stage_depends_on_every_monitoring_module(stages) -> None:
    """Omitting one would let DVC serve a stale report after that module changed."""
    deps = _paths(stages["monitor"]["deps"])
    for module in ("data_drift", "prediction_drift", "performance_monitor",
                   "fairness_monitor", "reporting"):
        assert f"src/monitoring/{module}.py" in deps, module


def test_the_stage_declares_the_params_it_reads(stages) -> None:
    assert set(stages["monitor"].get("params", [])) == {"monitoring", "data", "mlflow"}


def test_the_stage_declares_every_documented_output(stages) -> None:
    outs = _paths(stages["monitor"]["outs"])
    for expected in ("artifacts/monitoring/feature_drift.csv",
                     "artifacts/monitoring/prediction_drift.json",
                     "artifacts/monitoring/performance_metrics.json",
                     "artifacts/monitoring/fairness_metrics.csv",
                     "artifacts/monitoring/monitoring_summary.json",
                     "artifacts/monitoring/monitoring_report.md"):
        assert expected in outs, expected


def test_plots_go_in_their_own_directory(stages) -> None:
    """Never reports/figures itself: DVC prunes what it did not write in a directory it
    owns, and that directory carries a git-tracked .gitkeep."""
    outs = _paths(stages["monitor"]["outs"])
    assert "reports/figures/monitoring" in outs
    assert "reports/figures" not in outs


def test_monitor_produces_no_model(stages) -> None:
    assert not any(out.endswith(".joblib") for out in _paths(stages["monitor"]["outs"]))


def test_monitor_is_a_leaf(stages) -> None:
    """Nothing may depend on a monitoring result — an alert must not gate a score."""
    monitor_outs = _paths(stages["monitor"]["outs"])
    for name, stage in stages.items():
        if name == "monitor":
            continue
        assert monitor_outs.isdisjoint(_paths(stage.get("deps"))), \
            f"{name} depends on a monitor output"


def test_no_output_is_claimed_by_two_stages(stages) -> None:
    seen: dict[str, str] = {}
    for name, stage in stages.items():
        for path in _paths(stage.get("outs")) | _paths(stage.get("metrics")):
            assert path not in seen, f"{path} is written by both {seen[path]} and {name}"
            seen[path] = name


# ---------------------------------------------------------------------------
# Monitoring must not invalidate training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["src/utils/config_loader.py", "config/config.py"])
def test_monitoring_config_lives_outside_the_train_dependencies(stages, module) -> None:
    assert module in _paths(stages["train"]["deps"])
    source = (PROJECT_ROOT / module).read_text(encoding="utf-8").lower()
    assert "monitoring" not in source, (
        f"{module} is a train dependency and mentions monitoring; move it to the "
        f"monitoring package or scripts/monitor.py"
    )


def test_the_monitoring_section_is_not_a_train_dependency(stages) -> None:
    """Changing a drift threshold must not invalidate a multi-hour training run."""
    assert "monitoring" not in stages["train"].get("params", [])


def test_no_monitoring_module_is_a_train_dependency(stages) -> None:
    train_deps = _paths(stages["train"]["deps"])
    assert not any(dep.startswith("src/monitoring/") for dep in train_deps)


# ---------------------------------------------------------------------------
# The demonstration batch is labelled as such
# ---------------------------------------------------------------------------

def test_the_batch_builder_marks_its_output_as_a_demonstration() -> None:
    source = (PROJECT_ROOT / "scripts" / "create_monitoring_batch.py").read_text(
        encoding="utf-8")
    assert '"demonstration": True' in source
    assert "labels_are_observed_production_outcomes" in source


def test_the_batch_builder_records_which_features_it_perturbed() -> None:
    """Simulated drift must be separable from anything real."""
    source = (PROJECT_ROOT / "scripts" / "create_monitoring_batch.py").read_text(
        encoding="utf-8")
    assert "simulated_drift_features" in source


def test_the_batch_builder_never_writes_to_the_prepared_datasets() -> None:
    source = (PROJECT_ROOT / "scripts" / "create_monitoring_batch.py").read_text(
        encoding="utf-8")
    assert "to_parquet(CURRENT_BATCH_FILE" in source
    assert "to_parquet(TRAIN_FEATURES_FILE" not in source
