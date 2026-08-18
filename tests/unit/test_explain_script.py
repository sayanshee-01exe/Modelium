"""Step 8 — the explain stage's wiring, configuration and DVC declaration.

`scripts/explain.py` is orchestration: it decides what happens in what order and where
the results land. The SHAP mathematics is tested in `test_shap_explainer.py`; what
matters here is that the stage is wired so it cannot quietly do the wrong thing.

Three wiring mistakes would each produce a plausible-looking report:

*Explaining training rows and calling them holdout.* The sample must come from the same
test split the train stage held out, rebuilt with the same sizes, seed and stratification.

*Explaining a model nobody serves.* The champion must be resolved through the registry
alias, not read from whichever joblib happens to be on disk.

*Invalidating training to add an explanation.* The explain stage must not depend on files
the train stage depends on having *changed* — adding a constant to `config/config.py`
once started a three-hour retrain, so the paths and the settings validation deliberately
live outside it.

The full Home Credit data is never loaded here.
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

from src.explainability.shap_explainer import validate_explainability_settings
from src.utils.config_loader import load_params
from src.utils.exceptions import ConfigurationError

DVC_FILE = PROJECT_ROOT / "dvc.yaml"


@pytest.fixture(scope="module")
def stages() -> dict:
    with open(DVC_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["stages"]


def _paths(entries) -> set[str]:
    return {next(iter(e)) if isinstance(e, dict) else e for e in entries or []}


# ---------------------------------------------------------------------------
# The script imports and exposes a stage entry point
# ---------------------------------------------------------------------------

def test_the_stage_script_imports_without_side_effects() -> None:
    """Importing must not load data, load a model or compute SHAP."""
    spec = importlib.util.spec_from_file_location(
        "dvc_stage_explain", PROJECT_ROOT / "scripts" / "explain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_the_script_stays_orchestration_only() -> None:
    """SHAP implementation belongs in the module, so the stage can be reasoned about."""
    source = (PROJECT_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    # The script must not touch the shap library at all — calling the module's
    # compute_shap_values() is the orchestration this test wants to see.
    assert "import shap" not in source
    assert "shap.TreeExplainer" not in source
    assert "explainer.shap_values(" not in source
    assert "from src.explainability.shap_explainer import" in source
    assert "compute_shap_values(" in source


def test_the_script_never_fits_anything() -> None:
    """Explanation is inference-only; a fit call here would describe a different model."""
    source = (PROJECT_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    for forbidden in (".fit(", ".fit_transform(", "RandomizedSearchCV"):
        assert forbidden not in source, f"explain.py must not call {forbidden}"


def test_the_explainer_module_never_fits_anything() -> None:
    source = (PROJECT_ROOT / "src" / "explainability" / "shap_explainer.py").read_text(
        encoding="utf-8")
    for forbidden in (".fit(", ".fit_transform("):
        assert forbidden not in source, f"shap_explainer.py must not call {forbidden}"


def test_the_script_resolves_the_champion_through_the_registry() -> None:
    """Not from models/champion_pipeline.joblib, which is whatever ran last."""
    source = (PROJECT_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    assert "load_champion_from_registry" in source
    assert "champion_alias" in source
    assert "champion_pipeline.joblib" not in source


def test_the_script_explains_the_test_split() -> None:
    """Training rows presented as holdout explanations would be optimistically biased."""
    source = (PROJECT_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    assert "split_train_val_test" in source
    assert '"explained_split": "test"' in source


def test_the_identifier_is_not_a_model_feature() -> None:
    """SK_ID_CURR labels an explanation; explaining it as a feature would be nonsense."""
    source = (PROJECT_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    assert "drop(columns=[TARGET_COL, ID_COL]" in source


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------

def test_the_real_params_file_validates() -> None:
    settings = validate_explainability_settings(load_params()["explainability"])
    assert settings["enabled"] is True


@pytest.mark.parametrize("key", [
    "sample_size", "background_size", "max_display", "local_examples",
])
def test_a_non_positive_size_is_rejected(key) -> None:
    settings = dict(load_params()["explainability"])
    settings[key] = 0
    with pytest.raises(ConfigurationError, match=key):
        validate_explainability_settings(settings)


@pytest.mark.parametrize("key", [
    "sample_size", "background_size", "max_display", "local_examples", "random_state",
])
def test_a_missing_key_is_rejected(key) -> None:
    settings = {k: v for k, v in load_params()["explainability"].items() if k != key}
    with pytest.raises(ConfigurationError, match=key):
        validate_explainability_settings(settings)


def test_a_string_enabled_flag_is_rejected() -> None:
    """The string "false" is truthy and would silently switch explanation on."""
    settings = dict(load_params()["explainability"])
    settings["enabled"] = "false"
    with pytest.raises(ConfigurationError, match="enabled"):
        validate_explainability_settings(settings)


def test_a_negative_seed_is_rejected() -> None:
    settings = dict(load_params()["explainability"])
    settings["random_state"] = -1
    with pytest.raises(ConfigurationError, match="random_state"):
        validate_explainability_settings(settings)


def test_a_non_mapping_section_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="explainability"):
        validate_explainability_settings(None)


def test_params_yaml_declares_every_documented_key() -> None:
    settings = load_params()["explainability"]
    for key in ("enabled", "sample_size", "background_size", "random_state",
                "max_display", "local_examples"):
        assert key in settings, f"params.yaml: explainability.{key} is missing"


def test_the_sample_is_bounded_well_below_the_full_split() -> None:
    """46k rows x 537 features of float64 SHAP values is ~200 MB before plotting."""
    assert load_params()["explainability"]["sample_size"] <= 10_000


# ---------------------------------------------------------------------------
# DVC stage declaration
# ---------------------------------------------------------------------------

def test_the_explain_stage_is_declared(stages) -> None:
    assert "explain" in stages
    assert "scripts/explain.py" in stages["explain"]["cmd"]


def test_the_stage_depends_on_the_registry_record(stages) -> None:
    """This is what orders explain after register without hashing a large binary db."""
    deps = _paths(stages["explain"]["deps"])
    assert "artifacts/registry_record.json" in deps
    assert "src/explainability/shap_explainer.py" in deps
    assert not any(dep.endswith(".db") for dep in deps)


def test_the_stage_declares_the_params_sections_it_reads(stages) -> None:
    declared = set(stages["explain"].get("params", []))
    assert {"explainability", "data", "mlflow"} == declared


def test_the_stage_declares_all_four_machine_readable_outputs(stages) -> None:
    outs = _paths(stages["explain"]["outs"])
    for expected in ("artifacts/explainability/global_feature_importance.csv",
                     "artifacts/explainability/local_explanations.json",
                     "artifacts/explainability/explanation_report.json"):
        assert expected in outs, expected


def test_the_stage_declares_both_global_plots(stages) -> None:
    outs = _paths(stages["explain"]["outs"])
    assert "reports/figures/shap_summary.png" in outs
    assert "reports/figures/shap_bar.png" in outs


def test_the_stage_does_not_claim_the_whole_figures_directory(stages) -> None:
    """DVC deletes what it did not write inside a directory it owns, and
    reports/figures/.gitkeep is tracked by git. Claiming the parent removed it."""
    assert "reports/figures" not in _paths(stages["explain"]["outs"])


def test_local_plots_are_declared_so_stale_ones_are_cleaned(stages) -> None:
    """Filenames carry SK_ID_CURR, so a changed sample would otherwise leave the
    previous run's plots behind to be mistaken for current ones."""
    assert "reports/figures/shap_local" in _paths(stages["explain"]["outs"])


def test_explain_does_not_produce_a_model(stages) -> None:
    outs = _paths(stages["explain"]["outs"])
    assert not any(out.endswith(".joblib") for out in outs)


def test_explain_is_not_upstream_of_training_or_scoring(stages) -> None:
    """An explanation must never be a prerequisite for producing a model or a score."""
    explain_outs = _paths(stages["explain"]["outs"])
    for stage in ("train", "predict", "register", "prepare", "validate"):
        assert explain_outs.isdisjoint(_paths(stages[stage]["deps"])), \
            f"{stage} depends on an explain output"


def test_no_output_is_claimed_by_two_stages(stages) -> None:
    seen: dict[str, str] = {}
    for name, stage in stages.items():
        for path in _paths(stage.get("outs")) | _paths(stage.get("metrics")):
            assert path not in seen, f"{path} is written by both {seen[path]} and {name}"
            seen[path] = name


# ---------------------------------------------------------------------------
# Explaining must not invalidate training
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "src/utils/config_loader.py",
    "config/config.py",
])
def test_explainability_config_lives_outside_the_train_dependencies(stages, module) -> None:
    """These are train deps. Editing one to add an explainability constant or check
    invalidates a multi-hour training run, so neither may mention explainability."""
    assert module in _paths(stages["train"]["deps"])
    source = (PROJECT_ROOT / module).read_text(encoding="utf-8").lower()
    for token in ("explainability", "shap"):
        assert token not in source, (
            f"{module} is a train dependency and mentions {token!r}; move it to the "
            f"explainability module or the explain script"
        )


def test_the_explainability_section_is_not_a_train_dependency(stages) -> None:
    """Changing the sample size must not invalidate training."""
    assert "explainability" not in stages["train"].get("params", [])
