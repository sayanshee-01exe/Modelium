"""Step 6 — the DVC pipeline definition is correct and its stage scripts import.

`dvc.yaml` is not executed here: reproducing it needs 2.5 GB of raw data and hours of
tuning. What *is* checkable cheaply is everything that makes `dvc repro` behave —
whether the DAG is wired in the right order, whether each stage declares the inputs
that should invalidate it, and whether every script it names can actually be imported.

The dependency assertions matter more than they look. A stage that omits a dep silently
serves a stale cache: edit `tune.py` with `train` missing that dependency and DVC
reports "everything is up to date" while the champion on disk was built by the old code.
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

DVC_FILE = PROJECT_ROOT / "dvc.yaml"
STAGE_ORDER = ("validate", "prepare", "train", "predict")


@pytest.fixture(scope="module")
def pipeline() -> dict:
    with open(DVC_FILE, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def stages(pipeline) -> dict:
    return pipeline["stages"]


def _paths(entries) -> set[str]:
    """DVC entries are either a bare path or a single-key mapping with options."""
    out = set()
    for entry in entries or []:
        out.add(next(iter(entry)) if isinstance(entry, dict) else entry)
    return out


# ------------------------------------------------------------------- file & shape

def test_dvc_yaml_exists_and_parses(pipeline) -> None:
    assert isinstance(pipeline, dict) and "stages" in pipeline


def test_all_four_stages_are_defined(stages) -> None:
    assert set(stages) == set(STAGE_ORDER)


@pytest.mark.parametrize("stage", STAGE_ORDER)
def test_every_stage_has_a_command_and_dependencies(stages, stage) -> None:
    assert stages[stage]["cmd"].strip()
    assert stages[stage]["deps"], f"{stage} declares no dependencies"


@pytest.mark.parametrize("stage", STAGE_ORDER)
def test_no_stage_is_a_no_op(stages, stage) -> None:
    """A stage with no outputs or metrics produces nothing and re-runs every time."""
    assert _paths(stages[stage].get("outs")) or _paths(stages[stage].get("metrics"))


@pytest.mark.parametrize("stage", STAGE_ORDER)
def test_stage_scripts_exist(stages, stage) -> None:
    script = stages[stage]["cmd"].split()[-1]
    assert (PROJECT_ROOT / script).exists(), f"{stage} runs a missing script: {script}"


@pytest.mark.parametrize("stage", STAGE_ORDER)
def test_every_declared_dependency_that_is_code_exists(stages, stage) -> None:
    for dep in _paths(stages[stage]["deps"]):
        if dep.startswith(("src/", "scripts/", "config/")):
            assert (PROJECT_ROOT / dep).exists(), f"{stage} depends on missing {dep}"


# ------------------------------------------------------------------------ the DAG

def test_dag_is_wired_in_order(stages) -> None:
    """validate -> prepare -> train -> predict, each linked by a real artifact."""
    validate_out = _paths(stages["validate"]["outs"])
    assert validate_out & _paths(stages["prepare"]["deps"]), "prepare may run before validate"

    prepare_outs = _paths(stages["prepare"]["outs"])
    assert "data/processed/train_features.parquet" in prepare_outs & _paths(stages["train"]["deps"])
    assert "data/processed/test_features.parquet" in prepare_outs & _paths(stages["predict"]["deps"])

    train_outs = _paths(stages["train"]["outs"])
    assert "models/champion_pipeline.joblib" in train_outs & _paths(stages["predict"]["deps"])


def test_predict_depends_on_the_champion_and_its_metadata(stages) -> None:
    """Both, not just the model: the threshold lives in metadata, and a change to
    either must re-score the batch."""
    deps = _paths(stages["predict"]["deps"])
    assert "models/champion_pipeline.joblib" in deps
    assert "artifacts/deployment_meta.json" in deps


def test_predict_does_not_retrain(stages) -> None:
    """Scoring must not produce a model, and must not read the training features."""
    assert "scripts/train.py" not in stages["predict"]["cmd"]
    outs = _paths(stages["predict"].get("outs"))
    assert not any(o.endswith(".joblib") for o in outs)
    assert "data/processed/train_features.parquet" not in _paths(stages["predict"]["deps"])


def test_validate_stage_does_not_train(stages) -> None:
    outs = _paths(stages["validate"]["outs"])
    assert not any(o.endswith(".joblib") for o in outs)
    assert "params.yaml" not in _paths(stages["validate"]["deps"])


# ------------------------------------------------------------------------- params

def test_train_declares_the_params_sections_it_reads(stages) -> None:
    """Missing a section here means editing it would NOT re-run training."""
    from src.utils.config_loader import REQUIRED_SECTIONS

    assert set(stages["train"].get("params", [])) == set(REQUIRED_SECTIONS)


def test_parameter_free_stages_declare_no_params(stages) -> None:
    """prepare and predict genuinely read no params; declaring some would re-run the
    expensive aggregation whenever an unrelated search space changed."""
    for stage in ("validate", "prepare", "predict"):
        assert not stages[stage].get("params"), f"{stage} declares params it does not read"


# ---------------------------------------------------------------- outputs & metrics

def test_train_declares_its_artifacts(stages) -> None:
    outs = _paths(stages["train"]["outs"])
    assert "models/champion_pipeline.joblib" in outs
    assert "artifacts/deployment_meta.json" in outs
    assert "artifacts/metrics/validation_leaderboard.csv" in outs


def test_metrics_are_tracked(stages) -> None:
    assert "artifacts/metrics/test_metrics.json" in _paths(stages["train"].get("metrics"))


def test_predict_declares_the_predictions_output(stages) -> None:
    assert _paths(stages["predict"]["outs"]) == {"artifacts/predictions/test_predictions.csv"}


def test_no_output_is_claimed_by_two_stages(stages) -> None:
    """Two stages writing one path makes the DAG ambiguous and DVC refuses to build it."""
    seen: dict[str, str] = {}
    for name, stage in stages.items():
        for path in _paths(stage.get("outs")) | _paths(stage.get("metrics")):
            assert path not in seen, f"{path} is written by both {seen[path]} and {name}"
            seen[path] = name


# ---------------------------------------------------------------- script importability

@pytest.mark.parametrize("script", ["validate_data", "prepare_data", "train", "predict"])
def test_stage_script_imports_without_side_effects(script) -> None:
    """Importing must not read data or train — every script guards on __main__."""
    spec = importlib.util.spec_from_file_location(
        f"dvc_stage_{script}", PROJECT_ROOT / "scripts" / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


# ------------------------------------------------------------------------ dvcignore

def test_dvcignore_does_not_hide_pipeline_inputs() -> None:
    """Ignoring a path the pipeline reads would make DVC blind to real changes."""
    patterns = [
        line.strip() for line in (PROJECT_ROOT / ".dvcignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for needed in ("data/", "src/", "scripts/", "config/", "params.yaml",
                   "models/", "artifacts/"):
        assert needed not in patterns, f".dvcignore hides {needed}, which the pipeline uses"
