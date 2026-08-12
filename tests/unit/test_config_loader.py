"""Step 6 — params.yaml is the single source of truth, and it is validated at load.

Two things these pin.

*The real file is valid and is actually used.* A params file that loads but which no
module reads is decoration; several tests below assert the shipped values reach the
constants that Steps 3-5 consume.

*Invalid values fail at load, not mid-run.* A `cv_folds: 1` caught here fails in
milliseconds naming the field; the same value caught inside RandomizedSearchCV fails
minutes into a run with a message about split counts.

No DVC reproduction and no tuning: these are file-and-dict tests.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import (
    PARAMS_FILE,
    REQUIRED_MODEL_KEYS,
    REQUIRED_SECTIONS,
    get_section,
    load_params,
    validate_params,
)
from src.utils.exceptions import ConfigurationError


@pytest.fixture
def params() -> dict:
    """A deep copy of the real file, so a test can corrupt one field safely."""
    return copy.deepcopy(load_params())


def _write(tmp_path: Path, params: dict) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params))
    return path


# ------------------------------------------------------------------ the real file

def test_params_file_exists() -> None:
    assert PARAMS_FILE.exists(), f"params.yaml missing at {PARAMS_FILE}"


def test_real_params_file_loads_and_validates() -> None:
    assert isinstance(load_params(), dict)


def test_all_required_sections_are_present() -> None:
    params = load_params()
    assert set(REQUIRED_SECTIONS) <= set(params)


def test_every_tunable_model_has_a_search_space() -> None:
    models = get_section("models")
    assert set(REQUIRED_MODEL_KEYS) <= set(models)
    for name in REQUIRED_MODEL_KEYS:
        assert models[name], f"{name} has an empty search space"


def test_get_section_returns_the_section() -> None:
    assert get_section("tuning")["scoring"] == "average_precision"


def test_loading_is_cached() -> None:
    assert load_params() is load_params()


# --------------------------------------------- parameters actually reach the code

def test_tuning_defaults_come_from_params() -> None:
    from src.models import tune

    tuning = get_section("tuning")
    assert tune.DEFAULT_N_ITER == tuning["n_iter"]
    assert tune.DEFAULT_CV_FOLDS == tuning["cv_folds"]
    assert tune.DEFAULT_SCORING == tuning["scoring"]
    assert tune.DEFAULT_RANDOM_STATE == get_section("data")["random_state"]


def test_search_spaces_are_built_from_params() -> None:
    """The `model__` prefix is added in code so the YAML cannot half-forget it."""
    from src.models.tune import SEARCH_SPACES, build_search_spaces

    spaces = build_search_spaces()
    assert set(spaces) == {"Random Forest", "XGBoost", "LightGBM"}
    assert spaces == SEARCH_SPACES
    for name, space in spaces.items():
        assert all(k.startswith("model__") for k in space), f"{name} has unprefixed keys"

    yaml_rf = get_section("models")["random_forest"]
    assert SEARCH_SPACES["Random Forest"]["model__n_estimators"] == yaml_rf["n_estimators"]


def test_selection_gates_come_from_params() -> None:
    from src.models.selection import DEFAULT_OPERATIONAL_GATES, DEFAULT_QUALITY_GATES

    selection, threshold = get_section("selection"), get_section("threshold")
    assert DEFAULT_QUALITY_GATES["Average Precision"] == selection["min_average_precision"]
    assert DEFAULT_QUALITY_GATES["ROC-AUC"] == selection["min_roc_auc"]
    assert DEFAULT_OPERATIONAL_GATES["Recall"] == threshold["min_recall"]


def test_iqr_factor_reaches_the_preprocessor() -> None:
    import numpy as np
    import pandas as pd

    from src.features.data_preprocessing import build_preprocessor

    frame = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    clipper = build_preprocessor(frame).transformers[0][1].named_steps["iqr_clipper"]
    assert clipper.factor == get_section("preprocessing")["iqr_factor"]


def test_explicit_iqr_factor_overrides_params() -> None:
    import numpy as np
    import pandas as pd

    from src.features.data_preprocessing import build_preprocessor

    frame = pd.DataFrame({"a": np.arange(10.0)})
    clipper = build_preprocessor(frame, iqr_factor=3.0).transformers[0][1].named_steps["iqr_clipper"]
    assert clipper.factor == 3.0


def test_experiment_parameters_are_gone_from_python_config() -> None:
    """One source of truth: these moved to params.yaml and must not linger in config.py."""
    from config import config

    for removed in ("RANDOM_STATE", "VALIDATION_SIZE", "TEST_SIZE",
                    "TUNING_N_ITER", "TUNING_CV_FOLDS"):
        assert not hasattr(config, removed), f"config.py still defines {removed}"


def test_structural_constants_stay_in_python_config() -> None:
    """Paths and column names are not experiment parameters and belong in code."""
    from config import config

    for kept in ("TARGET_COL", "ID_COL", "DATA_DIR", "MODEL_DIR", "ARTIFACT_DIR",
                 "DATA_FILES", "TRAIN_FEATURES_FILE", "TEST_FEATURES_FILE"):
        assert hasattr(config, kept), f"config.py lost {kept}"


# ------------------------------------------------------------- structural failures

def test_missing_file_fails_clearly(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="No params.yaml"):
        load_params(tmp_path / "absent.yaml")


def test_malformed_yaml_fails_clearly(tmp_path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("data: {unclosed\n")
    with pytest.raises(ConfigurationError, match="valid YAML"):
        load_params(path)


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_each_required_section_is_enforced(params, section) -> None:
    del params[section]
    with pytest.raises(ConfigurationError, match=section):
        validate_params(params)


def test_all_missing_sections_are_reported_together(params) -> None:
    del params["tuning"], params["selection"]
    with pytest.raises(ConfigurationError) as exc:
        validate_params(params)
    assert "tuning" in str(exc.value) and "selection" in str(exc.value)


def test_a_non_mapping_section_is_rejected(params) -> None:
    params["tuning"] = ["n_iter", 20]
    with pytest.raises(ConfigurationError, match="mapping"):
        validate_params(params)


# ------------------------------------------------------------------ split settings

@pytest.mark.parametrize("field,value", [
    ("validation_size", 0.0), ("validation_size", -0.1), ("validation_size", 1.0),
    ("test_size", 0.0), ("test_size", -0.5), ("test_size", 1.5),
])
def test_invalid_split_size_raises(params, field, value) -> None:
    params["data"][field] = value
    with pytest.raises(ConfigurationError, match=field):
        validate_params(params)


def test_split_sizes_that_leave_no_training_data_raise(params) -> None:
    params["data"]["validation_size"] = 0.6
    params["data"]["test_size"] = 0.5
    with pytest.raises(ConfigurationError, match="leave training data"):
        validate_params(params)


def test_split_sizes_summing_to_exactly_one_raise(params) -> None:
    params["data"]["validation_size"] = 0.5
    params["data"]["test_size"] = 0.5
    with pytest.raises(ConfigurationError):
        validate_params(params)


def test_non_numeric_split_size_raises(params) -> None:
    params["data"]["test_size"] = "0.15"
    with pytest.raises(ConfigurationError, match="number"):
        validate_params(params)


# ---------------------------------------------------------------- tuning settings

@pytest.mark.parametrize("folds", [1, 0, -3])
def test_too_few_cv_folds_raises(params, folds) -> None:
    """Below 2 there is no held-out fold to score, so CV means nothing."""
    params["tuning"]["cv_folds"] = folds
    with pytest.raises(ConfigurationError, match="cv_folds"):
        validate_params(params)


@pytest.mark.parametrize("n_iter", [0, -1])
def test_non_positive_n_iter_raises(params, n_iter) -> None:
    params["tuning"]["n_iter"] = n_iter
    with pytest.raises(ConfigurationError, match="n_iter"):
        validate_params(params)


def test_non_integer_n_iter_raises(params) -> None:
    params["tuning"]["n_iter"] = 15.5
    with pytest.raises(ConfigurationError, match="integer"):
        validate_params(params)


def test_unknown_scoring_raises(params) -> None:
    params["tuning"]["scoring"] = "accuracy_at_0.5"
    with pytest.raises(ConfigurationError, match="scoring"):
        validate_params(params)


def test_scoring_must_match_the_selection_metric(params) -> None:
    """Optimising one estimator of the curve and ranking by another lets the model that
    wins the search lose the selection — the exact bug Step 4 removed."""
    params["tuning"]["scoring"] = "roc_auc"          # selection still says Average Precision
    with pytest.raises(ConfigurationError, match="primary_metric"):
        validate_params(params)


def test_nested_parallelism_is_rejected(params) -> None:
    """Both layers at -1 oversubscribes the machine and runs slower than either alone."""
    params["tuning"]["estimator_n_jobs"] = -1
    with pytest.raises(ConfigurationError, match="parallel"):
        validate_params(params)


def test_zero_n_jobs_is_rejected(params) -> None:
    params["tuning"]["search_n_jobs"] = 0
    with pytest.raises(ConfigurationError, match="search_n_jobs"):
        validate_params(params)


# ------------------------------------------------------- gates, threshold, models

@pytest.mark.parametrize("field", ["min_average_precision", "min_roc_auc"])
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_out_of_range_gate_raises(params, field, value) -> None:
    params["selection"][field] = value
    with pytest.raises(ConfigurationError, match=field):
        validate_params(params)


@pytest.mark.parametrize("value", [-0.2, 1.01])
def test_out_of_range_min_recall_raises(params, value) -> None:
    params["threshold"]["min_recall"] = value
    with pytest.raises(ConfigurationError, match="min_recall"):
        validate_params(params)


def test_unknown_threshold_strategy_raises(params) -> None:
    params["threshold"]["strategy"] = "youden"
    with pytest.raises(ConfigurationError, match="strategy"):
        validate_params(params)


@pytest.mark.parametrize("factor", [0.0, -1.5])
def test_non_positive_iqr_factor_raises(params, factor) -> None:
    params["preprocessing"]["iqr_factor"] = factor
    with pytest.raises(ConfigurationError, match="iqr_factor"):
        validate_params(params)


@pytest.mark.parametrize("model", REQUIRED_MODEL_KEYS)
def test_missing_model_search_space_raises(params, model) -> None:
    del params["models"][model]
    with pytest.raises(ConfigurationError, match=model):
        validate_params(params)


def test_empty_search_space_raises(params) -> None:
    params["models"]["xgboost"] = {}
    with pytest.raises(ConfigurationError, match="xgboost"):
        validate_params(params)


def test_scalar_instead_of_candidate_list_raises(params) -> None:
    """RandomizedSearchCV samples from lists; a bare scalar is silently iterated."""
    params["models"]["lightgbm"]["num_leaves"] = 31
    with pytest.raises(ConfigurationError, match="num_leaves"):
        validate_params(params)


def test_empty_candidate_list_raises(params) -> None:
    params["models"]["random_forest"]["max_depth"] = []
    with pytest.raises(ConfigurationError, match="max_depth"):
        validate_params(params)


# --------------------------------------------------- round trip through a real file

def test_a_valid_custom_file_loads(tmp_path, params) -> None:
    params["tuning"]["n_iter"] = 3
    loaded = load_params(_write(tmp_path, params))
    assert loaded["tuning"]["n_iter"] == 3


def test_an_invalid_custom_file_is_rejected_on_load(tmp_path, params) -> None:
    params["tuning"]["cv_folds"] = 1
    with pytest.raises(ConfigurationError):
        load_params(_write(tmp_path, params))
