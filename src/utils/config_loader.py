"""Load and validate `params.yaml`, the single source of truth for experiment values.

Deliberately small: a dict, a schema check, and typed accessors. No framework, no
Hydra, no dynamic composition — the project has one params file and four stages, and
anything more would be machinery to maintain rather than a problem solved.

Validation happens at load, not at use. A `cv_folds: 1` caught here fails in
milliseconds with a message naming the field; the same value caught inside
`RandomizedSearchCV` fails minutes into a run with a message about split counts.
`split_train_val_test` still re-checks the split sizes it is handed — that is defence
in depth against a caller that bypasses this loader, not a duplicated rule.

The parsed file is cached, so importing this from several modules reads the disk once
and every caller sees the same values within a run.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_FILE = PROJECT_ROOT / "params.yaml"

# Sections that change what the model *is*. The DVC train stage declares exactly these,
# so editing any of them invalidates training.
MODEL_SECTIONS: tuple[str, ...] = (
    "data", "preprocessing", "tuning", "selection", "threshold", "models",
)

# Observability only. Deliberately excluded from the DVC train stage's params: renaming
# an MLflow experiment or switching tracking off must not invalidate a 4-hour run, since
# it changes nothing about the resulting model.
TRACKING_SECTIONS: tuple[str, ...] = ("mlflow",)

REQUIRED_SECTIONS: tuple[str, ...] = MODEL_SECTIONS + TRACKING_SECTIONS

# Search spaces must exist for exactly the three tunable models, keyed as in
# src/models/tune.py. A typo'd section would otherwise silently tune nothing.
REQUIRED_MODEL_KEYS: tuple[str, ...] = ("random_forest", "xgboost", "lightgbm")

# sklearn scorer names this project is willing to optimise. Restricted on purpose: a
# threshold-dependent scorer such as "recall" would select a model by its behaviour at
# the arbitrary 0.5 default, which is the ordering bug Step 4 removed.
VALID_SCORINGS: frozenset[str] = frozenset({"average_precision", "roc_auc", "neg_log_loss"})

VALID_THRESHOLD_STRATEGIES: frozenset[str] = frozenset({"f1"})

# Registry keys the `mlflow:` section must declare, alongside enabled / experiment_name
# / tracking_uri. The two aliases are separate entries rather than one because
# "approved" and "recorded but refused" are different states, and a serving layer
# resolves only the first.
REGISTRY_KEYS: tuple[str, ...] = (
    "registered_model_name", "champion_alias", "candidate_alias",
)

# Metric name the leaderboard and gates use, paired with the sklearn scorer the search
# optimises. Ranking by a different estimator of the PR curve than the one being
# optimised lets the search winner lose the selection.
SCORING_TO_METRIC: dict[str, str] = {
    "average_precision": "Average Precision",
    "roc_auc": "ROC-AUC",
}


def _require(section: dict, key: str, where: str) -> Any:
    if key not in section or section[key] is None:
        raise ConfigurationError(f"params.yaml: '{where}.{key}' is required but missing")
    return section[key]


def _require_number(section: dict, key: str, where: str) -> float:
    value = _require(section, key, where)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(
            f"params.yaml: '{where}.{key}' must be a number, got {type(value).__name__} "
            f"({value!r})"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ConfigurationError(f"params.yaml: '{where}.{key}' must be finite, got {value}")
    return value


def _require_fraction(section: dict, key: str, where: str) -> float:
    """A value that is only meaningful as a proportion or a probability."""
    value = _require_number(section, key, where)
    if not 0.0 <= value <= 1.0:
        raise ConfigurationError(
            f"params.yaml: '{where}.{key}' must be within [0, 1], got {value}"
        )
    return value


def _require_int(section: dict, key: str, where: str, minimum: int) -> int:
    value = _require(section, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(
            f"params.yaml: '{where}.{key}' must be an integer, got "
            f"{type(value).__name__} ({value!r})"
        )
    if value < minimum:
        raise ConfigurationError(
            f"params.yaml: '{where}.{key}' must be >= {minimum}, got {value}"
        )
    return value


def validate_params(params: dict) -> dict:
    """Check the whole file, raising on the first structural or range violation.

    Args:
        params: Parsed `params.yaml` contents.

    Returns:
        The same mapping, unchanged, so this can wrap a load call.

    Raises:
        ConfigurationError: naming the offending field and the rule it broke.
    """
    if not isinstance(params, dict):
        raise ConfigurationError(
            f"params.yaml must parse to a mapping, got {type(params).__name__}"
        )

    missing = [s for s in REQUIRED_SECTIONS if s not in params or params[s] is None]
    if missing:
        raise ConfigurationError(
            f"params.yaml is missing required section(s): {sorted(missing)}"
        )
    for section in REQUIRED_SECTIONS:
        if not isinstance(params[section], dict):
            raise ConfigurationError(
                f"params.yaml: section '{section}' must be a mapping, got "
                f"{type(params[section]).__name__}"
            )

    # --- data -------------------------------------------------------------------
    data = params["data"]
    validation_size = _require_number(data, "validation_size", "data")
    test_size = _require_number(data, "test_size", "data")
    for name, size in (("validation_size", validation_size), ("test_size", test_size)):
        if not 0.0 < size < 1.0:
            raise ConfigurationError(
                f"params.yaml: 'data.{name}' must be strictly between 0 and 1, got {size}"
            )
    if validation_size + test_size >= 1.0:
        raise ConfigurationError(
            f"params.yaml: 'data.validation_size' + 'data.test_size' must be < 1 to "
            f"leave training data, got {validation_size} + {test_size} = "
            f"{validation_size + test_size}"
        )
    _require_int(data, "random_state", "data", minimum=0)

    # --- preprocessing ----------------------------------------------------------
    iqr_factor = _require_number(params["preprocessing"], "iqr_factor", "preprocessing")
    if iqr_factor <= 0:
        raise ConfigurationError(
            f"params.yaml: 'preprocessing.iqr_factor' must be positive, got {iqr_factor}; "
            f"a non-positive fence would clip every value to the median"
        )

    # --- tuning -----------------------------------------------------------------
    tuning = params["tuning"]
    _require_int(tuning, "n_iter", "tuning", minimum=1)
    _require_int(tuning, "cv_folds", "tuning", minimum=2)
    scoring = _require(tuning, "scoring", "tuning")
    if scoring not in VALID_SCORINGS:
        raise ConfigurationError(
            f"params.yaml: 'tuning.scoring' must be one of {sorted(VALID_SCORINGS)}, "
            f"got {scoring!r}"
        )
    for key in ("search_n_jobs", "estimator_n_jobs"):
        jobs = _require(tuning, key, "tuning")
        if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs == 0:
            raise ConfigurationError(
                f"params.yaml: 'tuning.{key}' must be a non-zero integer "
                f"(-1 means all cores), got {jobs!r}"
            )
    if tuning["search_n_jobs"] != 1 and tuning["estimator_n_jobs"] != 1:
        raise ConfigurationError(
            f"params.yaml: 'tuning.search_n_jobs'={tuning['search_n_jobs']} and "
            f"'tuning.estimator_n_jobs'={tuning['estimator_n_jobs']} both parallelise. "
            f"Nested parallelism oversubscribes the machine and runs slower than either "
            f"layer alone; set one of them to 1."
        )

    # --- selection --------------------------------------------------------------
    selection = params["selection"]
    primary_metric = _require(selection, "primary_metric", "selection")
    _require_fraction(selection, "min_average_precision", "selection")
    _require_fraction(selection, "min_roc_auc", "selection")

    expected_metric = SCORING_TO_METRIC.get(scoring)
    if expected_metric and primary_metric != expected_metric:
        raise ConfigurationError(
            f"params.yaml: 'tuning.scoring'={scoring!r} optimises {expected_metric!r} but "
            f"'selection.primary_metric' is {primary_metric!r}. Optimising one estimator "
            f"of the curve and ranking by another lets the model that wins the search "
            f"lose the selection."
        )

    # --- threshold --------------------------------------------------------------
    threshold = params["threshold"]
    strategy = _require(threshold, "strategy", "threshold")
    if strategy not in VALID_THRESHOLD_STRATEGIES:
        raise ConfigurationError(
            f"params.yaml: 'threshold.strategy' must be one of "
            f"{sorted(VALID_THRESHOLD_STRATEGIES)}, got {strategy!r}"
        )
    _require_fraction(threshold, "min_recall", "threshold")

    # --- mlflow -----------------------------------------------------------------
    tracking = params["mlflow"]
    enabled = _require(tracking, "enabled", "mlflow")
    if not isinstance(enabled, bool):
        raise ConfigurationError(
            f"params.yaml: 'mlflow.enabled' must be a boolean, got "
            f"{type(enabled).__name__} ({enabled!r}); the string \"false\" is truthy "
            f"and would switch tracking on"
        )
    for key in ("experiment_name", "tracking_uri"):
        value = _require(tracking, key, "mlflow")
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"params.yaml: 'mlflow.{key}' must be a non-empty string, got {value!r}"
            )

    # --- mlflow registry keys ---------------------------------------------------
    # Checked even when tracking is disabled: the values decide what a *later* enabled
    # run would be filed under, and a typo found at load costs milliseconds while the
    # same typo found after training costs the run.
    for key in REGISTRY_KEYS:
        value = _require(tracking, key, "mlflow")
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                f"params.yaml: 'mlflow.{key}' must be a non-empty string, got {value!r}"
            )
    if tracking["champion_alias"].strip() == tracking["candidate_alias"].strip():
        raise ConfigurationError(
            f"params.yaml: 'mlflow.champion_alias' and 'mlflow.candidate_alias' are both "
            f"{tracking['champion_alias']!r}. One alias for both states makes an approved "
            f"model indistinguishable from a rejected one, which is the whole point of "
            f"having two."
        )

    # --- models -----------------------------------------------------------------
    models = params["models"]
    missing_models = [m for m in REQUIRED_MODEL_KEYS if m not in models]
    if missing_models:
        raise ConfigurationError(
            f"params.yaml: 'models' is missing search space(s) for {sorted(missing_models)}"
        )
    for name in REQUIRED_MODEL_KEYS:
        space = models[name]
        if not isinstance(space, dict) or not space:
            raise ConfigurationError(
                f"params.yaml: 'models.{name}' must be a non-empty mapping of "
                f"parameter -> list of candidate values"
            )
        for param, values in space.items():
            if not isinstance(values, list) or not values:
                raise ConfigurationError(
                    f"params.yaml: 'models.{name}.{param}' must be a non-empty list of "
                    f"candidate values, got {values!r}"
                )

    return params


@lru_cache(maxsize=None)
def load_params(path: str | Path | None = None) -> dict:
    """Read and validate `params.yaml`. Cached per path.

    Args:
        path: Override for the params file; defaults to the repository's `params.yaml`.

    Returns:
        The validated parameter mapping.

    Raises:
        ConfigurationError: if the file is absent, unparseable, or violates a rule.
    """
    params_path = Path(path) if path is not None else PARAMS_FILE
    if not params_path.exists():
        raise ConfigurationError(
            f"No params.yaml at {params_path}. Experiment parameters live there, not in "
            f"Python; the file is tracked in the repository."
        )

    try:
        with open(params_path, encoding="utf-8") as handle:
            params = yaml.safe_load(handle)
    except yaml.YAMLError as err:
        raise ConfigurationError(f"params.yaml at {params_path} is not valid YAML: {err}") from err

    validate_params(params)
    logger.info("Loaded parameters from %s (sections: %s)", params_path, ", ".join(params))
    return params


def get_section(name: str, path: str | Path | None = None) -> dict:
    """One validated section, e.g. ``get_section("tuning")``."""
    return load_params(path)[name]
