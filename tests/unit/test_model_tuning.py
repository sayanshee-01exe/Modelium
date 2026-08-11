"""Step 4 — hyperparameter tuning.

The contract these pin: tuning happens with cross-validation *inside the training
split*, optimising average_precision, and the tuning functions cannot see validation or
test data even by accident — their signatures have nowhere to put it.

Searches here run with tiny n_iter/cv on synthetic data; the real search spaces are
exercised for shape, not for predictive quality.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tune import (
    SEARCH_SPACES,
    TUNED_SUFFIX,
    build_tunable_models,
    tune_candidates,
    tune_model,
)


@pytest.fixture
def imbalanced_xy():
    """~15% positive, with real signal so a tree can actually fit something."""
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    X = np.column_stack([signal, rng.normal(size=n), rng.normal(size=n)])
    y = (rng.random(n) < 1 / (1 + np.exp(-(signal * 2 - 2)))).astype(int)
    if y.sum() < 10:                       # guarantee both classes are well represented
        y[:15] = 1
    return X, y


# ------------------------------------------------------------------- search spaces

def test_only_the_three_intended_models_are_tunable() -> None:
    """LogReg stays an untuned baseline; SVM/DT/AdaBoost/GBM stay out of the hot path."""
    assert set(SEARCH_SPACES) == {"Random Forest", "XGBoost", "LightGBM"}


@pytest.mark.parametrize("params", [
    ("Random Forest", {"n_estimators", "max_depth", "min_samples_split",
                       "min_samples_leaf", "max_features"}),
    ("XGBoost", {"n_estimators", "learning_rate", "max_depth", "min_child_weight",
                 "subsample", "colsample_bytree", "gamma", "reg_alpha", "reg_lambda"}),
    ("LightGBM", {"n_estimators", "learning_rate", "num_leaves", "max_depth",
                  "min_child_samples", "subsample", "colsample_bytree",
                  "reg_alpha", "reg_lambda"}),
])
def test_search_space_covers_required_parameters(params) -> None:
    name, required = params
    assert required <= set(SEARCH_SPACES[name]), f"{name} missing {required - set(SEARCH_SPACES[name])}"


def test_tunable_models_are_built_unfitted() -> None:
    models = build_tunable_models(random_state=42, scale_pos_weight=3.0)
    assert set(models) == {"Random Forest", "XGBoost", "LightGBM"}
    for name, model in models.items():
        with pytest.raises(Exception):
            check_is_fitted(model)


def test_imbalance_handling_is_configured() -> None:
    """scale_pos_weight for the boosters, class_weight for the forest."""
    models = build_tunable_models(random_state=42, scale_pos_weight=11.5)
    assert models["XGBoost"].get_params()["scale_pos_weight"] == 11.5
    assert models["LightGBM"].get_params()["class_weight"] == "balanced"
    assert models["Random Forest"].get_params()["class_weight"] == "balanced"


# --------------------------------------------------------------------- tune_model

def test_tuning_returns_a_fitted_best_estimator(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    models = build_tunable_models(42, 2.0)
    search = tune_model(models["Random Forest"], SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    check_is_fitted(search.best_estimator_)
    assert search.best_estimator_.predict(X).shape == (len(y),)


def test_tuning_uses_stratified_kfold(imbalanced_xy) -> None:
    """Plain KFold can hand a fold zero positives, making average_precision undefined."""
    X, y = imbalanced_xy
    models = build_tunable_models(42, 2.0)
    search = tune_model(models["Random Forest"], SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=3, random_state=42)
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.get_n_splits() == 3


def test_scoring_is_average_precision(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    models = build_tunable_models(42, 2.0)
    search = tune_model(models["Random Forest"], SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    assert search.scoring == "average_precision"


def test_tuning_uses_randomized_not_grid_search(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    models = build_tunable_models(42, 2.0)
    search = tune_model(models["Random Forest"], SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    assert isinstance(search, RandomizedSearchCV)


def test_tuning_is_deterministic_for_a_fixed_seed(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    models = build_tunable_models(42, 2.0)
    a = tune_model(models["Random Forest"], SEARCH_SPACES["Random Forest"], X, y,
                   n_iter=3, cv_folds=2, random_state=7)
    b = tune_model(build_tunable_models(42, 2.0)["Random Forest"],
                   SEARCH_SPACES["Random Forest"], X, y,
                   n_iter=3, cv_folds=2, random_state=7)
    assert a.best_params_ == b.best_params_


# ------------------------------------------------------- no access to val/test data

@pytest.mark.parametrize("func", [tune_model, tune_candidates])
def test_tuning_functions_cannot_receive_test_data(func) -> None:
    """Leakage prevention by construction: there is no parameter to pass test data in."""
    params = set(inspect.signature(func).parameters)
    forbidden = {"X_test", "y_test", "X_val", "y_val", "X_eval", "y_eval"}
    assert not (params & forbidden), f"{func.__name__} exposes {params & forbidden}"


# ----------------------------------------------------------------- tune_candidates

def test_tune_candidates_tunes_all_three(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    tuned = tune_candidates(X, y, n_iter=2, cv_folds=2, random_state=42, scale_pos_weight=2.0)
    assert set(tuned) == {f"Random Forest{TUNED_SUFFIX}",
                          f"XGBoost{TUNED_SUFFIX}",
                          f"LightGBM{TUNED_SUFFIX}"}
    for search in tuned.values():
        check_is_fitted(search.best_estimator_)


def test_tune_candidates_can_tune_a_subset(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    tuned = tune_candidates(X, y, models=["LightGBM"], n_iter=2, cv_folds=2, random_state=42)
    assert set(tuned) == {f"LightGBM{TUNED_SUFFIX}"}


def test_tune_candidates_rejects_unknown_model(imbalanced_xy) -> None:
    X, y = imbalanced_xy
    with pytest.raises(ValueError, match="Unknown"):
        tune_candidates(X, y, models=["Nonexistent"], n_iter=2, cv_folds=2)
