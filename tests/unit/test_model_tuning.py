"""Step 4 — hyperparameter tuning with leak-free cross-validation.

The contract these pin: preprocessing lives *inside* the estimator handed to the search,
so every CV fold fits its own imputer/clipper/scaler/encoder on that fold's training
portion only. Fitting the preprocessor once on all of X_train and passing the transformed
matrix into the search — the previous implementation — let each fold's held-out rows
contribute to the statistics used to transform them, inflating CV scores.

`test_preprocessing_is_refitted_per_cv_fold` is the test that actually proves it: a
counting transformer shows folds+1 fits (one per fold, plus the final refit), not 1.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tune import (
    MODEL_STEP,
    PREPROCESSOR_STEP,
    SEARCH_SPACES,
    TUNED_SUFFIX,
    build_tunable_models,
    build_tuning_pipeline,
    tune_candidates,
    tune_model,
)


@pytest.fixture
def raw_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Raw, untransformed frame — with NaNs and a categorical, as tuning now receives."""
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    X = pd.DataFrame({
        "num_a": signal,
        "num_b": rng.normal(size=n),
        "cat_a": rng.choice(["x", "y", "z"], size=n),
    })
    X.loc[X.sample(frac=0.1, random_state=1).index, "num_b"] = np.nan
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(signal * 2 - 1.5)))).astype(int))
    if y.sum() < 20:
        y.iloc[:25] = 1
    return X, y


class _CountingPreprocessor(BaseEstimator, TransformerMixin):
    """Passthrough that records how many times `fit` was called across all instances."""

    fit_calls = 0

    def fit(self, X, y=None):
        type(self).fit_calls += 1
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


# --------------------------------------------------------------- search-space shape

def test_only_the_three_intended_models_are_tunable() -> None:
    assert set(SEARCH_SPACES) == {"Random Forest", "XGBoost", "LightGBM"}


@pytest.mark.parametrize("name", ["Random Forest", "XGBoost", "LightGBM"])
def test_every_search_space_key_uses_the_model_prefix(name) -> None:
    """Inside a Pipeline, a bare `n_estimators` addresses nothing and sklearn raises."""
    unprefixed = [k for k in SEARCH_SPACES[name] if not k.startswith(f"{MODEL_STEP}__")]
    assert unprefixed == [], f"{name} has unprefixed keys: {unprefixed}"


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
    present = {k.split("__", 1)[1] for k in SEARCH_SPACES[name]}
    assert required <= present, f"{name} missing {required - present}"


def test_tunable_models_are_built_unfitted() -> None:
    models = build_tunable_models(random_state=42, scale_pos_weight=3.0)
    assert set(models) == {"Random Forest", "XGBoost", "LightGBM"}
    for model in models.values():
        with pytest.raises(Exception):
            check_is_fitted(model)


def test_imbalance_handling_is_configured() -> None:
    models = build_tunable_models(random_state=42, scale_pos_weight=11.5)
    assert models["XGBoost"].get_params()["scale_pos_weight"] == 11.5
    assert models["LightGBM"].get_params()["class_weight"] == "balanced"
    assert models["Random Forest"].get_params()["class_weight"] == "balanced"


# ------------------------------------------------------------------ pipeline shape

def test_build_tuning_pipeline_wraps_preprocessing_and_model(raw_xy) -> None:
    X, _ = raw_xy
    pipe = build_tuning_pipeline(build_tunable_models(42)["Random Forest"], X)
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == [PREPROCESSOR_STEP, MODEL_STEP]


def test_search_estimator_is_a_pipeline_containing_preprocessing(raw_xy) -> None:
    """RandomizedSearchCV must clone a Pipeline, not a bare estimator."""
    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    assert isinstance(search.estimator, Pipeline)
    assert PREPROCESSOR_STEP in search.estimator.named_steps
    assert isinstance(search.best_estimator_, Pipeline)
    assert PREPROCESSOR_STEP in search.best_estimator_.named_steps


def test_pipeline_handed_to_the_search_is_unfitted(raw_xy) -> None:
    """`build_tuning_pipeline` reads column layout only — it learns no statistic.

    The complement of `test_preprocessing_is_refitted_per_cv_fold`: that one proves
    preprocessing is fitted *inside* every fold, this one proves it was not already
    fitted *before* the search ever saw the data.
    """
    X, _ = raw_xy
    pipe = build_tuning_pipeline(build_tunable_models(42)["Random Forest"], X)
    with pytest.raises(Exception):
        check_is_fitted(pipe.named_steps[PREPROCESSOR_STEP])


def test_search_never_fits_the_template_estimator(raw_xy) -> None:
    """The search clones its estimator per candidate; the template stays untouched.

    If the template's preprocessor came back fitted, something had fitted it on all of
    X_train rather than per fold — the leak this module exists to prevent.
    """
    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    with pytest.raises(Exception):
        check_is_fitted(search.estimator.named_steps[PREPROCESSOR_STEP])


def test_tuner_accepts_raw_dataframes(raw_xy) -> None:
    """Raw frame in — NaNs and a categorical column included, no pre-transform."""
    X, y = raw_xy
    assert X.isna().any().any() and X["cat_a"].dtype == object
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    check_is_fitted(search.best_estimator_)
    assert search.best_estimator_.predict(X).shape == (len(y),)


@pytest.mark.parametrize("name", ["Random Forest", "XGBoost", "LightGBM"])
def test_each_model_tunes_into_a_fitted_pipeline(name, raw_xy) -> None:
    """Every tunable model must survive the raw-frame contract on its own.

    Covered per model rather than only in aggregate: the three estimators come from
    three libraries with different input handling, so one of them silently failing to
    accept the preprocessor's output is a real failure mode.
    """
    X, y = raw_xy
    search = tune_model(build_tunable_models(42, scale_pos_weight=2.0)[name],
                        SEARCH_SPACES[name], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    best = search.best_estimator_
    assert isinstance(best, Pipeline)
    assert list(best.named_steps) == [PREPROCESSOR_STEP, MODEL_STEP]
    check_is_fitted(best)
    check_is_fitted(best.named_steps[PREPROCESSOR_STEP])
    proba = best.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),) and np.isfinite(proba).all()


def test_fitted_pipeline_predicts_from_raw_dataframes(raw_xy) -> None:
    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["LightGBM"], SEARCH_SPACES["LightGBM"],
                        X, y, n_iter=2, cv_folds=2, random_state=42)
    proba = search.best_estimator_.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),) and np.isfinite(proba).all()


# ------------------------------------------------- the leak fix, proved by counting

def test_preprocessing_is_refitted_per_cv_fold(raw_xy) -> None:
    """folds + 1 fits: one per CV fold, plus the final refit on all of X_train.

    A single fit would mean preprocessing statistics were shared across folds — the
    leak this change exists to remove.

    n_jobs=1 is required for the assertion, not for correctness: with parallel workers
    the fold fits happen in child processes and their counter increments never reach
    the parent, so the count would read 1 (the in-process refit) regardless.
    """
    X, y = raw_xy
    _CountingPreprocessor.fit_calls = 0
    tune_model(
        build_tunable_models(42)["Random Forest"],
        {f"{MODEL_STEP}__n_estimators": [10]},
        X.select_dtypes(include=np.number).fillna(0.0), y,
        n_iter=1, cv_folds=3, random_state=42, n_jobs=1,
        preprocessor=_CountingPreprocessor(),
    )
    assert _CountingPreprocessor.fit_calls == 4, (
        f"expected 3 fold fits + 1 refit, got {_CountingPreprocessor.fit_calls}"
    )


# ------------------------------------------------------------------ search settings

def test_tuning_uses_stratified_kfold(raw_xy) -> None:
    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=3, random_state=42)
    assert isinstance(search.cv, StratifiedKFold)
    assert search.cv.get_n_splits() == 3


def test_scoring_is_average_precision(raw_xy) -> None:
    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    assert search.scoring == "average_precision"


def test_tuning_uses_randomized_not_grid_search(raw_xy) -> None:
    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    assert isinstance(search, RandomizedSearchCV)


def test_tuning_is_deterministic_for_a_fixed_seed(raw_xy) -> None:
    X, y = raw_xy
    a = tune_model(build_tunable_models(42)["Random Forest"], SEARCH_SPACES["Random Forest"],
                   X, y, n_iter=3, cv_folds=2, random_state=7)
    b = tune_model(build_tunable_models(42)["Random Forest"], SEARCH_SPACES["Random Forest"],
                   X, y, n_iter=3, cv_folds=2, random_state=7)
    assert a.best_params_ == b.best_params_


# ------------------------------------------------------- no access to val/test data

@pytest.mark.parametrize("func", [tune_model, tune_candidates])
def test_tuning_functions_cannot_receive_test_data(func) -> None:
    """Leakage prevented by construction: no parameter exists to pass test data in."""
    params = set(inspect.signature(func).parameters)
    forbidden = {"X_test", "y_test", "X_val", "y_val", "X_eval", "y_eval"}
    assert not (params & forbidden), f"{func.__name__} exposes {params & forbidden}"


# ----------------------------------------------------------------- tune_candidates

def test_tune_candidates_tunes_all_three(raw_xy) -> None:
    X, y = raw_xy
    tuned = tune_candidates(X, y, n_iter=2, cv_folds=2, random_state=42, scale_pos_weight=2.0)
    assert set(tuned) == {f"Random Forest{TUNED_SUFFIX}",
                          f"XGBoost{TUNED_SUFFIX}",
                          f"LightGBM{TUNED_SUFFIX}"}
    for search in tuned.values():
        assert isinstance(search.best_estimator_, Pipeline)
        check_is_fitted(search.best_estimator_)


def test_tune_candidates_can_tune_a_subset(raw_xy) -> None:
    X, y = raw_xy
    tuned = tune_candidates(X, y, models=["LightGBM"], n_iter=2, cv_folds=2, random_state=42)
    assert set(tuned) == {f"LightGBM{TUNED_SUFFIX}"}


def test_tune_candidates_rejects_unknown_model(raw_xy) -> None:
    X, y = raw_xy
    with pytest.raises(ValueError, match="Unknown"):
        tune_candidates(X, y, models=["Nonexistent"], n_iter=2, cv_folds=2)


# ------------------------------------------------- §13: no nested CPU oversubscription

def test_tuned_estimators_do_not_self_parallelise() -> None:
    """One parallelism layer only.

    RandomizedSearchCV fans out across candidates. If each estimator also grabbed every
    core, the two layers would multiply into far more threads than the box has cores,
    which is slower than either strategy alone.
    """
    for name, model in build_tunable_models(random_state=42).items():
        assert model.get_params()["n_jobs"] == 1, f"{name} would oversubscribe inside the search"


def test_search_owns_the_parallelism(raw_xy) -> None:
    """The search parallelises, not the estimators — whatever worker count params sets.

    Asserted against params.yaml rather than a literal, because the right worker count
    is machine-specific: it is bounded by how many densely transformed folds fit in RAM
    at once, not by core count.
    """
    from src.utils.config_loader import get_section

    X, y = raw_xy
    search = tune_model(build_tunable_models(42)["Random Forest"],
                        SEARCH_SPACES["Random Forest"], X, y,
                        n_iter=2, cv_folds=2, random_state=42)
    assert search.n_jobs == get_section("tuning")["search_n_jobs"]
    assert search.n_jobs != 1, "the search must own the parallelism layer"


# ------------------------------------------------------ §15: baseline uses a Pipeline

def test_baseline_is_a_pipeline_with_preprocessing(raw_xy) -> None:
    """The baseline must share the tuned models' preprocessing contract, or the
    comparison measures two data treatments rather than two algorithms."""
    from src.models.train import build_baseline_pipeline

    X, y = raw_xy
    baseline = build_baseline_pipeline(X, random_state=42)
    assert isinstance(baseline, Pipeline)
    assert list(baseline.named_steps) == [PREPROCESSOR_STEP, MODEL_STEP]


def test_baseline_fits_and_predicts_from_raw_frames(raw_xy) -> None:
    from src.models.train import build_baseline_pipeline

    X, y = raw_xy
    baseline = build_baseline_pipeline(X, random_state=42)
    baseline.fit(X, y)
    proba = baseline.predict_proba(X)[:, 1]
    assert proba.shape == (len(y),) and np.isfinite(proba).all()
