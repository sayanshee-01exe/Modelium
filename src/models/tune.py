"""Hyperparameter tuning via randomised search over stratified CV folds.

Tuning consumes the **training split only**. Cross-validation folds are carved out of
X_train, so the validation split stays clean for model comparison and threshold tuning,
and the test split is never seen at all — these functions have no parameter through
which validation or test data could be passed.

Scoring is `average_precision` (PR-AUC): at a ~8% default rate, accuracy is useless and
ROC-AUC is optimistic, because both are dominated by the majority class.

Only Random Forest, XGBoost and LightGBM are tuned. Logistic Regression is kept as an
untuned interpretable baseline (see `src/models/train.py::build_baseline_model`), and
the remaining experimental estimators in `build_candidate_models` are deliberately left
out of the tuning path to keep a search tractable.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.utils.logger import get_logger

logger = get_logger(__name__)

TUNED_SUFFIX = " (Tuned)"

# Kept deliberately modest: with n_iter=20 and cv=3 this is 60 fits per model, which is
# the practical ceiling for a laptop run on the full feature table.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "Random Forest": {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [6, 8, 10, 12, 16],
        "min_samples_split": [2, 10, 25, 50],
        "min_samples_leaf": [10, 25, 50, 100],
        "max_features": ["sqrt", "log2", 0.3],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 500, 800],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5, 6, 8],
        "min_child_weight": [1, 5, 10, 25],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "gamma": [0.0, 0.1, 0.5, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    },
    "LightGBM": {
        "n_estimators": [300, 500, 800, 1200],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 7, 9],
        "min_child_samples": [10, 25, 50, 100],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 1.0, 5.0],
        "reg_lambda": [0.5, 1.0, 5.0, 10.0],
    },
}


def build_tunable_models(random_state: int = 42, scale_pos_weight: float = 1.0) -> dict:
    """Return unfitted base estimators for the three tunable models.

    Imbalance is handled by the estimator rather than by resampling: `class_weight` for
    the forest and LightGBM, `scale_pos_weight` for XGBoost. Resampling would have to
    happen inside each CV fold to be leak-free, which randomised search does not do for
    free — weighting is the correct default here.

    Args:
        random_state: Seed, so a rerun reproduces the same fits.
        scale_pos_weight: Typically ``(negatives / positives)`` computed on X_train.
    """
    return {
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
            random_state=random_state, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1, verbose=-1,
        ),
    }


def tune_model(
    estimator,
    param_distributions: dict[str, list[Any]],
    X_train,
    y_train,
    *,
    n_iter: int = 20,
    cv_folds: int = 3,
    scoring: str = "average_precision",
    random_state: int = 42,
    n_jobs: int = -1,
    verbose: int = 0,
) -> RandomizedSearchCV:
    """Randomised search over `param_distributions`, scored by stratified CV.

    Stratified folds matter at a ~8% positive rate: a plain KFold can hand a fold zero
    positives, which makes average_precision undefined for that split.

    Args:
        estimator: Unfitted estimator.
        param_distributions: Search space; see `SEARCH_SPACES`.
        X_train: Training features — *only* the training split.
        y_train: Training labels.
        n_iter: Parameter settings sampled. 15-30 is the practical range.
        cv_folds: Stratified folds per candidate.
        scoring: Optimisation metric. PR-AUC by default.
        random_state: Seed for both the sampler and the fold split.
        n_jobs: Parallel workers.
        verbose: Passed through to the search.

    Returns:
        The fitted `RandomizedSearchCV`, exposing `best_estimator_`, `best_params_`
        and `best_score_`.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True,
        error_score="raise",
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    logger.info(
        "Tuned %s: best %s=%.4f over %d candidates x %d folds in %.1fs",
        type(estimator).__name__, scoring, search.best_score_, n_iter, cv_folds,
        time.perf_counter() - started,
    )
    return search


def tune_candidates(
    X_train,
    y_train,
    *,
    models: Sequence[str] | None = None,
    n_iter: int = 20,
    cv_folds: int = 3,
    scoring: str = "average_precision",
    random_state: int = 42,
    scale_pos_weight: float = 1.0,
    n_jobs: int = -1,
) -> dict[str, RandomizedSearchCV]:
    """Tune each requested model on the training split.

    Args:
        X_train: Training features only.
        y_train: Training labels.
        models: Subset of `SEARCH_SPACES` keys; all three by default.
        n_iter: Parameter settings sampled per model.
        cv_folds: Stratified folds.
        scoring: Optimisation metric.
        random_state: Seed.
        scale_pos_weight: Imbalance ratio from the training split.
        n_jobs: Parallel workers.

    Returns:
        ``{"<name> (Tuned)": fitted RandomizedSearchCV}``.

    Raises:
        ValueError: if `models` names an estimator with no search space.
    """
    requested = list(models) if models is not None else list(SEARCH_SPACES)
    unknown = [name for name in requested if name not in SEARCH_SPACES]
    if unknown:
        raise ValueError(
            f"Unknown model(s) for tuning: {unknown}. Available: {sorted(SEARCH_SPACES)}"
        )

    base_models = build_tunable_models(random_state=random_state,
                                       scale_pos_weight=scale_pos_weight)
    results: dict[str, RandomizedSearchCV] = {}
    for name in requested:
        logger.info("Tuning %s (%d candidates x %d folds)...", name, n_iter, cv_folds)
        results[f"{name}{TUNED_SUFFIX}"] = tune_model(
            base_models[name], SEARCH_SPACES[name], X_train, y_train,
            n_iter=n_iter, cv_folds=cv_folds, scoring=scoring,
            random_state=random_state, n_jobs=n_jobs,
        )
    return results


def summarize_tuning(searches: dict[str, RandomizedSearchCV]) -> list[dict]:
    """Compact record of what each search chose, for logging and model metadata."""
    return [
        {
            "Model": name,
            "cv_best_score": float(search.best_score_),
            "cv_scoring": search.scoring,
            "cv_folds": int(search.cv.get_n_splits()),
            "best_params": {k: (v.item() if isinstance(v, np.generic) else v)
                            for k, v in search.best_params_.items()},
        }
        for name, search in searches.items()
    ]
