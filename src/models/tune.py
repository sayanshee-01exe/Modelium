"""Leak-free hyperparameter tuning via randomised search over stratified CV folds.

Preprocessing is part of the estimator, not something applied beforehand:

    RandomizedSearchCV(Pipeline([("preprocessor", ...), ("model", ...)]))

so each CV fold fits its own imputer, IQR bounds, scaler and encoder categories on that
fold's *training* portion and merely transforms its held-out portion. Fitting the
preprocessor once on all of X_train and searching over the transformed matrix — as this
module previously did — lets every fold's held-out rows contribute to the statistics
used to transform them, which inflates the CV score.

Because the estimator is a Pipeline, every search-space key is addressed through the
model step (`model__n_estimators`); a bare `n_estimators` addresses nothing and sklearn
raises.

Tuning consumes the **training split only**: these functions have no parameter through
which validation or test data could be passed. Scoring is `average_precision`, matching
the metric `src/models/selection.py` ranks by — optimising one estimator of the PR curve
and selecting by another would let the search winner differ from the selection winner.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.features.data_preprocessing import build_preprocessor
from src.utils.config_loader import get_section
from src.utils.logger import get_logger

logger = get_logger(__name__)

TUNED_SUFFIX = " (Tuned)"
PREPROCESSOR_STEP = "preprocessor"
MODEL_STEP = "model"

# params.yaml keys -> the display names used throughout the pipeline. The YAML side
# stays snake_case (conventional for config); the Python side keeps the names that
# already appear in leaderboards, metadata and logs.
PARAM_KEY_TO_MODEL: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def build_search_spaces(models_params: dict[str, dict[str, list[Any]]] | None = None
                        ) -> dict[str, dict[str, list[Any]]]:
    """Translate the `models:` section of params.yaml into sklearn search spaces.

    The `model__` prefix is applied here rather than written into the YAML, so the file
    stays readable and the prefix cannot be forgotten for one parameter out of thirty.
    Inside a Pipeline a bare `n_estimators` addresses nothing and sklearn raises.

    Args:
        models_params: The `models:` mapping; read from params.yaml when omitted.

    Returns:
        ``{"Random Forest": {"model__n_estimators": [...], ...}, ...}``
    """
    source = get_section("models") if models_params is None else models_params
    return {
        display: {f"{MODEL_STEP}__{param}": list(values)
                  for param, values in source[key].items()}
        for key, display in PARAM_KEY_TO_MODEL.items()
    }


# Module-level default so callers and tests keep a single obvious entry point. Budgets
# stay modest: n_iter x cv_folds fits per model is the practical laptop ceiling, and
# each fit also refits the preprocessor.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = build_search_spaces()

_TUNING = get_section("tuning")
DEFAULT_N_ITER: int = _TUNING["n_iter"]
DEFAULT_CV_FOLDS: int = _TUNING["cv_folds"]
DEFAULT_SCORING: str = _TUNING["scoring"]
DEFAULT_SEARCH_N_JOBS: int = _TUNING["search_n_jobs"]
DEFAULT_ESTIMATOR_N_JOBS: int = _TUNING["estimator_n_jobs"]
DEFAULT_RANDOM_STATE: int = get_section("data")["random_state"]


def build_tunable_models(random_state: int = DEFAULT_RANDOM_STATE,
                         scale_pos_weight: float = 1.0,
                         estimator_n_jobs: int = DEFAULT_ESTIMATOR_N_JOBS) -> dict:
    """Return unfitted base estimators for the three tunable models.

    Imbalance is handled by the estimator rather than by resampling: `class_weight` for
    the forest and LightGBM, `scale_pos_weight` for XGBoost. Resampling would have to
    happen inside each CV fold to stay leak-free, which randomised search does not do
    for free — weighting is the correct default here.

    Args:
        random_state: Seed, so a rerun reproduces the same fits.
        scale_pos_weight: Typically ``(negatives / positives)`` from the training split.
        estimator_n_jobs: Threads *per estimator*. Defaults to 1 on purpose: the search
            already parallelises across candidates with ``n_jobs=-1``, and letting each
            estimator also claim every core oversubscribes the machine (8 workers x 8
            threads on an 8-core box), which is slower than either strategy alone. Raise
            it only when fitting an estimator outside a search.
    """
    return {
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=estimator_n_jobs,
        ),
        "XGBoost": xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
            random_state=random_state, n_jobs=estimator_n_jobs,
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=estimator_n_jobs,
            verbose=-1,
        ),
    }


def build_tuning_pipeline(estimator, X_train, preprocessor=None) -> Pipeline:
    """Wrap an estimator behind preprocessing so the pair is fitted as one unit.

    Args:
        estimator: Unfitted classifier.
        X_train: Raw training frame — only its column layout and dtypes are read, to
            decide which columns are numeric and which are categorical. No statistic is
            learned here; that happens per fold inside the search.
        preprocessor: Override for the transformer, primarily so tests can inject a spy.
            Defaults to `build_preprocessor(X_train)`.

    Returns:
        ``Pipeline([("preprocessor", ...), ("model", estimator)])``.
    """
    transformer = build_preprocessor(X_train) if preprocessor is None else preprocessor
    return Pipeline([(PREPROCESSOR_STEP, transformer), (MODEL_STEP, estimator)])


def tune_model(
    estimator,
    param_distributions: dict[str, list[Any]],
    X_train,
    y_train,
    *,
    n_iter: int = DEFAULT_N_ITER,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_SEARCH_N_JOBS,
    verbose: int = 0,
    preprocessor=None,
) -> RandomizedSearchCV:
    """Randomised search over a preprocessing+model Pipeline, scored by stratified CV.

    Stratified folds matter at a ~8% positive rate: a plain KFold can hand a fold zero
    positives, which makes average_precision undefined for that split.

    Args:
        estimator: Unfitted classifier; wrapped in a Pipeline internally.
        param_distributions: Search space with `model__`-prefixed keys; see `SEARCH_SPACES`.
        X_train: **Raw** training features — untransformed, *only* the training split.
        y_train: Training labels.
        n_iter: Parameter settings sampled. 15-30 is the practical range.
        cv_folds: Stratified folds per candidate.
        scoring: Optimisation metric. Average Precision by default.
        random_state: Seed for both the sampler and the fold split.
        n_jobs: Parallel workers *for the search*. The estimators themselves are built
            single-threaded (see `build_tunable_models`) so the two layers do not
            compete for the same cores.
        verbose: Passed through to the search.
        preprocessor: Optional transformer override; see `build_tuning_pipeline`.

    Returns:
        The fitted `RandomizedSearchCV`. Its `best_estimator_` is a **full Pipeline**
        that accepts raw frames, so inference needs no separate preprocessing step.
    """
    pipeline = build_tuning_pipeline(estimator, X_train, preprocessor=preprocessor)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        estimator=pipeline,
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
    n_iter: int = DEFAULT_N_ITER,
    cv_folds: int = DEFAULT_CV_FOLDS,
    scoring: str = DEFAULT_SCORING,
    random_state: int = DEFAULT_RANDOM_STATE,
    scale_pos_weight: float = 1.0,
    n_jobs: int = DEFAULT_SEARCH_N_JOBS,
    preprocessor=None,
) -> dict[str, RandomizedSearchCV]:
    """Tune each requested model on the raw training split.

    Args:
        X_train: **Raw** training features only.
        y_train: Training labels.
        models: Subset of `SEARCH_SPACES` keys; all three by default.
        n_iter: Parameter settings sampled per model.
        cv_folds: Stratified folds.
        scoring: Optimisation metric.
        random_state: Seed.
        scale_pos_weight: Imbalance ratio from the training split.
        n_jobs: Parallel workers.
        preprocessor: Optional transformer override, mainly for tests.

    Returns:
        ``{"<name> (Tuned)": fitted RandomizedSearchCV}``, each `best_estimator_` a
        full preprocessing+model Pipeline.

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
        logger.info("Tuning %s (%d candidates x %d folds, preprocessing refit per fold)...",
                    name, n_iter, cv_folds)
        results[f"{name}{TUNED_SUFFIX}"] = tune_model(
            base_models[name], SEARCH_SPACES[name], X_train, y_train,
            n_iter=n_iter, cv_folds=cv_folds, scoring=scoring,
            random_state=random_state, n_jobs=n_jobs, preprocessor=preprocessor,
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
