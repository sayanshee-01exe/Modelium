from __future__ import annotations

from collections import OrderedDict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb


def build_baseline_model(random_state: int = 42):
    """Interpretable Logistic Regression baseline — deliberately not tuned.

    Every tuned model is judged against this on validation. A gradient booster that
    cannot beat a linear model on Average Precision is not worth its complexity, and
    without a fixed reference point that comparison is unavailable.
    """
    return LogisticRegression(
        class_weight="balanced", C=.1, solver="saga", max_iter=1000,
        random_state=random_state, n_jobs=-1,
    )


def build_baseline_pipeline(X_train, random_state: int = 42):
    """Baseline wrapped behind the same preprocessing contract as the tuned models.

    Returns a full ``Pipeline([("preprocessor", ...), ("model", LogisticRegression)])``
    so the baseline consumes raw frames exactly like the tuned pipelines do. Comparing a
    baseline fitted on a pre-transformed matrix against models that preprocess
    internally would compare two different data treatments, not two algorithms.
    """
    from src.models.tune import build_tuning_pipeline  # local: avoids a circular import

    return build_tuning_pipeline(build_baseline_model(random_state), X_train)


def build_candidate_models(random_state: int, scale_pos_weight: float = 1.0):
    """Full experimental portfolio from the original notebook.

    NOT used by the default training path. `scripts/train.py` runs
    `build_baseline_model` plus the three tuned models from `src/models/tune.py`;
    SVM, Decision Tree, AdaBoost and Gradient Boosting are kept here for ad-hoc
    experiments only, because fitting them on the full feature table costs hours and
    none has beaten the boosters.
    """
    return OrderedDict({
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", C=.1, solver="saga", max_iter=1000,
            random_state=random_state, n_jobs=-1,
        ),
        "SVM (Linear)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=.1, max_iter=2000, random_state=random_state), cv=3
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", max_depth=10, min_samples_leaf=50, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50, class_weight="balanced",
            random_state=random_state, n_jobs=-1, max_features="sqrt",
        ),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=.1, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=.05, max_depth=5, subsample=.8,
            min_samples_leaf=50, random_state=random_state,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=.05, max_depth=6, subsample=.8,
            colsample_bytree=.8, scale_pos_weight=scale_pos_weight,
            eval_metric="auc", tree_method="hist", random_state=random_state, n_jobs=-1,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=.05, max_depth=7, num_leaves=63,
            subsample=.8, colsample_bytree=.8, class_weight="balanced",
            random_state=random_state, n_jobs=-1, verbose=-1,
        ),
    })


def train_models(models, X_train, y_train):
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained
