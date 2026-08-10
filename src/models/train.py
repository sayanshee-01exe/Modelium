from __future__ import annotations

from collections import OrderedDict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
import xgboost as xgb


def build_candidate_models(random_state: int, scale_pos_weight: float = 1.0):
    """Return the baseline model portfolio used in the original notebook."""
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
