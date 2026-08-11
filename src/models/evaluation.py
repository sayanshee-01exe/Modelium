from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

# Average Precision, not a trapezoidal integral of the PR curve. sklearn's
# `average_precision_score` is a step-wise sum that does not interpolate between
# operating points, so it does not overstate performance the way `auc(recall,
# precision)` can. It is also exactly what RandomizedSearchCV optimises via
# scoring="average_precision", so the model that wins the search is the model that
# wins selection.
PRIMARY_METRIC = "Average Precision"


def get_probability_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def evaluate_model(model, X_eval, y_eval, model_name: str) -> dict:
    y_pred = model.predict(X_eval)
    y_prob = get_probability_scores(model, X_eval)
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_eval, y_pred),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_eval, y_prob),
        PRIMARY_METRIC: average_precision_score(y_eval, y_prob),
    }
    return metrics


def evaluate_at_threshold(y_true, y_prob, threshold: float, model_name: str) -> dict:
    """Score probabilities at an externally chosen threshold.

    `evaluate_model` calls `model.predict()`, which hard-codes 0.5. Final test
    evaluation must instead apply the threshold that was frozen on the validation
    split, or the decision-dependent metrics describe a cut-off the pipeline never
    actually selected. ROC-AUC and Average Precision are threshold-independent and are reported
    alongside for comparability with the validation leaderboard.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": model_name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        PRIMARY_METRIC: average_precision_score(y_true, y_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def classification_text(model, X_eval, y_eval) -> str:
    return classification_report(y_eval, model.predict(X_eval), target_names=["Non-Default", "Default"])
