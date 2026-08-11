from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, classification_report, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def get_probability_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = get_probability_scores(model, X_test)
    p, r, _ = precision_recall_curve(y_test, y_prob)
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "PR-AUC": auc(r, p),
    }
    return metrics


def evaluate_at_threshold(y_true, y_prob, threshold: float, model_name: str) -> dict:
    """Score probabilities at an externally chosen threshold.

    `evaluate_model` calls `model.predict()`, which hard-codes 0.5. Final test
    evaluation must instead apply the threshold that was frozen on the validation
    split, or the decision-dependent metrics describe a cut-off the pipeline never
    actually selected. ROC-AUC and PR-AUC are threshold-independent and are reported
    alongside for comparability with the validation leaderboard.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    p, r, _ = precision_recall_curve(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": model_name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": auc(r, p),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def classification_text(model, X_test, y_test) -> str:
    return classification_report(y_test, model.predict(X_test), target_names=["Non-Default", "Default"])
