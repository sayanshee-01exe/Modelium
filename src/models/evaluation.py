from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, auc, classification_report, f1_score,
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


def classification_text(model, X_test, y_test) -> str:
    return classification_report(y_test, model.predict(X_test), target_names=["Non-Default", "Default"])
