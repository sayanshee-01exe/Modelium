from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay


def plot_model_evaluation(model, X_test, y_test, title: str = "Model"):
    """Create separate evaluation figures (cleaner than one giant notebook cell)."""
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax1, cmap="Blues")
    ax1.set_title(f"Confusion Matrix — {title}")

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax2)
    ax2.set_title(f"ROC Curve — {title}")

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax3)
    ax3.set_title(f"Precision-Recall Curve — {title}")
    return fig1, fig2, fig3
