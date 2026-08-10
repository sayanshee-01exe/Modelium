from __future__ import annotations

import numpy as np
import pandas as pd


def global_shap_importance(model, X, feature_names, sample_size: int = 2000, random_state: int = 42):
    import shap
    rng = np.random.default_rng(random_state)
    n = min(sample_size, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    sample = X[idx] if not hasattr(X, "iloc") else X.iloc[idx]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    )
