from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def group_fairness_report(y_true, y_pred, groups, min_group_size: int = 30) -> pd.DataFrame:
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred); groups = np.asarray(groups)
    records = []
    overall_pred_positive = y_pred.mean()
    for group in pd.Series(groups).dropna().unique():
        mask = groups == group
        if mask.sum() < min_group_size:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        pred_positive = yp.mean()
        records.append({
            "group": str(group),
            "n": int(mask.sum()),
            "actual_default_rate": float(yt.mean()),
            "predicted_default_rate": float(pred_positive),
            "recall_tpr": float(recall_score(yt, yp, zero_division=0)),
            "fpr": float(fp / (fp + tn)) if fp + tn else 0.0,
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "disparate_impact_ratio": float(pred_positive / (overall_pred_positive + 1e-12)),
        })
    return pd.DataFrame(records)
