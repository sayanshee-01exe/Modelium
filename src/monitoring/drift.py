from __future__ import annotations

import numpy as np


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    eps = 1e-8
    breaks = np.unique(np.nanpercentile(expected, np.linspace(0, 100, buckets + 1)))
    if len(breaks) < 3:
        return 0.0
    exp = np.histogram(expected, bins=breaks)[0] + eps
    act = np.histogram(actual, bins=breaks)[0] + eps
    exp = exp / exp.sum(); act = act / act.sum()
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_status(value: float) -> str:
    if value > .20:
        return "RETRAIN"
    if value >= .10:
        return "MONITOR"
    return "STABLE"
