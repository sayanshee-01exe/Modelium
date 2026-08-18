"""Feature-level drift between a reference batch and a current batch.

Drift monitoring answers one question: *has the world the model was fitted on moved?*
It is measured on the raw feature frame, before preprocessing, because that is where a
change is interpretable — "AMT_INCOME_TOTAL shifted" is actionable, "column 287 of the
transformed matrix shifted" is not.

Two properties matter more than the choice of statistic:

*One bad feature must not lose the whole run.* With 407 features, something will always
be constant, all-null, or degenerate in a way that breaks a binning routine. Every
feature is scored inside its own guard and a failure is recorded as that feature's
status, so 406 usable results still reach the report.

*Reference defines the bins and the categories.* PSI compares a current distribution
against the reference's own quantile edges, and unseen-category rate is only meaningful
against the categories the reference actually contained. Re-deriving either from the
current batch would hide exactly the change being looked for.

Nothing here fits an encoder or a model. Drift is computed from the frames themselves.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Drift statuses, ordered by severity. `insufficient_data` and `error` are outcomes in
# their own right — reporting them is honest, silently scoring 0.0 would not be.
STABLE = "stable"
WARNING = "warning"
CRITICAL = "critical"
INSUFFICIENT = "insufficient_data"
ERROR = "error"

SEVERITY = {STABLE: 0, INSUFFICIENT: 1, ERROR: 1, WARNING: 2, CRITICAL: 3}

# Smoothing added to every bin before taking a log ratio. An empty bin on either side
# would otherwise make PSI infinite, which says "infinitely drifted" when it usually
# means "this bin was narrow".
EPS = 1e-8

# Below this many usable rows on either side, a distribution comparison describes
# sampling noise rather than drift.
MIN_ROWS = 30

# Effect-size floor for the KS test. A p-value answers "is there *any* difference",
# and at 10,000 rows per side it answers yes to differences far too small to act on:
# on a real run this alone flagged nine features whose largest CDF gap was under 0.05
# and whose PSI was below 0.011. The statistic is the effect size, so requiring both
# keeps "detectably different" from being reported as "meaningfully different".
MIN_KS_EFFECT = 0.05

# Unseen-category floor. A category absent from reference matters — one-hot encoding
# passes it through as all-zeros — but the reference is a *sample*, so a category that
# is merely rare can be missing from it by chance. One row in 10,000 is that artifact,
# not a schema change.
MIN_UNSEEN_CATEGORY_RATE = 0.005


def _clean_numeric(values: pd.Series) -> np.ndarray:
    """Finite float values only. Infinities are dropped, not clipped.

    An inf here comes from a ratio feature dividing by zero, and keeping it would put
    every other value into a single bin.
    """
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def compute_psi(reference: Sequence[float], current: Sequence[float],
                buckets: int = 10) -> float:
    """Population Stability Index using the reference's quantile bins.

    Returns:
        The PSI, or ``nan`` when it cannot be computed (either side empty). A constant
        reference returns 0.0: with a single distinct value there are no bins to shift
        between, so "no measurable drift" is the truthful answer rather than an error.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return float("nan")

    edges = np.unique(np.nanpercentile(reference, np.linspace(0, 100, buckets + 1)))
    if edges.size < 3:
        # Constant or near-constant reference: duplicated quantile edges collapse to
        # fewer than two usable bins.
        return 0.0

    # Open the outer edges so current values beyond the reference range are counted in
    # the end bins rather than dropped by np.histogram.
    edges[0], edges[-1] = -np.inf, np.inf

    expected = np.histogram(reference, bins=edges)[0].astype(float) + EPS
    actual = np.histogram(current, bins=edges)[0].astype(float) + EPS
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def jensen_shannon_distance(reference_probs: np.ndarray, current_probs: np.ndarray) -> float:
    """Symmetric, bounded distributional distance in [0, 1].

    Preferred over PSI as the reported `distribution_distance` because it is bounded, so
    a single value is comparable across features and across numeric/categorical types.
    """
    p = np.asarray(reference_probs, dtype=float) + EPS
    q = np.asarray(current_probs, dtype=float) + EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    divergence = 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))
    return float(np.sqrt(max(divergence, 0.0)))


def total_variation_distance(reference_probs: np.ndarray, current_probs: np.ndarray) -> float:
    """Half the L1 distance between two distributions, in [0, 1]."""
    p = np.asarray(reference_probs, dtype=float)
    q = np.asarray(current_probs, dtype=float)
    return float(0.5 * np.abs(p - q).sum())


def ks_two_sample(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value, or ``(nan, nan)`` on too few rows."""
    if reference.size < MIN_ROWS or current.size < MIN_ROWS:
        return float("nan"), float("nan")
    try:
        from scipy.stats import ks_2samp

        result = ks_2samp(reference, current)
        return float(result.statistic), float(result.pvalue)
    except Exception as err:                                     # pragma: no cover
        logger.debug("KS test failed: %s", err)
        return float("nan"), float("nan")


def numeric_feature_drift(reference: pd.Series, current: pd.Series,
                          buckets: int = 10) -> dict[str, Any]:
    """Drift measures for one numeric feature."""
    ref_clean, cur_clean = _clean_numeric(reference), _clean_numeric(current)

    row: dict[str, Any] = {
        "feature_type": "numeric",
        "reference_missing_rate": float(reference.isna().mean()),
        "current_missing_rate": float(current.isna().mean()),
        "reference_mean": float(np.mean(ref_clean)) if ref_clean.size else float("nan"),
        "current_mean": float(np.mean(cur_clean)) if cur_clean.size else float("nan"),
        "reference_std": float(np.std(ref_clean)) if ref_clean.size else float("nan"),
        "current_std": float(np.std(cur_clean)) if cur_clean.size else float("nan"),
        "reference_n": int(ref_clean.size),
        "current_n": int(cur_clean.size),
    }
    row["missing_rate_change"] = row["current_missing_rate"] - row["reference_missing_rate"]

    if ref_clean.size < MIN_ROWS or cur_clean.size < MIN_ROWS:
        row.update(psi=float("nan"), ks_statistic=float("nan"), ks_pvalue=float("nan"),
                   distribution_distance=float("nan"))
        return row

    row["psi"] = compute_psi(ref_clean, cur_clean, buckets)
    row["ks_statistic"], row["ks_pvalue"] = ks_two_sample(ref_clean, cur_clean)

    edges = np.unique(np.nanpercentile(ref_clean, np.linspace(0, 100, buckets + 1)))
    if edges.size >= 3:
        edges[0], edges[-1] = -np.inf, np.inf
        ref_hist = np.histogram(ref_clean, bins=edges)[0].astype(float)
        cur_hist = np.histogram(cur_clean, bins=edges)[0].astype(float)
        row["distribution_distance"] = jensen_shannon_distance(ref_hist, cur_hist)
    else:
        row["distribution_distance"] = 0.0
    return row


def categorical_feature_drift(reference: pd.Series, current: pd.Series) -> dict[str, Any]:
    """Drift measures for one categorical feature, with categories taken from reference.

    `unseen_category_rate` is the share of current rows whose value never appeared in the
    reference. It is reported separately from the distance because it means something
    different: not "the mix changed" but "a value the model has never been fitted on is
    now arriving", which one-hot encoding will silently pass through as all-zeros.
    """
    ref_values = reference.dropna().astype(str)
    cur_values = current.dropna().astype(str)

    row: dict[str, Any] = {
        "feature_type": "categorical",
        "reference_missing_rate": float(reference.isna().mean()),
        "current_missing_rate": float(current.isna().mean()),
        "reference_n": int(ref_values.size),
        "current_n": int(cur_values.size),
        "reference_categories": int(ref_values.nunique()),
        "current_categories": int(cur_values.nunique()),
        "reference_mean": float("nan"), "current_mean": float("nan"),
        "reference_std": float("nan"), "current_std": float("nan"),
        "ks_statistic": float("nan"), "ks_pvalue": float("nan"),
    }
    row["missing_rate_change"] = row["current_missing_rate"] - row["reference_missing_rate"]

    if ref_values.empty or cur_values.empty:
        row.update(psi=float("nan"), distribution_distance=float("nan"),
                   unseen_category_rate=float("nan"))
        return row

    known = set(ref_values.unique())
    row["unseen_category_rate"] = float((~cur_values.isin(known)).mean())

    # Align both distributions over the union so unseen categories contribute rather
    # than being dropped by a reference-only index.
    categories = sorted(known | set(cur_values.unique()))
    ref_probs = ref_values.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    cur_probs = cur_values.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()

    row["distribution_distance"] = jensen_shannon_distance(ref_probs, cur_probs)
    row["total_variation_distance"] = total_variation_distance(ref_probs, cur_probs)

    smoothed_ref, smoothed_cur = ref_probs + EPS, cur_probs + EPS
    smoothed_ref /= smoothed_ref.sum()
    smoothed_cur /= smoothed_cur.sum()
    row["psi"] = float(np.sum((smoothed_cur - smoothed_ref) * np.log(smoothed_cur / smoothed_ref)))
    return row


def classify_drift(row: dict[str, Any], thresholds: dict[str, float]) -> tuple[str, str]:
    """Turn the measures for one feature into a status and a human-readable reason.

    PSI leads because it is the measure the thresholds are calibrated on. The KS p-value
    is a secondary signal and deliberately cannot raise a feature to `critical` on its
    own: at 10,000 rows per side it rejects equality for differences far too small to
    matter, so treating it as decisive would mark most features drifted every run.
    """
    psi = row.get("psi", float("nan"))
    if row.get("reference_n", 0) < MIN_ROWS or row.get("current_n", 0) < MIN_ROWS:
        return INSUFFICIENT, (
            f"fewer than {MIN_ROWS} usable rows "
            f"(reference {row.get('reference_n', 0)}, current {row.get('current_n', 0)})"
        )
    if not np.isfinite(psi):
        return INSUFFICIENT, "PSI could not be computed"

    reasons: list[str] = []
    status = STABLE
    if psi >= thresholds["psi_critical_threshold"]:
        status = CRITICAL
        reasons.append(f"PSI {psi:.3f} >= critical {thresholds['psi_critical_threshold']}")
    elif psi >= thresholds["psi_warning_threshold"]:
        status = WARNING
        reasons.append(f"PSI {psi:.3f} >= warning {thresholds['psi_warning_threshold']}")

    # Significance AND effect size, never significance alone.
    ks_p = row.get("ks_pvalue", float("nan"))
    ks_stat = row.get("ks_statistic", float("nan"))
    if (np.isfinite(ks_p) and ks_p < thresholds["ks_pvalue_threshold"]
            and np.isfinite(ks_stat) and ks_stat >= MIN_KS_EFFECT):
        reasons.append(f"KS statistic {ks_stat:.3f} (p={ks_p:.2e})")
        if status == STABLE:
            status = WARNING

    unseen = row.get("unseen_category_rate", float("nan"))
    if np.isfinite(unseen) and unseen >= MIN_UNSEEN_CATEGORY_RATE:
        reasons.append(f"{unseen:.1%} of rows carry categories absent from reference")
        if status == STABLE:
            status = WARNING

    change = row.get("missing_rate_change", 0.0)
    if np.isfinite(change) and abs(change) >= 0.05:
        reasons.append(f"missing rate moved {change:+.1%}")
        if status == STABLE:
            status = WARNING

    return status, "; ".join(reasons) if reasons else "within thresholds"


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    thresholds: dict[str, float],
    buckets: int = 10,
) -> pd.DataFrame:
    """Score every feature, ordered most-drifted first.

    Neither input frame is modified. A feature that raises is recorded with status
    ``error`` and the exception text, so one degenerate column cannot cost the other 406.
    """
    records: list[dict[str, Any]] = []
    typed = [(c, "numeric") for c in numeric_columns] + \
            [(c, "categorical") for c in categorical_columns]

    for column, kind in typed:
        if column not in reference.columns or column not in current.columns:
            records.append({"feature": column, "feature_type": kind,
                            "drift_status": INSUFFICIENT,
                            "drift_reason": "absent from reference or current batch"})
            continue
        try:
            if kind == "numeric":
                row = numeric_feature_drift(reference[column], current[column], buckets)
            else:
                row = categorical_feature_drift(reference[column], current[column])
            status, reason = classify_drift(row, thresholds)
            row.update(feature=column, drift_status=status, drift_reason=reason)
        except Exception as err:
            logger.warning("Could not score drift for %r: %s", column, err)
            row = {"feature": column, "feature_type": kind, "drift_status": ERROR,
                   "drift_reason": f"{type(err).__name__}: {err}"}
        records.append(row)

    frame = pd.DataFrame.from_records(records)
    ordered = ["feature", "feature_type", "psi", "ks_statistic", "ks_pvalue",
               "distribution_distance", "reference_missing_rate", "current_missing_rate",
               "drift_status", "drift_reason"]
    for column in ordered:
        if column not in frame.columns:
            frame[column] = np.nan
    remaining = [c for c in frame.columns if c not in ordered]
    frame = frame[ordered + remaining]

    frame["_severity"] = frame["drift_status"].map(SEVERITY).fillna(0)
    frame = frame.sort_values(
        ["_severity", "psi"], ascending=[False, False], kind="mergesort", na_position="last"
    ).drop(columns="_severity").reset_index(drop=True)
    return frame


def summarise_drift(frame: pd.DataFrame, max_drifted_ratio: float) -> dict[str, Any]:
    """Counts, the drifted ratio, and the overall data-drift status."""
    counts = frame["drift_status"].value_counts().to_dict()
    monitored = int(len(frame))
    critical = int(counts.get(CRITICAL, 0))
    warning = int(counts.get(WARNING, 0))
    drifted_ratio = (critical + warning) / monitored if monitored else 0.0

    if critical > 0:
        status = CRITICAL
    elif drifted_ratio > max_drifted_ratio:
        status = CRITICAL
    elif warning > 0:
        status = WARNING
    else:
        status = STABLE

    return {
        "monitored_features": monitored,
        "stable_features": int(counts.get(STABLE, 0)),
        "warning_features": warning,
        "critical_features": critical,
        "insufficient_features": int(counts.get(INSUFFICIENT, 0)),
        "error_features": int(counts.get(ERROR, 0)),
        "drifted_feature_ratio": float(drifted_ratio),
        "max_drifted_feature_ratio": float(max_drifted_ratio),
        "status": status,
        "top_drifted": frame.head(10)[
            ["feature", "feature_type", "psi", "drift_status", "drift_reason"]
        ].to_dict(orient="records"),
    }
