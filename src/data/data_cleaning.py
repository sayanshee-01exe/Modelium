"""Data cleaning for the Home Credit application tables.

Each transformation encodes a specific EDA finding — the finding is named in the code
comment above it so a reviewer can trace the decision. Nothing here is aesthetic:
every change either fixes a data error, caps an outlier that would dominate a scaler,
or separates a sentinel from real values.

The cleaning runs *before* relational aggregation, so the child tables join onto an
application table that has already been corrected. It works identically for
`application_train` and `application_test`; the only difference is the
`missing_threshold` column-drop, which is training-time only.

Existing functions (``optimize_memory``, ``basic_checks``, ``drop_low_information_columns``)
are preserved for use by downstream modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Existing utility functions — used by prepare_data.py and other consumers
# ---------------------------------------------------------------------------

def optimize_memory(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns while preserving values."""
    df = df.copy()
    start = df.memory_usage(deep=True).sum() / 1024**2

    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        cmin, cmax = df[col].min(), df[col].max()
        if cmin >= np.iinfo(np.int8).min and cmax <= np.iinfo(np.int8).max:
            df[col] = df[col].astype(np.int8)
        elif cmin >= np.iinfo(np.int16).min and cmax <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
        elif cmin >= np.iinfo(np.int32).min and cmax <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype(np.float32)

    if verbose:
        end = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100 * (start - end) / max(start, 1e-12)
        print(f"Memory: {start:.1f} MB -> {end:.1f} MB ({reduction:.1f}% reduction)")
    return df


def basic_checks(df: pd.DataFrame, name: str) -> dict:
    """Return a compact data-quality report instead of only printing it."""
    missing = df.isna().sum()
    missing = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing.div(len(df)).mul(100),
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)

    return {
        "name": name,
        "shape": df.shape,
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "dtypes": df.dtypes.value_counts().astype(int).to_dict(),
        "missing": missing,
    }


def drop_low_information_columns(
    df: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
    target_col: str = "TARGET",
) -> tuple[pd.DataFrame, list[str]]:
    constant = [c for c in df.columns if df[c].dropna().nunique() <= 1]
    id_like = [c for c in df.columns if "SK_ID" in c and c not in {id_col, target_col}]
    drop_cols = sorted(set(constant + id_like))
    return df.drop(columns=drop_cols, errors="ignore").copy(), drop_cols


# ---------------------------------------------------------------------------
# EDA-driven application table cleaning
# ---------------------------------------------------------------------------

# Columns whose names start with "DAYS_" store negative-day counts from the
# application date. Flipping them positive makes downstream ratios and
# log-transforms work without sign confusion.
DAYS_COLUMNS = [
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
]

# EDA finding: DAYS_EMPLOYED == 365243 is a sentinel for "not employed"
# (~55,374 rows in application_train). The value is 1000 years and would
# dominate any scaler; it must be separated from real employment durations.
DAYS_EMPLOYED_SENTINEL = 365243

# Columns explicitly protected from the missing-threshold drop.
# EDA finding: EXT_SOURCE_1 is ~56% missing but is one of the strongest
# predictors of default. Dropping it loses signal that cannot be recovered
# from the other sources.
PROTECTED_COLUMNS = frozenset({"EXT_SOURCE_1"})


def clean_application_data(
    df: pd.DataFrame,
    *,
    missing_threshold: float = 0.70,
    income_cap_quantile: float = 0.999,
    children_cap: int = 10,
    drop_high_missing: bool = True,
) -> pd.DataFrame:
    """Apply EDA-driven cleaning to the application table.

    Each step has a code comment naming the EDA finding it addresses.

    Args:
        df:                 The raw application table (train or test).
        missing_threshold:  Fraction (0–1). Columns with more than this share of
                            missing values are dropped, unless protected.
        income_cap_quantile: Quantile for capping ``AMT_INCOME_TOTAL``.
        children_cap:       Hard cap for ``CNT_CHILDREN``.
        drop_high_missing:  If False, skip the missing-threshold column drop
                            (used when cleaning application_test, where the
                            training schema governs column selection).

    Returns:
        Cleaned DataFrame (copy, original untouched).
    """
    out = df.copy()
    n_before = len(out)
    logger.info("Cleaning application data: %d rows x %d columns", *out.shape)

    # ----- EDA: DAYS_EMPLOYED == 365243 is a sentinel for "not employed" -----
    # ~55,374 rows in application_train. Replace with NaN so it does not
    # distort statistics, and create a binary flag that preserves the signal.
    if "DAYS_EMPLOYED" in out.columns:
        sentinel_mask = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
        n_sentinel = int(sentinel_mask.sum())
        out["DAYS_EMPLOYED_ANOM"] = sentinel_mask.astype(int)
        out.loc[sentinel_mask, "DAYS_EMPLOYED"] = np.nan
        logger.info(
            "DAYS_EMPLOYED: %d sentinel values (365243) -> NaN + DAYS_EMPLOYED_ANOM flag",
            n_sentinel,
        )

    # ----- EDA: DAYS_* columns are stored as negative day counts ----------
    # Flip to positive so ratios and log transforms are meaningful.
    for col in DAYS_COLUMNS:
        if col in out.columns:
            # Only flip values that are actually negative (skip NaN and already-positive)
            mask = out[col] < 0
            if mask.any():
                out.loc[mask, col] = out.loc[mask, col].abs()

    # ----- EDA: CODE_GENDER == 'XNA' is a rare miscoding (4 rows) ---------
    # Replace with the training-set mode rather than dropping — 4 rows is
    # too few to affect the distribution but enough to crash OneHotEncoder
    # if it sees an unseen category at inference.
    if "CODE_GENDER" in out.columns:
        xna_mask = out["CODE_GENDER"] == "XNA"
        n_xna_gender = int(xna_mask.sum())
        if n_xna_gender > 0:
            mode_gender = out.loc[~xna_mask, "CODE_GENDER"].mode().iloc[0]
            out.loc[xna_mask, "CODE_GENDER"] = mode_gender
            logger.info(
                "CODE_GENDER: %d 'XNA' values -> '%s' (mode)", n_xna_gender, mode_gender,
            )

    # ----- EDA: ORGANIZATION_TYPE == 'XNA' is NOT an error ----------------
    # It correlates exactly with the DAYS_EMPLOYED_ANOM population (55,374
    # rows, 100% overlap). These are the "not employed" applicants — their
    # organisation type is legitimately unknown. Left unchanged deliberately.

    # ----- EDA: OCCUPATION_TYPE NaN has two distinct populations -----------
    # ~55,372 are the "not employed" sentinel group (DAYS_EMPLOYED_ANOM == 1),
    # ~41,019 are employed but did not report their occupation. Blanket mode
    # imputation would merge them; splitting preserves a real signal.
    if "OCCUPATION_TYPE" in out.columns and "DAYS_EMPLOYED_ANOM" in out.columns:
        occ_null = out["OCCUPATION_TYPE"].isna()
        not_employed = occ_null & (out["DAYS_EMPLOYED_ANOM"] == 1)
        unknown_employed = occ_null & (out["DAYS_EMPLOYED_ANOM"] == 0)
        out.loc[not_employed, "OCCUPATION_TYPE"] = "Not_Employed"
        out.loc[unknown_employed, "OCCUPATION_TYPE"] = "Unknown"
        logger.info(
            "OCCUPATION_TYPE: %d NaN -> 'Not_Employed', %d NaN -> 'Unknown'",
            int(not_employed.sum()), int(unknown_employed.sum()),
        )

    # ----- EDA: AMT_INCOME_TOTAL has extreme outliers ----------------------
    # The 117,000,000 outlier is > 1000x the median (~147,150). Cap at the
    # train p99.9 so it does not dominate standard scaling.
    if "AMT_INCOME_TOTAL" in out.columns:
        income_cap = out["AMT_INCOME_TOTAL"].quantile(income_cap_quantile)
        n_capped_income = int((out["AMT_INCOME_TOTAL"] > income_cap).sum())
        out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"].clip(upper=income_cap)
        logger.info(
            "AMT_INCOME_TOTAL: %d values capped at p%.1f (%.0f)",
            n_capped_income, income_cap_quantile * 100, income_cap,
        )

    # ----- EDA: CNT_CHILDREN has implausible values -----------------------
    # Max is 19 in application_train; anything > 10 is suspect and gets
    # capped to prevent it from creating a near-empty one-hot bin.
    if "CNT_CHILDREN" in out.columns:
        n_capped_children = int((out["CNT_CHILDREN"] > children_cap).sum())
        out["CNT_CHILDREN"] = out["CNT_CHILDREN"].clip(upper=children_cap)
        if n_capped_children > 0:
            logger.info(
                "CNT_CHILDREN: %d values capped at %d", n_capped_children, children_cap,
            )

    # ----- EDA: columns with >threshold missing add noise, not signal ------
    # Exception: EXT_SOURCE_1 (~56% missing) is one of the strongest
    # individual predictors and is explicitly protected.
    if drop_high_missing:
        n_rows = len(out)
        missing_frac = out.isna().mean()
        high_missing = [
            c for c in missing_frac.index
            if missing_frac[c] > missing_threshold and c not in PROTECTED_COLUMNS
        ]
        if high_missing:
            out = out.drop(columns=high_missing)
            logger.info(
                "Dropped %d columns with >%.0f%% missing: %s",
                len(high_missing), missing_threshold * 100, sorted(high_missing),
            )

    logger.info(
        "Cleaning complete: %d rows x %d columns (started %d x %d)",
        len(out), out.shape[1], n_before, df.shape[1],
    )
    return out


# ---------------------------------------------------------------------------
# Standalone entry point — not a DVC stage, but useful for debugging.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))

    from config.config import DATA_DIR, DATA_FILES

    INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = DATA_DIR / DATA_FILES["application_train"]
    if not raw_path.exists():
        logger.error("Raw file not found: %s", raw_path)
        sys.exit(1)

    raw = pd.read_csv(raw_path)
    logger.info("Loaded %s: %d rows x %d columns", raw_path.name, *raw.shape)

    cleaned = clean_application_data(raw)

    out_path = INTERIM_DIR / "application_clean.parquet"
    cleaned.to_parquet(out_path, index=False)
    logger.info("Wrote cleaned data to %s", out_path)
