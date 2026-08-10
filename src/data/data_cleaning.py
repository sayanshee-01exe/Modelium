from __future__ import annotations

import numpy as np
import pandas as pd


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
