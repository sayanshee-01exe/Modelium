from __future__ import annotations

import numpy as np
import pandas as pd


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create credit-risk features using lending-domain relationships."""
    out = df.copy()
    eps = 1e-6

    def has(*cols: str) -> bool:
        return all(c in out.columns for c in cols)

    if has("AMT_ANNUITY", "AMT_INCOME_TOTAL"):
        out["DEBT_BURDEN_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_INCOME_TOTAL"] + eps)
    if has("AMT_CREDIT", "AMT_INCOME_TOTAL"):
        out["CREDIT_INCOME_RATIO"] = out["AMT_CREDIT"] / (out["AMT_INCOME_TOTAL"] + eps)
    if has("AMT_ANNUITY", "AMT_CREDIT"):
        out["ANNUITY_CREDIT_RATIO"] = out["AMT_ANNUITY"] / (out["AMT_CREDIT"] + eps)
        out["PAYMENT_RATE"] = out["ANNUITY_CREDIT_RATIO"]
    if has("AMT_GOODS_PRICE", "AMT_CREDIT"):
        out["GOODS_CREDIT_RATIO"] = out["AMT_GOODS_PRICE"] / (out["AMT_CREDIT"] + eps)
        out["CREDIT_GOODS_DIFF"] = out["AMT_CREDIT"] - out["AMT_GOODS_PRICE"]

    if "DAYS_BIRTH" in out:
        out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365.25
    if "DAYS_EMPLOYED" in out:
        out["DAYS_EMPLOYED_CLEAN"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
        out["YEARS_EMPLOYED"] = -out["DAYS_EMPLOYED_CLEAN"] / 365.25
    if has("DAYS_BIRTH", "DAYS_EMPLOYED_CLEAN"):
        out["EMPLOYMENT_LIFE_FRACTION"] = (
            -out["DAYS_EMPLOYED_CLEAN"] / (-out["DAYS_BIRTH"] + eps)
        )
    if "DAYS_REGISTRATION" in out:
        out["YEARS_SINCE_REGISTRATION"] = -out["DAYS_REGISTRATION"] / 365.25
    if "DAYS_ID_PUBLISH" in out:
        out["YEARS_SINCE_ID_CHANGE"] = -out["DAYS_ID_PUBLISH"] / 365.25

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in out]
    if len(ext_cols) >= 2:
        ext = out[ext_cols]
        out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        out["EXT_SOURCE_SUM"] = ext.sum(axis=1)
        out["EXT_SOURCE_MIN"] = ext.min(axis=1)
        out["EXT_SOURCE_MAX"] = ext.max(axis=1)
        out["EXT_SOURCE_STD"] = ext.std(axis=1)
        out["EXT_SOURCE_RANGE"] = out["EXT_SOURCE_MAX"] - out["EXT_SOURCE_MIN"]
        weights = {"EXT_SOURCE_1": .25, "EXT_SOURCE_2": .50, "EXT_SOURCE_3": .25}
        denom = sum(weights[c] for c in ext_cols)
        out["EXT_SOURCE_WEIGHTED"] = sum(out[c] * weights[c] for c in ext_cols) / denom

    if has("CNT_FAM_MEMBERS", "AMT_INCOME_TOTAL"):
        out["INCOME_PER_PERSON"] = out["AMT_INCOME_TOTAL"] / (out["CNT_FAM_MEMBERS"] + eps)
    if has("CNT_FAM_MEMBERS", "AMT_CREDIT"):
        out["CREDIT_PER_PERSON"] = out["AMT_CREDIT"] / (out["CNT_FAM_MEMBERS"] + eps)

    return out
