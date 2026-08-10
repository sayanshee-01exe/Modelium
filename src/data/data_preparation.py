from __future__ import annotations

import pandas as pd


def aggregate_numeric_table(df: pd.DataFrame, group_key: str, prefix: str) -> pd.DataFrame:
    """Aggregate a one-to-many relational table to one row per group key."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    if group_key in numeric:
        numeric.remove(group_key)
    if not numeric:
        return df[[group_key]].drop_duplicates().reset_index(drop=True)

    agg_dict = {c: ["min", "max", "mean", "sum"] for c in numeric}
    agg_dict[numeric[0]].append("count")
    out = df.groupby(group_key).agg(agg_dict)
    out.columns = [f"{prefix}_{c}_{stat}" for c, stat in out.columns]
    return out.reset_index()


def build_relational_feature_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create one applicant-level table from the 7 Home Credit relations."""
    app = tables["application_train"].copy()
    bureau = tables["bureau"]
    bb = tables["bureau_balance"]

    bb_agg = aggregate_numeric_table(bb, "SK_ID_BUREAU", "bb")
    bureau_merged = bureau.merge(bb_agg, on="SK_ID_BUREAU", how="left")
    bureau_agg = aggregate_numeric_table(bureau_merged, "SK_ID_CURR", "bureau")

    rollups = [
        (bureau_agg, "bureau"),
        (aggregate_numeric_table(tables["previous_application"], "SK_ID_CURR", "prev"), "previous_application"),
        (aggregate_numeric_table(tables["pos_cash"], "SK_ID_CURR", "pos"), "pos_cash"),
        (aggregate_numeric_table(tables["credit_card"], "SK_ID_CURR", "cc"), "credit_card"),
        (aggregate_numeric_table(tables["installments"], "SK_ID_CURR", "inst"), "installments"),
    ]

    for agg_df, _ in rollups:
        app = app.merge(agg_df, on="SK_ID_CURR", how="left")
    return app
