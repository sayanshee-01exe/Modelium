from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.exceptions import DataValidationError


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


def build_relational_feature_table(
    tables: dict[str, pd.DataFrame],
    application_table: str = "application_train",
) -> pd.DataFrame:
    """Create one applicant-level table from the Home Credit relations.

    The applicant table is a parameter so training and inference share **one**
    implementation: ``application_train`` carries TARGET, ``application_test`` does not,
    but the aggregation and joins are identical either way. A second copy of this
    function for inference is how training/serving skew starts — the two drift, and the
    model is then scored on features it was never trained on.

    Nothing here reads TARGET, so the label is not required. It is simply carried
    through as a column when the applicant table happens to have one.

    Args:
        tables: Logical table name -> DataFrame, as returned by `load_home_credit_tables`.
        application_table: Key of the applicant-level table to build features for.

    Returns:
        One row per applicant, with the child tables aggregated and left-joined on.

    Raises:
        DataValidationError: if `application_table` is absent from `tables`.
    """
    if application_table not in tables:
        raise DataValidationError(
            f"Applicant table '{application_table}' is not among the loaded tables "
            f"({sorted(tables)}); cannot build the relational feature table"
        )

    app = tables[application_table].copy()
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


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split into three disjoint sets so the test data never informs model selection.

    The pipeline previously made a single train/test split and then used that one
    holdout for model comparison, champion selection, threshold tuning *and* final
    reporting — so the reported metrics were optimistically biased. The three-way
    split gives each concern its own data:

        train      fit the preprocessor and the candidate models
        validation compare models, tune hyperparameters, pick the threshold
        test       scored exactly once, after everything is frozen

    ``validation_size`` and ``test_size`` are fractions of the **original** frame,
    not of what remains after the test holdout.

    Args:
        X: Feature frame.
        y: Target aligned to ``X`` by position and index.
        validation_size: Fraction of the original rows held out for validation.
        test_size: Fraction of the original rows held out for the final test.
        random_state: Seed, so a rerun reproduces the same three sets.
        stratify: Preserve the class balance in every split. TARGET is ~8% positive,
            and an unstratified split can leave a small split with too few positives
            for PR-AUC to mean anything.

    Returns:
        ``(X_train, X_val, X_test, y_train, y_val, y_test)``

    Raises:
        ValueError: If the inputs are empty or misaligned, if either size is outside
            (0, 1), or if the two sizes leave no training data.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if len(X) == 0:
        raise ValueError("Cannot split an empty dataset")

    for name, size in (("validation_size", validation_size), ("test_size", test_size)):
        if not 0.0 < size < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {size}")
    if validation_size + test_size >= 1.0:
        raise ValueError(
            f"validation_size + test_size must be < 1 to leave training data, "
            f"got {validation_size} + {test_size} = {validation_size + test_size}"
        )

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    # validation_size is expressed against the original row count, so rescale it
    # against what actually survived the test holdout.
    val_fraction_of_rest = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest,
        test_size=val_fraction_of_rest,
        random_state=random_state,
        stratify=y_rest if stratify else None,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
