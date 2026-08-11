"""Raw data validation for the Home Credit relational tables.

Runs immediately after loading and before any cleaning or aggregation, so a bad input
fails with a message naming the table and the offending values — rather than surfacing
much later as a NaN column, an empty join, or a silently fanned-out row count.

Every check is available standalone, and `validate_raw_tables` runs all of them and
reports **every** failure in one raise. Fixing one problem per run is painful when a
single load of the real data is measured in gigabytes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns each table must carry for the pipeline to work at all. Deliberately a
# minimum contract, not the full schema: the raw tables have 3-122 columns each, and
# pinning all of them would break on any upstream addition without catching real faults.
REQUIRED_COLUMNS: dict[str, set[str]] = {
    "application_train": {"SK_ID_CURR", "TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"},
    "application_test": {"SK_ID_CURR", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"},
    "bureau": {"SK_ID_CURR", "SK_ID_BUREAU"},
    "bureau_balance": {"SK_ID_BUREAU"},
    "previous_application": {"SK_ID_CURR", "SK_ID_PREV"},
    "pos_cash": {"SK_ID_CURR", "SK_ID_PREV"},
    "credit_card": {"SK_ID_CURR", "SK_ID_PREV"},
    "installments": {"SK_ID_CURR", "SK_ID_PREV"},
}

# Join keys that must never be null. A null key silently drops the row from its merge.
IDENTIFIER_COLUMNS: dict[str, tuple[str, ...]] = {
    "application_train": ("SK_ID_CURR",),
    "application_test": ("SK_ID_CURR",),
    "bureau": ("SK_ID_CURR", "SK_ID_BUREAU"),
    "bureau_balance": ("SK_ID_BUREAU",),
    "previous_application": ("SK_ID_CURR", "SK_ID_PREV"),
    "pos_cash": ("SK_ID_CURR", "SK_ID_PREV"),
    "credit_card": ("SK_ID_CURR", "SK_ID_PREV"),
    "installments": ("SK_ID_CURR", "SK_ID_PREV"),
}

# Only the applicant tables are one-row-per-key. The child tables are one-to-many by
# design, so duplicate keys there are correct and must not be flagged.
UNIQUE_IDENTIFIERS: dict[str, str] = {
    "application_train": "SK_ID_CURR",
    "application_test": "SK_ID_CURR",
}

VALID_TARGET_VALUES = {0, 1}

# Tables batch inference needs. Same relational children as training — the aggregation
# is identical — but the applicant table is application_test and TARGET is absent by
# design, so `validate_raw_tables` (which demands application_train and a label) is the
# wrong contract here. application_train is deliberately excluded: scoring must not need
# the training applicants, and loading them would cost a 158 MB read for nothing.
INFERENCE_TABLES: tuple[str, ...] = (
    "application_test", "bureau", "bureau_balance", "previous_application",
    "pos_cash", "credit_card", "installments",
)


def validate_raw_files(data_dir: str | Path, file_map: dict[str, str]) -> None:
    """Check every expected CSV exists before anything tries to read one.

    Raises:
        DataValidationError: listing *all* missing paths, not just the first.
    """
    data_dir = Path(data_dir)
    missing = [str(data_dir / filename) for filename in file_map.values()
               if not (data_dir / filename).exists()]
    if missing:
        raise DataValidationError(
            f"{len(missing)} required data file(s) not found:\n  " + "\n  ".join(sorted(missing))
        )
    logger.info("File validation passed: %d files present in %s", len(file_map), data_dir)


def validate_not_empty(df: pd.DataFrame, name: str) -> None:
    """Reject an empty table: it produces an all-NaN left join, not a loud failure."""
    if df is None or len(df) == 0:
        raise DataValidationError(f"Table '{name}' is empty (0 rows)")


def validate_schema(df: pd.DataFrame, name: str, required_columns: set[str] | None = None) -> None:
    """Check the table carries its minimum required columns."""
    required = required_columns if required_columns is not None else REQUIRED_COLUMNS.get(name, set())
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Table '{name}' is missing required column(s): {sorted(missing)}"
        )


def validate_target(df: pd.DataFrame, name: str = "application_train",
                    target_col: str = "TARGET") -> None:
    """Check the label column is present, complete, binary, and has both classes."""
    if target_col not in df.columns:
        raise DataValidationError(f"Table '{name}' is missing the target column '{target_col}'")

    target = df[target_col]

    null_count = int(target.isna().sum())
    if null_count:
        raise DataValidationError(
            f"Table '{name}': target '{target_col}' has {null_count} missing value(s); "
            f"the label must be complete"
        )

    observed = set(target.unique())
    invalid = {v for v in observed if v not in VALID_TARGET_VALUES}
    if invalid:
        raise DataValidationError(
            f"Table '{name}': target '{target_col}' contains value(s) outside "
            f"{sorted(VALID_TARGET_VALUES)}: {sorted(invalid)}"
        )

    if len(observed) < 2:
        raise DataValidationError(
            f"Table '{name}': target '{target_col}' contains only one class "
            f"({sorted(observed)}); stratified splitting and PR-AUC are both undefined"
        )


def validate_identifiers(df: pd.DataFrame, name: str,
                         id_columns: tuple[str, ...] | None = None,
                         unique_column: str | None = None) -> None:
    """Check join keys are non-null, and unique where the table is one-row-per-key."""
    columns = id_columns if id_columns is not None else IDENTIFIER_COLUMNS.get(name, ())
    for column in columns:
        if column not in df.columns:
            raise DataValidationError(f"Table '{name}' is missing identifier column '{column}'")
        null_count = int(df[column].isna().sum())
        if null_count:
            raise DataValidationError(
                f"Table '{name}': identifier '{column}' has {null_count} null value(s); "
                f"null keys are silently dropped by the relational merges"
            )

    unique_col = unique_column if unique_column is not None else UNIQUE_IDENTIFIERS.get(name)
    if unique_col and unique_col in df.columns:
        duplicate_count = int(df[unique_col].duplicated().sum())
        if duplicate_count:
            raise DataValidationError(
                f"Table '{name}': identifier '{unique_col}' has {duplicate_count} duplicate "
                f"value(s); this table must be one row per key or the joins fan out"
            )


def validate_inference_tables(
    tables: dict[str, pd.DataFrame],
    *,
    application_table: str = "application_test",
) -> dict[str, dict]:
    """Validate the tables batch inference will score, and report all failures at once.

    Deliberately *not* `validate_raw_tables`: that one requires application_train and a
    complete, two-class TARGET, none of which exist at scoring time. Asking inference to
    satisfy the training contract would either fail on every run or force a fake label
    column into the input.

    The per-table checks themselves are reused rather than reimplemented, so the join
    keys, required columns and uniqueness rules cannot drift between the two contracts.

    Checks: every required table present and non-empty, required columns and join keys
    present, no null identifiers, and one row per applicant in `application_table`
    (a duplicate there fans the joins out and silently multiplies predictions).

    Args:
        tables: Logical table name -> DataFrame.
        application_table: Applicant-level table to score.

    Returns:
        Per-table ``{"rows": int, "columns": int}``.

    Raises:
        DataValidationError: with every problem found, one per line.
    """
    problems: list[str] = []
    report: dict[str, dict] = {}

    required = set(INFERENCE_TABLES) | {application_table}
    for name in sorted(required):
        if name not in tables:
            problems.append(f"Table '{name}' is required for inference but was not loaded")

    for name in sorted(required & set(tables)):
        df = tables[name]
        report[name] = {"rows": int(len(df)), "columns": int(df.shape[1])}
        for check in (
            lambda: validate_not_empty(df, name),
            lambda: validate_schema(df, name),
            lambda: validate_identifiers(df, name),
        ):
            try:
                check()
            except DataValidationError as err:
                problems.append(str(err))

    # No validate_target call anywhere in this function: TARGET is training-only.
    if problems:
        raise DataValidationError(
            f"Inference data validation failed with {len(problems)} problem(s):\n  "
            + "\n  ".join(problems)
        )

    logger.info(
        "Inference validation passed for %d table(s): %s",
        len(report),
        ", ".join(f"{n}({r['rows']:,}x{r['columns']})" for n, r in report.items()),
    )
    return report


def validate_raw_tables(tables: dict[str, pd.DataFrame], *,
                        target_col: str = "TARGET") -> dict[str, dict]:
    """Run every check against every table and report all failures at once.

    Args:
        tables: Mapping of logical table name to DataFrame, as returned by
            `load_home_credit_tables`.
        target_col: Label column, checked on `application_train` only.

    Returns:
        Per-table summary of ``{"rows": int, "columns": int}``, useful for logging.

    Raises:
        DataValidationError: with every problem found, one per line.
    """
    problems: list[str] = []
    report: dict[str, dict] = {}

    for name in REQUIRED_COLUMNS:
        if name not in tables:
            # application_test is registered but not required by the training path.
            if name == "application_test":
                continue
            problems.append(f"Table '{name}' is missing from the loaded tables")

    for name, df in tables.items():
        report[name] = {"rows": int(len(df)), "columns": int(df.shape[1])}
        for check in (
            lambda: validate_not_empty(df, name),
            lambda: validate_schema(df, name),
            lambda: validate_identifiers(df, name),
        ):
            try:
                check()
            except DataValidationError as err:
                problems.append(str(err))

        if name == "application_train":
            try:
                validate_target(df, name, target_col)
            except DataValidationError as err:
                problems.append(str(err))

    if problems:
        raise DataValidationError(
            f"Raw data validation failed with {len(problems)} problem(s):\n  "
            + "\n  ".join(problems)
        )

    logger.info(
        "Data validation passed for %d table(s): %s",
        len(report),
        ", ".join(f"{n}({r['rows']:,}x{r['columns']})" for n, r in report.items()),
    )
    return report
