"""Step 2 — raw data validation.

These pin the data contract for the seven Home Credit tables. The point is to fail at
load time with a message naming the table and the offending values, rather than letting
bad data surface as a NaN or a silently-empty join hundreds of lines downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_validation import (
    INFERENCE_TABLES,
    REQUIRED_COLUMNS,
    validate_identifiers,
    validate_inference_tables,
    validate_not_empty,
    validate_raw_files,
    validate_raw_tables,
    validate_schema,
    validate_target,
)
from src.utils.exceptions import DataValidationError


@pytest.fixture
def valid_tables() -> dict[str, pd.DataFrame]:
    """A minimal but structurally valid stand-in for the seven raw tables."""
    return {
        "application_train": pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "AMT_INCOME_TOTAL": [100_000.0, 200_000.0, 150_000.0, 90_000.0],
            "AMT_CREDIT": [500_000.0, 600_000.0, 450_000.0, 300_000.0],
            "AMT_ANNUITY": [25_000.0, 30_000.0, 22_000.0, 18_000.0],
        }),
        "application_test": pd.DataFrame({
            "SK_ID_CURR": [5, 6],
            "AMT_INCOME_TOTAL": [110_000.0, 120_000.0],
            "AMT_CREDIT": [400_000.0, 420_000.0],
            "AMT_ANNUITY": [20_000.0, 21_000.0],
        }),
        "bureau": pd.DataFrame({"SK_ID_CURR": [1, 2], "SK_ID_BUREAU": [10, 11]}),
        "bureau_balance": pd.DataFrame({"SK_ID_BUREAU": [10, 11], "MONTHS_BALANCE": [-1, -2]}),
        "previous_application": pd.DataFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [100]}),
        "pos_cash": pd.DataFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [100]}),
        "credit_card": pd.DataFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [100]}),
        "installments": pd.DataFrame({"SK_ID_CURR": [1], "SK_ID_PREV": [100]}),
    }


# ------------------------------------------------------------------- happy path

def test_valid_tables_pass(valid_tables) -> None:
    report = validate_raw_tables(valid_tables)
    assert report["application_train"]["rows"] == 4
    assert report["bureau"]["rows"] == 2


def test_report_covers_every_table(valid_tables) -> None:
    report = validate_raw_tables(valid_tables)
    assert set(report) == set(valid_tables)


# ------------------------------------------------------------------ file checks

def test_missing_file_fails(tmp_path) -> None:
    (tmp_path / "application_train.csv").write_text("SK_ID_CURR\n1\n")
    with pytest.raises(DataValidationError) as exc:
        validate_raw_files(tmp_path, {
            "application_train": "application_train.csv",
            "bureau": "bureau.csv",
        })
    assert "bureau.csv" in str(exc.value)


def test_all_files_present_passes(tmp_path) -> None:
    for name in ("application_train.csv", "bureau.csv"):
        (tmp_path / name).write_text("x\n1\n")
    validate_raw_files(tmp_path, {"application_train": "application_train.csv",
                                  "bureau": "bureau.csv"})


def test_missing_file_error_lists_every_missing_path(tmp_path) -> None:
    """Reporting one missing file at a time makes fixing a fresh clone tedious."""
    with pytest.raises(DataValidationError) as exc:
        validate_raw_files(tmp_path, {"a": "a.csv", "b": "b.csv", "c": "c.csv"})
    message = str(exc.value)
    assert "a.csv" in message and "b.csv" in message and "c.csv" in message


# ----------------------------------------------------------------- table checks

def test_missing_table_fails(valid_tables) -> None:
    del valid_tables["bureau"]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "bureau" in str(exc.value)


def test_empty_dataframe_fails(valid_tables) -> None:
    valid_tables["bureau"] = valid_tables["bureau"].iloc[0:0]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "bureau" in str(exc.value)


def test_validate_not_empty_direct() -> None:
    with pytest.raises(DataValidationError):
        validate_not_empty(pd.DataFrame({"a": []}), "some_table")


def test_missing_required_column_fails(valid_tables) -> None:
    valid_tables["application_train"] = valid_tables["application_train"].drop(columns=["AMT_CREDIT"])
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "AMT_CREDIT" in str(exc.value)


def test_validate_schema_direct(valid_tables) -> None:
    df = valid_tables["bureau"].drop(columns=["SK_ID_BUREAU"])
    with pytest.raises(DataValidationError) as exc:
        validate_schema(df, "bureau")
    assert "SK_ID_BUREAU" in str(exc.value)


def test_required_columns_registry_covers_all_tables(valid_tables) -> None:
    assert set(REQUIRED_COLUMNS) >= set(valid_tables)


# ---------------------------------------------------------------- target checks

def test_target_with_nan_fails(valid_tables) -> None:
    valid_tables["application_train"]["TARGET"] = [0, 1, np.nan, 1]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "TARGET" in str(exc.value)


def test_target_outside_binary_fails(valid_tables) -> None:
    valid_tables["application_train"]["TARGET"] = [0, 1, 2, 1]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "2" in str(exc.value)


def test_target_single_class_fails(valid_tables) -> None:
    """One class means PR-AUC and stratified splitting are both undefined."""
    valid_tables["application_train"]["TARGET"] = [0, 0, 0, 0]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "class" in str(exc.value).lower()


def test_validate_target_direct_accepts_valid(valid_tables) -> None:
    validate_target(valid_tables["application_train"], "application_train")


# ------------------------------------------------------------ identifier checks

def test_duplicate_sk_id_curr_fails(valid_tables) -> None:
    """application_train must be one row per applicant, or the relational joins fan out."""
    valid_tables["application_train"]["SK_ID_CURR"] = [1, 1, 3, 4]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "duplicate" in str(exc.value).lower()


def test_null_sk_id_curr_fails(valid_tables) -> None:
    valid_tables["application_train"]["SK_ID_CURR"] = [1, 2, None, 4]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "SK_ID_CURR" in str(exc.value)


def test_null_sk_id_bureau_fails(valid_tables) -> None:
    valid_tables["bureau_balance"]["SK_ID_BUREAU"] = [10, None]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "SK_ID_BUREAU" in str(exc.value)


def test_duplicate_ids_allowed_in_child_tables(valid_tables) -> None:
    """bureau is one-to-many by design — duplicates here are correct, not an error."""
    valid_tables["bureau"] = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2], "SK_ID_BUREAU": [10, 11, 12],
    })
    validate_raw_tables(valid_tables)


def test_application_test_duplicate_sk_id_curr_fails(valid_tables) -> None:
    """application_test is also one-row-per-applicant, so scoring cannot double-count."""
    valid_tables["application_test"]["SK_ID_CURR"] = [5, 5]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "duplicate" in str(exc.value).lower()
    assert "application_test" in str(exc.value)


def test_application_test_null_sk_id_curr_fails(valid_tables) -> None:
    valid_tables["application_test"]["SK_ID_CURR"] = [5, None]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    assert "application_test" in str(exc.value)


def test_application_test_does_not_require_target(valid_tables) -> None:
    """The holdout applicants are unlabelled; requiring TARGET would reject valid data."""
    assert "TARGET" not in REQUIRED_COLUMNS["application_test"]
    assert "TARGET" not in valid_tables["application_test"].columns
    validate_raw_tables(valid_tables)


def test_application_test_may_be_absent(valid_tables) -> None:
    """The training path never consumes it, so its absence must not block training."""
    del valid_tables["application_test"]
    validate_raw_tables(valid_tables)


def test_validate_identifiers_direct(valid_tables) -> None:
    df = valid_tables["bureau"].copy()
    df.loc[0, "SK_ID_CURR"] = None
    with pytest.raises(DataValidationError):
        validate_identifiers(df, "bureau")


# ------------------------------------------------------- aggregated error report

def test_all_problems_reported_together(valid_tables) -> None:
    """Fixing one broken table at a time across a 2.7 GB dataset is slow; the
    orchestrator should surface every failure in a single raise."""
    valid_tables["application_train"] = valid_tables["application_train"].drop(columns=["AMT_CREDIT"])
    valid_tables["bureau"] = valid_tables["bureau"].iloc[0:0]
    with pytest.raises(DataValidationError) as exc:
        validate_raw_tables(valid_tables)
    message = str(exc.value)
    assert "AMT_CREDIT" in message
    assert "bureau" in message


# ------------------------------------------- Step 5 §11: the inference-side contract

def _inference_tables(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """What a scoring run actually loads: no application_train, no TARGET anywhere."""
    return {name: df for name, df in tables.items() if name in INFERENCE_TABLES}


def test_inference_tables_pass_without_application_train(valid_tables) -> None:
    """The training contract demands application_train and a two-class TARGET; a
    scoring run has neither, so it needs its own validator."""
    report = validate_inference_tables(_inference_tables(valid_tables))
    assert report["application_test"]["rows"] == 2
    assert "application_train" not in report


def test_inference_does_not_require_target(valid_tables) -> None:
    """§7: no TARGET column anywhere in the inference input, and that is not an error."""
    tables = _inference_tables(valid_tables)
    assert all("TARGET" not in df.columns for df in tables.values())
    validate_inference_tables(tables)


def test_inference_rejects_a_missing_required_table(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    del tables["bureau_balance"]
    with pytest.raises(DataValidationError, match="bureau_balance"):
        validate_inference_tables(tables)


def test_inference_rejects_missing_application_test(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    del tables["application_test"]
    with pytest.raises(DataValidationError, match="application_test"):
        validate_inference_tables(tables)


def test_inference_rejects_empty_application_test(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    tables["application_test"] = tables["application_test"].iloc[0:0]
    with pytest.raises(DataValidationError, match="empty"):
        validate_inference_tables(tables)


def test_inference_rejects_missing_sk_id_curr(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    tables["application_test"] = tables["application_test"].drop(columns=["SK_ID_CURR"])
    with pytest.raises(DataValidationError, match="SK_ID_CURR"):
        validate_inference_tables(tables)


def test_inference_rejects_null_sk_id_curr(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    tables["application_test"].loc[0, "SK_ID_CURR"] = np.nan
    with pytest.raises(DataValidationError, match="null"):
        validate_inference_tables(tables)


def test_inference_rejects_duplicate_sk_id_curr(valid_tables) -> None:
    """A duplicate applicant fans the joins out and silently multiplies predictions."""
    tables = _inference_tables(valid_tables)
    tables["application_test"] = pd.concat([tables["application_test"]] * 2, ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_inference_tables(tables)


def test_inference_rejects_missing_join_keys(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    tables["bureau"] = tables["bureau"].drop(columns=["SK_ID_BUREAU"])
    with pytest.raises(DataValidationError, match="SK_ID_BUREAU"):
        validate_inference_tables(tables)


def test_inference_reports_all_problems_together(valid_tables) -> None:
    tables = _inference_tables(valid_tables)
    tables["application_test"] = tables["application_test"].drop(columns=["AMT_CREDIT"])
    tables["installments"] = tables["installments"].iloc[0:0]
    with pytest.raises(DataValidationError) as exc:
        validate_inference_tables(tables)
    message = str(exc.value)
    assert "AMT_CREDIT" in message and "installments" in message
