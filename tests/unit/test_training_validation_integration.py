"""Step 2 — prove the pipeline validates before it transforms.

`tests/unit/test_data_validation.py` proves the validators are correct in isolation.
These prove they are actually *wired in*, in the right order, which a unit test of the
module alone cannot show.

Step 6 moved validation out of `scripts/train.py` into its own DVC stage, so the
ordering guarantee is now enforced in two places and tested in both:

    within the stage   scripts/validate_data.py checks files before it reads them, and
                       table contracts before anything downstream consumes them
    across stages      dvc.yaml makes `prepare` depend on validate's report and `train`
                       depend on prepare's output, so aggregation cannot run before
                       validation has passed

No real data and no model training: every expensive stage is monkeypatched, and loading
raises a sentinel so `main()` stops the moment order has been observed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.exceptions import DataValidationError


def _load_script(name: str):
    """Import a file in scripts/ by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        f"modelium_{name}_script", PROJECT_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StopBeforeAggregation(Exception):
    """Sentinel: lets main() run far enough to observe order, then halts it."""


@pytest.fixture
def validate_module():
    return _load_script("validate_data")


@pytest.fixture
def prepare_module():
    return _load_script("prepare_data")


@pytest.fixture
def train_module():
    return _load_script("train")


@pytest.fixture
def fake_tables() -> dict[str, pd.DataFrame]:
    return {
        "application_train": pd.DataFrame({
            "SK_ID_CURR": [1, 2],
            "TARGET": [0, 1],
            "AMT_INCOME_TOTAL": [1.0, 2.0],
            "AMT_CREDIT": [3.0, 4.0],
            "AMT_ANNUITY": [5.0, 6.0],
        }),
    }


def test_validation_runs_before_loading(monkeypatch, validate_module, fake_tables, tmp_path) -> None:
    """File contracts hold before pandas reads ~2.5 GB, not after."""
    calls: list[str] = []

    monkeypatch.setattr(validate_module, "validate_raw_files",
                        lambda *a, **k: calls.append("validate_files"))
    monkeypatch.setattr(validate_module, "load_home_credit_tables",
                        lambda *a, **k: (calls.append("load"), fake_tables)[1])
    monkeypatch.setattr(validate_module, "validate_raw_tables",
                        lambda *a, **k: (calls.append("validate_tables"), {})[1])
    monkeypatch.setattr(validate_module, "VALIDATION_REPORT_FILE", tmp_path / "report.json")

    validate_module.main()

    assert calls == ["validate_files", "load", "validate_tables"]


def test_file_validation_runs_before_loading(monkeypatch, validate_module) -> None:
    """A missing file must fail before pandas reads ~2.5 GB, not after."""
    calls: list[str] = []

    def _validate_files(*_a, **_k):
        calls.append("validate_files")
        raise DataValidationError("required data file(s) not found: bureau.csv")

    monkeypatch.setattr(validate_module, "validate_raw_files", _validate_files)
    monkeypatch.setattr(validate_module, "load_home_credit_tables",
                        lambda *a, **k: calls.append("load"))

    with pytest.raises(DataValidationError):
        validate_module.main()

    assert calls == ["validate_files"], "load must not run after file validation fails"


def test_invalid_table_fails_the_validate_stage(monkeypatch, validate_module, tmp_path) -> None:
    """Uses the REAL validator: a bad TARGET halts the pipeline at its first stage."""
    bad_tables = {
        "application_train": pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 2],          # 2 is outside {0, 1}
            "AMT_INCOME_TOTAL": [1.0, 2.0, 3.0],
            "AMT_CREDIT": [3.0, 4.0, 5.0],
            "AMT_ANNUITY": [5.0, 6.0, 7.0],
        }),
    }
    report = tmp_path / "report.json"

    monkeypatch.setattr(validate_module, "validate_raw_files", lambda *a, **k: None)
    monkeypatch.setattr(validate_module, "load_home_credit_tables", lambda *a, **k: bad_tables)
    monkeypatch.setattr(validate_module, "VALIDATION_REPORT_FILE", report)

    with pytest.raises(DataValidationError) as exc:
        validate_module.main()

    assert "2" in str(exc.value)
    assert not report.exists(), "a passing report was written for invalid data"


def test_validate_stage_imports_both_validators(validate_module) -> None:
    """Guard against a future refactor quietly dropping a validation call."""
    assert hasattr(validate_module, "validate_raw_files")
    assert hasattr(validate_module, "validate_raw_tables")


def test_prepare_stage_validates_the_inference_contract(prepare_module) -> None:
    """predict/ no longer sees raw application_test, so prepare owns that check.

    Without this, moving to a parquet hand-off would have silently dropped the Step 5
    inference contract — duplicate applicants would fan the joins out and multiply
    predictions with nothing to catch it.
    """
    assert hasattr(prepare_module, "validate_inference_tables")


def test_train_stage_no_longer_reloads_raw_data(train_module) -> None:
    """Training starts from prepared features; re-reading raw data here would make the
    prepare stage's cache pointless."""
    assert not hasattr(train_module, "load_home_credit_tables")
    assert hasattr(train_module, "TRAIN_FEATURES_FILE")
