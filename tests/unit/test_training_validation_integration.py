"""Step 2 — prove the training entry point validates before it transforms.

`tests/unit/test_data_validation.py` proves the validators are correct in isolation.
These prove they are actually *wired into* `scripts/train.py` in the right order, which
a unit test of the module alone cannot show.

No real data and no model training: every expensive stage is monkeypatched, and
aggregation raises a sentinel so `main()` stops the moment order has been observed.
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


def _load_train_module():
    """Import scripts/train.py by path — scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(
        "modelium_train_script", PROJECT_ROOT / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StopBeforeAggregation(Exception):
    """Sentinel: lets main() run far enough to observe order, then halts it."""


@pytest.fixture
def train_module():
    return _load_train_module()


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


def test_validation_runs_before_transformation(monkeypatch, train_module, fake_tables) -> None:
    """The whole point of Step 2: nothing expensive happens before the contracts hold."""
    calls: list[str] = []

    monkeypatch.setattr(train_module, "validate_raw_files",
                        lambda *a, **k: calls.append("validate_files"))
    monkeypatch.setattr(train_module, "load_home_credit_tables",
                        lambda *a, **k: (calls.append("load"), fake_tables)[1])
    monkeypatch.setattr(train_module, "validate_raw_tables",
                        lambda *a, **k: calls.append("validate_tables"))
    monkeypatch.setattr(train_module, "optimize_memory",
                        lambda df, *a, **k: (calls.append("optimize"), df)[1])

    def _aggregate(_tables):
        calls.append("aggregate")
        raise _StopBeforeAggregation

    monkeypatch.setattr(train_module, "build_relational_feature_table", _aggregate)

    with pytest.raises(_StopBeforeAggregation):
        train_module.main()

    assert calls == ["validate_files", "load", "validate_tables", "optimize", "aggregate"]


def test_file_validation_runs_before_loading(monkeypatch, train_module) -> None:
    """A missing file must fail before pandas reads ~7 GB, not after."""
    calls: list[str] = []

    def _validate_files(*_a, **_k):
        calls.append("validate_files")
        raise DataValidationError("required data file(s) not found: bureau.csv")

    monkeypatch.setattr(train_module, "validate_raw_files", _validate_files)
    monkeypatch.setattr(train_module, "load_home_credit_tables",
                        lambda *a, **k: calls.append("load"))

    with pytest.raises(DataValidationError):
        train_module.main()

    assert calls == ["validate_files"], "load must not run after file validation fails"


def test_invalid_table_stops_before_aggregation(monkeypatch, train_module) -> None:
    """Uses the REAL validator: bad TARGET must halt the run before any transformation."""
    calls: list[str] = []
    bad_tables = {
        "application_train": pd.DataFrame({
            "SK_ID_CURR": [1, 2, 3],
            "TARGET": [0, 1, 2],          # 2 is outside {0, 1}
            "AMT_INCOME_TOTAL": [1.0, 2.0, 3.0],
            "AMT_CREDIT": [3.0, 4.0, 5.0],
            "AMT_ANNUITY": [5.0, 6.0, 7.0],
        }),
    }

    monkeypatch.setattr(train_module, "validate_raw_files", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "load_home_credit_tables", lambda *a, **k: bad_tables)
    monkeypatch.setattr(train_module, "optimize_memory",
                        lambda df, *a, **k: (calls.append("optimize"), df)[1])
    monkeypatch.setattr(train_module, "build_relational_feature_table",
                        lambda *a, **k: calls.append("aggregate"))

    with pytest.raises(DataValidationError) as exc:
        train_module.main()

    assert "2" in str(exc.value)
    assert "optimize" not in calls, "memory optimization ran on invalid data"
    assert "aggregate" not in calls, "aggregation ran on invalid data"


def test_train_script_imports_both_validators(train_module) -> None:
    """Guard against a future refactor quietly dropping a validation call."""
    assert hasattr(train_module, "validate_raw_files")
    assert hasattr(train_module, "validate_raw_tables")
