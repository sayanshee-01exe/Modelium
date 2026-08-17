"""Typed exceptions for the Modelium pipeline.

Every stage raises a typed error rather than returning None, printing a warning, or
letting bad input propagate — so a failure names its own cause instead of surfacing as
a NaN or an empty join several stages downstream.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


class ModeliumError(Exception):
    """Base class for every error this project raises deliberately.

    Catch this to distinguish a pipeline-detected problem from an unexpected crash.
    """


class ConfigurationError(ModeliumError):
    """Raised when `params.yaml` is absent, malformed, or holds an unusable value.

    Caught at load rather than at use: a `cv_folds: 1` surfaces here in milliseconds
    naming the field, instead of minutes into a run as an opaque split error.
    """


class DataValidationError(ModeliumError):
    """Raised when raw or intermediate data violates its expected contract.

    Covers missing files, missing or empty tables, absent required columns, invalid
    target values, and null or duplicated join keys.
    """


class InferenceSchemaError(DataValidationError):
    """Raised when scoring input cannot be reconciled with the training feature schema.

    A subclass of `DataValidationError` so callers that already catch data-contract
    failures keep working, while code that cares specifically about training/serving
    schema drift can catch this alone.

    Missing columns are fatal rather than imputed: a column the model was trained on but
    which inference cannot supply means the two sides disagree about what the features
    are, and filling it with a training median would hide that behind a plausible-looking
    prediction.
    """


class ModelArtifactError(ModeliumError):
    """Raised when a model artifact cannot fulfil its inference contract.

    Covers a champion that is not a full preprocessing+model pipeline, and therefore
    could not score raw data after being reloaded.
    """


class PipelineStageError(ModeliumError):
    """Raised when a DVC pipeline stage fails during orchestration.

    Wraps the underlying exception with the stage name and a human-readable reason,
    so the DVC log names which stage broke and why — not just a raw traceback from
    deep inside a library call.
    """

    def __init__(self, stage: str, reason: str, cause: Exception | None = None):
        self.stage = stage
        self.reason = reason
        msg = f"Stage '{stage}' failed: {reason}"
        super().__init__(msg)
        if cause is not None:
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# DataFrame validation helper
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_columns: Sequence[str] | None = None,
) -> None:
    """Raise ``DataValidationError`` if *df* is empty or missing required columns.

    This is the guard that sits at the top of every function that receives a
    DataFrame: cheaper to fail here with a clear message than to let an empty
    frame silently produce an empty output two stages later.

    Args:
        df:               The DataFrame to check.
        name:             Human-readable label used in error messages (e.g.
                          ``"application_train"``).
        required_columns: Optional iterable of column names that must be present.

    Raises:
        DataValidationError: with all violations described in a single raise.
    """
    errors: list[str] = []

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            f"'{name}' is not a DataFrame — got {type(df).__name__}"
        )

    if df.empty:
        errors.append(f"'{name}' is empty (0 rows)")

    if required_columns is not None:
        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            errors.append(
                f"'{name}' is missing {len(missing)} required column(s): {missing}"
            )

    if errors:
        raise DataValidationError("; ".join(errors))
