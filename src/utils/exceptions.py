"""Typed exceptions for the Modelium pipeline.

Every stage raises a typed error rather than returning None, printing a warning, or
letting bad input propagate — so a failure names its own cause instead of surfacing as
a NaN or an empty join several stages downstream.
"""

from __future__ import annotations


class ModeliumError(Exception):
    """Base class for every error this project raises deliberately.

    Catch this to distinguish a pipeline-detected problem from an unexpected crash.
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
