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


class ModelArtifactError(ModeliumError):
    """Raised when a model artifact cannot fulfil its inference contract.

    Covers a champion that is not a full preprocessing+model pipeline, and therefore
    could not score raw data after being reloaded.
    """
