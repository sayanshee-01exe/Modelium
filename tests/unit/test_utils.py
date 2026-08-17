"""Tests for src/utils/logger.py and src/utils/exceptions.py.

These tests validate:
- Logger configuration (console + file handlers, format, levels)
- Exception hierarchy and custom attributes
- validate_dataframe() helper behaviour
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_logger_state(monkeypatch):
    """Reset the module-level ``_configured`` flag before each test.

    Without this, a logger configured by an earlier test would cause
    ``get_logger`` to skip ``_configure_root_logger``, making handler-count
    assertions unpredictable.
    """
    import src.utils.logger as logger_mod

    monkeypatch.setattr(logger_mod, "_configured", False)

    root = logging.getLogger("modelium")
    root.handlers.clear()
    root.setLevel(logging.WARNING)  # stdlib default

    yield

    root.handlers.clear()


# ========================================================================
# Logger tests
# ========================================================================

class TestGetLogger:
    """Test ``get_logger`` returns properly configured loggers."""

    def test_returns_logger_instance(self):
        from src.utils.logger import get_logger

        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_name_prefixed_with_modelium(self):
        from src.utils.logger import get_logger

        logger = get_logger("some.module")
        assert logger.name.startswith("modelium.")

    def test_modelium_prefix_not_doubled(self):
        from src.utils.logger import get_logger

        logger = get_logger("modelium.train")
        assert logger.name == "modelium.train"

    def test_root_has_two_handlers(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")
        assert len(root.handlers) == 2

    def test_root_has_stream_handler(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_root_has_rotating_file_handler(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1

    def test_root_level_is_debug(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")
        assert root.level == logging.DEBUG

    def test_console_handler_level_is_info(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert stream_handlers[0].level == logging.INFO

    def test_file_handler_level_is_debug(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")
        file_handlers = [
            h for h in root.handlers
            if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert file_handlers[0].level == logging.DEBUG

    def test_log_directory_created(self):
        from src.utils.logger import LOG_DIR, get_logger

        get_logger("test_module")
        assert LOG_DIR.is_dir()

    def test_log_file_created_on_write(self, tmp_path, monkeypatch):
        """Verify that writing a log message creates the pipeline.log file."""
        import src.utils.logger as logger_mod

        log_dir = tmp_path / "logs"
        monkeypatch.setattr(logger_mod, "LOG_DIR", log_dir)
        monkeypatch.setattr(logger_mod, "LOG_FILE", log_dir / "pipeline.log")

        logger = logger_mod.get_logger("test_write")
        logger.info("test message")

        assert (log_dir / "pipeline.log").exists()

    def test_no_duplicate_handlers_on_repeated_calls(self):
        from src.utils.logger import get_logger

        get_logger("first")
        get_logger("second")
        get_logger("third")

        root = logging.getLogger("modelium")
        assert len(root.handlers) == 2

    def test_formatter_contains_expected_fields(self):
        from src.utils.logger import get_logger

        get_logger("test_module")
        root = logging.getLogger("modelium")

        for handler in root.handlers:
            fmt = handler.formatter._fmt
            assert "%(asctime)s" in fmt
            assert "%(name)s" in fmt
            assert "%(levelname)s" in fmt
            assert "%(message)s" in fmt


# ========================================================================
# Exception hierarchy tests
# ========================================================================

class TestExceptionHierarchy:
    """Ensure the exception tree is structured correctly."""

    def test_modelium_error_is_exception(self):
        from src.utils.exceptions import ModeliumError
        assert issubclass(ModeliumError, Exception)

    def test_configuration_error_is_modelium_error(self):
        from src.utils.exceptions import ConfigurationError, ModeliumError
        assert issubclass(ConfigurationError, ModeliumError)

    def test_data_validation_error_is_modelium_error(self):
        from src.utils.exceptions import DataValidationError, ModeliumError
        assert issubclass(DataValidationError, ModeliumError)

    def test_inference_schema_error_is_data_validation_error(self):
        from src.utils.exceptions import DataValidationError, InferenceSchemaError
        assert issubclass(InferenceSchemaError, DataValidationError)

    def test_model_artifact_error_is_modelium_error(self):
        from src.utils.exceptions import ModelArtifactError, ModeliumError
        assert issubclass(ModelArtifactError, ModeliumError)

    def test_pipeline_stage_error_is_modelium_error(self):
        from src.utils.exceptions import ModeliumError, PipelineStageError
        assert issubclass(PipelineStageError, ModeliumError)


class TestPipelineStageError:
    """Test the new PipelineStageError with its stage/reason attributes."""

    def test_stores_stage_name(self):
        from src.utils.exceptions import PipelineStageError
        err = PipelineStageError("train", "model diverged")
        assert err.stage == "train"

    def test_stores_reason(self):
        from src.utils.exceptions import PipelineStageError
        err = PipelineStageError("train", "model diverged")
        assert err.reason == "model diverged"

    def test_message_includes_stage_and_reason(self):
        from src.utils.exceptions import PipelineStageError
        err = PipelineStageError("validate", "missing file")
        assert "validate" in str(err)
        assert "missing file" in str(err)

    def test_chains_cause(self):
        from src.utils.exceptions import PipelineStageError
        original = ValueError("bad value")
        err = PipelineStageError("prepare", "conversion failed", cause=original)
        assert err.__cause__ is original

    def test_cause_defaults_to_none(self):
        from src.utils.exceptions import PipelineStageError
        err = PipelineStageError("train", "timeout")
        assert err.__cause__ is None

    def test_catchable_as_modelium_error(self):
        from src.utils.exceptions import ModeliumError, PipelineStageError
        with pytest.raises(ModeliumError):
            raise PipelineStageError("predict", "no model found")


# ========================================================================
# validate_dataframe tests
# ========================================================================

class TestValidateDataframe:
    """Test the ``validate_dataframe`` helper."""

    def test_valid_dataframe_passes(self):
        from src.utils.exceptions import validate_dataframe

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        validate_dataframe(df, "test_table", required_columns=["a", "b"])
        # no exception = pass

    def test_empty_dataframe_raises(self):
        from src.utils.exceptions import DataValidationError, validate_dataframe

        df = pd.DataFrame({"a": [], "b": []})
        with pytest.raises(DataValidationError, match="empty"):
            validate_dataframe(df, "empty_table")

    def test_missing_columns_raises(self):
        from src.utils.exceptions import DataValidationError, validate_dataframe

        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(DataValidationError, match="missing.*required column"):
            validate_dataframe(df, "test_table", required_columns=["a", "c", "d"])

    def test_missing_columns_listed_in_message(self):
        from src.utils.exceptions import DataValidationError, validate_dataframe

        df = pd.DataFrame({"a": [1]})
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataframe(df, "test_table", required_columns=["a", "x", "y"])

        msg = str(exc_info.value)
        assert "x" in msg
        assert "y" in msg

    def test_empty_and_missing_columns_both_reported(self):
        from src.utils.exceptions import DataValidationError, validate_dataframe

        df = pd.DataFrame({"a": pd.Series([], dtype="int64")})
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataframe(df, "bad_table", required_columns=["a", "z"])

        msg = str(exc_info.value)
        assert "empty" in msg
        assert "z" in msg

    def test_non_dataframe_raises(self):
        from src.utils.exceptions import DataValidationError, validate_dataframe

        with pytest.raises(DataValidationError, match="not a DataFrame"):
            validate_dataframe({"a": [1]}, "not_a_df")  # type: ignore[arg-type]

    def test_no_required_columns_skips_column_check(self):
        from src.utils.exceptions import validate_dataframe

        df = pd.DataFrame({"x": [1]})
        validate_dataframe(df, "simple_table")
        # no exception = pass

    def test_all_required_present_passes(self):
        from src.utils.exceptions import validate_dataframe

        df = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0], "EXT_SOURCE_1": [0.5]})
        validate_dataframe(
            df, "application_train",
            required_columns=["SK_ID_CURR", "TARGET"],
        )
        # no exception = pass
