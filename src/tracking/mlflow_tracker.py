"""MLflow experiment tracking, confined to one module.

Every `import mlflow` and every MLflow call in this project lives here. Callers get a
small interface — start a run, log params/metrics/tags/artifacts — and never branch on
whether tracking is switched on.

DVC and MLflow answer different questions and both are kept:

    DVC      can this run be reproduced? (data, stage graph, cached outputs)
    MLflow   what happened in this run? (params, metrics, artifacts, comparisons)

Two behaviours are deliberate rather than incidental:

*Disabled means silent, not broken.* With ``enabled: false`` every method is a no-op
and the run context managers still yield, so `scripts/train.py` needs no `if tracker:`
guards and training is never gated on an observability tool.

*Enabled means it must actually work.* If tracking is switched on and MLflow cannot be
initialised, that raises — a training run that silently discards four hours of metrics
is worse than one that refuses to start. Once running, a failure to log an individual
metric is warned about loudly and the run continues: losing a completed model to a
metric-logging hiccup would be a worse trade. Nothing is swallowed by a bare `except`.
"""

from __future__ import annotations

import numbers
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.utils.exceptions import ModeliumError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = "Modelium"

# MLflow rejects metric names outside a restricted character set, and the pipeline's
# metric keys are display names ("Average Precision", "ROC-AUC"). Normalising here keeps
# the leaderboard readable while the tracked names stay queryable.
_NON_ALNUM = re.compile(r"[^0-9a-zA-Z]+")

# MLflow truncates long param values; a 407-column feature list is not a param anyway.
MAX_PARAM_CHARS = 250

SQLITE_PREFIX = "sqlite:///"

# Where run artifacts land. MLflow 3 requires a database tracking backend — the plain
# filesystem store ("./mlruns") is in maintenance mode and raises unless explicitly
# opted into — so the database holds the run metadata and this directory holds the files.
ARTIFACT_DIRNAME = "mlartifacts"


class MLflowTrackingError(ModeliumError):
    """Raised when tracking is enabled but MLflow cannot be initialised.

    Deliberately fatal: the caller asked for the run to be tracked, and quietly
    proceeding would discard the record they asked for.
    """


def normalise_metric_name(name: str) -> str:
    """``"Average Precision"`` -> ``"average_precision"``, ``"ROC-AUC"`` -> ``"roc_auc"``."""
    return _NON_ALNUM.sub("_", str(name)).strip("_").lower()


def _is_number(value: Any) -> bool:
    """Numeric and loggable as a metric.

    `numbers.Real` also covers numpy scalars, which register with the ABC hierarchy.
    Booleans are excluded despite being ints: `promoted=True` is a tag, not a metric.
    """
    return not isinstance(value, bool) and isinstance(value, numbers.Real)


def get_git_commit() -> dict[str, str]:
    """Current commit and whether the tree was dirty, for run provenance.

    Absent git, or a non-repository directory, is not a tracking failure — it returns
    empty rather than raising, because a run outside version control is still a run
    worth recording.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip())
    except (subprocess.SubprocessError, OSError) as err:
        logger.warning("Could not read the git commit for run provenance: %s", err)
        return {}
    return {"git_commit": commit, "git_dirty": str(dirty).lower()}


class MLflowTracker:
    """Records one training execution to MLflow, or does nothing if disabled.

    Attributes:
        enabled: Whether any MLflow call is made at all.
        experiment_name: Experiment the runs are grouped under.
        tracking_uri: Resolved tracking location.
        degraded: True if a log call failed after the run started. The run continues,
            but the record is known to be incomplete.
    """

    def __init__(self, enabled: bool, experiment_name: str, tracking_uri: str):
        self.enabled = bool(enabled)
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.degraded = False
        self._mlflow = None

        if not self.enabled:
            logger.info("MLflow tracking is disabled; training will not be recorded")
            return

        try:
            import mlflow
        except ImportError as err:                            # pragma: no cover
            raise MLflowTrackingError(
                f"MLflow tracking is enabled but mlflow is not installed: {err}. "
                f"Install it, or set mlflow.enabled=false in params.yaml."
            ) from err

        try:
            mlflow.set_tracking_uri(self._resolve_uri(tracking_uri))
            self._ensure_experiment(mlflow, experiment_name)
        except Exception as err:
            raise MLflowTrackingError(
                f"MLflow tracking is enabled but could not be initialised "
                f"(uri={tracking_uri!r}, experiment={experiment_name!r}): {err}. "
                f"Fix the tracking configuration, or set mlflow.enabled=false to run "
                f"without it — training will not silently discard its metrics."
            ) from err

        self._mlflow = mlflow
        logger.info(
            "MLflow tracking enabled: experiment=%r uri=%s",
            experiment_name, self._resolve_uri(tracking_uri),
        )

    @staticmethod
    def _resolve_uri(tracking_uri: str) -> str:
        """Anchor a local tracking store to the repo root, not the caller's cwd.

        `sqlite:///mlflow.db` is a *relative* SQLite path, so without this the database
        would land wherever the process happened to start — one file when DVC runs the
        stage from the repository root, another when the script is run by hand from
        `scripts/`, and the run history silently split between them.

        Remote URIs (http, https, databricks) are passed through untouched.
        """
        if tracking_uri.startswith(SQLITE_PREFIX):
            rest = tracking_uri[len(SQLITE_PREFIX):]
            if rest and not rest.startswith("/"):
                return f"{SQLITE_PREFIX}{PROJECT_ROOT / rest}"
            return tracking_uri
        if "://" in tracking_uri:
            return tracking_uri
        path = Path(tracking_uri)
        return str(path if path.is_absolute() else (PROJECT_ROOT / path))

    @staticmethod
    def _ensure_experiment(mlflow, experiment_name: str) -> None:
        """Select the experiment, pinning its artifact root to the repo.

        MLflow otherwise defaults artifacts to `./mlartifacts` relative to the working
        directory — the same cwd-dependence `_resolve_uri` removes for the database.
        The location is only settable at creation, so it is applied once and reused.
        """
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=(PROJECT_ROOT / ARTIFACT_DIRNAME).as_uri(),
            )
        mlflow.set_experiment(experiment_name)

    @classmethod
    def from_params(cls, params: dict) -> "MLflowTracker":
        """Build from the validated `mlflow:` section of params.yaml."""
        section = params.get("mlflow") or {}
        return cls(
            enabled=section.get("enabled", False),
            experiment_name=section.get("experiment_name", "modelium"),
            tracking_uri=section.get("tracking_uri", "mlruns"),
        )

    # ------------------------------------------------------------------------- runs

    @contextmanager
    def start_run(self, run_name: str | None = None,
                  tags: dict[str, Any] | None = None) -> Iterator["MLflowTracker"]:
        """Parent run for one training execution. Yields self, enabled or not."""
        if not self.enabled:
            yield self
            return

        with self._mlflow.start_run(run_name=run_name):
            if tags:
                self.set_tags(tags)
            yield self

    @contextmanager
    def child_run(self, run_name: str,
                  tags: dict[str, Any] | None = None) -> Iterator["MLflowTracker"]:
        """Nested run for a single candidate model, grouped under the parent."""
        if not self.enabled:
            yield self
            return

        with self._mlflow.start_run(run_name=run_name, nested=True):
            if tags:
                self.set_tags(tags)
            yield self

    # ----------------------------------------------------------------------- logging

    def _guard(self, what: str, action) -> None:
        """Run a log call, surfacing any failure as a warning rather than a crash.

        A failed metric write must not destroy a completed training run, but it must
        also not pass unnoticed — hence the warning and the `degraded` flag.
        """
        if not self.enabled:
            return
        try:
            action()
        except Exception as err:
            self.degraded = True
            logger.warning("MLflow: failed to log %s: %s", what, err, exc_info=True)

    def log_params(self, params: dict[str, Any], prefix: str = "") -> None:
        """Log configuration values. The input mapping is never modified."""
        if not self.enabled or not params:
            return
        payload = {
            f"{prefix}{key}": self._as_param_value(value)
            for key, value in params.items() if value is not None
        }
        self._guard(f"{len(payload)} param(s)", lambda: self._mlflow.log_params(payload))

    @staticmethod
    def _as_param_value(value: Any) -> str:
        text = str(value)
        return text if len(text) <= MAX_PARAM_CHARS else text[:MAX_PARAM_CHARS - 3] + "..."

    def log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        """Log the numeric entries of a metrics mapping, normalising the names.

        Non-numeric entries (`"Model"`, and booleans, which belong in tags) are skipped
        rather than coerced. The input mapping is never modified.
        """
        if not self.enabled or not metrics:
            return
        payload = {
            f"{prefix}{normalise_metric_name(key)}": float(value)
            for key, value in metrics.items() if _is_number(value)
        }
        if not payload:
            return
        self._guard(f"{len(payload)} metric(s)", lambda: self._mlflow.log_metrics(payload))

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Log run tags. The input mapping is never modified."""
        if not self.enabled or not tags:
            return
        payload = {key: str(value) for key, value in tags.items() if value is not None}
        self._guard(f"{len(payload)} tag(s)", lambda: self._mlflow.set_tags(payload))

    def log_artifact(self, path, artifact_path: str | None = None) -> None:
        """Log one file. A missing file is warned about, not raised.

        The pipeline's artifacts are written by earlier steps; if one is absent, that is
        worth surfacing but is not a reason to fail a completed training run.
        """
        if not self.enabled:
            return
        path = Path(path)
        if not path.exists():
            logger.warning("MLflow: artifact %s does not exist; not logged", path)
            return
        self._guard(
            f"artifact {path.name}",
            lambda: self._mlflow.log_artifact(str(path), artifact_path=artifact_path),
        )

    def log_artifacts(self, paths, artifact_path: str | None = None) -> None:
        """Log several files, skipping any that are absent."""
        for path in paths:
            self.log_artifact(path, artifact_path=artifact_path)
