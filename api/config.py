"""API configuration: params.yaml defaults, overridable by environment variables.

Deployment must not require editing a tracked file, so every setting can be overridden
with a `MODELIUM_API_*` variable. params.yaml holds the defaults that make a local run
work out of the box; the environment holds whatever a particular deployment differs on.

No secret is read from or written to params.yaml. There are no credentials in this
service — it has no authentication, which the README states plainly rather than implying
otherwise by having a config slot for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_PREFIX = "MODELIUM_API_"

# Defaults used only if params.yaml cannot be read — a missing config file should not
# stop the service from starting and reporting itself unhealthy in a readable way.
FALLBACK = {
    "host": "127.0.0.1",
    "port": 8000,
    "max_batch_size": 100,
    "model_uri": "models:/modelium-credit-risk-champion@champion",
    "feature_store_path": "data/processed/test_features.parquet",
    "max_explain_rows": 1,
    "explain_top_n": 10,
    "log_level": "INFO",
}


@dataclass(frozen=True)
class ApiSettings:
    """Resolved settings for one process. Frozen: nothing rebinds these at runtime."""

    host: str
    port: int
    max_batch_size: int
    model_uri: str
    feature_store_path: Path
    max_explain_rows: int
    explain_top_n: int
    log_level: str
    registered_model_name: str
    champion_alias: str
    tracking_uri: str
    tracking_enabled: bool

    @property
    def safe_summary(self) -> dict[str, object]:
        """Settings that are safe to return to a client.

        Deliberately excludes `tracking_uri` and `feature_store_path`: both are
        filesystem locations on the host, and an API has no reason to disclose its own
        directory layout.
        """
        return {
            "registered_model": self.registered_model_name,
            "alias": self.champion_alias,
            "max_batch_size": self.max_batch_size,
        }


def _env(key: str) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{key.upper()}")


def _resolve_path(value: str) -> Path:
    """Anchor a relative path to the repo root, not the process working directory."""
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_settings() -> ApiSettings:
    """Build settings from params.yaml, then apply environment overrides."""
    api_section = dict(FALLBACK)
    registry = {
        "registered_model_name": "modelium-credit-risk-champion",
        "champion_alias": "champion",
        "tracking_uri": "sqlite:///mlflow.db",
        "enabled": True,
    }

    try:
        from src.utils.config_loader import load_params

        params = load_params()
        api_section.update(params.get("api") or {})
        mlflow_section = params.get("mlflow") or {}
        registry.update({
            "registered_model_name": mlflow_section.get(
                "registered_model_name", registry["registered_model_name"]),
            "champion_alias": mlflow_section.get(
                "champion_alias", registry["champion_alias"]),
            "tracking_uri": mlflow_section.get("tracking_uri", registry["tracking_uri"]),
            "enabled": mlflow_section.get("enabled", True),
        })
    except Exception:
        # A malformed params.yaml must not prevent the process from starting: it should
        # come up and report itself not-ready, which is far easier to diagnose than a
        # container that exits immediately.
        pass

    for key in ("host", "model_uri", "feature_store_path", "log_level"):
        override = _env(key)
        if override:
            api_section[key] = override
    for key in ("port", "max_batch_size", "max_explain_rows", "explain_top_n"):
        override = _env(key)
        if override:
            try:
                api_section[key] = int(override)
            except ValueError:
                pass

    # The tracking store is anchored the same way the pipeline anchors it, so the API
    # and the stages cannot end up reading different databases.
    tracking_uri = registry["tracking_uri"]
    try:
        from src.tracking.mlflow_tracker import MLflowTracker

        tracking_uri = MLflowTracker._resolve_uri(tracking_uri)
    except Exception:
        pass

    return ApiSettings(
        host=str(api_section["host"]),
        port=int(api_section["port"]),
        max_batch_size=int(api_section["max_batch_size"]),
        model_uri=str(api_section["model_uri"]),
        feature_store_path=_resolve_path(str(api_section["feature_store_path"])),
        max_explain_rows=int(api_section["max_explain_rows"]),
        explain_top_n=int(api_section["explain_top_n"]),
        log_level=str(api_section["log_level"]).upper(),
        registered_model_name=str(registry["registered_model_name"]),
        champion_alias=str(registry["champion_alias"]),
        tracking_uri=str(tracking_uri),
        tracking_enabled=bool(registry["enabled"]),
    )
