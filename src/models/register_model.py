"""File the champion into the MLflow Model Registry, under an alias that states its status.

Tracking already answers *what happened in this run*. The registry answers the question
a consumer actually asks: **which recorded run should I serve, and was it approved?**
Without it the only way to name a model is a file path on one machine, and "the
champion" means whatever `models/champion_pipeline.joblib` happens to contain.

Three decisions here are deliberate:

*Registration is not promotion.* Every run that logged a model gets registered, because
a refused champion is part of the history and deleting it would leave a gap where an
explanation should be. What promotion controls is the **alias**: the production alias is
moved onto a version only when that run passed every quality gate, so
``models:/<name>@<champion_alias>`` cannot resolve to a model the pipeline rejected.
This is the same rule `src/inference/predictor.py` enforces for batch scoring, applied
at the other end of the handoff.

*A reversal must move the alias, not just decline to set it.* Re-registering a version
that was previously approved and is now rejected removes the production alias. Leaving
it in place would keep serving an approved-looking model that no longer is one.

*Re-running is idempotent.* DVC re-runs a stage whenever any dependency changes, so
registering the same ``run_id`` twice reuses the version it already created rather than
stacking near-identical versions that differ only in when the command ran.

Registration is deliberately **not** a dependency of batch inference. Scoring loads the
local champion artifact, so an unreachable registry cannot stop the pipeline producing
predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.exceptions import ModelArtifactError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Keys `run_information.json` must carry for the register stage to act on it. Checked at
# load so a half-written handoff fails naming the missing field, rather than surfacing
# as an opaque error from inside an MLflow call.
RUN_INFO_REQUIRED_KEYS: tuple[str, ...] = (
    "tracking_enabled", "tracking_uri", "run_id", "experiment_id", "model_uri",
    "registered_model_name", "champion_model", "promoted", "optimal_threshold",
)

# Value of the `validation_status` tag on a registered version. The tag records the
# decision on the version itself, so it survives an alias being moved later.
PROMOTED_STATUS = "approved"
CANDIDATE_STATUS = "rejected"


# ---------------------------------------------------------------------------
# run_information.json — the handoff between the train and register stages
# ---------------------------------------------------------------------------

def build_run_information(
    tracker,
    *,
    model_uri: str | None,
    registered_model_name: str,
    champion_model: str,
    promoted: bool,
    optimal_threshold: float,
    test_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the run just completed, in the form the register stage consumes.

    Must be called *inside* the tracker's run context — the run id is only readable
    while the run is open.

    With tracking disabled this still returns a complete dict, with
    ``tracking_enabled: false`` and a null ``model_uri``. The train stage declares the
    file as a DVC output, so it has to exist either way; the register stage reads the
    flag and skips.

    Args:
        tracker: The `MLflowTracker` for the open run.
        model_uri: URI returned by `tracker.log_model`, or None if nothing was logged.
        registered_model_name: Name to file versions under, from params.yaml.
        champion_model: Display name of the winning model.
        promoted: Whether the run passed every quality gate.
        optimal_threshold: The decision threshold frozen on validation.
        test_metrics: Headline test metrics, recorded for auditability.

    Returns:
        A JSON-serialisable mapping carrying every key in `RUN_INFO_REQUIRED_KEYS`.
    """
    return {
        "tracking_enabled": bool(tracker.enabled),
        "tracking_uri": tracker.resolved_uri if tracker.enabled else None,
        "experiment_name": tracker.experiment_name,
        "experiment_id": tracker.active_experiment_id,
        "run_id": tracker.active_run_id,
        "model_uri": model_uri,
        "registered_model_name": registered_model_name,
        "champion_model": champion_model,
        "promoted": bool(promoted),
        "optimal_threshold": float(optimal_threshold),
        # True when the model was fitted but MLflow could not record all of it. The
        # register stage surfaces this rather than treating the run as fully tracked.
        "tracking_degraded": bool(getattr(tracker, "degraded", False)),
        "test_metrics": {
            key: value for key, value in (test_metrics or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_run_information(info: dict[str, Any], path: str | Path) -> Path:
    """Write the handoff file, creating its directory."""
    return _write_json(info, path, "run information")


def _write_json(payload: dict[str, Any], path: str | Path, what: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    logger.info("Wrote %s to %s", what, path)
    return path


def build_registry_record(
    run_info: dict[str, Any],
    version,
    *,
    champion_alias: str,
    candidate_alias: str,
) -> dict[str, Any]:
    """Summarise what registration did, for the stage's own output.

    The registry is external state DVC can neither cache nor restore, so the pipeline
    keeps its own record of the outcome. It is also what a reviewer reads to answer
    "what is currently serving" without a working MLflow install.

    Args:
        run_info: The handoff mapping the registration acted on.
        version: The `ModelVersion` created or reused, or None if nothing was registered.
        champion_alias: Alias reserved for approved versions.
        candidate_alias: Alias for recorded-but-refused versions.

    Returns:
        A JSON-serialisable mapping. ``registered: false`` with a ``skipped_reason`` when
        there was no run to file — a skip is an outcome worth recording, not an absence.
    """
    name = run_info.get("registered_model_name")
    if version is None:
        reason = (
            "MLflow tracking was disabled for this run"
            if not run_info.get("tracking_enabled")
            else "the run logged no model URI"
        )
        return {
            "registered": False,
            "skipped_reason": reason,
            "registered_model_name": name,
            "run_id": run_info.get("run_id"),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    promoted = bool(run_info["promoted"])
    alias = champion_alias if promoted else candidate_alias
    return {
        "registered": True,
        "registered_model_name": name,
        "version": str(version.version),
        "run_id": run_info.get("run_id"),
        "champion_model": run_info.get("champion_model"),
        "promoted": promoted,
        "validation_status": PROMOTED_STATUS if promoted else CANDIDATE_STATUS,
        "alias": alias,
        "model_uri": f"models:/{name}@{alias}",
        "optimal_threshold": run_info.get("optimal_threshold"),
        "test_metrics": run_info.get("test_metrics", {}),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def write_registry_record(record: dict[str, Any], path: str | Path) -> Path:
    """Write the register stage's output record, creating its directory."""
    return _write_json(record, path, "registry record")


def load_run_information(path: str | Path) -> dict[str, Any]:
    """Read and check the handoff file.

    Args:
        path: Location of `run_information.json`.

    Returns:
        The parsed mapping.

    Raises:
        ModelArtifactError: if the file is absent, unparseable, or missing a required
            key. All three mean the same thing — the train stage did not complete the
            handoff — and none should be discovered inside a registry call.
    """
    path = Path(path)
    if not path.exists():
        raise ModelArtifactError(
            f"No run_information.json at {path}. It is written by the train stage; run "
            f"`dvc repro train` before registering."
        )

    try:
        with open(path, encoding="utf-8") as handle:
            info = json.load(handle)
    except json.JSONDecodeError as err:
        raise ModelArtifactError(
            f"run_information.json at {path} is not valid JSON: {err}"
        ) from err

    if not isinstance(info, dict):
        raise ModelArtifactError(
            f"run_information.json at {path} must be a JSON object, got "
            f"{type(info).__name__}"
        )

    missing = [key for key in RUN_INFO_REQUIRED_KEYS if key not in info]
    if missing:
        raise ModelArtifactError(
            f"run_information.json at {path} is missing required key(s): {missing}. "
            f"The train stage did not finish writing the handoff."
        )
    return info


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _existing_version_for_run(client, name: str, run_id: str):
    """The version already registered from this run, if there is one.

    Registering the same run twice would otherwise create a second version identical to
    the first — which is what DVC re-running the stage would do on any dependency change.
    """
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception as err:
        # A registry that has never seen this name is not an error; nothing is registered.
        logger.debug("No existing versions for %r: %s", name, err)
        return None
    for version in versions:
        if version.run_id == run_id:
            return version
    return None


def _clear_alias(client, name: str, alias: str) -> None:
    """Remove an alias if it points anywhere. Absent is the desired end state either way."""
    try:
        client.delete_registered_model_alias(name=name, alias=alias)
    except Exception:
        # Nothing to remove. `get_model_version_by_alias` raises rather than returning
        # None for an unset alias, so asking first would need the same try/except.
        pass


def register_champion(
    run_info: dict[str, Any],
    *,
    champion_alias: str,
    candidate_alias: str,
):
    """Register the run's model and set the alias its promotion status earns.

    Args:
        run_info: Mapping from `load_run_information` / `build_run_information`.
        champion_alias: Alias for an approved version. Assigned only when
            ``run_info["promoted"]`` is true, and actively removed when it is not.
        candidate_alias: Alias for a version that was recorded but not approved.

    Returns:
        The `ModelVersion` that now exists, or None if there was nothing to register
        (tracking disabled, or model logging failed). None is a skip, not a failure —
        the registry is observability, and the pipeline already has its model on disk.
    """
    if not run_info.get("tracking_enabled"):
        logger.info(
            "MLflow tracking was disabled for this run; there is no run to register. "
            "Set mlflow.enabled=true in params.yaml to record and register runs."
        )
        return None

    model_uri = run_info.get("model_uri")
    run_id = run_info.get("run_id")
    if not model_uri or not run_id:
        logger.warning(
            "Run %s logged no model URI, so nothing can be registered. The champion "
            "pipeline on disk is unaffected; the run record is incomplete.", run_id,
        )
        return None

    import mlflow
    from mlflow import MlflowClient

    tracking_uri = run_info.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    name = run_info["registered_model_name"]
    promoted = bool(run_info["promoted"])

    version = _existing_version_for_run(client, name, run_id)
    if version is not None:
        logger.info(
            "Run %s is already registered as %s version %s; reusing it rather than "
            "creating a duplicate.", run_id, name, version.version,
        )
    else:
        version = mlflow.register_model(model_uri=model_uri, name=name)
        logger.info("Registered %s version %s from run %s", name, version.version, run_id)

    version_number = str(version.version)

    # Tags describe the version and stay with it; aliases point at it and move. Both are
    # rewritten on every call so a re-registration reflects the current decision rather
    # than whatever the first one recorded.
    for key, value in {
        "validation_status": PROMOTED_STATUS if promoted else CANDIDATE_STATUS,
        "champion_model": run_info.get("champion_model", ""),
        "optimal_threshold": run_info.get("optimal_threshold", ""),
        "primary_metric_value": run_info.get("test_metrics", {}).get("Average Precision", ""),
        "run_id": run_id,
        "tracking_degraded": str(bool(run_info.get("tracking_degraded", False))).lower(),
    }.items():
        client.set_model_version_tag(name=name, version=version_number,
                                     key=key, value=str(value))

    if promoted:
        _clear_alias(client, name, candidate_alias)
        client.set_registered_model_alias(name=name, alias=champion_alias,
                                          version=version_number)
        logger.info(
            "%s version %s passed every quality gate; alias '%s' now resolves to it "
            "(models:/%s@%s).", name, version_number, champion_alias, name,
            champion_alias,
        )
    else:
        # Removing rather than merely not setting: a version previously approved and now
        # rejected must stop resolving under the production alias.
        _clear_alias(client, name, champion_alias)
        client.set_registered_model_alias(name=name, alias=candidate_alias,
                                          version=version_number)
        logger.warning(
            "%s version %s did NOT pass its quality gates. It is registered and aliased "
            "'%s' for traceability, and the '%s' alias has been cleared — nothing should "
            "serve this version.", name, version_number, candidate_alias, champion_alias,
        )

    return version
