"""Rebase the MLflow store's artifact paths onto the container's mount points.

**Why this exists.** MLflow records where each artifact lives as an *absolute path on the
machine that wrote it*. On this project that is the developer's home directory, so the
database contains entries like::

    file:///Users/<someone>/.../smart_loan_approval_system/mlartifacts/models/m-748d…

Inside a Linux container that path does not exist. `mlflow.sklearn.load_model` would
either fail outright or — worse, and this is what happened while developing Step 11 —
appear to succeed on the host because it quietly read the original directory rather than
the relocated copy. A container has no such directory to fall back to, so the champion
simply would not load.

**What it does.** Copies the mounted store — both the database and the artifact tree — to
a writable location, then rewrites the four columns that hold paths so they point at that
copy. The host's files are mounted read-only and never modified: they stay the authority,
and the container derives its own view of them.

The artifact tree has to be copied rather than read in place because resolving a
`models:/<name>@<alias>` URI makes MLflow *write* a `registered_model_meta` file into the
model's own directory. Against a read-only mount that fails with `Errno 30` after the
alias has already resolved, which reads as a corrupt artifact rather than a mount
permission. The tree is ~16 MB, so copying it costs a moment at startup.

**How the rewrite avoids hard-coding anything.** It never looks for a particular home
directory. For each stored value it finds the `/mlartifacts` or `/mlruns` segment and
replaces everything *before* it with the container root, so the same code works whatever
machine wrote the database. Running it twice is a no-op.

Run automatically by the container entrypoint; safe to run by hand for debugging.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path

# (table, column) pairs that hold a filesystem location. Discovered by scanning the
# schema for the host path rather than assumed, so this list matches MLflow 3's layout.
PATH_COLUMNS = (
    ("runs", "artifact_uri"),
    ("logged_models", "artifact_location"),
    ("experiments", "artifact_location"),
    ("model_versions", "storage_location"),
)

# Directory names that mark where a stored path becomes relative to the store root.
MARKERS = ("mlartifacts", "mlruns")


def rebase_value(value: str, roots: dict[str, str]) -> str | None:
    """Point one stored location at the container mount, or None if unchanged.

    Preserves the original form: a `file://` URI stays a URI, a plain path stays plain.
    Everything before the marker segment is replaced, so the developer's home directory
    never needs to be known or matched.
    """
    if not value:
        return None
    for marker in MARKERS:
        needle = f"/{marker}"
        index = value.find(needle)
        if index == -1:
            continue
        suffix = value[index + len(needle):]
        root = roots[marker]
        rebased = f"file://{root}{suffix}" if value.startswith("file://") else f"{root}{suffix}"
        return rebased if rebased != value else None
    return None


def rebase_database(database: Path, roots: dict[str, str]) -> int:
    """Rewrite every stored location in place. Returns the number of rows changed."""
    changed = 0
    connection = sqlite3.connect(database)
    try:
        existing = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column in PATH_COLUMNS:
            if table not in existing:
                continue
            rows = connection.execute(
                f'SELECT rowid, "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL').fetchall()
            for rowid, value in rows:
                rebased = rebase_value(str(value), roots)
                if rebased is not None:
                    connection.execute(
                        f'UPDATE "{table}" SET "{column}" = ? WHERE rowid = ?',
                        (rebased, rowid))
                    changed += 1
        connection.commit()
    finally:
        connection.close()
    return changed


def sync_artifacts(source: Path, target: Path, force: bool = False) -> None:
    """Mirror the read-only artifact mount into a writable location."""
    if not source.exists():
        raise FileNotFoundError(
            f"No artifact directory at {source}. Mount the project's mlartifacts/ "
            f"there, or the champion's files cannot be read."
        )
    if target.exists() and not force:
        print(f"Reusing existing artifact copy at {target}", flush=True)
        return
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    print(f"Copied artifacts {source} -> {target}", flush=True)


def prepare(source: Path, target: Path, artifacts_root: Path, runs_root: Path,
            force: bool = False) -> int:
    """Copy the mounted store to a writable path and rebase it.

    Raises:
        FileNotFoundError: if the source database or the artifact directory is absent.
            Both are mount problems, and failing here with the missing path named is far
            easier to diagnose than an alias that mysteriously will not resolve.
    """
    if not source.exists():
        raise FileNotFoundError(
            f"No MLflow database at {source}. Mount the project's mlflow.db there — "
            f"see the compose file's volume list."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    if force or not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        shutil.copy2(source, target)
        print(f"Copied registry {source} -> {target}", flush=True)
    else:
        print(f"Reusing existing runtime registry at {target}", flush=True)

    changed = rebase_database(
        target, {"mlartifacts": str(artifacts_root), "mlruns": str(runs_root)})
    print(f"Rebased {changed} stored artifact location(s) onto the container mounts",
          flush=True)
    return changed


def verify(database: Path, registered_model_name: str, alias: str) -> int:
    """Confirm the alias resolves and its artifacts are readable from inside here.

    Checks the files exist rather than loading the model: loading pulls a large pipeline
    into memory, and the entrypoint only needs to know the mounts are wired correctly.
    """
    from urllib.parse import unquote, urlparse

    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        row = connection.execute(
            """
            SELECT mv.storage_location
            FROM registered_model_aliases a
            JOIN model_versions mv
              ON mv.name = a.name AND mv.version = a.version
            WHERE a.name = ? AND a.alias = ?
            """,
            (registered_model_name, alias),
        ).fetchone()
    except sqlite3.Error as err:
        print(f"ERROR: registry could not be queried: {err}", file=sys.stderr)
        return 1
    finally:
        connection.close()

    if not row:
        print(f"ERROR: alias {alias!r} does not resolve for {registered_model_name!r}. "
              f"Run the register stage on the host before starting the container.",
              file=sys.stderr)
        return 1

    location = str(row[0])
    path = Path(unquote(urlparse(location).path) if location.startswith("file://")
                else location)
    if not path.exists():
        print(f"ERROR: the champion's artifacts are not readable at {path}. The "
              f"mlartifacts mount is missing or points at the wrong directory.",
              file=sys.stderr)
        return 1

    print(f"Champion alias {alias!r} resolves and its artifacts are readable.", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", default=os.environ.get(
        "MODELIUM_MLFLOW_SOURCE_DB", "/app/runtime/mlflow.db"))
    parser.add_argument("--target", default=os.environ.get(
        "MODELIUM_MLFLOW_RUNTIME_DB", "/app/runtime/state/mlflow.db"))
    parser.add_argument("--artifacts-source", default=os.environ.get(
        "MODELIUM_MLARTIFACTS_SOURCE", "/app/runtime/mlartifacts"),
        help="read-only mount of the host's artifact tree")
    parser.add_argument("--artifacts-root", default=os.environ.get(
        "MODELIUM_MLARTIFACTS_ROOT", "/app/runtime/state/mlartifacts"),
        help="writable copy the registry is rebased onto")
    parser.add_argument("--runs-root", default=os.environ.get(
        "MODELIUM_MLRUNS_ROOT", "/app/runtime/mlruns"))
    parser.add_argument("--registered-model-name", default=os.environ.get(
        "MODELIUM_REGISTERED_MODEL_NAME", "modelium-credit-risk-champion"))
    parser.add_argument("--alias", default=os.environ.get(
        "MODELIUM_CHAMPION_ALIAS", "champion"))
    parser.add_argument("--force", action="store_true",
                        help="re-copy even when the runtime database looks current")
    args = parser.parse_args()

    try:
        sync_artifacts(Path(args.artifacts_source), Path(args.artifacts_root),
                       force=args.force)
        prepare(Path(args.source), Path(args.target), Path(args.artifacts_root),
                Path(args.runs_root), force=args.force)
    except FileNotFoundError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1

    return verify(Path(args.target), args.registered_model_name, args.alias)


if __name__ == "__main__":
    raise SystemExit(main())
