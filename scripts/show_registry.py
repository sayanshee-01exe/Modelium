"""Print what the MLflow Model Registry currently holds, without launching the UI.

The `mlflow ui` server answers the same questions, but a browser is not always the right
tool: this is what a CI check, a terminal session over SSH, or a reviewer wanting a
one-line answer to "what is serving?" can use. Read-only — it registers nothing, moves
no alias, and changes no run.

Reads the same local tracking store as the pipeline (`mlflow.tracking_uri` in
params.yaml), so it can never disagree with what the register stage wrote.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.mlflow_tracker import MLflowTracker
from src.utils.config_loader import load_params


def main() -> int:
    params = load_params()["mlflow"]
    name = params["registered_model_name"]

    # Reuse the tracker's URI anchoring so a relative sqlite path resolves to the repo
    # root here exactly as it does inside a stage.
    resolved = MLflowTracker._resolve_uri(params["tracking_uri"])

    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient(tracking_uri=resolved)
    print(f"tracking store : {resolved}")
    print(f"registered name: {name}\n")

    try:
        versions = client.search_model_versions(f"name='{name}'")
    except MlflowException as err:
        print(f"Nothing registered under {name!r} yet ({err}). Run `make register`.")
        return 0

    if not versions:
        print(f"Nothing registered under {name!r} yet. Run `make register`.")
        return 0

    # Aliases live on the registered model, not on the version rows, so they are
    # resolved separately rather than read off each version.
    alias_of: dict[str, list[str]] = {}
    for alias in (params["champion_alias"], params["candidate_alias"]):
        try:
            resolved_version = client.get_model_version_by_alias(name, alias)
        except MlflowException:
            continue
        alias_of.setdefault(str(resolved_version.version), []).append(alias)

    print(f"{'ver':>4}  {'status':<9} {'aliases':<22} {'run':<34} model")
    print("-" * 100)
    for version in sorted(versions, key=lambda v: int(v.version), reverse=True):
        number = str(version.version)
        print(
            f"{number:>4}  "
            f"{version.tags.get('validation_status', '-'):<9} "
            f"{','.join(alias_of.get(number, [])) or '-':<22} "
            f"{version.run_id:<34} "
            f"{version.tags.get('champion_model', '-')}"
        )

    champion_alias = params["champion_alias"]
    if champion_alias in {a for aliases in alias_of.values() for a in aliases}:
        print(f"\nServing URI: models:/{name}@{champion_alias}")
    else:
        print(
            f"\nNo version currently holds the '{champion_alias}' alias — nothing has "
            f"passed its quality gates. This is a refusal, not a gap."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
