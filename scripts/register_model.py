"""DVC stage 5 — register the champion in the MLflow Model Registry.

Reads the handoff `artifacts/run_information.json` written by the train stage and files
that run's logged model under the registered name from `params.yaml`, aliased according
to whether it passed its quality gates.

This stage trains nothing and reads no data. It is separated from training so that
re-registering — or registering after a registry outage — costs a second, not a
multi-hour search. It also sits *outside* the path to `predict`: batch scoring loads the
local champion artifact, so an unreachable registry cannot stop the pipeline producing
predictions.

Exit codes: 0 on success **and** on a deliberate skip (tracking disabled, or no model
logged). Non-zero only when registration was attempted and failed.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import REGISTRY_RECORD_FILE, RUN_INFO_FILE
from src.models.register_model import (
    build_registry_record, load_run_information, register_champion, write_registry_record,
)
from src.utils.config_loader import load_params
from src.utils.logger import get_logger

logger = get_logger("modelium.register")


def main() -> int:
    registry = load_params()["mlflow"]["registry"]
    run_info = load_run_information(RUN_INFO_FILE)

    version = register_champion(
        run_info,
        production_alias=registry["production_alias"],
        candidate_alias=registry["candidate_alias"],
    )

    # Written on every path, including a skip: DVC declares it as this stage's output,
    # and "nothing was registered, because tracking was off" is itself the outcome.
    record = build_registry_record(
        run_info, version,
        production_alias=registry["production_alias"],
        candidate_alias=registry["candidate_alias"],
    )
    write_registry_record(record, REGISTRY_RECORD_FILE)

    if version is None:
        print(f"Nothing registered — {record['skipped_reason']}. "
              f"The champion pipeline on disk is unaffected.")
        return 0

    name = run_info["registered_model_name"]
    status = "APPROVED" if run_info["promoted"] else "REJECTED"
    alias = registry["production_alias"] if run_info["promoted"] else registry["candidate_alias"]
    print(
        f"\nRegistered {name} version {version.version} [{status}]\n"
        f"  champion : {run_info['champion_model']}\n"
        f"  run       : {run_info['run_id']}\n"
        f"  threshold : {run_info['optimal_threshold']:.4f}\n"
        f"  resolves  : models:/{name}@{alias}"
    )
    if not run_info["promoted"]:
        print(
            "  NOTE      : this version failed its quality gates. The "
            f"'{registry['production_alias']}' alias was not assigned to it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
