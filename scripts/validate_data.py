"""DVC stage 1 — validate the raw data contract.

Loads every raw table and runs the Step 2 validators. Nothing is cleaned, aggregated,
transformed or trained here: this stage exists so a broken input fails in its own step,
naming the table and the offending values, rather than surfacing three stages later as
a NaN column or an empty join.

Its output is a small JSON report. That is what makes the stage cacheable — with no
output DVC would have no way to know the contract still holds and would re-read 2.5 GB
on every `dvc repro`.

Orchestration only; every check lives in `src/data/data_validation.py`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_DIR, DATA_FILES, TARGET_COL, VALIDATION_REPORT_FILE
from src.data.data_loader import load_home_credit_tables
from src.data.data_validation import validate_raw_files, validate_raw_tables
from src.utils.logger import get_logger

logger = get_logger("modelium.validate")


def main():
    logger.info("Validating raw dataset files...")
    validate_raw_files(DATA_DIR, DATA_FILES)

    tables = load_home_credit_tables(DATA_DIR, DATA_FILES)

    logger.info("Validating loaded dataframes...")
    report = validate_raw_tables(tables, target_col=TARGET_COL)

    VALIDATION_REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_column": TARGET_COL,
        "tables": report,
        "total_rows": int(sum(t["rows"] for t in report.values())),
    }
    with open(VALIDATION_REPORT_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Raw data validation passed for {len(report)} table(s):")
    for name, stats in sorted(report.items()):
        print(f"  {name:<22} {stats['rows']:>12,} rows x {stats['columns']:>3} cols")
    print(f"\nReport written to {VALIDATION_REPORT_FILE}")


if __name__ == "__main__":
    main()
