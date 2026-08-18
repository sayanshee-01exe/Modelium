"""Print the last monitoring run's status in the terminal.

Read-only. The JSON summary is the authority; this renders it so an operator can see
where a run landed without opening a file or the MLflow UI. Every section status is
shown alongside the overall one, because the overall status is by design only the worst
of them and says nothing about which section produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SUMMARY_FILE = PROJECT_ROOT / "artifacts" / "monitoring" / "monitoring_summary.json"

# Rendered next to each status so severity survives a terminal without colour.
MARKERS = {"stable": "ok", "healthy": "ok", "warning": "WARN", "critical": "CRIT",
           "insufficient_data": "n/a", "not_available": "n/a", "error": "ERR"}


def main() -> int:
    if not SUMMARY_FILE.exists():
        print(f"No monitoring summary at {SUMMARY_FILE.relative_to(PROJECT_ROOT)}.\n"
              f"Run `make monitor` first.")
        return 0

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    overall = summary.get("overall_status", "unknown")

    print(f"\nOverall status: {overall.upper()}  "
          f"[{MARKERS.get(overall, '?')}]  ({summary.get('monitoring_timestamp')})")
    print(f"model    : {summary.get('registered_model_name')} "
          f"v{summary.get('model_version')} (@{summary.get('model_alias')})")
    print(f"reference: {summary.get('reference_rows', 0):,} rows — "
          f"{summary.get('reference_dataset')}")
    print(f"current  : {summary.get('current_rows', 0):,} rows — "
          f"{summary.get('current_dataset')}")

    batch = summary.get("current_batch_metadata") or {}
    if batch.get("demonstration"):
        drifted = batch.get("simulated_drift_features") or []
        print(f"           DEMONSTRATION BATCH — labels are {batch.get('label_source')}, "
              f"not observed outcomes")
        if drifted:
            print(f"           simulated drift injected into: {', '.join(drifted)}")

    print("\nsection statuses")
    for name, status in (summary.get("section_statuses") or {}).items():
        print(f"  {name:<18} {status:<18} [{MARKERS.get(status, '?')}]")

    print(f"\nfeatures : {summary.get('monitored_features', 0)} monitored — "
          f"{summary.get('critical_features', 0)} critical, "
          f"{summary.get('warning_features', 0)} warning "
          f"({summary.get('drifted_feature_ratio', 0):.1%} drifted)")
    print(f"labels   : {'available' if summary.get('labels_available') else 'absent'}")
    print(f"\n{summary.get('action_taken', '')}")
    print(f"\nfull report: {(summary.get('artifacts') or {}).get('monitoring_report')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
