from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_home_credit_tables(data_dir: str | Path, file_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load the Home Credit relational CSV tables from one directory."""
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    missing = []
    for name, filename in file_map.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        tables[name] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))
    return tables
