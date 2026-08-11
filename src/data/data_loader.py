from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_home_credit_tables(data_dir: str | Path, file_map: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Load the Home Credit relational CSV tables from one directory.

    Note on the file-existence check below: primary responsibility for data contracts
    belongs to `src/data/data_validation.py::validate_raw_files`, which the training
    entry point calls *before* this function so a missing table fails before any I/O.
    The check here is a deliberate **secondary safety net** for callers that use the
    loader directly (notebooks, ad-hoc scripts) and therefore skip validation. It is
    intentionally duplicated rather than removed — deleting it would make those callers
    fail with a confusing pandas error instead of a clear one.
    """
    data_dir = Path(data_dir)
    tables: dict[str, pd.DataFrame] = {}
    missing = []
    logger.info("Loading Home Credit tables from %s", data_dir)
    for name, filename in file_map.items():
        path = data_dir / filename
        if not path.exists():
            missing.append(str(path))
            continue
        tables[name] = pd.read_csv(path)
    if missing:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(missing))
    logger.info("Loaded %d tables (%s)", len(tables), ", ".join(tables))
    return tables
