"""Centralised logging configuration for the Modelium pipeline.

Every module obtains its logger via ``get_logger(__name__)``.  On the first call the
root ``modelium`` logger is configured with:

* a **console** ``StreamHandler`` (INFO, to stdout)
* a **file** ``RotatingFileHandler`` (DEBUG, to ``logs/pipeline.log``)

Subsequent calls return child loggers that inherit these handlers — no per-module
boilerplate required.  The handler-guard pattern (``if not logger.handlers``) prevents
duplicate output when multiple modules import this during the same process.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "pipeline.log"

# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Rotating file settings
# ---------------------------------------------------------------------------
MAX_BYTES = 5 * 1024 * 1024   # 5 MB per file
BACKUP_COUNT = 3               # keep 3 rotated copies


def _configure_root_logger() -> logging.Logger:
    """One-time setup for the ``modelium`` root logger.

    Called exactly once; the guard in ``get_logger`` ensures subsequent imports
    reuse the existing handlers rather than stacking new ones.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("modelium")
    root.setLevel(logging.DEBUG)

    # --- console handler (INFO) -----------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # --- file handler (DEBUG) -------------------------------------------
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    return root


# Module-level flag — cheaper than inspecting ``root.handlers`` on every call.
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a named child of the ``modelium`` root logger.

    The first invocation configures the root with console + file handlers;
    subsequent calls skip that work.  All existing callers of
    ``get_logger(__name__)`` continue to work unchanged.

    Args:
        name: Dotted module path (typically ``__name__``).

    Returns:
        A ``logging.Logger`` whose output flows to both the console and
        ``logs/pipeline.log``.
    """
    global _configured
    if not _configured:
        _configure_root_logger()
        _configured = True

    # Ensure every caller lives under the ``modelium`` namespace so it
    # inherits the root handlers.  Names that already start with ``modelium``
    # (e.g. ``modelium.train``) are left untouched.
    if not name.startswith("modelium"):
        name = f"modelium.{name}"

    return logging.getLogger(name)
