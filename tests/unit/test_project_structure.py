"""Sprint v3 Task 1 — repo scaffolding, git, and environment.

These tests pin the acceptance criteria for the consolidation scaffold: the directory
layout the DVC stage graph writes into, the raw-data symlinks, ignore rules, the
dependency set, and the D6 fix (application_test.csv missing from config.DATA_FILES).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------------------------------- layout

REQUIRED_DIRS = [
    "data/raw",
    "data/interim",
    "data/processed",
    "logs",
    "models",
    "notebooks",
    "references",
    "reports/figures",
    "tests/unit",
]


@pytest.mark.parametrize("relpath", REQUIRED_DIRS)
def test_required_directory_exists(relpath: str) -> None:
    path = PROJECT_ROOT / relpath
    assert path.is_dir(), f"missing directory: {relpath}"


def test_tracked_output_dirs_have_gitkeep() -> None:
    """Empty output dirs must survive a clone, or DVC stages write into nothing.

    logs/ and data/raw/ are deliberately excluded: both are fully git-ignored, and git
    cannot re-include a file inside an excluded directory, so a .gitkeep there would
    never be tracked. The logger mkdir()s logs/ at runtime instead.
    """
    for relpath in ("data/interim", "data/processed", "reports/figures", "models"):
        assert (PROJECT_ROOT / relpath / ".gitkeep").exists(), f"{relpath}/.gitkeep missing"


# ------------------------------------------------------------------------- raw data

EXPECTED_RAW_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
]


@pytest.mark.parametrize("filename", EXPECTED_RAW_FILES)
def test_raw_csv_present_and_resolves(filename: str) -> None:
    """Symlinks must resolve to a real, non-empty file — a dangling link passes exists()
    checks on the link itself but breaks every downstream stage."""
    path = PROJECT_ROOT / "data" / "raw" / filename
    assert path.exists(), f"data/raw/{filename} missing"
    resolved = path.resolve(strict=False)
    assert resolved.is_file(), f"data/raw/{filename} does not resolve to a file"
    assert resolved.stat().st_size > 0, f"data/raw/{filename} resolves to an empty file"


def test_raw_data_is_symlinked_not_copied() -> None:
    """~2.7 GB — a copy would be a third one on disk (see sprints/v3/PRD.md)."""
    path = PROJECT_ROOT / "data" / "raw" / "application_train.csv"
    assert path.is_symlink(), "raw CSVs should be symlinks, not copies"


def test_column_dictionary_available_in_references() -> None:
    """Needed to write the per-table aggregation specs in Task 5."""
    path = PROJECT_ROOT / "references" / "HomeCredit_columns_description.csv"
    assert path.exists() and path.resolve(strict=False).is_file()


# -------------------------------------------------------------------------- ignores

REQUIRED_GITIGNORE_PATTERNS = [
    "data/raw/",
    ".venv/",
    "mlruns/",
    "logs/",
    "*.pkl",
    "*.joblib",
    "__pycache__/",
    ".DS_Store",
]


@pytest.mark.parametrize("pattern", REQUIRED_GITIGNORE_PATTERNS)
def test_gitignore_covers_pattern(pattern: str) -> None:
    gitignore = PROJECT_ROOT / ".gitignore"
    assert gitignore.exists(), ".gitignore missing"
    lines = {ln.strip() for ln in gitignore.read_text().splitlines()}
    assert pattern in lines, f".gitignore missing pattern: {pattern}"


def test_dvcignore_exists() -> None:
    assert (PROJECT_ROOT / ".dvcignore").exists()


def test_raw_data_not_staged_in_git() -> None:
    """A 2.7 GB accidental commit is painful to undo — assert the ignore actually works."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-q", "data/raw/application_train.csv"],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, "data/raw CSVs are NOT git-ignored"


# --------------------------------------------------------------------- dependencies

REQUIRED_PACKAGES = [
    "pandas", "numpy", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm",
    "catboost", "optuna", "mlflow", "dvc", "shap", "joblib", "matplotlib",
    "seaborn", "missingno", "pyarrow", "pytest",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_requirements_declares_package(package: str) -> None:
    text = (PROJECT_ROOT / "requirements.txt").read_text().lower()
    assert package.lower() in text, f"requirements.txt missing {package}"


@pytest.mark.parametrize(
    "module",
    ["pandas", "numpy", "sklearn", "imblearn", "xgboost", "lightgbm", "catboost",
     "optuna", "mlflow", "shap", "joblib", "matplotlib", "seaborn", "missingno",
     "pyarrow"],
)
def test_package_importable(module: str) -> None:
    """Declared in requirements is not the same as installed and importable."""
    pytest.importorskip(module, reason=f"{module} not installed in the active interpreter")


def test_dvc_cli_available() -> None:
    """Check the console script next to the running interpreter, then fall back to PATH.

    `.venv/bin/python -m pytest` does not put `.venv/bin` on PATH, so a bare
    shutil.which("dvc") would fail even with dvc correctly installed in the venv.
    """
    alongside = Path(sys.executable).parent / "dvc"
    assert alongside.exists() or shutil.which("dvc"), "dvc CLI not found"


def test_bare_venv_replaced() -> None:
    """The original venv/ held only pip; .venv/ is the real environment."""
    assert not (PROJECT_ROOT / "venv").exists(), "stale bare venv/ still present"
    assert (PROJECT_ROOT / ".venv").is_dir(), ".venv/ missing"


# ------------------------------------------------------------------------------ vcs


def test_git_repository_initialized() -> None:
    assert (PROJECT_ROOT / ".git").is_dir(), "git repository not initialized"


def test_dvc_repository_initialized() -> None:
    assert (PROJECT_ROOT / ".dvc").is_dir(), "dvc not initialized"


# --------------------------------------------------------------------- config (D6)


def test_config_includes_application_test() -> None:
    """Defect D6: DATA_FILES omitted application_test.csv, leaving no submission path."""
    from config.config import DATA_FILES

    assert "application_test" in DATA_FILES, "DATA_FILES missing application_test (D6)"
    assert DATA_FILES["application_test"] == "application_test.csv"


def test_config_data_files_all_resolve_on_disk() -> None:
    """Ties config to reality: every declared table must exist where DATA_DIR points."""
    from config.config import DATA_DIR, DATA_FILES

    missing = [fn for fn in DATA_FILES.values() if not (DATA_DIR / fn).exists()]
    assert not missing, f"DATA_FILES entries not found under {DATA_DIR}: {missing}"


def test_config_data_dir_points_at_raw() -> None:
    from config.config import DATA_DIR

    assert DATA_DIR == PROJECT_ROOT / "data" / "raw"
