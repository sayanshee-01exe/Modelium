"""Step 11 — the container definition says what it must, and omits what it must not.

These assertions are about deployment safety rather than syntax. A Dockerfile that runs
as root, ships a database, or health-checks by scoring an applicant is valid Docker and
a bad service, and none of those mistakes announce themselves at build time.

No Docker daemon is required: the files are parsed, not executed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCKERFILE = PROJECT_ROOT / "Dockerfile"
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
DOCKERIGNORE = PROJECT_ROOT / ".dockerignore"
ENTRYPOINT = PROJECT_ROOT / "docker" / "entrypoint.sh"
SMOKE_TEST = PROJECT_ROOT / "scripts" / "docker_smoke_test.sh"


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def api_service(compose) -> dict:
    return compose["services"]["api"]


@pytest.fixture(scope="module")
def dockerignore() -> list[str]:
    return [line.strip() for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

def test_the_dockerfile_exists() -> None:
    assert DOCKERFILE.exists()


def test_it_uses_a_slim_official_python_base(dockerfile) -> None:
    assert re.search(r"^FROM python:3\.\d+-slim", dockerfile, re.MULTILINE)


def test_the_python_version_matches_the_pickled_artifact(dockerfile) -> None:
    """Unpickling a fitted sklearn/LightGBM pipeline across minor versions is not safe."""
    image_version = re.search(r"FROM python:(3\.\d+)-slim", dockerfile).group(1)
    assert image_version == f"{sys.version_info.major}.{sys.version_info.minor}"


def test_it_runs_as_a_non_root_user(dockerfile) -> None:
    """Root in a container is root on a shared kernel."""
    assert re.search(r"^USER\s+modelium", dockerfile, re.MULTILINE)
    user_line = dockerfile.index("USER modelium")
    entrypoint_line = dockerfile.index("ENTRYPOINT")
    assert user_line < entrypoint_line, "USER must precede the entrypoint"


def test_a_dedicated_user_is_created(dockerfile) -> None:
    assert "useradd" in dockerfile and "groupadd" in dockerfile


def test_python_runtime_flags_are_set(dockerfile) -> None:
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert "PYTHONUNBUFFERED=1" in dockerfile


def test_port_8000_is_exposed(dockerfile) -> None:
    assert re.search(r"^EXPOSE\s+8000", dockerfile, re.MULTILINE)


def test_it_never_enables_autoreload(dockerfile) -> None:
    """The reloader watches the filesystem and forks workers; neither belongs in a
    served image.

    Comments are stripped first: the Dockerfile explains *why* it omits `--reload`, and
    matching the raw text would flag that explanation as the very thing it warns against.
    """
    instructions = "\n".join(line for line in dockerfile.splitlines()
                              if not line.strip().startswith("#"))
    assert "--reload" not in instructions


def test_a_healthcheck_is_declared(dockerfile) -> None:
    assert "HEALTHCHECK" in dockerfile


def test_the_healthcheck_probes_health_not_predict(dockerfile) -> None:
    """A probe that scores an applicant turns monitoring into load."""
    block = dockerfile[dockerfile.index("HEALTHCHECK"):]
    assert "/health" in block
    assert "/predict" not in block


def test_the_image_installs_the_serving_subset(dockerfile) -> None:
    """The development manifest pulls DVC, Optuna and CatBoost — none of which the API
    imports, and together most of a multi-gigabyte image."""
    assert "requirements-api.txt" in dockerfile
    # Comments are stripped first: the manifest documents *which* training packages it
    # drops, and matching the raw text would flag that explanation as the thing it warns
    # against.
    manifest = (PROJECT_ROOT / "requirements-api.txt").read_text(encoding="utf-8")
    pinned = "\n".join(line for line in manifest.splitlines()
                       if line.strip() and not line.strip().startswith("#"))
    for training_only in ("catboost", "optuna", "dvc", "pytest"):
        assert training_only not in pinned, f"{training_only} is not needed to serve"


def test_dependencies_are_copied_before_source(dockerfile) -> None:
    """Otherwise every source edit reinstalls the whole dependency tree."""
    requirements = dockerfile.index("COPY requirements-api.txt")
    source = dockerfile.index("COPY --chown=modelium:modelium api/")
    assert requirements < source


def test_the_build_uses_multiple_stages(dockerfile) -> None:
    assert "AS builder" in dockerfile and "AS runtime" in dockerfile


def test_the_compiler_toolchain_stays_in_the_builder(dockerfile) -> None:
    """Shipping gcc in a serving image is size and attack surface for no benefit."""
    runtime = dockerfile[dockerfile.index("AS runtime"):]
    assert "build-essential" not in runtime


def test_the_openmp_runtime_is_present(dockerfile) -> None:
    """LightGBM and XGBoost link against it; without it the model import fails."""
    runtime = dockerfile[dockerfile.index("AS runtime"):]
    assert "libgomp1" in runtime


def test_no_mutable_state_is_baked_into_the_image(dockerfile) -> None:
    """A database in an image makes the image a snapshot of a moment."""
    for state in ("mlflow.db", "mlartifacts", "COPY artifacts", "COPY data"):
        assert state not in dockerfile, f"{state} must be mounted, not copied"


def test_no_machine_specific_path_is_hard_coded(dockerfile) -> None:
    assert "/Users/" not in dockerfile
    assert "/home/shee" not in dockerfile


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

def test_the_dockerignore_exists() -> None:
    assert DOCKERIGNORE.exists()


@pytest.mark.parametrize("pattern", [
    ".git", ".venv", "__pycache__", ".pytest_cache", ".DS_Store", "notebooks",
    "tests", "reports", "logs", "data", "mlruns", "mlartifacts", "mlflow.db",
])
def test_heavy_or_irrelevant_paths_are_excluded(dockerignore, pattern) -> None:
    assert pattern in dockerignore, f"{pattern} should not enter the build context"


def test_the_dvc_cache_is_excluded(dockerignore) -> None:
    assert any(entry.startswith(".dvc/cache") for entry in dockerignore)


@pytest.mark.parametrize("needed", ["api", "src", "params.yaml", "config",
                                   "requirements.txt", "requirements-api.txt"])
def test_files_the_image_needs_are_not_excluded(dockerignore, needed) -> None:
    assert needed not in dockerignore, f"{needed} is required inside the image"


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def test_the_compose_file_is_valid_yaml(compose) -> None:
    assert isinstance(compose, dict) and "services" in compose


def test_the_api_service_exists(api_service) -> None:
    assert api_service["build"]["dockerfile"] == "Dockerfile"


def test_port_8000_is_published(api_service) -> None:
    assert "8000:8000" in api_service["ports"]


def test_the_service_has_a_healthcheck(api_service) -> None:
    check = api_service["healthcheck"]
    assert "/health" in " ".join(check["test"])
    assert "/predict" not in " ".join(check["test"])


def test_the_healthcheck_allows_a_slow_cold_start(api_service) -> None:
    """The champion is a large pickled pipeline; a short start period would mark a
    perfectly healthy container as failed while it is still loading."""
    start_period = api_service["healthcheck"]["start_period"]
    assert int(str(start_period).rstrip("s")) >= 30


def test_the_model_uri_points_at_the_champion_alias(api_service) -> None:
    assert api_service["environment"]["MODELIUM_MODEL_URI"].endswith("@champion")


def test_the_sqlite_uri_is_absolute(api_service) -> None:
    """sqlite:///app/... is a *relative* path named "app" and would silently create an
    empty database with no champion in it. The absolute form needs four slashes."""
    uri = api_service["environment"]["MLFLOW_TRACKING_URI"]
    assert uri.startswith("sqlite:////"), uri


def test_the_registry_and_artifacts_are_mounted(api_service) -> None:
    mounts = " ".join(api_service["volumes"])
    assert "mlflow.db" in mounts
    assert "mlartifacts" in mounts
    assert "artifacts" in mounts


def test_the_feature_store_is_mounted_as_one_file(api_service) -> None:
    """Not the whole data directory, which is ~3 GB of raw CSVs."""
    mounts = api_service["volumes"]
    assert any("test_features.parquet" in mount for mount in mounts)
    assert not any(mount.startswith("./data:") for mount in mounts)


@pytest.mark.parametrize("source", [
    "./mlflow.db", "./mlartifacts", "./artifacts",
    "./data/processed/test_features.parquet",
])
def test_state_the_api_only_reads_is_mounted_read_only(api_service, source) -> None:
    """The API reads the registry; it never registers, promotes or moves an alias."""
    mount = next(m for m in api_service["volumes"] if m.startswith(f"{source}:"))
    assert mount.endswith(":ro"), f"{source} should be read-only"


def test_the_whole_repository_is_not_mounted(api_service) -> None:
    for mount in api_service["volumes"]:
        assert not mount.startswith(".:"), "mounting the repo defeats the image"
        assert not mount.startswith("./:")


def test_a_writable_volume_exists_for_the_rebased_registry(compose, api_service) -> None:
    assert "modelium-runtime-state" in compose["volumes"]
    assert any("modelium-runtime-state" in mount for mount in api_service["volumes"])


def test_the_container_is_not_privileged(api_service) -> None:
    assert api_service.get("privileged") is not True


def test_resource_limits_leave_room_for_the_model(api_service) -> None:
    """Limits that prevent SHAP or model loading would be worse than none."""
    limits = api_service["deploy"]["resources"]["limits"]
    assert int(str(limits["memory"]).rstrip("G")) >= 4
    assert float(str(limits["cpus"]).strip('"')) >= 2


def test_no_secret_is_present_in_the_compose_file() -> None:
    text = COMPOSE_FILE.read_text(encoding="utf-8").lower()
    for marker in ("password", "secret", "api_key", "token", "aws_access"):
        assert marker not in text, f"{marker!r} should never appear here"


def test_no_machine_specific_path_in_the_compose_file() -> None:
    assert "/Users/" not in COMPOSE_FILE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The optional MLflow service shares one store
# ---------------------------------------------------------------------------

def test_the_mlflow_service_is_opt_in(compose) -> None:
    assert compose["services"]["mlflow"]["profiles"] == ["tools"]


def test_mlflow_and_the_api_read_the_same_database(compose, api_service) -> None:
    """Two stores would show a registry unrelated to what is being served."""
    mlflow_service = compose["services"]["mlflow"]
    command = " ".join(str(part) for part in mlflow_service["command"])
    assert api_service["environment"]["MLFLOW_TRACKING_URI"] in command
    assert any("modelium-runtime-state" in mount for mount in mlflow_service["volumes"])


# ---------------------------------------------------------------------------
# Entrypoint and smoke script
# ---------------------------------------------------------------------------

def test_the_entrypoint_rebases_the_registry_before_serving() -> None:
    """Mounted MLflow records point at the *host's* absolute paths, which do not exist
    in the container. Without this the champion cannot load."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "prepare_container_registry.py" in text
    assert "set -euo pipefail" in text
    assert text.index("prepare_container_registry.py") < text.index('exec "$@"')


def test_the_entrypoint_is_executable() -> None:
    assert ENTRYPOINT.stat().st_mode & 0o111


def test_the_smoke_script_cleans_up_on_failure() -> None:
    """A failed run must not leave a stack running."""
    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert "trap cleanup EXIT" in text
    # The script invokes compose through $COMPOSE, so match the teardown itself.
    assert "down --remove-orphans" in text


def test_the_smoke_script_asserts_no_exact_probability() -> None:
    """Pinning a probability would tie the test to one trained artifact."""
    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert "predicted_class" in text and "threshold" in text


# ---------------------------------------------------------------------------
# Serving stays out of the pipeline
# ---------------------------------------------------------------------------

def test_docker_is_not_a_dvc_stage() -> None:
    """Serving is a deployment concern, not a reproducible batch step."""
    stages = yaml.safe_load((PROJECT_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    for forbidden in ("serve", "api", "docker", "deploy"):
        assert forbidden not in stages
    for name, stage in stages.items():
        assert "docker" not in stage["cmd"], f"{name} invokes docker"
