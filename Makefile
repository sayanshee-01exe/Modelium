# Entry points for the pipeline, existing for one reason: DVC stage commands are plain
# `python scripts/....py`, and `python` resolves against PATH. Without the project venv
# on PATH that is whatever interpreter the shell happens to offer — on this machine
# /opt/anaconda3/bin/python, which has pandas but no mlflow, xgboost or lightgbm.
#
# The failure mode that makes this worth a Makefile is not the loud one. `train` dies
# immediately with "No module named 'mlflow'"; `validate` and `prepare` run to
# completion under the wrong interpreter and write real outputs, so a pipeline can be
# reproduced half under one Python and half under another without anyone noticing.
#
# VENV_BIN derives from this file's own location, so nothing here is specific to one
# machine or one user's home directory. Every expansion is quoted: the repository path
# on the author's machine contains spaces, and an unquoted $(PY) silently splits into
# two arguments and reports the environment as broken.

VENV_BIN := $(CURDIR)/.venv/bin
PY       := $(VENV_BIN)/python
DVC_ENV  := PATH="$(VENV_BIN):$$PATH"

.PHONY: help venv-check repro dag status train register register-force predict test compile mlflow-ui registry-show

help:
	@echo "make venv-check    verify .venv exists and carries the training dependencies"
	@echo "make repro         run every out-of-date stage (correct interpreter)"
	@echo "make train         reproduce the train stage only"
	@echo "make register      reproduce the register stage only"
	@echo "make register-force  re-register an unchanged run (safe single-stage force)"
	@echo "make predict       reproduce the predict stage only"
	@echo "make dag           show the stage graph"
	@echo "make status        show what DVC considers out of date"
	@echo "make test          run the unit suite"
	@echo "make compile       byte-compile src/ and scripts/"
	@echo "make mlflow-ui     browse runs and the model registry at 127.0.0.1:5000"
	@echo "make registry-show print the registered versions and their aliases"

# Fails with a readable message rather than letting a stage discover the gap mid-run.
venv-check:
	@test -x "$(PY)" || { echo "No interpreter at $(PY). Create it:"; \
	  echo "  python -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
	@"$(PY)" -c "import mlflow, xgboost, lightgbm, sklearn, dvc" || { \
	  echo "$(PY) is missing training dependencies. Install them:"; \
	  echo "  .venv/bin/pip install -r requirements.txt"; exit 1; }
	@echo "OK: $$("$(PY)" -c 'import sys; print(sys.executable)')"

repro: venv-check
	$(DVC_ENV) "$(VENV_BIN)/dvc" repro

train: venv-check
	$(DVC_ENV) "$(VENV_BIN)/dvc" repro train

register: venv-check
	$(DVC_ENV) "$(VENV_BIN)/dvc" repro register

# Re-run registration against an unchanged run, e.g. to confirm idempotency.
#
# Use this rather than `dvc repro register --force`. DVC's --force propagates *upstream*:
# it would force validate, prepare and train as well, and because DVC removes a stage's
# outputs before re-running it, an interrupted --force leaves the champion deleted. The
# -s/--single-item flag confines the force to this stage alone.
register-force: venv-check
	$(DVC_ENV) "$(VENV_BIN)/dvc" repro --single-item --force register

predict: venv-check
	$(DVC_ENV) "$(VENV_BIN)/dvc" repro predict

dag:
	$(DVC_ENV) "$(VENV_BIN)/dvc" dag

status:
	$(DVC_ENV) "$(VENV_BIN)/dvc" status

test:
	"$(PY)" -m pytest tests/unit -q

compile:
	"$(PY)" -m compileall -q src scripts

# Local SQLite backend, matching mlflow.tracking_uri in params.yaml. No remote server.
mlflow-ui:
	"$(VENV_BIN)/mlflow" ui --backend-store-uri sqlite:///mlflow.db

registry-show:
	@"$(PY)" scripts/show_registry.py
