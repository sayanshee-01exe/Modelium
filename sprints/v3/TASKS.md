# Sprint v3 — Tasks

## Status: In progress — 1 / 14 complete

Ordered by dependency. P0 = blocks the sprint goal, P1 = required for "production-style",
P2 = nice-to-have if time allows.

---

- [x] Task 1: Repo scaffolding, git, environment (P0)
  - Acceptance: `git init` done with a `.gitignore` covering `data/raw/*.csv`, `.venv/`,
    `mlruns/`, `logs/`, `models/*.pkl`, `models/*.joblib`, `__pycache__/`, `.DS_Store`;
    `data/{raw,interim,processed}`, `logs/`, `reports/figures/`, `notebooks/`, `tests/unit/`
    exist; all 7 Home Credit CSVs **plus `application_test.csv`** symlinked into `data/raw/`
    from `../smart_loan_approval_system_claude/home-credit-default-risk/`; the bare `venv/`
    (only `pip` installed) is deleted and replaced by a working `.venv` (Python 3.13);
    `requirements.txt` extended with `mlflow`, `dvc`, `catboost`, `optuna`, `pytest`,
    `pyarrow`, `seaborn`, `missingno` and installs cleanly; `dvc init` succeeds. **Fixes D6, D8.**
  - Files: `.gitignore`, `.dvcignore`, `requirements.txt`, `config/config.py`, `data/raw/`
  - Completed: 2026-08-11 — scaffolded the consolidated repo. `git init -b main` (parent dir
    confirmed not a repo first, so no nesting) + `dvc init`. Created `data/{raw,interim,
    processed}`, `logs/`, `models/`, `notebooks/`, `references/`, `reports/figures/`,
    `tests/unit/`, with `.gitkeep` in the tracked-but-empty output dirs. Symlinked all 8 raw
    CSVs into `data/raw/` and the column dictionary into `references/` — all 9 verified to
    resolve to non-empty files (2.7 GB total, not copied). Replaced the bare `venv/` (12 MB,
    contained only pip) with `.venv` on Python 3.13.5 — the same interpreter the superseded
    `_claude` project used — and installed the expanded `requirements.txt` (added catboost,
    optuna, mlflow, dvc, seaborn, missingno, pyarrow, pytest, pip-audit). **D6 fixed**:
    `application_test.csv` added to `config.DATA_FILES`, which previously had no entry for
    the holdout applicants and therefore no scoring path.
  - Tests: `tests/unit/test_project_structure.py`, 69 tests, written before implementation and
    confirmed RED first (33 failed / 35 passed / 1 skipped), now all passing. Covers directory
    layout, symlink *resolution* (a dangling link passes a naive `exists()` check but breaks
    every downstream stage), `.gitignore` patterns, a `git check-ignore` assertion that the
    2.7 GB of raw data genuinely cannot be committed, dependency declaration + importability,
    and that every `DATA_FILES` entry resolves on disk.
  - Two of my own tests were wrong and were corrected rather than worked around:
    `test_tracked_output_dirs_have_gitkeep` originally included `logs/`, but git cannot
    re-include a file inside a fully-excluded directory, so a `.gitkeep` there would never be
    tracked; and `test_dvc_cli_available` used a bare `shutil.which`, which fails under
    `.venv/bin/python -m pytest` because `.venv/bin` is not on `PATH`.
  - Security: semgrep `--config auto` clean (0 findings, 290 rules, 24 files). `pip-audit`
    found 2 transitive advisories, both **accepted with rationale recorded in
    `requirements.txt`**: `cryptography` 49.x PYSEC-2026-3552 (PKCS#7/S-MIME decryption
    oracle) is fixed in 50.0.0, but mlflow 3.15.1 — the newest release — pins
    `cryptography<50`, so requiring the fix makes `pip install -r requirements.txt`
    unresolvable, and the affected `pkcs7_decrypt_*` APIs are never called here.
    `diskcache` 5.6.3 PYSEC-2026-2447 (pickle deserialization) has no upstream fix at all —
    5.6.3 is the newest release — and requires write access to the local DVC cache. `pip
    check` reports no broken requirements and a `--dry-run` install resolves cleanly.
  - Follow-ups for later tasks (noted, not fixed here — one task per invocation):
    - Adding `application_test` to `DATA_FILES` means `scripts/train.py`'s
      `load_home_credit_tables(DATA_DIR, DATA_FILES)` now also reads a table the training
      path never consumes. Harmless today (that script cannot run yet because of D1/D2) and
      resolved by Task 5's per-stage loader rework.
    - An empty, 0-byte `.gitigone` file (typo for `.gitignore`) was found in the project root,
      created outside this workflow. Left in place and deliberately excluded from the commit
      rather than deleted; remove it if it was a slip.

- [ ] Task 2: Logging + typed exceptions, wired for real (P0)
  - Acceptance: `src/utils/logger.py` configures console + `RotatingFileHandler` to
    `logs/pipeline.log` (replacing the current bare `basicConfig` helper that nothing calls);
    `src/utils/exceptions.py` defines `DataValidationError`, `ConfigError`, `PipelineStageError`;
    a `validate_dataframe(df, name, required_columns)` helper raises `DataValidationError` on
    empty frames or missing columns. Every stage added in later tasks logs start/end, row and
    column counts, and elapsed time. **Starts fixing D4.**
  - Files: `src/utils/logger.py`, `src/utils/exceptions.py`, `tests/unit/test_utils.py`

- [ ] Task 3: `params.yaml` + config split (P0)
  - Acceptance: `params.yaml` holds every tunable grouped by stage (`Data_Cleaning`,
    `Data_Preparation`, `Feature_Engineering`, `Data_Preprocessing`, `Feature_Selection`,
    `Train`, `Evaluation`), including the missing-value threshold, income/children caps,
    aggregation specs, split ratios, correlation cutoff, and per-model hyperparameters;
    `config/config.py` is reduced to **path constants only** (the current `RANDOM_STATE`,
    `HIGH_MISSING_THRESHOLD`, `LOW_CARDINALITY_THRESHOLD`, `DOWNSAMPLE_RATIO` move out);
    a `load_params()` helper reads it and raises `ConfigError` on a missing key.
  - Files: `params.yaml`, `config/config.py`, `src/utils/__init__.py`, `tests/unit/test_params.py`

- [ ] Task 4: Port `_claude`'s cleaning decisions into `src/data/data_cleaning.py` (P0)
  - Acceptance: `clean_application_data()` implements, each with a code comment naming the EDA
    finding it came from: `DAYS_EMPLOYED == 365243` → NaN + `DAYS_EMPLOYED_ANOM` flag;
    `DAYS_*` flipped positive; `CODE_GENDER == 'XNA'` → train mode; `ORGANIZATION_TYPE == 'XNA'`
    **left unchanged** with an `assert` proving it is exactly the `DAYS_EMPLOYED_ANOM`
    population (100% overlap, 55,374 rows — real signal, not an error); `OCCUPATION_TYPE` NaN
    split into `'Not_Employed'` (~55,372) vs `'Unknown'` (~41,019) rather than blanket mode;
    `AMT_INCOME_TOTAL` capped at the train p99.9 (fixes the 117,000,000 outlier);
    `CNT_CHILDREN` capped at 10; columns >`missing_threshold` dropped **except `EXT_SOURCE_1`**,
    which is explicitly protected. Runnable as `python src/data/data_cleaning.py` (DVC stage
    entry point) writing `data/interim/application_clean.parquet`.
  - Files: `src/data/data_cleaning.py`, `tests/unit/test_data_cleaning.py`
  - Note: port the 17 assertions from `_claude`'s `tests/unit/test_data_cleaning.py`; they
    encode these findings and should pass against the new module output.

- [ ] Task 5: Rewrite relational aggregation — fix the silent data loss (P0)
  - Rescoped 2026-08-10 after validating `src/` against the real CSVs. The original premise
    ("1000+ columns, RAM exhausted by width") was **wrong**: measured output is 293 aggregated
    → 415 merged → ~535 post-OHE → ~0.98 GB dense matrix, which is fine. The genuine problems
    are the raw load footprint and two silent data-loss bugs.
  - Acceptance:
    - **D9** — every child table contributes an explicit, correctly-named record-count feature
      (`bureau_record_count`, `prev_application_count`, `pos_record_count`, `cc_record_count`,
      `inst_record_count`). Regression test asserts all six counts survive
      `drop_low_information_columns`; today 5 of 6 are destroyed because `count` is attached to
      `numeric[0]` (an `SK_ID_*` column) and then dropped by the `"SK_ID"` name rule.
    - **D10** — the 22 discarded child-table categoricals are encoded rather than dropped, at
      minimum: `bureau_balance.STATUS` → per-applicant DPD-bucket counts/shares and worst
      status; `previous_application.NAME_CONTRACT_STATUS` → approved/refused counts + refusal
      rate; `CODE_REJECT_REASON` → top-reason counts; `bureau.CREDIT_ACTIVE`/`CREDIT_TYPE` →
      active-loan counts by type. Test asserts a non-zero delinquency signal reaches the model.
    - **D11** — `bb_*` columns are not blindly re-aggregated through `bureau`; only meaningful
      composites (min-of-min, max-of-max, sum-of-count) are kept, dropping mean-of-sum noise.
    - **D1** — `load_home_credit_tables` no longer holds all seven tables resident (~7 GB, plus
      an `optimize_memory` `.copy()` per table). Each child table is loaded → aggregated →
      released before the next; `optimize_memory` downcasts in place instead of copying. Peak
      RSS measured and recorded in the task notes.
    - ID-like columns excluded *before* aggregation rather than dropped after; per-table agg
      specs declared in `params.yaml`.
  - Files: `src/data/data_preparation.py`, `src/data/data_loader.py`, `src/data/data_cleaning.py`,
    `params.yaml`, `tests/unit/test_data_preparation.py`
  - Verified-fine, do not "fix": all 7 filenames match disk; every join key present as assumed
    (incl. `bureau_balance` having only `SK_ID_BUREAU`); all 14 column names referenced in
    `feature_engineering.py` exist in `application_train`; float32-before-`groupby.sum()`
    precision loss is 1.3e-07 relative / 0.58 currency units worst case — negligible.

- [ ] Task 6: Feature engineering cleanup + three-way split (P0)
  - Acceptance: `PAYMENT_RATE` (identical to `ANNUITY_CREDIT_RATIO`) removed (**D5**);
    `EXT_SOURCE_WEIGHTED` given the same NaN semantics as `EXT_SOURCE_MEAN` — renormalize
    weights over the non-null sources instead of propagating NaN (**D7**; measured: currently
    NaN for **64.4%** of rows vs 0.1% for `EXT_SOURCE_MEAN`, and `EXT_SOURCE_STD` NaN for 12%.
    Test asserts weighted-vs-mean null rates match within 1pp); the `_claude` v1
    ratio features (credit/income, annuity/income, credit/annuity, employment-to-age, log
    transforms of skewed monetary columns) are present and reconciled with the existing
    domain features, deduplicated where they overlap; data split **train/val/test** (not
    train/test) stratified on `TARGET`, written to `data/interim/`.
  - Files: `src/features/feature_engineering.py`, `tests/unit/test_feature_engineering.py`

- [ ] Task 7: Preprocessing + feature selection stages (P0)
  - Acceptance: `build_preprocessor` fits on **train only** and is persisted to
    `models/preprocessor.joblib`; `IQRClipper` is either wired into the numeric pipeline or
    deleted — no dead class (**D4**); a new `src/features/feature_selection.py` implements
    variance / correlation-cutoff / importance pruning driven by `params.yaml` and writes
    `models/selected_features.json`. Transformed train/val/test land in `data/processed/`.
  - Files: `src/features/data_preprocessing.py`, `src/features/feature_selection.py`,
    `tests/unit/test_preprocessing.py`, `tests/unit/test_feature_selection.py`

- [ ] Task 8: Trim the model portfolio + MLflow tracking (P0)
  - Acceptance: portfolio cut from 8 to 5 — Logistic Regression, Random Forest, XGBoost,
    LightGBM, CatBoost — dropping `GradientBoostingClassifier`, calibrated `LinearSVC`,
    `AdaBoost`, and the standalone `DecisionTree` (**D2**); hyperparameters read from
    `params.yaml`; every fit logged to MLflow (params, metrics, model artifact, run tags) into
    the local `mlruns/` store with `MLFLOW_ALLOW_FILE_STORE=true` set as `_claude`'s
    `src/utils.py` documents; a full training run completes in a recorded, laptop-feasible
    wall time.
  - Files: `src/models/train.py`, `params.yaml`, `tests/unit/test_train.py`

- [ ] Task 9: Evaluation on disjoint splits + threshold fix (P0)
  - Acceptance: model selection and F1-optimal threshold are computed on the **validation**
    split; the **test** split is scored exactly once for the final report (**D3**);
    `reports/metrics.json` + `reports/model_leaderboard.csv` + `run_information.json` written;
    ROC/PR/confusion figures produced via the currently-orphaned
    `src/visualization/plots.py` (**D4**); README records the champion's holdout ROC-AUC
    **against the 0.7692 baseline**, stated plainly whether or not it beats it.
  - Files: `src/models/evaluation.py`, `src/models/threshold.py`, `src/visualization/plots.py`,
    `tests/unit/test_evaluation.py`

- [ ] Task 10: `dvc.yaml` — wire the whole thing together (P0)
  - Acceptance: all 8 stages from the PRD's stage graph declared with correct `deps` / `params`
    / `outs` / `metrics`; each `src/` module has a `if __name__ == "__main__":` entry point
    (they are currently pure-function modules with none); `dvc repro` runs the pipeline
    end-to-end from raw CSVs on a clean cache; changing one value in `params.yaml` re-runs only
    the affected downstream stages; `dvc.lock` committed.
  - Files: `dvc.yaml`, `dvc.lock`, all `src/**/*.py` (entry points), `scripts/train.py`
    (reduced to a thin local wrapper)

- [ ] Task 11: `register_model` stage (P1)
  - Acceptance: `src/models/register_model.py` reads `run_information.json` and registers the
    champion run into the MLflow model registry under a stable name, mirroring the reference
    repo's final stage; re-running is idempotent (new version, not a duplicate entry).
  - Files: `src/models/register_model.py`, `tests/unit/test_register_model.py`

- [ ] Task 12: Activate explainability + monitoring modules (P1)
  - Acceptance: `src/explainability/shap_explainer.py` extended with `_claude`'s
    `explain_prediction` (per-applicant top-N features with direction + 3-tier risk category)
    and actually invoked by the evaluation stage or a notebook, producing global beeswarm/bar
    plots into `reports/figures/`; `src/monitoring/drift.py` (PSI) and
    `src/monitoring/fairness.py` called against train-vs-test to emit
    `reports/drift_report.json` and `reports/fairness_report.csv`. No module in `src/` remains
    unreferenced — **closes D4**.
  - Files: `src/explainability/shap_explainer.py`, `src/monitoring/*.py`,
    `tests/unit/test_explain.py`, `tests/unit/test_monitoring.py`
  - Note: port `_claude`'s 9 SHAP tests, incl. the margin-reconstruction identity check
    (`sum(shap_values, axis=1) + expected_value == model.predict(X, raw_score=True)`).

- [ ] Task 13: Port notebooks for reporting (P2)
  - Acceptance: `_claude`'s `notebooks/1..7` copied in and reworked to **call into `src/`**
    rather than redefine logic, re-executed against the new 7-table data, with an added
    notebook section comparing single-table vs relational feature importance.
  - Files: `notebooks/*.ipynb`, `tests/unit/test_notebooks.py`

- [ ] Task 14: README + consolidation writeup (P1)
  - Acceptance: README rewritten for the consolidated project — how to run (`dvc repro`),
    the stage graph, the champion result vs the 0.7692 baseline, and an explicit note that
    `_claude` / `_codex` are superseded reference folders; the current README's "Important
    improvements over the original monolithic pipeline" section is updated so it no longer
    describes a pipeline that has since changed.
  - Files: `README.md`, `sprints/v3/TASKS.md` (completion notes)
