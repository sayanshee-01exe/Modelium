# Sprint v3 — Walkthrough

> **Status at time of writing: 1 of 14 tasks complete (Task 1).**
> This sprint is in progress. This document describes the scaffold that Task 1 built and
> the inherited-but-not-yet-reworked `src/` modules it sits around. **No part of the ML
> pipeline runs end-to-end yet.** Sections marked *(scaffold only)* have no working
> implementation behind them.

## Summary

Sprint v3 consolidates three divergent copies of this project into one canonical repository
and rebuilds it around the infrastructure pattern of the reference repo
(`sayanshee-01exe/swiggy-delivery-time-prediction`): a DVC pipeline, `params.yaml`,
MLflow tracking, centralized logging, and a real test suite.

Task 1 delivered the foundation only: a git + DVC repository, the directory layout the future
stage graph writes into, symlinked raw data, a working Python 3.13 environment, and 69 tests
that hold all of it in place. It also fixed defect **D6** (the holdout applicant table was
absent from the config registry, leaving no scoring path).

The 13 `src/` modules in this repo were **inherited unchanged** from a ChatGPT-generated
refactor and are documented below as-is, with the defect each one carries and the task that
will rewrite it. They have never been executed against the real dataset.

## Background: why three folders became one

Three sibling projects existed in `Project Ideas Chatgpt/`:

| Folder | What it had | What it lacked |
|---|---|---|
| `smart_loan_approval_system` **(this one)** | Clean `src/` module layout matching the reference repo; code for all 7 Home Credit tables | No data, no git, no tests, empty venv, nothing runnable |
| `smart_loan_approval_system_claude` | v1+v2 complete: 7 notebooks, 104 tests, champion `LGBMClassifier` @ **0.7692** holdout ROC-AUC, SHAP module, MLflow runs | Pipeline logic trapped in notebooks (`src/` held only `utils.py` + `explain.py`); only 1 of 7 tables; no DVC |
| `smart_loan_approval_system_codex` | A third variant | — |

This folder was chosen as the consolidation target: it already has the right *shape*, and the
7-table relational merge is the substantive win available (the `_claude` v2 README explicitly
disclosed leaving ~0.79–0.80 vs 0.77 ROC-AUC on the table by using `application_train.csv`
alone). **0.7692 is the number v3 must beat.**

## Architecture Overview

The target stage graph, with what actually exists today:

```
                        params.yaml  (Task 3 — NOT YET WRITTEN)
                             │  every tunable, one file
                             ▼
   data/raw/*.csv ──▶ ┌──────────────────┐
   (8 symlinks,       │ data_cleaning    │  Task 4   src/data/data_cleaning.py
    2.7 GB, EXISTS)   └────────┬─────────┘
                               ▼ data/interim/application_clean.parquet
                      ┌──────────────────┐
                      │ data_preparation │  Task 5   src/data/data_preparation.py
                      └────────┬─────────┘           (7 tables → 1 applicant row)
                               ▼ data/interim/relational_features.parquet
                      ┌──────────────────────┐
                      │ feature_engineering  │  Task 6   + train/val/test split
                      └────────┬─────────────┘
                               ▼ data/interim/{train,val,test}.parquet
                      ┌──────────────────────┐
                      │ data_preprocessing   │  Task 7  ─▶ models/preprocessor.joblib
                      └────────┬─────────────┘
                               ▼ data/processed/{train,val,test}_trans.parquet
                      ┌──────────────────────┐
                      │ feature_selection    │  Task 7  ─▶ models/selected_features.json
                      └────────┬─────────────┘
                               ▼
                      ┌──────────────────────┐        ┌─────────────┐
                      │ train                │───────▶│   MLflow    │  Task 8
                      │ LogReg/RF/XGB/LGBM/  │        │  mlruns/    │
                      │ CatBoost             │        └──────┬──────┘
                      └────────┬─────────────┘               │
                               ▼ models/*.joblib             │
                      ┌──────────────────────┐               │
                      │ evaluation           │  Task 9       │
                      │ select on VAL,       │──▶ reports/metrics.json
                      │ score TEST once      │    reports/model_leaderboard.csv
                      └────────┬─────────────┘               │
                               ▼ run_information.json        ▼
                      ┌──────────────────────┐        ┌─────────────┐
                      │ register_model       │───────▶│  Registry   │  Task 11
                      └──────────────────────┘        └─────────────┘

  EXISTS TODAY: the repo, .venv, data/raw symlinks, directory layout, 69 tests
  NOT YET:      dvc.yaml, params.yaml, every stage entry point, all 8 arrows above
```

Note the **three-way split**. The inherited code splits train/test only and tunes the decision
threshold on the same holdout it reports from (defect D3). Validation drives model choice and
thresholding; test is scored exactly once.

## Files Created/Modified

### Created this sprint (Task 1)

#### `.gitignore`
**Purpose**: Keep 2.7 GB of raw data, model binaries, and the virtualenv out of git while
keeping the *directories* that DVC stages write into.

**How it works**: The subtle part is the interaction between directory exclusion and
`.gitkeep`. Excluding a whole directory (`data/raw/`) makes it impossible to re-include
anything inside it — git never descends into an excluded directory, so a negation like
`!data/raw/.gitkeep` silently does nothing. Directories that must survive a clone are
therefore excluded by *contents*, not as a whole:

```gitignore
data/interim/*
!data/interim/.gitkeep
data/processed/*
!data/processed/.gitkeep
```

`logs/` and `data/raw/` are excluded outright, since neither needs to survive a clone —
the logger will `mkdir` its own directory, and Task 1 recreates the data symlinks.
`references/*.csv` is excluded for the same reason as `data/raw/`: it is a symlink into the
Kaggle extract and would be dangling on a fresh clone.

#### `requirements.txt`
**Purpose**: Declare the dependency set, plus a written record of accepted security advisories.

**How it works**: Grouped by role (Core / Modelling / Tracking / Explainability /
Visualisation / Serialisation / Testing / Security tooling). Task 1 added `catboost`,
`optuna`, `mlflow`, `dvc`, `seaborn`, `missingno`, `pyarrow`, `pytest`, and `pip-audit` to
the nine packages already present.

The file also carries a comment block recording two transitive advisories that were
**deliberately not fixed**, so the next `/dev` run re-evaluates rather than rediscovers them:

```
#   cryptography 49.x  PYSEC-2026-3552 — PKCS#7/S-MIME decryption oracle. Fixed in 50.0.0,
#     but mlflow pins cryptography<50, so requiring the fix makes the install unresolvable.
#   diskcache 5.6.3    PYSEC-2026-2447 — pickle deserialization via the cache dir. 5.6.3 is
#     the newest release; no upstream fix exists.
```

#### `tests/unit/test_project_structure.py`
**Purpose**: Pin every part of Task 1's acceptance criteria as an executable assertion.
**69 tests** (many parametrized), written before the implementation and confirmed failing first.

**Key tests**:
- `test_raw_csv_present_and_resolves()` — 8 params, one per table
- `test_gitignore_covers_pattern()` — 8 params
- `test_requirements_declares_package()` / `test_package_importable()` — 17 + 15 params
- `test_raw_data_not_staged_in_git()` — shells out to `git check-ignore`
- `test_config_includes_application_test()` — the D6 regression guard

**How it works**: Two tests are worth explaining because a naive version of each would pass
while the repo was actually broken.

The symlink test resolves the link rather than checking the link itself. `Path.exists()` on a
dangling symlink follows the link and returns `False`, but `is_symlink()` returns `True` and a
`Path(...).exists()` check written against the wrong object can pass on a link pointing at
nothing. Asserting on the *resolved* target and its size closes that:

```python
resolved = path.resolve(strict=False)
assert resolved.is_file(), f"data/raw/{filename} does not resolve to a file"
assert resolved.stat().st_size > 0, f"data/raw/{filename} resolves to an empty file"
```

The ignore test asserts the *effect* rather than the file's text. A `.gitignore` can contain
the right-looking pattern and still fail to match — the directory-exclusion trap above is
exactly that failure. Asking git directly is the only assertion that means anything:

```python
result = subprocess.run(["git", "check-ignore", "-q", "data/raw/application_train.csv"],
                        cwd=PROJECT_ROOT, capture_output=True)
assert result.returncode == 0, "data/raw CSVs are NOT git-ignored"
```

#### `.dvcignore`, `.dvc/config`, `.dvc/.gitignore`
**Purpose**: DVC repository metadata, generated by `dvc init`. `.dvc/config` is currently
empty — no remote is configured, which is intentional (the PRD scopes cloud remotes out of
this sprint; the local cache is enough).

#### `sprints/v3/PRD.md` and `sprints/v3/TASKS.md`
**Purpose**: The sprint plan. `PRD.md` carries the **Known Defects table (D1–D11)**, which is
the most load-bearing part of this sprint's documentation — it records defects found by
validating the inherited code against the real CSVs, so they are tracked items rather than
things to rediscover mid-implementation.

#### `.gitkeep` files
Five zero-byte files in `data/interim/`, `data/processed/`, `reports/figures/`, `models/`, and
`references/`, so those directories survive a clone.

### Modified this sprint

#### `config/config.py`
**Purpose**: Path constants and the raw-table registry.

**How it works**: Task 1's change was defect **D6** — `DATA_FILES` listed the seven training
tables but omitted `application_test.csv`, so the 48,744 holdout applicants were unreachable
and the project had no scoring path at all:

```python
DATA_FILES = {
    "application_train": "application_train.csv",
    # D6: the holdout applicants were previously absent from the registry, leaving no
    # scoring/submission path. The training pipeline splits application_train and does
    # not consume this table.
    "application_test": "application_test.csv",
    ...
}
```

**Known problem with this file**: it mixes paths with tunables, and four of its constants are
dead — `HIGH_MISSING_THRESHOLD`, `LOW_CARDINALITY_THRESHOLD`, `DOWNSAMPLE_RATIO`, and
`REPORT_DIR` have **zero references** anywhere outside this file. Task 3 moves the tunables to
`params.yaml` and reduces this file to paths only.

### Inherited unchanged — awaiting later tasks

These 13 modules (530 lines total) came with the folder and have **not been modified or
executed against real data**. Each is listed with its defect and owning task.

#### `src/data/data_loader.py` (20 lines) — Task 5
**Purpose**: Read the Home Credit CSVs into a dict of DataFrames.
`load_home_credit_tables(data_dir, file_map)` collects missing paths and raises a single
`FileNotFoundError` listing all of them, rather than failing on the first.

**Defect D1**: it holds every table resident simultaneously. Measured footprint: **~7 GB**,
dominated by `bureau_balance` (27.3M rows, 1.68 GB) and `previous_application` (1.67 GB), and
`optimize_memory` then `.copy()`s each one on top of that. The fix is load → aggregate →
release per table.

#### `src/data/data_cleaning.py` (57 lines) — Task 4
**Purpose**: Memory downcasting, a data-quality report, and low-information column pruning.
- `optimize_memory()` — int64→int8/16/32 by observed range, float64→float32
- `basic_checks()` — returns a report dict instead of printing (never called by anything)
- `drop_low_information_columns()` — drops constant columns and ID-like columns

**Defect D9 (with `data_preparation.py`)**: the ID-drop rule is `"SK_ID" in c`, which
destroys the relational count features. See below.

**Missing entirely**: none of the cleaning decisions the `_claude` project established through
EDA are here — the `DAYS_EMPLOYED == 365243` sentinel, the `ORGANIZATION_TYPE == 'XNA'`
overlap, the income and children caps. Task 4 ports them.

#### `src/data/data_preparation.py` (41 lines) — Task 5
**Purpose**: Collapse the 7 relational tables into one applicant-level row.
`build_relational_feature_table()` handles the two-level join correctly: `bureau_balance` keys
on `SK_ID_BUREAU` only, so it is aggregated to that key, merged into `bureau`, and the result
re-aggregated to `SK_ID_CURR`. **That topology was verified correct against the real files.**

**Defect D9 — silent loss of five count features.** `aggregate_numeric_table` attaches the
`count` statistic to whichever numeric column happens to be first:

```python
agg_dict = {c: ["min", "max", "mean", "sum"] for c in numeric}
agg_dict[numeric[0]].append("count")
```

For 5 of the 6 child tables, `numeric[0]` is an `SK_ID_*` column, producing names like
`prev_SK_ID_PREV_count` — which `drop_low_information_columns` then deletes because they
contain `"SK_ID"`. Verified outcome:

```
survives  bb_MONTHS_BALANCE_count
DROPPED   bureau_SK_ID_BUREAU_count      <- number of prior bureau records
DROPPED   prev_SK_ID_PREV_count          <- number of previous applications
DROPPED   pos_SK_ID_PREV_count
DROPPED   cc_SK_ID_PREV_count
DROPPED   inst_SK_ID_PREV_count
```

"How many prior loans does this applicant have" never reaches the model.

**Defect D10 — 22 categorical columns discarded.** `select_dtypes(include="number")` silently
drops every non-numeric child column, including `bureau_balance.STATUS` (the days-past-due
buckets — the core credit-history signal), `previous_application.NAME_CONTRACT_STATUS`
(approved/refused) and `CODE_REJECT_REASON`, and `bureau.CREDIT_ACTIVE`/`CREDIT_TYPE`.

**Defect D11**: the `bb_*` columns get aggregated a second time, yielding 20 columns like
`bureau_bb_MONTHS_BALANCE_sum_mean`. Min-of-min is meaningful; mean-of-sum is noise.

#### `src/features/feature_engineering.py` (58 lines) — Task 6
**Purpose**: Credit-risk domain features — debt burden, credit/income, payment rate, age and
tenure in years, `EXT_SOURCE_*` aggregates, per-person income. Every block is guarded by a
`has(*cols)` check, so it degrades gracefully on a partial frame. All 14 column names it
references were verified present in `application_train`.

**Defect D5**: `PAYMENT_RATE` is assigned the value of `ANNUITY_CREDIT_RATIO` — a
byte-identical, perfectly collinear duplicate column.

**Defect D7**: `EXT_SOURCE_MEAN` skips nulls but `EXT_SOURCE_WEIGHTED` propagates them:

```python
out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)            # skipna -> 99.9% of rows
out["EXT_SOURCE_WEIGHTED"] = sum(out[c] * weights[c] for c in ext_cols) / denom
```

Since `EXT_SOURCE_1` is 56.4% missing and `EXT_SOURCE_3` 19.8%, the weighted version is
**NaN for 64.4% of rows** while its sibling is populated for 99.9%. `EXT_SOURCE_STD` is NaN
for 12%. These are the champion model's top-3 features by SHAP, so this is not cosmetic.

#### `src/features/data_preprocessing.py` (55 lines) — Task 7
**Purpose**: A leak-free `ColumnTransformer` — median-impute + scale numerics, mode-impute +
one-hot categoricals with `handle_unknown="ignore"`.

Also defines `IQRClipper`, a `BaseEstimator`/`TransformerMixin` that learns IQR bounds on
train and clips later data to them — the right shape for avoiding leakage, but **it is never
added to the pipeline** (defect D4).

#### `src/models/train.py` (53 lines) — Task 8
**Purpose**: An 8-model candidate portfolio and a fit loop.

**Defect D2**: the portfolio includes `GradientBoostingClassifier(n_estimators=200)` and a
`CalibratedClassifierCV(LinearSVC(...), cv=3)`. On 246k rows × ~535 features those take
hours to days on a laptop. Task 8 cuts to LogReg / RF / XGBoost / LightGBM / CatBoost.

#### `src/models/evaluation.py` (34 lines) — Task 9
**Purpose**: Metrics per model. `get_probability_scores()` usefully falls back to a min-max
normalized `decision_function()` for classifiers without `predict_proba`. Reports Accuracy,
Precision, Recall, F1, ROC-AUC, PR-AUC.

#### `src/models/threshold.py` (16 lines) — Task 9
**Purpose**: `find_f1_optimal_threshold()` sweeps the precision-recall curve for the
F1-maximizing cut point. The function is correct in isolation; **defect D3 is in the caller** —
`scripts/train.py` feeds it the same holdout it reports final metrics from, which biases those
metrics optimistically.

#### `src/models/serialization.py` (15 lines) — Task 7/9
**Purpose**: `save_production_bundle()` writes model, preprocessor, and a JSON metadata
sidecar. Correctly saves the whole fitted **preprocessor**, not just a scaler, so inference
can reproduce training-time transforms.

#### `src/explainability/shap_explainer.py` (20 lines) — Task 12 *(orphaned)*
`global_shap_importance()` — samples rows, builds a `TreeExplainer`, returns a ranked
mean-|SHAP| table. Handles the older per-class-list SHAP return shape. **Zero import sites.**

#### `src/monitoring/drift.py` (28 lines) — Task 12 *(orphaned)*
`compute_psi()` — Population Stability Index on percentile bins, with `psi_status()`
thresholding to STABLE / MONITOR / RETRAIN at 0.10 / 0.20. Guards empty input and degenerate
bins. **Zero import sites.**

#### `src/monitoring/fairness.py` (29 lines) — Task 12 *(orphaned)*
`group_fairness_report()` — per-group TPR, FPR, precision, actual vs predicted default rate,
and a disparate-impact ratio, skipping groups under `min_group_size=30`. **Zero import sites.**

#### `src/visualization/plots.py` (20 lines) — Task 9 *(orphaned)*
`plot_model_evaluation()` — confusion matrix, ROC, and PR curves as three separate figures.
**Zero import sites.**

#### `src/utils/logger.py` (9 lines) — Task 2 *(orphaned)*
A bare `logging.basicConfig` wrapper. No file handler, no rotation. **Zero import sites** —
nothing in this project logs anything today. Task 2 replaces it.

#### `scripts/train.py` (75 lines) — Task 10
**Purpose**: The only end-to-end entry point: load → optimize → merge → engineer → drop →
split → fit preprocessor → train 8 models → rank by PR-AUC → tune threshold → save bundle.

Reads as a clean orchestration script and correctly fits the preprocessor on train only. It
**cannot currently run** — D1 (7 GB load) and D2 (intractable models) both block it. Task 10
reduces it to a thin wrapper once each module has a DVC entry point.

#### `README.md` — Task 14
Describes the original ChatGPT refactor and its improvements over the monolithic
`pipeline.py`. Now partly stale: it documents a `lendguard_modular/` tree that no longer
matches this repo. Task 14 rewrites it.

## Data Flow

**Today**, only one flow is real — the test suite:

```
pytest → tests/unit/test_project_structure.py
       → filesystem assertions (dirs, symlink resolution, .gitkeep)
       → subprocess `git check-ignore`  (are the 2.7 GB really unstageable?)
       → import config.config           (does DATA_FILES match what is on disk?)
       → importlib checks on 15 packages
```

**Intended**, once Tasks 2–11 land:

1. `dvc repro` reads `params.yaml`, hashes `data/raw/*.csv` (symlinks; DVC follows them, and
   they are stage *deps*, not `dvc add`-tracked outputs, so no second copy enters the cache).
2. `data_cleaning` validates `application_train`, applies the ported EDA fixes, writes
   `data/interim/application_clean.parquet`.
3. `data_preparation` loads each child table in turn, aggregates it to `SK_ID_CURR` (with
   record counts and encoded categoricals — D9/D10), releases it, and joins onto the applicant
   frame.
4. `feature_engineering` adds domain ratios and splits stratified train/val/test.
5. `data_preprocessing` fits the `ColumnTransformer` on **train only**, persists it, transforms
   all three splits.
6. `feature_selection` prunes by variance / correlation / importance → `selected_features.json`.
7. `train` fits 5 algorithms, logging params + metrics + artifacts to `mlruns/`.
8. `evaluation` picks the champion and threshold **on validation**, scores **test once**, emits
   `reports/metrics.json` and `run_information.json`.
9. `register_model` promotes the champion run into the MLflow registry.

Every stage logs to `logs/pipeline.log` and raises a typed exception on bad input rather than
letting a NaN propagate downstream.

## Test Coverage

- **Unit: 69 tests**, all passing in ~2s warm. Layout (9), `.gitkeep` (1), raw-CSV symlink
  resolution (8), symlink-not-copy (1), column dictionary (1), `.gitignore` patterns (8),
  `.dvcignore` (1), `git check-ignore` effect (1), requirements declared (17), packages
  importable (15), dvc CLI (1), venv replaced (1), git init (1), dvc init (1), D6 regression
  (1), `DATA_FILES` resolve on disk (1), `DATA_DIR` correctness (1).
- **Integration: 0** — nothing to integrate yet; arrives with Task 10 (`dvc repro`).
- **E2E: 0** — no UI or API in this sprint (scoped to v4).

Written before the implementation and confirmed RED first: **33 failed / 35 passed / 1
skipped** pre-implementation.

**Two tests were wrong and were fixed rather than worked around.**
`test_tracked_output_dirs_have_gitkeep` originally asserted `logs/.gitkeep`, which can never be
tracked because `logs/` is wholly excluded. `test_dvc_cli_available` used a bare
`shutil.which("dvc")`, which reports a false failure under `.venv/bin/python -m pytest` because
`.venv/bin` is not on `PATH`; it now checks for the console script next to `sys.executable`
first.

**Not covered**: every `src/` module. There is no test for any transformation, model, or
metric — that is the point of Tasks 4–12, each of which is specified test-first.

## Security Measures

- **semgrep** `--config auto`: 0 findings across 290 rules / 24 files.
- **pip-audit**: 2 transitive advisories, both accepted with written rationale in
  `requirements.txt` rather than silently ignored:
  - `cryptography` 49.x / **PYSEC-2026-3552** — a Bleichenbacher oracle in `pkcs7_decrypt_*`.
    Fixed in 50.0.0, but mlflow 3.15.1 (the newest release) pins `cryptography<50`, so
    requiring the fix makes `pip install -r requirements.txt` unresolvable. The affected
    S/MIME APIs are never called here; the package arrives only via mlflow, google-auth, and
    asyncssh. Revisit when mlflow relaxes the pin.
  - `diskcache` 5.6.3 / **PYSEC-2026-2447** — pickle deserialization. No upstream fix exists;
    5.6.3 is the newest release. Requires write access to the local DVC cache, which already
    implies local code execution.
- **`pip check`** reports no broken requirements; a `--dry-run` install resolves cleanly, so
  the committed `requirements.txt` reproduces this environment.
- **No secrets in the repo.** No credentials, tokens, or cloud config; `.dvc/config` is empty
  and no remote is configured.
- **2.7 GB of personal financial data cannot be committed** — enforced by an executable
  `git check-ignore` assertion, not merely by a `.gitignore` line that looks right.

## Known Limitations

Being blunt, since 13 of 14 tasks remain:

1. **Nothing in the ML pipeline runs.** No model has been trained in this repo. The only
   executable artifact is the test suite.
2. **No `dvc.yaml` and no `params.yaml`.** DVC is initialized but orchestrates nothing, and
   every tunable is still a literal in module bodies or a constant in `config/config.py`.
3. **No `src/` module has an entry point.** They are pure-function modules; DVC stages need
   `if __name__ == "__main__":` blocks that do not exist.
4. **Five modules are orphaned** — `shap_explainer`, `drift`, `fairness`, `plots`, `logger` all
   have zero import sites. ~40% of the `src/` code is unreachable.
5. **Nothing logs.** `get_logger` is never called; there is no `logs/pipeline.log`, no file
   handler, and no typed exceptions.
6. **Four dead config constants** — `HIGH_MISSING_THRESHOLD`, `LOW_CARDINALITY_THRESHOLD`,
   `DOWNSAMPLE_RATIO`, `REPORT_DIR` are referenced nowhere.
7. **Zero test coverage of `src/`.** 69 tests, none touching a transformation or a model.
8. **The validated work in `_claude` has not been ported yet** — 7 notebooks, 104 tests, and
   the cleaning decisions still live only in the superseded folder. Until Task 4 lands, those
   findings exist as prose, not code, and the old folders must not be deleted.
9. **Raw data is symlinked outside the repo.** Anyone cloning this gets dangling links and must
   re-run Task 1's symlink step; there is no data acquisition script.
10. **`application_test` is now loaded unnecessarily.** Adding it to `DATA_FILES` (the D6 fix)
    means `scripts/train.py` reads a table the training path never consumes. Harmless today
    since that script cannot run, resolved by Task 5.
11. **The `0.7692` baseline is unverified in this repo.** It is inherited from `_claude`'s
    reported holdout ROC-AUC and has not been reproduced here.
12. **A stray 0-byte `.gitigone`** (typo for `.gitignore`) sits untracked in the project root.
    Left in place rather than deleted, as it was created outside this workflow.

## What's Next

Immediate order is fixed by dependency, not preference:

- **Task 2 — logging + typed exceptions.** Everything downstream is specified to log stage
  timings and raise typed errors, so this is the cheapest way to stop building on silence.
- **Task 3 — `params.yaml`.** Must precede Tasks 4–8, since each stage's acceptance criteria
  read parameters from it. Also clears the four dead constants.
- **Task 4 — port the `_claude` cleaning decisions.** The highest-risk knowledge transfer in
  the sprint: those findings are currently prose in a superseded folder. Task 4 encodes them,
  including the assert proving the `ORGANIZATION_TYPE == 'XNA'` population is exactly the
  employment-anomaly population, so a later refactor cannot silently "fix" those 55,374 rows.
- **Task 5 — the relational rewrite.** Where the sprint is won or lost: D9, D10, D11, and D1
  all land here, and D9+D10 are the two defects most likely to explain a disappointing ROC-AUC.

Beyond that: Tasks 6–9 (features, preprocessing, training, evaluation), Task 10 (`dvc.yaml` —
the first point at which `dvc repro` means anything), Tasks 11–12 (registry, activating the
orphaned modules), Tasks 13–14 (notebooks, README).

**Deferred to v4** per the PRD: Docker, GitHub Actions CI/CD, AWS CodeDeploy, a FastAPI/Flask
serving layer, a frontend dashboard, and a DVC cloud remote. Deleting the `_claude` and
`_codex` folders stays off the table until v3 beats 0.7692 and is verified.
