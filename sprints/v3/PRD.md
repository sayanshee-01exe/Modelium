# Sprint v3 — PRD: Consolidation + Full Relational Pipeline under DVC

## Overview
Make `smart_loan_approval_system` (this folder) the **single canonical** loan-risk project,
retiring the parallel `smart_loan_approval_system_claude` and `..._codex` folders as reference
material only. Two things merge here:

1. **The validated ML work from `_claude`** (v1+v2 complete: 7 notebooks, 104 tests, champion
   `LGBMClassifier` @ 0.7692 holdout ROC-AUC, SHAP explainability, MLflow runs) — currently
   trapped in notebooks, with `src/` holding only `utils.py` and `explain.py`.
2. **This folder's module layout and 7-table relational code** — the right shape, but a
   skeleton: no data, no tests, no git, empty venv, and several defects that prevent it from
   running at all (see *Known Defects*).

On top of that merge, adopt the infrastructure that actually distinguishes the reference repo
(`sayanshee-01exe/swiggy-delivery-time-prediction`): a **DVC pipeline** (`dvc.yaml` /
`params.yaml` / `dvc.lock`) where each stage is a `python src/.../module.py` entry point,
MLflow tracking with an explicit `register_model` stage, centralized logging, typed
exceptions, and a real test suite.

The substantive ML win this sprint is the **7-table relational merge** — the `_claude` v2
README explicitly disclosed leaving ~0.79-0.80 vs 0.77 ROC-AUC on the table by using only
`application_train.csv`. This sprint closes that gap with the champion as the baseline to beat.

## Goals
- One canonical repo, under git, that runs end-to-end via `dvc repro` with no notebook-order
  knowledge required.
- Every transformation lives in an importable `src/` module that is *also* a DVC stage entry
  point (`python src/data/data_cleaning.py`), mirroring the reference repo. Notebooks are
  ported from `_claude` for reporting only and call into `src/`.
- `params.yaml` centralizes every tunable (missing thresholds, aggregation specs, correlation
  cutoffs, model hyperparameters, split ratios) — nothing hardcoded in module bodies.
  `config/config.py` retains **paths only**.
- All seven Home Credit tables merged into one applicant-level feature table, with **explicit
  per-table aggregation specs** rather than blanket min/max/mean/sum over every numeric column.
- The `_claude` cleaning decisions are preserved as code, not rediscovered: `DAYS_EMPLOYED`
  365243 sentinel + flag, the `ORGANIZATION_TYPE=='XNA'` / employment-anomaly 100% overlap
  (left intentionally unchanged — real signal), split `OCCUPATION_TYPE` imputation
  (`Not_Employed` vs `Unknown`), `AMT_INCOME_TOTAL` p99.9 cap, `CNT_CHILDREN` cap at 10,
  `CODE_GENDER=='XNA'` fix, and `EXT_SOURCE_1` protection from the high-missing drop rule.
- Every model run logged to MLflow; the champion registered by a dedicated stage.
- Centralized logging to console + `logs/pipeline.log`, typed exceptions in
  `src/utils/exceptions.py`, input validation at the top of every stage — no silent failures.
- Test suite ported from `_claude` and extended to cover the new relational code.
- **Success metric**: holdout ROC-AUC > 0.7692 (the `_claude` champion), with the leaderboard
  and threshold reported from *disjoint* splits.

## User Stories
- As the project owner, I want one folder that is unambiguously "the project", so I stop
  maintaining three divergent copies and can point a reviewer at a single repo.
- As the project owner, I want `dvc repro` to rebuild everything from raw CSVs, so a changed
  parameter re-runs exactly the affected stages and nothing else.
- As the project owner, I want the cleaning insights `_claude` found through careful EDA
  (the XNA/employment overlap, the 117M income outlier) encoded in versioned module code, so
  they survive the consolidation instead of living in notebook prose.
- As the project owner, I want the training run to finish on a laptop, so I can iterate — the
  current portfolio and aggregation strategy make a single run effectively unrunnable.
- As a reviewer, I want the repo to read like the Swiggy reference (DVC + modular src/ +
  MLflow + tests), not like a folder of scripts.

## Known Defects (must be fixed, not carried forward)
| # | Location | Problem |
|---|---|---|
| D1 | `src/data/data_loader.py` | **Verified 2026-08-10 against the real CSVs.** The *column* explosion is NOT the issue — measured output is 293 aggregated columns → 415 merged → ~535 post-OHE → ~0.98 GB dense train matrix, all acceptable. The actual problem is the **raw load footprint: ~7 GB resident**, because `load_home_credit_tables` reads all seven tables up front and `optimize_memory` then `.copy()`s each one (transient peak higher still). `bureau_balance` (27.3M rows, 1.68 GB) and `previous_application` (1.67 GB) dominate. Fix = stream/aggregate each child table and release it before loading the next, not a narrower agg spec. |
| D9 | `src/data/data_preparation.py:14` + `src/data/data_cleaning.py:52` | **Silent data loss, verified.** `agg_dict[numeric[0]].append("count")` attaches the count stat to the first numeric column, which for 5 of 6 tables is an `SK_ID_*` column; `drop_low_information_columns` then drops every column containing `"SK_ID"`. Result: `bureau_SK_ID_BUREAU_count`, `prev_SK_ID_PREV_count`, `pos_SK_ID_PREV_count`, `cc_SK_ID_PREV_count`, `inst_SK_ID_PREV_count` are all destroyed — only `bb_MONTHS_BALANCE_count` survives. Prior-loan/application counts are among the strongest relational signals in this dataset. |
| D10 | `src/data/data_preparation.py:10` | **Silent data loss, verified.** `select_dtypes(include="number")` discards **22 categorical columns** across the child tables, including `bureau_balance.STATUS` (the DPD delinquency buckets — the core credit-history signal), `previous_application.NAME_CONTRACT_STATUS` (approved/refused) and `CODE_REJECT_REASON` (prior rejection history), and `bureau.CREDIT_ACTIVE`/`CREDIT_TYPE`. Needs count/ratio encodings per group key (e.g. share of DPD statuses, refusal rate). |
| D11 | `src/data/data_preparation.py:24-26` | `bb_*` columns are aggregated a second time when `bureau_merged` is rolled up, producing 20 columns like `bureau_bb_MONTHS_BALANCE_sum_mean`. Min-of-min is meaningful; mean-of-sum and sum-of-max are noise. |
| D2 | `src/models/train.py` | 8-model portfolio incl. `GradientBoostingClassifier(n_estimators=200)` and calibrated `LinearSVC` on that matrix — hours-to-days per run. |
| D3 | `scripts/train.py:56-57` | F1-optimal threshold tuned on the same holdout used for reported metrics → optimistically biased. Needs a disjoint validation split. |
| D4 | `src/monitoring/`, `src/explainability/`, `src/visualization/`, `basic_checks`, `IQRClipper`, `get_logger` | Dead code — defined, never called by any entry point. |
| D5 | `src/features/feature_engineering.py:20-21` | `PAYMENT_RATE` is a byte-identical duplicate of `ANNUITY_CREDIT_RATIO` — perfectly collinear. |
| D6 | `config/config.py` | `DATA_FILES` omits `application_test.csv`; no Kaggle-submission path exists. |
| D7 | `src/features/feature_engineering.py` | `EXT_SOURCE_MEAN` skips NaN (`.mean(axis=1)`) but `EXT_SOURCE_WEIGHTED` propagates it. **Measured impact: `EXT_SOURCE_1` is 56.4% missing and `EXT_SOURCE_3` 19.8%, so `EXT_SOURCE_WEIGHTED` is NaN for 64.4% of rows while its sibling is computed for 99.9%. `EXT_SOURCE_STD` is NaN for 12%.** Not cosmetic — `EXT_SOURCE_*` are the top-3 SHAP features in the `_claude` champion. Fix = renormalize weights over the non-null sources. |
| D8 | repo root | No git, no `data/`, no tests; venv contains only `pip`; `requirements.txt` missing `mlflow`, `dvc`, `catboost`, `optuna`, `pytest`. |

## Technical Architecture
- **Language/Env**: Python 3.13, venv at `.venv/` (the existing bare `venv/` is discarded).
- **Core**: pandas, numpy, scikit-learn, pyarrow, matplotlib, seaborn, missingno
- **Modeling**: xgboost, lightgbm, catboost, imbalanced-learn, optuna
- **Tracking**: mlflow (local file store `mlruns/`, `MLFLOW_ALLOW_FILE_STORE=true` — see the
  `_claude` `src/utils.py` note about MLflow ≥3.x putting the file backend behind an opt-in)
- **Orchestration**: dvc (local cache; no cloud remote this sprint)
- **Explainability**: shap (TreeExplainer — champion is a tree ensemble)
- **Testing**: pytest

```
smart_loan_approval_system/
├── config/config.py             # PATHS ONLY — tunables move to params.yaml
├── data/
│   ├── raw/                     # 7 tables + application_test.csv (dvc-tracked)
│   ├── interim/                 # application_clean, relational_features, train/test split
│   └── processed/               # train_trans.parquet, test_trans.parquet
├── logs/pipeline.log            # gitignored
├── mlruns/                      # gitignored
├── models/                      # preprocessor.joblib, champion_model.pkl, selected_features.json
├── notebooks/                   # ported from _claude; reporting only, calls into src/
├── reports/                     # metrics.json, model_leaderboard.csv, figures/
├── scripts/train.py             # one-shot local runner (kept; thin wrapper over src/)
├── src/
│   ├── data/{data_loader,data_cleaning,data_preparation}.py
│   ├── features/{feature_engineering,data_preprocessing,feature_selection}.py
│   ├── models/{train,evaluation,threshold,serialization,register_model}.py
│   ├── explainability/shap_explainer.py
│   ├── monitoring/{drift,fairness}.py
│   ├── visualization/plots.py
│   └── utils/{logger,exceptions}.py
├── tests/unit/
├── params.yaml
├── dvc.yaml  ·  dvc.lock  ·  .dvcignore  ·  .gitignore
└── requirements.txt
```

**DVC stage graph** (each stage = a `__main__` entry point on an existing module, Swiggy-style):

```
data_cleaning        src/data/data_cleaning.py          → data/interim/application_clean.parquet
data_preparation     src/data/data_preparation.py       → data/interim/relational_features.parquet
feature_engineering  src/features/feature_engineering.py→ data/interim/{train,val,test}.parquet
data_preprocessing   src/features/data_preprocessing.py → data/processed/{train,val,test}_trans.parquet
                                                          models/preprocessor.joblib
feature_selection    src/features/feature_selection.py  → models/selected_features.json
train                src/models/train.py                → models/*.joblib  (+ MLflow runs)
evaluation           src/models/evaluation.py           → reports/metrics.json, run_information.json
register_model       src/models/register_model.py       → MLflow registry entry
```

Note the **three-way split** (train/val/test): validation drives threshold selection and model
choice, test is touched once for the final report — this is the fix for D3.

## Out of Scope
- Docker, GitHub Actions CI/CD, AWS CodeDeploy (`appspec.yml`), `Makefile`/`tox.ini`/Sphinx — v4
- FastAPI/Flask serving layer and `frontend/` dashboard — v4
- DVC cloud remote (S3/GCS) — local cache only this sprint
- Fraud/anomaly detection (Isolation Forest, autoencoders)
- Migrating git history from `_claude` — this folder starts a fresh repo; `_claude` stays on
  disk read-only as provenance
- Deleting the `_claude` / `_codex` folders (decide after v3 lands and is verified)

## Dependencies
- Home Credit CSVs already on disk at
  `../smart_loan_approval_system_claude/home-credit-default-risk/` (all 7 tables + test).
- `_claude` artifacts readable for porting/baseline comparison: `models/champion_model.pkl`,
  `models/selected_features.json`, `notebooks/1..7`, `tests/unit/`.
