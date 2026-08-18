# Modelium

End-to-end ML engineering system for credit default risk prediction using the Home Credit dataset.

Modelium takes the eight raw Home Credit relational tables and produces a single deployable
artifact: a fitted scikit-learn `Pipeline` that carries its own preprocessing, plus a decision
threshold frozen on validation data. The whole path is a reproducible DVC pipeline.

The engineering focus is on the parts that quietly go wrong in credit modelling: **data leakage**,
**metric choice under class imbalance** (~8% default rate), and **serving an artifact that was
never approved**. Each is addressed by a specific mechanism described below.

---

## Pipeline

```text
Raw Home Credit Data
        ↓
DVC Validate            required files, table schemas, join keys
        ↓
DVC Prepare
        ↓
Relational Aggregation  6 child tables → one row per applicant
        ↓
Feature Engineering     credit-risk domain ratios
        ↓
Train / Validation / Test
        ↓
Leak-free CV Pipelines  preprocessing refitted inside every fold
        ↓
Hyperparameter Tuning   RandomizedSearchCV, StratifiedKFold
        ↓
Champion Selection      validation only
        ↓
Threshold Optimization  validation only
        ↓
Final Test Evaluation   test read exactly once
        ↓
Champion Pipeline       preprocessing + model, one artifact
        ↓
        ├──→ Model Registry   versioned, aliased by promotion status
        │            ↓
        │       SHAP Explanations   global importance + per-applicant
        ↓
Batch Inference
```

DVC DAG:

```text
validate → prepare → train ─┬→ register → explain
                            └→ predict
```

Neither `register` nor `explain` sits upstream of `predict`. Batch scoring loads the local
champion artifact, so a registry outage cannot stop the pipeline producing predictions,
and an explanation is never a prerequisite for a score.

---

## ML methodology

**Three-way split — 70% training / 15% validation / 15% final test.** Stratified, so the ~8%
positive rate is preserved in each. The validation split exists so model comparison and threshold
selection have somewhere to happen that is not the test set; the test split is read exactly once,
after the champion and threshold are frozen.

**Preprocessing is fitted only on training data, and inside CV folds.** The estimator handed to
`RandomizedSearchCV` is a `Pipeline([("preprocessor", ...), ("model", ...)])`, so every fold fits
its own imputer, IQR bounds, scaler and encoder on that fold's *training* portion alone. Fitting
the preprocessor once on all of `X_train` and searching over the transformed matrix would let each
fold's held-out rows shape the statistics used to transform them, inflating CV scores.

**Average Precision is the primary metric.** At an ~8% positive rate, accuracy is uninformative and
ROC-AUC is optimistic. `average_precision_score` is a step-wise sum that does not interpolate
between operating points, unlike a trapezoidal PR-AUC. The same estimator is used for
`RandomizedSearchCV(scoring="average_precision")`, the validation leaderboard, champion selection
and the quality gates — so the model that wins the search is the model that wins selection.
ROC-AUC is reported as a secondary metric.

**Candidates.** Logistic Regression (baseline, deliberately untuned) versus Random Forest, XGBoost
and LightGBM (tuned). The baseline uses the same full-pipeline structure, so the comparison
measures two algorithms rather than two data treatments. A booster that cannot beat a linear model
is not worth its complexity.

**Champion selection and threshold tuning both use validation only.** Pre-threshold quality gates
are restricted to threshold-*independent* metrics (Average Precision, ROC-AUC). Recall, Precision
and F1 are only meaningful once a cut-off exists, so they are checked *after* the threshold is
tuned — gating on Recall beforehand would judge a model by the arbitrary 0.5 default it will never
ship with.

Class imbalance is handled by estimator weighting (`class_weight="balanced"`,
`scale_pos_weight`) rather than resampling, which would have to happen inside each CV fold to stay
leak-free.

---

## Data

Modelium uses all eight Home Credit tables:

| Table | Role |
| --- | --- |
| `application_train.csv` | Training applicants, carries `TARGET` |
| `application_test.csv` | Scoring applicants, no `TARGET` |
| `bureau.csv` | Prior credits at other institutions |
| `bureau_balance.csv` | Monthly balances for those credits |
| `previous_application.csv` | Prior Home Credit applications |
| `POS_CASH_balance.csv` | Point-of-sale / cash loan balances |
| `credit_card_balance.csv` | Credit card balances |
| `installments_payments.csv` | Repayment history |

Two applicant tables and six child tables. The child tables are one-to-many, so each is aggregated
(min/max/mean/sum/count) to one row per `SK_ID_CURR` and left-joined onto the applicant table;
`bureau_balance` is rolled up through `bureau` first. Domain features are then derived — debt
burden, credit/income ratios, payment rate, age and employment tenure, and `EXT_SOURCE` aggregates.

`application_train` and `application_test` go through the **same** aggregation and feature functions,
with only the applicant table switched. A second implementation for inference is how
training/serving skew starts.

The raw dataset is ~2.5 GB and is **not** tracked in Git. Place the CSVs (or symlinks) in
`data/raw/`; that directory is gitignored, as are all generated artifacts.

---

## Data validation

Two distinct contracts, because training and scoring genuinely need different things:

| Check | Training | Inference |
| --- | --- | --- |
| Required files present | ✅ | ✅ |
| Table schemas / required columns | ✅ | ✅ |
| `TARGET` present, complete, binary, two classes | ✅ | — not present by design |
| `SK_ID_CURR` non-null | ✅ | ✅ |
| `SK_ID_CURR` unique in the applicant table | ✅ | ✅ |
| Join keys present and non-null | ✅ | ✅ |

Validation runs before anything transforms or aggregates, and reports every failure in one raise
rather than one per run. A duplicate applicant matters more than it looks: it fans the joins out
and silently multiplies predictions.

---

## Preprocessing

| Numeric | Categorical |
| --- | --- |
| Median imputation | Most-frequent imputation |
| IQR clipping (Tukey fences from the training fold) | `OneHotEncoder(handle_unknown="ignore")` |
| Standard scaling | |

`handle_unknown="ignore"` means a category unseen during training encodes as all-zeros instead of
raising, so inference cannot crash on a new value.

At inference, a raw frame is aligned to the exact schema the champion was fitted on: **missing
columns fail loudly**, extras are dropped, and order is corrected. Missing columns are not imputed
— a column the model was trained on but which inference cannot supply means the two sides disagree
about what the features are, and filling it with a training median would hide that behind a
plausible-looking prediction.

---

## Batch inference

```text
application_test  +  relational historical tables
        ↓
same aggregation + feature engineering as training
        ↓
saved champion pipeline (preprocessing travels with the model)
        ↓
frozen threshold from deployment metadata
        ↓
artifacts/predictions/test_predictions.csv
```

Output columns: `SK_ID_CURR`, `DEFAULT_PROBABILITY`, `PREDICTED_CLASS`.

Inference is **transform-only** — there is no `fit`, no `fit_transform`, and no code path that
reaches one. The decision threshold comes from metadata, never sklearn's hard-coded 0.5.

**Production inference blocks unpromoted models by default.** A champion that failed its quality
gates is refused with the reason, and scoring it requires an explicit `--allow-unpromoted` flag
intended for local debugging. The DVC `predict` stage does not pass that flag, so a rejected model
stops the pipeline rather than reaching a decision. Loading also validates the metadata contract,
the threshold range, and that the classifier exposes binary `{0, 1}` classes.

There is no real-time API yet — batch scoring only.

---

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**The virtual environment must be active before you run DVC.** DVC stage commands are
plain `python scripts/....py`, and `python` resolves against `PATH`. Without `.venv` on
it you get whatever interpreter the shell offers — typically a system or Anaconda Python
that has pandas but no `mlflow`, `xgboost` or `lightgbm`.

The loud half of that failure is harmless: `train` dies immediately with
`No module named 'mlflow'`. The quiet half is not — `validate` and `prepare` run to
completion under the wrong interpreter and write real outputs, so a pipeline can end up
reproduced half under one Python and half under another with nothing to show for it.

Use the Makefile, which puts `.venv/bin` on `PATH` for you and checks the interpreter
first:

```bash
make venv-check               # confirm .venv exists and has the training deps
make repro                    # run every out-of-date stage
make dag                      # inspect the pipeline
make test                     # unit suite, no data required
```

Individual stages:

```bash
make train
make register
make predict
```

If you would rather not use `make`, either activate the environment first
(`source .venv/bin/activate`) or set `PATH` explicitly — never point DVC at an absolute
interpreter path, which would only be correct on one machine:

```bash
PATH="$PWD/.venv/bin:$PATH" dvc repro
```

One DVC flag is worth knowing before you reach for it. `dvc repro <stage> --force`
propagates **upstream** — it forces every stage the target depends on, not just the one
named — and DVC deletes a stage's outputs *before* re-running it. So a `--force` on
`register` will begin re-running `train`, and interrupting it leaves the champion
deleted. To force a single stage, confine it with `-s`:

```bash
dvc repro --single-item --force register    # or: make register-force
```

`dvc repro` skips stages whose dependencies are unchanged, and `dvc status` reports what is stale.

| File | Role |
| --- | --- |
| `params.yaml` | Experiment configuration — the source of truth |
| `dvc.yaml` | Stage definitions: commands, dependencies, outputs, metrics |
| `dvc.lock` | Record of what was actually reproduced, and from which inputs |

### Experiment tracking

DVC and MLflow answer different questions, and the project keeps both:

| | Question it answers |
| --- | --- |
| **DVC** | Can this run be reproduced? (data, stage graph, cached outputs) |
| **MLflow** | What happened in this run? (params, metrics, artifacts, comparisons) |

Training records one parent run per execution, with a nested run per candidate model
carrying its tuned hyperparameters, CV score and validation metrics. The parent holds
the run configuration, the champion, the frozen threshold, post-threshold validation
metrics, final test metrics, the promotion decision, and the git commit the run came
from. Metrics artifacts and the champion pipeline are attached to the run.

Browse the runs locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open <http://127.0.0.1:5000>.

Tracking is configured under `mlflow:` in `params.yaml` and can be switched off with
`enabled: false`, in which case training runs exactly as before and records nothing —
it is never a prerequisite for producing a model. Run metadata lives in a local SQLite
database (MLflow 3 retired the plain `./mlruns` file store) and artifacts in
`mlartifacts/`; both are gitignored. That section is deliberately **not** among the
params DVC tracks for the train stage, so renaming an experiment does not invalidate a
multi-hour run.

### Model registry

Tracking answers *what happened in this run*. The registry answers the question a
consumer actually asks: **which recorded run should I serve, and was it approved?**
Without it, "the champion" means whatever `models/champion_pipeline.joblib` happens to
contain on one machine.

The train stage logs the fitted champion both as a file artifact and as an MLflow
*model* — a raw file is something to download, a model carries its flavor and is the only
form a registry can version — and writes `artifacts/run_information.json`, the handoff
naming the run, experiment, model URI and promotion decision. The `register` stage reads
it and files that model under `mlflow.registered_model_name` from `params.yaml`.

**Registration is not promotion.** Every run that logged a model is registered, because a
refused champion is part of the history. What promotion controls is the *alias*:

| Outcome | Alias (`params.yaml`) | Version tag |
| --- | --- | --- |
| Passed every quality gate | `champion_alias` → `champion` | `validation_status=approved` |
| Failed a gate | `candidate_alias` → `candidate` | `validation_status=rejected` |

So `models:/modelium-credit-risk-champion@champion` cannot resolve to a model the
pipeline rejected — the same rule `src/inference/predictor.py` enforces for batch
scoring, applied at the other end of the handoff. A reversal *moves* the alias: a version
that was approved and is re-registered as rejected has the production alias removed, not
merely left unset.

**Re-running is idempotent.** DVC re-runs a stage whenever a dependency changes, so
registering the same `run_id` twice reuses the version it already created rather than
stacking versions that differ only in when the command ran.

Inspect what is registered, from the terminal:

```bash
make register                        # run the stage
make registry-show                   # versions, aliases and promotion status
cat artifacts/registry_record.json   # what the last run registered
```

`make registry-show` prints one row per version with its `validation_status`, aliases and
source run, then the serving URI — or an explicit statement that no version holds the
champion alias, which is a refusal rather than a gap.

Load the approved model the way a consumer would:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/modelium-credit-risk-champion@champion")
model.predict_proba(raw_dataframe)      # preprocessing travels with the pipeline
```

The same call with `@candidate` loads a version that failed its gates — useful for
debugging, never for serving. Browse the same store in a UI with `make mlflow-ui`
(<http://127.0.0.1:5000>), where the parent run, the nested candidate runs, their
parameters and metrics, and the registered model are all visible.

The registry is external state DVC can neither cache nor restore, so the stage outputs
`artifacts/registry_record.json` — the pipeline's own record of the outcome, readable
without a working MLflow install. With `mlflow.enabled: false` there is no run to
register; the stage records the skip and succeeds rather than failing the pipeline.

The store is the same local SQLite database used for tracking. There is no remote
tracking server.

## Explainability

The model explained is the one the **registry** says is approved — resolved through
`models:/modelium-credit-risk-champion@champion`, never read from whichever joblib is on
disk. A report about a model nobody is serving is worse than no report.

The stage is inference-only. The champion arrives as a fitted pipeline; SHAP needs the
transformed matrix and the estimator separately, so the preprocessor is used via
`transform` and the estimator is explained directly. Nothing calls `fit` — refitting
preprocessing on the explanation sample would describe a transformation the model was
never trained with.

```bash
make explain            # or: dvc repro explain
```

**What gets explained.** A deterministic sample (`explainability.sample_size`, seed
`random_state`) drawn from the **test** split, rebuilt with the same sizes, seed and
stratification the train stage used — so these are held-out rows, not training rows
presented as if they were. `SK_ID_CURR` is carried alongside to label local explanations
and is never a model input.

| Output | What it answers |
| --- | --- |
| `reports/figures/shap_summary.png` | Beeswarm — direction and spread per feature |
| `reports/figures/shap_bar.png` | Global ranking by mean \|SHAP\| |
| `artifacts/explainability/global_feature_importance.csv` | The full ranking: `feature`, `mean_abs_shap`, `rank` |
| `reports/figures/shap_local/shap_local_<SK_ID_CURR>.png` | One applicant's waterfall |
| `artifacts/explainability/local_explanations.json` | Per applicant: probability, threshold, predicted and actual class, base value, top contributors with feature values |
| `artifacts/explainability/explanation_report.json` | Provenance — model URI, version, source run, sample, explainer type, additivity check |

**Local examples span the decision, not the top of the risk ranking.** Five near-identical
high scores teach nothing, so the selection takes the extremes, the applicant nearest the
frozen threshold, and — where labels allow — a true positive and a false negative, which
are the two cases a credit reviewer actually asks about.

Two details worth knowing when reading the numbers:

**Explanations are in margin space.** Tree and linear explainers attribute to the model's
raw log-odds score, not its probability, so `base_value + sum(shap) == raw_margin` holds
while the same identity against `predict_proba` does not. The report records which space
it verified and the measured error rather than failing a valid explanation for
disagreeing with the wrong reference.

**The positive class is located, never assumed.** SHAP returns a 2-D array, a list of
per-class arrays, or a 3-D stack depending on estimator and library version. Each is
normalised to class `1` using the estimator's own `classes_`; taking index 1 on faith
would invert the sign of every contribution while still producing a well-formed report.

The explainer is chosen from the champion's family — `TreeExplainer` for LightGBM,
XGBoost, CatBoost and forests, `LinearExplainer` for linear models, and the general
`shap.Explainer` otherwise — so a change of champion does not need a code change.

Artifacts are also logged to a dedicated MLflow run tagged with `source_run_id`,
`registered_model_name`, `model_alias` and `model_version`, rather than appended to the
training run, so the historical model record is not disturbed.

### params.yaml

`params.yaml` is the single source of truth for experiment configuration — split sizes, seed, IQR
factor, tuning budget, gate thresholds and all three search spaces. Python config holds only
structural constants (paths, filenames, `TARGET_COL`, `ID_COL`).

Parameters are validated at load, so an invalid value fails in milliseconds naming the field rather
than minutes into a run. Two checks are worth knowing about: the loader rejects `tuning.scoring`
disagreeing with `selection.primary_metric` (optimising one estimator of the PR curve while ranking
by another lets the search winner lose the selection), and it rejects both parallelism layers
running at once.

DVC hashes the params sections each stage declares, so editing a search space re-runs training
without re-aggregating ~57 M child rows.

---

## Repository structure

```text
modelium/
├── config/config.py              # paths, filenames, column names
├── params.yaml                   # experiment configuration
├── dvc.yaml                      # validate → prepare → train → predict
├── dvc.lock                      # reproduced stage record
├── data/
│   ├── raw/                      # Home Credit CSVs (gitignored)
│   └── processed/                # prepared feature tables (DVC-tracked)
├── src/
│   ├── data/                     # loading, validation, cleaning, aggregation, splitting
│   ├── features/                 # domain features, preprocessing
│   ├── models/                   # training, tuning, evaluation, selection, threshold,
│   │                             #   serialization, registry
│   ├── inference/                # Predictor
│   ├── tracking/                 # MLflow experiment tracking
│   ├── explainability/           # SHAP global + local explanations
│   ├── monitoring/               # drift / fairness scaffolding (not wired in)
│   ├── visualization/
│   └── utils/                    # config loader, logger, exceptions
├── scripts/
│   ├── validate_data.py          # DVC stage 1
│   ├── prepare_data.py           # DVC stage 2
│   ├── train.py                  # DVC stage 3
│   ├── register_model.py         # DVC stage 4
│   ├── explain.py                # DVC stage 5
│   └── predict.py                # DVC stage 6
├── tests/unit/
├── artifacts/                    # metrics, deployment metadata, run/registry records,
│                                 #   predictions (gitignored)
└── models/                       # champion_pipeline.joblib (gitignored)
```

---

## Project status

**Implemented**

- Raw data validation (training and inference contracts)
- Relational aggregation and domain feature engineering
- Leak-free, production-safe preprocessing
- Train / validation / test separation
- Cross-validated hyperparameter tuning
- Champion selection on validation
- Threshold optimization on validation
- Reproducible DVC pipeline with `params.yaml`
- Batch inference with promotion safety
- MLflow experiment tracking (local)
- MLflow Model Registry with promotion-gated aliases (local)
- SHAP explainability — global importance and per-applicant explanations

**Not yet implemented**

- Remote MLflow tracking server
- Model monitoring (drift, fairness)
- FastAPI serving
- Docker packaging
- CI/CD
- AWS deployment

`src/monitoring/` contains scaffolding from an earlier iteration and is not part of the
pipeline.
