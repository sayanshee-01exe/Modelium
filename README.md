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
validate → prepare → train ─┬→ register ─┬→ explain
                            │             └→ monitor
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

## Monitoring

**Offline batch monitoring.** The stage compares one stored current batch against one
stored reference batch when it is run. There is no live traffic, no streaming and no
continuous evaluation — calling it production monitoring would overstate what it does.

```bash
make monitoring-batch   # rebuild the demonstration batch
make monitor            # or: dvc repro monitor
make monitoring-show    # print the last run's status
```

The model is the registry's approved champion, resolved through
`models:/modelium-credit-risk-champion@champion`. The stage calls `predict_proba` and
`transform` only — it never fits, retrains, registers a version, moves an alias, promotes
or rolls back. It reports, and stops there.

| Batch | What it is |
| --- | --- |
| Reference | A deterministic sample of the **training split** — the distribution the champion was fitted on |
| Current | `data/monitoring/current_batch.parquet`, built by `scripts/create_monitoring_batch.py` |

**The current batch is a demonstration.** This project has no production traffic, so the
script manufactures one from the held-out test split and marks it as manufactured: the
rows and labels are genuine and unseen, but any drift is injected, and the affected
feature names are written into the batch metadata and reproduced in the report. The
labels are recorded as `label_source: held_out_test_split`, never as observed production
outcomes.

### What it measures

| Section | Measures |
| --- | --- |
| Feature drift | PSI, KS statistic and p-value, Jensen-Shannon distance, missing-rate change per feature; unseen-category rate for categoricals |
| Prediction drift | Mean/median/std probability, positive rate, probability PSI — all at the **frozen threshold**, never 0.5 |
| Performance | AP, ROC-AUC, accuracy, precision, recall, F1, confusion matrix, FPR/FNR — **only when labels exist** |
| Fairness | Per-group rates and disparities across applicant segments |

Two behaviours are worth knowing:

**Labels are optional, and their absence is reported rather than filled in.** In credit
risk a default is observed months after scoring, so a batch usually has none. Without
them the stage still measures drift and prediction-only fairness, and records
`labels_available: false` with no metrics at all — a plausible Average Precision computed
without outcomes would be the most damaging thing it could produce.

**A significance test alone cannot flag a feature.** At 10,000 rows per side the KS
p-value rejects equality for differences far too small to act on — on a real run it alone
flagged nine features whose largest CDF gap was under 0.05. The KS statistic is the
effect size, so both are required.

### Outputs

```text
artifacts/monitoring/feature_drift.csv           one row per feature, worst first
artifacts/monitoring/prediction_drift.json       score distribution comparison
artifacts/monitoring/performance_metrics.json    metrics, or labels_available: false
artifacts/monitoring/fairness_metrics.csv        per-group rates and disparities
artifacts/monitoring/monitoring_summary.json     machine-readable status
artifacts/monitoring/monitoring_report.md        the written report
reports/figures/monitoring/*.png                 drift, prediction, missing-rate, group plots
```

`overall_status` is the **worst** section status, and every section status is carried
alongside it. A single green headline that absorbed a red section would be worse than no
headline, since it is the one line an operator reads.

### Fairness scope

The fairness figures are a **technical demonstration** of disparity measurement on a
public research dataset, not a legal compliance assessment. `CODE_GENDER`,
`NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE` and `NAME_INCOME_TYPE` are self-reported
application fields, not verified protected attributes; a disparity can arise from the
population, the sample or the model, and these metrics cannot separate those causes.
Groups below `minimum_group_size` are reported as `insufficient_data` and never merged
into another group to reach the threshold.

Each run is also logged to a dedicated MLflow run tagged `monitoring_run=true` and linked
to the champion by `source_training_run_id`, `model_version` and `model_alias`. The
training run's own metrics are never modified.

## API serving

A local FastAPI service that scores applicants with the **registered champion**.

```bash
make api                    # or: uvicorn api.main:app --host 127.0.0.1 --port 8000
make api-test
```

Interactive schema at <http://127.0.0.1:8000/docs>, ReDoc at `/redoc`, raw spec at
`/openapi.json`.

The model is resolved through the registry alias
`models:/modelium-credit-risk-champion@champion`, never from a local filename, so the
service answers with whatever version the pipeline actually promoted. It is loaded **once**
at startup — resolving an alias and unpickling a fitted pipeline takes seconds, and doing
that per request would make latency dominated by loading.

Scoring reuses `src/inference/predictor.py` rather than reimplementing it, so the API
inherits the same preprocessing, schema alignment, positive-class resolution, promotion
gate and **frozen decision threshold** that batch inference uses. A second implementation
of any of those is how training/serving skew starts.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness. Answers even when the model failed to load |
| `GET /ready` | Readiness — five named checks; 200 ready, 503 not |
| `GET /model/info` | Which version is answering, its threshold and metrics |
| `POST /predict` | Score one applicant |
| `POST /predict/batch` | Score a bounded list |
| `POST /explain` | Per-applicant SHAP contributions |

**Applicants are identified, not described.** The champion expects 407 engineered
features built by a relational aggregation over ~57 M child rows. Requiring a caller to
supply those by hand would be unusable, and rebuilding them per request would duplicate
the training feature pipeline. So the default contract is an id, resolved against the
precomputed feature store; a caller holding an engineered row may pass `features`
instead.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sk_id_curr": 100005}'
```

```json
{
  "sk_id_curr": 100005,
  "default_probability": 0.6025,
  "predicted_class": 0,
  "threshold": 0.6916,
  "model_version": "2",
  "request_id": "3f2a…"
}
```

That response shows why the frozen threshold matters: at 0.6025 this applicant is **not**
flagged, because the threshold tuned on validation is 0.6916. Under sklearn's default 0.5
the same score would have been a default prediction.

**Safety behaviours worth knowing:**

- An **unpromoted** model is refused, not served. `Predictor` performs that check, and
  the API never passes `allow_unpromoted`.
- If the registry pipeline and the on-disk metadata describe **different champions**, the
  service refuses to start serving — a model with another run's threshold and schema is
  silently wrong in the worst way.
- A failed load is a served **503**, not a crashed process: `/ready` names which of the
  five checks failed, which tells an operator more than a container that exits.
- Batch size is capped by `api.max_batch_size`, and duplicate identifiers are rejected
  because they make a response ambiguous to join back.
- Errors are structured with a stable `category`; **no stack trace ever reaches a
  client**, and the log carries no applicant records or feature vectors.
- Every response carries `X-Request-ID` (a supplied one is preserved) and
  `X-Response-Time-ms`.

`/explain` returns per-applicant SHAP contributions using the same
`src/explainability/shap_explainer.py` the pipeline uses, with the explainer built once
at startup. Contributions are in **margin (log-odds) space** — they sum to the raw score,
not to the probability. Global SHAP artifacts come from the `explain` pipeline stage, not
from a request.

### Limitations

This is **local serving**. There is **no authentication**, no rate limiting and no
transport security in this layer, so it is not safe to expose publicly as it stands. CORS
middleware is deliberately absent, so no browser origin is permitted by default. The API
is **not** a DVC stage — it is a long-running service, not a reproducible batch step, so
it has no place in the pipeline graph. Docker packaging, CI/CD and cloud deployment are
future steps and are not implemented.

## Docker

Containerized serving, so the champion can be served without depending on a host Python
environment.

**Prerequisites:** Docker Desktop (or any Docker engine) running, and a champion already
registered — the container reads the registry, it never trains or registers.

```bash
make docker-build      # build the image
make docker-up         # start the stack, detached
make docker-health     # check /health and /ready
make docker-logs       # follow the API log
make docker-down       # stop
```

| Service | URL |
| --- | --- |
| API | <http://127.0.0.1:8000> |
| Swagger | <http://127.0.0.1:8000/docs> |
| Health | <http://127.0.0.1:8000/health> |
| Readiness | <http://127.0.0.1:8000/ready> |
| MLflow UI (opt-in) | <http://127.0.0.1:5000> via `make docker-mlflow` |

### What is in the image, and what is not

The image holds **code only**. Every piece of mutable state is mounted at run time, so
the same image serves a newly trained champion without being rebuilt, and no image is
ever a snapshot of a particular database.

| Mounted | Mode | Why |
| --- | --- | --- |
| `./mlflow.db` | read-only | The registry the champion alias resolves through |
| `./mlartifacts` | read-only | The champion's serialized files |
| `./artifacts` | read-only | Deployment metadata: frozen threshold, schema, promotion |
| `./data/processed/test_features.parquet` | read-only | Feature lookup by `SK_ID_CURR` |
| `modelium-runtime-state` (named volume) | writable | The container's own rebased registry copy |

### The one thing that needs explaining

MLflow records artifact locations as **absolute paths on the machine that wrote them**.
On this project the database contains entries like
`file:///Users/<someone>/.../mlartifacts/models/m-748d…`, which do not exist inside a
Linux container — so the champion would not load.

`docker/entrypoint.sh` runs `scripts/prepare_container_registry.py` before the API
starts. It copies the mounted database to a writable location and rewrites the four
columns that hold paths (`runs.artifact_uri`, `logged_models.artifact_location`,
`experiments.artifact_location`, `model_versions.storage_location`) onto the container's
mounts. It never looks for a particular home directory — it finds the `/mlartifacts` or
`/mlruns` segment and replaces everything before it — so it works whatever machine wrote
the database. **The host database is opened read-only and never modified.**

This failure is worth knowing about because of how it hides: on the host, a relocated
store appears to load fine, because MLflow quietly falls back to reading the original
directory. A container has no such directory, so what looks correct in development fails
only once containerized.

### Environment variables

Defaults come from `params.yaml`; the container overrides them through the environment,
so no deployment needs to edit a tracked file.

| Variable | Default in compose |
| --- | --- |
| `MODELIUM_API_HOST` / `MODELIUM_API_PORT` | `0.0.0.0` / `8000` |
| `MODELIUM_MAX_BATCH_SIZE` | `100` |
| `MODELIUM_MODEL_URI` | `models:/modelium-credit-risk-champion@champion` |
| `MLFLOW_TRACKING_URI` | `sqlite:////app/runtime/state/mlflow.db` |
| `MODELIUM_FEATURE_STORE_PATH` | `/app/runtime/data/test_features.parquet` |
| `MODELIUM_DEPLOYMENT_METADATA_PATH` | `/app/runtime/artifacts/deployment_meta.json` |
| `MODELIUM_REGISTRY_RECORD_PATH` | `/app/runtime/artifacts/registry_record.json` |

Note the **four slashes** in the SQLite URI. `sqlite:///app/...` is a *relative* path
called `app`, and MLflow would silently create an empty database with no champion in it.

### Container properties

- Runs as a **non-root** user (`modelium`, uid 1001).
- Installs `requirements-api.txt`, the **serving subset**. The development manifest pulls
  DVC, Optuna and CatBoost, none of which the API imports — dropping them took the image
  from 4.3 GB to 3.4 GB. `xgboost` is kept despite its size (it drags in ~290 MB of CUDA
  libraries) so an XGBoost champion serves without a rebuild; the API must not assume
  LightGBM. Keep the two manifests in step when a serving dependency changes.
- **Multi-stage build**: wheels are compiled in a builder stage, so no compiler toolchain
  ships in the runtime image.
- Python **3.13**, matching the interpreter the champion was pickled under — unpickling a
  fitted sklearn/LightGBM pipeline across minor versions is not guaranteed.
- `HEALTHCHECK` probes `/health`, never `/predict`: a probe that scores an applicant
  turns monitoring into load.
- Resource limits of **4 GB / 2 CPUs** are a laptop-sized starting point, not a measured
  requirement. Raise them if `/explain` is slow or the container is OOM-killed at start.
- Serving is **not** a DVC stage. Docker is a deployment concern, not a reproducible
  batch step.

`/health` is liveness and answers even when the model failed to load; `/ready` is what an
orchestrator should gate traffic on, and still validates all five checks — model loaded,
metadata, threshold, schema, champion alias.

### Smoke test

```bash
bash scripts/docker_smoke_test.sh     # or: make docker-test
```

Builds, starts, waits for health and readiness, checks `/model/info` carries a promoted
champion with a non-0.5 threshold, scores applicant `100001`, verifies the class follows
the frozen threshold, checks the 404 and 422 contracts, and tears the stack down on any
exit path. It deliberately asserts **no exact probability** — that would tie the test to
one trained artifact.

### Troubleshooting

| Symptom | Cause and check |
| --- | --- |
| Container exits at start | The entrypoint failed to resolve the champion. `docker compose logs api` names the missing mount |
| `alias 'champion' does not resolve` | No promoted champion registered. Run `make register` on the host first |
| `No MLflow database at /app/runtime/mlflow.db` | `mlflow.db` missing from the repo root — run the pipeline before serving |
| `artifacts are not readable` | The `mlartifacts` mount is missing or empty |
| `/ready` returns 503 | `curl -s localhost:8000/ready` — each of the five checks reports its own reason |
| Applicant returns 404 | The feature-store parquet is not mounted, or that id is not in it |
| Permission denied | Something is writing outside `/app/runtime/state`; only that volume is writable |
| Port already in use | `lsof -ti :8000` — a local `make api` may still be running |
| Slow first request | Cold start loads a large pipeline; the healthcheck allows 90s before failing |
| Apple Silicon | The image builds natively for arm64; no `platform:` override is needed |

```bash
docker compose ps                  # service state and health
docker compose logs api            # startup and per-request logs
docker inspect modelium-api        # full container configuration
curl -s localhost:8000/ready       # which readiness check is failing
```

### Limitations

Local containerization only. There is still **no authentication**, no rate limiting and
no TLS, so this is not safe to expose publicly. CI/CD, Kubernetes and cloud deployment
are not implemented.

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
│   ├── monitoring/               # drift, prediction drift, performance, fairness
│   ├── visualization/
│   └── utils/                    # config loader, logger, exceptions
├── scripts/
│   ├── validate_data.py          # DVC stage 1
│   ├── prepare_data.py           # DVC stage 2
│   ├── train.py                  # DVC stage 3
│   ├── register_model.py         # DVC stage 4
│   ├── explain.py                # DVC stage 5
│   ├── monitor.py                # DVC stage 6
│   └── predict.py                # DVC stage 7
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
- Offline batch monitoring — data drift, prediction drift, performance, fairness
- FastAPI serving of the registered champion (local)
- Docker containerization of the serving API

**Not yet implemented**

- Remote MLflow tracking server
- API authentication
- Docker packaging
- CI/CD
- AWS deployment

Every module under `src/` is now referenced by a pipeline stage.
