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
Batch Inference
```

DVC DAG:

```text
validate → prepare → train → predict
```

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

Place the Home Credit CSVs in `data/raw/`, then:

```bash
pytest tests/unit -v          # no data required
dvc dag                       # inspect the pipeline
dvc repro                     # run every out-of-date stage
```

Individual stages:

```bash
dvc repro validate
dvc repro prepare
dvc repro train
dvc repro predict
```

`dvc repro` skips stages whose dependencies are unchanged, and `dvc status` reports what is stale.

| File | Role |
| --- | --- |
| `params.yaml` | Experiment configuration — the source of truth |
| `dvc.yaml` | Stage definitions: commands, dependencies, outputs, metrics |
| `dvc.lock` | Record of what was actually reproduced, and from which inputs |

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
│   ├── models/                   # training, tuning, evaluation, selection, threshold, serialization
│   ├── inference/                # Predictor
│   ├── explainability/           # SHAP scaffolding (not wired into the pipeline)
│   ├── monitoring/               # drift / fairness scaffolding (not wired in)
│   ├── visualization/
│   └── utils/                    # config loader, logger, exceptions
├── scripts/
│   ├── validate_data.py          # DVC stage 1
│   ├── prepare_data.py           # DVC stage 2
│   ├── train.py                  # DVC stage 3
│   └── predict.py                # DVC stage 4
├── tests/unit/
├── artifacts/                    # metrics, deployment metadata, predictions (gitignored)
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

**Not yet implemented**

- MLflow experiment tracking
- SHAP explainability reporting
- Model monitoring (drift, fairness)
- FastAPI serving
- Docker packaging
- CI/CD
- AWS deployment

`src/explainability/` and `src/monitoring/` contain scaffolding from an earlier iteration and are
not part of the pipeline.
