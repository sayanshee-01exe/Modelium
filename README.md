# LendGuard AI — Modular Refactor

This refactors the original 3,136-line notebook-style `pipeline.py` into responsibility-based modules inspired by the Swiggy delivery-time project's `src/` layout.

## Structure

```text
lendguard_modular/
├── config/
│   └── config.py
├── data/
│   └── raw/                       # put 7 Home Credit CSV files here
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   ├── data_cleaning.py
│   │   └── data_preparation.py
│   ├── features/
│   │   ├── feature_engineering.py
│   │   └── data_preprocessing.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluation.py
│   │   ├── threshold.py
│   │   └── serialization.py
│   ├── visualization/
│   │   └── plots.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   ├── monitoring/
│   │   ├── drift.py
│   │   └── fairness.py
│   └── utils/
│       └── logger.py
├── scripts/
│   └── train.py
├── requirements.txt
└── README.md
```

## Main execution flow

```text
load_home_credit_tables
        ↓
optimize_memory
        ↓
build_relational_feature_table
        ↓
add_domain_features
        ↓
train_test_split
        ↓
build_preprocessor.fit(X_train)
        ↓
transform train/test
        ↓
build_candidate_models
        ↓
train_models
        ↓
evaluate_model
        ↓
select best PR-AUC model
        ↓
find_f1_optimal_threshold
        ↓
save_production_bundle
```

## Important improvements over the original monolithic pipeline

- removes the hard-coded Windows dataset path
- adds the missing `json` concern through a dedicated serialization module
- avoids fitting LabelEncoder on the full dataset
- keeps preprocessing fitted on train only
- saves one reusable preprocessing object, not only a scaler
- keeps relational aggregation separate from domain feature engineering
- separates training from evaluation, thresholding, SHAP, fairness, and drift
- makes the code importable and testable

## Run

1. Put the seven CSV files in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python scripts/train.py`

This first refactor intentionally keeps model hyperparameter tuning, MLflow, API, and Streamlit out of the training script. Those should be separate entry points instead of being generated dynamically from the training file.
