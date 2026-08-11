"""Training entry point.

Flow: load -> validate -> clean -> aggregate -> features -> split -> baseline + tuning
-> validation comparison -> champion + pre-threshold gates -> threshold -> operational
gates -> final test evaluation -> serialize.

Preprocessing is not applied here. It is the first step of every model pipeline, so it
is refitted inside each CV fold and travels with the serialized champion.

Split discipline (Step 1 of the refactor plan):

    TRAIN       fit the baseline pipeline and run every CV fold of the randomised
                search; each fold fits its own preprocessing
    VALIDATION  compare models on Average Precision, select the champion, apply
                pre-threshold gates, tune the decision threshold, then apply the
                operational gates at that threshold
    TEST        scored exactly once, at the end, with model and threshold frozen

Nothing above the "FINAL TEST EVALUATION" banner may read the test split.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_DIR, DATA_FILES, RANDOM_STATE, TARGET_COL, ID_COL, MODEL_DIR, ARTIFACT_DIR,
    VALIDATION_SIZE, TEST_SIZE, TUNING_N_ITER, TUNING_CV_FOLDS,
)
from src.data.data_loader import load_home_credit_tables
from src.data.data_validation import validate_raw_files, validate_raw_tables
from src.data.data_cleaning import optimize_memory, drop_low_information_columns
from src.data.data_preparation import build_relational_feature_table, split_train_val_test
from src.features.feature_engineering import add_domain_features
from src.features.data_preprocessing import (
    get_input_feature_names, get_output_feature_names,
)
from src.models.train import build_baseline_pipeline
from src.models.tune import PREPROCESSOR_STEP, summarize_tuning, tune_candidates
from src.models.selection import check_operational_gates, select_champion
from src.models.evaluation import (
    PRIMARY_METRIC, evaluate_model, evaluate_at_threshold, get_probability_scores,
)
from src.models.threshold import find_f1_optimal_threshold
from src.models.serialization import save_production_bundle
from src.utils.logger import get_logger

logger = get_logger("modelium.train")


def main():
    # load -> validate -> clean -> aggregate. Files are checked before any read, so a
    # missing table fails in milliseconds rather than after ~7 GB of I/O, and table
    # contracts are checked before anything transforms or aggregates the data.
    logger.info("Validating raw dataset files...")
    validate_raw_files(DATA_DIR, DATA_FILES)

    tables = load_home_credit_tables(DATA_DIR, DATA_FILES)

    logger.info("Validating loaded dataframes...")
    validate_raw_tables(tables, target_col=TARGET_COL)

    logger.info("Starting memory optimization...")
    tables = {name: optimize_memory(df) for name, df in tables.items()}

    df = build_relational_feature_table(tables)
    df = add_domain_features(df)
    df, dropped = drop_low_information_columns(df, ID_COL, TARGET_COL)

    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        validation_size=VALIDATION_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(
        f"Split — train {len(X_train):,} ({y_train.mean():.3%} positive) | "
        f"val {len(X_val):,} ({y_val.mean():.3%}) | "
        f"test {len(X_test):,} ({y_test.mean():.3%}) [held out]"
    )

    # Preprocessing is NOT fitted here. It is a step inside every model pipeline, so
    # each CV fold fits its own imputer/clipper/scaler/encoder on that fold's training
    # portion only. Fitting once on all of X_train and searching over the transformed
    # matrix would let each fold's held-out rows shape their own transformation.
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    # ------------------------------------------------------- TRAIN: baseline + tuning
    # Both the baseline fit and every CV fold of the search live inside X_train.
    logger.info("Fitting Logistic Regression baseline pipeline...")
    baseline = build_baseline_pipeline(X_train, RANDOM_STATE)
    baseline.fit(X_train, y_train)

    logger.info("Tuning Random Forest / XGBoost / LightGBM on training CV folds...")
    searches = tune_candidates(
        X_train, y_train,
        n_iter=TUNING_N_ITER, cv_folds=TUNING_CV_FOLDS,
        random_state=RANDOM_STATE, scale_pos_weight=scale_pos_weight,
    )

    trained = {"Logistic Regression": baseline}
    trained.update({name: search.best_estimator_ for name, search in searches.items()})

    # ---------------------------------------------------------------- VALIDATION
    # Model comparison and champion selection happen here, never on test.
    results = [evaluate_model(model, X_val, y_val, name) for name, model in trained.items()]
    champion = select_champion(results)
    leaderboard = champion.leaderboard
    print("\nValidation leaderboard (selection metric: Average Precision)")
    print(leaderboard.to_string(index=False))

    best_name = champion.name
    best_model = trained[best_name]
    if not champion.promoted:
        print(f"\nWARNING: champion {best_name} failed {len(champion.gate_failures)} "
              f"pre-threshold quality gate(s):")
        for failure in champion.gate_failures:
            print(f"  - {failure}")

    # Threshold is tuned on validation probabilities, then frozen.
    val_probs = get_probability_scores(best_model, X_val)
    threshold_info = find_f1_optimal_threshold(y_val, val_probs)
    frozen_threshold = float(threshold_info["threshold"])
    print(
        f"\nChampion: {best_name}\n"
        f"Threshold tuned on validation: {frozen_threshold:.4f} "
        f"(val F1={threshold_info['f1']:.4f}, "
        f"precision={threshold_info['precision']:.4f}, "
        f"recall={threshold_info['recall']:.4f})"
    )

    # Operational gates run only now: Recall/Precision/F1 are threshold-dependent, so
    # before this point they describe the arbitrary 0.5 default rather than the cut-off
    # this pipeline will actually ship with.
    val_metrics_at_threshold = evaluate_at_threshold(y_val, val_probs, frozen_threshold, best_name)
    operational_passed, operational_failures = check_operational_gates(val_metrics_at_threshold)
    if not operational_passed:
        print(f"\nWARNING: champion failed {len(operational_failures)} operational gate(s) "
              f"at the tuned threshold:")
        for failure in operational_failures:
            print(f"  - {failure}")

    # -------------------------------------------------- FINAL TEST EVALUATION
    # First and only read of the test split. Everything above is frozen.
    test_probs = get_probability_scores(best_model, X_test)
    test_metrics = evaluate_at_threshold(y_test, test_probs, frozen_threshold, best_name)
    print("\nFinal test evaluation (test set touched exactly once)")
    for key, value in test_metrics.items():
        print(f"  {key:<10} {value:.4f}" if isinstance(value, float) else f"  {key:<10} {value}")

    # best_model IS the full preprocessing+model pipeline; this is the same fitted
    # transformer, extracted so its feature names can be recorded in metadata.
    champion_preprocessor = best_model.named_steps[PREPROCESSOR_STEP]

    metadata = {
        "model_name": best_name,
        "optimal_threshold": frozen_threshold,
        "threshold_selected_on": "validation",
        "champion_selected_on": "validation",
        "primary_metric": PRIMARY_METRIC,
        "promoted": bool(champion.promoted and operational_passed),
        "pre_threshold_gate_failures": list(champion.gate_failures),
        "operational_gate_failures": list(operational_failures),
        "validation_metrics_at_threshold": val_metrics_at_threshold,
        "tuning": summarize_tuning(searches),
        "split": {
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "validation_size": VALIDATION_SIZE,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
        },
        "validation_metrics": leaderboard.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "target": TARGET_COL,
        "id_column": ID_COL,
        "raw_feature_columns": X.columns.tolist(),
        # The schema contract an inference caller must satisfy, taken from the fitted
        # preprocessor rather than from X, plus the transformed names SHAP and feature
        # importance need to label an otherwise anonymous float matrix.
        "input_feature_columns": get_input_feature_names(champion_preprocessor),
        "transformed_feature_names": get_output_feature_names(champion_preprocessor),
        "n_transformed_features": len(get_output_feature_names(champion_preprocessor)),
        "dropped_columns": dropped,
    }
    # best_model IS the full preprocessing+model pipeline, so the serialized artifact
    # accepts raw frames directly: champion.predict_proba(raw_dataframe).
    save_production_bundle(best_model, champion_preprocessor, metadata, MODEL_DIR, ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(ARTIFACT_DIR / "validation_leaderboard.csv", index=False)
    pd.DataFrame(summarize_tuning(searches)).to_csv(
        ARTIFACT_DIR / "tuning_summary.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(ARTIFACT_DIR / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
