"""Training entry point.

Flow: load -> validate -> clean -> aggregate -> features -> split -> train -> evaluate.

Split discipline (Step 1 of the refactor plan):

    TRAIN       fit the preprocessor, fit every candidate model
    VALIDATION  compare models, select the champion, tune the decision threshold
    TEST        scored exactly once, at the end, with the threshold already frozen

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
    VALIDATION_SIZE, TEST_SIZE,
)
from src.data.data_loader import load_home_credit_tables
from src.data.data_validation import validate_raw_files, validate_raw_tables
from src.data.data_cleaning import optimize_memory, drop_low_information_columns
from src.data.data_preparation import build_relational_feature_table, split_train_val_test
from src.features.feature_engineering import add_domain_features
from src.features.data_preprocessing import (
    align_to_training_schema, build_preprocessor,
    get_input_feature_names, get_output_feature_names,
)
from src.models.train import build_candidate_models, train_models
from src.models.evaluation import evaluate_model, evaluate_at_threshold, get_probability_scores
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

    # Fit on TRAIN only. Validation and test are transform-only, so no statistic from
    # either can leak into the fitted imputers, encoder categories, or scaler.
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)

    # Align before transforming. The three splits come from one frame so their layouts
    # already match, making this a no-op here — it is kept because it exercises the same
    # schema contract inference will depend on, so a future layout drift fails loudly in
    # training rather than silently at serving time.
    training_columns = get_input_feature_names(preprocessor)
    X_val_t = preprocessor.transform(align_to_training_schema(X_val, training_columns))
    X_test_t = preprocessor.transform(align_to_training_schema(X_test, training_columns))

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    candidates = build_candidate_models(RANDOM_STATE, scale_pos_weight)
    trained = train_models(candidates, X_train_t, y_train)

    # ---------------------------------------------------------------- VALIDATION
    # Model comparison and champion selection happen here, never on test.
    results = [evaluate_model(model, X_val_t, y_val, name) for name, model in trained.items()]
    leaderboard = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    print("\nValidation leaderboard (selection metric: PR-AUC)")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["Model"]
    best_model = trained[best_name]

    # Threshold is tuned on validation probabilities, then frozen.
    val_probs = get_probability_scores(best_model, X_val_t)
    threshold_info = find_f1_optimal_threshold(y_val, val_probs)
    frozen_threshold = float(threshold_info["threshold"])
    print(
        f"\nChampion: {best_name}\n"
        f"Threshold tuned on validation: {frozen_threshold:.4f} "
        f"(val F1={threshold_info['f1']:.4f}, "
        f"precision={threshold_info['precision']:.4f}, "
        f"recall={threshold_info['recall']:.4f})"
    )

    # -------------------------------------------------- FINAL TEST EVALUATION
    # First and only read of the test split. Everything above is frozen.
    test_probs = get_probability_scores(best_model, X_test_t)
    test_metrics = evaluate_at_threshold(y_test, test_probs, frozen_threshold, best_name)
    print("\nFinal test evaluation (test set touched exactly once)")
    for key, value in test_metrics.items():
        print(f"  {key:<10} {value:.4f}" if isinstance(value, float) else f"  {key:<10} {value}")

    metadata = {
        "model_name": best_name,
        "optimal_threshold": frozen_threshold,
        "threshold_selected_on": "validation",
        "champion_selected_on": "validation",
        "primary_metric": "PR-AUC",
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
        "input_feature_columns": get_input_feature_names(preprocessor),
        "transformed_feature_names": get_output_feature_names(preprocessor),
        "n_transformed_features": len(get_output_feature_names(preprocessor)),
        "dropped_columns": dropped,
    }
    save_production_bundle(best_model, preprocessor, metadata, MODEL_DIR, ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(ARTIFACT_DIR / "validation_leaderboard.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(ARTIFACT_DIR / "test_metrics.csv", index=False)


if __name__ == "__main__":
    main()
