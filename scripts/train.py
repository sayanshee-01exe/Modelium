"""DVC stage 3 — training entry point.

Flow: read prepared features -> split -> baseline + tuning -> validation comparison ->
champion + pre-threshold gates -> threshold -> operational gates -> final test
evaluation -> serialize.

Raw loading, validation and feature building happen in the `validate` and `prepare`
stages. This script begins where the modelling begins, so re-running it after a
params.yaml edit costs a search rather than a full re-aggregation.

Every experiment value is read from `params.yaml`; nothing is tuned in this file.

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

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    TARGET_COL, ID_COL, MODEL_DIR, ARTIFACT_DIR, METRICS_DIR,
    PREPARE_REPORT_FILE, TRAIN_FEATURES_FILE,
)
from src.data.data_preparation import split_train_val_test
from src.features.data_preprocessing import (
    get_input_feature_names, get_output_feature_names,
)
from src.utils.config_loader import load_params
from src.models.train import build_baseline_pipeline
from src.models.tune import PREPROCESSOR_STEP, summarize_tuning, tune_candidates
from src.models.selection import check_operational_gates, select_champion
from src.models.evaluation import (
    PRIMARY_METRIC, evaluate_model, evaluate_at_threshold, get_probability_scores,
)
from src.models.threshold import find_f1_optimal_threshold
from src.models.serialization import save_champion_pipeline
from src.utils.logger import get_logger

logger = get_logger("modelium.train")


def main():
    params = load_params()
    data_params, tuning_params = params["data"], params["tuning"]
    random_state = int(data_params["random_state"])

    # Features come from the prepare stage. Validation and aggregation already ran in
    # their own DVC stages, so this script starts where the modelling starts.
    df = pd.read_parquet(TRAIN_FEATURES_FILE)
    dropped = json.loads(PREPARE_REPORT_FILE.read_text()).get("dropped_columns", [])

    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y,
        validation_size=float(data_params["validation_size"]),
        test_size=float(data_params["test_size"]),
        random_state=random_state,
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
    baseline = build_baseline_pipeline(X_train, random_state)
    baseline.fit(X_train, y_train)

    logger.info("Tuning Random Forest / XGBoost / LightGBM on training CV folds...")
    searches = tune_candidates(
        X_train, y_train,
        n_iter=int(tuning_params["n_iter"]), cv_folds=int(tuning_params["cv_folds"]),
        scoring=str(tuning_params["scoring"]),
        random_state=random_state, scale_pos_weight=scale_pos_weight,
        n_jobs=int(tuning_params["search_n_jobs"]),
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
        # Which training run produced the artifact being served. Without it, a stale
        # champion_pipeline.joblib on disk is indistinguishable from a fresh one.
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
            "validation_size": float(data_params["validation_size"]),
            "test_size": float(data_params["test_size"]),
            "random_state": random_state,
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
    # accepts raw frames directly: champion.predict_proba(raw_dataframe). The frozen
    # threshold rides in metadata, since a sklearn Pipeline cannot carry one.
    save_champion_pipeline(best_model, metadata, MODEL_DIR, ARTIFACT_DIR)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(METRICS_DIR / "validation_leaderboard.csv", index=False)
    pd.DataFrame(summarize_tuning(searches)).to_csv(
        METRICS_DIR / "tuning_summary.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(METRICS_DIR / "test_metrics.csv", index=False)

    # Flat scalar JSON for `dvc metrics show/diff`. Same numbers as the CSV above; DVC
    # cannot diff a CSV row, so the headline metrics get a shape it understands.
    with open(METRICS_DIR / "test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, handle, indent=2)


if __name__ == "__main__":
    main()
