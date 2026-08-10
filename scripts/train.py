from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    DATA_DIR, DATA_FILES, RANDOM_STATE, TARGET_COL, ID_COL, MODEL_DIR, ARTIFACT_DIR,
)
from src.data.data_loader import load_home_credit_tables
from src.data.data_cleaning import optimize_memory, drop_low_information_columns
from src.data.data_preparation import build_relational_feature_table
from src.features.feature_engineering import add_domain_features
from src.features.data_preprocessing import build_preprocessor
from src.models.train import build_candidate_models, train_models
from src.models.evaluation import evaluate_model, get_probability_scores
from src.models.threshold import find_f1_optimal_threshold
from src.models.serialization import save_production_bundle


def main():
    tables = load_home_credit_tables(DATA_DIR, DATA_FILES)
    tables = {name: optimize_memory(df) for name, df in tables.items()}

    df = build_relational_feature_table(tables)
    df = add_domain_features(df)
    df, dropped = drop_low_information_columns(df, ID_COL, TARGET_COL)

    X = df.drop(columns=[TARGET_COL, ID_COL], errors="ignore")
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    candidates = build_candidate_models(RANDOM_STATE, scale_pos_weight)
    trained = train_models(candidates, X_train_t, y_train)

    results = [evaluate_model(model, X_test_t, y_test, name) for name, model in trained.items()]
    leaderboard = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["Model"]
    best_model = trained[best_name]
    probs = get_probability_scores(best_model, X_test_t)
    threshold_info = find_f1_optimal_threshold(y_test, probs)

    metadata = {
        "model_name": best_name,
        "optimal_threshold": threshold_info["threshold"],
        "target": TARGET_COL,
        "id_column": ID_COL,
        "raw_feature_columns": X.columns.tolist(),
        "dropped_columns": dropped,
        "primary_metric": "PR-AUC",
    }
    save_production_bundle(best_model, preprocessor, metadata, MODEL_DIR, ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(ARTIFACT_DIR / "model_leaderboard.csv", index=False)
    print(f"\nBest model: {best_name}")
    print(f"Optimal threshold: {threshold_info['threshold']:.4f}")


if __name__ == "__main__":
    main()
