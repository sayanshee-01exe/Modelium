from __future__ import annotations

import json
from pathlib import Path
import joblib


def save_production_bundle(model, preprocessor, metadata: dict, model_dir, artifact_dir):
    model_dir = Path(model_dir); artifact_dir = Path(artifact_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "best_model.pkl", compress=3)
    joblib.dump(preprocessor, model_dir / "preprocessor.pkl", compress=3)
    with open(artifact_dir / "deployment_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
