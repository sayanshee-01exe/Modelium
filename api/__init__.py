"""Local FastAPI service for the registered champion model.

Serving is deliberately thin: every rule that decides *what a prediction means* —
preprocessing, schema alignment, the frozen threshold, the promotion gate — already
exists in `src/inference/predictor.py` and is reused rather than reimplemented. A second
implementation of any of them is how training/serving skew starts.
"""
