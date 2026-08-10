from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    """Fit IQR bounds on train data and clip future data using those bounds."""
    def __init__(self, columns: list[str] | None = None, factor: float = 1.5):
        self.columns = columns
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        cols = self.columns or X.select_dtypes(include=np.number).columns.tolist()
        self.bounds_ = {}
        for col in cols:
            if col not in X:
                continue
            q1, q3 = X[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            if col in X:
                X[col] = X[col].clip(lo, hi)
        return X


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Leak-free preprocessing fitted later only on training data."""
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)
