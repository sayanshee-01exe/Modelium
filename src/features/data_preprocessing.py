"""Leak-free preprocessing for the Home Credit feature table.

Fitted on training data only; validation and test are transform-only. The same fitted
object is persisted by `src/models/serialization.py`, so training and inference apply
byte-identical transformations rather than two implementations that drift apart.

Numeric:      median impute -> IQR clip (train-derived bounds) -> standardise
Categorical:  most-frequent impute -> one-hot (handle_unknown="ignore")
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IQRClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to Tukey bounds learned from the training data.

    Bounds are ``[Q1 - factor*IQR, Q3 + factor*IQR]``, computed once during `fit` and
    reused verbatim by every later `transform`. Recomputing them per split would let
    validation and test statistics influence their own transformation, which is exactly
    the leak this class exists to avoid.

    Operates positionally on the column axis, because inside a numeric `Pipeline` it
    receives whatever the upstream imputer emits — a NumPy array, not a DataFrame.
    (The previous implementation keyed its bounds by column *label* and tested
    ``if col in X``; against an array that tests membership among the *values*, so it
    silently clipped nothing.) DataFrames are still accepted and returned as DataFrames.

    Args:
        factor: IQR multiplier. 1.5 is the conventional Tukey fence.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    @staticmethod
    def _as_array(X) -> np.ndarray:
        values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
        if values.ndim != 2:
            raise ValueError(f"IQRClipper expects a 2-D input, got shape {values.shape}")
        return values.astype(float, copy=False)

    def fit(self, X, y=None):
        values = self._as_array(X)
        if values.shape[0] == 0:
            raise ValueError("IQRClipper cannot fit on an empty array")

        q1 = np.nanpercentile(values, 25, axis=0)
        q3 = np.nanpercentile(values, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr

        # Set by hand rather than via _validate_data so the class stays compatible
        # across sklearn versions; OneToOneFeatureMixin reads these for feature names.
        self.n_features_in_ = values.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        values = self._as_array(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(
                f"IQRClipper was fitted on {self.n_features_in_} feature(s) but received "
                f"{values.shape[1]}; the inference feature layout must match training"
            )

        clipped = np.clip(values, self.lower_, self.upper_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(clipped, index=X.index, columns=X.columns)
        return clipped


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build the (unfitted) preprocessor from the training frame's column layout.

    Only the *column names and dtypes* of `X_train` are read here — no statistic is
    learned until `.fit()` is called by the caller on the training split.

    Dense vs sparse output (measured, not assumed):
      - All 8 candidate models accept a sparse matrix.
      - But `StandardScaler(with_mean=True)` refuses sparse input ("Cannot center
        sparse matrices"), so the numeric branch must stay dense.
      - The numeric branch is ~75% of the output width on the real feature table, so
        `ColumnTransformer` would exceed its `sparse_threshold` and densify the hstack
        regardless of what the encoder returns.
      Sparse output would therefore cost compatibility without saving memory. Dense
      output measures ~0.98 GB for the 246k-row training split, which is acceptable.
    """
    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = X_train.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("iqr_clipper", IQRClipper()),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": a category unseen at training encodes as all-zeros
        # instead of raising, so inference cannot crash on a new category value.
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    logger.info(
        "Preprocessor built for %d numeric and %d categorical column(s)",
        len(numeric), len(categorical),
    )
    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop", verbose_feature_names_out=False)


def get_output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Names of the transformed columns, positionally aligned with the output matrix.

    Needed by SHAP, importance ranking, MLflow metadata and inference debugging — all
    of which otherwise see an anonymous float matrix.
    """
    check_is_fitted(preprocessor)
    return [str(name) for name in preprocessor.get_feature_names_out()]


def get_input_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Raw input columns the preprocessor was fitted on, in their expected order.

    Persist these alongside the model: they are the contract an inference caller has
    to satisfy, and `align_to_training_schema` enforces it.
    """
    check_is_fitted(preprocessor)
    return [str(name) for name in preprocessor.feature_names_in_]


def align_to_training_schema(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    """Coerce an inference frame to the exact training column layout.

    Handles the three ways a caller can drift from the training schema:
      - **missing** columns  -> raise, since imputing an absent feature would silently
        score the applicant against a value the model never saw them provide
      - **extra** columns    -> dropped, with a log line
      - **wrong order**      -> reordered

    Args:
        df: Frame to align.
        expected_columns: Output of `get_input_feature_names`.

    Returns:
        A new frame containing exactly `expected_columns`, in that order.

    Raises:
        DataValidationError: if any expected column is absent.
    """
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Input is missing {len(missing)} column(s) required by the fitted "
            f"preprocessor: {sorted(missing)}"
        )

    extra = [c for c in df.columns if c not in expected_columns]
    if extra:
        logger.warning("Dropping %d unexpected input column(s): %s", len(extra), sorted(extra))

    return df.loc[:, expected_columns].copy()
