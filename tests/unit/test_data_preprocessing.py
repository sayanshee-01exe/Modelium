"""Step 3 — production-safe preprocessing.

Two things these pin down. First, the leakage contract: the preprocessor is fitted on
train only, and transforming validation or test must not move a single fitted statistic.
Second, the regression that motivated the rewrite — `IQRClipper` used to key its bounds
by DataFrame column label, then receive a NumPy array from the upstream imputer, so
`if col in X` tested membership against *values* and the clipper silently did nothing.
A shape-only assertion would not have caught that, so the tests below check clipped
*values*.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.data_preprocessing import (
    IQRClipper,
    align_to_training_schema,
    build_preprocessor,
    get_input_feature_names,
    get_output_feature_names,
)


@pytest.fixture
def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test with missing values in both dtypes and an unseen test category."""
    train = pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 8.0],
        "num_b": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "cat_a": ["x", "y", "x", "y", "x", None, "y", "x"],
    })
    val = pd.DataFrame({
        "num_a": [2.5, np.nan, 4.5],
        "num_b": [11.5, 12.5, 13.5],
        "cat_a": ["x", "y", None],
    })
    test = pd.DataFrame({
        "num_a": [3.5, 999.0, np.nan],
        "num_b": [12.0, 13.0, 14.0],
        "cat_a": ["x", "zzz_unseen", "y"],   # unseen category
    })
    return train, val, test


# ------------------------------------------------------------------ 1-4: fit/transform

def test_preprocessor_fits_successfully(frames) -> None:
    train, _, _ = frames
    pre = build_preprocessor(train)
    assert pre.fit(train) is pre


def test_training_transformation_works(frames) -> None:
    train, _, _ = frames
    pre = build_preprocessor(train)
    out = pre.fit_transform(train)
    assert out.shape[0] == len(train)
    assert np.isfinite(np.asarray(out, dtype=float)).all(), "no NaNs may survive preprocessing"


def test_validation_transformation_does_not_refit(frames) -> None:
    train, val, _ = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    before = _fitted_state(pre)
    pre.transform(val)
    assert _fitted_state(pre) == before, "transform(val) mutated fitted state"


def test_test_transformation_does_not_refit(frames) -> None:
    train, val, test = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    before = _fitted_state(pre)
    pre.transform(val)
    pre.transform(test)
    assert _fitted_state(pre) == before, "transform(test) mutated fitted state"


def _fitted_state(pre) -> str:
    """Serialised snapshot of every learned statistic in the fitted preprocessor."""
    parts = []
    for name, trans, _cols in pre.transformers_:
        if not hasattr(trans, "named_steps"):
            continue
        for step_name, step in trans.named_steps.items():
            for attr in ("statistics_", "lower_", "upper_", "mean_", "scale_", "categories_"):
                if hasattr(step, attr):
                    parts.append(f"{name}.{step_name}.{attr}={np.asarray(getattr(step, attr), dtype=object)}")
    return "|".join(parts)


# ------------------------------------------------------------- 5-7: missing / unseen

def test_missing_numeric_values_are_imputed(frames) -> None:
    train, _, _ = frames
    pre = build_preprocessor(train)
    out = np.asarray(pre.fit_transform(train), dtype=float)
    assert not np.isnan(out).any()


def test_missing_categorical_values_are_imputed(frames) -> None:
    train, _, _ = frames
    assert train["cat_a"].isna().any(), "fixture must contain a missing category"
    pre = build_preprocessor(train)
    out = np.asarray(pre.fit_transform(train), dtype=float)
    assert not np.isnan(out).any()


def test_unseen_category_does_not_crash(frames) -> None:
    """handle_unknown='ignore' — an unseen category at inference must encode as all-zeros."""
    train, _, test = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    out = np.asarray(pre.transform(test), dtype=float)
    assert out.shape[0] == len(test)
    assert np.isfinite(out).all()


# --------------------------------------------------------------------- 8-9: IQRClipper

def test_iqr_clipper_learns_bounds_on_fit() -> None:
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clipper = IQRClipper(factor=1.5).fit(X)
    assert hasattr(clipper, "lower_") and hasattr(clipper, "upper_")
    assert clipper.upper_[0] < 100.0, "the outlier must fall outside the learned upper bound"


def test_iqr_clipper_actually_clips_values() -> None:
    """The regression: the previous implementation was a silent no-op on arrays."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [100.0]])
    clipper = IQRClipper(factor=1.5).fit(X)
    out = np.asarray(clipper.transform(X))
    assert out.max() < 100.0, "outlier was not clipped"
    assert out.max() == pytest.approx(clipper.upper_[0])


def test_iqr_clipper_reuses_fitted_bounds_on_transform() -> None:
    """Bounds must come from train; a wilder test set must not widen them."""
    train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    clipper = IQRClipper(factor=1.5).fit(train)
    learned_upper = float(clipper.upper_[0])

    wild = np.array([[1000.0], [2000.0], [3000.0]])
    out = np.asarray(clipper.transform(wild))

    assert float(clipper.upper_[0]) == learned_upper, "bounds were recomputed on transform"
    assert out.max() == pytest.approx(learned_upper)


def test_iqr_clipper_clips_inside_the_real_pipeline(frames) -> None:
    """End-to-end: test's 999.0 must be pinned to the train-derived upper bound.

    Note the bound is *not* the training maximum — Tukey fences sit at Q3 + 1.5*IQR,
    which here is 9.0 against an observed train max of 8.0. So the correct invariant is
    that the extreme value lands exactly on the scaled upper bound, not that it stays
    below the scaled train max.
    """
    train, _, test = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    names = get_output_feature_names(pre)
    out = pd.DataFrame(np.asarray(pre.transform(test), dtype=float), columns=names)

    num_pipe = pre.named_transformers_["num"]
    clipper = num_pipe.named_steps["iqr_clipper"]
    scaler = num_pipe.named_steps["scaler"]
    idx = list(train.select_dtypes(include=np.number).columns).index("num_a")

    scaled_bound = (clipper.upper_[idx] - scaler.mean_[idx]) / scaler.scale_[idx]
    assert out["num_a"].max() == pytest.approx(scaled_bound), "999.0 was not clipped to the bound"

    unclipped = (999.0 - scaler.mean_[idx]) / scaler.scale_[idx]
    assert out["num_a"].max() < unclipped / 10, "clipping had no meaningful effect"


def test_iqr_clipper_rejects_wrong_feature_count() -> None:
    clipper = IQRClipper().fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        clipper.transform(np.array([[1.0], [2.0]]))


def test_iqr_clipper_preserves_dataframe_type() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 400.0]})
    out = IQRClipper().fit(df).transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a"]


# -------------------------------------------------------------- 10: schema stability

def test_train_val_test_share_feature_count(frames) -> None:
    train, val, test = frames
    pre = build_preprocessor(train)
    tr = pre.fit_transform(train)
    va = pre.transform(val)
    te = pre.transform(test)
    assert tr.shape[1] == va.shape[1] == te.shape[1]


def test_feature_count_matches_feature_names(frames) -> None:
    train, _, _ = frames
    pre = build_preprocessor(train)
    out = pre.fit_transform(train)
    assert out.shape[1] == len(get_output_feature_names(pre))


# ------------------------------------------------------------------- 11: feature names

def test_feature_names_are_recoverable(frames) -> None:
    """Needed downstream by SHAP, importance ranking, MLflow and inference debugging."""
    train, _, _ = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    names = get_output_feature_names(pre)
    assert "num_a" in names and "num_b" in names
    assert any(n.startswith("cat_a") for n in names)


def test_input_feature_names_are_exposed(frames) -> None:
    train, _, _ = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    assert get_input_feature_names(pre) == ["num_a", "num_b", "cat_a"]


# -------------------------------------------------------------------- 12: no mutation

def test_input_dataframe_is_not_mutated(frames) -> None:
    train, _, _ = frames
    snapshot = train.copy(deep=True)
    pre = build_preprocessor(train)
    pre.fit_transform(train)
    pd.testing.assert_frame_equal(train, snapshot)


def test_transform_does_not_mutate_input(frames) -> None:
    train, val, _ = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    snapshot = val.copy(deep=True)
    pre.transform(val)
    pd.testing.assert_frame_equal(val, snapshot)


# ------------------------------------------------------- §6: input schema alignment

def test_align_reorders_columns(frames) -> None:
    train, _, _ = frames
    expected = ["num_a", "num_b", "cat_a"]
    shuffled = train[["cat_a", "num_b", "num_a"]]
    assert list(align_to_training_schema(shuffled, expected).columns) == expected


def test_align_drops_extra_columns(frames) -> None:
    train, _, _ = frames
    extra = train.assign(unexpected=1)
    assert list(align_to_training_schema(extra, ["num_a", "num_b", "cat_a"]).columns) == [
        "num_a", "num_b", "cat_a"
    ]


def test_align_adds_missing_columns_as_nan(frames) -> None:
    """A caller short of a feature still scores; the imputer fills the gap downstream."""
    train, _, _ = frames
    expected = ["num_a", "num_b", "cat_a"]
    aligned = align_to_training_schema(train.drop(columns=["num_b"]), expected)
    assert list(aligned.columns) == expected
    assert aligned["num_b"].isna().all(), "added column must be entirely NaN"
    assert len(aligned) == len(train)


def test_align_added_column_is_imputed_by_the_pipeline(frames) -> None:
    """The added NaNs must survive transform — proving alignment and preprocessing agree."""
    train, _, test = frames
    pre = build_preprocessor(train)
    pre.fit(train)
    expected = get_input_feature_names(pre)

    degraded = align_to_training_schema(test.drop(columns=["num_b"]), expected)
    out = np.asarray(pre.transform(degraded), dtype=float)
    assert np.isfinite(out).all(), "median imputation should have filled the added column"


def test_align_warns_when_adding_missing_columns(frames, caplog) -> None:
    """§4 requires the schema change not be silent, since an added NaN is imputed to
    the training median — the model then scores a value the caller never supplied."""
    train, _, _ = frames
    with caplog.at_level("WARNING"):
        align_to_training_schema(train.drop(columns=["num_b"]), ["num_a", "num_b", "cat_a"])
    assert "num_b" in caplog.text


def test_align_handles_missing_and_extra_together(frames) -> None:
    train, _, _ = frames
    expected = ["num_a", "num_b", "cat_a"]
    messy = train.drop(columns=["num_b"]).assign(unexpected=1)[["unexpected", "cat_a", "num_a"]]
    aligned = align_to_training_schema(messy, expected)
    assert list(aligned.columns) == expected
    assert aligned["num_b"].isna().all()
    assert "unexpected" not in aligned.columns


def test_align_does_not_mutate_input(frames) -> None:
    train, _, _ = frames
    snapshot = train.copy(deep=True)
    align_to_training_schema(train, ["num_a", "num_b", "cat_a"])
    pd.testing.assert_frame_equal(train, snapshot)
