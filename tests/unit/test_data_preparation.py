"""Step 1 — Train / Validation / Test separation.

The defect these tests exist to prevent: `scripts/train.py` previously made a single
80/20 split and then used that one holdout for model comparison, champion selection,
threshold tuning AND final reporting. The test data therefore participated in model
selection, so the reported metrics were optimistically biased.

These tests pin the split contract. The leakage guards (disjointness, exhaustiveness,
determinism) are the ones that actually matter — a split that returns the right *sizes*
but overlapping *rows* would pass a naive proportions check.
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

from src.data.data_preparation import build_relational_feature_table, split_train_val_test
from src.utils.exceptions import DataValidationError


@pytest.fixture
def imbalanced_data() -> tuple[pd.DataFrame, pd.Series]:
    """~8% positive rate, mirroring Home Credit's TARGET balance."""
    rng = np.random.default_rng(0)
    n = 10_000
    X = pd.DataFrame({
        "feat_a": rng.normal(size=n),
        "feat_b": rng.normal(size=n),
        "cat": rng.choice(["x", "y", "z"], size=n),
    })
    y = pd.Series((rng.random(n) < 0.08).astype(int), name="TARGET")
    return X, y


# ------------------------------------------------------------------ proportions

def test_split_returns_six_objects(imbalanced_data) -> None:
    X, y = imbalanced_data
    result = split_train_val_test(X, y)
    assert len(result) == 6


def test_default_proportions_are_70_15_15(imbalanced_data) -> None:
    X, y = imbalanced_data
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(X, y)
    n = len(X)
    assert X_tr.shape[0] / n == pytest.approx(0.70, abs=0.01)
    assert X_va.shape[0] / n == pytest.approx(0.15, abs=0.01)
    assert X_te.shape[0] / n == pytest.approx(0.15, abs=0.01)


def test_custom_proportions_are_respected(imbalanced_data) -> None:
    X, y = imbalanced_data
    X_tr, X_va, X_te, *_ = split_train_val_test(
        X, y, validation_size=0.10, test_size=0.20
    )
    n = len(X)
    assert X_tr.shape[0] / n == pytest.approx(0.70, abs=0.01)
    assert X_va.shape[0] / n == pytest.approx(0.10, abs=0.01)
    assert X_te.shape[0] / n == pytest.approx(0.20, abs=0.01)


def test_X_and_y_stay_aligned(imbalanced_data) -> None:
    X, y = imbalanced_data
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_train_val_test(X, y)
    for X_part, y_part in ((X_tr, y_tr), (X_va, y_va), (X_te, y_te)):
        assert len(X_part) == len(y_part)
        assert list(X_part.index) == list(y_part.index)


# ---------------------------------------------------------------- leakage guards

def test_splits_are_pairwise_disjoint(imbalanced_data) -> None:
    """The core guarantee: no row may appear in more than one split."""
    X, y = imbalanced_data
    X_tr, X_va, X_te, *_ = split_train_val_test(X, y)
    tr, va, te = set(X_tr.index), set(X_va.index), set(X_te.index)
    assert tr & va == set(), "train and validation overlap"
    assert tr & te == set(), "train and test overlap"
    assert va & te == set(), "validation and test overlap"


def test_splits_are_exhaustive(imbalanced_data) -> None:
    """Every original row lands in exactly one split — nothing silently dropped."""
    X, y = imbalanced_data
    X_tr, X_va, X_te, *_ = split_train_val_test(X, y)
    recovered = set(X_tr.index) | set(X_va.index) | set(X_te.index)
    assert recovered == set(X.index)
    assert len(X_tr) + len(X_va) + len(X_te) == len(X)


def test_split_is_deterministic_for_a_fixed_seed(imbalanced_data) -> None:
    X, y = imbalanced_data
    first = split_train_val_test(X, y, random_state=42)
    second = split_train_val_test(X, y, random_state=42)
    for a, b in zip(first, second):
        assert list(a.index) == list(b.index)


def test_different_seeds_give_different_splits(imbalanced_data) -> None:
    X, y = imbalanced_data
    a = split_train_val_test(X, y, random_state=1)[0]
    b = split_train_val_test(X, y, random_state=2)[0]
    assert list(a.index) != list(b.index)


# ---------------------------------------------------------------- stratification

def test_class_balance_preserved_in_every_split(imbalanced_data) -> None:
    """TARGET is ~8% positive; an unstratified split can starve a small split of
    positives entirely, making PR-AUC meaningless."""
    X, y = imbalanced_data
    *_, y_tr, y_va, y_te = split_train_val_test(X, y)
    overall = y.mean()
    for name, part in (("train", y_tr), ("val", y_va), ("test", y_te)):
        assert part.mean() == pytest.approx(overall, abs=0.01), f"{name} class rate drifted"


def test_both_classes_present_in_every_split(imbalanced_data) -> None:
    X, y = imbalanced_data
    *_, y_tr, y_va, y_te = split_train_val_test(X, y)
    for name, part in (("train", y_tr), ("val", y_va), ("test", y_te)):
        assert set(part.unique()) == {0, 1}, f"{name} is missing a class"


def test_stratify_can_be_disabled(imbalanced_data) -> None:
    X, y = imbalanced_data
    result = split_train_val_test(X, y, stratify=False)
    assert len(result) == 6


# -------------------------------------------------------------------- validation

@pytest.mark.parametrize(
    "kwargs",
    [
        {"validation_size": 0.0},
        {"test_size": 0.0},
        {"validation_size": 1.0},
        {"test_size": -0.1},
        {"validation_size": 0.6, "test_size": 0.5},  # sum >= 1 leaves no training data
    ],
)
def test_invalid_sizes_raise_value_error(imbalanced_data, kwargs) -> None:
    X, y = imbalanced_data
    with pytest.raises(ValueError):
        split_train_val_test(X, y, **kwargs)


def test_mismatched_lengths_raise_value_error(imbalanced_data) -> None:
    X, y = imbalanced_data
    with pytest.raises(ValueError):
        split_train_val_test(X, y.iloc[:-5])


def test_empty_input_raises_value_error() -> None:
    with pytest.raises(ValueError):
        split_train_val_test(pd.DataFrame({"a": []}), pd.Series([], dtype=int))


# ------------------------- Step 5 §6/§16: one relational builder, two applicant tables

@pytest.fixture
def relational_tables() -> dict[str, pd.DataFrame]:
    """Train and test applicants over the *same* child tables, as in production.

    application_train carries TARGET; application_test does not. Everything else is
    identical, which is the whole point — one aggregation implementation, two callers.
    """
    rng = np.random.default_rng(5)
    train_ids = np.arange(1, 21)
    test_ids = np.arange(101, 116)
    all_ids = np.concatenate([train_ids, test_ids])

    def applicants(ids):
        return pd.DataFrame({
            "SK_ID_CURR": ids,
            "AMT_INCOME_TOTAL": rng.uniform(50_000, 300_000, len(ids)),
            "AMT_CREDIT": rng.uniform(100_000, 900_000, len(ids)),
            "AMT_ANNUITY": rng.uniform(5_000, 50_000, len(ids)),
        })

    child = lambda col: pd.DataFrame({                                  # noqa: E731
        "SK_ID_CURR": np.repeat(all_ids, 2),
        "SK_ID_PREV": np.arange(2 * len(all_ids)),
        col: rng.uniform(0, 1000, 2 * len(all_ids)),
    })
    return {
        "application_train": applicants(train_ids).assign(
            TARGET=rng.integers(0, 2, len(train_ids))),
        "application_test": applicants(test_ids),
        "bureau": pd.DataFrame({
            "SK_ID_CURR": np.repeat(all_ids, 2),
            "SK_ID_BUREAU": np.arange(2 * len(all_ids)),
            "AMT_CREDIT_SUM": rng.uniform(1_000, 90_000, 2 * len(all_ids)),
        }),
        "bureau_balance": pd.DataFrame({
            "SK_ID_BUREAU": np.arange(2 * len(all_ids)),
            "MONTHS_BALANCE": rng.integers(-60, 0, 2 * len(all_ids)),
        }),
        "previous_application": child("AMT_APPLICATION"),
        "pos_cash": child("CNT_INSTALMENT"),
        "credit_card": child("AMT_BALANCE"),
        "installments": child("AMT_PAYMENT"),
    }


def test_builder_defaults_to_application_train(relational_tables) -> None:
    """Step 1-4 behaviour is unchanged: the default caller keeps working untouched."""
    default = build_relational_feature_table(relational_tables)
    explicit = build_relational_feature_table(relational_tables, "application_train")
    pd.testing.assert_frame_equal(default, explicit)


def test_builder_works_for_application_train(relational_tables) -> None:
    out = build_relational_feature_table(relational_tables, "application_train")
    assert len(out) == 20
    assert "TARGET" in out.columns


def test_builder_works_for_application_test(relational_tables) -> None:
    out = build_relational_feature_table(relational_tables, "application_test")
    assert len(out) == 15


def test_inference_table_does_not_require_target(relational_tables) -> None:
    """§7: TARGET is training-only. Aggregation must never read or invent it."""
    out = build_relational_feature_table(relational_tables, "application_test")
    assert "TARGET" not in out.columns


def test_inference_table_preserves_sk_id_curr(relational_tables) -> None:
    out = build_relational_feature_table(relational_tables, "application_test")
    assert list(out["SK_ID_CURR"]) == list(relational_tables["application_test"]["SK_ID_CURR"])


def test_both_applicant_tables_produce_the_same_features(relational_tables) -> None:
    """§12 parity: identical engineered columns, TARGET aside.

    A column present in one and not the other means the model would be scored on a
    feature space it was not trained on.
    """
    train = build_relational_feature_table(relational_tables, "application_train")
    test = build_relational_feature_table(relational_tables, "application_test")
    assert set(train.columns) - {"TARGET"} == set(test.columns)


def test_builder_rejects_an_unknown_applicant_table(relational_tables) -> None:
    with pytest.raises(DataValidationError, match="application_holdout"):
        build_relational_feature_table(relational_tables, "application_holdout")


def test_builder_does_not_mutate_input_tables(relational_tables) -> None:
    snapshot = {name: df.copy(deep=True) for name, df in relational_tables.items()}
    build_relational_feature_table(relational_tables, "application_test")
    for name, df in relational_tables.items():
        pd.testing.assert_frame_equal(df, snapshot[name])
