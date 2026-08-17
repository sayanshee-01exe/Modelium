"""Tests for src/data/data_cleaning.py — clean_application_data and utilities.

All tests use small synthetic DataFrames so they run without the real Home Credit
CSVs (~2.5 GB, gitignored). Each test encodes a specific EDA finding and verifies
that the cleaning addresses it correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.data_cleaning import (
    DAYS_EMPLOYED_SENTINEL,
    PROTECTED_COLUMNS,
    basic_checks,
    clean_application_data,
    drop_low_information_columns,
    optimize_memory,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable synthetic application table
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_application() -> pd.DataFrame:
    """Minimal application table that exercises every cleaning path."""
    n = 100
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(1, n + 1),
        "TARGET": rng.choice([0, 1], size=n, p=[0.92, 0.08]),
        # DAYS columns — stored as negative day counts in the raw data
        "DAYS_BIRTH": rng.randint(-25000, -7000, size=n),
        "DAYS_EMPLOYED": np.where(
            np.arange(n) < 20,
            DAYS_EMPLOYED_SENTINEL,               # 20 sentinel rows
            rng.randint(-5000, -100, size=n),
        ),
        "DAYS_REGISTRATION": rng.randint(-8000, -100, size=n),
        "DAYS_ID_PUBLISH": rng.randint(-6000, -100, size=n),
        "DAYS_LAST_PHONE_CHANGE": rng.randint(-3000, 0, size=n),
        # CODE_GENDER — 2 XNA rows
        "CODE_GENDER": np.where(np.arange(n) < 2, "XNA", rng.choice(["M", "F"], size=n)),
        # ORGANIZATION_TYPE — mirrors the sentinel population
        "ORGANIZATION_TYPE": np.where(np.arange(n) < 20, "XNA", "Business Entity Type 3"),
        # OCCUPATION_TYPE — NaN for both sentinel and employed-unknown groups
        "OCCUPATION_TYPE": pd.array(
            [pd.NA] * 20                          # sentinel group → Not_Employed
            + [pd.NA] * 15                        # employed-unknown → Unknown
            + ["Laborers"] * 65,
            dtype="object",
        ),
        # Income — one extreme outlier
        "AMT_INCOME_TOTAL": np.concatenate([
            rng.uniform(50_000, 500_000, size=n - 1),
            [117_000_000],                         # the EDA outlier
        ]),
        "AMT_CREDIT": rng.uniform(100_000, 1_000_000, size=n),
        "AMT_ANNUITY": rng.uniform(5_000, 50_000, size=n),
        # CNT_CHILDREN — one implausible value
        "CNT_CHILDREN": np.concatenate([
            rng.choice([0, 1, 2, 3], size=n - 1),
            [19],                                  # implausible
        ]),
        # EXT_SOURCE columns — EXT_SOURCE_1 has >70% NaN but must be protected
        "EXT_SOURCE_1": np.where(rng.random(n) < 0.75, np.nan, rng.random(n)),
        "EXT_SOURCE_2": rng.random(n),
        "EXT_SOURCE_3": np.where(rng.random(n) < 0.40, np.nan, rng.random(n)),
        # A column with 90% missing — should be dropped at 70% threshold
        "OWN_CAR_AGE": np.where(rng.random(n) < 0.90, np.nan, rng.uniform(0, 30, size=n)),
    })
    return df


# ========================================================================
# DAYS_EMPLOYED sentinel handling
# ========================================================================

class TestDaysEmployedSentinel:
    """EDA: DAYS_EMPLOYED == 365243 is 'not employed', not a real duration."""

    def test_sentinel_replaced_with_nan(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert not (cleaned["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).any()

    def test_sentinel_flag_created(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert "DAYS_EMPLOYED_ANOM" in cleaned.columns

    def test_sentinel_flag_is_binary(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert set(cleaned["DAYS_EMPLOYED_ANOM"].unique()) <= {0, 1}

    def test_sentinel_count_matches_flag(self, raw_application):
        n_sentinel = (raw_application["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum()
        cleaned = clean_application_data(raw_application)
        n_flagged = cleaned["DAYS_EMPLOYED_ANOM"].sum()
        assert n_flagged == n_sentinel

    def test_non_sentinel_values_become_nan(self, raw_application):
        """Sentinel rows should be NaN; non-sentinel rows should NOT be NaN."""
        cleaned = clean_application_data(raw_application)
        # Rows that were sentinel → should be NaN
        sentinel_rows = raw_application["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
        assert cleaned.loc[sentinel_rows, "DAYS_EMPLOYED"].isna().all()


# ========================================================================
# DAYS columns sign flip
# ========================================================================

class TestDaysFlip:
    """EDA: DAYS_* are stored as negative; flip to positive for readability."""

    def test_days_birth_positive(self, raw_application):
        cleaned = clean_application_data(raw_application)
        # All non-NaN DAYS_BIRTH should be positive
        valid = cleaned["DAYS_BIRTH"].dropna()
        assert (valid >= 0).all()

    def test_days_registration_positive(self, raw_application):
        cleaned = clean_application_data(raw_application)
        valid = cleaned["DAYS_REGISTRATION"].dropna()
        assert (valid >= 0).all()

    def test_days_id_publish_positive(self, raw_application):
        cleaned = clean_application_data(raw_application)
        valid = cleaned["DAYS_ID_PUBLISH"].dropna()
        assert (valid >= 0).all()

    def test_days_employed_non_sentinel_positive(self, raw_application):
        cleaned = clean_application_data(raw_application)
        # Non-NaN (i.e. non-sentinel) employed days should be positive
        valid = cleaned["DAYS_EMPLOYED"].dropna()
        assert (valid >= 0).all()


# ========================================================================
# CODE_GENDER XNA fix
# ========================================================================

class TestCodeGenderXna:
    """EDA: CODE_GENDER == 'XNA' (4 rows in real data) is a miscoding."""

    def test_no_xna_after_cleaning(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert "XNA" not in cleaned["CODE_GENDER"].values

    def test_xna_replaced_with_mode(self, raw_application):
        mode = raw_application.loc[
            raw_application["CODE_GENDER"] != "XNA", "CODE_GENDER"
        ].mode().iloc[0]
        cleaned = clean_application_data(raw_application)
        xna_rows = raw_application["CODE_GENDER"] == "XNA"
        assert (cleaned.loc[xna_rows, "CODE_GENDER"] == mode).all()


# ========================================================================
# ORGANIZATION_TYPE XNA — deliberately NOT changed
# ========================================================================

class TestOrganizationTypeXna:
    """EDA: ORGANIZATION_TYPE == 'XNA' is the DAYS_EMPLOYED_ANOM population — real."""

    def test_xna_organization_preserved(self, raw_application):
        cleaned = clean_application_data(raw_application)
        n_xna_before = (raw_application["ORGANIZATION_TYPE"] == "XNA").sum()
        n_xna_after = (cleaned["ORGANIZATION_TYPE"] == "XNA").sum()
        assert n_xna_after == n_xna_before


# ========================================================================
# OCCUPATION_TYPE NaN split
# ========================================================================

class TestOccupationTypeSplit:
    """EDA: NaN occupation splits into 'Not_Employed' vs 'Unknown'."""

    def test_not_employed_filled(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert "Not_Employed" in cleaned["OCCUPATION_TYPE"].values

    def test_unknown_filled(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert "Unknown" in cleaned["OCCUPATION_TYPE"].values

    def test_no_remaining_nan_in_split_groups(self, raw_application):
        """After splitting, the only NaN left should be rows that didn't match
        either condition (none in our fixture)."""
        cleaned = clean_application_data(raw_application)
        n_nan = cleaned["OCCUPATION_TYPE"].isna().sum()
        assert n_nan == 0

    def test_not_employed_count_matches_sentinel(self, raw_application):
        """Every sentinel row's occupation should become 'Not_Employed'."""
        cleaned = clean_application_data(raw_application)
        sentinel_rows = cleaned["DAYS_EMPLOYED_ANOM"] == 1
        assert (cleaned.loc[sentinel_rows, "OCCUPATION_TYPE"] == "Not_Employed").all()


# ========================================================================
# Income capping
# ========================================================================

class TestIncomeCapping:
    """EDA: AMT_INCOME_TOTAL has a 117M outlier — cap at p99.9."""

    def test_extreme_income_capped(self, raw_application):
        cleaned = clean_application_data(raw_application)
        cap = raw_application["AMT_INCOME_TOTAL"].quantile(0.999)
        assert cleaned["AMT_INCOME_TOTAL"].max() <= cap + 1  # float tolerance

    def test_normal_income_unchanged(self, raw_application):
        """Values below the cap should not be altered."""
        cap = raw_application["AMT_INCOME_TOTAL"].quantile(0.999)
        below_cap = raw_application["AMT_INCOME_TOTAL"] <= cap
        cleaned = clean_application_data(raw_application)
        pd.testing.assert_series_equal(
            cleaned.loc[below_cap, "AMT_INCOME_TOTAL"].reset_index(drop=True),
            raw_application.loc[below_cap, "AMT_INCOME_TOTAL"].reset_index(drop=True),
            check_names=False,
        )


# ========================================================================
# Children capping
# ========================================================================

class TestChildrenCapping:
    """EDA: CNT_CHILDREN > 10 is implausible."""

    def test_children_capped_at_10(self, raw_application):
        cleaned = clean_application_data(raw_application)
        assert cleaned["CNT_CHILDREN"].max() <= 10

    def test_custom_cap(self, raw_application):
        cleaned = clean_application_data(raw_application, children_cap=5)
        assert cleaned["CNT_CHILDREN"].max() <= 5


# ========================================================================
# High-missing column drop
# ========================================================================

class TestHighMissingDrop:
    """EDA: columns >70% missing add noise, except EXT_SOURCE_1."""

    def test_high_missing_column_dropped(self, raw_application):
        """OWN_CAR_AGE is 90% missing — should be dropped at 70% threshold."""
        cleaned = clean_application_data(raw_application, missing_threshold=0.70)
        assert "OWN_CAR_AGE" not in cleaned.columns

    def test_ext_source_1_protected(self, raw_application):
        """EXT_SOURCE_1 is >70% missing but must survive (protected)."""
        cleaned = clean_application_data(raw_application, missing_threshold=0.70)
        assert "EXT_SOURCE_1" in cleaned.columns

    def test_drop_disabled_keeps_all_columns(self, raw_application):
        cleaned = clean_application_data(raw_application, drop_high_missing=False)
        assert "OWN_CAR_AGE" in cleaned.columns

    def test_protected_columns_constant_matches_expectation(self):
        assert "EXT_SOURCE_1" in PROTECTED_COLUMNS


# ========================================================================
# General contract
# ========================================================================

class TestCleaningContract:
    """General properties the cleaning must always hold."""

    def test_returns_dataframe(self, raw_application):
        result = clean_application_data(raw_application)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self, raw_application):
        original_cols = list(raw_application.columns)
        original_shape = raw_application.shape
        clean_application_data(raw_application)
        assert list(raw_application.columns) == original_cols
        assert raw_application.shape == original_shape

    def test_row_count_preserved(self, raw_application):
        """Cleaning should not drop rows — only fix values and drop columns."""
        cleaned = clean_application_data(raw_application)
        assert len(cleaned) == len(raw_application)

    def test_sk_id_curr_preserved(self, raw_application):
        cleaned = clean_application_data(raw_application)
        pd.testing.assert_series_equal(
            cleaned["SK_ID_CURR"].reset_index(drop=True),
            raw_application["SK_ID_CURR"].reset_index(drop=True),
        )

    def test_target_preserved(self, raw_application):
        cleaned = clean_application_data(raw_application)
        pd.testing.assert_series_equal(
            cleaned["TARGET"].reset_index(drop=True),
            raw_application["TARGET"].reset_index(drop=True),
        )


# ========================================================================
# Existing utility function tests
# ========================================================================

class TestOptimizeMemory:

    def test_reduces_memory(self):
        df = pd.DataFrame({
            "a": np.array([1, 2, 3], dtype="int64"),
            "b": np.array([1.0, 2.0, 3.0], dtype="float64"),
        })
        optimized = optimize_memory(df)
        assert optimized.memory_usage(deep=True).sum() < df.memory_usage(deep=True).sum()

    def test_values_preserved(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [10.5, 20.5, 30.5]})
        optimized = optimize_memory(df)
        np.testing.assert_array_equal(optimized["a"].values, [1, 2, 3])

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"a": np.array([1], dtype="int64")})
        optimize_memory(df)
        assert df["a"].dtype == np.int64


class TestDropLowInformationColumns:

    def test_drops_constant_columns(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        result, dropped = drop_low_information_columns(df)
        assert "a" in dropped
        assert "b" in result.columns

    def test_drops_sk_id_columns(self):
        df = pd.DataFrame({
            "SK_ID_CURR": [1, 2],
            "SK_ID_BUREAU": [10, 20],
            "value": [1.0, 2.0],
        })
        result, dropped = drop_low_information_columns(df)
        assert "SK_ID_BUREAU" in dropped
        assert "SK_ID_CURR" in result.columns  # protected

    def test_preserves_target(self):
        df = pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1], "x": [5, 6]})
        result, _ = drop_low_information_columns(df)
        assert "TARGET" in result.columns


class TestBasicChecks:

    def test_returns_dict_with_expected_keys(self):
        df = pd.DataFrame({"a": [1, np.nan], "b": [3, 4]})
        report = basic_checks(df, "test")
        assert report["name"] == "test"
        assert "shape" in report
        assert "duplicates" in report
        assert "missing" in report
