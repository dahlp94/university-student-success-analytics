"""Unit tests for the 2024 cross-sectional peer-adjustment analysis."""

import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.analyze_peer_adjustment import (
    ARKANSAS_UNITID,
    analyze_peer_adjustment,
    calculate_weights,
    validate_peer_input,
)

# Each institution has 100 students. The order of counts is:
# pell, subsidized_no_pell, neither.
# Arkansas and Kansas have identical group-specific graduation rates
# (50%, 70%, 90%), but different group compositions.
SYNTHETIC_INSTITUTIONS = {
    106397: ("University of Arkansas", (20, 30, 50), (10, 21, 45)),
    155317: ("University of Kansas", (40, 20, 40), (20, 14, 36)),
    157085: ("University of Kentucky", (50, 20, 30), (30, 16, 24)),
    159391: ("Louisiana State University", (20, 50, 30), (12, 35, 24)),
    178396: ("University of Missouri", (60, 10, 30), (42, 7, 27)),
    207500: ("University of Oklahoma", (10, 30, 60), (6, 21, 54)),
}

AID_GROUPS = ("pell", "subsidized_no_pell", "neither")


@pytest.fixture
def cohort_df():
    """Six valid institutions x three aid groups for the 2018 cohort."""
    records = []
    for unitid, (institution, cohorts, completers) in SYNTHETIC_INSTITUTIONS.items():
        total_cohort = sum(cohorts)
        for aid_group, n, y in zip(AID_GROUPS, cohorts, completers):
            records.append(
                {
                    "unitid": unitid,
                    "institution": institution,
                    "cohort_year": 2018,
                    "reporting_year": 2024,
                    "graduation_horizon": 6,
                    "aid_group": aid_group,
                    "adjusted_cohort": n,
                    "bachelor_completers": y,
                    "graduation_rate": y / n,
                    "cohort_share": n / total_cohort,
                }
            )
    return pd.DataFrame.from_records(records)


def test_valid_synthetic_cohorts_pass_validation(cohort_df):
    assert len(cohort_df) == 18
    validate_peer_input(cohort_df)


def test_reference_weights_sum_to_one(cohort_df):
    _, weights = analyze_peer_adjustment(cohort_df)

    assert set(weights["aid_group"]) == set(AID_GROUPS)
    np.testing.assert_allclose(weights["peer_pooled_weight"].sum(), 1.0)
    np.testing.assert_allclose(weights["all_pooled_weight"].sum(), 1.0)


def test_primary_reference_excludes_arkansas(cohort_df):
    _, weights = analyze_peer_adjustment(cohort_df)
    by_group = weights.set_index("aid_group")

    # Five peers: Pell 180, subsidized/no Pell 130, neither 190 (total 500).
    expected_peer = {
        "pell": 180 / 500,
        "subsidized_no_pell": 130 / 500,
        "neither": 190 / 500,
    }
    # Including Arkansas: Pell 200, subsidized/no Pell 160, neither 240.
    expected_all = {
        "pell": 200 / 600,
        "subsidized_no_pell": 160 / 600,
        "neither": 240 / 600,
    }
    for group in AID_GROUPS:
        assert by_group.loc[group, "peer_pooled_weight"] == pytest.approx(expected_peer[group])
        assert by_group.loc[group, "all_pooled_weight"] == pytest.approx(expected_all[group])


def test_observed_rates_reconstruct_from_counts(cohort_df):
    results, _ = analyze_peer_adjustment(cohort_df)
    by_id = results.set_index("unitid")

    expected_completers = {
        106397: 76,
        155317: 70,
        157085: 70,
        159391: 71,
        178396: 76,
        207500: 81,
    }
    for unitid, expected_y in expected_completers.items():
        assert by_id.loc[unitid, "total_cohort"] == 100
        assert by_id.loc[unitid, "total_completers"] == expected_y
        assert by_id.loc[unitid, "observed_rate"] == pytest.approx(expected_y / 100)


def test_direct_standardization_matches_manual_calculation(cohort_df):
    results, _ = analyze_peer_adjustment(cohort_df)
    arkansas = results.set_index("unitid").loc[ARKANSAS_UNITID]

    # Arkansas subgroup rates: Pell=.5, subsidized/no Pell=.7, neither=.9.
    expected_peer = (180 / 500) * 0.5 + (130 / 500) * 0.7 + (190 / 500) * 0.9
    expected_all = (200 / 600) * 0.5 + (160 / 600) * 0.7 + (240 / 600) * 0.9

    assert arkansas["standardized_rate_peer"] == pytest.approx(expected_peer)
    assert arkansas["standardized_rate_all"] == pytest.approx(expected_all)
    assert arkansas["standardization_shift_peer"] == pytest.approx(expected_peer - 0.76)
    assert arkansas["standardization_shift_all"] == pytest.approx(expected_all - 0.76)


def test_every_institution_uses_same_common_weights(cohort_df):
    results, weights = analyze_peer_adjustment(cohort_df)
    by_id = results.set_index("unitid")
    weights_by_group = weights.set_index("aid_group")

    for unitid, rows in cohort_df.groupby("unitid"):
        expected_peer = sum(
            row.graduation_rate * weights_by_group.loc[row.aid_group, "peer_pooled_weight"]
            for row in rows.itertuples(index=False)
        )
        expected_all = sum(
            row.graduation_rate * weights_by_group.loc[row.aid_group, "all_pooled_weight"]
            for row in rows.itertuples(index=False)
        )
        assert by_id.loc[unitid, "standardized_rate_peer"] == pytest.approx(expected_peer)
        assert by_id.loc[unitid, "standardized_rate_all"] == pytest.approx(expected_all)


def test_same_subgroup_rates_yield_same_standardized_rate(cohort_df):
    results, _ = analyze_peer_adjustment(cohort_df)
    by_id = results.set_index("unitid")
    arkansas = by_id.loc[106397]
    kansas = by_id.loc[155317]

    assert arkansas["observed_rate"] == pytest.approx(0.76)
    assert kansas["observed_rate"] == pytest.approx(0.70)
    assert arkansas["observed_rate"] != kansas["observed_rate"]
    assert arkansas["standardized_rate_peer"] == pytest.approx(
        kansas["standardized_rate_peer"]
    )
    assert arkansas["standardized_rate_all"] == pytest.approx(
        kansas["standardized_rate_all"]
    )


def test_results_independent_of_input_row_order(cohort_df):
    original_results, original_weights = analyze_peer_adjustment(cohort_df)
    shuffled = cohort_df.sample(frac=1, random_state=2026).reset_index(drop=True)
    shuffled_results, shuffled_weights = analyze_peer_adjustment(shuffled)

    assert_frame_equal(
        original_results.sort_values("unitid").reset_index(drop=True),
        shuffled_results.sort_values("unitid").reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert_frame_equal(
        original_weights.sort_values("aid_group").reset_index(drop=True),
        shuffled_weights.sort_values("aid_group").reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_pooled_peer_benchmark_and_gaps(cohort_df, capsys):
    # Double Kansas's size while keeping its 70% graduation rate.
    # Peer average of institutional rates is 73.6%, but the pooled
    # count-weighted benchmark becomes (368 + 70) / (500 + 100) = 73.0%.
    larger_peer = cohort_df.copy()
    kansas = larger_peer["unitid"] == 155317
    larger_peer.loc[kansas, ["adjusted_cohort", "bachelor_completers"]] *= 2

    analyze_peer_adjustment(larger_peer)
    output = capsys.readouterr().out

    # New peer weights: Pell 220/600, subsidized/no Pell 150/600,
    # neither 230/600. Arkansas standardized rate is 70.333...%.
    assert re.search(r"Pooled peer graduation rate:\s+73\.00%", output)
    assert re.search(r"Observed gap:\s+\+3\.00 pp", output)
    assert re.search(r"Standardized gap:\s+-2\.67 pp", output)


@pytest.mark.parametrize("change,expected_message", [
    ("duplicate", "Duplicate institution-aid group"),
    ("missing", "Incomplete or unexpected financial-aid groups"),
    ("unknown_group", "Incomplete or unexpected financial-aid groups"),
])
def test_invalid_aid_group_structure_is_rejected(cohort_df, change, expected_message):
    invalid = cohort_df.copy()
    if change == "duplicate":
        invalid = pd.concat([invalid, invalid.iloc[[0]]], ignore_index=True)
    elif change == "missing":
        invalid = invalid.drop(index=0).reset_index(drop=True)
    elif change == "unknown_group":
        invalid.loc[0, "aid_group"] = "other"

    with pytest.raises(ValueError, match=expected_message):
        analyze_peer_adjustment(invalid)


@pytest.mark.parametrize("field,error_pattern", [
    ("graduation_rate", "Graduation rates do not match"),
    ("cohort_share", "Cohort shares do not match"),
])
def test_inconsistent_derived_fields_rejected(cohort_df, field, error_pattern):
    invalid = cohort_df.copy()
    invalid.loc[0, field] += 0.01

    with pytest.raises(ValueError, match=error_pattern):
        validate_peer_input(invalid)


def test_invalid_six_year_cohort_metadata_rejected(cohort_df):
    invalid = cohort_df.copy()
    invalid["reporting_year"] = 2023

    with pytest.raises(ValueError, match="Invalid six-year cohort metadata"):
        validate_peer_input(invalid)


def test_empty_reference_population_rejected(cohort_df):
    empty = cohort_df.iloc[0:0].copy()
    with pytest.raises(ValueError, match="Reference population is empty"):
        calculate_weights(empty, "peer_pooled")
