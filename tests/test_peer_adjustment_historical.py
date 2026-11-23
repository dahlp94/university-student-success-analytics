
"""Tests for longitudinal peer adjustment using fixed 2018 weights."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.analyze_peer_adjustment import (
    ARKANSAS_UNITID,
    analyze_peer_adjustment,
)
from src.analyze_peer_adjustment_historical import (
    analyze_historical_peer_adjustment,
)

UNITIDS = (106397, 155317, 157085, 159391, 178396, 207500)
AID_GROUPS = ("pell", "subsidized_no_pell", "neither")


@pytest.fixture
def historical_df():
    """Five cohorts; subgroup rates fixed, composition changes in 2018."""
    records = []

    for year in range(2014, 2019):
        for unitid in UNITIDS:
            arkansas = unitid == ARKANSAS_UNITID

            if arkansas:
                counts = (50, 20, 30) if year < 2018 else (20, 30, 50)
                rates = (0.5, 0.7, 0.9)
            else:
                counts = (20, 30, 50) if year < 2018 else (40, 20, 40)
                rates = (0.6, 0.7, 0.8)

            for aid_group, n, rate in zip(AID_GROUPS, counts, rates):
                completers = round(n * rate)
                records.append({
                    "unitid": unitid,
                    "institution": f"University {unitid}",
                    "cohort_year": year,
                    "reporting_year": year + 6,
                    "graduation_horizon": 6,
                    "aid_group": aid_group,
                    "adjusted_cohort": n,
                    "bachelor_completers": completers,
                    "graduation_rate": completers / n,
                    "cohort_share": n / sum(counts),
                })

    return pd.DataFrame(records)


def test_fixed_weights_and_complete_results(historical_df):
    results, weights = analyze_historical_peer_adjustment(historical_df)

    assert len(results) == 30
    assert not results.duplicated(["unitid", "cohort_year"]).any()
    assert set(results["cohort_year"]) == set(range(2014, 2019))
    assert set(results["unitid"]) == set(UNITIDS)
    assert set(weights["reference_cohort_year"]) == {2018}

    actual = weights.set_index("aid_group")["fixed_peer_weight"]
    for group, expected in zip(AID_GROUPS, (0.4, 0.2, 0.4)):
        assert actual[group] == pytest.approx(expected)

    assert actual.sum() == pytest.approx(1.0)


def test_fixed_standardization_and_peer_gaps(historical_df):
    results, _ = analyze_historical_peer_adjustment(historical_df)

    arkansas = results.loc[
        results["unitid"].eq(ARKANSAS_UNITID)
    ].set_index("cohort_year")

    # Fixed subgroup rates and weights imply constant standardized rates.
    np.testing.assert_allclose(
        arkansas["standardized_rate_fixed_peer"], 0.70
    )
    np.testing.assert_allclose(
        arkansas["peer_standardized_rate_fixed"], 0.70
    )
    np.testing.assert_allclose(
        arkansas["standardized_gap_vs_peers"], 0.0, atol=1e-10
    )

    # Observed gaps change because cohort composition changes.
    assert arkansas.loc[2014, "observed_gap_vs_peers"] == pytest.approx(-0.07)
    assert arkansas.loc[2018, "observed_gap_vs_peers"] == pytest.approx(+0.06)


def test_2018_matches_existing_single_cohort_analysis(historical_df):
    historical, _ = analyze_historical_peer_adjustment(historical_df)

    existing, _ = analyze_peer_adjustment(
        historical_df.loc[historical_df["cohort_year"].eq(2018)]
    )

    actual = historical.loc[
        historical["cohort_year"].eq(2018)
    ].set_index("unitid").sort_index()

    expected = existing.set_index("unitid").sort_index()

    np.testing.assert_allclose(
        actual["observed_rate"],
        expected["observed_rate"],
    )
    np.testing.assert_allclose(
        actual["standardized_rate_fixed_peer"],
        expected["standardized_rate_peer"],
    )


def test_results_do_not_depend_on_row_order(historical_df):
    original, weights = analyze_historical_peer_adjustment(historical_df)

    shuffled, shuffled_weights = analyze_historical_peer_adjustment(
        historical_df.sample(frac=1, random_state=24)
    )

    assert_frame_equal(original, shuffled)
    assert_frame_equal(
        weights.sort_values("aid_group").reset_index(drop=True),
        shuffled_weights.sort_values("aid_group").reset_index(drop=True),
    )


def test_rejects_incomplete_cohort(historical_df):
    bad = historical_df.loc[
        ~(
            historical_df["cohort_year"].eq(2016)
            & historical_df["unitid"].eq(UNITIDS[1])
            & historical_df["aid_group"].eq("pell")
        )
    ]

    with pytest.raises(
        ValueError,
        match="Incomplete or unexpected financial-aid groups"
    ):
        analyze_historical_peer_adjustment(bad)


def test_rejects_absent_reference_year(historical_df):
    with pytest.raises(
        ValueError,
        match="Reference cohort 2020 not found"
    ):
        analyze_historical_peer_adjustment(
            historical_df,
            reference_year=2020,
        )
