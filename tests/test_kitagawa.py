
"""Unit tests for the symmetric Kitagawa graduation-rate decomposition.

Run from the repository root: python -m pytest tests/test_kitagawa.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.analyze_kitagawa import (
    EXPECTED_UNITIDS,
    decompose_all_institutions,
    decompose_kitagawa,
)


GROUPS = ("pell", "subsidized_no_pell", "neither")


def make_cohort(unitid, year, cohorts, completers):
    """Create internally consistent, integer-count subgroup records."""
    assert len(cohorts) == len(completers) == len(GROUPS)
    total = sum(cohorts)
    return [
        {
            "unitid": unitid,
            "cohort_year": year,
            "reporting_year": year + 6,
            "graduation_horizon": 6,
            "aid_group": group,
            "adjusted_cohort": n,
            "bachelor_completers": c,
            "graduation_rate": c / n,
            "cohort_share": n / total,
        }
        for group, n, c in zip(GROUPS, cohorts, completers)
    ]


@pytest.fixture
def cohorts():
    # Arkansas in 2014: shares (1/3, 1/3, 1/3), rates (.4, .6, .8).
    # Arkansas in 2018: shares (.5, .25, .25), rates (.5, .6, .9).
    # The endpoints have observed rates .600 and .625, respectively.
    records = make_cohort(106397, 2014, [100, 100, 100], [40, 60, 80])
    records += make_cohort(106397, 2018, [200, 100, 100], [100, 60, 90])
    # Unrelated records must not influence Arkansas's endpoint comparison.
    records += make_cohort(106397, 2016, [100, 100, 100], [20, 30, 40])
    records += make_cohort(155317, 2014, [100, 100, 100], [10, 20, 30])
    records += make_cohort(155317, 2018, [100, 100, 100], [40, 50, 60])
    return pd.DataFrame(records)


def test_manual_kitagawa_calculation(cohorts):
    summary, details = decompose_kitagawa(cohorts)
    # Independent hand calculation using exact fractions.
    expected_composition = (1 / 6) * 0.45 - (1 / 12) * 0.60 - (1 / 12) * 0.85
    expected_within = (5 / 12) * 0.10 + (7 / 24) * 0.10
    assert summary["observed_start"] == pytest.approx(0.600)
    assert summary["observed_end"] == pytest.approx(0.625)
    assert summary["observed_change"] == pytest.approx(0.025)
    assert summary["composition_component"] == pytest.approx(expected_composition)
    assert summary["within_group_component"] == pytest.approx(expected_within)
    assert summary["reconciliation_residual"] == pytest.approx(0, abs=1e-12)
    assert details["aid_group"].tolist() == list(GROUPS)


def test_subgroup_contributions_match_independent_values(cohorts):
    _, details = decompose_kitagawa(cohorts)
    assert details["composition_component"].tolist() == pytest.approx([
        (1 / 6) * 0.45, -(1 / 12) * 0.6, -(1 / 12) * 0.85
    ])
    assert details["within_group_component"].tolist() == pytest.approx([
        (5 / 12) * 0.1, 0.0, (7 / 24) * 0.1
    ])
    assert np.allclose(
        details["total_contribution"],
        details["composition_component"] + details["within_group_component"],
    )


def test_only_composition_changes():
    records = make_cohort(106397, 2014, [100, 100, 100], [40, 60, 80])
    records += make_cohort(106397, 2018, [200, 100, 100], [80, 60, 80])
    summary, _ = decompose_kitagawa(pd.DataFrame(records))
    assert summary["within_group_component"] == pytest.approx(0, abs=1e-12)
    assert summary["composition_component"] == pytest.approx(summary["observed_change"])


def test_only_within_group_rates_change():
    records = make_cohort(106397, 2014, [100, 100, 100], [40, 60, 80])
    records += make_cohort(106397, 2018, [100, 100, 100], [50, 60, 90])
    summary, _ = decompose_kitagawa(pd.DataFrame(records))
    assert summary["composition_component"] == pytest.approx(0, abs=1e-12)
    assert summary["within_group_component"] == pytest.approx(summary["observed_change"])


def test_row_order_does_not_change_results(cohorts):
    base_summary, base_details = decompose_kitagawa(cohorts)
    shuffled = cohorts.sample(frac=1, random_state=42).reset_index(drop=True)
    new_summary, new_details = decompose_kitagawa(shuffled)
    for field in ("observed_change", "composition_component", "within_group_component"):
        assert new_summary[field] == pytest.approx(base_summary[field])
    pd.testing.assert_frame_equal(base_details, new_details)


def test_selects_only_requested_institution_and_years(cohorts):
    summary, _ = decompose_kitagawa(cohorts, unitid=155317, start_year=2014, end_year=2018)
    assert summary["unitid"] == 155317
    assert summary["observed_start"] == pytest.approx(0.2)
    assert summary["observed_end"] == pytest.approx(0.5)


def test_rejects_reversed_or_identical_years(cohorts):
    for start_year, end_year in [(2018, 2014), (2014, 2014)]:
        with pytest.raises(ValueError, match="start_year"):
            decompose_kitagawa(cohorts, start_year=start_year, end_year=end_year)


def test_rejects_missing_group(cohorts):
    bad = cohorts.drop(cohorts.index[(cohorts.unitid == 106397) &
                                     (cohorts.cohort_year == 2018) &
                                     (cohorts.aid_group == "pell")])
    with pytest.raises(ValueError, match="three financial-aid groups"):
        decompose_kitagawa(bad)


def test_rejects_duplicate_group(cohorts):
    row = cohorts.loc[(cohorts.unitid == 106397) &
                      (cohorts.cohort_year == 2018) &
                      (cohorts.aid_group == "pell")]
    bad = pd.concat([cohorts, row], ignore_index=True)
    with pytest.raises(ValueError, match="three financial-aid groups|Duplicate"):
        decompose_kitagawa(bad)


@pytest.mark.parametrize("column,error", [
    ("graduation_rate", "Graduation rates do not match"),
    ("cohort_share", "Cohort shares do not match"),
])
def test_rejects_inconsistent_derived_fields(cohorts, column, error):
    bad = cohorts.copy()
    bad.loc[0, column] += 0.01
    with pytest.raises(ValueError, match=error):
        decompose_kitagawa(bad)


def test_rejects_invalid_reporting_year(cohorts):
    bad = cohorts.copy()
    bad.loc[0, "reporting_year"] = 2023
    with pytest.raises(ValueError, match="six-year cohort metadata"):
        decompose_kitagawa(bad)


def test_rejects_impossible_completer_count(cohorts):
    bad = cohorts.copy()
    bad.loc[0, "bachelor_completers"] = bad.loc[0, "adjusted_cohort"] + 1
    with pytest.raises(ValueError, match="Invalid cohort or completer counts"):
        decompose_kitagawa(bad)


def test_rejects_nonfinite_rate(cohorts):
    bad = cohorts.copy()
    bad.loc[0, "graduation_rate"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        decompose_kitagawa(bad)


# ---------------------------------------------------------
# Six-university Kitagawa comparison tests
# ---------------------------------------------------------

@pytest.fixture
def six_institution_cohorts():
    """Create consistent endpoint data for all six universities."""
    records = []
    for offset, unitid in enumerate(EXPECTED_UNITIDS):
        rows = make_cohort(unitid, 2014, [100, 100, 100], [40, 60, 80])
        rows += make_cohort(unitid, 2018, [200, 100, 100], [100 + 5 * offset, 60, 90])
        records.extend({**row, "institution": f"University {unitid}"} for row in rows)
    return pd.DataFrame(records)


def test_all_institutions_have_complete_unique_outputs(six_institution_cohorts):
    summary, details = decompose_all_institutions(six_institution_cohorts)

    assert len(summary) == 6
    assert summary["unitid"].is_unique
    assert set(summary["unitid"]) == set(EXPECTED_UNITIDS)
    assert len(details) == 18
    assert not details.duplicated(["unitid", "aid_group"]).any()
    for unitid, group in details.groupby("unitid"):
        assert unitid in EXPECTED_UNITIDS
        assert set(group["aid_group"]) == set(GROUPS)
        assert set(group["start_year"]) == {2014}
        assert set(group["end_year"]) == {2018}


def test_all_institution_components_reconcile(six_institution_cohorts):
    summary, details = decompose_all_institutions(six_institution_cohorts)

    for _, university in summary.iterrows():
        group = details.loc[details["unitid"].eq(university["unitid"])]
        assert university["observed_change"] == pytest.approx(
            university["observed_end"] - university["observed_start"], abs=1e-12
        )
        assert university["observed_change"] == pytest.approx(
            university["composition_component"] + university["within_group_component"],
            abs=1e-12,
        )
        for component in ("composition_component", "within_group_component"):
            assert group[component].sum() == pytest.approx(university[component], abs=1e-12)
        assert group["total_contribution"].sum() == pytest.approx(
            university["observed_change"], abs=1e-12
        )


def test_all_institutions_rejects_missing_university(six_institution_cohorts):
    bad = six_institution_cohorts.loc[
        six_institution_cohorts["unitid"].ne(EXPECTED_UNITIDS[-1])
    ]
    with pytest.raises(ValueError, match="Missing institutions"):
        decompose_all_institutions(bad)


def test_all_institutions_rejects_inconsistent_name(six_institution_cohorts):
    bad = six_institution_cohorts.copy()
    bad.loc[0, "institution"] = "Different university name"
    with pytest.raises(ValueError, match="Expected one consistent institution name"):
        decompose_all_institutions(bad)


def test_all_institutions_rejects_missing_aid_group(six_institution_cohorts):
    bad = six_institution_cohorts.loc[
        ~((six_institution_cohorts["unitid"] == EXPECTED_UNITIDS[1])
          & (six_institution_cohorts["cohort_year"] == 2018)
          & (six_institution_cohorts["aid_group"] == "pell"))
    ]
    with pytest.raises(ValueError, match="three financial-aid groups"):
        decompose_all_institutions(bad)
