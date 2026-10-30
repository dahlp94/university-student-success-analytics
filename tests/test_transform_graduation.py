"""Tests for the graduation cohort data transformation layer."""

import pytest
import numpy as np
import pandas as pd

from src.transform_graduation import build_graduation_cohort

@pytest.fixture
def mock_ipeds_data():
    """Generate controlled test datasets for the six target institutions."""
    institutions = [
        (106397, "University of Arkansas"),
        (178396, "University of Missouri-Columbia"),
        (207500, "University of Oklahoma-Norman Campus"),
        (155317, "University of Kansas"),
        (157085, "University of Kentucky"),
        (159391, "Louisiana State University")
    ]
    
    inst_df = pd.DataFrame(institutions, columns=["UNITID", "INSTNM"])
    
    pell_rows = []
    gr_rows = []
    
    # Generate mock subgroup counts that mathematically reconcile perfectly
    for uid, _ in institutions:
        pg_cohort, pg_comp = 1000, 550  # Pell
        ss_cohort, ss_comp = 800, 500   # Subsidized, no Pell
        nr_cohort, nr_comp = 2200, 1600 # Neither
        
        tt_cohort = pg_cohort + ss_cohort + nr_cohort
        tt_comp = pg_comp + ss_comp + nr_comp
        
        # Main bachelor's cohort row
        pell_rows.append({
            "UNITID": uid, "PSGRTYPE": 2, 
            "PGADJCT": pg_cohort, "PGCMBAC": pg_comp,
            "SSADJCT": ss_cohort, "SSCMBAC": ss_comp,
            "NRADJCT": nr_cohort, "NRCMBAC": nr_comp,
            "TTADJCT": tt_cohort, "TTCMBAC": tt_comp
        })
        
        # Corresponding totals in the master GR dataset
        gr_rows.append({"UNITID": uid, "GRTYPE": 8, "GRTOTLT": tt_cohort})
        gr_rows.append({"UNITID": uid, "GRTYPE": 12, "GRTOTLT": tt_comp})
        
    pell_df = pd.DataFrame(pell_rows)
    gr_df = pd.DataFrame(gr_rows)
    
    return pell_df, gr_df, inst_df


@pytest.fixture
def transformed_df(mock_ipeds_data):
    """Execute the transformation function on the mock data."""
    pell_df, gr_df, inst_df = mock_ipeds_data
    return build_graduation_cohort(pell_df, gr_df, inst_df, year=2024)


# --- Tests ---

def test_exact_record_count(transformed_df):
    """Exactly 18 records exist (6 universities x 3 aid groups)."""
    assert len(transformed_df) == 18, f"Expected 18 records, got {len(transformed_df)}"


def test_composite_key_uniqueness(transformed_df):
    """The composite key is perfectly unique."""
    keys = transformed_df[['unitid', 'cohort_year', 'graduation_horizon', 'aid_group']]
    assert not keys.duplicated().any(), "Found duplicate composite keys in the output."


def test_metadata_assignments(transformed_df):
    """Cohort metadata correctly offset from the 2024 reporting year."""
    assert (transformed_df['cohort_year'] == 2018).all()
    assert (transformed_df['reporting_year'] == 2024).all()
    assert (transformed_df['graduation_horizon'] == 6).all()


def test_counts_are_nonnegative_integers(transformed_df):
    """Adjusted cohort and completer counts are valid nonnegative integers."""
    for col in ['adjusted_cohort', 'bachelor_completers']:
        # Ensure it's an integer type (or can be safely cast to one without data loss)
        assert pd.api.types.is_numeric_dtype(transformed_df[col])
        assert (transformed_df[col] % 1 == 0).all(), f"{col} contains fractional values."
        assert (transformed_df[col] >= 0).all(), f"{col} contains negative values."


def test_completers_logic(transformed_df):
    """Bachelor's completers never exceed the corresponding adjusted cohort."""
    assert (transformed_df['bachelor_completers'] <= transformed_df['adjusted_cohort']).all()


def test_graduation_rate_calculation(transformed_df):
    """Graduation rates exactly equal completers / cohort."""
    expected_rate = transformed_df['bachelor_completers'] / transformed_df['adjusted_cohort']
    np.testing.assert_allclose(
        transformed_df['graduation_rate'], 
        expected_rate, 
        err_msg="Graduation rate calculation failed float tolerance."
    )


def test_cohort_shares_sum_to_one(transformed_df):
    """Cohort shares sum exactly to 1.0 within each institution."""
    share_sums = transformed_df.groupby('unitid')['cohort_share'].sum()
    np.testing.assert_allclose(
        share_sums, 
        1.0, 
        err_msg="Cohort shares do not sum to 1.0 for all institutions."
    )


def test_reconstructed_overall_rates(transformed_df, mock_ipeds_data):
    """Reconstructed overall graduation rates agree with validated totals."""
    pell_df, _, _ = mock_ipeds_data
    
    # Re-aggregate the transformed data back up to the institution level
    reconstructed = transformed_df.groupby('unitid').agg(
        total_cohort=('adjusted_cohort', 'sum'),
        total_completers=('bachelor_completers', 'sum')
    )
    reconstructed['overall_rate'] = reconstructed['total_completers'] / reconstructed['total_cohort']
    
    # Calculate the expected overall rate directly from the mock source data
    source = pell_df.set_index('UNITID')
    source['expected_rate'] = source['TTCMBAC'] / source['TTADJCT']
    
    # Align indexes and compare
    reconstructed, source = reconstructed.align(source, axis=0)
    np.testing.assert_allclose(
        reconstructed['overall_rate'], 
        source['expected_rate'],
        err_msg="Reconstructed overall graduation rates do not match source totals."
    )

def test_validation_catches_bad_totals(mock_ipeds_data):
    """Deliberately corrupt the source totals to test the validation suite."""
    pell_df, gr_df, inst_df = mock_ipeds_data
    
    # Isolate the first record and deliberately break the mathematical identity
    # by inflating the reported total adjusted cohort by 1 student.
    pell_df.loc[0, 'TTADJCT'] += 1
    
    # The transformation suite must catch this discrepancy and halt execution.
    with pytest.raises(ValueError, match="do not mathematically sum"):
        build_graduation_cohort(pell_df, gr_df, inst_df, year=2024)

def test_validation_catches_completers_exceeding_cohort(mock_ipeds_data):
    """Deliberately corrupt a subgroup so completers > cohort."""
    pell_df, gr_df, inst_df = mock_ipeds_data
    
    # Make the number of Pell completers greater than the Pell adjusted cohort
    pell_df.loc[0, 'PGCMBAC'] = pell_df.loc[0, 'PGADJCT'] + 10
    
    # We must also inflate the TTCMBAC (Total Completers) by 10 so it passes the 
    # sum check, forcing the script to catch the completer > cohort violation instead.
    pell_df.loc[0, 'TTCMBAC'] += 10
    
    with pytest.raises(ValueError, match="Completers cannot exceed the adjusted cohort"):
        build_graduation_cohort(pell_df, gr_df, inst_df, year=2024)