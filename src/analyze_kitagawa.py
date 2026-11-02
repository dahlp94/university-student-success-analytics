"""Decompose Arkansas's six-year graduation-rate change using Kitagawa's method."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_FILE = ROOT / "data" / "processed" / "graduation_cohorts_historical.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "kitagawa_arkansas_2014_2018.csv"

ARKANSAS_UNITID = 106397
EXPECTED_UNITIDS = (106397, 155317, 157085, 159391, 178396, 207500)
SUMMARY_FILE = ROOT / "data" / "processed" / "kitagawa_institutions_2014_2018.csv"
GROUP_FILE = ROOT / "data" / "processed" / "kitagawa_aid_group_contributions_2014_2018.csv"
AID_GROUPS = ["pell", "subsidized_no_pell", "neither"]


def decompose_kitagawa(
    df: pd.DataFrame,
    unitid: int = ARKANSAS_UNITID,
    start_year: int = 2014,
    end_year: int = 2018,
) -> tuple[dict, pd.DataFrame]:
    """Split an aggregate rate change into composition and within-group effects.

    Components are proportions, not percentage points. This is a descriptive,
    symmetric accounting decomposition and does not identify causal effects.
    """
    if start_year >= end_year:
        raise ValueError("start_year must be earlier than end_year")

    required = {
        "unitid", "cohort_year", "reporting_year", "graduation_horizon",
        "aid_group", "adjusted_cohort", "bachelor_completers",
        "graduation_rate", "cohort_share",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = df.loc[
        df["unitid"].eq(unitid)
        & df["cohort_year"].isin([start_year, end_year])
    ].copy()

    if len(selected) != 2 * len(AID_GROUPS):
        raise ValueError("Expected three financial-aid groups in each cohort")
    if selected.duplicated(["cohort_year", "aid_group"]).any():
        raise ValueError("Duplicate cohort and aid-group records")
    for year in (start_year, end_year):
        year_df = selected.loc[selected["cohort_year"].eq(year)]
        if set(year_df["aid_group"]) != set(AID_GROUPS):
            raise ValueError(f"{year}: Missing or unexpected financial-aid groups")

    if not (
        selected["graduation_horizon"].eq(6).all()
        and selected["reporting_year"].eq(selected["cohort_year"] + 6).all()
    ):
        raise ValueError("Invalid six-year cohort metadata")

    numeric = [
        "adjusted_cohort", "bachelor_completers", "graduation_rate", "cohort_share"
    ]
    if any(not pd.api.types.is_numeric_dtype(selected[c]) for c in numeric):
        raise ValueError("Expected numeric counts, rates, and shares")
    if not np.isfinite(selected[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite counts, rates, or shares")
    if (
        selected["adjusted_cohort"].le(0).any()
        or selected["bachelor_completers"].lt(0).any()
        or selected["bachelor_completers"].gt(selected["adjusted_cohort"]).any()
    ):
        raise ValueError("Invalid cohort or completer counts")
    if any((selected[c] % 1 != 0).any() for c in ["adjusted_cohort", "bachelor_completers"]):
        raise ValueError("Cohort and completer counts must be integers")

    cohort_totals = selected.groupby("cohort_year")["adjusted_cohort"].transform("sum")
    calculated_rates = selected["bachelor_completers"] / selected["adjusted_cohort"]
    calculated_shares = selected["adjusted_cohort"] / cohort_totals
    if not np.allclose(selected["graduation_rate"], calculated_rates, rtol=0, atol=1e-10):
        raise ValueError("Graduation rates do not match the counts")
    if not np.allclose(selected["cohort_share"], calculated_shares, rtol=0, atol=1e-10):
        raise ValueError("Cohort shares do not match the counts")

    start = selected.loc[selected["cohort_year"].eq(start_year)].set_index("aid_group").loc[AID_GROUPS]
    end = selected.loc[selected["cohort_year"].eq(end_year)].set_index("aid_group").loc[AID_GROUPS]

    w0 = start["adjusted_cohort"] / start["adjusted_cohort"].sum()
    w1 = end["adjusted_cohort"] / end["adjusted_cohort"].sum()
    r0 = start["bachelor_completers"] / start["adjusted_cohort"]
    r1 = end["bachelor_completers"] / end["adjusted_cohort"]

    # Kitagawa's symmetric decomposition: composition + within-group rate change.
    composition = (w1 - w0) * (r1 + r0) / 2
    within_group = (r1 - r0) * (w1 + w0) / 2

    details = pd.DataFrame({
        "aid_group": AID_GROUPS,
        "start_cohort_share": w0.to_numpy(),
        "end_cohort_share": w1.to_numpy(),
        "start_graduation_rate": r0.to_numpy(),
        "end_graduation_rate": r1.to_numpy(),
        "composition_component": composition.to_numpy(),
        "within_group_component": within_group.to_numpy(),
    })
    details["total_contribution"] = (
        details["composition_component"] + details["within_group_component"]
    )

    observed_start = float(start["bachelor_completers"].sum() / start["adjusted_cohort"].sum())
    observed_end = float(end["bachelor_completers"].sum() / end["adjusted_cohort"].sum())
    observed_change = observed_end - observed_start
    composition_total = float(composition.sum())
    within_group_total = float(within_group.sum())
    residual = observed_change - composition_total - within_group_total

    if not np.isclose(residual, 0.0, rtol=0, atol=1e-10):
        raise ValueError("Kitagawa decomposition does not reconstruct the observed change")

    summary = {
        "unitid": unitid,
        "start_year": start_year,
        "end_year": end_year,
        "observed_start": observed_start,
        "observed_end": observed_end,
        "observed_change": observed_change,
        "composition_component": composition_total,
        "within_group_component": within_group_total,
        "reconciliation_residual": residual,
    }
    return summary, details


def decompose_all_institutions(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2018,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the validated endpoint decomposition to the six study universities.

    Returned rates and contributions are proportions, not percentage points.
    """
    required = {"unitid", "institution", "cohort_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    available = set(df["unitid"].dropna().unique())
    missing_institutions = set(EXPECTED_UNITIDS) - available
    if missing_institutions:
        raise ValueError(f"Missing institutions: {sorted(missing_institutions)}")

    summaries = []
    contributions = []

    for unitid in EXPECTED_UNITIDS:
        # Check that each university has one consistent display name.
        labels = df.loc[df["unitid"].eq(unitid), "institution"].dropna().unique()
        if len(labels) != 1:
            raise ValueError(f"{unitid}: Expected one consistent institution name")

        summary, details = decompose_kitagawa(
            df, unitid=unitid, start_year=start_year, end_year=end_year
        )
        institution = str(labels[0])
        summaries.append({"institution": institution, **summary})
        contributions.append(
            details.assign(
                unitid=unitid,
                institution=institution,
                start_year=start_year,
                end_year=end_year,
            )
        )

    summary_df = pd.DataFrame(summaries).sort_values(
        "observed_change", ascending=False
    ).reset_index(drop=True)
    group_df = pd.concat(contributions, ignore_index=True)

    if len(summary_df) != 6 or len(group_df) != 18:
        raise ValueError("Unexpected decomposition output size")

    return summary_df, group_df


def main() -> None:
    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(f"Missing historical dataset: {HISTORICAL_FILE}")

    df = pd.read_csv(HISTORICAL_FILE)
    summary, details = decompose_kitagawa(df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(OUTPUT_FILE, index=False)

    print("\nARKANSAS | KITAGAWA DECOMPOSITION | 2014 vs. 2018 entering cohorts")
    print(f"2014 graduation rate:     {summary['observed_start']:.2%}")
    print(f"2018 graduation rate:     {summary['observed_end']:.2%}")
    print(f"Observed change:          {summary['observed_change'] * 100:+.3f} pp")
    print(f"Composition component:    {summary['composition_component'] * 100:+.3f} pp")
    print(f"Within-group component:   {summary['within_group_component'] * 100:+.3f} pp")
    print(f"Reconciliation residual:  {summary['reconciliation_residual']:.2e}")
    print("\nAid-group contributions (percentage points):")
    display = details[["aid_group", "composition_component", "within_group_component", "total_contribution"]].copy()
    for column in display.columns[1:]:
        display[column] = (display[column] * 100).round(3)
    print(display.to_string(index=False))
    print(f"\nSaved subgroup contributions to {OUTPUT_FILE}")
    # Compare all six institutions using the same entering-cohort endpoints.
    peer_summary, peer_details = decompose_all_institutions(df)
    peer_summary.to_csv(SUMMARY_FILE, index=False)
    peer_details.to_csv(GROUP_FILE, index=False)

    print("\nALL SIX UNIVERSITIES | 2014 vs. 2018 entering cohorts")
    display = peer_summary[
        ["institution", "observed_change", "composition_component", "within_group_component"]
    ].copy()
    display["institution"] = display["institution"].replace({
        "University of Arkansas": "Arkansas",
        "University of Kansas": "Kansas",
        "University of Kentucky": "Kentucky",
        "Louisiana State University and Agricultural & Mechanical College": "LSU",
        "University of Missouri-Columbia": "Missouri",
        "University of Oklahoma-Norman Campus": "Oklahoma",
    })
    for column in display.columns[1:]:
        display[column] = (display[column] * 100).round(3)
    print(display.to_string(index=False))
    print(f"\nSaved comparison: {SUMMARY_FILE}")
    print(f"Saved group contributions: {GROUP_FILE}")
    print("Note: Decomposition is descriptive, not causal.")


if __name__ == "__main__":
    main()
