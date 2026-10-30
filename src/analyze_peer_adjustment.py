"""Analyze cross-sectional graduation rates using direct standardization."""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

COHORT_FILE = PROCESSED_DIR / "graduation_cohorts_2024.csv"
RESULTS_FILE = PROCESSED_DIR / "peer_adjustment_2024.csv"
WEIGHTS_FILE = PROCESSED_DIR / "peer_reference_weights_2024.csv"

ARKANSAS_UNITID = 106397

EXPECTED_GROUPS = {
    "pell",
    "subsidized_no_pell",
    "neither",
}

# --------------------------------------------------
# Input Validation
# --------------------------------------------------

def validate_peer_input(df: pd.DataFrame) -> None:
    """Validate the single-cohort analytical dataset."""

    required = {
        "unitid",
        "institution",
        "cohort_year",
        "reporting_year",
        "graduation_horizon",
        "aid_group",
        "adjusted_cohort",
        "bachelor_completers",
        "graduation_rate",
        "cohort_share",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing values in required analytical fields")

    # This analysis currently covers six institutions.
    if df["unitid"].nunique() != 6:
        raise ValueError("Expected exactly six institutions")

    if ARKANSAS_UNITID not in set(df["unitid"]):
        raise ValueError("University of Arkansas is missing")

    # One entering cohort and one graduation horizon.
    metadata = df[
        ["cohort_year", "reporting_year", "graduation_horizon"]
    ].drop_duplicates()

    if len(metadata) != 1:
        raise ValueError("Expected a single cohort and graduation horizon")

    cohort_year, reporting_year, horizon = metadata.iloc[0]

    if reporting_year - cohort_year != 6 or horizon != 6:
        raise ValueError("Invalid six-year cohort metadata")

    # One row per institution and aid group.
    if df.duplicated(["unitid", "aid_group"]).any():
        raise ValueError("Duplicate institution-aid group records")

    if df.groupby("unitid")["institution"].nunique().gt(1).any():
        raise ValueError("Inconsistent institution names")

    group_sets = df.groupby("unitid")["aid_group"].agg(set)

    if any(groups != EXPECTED_GROUPS for groups in group_sets):
        raise ValueError("Incomplete or unexpected financial-aid groups")

    if len(df) != 18:
        raise ValueError("Expected exactly 18 analytical records")

    # Numeric validation.
    numeric_columns = [
        "adjusted_cohort",
        "bachelor_completers",
        "graduation_rate",
        "cohort_share",
    ]

    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric")

        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} contains nonfinite values")

    for col in ["adjusted_cohort", "bachelor_completers"]:
        if (df[col] % 1 != 0).any():
            raise ValueError(f"{col} contains fractional counts")

    # Positive subgroup denominators are required for
    # directly standardized subgroup rates.
    if (df["adjusted_cohort"] <= 0).any():
        raise ValueError("Adjusted cohort must be positive")

    if (
        (df["bachelor_completers"] < 0)
        | (df["bachelor_completers"] > df["adjusted_cohort"])
    ).any():
        raise ValueError("Invalid bachelor completer counts")

    # Verify derived graduation rates.
    expected_rates = (
        df["bachelor_completers"] / df["adjusted_cohort"]
    )

    if not np.allclose(
        df["graduation_rate"],
        expected_rates,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Graduation rates do not match source counts")

    # Verify cohort shares.
    institution_totals = df.groupby("unitid")[
        "adjusted_cohort"
    ].transform("sum")

    expected_shares = df["adjusted_cohort"] / institution_totals

    if not np.allclose(
        df["cohort_share"],
        expected_shares,
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Cohort shares do not match source counts")

    share_sums = df.groupby("unitid")["cohort_share"].sum()

    if not np.allclose(share_sums, 1.0, rtol=0, atol=1e-10):
        raise ValueError("Cohort shares do not sum to one")


# --------------------------------------------------
# Reference Population Weights
# --------------------------------------------------

def calculate_weights(
    df: pd.DataFrame,
    population_name: str,
) -> pd.DataFrame:
    """Calculate pooled financial-aid composition."""

    pooled = (
        df.groupby("aid_group", as_index=False)["adjusted_cohort"]
        .sum()
    )

    total_cohort = pooled["adjusted_cohort"].sum()

    if total_cohort <= 0:
        raise ValueError("Reference population is empty")

    pooled["weight"] = pooled["adjusted_cohort"] / total_cohort

    weight_col = f"{population_name}_weight"

    pooled = pooled.rename(columns={"weight": weight_col})

    return pooled[["aid_group", weight_col]]

# --------------------------------------------------
# Peer Adjustment Analysis
# --------------------------------------------------
def analyze_peer_adjustment(df: pd.DataFrame):
    """Calculate observed and directly standardized graduation rates."""

    validate_peer_input(df)

    # 1. Observed institutional graduation rates.
    observed = (
        df.groupby(["unitid", "institution"], as_index=False)
        .agg(
            total_cohort=("adjusted_cohort", "sum"),
            total_completers=("bachelor_completers", "sum"),
        )
    )

    observed["observed_rate"] = (
        observed["total_completers"] / observed["total_cohort"]
    )

    # 2. Reference populations.
    peers_only = df[df["unitid"] != ARKANSAS_UNITID].copy()

    peer_weights = calculate_weights(peers_only, "peer_pooled")
    all_weights = calculate_weights(df, "all_pooled")

    weights_df = peer_weights.merge(
        all_weights,
        on="aid_group",
        how="inner",
        validate="one_to_one",
    )

    # 3. Direct standardization.
    group_rates = df[
        ["unitid", "institution", "aid_group", "graduation_rate"]
    ].copy()

    analysis = group_rates.merge(
        weights_df,
        on="aid_group",
        how="left",
        validate="many_to_one",
    )

    if analysis[
        ["peer_pooled_weight", "all_pooled_weight"]
    ].isna().any().any():
        raise ValueError("Missing reference weights after merge")

    analysis["contribution_peer"] = (
        analysis["graduation_rate"] * analysis["peer_pooled_weight"]
    )

    analysis["contribution_all"] = (
        analysis["graduation_rate"] * analysis["all_pooled_weight"]
    )

    standardized = (
        analysis.groupby(["unitid", "institution"], as_index=False)
        .agg(
            standardized_rate_peer=("contribution_peer", "sum"),
            standardized_rate_all=("contribution_all", "sum"),
        )
    )

    # 4. Combine observed and standardized results.
    results = observed.merge(
        standardized,
        on=["unitid", "institution"],
        how="inner",
        validate="one_to_one",
    )

    results["standardization_shift_peer"] = (
        results["standardized_rate_peer"] - results["observed_rate"]
    )

    results["standardization_shift_all"] = (
        results["standardized_rate_all"] - results["observed_rate"]
    )

    # 5. Institutional rankings.
    results["rank_obs"] = (
        results["observed_rate"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results["rank_peer"] = (
        results["standardized_rate_peer"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results["rank_all"] = (
        results["standardized_rate_all"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    results = results.sort_values(
        "standardized_rate_peer", ascending=False
    ).reset_index(drop=True)

    # 6. Pooled peer benchmark.
    peer_total_cohort = peers_only["adjusted_cohort"].sum()
    peer_total_completers = peers_only["bachelor_completers"].sum()

    pooled_peer_rate = peer_total_completers / peer_total_cohort

    arkansas = results.loc[
        results["unitid"] == ARKANSAS_UNITID
    ].iloc[0]

    observed_gap = arkansas["observed_rate"] - pooled_peer_rate

    standardized_gap = (
        arkansas["standardized_rate_peer"] - pooled_peer_rate
    )
    
    # --------------------------------------------------
    # Console Report
    # --------------------------------------------------
    print("\n" + "=" * 85)
    print("CROSS-SECTIONAL PEER ADJUSTMENT ANALYSIS")
    print("=" * 85)

    print("\n[1] REFERENCE POPULATION WEIGHTS")
    print("-" * 70)

    print(
        f"{'Financial Aid Group':<22}"
        f"{'Peer Pooled':>18}"
        f"{'All 6 Pooled':>18}"
    )

    for _, row in weights_df.iterrows():
        print(
            f"{row['aid_group']:<22}"
            f"{row['peer_pooled_weight'] * 100:>17.2f}%"
            f"{row['all_pooled_weight'] * 100:>17.2f}%"
        )

    print("\n[2] OBSERVED VS. PEER-STANDARDIZED RATES")
    print("-" * 90)

    print(
        f"{'Institution':<40}"
        f"{'Observed':>12}"
        f"{'Standardized':>15}"
        f"{'Shift (pp)':>13}"
    )

    for _, row in results.iterrows():
        inst = row["institution"][:38]

        print(
            f"{inst:<40}"
            f"{row['observed_rate'] * 100:>11.2f}%"
            f"{row['standardized_rate_peer'] * 100:>14.2f}%"
            f"{row['standardization_shift_peer'] * 100:>+13.2f}"
        )

    print("\n[3] ARKANSAS VS. POOLED PEER BENCHMARK")
    print("-" * 70)

    print(f"Arkansas observed rate:        {arkansas['observed_rate']:.2%}")
    print(f"Arkansas standardized rate:    {arkansas['standardized_rate_peer']:.2%}")
    print(f"Pooled peer graduation rate:   {pooled_peer_rate:.2%}")

    print(f"\nObserved gap:                 {observed_gap * 100:+.2f} pp")
    print(f"Standardized gap:             {standardized_gap * 100:+.2f} pp")

    print("\n[4] RANKING SENSITIVITY")
    print("-" * 85)

    print(
        f"{'Institution':<40}"
        f"{'Observed':>12}"
        f"{'Peer Std':>12}"
        f"{'All Std':>12}"
    )

    for _, row in results.sort_values("rank_obs").iterrows():
        print(
            f"{row['institution'][:38]:<40}"
            f"{row['rank_obs']:>12}"
            f"{row['rank_peer']:>12}"
            f"{row['rank_all']:>12}"
        )

    print("\n" + "=" * 85)
    print("METHODOLOGICAL NOTE:")
    print(
        "Direct standardization compares institutions under a common "
        "financial-aid-group composition, holding their subgroup-specific "
        "graduation rates fixed."
    )
    print(
        "These are descriptive comparisons, not causal estimates of "
        "institutional effectiveness or value-added."
    )
    print("=" * 85 + "\n")

    return results, weights_df


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

def main():
    if not COHORT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {COHORT_FILE}. "
            "Run src/transform_graduation.py first."
        )

    df = pd.read_csv(COHORT_FILE)

    results, weights = analyze_peer_adjustment(df)

    results.to_csv(RESULTS_FILE, index=False)
    weights.to_csv(WEIGHTS_FILE, index=False)

    print(f"Saved analytical results: {RESULTS_FILE}")
    print(f"Saved reference weights:  {WEIGHTS_FILE}")

if __name__ == "__main__":
    main()
