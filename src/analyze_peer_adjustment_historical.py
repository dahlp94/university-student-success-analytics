
"""Track graduation rates using a fixed five-peer financial-aid distribution."""

from pathlib import Path

import pandas as pd

from src.analyze_peer_adjustment import (
    ARKANSAS_UNITID,
    calculate_weights,
    validate_peer_input,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
HISTORICAL_FILE = PROCESSED_DIR / "graduation_cohorts_historical.csv"
RESULTS_FILE = PROCESSED_DIR / "peer_adjustment_historical.csv"
WEIGHTS_FILE = PROCESSED_DIR / "peer_reference_weights_fixed_2018.csv"


def analyze_historical_peer_adjustment(
    df: pd.DataFrame, reference_year: int = 2018
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize every cohort using one pooled five-peer reference mix.

    Rates and differences are proportions, not percentage points.
    """
    if df.empty or df["cohort_year"].isna().any():
        raise ValueError("Historical data is empty or has missing cohort years")

    # Reuse the tested single-cohort validation for each entering cohort.
    for _, cohort in df.groupby("cohort_year"):
        validate_peer_input(cohort)

    reference = df.loc[
        df["cohort_year"].eq(reference_year)
        & df["unitid"].ne(ARKANSAS_UNITID)
    ]
    if reference.empty:
        raise ValueError(f"Reference cohort {reference_year} not found")

    weights = calculate_weights(reference, "fixed_peer")
    weights["reference_cohort_year"] = reference_year

    working = df.merge(
        weights[["aid_group", "fixed_peer_weight"]],
        on="aid_group", how="left", validate="many_to_one"
    )
    working["standardized_contribution"] = (
        working["graduation_rate"] * working["fixed_peer_weight"]
    )

    # Institution-level observed and standardized rates.
    results = working.groupby(
        ["cohort_year", "unitid", "institution"], as_index=False
    ).agg(
        total_cohort=("adjusted_cohort", "sum"),
        total_completers=("bachelor_completers", "sum"),
        standardized_rate_fixed_peer=("standardized_contribution", "sum"),
    )
    results["observed_rate"] = (
        results["total_completers"] / results["total_cohort"]
    )
    results["standardization_shift"] = (
        results["standardized_rate_fixed_peer"] - results["observed_rate"]
    )

    # Each year's peer benchmark pools the five peers, excluding Arkansas.
    peers = working.loc[working["unitid"].ne(ARKANSAS_UNITID)]
    benchmark = peers.groupby("cohort_year", as_index=False).agg(
        peer_cohort=("adjusted_cohort", "sum"),
        peer_completers=("bachelor_completers", "sum"),
    )
    benchmark["peer_observed_rate"] = (
        benchmark["peer_completers"] / benchmark["peer_cohort"]
    )

    # Standardize pooled peer subgroup rates to the same fixed mix.
    peer_groups = peers.groupby(
        ["cohort_year", "aid_group"], as_index=False
    ).agg(
        group_cohort=("adjusted_cohort", "sum"),
        group_completers=("bachelor_completers", "sum"),
        weight=("fixed_peer_weight", "first"),
    )
    peer_groups["contribution"] = (
        peer_groups["group_completers"] / peer_groups["group_cohort"]
        * peer_groups["weight"]
    )
    peer_standardized = peer_groups.groupby(
        "cohort_year", as_index=False
    ).agg(
        peer_standardized_rate_fixed=("contribution", "sum")
    )
    benchmark = benchmark.merge(
        peer_standardized, on="cohort_year", validate="one_to_one"
    )

    results = results.merge(
        benchmark[
            ["cohort_year", "peer_observed_rate", "peer_standardized_rate_fixed"]
        ],
        on="cohort_year", validate="many_to_one"
    )
    results["observed_gap_vs_peers"] = (
        results["observed_rate"] - results["peer_observed_rate"]
    )
    results["standardized_gap_vs_peers"] = (
        results["standardized_rate_fixed_peer"]
        - results["peer_standardized_rate_fixed"]
    )

    return (
        results.sort_values(["cohort_year", "unitid"]).reset_index(drop=True),
        weights,
    )


def main() -> None:
    df = pd.read_csv(HISTORICAL_FILE)
    results, weights = analyze_historical_peer_adjustment(df)

    results.to_csv(RESULTS_FILE, index=False)
    weights.to_csv(WEIGHTS_FILE, index=False)

    print(f"Saved historical comparison: {RESULTS_FILE}")
    print(f"Saved fixed peer weights:    {WEIGHTS_FILE}")

    print("\nARKANSAS VS. FIVE POOLED PEERS (percentage-point gaps)")
    arkansas = results.loc[results["unitid"].eq(ARKANSAS_UNITID)]

    for row in arkansas.itertuples(index=False):
        print(
            f"{row.cohort_year}: observed "
            f"{100 * row.observed_gap_vs_peers:+.2f} pp; "
            f"standardized "
            f"{100 * row.standardized_gap_vs_peers:+.2f} pp"
        )

    print("Note: Descriptive comparisons, not causal estimates.")


if __name__ == "__main__":
    main()
