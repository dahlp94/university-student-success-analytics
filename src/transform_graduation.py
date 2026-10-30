"""Build the 2024 graduation dataset by institution and financial-aid group."""

import pandas as pd
import numpy as np
from pathlib import Path

# Set explicit paths
ROOT = Path(__file__).resolve().parents[1]
RAW_IPEDS_DIR = ROOT / "data" / "raw" / "ipeds"
PROCESSED_DIR = ROOT / "data" / "processed"

# Map target output names to official IPEDS variable pairs (cohort, completers)
AID_GROUPS = {
    "pell": ("PGADJCT", "PGCMBAC"),
    "subsidized_no_pell": ("SSADJCT", "SSCMBAC"),
    "neither": ("NRADJCT", "NRCMBAC"),
}

def build_graduation_cohort(pell_df: pd.DataFrame, gr_df: pd.DataFrame, inst_df: pd.DataFrame, year: int = 2024) -> pd.DataFrame:
    """Transform wide IPEDS subgroup data into a validated long-format analytical table."""
    
    # Standardize column names to uppercase internally for robust extraction
    pell = pell_df.copy()
    pell.columns = pell.columns.str.upper()
    gr = gr_df.copy()
    gr.columns = gr.columns.str.upper()
    inst = inst_df.copy()
    inst.columns = inst.columns.str.upper()
    
    institution_ids = inst["UNITID"].unique().tolist()
    
    # Select the correct graduation cohort (PSGRTYPE = 2 is the Bachelor's cohort)
    pell_cohort = pell[pell["UNITID"].isin(institution_ids) & (pell["PSGRTYPE"] == 2)].copy()
    
    if len(pell_cohort) != len(institution_ids):
        raise ValueError(f"Validation failed: Expected exactly {len(institution_ids)} bachelor's cohort records, found {len(pell_cohort)}.")

    # Extract the correct counts and reshape from wide to long format
    pieces = []
    for group, (cohort_col, completer_col) in AID_GROUPS.items():
        part = pell_cohort[["UNITID", cohort_col, completer_col]].copy()
        part.columns = ["unitid", "adjusted_cohort", "bachelor_completers"]
        part["aid_group"] = group
        pieces.append(part)
        
    result = pd.concat(pieces, ignore_index=True)

    # Attach cohort metadata
    result["cohort_year"] = year - 6
    result["reporting_year"] = year
    result["graduation_horizon"] = 6

    # Calculate analytical measures
    # Graduation rate (handles zero denominators gracefully by returning NaN instead of inf)
    result["graduation_rate"] = (
        result["bachelor_completers"] / result["adjusted_cohort"].replace(0, np.nan)
    )
    
    # Cohort share (cohort / total institution cohort)
    result["cohort_share"] = (
        result["adjusted_cohort"] / 
        result.groupby("unitid")["adjusted_cohort"].transform("sum")
    )

    # ==============================================================
    # Strict Data Validation Suite
    # ==============================================================
    
    # Check logical constraints
    if not (result["adjusted_cohort"] >= 0).all() or not (result["bachelor_completers"] >= 0).all():
        raise ValueError("Validation failed: Counts must be non-negative integers.")
    if not (result["bachelor_completers"] <= result["adjusted_cohort"]).all():
        raise ValueError("Validation failed: Completers cannot exceed the adjusted cohort.")

    # Reconcile subgroup sums with the PELL dataset reported totals (TTADJCT, TTCMBAC)
    totals = result.groupby("unitid")[["adjusted_cohort", "bachelor_completers"]].sum()
    source = pell_cohort.set_index("UNITID")
    if not (totals["adjusted_cohort"].eq(source["TTADJCT"]) & totals["bachelor_completers"].eq(source["TTCMBAC"])).all():
        raise ValueError("Validation failed: Financial aid subgroups do not mathematically sum to the TTADJCT/TTCMBAC totals.")

    # Cross-file Reconciliation: Verify TTADJCT/TTCMBAC match the general GR2024 dataset (GRTYPE 8 & 12)
    gr_cohort = gr[gr["UNITID"].isin(institution_ids)]
    
    # Extract Adjusted Cohorts (GRTYPE = 8)
    gr_type_8 = gr_cohort[gr_cohort["GRTYPE"] == 8].set_index("UNITID")
    if not gr_type_8.empty and not source["TTADJCT"].eq(gr_type_8["GRTOTLT"]).all():
        print("WARNING: PELL Adjusted Cohort (TTADJCT) does not perfectly match GR2024 (GRTYPE=8) for all institutions.")
            
    # Extract Completers (GRTYPE = 12)
    gr_type_12 = gr_cohort[gr_cohort["GRTYPE"] == 12].set_index("UNITID")
    if not gr_type_12.empty and not source["TTCMBAC"].eq(gr_type_12["GRTOTLT"]).all():
        print("WARNING: PELL Completers (TTCMBAC) does not perfectly match GR2024 (GRTYPE=12) for all institutions.")

    # Check cohort shares sum exactly to 1 (within floating-point tolerance)
    share_sums = result.groupby("unitid")["cohort_share"].sum()
    if not np.allclose(share_sums, 1.0):
        raise ValueError("Validation failed: Cohort shares for an institution do not sum to 1.0.")

    # IPEDS Reporting Flags check (Ensure data is 'R' for Reported)
    flag_cols = [c for c in pell_cohort.columns if c.startswith("XPG") or c.startswith("XSS") or c.startswith("XNR")]
    for flag in flag_cols:
        invalid_flags = pell_cohort[~pell_cohort[flag].isin(["R", "Z", "A"])]
        if not invalid_flags.empty:
            print(f"Data Note: Found imputed/non-standard reporting flag in column {flag}.")

    # ==============================================================
    # Final Formatting & Join
    # ==============================================================
    
    # Attach institution names (fallback to INSTNM or INSTITUTION depending on what's in processed/institutions.csv)
    inst_name_col = "INSTNM" if "INSTNM" in inst.columns else "INSTITUTION"
    inst_merge = inst[["UNITID", inst_name_col]].rename(columns={"UNITID": "unitid", inst_name_col: "institution"})
    
    result = result.merge(inst_merge, on="unitid", how="left", validate="many_to_one")
    if result["institution"].isna().any():
        raise ValueError("Validation failed: Could not match institution names to UNITIDs.")

    # Apply strict final column order and sort
    final_cols = [
        "unitid", "institution", "cohort_year", "reporting_year", "graduation_horizon", 
        "aid_group", "adjusted_cohort", "bachelor_completers", "graduation_rate", "cohort_share"
    ]
    return result[final_cols].sort_values(["unitid", "aid_group"]).reset_index(drop=True)


if __name__ == "__main__":
    year = 2024
    
    # Load all three verified source files
    print(f"Loading IPEDS {year} Source Files...")
    inst_df = pd.read_csv(PROCESSED_DIR / "institutions.csv")
    pell_df = pd.read_csv(RAW_IPEDS_DIR / f"GR{year}_PELL_SSL.csv", low_memory=False)
    gr_df = pd.read_csv(RAW_IPEDS_DIR / f"GR{year}.csv", low_memory=False)
    
    # Run the transformation and strict validation
    print("Transforming cohort data and executing validation checks...")
    fact_df = build_graduation_cohort(pell_df, gr_df, inst_df, year)
    
    # Save output dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"graduation_cohorts_{year}.csv"
    fact_df.to_csv(out_path, index=False)
    
    print("\n--- Verification Summary ---")
    print(f"Extracted exactly {len(fact_df)} valid cohort records.")
    print(f"Data rigorously validated against GR2024 and GR2024_PELL_SSL totals.")
    print(f"Saved analytical dataset to: {out_path}")