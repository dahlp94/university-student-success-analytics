
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

GR_FILE = ROOT / "data/raw/ipeds/GR2024_PELL_SSL.csv"
INST_FILE = ROOT / "data/processed/institutions.csv"

# Candidate fields from historical IPEDS documentation.
FIELDS = [
    "UNITID", "PSGRTYPE",
    "PGADJCT", "PGCMBAC",
    "SSADJCT", "SSCMBAC",
    "NRADJCT", "NRCMBAC",
    "TTADJCT", "TTCMBAC",
]


def main():
    institutions = pd.read_csv(INST_FILE)
    graduation = pd.read_csv(
        GR_FILE, dtype=str, low_memory=False
    )

    filtered = graduation[
        graduation["UNITID"].isin(
            institutions["UNITID"].astype(str)
        )
    ].copy()

    print("\n--- Dataset Overview ---")
    print("Full shape:", graduation.shape)
    print("Filtered shape:", filtered.shape)
    print("Institutions:", filtered["UNITID"].nunique())

    print("\n--- Rows by Institution and Cohort Type ---")
    print(
        filtered.groupby(["UNITID", "PSGRTYPE"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\n--- All Columns ---")
    print(graduation.columns.tolist())

    present = [c for c in FIELDS if c in filtered.columns]
    missing = [c for c in FIELDS if c not in filtered.columns]

    print("\nCandidate fields present:", present)
    print("Candidate fields missing:", missing)

    print("\n--- Arkansas: All Cohort Types ---")
    arkansas = filtered[filtered["UNITID"] == "106397"]
    print(arkansas[present].to_string(index=False))

    print("\n--- Six Universities: PSGRTYPE = 2 ---")
    bachelor = filtered[filtered["PSGRTYPE"] == "2"]
    print(bachelor[present].to_string(index=False))

    print("\n--- Graduation Reporting Flags ---")

    flag_columns = [
        "XPGADJCT", "XPGCMBAC",
        "XSSADJCT", "XSSCMBAC",
        "XNRADJCT", "XNRCMBAC",
        "XTTADJCT", "XTTCMBAC",
    ]

    print(
        bachelor[["UNITID"] + flag_columns]
        .to_string(index=False)
    )


    print("\n--- Potential Status/Flag Columns ---")
    flags = [
        c for c in graduation.columns
        if c.startswith("X")
    ]
    print(flags)


    print("\n--- Main Graduation File (GR2024) ---")

    gr_file = ROOT / "data/raw/ipeds/GR2024.csv"
    gr = pd.read_csv(gr_file, dtype=str, low_memory=False)

    gr_filtered = gr[
        gr["UNITID"].isin(institutions["UNITID"].astype(str))
    ].copy()

    print("Full shape:", gr.shape)
    print("Filtered shape:", gr_filtered.shape)
    print("Institutions:", gr_filtered["UNITID"].nunique())

    print("\n--- All Columns ---")
    print(gr.columns.tolist())

    print("\n--- GRTYPE Coverage ---")
    print(
        gr_filtered.groupby("GRTYPE")["UNITID"]
        .nunique()
        .sort_index()
        .to_string()
    )

    print("\n--- Arkansas Graduation Records ---")
    arkansas_gr = gr_filtered[
        gr_filtered["UNITID"] == "106397"
    ]

    columns = [
        c for c in ["UNITID", "GRTYPE", "GRTOTLT", "XGRTOTLT"]
        if c in gr.columns
    ]

    print(arkansas_gr[columns].to_string(index=False))


    print("\n--- Cross-File Graduation Reconciliation ---")

    # Bachelor's adjusted cohort and bachelor's completers
    # from the main graduation file.
    gr_selected = gr_filtered[
        gr_filtered["GRTYPE"].isin(["8", "12"])
    ][
        ["UNITID", "GRTYPE", "GRTOTLT", "XGRTOTLT"]
    ].copy()

    if gr_selected.duplicated(["UNITID", "GRTYPE"]).any():
        raise ValueError("Duplicate GR2024 cohort records")

    gr_counts = gr_selected.pivot(
        index="UNITID",
        columns="GRTYPE",
        values="GRTOTLT"
    )

    gr_flags = gr_selected.pivot(
        index="UNITID",
        columns="GRTYPE",
        values="XGRTOTLT"
    )

    # Corresponding totals from financial-aid graduation data.
    pell_totals = bachelor.set_index("UNITID")[
        ["TTADJCT", "TTCMBAC"]
    ].copy()

    if set(gr_counts.index) != set(pell_totals.index):
        raise ValueError("Institution coverage mismatch")

    if set(gr_counts.columns) != {"8", "12"}:
        raise ValueError("Missing required GRTYPE codes")

    comparison = pd.DataFrame({
        "GR_adjusted": pd.to_numeric(gr_counts["8"]),
        "PELL_adjusted": pd.to_numeric(pell_totals["TTADJCT"]),
        "GR_completers": pd.to_numeric(gr_counts["12"]),
        "PELL_completers": pd.to_numeric(pell_totals["TTCMBAC"]),
    })

    comparison["cohort_match"] = (
        comparison["GR_adjusted"]
        == comparison["PELL_adjusted"]
    )

    comparison["completers_match"] = (
        comparison["GR_completers"]
        == comparison["PELL_completers"]
    )

    print(comparison.to_string())

    print("\n--- Main GR Reporting Flags ---")
    print(gr_flags.to_string())

    print("\n--- Validation Summary ---")
    print("Cohort matches:", comparison["cohort_match"].sum(), "/ 6")
    print(
        "Completer matches:",
        comparison["completers_match"].sum(),
        "/ 6"
    )

    if not comparison[
        ["cohort_match", "completers_match"]
    ].all().all():
        raise ValueError("Cross-file reconciliation failed")

    print("\nCross-file count reconciliation passed.")


if __name__ == "__main__":
    main()
