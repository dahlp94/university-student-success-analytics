
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ENROLLMENT_FILE = ROOT / "data/raw/ipeds/EF2024A.csv"
INSTITUTIONS_FILE = ROOT / "data/processed/institutions.csv"


def main():
    institutions = pd.read_csv(INSTITUTIONS_FILE)
    enrollment = pd.read_csv(ENROLLMENT_FILE, low_memory=False)

    unitids = set(institutions["UNITID"])
    filtered = enrollment[enrollment["UNITID"].isin(unitids)]

    print("\n--- Dataset Overview ---")
    print(f"Full dataset shape: {enrollment.shape}")
    print(f"Filtered dataset shape: {filtered.shape}")
    print(f"Institutions found: {filtered['UNITID'].nunique()}")

    assert set(filtered["UNITID"]) == unitids

    print("\n--- Rows per Institution ---")
    counts = filtered.groupby("UNITID").size()
    print(counts.to_string())

    print("\n--- Available Columns ---")
    for column in enrollment.columns:
        print(column)

    print("\n--- Reporting Category Codes ---")
    for column in ["EFALEVEL", "LSTUDY", "EFTYPE"]:
        if column in filtered.columns:
            print(f"\n{column}:")
            print(filtered[column].value_counts(dropna=False).sort_index())

    print("\n--- Sample Records ---")
    print(filtered.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
