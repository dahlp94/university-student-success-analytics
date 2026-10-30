
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data/processed/institutions.csv"

EXPECTED_UNITIDS = {
    106397,  # Arkansas
    155317,  # Kansas
    157085,  # Kentucky
    159391,  # LSU
    178396,  # Missouri
    207500,  # Oklahoma
}


def test_processed_file_exists():
    assert OUTPUT_PATH.exists()


def test_institutions():
    df = pd.read_csv(OUTPUT_PATH)

    assert len(df) == 6
    assert "UNITID" in df.columns
    assert "INSTNM" in df.columns

    assert df["UNITID"].is_unique
    assert df["UNITID"].notna().all()
    assert df["INSTNM"].notna().all()

    assert set(df["UNITID"]) == EXPECTED_UNITIDS
