"""Stage IPEDS source files under data/raw/ipeds."""

import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

# Directory to store the raw and processed IPEDS data
RAW_IPEDS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "ipeds"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Data URLs
HD2024_URL = "https://nces.ed.gov/ipeds/complete-data-files/HD2024.zip"
HD2024_ZIP = "HD2024.zip"

EF2024A_URL = "https://nces.ed.gov/ipeds/complete-data-files/EF2024A.zip"
EF2024A_ZIP = "EF2024A.zip"

GR2024_PELL_URL = "https://nces.ed.gov/ipeds/complete-data-files/GR2024_PELL_SSL.zip"
GR2024_PELL_ZIP = "GR2024_PELL_SSL.zip"

GR2024_URL = "https://nces.ed.gov/ipeds/complete-data-files/GR2024.zip"
GR2024_ZIP = "GR2024.zip"

TARGET_INSTITUTIONS = {
    106397: "University of Arkansas",
    178396: "University of Missouri-Columbia",
    207500: "University of Oklahoma-Norman Campus",
    155317: "University of Kansas",
    157085: "University of Kentucky",
    159391: "Louisiana State University"
}

def ingest(url: str, zip_filename: str, destination: Path = RAW_IPEDS_DIR) -> Path:
    """Download an official IPEDS ZIP file."""
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / zip_filename
    
    if not zip_path.exists():
        print(f"Downloading {zip_filename}...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"Skipping download: {zip_filename} already exists.")
        
    return zip_path

def extract(zip_path: Path, destination: Path = RAW_IPEDS_DIR) -> Path:
    """Extract the CSV into the target directory."""
    print(f"Extracting {zip_path.name} to {destination}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Locate the actual CSV file inside the ZIP (ignoring case)
        csv_name = next(name for name in zip_ref.namelist() if name.lower().endswith('.csv'))
        zip_ref.extract(csv_name, path=destination)
        
    return destination / csv_name

def verify_and_filter(csv_path: Path) -> pd.DataFrame:
    """Identify institutions, verify presence, inspect columns, and filter."""
    print(f"\nReading {csv_path.name}...")
    
    # IPEDS files often use utf-8-sig encoding
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    
    # Filter dataset for the targeted UNITIDs
    filtered_df = df[df['UNITID'].isin(TARGET_INSTITUTIONS.keys())].copy()
    
    print("\n--- Verification Report ---")
    print(f"Expected: {len(TARGET_INSTITUTIONS)} | Found: {len(filtered_df)}")
    
    found_ids = filtered_df['UNITID'].tolist()
    missing_ids = set(TARGET_INSTITUTIONS.keys()) - set(found_ids)
    
    # Print the found institutions using their actual names from the CSV ('INSTNM')
    for uid in found_ids:
        actual_name = filtered_df.loc[filtered_df['UNITID'] == uid, 'INSTNM'].values[0]
        print(f"Found: {uid} - {actual_name}")
        
    for uid in missing_ids:
        print(f"Missing: {uid} - {TARGET_INSTITUTIONS[uid]}")
        
    print(f"\nColumns inspected ({len(filtered_df.columns)} total). First 10:")
    print(list(filtered_df.columns)[:10])
    
    return filtered_df


if __name__ == "__main__":
    # --- Process HD2024 (Institutional Characteristics) ---
    print("--- Processing Directory Data (HD2024) ---")
    hd_zip_path = ingest(HD2024_URL, HD2024_ZIP)
    hd_csv_path = extract(hd_zip_path)
    
    standardized_hd_csv = RAW_IPEDS_DIR / "HD2024.csv"
    if hd_csv_path != standardized_hd_csv:
        hd_csv_path.rename(standardized_hd_csv)
        hd_csv_path = standardized_hd_csv
    
    filtered_data = verify_and_filter(hd_csv_path)
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "institutions.csv"

    filtered_data.to_csv(out_path, index=False)
    print(f"\nSaved filtered dataset to: {out_path}\n")

    # --- Process EF2024A (Fall Enrollment) ---
    print("--- Processing Enrollment Data (EF2024A) ---")
    ef_zip_path = ingest(EF2024A_URL, EF2024A_ZIP)
    ef_csv_path = extract(ef_zip_path)
    
    standardized_ef_csv = RAW_IPEDS_DIR / "EF2024A.csv"
    if ef_csv_path != standardized_ef_csv:
        ef_csv_path.rename(standardized_ef_csv)
        ef_csv_path = standardized_ef_csv
        
    print(f"\nEnrollment data successfully staged at: {standardized_ef_csv}\n")

    # --- Process GR2024_PELL_SSL (Financial Aid Graduation Rates) ---
    print("--- Processing Financial Aid Graduation Data (GR2024_PELL_SSL) ---")
    gr_zip_path = ingest(GR2024_PELL_URL, GR2024_PELL_ZIP)
    gr_csv_path = extract(gr_zip_path)
    
    standardized_gr_csv = RAW_IPEDS_DIR / "GR2024_PELL_SSL.csv"
    if gr_csv_path != standardized_gr_csv:
        gr_csv_path.rename(standardized_gr_csv)
        gr_csv_path = standardized_gr_csv
        
    print(f"\nFinancial Aid Graduation data successfully staged at: {standardized_gr_csv}")

    # --- Process GR2024 (Overall Graduation Rates) ---
    print("--- Processing Overall Graduation Data (GR2024) ---")
    gr2024_zip_path = ingest(GR2024_URL, GR2024_ZIP)
    gr2024_csv_path = extract(gr2024_zip_path)
    
    standardized_gr2024_csv = RAW_IPEDS_DIR / "GR2024.csv"
    if gr2024_csv_path != standardized_gr2024_csv:
        gr2024_csv_path.rename(standardized_gr2024_csv)
        gr2024_csv_path = standardized_gr2024_csv
        
    print(f"\nOverall Graduation data successfully staged at: {standardized_gr2024_csv}")