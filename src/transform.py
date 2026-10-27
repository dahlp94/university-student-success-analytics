"""Clean raw IPEDS and Arkansas inputs into processed tables."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def transform(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with stripped, snake_case column names."""
    cleaned = frame.copy()
    cleaned.columns = [
        str(column).strip().lower().replace(" ", "_") for column in cleaned.columns
    ]
    return cleaned


def retention_rate(students_returned: int, cohort_size: int) -> float:
    """Share of a cohort that returned for the next year."""
    if cohort_size <= 0:
        raise ValueError("cohort_size must be greater than zero")
    return students_returned / cohort_size


def graduation_rate(graduates: int, cohort_size: int) -> float:
    """Share of a cohort that graduated within the measured window."""
    if cohort_size <= 0:
        raise ValueError("cohort_size must be greater than zero")
    return graduates / cohort_size


def write_processed(frame: pd.DataFrame, name: str, destination: Path = PROCESSED_DIR) -> Path:
    """Write a cleaned frame to data/processed and return the file path."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path
