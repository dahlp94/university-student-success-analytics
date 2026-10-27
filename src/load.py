"""Load processed tables into data/exports for reporting tools."""

from pathlib import Path

import pandas as pd

EXPORTS_DIR = Path(__file__).resolve().parents[1] / "data" / "exports"


def load(frame: pd.DataFrame, name: str, destination: Path = EXPORTS_DIR) -> Path:
    """Write a processed frame to CSV and return the file path."""
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path
