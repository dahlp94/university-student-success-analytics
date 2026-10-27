"""Data-quality checks for processed student-success tables."""

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    """Raise if any required column is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def null_rates(frame: pd.DataFrame) -> pd.Series:
    """Return the share of null values in each column."""
    if frame.empty:
        return pd.Series(dtype="float64")
    return frame.isna().mean()
