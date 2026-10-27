"""Stage IPEDS source files under data/raw/ipeds."""

from pathlib import Path

RAW_IPEDS_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "ipeds"


def ingest(destination: Path = RAW_IPEDS_DIR) -> Path:
    """Create the IPEDS raw directory and return its path."""
    destination.mkdir(parents=True, exist_ok=True)
    return destination


if __name__ == "__main__":
    print(ingest())
