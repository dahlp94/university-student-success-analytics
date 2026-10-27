# University Student Success Analytics

Pipeline and reporting workspace for institutional student-success metrics, built from IPEDS and Arkansas source files.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Pipeline

1. `src/ingest_ipeds.py` stages IPEDS extracts in `data/raw/ipeds/`.
2. `src/transform.py` cleans raw inputs into `data/processed/`.
3. `src/validate.py` checks required columns and null rates.
4. `src/load.py` writes processed tables to `data/exports/`.

Arkansas source files belong in `data/raw/arkansas/`. Institution identifiers used by the pipeline are listed in `config/institutions.csv`.

## Layout

| Path | Purpose |
| --- | --- |
| `config/` | Institution list and other project configuration |
| `data/raw/` | Untouched source extracts |
| `data/processed/` | Cleaned tables |
| `data/exports/` | Files handed to reporting tools |
| `sql/` | Schema, transformations, reporting queries, and quality checks |
| `tests/` | Unit tests for transforms and metric definitions |
| `dashboards/tableau/` | Tableau workbooks |
| `reports/` | Institutional brief and data-quality write-ups |
| `docs/` | Source notes, metric definitions, and the data dictionary |

## Tests

```bash
pytest
```
