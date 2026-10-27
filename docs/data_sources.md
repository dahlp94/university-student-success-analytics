# Data Sources

## IPEDS

Integrated Postsecondary Education Data System files from the National Center for Education Statistics. Staged, unmodified, in `data/raw/ipeds/`.

Use IPEDS for national comparisons: institutional characteristics, enrollment, retention, and graduation.

## Arkansas

State or institutional files that are not part of the IPEDS download. Staged, unmodified, in `data/raw/arkansas/`.

Record the file name, extract date, and provider for each drop before it is transformed.

## Rules

- Do not edit files in `data/raw/`.
- Write cleaned tables to `data/processed/`.
- Write files intended for Tableau or other reporting tools to `data/exports/`.
- Institutions in scope are listed in `config/institutions.csv`.
