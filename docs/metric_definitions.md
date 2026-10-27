# Metric Definitions

Published definitions. Code in `src/transform.py` and queries under `sql/reporting/` should match this page.

## First-year retention

Share of an entering cohort that is still enrolled at the same institution the following fall.

`students_returned / cohort_size`

The cohort size must be greater than zero. Students who were not in the entering cohort are excluded.

## Graduation rate

Share of an entering cohort that graduated from the same institution within the measured window (commonly 150 percent of normal time).

`graduates / cohort_size`

The cohort size must be greater than zero. State the window (for example, six-year) next to every published figure.
