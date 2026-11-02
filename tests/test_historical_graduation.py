"""Tests for the 2020–2024 historical IPEDS cohort transformation."""

import pandas as pd
import pytest

from src import transform_graduation as tg


UNITIDS = [106397, 155317, 157085, 159391, 178396, 207500]


@pytest.fixture
def historical_sources(tmp_path, monkeypatch):
    """Create five years of synthetic, internally consistent IPEDS files."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()

    institutions = pd.DataFrame({
        "UNITID": UNITIDS,
        "INSTNM": [f"Synthetic University {i}" for i in range(6)],
    })
    institutions.to_csv(processed / "institutions.csv", index=False)

    for year in range(2020, 2025):
        pell_rows = []
        gr_rows = []
        for i, unitid in enumerate(UNITIDS):
            # Vary completers across years while keeping denominators valid.
            pg, ss, nr = 30 + i, 20 + i, 50 + i
            pg_done = 12 + (year - 2020) + i
            ss_done = 10 + (year - 2020) + i
            nr_done = 35 + (year - 2020) + i
            cohort_total = pg + ss + nr
            completer_total = pg_done + ss_done + nr_done
            row = {
                "UNITID": unitid,
                "PSGRTYPE": 2,
                "PGADJCT": pg,
                "PGCMBAC": pg_done,
                "SSADJCT": ss,
                "SSCMBAC": ss_done,
                "NRADJCT": nr,
                "NRCMBAC": nr_done,
                "TTADJCT": cohort_total,
                "TTCMBAC": completer_total,
            }
            for col in [
                "PGADJCT", "PGCMBAC", "SSADJCT", "SSCMBAC",
                "NRADJCT", "NRCMBAC", "TTADJCT", "TTCMBAC",
            ]:
                row[f"X{col}"] = "R"
            pell_rows.append(row)
            gr_rows.extend([
                {"UNITID": unitid, "GRTYPE": 8, "GRTOTLT": cohort_total},
                {"UNITID": unitid, "GRTYPE": 12, "GRTOTLT": completer_total},
            ])

        pd.DataFrame(pell_rows).to_csv(
            raw / f"GR{year}_PELL_SSL.csv", index=False
        )
        pd.DataFrame(gr_rows).to_csv(raw / f"GR{year}.csv", index=False)

    monkeypatch.setattr(tg, "RAW_IPEDS_DIR", raw)
    monkeypatch.setattr(tg, "PROCESSED_DIR", processed)
    return raw, processed


def test_historical_transformation_generates_90_valid_records(historical_sources):
    _, processed = historical_sources

    actual = tg.build_historical_graduation()
    saved = pd.read_csv(processed / "graduation_cohorts_historical.csv")

    assert actual.shape == (90, 10)
    pd.testing.assert_frame_equal(actual, saved, check_dtype=False)
    assert set(actual["cohort_year"]) == set(range(2014, 2019))
    assert set(actual["reporting_year"]) == set(range(2020, 2025))
    assert set(actual["graduation_horizon"]) == {6}
    assert actual.groupby("cohort_year").size().eq(18).all()

    key = ["unitid", "cohort_year", "graduation_horizon", "aid_group"]
    assert not actual.duplicated(key).any()
    assert actual.groupby(["unitid", "cohort_year"])["cohort_share"].sum().sub(1).abs().lt(1e-10).all()
    assert (
        actual["graduation_rate"]
        - actual["bachelor_completers"] / actual["adjusted_cohort"]
    ).abs().lt(1e-10).all()


def test_historical_transformation_rejects_inconsistent_totals(historical_sources):
    raw, processed = historical_sources
    path = raw / "GR2022_PELL_SSL.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "PGCMBAC"] += 1  # Published total is now inconsistent.
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="do not mathematically sum"):
        tg.build_historical_graduation()

    assert not (processed / "graduation_cohorts_historical.csv").exists()


def test_historical_transformation_rejects_missing_institution(historical_sources):
    raw, processed = historical_sources
    path = raw / "GR2021_PELL_SSL.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["UNITID"] != UNITIDS[-1]]
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Expected exactly 6 bachelor's cohort records"):
        tg.build_historical_graduation()

    assert not (processed / "graduation_cohorts_historical.csv").exists()
