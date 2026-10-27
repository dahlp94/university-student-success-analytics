import pytest

from src.transform import graduation_rate, retention_rate


def test_retention_rate():
    assert retention_rate(80, 100) == 0.8


def test_graduation_rate():
    assert graduation_rate(55, 100) == 0.55


def test_rates_reject_empty_cohort():
    with pytest.raises(ValueError):
        retention_rate(0, 0)
    with pytest.raises(ValueError):
        graduation_rate(1, 0)
