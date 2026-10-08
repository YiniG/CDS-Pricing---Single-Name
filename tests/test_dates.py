from datetime import date

import pytest

from cdsengine import cash_settle_date, cds_maturity, cds_schedule, step_in_date


@pytest.mark.parametrize("trade, months, expected", [
    (date(2020, 12, 14), 60, date(2025, 12, 20)),
    (date(2026, 3, 19), 60, date(2030, 12, 20)),   # day before the roll
    (date(2026, 3, 20), 60, date(2031, 6, 20)),    # roll day
    (date(2026, 9, 21), 12, date(2027, 12, 20)),
])
def test_standard_maturity(trade, months, expected):
    assert cds_maturity(trade, months) == expected


def test_maturity_matches_quantlib():
    ql = pytest.importorskip("QuantLib")
    d = date(2019, 1, 1)
    while d < date(2022, 1, 1):
        for y in (1, 5, 10):
            q = ql.cdsMaturity(ql.Date(d.day, d.month, d.year), ql.Period(y, ql.Years),
                               ql.DateGeneration.CDS2015)
            assert cds_maturity(d, 12 * y) == date(q.year(), q.month(), q.dayOfMonth())
        d = date.fromordinal(d.toordinal() + 5)


def test_primer_trade_lifecycle_dates():
    """Markit primer: CDX.NA.HY.S35 bought 10 Nov 2020, unwound 8 Feb 2021."""
    first = cds_schedule(date(2020, 11, 10), date(2025, 12, 20))[0]
    assert first.accrual_start == date(2020, 9, 21)     # 20 Sep 2020 was a Sunday
    assert first.pay_date == date(2020, 12, 21)
    assert (first.pay_date - first.accrual_start).days == 91
    assert 10_000_000 * 0.05 * first.year_fraction == pytest.approx(126_388.89, abs=0.005)
    # the primer counts trade date - accrual start; the standard model adds the step-in day
    assert (date(2020, 11, 10) - first.accrual_start).days == 50
    assert (step_in_date(date(2020, 11, 10)) - first.accrual_start).days == 51
    second = cds_schedule(date(2021, 2, 8), date(2025, 12, 20))[0]
    assert second.accrual_start == date(2020, 12, 21)
    assert (date(2021, 2, 8) - second.accrual_start).days == 49


def test_final_period_accrues_through_maturity():
    periods = cds_schedule(date(2020, 12, 14), date(2021, 12, 20))
    assert len(periods) == 5
    assert periods[-1].accrual_end == date(2021, 12, 21)
    assert sum(p.year_fraction for p in periods) * 360 == (date(2021, 12, 21) - date(2020, 9, 21)).days


def test_cash_settle_skips_weekend():
    assert cash_settle_date(date(2020, 12, 10)) == date(2020, 12, 15)
