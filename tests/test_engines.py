from datetime import date

import pytest

from cdsengine import (CDS, ConventionalSpreadEngine, DiscountCurve, ParSpreadEngine,
                       cds_maturity)

VAL = date(2020, 12, 14)
DISC = DiscountCurve(VAL, [date(2020 + k, 12, 14) for k in (1, 2, 3, 5, 7)],
                     [.97, .94, .92, .86, .81])
TENORS = (1, 2, 3, 5, 7)
WFC = [.0050, .0077, .0094, .0125, .0133]
CDS_5Y = CDS(date(2025, 12, 20), 0.01, 0.50, notional=10_000_000)


def test_conventional_engine():
    eng = ConventionalSpreadEngine(VAL, DISC)
    r = eng.price(CDS_5Y, 0.0125)
    assert r.quote_type == "conventional"
    assert len(r.credit_curve.hazards) == 1                       # flat
    assert r.par_spread == pytest.approx(0.0125, abs=1e-12)       # reprices its own quote
    assert r.conventional_spread == 0.0125
    assert r.upfront * 1e7 == pytest.approx(110_847.78, abs=0.01)
    assert r.price == pytest.approx(100 * (1 - r.upfront))
    assert list(r.cs01_bucket_amounts) == [CDS_5Y.maturity]
    assert eng.implied_spread(CDS_5Y, upfront=r.upfront) == pytest.approx(0.0125, abs=1e-10)
    assert eng.implied_spread(CDS_5Y, price=r.price) == pytest.approx(0.0125, abs=1e-10)
    with pytest.raises(ValueError):
        eng.implied_spread(CDS_5Y)


def test_par_engine():
    eng = ParSpreadEngine.from_tenors(VAL, DISC, TENORS, WFC, 0.50)
    r = eng.price(CDS_5Y)
    assert r.quote_type == "par"
    assert len(r.credit_curve.hazards) == 5
    assert r.par_spread == pytest.approx(0.0125, abs=1e-12)
    assert r.upfront * 1e7 == pytest.approx(112_162.92, abs=0.01)
    assert r.conventional_spread * 1e4 == pytest.approx(125.30, abs=0.005)
    assert max(r.cs01_bucket_amounts, key=r.cs01_bucket_amounts.get) == cds_maturity(VAL, 60)
    assert sum(r.cs01_bucket_amounts.values()) == pytest.approx(r.cs01_amount, rel=1e-3)
    # the same curve values a seasoned contract at an unquoted maturity
    seasoned = eng.price(CDS(date(2024, 12, 20), 0.01, 0.50))
    assert WFC[2] < seasoned.par_spread < WFC[3]


def test_engines_agree_when_the_curve_is_flat():
    conv = ConventionalSpreadEngine(VAL, DISC).price(CDS_5Y, 0.0125)
    par = ParSpreadEngine(VAL, DISC, [CDS_5Y.maturity], [0.0125], 0.50).price(CDS_5Y)
    assert par.upfront == pytest.approx(conv.upfront, abs=1e-14)
    assert par.conventional_spread == pytest.approx(0.0125, abs=1e-10)


def test_side_and_cash_settlement():
    eng = ConventionalSpreadEngine(VAL, DISC)
    buy = eng.price(CDS_5Y, 0.0125)
    sell = eng.price(CDS(CDS_5Y.maturity, 0.01, 0.50, 10_000_000, buy_protection=False), 0.0125)
    assert sell.market_value_amount == pytest.approx(-buy.market_value_amount)
    assert sell.cs01_amount == pytest.approx(-buy.cs01_amount)
    assert buy.cash_settlement_amount == pytest.approx(1e7 * (buy.upfront - buy.accrued))
    assert buy.accrued == pytest.approx(0.01 * 85 / 360)       # 21 Sep -> 15 Dec step-in


def test_par_engine_rejects_inconsistent_recovery():
    eng = ParSpreadEngine.from_tenors(VAL, DISC, TENORS, WFC, 0.50)
    with pytest.raises(ValueError, match="recovery"):
        eng.price(CDS(date(2025, 12, 20), 0.01, 0.40))
