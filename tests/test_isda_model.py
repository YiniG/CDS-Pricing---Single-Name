"""Closed forms vs brute-force integration, and internal consistency."""
import math
from datetime import date, timedelta

import pytest
from scipy.integrate import quad

from cdsengine import (CDS, CalibrationError, CreditCurve, DiscountCurve, add_months,
                       bootstrap_credit_curve, cds_maturity, cds_schedule,
                       clean_price, conventional_spread_from_upfront,
                       premium_leg_annuities, protection_leg_pv, risk_report,
                       step_in_date, upfront_from_conventional_spread, value_cds)

VAL = date(2020, 12, 14)
TENORS = (1, 2, 3, 5, 7)
MATS = [cds_maturity(VAL, 12 * k) for k in TENORS]
DISC = DiscountCurve(VAL, [add_months(VAL, 12 * k) for k in TENORS], [.97, .94, .92, .86, .81])
WFC = [.0050, .0077, .0094, .0125, .0133]
CCMO = [.0751, .1164, .1874, .4156, .6083]


def _continuous(curve, t, log_fn):
    """Evaluate a date-based curve at a real-valued time (years, Act/365F)."""
    days = t * 365.0
    d0 = VAL + timedelta(days=math.floor(days))
    w = days - math.floor(days)
    return (1 - w) * log_fn(d0) + w * log_fn(d0 + timedelta(days=1))


def test_protection_leg_matches_numerical_integration():
    cred = bootstrap_credit_curve(VAL, MATS, WFC, 0.5, DISC)
    T = DISC.time(MATS[3])
    lnP = lambda t: _continuous(DISC, t, DISC.log_df)
    lnQ = lambda t: _continuous(cred, t, cred.log_survival)
    eps = 1e-6
    hazard = lambda t: -(lnQ(t + eps) - lnQ(t - eps)) / (2 * eps)
    breaks = sorted({DISC.time(d) for d in DISC.dates + cred.dates if 0 < DISC.time(d) < T})
    num, _ = quad(lambda t: math.exp(lnP(t) + lnQ(t)) * hazard(t), 0, T,
                  points=breaks, limit=500, epsabs=1e-13)
    assert protection_leg_pv(VAL, MATS[3], 0.5, DISC, cred) == pytest.approx(0.5 * num, abs=2e-8)


def test_accrual_on_default_matches_daily_sum():
    cred = bootstrap_credit_curve(VAL, MATS, WFC, 0.5, DISC)
    _, aod, _ = premium_leg_annuities(VAL, MATS[3], DISC, cred)
    stepin, total = step_in_date(VAL), 0.0
    for p in cds_schedule(VAL, MATS[3]):
        d = max(p.accrual_start, stepin)
        while d < p.accrual_end:               # default "during day d"
            prev = d - timedelta(days=1)
            pd_day = cred.survival(prev) - cred.survival(d)
            mid_accrual = ((d - p.accrual_start).days + 0.5) / 360
            total += mid_accrual * pd_day * math.sqrt(DISC.df(prev) * DISC.df(d))
            d += timedelta(days=1)
    assert aod == pytest.approx(total, rel=1e-6)


def test_taylor_branch_is_continuous():
    disc = DiscountCurve.flat(VAL, 1e-7)
    for h in (0.0, 1e-9, 1e-5):
        cred = CreditCurve(VAL, [MATS[0]], [h])
        t = disc.time(MATS[0])
        exact = 0.6 * h / (h + 1e-7) * (1 - math.exp(-(h + 1e-7) * t)) if h else 0.0
        assert protection_leg_pv(VAL, MATS[0], 0.4, disc, cred) == pytest.approx(exact, abs=1e-15)


def test_bootstrap_reprices_every_quote():
    cred = bootstrap_credit_curve(VAL, MATS, WFC, 0.5, DISC)
    for m, s in zip(MATS, WFC):
        v = value_cds(CDS(m, s, 0.5), VAL, DISC, cred)
        assert v.par_spread == pytest.approx(s, abs=1e-12)
        assert v.pv_clean == pytest.approx(0.0, abs=1e-12)
    assert all(h > 0 for h in cred.hazards)


def test_ccmo_quotes_are_rejected():
    with pytest.raises(CalibrationError, match="4156"):
        bootstrap_credit_curve(VAL, MATS, CCMO, 0.10, DISC)


def test_negative_forward_hazard_is_rejected():
    with pytest.raises(CalibrationError, match="negative"):
        bootstrap_credit_curve(VAL, MATS[:2], [0.05, 0.001], 0.4, DISC)


def test_clean_dirty_and_upfront_identities():
    cred = bootstrap_credit_curve(VAL, MATS, WFC, 0.5, DISC)
    v = value_cds(CDS(MATS[3], 0.01, 0.5), VAL, DISC, cred)
    assert v.accrued == pytest.approx((step_in_date(VAL) - date(2020, 9, 21)).days / 360)
    assert v.pv_clean - v.pv_dirty == pytest.approx(0.01 * v.accrued * v.df_settle)
    assert v.upfront == pytest.approx(v.pv_dirty / v.df_settle + 0.01 * v.accrued)
    assert v.pv_clean == pytest.approx((v.par_spread - 0.01) * v.rpv01_clean)
    assert v.upfront == pytest.approx(v.pv_clean / DISC.df(date(2020, 12, 17)))


def test_conventional_spread_round_trip():
    cds = CDS(MATS[3], 0.01, 0.4)
    for s in (0.002, 0.01, 0.05, 0.30):
        u = upfront_from_conventional_spread(cds, VAL, s, DISC)
        assert conventional_spread_from_upfront(cds, VAL, u, DISC) == pytest.approx(s, abs=1e-10)
    assert upfront_from_conventional_spread(cds, VAL, 0.01, DISC) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("val, spread, quoted_price", [
    (date(2020, 11, 10), 0.033280, 107.61),
    (date(2021, 2, 8), 0.028337, 109.60),
])
def test_primer_price_spread_pairs(val, spread, quoted_price):
    """Markit primer CDX.NA.HY.S35 quotes.  The primer does not give the swap
    curve, so a flat 0.4% rate is assumed; agreement within 2 cents per 100
    is a sanity check on conventions, not a precision test."""
    cds = CDS(date(2025, 12, 20), 0.05, 0.30)
    u = upfront_from_conventional_spread(cds, val, spread, DiscountCurve.flat(val, 0.004))
    assert clean_price(u) == pytest.approx(quoted_price, abs=0.02)


def test_risk_report_signs_and_consistency():
    cds = CDS(MATS[3], 0.01, 0.5)
    r = risk_report(cds, VAL, MATS, WFC, DISC)
    assert r.cs01 == pytest.approx(r.rpv01 * 1e-4, rel=0.03)      # CS01 ~ RPV01 x 1bp
    assert sum(r.cs01_buckets) == pytest.approx(r.cs01, rel=1e-3)
    assert max(r.cs01_buckets) == r.cs01_buckets[3]               # risk sits at the 5Y point
    assert r.jtd == pytest.approx(0.5 - r.pv_clean)
    assert r.ir01 < 0 < r.cs01
