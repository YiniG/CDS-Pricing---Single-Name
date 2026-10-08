"""Cross-check against QuantLib's IsdaCdsEngine on identical curves."""
from datetime import date

import pytest

from cdsengine import (CDS, DiscountCurve, bootstrap_credit_curve, cds_maturity,
                       cds_schedule, value_cds)

ql = pytest.importorskip("QuantLib")

VAL = date(2020, 12, 14)
SPREADS = [.005, .0077, .0094, .01095, .0125]
R, RATE = 0.4, 0.0295
MATS = [cds_maturity(VAL, 12 * k) for k in range(1, 6)]
DISC = DiscountCurve.flat(VAL, RATE)


def qd(d):
    return ql.Date(d.day, d.month, d.year)


@pytest.fixture(autouse=True)
def _eval_date():
    ql.Settings.instance().evaluationDate = qd(VAL)


def _ql_cds(maturity, spread, hazards, legacy):
    yts = ql.YieldTermStructureHandle(ql.FlatForward(qd(VAL), RATE, ql.Actual365Fixed()))
    hc = ql.HazardRateCurve([qd(VAL)] + [qd(m) for m in MATS], [hazards[0]] + list(hazards),
                            ql.Actual365Fixed())
    hc.enableExtrapolation()
    sch = ql.Schedule(qd(VAL), qd(maturity), ql.Period(3, ql.Months), ql.WeekendsOnly(),
                      ql.Following, ql.Unadjusted, ql.DateGeneration.CDS2015, False)
    cds = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, spread, sch, ql.Following,
                               ql.Actual360(), True, True, qd(VAL), ql.FaceValueClaim(),
                               ql.Actual360(True), True, qd(VAL), 3)
    E = ql.IsdaCdsEngine
    cds.setPricingEngine(E(ql.DefaultProbabilityTermStructureHandle(hc), R, yts, False, E.Taylor,
                           E.HalfDayBias if legacy else E.NoBias,
                           E.Flat if legacy else E.Piecewise))
    return cds, sch


def test_schedule_matches_quantlib():
    _, sch = _ql_cds(MATS[-1], 0.01, [0.01] * 5, False)
    ours = cds_schedule(VAL, MATS[-1])
    theirs = [date(d.year(), d.month(), d.dayOfMonth()) for d in sch]
    assert [p.accrual_start for p in ours] == theirs[:-1]
    assert ours[-1].pay_date == date(2025, 12, 22)        # 20 Dec 2025 is a Saturday


@pytest.mark.parametrize("legacy", [False, True])
def test_legs_match_quantlib_exactly_in_compat_mode(legacy):
    cred = bootstrap_credit_curve(VAL, MATS, SPREADS, R, DISC)
    for m, s in zip(MATS, SPREADS):
        q, _ = _ql_cds(m, s, cred.hazards, legacy)
        v = value_cds(CDS(m, s, R), VAL, DISC, cred, legacy_accrual=legacy, quantlib_compat=True)
        assert v.protection_pv == pytest.approx(q.defaultLegNPV(), abs=1e-14)
        assert s * v.rpv01_dirty == pytest.approx(-q.couponLegNPV(), abs=1e-14)
        assert v.par_spread == pytest.approx(q.fairSpread(), abs=1e-12)
        assert v.pv_clean == pytest.approx(q.NPV(), abs=1e-14)


def test_default_convention_is_within_a_thousandth_of_a_bp():
    """Default (accrual-end) observation vs QuantLib (payment-date) observation."""
    cred = bootstrap_credit_curve(VAL, MATS, SPREADS, R, DISC)
    for m, s in zip(MATS, SPREADS):
        q, _ = _ql_cds(m, s, cred.hazards, False)
        v = value_cds(CDS(m, s, R), VAL, DISC, cred)
        assert v.protection_pv == pytest.approx(q.defaultLegNPV(), abs=1e-14)
        assert v.rpv01_dirty == pytest.approx(-q.couponLegNPV() / s, abs=2.5e-7)
        assert v.par_spread == pytest.approx(q.fairSpread(), abs=1e-9)


def test_midpoint_engine_is_close_but_not_equal():
    """The course notebook's engine: same curves, midpoint default timing."""
    cred = bootstrap_credit_curve(VAL, MATS, SPREADS, R, DISC)
    q, _ = _ql_cds(MATS[-1], SPREADS[-1], cred.hazards, False)
    yts = ql.YieldTermStructureHandle(ql.FlatForward(qd(VAL), RATE, ql.Actual365Fixed()))
    hc = ql.HazardRateCurve([qd(VAL)] + [qd(m) for m in MATS],
                            [cred.hazards[0]] + cred.hazards, ql.Actual365Fixed())
    hc.enableExtrapolation()
    q.setPricingEngine(ql.MidPointCdsEngine(ql.DefaultProbabilityTermStructureHandle(hc), R, yts))
    v = value_cds(CDS(MATS[-1], SPREADS[-1], R), VAL, DISC, cred)
    diff = abs(q.defaultLegNPV() - v.protection_pv)
    assert 1e-9 < diff < 1e-4
