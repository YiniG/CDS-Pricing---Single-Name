"""Bump-and-reprice risk measures (per unit notional, protection-buyer view).

Every measure re-bootstraps the credit curve from the *quotes*, because the
market observable is the spread, not the hazard rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

from .calibration import bootstrap_credit_curve
from .curves import DiscountCurve
from .pricing import CDS, value_cds

BP = 1e-4


@dataclass(frozen=True)
class RiskReport:
    pv_clean: float
    rpv01: float            # clean risky annuity
    cs01: float             # +1bp on every par spread
    cs01_buckets: List[float]
    ir01: float             # +1bp on every zero rate
    rec01: float            # +1% recovery, spreads unchanged
    jtd: float              # jump to default


def _pv(cds, val, mats, spreads, rec_curve, disc):
    curve = bootstrap_credit_curve(val, mats, spreads, rec_curve, disc)
    return value_cds(cds, val, disc, curve)


def risk_report(cds: CDS, valuation_date: date, maturities: Sequence[date],
                par_spreads: Sequence[float], disc: DiscountCurve) -> RiskReport:
    spreads = list(par_spreads)
    base = _pv(cds, valuation_date, maturities, spreads, cds.recovery, disc)
    pv0 = base.pv_clean
    cs01 = _pv(cds, valuation_date, maturities, [s + BP for s in spreads],
               cds.recovery, disc).pv_clean - pv0
    buckets = []
    for i in range(len(spreads)):
        bumped = spreads.copy()
        bumped[i] += BP
        buckets.append(_pv(cds, valuation_date, maturities, bumped,
                           cds.recovery, disc).pv_clean - pv0)
    ir01 = _pv(cds, valuation_date, maturities, spreads, cds.recovery,
               disc.shifted(BP)).pv_clean - pv0
    cds_r = CDS(cds.maturity, cds.coupon, cds.recovery + 0.01)
    rec01 = _pv(cds_r, valuation_date, maturities, spreads, cds_r.recovery, disc).pv_clean - pv0
    jtd = (1.0 - cds.recovery) - pv0
    return RiskReport(pv0, base.rpv01_clean, cs01, buckets, ir01, rec01, jtd)
