"""ISDA-standard-model pricing of a single-name CDS.

All values are per unit notional and seen by the *protection buyer*
(positive = the buyer's contract is an asset).  ``CDS.pv`` applies the
notional and the side.

The key idea: between any two consecutive curve nodes both the forward rate
f and the hazard rate h are constant, so DF(t)*Q(t) is a single exponential
and every integral below has a closed form.  No time-stepping, no midpoint
rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import List, Sequence

from .curves import CreditCurve, DiscountCurve
from .dates import ONE_DAY, act360, cash_settle_date, cds_schedule, step_in_date

_TAYLOR_CUTOFF = 1e-4


@dataclass(frozen=True)
class CDS:
    maturity: date
    coupon: float              # running coupon, decimal (0.01 = 100bp)
    recovery: float = 0.40
    notional: float = 1.0
    buy_protection: bool = True


@dataclass(frozen=True)
class CDSValuation:
    protection_pv: float       # value of the contingent leg
    coupon_annuity: float      # PV of 1 unit of coupon, scheduled payments only
    default_accrual_annuity: float  # PV of 1 unit of coupon accrued to default
    accrued: float             # coupon accrued up to step-in, per unit coupon
    coupon: float
    df_settle: float

    @property
    def rpv01_dirty(self) -> float:
        return self.coupon_annuity + self.default_accrual_annuity

    @property
    def rpv01_clean(self) -> float:
        # the accrued coupon is handed back on the cash settlement date,
        # so today it is worth accrued * DF(settle)
        return self.rpv01_dirty - self.accrued * self.df_settle

    @property
    def pv_dirty(self) -> float:
        return self.protection_pv - self.coupon * self.rpv01_dirty

    @property
    def pv_clean(self) -> float:
        return self.protection_pv - self.coupon * self.rpv01_clean

    @property
    def par_spread(self) -> float:
        return self.protection_pv / self.rpv01_clean

    @property
    def upfront(self) -> float:
        """Clean amount the buyer pays on the cash settlement date.

        Equal to pv_dirty / DF(settle) + coupon * accrued.
        """
        return self.pv_clean / self.df_settle

    @property
    def accrued_amount(self) -> float:
        return self.coupon * self.accrued


def _check_base(valuation_date: date, disc: DiscountCurve, cred: CreditCurve) -> None:
    if disc.base_date != valuation_date or cred.base_date != valuation_date:
        raise ValueError("both curves must be built as of the valuation date")


def _grid(start: date, end: date, *curves) -> List[date]:
    """start, every curve node strictly inside, end."""
    inner = sorted({d for c in curves for d in c.dates if start < d < end})
    return [start, *inner, end]


def _default_pv_segment(lnP0, lnQ0, lnP1, lnQ1) -> float:
    """integral of DF(t) * (-dQ(t)) over one segment with constant f and h."""
    h = lnQ0 - lnQ1
    fh = (lnP0 - lnP1) + h
    b0 = math.exp(lnP0 + lnQ0)
    if abs(fh) < _TAYLOR_CUTOFF:   # h/(f+h)*(1-e^-(f+h)) is 0/0 as f+h -> 0
        return b0 * h * (1 - fh / 2 + fh ** 2 / 6 - fh ** 3 / 24 + fh ** 4 / 120)
    return h / fh * (b0 - math.exp(lnP1 + lnQ1))


def _accrual_segment(t0, t1, ts, lnP0, lnQ0, lnP1, lnQ1) -> float:
    """integral of (t - ts) * DF(t) * (-dQ(t)) over one segment."""
    h = lnQ0 - lnQ1
    fh = (lnP0 - lnP1) + h
    b0 = math.exp(lnP0 + lnQ0)
    if abs(fh) < _TAYLOR_CUTOFF:
        return h * b0 * ((t0 - ts) * (1 - fh / 2 + fh ** 2 / 6 - fh ** 3 / 24)
                         + (t1 - t0) * (0.5 - fh / 3 + fh ** 2 / 8 - fh ** 3 / 30))
    b1 = math.exp(lnP1 + lnQ1)
    return h / fh * ((t1 - t0) * ((b0 - b1) / fh - b1) + (t0 - ts) * (b0 - b1))


def protection_leg_pv(valuation_date: date, maturity: date, recovery: float,
                      disc: DiscountCurve, cred: CreditCurve) -> float:
    """(1-R) * integral from today to maturity of DF(t) * (-dQ(t))."""
    _check_base(valuation_date, disc, cred)
    if maturity <= valuation_date:
        return 0.0
    grid = _grid(valuation_date, maturity, disc, cred)
    total = 0.0
    for d0, d1 in zip(grid, grid[1:]):
        total += _default_pv_segment(disc.log_df(d0), cred.log_survival(d0),
                                     disc.log_df(d1), cred.log_survival(d1))
    return (1.0 - recovery) * total


def premium_leg_annuities(valuation_date: date, maturity: date,
                          disc: DiscountCurve, cred: CreditCurve,
                          *, legacy_accrual: bool = False,
                          quantlib_compat: bool = False):
    """Return (coupon annuity, accrual-on-default annuity, accrued fraction).

    ``legacy_accrual=True`` reproduces the original ISDA code (before the
    accrual-on-default fix): a half-day accrual bias and one flat segment per
    coupon period.

    ``quantlib_compat=True`` observes survival one day before the *payment*
    date in every period, as QuantLib's IsdaCdsEngine does.  The default
    observes one day before the *accrual end*, which for the final period is
    the maturity date itself (the period accrues through maturity).  The two
    differ by roughly 1e-7 in annuity terms.
    """
    _check_base(valuation_date, disc, cred)
    stepin = step_in_date(valuation_date)
    periods = cds_schedule(valuation_date, maturity)
    coupons = default_accrual = 0.0
    for p in periods:
        # 1) scheduled coupon: paid only if the name survives the whole period
        end = (p.pay_date if quantlib_compat else p.accrual_end) - ONE_DAY
        coupons += p.year_fraction * disc.df(p.pay_date) * cred.survival(end)
        # 2) coupon accrued from period start to the default date
        start = max(p.accrual_start, stepin) - ONE_DAY
        ts = disc.time(p.accrual_start - ONE_DAY) - (1 / 730 if legacy_accrual else 0.0)
        grid = [start, end] if legacy_accrual else _grid(start, end, disc, cred)
        for d0, d1 in zip(grid, grid[1:]):
            default_accrual += _accrual_segment(
                disc.time(d0), disc.time(d1), ts,
                disc.log_df(d0), cred.log_survival(d0),
                disc.log_df(d1), cred.log_survival(d1))
    default_accrual *= 365.0 / 360.0          # curve time -> Act/360 accrual
    accrued = act360(periods[0].accrual_start, stepin) if periods else 0.0
    return coupons, default_accrual, accrued


def value_cds(cds: CDS, valuation_date: date, disc: DiscountCurve, cred: CreditCurve,
              **conventions) -> CDSValuation:
    """Value one CDS.  ``conventions`` are passed to ``premium_leg_annuities``."""
    prot = protection_leg_pv(valuation_date, cds.maturity, cds.recovery, disc, cred)
    ann, aod, accrued = premium_leg_annuities(valuation_date, cds.maturity, disc, cred,
                                              **conventions)
    return CDSValuation(prot, ann, aod, accrued, cds.coupon,
                        disc.df(cash_settle_date(valuation_date)))


def signed(cds: CDS, per_unit_buyer_value: float) -> float:
    """Apply notional and side to a per-unit, buyer-view number."""
    return cds.notional * (1 if cds.buy_protection else -1) * per_unit_buyer_value
