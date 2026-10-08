"""Date conventions for standard (SNAC / ISDA standard model) CDS contracts.

Everything here is deliberately small and explicit: CDS pricing errors are
far more often date errors than maths errors.

Two day-count conventions are used and must never be mixed up:

* Act/360  -> coupon accrual (what the protection buyer actually pays)
* Act/365F -> the time axis of the curves (how we measure t in exp(-lambda t))
"""
from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

ONE_DAY = timedelta(days=1)
IMM_MONTHS = (3, 6, 9, 12)


def act360(d0: date, d1: date) -> float:
    """Accrual fraction between two dates: actual days / 360."""
    return (d1 - d0).days / 360.0


def act365(d0: date, d1: date) -> float:
    """Curve time between two dates: actual days / 365 (fixed)."""
    return (d1 - d0).days / 365.0


def is_business_day(d: date) -> bool:
    """Weekends-only calendar, as used by the ISDA standard model schedule."""
    return d.weekday() < 5


def adjust_following(d: date) -> date:
    """Roll a non-business day forward to the next business day."""
    while not is_business_day(d):
        d += ONE_DAY
    return d


def add_business_days(d: date, n: int) -> date:
    while n > 0:
        d += ONE_DAY
        if is_business_day(d):
            n -= 1
    return d


def add_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    m += 1
    return date(y, m, min(d.day, _cal.monthrange(y, m)[1]))


def previous_imm20(d: date) -> date:
    """Latest 20 Mar / Jun / Sep / Dec on or before ``d``."""
    c = date(d.year, 12, 20)
    while c > d:
        c = add_months(c, -3)
    return c


def step_in_date(valuation_date: date) -> date:
    """Protection and accrual are effective from T+1 calendar day."""
    return valuation_date + ONE_DAY


def cash_settle_date(valuation_date: date) -> date:
    """The upfront amount changes hands at T+3 business days."""
    return add_business_days(valuation_date, 3)


def cds_maturity(trade_date: date, tenor_months: int) -> date:
    """Standard maturity date for a tenor (post-2015 semi-annual roll).

    The on-the-run contract rolls on 20 March and 20 September, so every
    standard maturity is a 20 June or a 20 December.
    """
    anchor = previous_imm20(trade_date)
    if anchor.month in (6, 12):
        anchor = add_months(anchor, -3)
    return add_months(anchor, tenor_months + 3)


@dataclass(frozen=True)
class CouponPeriod:
    accrual_start: date   # first day that accrues
    accrual_end: date     # first day that does NOT accrue
    pay_date: date
    year_fraction: float  # Act/360


def cds_schedule(valuation_date: date, maturity: date) -> List[CouponPeriod]:
    """Remaining coupon periods of a standard CDS seen from ``valuation_date``.

    * unadjusted dates sit on the 20th of Mar/Jun/Sep/Dec;
    * each is rolled Following, except the maturity which stays unadjusted;
    * the final period accrues one extra day, because protection covers the
      maturity date itself;
    * the first period starts on the coupon date on or before the step-in
      date, so the buyer always owes a *full* first coupon.
    """
    stepin = step_in_date(valuation_date)
    if maturity < stepin:
        return []
    first = previous_imm20(stepin)
    if adjust_following(first) > stepin:      # 20th fell on a weekend
        first = add_months(first, -3)
    unadj = [first]
    while unadj[-1] < maturity:
        unadj.append(add_months(unadj[-1], 3))
    if unadj[-1] != maturity:
        raise ValueError(f"{maturity} is not a standard quarterly CDS date")
    adj = [adjust_following(d) for d in unadj]
    periods = []
    for i in range(len(unadj) - 1):
        last = i == len(unadj) - 2
        end = maturity + ONE_DAY if last else adj[i + 1]
        periods.append(CouponPeriod(adj[i], end, adj[i + 1], act360(adj[i], end)))
    return periods
