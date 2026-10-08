"""Conversions between the three ways a standard CDS is quoted.

par spread            <-> a full credit curve
conventional spread   <-> a *flat* curve built from that single number
upfront / price       <-> cash that changes hands for a fixed-coupon contract
"""
from __future__ import annotations

from datetime import date

from scipy.optimize import brentq

from .calibration import bootstrap_credit_curve
from .curves import DiscountCurve
from .pricing import CDS, value_cds


def upfront_from_conventional_spread(cds: CDS, valuation_date: date, spread: float,
                                     disc: DiscountCurve) -> float:
    """Clean upfront (fraction of notional, paid by the buyer at T+3)."""
    flat = bootstrap_credit_curve(valuation_date, [cds.maturity], [spread], cds.recovery, disc)
    return value_cds(cds, valuation_date, disc, flat).upfront


def conventional_spread_from_upfront(cds: CDS, valuation_date: date, upfront: float,
                                     disc: DiscountCurve) -> float:
    f = lambda s: upfront_from_conventional_spread(cds, valuation_date, s, disc) - upfront
    return brentq(f, 1e-7, 10.0, xtol=1e-14)


def clean_price(upfront: float) -> float:
    """Bond-style price per 100: 100 minus points upfront."""
    return 100.0 * (1.0 - upfront)
