"""Bootstrap a piecewise-constant hazard curve from par CDS spreads."""
from __future__ import annotations

from datetime import date
from typing import Sequence

from scipy.optimize import brentq

from .curves import CreditCurve, DiscountCurve
from .pricing import CDS, value_cds


class CalibrationError(ValueError):
    """The quotes admit no arbitrage-free (non-negative hazard) curve."""


def bootstrap_credit_curve(valuation_date: date, maturities: Sequence[date],
                           par_spreads: Sequence[float], recovery: float,
                           disc: DiscountCurve, *, max_hazard: float = 50.0,
                           **conventions) -> CreditCurve:
    """Solve one hazard rate per quote, shortest maturity first.

    For quote k the unknown is the hazard on (T_{k-1}, T_k]; every earlier
    hazard is already fixed.  We look for the value that makes a CDS paying
    the quoted spread worth zero (clean) today.
    """
    if list(maturities) != sorted(set(maturities)):
        raise ValueError("maturities must be strictly increasing")
    hazards: list[float] = []
    for k, (mat, spread) in enumerate(zip(maturities, par_spreads)):
        cds = CDS(mat, spread, recovery)

        def clean_pv(h, k=k, cds=cds):
            curve = CreditCurve(valuation_date, maturities[:k + 1], [*hazards, h])
            return value_cds(cds, valuation_date, disc, curve, **conventions).pv_clean

        lo, hi = 0.0, 1.0
        if clean_pv(lo) > 0:
            raise CalibrationError(
                f"quote {k} ({mat}, {spread * 1e4:.1f}bp): implied hazard is negative; "
                "the spread is too low relative to shorter maturities")
        while clean_pv(hi) < 0:
            hi *= 2
            if hi > max_hazard:
                raise CalibrationError(
                    f"quote {k} ({mat}, {spread * 1e4:.1f}bp): no hazard rate reprices the "
                    "quote; even immediate default after the previous node is not enough "
                    "protection value for this spread")
        hazards.append(brentq(clean_pv, lo, hi, xtol=1e-14, rtol=1e-13))
    return CreditCurve(valuation_date, maturities, hazards)
