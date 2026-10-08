"""Two pricing engines, one per kind of spread quote.

``ConventionalSpreadEngine``  one quoted spread for one contract
                              -> flat hazard curve -> upfront, price, risk
``ParSpreadEngine``           par spreads at several tenors for one name
                              -> bootstrapped curve -> any contract, bucketed risk

They differ only in how the credit curve is built.  Everything after that
(legs, accrued, upfront, risk) is shared, which is why both return the same
``PricingResult``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Sequence

from .calibration import bootstrap_credit_curve
from .curves import CreditCurve, DiscountCurve
from .dates import cash_settle_date, cds_maturity
from .pricing import CDS, signed, value_cds
from .quotes import clean_price, conventional_spread_from_upfront
from .risk import risk_report


@dataclass(frozen=True)
class PricingResult:
    """Per-unit numbers are in the protection buyer's view; ``*_amount``
    fields apply the contract's notional and side (positive = the holder of
    the position pays / owns value, as described on each field)."""
    quote_type: str                 # "conventional" or "par"
    credit_curve: CreditCurve
    par_spread: float               # fair spread of this contract on the curve
    conventional_spread: float      # flat-curve spread equivalent to the upfront
    upfront: float                  # clean, fraction of notional, buyer pays
    price: float                    # 100 * (1 - upfront)
    accrued: float                  # coupon accrued to step-in, fraction of notional
    protection_pv: float
    rpv01: float                    # clean risky annuity
    pv_clean: float
    pv_dirty: float
    settle_date: date
    market_value_amount: float      # clean value of the position to its holder
    cash_settlement_amount: float   # paid by the holder at settle to enter (negative = received)
    cs01_amount: float              # +1bp on every quote used to build the curve
    cs01_bucket_amounts: Dict[date, float]
    ir01_amount: float
    rec01_amount: float
    jtd_amount: float


def _result(quote_type: str, cds: CDS, val: date, disc: DiscountCurve, curve: CreditCurve,
            maturities: Sequence[date], spreads: Sequence[float],
            conventional_spread: Optional[float]) -> PricingResult:
    v = value_cds(cds, val, disc, curve)
    r = risk_report(cds, val, maturities, spreads, disc)
    if conventional_spread is None:
        conventional_spread = conventional_spread_from_upfront(cds, val, v.upfront, disc)
    return PricingResult(
        quote_type=quote_type, credit_curve=curve, par_spread=v.par_spread,
        conventional_spread=conventional_spread, upfront=v.upfront,
        price=clean_price(v.upfront), accrued=v.accrued_amount,
        protection_pv=v.protection_pv, rpv01=v.rpv01_clean,
        pv_clean=v.pv_clean, pv_dirty=v.pv_dirty, settle_date=cash_settle_date(val),
        market_value_amount=signed(cds, v.pv_clean),
        cash_settlement_amount=signed(cds, v.upfront - v.accrued_amount),
        cs01_amount=signed(cds, r.cs01),
        cs01_bucket_amounts={m: signed(cds, b) for m, b in zip(maturities, r.cs01_buckets)},
        ir01_amount=signed(cds, r.ir01), rec01_amount=signed(cds, r.rec01),
        jtd_amount=signed(cds, r.jtd))


class ConventionalSpreadEngine:
    """Price one contract from its own quoted (conventional) spread.

    The quote is turned into a flat hazard curve by solving the single
    hazard rate at which a CDS of the same maturity paying that spread is
    worth zero.  This is exact for the quoted contract by construction, and
    must not be reused for any other maturity.
    """

    def __init__(self, valuation_date: date, discount_curve: DiscountCurve):
        self.valuation_date = valuation_date
        self.discount_curve = discount_curve

    def flat_curve(self, cds: CDS, spread: float) -> CreditCurve:
        return bootstrap_credit_curve(self.valuation_date, [cds.maturity], [spread],
                                      cds.recovery, self.discount_curve)

    def price(self, cds: CDS, spread: float) -> PricingResult:
        curve = self.flat_curve(cds, spread)
        return _result("conventional", cds, self.valuation_date, self.discount_curve, curve,
                       [cds.maturity], [spread], conventional_spread=spread)

    def implied_spread(self, cds: CDS, *, upfront: Optional[float] = None,
                       price: Optional[float] = None) -> float:
        """Inverse direction: clean upfront (fraction) or price per 100 -> spread."""
        if (upfront is None) == (price is None):
            raise ValueError("give exactly one of upfront or price")
        if upfront is None:
            upfront = 1.0 - price / 100.0
        return conventional_spread_from_upfront(cds, self.valuation_date, upfront,
                                                self.discount_curve)


class ParSpreadEngine:
    """Price any contract on a name from its par spread term structure.

    The curve is bootstrapped once, when the engine is created, so that every
    contract on the name is valued on the same curve.
    """

    def __init__(self, valuation_date: date, discount_curve: DiscountCurve,
                 maturities: Sequence[date], par_spreads: Sequence[float], recovery: float):
        self.valuation_date = valuation_date
        self.discount_curve = discount_curve
        self.maturities = list(maturities)
        self.par_spreads = list(par_spreads)
        self.recovery = recovery
        self.credit_curve = bootstrap_credit_curve(valuation_date, self.maturities,
                                                   self.par_spreads, recovery, discount_curve)

    @classmethod
    def from_tenors(cls, valuation_date: date, discount_curve: DiscountCurve,
                    tenor_years: Sequence[int], par_spreads: Sequence[float],
                    recovery: float) -> "ParSpreadEngine":
        """Quotes given as standard tenors (1, 3, 5, ... years)."""
        mats = [cds_maturity(valuation_date, 12 * y) for y in tenor_years]
        return cls(valuation_date, discount_curve, mats, par_spreads, recovery)

    def price(self, cds: CDS) -> PricingResult:
        if abs(cds.recovery - self.recovery) > 1e-12:
            raise ValueError(
                f"contract recovery {cds.recovery:.0%} differs from the {self.recovery:.0%} "
                "used to build the curve; a par spread curve is only meaningful with the "
                "recovery it was calibrated to")
        return _result("par", cds, self.valuation_date, self.discount_curve, self.credit_curve,
                       self.maturities, self.par_spreads, conventional_spread=None)
