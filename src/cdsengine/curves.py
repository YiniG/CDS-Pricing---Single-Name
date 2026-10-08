"""Discount and credit curves with piecewise-constant instantaneous rates.

Both curves store the *negative log* of what they return (-ln DF, -ln Q) at
their node dates and interpolate it linearly in Act/365F time.  Linear
interpolation of a log is exactly a piecewise-constant rate:

    -ln DF(t) = integral of f(u) du   ->  f constant between nodes
    -ln Q(t)  = integral of lambda(u) du  ->  lambda constant between nodes

Beyond the last node the last rate is extended flat.
"""
from __future__ import annotations

import bisect
import math
from datetime import date
from typing import List, Sequence

from .dates import act365


class _LogLinearCurve:
    def __init__(self, base_date: date, dates: Sequence[date], neg_logs: Sequence[float]):
        if len(dates) == 0 or len(dates) != len(neg_logs):
            raise ValueError("need one value per node date")
        if any(b <= a for a, b in zip([base_date, *dates], dates)):
            raise ValueError("node dates must be increasing and after the base date")
        self.base_date = base_date
        self.dates: List[date] = list(dates)
        self._t = [act365(base_date, d) for d in dates]
        self._y = [float(v) for v in neg_logs]

    def time(self, d: date) -> float:
        return act365(self.base_date, d)

    def rates(self) -> List[float]:
        """The constant rate on each interval (previous node, node]."""
        t = [0.0, *self._t]
        y = [0.0, *self._y]
        return [(y[i + 1] - y[i]) / (t[i + 1] - t[i]) for i in range(len(self._t))]

    def _neg_log(self, d: date) -> float:
        t = self.time(d)
        if t <= 0.0:
            return 0.0
        i = bisect.bisect_left(self._t, t)
        if i >= len(self._t):                       # flat extrapolation
            return self._y[-1] + self.rates()[-1] * (t - self._t[-1])
        t0, y0 = (0.0, 0.0) if i == 0 else (self._t[i - 1], self._y[i - 1])
        return y0 + (self._y[i] - y0) * (t - t0) / (self._t[i] - t0)


class DiscountCurve(_LogLinearCurve):
    """Risk-free discount factors, log-linear in DF (flat forwards)."""

    def __init__(self, base_date: date, dates: Sequence[date], dfs: Sequence[float]):
        if any(x <= 0 for x in dfs):
            raise ValueError("discount factors must be positive")
        super().__init__(base_date, dates, [-math.log(x) for x in dfs])

    @classmethod
    def flat(cls, base_date: date, rate: float) -> "DiscountCurve":
        """Continuously compounded flat rate, Act/365F."""
        node = date(base_date.year + 1, base_date.month, min(base_date.day, 28))
        return cls(base_date, [node], [math.exp(-rate * act365(base_date, node))])

    def log_df(self, d: date) -> float:
        return -self._neg_log(d)

    def df(self, d: date) -> float:
        return math.exp(self.log_df(d))

    def shifted(self, bump: float) -> "DiscountCurve":
        """Parallel shift of every continuously compounded zero rate."""
        return DiscountCurve(self.base_date, self.dates,
                             [math.exp(-(y + bump * t)) for y, t in zip(self._y, self._t)])


class CreditCurve(_LogLinearCurve):
    """Survival probabilities with a piecewise-constant hazard rate.

    ``hazards[i]`` applies on the interval (dates[i-1], dates[i]].
    """

    def __init__(self, base_date: date, dates: Sequence[date], hazards: Sequence[float]):
        if any(h < 0 for h in hazards):
            raise ValueError("hazard rates must be non-negative")
        t = [0.0] + [act365(base_date, d) for d in dates]
        cum, y = 0.0, []
        for i, h in enumerate(hazards):
            cum += h * (t[i + 1] - t[i])
            y.append(cum)
        super().__init__(base_date, dates, y)
        self.hazards = [float(h) for h in hazards]

    def log_survival(self, d: date) -> float:
        return -self._neg_log(d)

    def survival(self, d: date) -> float:
        return math.exp(self.log_survival(d))
