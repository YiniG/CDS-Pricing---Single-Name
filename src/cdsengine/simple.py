"""The textbook CDS model: annual coupons, default paid at period end.

This reproduces the course notebook numbers exactly and exists so that the
jump from the classroom formula to the ISDA standard model can be measured.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np

from .calibration import CalibrationError


def df_log_linear(known_t: Sequence[float], known_df: Sequence[float],
                  t: Sequence[float]) -> np.ndarray:
    """Interpolate discount factors linearly in ln(DF); DF(0) = 1."""
    kt = np.concatenate([[0.0], np.asarray(known_t, float)])
    kl = np.concatenate([[0.0], np.log(np.asarray(known_df, float))])
    return np.exp(np.interp(np.asarray(t, float), kt, kl))


def bootstrap_simple(times: Sequence[float], dfs: Sequence[float],
                     spreads: Sequence[float], recovery: float,
                     *, strict: bool = True) -> List[float]:
    """Survival probabilities P(0)=1, P(t_1), ... from par spreads (decimals).

    ``times[0]`` must be 0.  With ``strict=False`` impossible quotes return
    the raw (possibly negative) numbers instead of raising, which is what the
    course notebook displays for CCMO.
    """
    L = 1.0 - recovery
    P = [1.0]
    for N in range(1, len(times)):
        S = spreads[N]
        num = 0.0
        for n in range(1, N):
            dt = times[n] - times[n - 1]
            num += dfs[n] * (L * P[n - 1] - (L + dt * S) * P[n])
        dtN = times[N] - times[N - 1]
        pN = (num + dfs[N] * L * P[N - 1]) / (dfs[N] * (L + dtN * S))
        if strict and not (0.0 < pN <= P[N - 1]):
            raise CalibrationError(
                f"t={times[N]}: implied survival {pN:.4f} after {P[N - 1]:.4f} is not a "
                "valid probability; these spreads cannot be par spreads at this recovery")
        P.append(pN)
    return P


def hazards_from_survival(times: Sequence[float], P: Sequence[float]) -> List[float]:
    out = [0.0]
    for m in range(1, len(times)):
        ok = P[m] > 0 and P[m - 1] > 0
        out.append(-math.log(P[m] / P[m - 1]) / (times[m] - times[m - 1]) if ok else float("nan"))
    return out
