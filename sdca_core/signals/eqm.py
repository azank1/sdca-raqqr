"""EQM signals — risk (0..99) and reversed valuation z-score (-3..+3).

Exact ports of `eqmRiskAtIndex` and `eqmZScoreAtIndex`.

Sign convention is reversed on purpose: cheap = positive z / low risk,
expensive = negative z / high risk. That keeps "buy more when the number is
high" intuitive for accumulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..coefficients import RAQQR_KEYS, RAQQR_RISK_MARKS

_MARKS = np.array(RAQQR_RISK_MARKS, dtype=float)


def eqm_risk(bands: pd.DataFrame, price: pd.Series) -> pd.Series:
    """Linear interpolation of price position in log-space between the 7 bands,
    mapped onto the risk marks [1,10,25,50,75,95,99]. Below Q1% -> 1, above
    Q99% -> 99. Returns NaN where price or any band is non-positive/missing.
    """
    levels = bands[RAQQR_KEYS].to_numpy(dtype=float)      # (n,7) ascending
    p = price.to_numpy(dtype=float)

    valid = (
        np.isfinite(p) & (p > 0)
        & np.isfinite(levels).all(axis=1) & (levels > 0).all(axis=1)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        lp = np.log(p)
        ll = np.log(levels)
        # segment index = index of last band <= price, clipped to [0,5]
        seg = np.clip((levels <= p[:, None]).sum(axis=1) - 1, 0, 5)
        lo = np.take_along_axis(ll, seg[:, None], 1)[:, 0]
        hi = np.take_along_axis(ll, (seg + 1)[:, None], 1)[:, 0]
        f = (lp - lo) / (hi - lo)
        m_lo = _MARKS[seg]
        m_hi = _MARKS[seg + 1]
        risk = np.clip(m_lo + f * (m_hi - m_lo), 1.0, 99.0)

    risk = np.where(p < levels[:, 0], 1.0, risk)
    risk = np.where(p > levels[:, -1], 99.0, risk)
    risk = np.where(valid, risk, np.nan)
    return pd.Series(risk, index=price.index, name="eqm_risk")


def eqm_zscore(bands: pd.DataFrame, price: pd.Series) -> pd.Series:
    """Reversed EQM valuation z-score using the low rail / median / high rail.

        price <= median:  +3 * (ln(mid)-ln(price)) / (ln(mid)-ln(low))   -> [0, 3]
        price >  median:  -3 * (ln(price)-ln(mid)) / (ln(high)-ln(mid))  -> [-3, 0]
    """
    low = bands["lowRail"].to_numpy(dtype=float)
    mid = bands["0.5"].to_numpy(dtype=float)
    high = bands["highRail"].to_numpy(dtype=float)
    p = price.to_numpy(dtype=float)

    valid = (
        np.isfinite(p) & (p > 0)
        & np.isfinite(low) & (low > 0)
        & np.isfinite(mid) & (mid > 0)
        & np.isfinite(high) & (high > 0)
        & (low < mid) & (mid < high)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        lp, ll, lm, lh = np.log(p), np.log(low), np.log(mid), np.log(high)
        lower = np.clip(3.0 * (lm - lp) / (lm - ll), 0.0, 3.0)
        upper = np.clip(-3.0 * (lp - lm) / (lh - lm), -3.0, 0.0)
        z = np.where(lp <= lm, lower, upper)
    z = np.where(valid, z, np.nan)
    return pd.Series(z, index=price.index, name="eqm_z")
