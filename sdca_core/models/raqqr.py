"""RAQQR valuation bands.

Vectorized, exact port of the artifact's `raqqrPricesAtMs` + `buildModel`.

Per timestamp the 7 quantile prices are evaluated from the closed form, then
**rearranged** (sorted ascending) to guarantee non-crossing bands — the CFG
monotone rearrangement. After the sort the named bands are assigned by *position*,
exactly as the frontend does:

    bands["0.01"] = sorted[0]   (also the low rail / Q1% extreme)
    bands["0.5"]  = sorted[3]   (the median band used by the z-score)
    bands["0.99"] = sorted[6]   (also the high rail / Q99% extreme)

So the model is a pure function of the *date* — it does not depend on price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..coefficients import (
    ONE_DAY_MS,
    RAQQR_MU,
    RAQQR_GENESIS_MS,
    RAQQR_KEYS,
    RAQQR_COEF,
)

_C = np.array([RAQQR_COEF[k][0] for k in RAQQR_KEYS])
_A = np.array([RAQQR_COEF[k][1] for k in RAQQR_KEYS])
_B = np.array([RAQQR_COEF[k][2] for k in RAQQR_KEYS])


def days_since_genesis(index: pd.DatetimeIndex) -> np.ndarray:
    """t = max(1, round((ms - genesis) / 86_400_000)), matching the JS exactly."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    ns = idx.to_numpy().astype("datetime64[ns]").astype("int64")  # ns since epoch
    ms = ns / 1e6
    t = np.maximum(1.0, np.round((ms - RAQQR_GENESIS_MS) / ONE_DAY_MS))
    return t


def raqqr_bands(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return a DataFrame of band prices indexed by `index`.

    Columns: the seven RAQQR_KEYS (ascending) plus ``lowRail`` (== "0.01")
    and ``highRail`` (== "0.99").
    """
    idx = pd.DatetimeIndex(index)
    t = days_since_genesis(idx)
    x = np.log(t) - RAQQR_MU                      # (n,)
    # (n, 7) raw prices in nominal key order
    raw = np.power(10.0, _C + _A * x[:, None] + _B * (x * x)[:, None])
    ordered = np.sort(raw, axis=1)                 # CFG rearrangement
    out = pd.DataFrame(ordered, index=idx, columns=RAQQR_KEYS)
    out["lowRail"] = ordered[:, 0]
    out["highRail"] = ordered[:, -1]
    return out
