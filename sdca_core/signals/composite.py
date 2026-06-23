"""Composite signal layer.

The frontend blends one or more indicator z-scores into a composite z, then maps
that to a composite risk via a *linear* rule (deliberately different from the
nonlinear EQM-risk interpolation). With only the price indicator enabled — the
default — the composite z equals the EQM z exactly, so this layer reproduces the
frontend's default numbers while leaving room to add indicators later
(e.g. Sharpe / CBPL, currently unimplemented stubs in the artifact).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Indicator:
    """A per-day z-score series and its weight in the composite."""
    name: str
    z: pd.Series
    weight: float = 1.0
    enabled: bool = True


def composite_z(indicators: list[Indicator]) -> pd.Series:
    """Weighted average of enabled, finite indicator z-scores, clipped to [-3, 3].

    NaN indicators are skipped per day (matching the JS isFinite guard). If no
    indicator is finite on a day, the composite is NaN there.
    """
    enabled = [i for i in indicators if i.enabled and i.weight > 0]
    if not enabled:
        idx = indicators[0].z.index if indicators else pd.Index([])
        return pd.Series(np.nan, index=idx, name="composite_z")

    idx = enabled[0].z.index
    zsum = np.zeros(len(idx))
    wsum = np.zeros(len(idx))
    for ind in enabled:
        zi = ind.z.reindex(idx).to_numpy(dtype=float)
        finite = np.isfinite(zi)
        w = float(np.clip(ind.weight, 0, 5))
        zsum = np.where(finite, zsum + w * zi, zsum)
        wsum = np.where(finite, wsum + w, wsum)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(wsum > 0, np.clip(zsum / wsum, -3, 3), np.nan)
    return pd.Series(z, index=idx, name="composite_z")


def composite_risk_from_z(z: pd.Series) -> pd.Series:
    """Linear z -> risk map: clip(50 - (z/3)*50, 0, 100)."""
    r = np.clip(50.0 - (z.to_numpy(dtype=float) / 3.0) * 50.0, 0.0, 100.0)
    return pd.Series(r, index=z.index, name="composite_risk")


def cqm_z_from_risk(risk: pd.Series) -> pd.Series:
    """Reverse map used for the CQM Z line: clip(3*(50-risk)/50, -3, 3)."""
    z = np.clip(3.0 * (50.0 - risk.to_numpy(dtype=float)) / 50.0, -3.0, 3.0)
    return pd.Series(z, index=risk.index, name="cqm_z")
