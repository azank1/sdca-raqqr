"""High-level entry point.

`analyze(ohlcv)` returns one DataFrame with the bands and every signal the
frontend shows, so a quant can go straight to a table without wiring the layers
by hand. The lower-level modules remain importable for custom pipelines.
"""
from __future__ import annotations

import pandas as pd

from .models.raqqr import raqqr_bands
from .signals.eqm import eqm_risk, eqm_zscore
from .signals.composite import (
    Indicator,
    composite_z,
    composite_risk_from_z,
    cqm_z_from_risk,
)
from .backtest.curve import run_curve_backtest


def analyze(ohlcv: pd.DataFrame, extra_indicators: list[Indicator] | None = None) -> pd.DataFrame:
    """Compute bands + EQM + composite signals for an OHLCV frame.

    Parameters
    ----------
    ohlcv : DataFrame with a DatetimeIndex and a ``close`` column.
    extra_indicators : optional additional Indicators to blend into the composite
        (Sharpe, CBPL, ...). With none supplied, composite z == EQM z, matching
        the frontend default.

    Returns a DataFrame with columns:
        close, <7 band keys>, lowRail, highRail,
        eqm_risk, eqm_z, composite_z, composite_risk, cqm_z
    """
    if "close" not in ohlcv.columns:
        raise ValueError("ohlcv must have a 'close' column")
    price = ohlcv["close"].astype(float)

    bands = raqqr_bands(price.index)
    er = eqm_risk(bands, price)
    ez = eqm_zscore(bands, price)

    indicators = [Indicator("price", ez, weight=1.0)]
    if extra_indicators:
        indicators += list(extra_indicators)
    cz = composite_z(indicators)
    crisk = composite_risk_from_z(cz)
    cqmz = cqm_z_from_risk(crisk)

    out = bands.copy()
    out.insert(0, "close", price)
    out["eqm_risk"] = er
    out["eqm_z"] = ez
    out["composite_z"] = cz
    out["composite_risk"] = crisk
    out["cqm_z"] = cqmz
    return out


def backtest_curve(ohlcv: pd.DataFrame, starting_cash=10_000.0, start=None,
                   nodes=None, values=None, extra_indicators=None):
    """Convenience: analyze then run the curve DCA backtest on composite risk."""
    table = analyze(ohlcv, extra_indicators=extra_indicators)
    kwargs = {"starting_cash": starting_cash, "start": start}
    if nodes is not None:
        kwargs["nodes"] = nodes
    if values is not None:
        kwargs["values"] = values
    return run_curve_backtest(table["close"], table["composite_risk"], **kwargs)
