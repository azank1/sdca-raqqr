"""Quant performance ratios computed from a BacktestResult equity curve.

All ratios are annualized (252 trading days).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ratios(equity_curve: pd.DataFrame, risk_free: float = 0.0) -> dict:
    """Compute Sharpe, Sortino, Omega, Calmar and Max Drawdown.

    Parameters
    ----------
    equity_curve : BacktestResult.equity_curve (must have a 'portfolio' column)
    risk_free    : annualized risk-free rate (default 0.0)

    Returns
    -------
    dict with keys: sharpe, sortino, omega, calmar, max_drawdown_pct
    """
    port  = equity_curve["portfolio"].dropna()
    rets  = port.pct_change().dropna()
    ann   = 252 ** 0.5
    rfday = risk_free / 252.0
    n     = len(rets)

    if n < 2:
        return dict(sharpe=float("nan"), sortino=float("nan"),
                    omega=float("nan"), calmar=float("nan"),
                    max_drawdown_pct=float("nan"))

    excess = rets - rfday

    # Sharpe
    sharpe = float(excess.mean() / (excess.std(ddof=1) + 1e-12) * ann)

    # Sortino — downside deviation uses only negative excess returns
    neg    = excess[excess < 0]
    downside_std = float(np.sqrt((neg ** 2).mean())) if len(neg) > 0 else 1e-12
    sortino = float(excess.mean() / (downside_std + 1e-12) * ann)

    # Omega — ratio of gains to losses (sum-based, threshold = risk-free)
    gains  = float(excess[excess > 0].sum())
    losses = float(abs(excess[excess < 0].sum()))
    omega  = gains / losses if losses > 1e-12 else float("inf")

    # Max drawdown
    cummax = port.cummax()
    dd     = (port - cummax) / (cummax + 1e-12)
    max_dd = float(dd.min())           # most negative value

    # Calmar — annualised return / abs(max drawdown)
    ann_ret = float(excess.mean() * 252)
    calmar  = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else float("inf")

    return dict(
        sharpe          = round(sharpe,  3),
        sortino         = round(sortino, 3),
        omega           = round(omega,   3),
        calmar          = round(calmar,  3),
        max_drawdown_pct= round(max_dd * 100, 2),   # as %, e.g. -42.3
    )
