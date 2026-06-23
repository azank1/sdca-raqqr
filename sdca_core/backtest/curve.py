"""Accumulation / Distribution curve backtest.

Faithful port of the artifact's `runCurveBacktest`. Each day the curve value at
that day's composite risk is the trade rate:

    rate > 0:  buy  rate%  of *current cash*
    rate < 0:  sell |rate|% of *current BTC*

This compounds against the remaining balance, not against starting capital — so
deployment decelerates as cash depletes. That is the frontend's exact behaviour
and is preserved here. The default curve is an illustrative starting shape, not a
fitted optimum; the whole point of the tab is that the user reshapes it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 21 nodes at risk = 0,5,...,100
CURVE_RISK_NODES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                    55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
CURVE_DEFAULT_VALUES = [10, 10, 10, 10, 3.5, 2.0, 1.2, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, -0.5, -1.5, -2.5, -3.5, -10]


def curve_value_at_risk(risk, nodes=CURVE_RISK_NODES, values=CURVE_DEFAULT_VALUES):
    """Piecewise-linear interpolation of the curve at a given risk in [0,100]."""
    r = float(np.clip(risk, 0, 100)) if np.isfinite(risk) else np.nan
    if not np.isfinite(r):
        return np.nan
    for i in range(len(nodes) - 1):
        x0, x1 = nodes[i], nodes[i + 1]
        if x0 <= r <= x1:
            y0, y1 = values[i], values[i + 1]
            t = (r - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + t * (y1 - y0)
    return values[-1]


@dataclass
class BacktestResult:
    days: int
    buy_days: int
    sell_days: int
    no_trade_days: int
    starting_cash: float
    cash: float
    btc: float
    portfolio_value: float
    pnl: float
    return_pct: float
    avg_buy_price: float
    avg_risk: float
    avg_rate: float
    lump_value: float
    lump_return_pct: float
    vs_lump: float
    vs_lump_pct: float
    equity_curve: pd.DataFrame  # index=date; cols: portfolio, cash, btc, price, risk, rate, trade

    def summary(self) -> pd.Series:
        d = {k: v for k, v in self.__dict__.items() if k != "equity_curve"}
        return pd.Series(d)


def run_curve_backtest(
    price: pd.Series,
    risk: pd.Series,
    starting_cash: float = 10_000.0,
    start=None,
    nodes=CURVE_RISK_NODES,
    values=CURVE_DEFAULT_VALUES,
) -> BacktestResult:
    """Run the curve DCA backtest.

    Parameters
    ----------
    price : daily close, DatetimeIndex.
    risk  : composite risk (0..100) aligned to `price`.
    starting_cash : USD at t0.
    start : optional date; rows before it are dropped.
    nodes, values : the allocation curve (drag-editable in the UI).
    """
    df = pd.DataFrame({"price": price, "risk": risk}).dropna(subset=["price"])
    df = df[df["price"] > 0]
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if df.empty:
        raise ValueError("No rows after filtering; check date range / data.")

    cash = float(starting_cash)
    btc = 0.0
    net_deployed = gross_bought = gross_sold = btc_bought_gross = 0.0
    buy_days = sell_days = no_trade = 0
    risk_sum = rate_sum = 0.0
    risk_count = 0

    rows = []
    for ts, row in df.iterrows():
        p = float(row["price"])
        r = float(row["risk"]) if np.isfinite(row["risk"]) else np.nan
        rate = curve_value_at_risk(r, nodes, values) if np.isfinite(r) else 0.0
        rate = rate if np.isfinite(rate) else 0.0
        trade = 0.0

        if rate > 0.0001:
            buy_usd = min(cash * (rate / 100.0), cash)
            if buy_usd > 0.005:
                bought = buy_usd / p
                cash -= buy_usd
                btc += bought
                net_deployed += buy_usd
                gross_bought += buy_usd
                btc_bought_gross += bought
                trade = buy_usd
                buy_days += 1
            else:
                no_trade += 1
        elif rate < -0.0001:
            sell_btc = min(btc * (-rate / 100.0), btc)
            if sell_btc > 1e-12:
                sell_usd = sell_btc * p
                cash += sell_usd
                btc -= sell_btc
                net_deployed -= sell_usd
                gross_sold += sell_usd
                trade = -sell_usd
                sell_days += 1
            else:
                no_trade += 1
        else:
            no_trade += 1

        if np.isfinite(r):
            risk_sum += r
            rate_sum += rate
            risk_count += 1

        rows.append((ts, cash + btc * p, cash, btc, p, r, rate, trade))

    eq = pd.DataFrame(
        rows, columns=["date", "portfolio", "cash", "btc", "price", "risk", "rate", "trade"]
    ).set_index("date")

    latest = float(df["price"].iloc[-1])
    first = float(df["price"].iloc[0])
    portfolio = cash + btc * latest
    pnl = portfolio - starting_cash
    lump_btc = starting_cash / first
    lump_value = lump_btc * latest

    return BacktestResult(
        days=len(df),
        buy_days=buy_days,
        sell_days=sell_days,
        no_trade_days=no_trade,
        starting_cash=float(starting_cash),
        cash=cash,
        btc=btc,
        portfolio_value=portfolio,
        pnl=pnl,
        return_pct=100.0 * pnl / starting_cash,
        avg_buy_price=(gross_bought / btc_bought_gross) if btc_bought_gross > 0 else float("nan"),
        avg_risk=(risk_sum / risk_count) if risk_count else float("nan"),
        avg_rate=(rate_sum / risk_count) if risk_count else float("nan"),
        lump_value=lump_value,
        lump_return_pct=100.0 * (lump_value - starting_cash) / starting_cash,
        vs_lump=portfolio - lump_value,
        vs_lump_pct=100.0 * (portfolio - lump_value) / lump_value,
        equity_curve=eq,
    )
