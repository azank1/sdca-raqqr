"""Generate docs/assets/equity_curve.png — DCA portfolio vs lump-sum.

Run from the repo root with the venv active:
    python docs/plot_equity.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sdca_core as sc

START = "2018-01-01"
CASH  = 10_000.0

print("Fetching BTC daily data from Binance...")
ohlcv = sc.data.load_binance("BTCUSDT")
print(f"  {len(ohlcv)} days loaded  "
      f"({ohlcv.index[0].date()} → {ohlcv.index[-1].date()})")

res = sc.backtest_curve(ohlcv, starting_cash=CASH, start=START)
eq  = res.equity_curve

# lump-sum reference: buy at first price, hold
first_price = float(ohlcv.loc[ohlcv.index >= START, "close"].iloc[0])
lump_btc    = CASH / first_price
lump_series = lump_btc * eq["price"]

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

ax.plot(eq.index, eq["portfolio"], color="#3fb950", linewidth=1.6,
        label=f"RAQQR DCA  (${res.portfolio_value:,.0f}  +{res.return_pct:.0f}%)")
ax.plot(eq.index, lump_series,  color="#58a6ff", linewidth=1.2,
        linestyle="--", alpha=0.8,
        label=f"Lump-sum   (${res.lump_value:,.0f}  +{res.lump_return_pct:.0f}%)")
ax.fill_between(eq.index, eq["portfolio"], lump_series,
                where=(eq["portfolio"].values >= lump_series.values),
                alpha=0.08, color="#3fb950")

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"${v:,.0f}" if v >= 1000 else f"${v:.0f}"))
for spine in ax.spines.values():
    spine.set_edgecolor("#30363d")
ax.tick_params(colors="#8b949e")
ax.grid(True, color="#21262d", linewidth=0.6)

ax.set_title("RAQQR DCA vs Lump-Sum  |  $10k starting capital  |  2018 – present",
             color="#e6edf3", fontsize=12, pad=12)
ax.legend(facecolor="#161b22", edgecolor="#30363d",
          labelcolor="#e6edf3", fontsize=10)

out = os.path.join(os.path.dirname(__file__), "assets", "equity_curve.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
