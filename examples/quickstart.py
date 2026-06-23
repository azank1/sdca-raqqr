"""Quickstart: load data, compute the valuation table, run the DCA backtest.

    python examples/quickstart.py path/to/btc_daily.csv
"""
import sys
import sdca_core as sc

if len(sys.argv) > 1:
    ohlcv = sc.data.load_csv(sys.argv[1])          # needs a 'close' column
else:
    ohlcv = sc.data.load_binance("BTCUSDT")        # needs network

table = sc.analyze(ohlcv)
print(table[["close", "0.5", "eqm_risk", "eqm_z", "composite_risk"]].tail())

res = sc.backtest_curve(ohlcv, starting_cash=10_000, start="2018-01-01")
print(res.summary())
res.equity_curve.to_csv("backtest_equity.csv")     # solves the CSV-sharing problem
print("wrote backtest_equity.csv")
