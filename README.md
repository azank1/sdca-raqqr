# sdca-core

Python parity port of the **Bitcoin Asymmetric Tail Curvature (RAQQR) Rainbow**
valuation model and its accumulation/distribution DCA backtest. The numbers this
library produces match the HTML artifact's default configuration — proven by a
cross-implementation parity test, not assumed.

This is the quant-facing layer: load OHLCV, get a pandas table of bands and
signals, run vectorised backtests, export CSV. The interactive HTML chart sits on
top of the same math (see `shared/` parity contract).

## Install

```bash
pip install -e .            # core (numpy, pandas)
pip install -e ".[binance,dev]"   # + live fetch + pytest
```

## Use

```python
import sdca_core as sc

ohlcv = sc.data.load_csv("btc_daily.csv")     # any CSV with a 'close' column
table = sc.analyze(ohlcv)
# columns: close, 0.01..0.99, lowRail, highRail,
#          eqm_risk, eqm_z, composite_z, composite_risk, cqm_z

res = sc.backtest_curve(ohlcv, starting_cash=10_000, start="2018-01-01")
print(res.summary())          # days, btc, portfolio_value, return_pct, vs_lump_pct, ...
res.equity_curve.to_csv("equity.csv")
```

Reshape the allocation curve (the draggable nodes in the UI) by passing values:

```python
sc.backtest_curve(ohlcv, values=[12,12,10,8,4,2,1,0,0,0,0,0,0,0,0,0,-1,-2,-3,-5,-12])
```

## What maps to what

| Artifact concept            | Library |
|-----------------------------|---------|
| RAQQR bands + rails         | `sc.raqqr_bands(index)` |
| EQM Risk / EQM Z-score      | `sc.eqm_risk`, `sc.eqm_zscore` |
| Composite Z / CQM Risk      | `sc.composite_z`, `composite_risk_from_z`, `cqm_z_from_risk` |
| Accum/Dist curve backtest   | `sc.run_curve_backtest`, `sc.backtest_curve` |
| Curve node shape            | `sc.CURVE_RISK_NODES`, `sc.CURVE_DEFAULT_VALUES` |

## Parity contract

`sdca_core/coefficients.py` is the single source of truth. The web frontend must
read the same values. `parity/dump_js_bands.mjs` emits golden band prices from the
**verbatim** artifact formula; `tests/test_raqqr_parity.py` asserts the Python
port matches to floating point.

```bash
node parity/dump_js_bands.mjs > /dev/null   # regenerate if coefficients change
pytest -q                                    # 10 tests, incl. JS parity
```

## Known properties (inherited from the model, by design)

- **Full-sample / look-ahead.** RAQQR coefficients are fit on the whole history,
  so historical band values are in-sample. Backtests are illustrative of the
  *rule shape*, not an out-of-sample track record.
- **Compounding against balance.** The curve buys/sells a % of *current* cash/BTC,
  not of starting capital, so deployment decelerates as the balance moves.
- **Sharpe / CBPL indicators are not yet implemented** (stubs in the artifact).
  The composite framework accepts them via `sc.Indicator(...)` once computed.

## Not financial advice

A valuation and backtesting tool. Outputs are model estimates, not
recommendations.
