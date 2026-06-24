# sdca-raqqr — v0.2 `az/feat/mltpi-integration`

> **This branch is under active development. Main branch and the live Streamlit deployment are untouched.**

![v0.2 in progress](assets/v02_wip.gif)

---

## What's cooking

v0.2 merges the **MLTPI (Machine Learning Trend Probability Indicator)** into the RAQQR valuation engine — producing a combined quant signal that uses long-term rainbow valuation *and* Bayesian-optimised medium-term trend confirmation.

```
v0.1 (main, live now)         v0.2 (this branch, coming)
─────────────────────         ──────────────────────────
RAQQR bands                   RAQQR bands
EQM risk score                EQM risk score
DCA backtest                  DCA backtest
                              + MLTPI H(α) signal blend
                              + Sharpe / Sortino / Omega / Calmar
                              + ISP annotation UI
                              + Walk-forward validation
                              + Three-way equity comparison
```

---

## Why it matters (or: the market doesn't care about your feelings)

> *"Compound interest is the eighth wonder of the world. He who understands it, earns it. He who doesn't, pays it."* — probably Buffett, definitely overused

The RAQQR model tells you if BTC is historically **cheap or expensive**.
The MLTPI asks: **is the trend even ready to move yet?**

Combining them prevents the classic quant mistake: buying into maximum cheapness during a year-long downtrend, watching every "historically cheap" candle get cheaper. MLTPI gates RAQQR — it says "yes, it's cheap AND the trend is turning." That's the signal worth acting on.

> *"In theory, theory and practice are the same. In practice, they're not."* — every backtest ever

Both systems are in-sample to some degree. Don't use this as a crystal ball. Use it as a framework for thinking. Then paper trade. Then risk 1%.

---

## New in v0.2

### MLTPI Signal Builder

Annotate your own ideal entry/exit signal directly on the candlestick chart (max 40 trades over 3 years). The system trains 6 Bayesian-optimised indicators to match your annotations, scores them by Sharpe × Omega, clusters them by behavioural similarity, and reconstructs a weighted composite:

```
H(α) = Σ wᵢ · Sᵢ    ∈ [-1, 1]
```

Walk-forward split (80/20) separates training from validation so you can see real out-of-sample performance before applying the signal to your backtest.

### Quant Ratios (now in every backtest)

| Ratio | Measures |
|---|---|
| **Sharpe** | Return per unit of total volatility |
| **Sortino** | Return per unit of *downside* volatility only |
| **Omega** | Total gains / total losses (threshold-free) |
| **Calmar** | Annualised return / max drawdown |
| **Max Drawdown** | Worst peak-to-trough loss |

### Combined Signal Tab

- Pure RAQQR composite risk vs MLTPI-blended risk overlaid
- Raw H(α) signal panel (green = long bias, red = short bias)
- Confluence heatmap: days where both signals agree vs disagree

### Three-Way Equity Comparison

Pure RAQQR / MLTPI-only / RAQQR + MLTPI — vs lump-sum hold. See which blend actually adds lift.

---

## Branch status

```
az/feat/mltpi-integration
├── sdca_core/signals/mltpi/       ✅ pipeline + 6 indicators
├── sdca_core/backtest/metrics.py  ✅ Sharpe, Sortino, Omega, Calmar
├── pages/signal_builder.py        ✅ ISP annotation + training UI
├── app.py                         ✅ Combined Signal tab, ratio grid, 3-way chart
└── requirements.txt               ✅ bayesian-optimization, scikit-learn, optuna, scipy
```

> *"The stock market is a device for transferring money from the impatient to the patient."*
> — Warren Buffett (still relevant for BTC, arguably more so)

---

## v0.1 (live now)

The stable release lives on `main` and is deployed at:
**[sdca-raqqr.streamlit.app](https://sdca-raqqr-ndabr4wdankxgrrpep9gzk.streamlit.app/)**

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Branch](https://img.shields.io/badge/branch-az%2Ffeat%2Fmltpi--integration-orange)](https://github.com/azank1/sdca-raqqr/tree/az/feat/mltpi-integration)

---

> **Not financial advice.** Outputs are model estimates. Past performance is the one thing that definitely doesn't predict future results — and yet here we all are, looking at backtests.
