"""RAQQR Dashboard — Streamlit front door for sdca-raqqr.

Run locally:
    streamlit run app.py

Deploy: push to GitHub and connect at share.streamlit.io (free).
"""
from __future__ import annotations

import io
import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sdca_core as sc
from sdca_core.backtest.curve import CURVE_RISK_NODES, CURVE_DEFAULT_VALUES

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAQQR Bitcoin Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ────────────────────────────────────────────────────────────
BAND_COLOURS = [
    "rgba(59,130,246,0.18)",   # 0.01 → blue
    "rgba(34,197,94,0.18)",    # 0.10 → green
    "rgba(132,204,22,0.18)",   # 0.25 → lime
    "rgba(234,179,8,0.18)",    # 0.50 → yellow
    "rgba(249,115,22,0.18)",   # 0.75 → orange
    "rgba(239,68,68,0.18)",    # 0.95 → red
]
BAND_KEYS  = ["0.01", "0.1", "0.25", "0.5", "0.75", "0.95", "0.99"]
BAND_NAMES = ["Q1%", "Q10%", "Q25%", "Q50% (median)", "Q75%", "Q95%", "Q99%"]

PRESETS = {
    "Conservative": [6, 6, 5, 4, 2, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.3, -0.8, -1.5, -2.5, -6],
    "Default":      CURVE_DEFAULT_VALUES,
    "Aggressive":   [15, 15, 14, 12, 6, 3.5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2.5, -4, -7, -15],
}

# ── helpers ───────────────────────────────────────────────────────────────────
def risk_label(r: float) -> str:
    if r < 25:  return "🟢 Accumulate"
    if r < 50:  return "🟡 Watch"
    if r < 75:  return "🟠 Caution"
    return "🔴 Distribute"

def risk_color(r: float) -> str:
    if r < 25:  return "#22c55e"
    if r < 50:  return "#eab308"
    if r < 75:  return "#f97316"
    return "#ef4444"

# ── data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching BTC price history…")
def fetch_ohlcv() -> tuple[pd.DataFrame, str]:
    """Try Binance; fall back to Yahoo Finance via yfinance.

    Binance is geo-blocked on Streamlit Cloud (AWS US). yfinance is globally
    accessible with no API key required.
    """
    try:
        df = sc.data.load_binance("BTCUSDT")
        return df, "Binance"
    except Exception:
        df = sc.data.load_yfinance("BTC-USD")
        return df, "Yahoo Finance"

@st.cache_data(show_spinner="Computing valuation table…")
def compute_table(ohlcv_hash: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
    return sc.analyze(ohlcv)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("RAQQR Dashboard")
    st.caption("Bitcoin Asymmetric Tail Curvature Rainbow")
    st.divider()

    data_source = st.radio("Data source", ["Live Binance", "Upload CSV"], index=0)

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("CSV with date + close columns", type="csv")
        if uploaded:
            ohlcv_raw = sc.data.load_csv(uploaded)
            data_label = "CSV"
        else:
            st.info("Upload a CSV to continue.")
            st.stop()
    else:
        ohlcv_raw, data_label = fetch_ohlcv()
        st.caption(f"Live data via **{data_label}**")

    min_date = ohlcv_raw.index[0].date()
    max_date = ohlcv_raw.index[-1].date()

    date_range = st.slider(
        "Chart date range",
        min_value=min_date,
        max_value=max_date,
        value=(pd.Timestamp("2018-01-01").date(), max_date),
        format="YYYY-MM-DD",
    )

    st.divider()
    st.subheader("Backtest settings")
    starting_cash = st.number_input("Starting cash (USD)", min_value=100, value=10_000, step=500)
    backtest_start = st.date_input(
        "Backtest start date",
        value=pd.Timestamp("2018-01-01").date(),
        min_value=min_date,
        max_value=max_date,
    )
    preset_name = st.selectbox("Allocation curve preset", list(PRESETS.keys()), index=1)
    curve_values = PRESETS[preset_name]

    st.divider()
    st.subheader("MLTPI Signal")
    mltpi_enabled = st.toggle("Enable MLTPI blend", value=False)
    mltpi_weight  = 0.0
    active_mltpi  = st.session_state.get("active_mltpi")
    if mltpi_enabled:
        if active_mltpi is not None:
            mltpi_weight = st.slider("MLTPI weight", 0.0, 2.0, 0.8, 0.1)
            st.caption(f"Signal covers {len(active_mltpi)} days")
        else:
            st.info("No signal loaded. Build one in **Signal Builder** (sidebar nav).")
            mltpi_enabled = False

    st.divider()
    st.caption("Built on [sdca-raqqr](https://github.com/azank1/sdca-raqqr)")

# ── compute ───────────────────────────────────────────────────────────────────
# Build extra_indicators list if MLTPI is active
_extra_indicators = []
if mltpi_enabled and active_mltpi is not None:
    mltpi_z = (active_mltpi * 3).rename("mltpi_z")
    _extra_indicators = [sc.Indicator("MLTPI", mltpi_z, weight=mltpi_weight)]

table_full = compute_table(str(len(ohlcv_raw)), ohlcv_raw)

# filtered view for charts
mask = (table_full.index.date >= date_range[0]) & (table_full.index.date <= date_range[1])
table = table_full[mask]

# backtest (full history from backtest_start)
@st.cache_data(show_spinner="Running backtest…")
def run_backtest(n_rows: int, cash: float, start: str, preset: str,
                 mltpi_key: str = ""):
    vals  = PRESETS[preset]
    extra = []
    if mltpi_key and st.session_state.get("active_mltpi") is not None:
        sig   = st.session_state["active_mltpi"]
        mz    = (sig * 3).rename("mltpi_z")
        extra = [sc.Indicator("MLTPI", mz, weight=mltpi_weight)]
    return sc.backtest_curve(
        ohlcv_raw, starting_cash=cash, start=start,
        values=vals, extra_indicators=extra or None,
    )

# Unique key so cache invalidates when MLTPI toggles
_bt_key = f"mltpi_{mltpi_enabled}_{mltpi_weight}" if mltpi_enabled else ""
res = run_backtest(len(ohlcv_raw), float(starting_cash),
                   str(backtest_start), preset_name, _bt_key)

# Pure RAQQR backtest (always computed, for three-way comparison)
res_pure = run_backtest(len(ohlcv_raw), float(starting_cash),
                        str(backtest_start), preset_name, "")

# ── current readings ──────────────────────────────────────────────────────────
latest = table_full.iloc[-1]
cur_price     = latest["close"]
cur_eqm_risk  = latest["eqm_risk"]
cur_comp_risk = latest["composite_risk"]
prev          = table_full.iloc[-2]

# ── top KPI strip ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("BTC Price",        f"${cur_price:,.0f}",
          f"{cur_price - prev['close']:+,.0f}")
k2.metric("EQM Risk",         f"{cur_eqm_risk:.1f} / 100",
          f"{cur_eqm_risk - prev['eqm_risk']:+.2f}")
k3.metric("Composite Risk",   f"{cur_comp_risk:.1f} / 100",
          f"{cur_comp_risk - prev['composite_risk']:+.2f}")
k4.metric("Signal",           risk_label(cur_eqm_risk))
k5.metric("Data through",     str(table_full.index[-1].date()))

st.divider()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_rainbow, tab_risk, tab_combined, tab_backtest = st.tabs(
    ["🌈  Rainbow Chart", "📡  Risk Signal", "🔀  Combined Signal", "📈  Backtest"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RAINBOW CHART
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rainbow:
    fig = go.Figure()

    # filled bands (paint from outermost inward)
    band_pairs = [
        ("0.95", "0.99", BAND_COLOURS[5], "Q95–Q99 (distribute)"),
        ("0.75", "0.95", BAND_COLOURS[4], "Q75–Q95 (caution)"),
        ("0.5",  "0.75", BAND_COLOURS[3], "Q50–Q75 (watch)"),
        ("0.25", "0.5",  BAND_COLOURS[2], "Q25–Q50 (neutral)"),
        ("0.1",  "0.25", BAND_COLOURS[1], "Q10–Q25 (accumulate)"),
        ("0.01", "0.1",  BAND_COLOURS[0], "Q1–Q10 (strong buy)"),
    ]
    for lo_key, hi_key, colour, label in band_pairs:
        fig.add_trace(go.Scatter(
            x=table.index, y=table[hi_key],
            fill=None, mode="lines",
            line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=table.index, y=table[lo_key],
            fill="tonexty",
            fillcolor=colour,
            mode="lines",
            line=dict(width=0),
            name=label,
            hoverinfo="skip",
        ))

    # price line
    fig.add_trace(go.Scatter(
        x=table.index, y=table["close"],
        mode="lines",
        name="BTC Price",
        line=dict(color="#ffffff", width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:,.0f}<extra></extra>",
    ))

    # low / high rails
    for rail_key, rail_name in [("lowRail", "Low Rail (Q1%)"), ("highRail", "High Rail (Q99%)")]:
        fig.add_trace(go.Scatter(
            x=table.index, y=table[rail_key],
            mode="lines",
            name=rail_name,
            line=dict(dash="dot", width=1, color="rgba(255,255,255,0.4)"),
            hoverinfo="skip",
        ))

    fig.update_layout(
        title="RAQQR Rainbow — BTC Price vs Quantile Bands",
        yaxis_type="log",
        yaxis_title="Price (USD, log scale)",
        xaxis_title="Date",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        height=560,
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How to read this chart"):
        st.markdown("""
- **Price inside a low band (blue/green)** → historically cheap by the RAQQR model → accumulation zone
- **Price in the middle (yellow)** → near fair value (Q50% median band)
- **Price in the upper bands (orange/red)** → historically expensive → distribution zone
- The bands are **not support/resistance** — they are quantile regression lines fit to the full history.
        """)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_risk:
    col_gauge, col_hist = st.columns([1, 2])

    with col_gauge:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cur_eqm_risk,
            delta={"reference": float(prev["eqm_risk"]), "valueformat": ".1f"},
            title={"text": "EQM Risk Score", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": risk_color(cur_eqm_risk), "thickness": 0.25},
                "steps": [
                    {"range": [0,  25], "color": "rgba(34,197,94,0.15)"},
                    {"range": [25, 50], "color": "rgba(234,179,8,0.15)"},
                    {"range": [50, 75], "color": "rgba(249,115,22,0.15)"},
                    {"range": [75,100], "color": "rgba(239,68,68,0.15)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": cur_eqm_risk,
                },
            },
            number={"suffix": " / 100", "font": {"size": 36}},
        ))
        gauge_fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

        zone_color_map = {
            "🟢 Accumulate": "green",
            "🟡 Watch":      "yellow",
            "🟠 Caution":    "orange",
            "🔴 Distribute": "red",
        }
        label = risk_label(cur_eqm_risk)
        st.markdown(
            f"<div style='text-align:center; font-size:1.4rem; font-weight:600; "
            f"padding:12px; border-radius:8px; background:rgba(255,255,255,0.06)'>"
            f"{label}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Composite risk: **{cur_comp_risk:.1f}** / 100")

    with col_hist:
        risk_fig = go.Figure()

        # zone bands
        zone_defs = [
            (0,  25, "rgba(34,197,94,0.07)",  "Accumulate"),
            (25, 50, "rgba(234,179,8,0.07)",   "Watch"),
            (50, 75, "rgba(249,115,22,0.07)",  "Caution"),
            (75,100, "rgba(239,68,68,0.07)",   "Distribute"),
        ]
        for lo, hi, colour, zname in zone_defs:
            risk_fig.add_hrect(y0=lo, y1=hi, fillcolor=colour,
                               line_width=0, annotation_text=zname,
                               annotation_position="right",
                               annotation_font_size=11,
                               annotation_font_color="rgba(255,255,255,0.5)")

        risk_fig.add_trace(go.Scatter(
            x=table.index, y=table["eqm_risk"],
            mode="lines", name="EQM Risk",
            line=dict(color="#58a6ff", width=1.4),
        ))
        risk_fig.add_trace(go.Scatter(
            x=table.index, y=table["composite_risk"],
            mode="lines", name="Composite Risk",
            line=dict(color="#f97316", width=1.4, dash="dot"),
        ))

        risk_fig.update_layout(
            title="Risk Score History",
            yaxis=dict(range=[0, 100], title="Risk (0–100)"),
            xaxis_title="Date",
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
            height=340,
            margin=dict(t=60, b=40),
        )
        st.plotly_chart(risk_fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMBINED SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════
with tab_combined:
    if not mltpi_enabled or active_mltpi is None:
        st.info(
            "Enable the MLTPI blend in the sidebar (and build a signal in **Signal Builder**) "
            "to see the combined view."
        )
    else:
        mltpi_z_full = (active_mltpi * 3).reindex(table_full.index).ffill().fillna(0)

        # ── side-by-side risk comparison ────────────────────────────────────
        st.subheader("Pure RAQQR vs MLTPI-Blended Composite Risk")
        comb_indicators = [
            sc.Indicator("price", table_full["eqm_z"], weight=1.0),
            sc.Indicator("MLTPI", mltpi_z_full, weight=mltpi_weight),
        ]
        blended_z    = sc.composite_z(comb_indicators)
        blended_risk = sc.composite_risk_from_z(blended_z)
        blended_t    = blended_risk[mask]
        pure_risk_t  = table["composite_risk"]

        cf = go.Figure()
        cf.add_trace(go.Scatter(x=table.index, y=pure_risk_t,
                                mode="lines", name="Pure RAQQR",
                                line=dict(color="#58a6ff", width=1.4)))
        cf.add_trace(go.Scatter(x=blended_t.index, y=blended_t,
                                mode="lines", name=f"MLTPI blended (w={mltpi_weight})",
                                line=dict(color="#f97316", width=1.6, dash="dot")))
        for lo, hi, c in [(0,25,"rgba(34,197,94,0.06)"),
                          (25,50,"rgba(234,179,8,0.06)"),
                          (50,75,"rgba(249,115,22,0.06)"),
                          (75,100,"rgba(239,68,68,0.06)")]:
            cf.add_hrect(y0=lo, y1=hi, fillcolor=c, line_width=0)
        cf.update_layout(template="plotly_dark", height=300,
                         yaxis=dict(range=[0,100]), hovermode="x unified",
                         margin=dict(t=20, b=40))
        st.plotly_chart(cf, use_container_width=True)

        # ── MLTPI H(α) panel ─────────────────────────────────────────────────
        st.subheader("MLTPI H(α) Signal")
        mltpi_t = active_mltpi.reindex(table.index).ffill()
        hf = go.Figure()
        hf.add_trace(go.Scatter(
            x=mltpi_t.index, y=mltpi_t.values,
            mode="lines", fill="tozeroy",
            fillcolor="rgba(63,185,80,0.12)",
            line=dict(color="#3fb950", width=1.4),
            name="H(α)",
        ))
        hf.add_hline(y=0, line=dict(color="white", width=0.6, dash="dot"))
        hf.update_layout(template="plotly_dark", height=220,
                         yaxis=dict(range=[-1.1, 1.1]),
                         margin=dict(t=10, b=40), showlegend=False)
        st.plotly_chart(hf, use_container_width=True)

        # ── confluence heatmap ────────────────────────────────────────────────
        st.subheader("Confluence — Where Both Signals Agree")
        raqqr_bull   = pure_risk_t < 30
        mltpi_bull   = mltpi_t > 0.1
        raqqr_bear   = pure_risk_t > 70
        mltpi_bear   = mltpi_t < -0.1

        confluence = pd.Series(0.0, index=pure_risk_t.index)
        confluence[raqqr_bull & mltpi_bull]   =  1.0   # both say buy
        confluence[raqqr_bear & mltpi_bear]   = -1.0   # both say sell
        confluence[(raqqr_bull & mltpi_bear) |
                   (raqqr_bear & mltpi_bull)]  =  0.0   # disagreement (grey)

        conf_col = confluence.map(
            lambda v: "#22c55e" if v > 0 else ("#ef4444" if v < 0 else "#374151")
        )
        heatf = go.Figure(go.Bar(
            x=confluence.index, y=abs(confluence),
            marker_color=conf_col.values,
            name="Confluence",
        ))
        heatf.update_layout(template="plotly_dark", height=180,
                            yaxis=dict(range=[0,1.2], visible=False),
                            margin=dict(t=10, b=40),
                            showlegend=False)
        st.plotly_chart(heatf, use_container_width=True)
        st.caption("🟢 Both say buy  |  🔴 Both say sell  |  ⬛ Disagreement")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_backtest:
    eq = res.equity_curve

    # first price for lump-sum line
    lump_btc = res.starting_cash / float(eq["price"].iloc[0])
    lump_line = lump_btc * eq["price"]

    # buy / sell markers
    buys  = eq[eq["trade"] > 0]
    sells = eq[eq["trade"] < 0]

    # ── equity curve chart ──────────────────────────────────────────────────
    eq_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=["Portfolio Value (log scale)", "BTC Price + Trades"],
    )

    eq_fig.add_trace(go.Scatter(
        x=eq.index, y=eq["portfolio"],
        mode="lines", name="DCA Portfolio",
        line=dict(color="#3fb950", width=2),
        hovertemplate="DCA: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    eq_fig.add_trace(go.Scatter(
        x=eq.index, y=lump_line,
        mode="lines", name="Lump-Sum Hold",
        line=dict(color="#58a6ff", width=1.4, dash="dash"),
        hovertemplate="Lump: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    eq_fig.add_trace(go.Scatter(
        x=eq.index, y=eq["price"],
        mode="lines", name="BTC Price",
        line=dict(color="rgba(255,255,255,0.5)", width=1),
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    eq_fig.add_trace(go.Scatter(
        x=buys.index, y=buys["price"],
        mode="markers", name="Buy",
        marker=dict(color="#3fb950", size=4, symbol="triangle-up"),
        hovertemplate="Buy @ $%{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    eq_fig.add_trace(go.Scatter(
        x=sells.index, y=sells["price"],
        mode="markers", name="Sell",
        marker=dict(color="#ef4444", size=4, symbol="triangle-down"),
        hovertemplate="Sell @ $%{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    eq_fig.update_yaxes(type="log", row=1, col=1,
                        tickprefix="$", tickformat=",.0f")
    eq_fig.update_yaxes(type="log", row=2, col=1,
                        tickprefix="$", tickformat=",.0f")
    eq_fig.update_layout(
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        height=560,
        margin=dict(t=80, b=40),
    )
    st.plotly_chart(eq_fig, use_container_width=True)

    # ── summary stats ────────────────────────────────────────────────────────
    st.subheader("Backtest Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Portfolio Value",  f"${res.portfolio_value:,.0f}")
    m2.metric("Total Return",     f"{res.return_pct:.1f}%")
    m3.metric("vs Lump-Sum",      f"+{res.vs_lump_pct:.1f}%")
    m4.metric("BTC Accumulated",  f"{res.btc:.4f} BTC")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Lump-Sum Value",   f"${res.lump_value:,.0f}")
    m6.metric("Avg Buy Price",    f"${res.avg_buy_price:,.0f}")
    m7.metric("Buy Days",         f"{res.buy_days:,}")
    m8.metric("Sell Days",        f"{res.sell_days:,}")

    # ── quant ratios ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Quant Ratios")
    r = res.ratios
    rp = res_pure.ratios
    ra1, ra2, ra3, ra4, ra5 = st.columns(5)
    ra1.metric("Sharpe",       f"{r['sharpe']:.2f}",
               f"{r['sharpe']-rp['sharpe']:+.2f} vs pure" if mltpi_enabled else "")
    ra2.metric("Sortino",      f"{r['sortino']:.2f}",
               f"{r['sortino']-rp['sortino']:+.2f} vs pure" if mltpi_enabled else "")
    ra3.metric("Omega",        f"{r['omega']:.2f}",
               f"{r['omega']-rp['omega']:+.2f} vs pure" if mltpi_enabled else "")
    ra4.metric("Calmar",       f"{r['calmar']:.2f}",
               f"{r['calmar']-rp['calmar']:+.2f} vs pure" if mltpi_enabled else "")
    ra5.metric("Max Drawdown", f"{r['max_drawdown_pct']:.1f}%",
               f"{r['max_drawdown_pct']-rp['max_drawdown_pct']:+.1f}pp vs pure" if mltpi_enabled else "")

    with st.expander("What do these ratios mean?"):
        st.markdown("""
| Ratio | Formula | Interpretation |
|---|---|---|
| **Sharpe** | `mean(r) / std(r) × √252` | Risk-adjusted return; >1 is good, >2 is strong |
| **Sortino** | `mean(r) / downside_std × √252` | Like Sharpe but only penalises downside volatility |
| **Omega** | `sum(gains) / sum(losses)` | Ratio of all positive to all negative daily returns; >1 means more won than lost |
| **Calmar** | `ann_return / max_drawdown` | Return per unit of worst drawdown; >1 is solid |
| **Max DD** | `min((port - peak) / peak)` | Worst peak-to-trough loss as a percentage |
""")

    # ── three-way comparison (only when MLTPI active) ────────────────────────
    if mltpi_enabled and active_mltpi is not None:
        st.divider()
        st.subheader("Three-Way Equity Comparison")

        eq_blend = res.equity_curve
        eq_pure  = res_pure.equity_curve

        # MLTPI-only: long when H(α) > 0, flat otherwise
        @st.cache_data(show_spinner="Computing MLTPI-only backtest…")
        def _mltpi_only_bt(n: int, cash: float, start: str, mw: float):
            sig   = st.session_state.get("active_mltpi")
            if sig is None:
                return None
            mz    = (sig * 3).rename("mltpi_z")
            # Use MLTPI as sole indicator; RAQQR weight 0
            extra = [sc.Indicator("MLTPI", mz, weight=5.0)]
            return sc.backtest_curve(ohlcv_raw, starting_cash=cash, start=start,
                                     extra_indicators=extra)

        res_mltpi_only = _mltpi_only_bt(len(ohlcv_raw), float(starting_cash),
                                         str(backtest_start), mltpi_weight)

        lump_btc   = float(starting_cash) / float(eq_blend["price"].iloc[0])
        lump_line  = lump_btc * eq_blend["price"]

        tway = go.Figure()
        tway.add_trace(go.Scatter(x=eq_pure.index, y=eq_pure["portfolio"],
                                  mode="lines", name="Pure RAQQR",
                                  line=dict(color="#58a6ff", width=1.4, dash="dash")))
        if res_mltpi_only:
            tway.add_trace(go.Scatter(
                x=res_mltpi_only.equity_curve.index,
                y=res_mltpi_only.equity_curve["portfolio"],
                mode="lines", name="MLTPI only",
                line=dict(color="#f97316", width=1.4, dash="dot")))
        tway.add_trace(go.Scatter(x=eq_blend.index, y=eq_blend["portfolio"],
                                  mode="lines", name="RAQQR + MLTPI",
                                  line=dict(color="#3fb950", width=2.0)))
        tway.add_trace(go.Scatter(x=lump_line.index, y=lump_line,
                                  mode="lines", name="Lump-sum hold",
                                  line=dict(color="rgba(255,255,255,0.3)",
                                            width=1.2, dash="dot")))
        tway.update_yaxes(type="log", tickprefix="$", tickformat=",.0f")
        tway.update_layout(template="plotly_dark", height=400,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02),
                           hovermode="x unified", margin=dict(t=40, b=40))
        st.plotly_chart(tway, use_container_width=True)

    # ── allocation curve ─────────────────────────────────────────────────────
    st.subheader(f"Allocation Curve — {preset_name} preset")
    curve_fig = go.Figure()

    curve_fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)", dash="dot"))
    curve_fig.add_trace(go.Scatter(
        x=CURVE_RISK_NODES, y=curve_values,
        mode="lines+markers",
        name=preset_name,
        line=dict(color="#58a6ff", width=2),
        marker=dict(size=7),
        hovertemplate="Risk %{x} → %{y:+.1f}% of cash/BTC<extra></extra>",
    ))

    curve_fig.add_vrect(x0=0,  x1=25, fillcolor="rgba(34,197,94,0.06)",  line_width=0)
    curve_fig.add_vrect(x0=25, x1=50, fillcolor="rgba(234,179,8,0.06)",  line_width=0)
    curve_fig.add_vrect(x0=50, x1=75, fillcolor="rgba(249,115,22,0.06)", line_width=0)
    curve_fig.add_vrect(x0=75, x1=100,fillcolor="rgba(239,68,68,0.06)",  line_width=0)

    curve_fig.update_layout(
        xaxis_title="Composite Risk (0–100)",
        yaxis_title="Trade rate (% of cash bought / % of BTC sold)",
        template="plotly_dark",
        height=280,
        margin=dict(t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(curve_fig, use_container_width=True)
    st.caption(
        "Positive values = buy that % of remaining cash. "
        "Negative values = sell that % of held BTC. "
        "Change the preset in the sidebar to reshape the curve and re-run the backtest."
    )

    # ── export ───────────────────────────────────────────────────────────────
    csv_buf = io.StringIO()
    eq.to_csv(csv_buf)
    st.download_button(
        "Download equity curve CSV",
        data=csv_buf.getvalue(),
        file_name="raqqr_equity_curve.csv",
        mime="text/csv",
    )
