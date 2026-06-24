"""MLTPI Signal Builder — interactive ISP annotation + training pipeline.

Page 2 of the RAQQR Dashboard. Users annotate entry/exit trades on a
live candlestick chart (max 40 trades over a 3-year window), then trigger
Bayesian-optimised MLTPI training. The resulting H(α) signal is stored in
session state and available in the main dashboard as an extra indicator.

Phases:
  A. Annotate ISP (this file, Part 1)
  B. Training with progress feedback
  C. Signal review, walk-forward validation, and export/apply
"""
from __future__ import annotations

import io
import json
import sys
import os
import threading
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sdca_core as sc
from sdca_core.signals.mltpi import run_full_pipeline, INDICATOR_NAMES
from sdca_core.backtest.metrics import compute_ratios

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLTPI Signal Builder",
    page_icon="🧠",
    layout="wide",
)

MAX_TRADES   = 40
WINDOW_YEARS = 3

# ── session state defaults ────────────────────────────────────────────────────
for key, val in {
    "sb_trades":       [],     # [{date: str, type: "entry"|"exit"}]
    "sb_mode":         "entry",
    "sb_training":     None,   # None | {"status": ..., "step": ..., "msg": ..., "signal": ...}
    "sb_signal":       None,   # pd.Series in [-1,1] after training
    "sb_settings":     None,   # dict of trained settings for download
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Fetching BTC data…")
def _fetch() -> tuple[pd.DataFrame, str]:
    try:
        return sc.data.load_binance("BTCUSDT"), "Binance"
    except Exception:
        return sc.data.load_yfinance("BTC-USD"), "Yahoo Finance"

ohlcv_raw, data_src = _fetch()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Signal Builder")
    st.caption("MLTPI ISP annotation + training")
    st.divider()

    today     = ohlcv_raw.index[-1].date()
    win_start = today - timedelta(days=WINDOW_YEARS * 365)
    ann_start = st.date_input("Annotation window start",
                              value=win_start,
                              min_value=ohlcv_raw.index[0].date(),
                              max_value=today - timedelta(days=90))
    ann_end   = st.date_input("Annotation window end",
                              value=today,
                              min_value=ann_start + timedelta(days=90),
                              max_value=today)

    st.divider()
    st.subheader("Mark mode")
    mode_choice = st.radio("Click action", ["Entry (long ▲)", "Exit (short ▼)"], index=0)
    st.session_state["sb_mode"] = "entry" if "Entry" in mode_choice else "exit"

    st.divider()
    used   = len(st.session_state["sb_trades"])
    remaining = MAX_TRADES - used
    st.metric("Trades annotated", f"{used} / {MAX_TRADES}")
    if remaining <= 5:
        st.warning(f"Only {remaining} trades remaining!")

    if st.button("🗑 Clear all annotations", use_container_width=True):
        st.session_state["sb_trades"] = []
        st.session_state["sb_signal"] = None
        st.session_state["sb_training"] = None
        st.rerun()

    st.divider()
    st.subheader("Indicators")
    sel_indicators = st.multiselect(
        "Indicators to train",
        INDICATOR_NAMES,
        default=["agma", "qtrend", "gstX"],
    )

    st.divider()
    st.caption(f"Data via **{data_src}**")

# ── filter to annotation window ───────────────────────────────────────────────
mask  = (ohlcv_raw.index.date >= ann_start) & (ohlcv_raw.index.date <= ann_end)
ohlcv = ohlcv_raw[mask].copy()

if len(ohlcv) < 30:
    st.error("Annotation window too short. Expand the date range.")
    st.stop()

# ── ISP helpers ───────────────────────────────────────────────────────────────
def trades_to_isp(trades: list[dict], index: pd.DatetimeIndex) -> pd.Series:
    """Convert annotated trades to a ±1 daily ISP series."""
    isp = pd.Series(-1, index=index, dtype=int)
    sorted_trades = sorted(trades, key=lambda t: t["date"])
    in_long = False
    for t in sorted_trades:
        d = pd.Timestamp(t["date"])
        if t["type"] == "entry" and not in_long:
            isp[isp.index >= d] = 1
            in_long = True
        elif t["type"] == "exit" and in_long:
            isp[isp.index >= d] = -1
            in_long = False
    return isp


def quality_score(trades: list[dict], ohlcv: pd.DataFrame) -> dict:
    """Compute ISP quality metrics."""
    isp = trades_to_isp(trades, ohlcv.index)
    price = ohlcv["close"]

    # Balance: fraction of days long
    long_frac = float((isp == 1).mean())

    # Regime coverage: detect rough bull/bear periods
    pct_chg = price.pct_change(90).dropna()
    bull_days  = int((pct_chg > 0.20).sum())
    bear_days  = int((pct_chg < -0.20).sum())
    total_days = len(pct_chg)
    bull_isp   = int(isp.reindex(pct_chg.index)[pct_chg > 0.20].eq(1).sum())
    bear_isp   = int(isp.reindex(pct_chg.index)[pct_chg < -0.20].eq(-1).sum())
    bull_cov = bull_isp / max(bull_days, 1)
    bear_cov = bear_days / max(total_days - bull_days, 1)  # rough bear coverage proxy

    # Spacing: any two adjacent trades within 5 days?
    dates = sorted([pd.Timestamp(t["date"]) for t in trades])
    min_gap = int(min((dates[i+1] - dates[i]).days for i in range(len(dates)-1))) if len(dates) > 1 else 999

    # Composite 0-100 score
    balance_score  = max(0, 1 - abs(long_frac - 0.5) * 4) * 40   # 40 pts
    bull_score     = min(bull_cov, 1.0) * 30                       # 30 pts
    spacing_score  = (30 if min_gap >= 5 else max(0, 30 - (5 - min_gap) * 6))  # 30 pts
    total          = balance_score + bull_score + spacing_score

    return dict(
        score=round(total, 1),
        long_frac=round(long_frac * 100, 1),
        bull_coverage=round(bull_cov * 100, 1),
        min_gap_days=min_gap,
        balance_score=round(balance_score, 1),
        spacing_score=round(spacing_score, 1),
    )

# ── header ────────────────────────────────────────────────────────────────────
st.title("🧠 MLTPI Signal Builder")
st.caption(
    "Draw your ideal entry/exit signal on the chart. The system trains "
    "Bayesian-optimised trend indicators to match your annotation, then blends "
    "the result into the RAQQR composite risk on the main dashboard."
)

phase_tab, review_tab, apply_tab = st.tabs(
    ["Phase A — Annotate", "Phase C — Review Signal", "Apply to Dashboard"]
)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE A — ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════
with phase_tab:

    # ── annotation controls row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    c1.info(f"Mode: **{'Entry ▲' if st.session_state['sb_mode'] == 'entry' else 'Exit ▼'}**")
    trades = st.session_state["sb_trades"]
    c2.metric("Trades", f"{len(trades)} / {MAX_TRADES}")
    if trades:
        q = quality_score(trades, ohlcv)
        col = "#22c55e" if q["score"] >= 70 else "#eab308" if q["score"] >= 40 else "#ef4444"
        c3.markdown(
            f"<div style='background:{col}22;border:1px solid {col};"
            f"border-radius:8px;padding:8px;text-align:center'>"
            f"<b>ISP quality: {q['score']}/100</b></div>",
            unsafe_allow_html=True,
        )
        c4.caption(
            f"Long {q['long_frac']}% · Bull cov {q['bull_coverage']}% · "
            f"Min gap {q['min_gap_days']}d"
        )

    # ── candlestick chart ──────────────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["open"],  high=ohlcv["high"],
        low=ohlcv["low"],    close=ohlcv["close"],
        name="BTC",
        increasing_line_color="#3fb950",
        decreasing_line_color="#ef4444",
    ))

    # ISP overlay shading
    if trades:
        isp = trades_to_isp(trades, ohlcv.index)
        long_blocks = isp == 1
        # shade long periods green
        in_block, block_start = False, None
        for dt, is_long in zip(ohlcv.index, long_blocks):
            if is_long and not in_block:
                block_start = dt
                in_block = True
            elif not is_long and in_block:
                fig.add_vrect(x0=block_start, x1=dt,
                              fillcolor="rgba(63,185,80,0.12)", line_width=0)
                in_block = False
        if in_block:
            fig.add_vrect(x0=block_start, x1=ohlcv.index[-1],
                          fillcolor="rgba(63,185,80,0.12)", line_width=0)

    # Trade markers
    for t in trades:
        d = pd.Timestamp(t["date"])
        if d not in ohlcv.index:
            continue
        price_at = float(ohlcv.loc[d, "close"])
        colour   = "#3fb950" if t["type"] == "entry" else "#ef4444"
        symbol   = "triangle-up" if t["type"] == "entry" else "triangle-down"
        fig.add_trace(go.Scatter(
            x=[d], y=[price_at * (0.97 if t["type"] == "entry" else 1.03)],
            mode="markers",
            marker=dict(symbol=symbol, size=14, color=colour),
            name=t["type"].capitalize(),
            showlegend=False,
            hovertemplate=f"<b>{t['type'].upper()}</b> @ {d.date()}<extra></extra>",
        ))

    fig.update_layout(
        title="Click a candle to add a trade marker",
        yaxis_type="log",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=500,
        margin=dict(t=60, b=40),
    )

    # Click handler — Streamlit built-in selection
    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="annot_chart")

    # Process click
    if event and event.get("selection") and event["selection"].get("points"):
        pt = event["selection"]["points"][0]
        clicked_date = str(pd.Timestamp(pt["x"]).date())

        # Don't duplicate
        existing_dates = {t["date"] for t in st.session_state["sb_trades"]}
        if clicked_date not in existing_dates and len(st.session_state["sb_trades"]) < MAX_TRADES:
            st.session_state["sb_trades"].append({
                "date": clicked_date,
                "type": st.session_state["sb_mode"],
            })
            st.rerun()

    # ── annotation table ──────────────────────────────────────────────────────
    if trades:
        with st.expander(f"Annotated trades ({len(trades)})"):
            df_trades = pd.DataFrame(trades).sort_values("date")
            df_trades.index = range(1, len(df_trades) + 1)
            st.dataframe(df_trades, use_container_width=True)

            # Remove individual trade
            remove_idx = st.number_input("Remove trade #", min_value=1,
                                         max_value=len(trades), step=1)
            if st.button("Remove selected trade"):
                sorted_t = sorted(trades, key=lambda t: t["date"])
                sorted_t.pop(remove_idx - 1)
                st.session_state["sb_trades"] = sorted_t
                st.rerun()

    # ── walk-forward info ─────────────────────────────────────────────────────
    st.info(
        "**Walk-forward split:** The first 80% of the annotation window is used "
        "for training; the final 20% is held out for out-of-sample validation. "
        "Both periods are shown after training."
    )

    # ── train button ──────────────────────────────────────────────────────────
    st.divider()
    can_train = (len(trades) >= 4 and
                 st.session_state["sb_training"] is None or
                 (st.session_state["sb_training"] or {}).get("status") != "running")

    if st.button("🚀 Train MLTPI Signal", type="primary",
                 disabled=len(trades) < 4, use_container_width=True):

        isp_full = trades_to_isp(trades, ohlcv.index)

        # Walk-forward split: train on first 80%
        split_idx = int(len(ohlcv) * 0.8)
        ohlcv_train = ohlcv.iloc[:split_idx]
        isp_train   = isp_full.iloc[:split_idx]

        def _bg_train(ohlcv_t, isp_t, indicators):
            st.session_state["sb_training"] = {"status": "running", "step": 0,
                                                "msg": "Starting…", "signal": None}
            def _cb(step, total, msg):
                st.session_state["sb_training"]["step"] = step
                st.session_state["sb_training"]["msg"]  = msg

            try:
                sig = run_full_pipeline(ohlcv_t, isp_t,
                                        indicator_names=indicators,
                                        progress_cb=_cb)
                # Extend signal to full window via forward-fill
                sig_full = sig.reindex(ohlcv.index, method="ffill").fillna(0)
                st.session_state["sb_signal"]   = sig_full
                st.session_state["sb_training"] = {"status": "done",
                                                    "step": 9, "msg": "Done!",
                                                    "signal": sig_full}
            except Exception as e:
                st.session_state["sb_training"] = {"status": "error",
                                                    "step": 0, "msg": str(e),
                                                    "signal": None}

        thread = threading.Thread(
            target=_bg_train,
            args=(ohlcv_train, isp_train, sel_indicators),
            daemon=True,
        )
        thread.start()
        st.rerun()

    # ── training progress ─────────────────────────────────────────────────────
    tr = st.session_state.get("sb_training")
    if tr and tr.get("status") == "running":
        step = tr.get("step", 0)
        msg  = tr.get("msg", "")
        st.progress(step / 9, text=f"Step {step}/9: {msg}")
        st.caption("Training takes 3–8 minutes. This page auto-refreshes.")
        time.sleep(3)
        st.rerun()
    elif tr and tr.get("status") == "error":
        st.error(f"Training failed: {tr['msg']}")
    elif tr and tr.get("status") == "done":
        st.success("Training complete! See the **Phase C — Review Signal** tab.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C — REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with review_tab:
    sig = st.session_state.get("sb_signal")

    if sig is None:
        st.info("No signal trained yet. Complete Phase A and click Train.")
    else:
        st.subheader("H(α) Signal vs BTC Price")

        split_date = ohlcv.index[int(len(ohlcv) * 0.8)]

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.6, 0.4], vertical_spacing=0.04,
                             subplot_titles=["BTC Price", "MLTPI H(α)"])

        fig2.add_trace(go.Scatter(x=ohlcv.index, y=ohlcv["close"],
                                  mode="lines", name="BTC",
                                  line=dict(color="white", width=1.2)), row=1, col=1)
        fig2.add_vline(x=split_date, line=dict(color="yellow", dash="dash", width=1),
                       annotation_text="train | validate", row=1, col=1)

        fig2.add_trace(go.Scatter(x=sig.index, y=sig.values,
                                  mode="lines", name="H(α)",
                                  line=dict(color="#58a6ff", width=1.4),
                                  fill="tozeroy",
                                  fillcolor="rgba(88,166,255,0.12)"), row=2, col=1)
        fig2.add_hline(y=0, line=dict(color="white", width=0.5, dash="dot"), row=2, col=1)
        fig2.add_vline(x=split_date, line=dict(color="yellow", dash="dash", width=1),
                       row=2, col=1)

        fig2.update_layout(template="plotly_dark", height=480,
                           margin=dict(t=60, b=40),
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        # Walk-forward backtest comparison
        st.subheader("Walk-Forward Backtest — MLTPI signal vs Buy & Hold")
        mltpi_z = (sig * 3).rename("mltpi_z")

        col_is, col_oos = st.columns(2)
        for label, start_d, end_d, col in [
            ("In-sample (train 80%)", ann_start, split_date.date(), col_is),
            ("Out-of-sample (validate 20%)", split_date.date(), ann_end, col_oos),
        ]:
            slice_ohlcv = ohlcv_raw[(ohlcv_raw.index.date >= start_d) &
                                     (ohlcv_raw.index.date <= end_d)]
            if len(slice_ohlcv) < 10:
                col.warning(f"{label}: insufficient data")
                continue
            slice_z = mltpi_z.reindex(slice_ohlcv.index).ffill().fillna(0)
            extra   = [sc.Indicator("MLTPI", slice_z, weight=1.0)]
            r       = sc.backtest_curve(slice_ohlcv, starting_cash=10_000,
                                        extra_indicators=extra)
            with col:
                st.markdown(f"**{label}**")
                st.metric("Return", f"{r.return_pct:.1f}%")
                st.metric("Sharpe",  f"{r.ratios['sharpe']:.2f}")
                st.metric("Sortino", f"{r.ratios['sortino']:.2f}")
                st.metric("Omega",   f"{r.ratios['omega']:.2f}")
                st.metric("Max DD",  f"{r.ratios['max_drawdown_pct']:.1f}%")

        # Download
        st.divider()
        csv_buf = io.StringIO()
        sig.to_csv(csv_buf, header=["mltpi_signal"])
        st.download_button("⬇ Download mltpi_signal.csv",
                           csv_buf.getvalue(),
                           file_name="mltpi_signal.csv",
                           mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# APPLY TO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with apply_tab:
    sig = st.session_state.get("sb_signal")

    if sig is None:
        st.info("Train a signal first (Phase A → Phase C).")
    else:
        st.success(
            "Signal is ready. Click below to activate it on the main dashboard. "
            "The MLTPI blend slider in the sidebar controls how much weight it carries."
        )
        if st.button("✅ Apply signal to dashboard", type="primary",
                     use_container_width=True):
            st.session_state["active_mltpi"] = sig
            st.balloons()
            st.success("Applied! Open the main dashboard and enable MLTPI in the sidebar.")

        st.divider()
        st.subheader("Or upload a previously saved signal")
        uploaded = st.file_uploader("Upload mltpi_signal.csv", type="csv")
        if uploaded:
            loaded = pd.read_csv(uploaded, index_col=0, parse_dates=True).squeeze()
            st.session_state["active_mltpi"] = loaded
            st.session_state["sb_signal"]    = loaded
            st.success(f"Loaded {len(loaded)} days of signal.")
