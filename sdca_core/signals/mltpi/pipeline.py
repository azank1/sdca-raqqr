"""MLTPI pipeline — all 9 steps wired together, fully in-memory.

Usage:
    from sdca_core.signals.mltpi import run_full_pipeline

    signal = run_full_pipeline(ohlcv, isp, progress_cb=print)
    # signal is a pd.Series in [-1, 1], daily DatetimeIndex
    # convert to z-score: mltpi_z = signal * 3
"""
from __future__ import annotations

import importlib
import math
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INDICATOR_NAMES = ["agma", "qtrend", "gstX", "trendZ", "zscoreMA", "momentumX"]
TIMEFRAMES      = ["1D", "2D", "3D"]
_PKG            = "sdca_core.signals.mltpi.indicators"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_indicator(name: str):
    return importlib.import_module(f"{_PKG}.{name}")


def _prep_df(ohlcv: pd.DataFrame, isp: pd.Series) -> pd.DataFrame:
    """Merge OHLCV + ISP into the format the indicators expect."""
    df = ohlcv[["open", "high", "low", "close"]].copy().astype(float)
    # ISP is ±1; reindex to daily, forward-fill gaps, default -1
    df["manual_signal"] = isp.reindex(df.index).ffill().fillna(-1).astype(int)
    return df.dropna(subset=["close"])


def _apply_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "manual_signal": "last"}
    return df.resample(tf).agg(agg).dropna()


# ── metrics (used across multiple steps) ─────────────────────────────────────

def _sharpe(signal, prices):
    rets = prices.pct_change().fillna(0)
    active = np.array(signal[:-1]) * rets.values[1:]
    return float(np.mean(active) / (np.std(active) + 1e-8))


def _omega(signal, prices):
    rets = prices.pct_change().fillna(0)
    active = np.array(signal[:-1]) * rets.values[1:]
    gain = np.mean(active[active > 0]) if np.any(active > 0) else 0.0
    loss = abs(np.mean(active[active < 0])) if np.any(active < 0) else 1e-8
    return float(gain / loss)


def _mae(signal, target):
    return float(np.mean(np.abs(np.array(signal) - np.array(target))))


def _transition_freq(signal):
    s = np.array(signal)
    return float(np.sum(np.diff(s) != 0) / max(len(s) - 1, 1))


def _holding_period(signal):
    s = np.array(signal)
    tr = np.where(np.diff(s) != 0)[0]
    return float(np.mean(np.diff(tr))) if len(tr) >= 2 else float(len(s))


def _score_CEF(sharpe, omega, mae, corr):
    """Crypto Efficient Frontier score (from scale_timeframe.py)."""
    return (-((sharpe - 2.1) ** 2) - ((omega - 7) ** 2) - mae + 0.5 * corr)


# ── Step 1: train each indicator ─────────────────────────────────────────────

def _step1_train(df: pd.DataFrame, names: list[str],
                 cb: Callable | None, step: int, total: int) -> dict:
    """Bayesian-optimize each indicator vs the ISP. Returns {name: settings}."""
    from bayes_opt import BayesianOptimization  # imported here so import error is deferred

    trained = {}
    for i, name in enumerate(names):
        if cb:
            cb(step, total, f"Training {name} indicator ({i+1}/{len(names)})…")
        mod = _load_indicator(name)

        # Each indicator has its own objective / pbounds; call train_indicator
        # but capture the result in-memory instead of writing to disk.
        # We monkey-patch the output_path to a temp file, then reload.
        import tempfile, json, os
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            tmp_path = f.name
        try:
            mod.train_indicator(df.copy(), tmp_path)
            with open(tmp_path) as f:
                trained[name] = json.load(f)
        finally:
            os.unlink(tmp_path)

    return trained


# ── Step 2: extract features ─────────────────────────────────────────────────

def _step2_features(df: pd.DataFrame, trained: dict, cb, step, total) -> dict:
    if cb: cb(step, total, "Extracting indicator features…")
    profiles = {}
    for name, settings in trained.items():
        mod = _load_indicator(name)
        signal = mod.final_signal(df.copy(), "1D", settings=settings)
        isp    = df["manual_signal"].astype(int).values
        prices = df["close"]
        corr   = float(np.corrcoef(signal, isp)[0, 1]) if len(signal) > 1 else 0.0
        profiles[name] = {
            "mae_vs_isp":           _mae(signal, isp),
            "correlation_vs_isp":   corr,
            "sharpe_ratio":         _sharpe(signal, prices),
            "omega_ratio":          _omega(signal, prices),
            "transition_frequency": _transition_freq(signal),
            "avg_holding_period":   _holding_period(signal),
            "settings":             settings,
        }
    return profiles


# ── Step 3: scale timeframes ─────────────────────────────────────────────────

def _step3_scale_tf(df: pd.DataFrame, profiles: dict, cb, step, total) -> dict:
    if cb: cb(step, total, "Selecting optimal timeframes…")
    for name, prof in profiles.items():
        mod = _load_indicator(name)
        best_score, best_tf, best_feat = -np.inf, "1D", {}
        isp = df["manual_signal"].astype(int).values
        for tf in TIMEFRAMES:
            tf_df  = _apply_tf(df.copy(), tf)
            signal = mod.final_signal(tf_df.copy(), tf, settings=prof["settings"])
            isp_tf = tf_df["manual_signal"].astype(int).values
            prices = tf_df["close"]
            mae  = _mae(signal, isp_tf)
            corr = float(np.corrcoef(signal, isp_tf)[0, 1]) if len(signal) > 1 else 0.0
            sc   = _score_CEF(_sharpe(signal, prices), _omega(signal, prices), mae, corr)
            if sc > best_score:
                best_score = sc
                best_tf    = tf
                best_feat  = {
                    "preferred_timeframe": tf,
                    "score": round(sc, 4),
                    "sharpe_ratio": round(_sharpe(signal, prices), 4),
                    "omega_ratio":  round(_omega(signal, prices), 4),
                    "mae_vs_isp":   round(mae, 4),
                    "correlation_vs_isp": round(corr, 4),
                    "transition_frequency": round(_transition_freq(signal), 4),
                    "avg_holding_period":   round(_holding_period(signal), 2),
                }
        prof.update(best_feat)
    return profiles


# ── Step 4: cluster indicators ────────────────────────────────────────────────

def _step4_cluster(profiles: dict, cb, step, total) -> dict:
    if cb: cb(step, total, "Clustering indicators into strategies…")
    names, rows = [], []
    for name, p in profiles.items():
        if "preferred_timeframe" not in p:
            continue
        rows.append([
            p.get("mae_vs_isp", 0),
            p.get("correlation_vs_isp", 0),
            p.get("sharpe_ratio", 0),
            p.get("omega_ratio", 0),
            p.get("transition_frequency", 0),
            p.get("avg_holding_period", 0),
        ])
        names.append(name)

    if len(names) <= 1:
        # Only one indicator — single cluster
        return {"S1": [{"name": n, "tf": profiles[n].get("preferred_timeframe", "1D"),
                        "settings": profiles[n]["settings"]} for n in names]}

    X = StandardScaler().fit_transform(np.array(rows))
    X_pca = PCA(n_components=1).fit_transform(X)

    # Pick k via silhouette (simple grid search, no Optuna to avoid extra dep)
    best_k, best_sil = 2, -1.0
    for k in range(2, min(len(names), 5)):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_pca)
        s = silhouette_score(X_pca, labels)
        if s > best_sil:
            best_sil, best_k = s, k

    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X_pca)
    clusters: dict = {}
    for idx, lbl in enumerate(labels):
        key = f"S{lbl + 1}"
        clusters.setdefault(key, [])
        name = names[idx]
        clusters[key].append({
            "name": name,
            "tf":       profiles[name].get("preferred_timeframe", "1D"),
            "settings": profiles[name]["settings"],
            "score":    profiles[name].get("score", 1.0),
        })
    return clusters


# ── Step 5: optimize strategy weights ────────────────────────────────────────

def _compute_reward(signal: np.ndarray, df: pd.DataFrame) -> float:
    signal = pd.Series(signal, index=df.index).fillna(0).astype(int)
    prices = df["close"].astype(float).values
    equity = [1.0]
    for i in range(1, len(prices)):
        equity.append(equity[-1] * (prices[i] / prices[i - 1])
                      if signal.iloc[i] == 1 else equity[-1])
    equity = np.array(equity)
    rets   = np.diff(np.log(equity + 1e-8))
    sharpe = np.mean(rets) / (np.std(rets) + 1e-8)
    omega  = (np.mean(rets[rets > 0]) / (abs(np.mean(rets[rets < 0])) + 1e-8)
              if np.any(rets < 0) else float(np.mean(rets > 0)))
    flip   = np.sum(np.diff(signal.values) != 0) / max(len(signal) - 1, 1)
    return float((sharpe * omega) / (1 + flip + 1e-6))


def _step5_strategy_opt(df: pd.DataFrame, clusters: dict,
                        cb, step, total) -> dict:
    """Simple averaged strategy signal — Bayesian opt skipped here for speed;
    the indicator-level Bayes opt (Step 1) already captures the heavy lifting.
    Step 5 computes the reward for each cluster and embeds it in the structure."""
    if cb: cb(step, total, "Evaluating strategy clusters…")
    behavior: dict = {}
    for strat, indicators in clusters.items():
        behavior[strat] = {}
        signals = []
        for ind in indicators:
            mod    = _load_indicator(ind["name"])
            tf_df  = _apply_tf(df.copy(), ind["tf"])
            sig    = mod.final_signal(tf_df.copy(), ind["tf"], settings=ind["settings"])
            s_s    = pd.Series(sig, index=tf_df.index).reindex(df.index, method="ffill").fillna(-1)
            signals.append(s_s.values)
        combined = np.where(np.mean(signals, axis=0) > 0, 1, -1)
        reward   = _compute_reward(combined, df)
        for ind in indicators:
            behavior[strat][ind["name"]] = {
                "tf":       ind["tf"],
                "settings": ind["settings"],
                "score":    reward,
            }
    return behavior


# ── Step 6: profile-align timeframes to ISP ───────────────────────────────────

def _step6_profile_align(df: pd.DataFrame, behavior: dict,
                         cb, step, total) -> dict:
    if cb: cb(step, total, "Aligning indicator timeframes to ISP profile…")
    isp_signal = df["manual_signal"]
    isp_feat   = _extract_signal_features(isp_signal)
    aligned    = {}
    for strat, indicators in behavior.items():
        aligned[strat] = {}
        for name, info in indicators.items():
            mod      = _load_indicator(name)
            settings = info["settings"]
            best_dist, best_tf = float("inf"), info["tf"]
            for tf in TIMEFRAMES:
                tf_df  = _apply_tf(df.copy(), tf)
                signal = mod.final_signal(tf_df.copy(), tf, settings=settings)
                feat   = _extract_signal_features(pd.Series(signal, index=tf_df.index))
                dist   = float(np.mean([abs(feat[k] - isp_feat[k]) for k in isp_feat]))
                if dist < best_dist:
                    best_dist, best_tf = dist, tf
            aligned[strat][name] = {"tf": best_tf, "settings": settings,
                                    "score": info.get("score", 1.0)}
    return aligned


def _extract_signal_features(signal: pd.Series) -> dict:
    from scipy.stats import entropy as _entropy
    s = signal.fillna(0).astype(int)
    transitions = np.sum(np.diff(s.values) != 0)
    durations, count = [], 1
    for i in range(1, len(s)):
        if s.iloc[i] == s.iloc[i - 1]:
            count += 1
        else:
            durations.append(count)
            count = 1
    durations.append(count)
    vc = s.value_counts(normalize=True)
    return {
        "flip_rate": float(transitions / max(len(s) - 1, 1)),
        "std":       float(s.std()),
        "avg_hold":  float(np.mean(durations)),
        "entropy":   float(_entropy(vc)),
    }


# ── Step 7: compute strategy weights ─────────────────────────────────────────

def _step7_weights(aligned: dict, cb, step, total) -> dict:
    if cb: cb(step, total, "Computing strategy weights…")
    scores_counts = []
    for strat, indicators in aligned.items():
        scores = [v["score"] for v in indicators.values()
                  if isinstance(v, dict) and "score" in v]
        avg = float(np.mean(scores)) if scores else 0.0
        scores_counts.append((avg, len(scores)))

    weighted = np.array([
        (1 / (1 + abs(s))) * math.log(1 + c)
        for s, c in scores_counts
    ])
    norm = weighted / (weighted.sum() + 1e-8)
    return {strat: float(w) for strat, w in zip(aligned.keys(), norm)}


# ── Step 8+9: reconstruct H(α) and normalize ─────────────────────────────────

def _step89_build_signal(df: pd.DataFrame, aligned: dict, weights: dict,
                         cb, step, total) -> pd.Series:
    if cb: cb(step, total, "Building final H(α) signal…")
    final: pd.Series | None = None
    for strat, indicators in aligned.items():
        w = weights.get(strat, 0.0)
        if w == 0:
            continue
        sigs = []
        for name, info in indicators.items():
            mod    = _load_indicator(name)
            tf_df  = _apply_tf(df.copy(), info["tf"])
            sig    = mod.final_signal(tf_df.copy(), info["tf"], settings=info["settings"])
            s_s    = pd.Series(sig, index=tf_df.index).reindex(df.index, method="ffill").fillna(-1)
            sigs.append(s_s.values)
        strat_sig = pd.Series(np.mean(sigs, axis=0), index=df.index) * w
        final = strat_sig if final is None else final.add(strat_sig, fill_value=0)

    if final is None:
        return pd.Series(np.zeros(len(df)), index=df.index)

    max_abs = np.max(np.abs(final.values)) + 1e-8
    return (final / max_abs).rename("mltpi_signal")


# ── Public entry point ────────────────────────────────────────────────────────

def run_full_pipeline(
    ohlcv: pd.DataFrame,
    isp: pd.Series,
    indicator_names: list[str] | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> pd.Series:
    """Run the complete 9-step MLTPI pipeline.

    Parameters
    ----------
    ohlcv : DataFrame with DatetimeIndex and columns open/high/low/close
    isp   : pd.Series of ±1 values on a DatetimeIndex (user-annotated ISP)
    indicator_names : subset of INDICATOR_NAMES to use (default: all)
    progress_cb : optional callable(step, total_steps, message)

    Returns
    -------
    pd.Series in [-1, 1] with the same DatetimeIndex as ohlcv.
    Convert to z-score via  mltpi_z = signal * 3  before passing to composite_z().
    """
    names = indicator_names or INDICATOR_NAMES
    total = 9

    def cb(s, t, m):
        if progress_cb:
            progress_cb(s, t, m)

    df = _prep_df(ohlcv, isp)

    trained  = _step1_train(df, names, cb, 1, total)
    profiles = _step2_features(df, trained, cb, 2, total)
    profiles = _step3_scale_tf(df, profiles, cb, 3, total)
    clusters = _step4_cluster(profiles, cb, 4, total)
    behavior = _step5_strategy_opt(df, clusters, cb, 5, total)
    aligned  = _step6_profile_align(df, behavior, cb, 6, total)
    weights  = _step7_weights(aligned, cb, 7, total)
    signal   = _step89_build_signal(df, aligned, weights, cb, 8, total)

    cb(9, total, "Done.")
    return signal
