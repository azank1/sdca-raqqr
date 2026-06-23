"""OHLCV loaders. CSV is the portable path; Binance is a convenience fetch.

All loaders return a DataFrame with a tz-naive UTC DatetimeIndex (one row per day)
and at least a ``close`` column, sorted ascending, positive closes only.
"""
from __future__ import annotations

import pandas as pd


def _normalize(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], utc=True).dt.tz_localize(None)
    out = out.rename(columns={close_col: "close"})
    out = out[["close"] + [c for c in out.columns if c not in (date_col, "close")]]
    out.index = pd.DatetimeIndex(df[date_col].pipe(pd.to_datetime, utc=True).dt.tz_localize(None))
    out = out[pd.to_numeric(out["close"], errors="coerce") > 0]
    out["close"] = out["close"].astype(float)
    return out.sort_index()


def load_csv(path, date_col="date", close_col="close") -> pd.DataFrame:
    """Load OHLCV from a CSV. Auto-detects common column names if defaults miss."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if date_col not in df.columns:
        for cand in ("date", "time", "timestamp", "datetime", "day"):
            if cand in cols:
                date_col = cols[cand]
                break
    if close_col not in df.columns:
        for cand in ("close", "price", "close_usd", "adj close", "adj_close"):
            if cand in cols:
                close_col = cols[cand]
                break
    return _normalize(df, date_col, close_col)


def load_binance(symbol="BTCUSDT", interval="1d", limit=1000) -> pd.DataFrame:
    """Fetch daily klines from Binance. Paginates back via endTime to cover full
    history. Requires network access (won't work in a sandbox)."""
    import requests

    base = "https://api.binance.com/api/v3/klines"
    rows, end = [], None
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end is not None:
            params["endTime"] = end
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows = batch + rows
        end = batch[0][0] - 1
        if len(batch) < 1000 or len(rows) >= 20000:
            break
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbav", "tbqv", "ignore"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df[["date", "open", "high", "low", "close", "volume"]].drop_duplicates("date")
    return _normalize(df, "date", "close")
