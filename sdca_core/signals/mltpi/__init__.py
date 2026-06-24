"""MLTPI — Bayesian-trained multi-indicator trend signal layer.

Provides run_full_pipeline(ohlcv, isp, ...) -> pd.Series in [-1, 1],
which can be converted to a z-score via mltpi_z = signal * 3 and
injected into sdca_core.signals.composite as an Indicator.
"""
from .pipeline import run_full_pipeline, INDICATOR_NAMES

__all__ = ["run_full_pipeline", "INDICATOR_NAMES"]
