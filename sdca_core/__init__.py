"""sdca_core — Python parity port of the Bitcoin Asymmetric Tail Curvature
(RAQQR) Rainbow valuation model and its accumulation/distribution DCA backtest.

The numbers produced here match the HTML artifact's default configuration. See
the parity harness in parity/ for the cross-implementation check.
"""
from .api import analyze, backtest_curve
from .models.raqqr import raqqr_bands, days_since_genesis
from .signals.eqm import eqm_risk, eqm_zscore
from .signals.composite import (
    Indicator, composite_z, composite_risk_from_z, cqm_z_from_risk,
)
from .backtest.curve import (
    run_curve_backtest, curve_value_at_risk,
    CURVE_RISK_NODES, CURVE_DEFAULT_VALUES, BacktestResult,
)

__version__ = "0.1.0"
__all__ = [
    "analyze", "backtest_curve", "raqqr_bands", "days_since_genesis",
    "eqm_risk", "eqm_zscore", "Indicator", "composite_z",
    "composite_risk_from_z", "cqm_z_from_risk", "run_curve_backtest",
    "curve_value_at_risk", "CURVE_RISK_NODES", "CURVE_DEFAULT_VALUES",
    "BacktestResult",
]

from . import data  # noqa: E402
