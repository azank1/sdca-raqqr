"""Unit tests for signals and the curve backtest, checked against independent
reference implementations transcribed from the artifact.
"""
import math

import numpy as np
import pandas as pd
import pytest

from sdca_core.models.raqqr import raqqr_bands
from sdca_core.signals.eqm import eqm_risk, eqm_zscore
from sdca_core.signals.composite import (
    Indicator, composite_z, composite_risk_from_z, cqm_z_from_risk,
)
from sdca_core.backtest.curve import (
    curve_value_at_risk, run_curve_backtest,
    CURVE_RISK_NODES, CURVE_DEFAULT_VALUES,
)
from sdca_core.coefficients import RAQQR_KEYS, RAQQR_RISK_MARKS


def _ref_eqm_risk(levels, price):
    marks = RAQQR_RISK_MARKS
    if price < levels[0]:
        return 1.0
    if price > levels[-1]:
        return 99.0
    lp = math.log(price)
    for i in range(len(levels) - 1):
        if levels[i] <= price <= levels[i + 1]:
            lo, hi = math.log(levels[i]), math.log(levels[i + 1])
            f = (lp - lo) / (hi - lo)
            return min(99.0, max(1.0, marks[i] + f * (marks[i + 1] - marks[i])))
    return 50.0


def _ref_eqm_z(low, mid, high, price):
    lp, ll, lm, lh = map(math.log, (price, low, mid, high))
    if lp <= lm:
        return min(3.0, max(0.0, 3 * (lm - lp) / (lm - ll)))
    return max(-3.0, min(0.0, -3 * (lp - lm) / (lh - lm)))


@pytest.fixture(scope="module")
def table():
    idx = pd.date_range("2015-01-01", "2026-01-01", freq="D")
    bands = raqqr_bands(idx)
    # Synthetic price path that wanders through the bands so all branches hit.
    mid = bands["0.5"].to_numpy()
    wob = 1 + 0.6 * np.sin(np.linspace(0, 30, len(idx)))
    price = pd.Series(mid * wob, index=idx, name="close")
    return bands, price


def test_eqm_risk_matches_reference(table):
    bands, price = table
    got = eqm_risk(bands, price).to_numpy()
    lvl = bands[RAQQR_KEYS].to_numpy()
    ref = np.array([_ref_eqm_risk(lvl[i], price.iloc[i]) for i in range(len(price))])
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-9)


def test_eqm_z_matches_reference(table):
    bands, price = table
    got = eqm_zscore(bands, price).to_numpy()
    low = bands["lowRail"].to_numpy(); mid = bands["0.5"].to_numpy(); high = bands["highRail"].to_numpy()
    ref = np.array([_ref_eqm_z(low[i], mid[i], high[i], price.iloc[i]) for i in range(len(price))])
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-9)


def test_composite_defaults_to_eqm_z(table):
    bands, price = table
    ez = eqm_zscore(bands, price)
    cz = composite_z([Indicator("price", ez, 1.0)])
    np.testing.assert_allclose(cz.to_numpy(), ez.to_numpy(), equal_nan=True)


def test_composite_risk_and_cqm_maps():
    z = pd.Series([3.0, 0.0, -3.0, 1.5, -1.5])
    r = composite_risk_from_z(z)
    np.testing.assert_allclose(r.to_numpy(), [0, 50, 100, 25, 75])
    # cqm_z is the inverse-style map of risk back to z
    np.testing.assert_allclose(cqm_z_from_risk(r).to_numpy(), [3, 0, -3, 1.5, -1.5])


def test_curve_value_endpoints_and_interp():
    assert curve_value_at_risk(0) == 10
    assert curve_value_at_risk(100) == -10
    assert curve_value_at_risk(50) == 0
    # halfway between node 0 (10) and node 5%? nodes are every 5; 2.5 -> 10
    assert curve_value_at_risk(2.5) == 10
    # between 20% (3.5) and 25% (2.0): midpoint 22.5 -> 2.75
    assert curve_value_at_risk(22.5) == pytest.approx(2.75)


def test_curve_backtest_compounds_against_cash():
    # Flat price, constant low risk -> buy 10% of *remaining* cash each day.
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    price = pd.Series(100.0, index=idx)
    risk = pd.Series(0.0, index=idx)  # curve = +10%/day
    res = run_curve_backtest(price, risk, starting_cash=1000.0)
    # cash after n days = 1000 * 0.9**5 ; portfolio stays ~1000 at flat price
    assert res.cash == pytest.approx(1000 * 0.9 ** 5, rel=1e-9)
    assert res.portfolio_value == pytest.approx(1000.0, rel=1e-9)
    assert res.buy_days == 5


def test_curve_backtest_lump_benchmark():
    idx = pd.date_range("2020-01-01", periods=3, freq="D")
    price = pd.Series([100.0, 100.0, 200.0], index=idx)
    risk = pd.Series(50.0, index=idx)  # flat zone, no trades
    res = run_curve_backtest(price, risk, starting_cash=1000.0)
    assert res.buy_days == 0 and res.sell_days == 0
    # lump: 10 BTC * 200 = 2000
    assert res.lump_value == pytest.approx(2000.0)
