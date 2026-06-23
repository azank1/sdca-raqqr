"""Cross-implementation parity: Python band prices vs. the JS golden output.

The golden file is produced by the verbatim artifact band function
(parity/dump_js_bands.mjs). Matching it to floating-point tolerance proves the
Python core reproduces the frontend's valuation surface exactly.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdca_core.models.raqqr import raqqr_bands
from sdca_core.coefficients import RAQQR_KEYS

GOLDEN = Path(__file__).resolve().parent.parent / "parity" / "golden_bands.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


def test_band_prices_match_js(golden):
    dates = list(golden.keys())
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    bands = raqqr_bands(idx)
    py = bands[RAQQR_KEYS].to_numpy()
    js = np.array([golden[d] for d in dates])
    # JS sorts ascending; our columns are ascending too.
    assert py.shape == js.shape
    np.testing.assert_allclose(py, js, rtol=1e-12, atol=1e-9)


def test_low_high_rail_are_outer_bands(golden):
    dates = list(golden.keys())
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    bands = raqqr_bands(idx)
    js = np.array([golden[d] for d in dates])
    np.testing.assert_allclose(bands["lowRail"].to_numpy(), js[:, 0], rtol=1e-12)
    np.testing.assert_allclose(bands["highRail"].to_numpy(), js[:, -1], rtol=1e-12)


def test_bands_are_monotone_nondecreasing(golden):
    idx = pd.DatetimeIndex(pd.to_datetime(list(golden.keys())))
    arr = raqqr_bands(idx)[RAQQR_KEYS].to_numpy()
    assert (np.diff(arr, axis=1) >= -1e-9).all()
