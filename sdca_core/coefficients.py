"""RAQQR coefficients — single source of truth.

Transcribed verbatim from the Bitcoin Asymmetric Tail Curvature Rainbow artifact
(Cowen 2026, Table 3, full-sample fit). These are the only numbers that define
the valuation surface; everything else is derived from them. The web frontend
must read these same values (see parity/ harness) so the two implementations
cannot drift.

Model:  Q_tau(log10 P) = c + a*x + b*x**2,  x = ln(t) - MU,  t = days since 2009-01-01
Curvature b is tied across tail groups: bLO (Q1/Q10/Q25), bMED (Q50),
bHI (Q75/Q95/Q99).

NOTE ON LOOK-AHEAD: these are full-sample fitted coefficients, so a band value at
any historical date embeds information from the entire fit window. Backtests built
on them are in-sample by construction. This is preserved deliberately for parity
with the frontend; it is not a bug, but it is a property to disclose.
"""

ONE_DAY_MS = 86_400_000
RAQQR_MU = 7.9914
# Jan 1 2009 00:00:00 UTC anchor for the Table-3 coefficients.
RAQQR_GENESIS_MS = 1_230_768_000_000  # Date.UTC(2009, 0, 1)

# Quantile keys in nominal order, and the risk mark (%) each maps to.
RAQQR_KEYS = ["0.01", "0.1", "0.25", "0.5", "0.75", "0.95", "0.99"]
RAQQR_RISK_MARKS = [1, 10, 25, 50, 75, 95, 99]

# key -> (c, a, b)
RAQQR_COEF = {
    "0.01": (2.837, 2.578, -0.0241),
    "0.1":  (2.933, 2.552, -0.0241),
    "0.25": (3.004, 2.554, -0.0241),
    "0.5":  (3.214, 2.482, -0.1126),
    "0.75": (3.562, 2.283, -0.3259),
    "0.95": (3.897, 1.964, -0.3259),
    "0.99": (4.028, 1.904, -0.3259),
}
