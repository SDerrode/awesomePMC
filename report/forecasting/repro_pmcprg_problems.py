"""
repro_pmcprg_problems.py — minimal reproductions of the three pmcprg problems that
stopped the forecasting study (forecasting). Each prints the library value and
the exact (or self-consistent) one; after a fix the two agree.

    PYTHONPATH=. .venv/bin/python report/forecasting/repro_pmcprg_problems.py

Status: all three are fixed at pmcprg 39f249f; its output there is recorded
in README.md ("P1–P3: fixed at 39f249f").

P1  gaps quadrature: strong copula dependence → forecasts / PIT after a missing
    row collapse (predictive sd 0 instead of 0.02 at ρ = 0.9999), silently.
P2  predictive_pit / flag_outliers: NaN PIT on every row after the first when a
    row of the PMC prior p is all zero (an emptied state) → nothing flagged.
P3  forecast (and impute) quantiles far from the law whose mean and sd are
    returned (median 15.8 for a law of mean 4.26, sd 1.42), silently.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
from scipy.stats import norm

from pmcprg.pmc import PMCModel, flag_outliers, forecast, predictive_pit
from pmcprg.pmc.gaps import reference_grid

logging.getLogger("pmcprg").setLevel(logging.ERROR)


def ar1_as_pmc(rho: float, p=((0.45, 0.05), (0.05, 0.45))) -> PMCModel:
    """Two identical regimes N(0, 1), Gaussian copula ρ on every pair: Y is the AR(1)(ρ)."""
    tau = 2 / math.pi * math.asin(rho)
    return PMCModel.from_dict({
        "model": {"name": "AR(1)", "K": 2, "variant": "PMC", "N_default": 10},
        "prior": {"p": [list(r) for r in p]},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}} for i in range(2)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]})


def p1() -> None:
    print("P1 — forecast / predictive_pit after a missing row, AR(1)(ρ) as a PMC (G = 64 nodes):")
    for rho in (0.99, 0.998, 0.999, 0.9999):
        M = ar1_as_pmc(rho)
        f = forecast(M, np.array([0.3]), 2)             # h = 2 crosses one unobserved row
        sd_exact = math.sqrt(1 - rho ** 4)
        pit = predictive_pit(M, np.array([0.3, np.nan, 0.31])).pit[2]
        pit_exact = norm.cdf(0.31, rho ** 2 * 0.3, sd_exact)
        print(f"   ρ = {rho}: 2-step sd {f.sd[1]:.4f} (exact {sd_exact:.4f}); "
              f"PIT after a gap {pit:.4f} (exact {pit_exact:.4f})")


def p2() -> None:
    print("P2 — predictive_pit / flag_outliers with an all-zero row of p (8 sd spike at row 100):")
    y = np.random.default_rng(0).normal(size=200)
    y[100] = 8.0
    for p in (((0.0, 0.0), (0.0, 1.0)), ((1e-12, 1e-12), (1e-12, 1 - 3e-12))):
        M = ar1_as_pmc(0.3, p)
        pp = predictive_pit(M, y)
        print(f"   p = {[list(r) for r in p]}: NaN PITs {int(np.isnan(pp.pit).sum())}/200, "
              f"log_lik {pp.log_lik:.3f}, flagged {flag_outliers(M, y).n_flagged}")


P3_MODEL = json.loads("""{"model": {"name": "pmc_pair K=3", "K": 3, "variant": "PMC", "margin_structure": "pair"},
 "prior": {"p": [[0.1684, 0.0393, 0.0124], [0.0393, 0.6546, 0.0368], [0.0124, 0.0368, 0.0]]},
 "margins": [{"i": 0, "j": 0, "dist": "norm", "params": {"loc": 2.994, "scale": 1.079}},
  {"i": 0, "j": 1, "dist": "norm", "params": {"loc": 3.665, "scale": 1.033}},
  {"i": 0, "j": 2, "dist": "norm", "params": {"loc": 3.046, "scale": 1.109}},
  {"i": 1, "j": 0, "dist": "norm", "params": {"loc": 3.58, "scale": 1.231}},
  {"i": 1, "j": 1, "dist": "norm", "params": {"loc": 4.274, "scale": 0.836}},
  {"i": 1, "j": 2, "dist": "norm", "params": {"loc": 4.332, "scale": 0.869}},
  {"i": 2, "j": 0, "dist": "norm", "params": {"loc": 9.629, "scale": 1.032}},
  {"i": 2, "j": 1, "dist": "norm", "params": {"loc": 10.841, "scale": 0.886}},
  {"i": 2, "j": 2, "dist": "norm", "params": {"loc": 5.125, "scale": 0.588}}],
 "copulas": [{"i": 0, "j": 0, "name": "GH", "tau": 0.71}, {"i": 0, "j": 1, "name": "GH", "tau": 0.602},
  {"i": 0, "j": 2, "name": "GH", "tau": 0.606}, {"i": 1, "j": 0, "name": "GH", "tau": 0.608},
  {"i": 1, "j": 1, "name": "GH", "tau": 0.889}, {"i": 1, "j": 2, "name": "GH", "tau": 0.881},
  {"i": 2, "j": 0, "name": "Frank", "tau": 0.667}, {"i": 2, "j": 1, "name": "GH", "tau": 0.88},
  {"i": 2, "j": 2, "name": "Gauss", "tau": 0.3}]}""")


def p3() -> None:
    print("P3 — forecast quantiles against the forecast's own node law (ICE fit of Aotizhongxin with 5 % spikes):")
    M = PMCModel.from_dict(P3_MODEL)
    for G in (32, 64, 128, 256):
        f = forecast(M, np.array([4.0]), 2, quantiles=(0.025, 0.5, 0.975), gap_nodes=G)
        mass = f.density[1] * reference_grid(M, G).omega
        cum = np.cumsum(mass) / mass.sum()
        med = f.nodes[np.searchsorted(cum, 0.5)]
        print(f"   G = {G:3d}: h = 2 mean {f.mean[1]:.3f}, sd {f.sd[1]:.3f}; returned 2.5/50/97.5 % "
              f"{np.round(f.quantile_values[1], 3)}; median of the node law {med:.3f}")


if __name__ == "__main__":
    p1()
    p2()
    p3()
