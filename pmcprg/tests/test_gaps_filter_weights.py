"""Filter-weighted proposals of the local grids (cause 4 of the gap quadrature).

Until this version the Gaussian pieces that place the local grid of a run of
missing rows were weighted by the state law of its neighbours given y alone.
Where the filter sits in a state whose margin puts y far in its tail (Intel
Lab mote 22: a state 5.4 sds out, a Gaussian copula at τ = 0.975) that law
gives the path that stays in it ~1e-7, no piece represents it and the grid
misses the integrand, silently; convergence in ``gap_nodes`` was not monotone
(mote 22: 1.1 nats at G = 64, 7.1 at G = 256). ``gaps`` module docstring,
"Filter-weighted proposals".

Exact references: the regime-switching AR(1) of ``test_gaps_leading.py``
(y_n = m_{x_n} + s_{x_n} z_n, z an AR(1) of coefficient ρ): log p(y_obs),
P(x_n | y_obs) and the law of every missing y_n from a K-state recursion over
the observed rows.

Every tolerance is set from the error measured at the stated G (quoted); the
code before this fix (ca6ac57) misses each test (quoted as "old").
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pmcprg.pmc import gaps

from test_gaps_leading import MODELS, _model, _moments, _regime_ar1_exact, _simulate


def _tau(rho):
    return 2.0 / math.pi * math.asin(rho)


# ---------------------------------------------------------------------------
# A filter in a narrow state far in its margin's tail (mote 22, synthetic)
# ---------------------------------------------------------------------------

TAIL_M, TAIL_S = [0.0, 3.0], [1.0, 0.2]
TAIL_P = [[0.495, 0.005], [0.005, 0.495]]


def _tail_series(depth):
    """y = 3 + 0.2 z with z drifting smoothly from 0 to −depth: the chain stays in state 1
    (a switch would move y by 3 margins), whose margin puts y ``depth`` sds out, where the
    law of state 0 given y alone is 1e6 times heavier; single-row gaps in the tail."""
    n = np.arange(400)
    t = np.clip((n - 50) / 200.0, 0.0, 1.0)
    z = -depth * t * t * (3.0 - 2.0 * t) + 0.01 * np.sin(n / 7.0)
    Y = TAIL_M[1] + TAIL_S[1] * z
    Y[260:390:10] = np.nan
    return Y


@pytest.mark.parametrize("rho,depth", [(0.999, 5.4), (0.9999, 5.4)])
def test_filter_in_a_tail_state_is_integrated(rho, depth):
    # Measured (errors of the mean and sd in exact sds):
    #   rho      G    log-lik   mean     sd        old (ca6ac57)
    #   0.999    64   8.3e-6    6.3e-5   3.1e-3    0.48 nats, sd 0.90
    #   0.999   256   3.1e-11   2.2e-10  1.0e-8    1.3e-3 nats, sd 0.91
    #   0.9999   64   4.1e-5    3.3e-3   0.15      0.69 nats, sd 0.97
    #   0.9999  256   2.4e-8    2.4e-6   1.2e-4    1.3e-3 nats, sd 0.97
    # The old quad_error stayed quiet (1.4e-3, 1.9e-6): the failure was silent.
    m = _model(TAIL_M, TAIL_S, TAIL_P, _tau(rho))
    Y = _tail_series(depth)
    miss = ~np.isfinite(Y)
    ll, g, mu, sd = _regime_ar1_exact(TAIL_M, TAIL_S, TAIL_P, rho, Y)
    tol = {(0.999, 64): (1e-4, 1e-3, 2e-2), (0.999, 256): (1e-8, 1e-7, 1e-6),
           (0.9999, 64): (4e-4, 3e-2, 0.3), (0.9999, 256): (2e-7, 2e-5, 1e-3)}
    for G in (64, 256):
        t_ll, t_m, t_s = tol[(rho, G)]
        post = gaps.gap_posterior(m, Y, gap_nodes=G, xi=False)
        mean, s_ = _moments(post)
        assert abs(post.log_lik - ll) < t_ll, (G, post.log_lik - ll)
        assert np.max(np.abs(mean - mu[miss]) / sd[miss]) < t_m, G
        assert np.max(np.abs(s_ / sd[miss] - 1.0)) < t_s, G
        assert post.quad_error < gaps.QUAD_WARN


# ---------------------------------------------------------------------------
# Two-regime AR(1): interior gaps, convergence in G
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,rho,idx,tols", [
    # Measured |log-lik error| at G = 64 / 128 / 256 / 512, new and (old):
    #   repro, rho 0.999, rows 40-49:  2.0e-3 1.9e-5 5.1e-13 8.0e-13  (0.11 1.8e-3 5.1e-13 8.0e-13)
    ("repro", 0.999, list(range(40, 50)), (1e-2, 2e-4, 1e-10, 1e-10)),
    #   sd, rho 0.9999, rows 40, 80, 120, 160:
    #                                   1.1e-4 1.9e-7 3.5e-12 4.1e-12  (3.7e-2 3.5e-2 5.4e-2 4.0e-2)
    ("sd", 0.9999, [40, 80, 120, 160], (1e-3, 2e-6, 1e-9, 1e-9)),
])
def test_two_regime_interior_gaps_converge(name, rho, idx, tols):
    mm, ss, pp = MODELS[name]
    m = _model(mm, ss, pp, _tau(rho))
    Y = _simulate(mm, ss, pp, rho, 200, 1)
    Y[idx] = np.nan
    ll = _regime_ar1_exact(mm, ss, pp, rho, Y)[0]
    errs = []
    for G, tol in zip((64, 128, 256, 512), tols):
        post = gaps.gap_posterior(m, Y, gap_nodes=G, xi=False)
        errs.append(abs(post.log_lik - ll))
        assert errs[-1] < tol, (G, errs[-1])
    # monotone up to rounding: no G is worse than a smaller one by more than 1e-9 nats
    assert all(b <= a + 1e-9 for a, b in zip(errs, errs[1:])), errs


# ---------------------------------------------------------------------------
# Leading gap: the law of the missing values inside it
# ---------------------------------------------------------------------------

def test_leading_gap_means_and_sds():
    # repro model, rho = 0.999, rows 0-16 missing (N = 200, seed 1). Measured,
    # errors in exact sds, max over the gap: means 8.6e-3 / 4.2e-6, sds
    # 1.4e-2 / 1.2e-5 at G = 64 / 128; log-lik 1.3e-4 / 3.6e-7 nats. Old: means
    # 0.16 sd off at G = 64 (sds 5.5e-2), for a log-likelihood 2.6e-3 off and a
    # quad_error of 0.015, below the WARNING threshold.
    mm, ss, pp = MODELS["repro"]
    rho = 0.999
    m = _model(mm, ss, pp, _tau(rho))
    Y = _simulate(mm, ss, pp, rho, 200, 1)
    L = 17
    Y[:L] = np.nan
    ll, g, mu, sd = _regime_ar1_exact(mm, ss, pp, rho, Y)
    for G, t_ll, t_m in ((64, 1e-3, 3e-2), (128, 3e-6, 1e-4)):
        post = gaps.gap_posterior(m, Y, gap_nodes=G, xi=False)
        mean, s_ = _moments(post)
        assert np.max(np.abs(mean - mu[:L]) / sd[:L]) < t_m, G
        assert np.max(np.abs(s_ / sd[:L] - 1.0)) < t_m, G
        assert abs(post.log_lik - ll) < t_ll, (G, post.log_lik - ll)
