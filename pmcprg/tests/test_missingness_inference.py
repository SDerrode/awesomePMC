"""Non-ignorable missingness (P6) — properties of the inference with a mechanism.

Complements ``test_missingness_references.py`` (path-enumeration references):

* **invariance** — a mechanism that does not depend on the state (all rates
  π, or all onsets a and persistences b equal) leaves γ, ξ, MPM and the
  imputations of the ignorable model unchanged and shifts the log-likelihood
  by exactly the log-probability of the mask, on every fixture, linear and
  log-space passes;
* **nesting** — ``"state-markov"`` with a = b = π is ``"state"`` with π;
* **consistency** across the entry points (``classify``, ``gap_posterior``,
  ``forward``/``backward``/``smooth``, user-supplied ``precompute_weights``
  tensors, ``impute``), and between the exact shortcut and the grid on a
  model where both apply;
* **forecast** — no factor on the appended rows, whatever the last mask;
* **Monte Carlo** — strongly state-dependent masks: the mechanism lowers the
  MPM error at the missing positions and its posterior probabilities there
  are calibrated; with bursty masks ``"state"`` is over-confident where
  ``"state-markov"`` is calibrated;
* **estimators** — ICE and SEM carry the mechanism fixed;
* **default bit-identity** — with ``missingness is None`` every missing-data
  entry point returns what the code before P6 returned, bit for bit.
"""
from __future__ import annotations

import math
import re
import subprocess
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest

from pmcprg.missing.patterns import state_dependent, state_markov
from pmcprg.pmc import (PMCModel, StateMarkovMissingness, StateMissingness, gaps, simulate)
from pmcprg.pmc import inference as inf

REPO = Path(__file__).resolve().parents[2]
MODELS = sorted((REPO / "pmcprg" / "pmc" / "models").glob("*.toml"))
GAPS = [0, 1, 17, 30, 31, 32, 44, 59]          # leading block, isolated, block of 3, trailing
N_SEQ = 60


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _gapped(m, N=N_SEQ, idx=GAPS, tag=""):
    X, Y = simulate(m, N=N, seed=_seed("P6", m.name, tag) % 2**31)
    Yn = np.array(Y, dtype=float, copy=True)
    Yn[idx] = np.nan
    return X, Y, Yn


def _mask_loglik_markov(miss, a, b):
    """log P(m) under a two-state Markov chain (P(1 | 0) = a, P(1 | 1) = b)
    started from its stationary law s = a / (1 − b + a)."""
    s = a / (1.0 - b + a)
    ll = math.log(s if miss[0] else 1.0 - s)
    for prev, cur in zip(miss[:-1], miss[1:]):
        p1 = b if prev else a
        ll += math.log(p1 if cur else 1.0 - p1)
    return ll


def _grid_log_chain(m, Y):
    """The log-space pass of the augmented chain, forced (as test_gaps_api)."""
    miss = gaps.missing_mask(Y)
    chain = gaps._Chain(m, Y, miss, gaps.reference_grid(m), log=True)
    alphas, ll = gaps._forward_chain(chain)
    betas = gaps._backward_chain(chain)
    return ll, gaps._chain_posterior(chain, alphas, betas, want_xi=True)


# ---------------------------------------------------------------------------
# Invariance: a state-independent mechanism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
@pytest.mark.parametrize("kind", ["state", "state-markov"])
def test_state_independent_mechanism_only_shifts_the_log_likelihood(path, kind):
    # Measured max over the 9 fixtures × 2 kinds: |Δγ|, |Δξ| 4.4e-16 (linear)
    # and 6.7e-16 (log space, node posteriors included); imputation mean /
    # sd / quantiles 8.9e-15; |Δ log-lik − shift| 1.3e-13 (complete data:
    # 2.8e-16 and 7.5e-14) — rounding of the N extra products.
    m0 = PMCModel(path)
    K = m0.K
    _, Y, Yn = _gapped(m0)
    miss = gaps.missing_mask(Yn)
    M, N = int(miss.sum()), len(Yn)
    if kind == "state":
        m = m0.with_missingness(StateMissingness(rates=[0.3] * K))
        shift = M * math.log(0.3) + (N - M) * math.log(0.7)
        shift_complete = N * math.log(0.7)
    else:
        m = m0.with_missingness(StateMarkovMissingness(onset=[0.2] * K, persistence=[0.7] * K))
        shift = _mask_loglik_markov(miss, 0.2, 0.7)
        shift_complete = _mask_loglik_markov(np.zeros(N, bool), 0.2, 0.7)

    p0, p1 = gaps.gap_posterior(m0, Yn), gaps.gap_posterior(m, Yn)
    assert p1.log_lik - p0.log_lik == pytest.approx(shift, abs=1e-12)
    np.testing.assert_allclose(p1.gamma, p0.gamma, atol=1e-13)
    np.testing.assert_allclose(p1.xi, p0.xi, atol=1e-13)
    X0, g0, _ = inf.classify(m0, Yn)
    X1, g1, _ = inf.classify(m, Yn)
    assert np.array_equal(X0[np.max(g0, axis=1) > 0.5 + 1e-9], X1[np.max(g0, axis=1) > 0.5 + 1e-9])
    i0 = gaps.impute(m0, Yn, quantiles=(0.1, 0.5, 0.9))
    i1 = gaps.impute(m, Yn, quantiles=(0.1, 0.5, 0.9))
    np.testing.assert_allclose(i1.mean, i0.mean, atol=1e-13)
    np.testing.assert_allclose(i1.sd, i0.sd, atol=1e-13)
    np.testing.assert_allclose(i1.quantile_values, i0.quantile_values, atol=1e-13)

    # log-space passes (forced)
    if p0.method == "exact":
        a0, l0 = inf._forward_log_space(m0, Yn)
        a1, l1 = inf._forward_log_space(m, Yn)
        g0l = inf.smooth(a0, inf._backward_log_space(m0, Yn))
        g1l = inf.smooth(a1, inf._backward_log_space(m, Yn))
        np.testing.assert_allclose(g1l, g0l, atol=1e-13)
    else:
        l0, q0 = _grid_log_chain(m0, Yn)
        l1, q1 = _grid_log_chain(m, Yn)
        for key in ("gamma", "xi", "node_post"):
            np.testing.assert_allclose(q1[key], q0[key], atol=1e-13)
    assert l1 - l0 == pytest.approx(shift, abs=1e-12)

    # complete data: the mask m = 0 is evidence too, uniform here
    _, gc0, lc0 = inf.classify(m0, Y)
    _, gc1, lc1 = inf.classify(m, Y)
    np.testing.assert_allclose(gc1, gc0, atol=1e-13)
    assert lc1 - lc0 == pytest.approx(shift_complete, abs=1e-12)


# ---------------------------------------------------------------------------
# Nesting: state-markov with a = b = π is state with π
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hmc_in_gauss_k3.toml", "pmc_in_gauss_k2.toml",
                                  "hmc_dn_gauss_k2.toml", "pmc_pair_gauss_k2.toml"])
def test_state_markov_with_equal_onset_and_persistence_is_state(name):
    # a = b = π gives s = π / (1 − π + π) — π up to the rounding of 1 − π + π.
    # Measured |Δ| = 0 for these rates (1 − π + π rounds to 1): factors,
    # log-lik, γ and ξ are bit-identical; the tolerances only allow for rates
    # where that sum rounds to 1 ± 1 ulp.
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / name)
    rates = [0.1, 0.45, 0.3][:m0.K]
    ms = m0.with_missingness(StateMissingness(rates=rates))
    mm = m0.with_missingness(StateMarkovMissingness(onset=rates, persistence=rates))
    _, _, Yn = _gapped(m0, tag="nesting")
    miss = gaps.missing_mask(Yn)
    np.testing.assert_allclose(mm.missingness.evidence(miss), ms.missingness.evidence(miss),
                               rtol=0, atol=1e-15)
    ps, pm = gaps.gap_posterior(ms, Yn), gaps.gap_posterior(mm, Yn)
    assert pm.log_lik == pytest.approx(ps.log_lik, abs=1e-13)
    np.testing.assert_allclose(pm.gamma, ps.gamma, atol=1e-14)
    np.testing.assert_allclose(pm.xi, ps.xi, atol=1e-14)


# ---------------------------------------------------------------------------
# Consistency across entry points
# ---------------------------------------------------------------------------

def _mechanism(kind, K):
    if kind == "state":
        return StateMissingness(rates=[0.05, 0.4, 0.2][:K])
    return StateMarkovMissingness(onset=[0.05, 0.3, 0.15][:K], persistence=[0.4, 0.85, 0.6][:K])


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
@pytest.mark.parametrize("kind", ["state", "state-markov"])
def test_entry_points_agree(path, kind):
    m0 = PMCModel(path)
    m = m0.with_missingness(_mechanism(kind, m0.K))
    _, _, Yn = _gapped(m0, tag="entry")
    miss = gaps.missing_mask(Yn)
    X_hat, g, ll = inf.classify(m, Yn)
    post = gaps.gap_posterior(m, Yn)
    a, ll_f = inf.forward(m, Yn)
    b = inf.backward(m, Yn)
    assert ll == post.log_lik and ll_f == pytest.approx(ll, abs=1e-12)
    np.testing.assert_allclose(post.gamma, g, atol=1e-12)
    np.testing.assert_allclose(inf.smooth(a, b), g, atol=1e-12)
    np.testing.assert_allclose(post.xi.sum(axis=2), g[:-1], atol=1e-10)
    np.testing.assert_allclose(post.xi.sum(axis=1), g[1:], atol=1e-10)
    assert np.array_equal(X_hat, inf.mpm(g))
    # the mechanism is used: γ moves at the missing positions
    g_ign = gaps.gap_posterior(m0, Yn).gamma
    assert np.abs(g - g_ign)[miss].max() > 1e-3
    imp = gaps.impute(m, Yn, n_samples=4, rng=1)
    np.testing.assert_array_equal(imp.gamma, post.gamma[imp.index])
    assert imp.log_lik == post.log_lik
    X, Yc = inf.sample_posterior(m, Yn, np.random.default_rng(0), return_y=True)
    np.testing.assert_array_equal(Yc[~miss], Yn[~miss])
    assert np.all(np.isfinite(Yc)) and X.shape == (N_SEQ,)
    if post.method == "exact":
        # user-supplied tensors carry the factors: same results as without them
        W, f = inf.precompute_weights(m, Yn)
        assert inf.forward(m, Yn, W=W, f_pdf=f)[1] == ll_f
        np.testing.assert_array_equal(inf.forward(m, Yn, W=W, f_pdf=f)[0], a)
        np.testing.assert_array_equal(inf.backward(m, Yn, W=W), b)
        np.testing.assert_allclose(inf.joint_posteriors(a, W, b), post.xi, atol=1e-15)
        r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
        np.testing.assert_array_equal(inf.sample_posterior(m, Yn, r1, W=W, f_pdf=f),
                                      inf.sample_posterior(m, Yn, r2))


@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "pmc_pair_gauss_k2.toml"])
def test_posterior_draws_follow_the_posterior_given_the_mask(name):
    """FFBS draws (``impute(n_samples)`` and ``sample_posterior``) are drawn
    given (y_obs, m): their state frequencies match γ with the factors, which
    differs from the ignorable γ by far more than the Monte-Carlo error."""
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / name)
    m = m0.with_missingness(_mechanism("state", 2))
    _, _, Yn = _gapped(m0, N=30, idx=[0, 8, 9, 10, 20, 29], tag="draws")
    post = gaps.gap_posterior(m, Yn)
    S = 4000
    imp = gaps.impute(m, Yn, n_samples=S, rng=_seed("draws", name))
    freq = (imp.x_samples == 1).mean(axis=0)
    se = np.sqrt(post.gamma[:, 1] * (1 - post.gamma[:, 1]) / S)
    np.testing.assert_array_less(np.abs(freq - post.gamma[:, 1]), 4.5 * se + 1e-12)
    g_ign = gaps.gap_posterior(m0, Yn).gamma[:, 1]
    assert np.max(np.abs(g_ign - post.gamma[:, 1]) / (se + 1e-12)) > 20
    np.testing.assert_array_less(np.abs(imp.y_samples.mean(axis=0) - imp.mean),
                                 4.5 * imp.sd / math.sqrt(S))
    if gaps.needs_grid(m):
        # sample_posterior runs the same FFBS on the same chain as impute: one
        # draw from the same generator is the same path and the same values
        for s in range(10):
            X, Yc = inf.sample_posterior(m, Yn, np.random.default_rng(s), return_y=True)
            one = gaps.impute(m, Yn, quantiles=(), n_samples=1, rng=s)
            np.testing.assert_array_equal(X, one.x_samples[0])
            np.testing.assert_array_equal(Yc[one.index], one.y_samples[0])
    else:
        # the K-state sampler draws with its own stream: one draw per call
        rng = np.random.default_rng(_seed("draws-sp", name))
        S2 = 2000
        X = np.array([inf.sample_posterior(m, Yn, rng) for _ in range(S2)])
        f2 = (X == 1).mean(axis=0)
        se2 = np.sqrt(post.gamma[:, 1] * (1 - post.gamma[:, 1]) / S2)
        np.testing.assert_array_less(np.abs(f2 - post.gamma[:, 1]), 4.5 * se2 + 1e-12)


def _independence_pair(kind):
    """HMC-IN (exact shortcut) and the same model as HMC-DN with product copulas
    (the grid): the same law of (x, y)."""
    margins = [{"i": 0, "dist": "norm", "params": {"loc": -0.8, "scale": 1.0}},
               {"i": 1, "dist": "norm", "params": {"loc": 0.9, "scale": 1.3}}]
    prior = {"A": [[0.85, 0.15], [0.25, 0.75]]}
    exact = PMCModel.from_dict({"model": {"variant": "HMC-IN", "K": 2}, "prior": prior,
                                "margins": margins})
    grid = PMCModel.from_dict({"model": {"variant": "HMC-DN", "K": 2}, "prior": prior,
                               "margins": margins,
                               "copulas": [{"i": i, "j": j, "name": "Prod", "tau": 0.0}
                                           for i in range(2) for j in range(2)]})
    mech = _mechanism(kind, 2)
    return exact.with_missingness(mech), grid.with_missingness(mech)


@pytest.mark.parametrize("kind", ["state", "state-markov"])
def test_exact_shortcut_and_grid_agree_where_both_apply(kind):
    # With product copulas the grid blocks are rescaled to the exact A_ij, so
    # the state posteriors agree to rounding: measured 0 (log-lik), 2.2e-16
    # (γ, ξ, forecast probabilities). The moments of y are G = 64 quadratures
    # of the exact mixtures: measured 9.5e-9 (mean), 3.2e-7 (sd), 5.2e-9 (5,
    # 50, 95 % quantiles) — the ignorable model on the same data: 6.0e-9,
    # 2.2e-7, 6.1e-9, i.e. the quadrature error of the grid itself.
    exact, grid = _independence_pair(kind)
    assert not gaps.needs_grid(exact) and gaps.needs_grid(grid)
    _, _, Yn = _gapped(exact, N=40, idx=[0, 1, 9, 20, 21, 22, 39], tag="indep")
    pe, pg = gaps.gap_posterior(exact, Yn), gaps.gap_posterior(grid, Yn)
    assert (pe.method, pg.method) == ("exact", "grid")
    assert pg.log_lik == pytest.approx(pe.log_lik, abs=1e-12)
    np.testing.assert_allclose(pg.gamma, pe.gamma, atol=1e-12)
    np.testing.assert_allclose(pg.xi, pe.xi, atol=1e-12)
    ie, ig = gaps.impute(exact, Yn), gaps.impute(grid, Yn)
    np.testing.assert_allclose(ig.mean, ie.mean, atol=4e-8)
    np.testing.assert_allclose(ig.sd, ie.sd, atol=1e-6)
    np.testing.assert_allclose(ig.quantile_values, ie.quantile_values, atol=3e-8)
    fe, fg = gaps.forecast(exact, Yn, 3), gaps.forecast(grid, Yn, 3)
    np.testing.assert_allclose(fg.state_probs, fe.state_probs, atol=1e-12)
    assert fg.log_lik == pytest.approx(fe.log_lik, abs=1e-12)


# ---------------------------------------------------------------------------
# Forecast: the appended rows carry no factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hmc_in_gauss_k3.toml", "pmc_in_gauss_k2.toml",
                                  "hmc_dn_gauss_k2.toml", "pmc_gauss_k2.toml"])
@pytest.mark.parametrize("kind", ["state", "state-markov"])
@pytest.mark.parametrize("last_missing", [False, True])
def test_forecast_gives_no_factor_to_the_future_rows(name, kind, last_missing):
    # X is Markov for these fixtures (state margins), so the manual filter is
    # P(x_{N+k} | y_obs, m) = α̂_N A^k with α̂_N the forward filter at the last
    # row (its factors included): exact for the shortcut, and for the grid
    # too, whose blocks carry the exact A_ij. Measured max |Δ| 3.3e-16
    # (probabilities), 0 (log-lik); with a factor on the future rows the
    # probabilities would move by at least 0.22 over these 16 cases.
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / name)
    m = m0.with_missingness(_mechanism(kind, m0.K))
    idx = [3, 10, 11] + ([27, 28, 29] if last_missing else [])
    _, _, Yn = _gapped(m0, N=30, idx=idx, tag="forecast")
    h = 4
    fc = gaps.forecast(m, Yn, h)
    a, ll = inf.forward(m, Yn)
    P = a[-1]
    for k in range(h):
        P = P @ m.transition_A
        np.testing.assert_allclose(fc.state_probs[k], P, atol=1e-13)
    assert fc.log_lik == pytest.approx(ll, abs=1e-12)
    # a factor on the future rows (their mask taken as "missing") would move them
    ext = np.concatenate([Yn, np.full(h, np.nan)])
    wrong = gaps.gap_posterior(m, ext).alpha_hat[len(Yn):]
    assert np.abs(wrong - fc.state_probs).max() > 1e-3


# ---------------------------------------------------------------------------
# Monte Carlo: strongly state-dependent masks
# ---------------------------------------------------------------------------

def _hmc_in(A, mu, sd=1.0):
    return PMCModel.from_dict({
        "model": {"variant": "HMC-IN", "K": 2}, "prior": {"A": A},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": float(mu[i]), "scale": sd}}
                    for i in range(2)]})


def _calibration(X, P, miss):
    """z-scores of Σ_{n missing, p_n in bin} (1[x_n = 1] − p_n).

    E[1[x_n = 1] − p_n | y_obs, m] = 0 for the exact posterior p_n, and the bin
    only depends on (y_obs, m): each replication's sum has mean 0. The
    replications are independent, so the z-score uses their empirical
    standard error (positions within one sequence are correlated). Returns
    the overall z and [(count, z)] per bin of p_n (bins with at least 8
    replications contributing).
    """
    R = len(X)
    edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-12]

    def z_of(select):
        S = np.array([float(((X[r] == 1) - P[r])[select(r)].sum()) for r in range(R)])
        n = sum(int(select(r).sum()) for r in range(R))
        used = sum(bool(select(r).any()) for r in range(R))
        sd = S.std(ddof=1)
        return n, used, (S.mean() / (sd / math.sqrt(R)) if sd > 0 else 0.0)

    _, _, z_all = z_of(lambda r: miss[r])
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n, used, z = z_of(lambda r, lo=lo, hi=hi: miss[r] & (P[r] >= lo) & (P[r] < hi))
        if used >= 8:
            bins.append((n, z))
    return z_all, bins


def _mc(m0, fits, gen, R, N, tag):
    """MPM error at the missing positions and calibration of each fit."""
    X_all, P_all, M_all = [], {k: [] for k in fits}, []
    err = {k: 0 for k in fits}
    for r in range(R):
        X, Y = simulate(m0, N=N, seed=_seed(tag, "x", r) % 2**31)
        Yn, _ = gen(Y, X, _seed(tag, "mask", r))
        miss = gaps.missing_mask(Yn)
        X_all.append(X)
        M_all.append(miss)
        for k, mk in fits.items():
            X_hat, g, _ = inf.classify(mk, Yn)
            P_all[k].append(g[:, 1])
            err[k] += int((X_hat != X)[miss].sum())
    M = int(sum(mm.sum() for mm in M_all))
    return ({k: err[k] / M for k in fits},
            {k: _calibration(X_all, P_all[k], M_all) for k in fits}, M)


def test_monte_carlo_state_mechanism_lowers_the_error_and_is_calibrated():
    # HMC-IN, A = [[0.8, 0.2], [0.2, 0.8]], margins N(∓1, 1), π = (0.01, 0.3),
    # 40 sequences of N = 1000 (6 225 missing rows, 1.7 s). Measured:
    #   MPM error at the missing rows   ignorable 0.311   state 0.034
    #   calibration z (overall)         ignorable +56     state −0.66
    #   per-bin z (state)               −1.18 (259 rows in [0.6, 0.8)),
    #                                   −0.12 (5 966 rows in [0.8, 1])
    # The ignorable posterior ignores that a missing row is 30× likelier in
    # state 1: under-confident there, z ≫ 0. |z| < 4.5: two-sided level ~1e-5.
    rates = (0.01, 0.3)
    m0 = _hmc_in([[0.8, 0.2], [0.2, 0.8]], (-1.0, 1.0))
    fits = {"ignorable": m0, "state": m0.with_missingness(StateMissingness(rates=rates))}
    err, cal, M = _mc(m0, fits, lambda Y, X, s: state_dependent(Y, X, rates, seed=s),
                      R=40, N=1000, tag="mc-state")
    assert M > 4000
    assert err["state"] < 0.25 * err["ignorable"]
    z_all, bins = cal["state"]
    assert abs(z_all) < 4.5 and all(abs(z) < 4.5 for _, z in bins)
    assert cal["ignorable"][0] > 4.5


def test_monte_carlo_bursty_masks_state_markov_is_calibrated_state_is_not():
    # HMC-IN, A = [[0.98, 0.02], [0.02, 0.98]], margins N(∓0.5, 1); Markov
    # masks a = (0.002, 0.02), b = (0.8, 0.95) (bursts of mean length 5 and
    # 20); 40 sequences of N = 2000 (10 663 missing rows, 6 s). "state" uses
    # the marginal missing rate of each state measured on the same data,
    # (0.0249, 0.2427) — s = a / (1 − b + a) = (0.0099, 0.286) under a
    # constant state. Measured:
    #                     MPM error at missing   calibration z (overall)
    #   ignorable         0.193                  +14.9 (under-confident)
    #   state             0.090                  −13.9 (over-confident)
    #   state-markov      0.062                  +0.69, per bin |z| ≤ 1.14
    # A burst of L rows is L independent pieces of evidence (π_1/π_0)^L for
    # "state", one onset and L − 1 continuations for "state-markov".
    a, b = (0.002, 0.02), (0.8, 0.95)
    m0 = _hmc_in([[0.98, 0.02], [0.02, 0.98]], (-0.5, 0.5))

    def gen(Y, X, s):
        return state_markov(Y, X, a, b, seed=s)

    # marginal missing rate of each state, measured on the data of the study
    counts = np.zeros((2, 2))
    for r in range(40):
        X, Y = simulate(m0, N=2000, seed=_seed("mc-markov", "x", r) % 2**31)
        _, mk = gen(Y, X, _seed("mc-markov", "mask", r))
        for i in range(2):
            counts[i] += [mk[X == i].sum(), (X == i).sum()]
    matched = counts[:, 0] / counts[:, 1]
    fits = {"ignorable": m0,
            "state": m0.with_missingness(StateMissingness(rates=matched.tolist())),
            "state-markov": m0.with_missingness(StateMarkovMissingness(onset=a, persistence=b))}
    err, cal, M = _mc(m0, fits, gen, R=40, N=2000, tag="mc-markov")
    assert M > 8000
    assert err["state-markov"] < 0.5 * err["ignorable"]
    assert err["state-markov"] <= err["state"]
    z_all, bins = cal["state-markov"]
    assert abs(z_all) < 4.5 and all(abs(z) < 4.5 for _, z in bins)
    assert cal["state"][0] < -4.5          # over-confident for state 1
    assert cal["ignorable"][0] > 4.5


# ---------------------------------------------------------------------------
# Estimators: the mechanism is carried fixed
# ---------------------------------------------------------------------------

def _fit(algo, m, Y, **cfg):
    from pmcprg.pmc.ice import ice
    from pmcprg.pmc.sem import sem
    if algo == "sem":
        return sem(m, Y, {"max_iter": 3, "sem_seed": 4, **cfg})
    strategy = algo.split("-")[1]
    return ice(m, Y, {"max_iter": 3, "tol": 1e-12, "missing_strategy": strategy,
                      "missing_draws": 2, **cfg})


@pytest.mark.parametrize("algo", ["ice-available", "ice-impute", "sem"])
@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "pmc_gauss_k2.toml"])
def test_estimators_carry_the_mechanism_and_use_it(algo, name):
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / name)
    m = m0.with_missingness(_mechanism("state-markov", 2))
    _, Y, Yn = _gapped(m0, N=150, idx=list(range(20, 26)) + [60, 90, 91], tag="estim")
    fitted, trace = _fit(algo, m, Yn, fit_margins=True)
    assert fitted.missingness == m.missingness
    assert fitted.raw["missingness"] == m.raw["missingness"]
    assert trace.log_liks[0] == pytest.approx(gaps.gap_posterior(m, Yn).log_lik, rel=1e-12)
    assert abs(trace.log_liks[0] - gaps.gap_posterior(m0, Yn).log_lik) > 1.0
    # complete data: the mask m = 0 is evidence, the trace starts at its likelihood
    fitted_c, trace_c = _fit(algo, m, Y)
    assert fitted_c.missingness == m.missingness
    assert trace_c.log_liks[0] == pytest.approx(inf.classify(m, Y)[2], rel=1e-12)


@pytest.mark.parametrize("algo", ["ice-available", "ice-impute", "sem"])
def test_estimators_with_a_state_independent_mechanism_match_the_ignorable_fit(algo):
    # Uniform rates leave every posterior unchanged (to rounding), so the fit
    # is the ignorable one and its trace is shifted by the mask's
    # log-probability. Measured after 3 iterations (ICE available / impute /
    # SEM): trace 3.5e-11 / 1.9e-11 / 1.3e-13 off the shift; prior 2.6e-13 /
    # 6.6e-13 / 0; margin parameters (relative) 1.2e-12 / 1.1e-12 / 0; τ
    # 2.2e-12 / 2.8e-12 / 0. ICE's rounding differences go through the τ and
    # margin optimisers and grow to ~1e-12; SEM draws the same paths and
    # refits them identically. Tolerances ≥ 14× the measured values.
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / "pmc_gauss_k2.toml")
    m = m0.with_missingness(StateMissingness(rates=[0.25, 0.25]))
    _, _, Yn = _gapped(m0, N=150, idx=list(range(20, 26)) + [60, 90, 91], tag="estim")
    miss = gaps.missing_mask(Yn)
    shift = miss.sum() * math.log(0.25) + (~miss).sum() * math.log(0.75)
    f0, t0 = _fit(algo, m0, Yn, fit_margins=True)
    f1, t1 = _fit(algo, m, Yn, fit_margins=True)
    assert len(t1.log_liks) == len(t0.log_liks)
    np.testing.assert_allclose(np.array(t1.log_liks) - np.array(t0.log_liks), shift, atol=5e-10)
    np.testing.assert_allclose(f1.prior_p, f0.prior_p, atol=1e-11)
    for b0, b1 in zip(f0.margin_blocks(), f1.margin_blocks()):
        assert b0["dist"] == b1["dist"]
        for k in b0["params"]:
            assert b1["params"][k] == pytest.approx(b0["params"][k], rel=1e-10, abs=1e-10)
    for c0, c1 in zip(f0.copula_blocks(), f1.copula_blocks()):
        assert c0["name"] == c1["name"] and c1["tau"] == pytest.approx(c0["tau"], abs=1e-10)


def test_impute_strategy_keeps_the_observed_mask_on_the_completed_series(monkeypatch):
    """The complete-data E-step of each completed series is P(x | y^(d), m):
    the factors of the observed mask, not those of the (complete) series."""
    import importlib
    ICE = importlib.import_module("pmcprg.pmc.ice")
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / "hmc_in_gauss_k2.toml")
    m = m0.with_missingness(_mechanism("state", 2))
    _, _, Yn = _gapped(m0, N=80, idx=[5, 6, 7, 40], tag="impute-mask")
    miss = gaps.missing_mask(Yn)
    seen = []
    real = inf._weights

    def spy(model, Y, ev):
        seen.append(None if ev is None else np.array(ev))
        return real(model, Y, ev)

    monkeypatch.setattr(inf, "_weights", spy)
    gap = ICE._gap_e_step(m, Yn, miss, gap_nodes=None, posterior=True, n_draws=2,
                          rng=np.random.default_rng(0))
    seen.clear()
    ICE._m_step_impute(m.raw, m, gap.Y_draws, gap.xi, fit_margins=True, candidates=[],
                       selection_criterion="aic", margin_selection_rule="mle", miss=miss)
    assert len(seen) == 2
    for ev in seen:
        np.testing.assert_array_equal(ev, m.missingness.evidence(miss))


def test_kmeans_start_and_multistart_keep_the_mechanism():
    from pmcprg.pmc.ice import ice
    m0 = PMCModel(REPO / "pmcprg" / "pmc" / "models" / "hmc_in_gauss_k2.toml")
    m = m0.with_missingness(_mechanism("state", 2))
    _, _, Yn = _gapped(m0, N=120, tag="kmeans")
    fitted, trace = ice(m, Yn, {"max_iter": 2, "init": "kmeans", "n_starts": 2})
    assert fitted.missingness == m.missingness


# ---------------------------------------------------------------------------
# Default bit-identity: missingness None gives the pre-P6 results
# ---------------------------------------------------------------------------

BASE_COMMIT = "e24470e"      # the last commit before P6


@pytest.fixture(scope="module")
def pre_p6():
    """``inference.py`` and ``gaps.py`` at BASE_COMMIT, wired to each other."""
    try:
        srcs = {f: subprocess.run(["git", "show", f"{BASE_COMMIT}:pmcprg/pmc/{f}.py"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout
                for f in ("inference", "gaps")}
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git history with commit {BASE_COMMIT} unavailable")
    names = {"inference": "_pre_p6_inference", "gaps": "_pre_p6_gaps"}
    mods = {f: types.ModuleType(n) for f, n in names.items()}
    srcs["inference"] = re.sub(r"from pmcprg\.pmc import gaps\b", "gaps = _PRE_P6_GAPS",
                               srcs["inference"])
    srcs["gaps"] = srcs["gaps"].replace("from pmcprg.pmc import inference as _inf",
                                        "_inf = _PRE_P6_INF")
    mods["inference"].__dict__["_PRE_P6_GAPS"] = mods["gaps"]
    mods["gaps"].__dict__["_PRE_P6_INF"] = mods["inference"]
    for f, mod in mods.items():
        sys.modules[names[f]] = mod               # dataclasses resolve their module
    try:
        for f in ("inference", "gaps"):
            exec(compile(srcs[f], f"{names[f]}.py", "exec"), mods[f].__dict__)
        yield mods["inference"], mods["gaps"]
    finally:
        for n in names.values():
            sys.modules.pop(n, None)


def _same(a, b, exclude=()):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if a is None or b is None:
        return a is b
    if isinstance(a, (float, int, np.floating, np.integer, np.ndarray)):
        return np.array_equal(np.asarray(a), np.asarray(b))
    fields = getattr(a, "__dataclass_fields__", None)
    if fields is not None:
        # fields added since BASE_COMMIT (e.g. the per-position ``nodes`` of
        # the local grids) have no pre-P6 counterpart to compare with
        old = getattr(b, "__dataclass_fields__", {})
        return all(_same(getattr(a, k), getattr(b, k)) for k in fields
                   if k not in ("grid", *exclude) and k in old)
    return a == b


# The gap-quadrature fix (P1: local grids where the stationary grid cannot
# resolve a strongly dependent transition; P3: the quantile density propagated
# through the exact kernel) changes, on purpose, two things this test used to
# pin. Everything else stays bit-identical to the pre-P6 code. Measured
# against the converged value (G = 2048), decision of the author, 2026-09-23:
# * the posterior/predictive **quantiles** of every grid fixture: hmc_dn and
#   pmc forecast error 3.5e-8 → 2.2e-9 and 5.0e-9 → 9.3e-10; pmc_pair forecast
#   7.4e-6 → 5.6e-6, impute 2.3e-4 → 1.6e-4 — compared with the rest excluded;
# * **sp2016_gice_k2**, where the local grids switch on: the pre-P6 values
#   were wrong (log-lik 0.092 nat off, γ 5.1e-2, quantiles 0.97; now 2.5e-3,
#   8.0e-4, 9.1e-3) — not compared.
_P1P3_CHANGED = ("quantile_values",)
_P1P3_WRONG_BEFORE = {"sp2016_gice_k2"}


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_missing_data_results_are_bit_identical_to_pre_p6(pre_p6, path):
    old_inf, old_gaps = pre_p6
    if path.stem in _P1P3_WRONG_BEFORE:
        pytest.skip("pre-P6 values were wrong for this model (gap-quadrature fix, P1/P3)")
    m = PMCModel(path)
    assert m.missingness is None
    _, Y, Yn = _gapped(m, N=80, idx=[0, 1, 17, 40, 41, 42, 60, 79], tag="bit")
    for Z in (Yn, Y):
        assert _same(inf.classify(m, Z), old_inf.classify(m, Z))
        assert _same(inf.forward(m, Z), old_inf.forward(m, Z))
        assert _same(inf.backward(m, Z), old_inf.backward(m, Z))
        assert _same(inf.sample_posterior(m, Z, np.random.default_rng(3), return_y=True),
                     old_inf.sample_posterior(m, Z, np.random.default_rng(3), return_y=True))
        assert _same(gaps.gap_posterior(m, Z), old_gaps.gap_posterior(m, Z))
        assert _same(gaps.impute(m, Z, n_samples=5, rng=2), old_gaps.impute(m, Z, n_samples=5, rng=2),
                     exclude=_P1P3_CHANGED)
        assert _same(gaps.forecast(m, Z, 3), old_gaps.forecast(m, Z, 3), exclude=_P1P3_CHANGED)
    if not gaps.needs_grid(m):
        assert _same(inf.precompute_weights(m, Yn), old_inf.precompute_weights(m, Yn))
        assert _same(inf._forward_log_space(m, Yn), old_inf._forward_log_space(m, Yn))
        assert _same(inf._backward_log_space(m, Yn), old_inf._backward_log_space(m, Yn))
    else:
        miss = gaps.missing_mask(Yn)
        grid = gaps.reference_grid(m)
        for log in (False, True):
            c_new = gaps._Chain(m, Yn, miss, grid, log=log)
            c_old = old_gaps._Chain(m, Yn, miss, old_gaps.reference_grid(m), log=log)
            assert _same(c_new.init, c_old.init) and _same(list(c_new.trans), list(c_old.trans))
