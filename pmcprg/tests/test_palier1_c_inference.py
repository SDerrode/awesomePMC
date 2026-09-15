"""Forward–backward on transition weights that underflow (AUDIT_COPULES FR-2 / RB-9).

Copula densities are no longer floored at EPS, so a pair density can be
exactly 0.0 in float64 — a Gaussian copula at τ ≈ 0.99 gives log c of the
order of −10⁴ on a discordant jump — or a few 1e-300. These tests build
small models where that happens and compare ``pmcprg.pmc.inference`` with a
log-space forward–backward written here from the margins' log-densities and
the copulas' ``logpdf_array``: the reference shares no code with the
recursions under test.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

from pmcprg.exceptions import IncompatibleObservationError
from pmcprg.pmc.inference import (
    backward,
    forward,
    joint_posteriors,
    precompute_weights,
    smooth,
)
from pmcprg.pmc.model import PMCModel


def _pmc(locs, taus, p):
    raw = {
        "model": {"name": "underflow", "variant": "PMC", "K": 2, "N_default": 10},
        "margins": [{"i": k, "dist": "norm", "params": {"loc": float(locs[k]), "scale": 1.0}}
                    for k in range(2)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": float(taus[i][j])}
                    for i in range(2) for j in range(2)],
        "prior": {"p": [list(map(float, row)) for row in p]},
    }
    return PMCModel.from_dict(raw)


def _reference(mdl, Y, locs):
    """Log-space forward–backward: (log-lik, α̂, β̂ normalised, γ, ξ, log W)."""
    K, N = 2, len(Y)
    p = np.asarray(mdl.prior_p)
    lf = np.column_stack([stats.norm.logpdf(Y, loc=locs[k]) for k in range(K)])
    F = np.column_stack([stats.norm.cdf(Y, loc=locs[k]) for k in range(K)])
    logW = np.empty((N - 1, K, K))
    for i in range(K):
        for j in range(K):
            logc = mdl.copula(i, j).logpdf_array(np.column_stack((F[:-1, i], F[1:, j])))
            logW[:, i, j] = np.log(p[i, j] / p[i].sum()) + lf[1:, j] + logc
    la = np.empty((N, K))
    a = np.array([logsumexp(np.log(p[:, j])) + lf[0, j] for j in range(K)])
    ll = logsumexp(a)
    la[0] = a - ll
    for n in range(N - 1):
        r = logsumexp(la[n][:, None] + logW[n], axis=0)
        c = logsumexp(r)
        ll += c
        la[n + 1] = r - c
    lb = np.empty((N, K))
    lb[-1] = -np.log(K)
    for n in range(N - 2, -1, -1):
        r = logsumexp(logW[n] + lb[n + 1][None, :], axis=1)
        lb[n] = r - logsumexp(r)
    lg = la + lb
    gamma = np.exp(lg - logsumexp(lg, axis=1, keepdims=True))
    lx = la[:-1, :, None] + logW + lb[1:, None, :]
    xi = np.exp(lx - logsumexp(lx, axis=(1, 2), keepdims=True))
    return ll, np.exp(la), np.exp(lb), gamma, xi, logW


# Four strongly concordant Gaussian copulas: a discordant jump (2.6 → −2.5)
# drives every pair density below 1e-308, with log c from about −1.3e4 to
# −2e5 depending on the pair — so the exact posterior at the jump is far
# from uniform.
_LOCS = (-1.0, 1.0)
_TAUS = ((0.995, 0.99), (0.99, 0.98))
_P = ((0.40, 0.10), (0.10, 0.40))
_Y = np.array([2.5, 2.4, 2.6, -2.5, -2.4, -2.6, 2.5, 0.0, 0.1])


def test_an_all_underflow_step_is_recomputed_exactly_in_log_space(caplog):
    mdl = _pmc(_LOCS, _TAUS, _P)
    W, f_pdf = precompute_weights(mdl, _Y)
    assert np.all(np.isfinite(W))
    void = np.nonzero(W.sum(axis=(1, 2)) == 0.0)[0]
    assert void.size >= 1, "premise: some step has every transition weight at 0.0"

    ll_ref, a_ref, b_ref, g_ref, xi_ref, logW = _reference(mdl, _Y, _LOCS)
    assert np.all(np.isfinite(logW[void])), "premise: the log weights are finite there"

    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.inference"):
        alpha, ll = forward(mdl, _Y, W=W, f_pdf=f_pdf)
        beta = backward(mdl, _Y, W=W)
    assert any("log space" in r.message for r in caplog.records)

    # The former floor gave ll ≈ -3e3 (C := MIN_POSITIVE) and α̂ = 0 after the jump.
    assert np.isfinite(ll)
    assert ll == pytest.approx(ll_ref, rel=1e-12)
    np.testing.assert_allclose(alpha, a_ref, atol=1e-12)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(beta, b_ref, atol=1e-12)

    gamma = smooth(alpha, beta)
    np.testing.assert_allclose(gamma, g_ref, atol=1e-12)
    assert np.abs(gamma[void] - 0.5).max() > 0.1, "the posterior at the jump is informative"

    xi = joint_posteriors(alpha, W, beta)
    assert np.all(np.isfinite(xi))
    np.testing.assert_allclose(xi.sum(axis=(1, 2)), 1.0, atol=1e-12)
    ok = np.setdiff1d(np.arange(len(_Y) - 1), void)
    np.testing.assert_allclose(xi[ok], xi_ref[ok], atol=1e-10)
    # At the void steps W carries no information; ξ falls back to γ_n ⊗ γ_{n+1},
    # which keeps both marginals of ξ consistent with γ.
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)
    np.testing.assert_allclose(xi.sum(axis=1), gamma[1:], atol=1e-10)


def test_densities_of_a_few_1e300_stay_on_the_linear_path_and_exact(caplog):
    """A jump whose best pair density is ~1e-290: C stays representable,
    no log-space recomputation is needed, and the result is still exact."""
    locs = (0.0, 0.0)
    taus = ((0.9, 0.9), (0.9, 0.9))
    mdl = _pmc(locs, taus, ((0.25, 0.25), (0.25, 0.25)))
    Y = np.array([0.3, 2.9, -2.9, 0.1, 0.5])
    W, f_pdf = precompute_weights(mdl, Y)
    ll_ref, a_ref, _, g_ref, xi_ref, logW = _reference(mdl, Y, locs)
    step_max = logW.max(axis=(1, 2))
    assert step_max.min() < -600.0 and np.all(W.sum(axis=(1, 2)) > 0.0), \
        "premise: a step of tiny but representable weights"

    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.inference"):
        alpha, ll = forward(mdl, Y, W=W, f_pdf=f_pdf)
        beta = backward(mdl, Y, W=W)
    assert not any("log space" in r.message for r in caplog.records)
    assert ll == pytest.approx(ll_ref, rel=1e-10)
    np.testing.assert_allclose(alpha, a_ref, atol=1e-10)
    np.testing.assert_allclose(smooth(alpha, beta), g_ref, atol=1e-10)
    np.testing.assert_allclose(joint_posteriors(alpha, W, beta), xi_ref, atol=1e-10)


def test_rows_of_zero_pair_density_do_not_disturb_the_linear_recursion():
    """Opposite copulas per source state: at every jump half the pair densities
    are 0.0, but each step keeps a reachable positive weight."""
    locs = (0.0, 0.0)
    taus = ((0.995, 0.995), (-0.995, -0.995))
    mdl = _pmc(locs, taus, ((0.45, 0.05), (0.05, 0.45)))
    Y = np.array([1.5, 1.5, -1.5, 1.5, 1.5, 1.5, -1.5])
    W, f_pdf = precompute_weights(mdl, Y)
    assert (W == 0.0).any() and np.all(W.sum(axis=(1, 2)) > 0.0)
    ll_ref, a_ref, _, g_ref, _, _ = _reference(mdl, Y, locs)
    alpha, ll = forward(mdl, Y, W=W, f_pdf=f_pdf)
    beta = backward(mdl, Y, W=W)
    assert ll == pytest.approx(ll_ref, rel=1e-12)
    np.testing.assert_allclose(smooth(alpha, beta), g_ref, atol=1e-12)


def test_an_observation_impossible_even_in_log_space_is_refused():
    """Zero density under every state mid-sequence is a modelling error, as at Y[0]."""
    raw = {
        "model": {"name": "bounded", "variant": "HMC-IN", "K": 2, "N_default": 4},
        "margins": [{"i": 0, "dist": "uniform", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 1, "dist": "uniform", "params": {"loc": 0.0, "scale": 2.0}}],
        "prior": {"A": [[0.5, 0.5], [0.5, 0.5]]},
    }
    mdl = PMCModel.from_dict(raw)
    with pytest.raises(IncompatibleObservationError, match=r"Y\[2\]"):
        forward(mdl, np.array([0.5, 0.2, 3.0, 0.3]))


def test_joint_posterior_fallback_keeps_both_marginals():
    """Unit check of the void-step coupling with non-uniform α̂, β̂."""
    alpha = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    beta = np.array([[0.5, 0.5], [0.9, 0.1], [0.5, 0.5]])
    W = np.array([[[0.3, 0.7], [0.4, 0.6]], [[0.0, 0.0], [0.0, 0.0]]])
    xi = joint_posteriors(alpha, W, beta)
    gamma = smooth(alpha, beta)
    np.testing.assert_allclose(xi[1], np.outer(gamma[1], gamma[2]))
    np.testing.assert_allclose(xi[1].sum(axis=1), gamma[1])
    np.testing.assert_allclose(xi[1].sum(axis=0), gamma[2])
