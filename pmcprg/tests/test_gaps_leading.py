"""Leading gaps (P5), batch / filter agreement and the calibrated quadrature diagnostic (P6).

P5 (``report/erroneous_data/intel_lab``, library problem 3): a series that
starts with missing rows was misintegrated under strong serial dependence
(−22 nats at τ = 0.999, G = 64), silently (``quad_error`` 3e-5), and the batch
pass and the PIT filter disagreed (their state filters by up to 0.26).

Exact references, independent of the quadrature grid:

* **The gap-free pass.** A stationary PMC whose first L rows are missing has,
  after the gap, the law of the series started at row L + 1: log p(y_obs) is
  the forward pass on Y[L:], and P(x_n | nothing observed) = π inside the gap.
* **Regime-switching AR(1).** y_n = m_{x_n} + s_{x_n} z_n with x a stationary
  Markov chain and z an AR(1) of coefficient ρ independent of x — a PMC with
  state margins N(m_i, s_i²) and Gaussian copulas ρ on every pair. The AR(1)
  sampled at the observed rows is Markov, so log p(y_obs), P(x_n | y_obs) and
  the law of every missing y_n follow from a K-state recursion over the
  observed rows (:func:`_regime_ar1_exact`). Unlike the AR(1) of
  ``test_gaps_local_grids.py`` the state posterior is not trivial, and a
  switch moves y by whole margins.

Every tolerance is set from the error measured at the stated G (quoted); the
code before this fix misses each of them (quoted as "old").
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy import stats

from pmcprg.pmc import PMCModel, classify, predictive_pit, simulate
from pmcprg.pmc import gaps

P_REPRO = [[0.49, 0.01], [0.01, 0.49]]
MODELS = {  # (state means, state sds, p)
    "repro": ([0.0, 1.0], [1.0, 1.0], P_REPRO),
    "sd": ([0.0, 0.5], [1.0, 2.0], [[0.45, 0.05], [0.05, 0.45]]),
}


def _model(m, s, p, tau):
    K = len(m)
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": K}, "prior": {"p": p},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": m[i], "scale": s[i]}} for i in range(K)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(K) for j in range(K)]})


def _simulate(m, s, p, rho, N, seed):
    rng = np.random.default_rng(seed)
    p = np.asarray(p, float)
    pi = p.sum(axis=1)
    A = p / pi[:, None]
    x = np.empty(N, int)
    x[0] = rng.choice(len(pi), p=pi)
    for n in range(1, N):
        x[n] = rng.choice(len(pi), p=A[x[n - 1]])
    z = np.empty(N)
    z[0] = rng.standard_normal()
    for n in range(1, N):
        z[n] = rho * z[n - 1] + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
    return np.asarray(m)[x] + np.asarray(s)[x] * z


def _regime_ar1_exact(m, s, p, rho, Y):
    """log p(y_obs), P(x_n | y_obs) (N, K), mean and sd of every missing y_n (NaN elsewhere)."""
    m, s, p = np.asarray(m, float), np.asarray(s, float), np.asarray(p, float)
    pi = p.sum(axis=1)
    A = p / pi[:, None]
    K, N = m.size, len(Y)
    obs = np.nonzero(np.isfinite(Y))[0]

    def Ad(d):
        return np.linalg.matrix_power(A, d)

    def emis(a, n):            # density of y_n given (x_a, x_n) = (i, j), y_a observed
        r = rho ** (n - a)
        return (stats.norm.pdf(((Y[n] - m) / s)[None, :], r * ((Y[a] - m) / s)[:, None],
                               math.sqrt(1.0 - r * r)) / s[None, :])

    al = {obs[0]: pi * stats.norm.pdf((Y[obs[0]] - m) / s) / s}
    ll = math.log(al[obs[0]].sum())
    al[obs[0]] = al[obs[0]] / al[obs[0]].sum()
    E = {}
    for a, n in zip(obs[:-1], obs[1:]):
        E[n] = Ad(n - a) * emis(a, n)
        r = al[a] @ E[n]
        ll += math.log(r.sum())
        al[n] = r / r.sum()
    be = {obs[-1]: np.ones(K)}
    for a, n in zip(obs[-2::-1], obs[:0:-1]):
        b = E[n] @ be[n]
        be[a] = b / b.sum()
    gam = np.full((N, K), np.nan)
    mean, sd = np.full(N, np.nan), np.full(N, np.nan)
    for n in obs:
        gam[n] = al[n] * be[n] / (al[n] * be[n]).sum()
    for n in np.nonzero(~np.isfinite(Y))[0]:
        left, right = obs[obs < n], obs[obs > n]
        a = left[-1] if left.size else None
        b = right[0] if right.size else None
        comps = []                                   # (weight, state, mean, var) of y_n
        for i in (range(K) if a is not None else [None]):
            for k in range(K):
                for j in (range(K) if b is not None else [None]):
                    w = al[a][i] * Ad(n - a)[i, k] if a is not None else pi[k]
                    mz, vz = 0.0, 1.0
                    if a is not None:
                        mz = rho ** (n - a) * (Y[a] - m[i]) / s[i]
                        vz = 1.0 - rho ** (2 * (n - a))
                    if b is not None:
                        r = rho ** (b - n)
                        zb = (Y[b] - m[j]) / s[j]
                        pm, pv = r * mz, r * r * vz + 1.0 - r * r
                        w *= Ad(b - n)[k, j] * stats.norm.pdf(zb, pm, math.sqrt(pv)) / s[j] * be[b][j]
                        kg = vz * r / pv
                        mz, vz = mz + kg * (zb - pm), vz - kg * r * vz
                    comps.append((w, k, m[k] + s[k] * mz, s[k] ** 2 * vz))
        W = np.array([c[0] for c in comps])
        W = W / W.sum()
        M = np.array([c[2] for c in comps])
        V = np.array([c[3] for c in comps])
        mean[n] = W @ M
        sd[n] = math.sqrt(W @ (V + M * M) - mean[n] ** 2)
        gam[n] = np.bincount([c[1] for c in comps], weights=W, minlength=K)
    return ll, gam, mean, sd


def _moments(post):
    mass = post.node_post.sum(axis=1)
    mean = (mass * post.nodes).sum(axis=1)
    return mean, np.sqrt(np.maximum((mass * post.nodes ** 2).sum(axis=1) - mean ** 2, 0.0))


# ---------------------------------------------------------------------------
# The leading gap of report/erroneous_data/intel_lab/repro_leading_gap.py
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.997, 0.999])
def test_leading_gap_matches_the_gap_free_pass(tau):
    # Measured |log-lik error| at G = 64 / 256, max over L = 2, 5, 17 and both
    # passes: 2.5e-4 / 4.8e-8; the batch state filter equals π to 1e-15 in
    # the gap. Old: 4.3 and 22 nats at L = 17, G = 64 (1.7 at G = 256), the
    # batch filter off π by 0.29, the two passes 0.92 nats apart.
    m = _model(*MODELS["repro"][:2], P_REPRO, tau)
    pi = np.asarray(m.prior_p).sum(axis=1)
    _, Y = simulate(m, 400, seed=1)
    Y = np.asarray(Y, float)
    for L in (2, 5, 17):
        Yn = Y.copy()
        Yn[:L] = np.nan
        exact = float(classify(m, Y[L:])[2])
        for G, tol in ((64, 1e-3), (256, 1e-6)):
            post = gaps.gap_posterior(m, Yn, gap_nodes=G, xi=False)
            P = predictive_pit(m, Yn, gap_nodes=G)
            assert abs(post.log_lik - exact) < tol, (L, G, post.log_lik - exact)
            assert abs(P.log_lik - exact) < tol, (L, G, P.log_lik - exact)
            np.testing.assert_allclose(post.alpha_hat[:L], np.broadcast_to(pi, (L, 2)), atol=1e-12)
            np.testing.assert_allclose(P.alpha_hat[:L], np.broadcast_to(pi, (L, 2)), atol=1e-12)


def _ar1_model(kind, rho):
    tau = 2.0 / math.pi * math.asin(rho)
    cops = [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]
    std = {"dist": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    if kind == "HMC-DN":
        raw = {"model": {"variant": "HMC-DN", "K": 2}, "prior": {"A": [[0.9, 0.1], [0.2, 0.8]]},
               "margins": [{"i": i, **std} for i in range(2)], "copulas": cops}
    elif kind == "PMC-state":
        raw = {"model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
               "margins": [{"i": i, **std} for i in range(2)], "copulas": cops}
    else:
        raw = {"model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.5, 0.05], [0.05, 0.4]]},
               "margins": [{"i": i, "j": j, **std} for i in range(2) for j in range(2)], "copulas": cops}
    return PMCModel.from_dict(raw)


@pytest.mark.parametrize("kind", ["HMC-DN", "PMC-state", "PMC-pair"])
@pytest.mark.parametrize("rho", [0.999, 0.9999])
def test_leading_gap_ar1_closed_form(kind, rho):
    """AR(1) written as a PMC (identical regimes), leading gaps of 1, 5 and 20 rows."""
    # Measured at G = 64, max over L (and the kinds, identical): log-lik
    # 2.7e-7 (batch and filter), means 1.7e-10 and sds 1.8e-4 of the exact
    # conditional sd. Old: 2.5e-3 nats and sds 4.2e-3 off at L = 20.
    m = _ar1_model(kind, rho)
    rng = np.random.default_rng(0)
    Y = np.empty(80)
    Y[0] = rng.standard_normal()
    for n in range(1, 80):
        Y[n] = rho * Y[n - 1] + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
    idx = np.arange(80)
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    for L in (1, 5, 20):
        miss = np.zeros(80, bool)
        miss[:L] = True
        miss[[40, 41, 60]] = True
        o = ~miss
        ll = stats.multivariate_normal(mean=np.zeros(o.sum()), cov=S[np.ix_(o, o)]).logpdf(Y[o])
        Kg = S[np.ix_(miss, o)] @ np.linalg.inv(S[np.ix_(o, o)])
        mu = Kg @ Y[o]
        sd = np.sqrt(np.diag(S[np.ix_(miss, miss)] - Kg @ S[np.ix_(o, miss)]))
        Yn = np.where(miss, np.nan, Y)
        post = gaps.gap_posterior(m, Yn, xi=False)
        mean, s_ = _moments(post)
        assert abs(post.log_lik - ll) < 1e-6
        assert abs(predictive_pit(m, Yn).log_lik - ll) < 1e-6
        assert np.max(np.abs(mean - mu) / sd) < 1e-8
        assert np.max(np.abs(s_ / sd - 1.0)) < 1e-3


@pytest.mark.parametrize("name", ["repro", "sd"])
@pytest.mark.parametrize("tau", [0.99, 0.999])
def test_leading_gap_against_the_regime_ar1(name, tau):
    """Log-likelihood, state posteriors and the law of the missing values inside the gap."""
    # Measured, max over L = 5, 17 (N = 200, seed 1), errors of the mean and
    # sd in units of the exact sd:
    #            log-lik   gamma    mean     sd        (G = 64 / 128)
    #   repro    6.4e-4    9.3e-6   2.8e-3   4.6e-3   / 3.3e-5 8.3e-7 3.6e-4 4.4e-4
    #   sd       2.3e-3    2.7e-4   2.0e-2   1.2e-2   / 3.1e-6 8.8e-7 2.6e-3 2.8e-3
    # Old (sd model, ρ = 0.9999, L = 17): 0.65 nats, γ off by 0.11 at G = 64,
    # 3.4e-2 nats at G = 256.
    mm, ss, pp = MODELS[name]
    rho = math.sin(math.pi * tau / 2.0)
    m = _model(mm, ss, pp, tau)
    Y = _simulate(mm, ss, pp, rho, 200, 1)
    for L in (5, 17):
        Yn = Y.copy()
        Yn[:L] = np.nan
        ll, g, mu, sd = _regime_ar1_exact(mm, ss, pp, rho, Yn)
        for G, t_ll, t_g, t_m in ((64, 5e-3, 1e-3, 5e-2), (128, 1e-4, 5e-6, 1e-2)):
            post = gaps.gap_posterior(m, Yn, gap_nodes=G, xi=False)
            mean, s_ = _moments(post)
            assert abs(post.log_lik - ll) < t_ll, (L, G, post.log_lik - ll)
            assert np.abs(post.gamma - g).max() < t_g, (L, G)
            assert np.max(np.abs(mean - mu[:L]) / sd[:L]) < t_m, (L, G)
            assert np.max(np.abs(s_ / sd[:L] - 1.0)) < t_m, (L, G)


# ---------------------------------------------------------------------------
# Batch pass and PIT filter: the same chain
# ---------------------------------------------------------------------------

GAPS = {"leading": list(range(17)), "interior 2-3": [50, 51, 90, 91, 92, 150, 151, 200, 201, 202],
        "trailing": [297, 298, 299]}


@pytest.mark.parametrize("name", ["repro", "sd"])
@pytest.mark.parametrize("tau", [0.997, 0.999])
@pytest.mark.parametrize("pattern", list(GAPS))
def test_batch_and_filter_agree(name, tau, pattern):
    # Measured (G = 64, N = 300, seed 3): |log-lik difference| ≤ 2.3e-13,
    # state filters ≤ 1.7e-15 apart. Old: 2.4 nats (leading gap, τ = 0.999)
    # and state filters 0.26 apart after interior runs of 2–3 rows (repro,
    # τ = 0.997): blocks whose linear-space weights underflowed lost their mass.
    mm, ss, pp = MODELS[name]
    rho = math.sin(math.pi * tau / 2.0)
    m = _model(mm, ss, pp, tau)
    Y = _simulate(mm, ss, pp, rho, 300, 3)
    Y[GAPS[pattern]] = np.nan
    post = gaps.gap_posterior(m, Y, xi=False)
    P = predictive_pit(m, Y)
    assert abs(post.log_lik - P.log_lik) < 1e-9
    assert np.abs(post.alpha_hat - P.alpha_hat).max() < 1e-12


def test_linear_and_log_chains_agree():
    # Measured: 0.0 nats and 1.1e-15 (leading gap), 0.0 and 1.7e-15 (runs of
    # 2–3 rows). Old: 5.9e-2 nats and 2.4e-2 (leading gap, τ = 0.999).
    m = _model(*MODELS["repro"][:2], P_REPRO, 0.999)
    rho = math.sin(math.pi * 0.999 / 2.0)
    Y = _simulate(*MODELS["repro"], rho, 300, 3)
    for idx in (GAPS["leading"], GAPS["interior 2-3"]):
        Yn = Y.copy()
        Yn[idx] = np.nan
        miss = gaps.missing_mask(Yn)
        ref = gaps.reference_grid(m, 64)
        out = []
        for log in (False, True):
            ch = gaps._Chain(m, Yn, miss, ref, log=log)
            al, ll = gaps._forward_chain(ch)
            out.append((ll, gaps._marginal_alpha(ch, al)))
        assert abs(out[0][0] - out[1][0]) < 1e-9
        assert np.abs(out[0][1] - out[1][1]).max() < 1e-12


# ---------------------------------------------------------------------------
# The quadrature diagnostic
# ---------------------------------------------------------------------------

def test_diagnostic_covers_leading_gaps(caplog):
    # Measured, repro model τ = 0.99 (N = 200, seed 1), a 40-row leading gap:
    # quad_error 0.32 at G = 64, where the posterior means inside the gap are
    # 7.9e-2 exact sds off (the log-likelihood 4.5e-4 nats); 2.5e-6 at
    # G = 256 (errors 8e-6). Old: 3e-5 at both G (the rows of a leading gap
    # were not checked).
    mm, ss, pp = MODELS["repro"]
    rho = math.sin(math.pi * 0.99 / 2.0)
    m = _model(mm, ss, pp, 0.99)
    Y = _simulate(mm, ss, pp, rho, 200, 1)
    Y[:40] = np.nan
    _, _, mu, sd = _regime_ar1_exact(mm, ss, pp, rho, Y)
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        post = gaps.gap_posterior(m, Y, gap_nodes=64, xi=False)
    mean, _ = _moments(post)
    assert np.max(np.abs(mean - mu[:40]) / sd[:40]) > 2e-2
    assert post.quad_error > gaps.QUAD_WARN
    assert any("not converged" in r.message for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        post = gaps.gap_posterior(m, Y, gap_nodes=256, xi=False)
    mean, _ = _moments(post)
    assert np.max(np.abs(mean - mu[:40]) / sd[:40]) < 1e-4
    assert post.quad_error < 1e-2 * gaps.QUAD_WARN
    assert not any("not converged" in r.message for r in caplog.records)


def _eval_series(name, rho, seed):
    """The calibration series of the CHANGELOG: N = 300, a leading gap, runs of 1-6 rows."""
    mm, ss, pp = MODELS[name]
    Y = _simulate(mm, ss, pp, rho, 300, seed)
    rng = np.random.default_rng(100 + seed)
    miss = np.zeros(300, bool)
    miss[:rng.integers(1, 12)] = True
    n = 20
    while n < 285:
        L = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 6]))
        miss[n:n + L] = True
        n += L + int(rng.integers(3, 12))
    return np.where(miss, np.nan, Y)


@pytest.mark.parametrize("name,rho,G,off", [
    # Measured: |log-lik error| (exact) and quad_error. The filter-weighted
    # proposals put the first case, 0.17 nats off at G = 64 before them
    # (quad_error 0.92), at 2.6e-4 nats (quad_error 5.8e-3, quiet): it is
    # taken at G = 32 instead.
    ("sd", 0.999, 32, True),        # 0.18 nats; 1.02
    ("repro", 0.9999, 128, False),  # 2.8e-5 nats; 8.9e-4 (P5: 1.1e-3 nats; 0.026)
    ("sd", 0.99, 128, False),       # 2.3e-8 nats; 4.1e-12
])
def test_diagnostic_warns_when_the_answer_is_off(caplog, name, rho, G, off):
    """The WARNING fires on a pass off by more than 0.01 nats, not on one within 2e-3."""
    mm, ss, pp = MODELS[name]
    tau = 2.0 / math.pi * math.asin(rho)
    m = _model(mm, ss, pp, tau)
    Yn = _eval_series(name, rho, 1)
    ll = _regime_ar1_exact(mm, ss, pp, rho, Yn)[0]
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        post = gaps.gap_posterior(m, Yn, gap_nodes=G, xi=False)
    warned = any("not converged" in r.message for r in caplog.records)
    assert (abs(post.log_lik - ll) > 1e-2) is off
    assert warned is off and (post.quad_error > gaps.QUAD_WARN) is off


def test_report_parts_are_per_run():
    m = _model(*MODELS["sd"][:2], MODELS["sd"][2], 2.0 / math.pi * math.asin(0.999))
    Yn = _eval_series("sd", 0.999, 1)
    post, (chain, alphas, betas) = gaps._posterior(m, Yn, 64, False)
    total, parts = gaps._quadrature_report(chain, alphas, betas, per_run=True)
    starts = gaps._runs(gaps.missing_mask(Yn))[0]
    assert sorted(parts) == sorted(int(a) for a in starts)
    assert total == pytest.approx(sum(parts.values()), rel=1e-12)
    assert total == pytest.approx(post.quad_error, rel=1e-12)
