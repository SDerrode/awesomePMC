"""Missing observations on the quadrature grid — accuracy against exact references.

The grid variants (HMC-DN, PMC, PMC-IN with pair margins) integrate a missing
y_n on quadrature nodes (:mod:`pmcprg.pmc.gaps`). Every test here compares with
a reference computed independently of that grid:

* **Gaussian AR(1)** — identical states, N(0, 1) margins, Gaussian copulas of
  the same ρ: Y is a stationary Gaussian AR(1) (covariance ρ^|s−t|) and X is
  independent of Y. Observed-data log-likelihood, conditional mean / sd /
  quantiles of the missing values (closed forms for an isolated gap and for
  the h-step forecast) and γ = prior, for HMC-DN, PMC (state margins) and PMC
  with pair margins, over every gap pattern.
* **Distinct regimes (Gaussian HMC-DN / PMC with state margins)** — margins
  N(μ_i, σ_i²), Gaussian copulas ρ_ij: given a state path, y is a Gaussian
  Markov chain, so p(y_obs) = Σ_paths p(path) N(y_obs; μ_path, Σ_path) exactly
  (marginalising a Gaussian drops coordinates). N = 8, 256 paths: likelihood,
  γ, ξ, imputation moments and quantiles, forecast.
* **General PMC with pair margins, Clayton / Gumbel τ = 0.7**
  (CSDA-2013 Table 1) — brute force over the state paths with the missing y
  integrated by adaptive ``scipy.integrate.quad`` (isolated gap) or a fine
  composite Gauss–Legendre tensor rule (block of two), on transition densities
  written from the formulas (A16 Eqs. 12–14) with from-scratch copula
  densities.

Tolerances are set from measured errors at G = 64 (default), with a margin;
the measured tables are in the comments of each test.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import integrate, stats

from pmcprg.numerics import EPS, ONE_MINUS_EPS
from pmcprg.pmc import PMCModel
from pmcprg.pmc import gaps
from pmcprg.pmc.inference import backward, classify, forward, smooth

QS = (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# Gaussian AR(1): identical states
# ---------------------------------------------------------------------------

def _tau(rho):
    return 2.0 / math.pi * math.asin(rho)


def _ar1_model(kind, rho):
    tau = _tau(rho)
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
               "margins": [{"i": i, "j": j, **std} for i in range(2) for j in range(2)],
               "copulas": cops}
    return PMCModel.from_dict(raw)


def _ar1(N, rho, seed):
    rng = np.random.default_rng(seed)
    y = np.empty(N)
    y[0] = rng.standard_normal()
    for n in range(1, N):
        y[n] = rho * y[n - 1] + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
    return y


def _ar1_reference(Y, miss, rho):
    N = len(Y)
    idx = np.arange(N)
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    o, m = ~miss, miss
    yo = Y[o]
    ll = stats.multivariate_normal(mean=np.zeros(o.sum()), cov=S[np.ix_(o, o)]).logpdf(yo)
    Kg = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
    mu = Kg @ yo
    sd = np.sqrt(np.diag(S[np.ix_(m, m)] - Kg @ S[np.ix_(o, m)]))
    q = mu[:, None] + sd[:, None] * stats.norm.ppf(QS)[None, :]
    return ll, mu, sd, q


# Leading gap, isolated gap, block of five, every other position, trailing gap.
AR1_MISSING = [0, 1, 2, 5, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 30, 37, 38, 39]


@pytest.mark.parametrize("kind", ["HMC-DN", "PMC-state", "PMC-pair"])
@pytest.mark.parametrize("rho", [0.2, 0.5, 0.9, -0.7])
@pytest.mark.parametrize("seed", [0, 1])
def test_gaussian_ar1_all_gap_patterns(kind, rho, seed):
    # Measured, max over the three kinds and both seeds (seed 1 puts y = 2.8
    # next to the isolated gap):
    #              G     log-lik  mean     sd       quantiles
    #   ρ = 0.2    32    7.9e-7   2.0e-6   6.9e-6   3.1e-7
    #              64    5.5e-8   1.2e-7   4.8e-7   2.0e-8
    #              128   3.7e-9   7.0e-9   3.1e-8   1.2e-9
    #   ρ = 0.5    32    4.2e-6   4.5e-6   5.6e-6   1.7e-6
    #              64    5.7e-8   2.4e-7   4.9e-7   8.8e-8
    #              128   6.4e-9   9.5e-9   2.4e-8   4.1e-9
    #   ρ = 0.9    32    2.0e-6   5.3e-9   1.1e-8   1.3e-8
    #              64    2.7e-9   2.4e-11  6.6e-11  4.0e-11
    #              128   2.7e-12  1.2e-13  3.0e-13  2.3e-13
    #   ρ = −0.7   32    1.1e-5   3.9e-7   7.0e-7   6.2e-7
    #              64    4.5e-7   2.1e-8   3.8e-8   3.3e-8
    #              128   1.3e-8   1.0e-9   2.5e-9   2.5e-9
    # γ equals the prior to 1.2e-12 in every case.
    m = _ar1_model(kind, rho)
    N = 40
    Y = _ar1(N, rho, seed)
    if seed == 1:
        Y[4] = 2.8
    miss = np.zeros(N, bool)
    miss[AR1_MISSING] = True
    Yn = np.where(miss, np.nan, Y)
    ll_ref, mu_ref, sd_ref, q_ref = _ar1_reference(Y, miss, rho)

    imp = gaps.impute(m, Yn, quantiles=QS)
    assert imp.method == "grid"
    assert imp.log_lik == pytest.approx(ll_ref, abs=2e-6)
    np.testing.assert_allclose(imp.mean, mu_ref, atol=1e-6)
    np.testing.assert_allclose(imp.sd, sd_ref, atol=2e-6)
    np.testing.assert_allclose(imp.quantile_values, q_ref, atol=2e-6)

    # X is independent of Y: every posterior marginal is the prior P(x_n).
    _, gamma, ll = classify(m, Yn)
    assert ll == imp.log_lik
    pi = m.prior_p.sum(axis=0)
    np.testing.assert_allclose(gamma, np.broadcast_to(pi, gamma.shape), atol=1e-10)


@pytest.mark.parametrize("rho,tol", [(0.9, {32: 1e-5, 64: 1e-8, 128: 1e-11}),
                                     (0.99, {32: 1.0, 64: 2e-5, 128: 1e-9})])
def test_gaussian_ar1_convergence_in_G(rho, tol):
    # All gap patterns, max over the error kinds (HMC-DN, seed 0). Measured:
    #   ρ = 0.9:  G=32 2.3e-9 (quantiles 1.3e-8)  G=64 1.0e-11  G=128 4.8e-14
    #   ρ = 0.99: G=32 2.2e-1                     G=64 3.7e-6   G=128 1.4e-14
    # (ρ = 0.99 needs G ≥ 64: the conditional sd 0.14 spans few nodes.)
    m = _ar1_model("HMC-DN", rho)
    Y = _ar1(40, rho, 0)
    miss = np.zeros(40, bool)
    miss[AR1_MISSING] = True
    Yn = np.where(miss, np.nan, Y)
    ll_ref, mu_ref, sd_ref, q_ref = _ar1_reference(Y, miss, rho)
    err = {}
    for G in (32, 64, 128):
        imp = gaps.impute(m, Yn, gap_nodes=G, quantiles=QS)
        err[G] = max(abs(imp.log_lik - ll_ref), np.abs(imp.mean - mu_ref).max(),
                     np.abs(imp.sd - sd_ref).max(), np.abs(imp.quantile_values - q_ref).max())
    for G in (32, 64, 128):
        assert err[G] < tol[G]
    assert err[128] < err[64] < err[32]


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_gaussian_ar1_closed_forms_isolated_gap_and_block(rho):
    """Isolated gap: mean ρ(y_{m−1}+y_{m+1})/(1+ρ²), var (1−ρ²)/(1+ρ²).
    Gap of g values, L = g+1, position k: mean [ρ^k(1−ρ^{2(L−k)}) y_a +
    ρ^{L−k}(1−ρ^{2k}) y_b]/(1−ρ^{2L}), var (1−ρ^{2k})(1−ρ^{2(L−k)})/(1−ρ^{2L})."""
    m = _ar1_model("PMC-pair", rho)
    Y = _ar1(20, rho, 3)
    Yn = Y.copy()
    Yn[5] = np.nan
    Yn[10:14] = np.nan                               # g = 4, L = 5, y_a = Y[9], y_b = Y[14]
    imp = gaps.impute(m, Yn)
    r2 = rho * rho
    np.testing.assert_allclose(imp.mean[0], rho * (Y[4] + Y[6]) / (1 + r2), atol=1e-6)
    np.testing.assert_allclose(imp.sd[0] ** 2, (1 - r2) / (1 + r2), atol=1e-6)
    L = 5
    for k in range(1, 5):
        mean = (rho ** k * (1 - rho ** (2 * (L - k))) * Y[9]
                + rho ** (L - k) * (1 - rho ** (2 * k)) * Y[14]) / (1 - rho ** (2 * L))
        var = (1 - rho ** (2 * k)) * (1 - rho ** (2 * (L - k))) / (1 - rho ** (2 * L))
        np.testing.assert_allclose(imp.mean[k], mean, atol=1e-6)
        np.testing.assert_allclose(imp.sd[k] ** 2, var, atol=1e-6)


@pytest.mark.parametrize("rho", [0.5, 0.9])
def test_gaussian_ar1_forecast_closed_form(rho):
    """h-step forecast N(ρ^h y_N, 1 − ρ^{2h}); state probabilities = prior."""
    m = _ar1_model("HMC-DN", rho)
    Y = _ar1(30, rho, 4)
    Y[7] = np.nan
    fc = gaps.forecast(m, Y, 5, quantiles=QS)
    h = np.arange(1, 6)
    mean = rho ** h * Y[-1]
    sd = np.sqrt(1 - rho ** (2 * h))
    np.testing.assert_allclose(fc.mean, mean, atol=1e-6)
    np.testing.assert_allclose(fc.sd, sd, atol=1e-6)
    np.testing.assert_allclose(fc.quantile_values, mean[:, None] + sd[:, None] * stats.norm.ppf(QS),
                               atol=1e-6)
    np.testing.assert_allclose(fc.state_probs, np.broadcast_to(m.stationary_pi, (5, 2)), atol=1e-10)
    # the density on the grid integrates to 1 and matches the normal density
    np.testing.assert_allclose((fc.density * fc.nodes[None, :] ** 0) @ gaps.reference_grid(m).omega,
                               np.ones(5), atol=1e-10)
    np.testing.assert_allclose(fc.density[0], stats.norm.pdf(fc.nodes, mean[0], sd[0]),
                               atol=1e-6)


# ---------------------------------------------------------------------------
# Distinct regimes: Gaussian HMC-DN / state-margin PMC with Gaussian copulas
# ---------------------------------------------------------------------------

MU = np.array([-1.0, 1.2])
SIG = np.array([0.8, 1.3])
TAU = np.array([[0.6, -0.2], [0.3, 0.8]])
RHO = np.sin(np.pi * TAU / 2.0)


def _regime_model(variant):
    margins = [{"i": i, "dist": "norm", "params": {"loc": float(MU[i]), "scale": float(SIG[i])}}
               for i in range(2)]
    cops = [{"i": i, "j": j, "name": "Gauss", "tau": float(TAU[i, j])} for i in range(2) for j in range(2)]
    if variant == "HMC-DN":
        prior = {"A": [[0.85, 0.15], [0.25, 0.75]]}
    else:
        prior = {"p": [[0.42, 0.08], [0.08, 0.42]]}
    return PMCModel.from_dict({"model": {"variant": variant, "K": 2}, "prior": prior,
                               "margins": margins, "copulas": cops})


def _regime_reference(m, Y):
    """Exact log p(y_obs), γ, ξ and the Gaussian-mixture law of every missing y."""
    N = len(Y)
    miss = ~np.isfinite(Y)
    o = ~miss
    p = m.prior_p
    init = p.sum(axis=0)                           # α_1(j) = Σ_i p_ij f_j
    A = p / p.sum(axis=1, keepdims=True)
    logw, comps, paths = [], [], list(itertools.product(range(2), repeat=N))
    for path in paths:
        x = np.array(path)
        lp = math.log(init[x[0]]) + sum(math.log(A[x[n], x[n + 1]]) for n in range(N - 1))
        mu = MU[x]
        sd = SIG[x]
        # corr(z_s, z_t) = Π_{k=s}^{t-1} ρ_{x_k x_{k+1}}
        r = np.array([RHO[x[n], x[n + 1]] for n in range(N - 1)])
        C = np.eye(N)
        for s in range(N):
            for t in range(s + 1, N):
                C[s, t] = C[t, s] = np.prod(r[s:t])
        S = C * np.outer(sd, sd)
        if o.any():
            lp += stats.multivariate_normal(mean=mu[o], cov=S[np.ix_(o, o)]).logpdf(Y[o])
            Kg = S[np.ix_(miss, o)] @ np.linalg.inv(S[np.ix_(o, o)])
            cm = mu[miss] + Kg @ (Y[o] - mu[o])
            cv = np.diag(S[np.ix_(miss, miss)] - Kg @ S[np.ix_(o, miss)])
        else:
            cm, cv = mu, np.diag(S)
        logw.append(lp)
        comps.append((cm, cv))
    logw = np.array(logw)
    ll = float(np.logaddexp.reduce(logw))
    w = np.exp(logw - ll)
    gamma = np.zeros((N, 2))
    xi = np.zeros((N - 1, 2, 2))
    for wk, path in zip(w, paths):
        for n in range(N):
            gamma[n, path[n]] += wk
        for n in range(N - 1):
            xi[n, path[n], path[n + 1]] += wk
    CM = np.array([c[0] for c in comps])          # (paths, M)
    CV = np.array([c[1] for c in comps])
    mean = w @ CM
    sd = np.sqrt(w @ (CV + CM ** 2) - mean ** 2)
    q = np.empty((CM.shape[1], len(QS)))
    for k in range(CM.shape[1]):
        for a, lev in enumerate(QS):
            def F(y, k=k, lev=lev):
                return w @ stats.norm.cdf(y, CM[:, k], np.sqrt(CV[:, k])) - lev
            q[k, a] = __import__("scipy.optimize", fromlist=["brentq"]).brentq(F, -30, 30, xtol=1e-14)
    return ll, gamma, xi, mean, sd, q


REGIME_PATTERNS = {
    "isolated": [3],
    "block3": [2, 3, 4],
    "leading2": [0, 1],
    "trailing2": [6, 7],
    "several": [0, 2, 5, 6],
    "every-other": [1, 3, 5, 7],
    "all-but-one": [0, 1, 2, 4, 5, 6, 7],
}
Y_REGIME = np.array([-0.7, -1.4, 0.3, 1.9, 2.2, 0.9, -0.2, -1.1])


@pytest.mark.parametrize("variant", ["HMC-DN", "PMC"])
@pytest.mark.parametrize("pattern", sorted(REGIME_PATTERNS))
def test_distinct_regimes_against_gaussian_path_mixture(variant, pattern):
    # Measured, max over the 7 patterns and both variants:
    #           ll      γ       ξ       mean    sd      quantiles
    #   G=32    7.1e-4  5.6e-4  6.1e-4  7.0e-4  5.7e-4  1.0e-3
    #   G=64    6.0e-8  4.6e-8  5.8e-8  4.8e-7  1.5e-6  2.5e-6
    #   G=128   1.2e-9  9.0e-10 7.6e-10 7.5e-9  4.9e-8  8.2e-8
    # (Quantiles from the Legendre interpolant of the G node masses alone,
    # without the Nyström interpolation, were at 1.3e-4 for G = 64.)
    m = _regime_model(variant)
    Y = Y_REGIME.copy()
    Y[REGIME_PATTERNS[pattern]] = np.nan
    ll_ref, g_ref, xi_ref, mu_ref, sd_ref, q_ref = _regime_reference(m, Y)

    post = gaps.gap_posterior(m, Y)
    assert post.method == "grid"
    assert post.log_lik == pytest.approx(ll_ref, abs=5e-7)
    np.testing.assert_allclose(post.gamma, g_ref, atol=5e-7)
    np.testing.assert_allclose(post.xi, xi_ref, atol=5e-7)
    imp = gaps.impute(m, Y, quantiles=QS)
    np.testing.assert_allclose(imp.mean, mu_ref, atol=3e-6)
    np.testing.assert_allclose(imp.sd, sd_ref, atol=1e-5)
    np.testing.assert_allclose(imp.quantile_values, q_ref, atol=1e-5)
    # the K-state view: forward/backward/smooth reproduce γ and the likelihood
    a, ll = forward(m, Y)
    assert ll == pytest.approx(post.log_lik, abs=1e-12)
    np.testing.assert_allclose(smooth(a, backward(m, Y)), post.gamma, atol=1e-12)


@pytest.mark.parametrize("variant", ["HMC-DN", "PMC"])
def test_distinct_regimes_all_missing_gives_the_prior(variant):
    m = _regime_model(variant)
    Y = np.full(8, np.nan)
    ll_ref, g_ref, xi_ref, mu_ref, sd_ref, q_ref = _regime_reference(m, Y)
    post = gaps.gap_posterior(m, Y)
    # No observation: log-likelihood 0 and the prior as posterior, exactly —
    # the quadrature blocks are rescaled to the exact transition probabilities.
    assert ll_ref == pytest.approx(0.0, abs=1e-12)
    assert post.log_lik == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(post.gamma, g_ref, atol=1e-12)
    np.testing.assert_allclose(post.xi, xi_ref, atol=1e-12)
    # y marginals: the moments drift with the quadrature error of each step
    # (measured at G = 64: mean 3.1e-8, sd 6.5e-7, quantiles 1.2e-6).
    imp = gaps.impute(m, Y, quantiles=QS)
    np.testing.assert_allclose(imp.mean, mu_ref, atol=2e-6)
    np.testing.assert_allclose(imp.sd, sd_ref, atol=5e-6)
    np.testing.assert_allclose(imp.quantile_values, q_ref, atol=1e-5)


@pytest.mark.parametrize("variant", ["HMC-DN", "PMC"])
def test_distinct_regimes_forecast_equals_trailing_gap_reference(variant):
    # Measured at G = 64 (max of the two variants): ll 7.3e-9, P 4.8e-10,
    # mean 4.7e-7, sd 1.5e-6, quantiles 7.7e-7.
    m = _regime_model(variant)
    Y = Y_REGIME[:6].copy()
    Y[2] = np.nan
    fc = gaps.forecast(m, Y, 2, quantiles=QS)
    ext = np.concatenate([Y, [np.nan, np.nan]])
    ll_ref, g_ref, _, mu_ref, sd_ref, q_ref = _regime_reference(m, ext)
    assert fc.log_lik == pytest.approx(ll_ref, abs=5e-7)
    np.testing.assert_allclose(fc.state_probs, g_ref[6:], atol=5e-7)
    np.testing.assert_allclose(fc.mean, mu_ref[1:], atol=3e-6)
    np.testing.assert_allclose(fc.sd, sd_ref[1:], atol=1e-5)
    np.testing.assert_allclose(fc.quantile_values, q_ref[1:], atol=1e-5)


# ---------------------------------------------------------------------------
# General PMC with pair margins: Clayton / Gumbel τ = 0.7, brute force
# ---------------------------------------------------------------------------

TABLE1 = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}
P_T1 = np.array([[0.50, 0.05], [0.05, 0.40]])


def _pair_model(copula, variant="PMC"):
    if copula == "Clayton" and variant == "PMC":
        return PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml")
    raw = {"model": {"variant": variant, "K": 2}, "prior": {"p": P_T1.tolist()},
           "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                       for (i, j), (mu, sd) in sorted(TABLE1.items())]}
    if variant == "PMC":
        raw["copulas"] = [{"i": i, "j": j, "name": copula, "tau": 0.7} for i in range(2) for j in range(2)]
    return PMCModel.from_dict(raw)


def _clayton_pdf(u, v, tau):
    th = 2.0 * tau / (1.0 - tau)
    return (1 + th) * (u * v) ** (-1 - th) * (u ** -th + v ** -th - 1) ** (-2 - 1 / th)


def _gumbel_pdf(u, v, tau):
    # Nelsen (2006) family (4.2.4): c = C (uv)^{-1} (x y)^{θ-1} s^{-2+1/θ} (s^{1/θ} + θ − 1)
    th = 1.0 / (1.0 - tau)
    x, y = -np.log(u), -np.log(v)
    s = x ** th + y ** th
    return np.exp(-s ** (1 / th)) / (u * v) * (x * y) ** (th - 1) * s ** (-2 + 1 / th) * (s ** (1 / th) + th - 1)


def _pair_density(copula):
    """q(j, y1 | i, y0) and μ(i, y) from A16 Eqs. 12–14, vectorised in y0, y1."""
    cpdf = {"Clayton": _clayton_pdf, "GH": _gumbel_pdf, None: None}[copula]

    def f(i, j, y):
        return stats.norm.pdf(y, *TABLE1[(i, j)])

    def F(i, j, y):
        return np.clip(stats.norm.cdf(y, *TABLE1[(i, j)]), EPS, ONE_MINUS_EPS)

    def q(j, y1, i, y0):
        with np.errstate(invalid="ignore", divide="ignore", over="ignore", under="ignore"):
            den = sum(P_T1[i, k] * f(i, k, y0) for k in range(2))
            val = P_T1[i, j] * f(i, j, y0) / den * f(j, i, y1)
            if cpdf is not None:
                val = val * cpdf(F(i, j, y0), F(j, i, y1), 0.7)
        return np.nan_to_num(val, nan=0.0, posinf=0.0)

    def mu(i, y):
        return sum(P_T1[i, j] * f(i, j, y) for j in range(2))
    return q, mu


PANELS = [(-14.0, -4.0), (-4.0, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 16.0)]


def _brute_force_gap(copula, Y, gap):
    """log p(y_obs), γ for a gap of one or two consecutive missing indices."""
    q, mu = _pair_density(copula)
    N = len(Y)
    m0 = gap[0]
    L = len(gap)
    integrals = {}
    if L == 1:
        for a, b, c in itertools.product(range(2), repeat=3):
            def g(y, a=a, b=b, c=c):
                return q(b, y, a, Y[m0 - 1]) * q(c, Y[m0 + 1], b, y)
            integrals[(a, b, c)] = sum(integrate.quad(g, lo, hi, epsabs=0, epsrel=1e-12, limit=500)[0]
                               for lo, hi in PANELS)
    else:
        # Composite Gauss–Legendre, 8 nodes per 0.1-wide panel on [-10, 12]
        # (1760 nodes per axis; halving the panels changes log p by < 1e-14).
        xg, wg = np.polynomial.legendre.leggauss(8)
        edges = np.arange(-10.0, 12.0 + 1e-9, 0.1)
        a_, b_ = edges[:-1, None], edges[1:, None]
        z = (0.5 * (b_ - a_) * xg[None, :] + 0.5 * (a_ + b_)).ravel()
        wz = (0.5 * (b_ - a_) * wg[None, :]).ravel()
        mid = {(b, c): q(c, z[None, :], b, z[:, None]) for b in range(2) for c in range(2)}
        for a, b, c, d in itertools.product(range(2), repeat=4):
            left = q(b, z, a, Y[m0 - 1]) * wz
            right = q(d, Y[m0 + 2], c, z) * wz
            integrals[(a, b, c, d)] = float(left @ mid[(b, c)] @ right)
    joint = {}
    for path in itertools.product(range(2), repeat=N):
        pr = mu(path[0], Y[0])
        for n in range(N - 1):
            if m0 - 1 <= n <= m0 + L - 1:
                continue
            pr *= q(path[n + 1], Y[n + 1], path[n], Y[n])
        pr *= integrals[tuple(path[m0 - 1: m0 + L + 1])]
        joint[path] = pr
    Z = sum(joint.values())
    gamma = np.zeros((N, 2))
    for path, pr in joint.items():
        for n in range(N):
            gamma[n, path[n]] += pr / Z
    return math.log(Z), gamma


Y_PAIR = {
    "central": np.array([0.2, -0.4, 1.9, 1.1, 2.4]),
    "lower-tail": np.array([-1.2, -1.5, 0.3, -1.4, 0.1]),
    "upper-tail": np.array([2.5, 3.1, 0.0, 2.9, 1.0]),
}


@pytest.mark.parametrize("copula", ["Clayton", "GH"])
@pytest.mark.parametrize("case", sorted(Y_PAIR))
def test_pair_margins_isolated_gap_against_quad(copula, case):
    # Measured max(|Δ log-lik|, |Δγ|):
    #                         G=32     G=64     G=128    G=256
    #   Clayton central       2.0e-4   1.4e-8   1.0e-11  5.2e-13
    #   Clayton lower-tail    2.3e-2   1.4e-4   3.7e-11  2.7e-14
    #   Clayton upper-tail    4.8e-7   2.7e-8   1.6e-9   8.1e-11
    #   Gumbel  central       4.2e-5   2.9e-9   7.3e-12  2.9e-13
    #   Gumbel  lower-tail    3.7e-7   3.1e-9   1.2e-10  5.0e-12
    #   Gumbel  upper-tail    3.2e-3   2.7e-7   2.3e-12  3.6e-15
    # (lower-tail: y ≈ −1.4 on both sides, where Clayton's lower-tail
    # dependence concentrates the conditional law between nodes).
    m = _pair_model(copula)
    Y = Y_PAIR[case]
    Yn = Y.copy()
    Yn[2] = np.nan
    ll_ref, g_ref = _brute_force_gap(copula, Y, [2])
    err = {}
    for G in (32, 64, 128):
        post = gaps.gap_posterior(m, Yn, gap_nodes=G, xi=False)
        err[G] = max(abs(post.log_lik - ll_ref), np.abs(post.gamma - g_ref).max())
    assert err[64] < 5e-4
    assert err[128] < 1e-8
    assert err[128] < err[32]


def test_pmc_in_pair_margins_isolated_gap_against_quad():
    # PMC-IN with pair margins: the Eq. 13 factor A_ij(y) needs the grid too.
    # Measured max(|Δ log-lik|, |Δγ|): G=32 3.3e-7, G=64 1.0e-8, G=128 6.1e-10.
    m = _pair_model(None, variant="PMC-IN")
    assert gaps.needs_grid(m)
    Y = Y_PAIR["central"]
    Yn = Y.copy()
    Yn[2] = np.nan
    ll_ref, g_ref = _brute_force_gap(None, Y, [2])
    post = gaps.gap_posterior(m, Yn, xi=False)
    assert post.log_lik == pytest.approx(ll_ref, abs=5e-8)
    np.testing.assert_allclose(post.gamma, g_ref, atol=5e-8)


@pytest.mark.parametrize("copula", ["Clayton", "GH"])
@pytest.mark.parametrize("case", ["central", "lower"])
def test_pair_margins_block_of_two_against_tensor_rule(copula, case):
    # Measured max(|Δ log-lik|, |Δγ|):
    #                       G=32     G=64     G=128    G=256
    #   Clayton central     5.8e-5   9.2e-8   1.8e-10  9.0e-12
    #   Clayton lower       3.1e-2   4.8e-6   6.9e-11  6.2e-15
    #   Gumbel  central     1.2e-5   5.7e-9   6.0e-12  2.5e-13
    #   Gumbel  lower       2.3e-7   4.8e-9   2.0e-10  9.0e-12
    m = _pair_model(copula)
    Y = (np.array([0.4, -0.2, 1.3, 0.8, 1.9, 1.2]) if case == "central"
         else np.array([-1.0, -1.3, 0.0, 0.0, -1.2, -0.5]))
    Yn = Y.copy()
    Yn[[2, 3]] = np.nan
    ll_ref, g_ref = _brute_force_gap(copula, Y, [2, 3])
    for G, tol in ((64, 2e-5), (128, 2e-9)):
        post = gaps.gap_posterior(m, Yn, gap_nodes=G, xi=False)
        assert post.log_lik == pytest.approx(ll_ref, abs=tol)
        np.testing.assert_allclose(post.gamma, g_ref, atol=tol)
