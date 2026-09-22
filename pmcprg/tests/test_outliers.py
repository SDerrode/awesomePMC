"""Erroneous data: predictive PIT, outlier flags, flag-and-mask estimation.

:mod:`pmcprg.pmc.outliers` (erroneous-data pilot). Every reference here is
computed independently of the module:

* **Exactness** — the PIT against brute force over the state paths: HMC-IN
  (mixture of the state CDFs weighted by the path posteriors), Gaussian
  regimes with Gaussian copulas (given a path, y is a Gaussian Markov chain:
  the PIT is a mixture of Gaussian conditional CDFs, also across gaps), and
  a general PMC with pair margins written from DerrodePieczynski_CSDA2013
  Eqs. 12–14 with a from-scratch Gaussian copula; a Gaussian AR(1) regime
  (identical states) in closed form, gaps included.
* **h-function convention** — the PIT against the quadrature of the
  predictive density built from ``precompute_weights`` (Clayton and the
  non-exchangeable Clayton90): the copula argument of y_{n−1} is the
  conditioning one.
* **Consistency** — with ``forecast(h=1)`` on prefixes (the PIT of its
  quantile at level q is q), with ``gap_posterior`` (filter, log-likelihood)
  and with its own log predictive density (derivative of the PIT).
* **Calibration** — under the true model the PITs are uniform and
  independent (KS, Ljung–Box), and a misspecified model is rejected.
* **Sequential gating** — a single spike flags its neighbour without gating
  and not with it; the gated neighbour's PIT is the two-step closed form.

Seeds are fixed integers or ``zlib.crc32`` of a label; simulation and
contamination use distinct seeds. Tolerances come from the errors measured
at G = 64 (default) on macOS arm64, quoted next to each assertion with the
margin taken.
"""
from __future__ import annotations

import itertools
import math
import zlib

import numpy as np
import pytest
from scipy import integrate, stats

from pmcprg.pmc import (
    PMCModel,
    StateMissingness,
    flag_outliers,
    forecast,
    forward,
    gap_posterior,
    pit_checks,
    predictive_pit,
    robust_estimate,
    simulate,
)
from pmcprg.pmc.inference import precompute_weights
from pmcprg.pmc.outliers import _bh_threshold

MODELS = "pmcprg/pmc/models"


def _seed(label: str) -> int:
    return zlib.crc32(label.encode())


def _fixture(name: str) -> PMCModel:
    return PMCModel(f"{MODELS}/{name}.toml")


def _tau(rho: float) -> float:
    return 2.0 / math.pi * math.asin(rho)


# ---------------------------------------------------------------------------
# Models built in code
# ---------------------------------------------------------------------------

def _gauss_regimes(kind: str, mu, sig, rho, prior):
    """K = 2 Gaussian margins N(mu_i, sig_i²) and Gaussian copulas rho[i][j]."""
    K = len(mu)
    cops = [{"i": i, "j": j, "name": "Gauss", "tau": _tau(rho[i][j])}
            for i in range(K) for j in range(K)]
    margins = [{"i": i, "dist": "norm", "params": {"loc": mu[i], "scale": sig[i]}}
               for i in range(K)]
    if kind == "HMC-DN":
        raw = {"model": {"variant": "HMC-DN", "K": K}, "prior": {"A": prior},
               "margins": margins, "copulas": cops}
    else:
        raw = {"model": {"variant": "PMC", "K": K}, "prior": {"p": prior},
               "margins": margins, "copulas": cops}
    return PMCModel.from_dict(raw)


def _ar1_model(kind: str, rho: float) -> PMCModel:
    """Identical N(0, 1) states, Gaussian copulas ρ: y is a Gaussian AR(1)."""
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


def _ar1_series(N: int, rho: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.empty(N)
    y[0] = rng.standard_normal()
    for n in range(1, N):
        y[n] = rho * y[n - 1] + math.sqrt(1.0 - rho * rho) * rng.standard_normal()
    return y


def _ar1_pit(y: np.ndarray, rho: float) -> np.ndarray:
    """Closed form: y_n | y_m ~ N(ρ^L y_m, 1 − ρ^{2L}), m the last observed row."""
    out = np.full(len(y), np.nan)
    last = None
    for n in range(len(y)):
        if not np.isfinite(y[n]):
            continue
        if last is None:
            out[n] = stats.norm.cdf(y[n])
        else:
            L = n - last
            out[n] = stats.norm.cdf((y[n] - rho ** L * y[last]) / math.sqrt(1.0 - rho ** (2 * L)))
        last = n
    return out


# ---------------------------------------------------------------------------
# Brute-force references over the state paths
# ---------------------------------------------------------------------------

def _hmc_in_bruteforce(model: PMCModel, y: np.ndarray) -> np.ndarray:
    """PIT of every observed row of an HMC-IN by enumeration of x_{0:n}."""
    K = model.K
    A = model.transition_A
    pi = model.prior_p.sum(axis=0)
    dists = [stats.norm(**model.margin(k).params) for k in range(K)]
    out = np.full(len(y), np.nan)
    for n in range(len(y)):
        if not np.isfinite(y[n]):
            continue
        num = den = 0.0
        for path in itertools.product(range(K), repeat=n + 1):
            w = pi[path[0]] * math.prod(A[path[k - 1], path[k]] for k in range(1, n + 1))
            for k in range(n):
                if np.isfinite(y[k]):
                    w *= dists[path[k]].pdf(y[k])
            num += w * dists[path[n]].cdf(y[n])
            den += w
        out[n] = num / den
    return out


def _gauss_regime_bruteforce(mu, sig, rho, init, A, y: np.ndarray) -> np.ndarray:
    """PIT of the observed rows of a Gaussian-regime HMC-DN / state-margin PMC.

    Given a path, d_k = y_k − μ_{x_k} is a Gaussian AR chain, d_k = b_k d_{k−1}
    + s_k ε_k with b_k = ρ σ_{x_k}/σ_{x_{k−1}}, s_k = σ_{x_k} √(1 − ρ²) (the
    Gaussian copula between Gaussian margins); missing rows are dropped from
    the joint Gaussian (marginalisation), and the PIT is the path mixture of
    the conditional CDFs Φ((d_n − m)/s) weighted by p(path) N(d_obs; 0, Σ).
    """
    mu, sig, rho = np.asarray(mu), np.asarray(sig), np.asarray(rho)
    K = len(mu)
    out = np.full(len(y), np.nan)
    for n in range(len(y)):
        if not np.isfinite(y[n]):
            continue
        obs = [k for k in range(n) if np.isfinite(y[k])]
        num = den = 0.0
        for path in itertools.product(range(K), repeat=n + 1):
            path = np.array(path)
            w = init[path[0]] * math.prod(A[path[k - 1], path[k]] for k in range(1, n + 1))
            s = np.empty(n + 1)
            b = np.zeros(n + 1)
            s[0] = sig[path[0]]
            for k in range(1, n + 1):
                r = rho[path[k - 1], path[k]]
                s[k] = sig[path[k]] * math.sqrt(1.0 - r * r)
                b[k] = r * sig[path[k]] / sig[path[k - 1]]
            L = np.zeros((n + 1, n + 1))
            for m in range(n + 1):
                L[m, m] = s[m]
                for k in range(m + 1, n + 1):
                    L[k, m] = L[k - 1, m] * b[k]
            S = L @ L.T
            d = y[: n + 1] - mu[path]
            if obs:
                Soo = S[np.ix_(obs, obs)]
                w *= stats.multivariate_normal(mean=np.zeros(len(obs)), cov=Soo).pdf(d[obs])
                sol = np.linalg.solve(Soo, S[obs, n])
                m_c = float(sol @ d[obs])
                v_c = float(S[n, n] - S[n, obs] @ sol)
            else:
                m_c, v_c = 0.0, float(S[n, n])
            num += w * stats.norm.cdf((d[n] - m_c) / math.sqrt(v_c))
            den += w
        out[n] = num / den
    return out


def _gauss_copula_pdf(u, v, r):
    a, b = stats.norm.ppf(u), stats.norm.ppf(v)
    q = (a * a - 2 * r * a * b + b * b) / (1 - r * r)
    return math.exp(-0.5 * q + 0.5 * (a * a + b * b)) / math.sqrt(1 - r * r)


def _gauss_copula_h(v, u, r):
    return stats.norm.cdf((stats.norm.ppf(v) - r * stats.norm.ppf(u)) / math.sqrt(1 - r * r))


def _pair_pmc_bruteforce(p, mu, sig, rho, y: np.ndarray) -> np.ndarray:
    """General PMC (pair margins N(mu_ij, sig_ij²), Gaussian copulas rho_ij),
    complete data, written from DerrodePieczynski_CSDA2013 Eqs. 12–14."""
    K = len(p)
    f = lambda i, j, t: stats.norm.pdf(t, mu[i][j], sig[i][j])       # noqa: E731
    F = lambda i, j, t: stats.norm.cdf(t, mu[i][j], sig[i][j])       # noqa: E731

    def T(i, j, t):                                                   # Eq. 13
        return p[i][j] * f(i, j, t) / sum(p[i][k] * f(i, k, t) for k in range(K))

    def q(i, j, t, t2):                                               # Eqs. 13–14
        return T(i, j, t) * f(j, i, t2) * _gauss_copula_pdf(F(i, j, t), F(j, i, t2), rho[i][j])

    out = np.empty(len(y))
    out[0] = sum(p[i][j] * F(i, j, y[0]) for i in range(K) for j in range(K))
    for n in range(1, len(y)):
        num = den = 0.0
        for path in itertools.product(range(K), repeat=n + 1):
            w = sum(p[path[0]][j] * f(path[0], j, y[0]) for j in range(K))
            for k in range(1, n):
                w *= q(path[k - 1], path[k], y[k - 1], y[k])
            i, j = path[n - 1], path[n]
            w *= T(i, j, y[n - 1])
            num += w * _gauss_copula_h(F(j, i, y[n]), F(i, j, y[n - 1]), rho[i][j])
            den += w
        out[n] = num / den
    return out


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------

def test_hmc_in_pit_matches_path_enumeration():
    """HMC-IN: mixture of the state CDFs, with and without missing rows.

    Measured max |error| 2.2e-16 complete, 1.1e-16 with four missing rows
    (N = 9, 512 paths); tolerance 1e-13.
    """
    model = _fixture("hmc_in_gauss_k2")
    _, Y = simulate(model, 9, seed=_seed("hmc-in-exact"))
    ref = _hmc_in_bruteforce(model, Y)
    res = predictive_pit(model, Y)
    assert res.method == "exact"
    np.testing.assert_allclose(res.pit, ref, rtol=0, atol=1e-13)
    Ym = Y.copy()
    Ym[[0, 3, 4, 7]] = np.nan
    ref = _hmc_in_bruteforce(model, Ym)
    res = predictive_pit(model, Ym)
    assert np.isnan(res.pit[[0, 3, 4, 7]]).all()
    np.testing.assert_allclose(res.pit, ref, rtol=0, atol=1e-13)


REGIMES = dict(mu=[-1.0, 1.5], sig=[0.8, 1.3], rho=[[0.7, -0.3], [0.2, 0.85]])


@pytest.mark.parametrize("kind", ["HMC-DN", "PMC-state"])
def test_gaussian_regimes_pit_matches_path_enumeration(kind):
    """Gaussian regimes, Gaussian copulas: exact on complete data, quadrature
    error across gaps.

    Measured max |error|: complete data 5.6e-16 (HMC-DN) and 3.3e-16 (PMC);
    with an isolated gap and a block of two, 1.6e-8 and 5.2e-9 at G = 64,
    1.1e-10 and 3.0e-11 at G = 128, 4.8e-12 and 3.8e-12 at G = 256 (the
    quadrature of the grid). Tolerances 1e-13, 5e-8 (G = 64) and 1e-9
    (G = 128).
    """
    if kind == "HMC-DN":
        prior = [[0.85, 0.15], [0.25, 0.75]]
        A = np.array(prior)
    else:
        # x_1 ~ Σ_i p_ij, then A = p / Σ_j p_ij (state margins: X Markov).
        prior = [[0.40, 0.10], [0.15, 0.35]]
        P = np.array(prior)
        init, A = P.sum(axis=0), P / P.sum(axis=1, keepdims=True)
    model = _gauss_regimes(kind, REGIMES["mu"], REGIMES["sig"], REGIMES["rho"], prior)
    if kind == "HMC-DN":
        # The forward's law of x_1, Σ_i p_ij = (πA)_j with π the model's
        # stationary law (power iteration stopped at a 1e-12 step: πA − π is
        # 4e-13 here, which a brute force started from π would see at row 1).
        init = model.prior_p.sum(axis=0)
    _, Y = simulate(model, 8, seed=_seed(f"regimes-{kind}"))
    ref = _gauss_regime_bruteforce(REGIMES["mu"], REGIMES["sig"], REGIMES["rho"], init, A, Y)
    res = predictive_pit(model, Y)
    np.testing.assert_allclose(res.pit, ref, rtol=0, atol=1e-13)
    Ym = Y.copy()
    Ym[[2, 5, 6]] = np.nan
    ref = _gauss_regime_bruteforce(REGIMES["mu"], REGIMES["sig"], REGIMES["rho"], init, A, Ym)
    res = predictive_pit(model, Ym)
    assert res.method == "grid"
    np.testing.assert_allclose(res.pit, ref, rtol=0, atol=5e-8)
    fine = predictive_pit(model, Ym, gap_nodes=128)
    np.testing.assert_allclose(fine.pit, ref, rtol=0, atol=1e-9)


def test_pair_margin_pmc_pit_matches_path_enumeration():
    """General PMC with pair margins: T_ij(y) = p_ij f_ij(y)/D_i(y) and
    h_ij(F_ji(y') | F_ij(y)). Measured max |error| 4.5e-14 (the from-scratch
    copula density rounds at that level); tolerance 1e-12."""
    p = [[0.42, 0.08], [0.12, 0.38]]
    mu = [[-1.0, 0.2], [0.8, 1.6]]
    sig = [[0.9, 1.4], [1.2, 0.7]]
    rho = [[0.75, -0.4], [0.3, 0.6]]
    raw = {"model": {"variant": "PMC", "K": 2}, "prior": {"p": p},
           "margins": [{"i": i, "j": j, "dist": "norm",
                        "params": {"loc": mu[i][j], "scale": sig[i][j]}}
                       for i in range(2) for j in range(2)],
           "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": _tau(rho[i][j])}
                       for i in range(2) for j in range(2)]}
    model = PMCModel.from_dict(raw)
    assert model.margin_structure == "pair"
    _, Y = simulate(model, 8, seed=_seed("pair-exact"))
    ref = _pair_pmc_bruteforce(p, mu, sig, rho, Y)
    np.testing.assert_allclose(predictive_pit(model, Y).pit, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["HMC-DN", "PMC-state", "PMC-pair"])
def test_gaussian_ar1_pit_closed_form(kind):
    """Identical Gaussian states: the AR(1) closed form, across gaps of 1–3.

    Measured max |error| of the PIT 1.3e-10 and of log_pred 1.1e-9 (grid
    G = 64, ρ = 0.8, the three kinds alike); tolerances 1e-9 and 1e-8.
    """
    rho = 0.8
    y = _ar1_series(60, rho, _seed("ar1-series"))
    y[[5, 6, 7, 20, 40, 41]] = np.nan
    model = _ar1_model(kind, rho)
    res = predictive_pit(model, y)
    ref = _ar1_pit(y, rho)
    np.testing.assert_allclose(res.pit, ref, rtol=0, atol=1e-9)
    # log predictive density, closed form
    lp = np.full(len(y), np.nan)
    last = None
    for n in range(len(y)):
        if np.isnan(y[n]):
            continue
        if last is None:
            lp[n] = stats.norm.logpdf(y[n])
        else:
            L = n - last
            lp[n] = stats.norm.logpdf(y[n], rho ** L * y[last], math.sqrt(1 - rho ** (2 * L)))
        last = n
    np.testing.assert_allclose(res.log_pred, lp, rtol=0, atol=1e-8)


# ---------------------------------------------------------------------------
# The h-function convention: quadrature of the predictive density
# ---------------------------------------------------------------------------

def _copula_model(kind: str, fam00: str, tau00: float, fam01: str, tau01: float) -> PMCModel:
    cops = [{"i": 0, "j": 0, "name": fam00, "tau": tau00},
            {"i": 0, "j": 1, "name": fam01, "tau": tau01},
            {"i": 1, "j": 0, "name": "Gauss", "tau": 0.2},
            {"i": 1, "j": 1, "name": fam00, "tau": tau00}]
    if kind == "HMC-DN":
        return PMCModel.from_dict({
            "model": {"variant": "HMC-DN", "K": 2}, "prior": {"A": [[0.8, 0.2], [0.3, 0.7]]},
            "margins": [{"i": 0, "dist": "norm", "params": {"loc": -0.5, "scale": 1.0}},
                        {"i": 1, "dist": "gamma", "params": {"a": 3.0, "scale": 0.8}}],
            "copulas": cops})
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.1], [0.1, 0.35]]},
        "margins": [{"i": 0, "j": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 0, "j": 1, "dist": "norm", "params": {"loc": 0.5, "scale": 1.5}},
                    {"i": 1, "j": 0, "dist": "norm", "params": {"loc": 1.0, "scale": 1.2}},
                    {"i": 1, "j": 1, "dist": "norm", "params": {"loc": 1.8, "scale": 0.9}}],
        "copulas": cops})


@pytest.mark.parametrize("kind", ["HMC-DN", "PMC-pair"])
@pytest.mark.parametrize("fams", [("Clayton", 0.6, "Clayton90", -0.4),
                                  ("Clayton90", -0.5, "Frank", 0.3)])
def test_pit_is_the_integral_of_the_predictive_density(kind, fams):
    """PIT_n = ∫_{−∞}^{y_n} Σ_ij α̂_{n−1}(i) W(i, j; y_{n−1}, t) dt with W from
    precompute_weights and α̂ from forward — a swapped copula argument order
    gives h(u | v) instead of h(v | u) and fails (Clayton90 is not even
    exchangeable). Measured max |error| 1.0e-15 over the 12 cases (adaptive
    quad, epsabs 1e-12); tolerance 1e-11, from quad's requested accuracy.
    """
    model = _copula_model(kind, *fams)
    _, Y = simulate(model, 6, seed=_seed(f"hconv-{kind}-{fams[0]}"))
    res = predictive_pit(model, Y)
    for n in (1, 3, 5):
        # α̂_{n−1}: the last row of a forward pass on Y[:n] (N ≥ 2 required).
        a = forward(model, Y[: max(n, 2)])[0][n - 1]

        def dens(t, n=n, a=a):
            W, _ = precompute_weights(model, np.array([Y[n - 1], t]))
            return float(a @ W[0].sum(axis=1))

        val = integrate.quad(dens, -40.0, Y[n], points=[0.0] if Y[n] > 0 else None,
                             epsabs=1e-12, epsrel=1e-12, limit=200)[0]
        assert abs(res.pit[n] - val) < 1e-11, (n, res.pit[n], val)


# ---------------------------------------------------------------------------
# Consistency with the gap machinery and with forecast(h=1)
# ---------------------------------------------------------------------------

FIXTURES = ["hmc_in_gauss_k2", "hmc_dn_gauss_k2", "pmc_gauss_k2", "pmc_pair_gauss_k2",
            "pmc_in_gauss_k2"]


@pytest.mark.parametrize("name", FIXTURES)
def test_filter_matches_gap_posterior(name):
    """The filter of predictive_pit is that of gap_posterior on the same
    missing rows. Measured over the five fixtures: α̂ ≤ 6.1e-16,
    log-likelihood ≤ 5.7e-14 (N = 300); tolerances 1e-12 and 1e-10."""
    model = _fixture(name)
    _, Y = simulate(model, 300, seed=_seed(f"gp-{name}"))
    Y[[0, 30, 31, 32, 100, 299]] = np.nan
    res = predictive_pit(model, Y)
    gp = gap_posterior(model, Y, xi=False)
    np.testing.assert_allclose(res.alpha_hat, gp.alpha_hat, rtol=0, atol=1e-12)
    assert abs(res.log_lik - gp.log_lik) < 1e-10
    assert np.isnan(res.pit[[0, 30, 31, 32, 100, 299]]).all()
    assert np.isfinite(res.pit[~np.isnan(Y)]).all()


def test_filter_matches_gap_posterior_with_missingness_mechanism():
    """Non-ignorable missingness: the filter is given (y_obs, m), as
    gap_posterior. Measured: α̂ ≤ 3.9e-16, log-likelihood 0; tolerances as
    above."""
    for name in ("hmc_in_gauss_k2", "pmc_gauss_k2"):
        model = _fixture(name).with_missingness(StateMissingness(rates=[0.05, 0.3]))
        _, Y = simulate(model, 200, seed=_seed(f"mnar-{name}"))
        Y[[10, 11, 50, 120]] = np.nan
        res = predictive_pit(model, Y)
        gp = gap_posterior(model, Y, xi=False)
        np.testing.assert_allclose(res.alpha_hat, gp.alpha_hat, rtol=0, atol=1e-12)
        assert abs(res.log_lik - gp.log_lik) < 1e-10


@pytest.mark.parametrize("name", ["hmc_in_gauss_k2", "pmc_gauss_k2", "pmc_pair_gauss_k2"])
def test_pit_derivative_is_the_predictive_density(name):
    """d PIT_n / d y_n = exp(log_pred_n), and log_pred_n = log p(y_1:n) −
    log p(y_1:n−1) from gap_posterior, after a gap and on observed rows.

    Measured: central difference (h = 1e-5) relative error ≤ 2.6e-10; prefix
    likelihood difference ≤ 8.5e-15. Tolerances 1e-8 and 1e-12.
    """
    model = _fixture(name)
    _, Y = simulate(model, 200, seed=_seed(f"deriv-{name}"))
    Y[[30, 31, 32, 100]] = np.nan
    res = predictive_pit(model, Y)
    for n in (33, 101, 150):
        h = 1e-5
        Yp, Ym = Y[: n + 1].copy(), Y[: n + 1].copy()
        Yp[n] += h
        Ym[n] -= h
        d = (predictive_pit(model, Yp).pit[n] - predictive_pit(model, Ym).pit[n]) / (2 * h)
        assert abs(d / math.exp(res.log_pred[n]) - 1.0) < 1e-8
        ll1 = gap_posterior(model, Y[: n + 1], xi=False).log_lik
        ll0 = gap_posterior(model, Y[:n], xi=False).log_lik
        assert abs((ll1 - ll0) - res.log_pred[n]) < 1e-12


QS = (0.001, 0.05, 0.3, 0.5, 0.9, 0.999)


@pytest.mark.parametrize("name, G, tol", [("hmc_in_gauss_k2", 64, 1e-12),
                                          ("pmc_gauss_k2", 64, 5e-9),
                                          ("pmc_pair_gauss_k2", 128, 5e-8)])
def test_pit_of_forecast_quantiles(name, G, tol):
    """PIT_n(forecast(Y[:n], 1) quantile at level q) = q, on prefixes that end
    observed (n = 150), end in a gap of three (n = 33) or of one (n = 101).

    Measured max |PIT − q| over the six levels: HMC-IN 6.9e-15 (exact
    mixture quantiles, bisection to 1e-13); pmc_gauss_k2 1.3e-9 (G = 64);
    pmc_pair_gauss_k2 (Clayton τ = 0.7) 7.5e-6 at G = 64, 1.3e-8 at G = 128,
    1.0e-11 at G = 256 — the quadrature of the gap of three, shared by both
    sides and converging with G (on an observed prefix the PIT is exact and
    the error 7.4e-10 is forecast's Nyström quantile). Tolerances 1e-12,
    5e-9 and 5e-8 (G = 128).
    """
    model = _fixture(name)
    _, Y = simulate(model, 160, seed=_seed(f"fc-{name}"))
    Y[[30, 31, 32, 100]] = np.nan
    for n in (33, 101, 150):
        fc = forecast(model, Y[:n], 1, quantiles=QS, gap_nodes=G)
        for q, yq in zip(QS, fc.quantile_values[0]):
            Yq = Y[: n + 1].copy()
            Yq[n] = yq
            assert abs(predictive_pit(model, Yq, gap_nodes=G).pit[n] - q) < tol, (n, q)


def test_upper_tail_without_cancellation():
    """HMC-IN: 1 − PIT is summed from the margins' survival functions, so a
    +12 sd observation keeps a finite normal score and a p-value far below
    EPS (closed form: Σ_j w_j S_j(y))."""
    model = _fixture("hmc_in_gauss_k2")
    Y = np.array([-1.0, 1.0, 14.0])
    res = predictive_pit(model, Y)
    a = forward(model, Y[:2])[0][-1] @ model.transition_A
    sf = a[0] * stats.norm.sf(14.0, -1.0, 1.0) + a[1] * stats.norm.sf(14.0, 1.0, 1.0)
    assert res.pvalue[2] == pytest.approx(2 * sf, rel=1e-12)
    assert res.z[2] == pytest.approx(-stats.norm.ppf(sf), rel=1e-12)
    assert res.pvalue[2] < 1e-30


# ---------------------------------------------------------------------------
# Calibration under the true model
# ---------------------------------------------------------------------------

CALIB = ["hmc_in_gauss_k2", "hmc_dn_gauss_k2", "pmc_gauss_k2", "pmc_pair_gauss_k2"]


def test_pit_uniform_and_independent_under_the_model():
    """Four fixtures × 3 seeds, N = 1500 (10 % MCAR gaps on one seed per
    fixture): KS on the PITs and Ljung–Box (10 lags) on z and z².

    Under the model every p-value is U(0, 1): measured minimum over the 36
    p-values 0.030, over the four pooled KS tests (4500 PITs each) 0.0034
    (pmc_gauss_k2; with 20 series of 2000 per fixture the pooled KS p-values
    are 0.61, 0.99 and 0.65 for pmc_gauss_k2, pmc_pair_gauss_k2 and
    hmc_dn_gauss_k2 — no systematic miscalibration). Assert each > 1e-3, a
    Bonferroni level of 4 % for the 40 tests. A misspecified model (margin
    scales × 1.25) is rejected on one of the same series: measured KS
    p = 1.4e-5, asserted < 1e-3.
    """
    pvals = []
    for name in CALIB:
        model = _fixture(name)
        pooled = []
        for s in range(3):
            _, Y = simulate(model, 1500, seed=_seed(f"calib-{name}-{s}"))
            if s == 2:
                rng = np.random.default_rng(_seed(f"calib-mask-{name}"))
                Y[rng.random(1500) < 0.1] = np.nan
            res = predictive_pit(model, Y)
            c = pit_checks(res)
            pvals += [c.ks_pvalue, c.lb_pvalue, c.lb2_pvalue]
            pooled.append(res.pit[np.isfinite(res.pit)])
        assert stats.kstest(np.concatenate(pooled), "uniform").pvalue > 1e-3, name
    assert min(pvals) > 1e-3, pvals

    # Power: the same series under a model with scales × 1.25.
    model = _fixture("pmc_gauss_k2")
    raw = model.raw
    for b in raw["margins"]:
        b["params"]["scale"] *= 1.25
    wrong = PMCModel.from_dict(raw)
    _, Y = simulate(model, 1500, seed=_seed("calib-pmc_gauss_k2-0"))
    assert pit_checks(predictive_pit(wrong, Y)).ks_pvalue < 1e-3


# ---------------------------------------------------------------------------
# Flags: sequential gating, multiplicity
# ---------------------------------------------------------------------------

def test_sequential_gating_spares_the_spike_neighbour():
    """AR(1) regime ρ = 0.9 with a +10 spike at n = 100: without gating the
    clean y_101 is predicted from the spike and flagged; with gating it is
    predicted from y_99 across the gated row — the two-step closed form
    Φ((y_101 − ρ² y_99)/√(1 − ρ⁴)): measured error 3.9e-16 (the quadrature
    of one gated row of a Gaussian AR(1) converges to rounding at G = 64);
    tolerance 1e-12. Without gating the neighbour's p-value is 5.5e-65.
    """
    rho = 0.9
    model = _ar1_model("HMC-DN", rho)
    y = _ar1_series(300, rho, _seed("gating-series"))
    y[100] += 10.0
    seq = flag_outliers(model, y)
    raw = flag_outliers(model, y, sequential=False)
    assert seq.flagged[100] and raw.flagged[100]
    assert raw.flagged[101]
    assert not seq.flagged[101]
    ref = stats.norm.cdf((y[101] - rho ** 2 * y[99]) / math.sqrt(1 - rho ** 4))
    assert abs(seq.pit[101] - ref) < 1e-12
    assert set(seq.index) == {100}
    # Non-sequential p-values are predictive_pit's.
    np.testing.assert_array_equal(raw.pvalue, predictive_pit(model, y).pvalue)
    # The gated filter's likelihood integrates the flagged row out.
    ym = y.copy()
    ym[100] = np.nan
    assert abs(seq.log_lik - gap_posterior(model, ym, xi=False).log_lik) < 1e-10


def test_bh_threshold_and_fixed_point():
    """BH threshold against the step-up definition; the sequential BH flags
    are the BH set of their own p-values (fixed point)."""
    rng = np.random.default_rng(_seed("bh"))
    p = np.concatenate([rng.random(990), rng.random(10) * 1e-5])
    thr = _bh_threshold(p, 0.05)
    ps = np.sort(p)
    k = max(i + 1 for i in range(p.size) if ps[i] <= 0.05 * (i + 1) / p.size)
    assert thr == np.nextafter(0.05 * k / p.size, np.inf)
    assert _bh_threshold(np.array([0.5, 0.9]), 0.05) == 0.0

    model = _fixture("pmc_gauss_k2")
    _, Y = simulate(model, 1000, seed=_seed("bh-series"))
    rng = np.random.default_rng(_seed("bh-spikes"))
    pos = rng.choice(np.arange(2, 998, 3), 12, replace=False)
    Y[pos] += 5.0 * Y.std()
    f = flag_outliers(model, Y, alpha=0.05, correction="bh")
    obs = np.isfinite(f.pvalue)
    assert f.threshold == _bh_threshold(f.pvalue[obs], 0.05)
    np.testing.assert_array_equal(f.flagged, obs & (f.pvalue < f.threshold))
    bonf = flag_outliers(model, Y, alpha=0.05, correction="bonferroni")
    assert bonf.threshold == 0.05 / 1000
    assert set(bonf.index) <= set(f.index)


def test_null_false_flag_count():
    """Under the model the flags at α are Binomial(m, α) (independent
    p-values): 3 fixtures × 2 seeds × N = 2000 at α = 1e-2 give 120 expected
    flags; measured 123 (sd 10.9). Assert within ±4 sd of the Binomial."""
    total = m = 0
    for name in ("hmc_in_gauss_k2", "pmc_gauss_k2", "pmc_pair_gauss_k2"):
        model = _fixture(name)
        for s in range(2):
            _, Y = simulate(model, 2000, seed=_seed(f"null-{name}-{s}"))
            f = flag_outliers(model, Y, alpha=1e-2)
            total += f.n_flagged
            m += f.n_tests
    sd = math.sqrt(m * 1e-2 * (1 - 1e-2))
    assert abs(total - m * 1e-2) < 4 * sd, total


# ---------------------------------------------------------------------------
# Robust estimation
# ---------------------------------------------------------------------------

def test_robust_estimate_recovers_isolated_spikes():
    """HMC-IN, N = 1500, 15 isolated +8 sd spikes: the fixed point masks every
    spike (and one clean row, 3 fits: masks of 0, 15, 16 rows), the flags of
    the last fit equal its mask, and the fitted means are those of the fit on
    the clean series (measured max |Δμ| 0.0045; tolerance 0.02), where the
    raw fit is off by 0.105 (asserted > 0.08)."""
    model = _fixture("hmc_in_gauss_k2")
    _, Y = simulate(model, 1500, seed=_seed("robust-series"))
    rng = np.random.default_rng(_seed("robust-spikes"))
    pos = rng.choice(np.arange(1, 1499, 3), 15, replace=False)
    Yc = Y.copy()
    Yc[pos] += 8.0 * Y.std()
    cfg = {"fit_margins": True}
    rf = robust_estimate(model, Yc, cfg)
    assert rf.converged
    assert not rf.masks[0].any()
    assert set(pos) <= set(np.nonzero(rf.mask)[0])
    np.testing.assert_array_equal(rf.flags.flagged, rf.mask)
    assert len(rf.masks) == len(rf.traces) == rf.n_fits
    from pmcprg.pmc import ice
    clean, _ = ice(model, Y, cfg)
    raw, _ = ice(model, Yc, cfg)
    mu = lambda m: np.array([b["params"]["loc"] for b in m.margin_blocks()])  # noqa: E731
    assert np.abs(mu(rf.model) - mu(clean)).max() < 0.02
    assert np.abs(mu(raw) - mu(clean)).max() > 0.08


def test_robust_estimate_with_sem():
    """algorithm = "sem": the same loop around SEM (seeded, deterministic);
    every +8 sd spike of an HMC-IN series is masked at the fixed point."""
    model = _fixture("hmc_in_gauss_k2")
    _, Y = simulate(model, 800, seed=_seed("sem-series"))
    rng = np.random.default_rng(_seed("sem-spikes"))
    pos = rng.choice(np.arange(1, 799, 3), 8, replace=False)
    Y[pos] += 8.0 * Y.std()
    rf = robust_estimate(model, Y, {"fit_margins": True, "max_iter": 15}, algorithm="sem")
    assert type(rf.trace).__name__ == "SemTrace"
    assert rf.converged
    assert set(pos) <= set(np.nonzero(rf.mask)[0])


def test_robust_estimate_releases_clean_rows_of_the_initial_mask():
    """Rows of initial_mask the model does not flag are released after the
    first fit (clean series: masks of 4 then 0 rows); masks[0] is the
    initial mask."""
    model = _fixture("hmc_in_gauss_k2")
    _, Y = simulate(model, 800, seed=_seed("release-series"))
    init = np.zeros(800, bool)
    init[[10, 200, 201, 500]] = True
    rf = robust_estimate(model, Y, {"fit_margins": True}, initial_mask=init)
    np.testing.assert_array_equal(rf.masks[0], init)
    assert rf.converged
    assert not rf.mask[[10, 200, 201, 500]].any()


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def test_refusals():
    mvn = _fixture("hmc_in_mvn_k2_d3")
    with pytest.raises(NotImplementedError, match="scalar"):
        predictive_pit(mvn, np.zeros((10, 3)))
    model = _fixture("hmc_in_gauss_k2")
    Y = simulate(model, 50, seed=1)[1]
    with pytest.raises(ValueError, match="alpha"):
        flag_outliers(model, Y, alpha=0.0)
    with pytest.raises(ValueError, match="correction"):
        flag_outliers(model, Y, correction="holm")
    with pytest.raises(ValueError, match="flag_cfg"):
        robust_estimate(model, Y, flag_cfg={"level": 0.01})
    mnar = model.with_missingness(StateMissingness(rates=[0.1, 0.2]))
    with pytest.raises(NotImplementedError, match="missingness"):
        robust_estimate(mnar, Y)
