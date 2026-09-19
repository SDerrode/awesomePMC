"""Non-ignorable missingness (P6) — references independent of the implementation.

With a state-dependent mechanism (``[missingness]``, :mod:`pmcprg.pmc.missingness`)
the mask m is evidence on the states: p(m | x) = Π_n e_n(x_n) and

    p(y_obs, m) = Σ_x ∫ p(x, y) Π_n e_n(x_n) dy_miss.

Every reference here enumerates the K^N state paths of a short sequence
(N = 6), weights each path by its integrated joint density times
Π_n e_n(x_n) — the factors written below from the definitions of the two
mechanisms, not taken from the package — and sums:

* **exact-shortcut variants** (HMC-IN, HMC-IN2, PMC-IN with state margins,
  d = 1 and d = 3): the integral of a missing y along a fixed path is 1;
* **Gaussian-copula regimes** (HMC-DN, PMC with state margins): given a path,
  y is a Gaussian Markov chain, so p(y_obs | x) is a Gaussian density and the
  law of a missing y given (x, y_obs) a Gaussian conditional — exact, which
  also gives the posterior mean, sd and quantiles of the missing values;
* **Clayton PMC (state and pair margins) and PMC-IN with pair margins**:
  isolated gaps (leading, interior, trailing, or two separate interior ones)
  integrated by adaptive ``scipy.integrate.quad`` on transition densities
  written from DerrodePieczynski_CSDA2013 Eqs. 12–14 with a from-scratch
  Clayton density.

Mask patterns include a leading, an interior and a trailing gap, a block of
two, two separate gaps, and a complete Y (whose mask m = 0 still carries
e_n(i) = p(m_n = 0 | …)). The grid tolerances are set from the errors
measured at G = 64 for these very cases, next to the errors of the
ignorable model on the same data (the quadrature error the factors inherit),
reported in each test.
"""
from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest
from scipy import integrate, optimize, stats

from pmcprg.numerics import EPS, ONE_MINUS_EPS
from pmcprg.pmc import PMCModel, gaps, simulate
from pmcprg.pmc import inference as inf

QS = (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# Evidence factors from the definitions (independent of pmcprg.pmc.missingness)
# ---------------------------------------------------------------------------

def _ev_state(miss, rates):
    """e_n(i) = π_i if m_n = 1 else 1 − π_i."""
    return np.array([[r if m else 1.0 - r for r in rates] for m in miss], dtype=float)


def _ev_markov(miss, onset, persistence):
    """e_0(i) = s_i^{m_0} (1 − s_i)^{1 − m_0}, s_i = a_i / (1 − b_i + a_i);
    e_n(i) = P(m_n | m_{n−1}, x_n = i) with P(m_n = 1 | m_{n−1} = 0) = a_i and
    P(m_n = 1 | m_{n−1} = 1) = b_i."""
    out = np.empty((len(miss), len(onset)))
    for i, (a, b) in enumerate(zip(onset, persistence)):
        s = a / (1.0 - b + a)
        out[0, i] = s if miss[0] else 1.0 - s
        for n in range(1, len(miss)):
            p1 = b if miss[n - 1] else a
            out[n, i] = p1 if miss[n] else 1.0 - p1
    return out


RATES = [0.15, 0.6, 0.35]
ONSET = [0.1, 0.35, 0.2]
PERSIST = [0.3, 0.8, 0.55]


def _mechanism(kind, K):
    """(``[missingness]`` table, evidence function of the mask) for K states."""
    if kind == "state":
        return ({"mechanism": "state", "rates": RATES[:K]},
                lambda miss: _ev_state(miss, RATES[:K]))
    return ({"mechanism": "state-markov", "onset": ONSET[:K], "persistence": PERSIST[:K]},
            lambda miss: _ev_markov(miss, ONSET[:K], PERSIST[:K]))


def _with(m, kind):
    table, ev = _mechanism(kind, m.K)
    raw = m.raw
    raw["missingness"] = table
    return PMCModel.from_dict(raw), ev


def _posteriors(lw, paths, N, K):
    """log Σ exp(lw), γ and ξ of the path weights."""
    ll = float(np.logaddexp.reduce(lw))
    w = np.exp(lw - ll)
    gamma = np.zeros((N, K))
    xi = np.zeros((N - 1, K, K))
    for wk, x in zip(w, paths):
        gamma[np.arange(N), x] += wk
        xi[np.arange(N - 1), x[:-1], x[1:]] += wk
    return ll, gamma, xi, w


# ---------------------------------------------------------------------------
# Exact-shortcut variants
# ---------------------------------------------------------------------------

def _shortcut_reference(m, Y, ev):
    """Σ over paths of (Σ_i p_{i x_1}) Π A Π f^{obs} Π e — the integral of a
    missing y along a fixed path is 1."""
    N, K = len(Y), m.K
    fin = np.isfinite(Y) if Y.ndim == 1 else np.isfinite(Y).all(axis=1)
    p = m.prior_p
    init = p.sum(axis=0)
    A = m.transition_A if m.variant.has_markov_prior else p / p.sum(axis=1, keepdims=True)
    logf = np.zeros((N, K))
    for n in range(N):
        if fin[n]:
            logf[n] = [m.margin(k).logpdf(Y[n]) for k in range(K)]
    with np.errstate(divide="ignore"):
        loge, logA, loginit = np.log(ev), np.log(A), np.log(init)
    paths = [np.array(x) for x in itertools.product(range(K), repeat=N)]
    lw = np.array([loginit[x[0]] + logA[x[:-1], x[1:]].sum()
                   + logf[np.arange(N), x].sum() + loge[np.arange(N), x].sum() for x in paths])
    return _posteriors(lw, paths, N, K)[:3]


SHORTCUT = ["hmc_in_gauss_k2.toml", "hmc_in2_gauss_k2.toml", "hmc_in_gauss_k3.toml",
            "pmc_in_gauss_k2.toml", "hmc_in_mvn_k2_d3.toml"]
PATTERNS = {"lead-mid-trail": [0, 3, 5], "block-and-isolated": [1, 2, 4], "isolated": [2],
            "complete": []}


@pytest.mark.parametrize("name", SHORTCUT)
@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("kind", ["state", "state-markov"])
def test_shortcut_against_path_enumeration(name, pattern, kind):
    # Exact on both sides: measured max |Δ| over the 40 cases 7.1e-15
    # (log-lik) and 3.8e-15 (γ, ξ), linear and log-space passes alike —
    # rounding only; 1e-12 leaves a factor > 100.
    tol = 1e-12
    m0 = PMCModel(f"pmcprg/pmc/models/{name}")
    m, ev_of = _with(m0, kind)
    _, Y = simulate(m0, N=6, seed=17)
    Y = np.array(Y, dtype=float)
    Y[PATTERNS[pattern]] = np.nan
    miss = gaps.missing_mask(Y)
    ll_ref, g_ref, xi_ref = _shortcut_reference(m, Y, ev_of(miss))

    W, f = inf.precompute_weights(m, Y)
    a, ll = inf.forward(m, Y, W=W, f_pdf=f)
    b = inf.backward(m, Y, W=W)
    assert ll == pytest.approx(ll_ref, abs=tol)
    np.testing.assert_allclose(inf.smooth(a, b), g_ref, atol=tol)
    np.testing.assert_allclose(inf.joint_posteriors(a, W, b), xi_ref, atol=tol)
    _, g, ll2 = inf.classify(m, Y)
    np.testing.assert_allclose(g, g_ref, atol=tol)
    assert ll2 == pytest.approx(ll_ref, abs=tol)
    post = gaps.gap_posterior(m, Y)
    assert post.method == "exact" and post.log_lik == pytest.approx(ll_ref, abs=tol)
    np.testing.assert_allclose(post.xi, xi_ref, atol=tol)
    # the log-space passes carry the factors too
    a_log, ll_log = inf._forward_log_space(m, Y)
    assert ll_log == pytest.approx(ll_ref, abs=tol)
    np.testing.assert_allclose(inf.smooth(a_log, inf._backward_log_space(m, Y)), g_ref, atol=tol)


# ---------------------------------------------------------------------------
# Gaussian-copula regimes: exact Gaussian path mixture
# ---------------------------------------------------------------------------

MU = np.array([-1.0, 1.2])
SIG = np.array([0.8, 1.3])
TAU = np.array([[0.6, -0.2], [0.3, 0.8]])
RHO = np.sin(np.pi * TAU / 2.0)          # Gaussian copula: τ = (2/π) arcsin ρ


def _regime_model(variant):
    margins = [{"i": i, "dist": "norm", "params": {"loc": float(MU[i]), "scale": float(SIG[i])}}
               for i in range(2)]
    cops = [{"i": i, "j": j, "name": "Gauss", "tau": float(TAU[i, j])}
            for i in range(2) for j in range(2)]
    prior = ({"A": [[0.85, 0.15], [0.25, 0.75]]} if variant == "HMC-DN"
             else {"p": [[0.42, 0.08], [0.08, 0.42]]})
    return PMCModel.from_dict({"model": {"variant": variant, "K": 2}, "prior": prior,
                               "margins": margins, "copulas": cops})


def _regime_reference(m, Y, ev):
    """log p(y_obs, m), γ, ξ and the mean, sd, quantiles of every missing y.

    Given the path x, z_n = (y_n − μ_{x_n}) / σ_{x_n} is a Gaussian AR chain with
    corr(z_s, z_t) = Π_{k=s}^{t−1} ρ_{x_k x_{k+1}}; the law of y_miss given
    (x, y_obs) does not involve the mask.
    """
    N = len(Y)
    miss = ~np.isfinite(Y)
    o = ~miss
    p = m.prior_p
    init = p.sum(axis=0)
    A = p / p.sum(axis=1, keepdims=True)
    paths = [np.array(x) for x in itertools.product(range(2), repeat=N)]
    lw, comps = [], []
    for x in paths:
        lp = math.log(init[x[0]]) + np.log(A[x[:-1], x[1:]]).sum() + np.log(ev[np.arange(N), x]).sum()
        r = RHO[x[:-1], x[1:]]
        C = np.eye(N)
        for s in range(N):
            for t in range(s + 1, N):
                C[s, t] = C[t, s] = np.prod(r[s:t])
        S = C * np.outer(SIG[x], SIG[x])
        mu = MU[x]
        if o.any():
            lp += stats.multivariate_normal(mean=mu[o], cov=S[np.ix_(o, o)]).logpdf(Y[o])
            Kg = S[np.ix_(miss, o)] @ np.linalg.inv(S[np.ix_(o, o)])
            cm = mu[miss] + Kg @ (Y[o] - mu[o])
            cv = np.diag(S[np.ix_(miss, miss)] - Kg @ S[np.ix_(o, miss)])
        else:
            cm, cv = mu, np.diag(S)
        lw.append(lp)
        comps.append((cm, cv))
    ll, gamma, xi, w = _posteriors(np.array(lw), paths, N, 2)
    CM = np.array([c[0] for c in comps])
    CS = np.sqrt(np.array([c[1] for c in comps]))
    mean = w @ CM
    sd = np.sqrt(w @ (CS ** 2 + CM ** 2) - mean ** 2)
    q = np.array([[optimize.brentq(lambda y, k=k, lev=lev: w @ stats.norm.cdf(y, CM[:, k], CS[:, k]) - lev,
                                   -30.0, 30.0, xtol=1e-14) for lev in QS]
                  for k in range(CM.shape[1])]).reshape(CM.shape[1], len(QS))
    return ll, gamma, xi, mean, sd, q


Y_REGIME = np.array([-0.7, -1.4, 0.3, 1.9, 2.2, 0.9])


@pytest.mark.parametrize("variant", ["HMC-DN", "PMC"])
@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("kind", ["state", "state-markov"])
def test_gaussian_regimes_against_path_mixture(variant, pattern, kind):
    # Measured at G = 64, max over the 2 variants × 4 patterns:
    #                 log-lik  γ, ξ     mean     sd       quantiles
    #   ignorable     7.2e-8   6.0e-8   1.1e-7   1.1e-7   1.4e-6
    #   state         9.9e-8   1.0e-7   2.9e-7   9.1e-7   1.1e-7
    #   state-markov  1.2e-7   1.1e-7   2.8e-7   8.6e-7   2.9e-7
    # The factors reweight the paths; the errors stay at the level of the
    # ignorable model's quadrature error (first row, same data). Tolerances
    # 4–5× the largest measured value of each column.
    m0 = _regime_model(variant)
    m, ev_of = _with(m0, kind)
    Y = Y_REGIME.copy()
    Y[PATTERNS[pattern]] = np.nan
    ll_ref, g_ref, xi_ref, mu_ref, sd_ref, q_ref = _regime_reference(m, Y, ev_of(gaps.missing_mask(Y)))

    post = gaps.gap_posterior(m, Y)
    assert post.log_lik == pytest.approx(ll_ref, abs=5e-7)
    np.testing.assert_allclose(post.gamma, g_ref, atol=5e-7)
    np.testing.assert_allclose(post.xi, xi_ref, atol=5e-7)
    imp = gaps.impute(m, Y, quantiles=QS)
    np.testing.assert_allclose(imp.mean, mu_ref, atol=1.5e-6)
    np.testing.assert_allclose(imp.sd, sd_ref, atol=4e-6)
    np.testing.assert_allclose(imp.quantile_values, q_ref, atol=6e-6)
    np.testing.assert_allclose(imp.gamma, g_ref[imp.index], atol=5e-7)
    # the K-state view of the same run
    a, ll = inf.forward(m, Y)
    assert ll == pytest.approx(post.log_lik, abs=1e-12)
    np.testing.assert_allclose(inf.smooth(a, inf.backward(m, Y)), post.gamma, atol=1e-12)


# ---------------------------------------------------------------------------
# Clayton PMC (state and pair margins), PMC-IN with pair margins: quad
# ---------------------------------------------------------------------------

P_SYM = np.array([[0.50, 0.05], [0.05, 0.40]])
PAIR_MARGINS = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}
STATE_MARGINS = {0: (-0.6, 0.9), 1: (1.0, 1.2)}


def _clayton_pdf(u, v, tau):
    # C(u, v) = (u^−θ + v^−θ − 1)^(−1/θ), θ = 2τ/(1 − τ);
    # ∂²C/∂u∂v = (1 + θ) (uv)^(−1−θ) (u^−θ + v^−θ − 1)^(−2−1/θ).
    th = 2.0 * tau / (1.0 - tau)
    return (1.0 + th) * (u * v) ** (-1.0 - th) * (u ** -th + v ** -th - 1.0) ** (-2.0 - 1.0 / th)


def _quad_case(kind):
    """(model, q, μ): q(j, y1 | i, y0) and μ(i, y) from DerrodePieczynski_CSDA2013
    Eqs. 12–14 — p_ij f_ij(y0) / Σ_k p_ik f_ik(y0) · f_ji(y1) [· c_ij(F_ij(y0),
    F_ji(y1))], μ(i, y) = Σ_j p_ij f_ij(y) — with f_ij = f_i for state margins."""
    if kind == "state-clayton":
        par = {(i, j): STATE_MARGINS[i] for i in range(2) for j in range(2)}
        margins = [{"i": i, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                   for i, (mu, sd) in sorted(STATE_MARGINS.items())]
    else:
        par = PAIR_MARGINS
        margins = [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                   for (i, j), (mu, sd) in sorted(PAIR_MARGINS.items())]
    tau = None if kind == "pair-in" else 0.7
    raw = {"model": {"variant": "PMC-IN" if tau is None else "PMC", "K": 2},
           "prior": {"p": P_SYM.tolist()}, "margins": margins}
    if tau is not None:
        raw["copulas"] = [{"i": i, "j": j, "name": "Clayton", "tau": tau}
                          for i in range(2) for j in range(2)]

    def f(i, j, y):
        return stats.norm.pdf(y, *par[(i, j)])

    def F(i, j, y):
        return min(max(stats.norm.cdf(y, *par[(i, j)]), EPS), ONE_MINUS_EPS)

    def q(j, y1, i, y0):
        val = P_SYM[i, j] * f(i, j, y0) / sum(P_SYM[i, k] * f(i, k, y0) for k in range(2)) * f(j, i, y1)
        if tau is not None:
            val *= _clayton_pdf(F(i, j, y0), F(j, i, y1), tau)
        return val

    def mu(i, y):
        return sum(P_SYM[i, j] * f(i, j, y) for j in range(2))
    return PMCModel.from_dict(raw), q, mu


Y_QUAD = np.array([0.2, -0.4, 1.9, 1.1, 2.4, 0.7])
QUAD_PATTERNS = {"lead-mid-trail": [0, 2, 5], "two-interior": [1, 4]}


@functools.lru_cache(maxsize=None)
def _quad_path_joint(case, pattern):
    """log ∫ p(x, y) dy_miss for every path x — isolated missing positions.

    Along a path, a missing y_m enters μ(x_0, y_m) (m = 0) or q(x_m, y_m | x_{m−1},
    y_{m−1}), and q(x_{m+1}, y_{m+1} | x_m, y_m) (m < N−1): one integral per gap
    and per triple of states, by adaptive quad (relative tolerance 1e-12) on
    panels split at −4, 0, 2, 5 and at the neighbouring observations. Cached:
    the mechanisms only reweight these paths.
    """
    _, q, mu = _quad_case(case)
    Y = Y_QUAD.copy()
    Y[QUAD_PATTERNS[pattern]] = np.nan
    N, K = len(Y), 2
    miss = np.isnan(Y)
    idx = np.nonzero(miss)[0]
    assert not np.any(np.diff(idx) == 1)
    cache = {}

    def integral(mpos, xl, xm, xr):
        key = (mpos, xl, xm, xr)
        if key not in cache:
            def g(y):
                left = mu(xm, y) if mpos == 0 else q(xm, y, xl, Y[mpos - 1])
                return left * (q(xr, Y[mpos + 1], xm, y) if mpos < N - 1 else 1.0)
            cuts = {-14.0, -4.0, 0.0, 2.0, 5.0, 16.0}
            cuts |= {float(Y[k]) for k in (mpos - 1, mpos + 1) if 0 <= k < N}
            cuts = sorted(cuts)
            cache[key] = sum(integrate.quad(g, lo, hi, epsabs=0.0, epsrel=1e-12, limit=500)[0]
                             for lo, hi in zip(cuts[:-1], cuts[1:]))
        return cache[key]

    paths = tuple(itertools.product(range(K), repeat=N))
    lj = []
    for x in paths:
        w = 1.0 if miss[0] else mu(x[0], Y[0])
        for n in range(N - 1):
            if not (miss[n] or miss[n + 1]):
                w *= q(x[n + 1], Y[n + 1], x[n], Y[n])
        for mp in idx:
            w *= integral(mp, x[mp - 1] if mp > 0 else None, x[mp], x[mp + 1] if mp < N - 1 else None)
        lj.append(math.log(w))
    return paths, np.array(lj)


def _quad_reference(case, pattern, ev):
    """log p(y_obs, m), γ, ξ: the path joints of :func:`_quad_path_joint` × Π e."""
    paths, lj = _quad_path_joint(case, pattern)
    N = len(paths[0])
    paths = [np.array(x) for x in paths]
    lw = lj + np.array([np.log(ev[np.arange(N), x]).sum() for x in paths])
    return _posteriors(lw, paths, N, 2)[:3]


@pytest.mark.parametrize("case", ["state-clayton", "pair-clayton", "pair-in"])
@pytest.mark.parametrize("pattern", sorted(QUAD_PATTERNS))
@pytest.mark.parametrize("kind", [None, "state", "state-markov"])
def test_quadrature_cases_against_path_enumeration(case, pattern, kind):
    # Measured max(|Δ log-lik|, |Δγ|, |Δξ|) at G = 64 (G = 128), over the two
    # patterns:
    #                    ignorable          state              state-markov
    #   state-clayton    5.9e-9 (4.6e-10)   4.8e-9 (3.2e-10)   4.5e-9 (3.3e-10)
    #   pair-clayton     4.7e-7 (2.2e-10)   5.7e-7 (5.9e-10)   3.9e-7 (6.2e-10)
    #   pair-in          1.2e-8 (7.0e-10)   7.9e-9 (4.6e-10)   8.5e-9 (5.0e-10)
    # The ignorable column (kind None) is the quadrature error the factors
    # inherit: same order with them. Tolerances 2e-6 at G = 64 and 3e-9 at
    # G = 128 (3.5× and 5× the largest measured value).
    m, _, _ = _quad_case(case)
    Y = Y_QUAD.copy()
    Y[QUAD_PATTERNS[pattern]] = np.nan
    miss = gaps.missing_mask(Y)
    if kind is None:
        ev = np.ones((len(Y), 2))
    else:
        m, ev_of = _with(m, kind)
        ev = ev_of(miss)
    ll_ref, g_ref, xi_ref = _quad_reference(case, pattern, ev)
    for G, tol in ((64, 2e-6), (128, 3e-9)):
        post = gaps.gap_posterior(m, Y, gap_nodes=G)
        assert post.method == "grid"
        assert post.log_lik == pytest.approx(ll_ref, abs=tol)
        np.testing.assert_allclose(post.gamma, g_ref, atol=tol)
        np.testing.assert_allclose(post.xi, xi_ref, atol=tol)
