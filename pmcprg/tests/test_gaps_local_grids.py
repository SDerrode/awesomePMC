"""Local quadrature grids around gaps (P1) and the quantile path (P3) — exact references.

P1: the reference grid of :mod:`pmcprg.pmc.gaps` puts its nodes in the
quantiles of the stationary law; under strong serial dependence the law of a
missing y given its neighbours is far narrower than the node spacing and the
quadrature fails silently (2-step forecast sd 0 instead of 0.02 at ρ =
0.9999). Local grids (module docstring, "Local grids") fix it. P3: the
quantiles of ``forecast`` / ``impute`` could disagree with the law whose mean
and sd are returned (median 15.8 for a law of mean 4.26, sd 1.42).

References, all independent of the quadrature grid:

* **Gaussian AR(1) as a PMC** (identical regimes, Gaussian copulas ρ): closed
  forms of the log-likelihood, conditional means, sds and quantiles for
  leading, isolated, block-of-five, every-other and trailing gaps; h-step
  forecasts; the PIT after a gap of length L, Φ((y − ρ^{L+1} y_a) /
  √(1 − ρ^{2(L+1)})).
* **Pair-margin PMC at high τ** (Gauss 0.99, Clayton 0.9 / 0.95, Gumbel 0.9 /
  0.95; margins of DerrodePieczynski_CSDA2013 Table 1): brute force over the
  state paths, the missing y integrated by composite Gauss–Legendre on
  0.001-wide panels over [−10, 12] (halving the panels changes log p by
  < 7e-14), with copula densities written from their formulas.
* **P3**: the ICE fit of ``report/forecasting`` (3-state pair margins),
  quantiles against their converged values (G = 512) and against the node
  law they come from; randomised 3-state pair-margin models.

Every tolerance is set from the error measured at the stated G, with the
measured value quoted; the pre-fix quadrature misses each of them by one to
eight orders of magnitude (quoted as "old").
"""
from __future__ import annotations

import itertools
import logging
import math

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtri

from pmcprg.pmc import PMCModel, flag_outliers, predictive_pit, simulate
from pmcprg.pmc import gaps

ICE = pytest.importorskip("pmcprg.pmc.ice")
QS = (0.05, 0.5, 0.95)
EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# Gaussian AR(1) at extreme ρ
# ---------------------------------------------------------------------------

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
    idx = np.arange(len(Y))
    S = rho ** np.abs(idx[:, None] - idx[None, :])
    o, m = ~miss, miss
    ll = stats.multivariate_normal(mean=np.zeros(o.sum()), cov=S[np.ix_(o, o)]).logpdf(Y[o])
    Kg = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])
    mu = Kg @ Y[o]
    sd = np.sqrt(np.diag(S[np.ix_(m, m)] - Kg @ S[np.ix_(o, m)]))
    return ll, mu, sd, mu[:, None] + sd[:, None] * stats.norm.ppf(QS)[None, :]


# Leading gap, isolated gap, block of five, every other position, trailing gap.
AR1_MISSING = [0, 1, 2, 5, 10, 11, 12, 13, 14, 20, 22, 24, 26, 28, 30, 37, 38, 39]


@pytest.mark.parametrize("kind", ["HMC-DN", "PMC-state", "PMC-pair"])
@pytest.mark.parametrize("rho", [0.998, 0.999, 0.9999])
def test_ar1_extreme_rho_every_gap_pattern(kind, rho):
    # Measured at G = 64, max over the three kinds and both seeds (errors of
    # the mean, sd and quantiles in units of the exact conditional sd):
    #              log-lik   mean     sd       quantiles   old (log-lik, sd)
    #   ρ = 0.998  5.0e-7    2.8e-8   4.4e-8   9.3e-8      0.24, 0.41
    #   ρ = 0.999  1.8e-7    5.7e-8   1.6e-7   1.4e-7      2.5,  0.89
    #   ρ = 0.9999 2.4e-7    2.8e-7   5.7e-7   8.8e-7      35,   1.8
    m = _ar1_model(kind, rho)
    for seed in (0, 1):
        Y = _ar1(40, rho, seed)
        miss = np.zeros(40, bool)
        miss[AR1_MISSING] = True
        Yn = np.where(miss, np.nan, Y)
        ll, mu, sd, q = _ar1_reference(Y, miss, rho)
        imp = gaps.impute(m, Yn, quantiles=QS)
        assert imp.log_lik == pytest.approx(ll, abs=5e-6)
        assert np.max(np.abs(imp.mean - mu) / sd) < 3e-6
        assert np.max(np.abs(imp.sd / sd - 1.0)) < 6e-6
        assert np.max(np.abs(imp.quantile_values - q) / sd[:, None]) < 1e-5
        post = gaps.gap_posterior(m, Yn, xi=False)
        assert post.quad_error < gaps.QUAD_WARN


@pytest.mark.parametrize("rho", [0.998, 0.999, 0.9999])
def test_ar1_extreme_rho_forecast(rho):
    """h-step forecast N(ρ^h y_N, 1 − ρ^{2h}), from an observed and from a missing last row."""
    # Measured at G = 64 (units of the exact sd): mean 1.1e-14, sd 1.8e-9,
    # quantiles 1.1e-9. Old: sd off by 3.8e-2, 0.36 and 1.0 (sd 0 instead
    # of 0.02 at ρ = 0.9999, h = 2).
    m = _ar1_model("PMC-state", rho)
    Y = _ar1(30, rho, 4)
    Y[7] = np.nan
    for Yc, off in ((Y, 0), (np.concatenate([Y, [np.nan]]), 1)):
        fc = gaps.forecast(m, Yc, 5, quantiles=QS)
        h = np.arange(1, 6) + off
        mean = rho ** h * Y[-1]
        sd = np.sqrt(1.0 - rho ** (2 * h))
        np.testing.assert_allclose((fc.mean - mean) / sd, 0.0, atol=1e-9)
        np.testing.assert_allclose(fc.sd / sd, 1.0, atol=2e-8)
        q = mean[:, None] + sd[:, None] * stats.norm.ppf(QS)
        np.testing.assert_allclose((fc.quantile_values - q) / sd[:, None], 0.0, atol=2e-8)


def test_pit_and_flags_after_gaps_at_rho_09999():
    """The PIT of a row after a gap of length L is Φ((y − ρ^{L+1} y_a) / √(1 − ρ^{2(L+1)}))."""
    # Measured (N = 400, 15 % gaps): max |PIT − exact| 2.6e-12, the filter's
    # log-likelihood equal to gap_posterior's (0.0 difference), 1 flag at
    # α = 1e-3 (the exact PITs give 1). Old: PIT off by 0.85, 119 flags.
    rho = 0.9999
    m = _ar1_model("PMC-state", rho)
    Y = _ar1(400, rho, 11)
    miss = np.random.default_rng(12).random(400) < 0.15
    miss[0] = False
    Yn = np.where(miss, np.nan, Y)
    last = np.maximum.accumulate(np.where(~miss, np.arange(400), 0))
    P = predictive_pit(m, Yn)
    rows = np.nonzero(~miss)[0][1:]
    prev = last[rows - 1]
    L = rows - prev
    exact = stats.norm.cdf(Y[rows], rho ** L * Y[prev], np.sqrt(1.0 - rho ** (2 * L)))
    assert np.max(np.abs(P.pit[rows] - exact)) < 1e-9
    assert np.any(L > 1)
    assert P.log_lik == pytest.approx(gaps.gap_posterior(m, Yn, xi=False).log_lik, abs=1e-9)
    pv = 2.0 * np.minimum(exact, 1.0 - exact)
    flags = flag_outliers(m, Yn, sequential=False)
    np.testing.assert_array_equal(np.nonzero(flags.flagged)[0], rows[pv < 1e-3])


def test_long_gap_and_jumps_at_rho_09999():
    """A 20-row gap and a 10 ε jump across a missing row, against the AR(1) closed forms."""
    # Measured at G = 64: gap of 20 rows 2.7e-5 (log-lik), 5.0e-5 (mean / sd,
    # sd units); jump of 10 innovations 2.6e-4 (log-lik), 2.4e-7 (mean).
    # Old: 9.9 and 7.7 nats.
    rho = 0.9999
    eps = math.sqrt(1.0 - rho * rho)
    m = _ar1_model("PMC-state", rho)
    Y = _ar1(40, rho, 3)
    miss = np.zeros(40, bool)
    miss[10:30] = True
    ll, mu, sd, _ = _ar1_reference(Y, miss, rho)
    post = gaps.gap_posterior(m, np.where(miss, np.nan, Y), xi=False)
    mass = post.node_post.sum(axis=1)
    mean = (mass * post.nodes).sum(axis=1)
    assert post.log_lik == pytest.approx(ll, abs=3e-4)
    assert np.max(np.abs(mean - mu) / sd) < 5e-4
    Y = _ar1(21, rho, 4)
    Y[11:] += 10.0 * eps
    miss = np.zeros(21, bool)
    miss[10] = True
    ll, mu, sd, _ = _ar1_reference(Y, miss, rho)
    post = gaps.gap_posterior(m, np.where(miss, np.nan, Y), xi=False)
    mass = post.node_post.sum(axis=1)
    assert post.log_lik == pytest.approx(ll, abs=3e-3)
    assert abs((mass * post.nodes).sum() - mu[0]) / sd[0] < 1e-5


# ---------------------------------------------------------------------------
# Pair-margin PMC at high τ: brute force
# ---------------------------------------------------------------------------

TABLE1 = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}
P_T1 = np.array([[0.50, 0.05], [0.05, 0.40]])


def _pair_model(cop, tau):
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": P_T1.tolist()},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                    for (i, j), (mu, sd) in sorted(TABLE1.items())],
        "copulas": [{"i": i, "j": j, "name": cop, "tau": tau} for i in range(2) for j in range(2)]})


def _log_copula(name, tau, u, v):
    """log c(u, v), from the formulas (Nelsen 2006)."""
    with np.errstate(all="ignore"):
        lu, lv = np.log(u), np.log(v)
        if name == "Gauss":
            r = math.sin(math.pi * tau / 2.0)
            x, y = ndtri(u), ndtri(v)
            return -0.5 * math.log1p(-r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))
        if name == "Clayton":
            th = 2.0 * tau / (1.0 - tau)
            a, b = -th * lu, -th * lv
            mx = np.maximum(a, b)
            ls = mx + np.log(np.exp(a - mx) + np.exp(b - mx) - np.exp(-mx))    # log(u^-θ + v^-θ − 1)
            return math.log1p(th) - (1 + th) * (lu + lv) - (2 + 1 / th) * ls
        th = 1.0 / (1.0 - tau)                                                 # Gumbel, (4.2.4)
        x, y = -lu, -lv
        s = x ** th + y ** th
        return (-s ** (1 / th) - lu - lv + (th - 1) * (np.log(x) + np.log(y))
                + (-2 + 1 / th) * np.log(s) + np.log(s ** (1 / th) + th - 1))


def _brute_force(name, tau, Y):
    """log p(y_obs), γ, E and sd of the missing y (isolated rows) — K = 2, pair margins."""
    def f(i, j, y):
        return stats.norm.pdf(y, *TABLE1[(i, j)])

    def F(i, j, y):
        return np.clip(stats.norm.cdf(y, *TABLE1[(i, j)]), EPS, 1.0 - EPS)

    def q(j, y1, i, y0):                         # DerrodePieczynski_CSDA2013 Eqs. 12–14
        den = sum(P_T1[i, k] * f(i, k, y0) for k in range(2))
        with np.errstate(all="ignore"):
            val = (P_T1[i, j] * f(i, j, y0) / den * f(j, i, y1)
                   * np.exp(_log_copula(name, tau, F(i, j, y0), F(j, i, y1))))
        return np.nan_to_num(val, nan=0.0, posinf=0.0)

    def mu(i, y):
        return sum(P_T1[i, j] * f(i, j, y) for j in range(2))

    def log(x):
        with np.errstate(divide="ignore"):
            return float(np.log(x))

    xg, wg = np.polynomial.legendre.leggauss(8)
    e = np.arange(-10.0, 12.0 + 1e-12, 1e-3)
    z = (0.5 * (e[1:, None] - e[:-1, None]) * xg + 0.5 * (e[:-1, None] + e[1:, None])).ravel()
    wz = (0.5 * (e[1:, None] - e[:-1, None]) * wg).ravel()
    N = len(Y)
    miss = ~np.isfinite(Y)
    integ = {}
    for n in np.nonzero(miss)[0]:
        for a, b, c in itertools.product(range(2), repeat=3):
            left = q(b, z, a, Y[n - 1]) if n > 0 else mu(b, z)
            right = q(c, Y[n + 1], b, z) if n < N - 1 else np.ones_like(z)
            g = left * right * wz
            integ[(n, a, b, c)] = (g.sum(), (g * z).sum(), (g * z * z).sum())
    logw, recs = [], []
    for xs in itertools.product(range(2), repeat=N):
        lp, mom = 0.0, []
        for n in range(N):
            if miss[n]:
                v = integ[(n, xs[n - 1] if n > 0 else 0, xs[n], xs[n + 1] if n < N - 1 else 0)]
                lp += log(v[0])
                mom.append((v[1] / v[0], v[2] / v[0]) if v[0] > 0 else (0.0, 0.0))
            elif n == 0:
                lp += log(mu(xs[0], Y[0]))
            elif not miss[n - 1]:
                lp += log(q(xs[n], Y[n], xs[n - 1], Y[n - 1]))
        logw.append(lp)
        recs.append((xs, mom))
    logw = np.array(logw)
    ll = float(np.logaddexp.reduce(logw))
    w = np.exp(logw - ll)
    gam = np.zeros((N, 2))
    m1 = np.zeros(miss.sum())
    m2 = np.zeros(miss.sum())
    for wk, (xs, mom) in zip(w, recs):
        for n in range(N):
            gam[n, xs[n]] += wk
        for k, (e1, e2) in enumerate(mom):
            m1[k] += wk * e1
            m2[k] += wk * e2
    return ll, gam, m1, np.sqrt(m2 - m1 ** 2)


PAIR_Y = {
    "central": [0.2, -0.4, np.nan, 0.1, 0.4],
    "lower": [-1.2, -1.5, np.nan, -1.4, -0.8],
    "upper": [2.5, 3.1, np.nan, 2.9, 1.0],
    "lead-trail": [np.nan, 0.3, 0.5, 0.45, np.nan],
}


@pytest.mark.parametrize("cop,tau,case,tol64,tol128", [
    # max(|Δ log p|, |Δγ|, |Δ mean|, |Δ sd|) measured at G = 64 / 128, new and (old);
    # with the filter-weighted proposals (test_gaps_filter_weights.py) three changed:
    # Gauss lower 4.6e-4 -> 9.6e-5, Clayton 0.95 lower 3.3e-4 -> 1.6e-3 (its log-lik;
    # gamma, mean and sd 4 times smaller), Gumbel 0.95 upper 5.4e-4 -> 6.2e-5 at G = 64
    ("Gauss", 0.99, "central", None, 2e-3),      # 3.8e-2 / 2.4e-4   (4.2 / 1.0)
    ("Gauss", 0.99, "lower", 3e-3, 5e-6),        # 4.6e-4 / 6.6e-7   (5.6 / 0.53)
    ("Gauss", 0.99, "lead-trail", 1e-2, 3e-6),   # 1.5e-3 / 4.2e-7   (2.3 / 0.52)
    ("Clayton", 0.9, "central", 2e-4, 2e-8),     # 2.4e-5 / 2.0e-9   (7.4e-3 / 6.4e-5)
    ("Clayton", 0.9, "lower", 3e-3, 5e-6),       # 4.0e-4 / 8.0e-7   (0.14 / 4.7e-3)
    ("Clayton", 0.95, "central", 2e-2, 3e-4),    # 3.4e-3 / 3.5e-5   (0.18 / 1.9e-2)
    ("Clayton", 0.95, "lower", 3e-3, 3e-6),      # 3.3e-4 / 3.0e-7   (0.51 / 0.15)
    ("GH", 0.9, "upper", 1e-6, 1e-10),           # 6.7e-8 / 1.6e-13  (5.7e-3 / 2.6e-5)
    ("GH", 0.95, "upper", 4e-3, 2e-5),           # 5.4e-4 / 2.3e-6   (0.14 / 1.7e-3)
    ("GH", 0.95, "lead-trail", 5e-3, 5e-6),      # 7.7e-4 / 5.1e-7   (2.4e-2 / 6.0e-4)
])
def test_pair_margins_high_tau_against_brute_force(cop, tau, case, tol64, tol128):
    m = _pair_model(cop, tau)
    Y = np.array(PAIR_Y[case], dtype=float)
    ll, g, mu, sd = _brute_force(cop, tau, Y)
    for G, tol in ((64, tol64), (128, tol128)):
        if tol is None:
            continue
        post = gaps.gap_posterior(m, Y, gap_nodes=G, xi=False)
        mass = post.node_post.sum(axis=1)
        mean = (mass * post.nodes).sum(axis=1)
        s = np.sqrt((mass * post.nodes ** 2).sum(axis=1) - mean ** 2)
        err = max(abs(post.log_lik - ll), np.abs(post.gamma - g).max(), np.abs(mean - mu).max(),
                  np.abs(s - sd).max())
        assert err < tol, (G, err)


# ---------------------------------------------------------------------------
# P3: quantiles consistent with the returned law
# ---------------------------------------------------------------------------

P3_MODEL = {
    "model": {"name": "pmc_pair K=3", "K": 3, "variant": "PMC", "margin_structure": "pair"},
    "prior": {"p": [[0.1684, 0.0393, 0.0124], [0.0393, 0.6546, 0.0368], [0.0124, 0.0368, 0.0]]},
    "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": lo, "scale": sc}}
                for (i, j), (lo, sc) in {
                    (0, 0): (2.994, 1.079), (0, 1): (3.665, 1.033), (0, 2): (3.046, 1.109),
                    (1, 0): (3.58, 1.231), (1, 1): (4.274, 0.836), (1, 2): (4.332, 0.869),
                    (2, 0): (9.629, 1.032), (2, 1): (10.841, 0.886), (2, 2): (5.125, 0.588)}.items()],
    "copulas": [{"i": i, "j": j, "name": nm, "tau": t} for (i, j), (nm, t) in {
        (0, 0): ("GH", 0.71), (0, 1): ("GH", 0.602), (0, 2): ("GH", 0.606),
        (1, 0): ("GH", 0.608), (1, 1): ("GH", 0.889), (1, 2): ("GH", 0.881),
        (2, 0): ("Frank", 0.667), (2, 1): ("GH", 0.88), (2, 2): ("Gauss", 0.3)}.items()],
}
P3_QS = (0.025, 0.5, 0.975)


def _node_law_distance(qv, mass, nodes, levels):
    """How far each level lies outside the CDF bracket of the node law at its quantile."""
    d = 0.0
    for r in range(qv.shape[0]):
        cum = np.cumsum(mass[r])
        for a, lev in enumerate(levels):
            k = int(np.searchsorted(nodes[r], qv[r, a]))
            lo = cum[k - 2] if k >= 2 else 0.0
            hi = cum[min(k, cum.size - 1)]
            d = max(d, lo - lev, lev - hi)
    return d


def test_p3_forecast_quantiles_match_the_law_and_converge():
    # Measured (h = 2 from y = 4): quantiles 3.040 / 4.003 / 10.468 at every
    # G ≥ 128; |q(G) − q(512)| = 4.3e-3, 2.7e-3, 4.3e-5, 4.3e-5 at G = 32,
    # 64, 128, 256; every quantile inside the CDF bracket of the node law.
    # Old (G = 64): 13.40 / 15.77 / 16.69 for a law of mean 4.26, sd 1.42.
    m = PMCModel.from_dict(P3_MODEL)
    ref = gaps.forecast(m, np.array([4.0]), 2, quantiles=P3_QS, gap_nodes=512).quantile_values
    for G, tol in ((32, 2e-2), (64, 1e-2), (128, 3e-4), (256, 3e-4)):
        fc = gaps.forecast(m, np.array([4.0]), 2, quantiles=P3_QS, gap_nodes=G)
        np.testing.assert_allclose(fc.quantile_values, ref, atol=tol)
        assert np.all(np.diff(fc.quantile_values, axis=1) > 0.0)
        assert _node_law_distance(fc.quantile_values, fc.grid_mass, fc.grid_nodes, P3_QS) <= 1e-12
        assert abs(fc.quantile_values[1, 1] - fc.mean[1]) < fc.sd[1]


def test_p3_impute_quantiles_match_the_law_and_converge():
    # Measured: |q(G) − q(512)| = 7.3e-3, 1.0e-3, 1.2e-5, 1.5e-5 at G = 32,
    # 64, 128, 256; the node-law bracket is met at G ≥ 64 (1.3e-3 at 32).
    m = PMCModel.from_dict(P3_MODEL)
    Y = np.array([4.0, np.nan, 4.2, np.nan, np.nan, 10.5, 4.1])
    ref = gaps.impute(m, Y, quantiles=P3_QS, gap_nodes=512).quantile_values
    for G, tol in ((64, 1e-2), (128, 2e-4), (256, 2e-4)):
        imp = gaps.impute(m, Y, quantiles=P3_QS, gap_nodes=G)
        np.testing.assert_allclose(imp.quantile_values, ref, atol=tol)
        assert np.all(np.diff(imp.quantile_values, axis=1) > 0.0)
        assert _node_law_distance(imp.quantile_values, imp.grid_mass, imp.grid_nodes, P3_QS) <= 1e-12


@pytest.mark.parametrize("seed", range(6))
def test_random_pair_margin_models_quantiles_are_consistent(seed):
    """Monotone in the level and inside the CDF bracket of the node law (measured: always, 0.0)."""
    fams = ["Gauss", "Clayton", "GH", "Frank"]
    rng = np.random.default_rng(seed)
    K = 3
    p = rng.dirichlet(np.full(K * K, 2.0)).reshape(K, K)
    p = 0.5 * (p + p.T)
    m = PMCModel.from_dict({
        "model": {"variant": "PMC", "K": K, "margin_structure": "pair"}, "prior": {"p": p.tolist()},
        "margins": [{"i": i, "j": j, "dist": "norm",
                     "params": {"loc": float(rng.uniform(0, 10)), "scale": float(rng.uniform(0.5, 1.5))}}
                    for i in range(K) for j in range(K)],
        "copulas": [{"i": i, "j": j, "name": fams[rng.integers(4)], "tau": float(rng.uniform(0.3, 0.9))}
                    for i in range(K) for j in range(K)]})
    _, Y = simulate(m, 40, seed=seed)
    Y = np.asarray(Y, dtype=float)
    Y[[3, 10, 11, 20, 21, 22, 39]] = np.nan
    for obj in (gaps.impute(m, Y, quantiles=P3_QS), gaps.forecast(m, Y[:-1], 3, quantiles=P3_QS)):
        assert np.all(np.diff(obj.quantile_values, axis=1) >= 0.0)
        assert _node_law_distance(obj.quantile_values, obj.grid_mass, obj.grid_nodes, P3_QS) <= 1e-12


def test_quantile_safety_net_on_a_spiky_law(caplog):
    """A Dirac-like node law makes the Legendre CDF oscillate: the monotone fallback is used."""
    m = _ar1_model("PMC-state", 0.5)
    g = gaps.reference_grid(m, 64)
    mass = np.full((1, 64), 1e-300)
    mass[0, 40] = 1.0
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        _, _, qv = gaps._grid_summary(mass, g.nodes[None], QS, mass, [g], [g])
    assert any("monotone CDF" in r.message for r in caplog.records)
    assert np.all(np.diff(qv[0]) >= 0.0)
    assert g.nodes[39] <= qv[0, 1] <= g.nodes[41]


# ---------------------------------------------------------------------------
# Convergence diagnostic
# ---------------------------------------------------------------------------

def test_diagnostic_is_quiet_when_converged_and_warns_otherwise(caplog, monkeypatch):
    # Measured quad_error, AR(1) ρ = 0.9999, the gap patterns above, G = 64:
    # 3.2e-7 with local grids, 30 on the reference grid alone (the pre-fix
    # quadrature, errors of 35 nats).
    m = _ar1_model("PMC-state", 0.9999)
    Y = _ar1(40, 0.9999, 0)
    miss = np.zeros(40, bool)
    miss[AR1_MISSING] = True
    Yn = np.where(miss, np.nan, Y)
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        post = gaps.gap_posterior(m, Yn)
    assert post.quad_error < 1e-5
    assert not any("not converged" in r.message for r in caplog.records)
    caplog.clear()
    monkeypatch.setattr(gaps, "_LOCAL_GRIDS", False)
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        post = gaps.gap_posterior(m, Yn)
    assert post.quad_error > gaps.QUAD_WARN
    assert any("not converged" in r.message for r in caplog.records)


def test_diagnostic_warns_on_a_gap_too_long_for_the_grid(caplog):
    # Measured, ρ = 0.9999, a 100-row gap: log-lik error 0.16 and
    # quad_error 2.1 at G = 64; 1.6e-4 and < QUAD_WARN at G = 128. Since the
    # moment-matched steps of long runs (gaps docstring, "Moment-matched
    # rows") G = 64 is 0.015 nats off (quad_error 0.015, below QUAD_WARN):
    # the case too coarse for the grid is now G = 48, 0.075 nats off
    # (quad_error 0.080); G = 128, 9.1e-5 (quad_error 8.8e-5).
    rho = 0.9999
    m = _ar1_model("PMC-state", rho)
    Y = _ar1(120, rho, 3)
    miss = np.zeros(120, bool)
    miss[10:110] = True
    Yn = np.where(miss, np.nan, Y)
    ll = _ar1_reference(Y, miss, rho)[0]
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        post = gaps.gap_posterior(m, Yn, gap_nodes=48, xi=False)
    assert post.quad_error > gaps.QUAD_WARN and abs(post.log_lik - ll) > 1e-2
    assert any("not converged" in r.message for r in caplog.records)
    post = gaps.gap_posterior(m, Yn, gap_nodes=128, xi=False)
    assert post.quad_error < gaps.QUAD_WARN and abs(post.log_lik - ll) < 1e-3


# ---------------------------------------------------------------------------
# Draws of the missing values sit on the nodes of their own position
# ---------------------------------------------------------------------------

def test_gap_e_step_draws_lie_on_the_local_nodes():
    m = _ar1_model("PMC-state", 0.999)
    Y = _ar1(60, 0.999, 2)
    Y[[5, 20, 21, 22, 40]] = np.nan
    miss = gaps.missing_mask(Y)
    est = ICE._gap_e_step(m, Y, miss, gap_nodes=None, posterior=True, n_draws=4,
                          rng=np.random.default_rng(0))
    post = gaps.gap_posterior(m, Y)
    idx = post.index
    for k, n in enumerate(idx):
        assert np.isin(est.Y_draws[:, n], post.nodes[k]).all()
    # local grids: the draws are near the observed neighbours, not on the
    # stationary quantiles (the reference nodes are 0.03–0.1 apart here)
    assert np.max(np.abs(est.Y_draws[:, 5] - 0.5 * (Y[4] + Y[6]))) < 0.2
