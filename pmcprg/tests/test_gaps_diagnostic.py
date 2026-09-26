"""The quadrature diagnostic and long runs of missing rows — exact references.

Two defects of the gap quadrature (``pmcprg.pmc.gaps``), found on the Intel
Lab study (``report/erroneous_data/intel_lab``, library problem 4):

* **The diagnostic.** ``GapPosterior.quad_error`` summed, over every missing
  position, the relative errors of the raw block masses (capped at 1): about
  1 per missing row of a long run, whatever the error. A trailing gap, which
  adds exactly 0 to log p(y_obs), added 1 982 for 2 000 rows on Intel mote 47;
  a 100-row gap of an AR(1) at ρ = 0.9999 reported 2.2 for 0.24 nats. It now
  sums, per run, the local errors of the steps into its positions — the row
  of each block against the exact one-step law, both on the local quartic of
  the backward message — and tracks the error of the run's log-likelihood
  factor (``gaps`` docstring, "Convergence diagnostic").
* **Long interior gaps.** Where the one-step law is narrower than the node
  spacing, the renormalised (Tauchen–Hussey) rows keep a node's mass on
  itself (no diffusion, no mean reversion) or move it to the nearest node of
  another grid, and which grid a position gets changed with G: a 1 000-row
  gap of an AR(1) at ρ = 0.9999 was −2.7 / −0.028 / −0.19 nats off at G = 64 /
  128 / 256. The steps of long runs are now tilted to the exact conditional
  mean and variance, and kernel test pieces of components no grid resolves
  no longer decide the grids (``gaps`` docstring, "Moment-matched rows").

References, independent of the quadrature:

* **Gaussian AR(1)** written as a PMC (two identical regimes, N(0, 1)
  margins, Gaussian copulas ρ): log p(y_obs) and the factor of every run of
  missing rows in closed form (the AR(1) marginalised over the gap).
* **State-specific dependence, independent switches** — the structure of the
  Intel mote-47 fit (its margins, prior and stay dependence τ = 0.996 /
  0.983 / 0.969 as Gaussian copulas on (i, i), independence on i ≠ j): given
  the state path, y_R depends on y_L only if the chain stayed in one state,
  so p(y_R, x_R = k | y_L, x_L = i) = [i = k] A_ii^d N(y_R; m_i + s_i r_i^d
  z_L, s_i² (1 − r_i^{2d})) + ((A^d)_ik − [i = k] A_ii^d) N(y_R; m_k, s_k²),
  and a K-state recursion over the observed rows is exact.

Every tolerance is set from the value measured at the stated G (quoted); the
code before this fix misses each test (quoted as "old").
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.integrate import trapezoid   # np.trapezoid needs NumPy >= 2
from scipy import stats

from pmcprg.pmc import PMCModel, StateMissingness, predictive_pit
from pmcprg.pmc import gaps


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------

def _ar1_model(rho):
    tau = 2.0 / math.pi * math.asin(rho)
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}} for i in range(2)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]})


def _ar1_path(N, rho, seed):
    rng = np.random.default_rng(seed)
    y = np.empty(N)
    y[0] = rng.standard_normal()
    e = rng.standard_normal(N)
    c = math.sqrt(1.0 - rho * rho)
    for n in range(1, N):
        y[n] = rho * y[n - 1] + c * e[n]
    return y


def _ar1_exact(Yn, rho):
    """log p(y_obs) and {first row of each run: exact log factor}."""
    obs = np.nonzero(np.isfinite(Yn))[0]
    ll = float(stats.norm.logpdf(Yn[obs[0]]))
    for p, n in zip(obs[:-1], obs[1:]):
        r = rho ** (n - p)
        ll += float(stats.norm.logpdf(Yn[n], r * Yn[p], math.sqrt(1.0 - r * r)))
    fac = {}
    a_, b_ = gaps._runs(~np.isfinite(Yn))
    for a, b in zip(a_, b_):
        a, b = int(a), int(b)
        if b == len(Yn) - 1:
            fac[a] = 0.0
        elif a == 0:
            fac[a] = float(stats.norm.logpdf(Yn[b + 1]))
        else:
            r = rho ** (b + 2 - a)
            fac[a] = float(stats.norm.logpdf(Yn[b + 1], r * Yn[a - 1], math.sqrt(1.0 - r * r)))
    return ll, fac


SW_M = [18.71, 21.61, 25.49]
SW_S = [1.95, 1.89, 2.40]
SW_P = np.array([[0.6305, 0.00065, 0.0], [0.00065, 0.2964, 0.00125], [0.0, 0.00125, 0.0693]])
SW_P = SW_P / SW_P.sum()
SW_R = [math.sin(math.pi * t / 2.0) for t in (0.9961, 0.9830, 0.9688)]


def _switch_model():
    cops = [{"i": i, "j": j, "name": "Gauss",
             "tau": (2.0 / math.pi * math.asin(SW_R[i]) if i == j else 0.0)}
            for i in range(3) for j in range(3)]
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 3}, "prior": {"p": SW_P.tolist()},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": SW_M[i], "scale": SW_S[i]}} for i in range(3)],
        "copulas": cops})


def _switch_path(N, seed):
    rng = np.random.default_rng(seed)
    pi = SW_P.sum(axis=1)
    A = SW_P / pi[:, None]
    x = np.empty(N, int)
    z = np.empty(N)
    x[0], z[0] = rng.choice(3, p=pi), rng.standard_normal()
    for n in range(1, N):
        x[n] = rng.choice(3, p=A[x[n - 1]])
        if x[n] == x[n - 1]:
            r = SW_R[x[n]]
            z[n] = r * z[n - 1] + math.sqrt(1.0 - r * r) * rng.standard_normal()
        else:
            z[n] = rng.standard_normal()
    return np.asarray(SW_M)[x] + np.asarray(SW_S)[x] * z


def _switch_exact(Yn):
    """log p(y_obs) of the switch model (module docstring)."""
    m, s, r = np.asarray(SW_M), np.asarray(SW_S), np.asarray(SW_R)
    pi = SW_P.sum(axis=1)
    A = SW_P / pi[:, None]
    obs = np.nonzero(np.isfinite(Yn))[0]
    al = pi * stats.norm.pdf(Yn[obs[0]], m, s)
    ll = math.log(al.sum())
    al = al / al.sum()
    for a, n in zip(obs[:-1], obs[1:]):
        d = int(n - a)
        Ad = np.linalg.matrix_power(A, d)
        stay = np.diag(A) ** d
        rd = r ** d
        own = stats.norm.pdf(Yn[n], m + s * rd * (Yn[a] - m) / s, s * np.sqrt(1.0 - rd * rd))
        E = (Ad - np.diag(stay)) * stats.norm.pdf(Yn[n], m, s)[None, :]
        E[np.arange(3), np.arange(3)] += stay * own
        v = al @ E
        ll += math.log(v.sum())
        al = v / v.sum()
    return ll


def _run_factor(chain, alphas, a, b):
    """log of the chain's mass from x_{a−1} (its own filter) to the observed row b + 1."""
    v = np.asarray(alphas[a - 1], float)
    ls = 0.0
    for n in range(a - 1, b + 1):
        T = chain.trans[n]
        if chain.log:
            r = np.log(np.maximum(v, 1e-300))[:, None] + T
            mx = np.max(r)
            v = np.exp(r - mx).sum(axis=0)
            ls += mx
        else:
            v = v @ T
        s = v.sum()
        ls += math.log(s)
        v = v / s
    return ls


# ---------------------------------------------------------------------------
# Defect 1: a trailing gap contributes nothing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [64, 128])
def test_trailing_gap_contributes_nothing(G):
    # A trailing gap adds exactly 0 to log p(y_obs): its rows sum to their
    # exact masses. Measured: its part of quad_error 1e-16 (old: 185 for 300
    # rows at G = 64, 16 at G = 128 — about 1 per row, the per-block cap).
    rho = 0.9999
    m = _ar1_model(rho)
    Y = _ar1_path(500, rho, 0)
    Y[40:45] = np.nan
    Y[200:] = np.nan
    post, (chain, alphas, betas) = gaps._posterior(m, Y, G, False)
    total, parts = gaps._quadrature_report(chain, alphas, betas, per_run=True)
    assert parts[200] < 1e-12
    assert total == pytest.approx(parts[40], rel=1e-9, abs=1e-15)
    # the log-likelihood is that of the series without the trailing rows
    ref = gaps.gap_posterior(m, Y[:200], gap_nodes=G, xi=False)
    assert post.log_lik == pytest.approx(ref.log_lik, abs=1e-9)


def test_trailing_gap_with_state_missingness_contributes_nothing():
    # With state-dependent missingness the backward message of a trailing
    # gap differs across states but not across the nodes of a state (state
    # margins): the part stays 0. Measured 1e-17 (old: 45 for 60 rows).
    rho = 0.999
    tau = 2.0 / math.pi * math.asin(rho)
    m = PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 1, "dist": "norm", "params": {"loc": 1.5, "scale": 0.8}}],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]}
    ).with_missingness(StateMissingness(rates=[0.05, 0.4]))
    Y = _ar1_path(200, rho, 3)
    Y[140:] = np.nan
    post, (chain, alphas, betas) = gaps._posterior(m, Y, 64, False)
    _, parts = gaps._quadrature_report(chain, alphas, betas, per_run=True)
    assert parts[140] < 1e-12


# ---------------------------------------------------------------------------
# Defect 1: the diagnostic tracks the error of each run's factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("L,G,seed", [
    # measured |error of the run's factor| / its part of quad_error
    (20, 32, 0),     # 5.5e-02 / 5.8e-02 (old quad_error 0.18)
    (100, 48, 3),    # 7.4e-02 / 7.9e-02
    (100, 64, 3),    # 1.2e-02 / 1.3e-02  (old: 0.24 nats off, quad_error 2.2)
    (100, 64, 0),    # 1.6e-02 / 1.5e-02
    (100, 128, 2),   # 1.2e-04 / 1.2e-04
    (500, 64, 0),    # 1.1e-01 / 1.2e-01
])
def test_diagnostic_tracks_the_error_of_the_run(L, G, seed):
    rho = 0.9999
    m = _ar1_model(rho)
    Y = _ar1_path(L + 40, rho, seed)
    Y[20:20 + L] = np.nan
    ll, fac = _ar1_exact(Y, rho)
    post, (chain, alphas, betas) = gaps._posterior(m, Y, G, False)
    _, parts = gaps._quadrature_report(chain, alphas, betas, per_run=True)
    err = abs(_run_factor(chain, alphas, 20, 19 + L) - fac[20])
    assert err > 1e-5
    assert err / 4.0 < parts[20] < 4.0 * err, (err, parts[20])


def test_diagnostic_is_quiet_on_a_small_error():
    # Moderate dependence (ρ = 0.9), a 60-row gap at G = 64: the run is
    # 1.6e-8 nats off, its part 1.9e-8 (kernels the grid resolves).
    rho = 0.9
    m = _ar1_model(rho)
    Y = _ar1_path(300, rho, 1)
    Y[100:160] = np.nan
    ll, fac = _ar1_exact(Y, rho)
    post, (chain, alphas, betas) = gaps._posterior(m, Y, 64, False)
    _, parts = gaps._quadrature_report(chain, alphas, betas, per_run=True)
    err = abs(_run_factor(chain, alphas, 100, 159) - fac[100])
    assert err < 1e-7 and parts[100] < 1e-6, (err, parts[100])


def test_diagnostic_catches_a_bridge_far_in_the_kernel_tail():
    # A single missing row across a jump of 32 kernel sds (pair margins,
    # Gaussian copulas τ = 0.99, the brute-force case "central" of
    # test_gaps_local_grids.py): the block of the entry kernel is rescaled
    # by target / raw mass, and that factor is the error. Measured at G = 64:
    # 3.8e-2 nats off, quad_error 3.3e-2; G = 128: 2.4e-4 and 1.9e-4.
    from test_gaps_local_grids import PAIR_Y, _brute_force, _pair_model
    m = _pair_model("Gauss", 0.99)
    Y = np.array(PAIR_Y["central"], dtype=float)
    ll = _brute_force("Gauss", 0.99, Y)[0]
    for G in (64, 128):
        post = gaps.gap_posterior(m, Y, gap_nodes=G, xi=False)
        err = abs(post.log_lik - ll)
        assert err / 2.0 < post.quad_error < 2.0 * err, (G, err, post.quad_error)


# ---------------------------------------------------------------------------
# Defect 2: long interior gaps converge in G
# ---------------------------------------------------------------------------

def test_long_ar1_gap_converges_in_G():
    # AR(1) at ρ = 0.9999, a 500-row interior gap. Measured |log-lik error|
    # at G = 64 / 128 / 256: 0.11 / 1.4e-2 / 1.1e-5 (old: 0.35 / 0.88 / 0.099,
    # not monotone).
    rho = 0.9999
    m = _ar1_model(rho)
    Y = _ar1_path(900, rho, 7)
    Y[200:700] = np.nan
    ll, _ = _ar1_exact(Y, rho)
    err = [abs(gaps.gap_posterior(m, Y, gap_nodes=G, xi=False).log_lik - ll) for G in (64, 128, 256)]
    assert err[0] > err[1] > err[2]
    assert err[0] < 0.3 and err[1] < 0.05 and err[2] < 1e-3, err


def test_switch_model_long_gap_converges_in_G():
    # The structure of Intel mote 47 (module docstring), a 300-row interior
    # gap. Measured |log-lik error| and quad_error at G = 64 / 128: 3.1e-2
    # and 3.7e-2 / 6.1e-3 and 6.0e-3 (3.5e-3 and 3.4e-3 at G = 256). Old: the
    # error grew from G = 128 to 256 on 2 000-row gaps (0.058 → 1.71 nats:
    # the reference grid kept at the end of the gap, chosen by a kernel test
    # piece at the centre of a broad law), and quad_error was ~1 per row.
    m = _switch_model()
    Y = _switch_path(700, 4)
    Y[200:500] = np.nan
    ll = _switch_exact(Y)
    posts = [gaps.gap_posterior(m, Y, gap_nodes=G, xi=False) for G in (64, 128)]
    err = [abs(p.log_lik - ll) for p in posts]
    assert err[1] < err[0] and err[1] < 1e-2, err
    for e, p in zip(err, posts):
        assert e / 3.0 < p.quad_error < 3.0 * e, (e, p.quad_error)


def test_batch_and_filter_agree_on_a_long_gap():
    # The PIT filter takes the same moment-matched transitions: its
    # log-likelihood equals gap_posterior's (measured: to the last bit).
    rho = 0.9999
    m = _ar1_model(rho)
    Y = _ar1_path(300, rho, 2)
    Y[50:250] = np.nan
    post = gaps.gap_posterior(m, Y, gap_nodes=64, xi=False)
    P = predictive_pit(m, Y, gap_nodes=64)
    assert P.log_lik == pytest.approx(post.log_lik, abs=1e-9)


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

def test_short_runs_are_not_tilted():
    # Runs shorter than _TILT_RUN keep their renormalised rows bit for bit.
    rho = 0.9999
    m = _ar1_model(rho)
    Y = _ar1_path(200, rho, 5)
    for a, L in ((10, 1), (30, 5), (60, 20), (120, gaps._TILT_RUN - 1)):
        Y[a:a + L] = np.nan
    out = []
    for tilt in (True, False):
        old = gaps._TILT
        try:
            gaps._TILT = tilt
            gaps._GRID_CACHE.clear()
            out.append(gaps.gap_posterior(m, Y, gap_nodes=64, xi=True))
        finally:
            gaps._TILT = old
    for f in ("log_lik", "alpha_hat", "beta_hat", "gamma", "xi", "node_post"):
        assert np.array_equal(getattr(out[0], f), getattr(out[1], f)), f


def test_moment_tilt_matches_mean_and_variance():
    # A law far narrower than the node spacing (sd 0.1, spacing 1), its mean
    # 0.005 from a node: the tilted row has the exact mean and variance on
    # the node and its two neighbours; where the variance is out of reach
    # (mean 0.4 from a node: the least variance with that mean is 0.4 × 0.6
    # = 0.24 > 0.01), the mean alone.
    y = np.arange(-10.0, 11.0)[None, :]
    for m0, flag in ((0.005, 2), (0.4, 1)):
        m, sd = np.array([m0]), np.array([0.1])
        LB = stats.norm.logpdf(y, m0, 0.1)
        out, fl = gaps._moment_tilt(LB, y, m, sd)
        p = np.exp(out[0])
        assert fl[0] == flag
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert (p * y[0]).sum() == pytest.approx(m0, abs=1e-9)
        if flag == 2:
            assert (p * (y[0] - m0) ** 2).sum() == pytest.approx(0.01, rel=1e-8)
            assert np.count_nonzero(p > 1e-6) == 3


@pytest.mark.parametrize("cop,tau", [("Gauss", 0.99), ("GH", 0.98), ("Clayton", 0.95), ("Frank", 0.9)])
def test_kernel_moments_match_direct_integration(cop, tau):
    # Mean, variance, third and fourth central moments of y' | (x = 0, y,
    # x' = 1) with pair margins, from the Gauss–Hermite table against the
    # density integrated on a fine grid. Measured (24 nodes): mean 5.7e-7 sd,
    # variance 1.5e-6, third moment 6.7e-6 sd³, fourth 8.8e-6 (Clayton 0.95,
    # the worst; the Gaussian copula to 1e-13).
    m = PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.4, 0.1], [0.1, 0.4]]},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": 0.3 * i - 0.2 * j, "scale": 1.0 + 0.2 * j}}
                    for i in range(2) for j in range(2)],
        "copulas": [{"i": i, "j": j, "name": cop, "tau": tau} for i in range(2) for j in range(2)]})
    mom = gaps._KernelMoments(m)
    yy = np.linspace(-15.0, 15.0, 600001)
    fj = gaps._frozen(m.margin(1, 0))
    Fv = np.clip(fj.cdf(yy), gaps.EPS, gaps.ONE_MINUS_EPS)
    for y0 in (-2.0, 0.1, 1.7):
        u = float(np.clip(gaps._frozen(m.margin(0, 1)).cdf(y0), gaps.EPS, gaps.ONE_MINUS_EPS))
        got = mom.at([0], [1], [u])[0]
        dens = fj.pdf(yy) * m.copula(0, 1).pdf_array(np.column_stack([np.full(yy.size, u), Fv]))
        Z = trapezoid(dens, yy)
        mu = trapezoid(dens * yy, yy) / Z
        c = [trapezoid(dens * (yy - mu) ** k, yy) / Z for k in (2, 3, 4)]
        sd = math.sqrt(c[0])
        assert abs(got[0] - mu) < 2e-6 * sd
        assert got[1] == pytest.approx(c[0], rel=5e-6)
        assert abs(got[2] - c[1]) < 3e-5 * sd ** 3
        assert got[3] == pytest.approx(c[2], rel=5e-5)


def test_warning_names_the_estimated_error(caplog):
    # The message keeps the form the studies parse ("relative error X > Y …
    # (N missing rows, gap_nodes = G)").
    rho = 0.9999
    m = _ar1_model(rho)
    Y = _ar1_path(140, rho, 3)
    Y[20:120] = np.nan
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        gaps.gap_posterior(m, Y, gap_nodes=32, xi=False)
    import re
    msg = [r.message for r in caplog.records if "not converged" in r.message]
    pat = re.compile(r"relative error ([0-9.eE+-]+) > ([0-9.eE+-]+) .*\((\d+) missing rows, gap_nodes = (\d+)\)")
    got = pat.search(msg[0]) if msg else None
    assert got is not None and got.group(3) == "100" and got.group(4) == "32"


def test_local_derivs_survive_an_exactly_singular_system():
    """Distinct nodes far closer to m than the farthest of the five nearest
    make the scaled powers underflow and the system exactly singular (the
    Intel Lab detection run of 2026-09-26 stopped on it, inside a narrow
    local grid). That row goes to least squares; the other rows keep the
    numbers they have without it."""
    y = np.array([[-1.0, 1e-200, 2e-200, 3e-200, 4e-200, 1.0],
                  [-2.0, -1.0, 0.1, 1.0, 2.0, 3.0]])
    b = np.array([[0.3, 0.5, 0.6, 0.7, 0.8, 0.2],
                  [0.1, 0.4, 0.5, 0.45, 0.3, 0.1]])
    m = np.zeros(2)
    x = y[0, :5]
    V = (x / np.max(np.abs(x)))[:, None] ** np.arange(5)[None, :]
    with pytest.raises(np.linalg.LinAlgError):   # premise: exactly singular
        np.linalg.solve(V, np.ones(5))
    h, _ = gaps._local_derivs(y, b, m)
    assert np.all(np.isfinite(h))
    alone, _ = gaps._local_derivs(y[1:], b[1:], m[1:])
    assert np.array_equal(h[1], alone[0])
