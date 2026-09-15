"""Missing observations (NaN) — API contract, exact shortcut, sampling, forecasting.

Complements ``test_gaps_references.py`` (accuracy of the quadrature grid):

* complete data takes the unchanged code path — bit-identical to the
  implementation before missing-data support (all fixtures × 2 seeds);
* exact shortcut (HMC-IN, HMC-IN2, PMC-IN with state margins, d = 1 and d = 3):
  likelihood, γ, ξ and MPM against brute force over all state paths, exact to
  1e-10, including the log-space passes;
* NaN-aware entry points on every fixture (no NaN leaks out), the refusals
  (``precompute_weights`` and ``W`` for the grid variants, ICE/SEM), inf as
  missing, all-missing sequences, a multivariate grid model;
* ``forecast`` against a Monte-Carlo simulation of the model's transition,
  ``impute``/``sample_posterior`` draws against the posterior moments;
* the log-space fallback of the augmented chain.
"""
from __future__ import annotations

import itertools
import logging
import math
import re
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
from scipy import optimize, stats

from pmcprg.numerics import EPS, ONE_MINUS_EPS
from pmcprg.pmc import PMCModel, simulate
from pmcprg.pmc import gaps
from pmcprg.pmc import inference as inf

REPO = Path(__file__).resolve().parents[2]
MODELS = sorted((REPO / "pmcprg" / "pmc" / "models").glob("*.toml"))
SHORTCUT_FIXTURES = ["hmc_in_gauss_k2.toml", "hmc_in2_gauss_k2.toml", "hmc_in_gauss_k3.toml",
                     "pmc_in_gauss_k2.toml", "hmc_in_mvn_k2_d3.toml"]
GRID_FIXTURES = ["hmc_dn_gauss_k2.toml", "pmc_gauss_k2.toml", "pmc_pair_gauss_k2.toml",
                 "sp2016_gice_k2.toml"]
BASE_COMMIT = "fc58cde"   # last commit before missing-data support


def _model(name):
    return PMCModel(REPO / "pmcprg" / "pmc" / "models" / name)


def _with_gaps(Y, idx):
    Yn = np.array(Y, dtype=float, copy=True)
    Yn[idx] = np.nan
    return Yn


# ---------------------------------------------------------------------------
# Complete data: bit-identical to the implementation before this change
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pristine_inference():
    """``inference.py`` at BASE_COMMIT, loaded as a module (git needed).

    BASE_COMMIT predates the rename of the import package ``prg`` → ``pmcprg``:
    the file is read from its old path and its imports are rewritten.
    """
    old_pkg = "p" + "rg"          # spelled apart so the rename script leaves it alone
    try:
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:{old_pkg}/pmc/inference.py"],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git history with commit {BASE_COMMIT} unavailable")
    src = re.sub(rf"(?<![A-Za-z0-9_]){old_pkg}(?![A-Za-z0-9_])", "pmcprg", src)
    mod = types.ModuleType("inference_pristine")
    exec(compile(src, "inference_pristine.py", "exec"), mod.__dict__)
    return mod


def _same(a, b):
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    return len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b))


@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
@pytest.mark.parametrize("seed", [0, 7])
def test_complete_data_is_bit_identical(pristine_inference, path, seed):
    old, new = pristine_inference, inf
    m = PMCModel(path)
    _, Y = simulate(m, N=300, seed=seed)
    Wn, fn = new.precompute_weights(m, Y)
    Wo, fo = old.precompute_weights(m, Y)
    assert _same((Wn, fn), (Wo, fo))
    assert _same(new.forward(m, Y), old.forward(m, Y))
    assert _same(new.forward(m, Y, W=Wn, f_pdf=fn), old.forward(m, Y, W=Wo, f_pdf=fo))
    assert _same(new.backward(m, Y), old.backward(m, Y))
    assert _same(new.classify(m, Y), old.classify(m, Y))
    a, _ = new.forward(m, Y)
    b = new.backward(m, Y)
    assert _same(new.smooth(a, b), old.smooth(a, b))
    assert _same(new.joint_posteriors(a, Wn, b), old.joint_posteriors(a, Wo, b))
    assert _same(new.sample_posterior(m, Y, np.random.default_rng(3)),
                 old.sample_posterior(m, Y, np.random.default_rng(3)))
    assert _same(new._forward_log_space(m, Y), old._forward_log_space(m, Y))
    assert _same(new._backward_log_space(m, Y), old._backward_log_space(m, Y))
    assert _same(new._log_transition_weights(m, Y), old._log_transition_weights(m, Y))


# ---------------------------------------------------------------------------
# Exact shortcut: brute force over the state paths
# ---------------------------------------------------------------------------

def _shortcut_brute_force(m, Y):
    """log p(y_obs), γ, ξ: Σ over paths of (Σ_i p_{i x1}) Π A f^{obs}."""
    N, K = len(Y), m.K
    miss = gaps.missing_mask(Y)
    p = m.prior_p
    init = p.sum(axis=0)
    A = m.transition_A if m.variant.has_markov_prior else p / p.sum(axis=1, keepdims=True)
    logf = np.zeros((N, K))
    for k in range(K):
        for n in range(N):
            if not miss[n]:
                logf[n, k] = m.margin(k).logpdf(Y[n])
    paths = list(itertools.product(range(K), repeat=N))
    lw = np.array([math.log(init[x[0]]) + logf[0, x[0]]
                   + sum(math.log(A[x[n], x[n + 1]]) + logf[n + 1, x[n + 1]] for n in range(N - 1))
                   for x in paths])
    ll = float(np.logaddexp.reduce(lw))
    w = np.exp(lw - ll)
    gamma = np.zeros((N, K))
    xi = np.zeros((N - 1, K, K))
    for wk, x in zip(w, paths):
        gamma[np.arange(N), x] += wk
        xi[np.arange(N - 1), x[:-1], x[1:]] += wk
    return ll, gamma, xi


@pytest.mark.parametrize("name", SHORTCUT_FIXTURES)
@pytest.mark.parametrize("missing", [[1, 4], [0, 3, 6], [5, 6], [0, 1, 2], [1, 3, 5],
                                     [0, 1, 2, 3, 5, 6], list(range(7))],
                         ids=["two", "three-spread", "trailing", "leading", "every-other",
                              "all-but-one", "all"])
def test_shortcut_is_exact_against_brute_force(name, missing):
    m = _model(name)
    assert not gaps.needs_grid(m)
    _, Y = simulate(m, N=7, seed=11)
    Yn = _with_gaps(Y, missing)
    ll_ref, g_ref, xi_ref = _shortcut_brute_force(m, Yn)

    W, f = inf.precompute_weights(m, Yn)
    a, ll = inf.forward(m, Yn, W=W, f_pdf=f)
    b = inf.backward(m, Yn, W=W)
    g = inf.smooth(a, b)
    assert ll == pytest.approx(ll_ref, abs=1e-10)
    np.testing.assert_allclose(g, g_ref, atol=1e-10)
    np.testing.assert_allclose(inf.joint_posteriors(a, W, b), xi_ref, atol=1e-10)
    X_hat, g2, ll2 = inf.classify(m, Yn)
    assert np.array_equal(X_hat, np.argmax(g_ref, axis=1)) and ll2 == ll
    np.testing.assert_array_equal(g2, g)
    # the log-space passes accept the gaps too
    a_log, ll_log = inf._forward_log_space(m, Yn)
    assert ll_log == pytest.approx(ll_ref, abs=1e-10)
    np.testing.assert_allclose(inf.smooth(a_log, inf._backward_log_space(m, Yn)), g_ref, atol=1e-10)
    # marginalised tensors: density 1 at a missing row, A_ij into it
    np.testing.assert_array_equal(f[missing], 1.0)
    post = gaps.gap_posterior(m, Yn)
    assert post.method == "exact" and post.log_lik == ll


def test_shortcut_log_space_fallback_with_gaps():
    """An outlier whose density underflows next to a gap: exact log-space pass."""
    m = _model("hmc_in_gauss_k2.toml")
    Y = np.array([-0.9, np.nan, 40.0, 1.1, np.nan, np.nan, -1.2])
    ll_ref, g_ref, xi_ref = _shortcut_brute_force(m, Y)
    a, ll = inf.forward(m, Y)
    assert np.isfinite(ll) and ll == pytest.approx(ll_ref, abs=1e-9)
    np.testing.assert_allclose(inf.smooth(a, inf.backward(m, Y)), g_ref, atol=1e-10)


def test_shortcut_imputation_and_forecast_are_exact_mixtures():
    m = _model("hmc_in_gauss_k3.toml")
    _, Y = simulate(m, N=40, seed=2)
    Yn = _with_gaps(Y, [0, 7, 8, 9, 39])
    imp = gaps.impute(m, Yn, quantiles=(0.1, 0.5, 0.9))
    post = gaps.gap_posterior(m, Yn)
    mus = np.array([m.margin(k).params["loc"] for k in range(3)])
    sds = np.array([m.margin(k).params["scale"] for k in range(3)])
    w = post.gamma[imp.index]
    mean = w @ mus
    np.testing.assert_allclose(imp.mean, mean, atol=1e-12)
    np.testing.assert_allclose(imp.sd, np.sqrt(w @ (sds ** 2 + mus ** 2) - mean ** 2), atol=1e-12)
    for r in range(len(imp.index)):
        for c, lev in enumerate((0.1, 0.5, 0.9)):
            ref = optimize.brentq(lambda y: w[r] @ stats.norm.cdf(y, mus, sds) - lev, -20, 20, xtol=1e-13)
            assert imp.quantile_values[r, c] == pytest.approx(ref, abs=1e-9)
    np.testing.assert_allclose(imp.Y_mean[imp.index], imp.mean)

    fc = gaps.forecast(m, Yn, 4)
    a, _ = inf.forward(m, Yn)
    P = a[-1]
    for k in range(4):
        P = P @ m.transition_A
        np.testing.assert_allclose(fc.state_probs[k], P, atol=1e-12)
        assert fc.mean[k] == pytest.approx(P @ mus, abs=1e-12)
    assert fc.method == "exact"


def test_multivariate_shortcut_imputation_shapes_and_moments():
    m = _model("hmc_in_mvn_k2_d3.toml")
    _, Y = simulate(m, N=30, seed=1)
    Y[4, 1] = np.nan          # a partially observed row is missing as a whole
    Y[10:12] = np.nan
    assert gaps.missing_mask(Y).sum() == 3
    imp = gaps.impute(m, Y, n_samples=500, rng=0)
    assert imp.mean.shape == (3, 3) and imp.sd.shape == (3, 3)
    assert imp.quantile_values.shape == (3, 3, 3)
    assert imp.y_samples.shape == (500, 3, 3)
    mus = np.array([m.margin(k).params["mean"] for k in range(2)])
    np.testing.assert_allclose(imp.mean, imp.gamma @ mus, atol=1e-12)
    assert imp.nodes is None and imp.density is None
    fc = gaps.forecast(m, Y, 2)
    assert fc.mean.shape == (2, 3) and fc.quantile_values.shape == (2, 3, 3)


# ---------------------------------------------------------------------------
# NaN-aware entry points on every fixture; refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", MODELS, ids=lambda p: p.stem)
def test_nan_aware_entry_points_on_every_fixture(path):
    m = PMCModel(path)
    _, Y = simulate(m, N=120, seed=5)
    Yn = _with_gaps(Y, [0, 1, 17, 40, 41, 42, 43, 60, 62, 64, 119])
    a, ll = inf.forward(m, Yn)
    b = inf.backward(m, Yn)
    X_hat, g, ll2 = inf.classify(m, Yn)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.isfinite(ll)
    assert ll2 == ll
    np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(inf.smooth(a, b), g, atol=1e-12)
    np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-12)
    post = gaps.gap_posterior(m, Yn)
    np.testing.assert_allclose(post.gamma, g, atol=1e-12)
    # ξ is consistent with γ on both sides
    np.testing.assert_allclose(post.xi.sum(axis=2), g[:-1], atol=1e-10)
    np.testing.assert_allclose(post.xi.sum(axis=1), g[1:], atol=1e-10)
    X, Yc = inf.sample_posterior(m, Yn, np.random.default_rng(0), return_y=True)
    assert X.shape == (120,) and set(np.unique(X)) <= set(range(m.K))
    obs = ~gaps.missing_mask(Yn)
    np.testing.assert_array_equal(Yc[obs], Y[obs])
    assert np.all(np.isfinite(Yc))
    fc = gaps.forecast(m, Yn, 3)
    assert np.all(np.isfinite(fc.mean)) and fc.state_probs.shape == (3, m.K)
    assert (post.method == "grid") == gaps.needs_grid(m)


@pytest.mark.parametrize("name", GRID_FIXTURES)
def test_grid_variants_refuse_weight_tensors(name):
    m = _model(name)
    _, Y = simulate(m, N=20, seed=0)
    Yn = _with_gaps(Y, [5])
    with pytest.raises(ValueError, match="augmented state"):
        inf.precompute_weights(m, Yn)
    W, f = inf.precompute_weights(m, Y)
    with pytest.raises(ValueError, match="augmented state"):
        inf.forward(m, Yn, W=W, f_pdf=f)
    with pytest.raises(ValueError, match="augmented state"):
        inf.backward(m, Yn, W=W)
    with pytest.raises(ValueError, match="augmented state"):
        inf.sample_posterior(m, Yn, np.random.default_rng(0), W=W, f_pdf=f)


def test_infinite_values_are_missing():
    m = _model("pmc_pair_gauss_k2.toml")
    _, Y = simulate(m, N=50, seed=1)
    Ynan = _with_gaps(Y, [3, 20, 21])
    Yinf = Y.copy()
    Yinf[[3, 20, 21]] = [np.inf, -np.inf, np.nan]
    r1, r2 = inf.classify(m, Ynan), inf.classify(m, Yinf)
    assert _same(r1, r2)


@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "hmc_dn_gauss_k2.toml", "pmc_pair_gauss_k2.toml"])
def test_all_missing_sequence_gives_the_prior(name):
    """No observation: log-likelihood 0 and P(x_n = i) = Σ_j p_ij. Exact when X
    is Markov (state margins: the quadrature blocks carry the exact A_ij); with
    pair margins P(x_{n+1}) = Σ_i ∫ μ(i, y) A_ij(y) dy is itself a quadrature —
    measured drift 3.6e-6 after 12 steps at G = 64 (5.7e-6 at N = 100)."""
    m = _model(name)
    tol = 2e-5 if m.margin_structure == "pair" else 1e-12
    Y = np.full(12, np.nan)
    X_hat, g, ll = inf.classify(m, Y)
    assert ll == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(g, np.broadcast_to(m.prior_p.sum(axis=0), g.shape), atol=tol)
    fc = gaps.forecast(m, Y, 2)
    np.testing.assert_allclose(fc.state_probs, np.broadcast_to(m.prior_p.sum(axis=0), (2, m.K)),
                               atol=tol)


def test_trailing_gap_leaves_the_observed_posterior_unchanged():
    """β̃ = 1 in a trailing gap: likelihood and γ of the observed part are those
    of the truncated sequence (exactly, thanks to the block renormalisation)."""
    m = _model("pmc_pair_gauss_k2.toml")
    _, Y = simulate(m, N=60, seed=4)
    Y[10] = np.nan
    Yn = np.concatenate([Y, np.full(5, np.nan)])
    _, g_trunc, ll_trunc = inf.classify(m, Y)
    _, g, ll = inf.classify(m, Yn)
    assert ll == pytest.approx(ll_trunc, abs=1e-12)
    np.testing.assert_allclose(g[:60], g_trunc, atol=1e-12)


def test_grid_needs_scalar_observations():
    raw = {"model": {"variant": "PMC-IN", "K": 2, "d": 2},
           "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
           "margins": [{"i": i, "j": j, "dist": "multivariate_normal",
                        "params": {"mean": [float(i), float(j)], "cov": [[1.0, 0.0], [0.0, 1.0]]}}
                       for i in range(2) for j in range(2)]}
    try:
        m = PMCModel.from_dict(raw)
    except (ValueError, KeyError, TypeError) as exc:     # pragma: no cover — model refused upstream
        pytest.skip(f"pair-indexed multivariate model not constructible: {exc}")
    assert gaps.needs_grid(m)
    Y = np.zeros((6, 2))
    Y[2] = np.nan
    with pytest.raises(NotImplementedError, match="scalar observations"):
        inf.classify(m, Y)


def test_estimators_accept_missing_data():
    """ICE and SEM estimate from the observed-data likelihood (test_estim_missing.py)."""
    from pmcprg.pmc.ice import ice
    from pmcprg.pmc.sem import sem
    m = _model("hmc_dn_gauss_k2.toml")
    _, Y = simulate(m, N=50, seed=0)
    Yn = _with_gaps(Y, [7])
    for fit in (ice, sem):
        _, trace = fit(m, Yn, {"max_iter": 2})
        assert trace.log_liks[0] == pytest.approx(gaps.gap_posterior(m, Yn).log_lik, rel=1e-12)


def test_gap_nodes_and_quantiles_are_validated():
    m = _model("hmc_dn_gauss_k2.toml")
    Y = _with_gaps(np.zeros(5), [2])
    with pytest.raises(ValueError, match="gap_nodes"):
        inf.classify(m, Y, gap_nodes=1)
    with pytest.raises(ValueError, match="quantiles"):
        gaps.impute(m, Y, quantiles=(0.0, 0.5))
    with pytest.raises(ValueError, match="horizon"):
        gaps.forecast(m, Y, 0)
    imp = gaps.impute(m, np.zeros(5))
    assert imp.index.size == 0 and imp.mean.shape == (0,)
    pair = _model("pmc_pair_gauss_k2.toml")
    imp = gaps.impute(pair, np.linspace(-1, 2, 6), n_samples=3, rng=0)
    assert imp.index.size == 0 and imp.quantile_values.shape == (0, 3)
    assert imp.x_samples.shape == (3, 6) and imp.y_samples.shape == (3, 0)


# ---------------------------------------------------------------------------
# Log-space fallback of the augmented chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hmc_dn_gauss_k2.toml", "pmc_pair_gauss_k2.toml", "sp2016_gice_k2.toml"])
def test_log_space_chain_agrees_with_linear_chain(name):
    m = _model(name)
    _, Y = simulate(m, N=80, seed=9)
    Yn = _with_gaps(Y, [0, 10, 11, 12, 30, 32, 79])
    miss = gaps.missing_mask(Yn)
    grid = gaps.reference_grid(m)
    out = {}
    for log in (False, True):
        chain = gaps._Chain(m, Yn, miss, grid, log=log)
        alphas, ll = gaps._forward_chain(chain)
        betas = gaps._backward_chain(chain)
        post = gaps._chain_posterior(chain, alphas, betas, want_xi=True)
        out[log] = (ll, post)
    assert out[True][0] == pytest.approx(out[False][0], abs=1e-9)
    # β̂ itself is not compared: where α̂_n(i) = 0 (y_n outside the support of
    # f_i, as in the SP-2016 fixture) the linear weights of precompute_weights
    # give W[n, i, ·] = 0 while the log weights cancel f_i(y_n) symbolically —
    # a convention that leaves every posterior quantity unchanged.
    for key in ("gamma", "xi", "node_post", "alpha_hat"):
        np.testing.assert_allclose(out[True][1][key], out[False][1][key], atol=1e-9)


def test_log_space_fallback_on_underflow_next_to_a_gap(caplog):
    m = _model("hmc_dn_gauss_k2.toml")
    Y = np.array([-0.5, np.nan, np.nan, 45.0, 1.2, np.nan, 0.8])
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        X_hat, g, ll = inf.classify(m, Y)
    assert any("log space" in r.message for r in caplog.records)
    assert np.isfinite(ll) and np.all(np.isfinite(g))
    np.testing.assert_allclose(g.sum(axis=1), 1.0, atol=1e-12)
    imp = gaps.impute(m, Y)
    assert np.all(np.isfinite(imp.mean))


# ---------------------------------------------------------------------------
# Forecast against Monte-Carlo simulation of the model's transition
# ---------------------------------------------------------------------------

def _mc_forecast(m, Y, h, S, rng):
    """Draw x_N from the filter, then (x, y) forward with Eqs. 13–14."""
    K = m.K
    a, _ = inf.forward(m, Y)
    x = rng.choice(K, size=S, p=a[-1] / a[-1].sum())
    y = np.full(S, float(Y[-1]))
    p = m.prior_p
    xs, ys = [], []
    for _ in range(h):
        # x_{n+1} | x_n = i, y_n ∝ p_ij f_ij(y_n)   (A_ij for state margins)
        P = np.empty((S, K))
        for j in range(K):
            if m.margin_structure == "pair":
                P[:, j] = np.choose(x, [p[i, j] * m.margin(i, j).pdf_vec(y) for i in range(K)])
            else:
                P[:, j] = m.transition_A[x, j]
        P /= P.sum(axis=1, keepdims=True)
        xn = (rng.random(S)[:, None] > np.cumsum(P, axis=1)).sum(axis=1)
        yn = np.empty(S)
        for i in range(K):
            for j in range(K):
                sel = (x == i) & (xn == j)
                if sel.any():
                    u = np.clip(m.margin(i, j).cdf_vec(y[sel]), EPS, ONE_MINUS_EPS)
                    w = rng.uniform(EPS, ONE_MINUS_EPS, sel.sum())
                    v = m.copula(i, j).inv_h_array(w, u)
                    yn[sel] = m.margin(j, i)._frozen.ppf(v)
        x, y = xn, yn
        xs.append(x.copy())
        ys.append(y.copy())
    return np.array(xs), np.array(ys)


@pytest.mark.parametrize("name", ["pmc_pair_gauss_k2.toml", "hmc_dn_gauss_k2.toml"])
def test_forecast_against_monte_carlo(name):
    from pmcprg.pmc.simulate import _sample_copula_conditional
    m = _model(name)
    _, Y = simulate(m, N=80, seed=21)
    Y = _with_gaps(Y, [30, 31, 50])
    qs = (0.1, 0.5, 0.9)
    fc = gaps.forecast(m, Y, 5, quantiles=qs)
    S = 20_000
    rng = np.random.default_rng(12345)
    xs, ys = _mc_forecast(m, Y, 5, S, rng)
    # the vectorised draw is the simulator's conditional-inversion step
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    u = float(np.clip(m.margin(0, 1).cdf(0.3), EPS, ONE_MINUS_EPS))
    v = m.copula(0, 1).inv_h_array(np.array([r2.uniform(EPS, ONE_MINUS_EPS)]), np.array([u]))[0]
    assert _sample_copula_conditional(m, 0, 1, 0.3, r1) == pytest.approx(m.margin(1, 0).ppf(v), abs=1e-9)
    for k in (0, 1, 4):                                   # h = 1, 2, 5
        probs = np.bincount(xs[k], minlength=m.K) / S
        se_p = np.sqrt(fc.state_probs[k] * (1 - fc.state_probs[k]) / S)
        assert np.all(np.abs(probs - fc.state_probs[k]) < 4.5 * se_p + 1e-3)
        assert abs(ys[k].mean() - fc.mean[k]) < 4.5 * fc.sd[k] / math.sqrt(S)
        assert abs(ys[k].std() - fc.sd[k]) < 4.5 * fc.sd[k] / math.sqrt(2 * S)
        dens_at_q = np.interp(fc.quantile_values[k], fc.nodes, fc.density[k])
        emp_q = np.quantile(ys[k], qs)
        se_q = np.sqrt(np.array(qs) * (1 - np.array(qs)) / S) / dens_at_q
        assert np.all(np.abs(emp_q - fc.quantile_values[k]) < 4.5 * se_q)


# ---------------------------------------------------------------------------
# Posterior draws: impute(n_samples) and sample_posterior(return_y)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["pmc_pair_gauss_k2.toml", "hmc_in_gauss_k2.toml"])
def test_impute_draws_match_the_posterior_moments(name):
    m = _model(name)
    _, Y = simulate(m, N=60, seed=8)
    Yn = _with_gaps(Y, [0, 20, 21, 22, 40, 59])
    S = 6000
    imp = gaps.impute(m, Yn, n_samples=S, rng=7)
    assert imp.x_samples.shape == (S, 60) and imp.y_samples.shape == (S, 6)
    se = imp.sd / math.sqrt(S)
    np.testing.assert_array_less(np.abs(imp.y_samples.mean(axis=0) - imp.mean), 4.5 * se)
    np.testing.assert_array_less(np.abs(imp.y_samples.std(axis=0) - imp.sd),
                                 4.5 * imp.sd / math.sqrt(2 * S) * 1.5)
    freq = np.stack([(imp.x_samples[:, imp.index] == k).mean(axis=0) for k in range(m.K)], axis=1)
    se_p = np.sqrt(imp.gamma * (1 - imp.gamma) / S)
    np.testing.assert_array_less(np.abs(freq - imp.gamma), 4.5 * se_p + 1e-3)
    # at observed positions the state draws follow γ as well
    post = gaps.gap_posterior(m, Yn)
    f_obs = (imp.x_samples[:, 10] == 0).mean()
    assert abs(f_obs - post.gamma[10, 0]) < 4.5 * math.sqrt(post.gamma[10, 0] * (1 - post.gamma[10, 0]) / S) + 1e-3


def test_grid_draws_reproduce_the_conditional_covariance_of_a_block():
    """Joint draws, not marginal ones: AR(1) ρ = 0.9, a block of three."""
    rho = 0.9
    tau = 2.0 / math.pi * math.asin(rho)
    raw = {"model": {"variant": "HMC-DN", "K": 2}, "prior": {"A": [[0.9, 0.1], [0.2, 0.8]]},
           "margins": [{"i": i, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}} for i in range(2)],
           "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]}
    m = PMCModel.from_dict(raw)
    Y = np.array([0.3, -0.2, 0.5, 1.1, np.nan, np.nan, np.nan, 0.9, 0.4, -0.3])
    idx = np.arange(10)
    Sig = rho ** np.abs(idx[:, None] - idx[None, :])
    o = np.isfinite(Y)
    Kg = Sig[np.ix_(~o, o)] @ np.linalg.inv(Sig[np.ix_(o, o)])
    C = Sig[np.ix_(~o, ~o)] - Kg @ Sig[np.ix_(o, ~o)]
    S = 20_000
    imp = gaps.impute(m, Y, n_samples=S, rng=3)
    emp = np.cov(imp.y_samples.T)
    se = np.sqrt((np.outer(np.diag(C), np.diag(C)) + C ** 2) / S)
    np.testing.assert_array_less(np.abs(emp - C), 4.5 * se)


def test_sample_posterior_return_y():
    m = _model("hmc_in2_gauss_k2.toml")
    _, Y = simulate(m, N=50, seed=3)
    # complete data: same X as before, Y copied
    X0 = inf.sample_posterior(m, Y, np.random.default_rng(5))
    X1, Y1 = inf.sample_posterior(m, Y, np.random.default_rng(5), return_y=True)
    assert np.array_equal(X0, X1) and np.array_equal(Y1, Y) and Y1 is not Y
    # shortcut with gaps: missing y drawn from f_{X̃_n}
    Yn = _with_gaps(Y, [4, 5, 30])
    S = 3000
    rng = np.random.default_rng(0)
    draws = np.array([inf.sample_posterior(m, Yn, rng, return_y=True)[1][4] for _ in range(S)])
    imp = gaps.impute(m, Yn)
    assert abs(draws.mean() - imp.mean[0]) < 4.5 * imp.sd[0] / math.sqrt(S)
