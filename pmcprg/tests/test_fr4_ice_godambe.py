"""Godambe's (IFM) observed information for a copula τ fitted inside ICE,
alongside margins re-estimated by weighted MLE (AUDIT_COPULES FR-4, "Reste":
standard errors inside ICE) — :mod:`pmcprg.pmc._godambe`.

What is checked
----------------
* an unsupported family (Clayton, BB1, …) raises ``NotImplementedError`` —
  the pilot scope is Gauss only, K = 2, state-indexed Gaussian margins;
* a non-Gaussian margin, or a ``"pair"``-structured model, raises
  ``NotImplementedError`` too;
* missing rows are refused, as in :mod:`pmcprg.pmc._oakes`;
* the sensitivity matrix ``D`` is block lower-triangular by construction
  (the (η, θ) block, ``D_ηθ``, is exactly zero — the margin score does not
  involve τ) and its diagonal margin blocks match the closed-form Gaussian
  weighted-MLE curvature computed independently from ``gamma``;
* ``D``'s copula row (``D_θθ`` and the IFM correction ``D_θη``) matches an
  independently-coded finite difference that bypasses this module's own
  ``_margin_cdfs``/``_pair_pseudo_obs`` machinery (uses ``scipy.stats.norm``
  directly);
* Monte-Carlo coverage of the 95 %/90 % Wald intervals built from Godambe's
  SE, against (a) the naive :mod:`pmcprg.copulas._stderr` sandwich applied
  to the final ICE weights *and* margins as if both were fixed, and (b)
  :mod:`pmcprg.pmc._oakes` — which corrects for ξ̂'s own uncertainty but not
  for the margins' re-estimation — in two regimes: the package's standard
  K = 2 HMC-DN fixture (heavily overlapping state margins) and a
  well-separated-states variant (near-hard classification, i.e. the
  E-step's own latent-state contribution to the total uncertainty is
  small).

Measured with the seeds below, N = 600, pair (0, 0), τ = 0.6,
``fit_margins=True``, R = 300 replicates:

    Standard fixture (``hmc_dn_gauss_k2.toml``, μ = ∓1, σ = 1 — states
    overlap a great deal): SE ratio (RMS reported / empirical SD) 0.578
    (Godambe) vs 0.494 (Oakes) vs 0.403 (naive); 95 % coverage 0.737
    (Godambe) vs 0.680 (Oakes) vs 0.580 (naive); 90 % coverage 0.673
    (Godambe) vs 0.603 (Oakes) vs 0.520 (naive).

    Well-separated states (μ = ∓4, σ = 1, A close to the identity — the
    E-step classifies states almost exactly): SE ratio 0.860 (Godambe) vs
    0.482 (Oakes) vs 0.473 (naive); 95 % coverage 0.913 (Godambe) vs 0.627
    (Oakes) vs 0.637 (naive); 90 % coverage 0.827 (Godambe) vs 0.533 (both
    Oakes and naive).

Interpretation — a genuine, but partial, correction
------------------------------------------------------
Godambe's SE is closer to the empirical SD, and its coverage closer to
nominal, than *both* the naive sandwich and Oakes' identity, in *both*
regimes, confirming the module docstring's central claim: accounting for
the margin/copula score covariance (Joe's IFM correction, ``D_θη`` above)
recovers real uncertainty neither of the other two methods sees. In the
well-separated regime — where the E-step's own latent-state uncertainty is
small, isolating the margin-reestimation channel this module targets —
Godambe's coverage is close to nominal (0.91 at 95 %, 0.83 at 90 %),
confirming the sandwich is not merely "bigger", but the *right* correction
for the channel it targets.

In the standard, heavily-overlapping-states fixture Godambe still
under-covers materially (0.74 at 95 %, 0.67 at 90 %). This module's own
docstring flags why: it corrects the margin/copula *cross-covariance*
(Joe's IFM channel) but not the EM *latent-state* channel Oakes' identity
addresses for τ alone — Oakes' correction is not, in this pilot, extended
to the margin parameters themselves (an "Oakes for margins" term, i.e. how
γ̂_n(k) reacts to a margin perturbation, analogous to
:func:`pmcprg.pmc._oakes._mixed_info`, but for η instead of θ). When the
states are hard to tell apart, that missing channel is the dominant one,
and neither Godambe (this module) nor Oakes alone recovers the true
variance; Oakes and the naive sandwich are statistically indistinguishable
from each other there (both correct for none, or only part, of what
matters), while Godambe is measurably — if not fully — better. This is the
FR-4 "Reste" note's explicit next step, not a defect of the two-stage
partial corrections computed so far.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.stats import norm

from pmcprg.copulas import CopulaGaussian
from pmcprg.pmc._godambe import H_ETA, H_PSI, ice_godambe_tau_se
from pmcprg.pmc._oakes import ice_oakes_tau_se
from pmcprg.pmc.ice import ice
from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights, smooth
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

Z95 = float(norm.ppf(0.975))
Z90 = float(norm.ppf(0.95))

_GAUSS_TOML = "pmcprg/pmc/models/hmc_dn_gauss_k2.toml"


def _separated_model(tau_diag: float = 0.6) -> PMCModel:
    """K = 2 HMC-DN, Gaussian margins far apart (μ = ∓4, σ = 1) and a
    near-identity transition matrix — the E-step classifies states almost
    exactly, isolating the margin-reestimation channel Godambe targets from
    the EM latent-state channel Oakes targets (see the module docstring)."""
    raw = {
        "model": {"name": "hmc-dn-gauss-separated-k2", "variant": "HMC-DN", "K": 2, "N_default": 600},
        "prior": {"A": [[0.98, 0.02], [0.02, 0.98]]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -4.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 4.0, "scale": 1.0}},
        ],
        "copulas": [
            {"i": 0, "j": 0, "name": "Gauss", "tau": tau_diag},
            {"i": 0, "j": 1, "name": "Gauss", "tau": 0.0},
            {"i": 1, "j": 0, "name": "Gauss", "tau": 0.0},
            {"i": 1, "j": 1, "name": "Gauss", "tau": tau_diag},
        ],
    }
    return PMCModel.from_dict(raw)


def _fit_gauss(N=600, seed=0, max_iter=30, mdl=None):
    mdl = mdl if mdl is not None else PMCModel(_GAUSS_TOML)
    _, Y = simulate(mdl, N=N, seed=seed)
    fitted, _ = ice(mdl, Y, {"max_iter": max_iter, "tol": 1e-4,
                              "candidates": ["Gauss"], "fit_margins": True})
    return fitted, Y


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------

def test_unsupported_family_raises():
    """Clayton is not in this pilot's scope (Gauss only)."""
    mdl = PMCModel(_GAUSS_TOML)
    raw = mdl.raw
    for blk in raw["copulas"]:
        blk["name"], blk["tau"] = "Clayton", 0.3
    clayton_mdl = PMCModel.from_dict(raw)
    _, Y = simulate(clayton_mdl, N=200, seed=0)
    with pytest.raises(NotImplementedError):
        ice_godambe_tau_se(clayton_mdl, Y, pairs=[(0, 0)])


def test_non_gaussian_margin_raises():
    mdl = PMCModel(_GAUSS_TOML)
    raw = mdl.raw
    raw["margins"][0]["dist"] = "expon"
    raw["margins"][0]["params"] = {"loc": 0.0, "scale": 1.0}
    bad_mdl = PMCModel.from_dict(raw)
    _, Y = simulate(bad_mdl, N=200, seed=0)
    with pytest.raises(NotImplementedError):
        ice_godambe_tau_se(bad_mdl, Y, pairs=[(0, 0)])


def test_missing_rows_are_refused():
    fitted, Y = _fit_gauss(N=300, seed=1, max_iter=15)
    Y = Y.copy()
    Y[5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ice_godambe_tau_se(fitted, Y)


# ---------------------------------------------------------------------------
# Mechanics: D's block structure, and the margin blocks' closed form
# ---------------------------------------------------------------------------

def test_D_is_block_lower_triangular():
    """The margin score does not involve τ: D's (η, θ) block is exactly 0."""
    fitted, Y = _fit_gauss(N=600, seed=7, max_iter=30)
    res = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    dim = len(res.margin_names)
    assert res.D[:dim, dim] == pytest.approx(0.0, abs=1e-12)


def test_diagonal_pair_has_one_margin_block():
    """Pair (0, 0) uses only state 0's margin — 2 free parameters, not 4."""
    fitted, Y = _fit_gauss(N=600, seed=7, max_iter=30)
    res = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert res.margin_names == ("mu_0", "sigma_0")
    assert res.D.shape == (3, 3)


def test_off_diagonal_pair_has_two_margin_blocks():
    fitted, Y = _fit_gauss(N=600, seed=7, max_iter=30)
    res = ice_godambe_tau_se(fitted, Y, pairs=[(0, 1)])[(0, 1)]
    assert res.margin_names == ("mu_0", "sigma_0", "mu_1", "sigma_1")
    assert res.D.shape == (5, 5)


def test_result_is_well_formed():
    fitted, Y = _fit_gauss(N=600, seed=7, max_iter=30)
    res = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert not res.at_boundary
    assert res.se_tau > 0.0 and math.isfinite(res.se_tau)
    assert all(v > 0.0 and math.isfinite(v) for v in res.margin_se.values())
    assert np.allclose(res.cov, res.cov.T)
    assert res.n_eff > 0.0


# ---------------------------------------------------------------------------
# Cross-check: D independently recomputed, bypassing this module's own
# pseudo-observation / CDF helpers.
# ---------------------------------------------------------------------------

def test_D_matches_independent_recomputation():
    fitted, Y = _fit_gauss(N=600, seed=11, max_iter=30)
    res = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]

    W, f_pdf = precompute_weights(fitted, Y)
    alpha_hat, _ = forward(fitted, Y, W=W, f_pdf=f_pdf)
    beta_hat = backward(fitted, Y, W=W)
    gamma = smooth(alpha_hat, beta_hat)
    xi = joint_posteriors(alpha_hat, W, beta_hat)
    xi_hat = xi[:, 0, 0]

    blk = next(m for m in fitted.margin_blocks() if m["i"] == 0)
    mu0, sigma0 = float(blk["params"]["loc"]), float(blk["params"]["scale"])
    tau_hat = res.tau_hat

    # D_ηη, closed form (module docstring): diag(G_0/σ0², 2 G_0).
    G0 = float(gamma[:, 0].sum())
    assert res.D[0, 0] == pytest.approx(G0 / sigma0 ** 2, rel=1e-9)
    assert res.D[1, 1] == pytest.approx(2.0 * G0, rel=1e-9)

    # D_θθ, an independent finite difference (plain scipy.stats.norm, not
    # this module's _margin_cdfs / _pair_pseudo_obs).
    def uv_of(mu, sigma):
        u = np.clip(norm.cdf(Y[:-1], mu, sigma), 1e-10, 1.0 - 1e-10)
        v = np.clip(norm.cdf(Y[1:], mu, sigma), 1e-10, 1.0 - 1e-10)
        return np.column_stack((u, v))

    psi_hat = 2.0 * math.atanh(tau_hat)
    h = H_PSI

    def tau_of(psi):
        return math.tanh(psi / 2.0)

    uv0 = uv_of(mu0, sigma0)
    ld0 = CopulaGaussian(tau_k=tau_hat).logpdf_array(uv0)
    ldp = CopulaGaussian(tau_k=tau_of(psi_hat + h)).logpdf_array(uv0)
    ldm = CopulaGaussian(tau_k=tau_of(psi_hat - h)).logpdf_array(uv0)
    curvature = (ldp - 2.0 * ld0 + ldm) / (h * h)
    D_tt_direct = -float(np.dot(xi_hat, curvature))
    assert res.D[2, 2] == pytest.approx(D_tt_direct, rel=1e-6)

    # D_θη (the IFM correction itself), an independent finite difference.
    def S(mu, sigma):
        uv = uv_of(mu, sigma)
        ldp = CopulaGaussian(tau_k=tau_of(psi_hat + h)).logpdf_array(uv)
        ldm = CopulaGaussian(tau_k=tau_of(psi_hat - h)).logpdf_array(uv)
        phi = (ldp - ldm) / (2.0 * h)
        return float(np.dot(xi_hat, phi))

    h_eta = H_ETA
    D_theta_mu_direct = -(S(mu0 + h_eta, sigma0) - S(mu0 - h_eta, sigma0)) / (2.0 * h_eta)
    sigma_p = sigma0 * math.exp(h_eta)
    sigma_m = sigma0 * math.exp(-h_eta)
    D_theta_ls_direct = -(S(mu0, sigma_p) - S(mu0, sigma_m)) / (2.0 * h_eta)

    assert res.D[2, 0] == pytest.approx(D_theta_mu_direct, rel=1e-3)
    assert res.D[2, 1] == pytest.approx(D_theta_ls_direct, rel=1e-3)


# ---------------------------------------------------------------------------
# Monte-Carlo coverage (slow)
# ---------------------------------------------------------------------------

def _coverage_study(build_model, tau_true: float, *, R: int, N: int, seed0: int):
    est, se_god, se_naive, se_oakes = [], [], [], []
    c95g = c95n = c95o = c90g = c90n = c90o = 0
    for r in range(R):
        mdl = build_model()
        _, Y = simulate(mdl, N=N, seed=seed0 + r)
        fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                                   "candidates": ["Gauss"], "fit_margins": True})
        g = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
        if g.at_boundary or not math.isfinite(g.se_tau):
            continue
        o = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
        est.append(g.tau_hat)
        se_god.append(g.se_tau)
        se_naive.append(g.se_tau_naive)
        se_oakes.append(o.se_tau)
        c95g += abs(g.tau_hat - tau_true) <= Z95 * g.se_tau
        c95n += abs(g.tau_hat - tau_true) <= Z95 * g.se_tau_naive
        c95o += abs(g.tau_hat - tau_true) <= Z95 * o.se_tau
        c90g += abs(g.tau_hat - tau_true) <= Z90 * g.se_tau
        c90n += abs(g.tau_hat - tau_true) <= Z90 * g.se_tau_naive
        c90o += abs(g.tau_hat - tau_true) <= Z90 * o.se_tau
    n = len(est)
    est, se_god, se_naive, se_oakes = (np.asarray(x) for x in (est, se_god, se_naive, se_oakes))
    sd = np.std(est, ddof=1)
    return {
        "n": n,
        "ratio_god": math.sqrt(np.mean(se_god ** 2)) / sd,
        "ratio_naive": math.sqrt(np.mean(se_naive ** 2)) / sd,
        "ratio_oakes": math.sqrt(np.mean(se_oakes ** 2)) / sd,
        "cov95_god": c95g / n, "cov95_naive": c95n / n, "cov95_oakes": c95o / n,
        "cov90_god": c90g / n, "cov90_naive": c90n / n, "cov90_oakes": c90o / n,
    }


@pytest.mark.slow
def test_mc_coverage_overlapping_states():
    """R = 300, N = 600, τ = 0.6, pair (0, 0), ``fit_margins=True``, the
    package's standard K = 2 fixture (states overlap a great deal). See the
    module docstring for the measured numbers and their interpretation:
    Godambe beats both naive and Oakes but does not reach nominal coverage
    here — the EM latent-state channel this pilot does not correct for
    dominates when the states are hard to classify."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study(lambda: PMCModel(_GAUSS_TOML), 0.6, R=300, N=600, seed0=30_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 280
    assert r["ratio_god"] > r["ratio_oakes"] > r["ratio_naive"]
    assert r["cov95_god"] > r["cov95_oakes"] > r["cov95_naive"]
    assert r["cov90_god"] > r["cov90_oakes"] > r["cov90_naive"]
    assert 0.45 <= r["ratio_god"] <= 0.72, r
    assert 0.62 <= r["cov95_god"] <= 0.86, r
    assert 0.55 <= r["cov90_god"] <= 0.80, r


@pytest.mark.slow
def test_mc_coverage_separated_states():
    """R = 300, N = 600, τ = 0.6, pair (0, 0), ``fit_margins=True``, states
    far apart (μ = ∓4, near-identity transition matrix): the E-step's own
    latent-state uncertainty is small, isolating the margin-reestimation
    channel Godambe targets. Coverage should be close to nominal here — the
    module docstring's confirmation that the sandwich, not just its size,
    is right for the channel it targets."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study(_separated_model, 0.6, R=300, N=600, seed0=70_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 280
    assert r["ratio_god"] > r["ratio_oakes"]
    assert r["ratio_god"] > r["ratio_naive"]
    assert r["cov95_god"] > r["cov95_oakes"]
    assert r["cov95_god"] > r["cov95_naive"]
    assert 0.70 <= r["ratio_god"] <= 1.00, r
    assert 0.80 <= r["cov95_god"] <= 0.99, r
    assert 0.70 <= r["cov90_god"] <= 0.95, r
