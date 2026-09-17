"""Lystig & Hughes' (2002) exact observed information of an ICE fit
(AUDIT_COPULES FR-4, "Reste": standard errors inside ICE, third option) —
:mod:`pmcprg.pmc._lystig_hughes`.

What is checked
----------------
* the prior parametrisation (softmax over the distinct entries of the
  symmetric joint p) round-trips, and its analytic derivatives match finite
  differences; likewise the Gaussian margins' analytic ``∂ log f`` and the
  perturbed-CDF helper against :func:`pmcprg.pmc.ice._margin_cdfs`;
* the log-likelihood of the scaled recursion is :func:`forward`'s, to the
  last digits;
* the exact gradient matches a central difference of :func:`forward`'s
  log-likelihood, and the exact Hessian matches both a central difference of
  the exact gradient and a 4-point second difference of the log-likelihood,
  on **seven** models at parameters away from any fit (non-zero score, so
  every term of the recursion counts): HMC-DN Gauss, HMC-DN Clayton and
  state-margin PMC Gauss (the original one-parameter, margins-fixed pilot),
  HMC-DN BB1 and HMC-DN Student (two coordinates per pair), and HMC-DN
  Gauss and BB1 with ``fit_margins=True`` (the margin coordinates in θ);
* the **pre-existing path is bit-identical**: on two fixed fixtures the
  one-parameter, margins-fixed ``log_lik``/``grad``/``newton_step``/SEs
  equal, to the last bit, the constants the module produced at commit
  ``df4e662`` (before the extension);
* the diagonal ("partial") SE equals Oakes' SE (:mod:`pmcprg.pmc._oakes`) —
  Oakes' identity gives exactly −∂²ℓ/∂ψ∂ψᵀ — for one-parameter families
  *and* for the 2×2 blocks of BB1/Student against ``OakesResultMulti``; the
  joint SE is never smaller;
* scope guards (family, variant, pair margins, missing rows, non-Gaussian
  margins under ``fit_margins``) and the boundary convention (a pair at the
  edge of its range, or a BB1/Student whose ``_spec_of`` flags a boundary,
  is held fixed);
* Monte-Carlo coverage (slow): with margins known, the pilot's own settings
  and seeds; with ``fit_margins=True``, :mod:`pmcprg.pmc._godambe`'s two
  fixtures and seeds, so LH, Godambe, Oakes and the naive sandwich are
  compared on exactly the same replicates; and, for two-parameter families,
  :mod:`pmcprg.pmc._oakes`'s own BB1 and Student fixtures and seeds.

Exactness, measured
--------------------
Maximum relative error against finite differences (N = 600, the perturbed
models of :func:`_truth_and_perturbed`; gradient step 1e-5, Hessian steps
1e-4 against the exact gradient and 1e-3 for the 4-point difference of ℓ):

    model                       grad      Hess vs ∂g   Hess vs 4-pt ℓ
    HMC-DN Gauss  (P = 6)       6e-7      2e-7         2e-5
    HMC-DN BB1    (P = 10)      2.0e-7    9.5e-7       1.8e-6
    HMC-DN Student(P = 10)      1.6e-7    2.2e-5       6.7e-5
    HMC-DN Gauss, fit_margins   2.1e-8    5.1e-7       4.0e-5
      (P = 10)
    HMC-DN BB1,   fit_margins   8.7e-8    2.0e-6       2.2e-5
      (P = 14)

The residuals are the O(h²) error of the per-observation copula and
margin/copula difference stencils, not of the recursion, which is exact.

Cost (N = 600, K = 2, ms per call; ``ice_lh_information`` returns the whole
matrix, Oakes and Godambe one pair or one 2×2 block at a time)

    Gauss, margins fixed   (P = 6) : LH 17 · Oakes 4 pairs 47 · 1 pair 12
    Gauss, fit_margins    (P = 10) : LH 31 · Godambe 4 pairs 51 · 1 pair 10
    BB1,   margins fixed  (P = 10) : LH 20 · Oakes multi 4 pairs 52 · 1 pair 23
    BB1,   fit_margins    (P = 14) : LH 32
    Student, margins fixed(P = 10) : LH 31 · Oakes multi 1 pair 44

Monte-Carlo coverage, margins known (R = 300, pilot's seeds, pair (0, 0);
SE ratio = RMS reported SE / empirical SD of the ICE τ̂; joint (LH) /
partial (Oakes) / naive):

    Gauss,   τ = 0.6: ratio 0.999 / 0.950 / 0.772; 95 % coverage
             0.943 / 0.936 / 0.863; 90 % coverage 0.890 / 0.866 / 0.789.
    Clayton, τ = 0.5: ratio 1.028 / 1.009 / 0.913; 95 % coverage
             0.947 / 0.943 / 0.927; 90 % coverage 0.910 / 0.903 / 0.863.

Monte-Carlo coverage, two-parameter families (R = 150, N = 600, margins
known, Oakes' own BB1 and Student fixtures and seeds; n = 135 and 132 usable
after the boundary exclusions Oakes' own study makes). LH's *partial* SE
reproduces Oakes' matrix SE to three decimals, the cross-check this study
exists for; the joint SE adds the prior's and the other three pairs'
contribution:

    BB1, τ = 0.5, δ = 1.5 — τ: ratio 1.092 (LH joint) / 0.994 (LH partial =
        Oakes) / 0.774 (naive); 95 % coverage 0.933 / 0.911 / 0.859; 90 %
        0.919 / 0.904 / 0.815. δ: ratio 1.063 / 0.957 / 0.669; 95 %
        0.956 / 0.956 / 0.822; 90 % 0.926 / 0.896 / 0.748.
    Student, τ = 0.5, ν = 6 — τ: ratio 1.157 / 0.964 / 0.732; 95 % coverage
        0.955 / 0.939 / 0.826; 90 % 0.917 / 0.886 / 0.735. ν (``df``): 95 %
        coverage 0.924 / 0.917 / 0.894; 90 % 0.902 / 0.902 / 0.864. The SE
        ratio is not reported for ν, for the reason Oakes' own study gives
        (ν's sampling distribution is heavy right-tailed at N = 600, so a
        few replicates dominate the RMS: measured 4.99 here).

Monte-Carlo bands at R = 300: binomial SE 1.3 points at 95 %, 1.7 at 90 %;
the SE ratio has a relative SE of about 4 %. At R = 150 those are 1.8, 2.4
and 5.8 %.

Margins re-estimated: the exact joint information closes Godambe's gap
--------------------------------------------------------------------------
The open question :mod:`pmcprg.pmc._godambe` left is whether the residual
under-coverage it attributed to the **latent-state channel it does not
model** is really that. Run on Godambe's own two fixtures, its own seeds and
its own ``fit_margins=True`` ICE configuration (R = 300, N = 600, pair
(0, 0), τ = 0.6), with LH's margin coordinates in θ — the Godambe, Oakes and
naive columns below reproduce :mod:`test_fr4_ice_godambe`'s published
numbers to the third decimal, i.e. the comparison is on identical
replicates:

    Standard fixture (``hmc_dn_gauss_k2.toml``, μ = ∓1, σ = 1, states
    overlap a great deal; n = 291/300):
        ratio  1.128 LH · 0.579 Godambe · 0.494 Oakes · 0.403 naive
                     · 0.512 LH with the margins *held fixed*
        95 %   0.948 LH · 0.735 Godambe · 0.677 Oakes · 0.581 naive
        90 %   0.904 LH · 0.674 Godambe · 0.605 Oakes · 0.519 naive

    Well-separated states (μ = ∓4, near-identity A; n = 300/300):
        ratio  0.977 LH · 0.860 Godambe · 0.482 Oakes · 0.473 naive
        95 %   0.937 LH · 0.913 Godambe · 0.627 Oakes · 0.637 naive
        90 %   0.880 LH · 0.827 Godambe · 0.533 Oakes · 0.533 naive

**Yes — it closes.** On the hard, overlapping fixture the exact joint
information takes 95 % coverage from 0.735 (Godambe) to 0.948, within the
Monte-Carlo band of nominal (binomial SE 1.3 points at R = 300), and 90 %
from 0.674 to 0.904. On the easy fixture it takes 0.913 to 0.937. Godambe's
diagnosis was right: the missing channel was the latent-state one, and it is
*large* exactly where the states overlap — note that LH with the margins
held fixed scores 0.512/0.687 there, i.e. neither channel alone suffices;
only the joint matrix, which carries the margins, the latent states and the
copula together, reaches nominal.

The remaining 13 % over-statement of the SE on the overlapping fixture
(ratio 1.128, a conservative error) is the ICE/MLE gap this module
documents: ``I(θ̂)⁻¹`` is the MLE's variance, and ICE's margin M-step is not
a Q-maximiser (it is the γ-weighted Gaussian MLE, blind to the copula's
dependence on μ, σ). The diagnostic is measured in
:func:`test_margin_score_is_larger_than_the_copula_score`: at an ICE fit
with ``fit_margins=True`` the median one-Newton-step distance is 0.51 SE in
the margin coordinates against 0.21 SE in ψ_00 (0.40 vs 0.12 on the
separated fixture), and the median score is 4.6 against 1.7. With
``fit_margins=False`` — where ICE's copula M-step *is* an exact EM step —
the same diagnostic is 0.2 SE and the ratio is 0.999, as the pilot reported.
Chasing the last 13 % would mean modelling ICE's own fixed-point map, not
the likelihood's curvature; that is a different object, and out of FR-4's
scope.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.stats import norm

from pmcprg.pmc._godambe import ice_godambe_tau_se
from pmcprg.pmc._lystig_hughes import (
    H_ETA,
    _cdf_column,
    _eta_of_model,
    _log_f_derivatives,
    _prior_from_eta,
    ice_lh_information,
    lh_loglik_derivatives,
    model_from_theta,
)
from pmcprg.pmc._oakes import _tau_of_psi, ice_oakes_tau_se
from pmcprg.pmc.ice import _margin_cdfs, ice
from pmcprg.pmc.inference import forward
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

Z95 = float(norm.ppf(0.975))
Z90 = float(norm.ppf(0.95))

_GAUSS_TOML = "pmcprg/pmc/models/hmc_dn_gauss_k2.toml"


def _hmc_dn(family: str, blocks, A=((0.85, 0.15), (0.25, 0.75))) -> PMCModel:
    raw = {
        "model": {"name": f"hmc-dn-{family}-k2", "variant": "HMC-DN", "K": 2, "N_default": 600},
        "prior": {"A": [list(r) for r in A]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}},
        ],
        "copulas": blocks,
    }
    return PMCModel.from_dict(raw)


def _one_param(family: str, taus, A=((0.85, 0.15), (0.25, 0.75))) -> PMCModel:
    return _hmc_dn(family, [{"i": i, "j": j, "name": family, "tau": t}
                            for (i, j), t in zip(((0, 0), (0, 1), (1, 0), (1, 1)), taus)], A=A)


def _pmc_state(taus) -> PMCModel:
    raw = {
        "model": {"name": "pmc-state-gauss-k2", "variant": "PMC", "K": 2, "N_default": 600},
        "prior": {"p": [[0.40, 0.08], [0.08, 0.44]]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.5}},
        ],
        "copulas": [
            {"i": i, "j": j, "name": "Gauss", "tau": t}
            for (i, j), t in zip(((0, 0), (0, 1), (1, 0), (1, 1)), taus)
        ],
    }
    return PMCModel.from_dict(raw)


def _clayton_model(tau_diag: float = 0.5, tau_off: float = 0.15) -> PMCModel:
    """The Oakes pilot's Clayton fixture (:mod:`test_fr4_ice_oakes`)."""
    return _one_param("Clayton", (tau_diag, tau_off, tau_off, tau_diag),
                      A=((0.90, 0.10), (0.10, 0.90)))


def _bb1_model(tau_diag: float = 0.5, delta_diag: float = 1.5,
               tau_off: float = 0.15, delta_off: float = 1.05) -> PMCModel:
    """The Oakes two-parameter pilot's BB1 fixture
    (:mod:`test_fr4_ice_oakes_bb1_student`), reproduced so the coverage study
    below runs on identical replicates."""
    return _hmc_dn("BB1", [
        {"i": 0, "j": 0, "name": "BB1", "tau": tau_diag, "delta": delta_diag},
        {"i": 0, "j": 1, "name": "BB1", "tau": tau_off, "delta": delta_off},
        {"i": 1, "j": 0, "name": "BB1", "tau": tau_off, "delta": delta_off},
        {"i": 1, "j": 1, "name": "BB1", "tau": tau_diag, "delta": delta_diag},
    ], A=((0.90, 0.10), (0.10, 0.90)))


def _student_model(tau_diag: float = 0.5, df_diag: float = 6.0,
                   tau_off: float = 0.15, df_off: float = 6.0) -> PMCModel:
    """The Oakes two-parameter pilot's Student fixture."""
    return _hmc_dn("Student", [
        {"i": 0, "j": 0, "name": "Student", "tau": tau_diag, "df": df_diag},
        {"i": 0, "j": 1, "name": "Student", "tau": tau_off, "df": df_off},
        {"i": 1, "j": 0, "name": "Student", "tau": tau_off, "df": df_off},
        {"i": 1, "j": 1, "name": "Student", "tau": tau_diag, "df": df_diag},
    ], A=((0.90, 0.10), (0.10, 0.90)))


def _separated_model(tau_diag: float = 0.6) -> PMCModel:
    """:mod:`test_fr4_ice_godambe`'s well-separated-states fixture (μ = ∓4,
    near-identity A), reproduced verbatim so that the ``fit_margins`` coverage
    study below runs on identical replicates."""
    raw = {
        "model": {"name": "hmc-dn-gauss-separated-k2", "variant": "HMC-DN", "K": 2,
                  "N_default": 600},
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


#: ``case -> (build, simulation seed, fit_margins)``. The first three are the
#: original pilot's; the rest exercise the two extensions.
_CASES = {
    "hmc_dn_gauss": (lambda: _one_param("Gauss", (0.6, 0.2, -0.1, 0.5)), 5, False),
    "hmc_dn_clayton": (lambda: _one_param("Clayton", (0.5, 0.2, 0.1, 0.4)), 6, False),
    "pmc_state_gauss": (lambda: _pmc_state((0.5, -0.2, 0.1, 0.4)), 7, False),
    "hmc_dn_bb1": (lambda: _bb1_model(0.5, 1.5, 0.2, 1.05), 21, False),
    "hmc_dn_student": (lambda: _student_model(0.5, 6.0, 0.2, 7.0), 22, False),
    "hmc_dn_gauss_margins": (lambda: _one_param("Gauss", (0.6, 0.2, -0.1, 0.5)), 23, True),
    "hmc_dn_bb1_margins": (lambda: _bb1_model(0.5, 1.5, 0.2, 1.05), 24, True),
}

#: Shift applied to θ in :func:`_truth_and_perturbed`. Its first six entries
#: are the pilot's, unchanged, so the three original cases are unmoved.
_SHIFT = np.array([0.3, -0.2, 0.15, -0.1, 0.2, -0.15,
                   0.25, -0.05, 0.1, -0.12, 0.18, -0.22, 0.08, -0.16])


def _truth_and_perturbed(case: str):
    """The simulating model, a model at a θ shifted away from it (so the
    score is far from zero and every term of the recursion is exercised),
    the data, and the case's ``fit_margins``."""
    build, seed, fit_margins = _CASES[case]
    mdl = build()
    _, Y = simulate(mdl, N=600, seed=seed)
    theta = ice_lh_information(mdl, Y, fit_margins=fit_margins).theta
    shift = _SHIFT[: theta.size]
    return (model_from_theta(mdl, theta + shift, fit_margins=fit_margins), Y, fit_margins)


# ---------------------------------------------------------------------------
# Parametrisation
# ---------------------------------------------------------------------------

def test_prior_derivatives_match_finite_differences():
    eta = np.array([-0.7, 0.4])
    _, pi, dlogA, d2logA, dpi, d2pi = _prior_from_eta(eta, 2)
    h = 1e-6
    for t in range(2):
        e = np.eye(2)[t] * h
        pp, pip, *_ = _prior_from_eta(eta + e, 2)
        pm, pim, *_ = _prior_from_eta(eta - e, 2)
        logA_p = np.log(pp / pip[:, None])
        logA_m = np.log(pm / pim[:, None])
        np.testing.assert_allclose(dlogA[t], (logA_p - logA_m) / (2 * h), atol=1e-8)
        np.testing.assert_allclose(dpi[t], (pip - pim) / (2 * h), atol=1e-8)
        _, _, dlogA_p, _, dpi_p, _ = _prior_from_eta(eta + e, 2)
        _, _, dlogA_m, _, dpi_m, _ = _prior_from_eta(eta - e, 2)
        np.testing.assert_allclose(d2logA[:, t], (dlogA_p - dlogA_m) / (2 * h), atol=1e-7)
        np.testing.assert_allclose(d2pi[:, t], (dpi_p - dpi_m) / (2 * h), atol=1e-7)


def test_prior_is_symmetric_and_sums_to_one():
    p, pi, *_ = _prior_from_eta(np.array([0.3, -1.1]), 2)
    assert np.allclose(p, p.T)
    assert p.sum() == pytest.approx(1.0, abs=1e-15)
    assert np.allclose(pi, p.sum(axis=1))


def test_gaussian_margin_log_density_derivatives_match_finite_differences():
    """The analytic ``∂ log f``/``∂² log f`` in (μ, log σ) of the emission
    path — the first of the two channels a margin coordinate opens."""
    y = np.linspace(-3.0, 4.0, 11)
    mu, sigma = 0.4, 1.3
    d1, d2 = _log_f_derivatives(y, mu, sigma)

    def logf(m, ls):
        return norm.logpdf(y, loc=m, scale=math.exp(ls))

    ls = math.log(sigma)
    h = 1e-6                                   # first differences
    np.testing.assert_allclose(d1[0], (logf(mu + h, ls) - logf(mu - h, ls)) / (2 * h), atol=1e-7)
    np.testing.assert_allclose(d1[1], (logf(mu, ls + h) - logf(mu, ls - h)) / (2 * h), atol=1e-7)
    h = 1e-4                                   # second differences (rounding ∝ h⁻²)
    np.testing.assert_allclose(
        d2[0, 0], (logf(mu + h, ls) - 2 * logf(mu, ls) + logf(mu - h, ls)) / (h * h), atol=1e-6)
    np.testing.assert_allclose(
        d2[1, 1], (logf(mu, ls + h) - 2 * logf(mu, ls) + logf(mu, ls - h)) / (h * h), atol=1e-5)
    mixed = (logf(mu + h, ls + h) - logf(mu + h, ls - h)
             - logf(mu - h, ls + h) + logf(mu - h, ls - h)) / (4 * h * h)
    np.testing.assert_allclose(d2[0, 1], mixed, atol=1e-6)


def test_unperturbed_cdf_column_matches_margin_cdfs():
    """The perturbed-CDF helper reproduces the package's own clipped margin
    CDF at zero shift — the second channel starts from the same numbers ICE
    and Oakes use."""
    mdl = _one_param("Gauss", (0.6, 0.2, -0.1, 0.5))
    _, Y = simulate(mdl, N=200, seed=2)
    f_cdf = _margin_cdfs(mdl, Y)
    for k, blk in enumerate(mdl.margin_blocks()):
        col = _cdf_column(Y, float(blk["params"]["loc"]), float(blk["params"]["scale"]),
                          (0, 0), H_ETA)
        np.testing.assert_array_equal(col, f_cdf[:, k])


@pytest.mark.parametrize("case", sorted(_CASES))
def test_theta_round_trip(case):
    mdl, Y, fm = _truth_and_perturbed(case)
    info = ice_lh_information(mdl, Y, fit_margins=fm)
    back = model_from_theta(mdl, info.theta, fit_margins=fm)
    np.testing.assert_allclose(back.prior_p, mdl.prior_p, atol=1e-11)
    np.testing.assert_allclose(_eta_of_model(back), info.theta[:2], atol=1e-9)
    if fm:
        for a, b in zip(back.margin_blocks(), mdl.margin_blocks()):
            assert a["params"]["loc"] == pytest.approx(b["params"]["loc"], abs=1e-11)
            assert a["params"]["scale"] == pytest.approx(b["params"]["scale"], rel=1e-11)


def test_names_of_each_layout():
    """θ's layout: prior, then (with ``fit_margins``) the margins, then one
    or two coordinates per free pair."""
    mdl, Y, _ = _truth_and_perturbed("hmc_dn_gauss")
    assert ice_lh_information(mdl, Y).names == (
        "eta_01", "eta_11", "psi_00", "psi_01", "psi_10", "psi_11")
    assert ice_lh_information(mdl, Y, fit_margins=True).names == (
        "eta_01", "eta_11", "mu_0", "log_sigma_0", "mu_1", "log_sigma_1",
        "psi_00", "psi_01", "psi_10", "psi_11")
    mdl, Y, _ = _truth_and_perturbed("hmc_dn_bb1")
    assert ice_lh_information(mdl, Y).names == (
        "eta_01", "eta_11", "psi_00_0", "psi_00_1", "psi_01_0", "psi_01_1",
        "psi_10_0", "psi_10_1", "psi_11_0", "psi_11_1")


# ---------------------------------------------------------------------------
# The recursion: value, gradient, Hessian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(_CASES))
def test_loglik_matches_forward(case):
    mdl, Y, fm = _truth_and_perturbed(case)
    ll, *_ = lh_loglik_derivatives(mdl, Y, fit_margins=fm)
    assert ll == pytest.approx(forward(mdl, Y)[1], rel=1e-13)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_gradient_matches_finite_difference_of_loglik(case):
    mdl, Y, fm = _truth_and_perturbed(case)
    _, g, _, _ = lh_loglik_derivatives(mdl, Y, fit_margins=fm)
    theta = ice_lh_information(mdl, Y, fit_margins=fm).theta
    h = 1e-5

    def ll(x):
        return forward(model_from_theta(mdl, x, fit_margins=fm), Y)[1]

    g_fd = np.array([(ll(theta + h * e) - ll(theta - h * e)) / (2 * h)
                     for e in np.eye(theta.size)])
    assert np.max(np.abs(g)) > 1.0              # away from a stationary point
    rel = np.max(np.abs(g - g_fd) / np.maximum(1.0, np.abs(g)))
    assert rel < 1e-5, (g, g_fd)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_hessian_matches_finite_differences(case):
    mdl, Y, fm = _truth_and_perturbed(case)
    _, _, H, _ = lh_loglik_derivatives(mdl, Y, fit_margins=fm)
    theta = ice_lh_information(mdl, Y, fit_margins=fm).theta
    P = theta.size
    E = np.eye(P)

    def grad(x):
        return lh_loglik_derivatives(model_from_theta(mdl, x, fit_margins=fm),
                                     Y, fit_margins=fm)[1]

    h = 1e-4
    H_g = np.array([(grad(theta + h * e) - grad(theta - h * e)) / (2 * h) for e in E])
    assert np.max(np.abs(H - H_g) / np.maximum(1.0, np.abs(H))) < 1e-4

    def ll(x):
        return forward(model_from_theta(mdl, x, fit_margins=fm), Y)[1]

    h = 1e-3
    H_l = np.empty((P, P))
    for r in range(P):
        for s in range(r, P):
            H_l[r, s] = H_l[s, r] = (
                ll(theta + h * (E[r] + E[s])) - ll(theta + h * (E[r] - E[s]))
                - ll(theta - h * (E[r] - E[s])) + ll(theta - h * (E[r] + E[s]))
            ) / (4 * h * h)
    assert np.max(np.abs(H - H_l) / np.maximum(1.0, np.abs(H))) < 1e-3


# ---------------------------------------------------------------------------
# The pre-existing path is bit-identical
# ---------------------------------------------------------------------------

#: ``ice_lh_information`` at the ICE fit of ``_CASES[case]``'s model on
#: ``simulate(mdl, N=600, seed=5)``, as the module produced them at commit
#: ``df4e662`` — *before* the two-parameter / ``fit_margins`` extension. Both
#: code paths (one-parameter ψ stencil, scalar delta method) were kept
#: verbatim precisely so that these compare equal to the last bit.
_PILOT_GOLDEN = {
    "Gauss": dict(
        log_lik=-804.4405599384902,
        grad=[0.910334315160524, -0.8665567135133961, -0.4045615982344939,
              -0.23634175899118215, 0.013437557697517305, 0.5565760577224136],
        newton_step=[0.002551379148912642, -0.06283834719072595, -0.0025246199700809276,
                     -0.01724087761125474, 0.004214501616900599, 0.011932453261647167],
        se_tau={(0, 0): 0.021085512828429735, (0, 1): 0.109044305610301,
                (1, 0): 0.09621967161404389, (1, 1): 0.042682916794981526},
        se_tau_partial={(0, 0): 0.02039055608431679, (0, 1): 0.1064296303595652,
                        (1, 0): 0.09540720797364931, (1, 1): 0.04124170994065751},
    ),
    "Clayton": dict(
        log_lik=-841.6659842023837,
        grad=[0.635587634158395, -0.42246551291941076, 0.09845254713625626,
              -0.06359539681682813, -0.006893378685249089, 0.048698313717369726],
        newton_step=[0.012782460365284529, -0.011307544128622473, 0.0020481006020818104,
                     -0.021751280881567494, -0.009357311621071118, 0.008345712394804273],
        se_tau={(0, 0): 0.022299184698433644, (0, 1): 0.08507218681912901,
                (1, 0): 0.06458889123628396, (1, 1): 0.05622126731947303},
        se_tau_partial={(0, 0): 0.021656948289112077, (0, 1): 0.08409821268839802,
                        (1, 0): 0.0642678937320133, (1, 1): 0.054507101253361244},
    ),
}


@pytest.mark.parametrize("family, taus", [
    ("Gauss", (0.6, 0.2, -0.1, 0.5)),
    ("Clayton", (0.5, 0.2, 0.1, 0.4)),
])
def test_pilot_path_is_bit_identical(family, taus):
    """The one-parameter, margins-fixed path must be untouched by the
    extension: same inputs, same numbers, to the last bit (module docstring
    of :mod:`pmcprg.pmc._lystig_hughes`)."""
    mdl = _one_param(family, taus)
    _, Y = simulate(mdl, N=600, seed=5)
    fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                             "candidates": [family], "fit_margins": False})
    info = ice_lh_information(fitted, Y)
    want = _PILOT_GOLDEN[family]
    assert info.log_lik == want["log_lik"]
    np.testing.assert_array_equal(info.grad, np.array(want["grad"]))
    np.testing.assert_array_equal(info.newton_step, np.array(want["newton_step"]))
    assert info.se_tau == want["se_tau"]
    assert info.se_tau_partial == want["se_tau_partial"]


# ---------------------------------------------------------------------------
# At an ICE fit: SEs, Oakes' diagonal block, the score
# ---------------------------------------------------------------------------

def _fit(build, family, seed, N=600, fit_margins=False):
    mdl = build()
    _, Y = simulate(mdl, N=N, seed=seed)
    fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                             "candidates": [family], "fit_margins": fit_margins})
    return fitted, Y


@pytest.mark.parametrize("family, build", [
    ("Gauss", lambda: PMCModel(_GAUSS_TOML)),
    ("Clayton", _clayton_model),
])
def test_partial_se_is_oakes_and_joint_se_is_larger(family, build):
    fitted, Y = _fit(build, family, seed=11)
    info = ice_lh_information(fitted, Y)
    oakes = ice_oakes_tau_se(fitted, Y)
    assert np.all(np.linalg.eigvalsh(info.info) > 0.0)
    for pair, res in oakes.items():
        if pair in info.fixed_pairs:
            assert math.isnan(info.se_tau[pair])
            continue
        assert info.se_tau_partial[pair] == pytest.approx(res.se_tau, rel=1e-5)
        assert info.se_tau[pair] >= info.se_tau_partial[pair] * (1.0 - 1e-12)
        assert info.se_tau[pair] > res.se_tau_naive


@pytest.mark.parametrize("family, build, name2", [
    ("BB1", _bb1_model, "delta"),
    ("Student", _student_model, "df"),
])
def test_partial_block_matches_oakes_multi(family, build, name2):
    """For a two-parameter family, LH's own 2×2 block of the information is
    Oakes' ``info_psi``, so the SEs it gives for *every* reported quantity
    (τ and δ or ν) must match :class:`OakesResultMulti`'s — the cross-check
    that the ψ vector, the stencil and the Jacobian are the shared ones."""
    fitted, Y = _fit(build, family, seed=11)
    info = ice_lh_information(fitted, Y)
    res = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert info.pair_names[(0, 0)] == res.names
    assert (0, 0) not in info.fixed_pairs and not res.at_boundary
    for name in ("tau_k", name2):
        assert info.se_partial[(0, 0)][name] == pytest.approx(res.se[name], rel=1e-5)
        assert info.se[(0, 0)][name] >= info.se_partial[(0, 0)][name] * (1.0 - 1e-12)
        assert info.se[(0, 0)][name] > res.se_naive[name]
    assert info.se_tau[(0, 0)] == info.se[(0, 0)]["tau_k"]


def test_two_parameter_ci_uses_the_right_estimate():
    fitted, Y = _fit(_bb1_model, "BB1", seed=11)
    info = ice_lh_information(fitted, Y)
    lo, hi = info.ci((0, 0), 0.95, "delta")
    assert 0.5 * (lo + hi) == pytest.approx(info.estimate[(0, 0)]["delta"])
    assert hi - lo == pytest.approx(2 * Z95 * info.se[(0, 0)]["delta"])
    with pytest.raises(KeyError):
        info.ci((0, 0), 0.95, "df")


def test_score_at_ice_fit_is_small_against_the_se():
    """ICE is not the exact MLE (module docstring): the score is not zero,
    but one Newton step from the fit is a small fraction of the SE."""
    fitted, Y = _fit(lambda: PMCModel(_GAUSS_TOML), "Gauss", seed=7)
    info = ice_lh_information(fitted, Y)
    sd = np.sqrt(np.diag(info.cov))
    assert np.all(np.abs(info.newton_step) < 0.5 * sd), (info.newton_step, sd)


def test_ci_is_centred_on_tau_hat():
    fitted, Y = _fit(lambda: PMCModel(_GAUSS_TOML), "Gauss", seed=3)
    info = ice_lh_information(fitted, Y)
    lo, hi = info.ci((0, 0), 0.95)
    assert 0.5 * (lo + hi) == pytest.approx(info.tau_hat[(0, 0)])
    assert hi - lo == pytest.approx(2 * Z95 * info.se_tau[(0, 0)])


# ---------------------------------------------------------------------------
# Margins re-estimated
# ---------------------------------------------------------------------------

def test_fit_margins_widens_every_interval_and_reports_margin_ses():
    """Adding the margins to θ can only enlarge the copula block of the
    inverse information (Schur complement), and the margins get SEs of their
    own from the same matrix."""
    fitted, Y = _fit(lambda: PMCModel(_GAUSS_TOML), "Gauss", seed=7, fit_margins=True)
    fixed = ice_lh_information(fitted, Y)
    joint = ice_lh_information(fitted, Y, fit_margins=True)
    assert joint.fit_margins and not fixed.fit_margins
    assert joint.margin_names == ("mu_0", "sigma_0", "mu_1", "sigma_1")
    assert all(v > 0.0 and math.isfinite(v) for v in joint.margin_se.values())
    assert np.all(np.linalg.eigvalsh(joint.info) > 0.0)
    for pair in fixed.tau_hat:
        if pair in joint.fixed_pairs:
            continue
        assert joint.se_tau[pair] > fixed.se_tau[pair]
        # The pair's own block is unchanged by adding margins to θ.
        assert joint.se_tau_partial[pair] == pytest.approx(
            fixed.se_tau_partial[pair], rel=1e-9)


def test_fit_margins_beats_godambe_and_oakes_on_one_fit():
    """One replicate of the study in the module docstring: with the margins
    re-estimated, the exact joint SE exceeds Godambe's IFM sandwich, which
    exceeds Oakes' and the naive one."""
    fitted, Y = _fit(lambda: PMCModel(_GAUSS_TOML), "Gauss", seed=7, fit_margins=True)
    lh = ice_lh_information(fitted, Y, fit_margins=True).se_tau[(0, 0)]
    god = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    oak = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert lh > god.se_tau > oak.se_tau_naive
    assert lh > oak.se_tau


def test_margin_score_is_larger_than_the_copula_score():
    """The ICE/MLE diagnostic of the module docstring: ICE's margin M-step is
    the γ-weighted Gaussian MLE, blind to the copula's dependence on (μ, σ),
    so at the fit the one-Newton-step distance to the MLE is markedly larger
    in the margin coordinates than in ψ — which is *not* the case with
    ``fit_margins=False``, where the copula M-step is an exact EM step.

    Over the 300 replicates of :func:`test_mc_coverage_fit_margins_overlapping_states`
    the medians are 0.51 SE (margins) against 0.21 SE (ψ_00); this single
    replicate is one draw from that, so only the ordering and an order-of-
    magnitude bound are asserted."""
    fitted, Y = _fit(lambda: PMCModel(_GAUSS_TOML), "Gauss", seed=7, fit_margins=True)
    info = ice_lh_information(fitted, Y, fit_margins=True)
    idx = {nm: i for i, nm in enumerate(info.names)}
    m = [idx[n] for n in ("mu_0", "log_sigma_0", "mu_1", "log_sigma_1")]
    sd = np.sqrt(np.diag(info.cov))
    step_margin = np.max(np.abs(info.newton_step[m] / sd[m]))
    step_psi = abs(info.newton_step[idx["psi_00"]] / sd[idx["psi_00"]])
    assert step_margin > step_psi
    assert step_margin < 3.0        # of the order of an SE, not of the estimate
    # With the margins held fixed, ICE's remaining M-steps are much closer to
    # a stationary point of ℓ (the pilot's own diagnostic).
    fixed_fit, Yf = _fit(lambda: PMCModel(_GAUSS_TOML), "Gauss", seed=7)
    fixed = ice_lh_information(fixed_fit, Yf)
    assert np.max(np.abs(fixed.newton_step) / np.sqrt(np.diag(fixed.cov))) < 0.5


def test_fit_margins_requires_gaussian_margins():
    mdl = PMCModel(_GAUSS_TOML)
    raw = mdl.raw
    raw["margins"][0]["dist"] = "expon"
    raw["margins"][0]["params"] = {"loc": 0.0, "scale": 1.0}
    bad = PMCModel.from_dict(raw)
    _, Y = simulate(bad, N=200, seed=0)
    with pytest.raises(NotImplementedError, match="Gaussian"):
        ice_lh_information(bad, Y, fit_margins=True)
    # …while the margins-fixed path does not care about the margin family.
    assert math.isfinite(ice_lh_information(bad, Y).log_lik)


# ---------------------------------------------------------------------------
# Scope guards and boundary
# ---------------------------------------------------------------------------

def test_unsupported_family_raises():
    mdl = _one_param("Frank", (0.5, 0.2, 0.2, 0.5))
    _, Y = simulate(mdl, N=100, seed=0)
    with pytest.raises(NotImplementedError):
        ice_lh_information(mdl, Y)


def test_tawn_is_out_of_scope():
    """Tawn is a two-parameter family, but ``_spec_of`` has no working
    coordinate for it — out of scope, and refused rather than guessed at."""
    mdl = _hmc_dn("Tawn1", [{"i": i, "j": j, "name": "Tawn1", "tau": 0.3}
                            for (i, j) in ((0, 0), (0, 1), (1, 0), (1, 1))])
    _, Y = simulate(mdl, N=100, seed=0)
    with pytest.raises(NotImplementedError, match="Tawn"):
        ice_lh_information(mdl, Y)


def test_pair_margins_raise():
    mdl = PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml")
    _, Y = simulate(mdl, N=100, seed=0)
    with pytest.raises(NotImplementedError):
        ice_lh_information(mdl, Y)


def test_missing_rows_are_refused():
    """Gaps (:mod:`pmcprg.pmc.gaps`) are out of scope on both axes."""
    mdl = PMCModel(_GAUSS_TOML)
    _, Y = simulate(mdl, N=100, seed=0)
    Y = Y.copy()
    Y[5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ice_lh_information(mdl, Y)
    with pytest.raises(ValueError, match="missing"):
        ice_lh_information(mdl, Y, fit_margins=True)


def test_boundary_pair_is_held_fixed():
    mdl = _clayton_model(tau_off=1e-4)
    _, Y = simulate(mdl, N=300, seed=3)
    info = ice_lh_information(mdl, Y)
    assert set(info.fixed_pairs) == {(0, 1), (1, 0)}
    assert info.names == ("eta_01", "eta_11", "psi_00", "psi_11")
    assert math.isnan(info.se_tau[(0, 1)])
    assert math.isfinite(info.se_tau[(0, 0)])


def test_boundary_two_parameter_pair_is_held_fixed():
    """A BB1 at the Clayton limit δ = 1 is a ``_spec_of`` boundary: the pair
    keeps its copula in W (so the other coordinates stay right) but drops out
    of θ (Self & Liang 1987)."""
    mdl = _bb1_model(delta_off=1.0 + 1e-9)
    _, Y = simulate(mdl, N=300, seed=3)
    info = ice_lh_information(mdl, Y)
    assert set(info.fixed_pairs) == {(0, 1), (1, 0)}
    assert info.names == ("eta_01", "eta_11", "psi_00_0", "psi_00_1",
                          "psi_11_0", "psi_11_1")
    assert math.isnan(info.se_tau[(0, 1)])
    assert math.isfinite(info.se_tau[(0, 0)])


# ---------------------------------------------------------------------------
# Monte-Carlo coverage (slow)
# ---------------------------------------------------------------------------

def _coverage_study(build_model, tau_true: float, family: str, *, R: int, N: int,
                    seed0: int, pair=(0, 0)):
    """Coverage of the joint (LH), partial (Oakes) and naive SEs for ``pair``;
    also the one-Newton-step MLE (``tau_mle1``), for the ICE-vs-MLE remark."""
    est, est_mle1 = [], []
    se = {"lh": [], "oak": [], "naive": []}
    cov95 = dict.fromkeys(se, 0)
    cov90 = dict.fromkeys(se, 0)
    cov95_mle1 = 0
    for r in range(R):
        mdl = build_model()
        _, Y = simulate(mdl, N=N, seed=seed0 + r)
        fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                                 "candidates": [family], "fit_margins": False})
        o = ice_oakes_tau_se(fitted, Y, pairs=[pair])[pair]
        if o.at_boundary or not math.isfinite(o.se_tau):
            continue
        info = ice_lh_information(fitted, Y)
        if not math.isfinite(info.se_tau[pair]):
            continue
        tau_hat = o.tau_hat
        k = info.names.index(f"psi_{pair[0]}{pair[1]}")
        # One Newton step θ + I⁻¹g in ψ, mapped back to τ exactly.
        a, b = (-1.0, 1.0) if family == "Gauss" else (0.0, 1.0)
        tau_mle1 = _tau_of_psi(info.theta[k] + info.newton_step[k], a, b)
        est.append(tau_hat)
        est_mle1.append(tau_mle1)
        vals = {"lh": info.se_tau[pair], "oak": o.se_tau, "naive": o.se_tau_naive}
        for key, s in vals.items():
            se[key].append(s)
            cov95[key] += abs(tau_hat - tau_true) <= Z95 * s
            cov90[key] += abs(tau_hat - tau_true) <= Z90 * s
        cov95_mle1 += abs(tau_mle1 - tau_true) <= Z95 * info.se_tau[pair]
    n = len(est)
    est = np.asarray(est)
    sd = float(np.std(est, ddof=1))
    out = {"n": n, "sd": sd, "sd_mle1": float(np.std(est_mle1, ddof=1)),
           "bias": float(est.mean() - tau_true),
           "cov95_lh_mle1": cov95_mle1 / n}
    for key in se:
        arr = np.asarray(se[key])
        out[f"ratio_{key}"] = math.sqrt(float(np.mean(arr ** 2))) / sd
        out[f"cov95_{key}"] = cov95[key] / n
        out[f"cov90_{key}"] = cov90[key] / n
    return out


@pytest.mark.slow
def test_mc_coverage_gauss():
    """R = 300, N = 600, τ = 0.6, pair (0, 0), margins known — the Oakes
    pilot's setting and seeds. Bands as there (module docstring of
    :mod:`test_fr4_ice_oakes`); the joint SE is at least Oakes' by
    construction."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study(lambda: PMCModel(_GAUSS_TOML), 0.6, "Gauss",
                            R=300, N=600, seed0=30_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 280
    assert r["ratio_lh"] >= r["ratio_oak"] > r["ratio_naive"], r
    assert 0.85 <= r["ratio_lh"] <= 1.18, r
    assert 0.89 <= r["cov95_lh"] <= 0.99, r
    assert 0.80 <= r["cov90_lh"] <= 0.97, r


@pytest.mark.slow
def test_mc_coverage_clayton():
    """R = 300, N = 600, τ = 0.5, pair (0, 0), margins known — same bands."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study(_clayton_model, 0.5, "Clayton",
                            R=300, N=600, seed0=40_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 280
    assert r["ratio_lh"] >= r["ratio_oak"] > r["ratio_naive"], r
    assert 0.85 <= r["ratio_lh"] <= 1.18, r
    assert 0.89 <= r["cov95_lh"] <= 0.99, r
    assert 0.80 <= r["cov90_lh"] <= 0.97, r


def _coverage_study_margins(build_model, tau_true: float, *, R: int, N: int, seed0: int):
    """LH (margins in θ) vs Godambe vs Oakes vs naive vs LH-with-margins-fixed,
    on the same replicates — :mod:`test_fr4_ice_godambe`'s own study, with LH
    added (and its ``fit_margins=False`` sibling as the control)."""
    keys = ("lh", "god", "oak", "naive", "lhfix")
    est, se = [], {k: [] for k in keys}
    cov95 = dict.fromkeys(keys, 0)
    cov90 = dict.fromkeys(keys, 0)
    for r in range(R):
        mdl = build_model()
        _, Y = simulate(mdl, N=N, seed=seed0 + r)
        fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                                 "candidates": ["Gauss"], "fit_margins": True})
        g = ice_godambe_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
        if g.at_boundary or not math.isfinite(g.se_tau):
            continue
        o = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
        info = ice_lh_information(fitted, Y, fit_margins=True)
        infofix = ice_lh_information(fitted, Y)
        if not (math.isfinite(info.se_tau[(0, 0)]) and math.isfinite(infofix.se_tau[(0, 0)])):
            continue
        est.append(g.tau_hat)
        vals = {"lh": info.se_tau[(0, 0)], "god": g.se_tau, "oak": o.se_tau,
                "naive": g.se_tau_naive, "lhfix": infofix.se_tau[(0, 0)]}
        for k, s in vals.items():
            se[k].append(s)
            cov95[k] += abs(g.tau_hat - tau_true) <= Z95 * s
            cov90[k] += abs(g.tau_hat - tau_true) <= Z90 * s
    n = len(est)
    sd = float(np.std(np.asarray(est), ddof=1))
    out = {"n": n, "sd": sd}
    for k in keys:
        a = np.asarray(se[k])
        out[f"ratio_{k}"] = math.sqrt(float(np.mean(a ** 2))) / sd
        out[f"cov95_{k}"] = cov95[k] / n
        out[f"cov90_{k}"] = cov90[k] / n
    return out


@pytest.mark.slow
def test_mc_coverage_fit_margins_overlapping_states():
    """R = 300, N = 600, τ = 0.6, pair (0, 0), ``fit_margins=True`` —
    :mod:`test_fr4_ice_godambe`'s standard fixture and seeds, so the Godambe,
    Oakes and naive columns reproduce that module's published numbers and the
    comparison is on identical replicates.

    The FR-4 question this study answers: **the exact joint information does
    close Godambe's residual gap** (0.948 vs 0.735 at 95 %, measured; module
    docstring). Bands: binomial SE 1.3 points at 95 % and 1.7 at 90 % for
    R = 300, widened for the O(1/N) finite-sample gap and for the 13 %
    conservative bias the ICE/MLE difference leaves."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study_margins(lambda: PMCModel(_GAUSS_TOML), 0.6,
                                    R=300, N=600, seed0=30_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 270, r
    # The gap closes: LH beats Godambe, which beats Oakes, which beats naive.
    assert r["cov95_lh"] > r["cov95_god"] > r["cov95_oak"] > r["cov95_naive"], r
    assert r["ratio_lh"] > r["ratio_god"] > r["ratio_oak"] > r["ratio_naive"], r
    # …and reaches nominal, which none of the others does.
    assert 0.90 <= r["cov95_lh"] <= 0.99, r
    assert 0.84 <= r["cov90_lh"] <= 0.96, r
    assert 0.95 <= r["ratio_lh"] <= 1.35, r
    # The margins-fixed control stays with Oakes: neither channel alone does it.
    assert r["cov95_lhfix"] < 0.80, r


@pytest.mark.slow
def test_mc_coverage_fit_margins_separated_states():
    """R = 300, N = 600, τ = 0.6, pair (0, 0), ``fit_margins=True`` —
    :mod:`test_fr4_ice_godambe`'s well-separated-states fixture and seeds,
    where the latent-state channel is small and Godambe already reaches 0.913.
    LH improves on it (0.937 measured) without over-shooting."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study_margins(_separated_model, 0.6, R=300, N=600, seed0=70_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 270, r
    assert r["cov95_lh"] >= r["cov95_god"] > r["cov95_oak"], r
    assert r["ratio_lh"] > r["ratio_god"] > r["ratio_oak"], r
    assert 0.89 <= r["cov95_lh"] <= 0.99, r
    assert 0.82 <= r["cov90_lh"] <= 0.96, r
    assert 0.85 <= r["ratio_lh"] <= 1.20, r


def _coverage_study_two_param(build_model, truth: dict, family: str, name2: str, *,
                              R: int, N: int, seed0: int):
    """LH joint, LH partial, Oakes multi and naive, for both parameters of a
    two-parameter family — :mod:`test_fr4_ice_oakes_bb1_student`'s study with
    the two LH columns added."""
    keys = ("lh", "lhpart", "oak", "naive")
    rows = {q: {"est": [], **{k: [] for k in keys},
                **{f"c95_{k}": 0 for k in keys}, **{f"c90_{k}": 0 for k in keys}}
            for q in ("tau_k", name2)}
    n = 0
    for r in range(R):
        mdl = build_model()
        _, Y = simulate(mdl, N=N, seed=seed0 + r)
        try:
            fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                                     "candidates": [family], "fit_margins": False})
            o = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
            info = ice_lh_information(fitted, Y)
        except Exception:
            continue
        if o.at_boundary or (0, 0) in info.fixed_pairs:
            continue
        vals = {"oak": o.se, "naive": o.se_naive,
                "lh": info.se[(0, 0)], "lhpart": info.se_partial[(0, 0)]}
        if not all(math.isfinite(vals[k][q]) for k in keys for q in ("tau_k", name2)):
            continue
        n += 1
        for q in ("tau_k", name2):
            rows[q]["est"].append(o.estimate[q])
            err = abs(o.estimate[q] - truth[q])
            for k in keys:
                rows[q][k].append(vals[k][q])
                rows[q][f"c95_{k}"] += err <= Z95 * vals[k][q]
                rows[q][f"c90_{k}"] += err <= Z90 * vals[k][q]
    out = {"n": n}
    for q in ("tau_k", name2):
        sd = float(np.std(np.asarray(rows[q]["est"]), ddof=1))
        out[q] = {"sd": sd}
        for k in keys:
            a = np.asarray(rows[q][k])
            out[q][f"ratio_{k}"] = math.sqrt(float(np.mean(a ** 2))) / sd
            out[q][f"cov95_{k}"] = rows[q][f"c95_{k}"] / n
            out[q][f"cov90_{k}"] = rows[q][f"c90_{k}"] / n
    return out


@pytest.mark.slow
def test_mc_coverage_bb1_joint():
    """R = 150, N = 600, τ = 0.5, δ = 1.5, pair (0, 0), margins known —
    :mod:`test_fr4_ice_oakes_bb1_student`'s BB1 fixture and seeds. Bands are
    that module's (binomial SE 1.8 points at 95 %, 2.4 at 90 % for R = 150,
    widened for the finite-difference noise of the matrix terms). The point
    of this study is the *identity* LH partial ≡ Oakes, asserted tightly,
    and the joint SE's extra margin over it."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study_two_param(lambda: _bb1_model(0.5, 1.5),
                                      {"tau_k": 0.5, "delta": 1.5},
                                      "BB1", "delta", R=150, N=600, seed0=50_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 100, r
    for q in ("tau_k", "delta"):
        assert r[q]["ratio_lhpart"] == pytest.approx(r[q]["ratio_oak"], rel=5e-3), (q, r)
        assert r[q]["ratio_lh"] >= r[q]["ratio_lhpart"] > r[q]["ratio_naive"], (q, r)
        assert 0.85 <= r[q]["ratio_lh"] <= 1.35, (q, r)
        assert 0.86 <= r[q]["cov95_lh"] <= 1.00, (q, r)
        assert 0.80 <= r[q]["cov90_lh"] <= 1.00, (q, r)
        assert r[q]["cov95_naive"] <= r[q]["cov95_lh"] + 0.05, (q, r)


@pytest.mark.slow
def test_mc_coverage_student_joint():
    """R = 150, N = 600, τ = 0.5, ν = 6, pair (0, 0), margins known —
    :mod:`test_fr4_ice_oakes_bb1_student`'s Student fixture and seeds. As
    there, the SE *ratio* is not asserted for ``df``: ν's sampling
    distribution is heavy right-tailed at N = 600, so a handful of replicates
    with a very large reported SE dominate the RMS; coverage is the robust
    diagnostic and is what FR-4 is about."""
    logging.disable(logging.WARNING)
    try:
        r = _coverage_study_two_param(lambda: _student_model(0.5, 6.0),
                                      {"tau_k": 0.5, "df": 6.0},
                                      "Student", "df", R=150, N=600, seed0=60_000)
    finally:
        logging.disable(logging.NOTSET)
    assert r["n"] >= 80, r
    for q in ("tau_k", "df"):
        assert r[q]["ratio_lhpart"] == pytest.approx(r[q]["ratio_oak"], rel=5e-3), (q, r)
        assert r[q]["ratio_lh"] >= r[q]["ratio_lhpart"], (q, r)
        assert 0.80 <= r[q]["cov95_lh"] <= 1.00, (q, r)
        assert 0.68 <= r[q]["cov90_lh"] <= 1.00, (q, r)
        assert r[q]["cov95_naive"] <= r[q]["cov95_lh"] + 0.05, (q, r)
    assert 0.65 <= r["tau_k"]["ratio_lh"] <= 1.40, r
