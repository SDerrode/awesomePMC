"""Lystig & Hughes' (2002) exact observed information of an ICE fit
(AUDIT_COPULES FR-4, "Reste": standard errors inside ICE, third option) —
:mod:`pmcprg.pmc._lystig_hughes`.

What is checked
----------------
* the prior parametrisation (softmax over the distinct entries of the
  symmetric joint p) round-trips, and its analytic derivatives match finite
  differences;
* the log-likelihood of the scaled recursion is :func:`forward`'s, to the
  last digits;
* the exact gradient matches a central difference of :func:`forward`'s
  log-likelihood, and the exact Hessian matches both a central difference of
  the exact gradient and a 4-point second difference of the log-likelihood —
  HMC-DN Gauss, HMC-DN Clayton and state-margin PMC Gauss, at parameters
  away from any fit (non-zero score, so every term of the recursion counts);
* the diagonal ("partial") SE equals Oakes' SE (:mod:`pmcprg.pmc._oakes`) —
  Oakes' identity gives exactly −∂²ℓ/∂ψ² — and the joint SE is never
  smaller;
* scope guards (family, variant, pair margins, missing rows) and the
  boundary convention (a pair at the edge of its range is held fixed);
* Monte-Carlo coverage (slow), same setting and seeds as the Oakes pilot
  (K = 2 HMC-DN, N = 600, pair (0, 0), margins known), comparing the full
  (joint) SE, Oakes' partial SE and the naive sandwich.

Measured (``lh_loglik_derivatives``, N = 600, first model below): maximum
relative gradient error 6e-7 against the central difference of ``forward``
(h = 1e-5; the residual is the O(h²) error of the per-observation copula
derivative, not of the recursion); Hessian 2e-7 against the difference of
the exact gradient, 2e-5 against the 4-point second difference of ℓ
(h = 1e-3). Partial SE vs Oakes: relative difference 1e-9. Cost at K = 2,
P = 6, N = 600: 18 ms for the whole 6×6 matrix, vs 47 ms for Oakes' four
one-pair computations.

Monte-Carlo coverage, R = 300 (299 usable for Gauss, one boundary τ̂),
seeds as the Oakes pilot, pair (0, 0); SE ratio = RMS reported SE /
empirical SD of the ICE τ̂; joint (LH) / partial (Oakes) / naive:

    Gauss,   τ = 0.6: ratio 0.999 / 0.950 / 0.772; 95 % coverage
             0.943 / 0.936 / 0.863; 90 % coverage 0.890 / 0.866 / 0.789.
             Pair (1, 1), same runs: ratio 1.044 / 1.004 / 0.813; 95 %
             0.957 / 0.946 / 0.870; 90 % 0.890 / 0.883 / 0.813.
    Clayton, τ = 0.5: ratio 1.028 / 1.009 / 0.913; 95 % coverage
             0.947 / 0.943 / 0.927; 90 % coverage 0.910 / 0.903 / 0.863.

Monte-Carlo bands at R = 300: binomial SE 1.3 points at 95 %, 1.7 at 90 %;
the SE ratio has a relative SE of about 4 %. The joint SE is 2-5 % above
Oakes' partial one: on this fixture the prior and the other pairs' τ carry
little information about τ_00 (small off-diagonal information), so Oakes'
partial profile was already close; the joint one brings the Gauss ratio from
0.95 to 1.00 and its 90 % coverage from 0.87 to 0.89, both within the band
of nominal. The empirical SD is that of the ICE estimate, not of the MLE;
a one-Newton-step MLE (``θ + I⁻¹g``) has coverage 0.933 (Gauss) / 0.940
(Clayton) at 95 % with the same SE, i.e. the ICE/MLE difference (a small
fraction of an SE per replicate, see
:func:`test_score_at_ice_fit_is_small_against_the_se`) does not move the
comparison.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.stats import norm

from pmcprg.pmc._lystig_hughes import (
    _eta_of_model,
    _prior_from_eta,
    ice_lh_information,
    lh_loglik_derivatives,
    model_from_theta,
)
from pmcprg.pmc._oakes import _tau_of_psi, ice_oakes_tau_se
from pmcprg.pmc.ice import ice
from pmcprg.pmc.inference import forward
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

Z95 = float(norm.ppf(0.975))
Z90 = float(norm.ppf(0.95))

_GAUSS_TOML = "pmcprg/pmc/models/hmc_dn_gauss_k2.toml"


def _hmc_dn(family: str, taus, A=((0.85, 0.15), (0.25, 0.75))) -> PMCModel:
    raw = {
        "model": {"name": f"hmc-dn-{family}-k2", "variant": "HMC-DN", "K": 2, "N_default": 600},
        "prior": {"A": [list(r) for r in A]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}},
        ],
        "copulas": [
            {"i": i, "j": j, "name": family, "tau": t}
            for (i, j), t in zip(((0, 0), (0, 1), (1, 0), (1, 1)), taus)
        ],
    }
    return PMCModel.from_dict(raw)


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
    return _hmc_dn("Clayton", (tau_diag, tau_off, tau_off, tau_diag),
                   A=((0.90, 0.10), (0.10, 0.90)))


_CASES = {
    "hmc_dn_gauss": (lambda: _hmc_dn("Gauss", (0.6, 0.2, -0.1, 0.5)), 5),
    "hmc_dn_clayton": (lambda: _hmc_dn("Clayton", (0.5, 0.2, 0.1, 0.4)), 6),
    "pmc_state_gauss": (lambda: _pmc_state((0.5, -0.2, 0.1, 0.4)), 7),
}


def _truth_and_perturbed(case: str):
    """The simulating model, and a model at a θ shifted away from it (so the
    score is far from zero and every term of the recursion is exercised)."""
    build, seed = _CASES[case]
    mdl = build()
    _, Y = simulate(mdl, N=600, seed=seed)
    theta = ice_lh_information(mdl, Y).theta
    shift = np.array([0.3, -0.2, 0.15, -0.1, 0.2, -0.15])[: theta.size]
    return model_from_theta(mdl, theta + shift), Y


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


@pytest.mark.parametrize("case", sorted(_CASES))
def test_theta_round_trip(case):
    mdl, Y = _truth_and_perturbed(case)
    info = ice_lh_information(mdl, Y)
    back = model_from_theta(mdl, info.theta)
    np.testing.assert_allclose(back.prior_p, mdl.prior_p, atol=1e-11)
    np.testing.assert_allclose(_eta_of_model(back), info.theta[:2], atol=1e-9)
    assert info.names == ("eta_01", "eta_11", "psi_00", "psi_01", "psi_10", "psi_11")


# ---------------------------------------------------------------------------
# The recursion: value, gradient, Hessian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(_CASES))
def test_loglik_matches_forward(case):
    mdl, Y = _truth_and_perturbed(case)
    ll, *_ = lh_loglik_derivatives(mdl, Y)
    assert ll == pytest.approx(forward(mdl, Y)[1], rel=1e-13)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_gradient_matches_finite_difference_of_loglik(case):
    mdl, Y = _truth_and_perturbed(case)
    _, g, _, _ = lh_loglik_derivatives(mdl, Y)
    theta = ice_lh_information(mdl, Y).theta
    h = 1e-5

    def ll(x):
        return forward(model_from_theta(mdl, x), Y)[1]

    g_fd = np.array([(ll(theta + h * e) - ll(theta - h * e)) / (2 * h)
                     for e in np.eye(theta.size)])
    assert np.max(np.abs(g)) > 1.0              # away from a stationary point
    rel = np.max(np.abs(g - g_fd) / np.maximum(1.0, np.abs(g)))
    assert rel < 1e-5, (g, g_fd)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_hessian_matches_finite_differences(case):
    mdl, Y = _truth_and_perturbed(case)
    _, _, H, _ = lh_loglik_derivatives(mdl, Y)
    theta = ice_lh_information(mdl, Y).theta
    P = theta.size
    E = np.eye(P)

    def grad(x):
        return lh_loglik_derivatives(model_from_theta(mdl, x), Y)[1]

    h = 1e-4
    H_g = np.array([(grad(theta + h * e) - grad(theta - h * e)) / (2 * h) for e in E])
    assert np.max(np.abs(H - H_g) / np.maximum(1.0, np.abs(H))) < 1e-5

    def ll(x):
        return forward(model_from_theta(mdl, x), Y)[1]

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
# At an ICE fit: SEs, Oakes' diagonal, the score
# ---------------------------------------------------------------------------

def _fit(build, family, seed, N=600):
    mdl = build()
    _, Y = simulate(mdl, N=N, seed=seed)
    fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                              "candidates": [family], "fit_margins": False})
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
# Scope guards and boundary
# ---------------------------------------------------------------------------

def test_unsupported_family_raises():
    mdl = _hmc_dn("Frank", (0.5, 0.2, 0.2, 0.5))
    _, Y = simulate(mdl, N=100, seed=0)
    with pytest.raises(NotImplementedError):
        ice_lh_information(mdl, Y)


def test_pair_margins_raise():
    mdl = PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml")
    _, Y = simulate(mdl, N=100, seed=0)
    with pytest.raises(NotImplementedError):
        ice_lh_information(mdl, Y)


def test_missing_rows_are_refused():
    mdl = PMCModel(_GAUSS_TOML)
    _, Y = simulate(mdl, N=100, seed=0)
    Y = Y.copy()
    Y[5] = np.nan
    with pytest.raises(ValueError, match="missing"):
        ice_lh_information(mdl, Y)


def test_boundary_pair_is_held_fixed():
    mdl = _clayton_model(tau_off=1e-4)
    _, Y = simulate(mdl, N=300, seed=3)
    info = ice_lh_information(mdl, Y)
    assert set(info.fixed_pairs) == {(0, 1), (1, 0)}
    assert info.names == ("eta_01", "eta_11", "psi_00", "psi_11")
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
