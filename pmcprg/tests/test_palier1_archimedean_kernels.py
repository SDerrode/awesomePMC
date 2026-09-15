"""Log-space kernels of Clayton, Gumbel–Hougaard, Joe and their survival copulas.

Complements ``test_copula_limits.py`` (the high-precision references of
palier 1) on what that file does not reach: Joe's τ(θ) map on its whole
domain (RB-2), the RB-1 log-likelihood itself, the inverse h-functions in the
tails, the survival wrapper's exact complements (FR-1) and the absence of
sentinel values.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate

from pmcprg.copulas import (CopulaClayton, CopulaGH, CopulaJoe, SurvivalClayton,
                         SurvivalGH, SurvivalJoe)
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta, _joe_theta_from_tau
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS

_FAMILIES = [CopulaClayton, SurvivalClayton, CopulaGH, SurvivalGH, CopulaJoe, SurvivalJoe]
_IDS = [c.__name__ for c in _FAMILIES]


# --------------------------------------------------------------------------
# Joe: Kendall's τ ↔ θ (RB-2)
# --------------------------------------------------------------------------

def _joe_tau_quadrature(theta):
    """τ = 1 + 4 ∫₀¹ φ/φ′ (Genest & MacKay 1986), φ(t) = −log(1 − (1−t)^θ)."""
    def ratio(t):
        a = (1.0 - t) ** theta
        return math.log1p(-a) * (1.0 - a) / (theta * (1.0 - t) ** (theta - 1.0))
    val, _ = integrate.quad(ratio, 0.0, 1.0, epsabs=1e-15, epsrel=1e-13, limit=200)
    return 1.0 + 4.0 * val


@pytest.mark.parametrize("theta", [1.2, 4.0 / 3.0, 1.5, 2.0, 2.5, 8.0 / 3.0, 3.0, 10.0])
def test_joe_tau_matches_genest_mackay_integral(theta):
    assert _joe_tau_from_theta(theta) == pytest.approx(_joe_tau_quadrature(theta), abs=1e-12)


def test_joe_tau_is_exact_at_independence():
    """τ(1) = 0 exactly and τ(1 + δ)/δ → τ′(1) = 2(π²/3 − 3) with relative precision."""
    assert _joe_tau_from_theta(1.0) == 0.0
    slope = 2.0 * (math.pi ** 2 / 3.0 - 3.0)
    for k in (52, 40, 30):
        delta = 2.0 ** -k                      # 1 + δ is exact
        assert _joe_tau_from_theta(1.0 + delta) / delta == pytest.approx(slope, rel=1e-8)


@pytest.mark.parametrize("theta", [4.0 / 3.0, 2.0, 8.0 / 3.0])
def test_joe_tau_is_continuous_and_increasing_across_expansions(theta):
    """No jump where the evaluation switches expansion: the increments on a
    grid straddling the switch are the slope times the step, to rounding."""
    step = 1e-9
    grid = theta + np.arange(-10, 11) * step
    taus = np.array([_joe_tau_from_theta(t) for t in grid])
    inc = np.diff(taus)
    assert np.all(inc > 0.0)
    np.testing.assert_allclose(inc, np.median(inc), rtol=1e-5)


@pytest.mark.parametrize("tau", [5e-324, 1e-300, 1e-16, 1e-12, 1e-7, 1e-3, 0.15888, 0.3550659, 0.47406,
                                 0.5, 0.9, 0.999, 1 - 1e-9, 1 - 1e-13, 1 - 2.0 ** -52])
def test_joe_theta_inverts_tau_on_the_whole_domain(tau):
    theta = _joe_theta_from_tau(tau)
    assert theta >= 1.0 and math.isfinite(theta)
    back = _joe_tau_from_theta(theta)
    assert abs(back - tau) <= 2e-16 + 1e-12 * tau
    if tau > 0.5:                              # 1 − τ keeps its relative precision
        assert abs((1.0 - back) - (1.0 - tau)) <= 1e-9 * (1.0 - tau)


def test_joe_theta_is_refused_where_it_does_not_exist():
    assert _joe_theta_from_tau(0.0) == 1.0
    for tau in (1.0, 1.5, -1e-12, float("nan")):
        with pytest.raises(CopulaParameterError):
            _joe_theta_from_tau(tau)


# --------------------------------------------------------------------------
# Clayton (RB-1)
# --------------------------------------------------------------------------

def test_clayton_log_likelihood_vanishes_at_independence():
    """RB-1: on 200 independent uniforms at θ = 10⁻¹⁵ the log-likelihood was +0.58."""
    uv = np.random.default_rng(0).random((200, 2))
    for tau in (5e-16, 1e-12, 1e-9):
        cop = CopulaClayton(tau_k=tau)
        ll = float(np.sum(cop.logpdf_array(uv)))
        # log c = θ(1 + log u)(1 + log v) + O(θ²); the residual cancellation is
        # absolute, ε·|log u + log v| per point, not ε/θ
        first_order = cop.theta * float(np.sum((1 + np.log(uv[:, 0])) * (1 + np.log(uv[:, 1]))))
        assert ll == pytest.approx(first_order, rel=1e-6, abs=1e-12)
        # C = uv·exp(θ log u log v + O(θ²))
        lu, lv = np.log(uv[:, 0]), np.log(uv[:, 1])
        np.testing.assert_allclose(cop.cdf_array(uv), uv[:, 0] * uv[:, 1] * np.exp(cop.theta * lu * lv),
                                   rtol=1e-12)


@pytest.mark.parametrize("cls", [CopulaClayton, SurvivalClayton], ids=["Clayton", "SClayton"])
def test_clayton_strong_dependence_has_no_overflow(cls):
    """u^{−θ} overflows below exp(−709.78/θ) = 9.8·10⁻¹² at θ = 28."""
    cop = cls(tau_k=28.0 / 30.0)
    assert cop.theta == pytest.approx(28.0)
    edge = np.array([EPS, 1e-300, 1e-12, 1e-6, 0.5, 1 - 1e-6, ONE_MINUS_EPS])
    uv = np.array([(a, b) for a in edge for b in edge])
    logc = cop.logpdf_array(uv)
    assert np.all(np.isfinite(logc))
    assert np.all((cop.cdf_array(uv) >= 0.0) & (cop.cdf_array(uv) <= 1.0))
    h = np.array([cop.conditional_cdf(b, a) for a, b in uv])
    assert np.all(np.isfinite(h)) and np.all((h >= 0.0) & (h <= 1.0))
    v = cop.inv_h_array(np.full(len(edge), 0.3), edge)
    assert np.all(np.isfinite(v))
    if cls is CopulaClayton:
        for a in edge:
            m = cop.majorant(a)
            assert math.isfinite(m)
            assert m >= max(cop.pdf([a, b]) for b in np.linspace(0.01, 0.99, 50))


# --------------------------------------------------------------------------
# Inverse h-functions in the tails
# --------------------------------------------------------------------------

def _tail_sample(n, seed):
    rng = np.random.default_rng(seed)
    e = 10.0 ** rng.uniform(-14, -0.3, n)
    return np.where(rng.random(n) < 0.5, e, 1.0 - e)


@pytest.mark.parametrize("tau", [1e-12, 0.3, 0.95])
@pytest.mark.parametrize("cls", _FAMILIES, ids=_IDS)
def test_inv_h_brackets_w_in_the_tails(cls, tau):
    """h(v⁻|u) ≤ w ≤ h(v⁺|u) at the float neighbours of v = inv_h(w|u)."""
    cop = cls(tau_k=tau)
    u, w = _tail_sample(300, 1), _tail_sample(300, 2)
    v = cop.inv_h_array(w, u)
    assert np.all(np.isfinite(v))
    lo = np.nextafter(np.nextafter(v, 0.0), 0.0)
    hi = np.nextafter(np.nextafter(v, 1.0), 1.0)
    inside = (lo > EPS) & (hi < ONE_MINUS_EPS)
    assert inside.sum() > 100
    h_lo = np.array([cop.conditional_cdf(b, a) for a, b in zip(u, lo)])
    h_hi = np.array([cop.conditional_cdf(b, a) for a, b in zip(u, hi)])
    # Relative slack, but never below two ulps of w: near w = 1 the h-function
    # is flat at float resolution (h(v⁻|u) = w = h(v⁺|u)), and a one-ulp
    # difference between libm implementations (Linux CI, NumPy 1.24) must not fail.
    tol = np.maximum(1e-10 * np.minimum(w, 1.0 - w), 2.0 * np.spacing(w))
    ok = (h_lo - tol <= w) & (w <= h_hi + tol)
    assert np.all(ok[inside]), (u[inside & ~ok][:3], w[inside & ~ok][:3])
    scalar = np.array([cop.inv_h(a, b) for a, b in zip(w[:20], u[:20])])
    np.testing.assert_array_equal(scalar, v[:20])


# --------------------------------------------------------------------------
# Survival wrapper: exact complements (FR-1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [1e-8, 0.5, 0.95])
@pytest.mark.parametrize("cls", [SurvivalClayton, SurvivalGH, SurvivalJoe], ids=_IDS[1::2])
def test_survival_is_the_base_at_the_complement(cls, tau):
    """Where 1 − u is exact (u ≥ 1/2) the survival copula is the base at (1−u, 1−v)."""
    cop = cls(tau_k=tau)
    base = cop._base
    g = np.array([0.5, 0.7, 0.9, 1 - 1e-6, 1 - 1e-12])
    uv = np.array([(a, b) for a in g for b in g])
    ref = 1.0 - uv                                   # exact (Sterbenz)
    np.testing.assert_allclose(cop.logpdf_array(uv), base.logpdf_array(ref), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(cop.cdf_array(uv), uv.sum(axis=1) - 1.0 + base.cdf_array(ref),
                               rtol=1e-12, atol=1e-15)
    for (a, b), (ra, rb) in zip(uv, ref):
        assert cop.conditional_cdf(b, a) == pytest.approx(1.0 - base.conditional_cdf(rb, ra),
                                                          rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("cls", _FAMILIES, ids=_IDS)
def test_kernel_pairs_are_complementary(cls):
    """(C, 1 − C), (h, 1 − h) and (b, 1 − b) returned by the kernel sum to one."""
    cop = cls(tau_k=0.6)
    k = cop._base if hasattr(cop, "_base") else cop
    g = np.array([1e-12, 1e-4, 0.3, 0.8, 1 - 1e-9])
    a, b = (x.ravel() for x in np.meshgrid(g, g))
    ka, kb = k._kcoord(a), k._kcoord(b)
    for pair in (k._k_cdf(ka, kb), k._k_h(kb, ka), k._k_inv_h(np.log(b), ka)):
        np.testing.assert_allclose(pair[0] + pair[1], 1.0, atol=4e-16)


# --------------------------------------------------------------------------
# No sentinels
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _FAMILIES, ids=_IDS)
def test_array_paths_propagate_nan_instead_of_a_sentinel(cls):
    cop = cls(tau_k=0.5)
    uv = np.array([[np.nan, 0.4], [0.3, 0.6]])
    for out in (cop.logpdf_array(uv), cop.pdf_array(uv), cop.cdf_array(uv),
                cop.inv_h_array(np.array([np.nan, 0.4]), np.array([0.3, 0.6]))):
        assert np.isnan(out[0]) and np.isfinite(out[1])


@pytest.mark.parametrize("cls", _FAMILIES, ids=_IDS)
def test_pdf_array_is_unfloored_exp_of_logpdf_array_in_the_corners(cls):
    cop = cls(tau_k=0.95)
    g = np.array([1e-12, 1e-6, 0.5, 1 - 1e-6, 1 - 1e-12])
    uv = np.array([(a, b) for a in g for b in g])
    logc = cop.logpdf_array(uv)
    assert np.all(np.isfinite(logc))
    assert logc.min() < math.log(EPS)                # the old floor would have bitten
    np.testing.assert_array_equal(cop.pdf_array(uv), np.exp(logc))
