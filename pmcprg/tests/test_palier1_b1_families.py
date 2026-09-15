"""Palier 1, lot B1 — A12, A14, BB1, AMH, FGM, cubic sections, product.

Complements ``test_copula_limits.py`` (FR-1) where its grid does not reach:
the registered parameter bounds (AMH θ = −1 and θ → 1, FGM |θ| = 1, cubic
sections θ = 1/4, where the density itself goes to 0 at a corner), the
τ ↔ θ maps near independence (RB-10), BB1 at τ = 0.999 (RB-8), and the
sentinels the scalar paths used to return (RB-9).

References are computed with ``decimal`` from the copula CDFs — the density
as a mixed central difference of C at a step 10⁻²⁰ of the distance to the
edge, the precision raised until two successive precisions agree — so they
do not share a formula with the package.
"""
from __future__ import annotations

import math
from decimal import Decimal as D, localcontext

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_ONE = D(1)


def _c_amh(u, v, th):
    return u * v / (_ONE - th * (_ONE - u) * (_ONE - v))


def _c_fgm(u, v, th):
    return u * v * (_ONE + th * (_ONE - u) * (_ONE - v))


def _c_cubsec(u, v, th):
    return u * v * (_ONE + 2 * th * (_ONE - u) * (_ONE - v) * (_ONE + u + v - 2 * u * v))


def _ref_logc(C, u, v, th):
    """log of ∂²C/∂u∂v at (u, v), converged to 12 digits in ``decimal``."""
    prev = None
    for prec in (60, 120, 240, 480):
        with localcontext() as ctx:
            ctx.prec = prec
            U, V, TH = D(u), D(v), D(th)
            hu = min(U, _ONE - U) * D("1e-20")
            hv = min(V, _ONE - V) * D("1e-20")
            mixed = (C(U + hu, V + hv, TH) - C(U + hu, V - hv, TH)
                     - C(U - hu, V + hv, TH) + C(U - hu, V - hv, TH)) / (4 * hu * hv)
        if prev is not None and abs(mixed - prev) <= abs(prev) * D("1e-12"):
            return float(mixed.ln()) if mixed > 0 else -math.inf
        prev = mixed
    raise AssertionError(f"reference did not converge at {(u, v, th)}")


_EDGE = (EPS, 1e-12, 1e-6, 0.3, 0.5, 1 - 1e-6, 1 - 1e-12, 1 - EPS)
_EDGE_UV = np.array([(u, v) for u in _EDGE for v in _EDGE])

_ENTRY = {e.value.SHORT_NAME: e for e in CopulaEnum}


# --------------------------------------------------------------------------
# Densities at the registered parameter bounds
# --------------------------------------------------------------------------

@pytest.mark.parametrize("short,tau,C", [
    pytest.param("AMH", CopulaEnum.AMH.value.TAU_MIN_MAX[0], _c_amh, id="AMH-theta=-1"),
    pytest.param("AMH", CopulaEnum.AMH.value.TAU_MIN_MAX[1], _c_amh, id="AMH-theta->1"),
    pytest.param("FGM", -2.0 / 9.0, _c_fgm, id="FGM-theta=-1"),
    pytest.param("FGM", 2.0 / 9.0, _c_fgm, id="FGM-theta=+1"),
    pytest.param("CubSec", 33.0 / 200.0, _c_cubsec, id="CubSec-theta=1/4"),
])
def test_density_at_registered_bound_matches_high_precision_reference(short, tau, C):
    """Where the density approaches 0 at a corner, no floor and no NaN."""
    cop = _ENTRY[short].klass(tau_k=tau)
    logc = cop.logpdf_array(_EDGE_UV)
    ref = np.array([_ref_logc(C, u, v, cop.theta) for u, v in _EDGE_UV])
    assert not np.any(np.isnan(logc))
    np.testing.assert_allclose(logc, ref, rtol=1e-9, atol=1e-9)
    pdf = cop.pdf_array(_EDGE_UV)
    np.testing.assert_allclose(pdf, np.exp(logc), rtol=1e-12)
    np.testing.assert_allclose([cop.pdf(p) for p in _EDGE_UV], pdf, rtol=1e-13)


def test_cubic_section_density_vanishes_exactly_at_the_corner():
    from pmcprg.copulas.explicit.cubic_section import _cubsec_logpdf, _cubsec_pdf
    u, v = np.array([0.0, 1.0]), np.array([1.0, 0.0])
    assert np.all(_cubsec_pdf(u, v, 0.25) == 0.0)
    assert np.all(_cubsec_logpdf(u, v, 0.25) == -np.inf)


# --------------------------------------------------------------------------
# τ ↔ θ near independence and at the bounds (RB-10)
# --------------------------------------------------------------------------

def _amh_tau_decimal(th: float) -> float:
    with localcontext() as ctx:
        ctx.prec = 80
        t = D(th)
        return float(_ONE - 2 * (t + (_ONE - t) ** 2 * (_ONE - t).ln()) / (3 * t * t))


@pytest.mark.parametrize("theta", [1e-300, -1e-12, 1e-8, -0.01, 0.05, -0.2, 0.5, -0.5000001, -0.9, 0.99])
def test_amh_tau_of_theta_matches_high_precision(theta):
    """The closed form lost up to 10⁻¹¹ for |θ| ≤ 0.05 and everything below 10⁻⁸."""
    from pmcprg.copulas.archimedean.amh import _amh_tau_from_theta
    ref = 2.0 * theta / 9.0 if abs(theta) < 1e-100 else _amh_tau_decimal(theta)
    assert _amh_tau_from_theta(theta) == pytest.approx(ref, rel=1e-13, abs=0.0)


@pytest.mark.parametrize("tau", [-1e-8, -1e-15, 1e-12, 1e-4, 0.2,
                                 CopulaEnum.AMH.value.TAU_MIN_MAX[0],
                                 CopulaEnum.AMH.value.TAU_MIN_MAX[1]])
def test_amh_stored_tau_is_the_realised_tau(tau):
    """τ = −10⁻⁸ used to build θ = −1.2·10⁻⁶, a copula realising τ = −2.7·10⁻⁷."""
    cop = CopulaEnum.AMH.klass(tau_k=tau)
    realised = 2.0 * cop.theta / 9.0 if abs(cop.theta) < 1e-100 else _amh_tau_decimal(cop.theta)
    assert realised == pytest.approx(tau, rel=1e-12, abs=1e-16)


def test_amh_theta_outside_the_family_is_refused():
    from pmcprg.copulas.archimedean.amh import _amh_theta_from_tau
    assert _amh_theta_from_tau(0.0) == 0.0
    for tau in (-0.19, 0.3334, 0.4):
        with pytest.raises(CopulaParameterError):
            _amh_theta_from_tau(tau)


@pytest.mark.parametrize("tau", [1e-12, 1e-6, 0.1, 33.0 / 200.0])
def test_cubic_section_theta_solves_the_tau_equation(tau):
    """θ = (75/4)(2/3 − √…) kept three digits at τ = 10⁻¹²; θ ≤ 1/4 at the bound."""
    cop = CopulaEnum.CUBSEC.klass(tau_k=tau)
    with localcontext() as ctx:
        ctx.prec = 50
        t = D(tau)
        ref = float((D(75) / 4) * (D(2) / 3 - (D(4) / 9 - 8 * t / 75).sqrt()))
    assert cop.theta == pytest.approx(ref, rel=1e-14)
    assert 0.0 <= cop.theta <= 0.25


# --------------------------------------------------------------------------
# BB1 in extreme dependence and at the edge (RB-8, K-12)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.999, 0.9999])
def test_bb1_log_density_is_finite_in_extreme_dependence(tau):
    """RB-8: the former kernel gave NaN on 19/20 and 20/20 of these points."""
    cop = CopulaEnum.BB1.klass(tau_k=tau, delta=1.5)
    uv = np.vstack([np.random.default_rng(3).random((20, 2)), _EDGE_UV])
    logc = cop.logpdf_array(uv)
    assert np.all(np.isfinite(logc))
    ok = logc > -700.0
    np.testing.assert_allclose(cop.pdf_array(uv)[ok], np.exp(logc[ok]), rtol=1e-12)
    assert np.all(np.isfinite(cop.cdf_array(uv)))
    assert all(math.isfinite(cop.conditional_cdf(v, u)) for u, v in uv)


def test_bb1_cdf_at_the_upper_edge_is_the_margin():
    """K-12: C(0.7, 1 − 10⁻¹⁵) was 2·10⁻¹⁶ (τ = 0.34, δ = 1.5)."""
    cop = CopulaEnum.BB1.klass(tau_k=0.34, delta=1.5)
    assert cop.cdf([0.7, 1 - 1e-15]) == pytest.approx(0.7, rel=1e-14)
    assert cop.cdf([1 - 1e-15, 0.7]) == pytest.approx(0.7, rel=1e-14)
    near_gumbel = CopulaEnum.BB1.klass(tau_k=1 - 1 / 1.5 + 1e-9, delta=1.5)   # θ ≈ 3·10⁻⁹
    assert near_gumbel.cdf([0.4, 1 - EPS]) == pytest.approx(0.4, rel=1e-14)


# --------------------------------------------------------------------------
# A12 tail dependence and scalar sentinels (RB-9, RB-10)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.4, 0.7, 0.95])
def test_a12_tail_dependence_is_closed_form(tau):
    cop = CopulaEnum.A12.klass(tau_k=tau)
    lam_l, lam_u = cop.tail_dependence()
    assert lam_l == pytest.approx(2.0 ** (-1.0 / cop.theta), rel=1e-15)
    assert lam_u == pytest.approx(2.0 - 2.0 ** (1.0 / cop.theta), rel=1e-15)
    u = 1e-9                                   # the diagonal limits, now reachable
    assert cop.cdf([u, u]) / u == pytest.approx(lam_l, abs=1e-6)
    w = 1 - 1e-9
    assert (1 - 2 * w + cop.cdf([w, w])) / (1 - w) == pytest.approx(lam_u, abs=1e-6)


def test_scalar_paths_return_the_value_not_a_sentinel():
    """The former fallbacks: ``EPS`` for a density, a 0/1 step for h."""
    bb1 = CopulaEnum.BB1.klass(tau_k=0.9, delta=1.5)
    assert bb1.pdf([1e-12, 1e-12]) == pytest.approx(4.320210091e12, rel=1e-8)   # was EPS
    a14 = CopulaEnum.A14.klass(tau_k=0.99)
    assert a14.logpdf_array(np.array([[0.5, 1 - 1e-12]]))[0] < -745.0
    assert a14.pdf([0.5, 1 - 1e-12]) == 0.0                                    # was EPS
    # A12 on the diagonal: h(u|u) = 2^{1/θ−1} / (u + 2^{1/θ}(1 − u))², no
    # cancellation. θ = 200/3 overflowed (1/v − 1)^θ and returned the step 1.
    a12 = CopulaEnum.A12.klass(tau_k=0.99)
    for u in (1e-12, 1e-6, 1 - 1e-6):
        k = 2.0 ** (1.0 / a12.theta)
        assert a12.conditional_cdf(u, u) == pytest.approx(0.5 * k / (u + k * (1 - u)) ** 2, rel=1e-10)


def test_product_array_paths_are_closed_forms():
    cop = CopulaEnum.PRODUCT.klass(tau_k=0.0)
    uv = np.random.default_rng(0).random((1000, 2))
    assert np.all(cop.pdf_array(uv) == 1.0)
    assert np.all(cop.logpdf_array(uv) == 0.0)
    np.testing.assert_array_equal(cop.cdf_array(uv), uv[:, 0] * uv[:, 1])
    with pytest.raises(ValueError):
        cop.logpdf_array(uv[:, 0])
