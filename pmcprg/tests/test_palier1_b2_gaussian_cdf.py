"""Gaussian copula CDF — accuracy in the tails and determinism (AUDIT_COPULES FR-1).

``CopulaGaussian.cdf`` used to go through scipy's randomised quasi-Monte-Carlo
``multivariate_normal.cdf`` (absolute tolerance 10⁻⁵): 0 instead of 7.4·10⁻²⁰
at u = v = 10⁻¹⁴, and a different value at every call. It is now the
deterministic Drezner–Wesolowsky/Genz one-dimensional integral.

The reference here is independent of that formula: the probability
P(X ≤ x, Y ≤ y) = ∫_{−∞}^{x} φ(s) Φ((y − ρs)/√(1−ρ²)) ds, whose log-integrand
is concave, evaluated in log space (``log_ndtr``) by scipy's adaptive
quadrature on the integrand scaled by its maximum (found by bisection on the
derivative), split at the maximum. Integrating along x and along y agree to
~10⁻¹² on this grid.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate, special

from pmcprg.copulas.elliptical.gaussian import CopulaGaussian

_LOG2PI = math.log(2.0 * math.pi)


def _z(p: float) -> float:
    return float(special.ndtri(p)) if p <= 0.5 else -float(special.ndtri(1.0 - p))


def _reference_cdf(u: float, v: float, rho: float) -> float:
    x, y = _z(u), _z(v)
    sig = math.sqrt((1.0 - rho) * (1.0 + rho))

    def logf(s):
        return -0.5 * s * s - 0.5 * _LOG2PI + float(special.log_ndtr((y - rho * s) / sig))

    def dlogf(s):
        z = (y - rho * s) / sig
        mills = math.exp(-0.5 * z * z - 0.5 * _LOG2PI - float(special.log_ndtr(z)))
        return -s - rho / sig * mills

    if dlogf(x) >= 0.0:
        mode = x
    else:
        lo, hi = x - 400.0, x
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if dlogf(mid) > 0.0 else (lo, mid)
        mode = 0.5 * (lo + hi)
    lmax = logf(mode)
    total = 0.0
    for a, b in ((mode - 45.0, mode), (mode, x)):
        if b > a:
            total += integrate.quad(lambda s: math.exp(min(logf(s) - lmax, 700.0)), a, b,
                                    epsabs=0.0, epsrel=1e-13, limit=500)[0]
    return math.exp(lmax + math.log(total)) if total > 0.0 else 0.0


_G = (1e-14, 1e-10, 1e-6, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9,
      1 - 1e-3, 1 - 1e-6, 1 - 1e-10, 1 - 1e-14)
_UV = np.array([(u, v) for u in _G for v in _G])
_RHOS = (-0.99, -0.9, -0.5, -0.1, 0.0, 0.1, 0.5, 0.9, 0.99)


def _copula(rho: float) -> CopulaGaussian:
    return CopulaGaussian(tau_k=2.0 / math.pi * math.asin(rho))


@pytest.mark.parametrize("rho", _RHOS)
def test_cdf_array_matches_log_space_quadrature(rho):
    cop = _copula(rho)
    ref = np.array([_reference_cdf(u, v, cop.theta) for u, v in _UV])
    got = cop.cdf_array(_UV)
    big = ref > 1e-300
    rel = np.abs(got[big] - ref[big]) / ref[big]
    worst = int(np.argmax(rel))
    assert rel[worst] <= 1e-8, (f"rel. error {rel[worst]:.2e} at (u, v) = {_UV[big][worst]}: "
                                f"got {got[big][worst]:.17g}, ref {ref[big][worst]:.17g}")
    # Below the double-precision floor the value must be a genuine underflow, not a sentinel.
    assert np.all(got[~big] <= 1e-290)


def test_the_qmc_regression_point():
    """u = v = 10⁻¹⁴, ρ = ½: QMC returned 0; the value is 7.448·10⁻²⁰."""
    cop = _copula(0.5)
    ref = _reference_cdf(1e-14, 1e-14, cop.theta)
    assert ref == pytest.approx(7.448e-20, rel=1e-3)
    assert cop.cdf([1e-14, 1e-14]) == pytest.approx(ref, rel=1e-10)


@pytest.mark.parametrize("rho", (-0.95, 0.3, 0.95))
def test_cdf_is_deterministic_and_independent_of_the_batch(rho):
    uv = np.random.default_rng(3).random((5000, 2))
    first = _copula(rho).cdf_array(uv)
    second = _copula(rho).cdf_array(uv)
    assert np.array_equal(first, second)
    assert np.array_equal(_copula(rho).cdf_array(uv[::-1]), first[::-1])
    scalar = np.array([_copula(rho).cdf(p) for p in uv[:50]])
    assert np.array_equal(scalar, first[:50])


@pytest.mark.parametrize("rho", (-0.9, -0.3, 0.3, 0.9))
def test_cdf_symmetry_and_rotation(rho):
    """C(u, v) = C(v, u) and C_ρ(u, v) + C_{−ρ}(u, 1 − v) = u."""
    uv = np.random.default_rng(5).random((2000, 2))
    cop, rot = _copula(rho), _copula(-rho)
    np.testing.assert_array_equal(cop.cdf_array(uv), cop.cdf_array(uv[:, ::-1]))
    np.testing.assert_allclose(cop.cdf_array(uv) + rot.cdf_array(np.column_stack((uv[:, 0], 1.0 - uv[:, 1]))),
                               uv[:, 0], rtol=0, atol=1e-12)


def test_cdf_array_is_faster_than_the_qmc_backend():
    """10⁵ points: the QMC backend took ≈ 0.4 s; the quadrature must stay vectorised."""
    import time
    uv = np.random.default_rng(0).random((100_000, 2))
    cop = _copula(0.5)
    t0 = time.perf_counter()
    cop.cdf_array(uv)
    assert time.perf_counter() - t0 < 2.0


@pytest.mark.parametrize("rho", (-0.9, 0.9))
def test_h_function_keeps_relative_precision_in_both_tails(rho):
    """h(v|u) + h(1−v|1−u) = 1 by radial symmetry; each is computed from its own
    lower tail, so tiny values are exact to relative precision, not rounded to 0."""
    cop = _copula(rho)
    t = 2.0 ** -40                      # 1 − t is exact in binary
    for u, v in ((t, t), (t, 0.5), (0.5, t), (1 - t, t), (t, 1 - t)):
        lo = cop.conditional_cdf(v, u)
        hi = cop.conditional_cdf(1.0 - v, 1.0 - u)
        x, y = _z(u), _z(v)
        expect = float(special.ndtr((y - cop.theta * x) / math.sqrt(1.0 - cop.theta ** 2)))
        assert lo == pytest.approx(expect, rel=1e-12)
        assert lo + hi == pytest.approx(1.0, abs=1e-15)
    np.testing.assert_array_equal(cop.logpdf_array(np.array([[t, t], [t, 1 - t]])),
                                  cop.logpdf_array(np.array([[1 - t, 1 - t], [1 - t, t]])))
