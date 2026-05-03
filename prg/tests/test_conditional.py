"""
Verification of the conditional CDF  h(v|u) = ∂C(u,v)/∂u.

Checks: boundary values, normalization ∫h dv=1, monotonicity,
and consistency with numerical finite-differences on the CDF.
"""
import numpy as np
import pytest

from prg.copulas import (
    CopulaGaussian, CopulaStudent,
    CopulaGH, CopulaClayton, CopulaA12, CopulaA14,
    CopulaProduct, CopulaFGM, CopulaCubSec,
)
from prg.tools.tools import EPS, ONE_MINUS_EPS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ALL = [
    pytest.param(CopulaGaussian(tau_k= 0.5),  id='Gaussian'),
    pytest.param(CopulaStudent (tau_k= 0.5),  id='Student'),
    pytest.param(CopulaGH      (tau_k= 0.5),  id='GH'),
    pytest.param(CopulaClayton (tau_k= 0.5),  id='Clayton'),
    pytest.param(CopulaA12     (tau_k= 0.5),  id='A12'),
    pytest.param(CopulaA14     (tau_k= 0.5),  id='A14'),
    pytest.param(CopulaProduct (tau_k= 0.0),  id='Product'),
    pytest.param(CopulaFGM     (tau_k= 0.15), id='FGM'),
    pytest.param(CopulaCubSec  (tau_k= 0.15), id='CubSec'),
]

# Copulas with analytical CDF → can cross-validate conditional_cdf numerically
_WITH_CDF = [p for p in _ALL if p.id != 'Student']

_U_TEST = [0.2, 0.5, 0.8]
_V_TEST = [0.2, 0.5, 0.8]

# ---------------------------------------------------------------------------
# 1. Boundary  h(1|u) = 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_conditional_upper_boundary(cop):
    """h(1−ε | u) must be ≈ 1 for all u."""
    v_top = 1.0 - 1e-6
    for u in _U_TEST:
        h = cop.conditional_cdf(v_top, u)
        assert abs(h - 1.0) < 1e-4, f"h({v_top}|{u}) = {h:.6f}, expected ≈ 1"


# ---------------------------------------------------------------------------
# 2. Boundary  h(0|u) = 0
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_conditional_lower_boundary(cop):
    """h(ε | u) must be ≈ 0 for all u."""
    v_bot = 1e-6
    for u in _U_TEST:
        h = cop.conditional_cdf(v_bot, u)
        assert abs(h) < 1e-4, f"h({v_bot}|{u}) = {h:.6f}, expected ≈ 0"


# ---------------------------------------------------------------------------
# 3. Marginal identity  ∫₀¹ h(v|u) du = v
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_conditional_marginal_identity(cop):
    """∫₀¹ h(v|u) du = v for each fixed v (fundamental identity of h-functions)."""
    u_vals = np.linspace(0.005, 0.995, 80)
    for v in [0.3, 0.5, 0.7]:
        h_vals = np.array([cop.conditional_cdf(v, u) for u in u_vals])
        integral = float(np.trapezoid(h_vals, u_vals))
        assert abs(integral - v) < 0.02, (
            f"∫h({v}|u) du = {integral:.4f} (expected {v:.2f} ± 0.02)"
        )


# ---------------------------------------------------------------------------
# 4. Monotonicity in v
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_conditional_monotone(cop):
    """h(v|u) must be non-decreasing in v for each fixed u."""
    v_vals = np.linspace(0.01, 0.99, 40)
    for u in _U_TEST:
        h_vals = np.array([cop.conditional_cdf(v, u) for v in v_vals])
        diffs = np.diff(h_vals)
        assert np.all(diffs >= -1e-6), (
            f"h(v|{u}) not monotone: min diff = {diffs.min():.2e}"
        )


# ---------------------------------------------------------------------------
# 5. Consistency with numerical derivative of CDF  (CDF-available copulas only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _WITH_CDF)
def test_conditional_agrees_with_cdf_derivative(cop):
    """Analytical h(v|u) agrees with (C(u+h,v)−C(u−h,v))/(2h) within 1%."""
    delta = 5e-4
    for u in _U_TEST:
        for v in _V_TEST:
            u_lo = max(EPS, u - delta)
            u_hi = min(ONE_MINUS_EPS, u + delta)
            fd_h = (cop.cdf([u_hi, v]) - cop.cdf([u_lo, v])) / (u_hi - u_lo)
            ana_h = cop.conditional_cdf(v, u)
            rel_err = abs(ana_h - fd_h) / (abs(fd_h) + 1e-10)
            assert rel_err < 0.01, (
                f"h({v}|{u}): analytical={ana_h:.6f}, FD={fd_h:.6f}, "
                f"rel_err={rel_err:.4f}"
            )


# ---------------------------------------------------------------------------
# 6. Non-negativity  h(v|u) ∈ [0,1]
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_conditional_in_unit_interval(cop):
    """h(v|u) ∈ [0,1] on the interior grid."""
    v_vals = np.linspace(0.05, 0.95, 15)
    for u in _U_TEST:
        for v in v_vals:
            h = cop.conditional_cdf(v, u)
            assert 0.0 - 1e-9 <= h <= 1.0 + 1e-9, (
                f"h({v:.2f}|{u:.2f}) = {h:.6f} out of [0,1]"
            )
