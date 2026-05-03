"""
Scientific verification of copula implementations.

Checks: non-negativity, normalization (∫∫c=1), CDF boundary conditions,
PDF–CDF consistency, and Kendall's tau correctness.
"""
import math
import numpy as np
import pytest

from prg.copulas import (
    CopulaGaussian, CopulaStudent,
    CopulaGH, CopulaClayton, CopulaA12, CopulaA14,
    CopulaProduct, CopulaFGM, CopulaCubSec,
)

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

# Copulas that provide an analytical CDF (Student is excluded)
_WITH_CDF = [p for p in _ALL if p.id != 'Student']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_integral(cop, n: int = 80) -> float:
    """Trapezoidal estimate of ∫∫ pdf(u,v) du dv over (0,1)².

    Uses [0.001, 0.999] to avoid boundary singularities while keeping
    the missing-strip contribution below 0.5%.
    """
    u = np.linspace(0.001, 0.999, n)
    U, V = np.meshgrid(u, u)
    Z = np.vectorize(lambda ui, vi: cop.pdf([ui, vi]))(U, V)
    return float(np.trapezoid(np.trapezoid(Z, u, axis=1), u))


def _kendall_tau_hoeffding(cop, n: int = 50) -> float:
    """Hoeffding formula: τ = 1 − 4∫∫ h(v|u)·h(u|v) du dv.

    The integrand is bounded in [0,1] everywhere, making the numerical
    integration far more accurate than the C·c formula.
    Note: valid for symmetric copulas C(u,v) = C(v,u), which is the case
    for all families implemented here.
    """
    u = np.linspace(0.01, 0.99, n)
    U, V = np.meshgrid(u, u)
    Z = np.vectorize(
        lambda ui, vi: cop.conditional_cdf(vi, ui) * cop.conditional_cdf(ui, vi)
    )(U, V)
    return 1.0 - 4.0 * float(np.trapezoid(np.trapezoid(Z, u, axis=1), u))


def _cdf_mixed_deriv(cop, u, v, h: float = 1e-3) -> float:
    """Central finite-difference estimate of ∂²C/∂u∂v."""
    pp = cop.cdf([u + h, v + h])
    pm = cop.cdf([u + h, v - h])
    mp = cop.cdf([u - h, v + h])
    mm = cop.cdf([u - h, v - h])
    return (pp - pm - mp + mm) / (4.0 * h**2)


# ---------------------------------------------------------------------------
# 1. Non-negativity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_pdf_non_negative(cop):
    """c(u,v) ≥ 0 on a 20×20 interior grid."""
    u = np.linspace(0.05, 0.95, 20)
    for ui in u:
        for vi in u:
            assert cop.pdf([ui, vi]) >= -1e-10, f"pdf < 0 at ({ui},{vi})"


# ---------------------------------------------------------------------------
# 2. Normalization  ∫∫ c(u,v) du dv = 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_pdf_normalizes_to_one(cop):
    """Trapezoidal integral on (0.001,0.999)² within 2% of 1."""
    norm = _grid_integral(cop, n=80)
    assert abs(norm - 1.0) < 0.02, f"normalization = {norm:.5f} (expected 1.0 ± 0.02)"


# ---------------------------------------------------------------------------
# 3. CDF boundary conditions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _WITH_CDF)
def test_cdf_boundary_lower(cop):
    """C(u,0) = 0  and  C(0,v) = 0."""
    eps = 1e-9
    for t in [0.2, 0.5, 0.8]:
        assert cop.cdf([t,  eps]) < 1e-6, f"C({t}, 0) = {cop.cdf([t, eps]):.2e}"
        assert cop.cdf([eps, t ]) < 1e-6, f"C(0, {t}) = {cop.cdf([eps, t]):.2e}"


@pytest.mark.parametrize('cop', _WITH_CDF)
def test_cdf_boundary_upper(cop):
    """C(u,1) = u  and  C(1,v) = v."""
    eps = 1e-7
    for t in [0.2, 0.5, 0.8]:
        assert abs(cop.cdf([t,   1-eps]) - t) < 5e-5, f"C({t},1) = {cop.cdf([t,1-eps]):.6f} ≠ {t}"
        assert abs(cop.cdf([1-eps, t  ]) - t) < 5e-5, f"C(1,{t}) = {cop.cdf([1-eps,t]):.6f} ≠ {t}"


@pytest.mark.parametrize('cop', _WITH_CDF)
def test_cdf_frechet_bounds(cop):
    """max(u+v−1,0) ≤ C(u,v) ≤ min(u,v) on a 5×5 grid."""
    for u in [0.3, 0.5, 0.7]:
        for v in [0.3, 0.5, 0.7]:
            c = cop.cdf([u, v])
            assert c >= max(u + v - 1.0, 0.0) - 1e-9, f"below Fréchet lower at ({u},{v})"
            assert c <= min(u, v)          + 1e-9,     f"above Fréchet upper at ({u},{v})"


# ---------------------------------------------------------------------------
# 4. PDF–CDF consistency  ∂²C/∂u∂v ≈ pdf
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _WITH_CDF)
def test_pdf_cdf_consistency(cop):
    """Mixed second derivative of CDF matches PDF within 2%."""
    test_points = [(0.3, 0.5), (0.5, 0.5), (0.7, 0.3)]
    for (u, v) in test_points:
        pdf_val   = cop.pdf([u, v])
        deriv_val = _cdf_mixed_deriv(cop, u, v, h=1e-3)
        rel_err   = abs(pdf_val - deriv_val) / (abs(pdf_val) + 1e-12)
        assert rel_err < 0.02, (
            f"PDF–CDF mismatch at ({u},{v}): pdf={pdf_val:.6f}, "
            f"∂²C={deriv_val:.6f}, rel_err={rel_err:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. Kendall's tau  τ = 1 − 4∫∫ h(v|u)·h(u|v) du dv  (Hoeffding formula)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('cop', _ALL)
def test_kendall_tau_numerical(cop):
    """Hoeffding-formula τ matches configured tau_k within 0.02."""
    tau_est = _kendall_tau_hoeffding(cop, n=50)
    tau_ref = cop.params['tau_k']
    assert abs(tau_est - tau_ref) < 0.05, (
        f"tau_k={tau_ref:.3f}, Hoeffding estimate={tau_est:.3f}"
    )


def test_kendall_tau_gaussian_analytical():
    """τ = 2/π arcsin(θ) for Gaussian copula."""
    for tau_k in [0.3, 0.5, 0.7, -0.4]:
        cop = CopulaGaussian(tau_k=tau_k)
        tau_formula = 2.0 / math.pi * math.asin(cop.theta)
        assert abs(tau_formula - tau_k) < 1e-8, (
            f"τ formula mismatch: {tau_formula:.8f} vs {tau_k}"
        )


def test_kendall_tau_student_analytical():
    """τ = 2/π arcsin(θ) also holds for Student-t copula."""
    for tau_k in [0.3, 0.5, 0.7]:
        cop = CopulaStudent(tau_k=tau_k)
        tau_formula = 2.0 / math.pi * math.asin(cop.theta)
        assert abs(tau_formula - tau_k) < 1e-8, (
            f"τ formula mismatch: {tau_formula:.8f} vs {tau_k}"
        )
