"""
Scientific verification of copula implementations.

Checks: non-negativity, normalization (∫∫c=1), CDF boundary conditions,
PDF–CDF consistency, and Kendall's tau correctness.
"""
import importlib
import math

import numpy as np
from scipy.integrate import trapezoid   # np.trapezoid needs NumPy >= 2
import pytest

from pmcprg.copulas import CopulaEnum, CopulaGaussian, CopulaStudent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Target dependence for the sweep, clipped into each family's own τ range
# (Product is degenerate at 0, FGM caps at 2/9, CubSec at 0.165, AMH at 1/3).
_TARGET_TAU = 0.5


def _registered_families() -> list:
    """One instance per registered, available copula family.

    Derived from :class:`CopulaEnum` instead of a hand-written list: this
    suite used to name only 9 of the 17 families, so the 8 it omitted were
    never checked for normalisation or PDF–CDF consistency. That is how a
    spurious θ factor in the Joe density — which made ∫∫c integrate to θ
    rather than 1 — survived here undetected. A family added to the registry
    is now covered automatically.
    """
    params = []
    for entry in CopulaEnum:
        meta = entry.value
        if not meta.MODULE or not meta.AVAILABLE:
            continue
        cls = getattr(importlib.import_module(meta.MODULE), meta.CLASS_NAME)
        tau_min, tau_max = meta.TAU_MIN_MAX
        tau = float(np.clip(_TARGET_TAU, tau_min, tau_max))
        # constructible_params (identity for every family without a joint
        # τ/extra-parameter constraint) repairs a clipped τ that lands where
        # the default extra parameter is inadmissible — BB190/BB1270 (audit
        # FR-8, BB1 round): clipping 0.5 into their negative range lands at
        # −EPS, where the default δ = 1.5 needs θ_base > 0 at τ_base ≈ 0,
        # i.e. δ ≈ 1, exactly the same joint constraint BB1 itself needed
        # this hook for.
        cop_params = cls.constructible_params({"tau_k": tau})
        params.append(pytest.param(cls(**cop_params), id=meta.SHORT_NAME))
    return params


_ALL = _registered_families()


def _provides_cdf(cop) -> bool:
    """Student's CDF is a bivariate-t integral with no closed form."""
    try:
        cop.cdf([0.5, 0.5])
    except NotImplementedError:
        return False
    return True


_WITH_CDF = [p for p in _ALL if _provides_cdf(p.values[0])]

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
    return float(trapezoid(trapezoid(Z, u, axis=1), u))


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
    return 1.0 - 4.0 * float(trapezoid(trapezoid(Z, u, axis=1), u))


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


def test_joe_density_carries_no_theta_factor():
    """Regression: the Joe density once carried a spurious leading θ.

    ``∫∫c`` then integrated to θ (≈ 2.86 at τ=0.5) instead of 1, inflating
    every Joe log-density by log θ ≈ 1.05 nat. Since Joe belongs to the
    default ICE candidate list, it won family selection almost
    unconditionally. Verified by high-accuracy quadrature rather than the
    coarse trapezoid grid above, at several dependence levels.
    """
    from scipy import integrate

    from pmcprg.copulas import CopulaJoe

    for tau in (0.2, 0.5, 0.8):
        cop = CopulaJoe(tau_k=tau)
        mass, _ = integrate.dblquad(
            lambda v, u: cop.pdf([u, v]),
            1e-9, 1 - 1e-9, lambda u: 1e-9, lambda u: 1 - 1e-9,
            epsabs=1e-6,
        )
        assert mass == pytest.approx(1.0, abs=2e-3), (
            f"Joe(τ={tau}) integrates to {mass:.6f}; θ={cop.theta:.4f} "
            f"(a value near θ means the spurious factor is back)"
        )


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


@pytest.mark.slow
@pytest.mark.parametrize("cop", _ALL)
def test_realised_kendall_tau_matches_the_requested_one(cop):
    """The τ a family *realises* must be the τ it was asked for.

    This is the one property the rest of the suite structurally cannot check.
    Every other test compares the family against *itself* — pdf against the
    derivative of its own cdf, pdf_array against pdf, the survival copula
    against its base — and a self-consistent family with a wrong τ→θ map
    passes all of them. Only sampling, whose Kendall τ is measured
    independently of the implementation, exposes it. Two such defects were
    found this way: Plackett inverted Spearman's ρ, and A14 used
    ``2/(3(1−τ))``, an expression that coincides with the correct
    ``1/(1−τ) − 1/2`` only at the family's lower τ bound.
    """
    from scipy.stats import kendalltau

    tau_ref = cop.params["tau_k"]
    if tau_ref == 0.0:                      # Product: nothing to invert
        pytest.skip("independence copula has no τ↔θ map")

    sample = cop.sample(20_000, seed=5)
    tau_emp = kendalltau(sample[:, 0], sample[:, 1])[0]
    assert tau_emp == pytest.approx(tau_ref, abs=0.02), (
        f"requested τ={tau_ref:.4f}, realised τ={tau_emp:.4f} "
        f"(θ={getattr(cop, 'theta', float('nan')):.4f})"
    )


def test_plackett_tau_is_kendall_not_spearman():
    """Regression: Plackett's τ↔θ map used to invert Spearman's ρ_S.

    ``(θ+1)/(θ−1) − 2θ log θ/(θ−1)²`` is the closed form of ρ_S, not of τ
    (which has none for this family), so ``CopulaPlackett(tau_k=0.5)``
    realised τ ≈ 0.35 while every other family honoured Kendall's τ.
    Ground truth here is the *empirical* τ of a sample, independent of the
    quadrature the implementation itself uses.
    """
    from scipy.stats import kendalltau

    from pmcprg.copulas import CopulaPlackett
    from pmcprg.copulas.explicit.plackett import _plackett_rho_s_from_theta

    for tau in (-0.4, 0.35, 0.5, 0.7):
        cop = CopulaPlackett(tau_k=tau)
        sample = cop.sample(50_000, seed=13)
        tau_emp = kendalltau(sample[:, 0], sample[:, 1])[0]
        assert tau_emp == pytest.approx(tau, abs=0.01), (
            f"requested τ={tau}, realised τ={tau_emp:.4f} (θ={cop.theta:.4f})"
        )
        # |ρ_S| > |τ| strictly: catches a regression to the ρ_S inversion,
        # which would place ρ_S — not τ — at the requested value.
        assert abs(_plackett_rho_s_from_theta(cop.theta)) > abs(tau) + 0.05


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
