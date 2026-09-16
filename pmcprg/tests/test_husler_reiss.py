"""Hüsler–Reiss copula (FR-9, round 2) — construction, round-trips, τ(λ), fit.

Mirrors ``test_galambos.py``'s depth and structure. Complements
``test_copula_limits.py`` (which added Hüsler–Reiss to its own grid and
high-precision reference file) with the checks that are specific to this
family: boundary rejection, the direction of λ (the module's own "Direction
of λ" derivation, re-checked here numerically), the h/h⁻¹ round-trip, τ(λ)
against a simulated Kendall's τ, sampling's marginal uniformity, ``fit``
recovery, and that the reference file addition covers exactly the
registered cases.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest, norm

from pmcprg.copulas import CopulaEnum, CopulaHuslerReiss
from pmcprg.copulas.extreme_value.husler_reiss import (
    _husler_reiss_A_terms,
    _husler_reiss_tau_from_lambda,
    _husler_reiss_lambda_from_tau,
    _LAMBDA_HI,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

HUSLER_REISS = CopulaEnum.HUSLER_REISS


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    assert HUSLER_REISS.value.SHORT_NAME == "HuslerReiss"
    assert HUSLER_REISS.value.CLASS_NAME == "CopulaHuslerReiss"
    assert HUSLER_REISS.klass is CopulaHuslerReiss
    assert HUSLER_REISS.value.AVAILABLE
    assert HUSLER_REISS in CopulaEnum.available()


def test_tau_range_is_one_sided():
    """Extreme-value copulas have no negative dependence (module docstring)."""
    lo, hi = HUSLER_REISS.value.TAU_MIN_MAX
    assert lo == pytest.approx(EPS)
    assert hi == 1.0


# --------------------------------------------------------------------------
# Construction / boundary rejection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.0, -0.1, -1.0])
def test_non_positive_tau_is_refused(tau):
    with pytest.raises(CopulaParameterError):
        CopulaHuslerReiss(tau_k=tau)


def test_tau_at_or_beyond_one_is_refused():
    with pytest.raises(CopulaParameterError):
        CopulaHuslerReiss(tau_k=1.0)
    with pytest.raises(CopulaParameterError):
        CopulaHuslerReiss(tau_k=1.5)


def test_a_tau_beyond_the_reachable_bound_is_clamped_and_stored():
    """Same RB-10/F1/G2/Galambos convention: λ is capped at ``_LAMBDA_HI``,
    so a τ beyond ``reachable_tau_bounds()[1]`` (a registered bound the base
    class's ``_check_tau_k`` would otherwise accept, since < 1.0) builds at
    the cap and stores the τ it realises, not the one asked for."""
    lo, hi = HUSLER_REISS.klass.reachable_tau_bounds()
    assert lo == pytest.approx(EPS)
    assert 0.999 < hi < 1.0
    assert HUSLER_REISS.reachable_tau(0.999999) == pytest.approx(hi)
    assert HUSLER_REISS.reachable_tau(0.5) == 0.5
    assert HUSLER_REISS.constructible_tau_range()[1] <= hi

    cop = CopulaHuslerReiss(tau_k=0.999999)
    assert cop.theta == _LAMBDA_HI
    assert cop.params["tau_k"] == pytest.approx(hi, rel=1e-9)
    assert cop.params["tau_k"] < 0.999999


# --------------------------------------------------------------------------
# Direction of λ — re-checked numerically (module docstring's key finding:
# λ runs the SAME way as Galambos's θ, not the opposite, contrary to a
# plausible first reading of the Pickands formula).
# --------------------------------------------------------------------------

def test_lambda_zero_limit_is_independence_not_comonotone():
    """A(t) -> 1 (independence) as λ -> 0+, for fixed interior t — module
    docstring's "Direction of λ" derivation, re-checked numerically."""
    for t in (0.2, 0.4, 0.5, 0.6, 0.8):
        A_small, _, _ = _husler_reiss_A_terms(t, 1e-3)
        assert A_small == pytest.approx(1.0, abs=1e-2)


def test_lambda_infinity_limit_is_comonotone_not_independence():
    """A(t) -> max(t, 1-t) (comonotone) as λ -> ∞, for fixed interior t."""
    for t in (0.2, 0.4, 0.6, 0.8):
        A_large, _, _ = _husler_reiss_A_terms(t, 1e4)
        assert A_large == pytest.approx(max(t, 1.0 - t), abs=1e-2)


def test_tau_increases_with_lambda():
    """λ -> 0 is independence (τ -> 0), λ -> ∞ is comonotone (τ -> 1) — the
    SAME direction as Galambos's θ, not the opposite (module docstring)."""
    lambdas = [1e-2, 0.1, 0.5, 1.0, 10.0, 1e3, 1e6]
    taus = [_husler_reiss_tau_from_lambda(lam) for lam in lambdas]
    assert all(t >= 0.0 for t in taus)
    assert all(a <= b for a, b in zip(taus, taus[1:]))
    assert taus[0] < 1e-4
    assert taus[-1] > 0.999


def test_upper_tail_dependence_increases_with_lambda_towards_one():
    """λ_U = 2Φ(-1/λ) -> 0 as λ -> 0, -> 1 as λ -> ∞ (module docstring):
    the tail-dependence formula independently confirms the direction."""
    lambdas = [0.3, 1.0, 10.0, 1e4]
    lam_us = [CopulaHuslerReiss(tau_k=_husler_reiss_tau_from_lambda(lam)).tail_dependence()[1]
             for lam in lambdas]
    assert lam_us[0] < 1e-3
    assert lam_us[-1] > 0.999
    assert all(a <= b for a, b in zip(lam_us, lam_us[1:]))


# --------------------------------------------------------------------------
# τ(λ) — closed-form claims re-derived and cross-checked (module docstring)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lam,tau_expected", [
    # 50-digit mpmath, Genest & MacKay (1986) integral of A''; see the
    # module docstring for the closed-form A, A', A'' this integrates and
    # their own cross-check against mpmath numerical differentiation.
    (1.0, 0.25544930692454965),
    (2.0, 0.5386784028894569),
    (5.0, 0.7913872794127044),
    (20.0, 0.9446652597558867),
])
def test_tau_from_lambda_matches_high_precision_quadrature(lam, tau_expected):
    assert _husler_reiss_tau_from_lambda(lam) == pytest.approx(tau_expected, rel=1e-6)


def test_tau_lambda_round_trip():
    for tau in (1e-8, 1e-4, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95):
        lam = _husler_reiss_lambda_from_tau(tau)
        assert _husler_reiss_tau_from_lambda(lam) == pytest.approx(tau, rel=1e-6, abs=1e-9)


def test_upper_tail_dependence_matches_the_pickands_identity():
    """λ_U = 2(1 − A(1/2)) for any EV copula (Nelsen 2006, Thm 5.7.3);
    A(1/2) = Φ(1/λ) here ⇒ λ_U = 2Φ(−1/λ) (module docstring)."""
    for lam in (0.2, 1.0, 3.0, 10.0):
        cop = CopulaHuslerReiss(tau_k=_husler_reiss_tau_from_lambda(lam))
        lam_l, lam_u = cop.tail_dependence()
        assert lam_l == 0.0
        assert lam_u == pytest.approx(2.0 * norm.cdf(-1.0 / lam))


# --------------------------------------------------------------------------
# A, A', A'' — internal consistency (numerical differentiation cross-check)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.3, 1.0, 5.0, 50.0])
def test_A_prime_matches_numerical_differentiation(lam):
    h = 1e-6
    ts = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    A_lo, _, _ = _husler_reiss_A_terms(ts - h, lam)
    A_hi, _, _ = _husler_reiss_A_terms(ts + h, lam)
    _, Ap, _ = _husler_reiss_A_terms(ts, lam)
    numeric = (A_hi - A_lo) / (2 * h)
    np.testing.assert_allclose(Ap, numeric, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("lam", [0.3, 1.0, 5.0])
def test_A_double_prime_matches_numerical_differentiation(lam):
    h = 1e-4
    ts = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
    _, Ap_lo, _ = _husler_reiss_A_terms(ts - h, lam)
    _, Ap_hi, _ = _husler_reiss_A_terms(ts + h, lam)
    _, _, App = _husler_reiss_A_terms(ts, lam)
    numeric = (Ap_hi - Ap_lo) / (2 * h)
    np.testing.assert_allclose(App, numeric, rtol=1e-3, atol=1e-6)


def test_A_boundary_values_are_one():
    """A(0) = A(1) = 1 for any Pickands function."""
    A0, _, _ = _husler_reiss_A_terms(1e-10, 2.0)
    A1, _, _ = _husler_reiss_A_terms(1.0 - 1e-10, 2.0)
    assert A0 == pytest.approx(1.0, abs=1e-6)
    assert A1 == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# pdf / cdf / h / h⁻¹
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [1e-4, 0.3, 0.7, 0.95])
def test_h_inv_h_round_trip(tau):
    """``inv_h(h(v|u), u) == v`` — checked on the *w = h(v|u)* scale (same
    rationale as Galambos's version of this test: at strong dependence
    h(·|u) can be nearly flat over part of its range)."""
    cop = CopulaHuslerReiss(tau_k=tau)
    rng = np.random.default_rng(0)
    us = rng.uniform(0.01, 0.99, 40)
    vs = rng.uniform(0.01, 0.99, 40)
    for u, v in zip(us, vs):
        w = cop.conditional_cdf(v, u)
        v_back = cop.inv_h(w, u)
        w_back = cop.conditional_cdf(v_back, u)
        assert w_back == pytest.approx(w, abs=1e-6)


@pytest.mark.parametrize("tau", [1e-4, 0.3, 0.7, 0.95])
def test_pdf_array_is_exp_of_logpdf_array(tau):
    cop = CopulaHuslerReiss(tau_k=tau)
    rng = np.random.default_rng(1)
    uv = rng.uniform(0.001, 0.999, (200, 2))
    logc = cop.logpdf_array(uv)
    finite = np.isfinite(logc) & (logc > -700) & (logc < 700)
    np.testing.assert_allclose(cop.pdf_array(uv)[finite], np.exp(logc[finite]), rtol=1e-12)


@pytest.mark.parametrize("tau", [1e-4, 0.3, 0.7, 0.95])
def test_cdf_is_a_valid_copula_cdf(tau):
    """C(u,0)=C(0,v)=0-ish, C(u,1)=u, C(1,v)=v, and C nondecreasing in each arg."""
    cop = CopulaHuslerReiss(tau_k=tau)
    for u in (0.2, 0.5, 0.8):
        assert cop.cdf([u, 1.0 - EPS]) == pytest.approx(u, abs=1e-6)
        assert cop.cdf([1.0 - EPS, u]) == pytest.approx(u, abs=1e-6)
    grid = np.linspace(0.05, 0.95, 10)
    for u in grid:
        vals = [cop.cdf([u, v]) for v in grid]
        assert all(a <= b + 1e-12 for a, b in zip(vals, vals[1:]))


# --------------------------------------------------------------------------
# Sampling: marginal uniformity and realised τ (Monte-Carlo, module docstring)
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tau", [0.2, 0.5, 0.8])
def test_sample_has_uniform_margins_and_the_right_tau(tau):
    cop = CopulaHuslerReiss(tau_k=tau)
    uv = cop.sample(n=20000, seed=42)
    assert kstest(uv[:, 0], "uniform").pvalue > 0.001
    assert kstest(uv[:, 1], "uniform").pvalue > 0.001
    tau_hat = kendalltau(uv[:, 0], uv[:, 1]).statistic
    # Monte-Carlo s.e. of Kendall's tau at n=20000 is ~0.006; allow 5 s.e.
    assert abs(tau_hat - tau) < 0.03


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau_true", [0.2, 0.5, 0.8])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_recovers_tau_from_simulated_data(tau_true, method):
    cop = CopulaHuslerReiss(tau_k=tau_true)
    uv = cop.sample(n=3000, seed=7)
    r = CopulaHuslerReiss.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.06)
    assert r.copula.params["tau_k"] == r.tau_k


def test_fit_best_includes_husler_reiss():
    from pmcprg.copulas._base import CopulaVirt
    cop = CopulaHuslerReiss(tau_k=0.6)
    uv = cop.sample(n=800, seed=3)
    results = CopulaVirt.fit_best(uv, families=[CopulaHuslerReiss], method="tau")
    assert len(results) == 1
    assert results[0].copula.__class__ is CopulaHuslerReiss


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_husler_reiss_cases_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _TAIL_TAUS, _INDEP_TAUS, _file_table, _required_param_keys, _match_file_key,
    )
    assert "HuslerReiss" in _TAIL_TAUS
    assert "HuslerReiss" in _INDEP_TAUS
    table = _file_table()
    need = {k for k in _required_param_keys() if k[0] == "HuslerReiss"}
    assert need, "no Hüsler–Reiss cases registered in test_copula_limits.py"
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
