"""t-EV copula (FR-9, last round) — model, τ(ρ, ν), parametrisation, fit.

Complements ``test_copula_limits.py`` (which added t-EV to its
high-precision ``decimal`` reference grid, ρ as the file's θ and ν in its
``delta`` slot) with the checks specific to this family: the Pickands
function's defining properties and its exchangeability, the identity of the
stable tail dependence function with the upper-tail limit of the package's
own Student-t copula, λ_U, the Hüsler–Reiss limit, τ against ``mpmath`` and
against a simulated Kendall's τ, the reachable-τ cap, the ``nu`` parameter's
own fitting branch (not Student's ``df``), recovery of τ and ν by ``fit``,
and multistart starts.

Every seed below is an integer literal (never ``hash()`` of anything).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.special import stdtr
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import CopulaEnum, CopulaHuslerReiss, CopulaStudent, CopulaTEV
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.copulas.extreme_value.husler_reiss import _husler_reiss_tau_from_lambda
from pmcprg.copulas.extreme_value.t_ev import (
    _NU_MAX,
    _NU_MIN,
    _TAU_CAP,
    _lambda_u,
    _log_tcdf,
    _s_of,
    _tau_of,
    _tev_A_terms,
    _tev_log_cdf,
    _tev_logpdf,
    _tev_tau_quad,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_G = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12])
_UV = np.array([(u, v) for u in _G for v in _G])


def _s_of_sp(sp, nu):
    """s = ln η from s' = ln((ν + 1)·η)."""
    return sp - math.log(nu + 1.0)


# --------------------------------------------------------------------------
# Registration, and the ``nu`` / ``df`` separation
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    entry = CopulaEnum.TEV
    assert entry.value.SHORT_NAME == "tEV"
    assert entry.klass is CopulaTEV
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "nu"]
    assert entry.value.TAU_MIN_MAX == [EPS, 1.0]
    assert entry in CopulaEnum.available()
    assert CopulaTEV.n_params == 2


def test_nu_bounds_are_registered_and_student_df_is_unchanged():
    from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
    assert EXTRA_PARAM_BOUNDS_BY_PARAM["nu"] == (0.5, 100.0, 4.0)
    assert EXTRA_PARAM_BOUNDS["CopulaTEV"] == {"nu": (0.5, 100.0, 4.0)}
    assert EXTRA_PARAM_BOUNDS["CopulaStudent"] == {"df": (2.001, 100.0, 4.0)}
    lo, hi, init = EXTRA_PARAM_BOUNDS_BY_PARAM["nu"]
    assert _NU_MIN <= lo < init < hi <= _NU_MAX


def test_t_ev_does_not_take_the_student_df_branch():
    """``_two_parameter_spec`` dispatches on the parameter *name*: under
    ``df`` t-EV would have gone through Student's (atanh τ, 1/ν) box with
    Student's ν > 2 bounds. ``nu`` has its own (logit λ_U, 1/ν) branch."""
    from pmcprg.copulas._fit import _two_parameter_spec
    assert "df" not in CopulaEnum.TEV.value.PARAMETERS_SET_NAME
    with pytest.raises(CopulaParameterError):
        CopulaTEV(tau_k=0.4, df=4.0)
    p0, stages, params_of = _two_parameter_spec(CopulaTEV, CopulaEnum.TEV, 0.4)
    _, s_stages, s_params_of = _two_parameter_spec(CopulaStudent, CopulaEnum.STUDENT, 0.4)
    assert len(stages) == 2 and stages[0] is stages[1]          # a restart
    (y_lo, y_hi), (e_lo, e_hi) = stages[0][2]
    assert (e_lo, e_hi) == (0.01, 2.0)                           # 1/ν on [0.5, 100]
    assert y_lo == pytest.approx(math.log(1e-6) - math.log1p(-1e-6), rel=1e-15)
    assert stages[0][2] != s_stages[0][2]
    assert set(params_of(p0)) == {"tau_k", "nu"}
    assert set(s_params_of((0.4, 0.25))) == {"tau_k", "df"}
    assert p0 == (pytest.approx(math.log(0.4 / 0.6)), pytest.approx(0.25))
    # every point of the box is a t-EV copula, built without clamping, whose
    # λ_U is the box coordinate and whose τ lies in [λ_U/2, λ_U]
    for y in (y_lo, -3.0, 0.0, 4.0, y_hi):
        for e in (e_lo, 0.25, 1.0, e_hi):
            params = params_of((y, e))
            cop = CopulaTEV(**params)
            assert cop.params == params
            lam = 1.0 / (1.0 + math.exp(-y))
            # λ_U → (τ, ν) → λ_U goes through SciPy's Student-t quantile and
            # cdf: 1.1·10⁻⁹ relative at λ_U ≈ 0.047 with the lowest supported
            # dependency versions (CI min-versions job), < 10⁻⁹ with current ones
            assert cop.tail_dependence()[1] == pytest.approx(lam, rel=1e-8)
            assert 0.5 * lam <= params["tau_k"] <= lam


def test_ice_m_step_fits_nu_not_df():
    from scipy.stats import rankdata

    from pmcprg.pmc.ice import FIT_FAILED_KEY, _fit_copula_params, _resolve_candidate
    uv = CopulaTEV(tau_k=0.5, nu=2.0).sample(n=1500, seed=11)
    pobs = np.column_stack([rankdata(uv[:, 0]), rankdata(uv[:, 1])]) / (len(uv) + 1)
    entry, cls = _resolve_candidate("tEV")
    assert cls is CopulaTEV
    p = _fit_copula_params(cls, entry, pobs[:, 0], pobs[:, 1], np.ones(len(uv)))
    assert FIT_FAILED_KEY not in p
    assert set(p) == {"tau_k", "nu"}
    assert p["tau_k"] == pytest.approx(0.5, abs=0.03) and p["nu"] == pytest.approx(2.0, rel=0.35)
    r = CopulaTEV.fit(uv)
    assert p["tau_k"] == pytest.approx(r.tau_k, rel=1e-6)
    assert p["nu"] == pytest.approx(r.copula.nu, rel=1e-5)
    # doubled weights: the same optimum (the objective is the total weighted LL)
    p2 = _fit_copula_params(cls, entry, pobs[:, 0], pobs[:, 1], np.full(len(uv), 2.0))
    assert p2["nu"] == pytest.approx(p["nu"], rel=1e-4)


# --------------------------------------------------------------------------
# Construction, reachable τ
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {"tau_k": 0.0}, {"tau_k": -0.1}, {"tau_k": 1.0},
    {"tau_k": 0.3, "nu": 0.0}, {"tau_k": 0.3, "nu": -1.0}, {"tau_k": 0.3, "nu": 0.04},
    {"tau_k": 0.3, "nu": 1001.0}, {"tau_k": 0.3, "nu": float("nan")},
    {"tau_k": 0.3, "nu": float("inf")}, {"tau_k": 0.3, "nu": True}, {"tau_k": 0.3, "nu": "four"},
])
def test_refused_parameters(kw):
    with pytest.raises(CopulaParameterError):
        CopulaTEV(**kw)


def test_default_nu_is_students_default():
    cop = CopulaTEV(tau_k=0.6)
    assert cop.nu == 4.0 and cop.params["nu"] == 4.0
    assert cop.theta == CopulaTEV(tau_k=0.6, nu=4.0).theta


@pytest.mark.parametrize("nu", [_NU_MIN, 1.0, 4.0, 100.0, _NU_MAX])
def test_the_tau_cap_is_the_same_for_every_nu(nu):
    """Every τ ∈ (0, 1) is a t-EV τ at every ν; the numerical cap 1 − 1.25·10⁻⁶
    is ν-free. At the cap the copula keeps τ; above it, stores the cap (RB-10)."""
    assert CopulaTEV.reachable_tau_bounds() == (EPS, _TAU_CAP)
    lo, hi = CopulaEnum.TEV.constructible_tau_range()
    assert lo == EPS and hi <= _TAU_CAP
    at = CopulaTEV(tau_k=_TAU_CAP, nu=nu)
    assert at.params["tau_k"] == _TAU_CAP
    assert _tev_tau_quad(at._s, nu) == pytest.approx(_TAU_CAP, abs=1e-13)
    above = CopulaTEV(tau_k=1.0 - 1e-9, nu=nu)
    assert above.params["tau_k"] == _TAU_CAP and above._s == at._s
    # x = √((ν+1)η) at the cap stays near 10⁻⁶ (Hüsler–Reiss λ ≈ 10⁶)
    assert 0.8e-6 < math.exp(0.5 * (at._s + math.log(nu + 1.0))) < 1.2e-6


@pytest.mark.parametrize("nu", [_NU_MIN, 0.5, 1.0, 2.5, 10.0, 100.0, _NU_MAX])
def test_tau_s_round_trip(nu):
    for tau in (1e-12, 1e-8, 1e-4, 0.05, 0.3, 0.5, 0.8, 0.99, 0.9999):
        cop = CopulaTEV(tau_k=tau, nu=nu)
        assert cop.params["tau_k"] == tau
        # θ = ρ is informative only: at ν = 0.05, τ = 1e-12 it rounds to −1
        # while ln η (the copula's own parameter) is finite
        assert -1.0 <= cop.theta < 1.0 and math.isfinite(cop._s)
        assert _tev_tau_quad(cop._s, nu) == pytest.approx(tau, rel=1e-9)


def test_s_inversion_refuses_non_positive_tau():
    with pytest.raises(CopulaParameterError):
        _s_of(0.0, 4.0)


# --------------------------------------------------------------------------
# The Pickands function
# --------------------------------------------------------------------------

_PAIRS = [(1e-6, 1.0), (0.05, 0.5), (0.3, 2.0), (0.7, 50.0), (0.95, 100.0),
          (0.5, 1000.0), (0.9, 0.05), (0.999, 4.0)]


@pytest.mark.parametrize("tau,nu", _PAIRS)
def test_pickands_bounds_symmetry_and_convexity(tau, nu):
    s = CopulaTEV(tau_k=tau, nu=nu)._s
    t = np.linspace(1e-6, 1 - 1e-6, 4001)
    A, Ap, App = _tev_A_terms(t, s, nu)
    assert np.all(A <= 1.0 + 2e-16)
    assert np.all(A >= np.maximum(t, 1 - t) - 2e-16)
    assert np.all(App >= 0.0)
    A_rev, Ap_rev, _ = _tev_A_terms(1.0 - t, s, nu)
    np.testing.assert_allclose(A_rev, A, rtol=0, atol=4e-16)     # A(t) = A(1 − t)
    # A' is steep near ½ at τ → 1: 1 − t and t are not exact mirrors in floats
    np.testing.assert_allclose(Ap_rev, -Ap, rtol=0, atol=1e-12)
    ends, _, _ = _tev_A_terms(np.array([1e-300, 1e-15, 1 - 1e-15]), s, nu)
    np.testing.assert_allclose(ends, 1.0, atol=2e-15)


@pytest.mark.parametrize("tau,nu", [(0.05, 0.5), (0.3, 2.0), (0.7, 50.0), (0.9, 0.05)])
def test_pickands_derivatives_match_finite_differences(tau, nu):
    s = CopulaTEV(tau_k=tau, nu=nu)._s
    t = np.array([0.1, 0.25, 0.4, 0.55, 0.8, 0.9])
    h = 1e-6
    A_p, Ap_p, _ = _tev_A_terms(t + h, s, nu)
    A_m, Ap_m, _ = _tev_A_terms(t - h, s, nu)
    _, Ap, App = _tev_A_terms(t, s, nu)
    np.testing.assert_allclose(Ap, (A_p - A_m) / (2 * h), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(App, (Ap_p - Ap_m) / (2 * h), rtol=1e-4, atol=1e-6)


def test_the_other_argument_assignment_is_not_a_pickands_function():
    """With (1 − t)/t in the t-term instead of t/(1 − t), A(1) = T_{ν+1}(−kρ) ≠ 1:
    the assignment t = w/(w + z) ↔ (w/z)^{1/ν} in the w-term is forced."""
    rho, nu = 0.5, 1.0
    k = math.sqrt((nu + 1) / (1 - rho * rho))
    t = 1 - 1e-12
    swapped = (t * stdtr(nu + 1, k * (((1 - t) / t) ** (1 / nu) - rho))
               + (1 - t) * stdtr(nu + 1, k * ((t / (1 - t)) ** (1 / nu) - rho)))
    assert swapped == pytest.approx(stdtr(nu + 1, -k * rho), abs=1e-10)
    assert swapped < 0.3
    s = math.log((1 - rho) / (1 + rho))
    assert float(_tev_A_terms(np.array([t]), s, nu)[0][0]) == pytest.approx(1.0, abs=1e-11)


@pytest.mark.parametrize("tau,nu", [(0.05, 0.5), (0.4, 3.0), (0.95, 100.0)])
def test_the_copula_is_exchangeable(tau, nu):
    cop = CopulaTEV(tau_k=tau, nu=nu)
    sw = _UV[:, ::-1]
    np.testing.assert_allclose(cop.logpdf_array(sw), cop.logpdf_array(_UV), rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(cop.cdf_array(sw), cop.cdf_array(_UV), rtol=1e-14, atol=0)


@pytest.mark.parametrize("w,z", [(1.0, 2.0), (0.5, 0.3), (2.0, 1.0)])
@pytest.mark.parametrize("rho,nu", [(0.5, 3.0), (-0.3, 5.0), (0.9, 3.0)])
def test_ell_w_is_the_upper_tail_limit_of_the_student_copula(w, z, rho, nu):
    """ℓ_w(w, z) = lim_{s→0} P(V ≤ 1 − s z | U = 1 − s w) for the Student-t copula
    (Demarta & McNeil 2005) — here the package's own Student h-function — is
    T_{ν+1}(k((w/z)^{1/ν} − ρ)), the t-EV conditional term (module docstring)."""
    from pmcprg.copulas.extreme_value.t_ev import _tev_logh
    stud = CopulaStudent(tau_k=2.0 / math.pi * math.asin(rho), df=nu)
    k = math.sqrt((nu + 1) / (1 - rho * rho))
    lw = float(stdtr(nu + 1, k * ((w / z) ** (1 / nu) - rho)))
    # the gap closes like s^{2/ν}; below s ≈ 1e-10 the rounding of 1 − s·z takes over
    errs = [abs(stud.conditional_cdf(1 - s * z, 1 - s * w) - lw) / lw for s in (1e-3, 1e-7, 1e-10)]
    assert errs[2] < 2e-5 and errs[2] < errs[0] / 100
    # the same ℓ_w from the t-EV copula at u = e^{−w}, v = e^{−z}: h·u/C = ℓ_w
    s = math.log((1 - rho) / (1 + rho))
    logC = float(_tev_log_cdf(np.array([-w]), np.array([-z]), s, nu)[0])
    logh = float(_tev_logh(np.array([-z]), np.array([-w]), s, nu)[0])
    assert math.exp(logh - logC - w) == pytest.approx(lw, rel=1e-13)


# --------------------------------------------------------------------------
# λ_U and the Hüsler–Reiss limit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,nu", _PAIRS)
def test_tail_dependence_matches_the_pickands_identity(tau, nu):
    cop = CopulaTEV(tau_k=tau, nu=nu)
    A_half = float(_tev_A_terms(np.array([0.5]), cop._s, nu)[0][0])
    lam_l, lam_u = cop.tail_dependence()
    assert lam_l == 0.0
    assert lam_u == pytest.approx(2.0 * (1.0 - A_half), rel=1e-12)
    # the Student-t copula's own coefficient at the same (ρ, ν) (ν > 2 there)
    if nu > 2.0 and abs(cop.theta) < 0.999:
        stud = CopulaStudent(tau_k=2.0 / math.pi * math.asin(cop.theta), df=nu)
        assert stud.tail_dependence()[1] == pytest.approx(lam_u, rel=1e-10)


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_husler_reiss_limit(tau):
    """ν → ∞ with (ν + 1)(1 − ρ)/(1 + ρ) = 1/λ² fixed is Hüsler–Reiss(λ): at
    this scaling λ_U is 2T_{ν+1}(−1/λ) → 2Φ(−1/λ), and C and τ converge at rate 1/ν."""
    hr = CopulaHuslerReiss(tau_k=tau)
    lam = hr.theta
    uv = _UV
    ref_C = hr.cdf_array(uv)
    ref_tau = _husler_reiss_tau_from_lambda(lam)
    errs_C, errs_tau = [], []
    for nu in (100.0, 1000.0):
        s = -2.0 * math.log(lam) - math.log(nu + 1.0)
        C = np.exp(_tev_log_cdf(np.log(uv[:, 0]), np.log(uv[:, 1]), s, nu))
        errs_C.append(np.max(np.abs(C - ref_C)))
        errs_tau.append(abs(_tau_of(s, nu) - ref_tau))
        cop = CopulaTEV(tau_k=_tau_of(s, nu), nu=nu)
        assert cop.tail_dependence()[1] == pytest.approx(
            2.0 * stdtr(nu + 1.0, -1.0 / lam), rel=1e-9)
    assert errs_C[1] < 1e-4 and errs_tau[1] < 3e-4
    assert 8.0 < errs_C[0] / errs_C[1] < 12.0
    assert 8.0 < errs_tau[0] / errs_tau[1] < 12.0
    # the density, where it is not a far tail (ln c > −5 under Hüsler–Reiss;
    # deeper, the t-tails are polynomial and the normal ones are not, so the
    # convergence is pointwise only): also at rate 1/ν
    g = np.linspace(0.05, 0.95, 19)
    mid = np.array([(a, b) for a in g for b in g])
    ref = hr.logpdf_array(mid)
    keep = ref > -5.0
    errs = []
    for nu in (100.0, 1000.0):
        s = -2.0 * math.log(lam) - math.log(nu + 1.0)
        lc = _tev_logpdf(np.log(mid[:, 0]), np.log(mid[:, 1]), s, nu)
        errs.append(np.max(np.abs(lc - ref)[keep]))
    assert errs[1] < 0.05 and 8.0 < errs[0] / errs[1] < 12.0


# --------------------------------------------------------------------------
# τ(s, ν)
# --------------------------------------------------------------------------

# 30-digit mpmath: mp.quad over L = logit t of (B'' − (1 − 2σ(L))B')/B,
# B(L) = A(σ(L)), B', B'' by mp.diff of the Demarta–McNeil A (not the module's
# closed forms), T by mpmath's incomplete Beta; (ν, s' = ln((ν + 1)η), τ).
_TAU_MPMATH = [
    (0.5, -20.0, 0.99994835663666333), (0.5, -2.0, 0.65516796763439062),
    (0.5, 0.0, 0.33474556376119229), (0.5, 5.0, 0.010927547411061075),
    (1.0, -6.0, 0.94593933530292312), (1.0, 2.0, 0.077915047623049448),
    (2.0, -20.0, 0.99994994036612027), (2.0, 0.0, 0.30740620366628203),
    (2.0, 5.0, 0.00084787990474716743), (10.0, -2.0, 0.64644260696965593),
    (10.0, 5.0, 7.637269322627051e-8), (50.0, -6.0, 0.94503170674957039),
    (50.0, 2.0, 0.0068422753580762349), (50.0, 5.0, 7.927407601761245e-17),
    (100.0, -20.0, 0.99994883529249752), (100.0, 0.0, 0.2572855622151798),
    (100.0, 2.0, 0.005918717124084124), (100.0, 5.0, 1.1907489277563293e-21),
]


@pytest.mark.parametrize("nu,sp,tau", _TAU_MPMATH)
def test_tau_quadrature_matches_mpmath(nu, sp, tau):
    # rel=1e-12, not 1e-14: the deepest-tail case (nu=100, sp=5, tau~1.19e-21)
    # measured a 1.2e-14 relative drift across GitHub Actions runners — the
    # quadrature's own exp/log chain at that magnitude is sensitive to the
    # CPU's SIMD/FMA codepath, not a platform this package controls.
    assert _tev_tau_quad(_s_of_sp(sp, nu), nu) == pytest.approx(tau, rel=1e-12, abs=0.0)


def test_tau_is_decreasing_in_s_and_bracketed_by_lambda_u():
    """τ decreases from the comonotone end (ρ → 1) to independence (ρ → −1),
    and λ_U/2 ≤ τ ≤ λ_U (the left inequality and τ ≤ 2λ_U hold for every
    symmetric EV copula; τ ≤ λ_U is observed)."""
    sps = np.linspace(math.log(1e-14), 12.0, 120)
    for nu in (_NU_MIN, 0.5, 1.0, 4.0, 30.0, 100.0, _NU_MAX):
        taus = np.array([_tev_tau_quad(_s_of_sp(sp, nu), nu) for sp in sps])
        lams = np.array([_lambda_u(_s_of_sp(sp, nu), nu) for sp in sps])
        pos = taus > 1e-250
        assert np.all(np.diff(taus[pos]) < 0.0)
        assert np.all(taus[pos] >= 0.5 * lams[pos]) and np.all(taus[pos] <= lams[pos])
        assert taus[0] > 1.0 - 2e-7


def test_negative_rho_is_admissible_and_weak():
    """ρ < 0 still gives an EV copula with τ > 0; τ → 0 only as ρ → −1."""
    for nu in (0.5, 4.0):
        taus = [_tev_tau_quad(math.log((1 - r) / (1 + r)), nu) for r in (0.0, -0.5, -0.9, -0.999)]
        assert all(t > 0 for t in taus) and all(np.diff(taus) < 0)
    assert CopulaTEV(tau_k=0.01, nu=1.0).theta < 0.0


# --------------------------------------------------------------------------
# pdf / cdf / h / h⁻¹
# --------------------------------------------------------------------------

# (τ, ν, u, v, C, h(v|u), ln c): mpmath, C from ℓ, h and c by numerical
# differentiation of C in (ln w, ln z) at 40–640 digits (raised until two
# precisions agree to 1e-15), at the ln η each copula builds.
_POINT_MPMATH = [
    (0.3, 1.0, 0.2, 0.7, 0.17215982770272192, 0.8501214680405251, -0.4065773768244633),
    (0.3, 1.0, 0.7, 0.2, 0.17215982770272192, 0.1171244526476099, -0.4065773768244633),
    (0.05, 0.5, 1e-12, 0.3, 3.4265176253243374e-13, 0.34264883172617977, 0.01617214036688241),
    (0.7, 50.0, 0.9, 0.95, 0.8978177561392328, 0.932078688927788, 1.508934464607806),
    (0.7, 50.0, 1e-06, 1e-06, 3.896694767462601e-08, 0.024059828845068763, 9.721639147828329),
    (0.95, 4.0, 0.999999, 0.5, 0.4999999999999885, 1.1928958585263879e-08, -17.488457077576925),
    (0.95, 4.0, 1e-12, 1e-06, 9.996013217828133e-13, 0.9995282683706697, 5.369314096146682),
    (1e-08, 100.0, 0.3, 0.9, 0.2700000012029102, 0.9000000024147955, -1.0283334357693901e-08),
    (0.0001, 2.0, 1e-12, 0.999999999999, 9.999999999990005e-13, 0.9999999999990006, -0.0005601315635434411),
    (0.5, 1000.0, 0.01, 0.99, 0.009999999992623318, 0.9999999985203278, -14.006262680662102),
    (0.9, 0.05, 0.4, 0.45, 0.38514903044367965, 0.960177891591179, -1.2873555136208772),
    (0.4, 3.0, 0.999, 0.001, 0.0009999491319254352, 5.317205432735814e-05, -2.9256558749143364),
]


@pytest.mark.parametrize("tau,nu,u,v,C,h,logc", _POINT_MPMATH)
def test_kernel_matches_mpmath(tau, nu, u, v, C, h, logc):
    cop = CopulaTEV(tau_k=tau, nu=nu)
    assert cop.cdf([u, v]) == pytest.approx(C, rel=1e-12)
    assert cop.conditional_cdf(v, u) == pytest.approx(h, rel=1e-11)
    assert float(cop.logpdf_array(np.array([[u, v]]))[0]) == pytest.approx(logc, rel=1e-12, abs=1e-12)
    assert cop.pdf([u, v]) == pytest.approx(math.exp(logc), rel=1e-11)


def test_log_t_cdf_in_the_far_lower_tail():
    """ln T_n(x) where ``stdtr`` underflows or loses the tail: the Cauchy tail
    ln T_1(x) = −ln(π|x|) + O(x⁻²), and 50-digit mpmath values."""
    x = np.array([-1e10, -1e100, -1e300])
    got = _log_tcdf(1.0, x)
    np.testing.assert_allclose(got, -np.log(np.pi * np.abs(x)), rtol=1e-15)
    for n, x, ref in ((101.0, -1000.0, -467.85372180099813),
                      (1001.0, -40.0, -482.0569335905918),
                      (1.05, -1e5, -13.222811534012957)):
        assert float(_log_tcdf(n, np.array([x]))[0]) == pytest.approx(ref, rel=1e-14)


@pytest.mark.parametrize("tau,nu", [(1e-4, 2.0), (0.3, 1.0), (0.6, 50.0), (0.95, 4.0), (0.8, 0.05)])
def test_h_inv_h_round_trip_and_saturation(tau, nu):
    cop = CopulaTEV(tau_k=tau, nu=nu)
    rng = np.random.default_rng(20260917)
    u = rng.uniform(1e-6, 1 - 1e-6, 300)
    v = rng.uniform(1e-6, 1 - 1e-6, 300)
    w = np.array([cop.conditional_cdf(b, a) for a, b in zip(u, v)])
    v_back = cop.inv_h_array(w, u)
    w_back = np.array([cop.conditional_cdf(b, a) for a, b in zip(u, v_back)])
    np.testing.assert_allclose(w_back, w, rtol=1e-9, atol=1e-12)
    assert cop.inv_h(0.0, 0.3) == EPS and cop.inv_h(1.0, 0.3) == 1.0 - EPS
    assert cop.inv_h(float(w[0]), float(u[0])) == v_back[0]


@pytest.mark.parametrize("tau,nu", [(1e-4, 2.0), (0.3, 1.0), (0.6, 50.0)])
def test_cdf_margins_monotonicity_and_density_normalisation(tau, nu):
    cop = CopulaTEV(tau_k=tau, nu=nu)
    for x in (0.2, 0.5, 0.8):
        assert cop.cdf([x, 1.0 - EPS]) == pytest.approx(x, abs=1e-12)
        assert cop.cdf([1.0 - EPS, x]) == pytest.approx(x, abs=1e-12)
    g = np.linspace(0.02, 0.98, 25)
    grid = np.array([(a, b) for a in g for b in g])
    C = cop.cdf_array(grid).reshape(25, 25)
    assert np.all(np.diff(C, axis=0) >= -1e-15) and np.all(np.diff(C, axis=1) >= -1e-15)
    m = (np.arange(400) + 0.5) / 400 * 0.96 + 0.02
    mm = np.array(np.meshgrid(m, m)).reshape(2, -1).T
    vol = cop.pdf_array(mm).sum() * (0.96 / 400) ** 2
    rect = cop.cdf([0.98, 0.98]) - cop.cdf([0.02, 0.98]) - cop.cdf([0.98, 0.02]) + cop.cdf([0.02, 0.02])
    assert vol == pytest.approx(rect, rel=2e-3)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,nu", [(0.1, 1.0), (0.4, 0.5), (0.5, 4.0), (0.8, 50.0)])
def test_sample_margins_and_monte_carlo_tau(tau, nu):
    uv = CopulaTEV(tau_k=tau, nu=nu).sample(n=20000, seed=4242)
    assert kstest(uv[:, 0], "uniform").pvalue > 1e-3
    assert kstest(uv[:, 1], "uniform").pvalue > 1e-3
    # s.e. of Kendall's τ at n = 20000 is ≤ 0.005 here; 5 s.e.
    assert kendalltau(uv[:, 0], uv[:, 1]).statistic == pytest.approx(tau, abs=0.025)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,nu,rel_nu", [(0.3, 1.0, 0.25), (0.5, 2.0, 0.3),
                                           (0.7, 5.0, 0.4), (0.9, 3.0, 0.35)])
def test_fit_recovers_tau_and_nu(tau, nu, rel_nu):
    uv = CopulaTEV(tau_k=tau, nu=nu).sample(n=3000, seed=31337)
    r = CopulaTEV.fit(uv)
    assert r.converged and r.method == "mle"
    assert r.tau_k == pytest.approx(tau, abs=0.04)
    assert r.copula.nu == pytest.approx(nu, rel=rel_nu)
    assert r.copula.params["tau_k"] == r.tau_k


def test_fit_tau_method_falls_back_to_mle(caplog):
    uv = CopulaTEV(tau_k=0.4, nu=3.0).sample(n=800, seed=7)
    with caplog.at_level(logging.WARNING):
        r_tau = CopulaTEV.fit(uv, method="tau")
    assert "falling back to MLE" in caplog.text
    r_mle = CopulaTEV.fit(uv, method="mle")
    assert r_tau.tau_k == r_mle.tau_k and r_tau.copula.nu == r_mle.copula.nu


@pytest.mark.parametrize("tau,nu,seed", [(0.35, 3.0, 77), (0.2, 20.0, 3), (0.8, 1.5, 4),
                                         (0.05, 2.0, 5)])
def test_joint_mle_does_not_depend_on_its_start(tau, nu, seed):
    from pmcprg.copulas._fit import _fit_two_parameter_mle
    uv = CopulaTEV(tau_k=tau, nu=nu).sample(n=1500, seed=seed)
    fits = [_fit_two_parameter_mle(CopulaTEV, CopulaEnum.TEV, uv, None, t0)
            for t0 in (1e-6, 0.02, 0.35, 0.9, 0.95)]
    assert all(f.converged for f in fits)
    best = max(f.log_likelihood for f in fits)
    for f in fits:
        assert f.log_likelihood == pytest.approx(best, abs=1e-6)
        assert f.params["nu"] == pytest.approx(fits[0].params["nu"], rel=1e-3)


def test_nu_at_the_upper_bound_on_husler_reiss_like_data():
    """Large ν is weakly identified: on data from ν = 80 the fit may stop at
    the box end ν = 100 — converged, with the right τ."""
    uv = CopulaTEV(tau_k=0.6, nu=80.0).sample(n=1500, seed=6)
    r = CopulaTEV.fit(uv)
    assert r.converged and r.tau_k == pytest.approx(0.6, abs=0.03)
    assert r.copula.nu > 20.0


def test_fit_best_includes_t_ev():
    from pmcprg.copulas._base import CopulaVirt
    uv = CopulaTEV(tau_k=0.5, nu=1.0).sample(n=1500, seed=3)
    res = CopulaVirt.fit_best(uv, families=[CopulaTEV, CopulaHuslerReiss, CopulaStudent],
                              method="mle")
    assert res[0].copula.__class__ is CopulaTEV


def test_standard_errors_are_not_implemented():
    cop = CopulaTEV(tau_k=0.4, nu=3.0)
    uv = cop.sample(n=200, seed=1)
    with pytest.raises(NotImplementedError):
        cop.standard_errors(uv)


# --------------------------------------------------------------------------
# Multistart: every start builds (no joint constraint to repair)
# --------------------------------------------------------------------------

MODELS = "pmcprg/pmc/models"


def _tev_model(tau=0.6, nu=3.0):
    from pmcprg.pmc.model import PMCModel
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    for blk in raw["copulas"]:
        blk.update({"name": "tEV", "tau": tau, "nu": nu})
    return PMCModel.from_dict(raw)


def test_constructible_params_is_the_identity():
    for params in ({"tau_k": 0.3, "nu": 0.5}, {"tau_k": 1e-9}, {"tau_k": 0.99, "nu": 100.0}):
        assert CopulaTEV.constructible_params(params) is params


@pytest.mark.parametrize("jitter", [0.1, 0.5])
def test_jittered_starts_always_build(jitter):
    from pmcprg.pmc._estim_common import perturb_initial_model
    model = _tev_model()
    nus = set()
    for seed in range(40):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=jitter)
        for blk in out.copula_blocks():
            cop = CopulaTEV(tau_k=blk["tau"], nu=blk["nu"])
            assert cop.params["tau_k"] == blk["tau"]
            nus.add(blk["nu"])
    assert len(nus) > 40


@pytest.mark.parametrize("mode", ["random", "sweep"])
def test_family_starts_with_t_ev_candidates_build(mode):
    from pmcprg.pmc._estim_common import build_multistart_inits
    from pmcprg.pmc.model import PMCModel
    cands = ["Gauss", "tEV", "Student"]
    for model in (PMCModel(f"{MODELS}/pmc_gauss_k2.toml"), _tev_model(0.9, 90.0)):
        for seed in range(3):
            cfg = {"n_starts": 1 + len(cands) ** 2 + 3, "multistart_seed": seed,
                   "multistart_jitter": 0.25, "multistart_families": mode, "candidates": cands}
            for m, _ in build_multistart_inits(model, cfg, label="ICE"):
                for blk in m.copula_blocks():
                    if blk["name"] == "tEV":
                        assert "df" not in blk
                        CopulaTEV(tau_k=blk["tau"], nu=blk.get("nu", 4.0))


def test_selection_placeholder_builds():
    from pmcprg.pmc.ice import FIT_FAILED_KEY, _copula_candidate_fits, _copula_placeholder
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=50), rng.uniform(size=50)
    cands = ["tEV", "Gauss"]
    resolved, _ = _copula_candidate_fits(cands, u, v, np.ones(50), "mle")
    blk = _copula_placeholder(cands, resolved, "mle", {"name": "tEV", "tau": 0.0})
    assert blk[FIT_FAILED_KEY] is True
    CopulaTEV(tau_k=blk["tau"], nu=blk.get("nu", 4.0))


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_t_ev_cases_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _INDEP_TAUS, _TAIL_TAUS, _file_table, _match_file_key, _required_param_keys,
    )
    table = _file_table()
    assert "tEV" in _TAIL_TAUS and "tEV" in _INDEP_TAUS
    need = {k for k in _required_param_keys() if k[0] == "tEV"}
    assert len(need) == 6
    assert {k[2] for k in need} == {"1.0", "50.0", "4.0", "10.0", "100.0", "2.0"}
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
