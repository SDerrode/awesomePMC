"""Tawn copulas, types 1 and 2 (FR-9, round 3) — model, τ(θ, ψ), joint constraint, fit.

Complements ``test_copula_limits.py`` (which added both types to its
high-precision ``decimal`` reference grid, ψ carried in its ``delta`` slot)
with the checks specific to this family: the Pickands function's defining
properties, the reachable-τ cap τ < ψ, the ψ = 1 → Gumbel limit, the
transposition between the two types, τ(θ, ψ) against ``mpmath`` and against
a simulated Kendall's τ, the asymmetry seen by the exchangeability test,
recovery of both τ and ψ by ``fit``, and the multistart repair of refused
(τ, ψ) pairs — mirroring ``test_multistart_joint_constraints.py`` for BB1.

Every seed below is an integer literal (never ``hash()`` of anything).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import CopulaEnum, CopulaGH, CopulaTawn1, CopulaTawn2
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.copulas.extreme_value.tawn import (
    _THETA_HI,
    _tau_of,
    _tawn_A_terms,
    _tawn_logpdf,
    _tawn_tau_quad,
    _theta_of,
)
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

TYPES = [CopulaTawn1, CopulaTawn2]
_G = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12])
_UV = np.array([(u, v) for u in _G for v in _G])


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry,klass,short", [
    (CopulaEnum.TAWN1, CopulaTawn1, "Tawn1"), (CopulaEnum.TAWN2, CopulaTawn2, "Tawn2")])
def test_registered_in_copula_enum(entry, klass, short):
    assert entry.value.SHORT_NAME == short
    assert entry.klass is klass
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "psi"]
    assert entry.value.TAU_MIN_MAX == [EPS, 1.0]
    assert entry in CopulaEnum.available()
    assert klass.n_params == 2


def test_psi_bounds_are_registered_for_both_types():
    from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
    assert EXTRA_PARAM_BOUNDS_BY_PARAM["psi"] == (0.01, 1.0, 1.0)
    for name in ("CopulaTawn1", "CopulaTawn2"):
        assert EXTRA_PARAM_BOUNDS[name] == {"psi": (0.01, 1.0, 1.0)}


@pytest.mark.parametrize("klass", TYPES)
def test_two_parameter_spec_takes_the_psi_branch(klass):
    """``psi`` must not fall into the generic (τ, extra) box branch: the
    optimiser box is (ln(θ − 1), ψ), and every point of it builds."""
    from pmcprg.copulas._fit import _two_parameter_spec
    entry = next(e for e in CopulaEnum if e.value.CLASS_NAME == klass.__name__)
    p0, stages, params_of = _two_parameter_spec(klass, entry, 0.4)
    assert len(stages) == 2 and stages[0] is stages[1]     # a restart
    (s_lo, s_hi), (psi_lo, psi_hi) = stages[0][2]
    assert (psi_lo, psi_hi) == (0.01, 1.0)
    assert s_lo == pytest.approx(np.log(1e-6)) and s_hi > np.log(9e3)
    assert p0[1] == pytest.approx(0.7)
    for s in (s_lo, 0.0, s_hi):
        for psi in (psi_lo, 0.3, psi_hi):
            params = params_of((s, psi))
            cop = klass(**params)
            assert cop.params["tau_k"] == params["tau_k"]          # no clamping
            assert cop.theta == pytest.approx(1.0 + np.exp(s), rel=1e-12)


# --------------------------------------------------------------------------
# Construction and the joint constraint τ < ψ
# --------------------------------------------------------------------------

@pytest.mark.parametrize("klass", TYPES)
def test_refused_parameters(klass):
    for kw in ({"tau_k": 0.0}, {"tau_k": -0.1}, {"tau_k": 1.0},
               {"tau_k": 0.5, "psi": 0.5}, {"tau_k": 0.7, "psi": 0.6},
               {"tau_k": 0.3, "psi": 0.0}, {"tau_k": 0.3, "psi": 1.2},
               {"tau_k": 0.3, "psi": float("nan")}, {"tau_k": 0.3, "psi": True}):
        with pytest.raises(CopulaParameterError):
            klass(**kw)


@pytest.mark.parametrize("klass", TYPES)
def test_default_psi_is_the_gumbel_member(klass):
    cop = klass(tau_k=0.6)
    assert cop.psi == 1.0 and cop.params["psi"] == 1.0
    assert cop.theta == CopulaGH(tau_k=0.6).theta


@pytest.mark.parametrize("klass", TYPES)
def test_a_tau_beyond_the_theta_cap_is_clamped_and_stored(klass):
    """For τ(10⁶, ψ) ≤ τ < ψ the copula is built at θ = 10⁶ and stores the τ it
    realises (RB-10); the family-wide reachable bound is the ψ = 1 one."""
    cop = klass(tau_k=0.6 - 1e-9, psi=0.6)
    assert cop.theta == _THETA_HI
    assert cop.params["tau_k"] == _tau_of(_THETA_HI, cop._pu, cop._pv)
    assert 0.6 - 1e-6 < cop.params["tau_k"] < 0.6 - 1e-9
    lo, hi = klass.reachable_tau_bounds()
    assert lo == EPS and hi == pytest.approx(1.0 - 1e-6, abs=1e-15)
    gum = klass(tau_k=0.9999999)
    assert gum.theta == _THETA_HI and gum.params["tau_k"] == hi


@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("psi", [1e-3, 0.05, 0.5, 0.97, 1.0])
def test_tau_theta_round_trip(klass, psi):
    for frac in (1e-10, 1e-4, 0.1, 0.5, 0.9, 0.99):
        tau = frac * psi
        cop = klass(tau_k=tau, psi=psi)
        assert cop.params["tau_k"] == tau
        assert _tawn_tau_quad(cop.theta, cop._pu, cop._pv) == pytest.approx(tau, rel=1e-7)


# --------------------------------------------------------------------------
# The Pickands function (module docstring, "Verified, not transcribed")
# --------------------------------------------------------------------------

_PARAMS = [(th, pu, pv) for th in (1.05, 2.0, 7.0, 60.0)
           for pu, pv in ((1.0, 0.02), (1.0, 0.4), (0.3, 1.0), (0.7, 0.6), (1.0, 1.0))]


@pytest.mark.parametrize("th,pu,pv", _PARAMS)
def test_pickands_bounds_and_convexity(th, pu, pv):
    t = np.linspace(1e-6, 1 - 1e-6, 2001)
    A, _, App = _tawn_A_terms(t, th, pu, pv)
    assert np.all(A <= 1.0 + 1e-15)
    assert np.all(A >= np.maximum(t, 1 - t) - 1e-14)
    assert np.all(App >= 0.0)
    ends, _, _ = _tawn_A_terms(np.array([1e-15, 1 - 1e-15]), th, pu, pv)
    np.testing.assert_allclose(ends, 1.0, atol=1e-13)


@pytest.mark.parametrize("th,pu,pv", _PARAMS)
def test_pickands_derivatives_match_finite_differences(th, pu, pv):
    t = np.array([0.1, 0.25, 0.4, 0.55, 0.8, 0.9])
    h = 1e-6
    A_p, Ap_p, _ = _tawn_A_terms(t + h, th, pu, pv)
    A_m, Ap_m, _ = _tawn_A_terms(t - h, th, pu, pv)
    _, Ap, App = _tawn_A_terms(t, th, pu, pv)
    np.testing.assert_allclose(Ap, (A_p - A_m) / (2 * h), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(App, (Ap_p - Ap_m) / (2 * h), rtol=1e-4, atol=1e-6)


def test_independence_limits():
    """θ = 1 (the norm is linear) and ψ → 0 give A ≡ 1."""
    t = np.linspace(0.01, 0.99, 50)
    A1, _, _ = _tawn_A_terms(t, 1.0 + 1e-12, 1.0, 0.4)
    np.testing.assert_allclose(A1, 1.0, atol=1e-11)
    A0, _, _ = _tawn_A_terms(t, 5.0, 1.0, 1e-12)
    np.testing.assert_allclose(A0, 1.0, atol=1e-11)


# --------------------------------------------------------------------------
# τ(θ, ψ): mpmath references, monotonicity, the cap τ → ψ
# --------------------------------------------------------------------------

# 60-digit mpmath: mp.quad of t(1−t)A''/A with A'' a central second difference
# of A itself (not this module's closed form), split at t* ± k·t*(1−t*)/θ.
_TAU_MPMATH = [
    # (θ, ψ_u, ψ_v, τ)
    (1.000001, 1.0, 0.01, 4.651674740617487e-08),
    (1.000001, 1.0, 0.05, 1.576698076566826e-07),
    (1.000001, 1.0, 0.5, 6.931463580367907e-07),
    (1.000001, 1.0, 0.95, 9.745716061432026e-07),
    (1.000001, 1.0, 1.0, 9.999989999187335e-07),
    (1.2, 1.0, 0.01, 0.005819165136499953),
    (1.2, 1.0, 0.05, 0.022086705414096),
    (1.2, 1.0, 0.5, 0.11168642734821134),
    (1.2, 1.0, 0.95, 0.162048447967036),
    (1.2, 1.0, 1.0, 0.16666666666666663),
    (2.0, 1.0, 0.01, 0.00963114272156024),
    (2.0, 1.0, 0.05, 0.04433315159680335),
    (2.0, 1.0, 0.5, 0.3068528194400547),
    (2.0, 1.0, 0.95, 0.48312072609425744),
    (2.0, 1.0, 1.0, 0.5),
    (5.0, 1.0, 0.01, 0.009966991077419551),
    (5.0, 1.0, 0.05, 0.04920386850273793),
    (5.0, 1.0, 0.5, 0.4392553889064479),
    (5.0, 1.0, 0.95, 0.7664255446197309),
    (5.0, 1.0, 1.0, 0.8),
    (50.0, 1.0, 0.01, 0.009997917552613835),
    (50.0, 1.0, 0.05, 0.04994802712251124),
    (50.0, 1.0, 0.5, 0.49489902078267106),
    (50.0, 1.0, 0.95, 0.931914505363328),
    (50.0, 1.0, 1.0, 0.98),
    (1000.0, 1.0, 0.01, 0.009999899801609176),
    (1000.0, 1.0, 0.05, 0.0499974952411969),
    (1000.0, 1.0, 0.5, 0.49974974987512527),
    (1000.0, 1.0, 0.95, 0.9490974098266601),
    (1000.0, 1.0, 1.0, 0.999),
    (1000000.0, 1.0, 0.01, 0.009999999899999803),
    (1000000.0, 1.0, 0.05, 0.04999999749999525),
    (1000000.0, 1.0, 0.5, 0.49999974999975),
    (1000000.0, 1.0, 0.95, 0.9499990974999097),
    (1000000.0, 1.0, 1.0, 0.999999),
    (3.0, 0.4, 1.0, 0.3207648781476475),
    (1000000.0, 0.05, 1.0, 0.04999999749999525),
]


@pytest.mark.parametrize("th,pu,pv,tau", _TAU_MPMATH)
def test_tau_quadrature_matches_mpmath(th, pu, pv, tau):
    rel = 5e-13 if th <= 50.0 else 2e-10
    assert _tawn_tau_quad(th, pu, pv) == pytest.approx(tau, rel=rel, abs=0.0)


@pytest.mark.parametrize("th", [1.0 + 1e-9, 1.5, 2.0, 5.0, 50.0, 1e4, 1e6])
def test_psi_one_quadrature_reproduces_gumbel(th):
    """The Genest–MacKay quadrature itself (not the ψ = 1 shortcut)."""
    rel = 1e-12 if th <= 50.0 else 2e-11
    assert _tawn_tau_quad(th, 1.0, 1.0) == pytest.approx((th - 1.0) / th, rel=rel)


def test_tau_is_increasing_in_theta_and_psi():
    thetas = [1.0 + 1e-6, 1.01, 1.3, 2.0, 5.0, 30.0, 1e3, 1e6]
    psis = [0.01, 0.05, 0.2, 0.5, 0.8, 0.99, 1.0]
    tab = np.array([[_tau_of(th, 1.0, p) for p in psis] for th in thetas])
    assert np.all(np.diff(tab, axis=0) > 0.0)
    assert np.all(np.diff(tab, axis=1) > 0.0)
    assert np.all(tab < np.array(psis))


@pytest.mark.parametrize("psi", [0.01, 0.2, 0.6, 0.95])
def test_reachable_tau_tends_to_psi(psi):
    """θ → ∞: the Marshall–Olkin limit has τ = ψ exactly (module docstring)."""
    gaps = [psi - _tau_of(th, 1.0, psi) for th in (1e2, 1e4, 1e6)]
    assert all(g > 0.0 for g in gaps)
    assert gaps[2] < 1e-5 and gaps[1] / gaps[2] == pytest.approx(100.0, rel=0.05)
    # the general Marshall–Olkin formula, both weights < 1
    assert _tau_of(1e6, 0.3, 0.6) == pytest.approx(0.18 / (0.9 - 0.18), abs=1e-5)


def test_the_two_types_share_tau():
    for th, psi in ((1.3, 0.2), (3.0, 0.4), (40.0, 0.9)):
        assert _tau_of(th, psi, 1.0) == pytest.approx(_tau_of(th, 1.0, psi), rel=1e-12)


# --------------------------------------------------------------------------
# ψ = 1 is Gumbel–Hougaard; type 2 is the transpose of type 1
# --------------------------------------------------------------------------

@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau", [1e-8, 0.1, 0.5, 0.9, 0.99])
def test_psi_one_is_gumbel_to_machine_precision(klass, tau):
    t, g = klass(tau_k=tau, psi=1.0), CopulaGH(tau_k=tau)
    assert t.theta == g.theta and t.params["tau_k"] == g.params["tau_k"]
    # Two different closed forms of one density, equal to rounding. Against a
    # 60-digit mpmath value on this grid Tawn's is within 8e-15 nat, GH's within
    # 1.7e-14 for τ ≤ 0.5 but 8.6e-13 at τ = 0.99 on the diagonal (its
    # (1 − 2θ)·ln A term amplifies the rounding of ln A by 2θ − 1 = 199), where
    # Tawn's value is the correctly rounded one.
    ok = np.isfinite(g.logpdf_array(_UV))
    np.testing.assert_allclose(t.logpdf_array(_UV)[ok], g.logpdf_array(_UV)[ok],
                               rtol=3e-14, atol=4e-14)
    assert np.array_equal(t.cdf_array(_UV), g.cdf_array(_UV))
    ht = np.array([t.conditional_cdf(v, u) for u, v in _UV])
    hg = np.array([g.conditional_cdf(v, u) for u, v in _UV])
    np.testing.assert_allclose(ht, hg, rtol=2e-13, atol=1e-300)
    np.testing.assert_allclose(t.tail_dependence(), g.tail_dependence(), rtol=1e-15, atol=0)


@pytest.mark.parametrize("tau,psi", [(1e-6, 0.01), (0.2, 0.3), (0.5, 0.9), (0.9, 0.95)])
def test_type2_is_the_transpose_of_type1(tau, psi):
    c1, c2 = CopulaTawn1(tau_k=tau, psi=psi), CopulaTawn2(tau_k=tau, psi=psi)
    assert c2.theta == pytest.approx(c1.theta, rel=1e-10)
    swapped = _UV[:, ::-1]
    # at one and the same θ, the kernels are exact transposes
    np.testing.assert_allclose(_tawn_logpdf(np.log(_UV[:, 0]), np.log(_UV[:, 1]), c1.theta, 1.0, psi),
                               _tawn_logpdf(np.log(swapped[:, 0]), np.log(swapped[:, 1]), c1.theta, psi, 1.0),
                               rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(c2.cdf_array(_UV), c1.cdf_array(swapped), rtol=1e-9, atol=1e-300)
    assert c1.tail_dependence() == pytest.approx(c2.tail_dependence(), rel=1e-10)


@pytest.mark.parametrize("klass", TYPES)
def test_tail_dependence_matches_the_pickands_identity(klass):
    for tau, psi in ((0.05, 0.1), (0.3, 0.5), (0.7, 0.8), (0.5, 1.0)):
        cop = klass(tau_k=tau, psi=psi)
        A_half, _, _ = _tawn_A_terms(0.5, cop.theta, cop._pu, cop._pv)
        lam_l, lam_u = cop.tail_dependence()
        assert lam_l == 0.0
        assert lam_u == pytest.approx(2.0 * (1.0 - float(A_half)), rel=1e-12)
    # θ → ∞: λ_U → ψ (the Marshall–Olkin value)
    assert klass(tau_k=0.4 - 1e-12, psi=0.4).tail_dependence()[1] == pytest.approx(0.4, abs=1e-6)


# --------------------------------------------------------------------------
# pdf / cdf / h / h⁻¹ — mpmath spot values and copula properties
# --------------------------------------------------------------------------

# (class, τ, ψ, u, v, C, h(v|u), ln c): independent mpmath values (C from the
# definition, h and c by high-precision central differences in (−ln u, −ln v)),
# at the θ each copula builds.
_POINT_MPMATH = [
    ('CopulaTawn1', 0.3, 0.35, 0.2, 0.7, 0.15860696831118137, 0.7929723502309204, -0.3041043132849931),
    ('CopulaTawn1', 0.3, 0.35, 0.7, 0.2, 0.19428835222490165, 0.07550259226037832, -0.14498843801447694),
    ('CopulaTawn2', 0.3, 0.35, 0.2, 0.7, 0.19428835222490168, 0.9303665040922993, -0.1449884380144778),
    ('CopulaTawn1', 0.95, 0.97, 1e-12, 1e-06, 6.606934480075951e-13, 0.6606934480075941, 9.894487343905643),
    ('CopulaTawn2', 0.95, 0.97, 0.999999, 0.5, 0.4999999849999927, 0.015000014550014346, -3.5065569273194956),
    ('CopulaTawn1', 0.0001, 0.05, 1e-12, 0.999999999999, 9.999999999990011e-13, 0.9999999999990011, -0.0010657409409218553),
    ('CopulaTawn2', 1e-08, 0.01, 0.3, 0.9, 0.2700000022531962, 0.9000000031044246, -1.499878218671285e-09),
    ('CopulaTawn1', 0.6, 0.9, 0.9, 0.95, 0.8923405457846146, 0.9346436375296421, 1.213273320064978),
    ('CopulaTawn2', 0.6, 0.9, 1e-06, 1e-06, 1.2934636227191984e-08, 0.0079788374883695, 8.713124680614936),
    ('CopulaTawn1', 0.05, 0.999, 0.01, 0.99, 0.009931068690705595, 0.9930284945588849, -0.30802393329654376),
]


@pytest.mark.parametrize("name,tau,psi,u,v,C,h,logc", _POINT_MPMATH)
def test_kernel_matches_mpmath(name, tau, psi, u, v, C, h, logc):
    cop = {"CopulaTawn1": CopulaTawn1, "CopulaTawn2": CopulaTawn2}[name](tau_k=tau, psi=psi)
    assert cop.cdf([u, v]) == pytest.approx(C, rel=1e-12)
    assert cop.conditional_cdf(v, u) == pytest.approx(h, rel=1e-11)
    assert float(cop.logpdf_array(np.array([[u, v]]))[0]) == pytest.approx(logc, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau,psi", [(1e-4, 0.05), (0.3, 0.35), (0.6, 0.9), (0.95, 0.97)])
def test_h_inv_h_round_trip_and_saturation(klass, tau, psi):
    cop = klass(tau_k=tau, psi=psi)
    rng = np.random.default_rng(20260916)
    u = rng.uniform(1e-6, 1 - 1e-6, 300)
    v = rng.uniform(1e-6, 1 - 1e-6, 300)
    w = np.array([cop.conditional_cdf(b, a) for a, b in zip(u, v)])
    v_back = cop.inv_h_array(w, u)
    w_back = np.array([cop.conditional_cdf(b, a) for a, b in zip(u, v_back)])
    np.testing.assert_allclose(w_back, w, rtol=1e-9, atol=1e-12)
    assert cop.inv_h(0.0, 0.3) == EPS and cop.inv_h(1.0, 0.3) == 1.0 - EPS
    assert cop.inv_h(float(w[0]), float(u[0])) == v_back[0]


@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau,psi", [(1e-4, 0.05), (0.3, 0.35), (0.6, 0.9)])
def test_cdf_margins_monotonicity_and_density_normalisation(klass, tau, psi):
    cop = klass(tau_k=tau, psi=psi)
    for x in (0.2, 0.5, 0.8):
        assert cop.cdf([x, 1.0 - EPS]) == pytest.approx(x, abs=1e-12)
        assert cop.cdf([1.0 - EPS, x]) == pytest.approx(x, abs=1e-12)
    g = np.linspace(0.02, 0.98, 25)
    grid = np.array([(a, b) for a in g for b in g])
    C = cop.cdf_array(grid).reshape(25, 25)
    assert np.all(np.diff(C, axis=0) >= -1e-15) and np.all(np.diff(C, axis=1) >= -1e-15)
    # ∫∫ c over [0.02, 0.98]² equals the rectangle's C-volume (midpoint rule, 400²)
    m = (np.arange(400) + 0.5) / 400 * 0.96 + 0.02
    mm = np.array(np.meshgrid(m, m)).reshape(2, -1).T
    vol = cop.pdf_array(mm).sum() * (0.96 / 400) ** 2
    rect = cop.cdf([0.98, 0.98]) - cop.cdf([0.02, 0.98]) - cop.cdf([0.98, 0.02]) + cop.cdf([0.02, 0.02])
    assert vol == pytest.approx(rect, rel=2e-3)


def test_asymmetry_direction():
    """Type 1 puts less mass below-left of the diagonal on the u < v side:
    C₁(u, v) ≤ C₁(v, u) for u < v (θ → ∞: ℓ(w, z) − ℓ(z, w) = (1 − ψ) z > 0
    when ψw > z), with equality only at θ = 1 or ψ = 1; type 2 the reverse."""
    g = np.linspace(0.02, 0.98, 30)
    lower = np.array([(a, b) for a in g for b in g if a < b])
    for tau, psi in ((0.05, 0.1), (0.3, 0.35), (0.5, 0.8), (0.8, 0.9)):
        d1 = CopulaTawn1(tau_k=tau, psi=psi).cdf_array(lower) \
            - CopulaTawn1(tau_k=tau, psi=psi).cdf_array(lower[:, ::-1])
        d2 = CopulaTawn2(tau_k=tau, psi=psi).cdf_array(lower) \
            - CopulaTawn2(tau_k=tau, psi=psi).cdf_array(lower[:, ::-1])
        assert np.all(d1 <= 1e-15) and d1.min() < -1e-3
        np.testing.assert_allclose(d2, -d1, atol=1e-9)


# --------------------------------------------------------------------------
# Sampling: margins, realised τ against the quadrature, asymmetry
# --------------------------------------------------------------------------

@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau,psi", [(0.1, 0.2), (0.3, 0.35), (0.5, 0.9), (0.7, 0.75)])
def test_sample_margins_and_monte_carlo_tau(klass, tau, psi):
    uv = klass(tau_k=tau, psi=psi).sample(n=20000, seed=4242)
    assert kstest(uv[:, 0], "uniform").pvalue > 1e-3
    assert kstest(uv[:, 1], "uniform").pvalue > 1e-3
    # s.e. of Kendall's τ at n = 20000 is ≤ 0.005 here; 5 s.e.
    assert kendalltau(uv[:, 0], uv[:, 1]).statistic == pytest.approx(tau, abs=0.025)


def test_exchangeability_test_rejects_on_asymmetric_samples():
    from pmcprg.diagnostics.exchangeability import exchangeability_test
    for klass, seed in ((CopulaTawn1, 101), (CopulaTawn2, 202)):
        uv = klass(tau_k=0.3, psi=0.35).sample(n=500, seed=seed)
        res = exchangeability_test(uv[:, 0], uv[:, 1], B=100, seed=seed)
        assert res.reject and res.p_value < 0.02


# --------------------------------------------------------------------------
# fit: both parameters
# --------------------------------------------------------------------------

@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau,psi,tol_psi", [(0.2, 0.3, 0.12), (0.4, 0.6, 0.12),
                                             (0.5, 1.0, 0.08), (0.7, 0.8, 0.06)])
def test_fit_recovers_tau_and_psi(klass, tau, psi, tol_psi):
    uv = klass(tau_k=tau, psi=psi).sample(n=3000, seed=31337)
    r = klass.fit(uv)
    assert r.converged and r.method == "mle"
    assert r.tau_k == pytest.approx(tau, abs=0.04)
    assert r.copula.psi == pytest.approx(psi, abs=tol_psi)
    assert r.copula.params["tau_k"] == r.tau_k


def test_fit_tau_method_falls_back_to_mle(caplog):
    uv = CopulaTawn1(tau_k=0.4, psi=0.6).sample(n=800, seed=7)
    with caplog.at_level(logging.WARNING):
        r_tau = CopulaTawn1.fit(uv, method="tau")
    assert "falling back to MLE" in caplog.text
    r_mle = CopulaTawn1.fit(uv, method="mle")
    assert r_tau.tau_k == r_mle.tau_k and r_tau.copula.psi == r_mle.copula.psi


def test_ice_m_step_fits_psi():
    """The weighted M-step goes through the same joint optimiser as ``fit``
    (which works on rank pseudo-observations: fed the same ones, both agree)."""
    from scipy.stats import rankdata

    from pmcprg.pmc.ice import FIT_FAILED_KEY, _fit_copula_params, _resolve_candidate
    uv = CopulaTawn2(tau_k=0.5, psi=0.7).sample(n=1500, seed=11)
    pobs = np.column_stack([rankdata(uv[:, 0]), rankdata(uv[:, 1])]) / (len(uv) + 1)
    entry, cls = _resolve_candidate("Tawn2")
    p = _fit_copula_params(cls, entry, pobs[:, 0], pobs[:, 1], np.ones(len(uv)))
    assert FIT_FAILED_KEY not in p
    assert set(p) == {"tau_k", "psi"}
    assert p["tau_k"] == pytest.approx(0.5, abs=0.05) and p["psi"] == pytest.approx(0.7, abs=0.12)
    r = CopulaTawn2.fit(uv)
    assert p["tau_k"] == pytest.approx(r.tau_k, rel=1e-6)
    assert p["psi"] == pytest.approx(r.copula.psi, rel=1e-6)
    # doubled weights: the same optimum (the objective is the total weighted LL)
    p2 = _fit_copula_params(cls, entry, pobs[:, 0], pobs[:, 1], np.full(len(uv), 2.0))
    assert p2["psi"] == pytest.approx(p["psi"], rel=1e-5)


@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau,psi,seed", [(0.35, 0.45, 77), (0.2, 0.9, 3), (0.8, 0.95, 4)])
def test_joint_mle_does_not_depend_on_its_start(klass, tau, psi, seed):
    """Contrived starts far from the data's τ reach the same optimum (a single
    L-BFGS-B pass stopped 131 nat short from τ_start = 0.9 on the first data
    set, 1.1 nat on the second — hence the restart stage)."""
    from pmcprg.copulas._fit import _fit_two_parameter_mle
    entry = next(e for e in CopulaEnum if e.value.CLASS_NAME == klass.__name__)
    uv = klass(tau_k=tau, psi=psi).sample(n=1500, seed=seed)
    fits = [_fit_two_parameter_mle(klass, entry, uv, None, t0) for t0 in (0.02, 0.35, 0.9, 0.95)]
    assert all(f.converged for f in fits)
    best = max(f.log_likelihood for f in fits)
    for f in fits:
        assert f.log_likelihood == pytest.approx(best, abs=1e-5)
        assert f.params["psi"] == pytest.approx(fits[0].params["psi"], abs=2e-4)


def test_fit_best_includes_tawn():
    from pmcprg.copulas._base import CopulaVirt
    uv = CopulaTawn1(tau_k=0.4, psi=0.5).sample(n=800, seed=3)
    res = CopulaVirt.fit_best(uv, families=[CopulaTawn1, CopulaTawn2, CopulaGH], method="mle")
    assert res[0].copula.__class__ is CopulaTawn1


# --------------------------------------------------------------------------
# Joint-constraint repair (mirrors test_multistart_joint_constraints.py)
# --------------------------------------------------------------------------

def _refused(klass, tau, psi) -> bool:
    try:
        klass(tau_k=tau, psi=psi)
    except CopulaParameterError:
        return True
    return False


@pytest.mark.parametrize("klass", TYPES)
def test_accepted_pairs_are_returned_untouched(klass):
    rng = np.random.default_rng(5)
    n_accepted = 0
    for tau in np.concatenate([[EPS, 1e-9, 0.1, 0.5, 0.9, 0.9999], rng.uniform(0, 1, 15)]):
        for psi in (0.01, 0.3, 1.0, float(tau), float(np.nextafter(tau, 2.0)), float(rng.uniform())):
            params = {"tau_k": float(tau), "psi": psi}
            out = klass.constructible_params(params)
            assert (out is params) == (not _refused(klass, float(tau), psi))
            n_accepted += out is params
    assert n_accepted > 40
    params = {"tau_k": 0.99}
    assert klass.constructible_params(params) is params


@pytest.mark.parametrize("klass", TYPES)
@pytest.mark.parametrize("tau", [EPS, 1e-6, 0.2, 0.5, 0.9, 0.999, 0.9999, 1.0 - 1.1e-6])
@pytest.mark.parametrize("psi", [-0.5, 0.0, 1e-9, 0.1, 1.5, float("inf")])
def test_refused_pairs_move_only_psi(klass, tau, psi):
    params = {"tau_k": tau, "psi": psi}
    if not _refused(klass, tau, psi):
        assert klass.constructible_params(params) is params
        return
    out = klass.constructible_params(params)
    assert out is not params and out["tau_k"] == tau
    assert out["psi"] == klass.repaired_psi(tau)
    cop = klass(**out)
    assert cop.params["tau_k"] == tau                  # built below the θ cap
    assert tau < out["psi"] <= 1.0
    if out["psi"] < 1.0:
        assert out["psi"] == 0.5 * (1.0 + tau)


@pytest.mark.parametrize("klass", TYPES)
def test_constrain_params_projects_into_the_box_and_the_admissible_set(klass):
    assert klass.constrain_params({"tau_k": 0.5, "psi": 5.0}) == {"tau_k": 0.5, "psi": 1.0}
    assert klass.constrain_params({"tau_k": 0.5, "psi": 0.2}) == {"tau_k": 0.5, "psi": 0.75}
    assert klass.constrain_params({"tau_k": 0.001, "psi": 0.0}) == {"tau_k": 0.001, "psi": 0.01}


MODELS = "pmcprg/pmc/models"


def _tawn_model(name="Tawn1", tau=0.6, psi=0.65):
    from pmcprg.pmc.model import PMCModel
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    for blk in raw["copulas"]:
        blk.update({"name": name, "tau": tau, "psi": psi})
    return PMCModel.from_dict(raw)


@pytest.mark.parametrize("name", ["Tawn1", "Tawn2"])
@pytest.mark.parametrize("jitter", [0.1, 0.25, 0.5])
def test_jittered_starts_always_build(name, jitter):
    from pmcprg.pmc._estim_common import perturb_initial_model
    klass = CopulaEnum.from_short_name(name).klass
    model = _tawn_model(name)
    repaired = 0
    for seed in range(60):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=jitter)
        for blk in out.copula_blocks():
            assert not _refused(klass, blk["tau"], blk["psi"])
            repaired += blk["psi"] == klass.repaired_psi(blk["tau"])
    assert repaired > 0


@pytest.mark.parametrize("mode", ["random", "sweep"])
@pytest.mark.parametrize("cands", [["Gauss", "Tawn1"], ["Tawn1", "Tawn2", "BB1", "Clayton"]])
def test_family_starts_with_tawn_candidates_build(mode, cands):
    from pmcprg.pmc._estim_common import build_multistart_inits
    from pmcprg.pmc.model import PMCModel
    for model in (PMCModel(f"{MODELS}/pmc_gauss_k2.toml"), _tawn_model("Tawn2", 0.8, 0.85)):
        for seed in range(3):
            cfg = {"n_starts": 1 + len(cands) ** 2 + 3, "multistart_seed": seed,
                   "multistart_jitter": 0.25, "multistart_families": mode, "candidates": cands}
            for m, _ in build_multistart_inits(model, cfg, label="ICE"):
                for blk in m.copula_blocks():
                    if blk["name"].startswith("Tawn"):
                        klass = CopulaEnum.from_short_name(blk["name"]).klass
                        assert not _refused(klass, blk["tau"], blk.get("psi", 1.0))


@pytest.mark.parametrize("candidates", [["Tawn1", "Gauss"], ["Tawn2", "BB1"]])
def test_selection_placeholder_builds(candidates):
    from pmcprg.pmc.ice import FIT_FAILED_KEY, _copula_candidate_fits, _copula_placeholder
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=50), rng.uniform(size=50)
    resolved, _ = _copula_candidate_fits(candidates, u, v, np.ones(50), "mle")
    blk = _copula_placeholder(candidates, resolved, "mle", {"name": candidates[0], "tau": 0.0})
    assert blk[FIT_FAILED_KEY] is True
    stored = {k: val for k, val in blk.items() if k != FIT_FAILED_KEY}
    CopulaEnum.from_short_name(candidates[0]).klass(tau_k=stored["tau"], psi=stored.get("psi", 1.0))


def test_theta_inversion_refuses_non_positive_tau():
    with pytest.raises(CopulaParameterError):
        _theta_of(0.0, 1.0, 0.5, "Tawn1")


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_tawn_cases_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _INDEP_TAUS, _TAIL_TAUS, _file_table, _match_file_key, _required_param_keys,
    )
    table = _file_table()
    for short in ("Tawn1", "Tawn2"):
        assert short in _TAIL_TAUS and short in _INDEP_TAUS
        need = {k for k in _required_param_keys() if k[0] == short}
        assert len(need) == 6
        assert {k[2] for k in need} == {"0.001", "0.05", "0.5", "0.35", "0.9", "0.97"}
        for key in need:
            assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
