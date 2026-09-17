"""Numerical fidelity of every copula family's evaluation paths.

Audit K-1..K-3 found three families whose *vectorised* paths — the ones ICE,
the Huard evidence and the GoF reference consume — silently disagreed with
their scalar paths in strong dependence: an ``EPS`` floor on an intermediate
that is legitimately far below ``EPS`` (Joe's ``W``, A14's ``U_i``), and the
textbook cancellation ``e^{−θ} − 1`` in Frank. None of the existing tests
looked at τ ≥ 0.8, so the errors (up to 290 nat on log c) went unnoticed.

Every family is therefore checked at strong dependence, on both sides where
the family allows it, for

* array paths == scalar paths (``logpdf_array``/``pdf_array``/``cdf_array``);
* density == mixed finite difference of the CDF (independent of the density
  formula);
* h(v|u) == ∂C/∂u by finite difference, and ``inv_h ∘ h = id``;

plus regression pins on the exact numbers the audit reported.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum
from pmcprg.numerics import EPS

# --------------------------------------------------------------------------
# Cases: every available family at strong dependence (both signs where the
# family allows it) plus one moderate value.
# --------------------------------------------------------------------------

_CANDIDATE_TAUS = (0.95, 0.9, 0.8, 0.5, 0.15, -0.15, -0.5, -0.8, -0.9, -0.95)


def _build(entry: CopulaEnum, tau: float):
    kwargs: dict = {"tau_k": float(tau)}
    if "delta" in entry.value.PARAMETERS_SET_NAME:
        kwargs["delta"] = 1.5          # BB1: effective lower τ-bound ≈ 1/3
    if "df" in entry.value.PARAMETERS_SET_NAME:
        kwargs["df"] = 4.0
    # Any other extra keeps its constructor default (Tawn's ψ = 1, t-EV's ν).
    # A family whose parameters are *jointly* constrained may still refuse the
    # pair — Tawn 3 does not at ψ_u = ψ_v = 1 (its cap is then 1), but that is
    # a property of its defaults, not something this sweep should rely on — so
    # route through the family's own repair hook, which is the identity at
    # every triple the constructor accepts (FR-9, round 5).
    return entry.klass(**entry.klass.constructible_params(kwargs))


def _taus(entry: CopulaEnum) -> list[float]:
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    if tau_max - tau_min < 1e-9:                       # Product
        return [0.0]
    lo = tau_min + 1e-3
    if "delta" in entry.value.PARAMETERS_SET_NAME:
        lo = max(lo, 1.0 / 3.0 + 1e-3)                 # BB1 at δ = 1.5
    return [t for t in _CANDIDATE_TAUS if lo <= t <= tau_max - 1e-3]


CASES = [
    pytest.param(entry, tau, id=f"{entry.value.SHORT_NAME}-tau{tau:+.2f}")
    for entry in CopulaEnum if entry.value.AVAILABLE
    for tau in _taus(entry)
]

# Interior grid plus a ring close to the edges — the corners are where the
# floors used to bite.
_G = np.concatenate(([0.005, 0.02], np.linspace(0.08, 0.92, 8), [0.98, 0.995]))
_U, _V = (a.ravel() for a in np.meshgrid(_G, _G))
_UV = np.column_stack((_U, _V))


def _has_cdf(cop) -> bool:
    try:
        cop.cdf([0.3, 0.7])
    except NotImplementedError:                        # Student
        return False
    return True


# --------------------------------------------------------------------------
# Array paths against scalar paths
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry,tau", CASES)
def test_logpdf_array_matches_scalar_pdf(entry, tau):
    cop = _build(entry, tau)
    scalar = np.array([cop.pdf([u, v]) for u, v in _UV])
    logc   = cop.logpdf_array(_UV)
    # A scalar 0.0 is acceptable only as a genuine underflow of exp: the log
    # path must agree that the density is below the double-precision floor.
    underflow = scalar == 0.0
    assert np.all(logc[underflow] < -700.0), "scalar pdf returned 0.0 where log c is representable"
    fallback = scalar == EPS                            # the scalar paths' fallback sentinel
    assert fallback.mean() <= 0.1, "scalar pdf falls back on more than 10 % of the grid"
    ok = ~underflow & ~fallback
    np.testing.assert_allclose(logc[ok], np.log(scalar[ok]), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(cop.pdf_array(_UV)[ok], scalar[ok], rtol=1e-7)


@pytest.mark.parametrize("entry,tau", CASES)
def test_cdf_array_matches_scalar_cdf(entry, tau):
    cop = _build(entry, tau)
    if not _has_cdf(cop):
        pytest.skip("no closed-form CDF")
    scalar = np.array([cop.cdf([u, v]) for u, v in _UV])
    np.testing.assert_allclose(cop.cdf_array(_UV), scalar, atol=1e-9)


# --------------------------------------------------------------------------
# Density and h-function against derivatives of the CDF (formula-independent)
# --------------------------------------------------------------------------

_GI = np.linspace(0.06, 0.94, 9)
_UI, _VI = (a.ravel() for a in np.meshgrid(_GI, _GI))


@pytest.mark.parametrize("entry,tau", CASES)
def test_density_is_mixed_derivative_of_cdf(entry, tau):
    cop = _build(entry, tau)
    if not _has_cdf(cop):
        pytest.skip("no closed-form CDF")
    d = 1e-4
    C = lambda a, b: cop.cdf_array(np.column_stack((a, b)))   # noqa: E731
    mixed = (C(_UI + d, _VI + d) - C(_UI + d, _VI - d)
             - C(_UI - d, _VI + d) + C(_UI - d, _VI - d)) / (4 * d * d)
    dens = cop.pdf_array(np.column_stack((_UI, _VI)))
    sel = dens > 1e-2                                  # where the FD is resolvable
    assert sel.sum() > 10
    np.testing.assert_allclose(dens[sel], mixed[sel], rtol=2e-2)


@pytest.mark.parametrize("entry,tau", CASES)
def test_h_is_derivative_of_cdf(entry, tau):
    cop = _build(entry, tau)
    if not _has_cdf(cop):
        pytest.skip("no closed-form CDF")
    d = 1e-6
    for u, v in zip(_UI, _VI):
        fd = (cop.cdf([u + d, v]) - cop.cdf([u - d, v])) / (2 * d)
        assert abs(cop.conditional_cdf(v, u) - fd) < 2e-5, (u, v)


@pytest.mark.parametrize("entry,tau", CASES)
def test_inv_h_roundtrip(entry, tau):
    cop = _build(entry, tau)
    n_checked = 0
    for u, v in zip(_UI, _VI):
        w = cop.conditional_cdf(v, u)
        if 1e-6 < w < 1 - 1e-6:                        # not saturated in double
            assert abs(cop.inv_h(w, u) - v) < 1e-5, (u, v, w)
            n_checked += 1
    assert n_checked > 10


# --------------------------------------------------------------------------
# Regression pins — the numbers the audit reported (K-1, K-2, K-3)
# --------------------------------------------------------------------------

def test_joe_strong_dependence_arrays_no_longer_flattened():
    """K-1: cdf_array used to return 0.606 on the whole upper corner."""
    cop = CopulaEnum.JOE.klass(tau_k=0.95)
    pts = np.array([[0.90, 0.90], [0.95, 0.97], [0.99, 0.985]])
    np.testing.assert_allclose(cop.cdf_array(pts), [0.898194, 0.950000, 0.985000], atol=2e-6)
    np.testing.assert_allclose(cop.logpdf_array(pts), [4.564, -12.645, -7.466], atol=2e-3)


def test_a14_strong_dependence_logpdf_and_tails():
    """K-2: logpdf_array was off by 0.77 nat at the centre, ~100 nat in the corners."""
    cop = CopulaEnum.A14.klass(tau_k=0.95)
    pts = np.array([[0.5, 0.5], [0.02, 0.98], [0.008, 0.992]])
    np.testing.assert_allclose(cop.logpdf_array(pts), [2.658, -97.642, -119.269], atol=2e-3)
    for tau in (0.6, 0.9, 0.95, 0.99):
        c = CopulaEnum.A14.klass(tau_k=tau)
        lam_l, lam_u = c.tail_dependence()
        assert lam_l == 0.5
        assert lam_u == pytest.approx(2.0 - 2.0 ** (1.0 / c.theta))
    # The limit itself, where it is numerically reachable (θ = 2 → u^{1/θ} = 10⁻³).
    c = CopulaEnum.A14.klass(tau_k=0.6)
    assert c.cdf([1e-6, 1e-6]) / 1e-6 == pytest.approx(0.5, abs=2e-3)


def test_frank_strong_dependence_scalar_paths():
    """K-3: the scalar pdf returned EPS everywhere at τ = 0.9; cdf and h broke in the corner."""
    cop = CopulaEnum.FRANK.klass(tau_k=0.9)
    assert cop.pdf([0.5, 0.5]) == pytest.approx(9.57, abs=0.01)
    assert cop.pdf([0.9, 0.95]) == pytest.approx(4.455, abs=0.01)
    assert cop.cdf([0.98, 0.99]) == pytest.approx(0.974872, abs=1e-5)
    assert cop.conditional_cdf(0.99, 0.98) == pytest.approx(0.821762, abs=1e-5)
    assert cop.conditional_cdf(0.95, 0.90) == pytest.approx(0.888313, abs=1e-5)


@pytest.mark.parametrize("tau", [0.9, 0.95, 0.994, -0.9, -0.994])
def test_frank_inverse_h_roundtrip_strong_dependence(tau):
    """K-3: inv_h (which ``simulate`` uses) was off by 1e-2 at τ = 0.9, degenerate at 0.95."""
    cop = CopulaEnum.FRANK.klass(tau_k=tau)
    for u in (0.05, 0.5, 0.95):
        for v in (0.05, 0.3, 0.7, 0.95):
            w = cop.conditional_cdf(v, u)
            if 1e-9 < w < 1 - 1e-9:
                assert abs(cop.inv_h(w, u) - v) < 1e-7, (u, v, w)
    w = np.array([0.05, 0.3, 0.7, 0.95])
    u = np.full_like(w, 0.5)
    v = cop.inv_h_array(w, u)
    hv = np.array([cop.conditional_cdf(vi, ui) for vi, ui in zip(v, u)])
    np.testing.assert_allclose(hv, w, atol=1e-7)


@pytest.mark.parametrize("tau", [0.5, 0.9, 0.994])
def test_frank_negative_theta_is_the_rotation(tau):
    """C_{−θ}(u, v) = u − C_θ(u, 1 − v) and c_{−θ}(u, v) = c_θ(u, 1 − v)."""
    pos = CopulaEnum.FRANK.klass(tau_k=tau)
    neg = CopulaEnum.FRANK.klass(tau_k=-tau)
    assert neg.theta == pytest.approx(-pos.theta)
    rot = np.column_stack((_UV[:, 0], 1.0 - _UV[:, 1]))
    np.testing.assert_allclose(neg.cdf_array(_UV), _UV[:, 0] - pos.cdf_array(rot), atol=1e-12)
    np.testing.assert_allclose(neg.logpdf_array(_UV), pos.logpdf_array(rot), atol=1e-10)


@pytest.mark.parametrize("tau", [0.9, 0.994, -0.994])
def test_frank_density_integrates_to_one_in_strong_dependence(tau):
    cop = CopulaEnum.FRANK.klass(tau_k=tau)
    g = (np.arange(600) + 0.5) / 600
    u, v = (a.ravel() for a in np.meshgrid(g, g))
    mass = float(np.mean(cop.pdf_array(np.column_stack((u, v)))))
    assert math.isfinite(mass)
    assert mass == pytest.approx(1.0, abs=3e-3)       # midpoint rule on a sharp ridge
