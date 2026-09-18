"""BB8 copula (FR-9) — generator, sub-models, τ quadrature, tails, fit.

Complements ``test_copula_limits.py`` (which adds BB8 to its high-precision
``decimal`` reference grid at four τ, its δ carried in the file's ``delta``
slot, one of the four being the Joe sub-model exactly) with the checks
specific to this family:

* the **τ quadrature** against a 60-digit ``mpmath`` Genest–MacKay reference
  computed in the same coordinate but with independent breakpoints, and
  against a simulated Kendall's τ. BB8's τ has no elementary closed form — it
  has a ``₃F₂`` one, verified in the module docstring and unusable
  numerically — so this table is the only thing standing between the family
  and a silently wrong dependence scale;
* C, h and ln c against an adaptive-precision ``mpmath`` ground truth (C from
  the generator alone, h and c as its finite differences at a step 10⁻²⁰ of
  the distance to the edge, the precision raised from 480 to 7680 digits until
  **three** successive ones agree to 25 digits — ``test_copula_limits``'s
  recipe, hardened: the usual "two successive agree to 12 digits" stopped on
  two values that were *both* wrong at the hardest corner of this family, and
  the double-precision kernel was right where the reference was not, see the
  module docstring of ``bb8.py``);
* **δ = 1 ≡ Joe** to machine precision, and **θ = 1 ≡ independence** — the
  second being the negative result of this round: the "Joe–Frank" nickname
  does *not* mean that θ = 1 is Frank. Frank is only the joint limit
  θ → ∞, δ → 0 at fixed θδ, tested here as a rate, not as a member;
* λ_U = 0 for every δ < 1, 2 − 2^{1/θ} at δ = 1, λ_L = 0 — the family's
  discontinuity at δ = 1, on the diagonal;
* the **absence** of a joint (τ, δ) constraint, unlike BB1/BB6/Tawn: every
  registered τ builds at every δ;
* that ``delta8`` takes **its own** branch of ``_two_parameter_spec`` and
  neither BB1's ``delta`` nor BB6's ``delta6`` branch, which carry those
  families' τ maps;
* recovery of τ by ``fit``, the ``method='tau'`` fallback, and the ICE
  integration path.

Every seed below is an integer literal or ``zlib.crc32`` of a repr — never
``hash()``, which is salted per process (commit 5e56fda).
"""
from __future__ import annotations

import logging
import math
import zlib

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import (
    CopulaBB1,
    CopulaBB6,
    CopulaBB8,
    CopulaEnum,
    CopulaFrank,
    CopulaJoe,
)
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.copulas.archimedean.bb8 import (
    _DELTA_DEFAULT,
    _THETA_HI,
    _bb8_log_p,
    _bb8_tau_quad,
    _tau_of,
)
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_G = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12])
_UV = np.array([(u, v) for u in _G for v in _G])

# (τ, δ) points used throughout: interior, the Joe end δ = 1, and δ small
# enough that θ runs into the thousands.
_CASES = [(0.2, 1.0), (0.3, 0.5), (0.4, 0.2), (0.5, 0.05),
          (0.7, 1.0), (0.7, 0.6), (0.9, 0.3), (0.95, 0.01)]


def _seed(*parts: object) -> int:
    """A reproducible seed from ``parts`` — ``zlib.crc32``, never ``hash()``."""
    return zlib.crc32(repr(parts).encode())


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    entry = CopulaEnum.BB8
    assert entry.value.SHORT_NAME == "BB8"
    assert entry.value.CLASS_NAME == "CopulaBB8"
    assert entry.klass is CopulaBB8
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "delta8"]
    assert entry.value.TAU_MIN_MAX == [EPS, 1.0]
    assert entry in CopulaEnum.available()
    assert CopulaBB8.n_params == 2
    assert CopulaEnum.from_short_name("BB8") is entry


def test_delta8_bounds_are_registered_and_reuse_no_other_familys():
    from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
    assert EXTRA_PARAM_BOUNDS_BY_PARAM["delta8"] == (0.01, 1.0, 1.0)
    assert EXTRA_PARAM_BOUNDS["CopulaBB8"] == {"delta8": (0.01, 1.0, 1.0)}
    # BB1's and BB6's own entries are untouched, and BB8 appears under neither.
    assert EXTRA_PARAM_BOUNDS["CopulaBB1"] == {"delta": (1.0, 10.0, 1.5)}
    assert EXTRA_PARAM_BOUNDS["CopulaBB6"] == {"delta6": (1.0, 10.0, 1.0)}
    names = CopulaEnum.BB8.value.PARAMETERS_SET_NAME
    assert "delta" not in names and "delta6" not in names


def test_two_parameter_spec_takes_its_own_branch_not_bb1s_or_bb6s():
    """The silent-bug guard: ``_two_parameter_spec`` dispatches on the
    parameter *name*, so a BB8 registered under ``delta`` or ``delta6`` would
    be optimised with another family's τ map — every (θ, δ) pair being
    numerically plausible, nothing would ever complain."""
    from pmcprg.copulas._fit import _two_parameter_spec

    p0, stages, params_of = _two_parameter_spec(CopulaBB8, CopulaEnum.BB8, 0.6)
    assert set(params_of(p0)) == {"tau_k", "delta8"}
    assert len(stages) == 2
    # Box = the admissible set: s = ln(θ − 1) and δ ∈ [0.01, 1], every point
    # buildable, with *no* projection (BB8 has no joint constraint).
    (s_lo, s_hi), (d_lo, d_hi) = stages[0][2]
    assert s_lo == math.log(1e-10)
    assert s_hi == pytest.approx(math.log(_THETA_HI - 1.0))
    assert (d_lo, d_hi) == (0.01, 1.0)
    rng = np.random.default_rng(20260918)
    for _ in range(200):
        p = (rng.uniform(s_lo, s_hi), rng.uniform(d_lo, d_hi))
        kw = params_of(p)
        cop = CopulaBB8(**kw)                       # must not raise
        assert 0.0 < kw["tau_k"] < 1.0
        # ... and the τ it returns is BB8's own quadrature, not BB1's or
        # BB6's closed form.
        theta = 1.0 + math.exp(min(max(p[0], s_lo), s_hi))
        assert kw["tau_k"] == pytest.approx(
            _bb8_tau_quad(theta, kw["delta8"]), rel=1e-12, abs=1e-15)
        assert cop.theta >= 1.0

    # BB1's and BB6's branches are unchanged and keyed on their own names.
    q0, _, pof1 = _two_parameter_spec(CopulaBB1, CopulaEnum.BB1, 0.6)
    assert set(pof1(q0)) == {"tau_k", "delta"}
    r0, _, pof6 = _two_parameter_spec(CopulaBB6, CopulaEnum.BB6, 0.6)
    assert set(pof6(r0)) == {"tau_k", "delta6"}


# --------------------------------------------------------------------------
# Parameters — and the joint constraint BB8 does *not* have
# --------------------------------------------------------------------------

def test_default_delta8_is_the_joe_member():
    assert _DELTA_DEFAULT == 1.0
    cop = CopulaBB8(tau_k=0.4)
    assert cop.delta8 == 1.0
    assert cop.theta == CopulaJoe(tau_k=0.4).theta


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.0 + 1e-9, 2.0, math.nan, math.inf])
def test_delta8_outside_zero_one_is_refused(bad):
    with pytest.raises(CopulaParameterError, match="delta8"):
        CopulaBB8(tau_k=0.5, delta8=bad)


@pytest.mark.parametrize("bad", [True, False, "not a number", None, object()])
def test_delta8_that_is_not_a_number_is_refused(bad):
    """Booleans included: ``float(True)`` is 1.0, a *valid* δ, so the check has
    to precede the conversion (BB6's and Tawn's precedent). A numeric string
    is accepted, as it is for every other family here — ``float("0.5")``."""
    with pytest.raises(CopulaParameterError):
        CopulaBB8(tau_k=0.5, delta8=bad)
    assert CopulaBB8(tau_k=0.5, delta8="0.5").delta8 == 0.5


def test_no_joint_tau_delta_constraint():
    """BB8 is the package's only two-parameter family whose whole registered
    τ-range is reachable at *every* value of the second parameter — BB1, BB6
    and the Tawn models all restrict one given the other. Every (τ, δ) below
    must build, and none must be repaired."""
    lo, hi = CopulaEnum.BB8.constructible_tau_range()
    for tau in (lo, 1e-9, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.9, 0.99, hi):
        for delta in (0.01, 0.05, 0.2, 0.5, 0.9, 1.0):
            params = {"tau_k": tau, "delta8": delta}
            assert CopulaBB8.constructible_params(params) is params
            cop = CopulaBB8(tau_k=tau, delta8=delta)
            assert cop.theta >= 1.0


@pytest.mark.parametrize("delta,fixed", [(0.0, 0.01), (-3.0, 0.01), (5.0, 1.0)])
def test_constructible_params_moves_only_delta_into_the_box(delta, fixed):
    out = CopulaBB8.constructible_params({"tau_k": 0.6, "delta8": delta})
    assert out["tau_k"] == 0.6                    # τ is the caller's, untouched
    assert out["delta8"] == fixed
    CopulaBB8(**out)                              # and it builds


def test_constrain_params_is_the_plain_box():
    assert CopulaBB8.constrain_params({"tau_k": 0.6})["delta8"] == 1.0
    assert CopulaBB8.constrain_params({"tau_k": 0.6, "delta8": 3.0})["delta8"] == 1.0
    assert CopulaBB8.constrain_params({"tau_k": 0.6, "delta8": 1e-9})["delta8"] == 0.01


def test_tau_beyond_the_theta_cap_stores_the_tau_realised():
    """RB-10 (Galambos's and Tawn's convention): a τ the family cannot reach
    at this δ builds at the θ cap and the copula reports the τ it *has*."""
    cap = _tau_of(_THETA_HI, 0.001)
    assert cap < 1.0
    cop = CopulaBB8(tau_k=min(cap + 1e-4, 1.0 - 1e-9), delta8=0.001)
    assert cop.theta == _THETA_HI
    assert cop.params["tau_k"] == pytest.approx(cap, rel=1e-12)


def test_reachable_tau_bounds_is_the_joe_cap():
    """BB8 is the first *Archimedean* family here to declare
    ``reachable_tau_bounds`` — the mechanism Galambos, Hüsler–Reiss, Tawn and
    t-EV use, for their reason: τ is a quadrature inverted by Brent on a
    bounded θ, so the registered τ = 1 is not attained. It is excluded from
    ``test_frank_reachable_tau.test_other_families_are_unchanged`` for exactly
    that, and this test is what replaces it."""
    lo, hi = CopulaBB8.reachable_tau_bounds()
    assert lo == EPS
    assert hi == pytest.approx(_joe_tau_from_theta(_THETA_HI), rel=1e-15)
    assert hi == pytest.approx(1.0 - 2e-7, rel=1e-6)
    # It covers the whole padded τ-range the optimisers use ...
    assert hi > CopulaEnum.BB8.constructible_tau_range()[1]
    # ... and beyond it, τ is capped rather than refused (the RB-10 rule).
    assert CopulaEnum.BB8.reachable_tau(1.0) == hi
    assert CopulaEnum.BB8.reachable_tau(0.9) == 0.9
    assert CopulaBB8.reachable_tau_abs is None


# --------------------------------------------------------------------------
# Kendall's τ — the quadrature against a 60-digit mpmath reference
# --------------------------------------------------------------------------

# 60-digit ``mpmath``: τ = 1 + (4/(θ²δ²))·quad(g) with
# g(z) = (1 − e^{−z})·ln((1 − e^{−z})/η)·e^{z(1 − 2/θ)} on [0, −θ ln(1 − δ)],
# η = 1 − (1 − δ)^θ, on geometric breakpoint sets graded towards *both* ends
# at ratios 10 and 3 — two independent meshes agreeing to 10⁻⁵².
#
# Two lessons are recorded in this table rather than in prose. (1) The naive
# ``ln A₁ − ln η`` loses 44 of 50 digits at θ = 10³, δ = 0.2: the reference
# that produced it looked converged and was wrong in the third digit, so the
# reference uses the same cancellation-free ``log1p`` form the module does.
# (2) δ = 1 is deliberately absent: the m-coordinate's upper limit is
# infinite there and the family *is* Joe, whose closed form the module uses —
# ``test_delta_one_is_joes_own_tau`` pins that separately.
_TAU_MPMATH = [
    # (theta, delta, tau)
    (1.0, 0.02, 0.0),
    (1.0, 0.15, 0.0),
    (1.0, 0.5, 0.0),
    (1.0, 0.9, 0.0),
    (1.0, 0.999, 0.0),
    (1.0000001, 0.02, 2.25604277856524e-10),
    (1.0000001, 0.15, 1.8769596815168978e-09),
    (1.0000001, 0.5, 8.843714553327214e-09),
    (1.0000001, 0.9, 3.202171675622449e-08),
    (1.0000001, 0.999, 5.690434150244064e-08),
    (1.2, 0.02, 0.00045120664342848796),
    (1.2, 0.15, 0.00375290294371552),
    (1.2, 0.5, 0.01760529114068087),
    (1.2, 0.9, 0.06155830183570701),
    (1.2, 0.999, 0.10137607824828032),
    (2.0, 0.02, 0.002255987623926643),
    (2.0, 0.15, 0.018740316505926887),
    (2.0, 0.5, 0.08612242827877559),
    (2.0, 0.9, 0.2609562369427465),
    (2.0, 0.999, 0.35378797082316477),
    (3.7, 0.02, 0.0060907995486837625),
    (3.7, 0.15, 0.05040523545917583),
    (3.7, 0.5, 0.21869598660620568),
    (3.7, 0.9, 0.5072461336499174),
    (3.7, 0.999, 0.5885165196080254),
    (8.0, 0.02, 0.015786896719050647),
    (8.0, 0.15, 0.12859391275191312),
    (8.0, 0.5, 0.46217490963180013),
    (8.0, 0.9, 0.7365276220597808),
    (8.0, 0.999, 0.7828202348245034),
    (50.0, 0.02, 0.10937453993048687),
    (50.0, 0.15, 0.6043527401593342),
    (50.0, 0.5, 0.885656797664117),
    (50.0, 0.9, 0.9523632091555917),
    (50.0, 0.999, 0.9609194524013797),
    (200.0, 0.02, 0.3903924151233175),
    (200.0, 0.15, 0.8827520570293406),
    (200.0, 0.5, 0.9703567760417317),
    (200.0, 0.9, 0.9878579649212228),
    (200.0, 0.999, 0.9900440496414237),
    (5000.0, 0.02, 0.9610421982461754),
    (5000.0, 0.15, 0.995076391114079),
    (5000.0, 0.5, 0.9988005726792587),
    (5000.0, 0.9, 0.9995112404375908),
    (5000.0, 0.999, 0.9995993025597854),
]


@pytest.mark.parametrize("theta,delta,tau", _TAU_MPMATH)
def test_tau_quadrature_matches_the_mpmath_reference(theta, delta, tau):
    """The measured bound over the whole table is 6·10⁻¹⁶ absolute; the
    relative one is loose only where τ itself is ≈ 10⁻¹⁰."""
    assert _bb8_tau_quad(theta, delta) == pytest.approx(tau, rel=1e-9, abs=2e-15)


@pytest.mark.parametrize("theta", [1.0, 1.0000001, 1.2, 2.0, 3.7, 8.0, 50.0, 200.0, 5e3, 1e6])
def test_delta_one_is_joes_own_tau(theta):
    """δ = 1 does not go through the quadrature at all: BB8 *is* Joe there, so
    τ must be ``_joe_tau_from_theta`` bit for bit — otherwise the Joe member
    would build a θ differing from ``CopulaJoe``'s in its last digits."""
    assert _bb8_tau_quad(theta, 1.0) == _joe_tau_from_theta(theta)


def test_tau_is_zero_at_theta_one_for_every_delta():
    """θ = 1 is the independence copula whatever δ — the generator is −ln t."""
    for delta in (0.001, 0.01, 0.1, 0.5, 0.9, 1.0):
        assert _bb8_tau_quad(1.0, delta) == 0.0


def test_tau_increases_in_theta_and_in_delta():
    thetas = [1.0 + 1e-9, 1.01, 1.3, 2.0, 5.0, 30.0, 1e3, 1e5, _THETA_HI]
    deltas = [0.01, 0.05, 0.2, 0.5, 0.9, 1.0]
    for delta in deltas:
        row = [_bb8_tau_quad(t, delta) for t in thetas]
        assert all(a < b for a, b in zip(row, row[1:])), (delta, row)
    for theta in thetas:
        col = [_bb8_tau_quad(theta, d) for d in deltas]
        assert all(a < b for a, b in zip(col, col[1:])), (theta, col)


@pytest.mark.parametrize("delta,const", [(1.0, 2.0), (0.9, 4 / 0.9 - 2), (0.5, 6.0),
                                         (0.2, 18.0), (0.05, 78.0), (0.01, 398.0)])
def test_tau_approaches_one_at_joes_rate_with_a_delta_dependent_constant(delta, const):
    """``1 − τ ≃ (4/δ − 2)/θ`` as θ → ∞ — the rate is Joe's ``O(1/θ)``, not the
    ``θ^{-2}`` this round first guessed from a mis-converged reference, and it
    is the *constant* that carries δ. At δ = 1 it is Joe's own ``1 − 2/θ``.
    This is what makes the θ cap 10⁷ (and the RB-10 convention behind it)
    necessary rather than decorative: at δ = 0.01 the cap still leaves
    1 − τ = 4·10⁻⁵."""
    assert (1.0 - _bb8_tau_quad(1e7, delta)) * 1e7 == pytest.approx(const, rel=2e-5)
    # The rate itself: one decade of θ, one decade of 1 − τ (the O(1/θ²) term
    # is still worth 1.5 % of the ratio at δ = 0.01, θ = 10⁶).
    a = 1.0 - _bb8_tau_quad(1e6, delta)
    b = 1.0 - _bb8_tau_quad(1e7, delta)
    assert a / b == pytest.approx(10.0, rel=2e-3)


@pytest.mark.parametrize("tau,delta", _CASES)
def test_tau_round_trip(tau, delta):
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    assert cop.tau_of() == pytest.approx(tau, rel=0.0, abs=1e-12)
    assert cop.params["tau_k"] == tau                 # stored, not re-derived


# --------------------------------------------------------------------------
# Sub-models: Joe at δ = 1, independence at θ = 1, Frank only as a limit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.05, 0.2, 0.5, 0.8, 0.95])
def test_delta_one_is_joe_to_machine_precision(tau):
    bb8, joe = CopulaBB8(tau_k=tau, delta8=1.0), CopulaJoe(tau_k=tau)
    assert bb8.theta == joe.theta                       # bit for bit
    np.testing.assert_allclose(bb8.cdf_array(_UV), joe.cdf_array(_UV),
                               rtol=1e-13, atol=1e-15)
    lb, lj = bb8.logpdf_array(_UV), joe.logpdf_array(_UV)
    ok = np.isfinite(lb) & np.isfinite(lj)
    np.testing.assert_allclose(lb[ok], lj[ok], rtol=1e-11, atol=1e-11)
    assert bb8.tail_dependence() == pytest.approx(joe.tail_dependence(), rel=1e-14)


@pytest.mark.parametrize("delta", [0.02, 0.2, 0.5, 0.9, 1.0])
def test_theta_one_is_independence_not_frank(delta):
    """The negative result of this round. BB8 at θ = 1 is Π for every δ — the
    "Joe–Frank" name notwithstanding, no parameter value is a Frank copula,
    and a Frank copula at the same (near-zero) τ is *not* the same thing
    either (it differs from Π only at order τ, so the check is that BB8 is
    closer to Π than Frank is, by many orders)."""
    from pmcprg.copulas.archimedean.bb8 import _bb8_cdf_k, _bb8_logpdf_k

    prod = _UV[:, 0] * _UV[:, 1]
    ka, kb = np.log1p(-_UV[:, 0]), np.log1p(-_UV[:, 1])
    # θ = 1 *exactly*, which is what the algebra says: ln c ≡ 0 and C ≡ uv, to
    # the last bit, for every δ.
    # (The only residual is ``ln δ − ln η``, two separately rounded logs of
    # the same number: η = 1 − (1 − δ)^1 = δ at θ = 1.)
    assert np.max(np.abs(_bb8_logpdf_k(ka, kb, 1.0, delta))) < 1e-15
    assert np.max(np.abs(_bb8_cdf_k(ka, kb, 1.0, delta)[0] - prod)) < 1e-15
    # The constructor cannot reach θ = 1 (τ = 0 is outside the range), but a
    # τ of 10⁻¹² lands within 2·10⁻¹² of it and C follows.
    cop = CopulaBB8(tau_k=1e-12, delta8=delta)
    assert cop.theta - 1.0 < 1e-9
    assert np.max(np.abs(cop.cdf_array(_UV) - prod)) < 1e-11
    # Frank at a *visible* τ is nowhere near, which is the point: θ = 1 is not
    # "a Frank copula with some parameter", it is independence.
    frank = CopulaFrank(tau_k=0.3)
    assert np.max(np.abs(frank.cdf_array(_UV) - prod)) > 1e-3


@pytest.mark.parametrize("theta", [4.0, 12.0])
def test_delta_to_zero_at_fixed_theta_is_independence_linearly(theta):
    """|C/uv − 1| falls like δ (measured 1.5·10⁻³, 1.5·10⁻⁶, 1.5·10⁻⁹ at θ = 4).

    On the kernel at a **fixed** θ, not through the constructor: at these δ the
    τ is far below what a double can invert back to θ, so a round trip through
    ``tau_k`` would be measuring Brent's resolution, not the family's limit.
    """
    from pmcprg.copulas.archimedean.bb8 import _bb8_cdf_k

    ka, kb = np.log1p(-_UV[:, 0]), np.log1p(-_UV[:, 1])
    prod = _UV[:, 0] * _UV[:, 1]
    errs = [float(np.max(np.abs(_bb8_cdf_k(ka, kb, theta, d)[0] / prod - 1.0)))
            for d in (1e-3, 1e-6, 1e-9)]
    for a, b in zip(errs, errs[1:]):
        assert b < a / 100.0                          # δ down 10³, err down ≥ 10²
    assert errs[-1] < 1e-8
    # The whole corner grid holds it, including (10⁻¹², 10⁻¹²), where C is
    # 10⁻²⁴ — the point the ``ln A`` branch of ``_bb8_terms_k`` exists for.


@pytest.mark.parametrize("kappa", [1.0, 5.0])
def test_frank_is_only_a_joint_limit(kappa):
    """Frank appears as θ → ∞, δ → 0 with θδ = κ fixed, and the approach is
    O(1/θ). Tested on the *kernel* at fixed (θ, δ), not through τ: the point
    is that no admissible parameter value is Frank, and that the limit is
    genuinely Frank's copula and not merely "Frank-like"."""
    from pmcprg.copulas.archimedean.bb8 import _bb8_cdf_k

    frank = CopulaFrank(tau_k=0.5)
    frank.theta = kappa                                # Frank's own parameter
    ka, kb = np.log1p(-_UV[:, 0]), np.log1p(-_UV[:, 1])
    ref = frank.cdf_array(_UV)
    errs = []
    for theta in (1e2, 1e3, 1e4):
        c, _ = _bb8_cdf_k(ka, kb, theta, kappa / theta)
        errs.append(float(np.max(np.abs(c - ref))))
    assert errs[0] < 2e-2
    for a, b in zip(errs, errs[1:]):
        assert b < a / 5.0                             # ≈ 1/10 per decade of θ
    assert errs[-1] < 1e-4


# --------------------------------------------------------------------------
# C, h, ln c against the adaptive-precision mpmath ground truth
# --------------------------------------------------------------------------

# (τ, δ, u, v, C, h(v|u), ln c). C from φ^{-1}(φ(u) + φ(v)) alone; h and c its
# first and mixed central differences at a step 10⁻²⁰ of the distance to the
# edge; the working precision raised over (480, 960, 1920, 3840, 7680) digits
# until **three** successive ones agree to 25 digits (the precision each row
# converged at is in the comment, with the θ the constructor built). Every row
# was re-derived under that rule after the looser one (two precisions, 12
# digits) was caught stopping on two wrong values elsewhere on this family's
# grid: 0 rows changed, but the rule is the one recorded.
_POINT_MPMATH = [
    (0.4, 0.5, 0.3, 0.7, 0.2773302089786293, 0.8838762560844157, -0.4063033532531152),          # θ = 6.6654364440, 240
    (0.4, 0.5, 0.7, 0.3, 0.2773302089786293, 0.13557731477463023, -0.4063033532531152),         # exchangeable, 240
    (0.5, 0.05, 1e-12, 1e-06, 5.632708605596951e-18, 5.632708605581279e-06, 1.7285876462029492),  # θ = 112.29957509, 240
    (0.7, 1.0, 0.999999, 0.5, 0.5, 3.5582056628644027e-26, -56.380834809859074),                # δ = 1 (Joe), 240
    (0.9, 0.3, 1e-12, 0.999999999999, 1e-12, 1.0, -34.92801817125117),                          # θ = 108.69639024, 240
    (0.9, 0.3, 0.9, 0.95, 0.8979829941119751, 0.9146321987044728, 1.3747673846211557),          # 240
    (0.95, 0.01, 1e-06, 1e-06, 7.79256546580747e-11, 7.792261888327716e-05, 4.355677313623291),  # θ = 7793.1726720, 240
    (0.2, 1.0, 0.01, 0.99, 0.009987017949961239, 0.9986988770083557, -1.672075673591778),        # δ = 1 (Joe), 240
    (1e-4, 0.5, 1e-12, 0.999999999999, 9.999999999990005e-13, 0.9999999999990005, -0.0004368371010942967),  # 240
    (0.99, 0.8, 0.999999, 0.999999, 0.9999980011888739, 0.9988118349296707, 7.079569655866932),  # θ = 298.57343443, 960
    (0.1, 0.95, 0.5, 0.5, 0.27509948710658944, 0.521198097465723, 0.039116346786064383),         # 240
    (0.6, 0.2, 1e-12, 1e-12, 7.21203878192718e-24, 7.212038781901905e-12, 1.9757516827360173),   # θ = 36.048617976, 240
    (0.3, 0.999, 1e-12, 0.999999999999, 9.999999999999917e-13, 0.9999999999999918, -4.795273234583475),  # 240
    (0.8, 0.6, 0.999999999999, 0.999999999999, 0.999999999998, 0.9999999999695243, 3.4169532152380735),  # θ = 21.317612155, 240
]


@pytest.mark.parametrize("tau,delta,u,v,C,h,logc", _POINT_MPMATH)
def test_kernel_matches_mpmath(tau, delta, u, v, C, h, logc):
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    assert cop.cdf([u, v]) == pytest.approx(C, rel=1e-11)
    assert cop.conditional_cdf(v, u) == pytest.approx(h, rel=1e-10)
    assert float(cop.logpdf_array(np.array([[u, v]]))[0]) == pytest.approx(
        logc, rel=1e-11, abs=1e-11)
    assert cop.pdf([u, v]) == pytest.approx(math.exp(logc), rel=1e-10)


@pytest.mark.parametrize("tau,delta", _CASES)
def test_no_nan_on_the_edge_grid_and_pdf_is_exp_of_logpdf(tau, delta):
    """RB-8/RB-9: the whole [10⁻¹², 1 − 10⁻¹²] grid evaluates, and the linear
    density is exactly ``exp`` of the log one (no ``EPS`` floor)."""
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    logc = cop.logpdf_array(_UV)
    assert not np.any(np.isnan(logc))
    ok = np.isfinite(logc) & (logc > -700.0) & (logc < 700.0)
    np.testing.assert_allclose(cop.pdf_array(_UV)[ok], np.exp(logc[ok]), rtol=1e-12)
    scalar = np.array([cop.pdf([u, v]) for u, v in _UV])
    np.testing.assert_allclose(scalar[ok], np.exp(logc[ok]), rtol=1e-9)
    C = cop.cdf_array(_UV)
    assert np.all((C >= 0.0) & (C <= 1.0)) and not np.any(np.isnan(C))


@pytest.mark.parametrize("delta", [0.01, 0.3, 0.7, 0.999, 1.0])
def test_log_p_two_branches_agree_across_the_switch(delta):
    """``ln P`` switches form at P = ½ (module docstring, "Numerics"). Either
    side of it the two exact forms must agree, and at δ = 1 the second must
    return ``ka`` itself — the Joe coordinate, bit for bit."""
    ka = np.log1p(-np.array([1e-16, 1e-12, 1e-6, 0.3, 0.5, 0.7, 0.9,
                             1 - 1e-6, 1 - 1e-12, 1 - EPS]))
    got = _bb8_log_p(ka, delta)
    u = -np.expm1(ka)
    assert np.all(np.isfinite(got))
    # Against the definition, computed the other way round in each regime.
    np.testing.assert_allclose(got, np.log(1.0 - delta + delta * np.exp(ka)),
                               rtol=1e-9, atol=1e-16)
    assert np.all(got <= 0.0)
    assert np.all(np.diff(got) <= 0.0)              # decreasing in u
    if delta == 1.0:
        np.testing.assert_array_equal(got, ka)       # exactly Joe's coordinate
    # u → 0: ln P ≈ −δu, which ``ln(1 − u)`` alone would get wrong by 1/δ.
    tiny = u < 1e-10
    np.testing.assert_allclose(got[tiny], -delta * u[tiny], rtol=1e-9)


# --------------------------------------------------------------------------
# Tail dependence — and its discontinuity at δ = 1
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,delta", _CASES)
def test_tail_dependence_matches_the_diagonal_limits(tau, delta):
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    lam_l, lam_u = cop.tail_dependence()
    assert lam_l == 0.0
    expected = 2.0 - 2.0 ** (1.0 / cop.theta) if delta >= 1.0 else 0.0
    assert lam_u == pytest.approx(expected, rel=1e-14, abs=1e-15)

    # λ_U from the diagonal: (1 − 2u + C(u, u))/(1 − u) = 2 − (1 − C)/(1 − u)
    # as u → 1. Written through the kernel's **complement** 1 − C: formed as
    # ``1 − 2u + C`` in linear scale it cancels to nothing (at 1 − u = 10⁻¹²
    # the ratio came out −1.1·10⁻⁴ where it must be 0 — the limit of the
    # double, not of the family).
    # The mesh is written in the kernel's own coordinate ka = ln(1 − u), so
    # 1 − u is the *exact* power of ten the ratio divides by, and stops at
    # 10⁻⁸: below that the double's resolution of ln(1 − A) (whose scale is
    # θ ln E) is coarser than the quantity being measured, and the ratio walks
    # away again — 1.7 at 1 − u = 10⁻¹⁵, τ = 0.95, δ = 0.01. Down to 10⁻⁸ it
    # falls exactly like 1 − u, the λ_U = 0 signature.
    ks = np.arange(3, 9)
    q = 10.0 ** -ks.astype(float)
    ka = np.log(q)
    _, comp = cop._k_cdf(ka, ka)
    ratio = 2.0 - comp / q
    assert ratio[-1] == pytest.approx(expected, abs=2e-6)
    if delta < 1.0:
        # λ_U = 0: the ratio itself is O(1 − u) and falls a decade per decade.
        assert np.all(np.diff(np.abs(ratio)) < 0.0)
        assert abs(ratio[0]) / abs(ratio[-1]) > 1e4
    else:
        # λ_U > 0: the ratio converges to the limit from 1 − u = 10⁻³ on, at
        # the family's own algebraic rate (7·10⁻⁵ relative at 10⁻³, under
        # 10⁻⁸ by 10⁻⁶).
        np.testing.assert_allclose(ratio, expected, rtol=1e-4)
        assert ratio[-1] == pytest.approx(expected, rel=1e-8)
    # λ_L from C(u, u)/u as u → 0: zero for every member.
    ls = np.array([10.0 ** -k for k in range(6, 13)])
    assert np.max(cop.cdf_array(np.column_stack([ls, ls])) / ls) < 1e-4


def test_lambda_u_is_discontinuous_at_delta_one():
    """Not an approximation: the family really has λ_U = 0 up to δ = 1
    exclusive and 2 − 2^{1/θ} at δ = 1. The diagonal ratio confirms both
    sides at a τ where the Joe member's λ_U is far from 0."""
    joe_side = CopulaBB8(tau_k=0.8, delta8=1.0)
    assert joe_side.tail_dependence()[1] > 0.8
    for delta in (1 - 1e-6, 1 - 1e-12):
        assert CopulaBB8(tau_k=0.8, delta8=delta).tail_dependence()[1] == 0.0


# --------------------------------------------------------------------------
# h, inverse h, sampling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,delta", _CASES)
def test_h_inv_h_round_trip_and_saturation(tau, delta):
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    rng = np.random.default_rng(_seed("invh", tau, delta))
    u = rng.uniform(1e-6, 1 - 1e-6, 500)
    w = rng.uniform(1e-6, 1 - 1e-6, 500)
    v = cop.inv_h_array(w, u)
    back = np.array([cop.conditional_cdf(vi, ui) for vi, ui in zip(v, u)])
    np.testing.assert_allclose(back, w, atol=1e-9)
    # Monotone in w, inside the open square, and saturating at the ends of the
    # *w* range — not at v = ε: ``inv_h_array`` clips w into [ε, 1 − ε] first,
    # and the v solving h(v|u) = ε is a perfectly ordinary interior point once
    # θ is large (0.039 at τ = 0.95, δ = 0.01).
    ws = np.array([1e-300, 1e-12, 0.01, 0.5, 0.99, 1 - 1e-12, 1 - 1e-16])
    vs = cop.inv_h_array(ws, np.full(ws.shape, 0.5))
    assert np.all((vs > 0.0) & (vs < 1.0))
    assert np.all(np.diff(vs) >= 0.0)


@pytest.mark.parametrize("tau,delta", _CASES)
def test_h_is_a_cdf_in_v(tau, delta):
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    vs = np.array([1e-12, 1e-6, 0.2, 0.5, 0.8, 1 - 1e-6, 1 - 1e-12])
    for u in (1e-6, 0.3, 0.7, 1 - 1e-6):
        h = np.array([cop.conditional_cdf(v, u) for v in vs])
        assert np.all((h >= 0.0) & (h <= 1.0))
        assert np.all(np.diff(h) >= -1e-12)
    assert cop.conditional_cdf(1.0, 0.4) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("tau,delta", _CASES)
def test_sample_margins_and_monte_carlo_tau(tau, delta):
    cop = CopulaBB8(tau_k=tau, delta8=delta)
    uv = cop.sample(n=4000, seed=_seed("sample", tau, delta))
    assert kstest(uv[:, 0], "uniform").pvalue > 1e-3
    assert kstest(uv[:, 1], "uniform").pvalue > 1e-3
    got, _ = kendalltau(uv[:, 0], uv[:, 1])
    assert got == pytest.approx(tau, abs=0.05)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_fit_tau_method_falls_back_to_mle(caplog):
    uv = CopulaBB8(tau_k=0.5, delta8=0.4).sample(n=400, seed=5)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.archimedean.bb8"):
        res = CopulaBB8.fit(uv, method="tau")
    assert res.method == "mle"
    assert "cannot identify delta8" in caplog.text


@pytest.mark.parametrize("tau,delta", [(0.3, 0.5), (0.5, 0.2), (0.7, 1.0), (0.8, 0.7)])
def test_fit_recovers_tau(tau, delta):
    """τ is recovered tightly; δ is *not* pinned down at this n and the test
    says so rather than pretending otherwise — the (θ, δ) likelihood ridge is
    long and flat (module docstring / CHANGELOG, "Known limitations")."""
    uv = CopulaBB8(tau_k=tau, delta8=delta).sample(n=3000, seed=_seed("fit", tau, delta))
    res = CopulaBB8.fit(uv, method="mle")
    assert res.converged
    assert res.copula.params["tau_k"] == pytest.approx(tau, abs=0.05)
    assert 0.01 <= res.copula.params["delta8"] <= 1.0
    # The fitted model is at least as likely as the truth's own parameters.
    truth = CopulaBB8(tau_k=tau, delta8=delta)
    assert res.log_likelihood >= float(np.sum(truth.logpdf_array(res.uv))) - 1e-6


def test_fit_reaches_the_joe_boundary():
    """δ̂ must be able to sit *on* δ = 1 — the Joe sub-model is the upper end
    of the box, and a fit that cannot reach it makes the LR test meaningless."""
    uv = CopulaJoe(tau_k=0.6).sample(n=3000, seed=_seed("joe-boundary"))
    res = CopulaBB8.fit(uv, method="mle")
    assert res.copula.params["delta8"] > 0.95


def test_submodel_lr_test_accepts_bb8s_nesting():
    from pmcprg.copulas import submodel_lr_test
    from pmcprg.copulas._stderr import _SUBMODEL_NESTING

    assert ("CopulaBB8", "CopulaJoe") in _SUBMODEL_NESTING
    # BB8's other edge (θ = 1) is independence, not a two-parameter nesting.
    assert ("CopulaBB8", "CopulaProduct") not in _SUBMODEL_NESTING
    uv = CopulaBB8(tau_k=0.55, delta8=0.35).sample(n=600, seed=_seed("lr"))
    res = submodel_lr_test(CopulaBB8, CopulaJoe, uv)
    assert res.boundary is True
    assert res.null_distribution == "0.5*chi2(0) + 0.5*chi2(1)"
    assert res.p_value < 0.01                       # genuine BB8 data, δ ≪ 1


def test_fit_best_includes_bb8():
    from pmcprg.copulas._base import CopulaVirt
    uv = CopulaBB8(tau_k=0.5, delta8=0.3).sample(n=800, seed=_seed("best"))
    out = CopulaVirt.fit_best(uv, families=[CopulaBB8, CopulaJoe], method="mle")
    assert [r.copula.class_name for r in out]
    assert "CopulaBB8" in {r.copula.class_name for r in out}
    assert not out.failures


def test_ice_m_step_fits_delta8():
    """The ICE M-step must reach BB8 through its own branch and return both
    parameters (RB-3's failure mode: an extra parameter stuck at its start)."""
    from pmcprg.pmc.ice import _fit_copula_params, _resolve_candidate
    uv = CopulaBB8(tau_k=0.7, delta8=0.25).sample(n=2000, seed=_seed("ice"))
    entry, cls = _resolve_candidate("BB8")
    assert cls is CopulaBB8 and entry is CopulaEnum.BB8
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(len(uv)))
    assert set(p) == {"tau_k", "delta8"}
    assert 0.01 <= p["delta8"] < 1.0                # moved off the 1.0 start
    assert p["tau_k"] == pytest.approx(0.7, abs=0.06)


def test_bb8_cases_are_covered_by_the_reference_file():
    """The decimal reference grid of ``test_copula_limits`` holds BB8."""
    import test_copula_limits as tcl
    keys = {k for k in tcl._file_table() if k[0] == "BB8"}
    assert len(keys) == len(tcl._TAIL_TAUS["BB8"])
