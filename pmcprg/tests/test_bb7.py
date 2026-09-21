"""BB7 copula (FR-9) — generator, sub-models, τ, tails, joint constraint, fit.

Complements ``test_copula_limits.py`` (which adds BB7 to its high-precision
``decimal`` reference grid at four τ, its *recovered* δ carried in the file's
``delta`` slot, one of the four cases being the Clayton sub-model exactly and
one the Joe end) with the checks specific to this family:

* the **Beta closed form** of τ — ``τ = 1 − 2[1 − (2/θ)(δ+1)B(2/θ, δ+1)] /
  (δ(2 − θ))``, with its removable singularity at θ = 2 — against a 120-digit
  ``mpmath`` Genest–MacKay quadrature ``1 + 4∫φ/φ′`` of BB7's *own* generator,
  and against a simulated Kendall's τ;
* the two algebraic τ identities the closed form must reproduce **exactly**
  and which no quadrature was used for: ``τ(1, δ) = δ/(δ + 2)`` (Clayton) and
  ``τ(θ, 1) = 1 − 2/(θ + 2)``;
* the property that decides this family's parametrisation: τ **is** strictly
  increasing in δ at fixed θ, and **is not** monotone in θ at fixed δ — it
  dips below the Clayton value for δ ≳ 3.44. Hence ``theta7``, not a
  ``delta7``, is the registered extra parameter;
* C, h and ln c against an adaptive-precision ``mpmath`` ground truth (C from
  the generator alone, h and c as its finite differences at a step 10⁻²⁰ of
  the distance to the edge, the precision raised until two successive ones
  agree to 12 digits — the recipe of ``test_copula_limits``, applied off-line
  and recorded below);
* θ = 1 ≡ Clayton to machine precision, and θ → θ_Joe(τ) ≡ Joe;
* λ_U = 2 − 2^{1/θ} and λ_L = 2^{−1/δ}, each depending on one parameter only;
* the joint (τ, θ) constraint θ < θ_Joe(τ), its repair hooks and the
  multistart draws — mirroring ``test_multistart_joint_constraints.py`` for
  BB1 — including the fact that both of BB7's ends are attained *doubles*;
* that ``theta7`` takes **its own** branch of ``_two_parameter_spec`` and not
  BB1's ``delta`` or BB6's ``delta6`` branch, which would fit BB7 with
  another family's τ map;
* recovery of both τ and θ by ``fit``, and ``method='tau'`` (itau with a profile MLE, FR-12).

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
    CopulaBB7,
    CopulaClayton,
    CopulaEnum,
    CopulaJoe,
)
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.copulas.archimedean.bb7 import (
    _DELTA_FLOOR,
    _THETA_DEFAULT,
    _bb7_cdf_k,
    _bb7_delta_from_tau,
    _bb7_log_x_minus_one,
    _bb7_tau_from_theta,
    _bb7_terms_k,
)
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_G = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12])
_UV = np.array([(u, v) for u in _G for v in _G])

# (τ, θ) points used throughout: the Clayton end θ = 1 at three τ, the Joe end
# (0.7, 5.0) — θ_Joe(0.7) = 5.4638 — and four interior points.
_CASES = [(0.2, 1.0), (0.3, 1.2), (0.4, 1.0), (0.5, 1.5),
          (0.5, 2.5), (0.7, 1.0), (0.7, 5.0), (0.9, 3.0)]


def _seed(*parts: object) -> int:
    """A reproducible seed from ``parts`` — ``zlib.crc32``, never ``hash()``."""
    return zlib.crc32(repr(parts).encode())


def _theta_at_delta(tau: float, delta: float) -> float:
    """The θ that realises ``(τ, δ)`` — bisection on θ ∈ [1, theta_max(τ)].

    Used only to walk *into* the Joe corner along a chosen δ; the family
    itself never needs it. δ is decreasing in θ at fixed τ (the inverse of
    "τ increasing in δ at fixed θ"), which is what makes the bisection valid.
    """
    lo, hi = 1.0, CopulaBB7.theta_max(tau)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if CopulaBB7(tau_k=tau, theta7=mid).delta7 > delta:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    entry = CopulaEnum.BB7
    assert entry.value.SHORT_NAME == "BB7"
    assert entry.value.CLASS_NAME == "CopulaBB7"
    assert entry.klass is CopulaBB7
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "theta7"]
    assert entry.value.TAU_MIN_MAX == [EPS, 1.0]
    assert entry in CopulaEnum.available()
    assert CopulaBB7.n_params == 2
    assert CopulaEnum.from_short_name("BB7") is entry


def test_theta7_bounds_are_registered_and_do_not_reuse_bb1s_or_bb6s():
    from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
    assert EXTRA_PARAM_BOUNDS_BY_PARAM["theta7"] == (1.0, 10.0, 1.0)
    assert EXTRA_PARAM_BOUNDS["CopulaBB7"] == {"theta7": (1.0, 10.0, 1.0)}
    # BB1's and BB6's own entries are untouched.
    assert EXTRA_PARAM_BOUNDS["CopulaBB1"] == {"delta": (1.0, 10.0, 1.5)}
    assert EXTRA_PARAM_BOUNDS["CopulaBB6"] == {"delta6": (1.0, 10.0, 1.0)}
    names = CopulaEnum.BB7.value.PARAMETERS_SET_NAME
    assert "delta" not in names and "delta6" not in names


def test_two_parameter_spec_takes_its_own_branch():
    """The silent-bug guard: ``_two_parameter_spec`` dispatches on the
    parameter *name*, so a BB7 registered under ``delta`` would be optimised
    with BB1's ``τ = 1 − 2/(δ(θ + 2))`` map and under ``delta6`` with BB6's
    outer-power one. BB7's branch must produce its own kwargs key and its own
    τ map."""
    from pmcprg.copulas._fit import _two_parameter_spec

    p0, stages, params_of = _two_parameter_spec(CopulaBB7, CopulaEnum.BB7, 0.6)
    assert set(params_of(p0)) == {"tau_k", "theta7"}
    assert len(stages) == 2
    # Box = the admissible set: s = ln(θ − 1) and ln δ, every point buildable.
    (s_lo, s_hi), (ld_lo, ld_hi) = stages[0][2]
    assert s_lo == math.log(1e-10) and s_hi == pytest.approx(math.log(9.0))
    assert ld_lo == pytest.approx(math.log(1e-6)) and ld_hi == pytest.approx(math.log(1e6))
    rng = np.random.default_rng(20260918)
    for _ in range(200):
        p = (rng.uniform(s_lo, s_hi), math.exp(rng.uniform(ld_lo, ld_hi)))
        kw = params_of(p)
        cop = CopulaBB7(**kw)                       # must not raise
        assert 0.0 < kw["tau_k"] < 1.0
        # ... and the τ it returns is BB7's own, not BB1's or BB6's map.
        theta = 1.0 + math.exp(min(max(p[0], s_lo), s_hi))
        assert kw["tau_k"] == pytest.approx(
            _bb7_tau_from_theta(theta, p[1]), rel=1e-12)
        assert cop.theta >= 1.0

    # BB1's and BB6's branches are unchanged and keyed on their own names.
    q0, _, pof1 = _two_parameter_spec(CopulaBB1, CopulaEnum.BB1, 0.6)
    assert set(pof1(q0)) == {"tau_k", "delta"}
    r0, _, pof6 = _two_parameter_spec(CopulaBB6, CopulaEnum.BB6, 0.6)
    assert set(pof6(r0)) == {"tau_k", "delta6"}


# --------------------------------------------------------------------------
# Why θ and not δ: the monotonicity that decides the parametrisation
# --------------------------------------------------------------------------

def test_tau_is_strictly_increasing_in_delta_at_fixed_theta():
    """The property the (τ, θ) parametrisation rests on: δ is recoverable."""
    deltas = [1e-8, 1e-4, 0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e3, 1e4]
    for theta in (1.0, 1.0001, 1.05, 1.3, 1.9, 2.0, 2.1, 3.0, 5.0, 12.0, 60.0, 500.0):
        vals = [_bb7_tau_from_theta(theta, d) for d in deltas]
        assert all(b > a for a, b in zip(vals, vals[1:])), (theta, vals)
        # the low-δ end of the range is the Joe limit
        assert vals[0] == pytest.approx(_joe_tau_from_theta(theta), abs=1e-7)


def test_tau_is_not_monotone_in_theta_at_fixed_delta():
    """The property that rules **δ** out as the stored parameter, and the
    reason BB7 does not simply copy BB1's and BB6's choice.

    At fixed δ, τ starts at Clayton's δ/(δ + 2), *dips*, and only then rises
    to 1. The dip appears at δ* ≈ 3.44121781421 (bisection at 40 digits in
    ``mpmath``) and deepens with δ. Were δ the extra parameter, (τ, δ) would
    be two-to-one over the dip and a whole slab of genuine BB7 copulas would
    have no (τ, δ) name at all.
    """
    thetas = [1.0, 1.001, 1.01, 1.05, 1.1, 1.3, 1.6, 2.0, 3.0, 6.0, 20.0]
    # below δ*: increasing all the way
    for delta in (0.1, 1.0, 2.0, 3.0):
        vals = [_bb7_tau_from_theta(t, delta) for t in thetas]
        assert all(b > a for a, b in zip(vals, vals[1:])), (delta, vals)
    # above δ*: the minimum is strictly inside, not at θ = 1
    for delta, want_drop in ((5.0, 8.23e-3), (7.0, 2.32e-2), (10.0, 3.89e-2), (20.0, 5.95e-2)):
        vals = [_bb7_tau_from_theta(t, delta) for t in thetas]
        drop = vals[0] - min(vals)
        assert drop == pytest.approx(want_drop, rel=0.05), (delta, drop)
        assert vals[-1] > vals[0]                   # and it still rises to 1

    # δ* itself, to three digits: ∂τ/∂θ at θ = 1 changes sign there
    def slope(d):
        return _bb7_tau_from_theta(1.0 + 1e-6, d) - _bb7_tau_from_theta(1.0, d)

    assert slope(3.43) > 0.0 and slope(3.45) < 0.0


def test_delta_from_tau_round_trip():
    """δ(τ, θ) inverts τ(θ, ·) over the whole admissible set."""
    rng = np.random.default_rng(161803)
    worst = 0.0
    for _ in range(600):
        theta = float(1.0 + 10.0 ** rng.uniform(-3, 1))
        lo = _joe_tau_from_theta(theta)
        tau = float(lo + (1.0 - lo) * rng.uniform(1e-4, 1 - 1e-4))
        delta = _bb7_delta_from_tau(tau, theta)
        assert delta > 0.0
        worst = max(worst, abs(_bb7_tau_from_theta(theta, delta) - tau))
    assert worst < 1e-12


# --------------------------------------------------------------------------
# Parameters, the joint (τ, θ) constraint
# --------------------------------------------------------------------------

def test_default_theta7_is_the_clayton_member():
    """A block without θ must build at *every* registered τ — BB6's ``δ = 1``
    principle, applied to the sub-model that is universally admissible here."""
    assert _THETA_DEFAULT == 1.0
    for tau in (EPS, 1e-9, 1e-3, 0.1, 1 / 3, 0.5, 0.9, 1 - 1e-6):
        cop = CopulaBB7(tau_k=tau)
        assert cop.theta == 1.0
        assert cop.params["theta7"] == 1.0
        assert cop.delta7 == pytest.approx(2.0 * tau / (1.0 - tau), rel=1e-15)


@pytest.mark.parametrize("tau", [1e-9, 1e-3, 0.1, 1 / 3, 0.5, 0.75, 0.9, 0.99])
def test_theta_max_is_the_largest_admissible_double(tau):
    """``theta_max`` must agree with the constructor to the last bit, and the
    end itself must be an admissible *double* (a BB7 with a tiny δ), so the
    repair can sit on it without BB1's pull-in."""
    t = CopulaBB7.theta_max(tau)
    cop = CopulaBB7(tau_k=tau, theta7=t)            # must not raise
    assert cop.delta7 > 0.0
    assert _joe_tau_from_theta(t) < tau
    with pytest.raises(CopulaParameterError):
        CopulaBB7(tau_k=tau, theta7=float(np.nextafter(t, math.inf)))


@pytest.mark.parametrize("tau,theta", [(0.2, 2.0), (0.5, 4.0), (0.1, 1.5), (0.9, 30.0)])
def test_refused_pairs(tau, theta):
    with pytest.raises(CopulaParameterError, match=r"tau > tau_Joe\(theta7\)"):
        CopulaBB7(tau_k=tau, theta7=theta)


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, float("nan"), float("inf")])
def test_theta7_below_one_is_refused(bad):
    with pytest.raises(CopulaParameterError):
        CopulaBB7(tau_k=0.5, theta7=bad)


def test_accepted_pairs_are_returned_untouched():
    """``constructible_params`` is the identity — same object — on every pair
    the constructor accepts, and only then (the hook's contract)."""
    rng = np.random.default_rng(20260902)
    taus = np.concatenate([[EPS, 1e-9, 0.1, 1 / 3, 0.5, 0.9, 0.9999],
                           rng.uniform(0, 1, 40)])
    n_accepted = 0
    for tau in taus:
        hi = CopulaBB7.theta_max(float(tau))
        for theta in [1.0, 1.5, 3.0, 10.0, hi, float(np.nextafter(hi, 1.0)),
                      float(np.nextafter(hi, math.inf)),
                      float(rng.uniform(1.0, 10.0))]:
            params = {"tau_k": float(tau), "theta7": float(theta)}
            try:
                CopulaBB7(**params)
                refused = False
            except CopulaParameterError:
                refused = True
            out = CopulaBB7.constructible_params(params)
            assert (out is params) == (not refused)
            n_accepted += out is params
    assert n_accepted > 80


@pytest.mark.parametrize("tau", [EPS, 1e-9, 1e-6, 0.05, 0.2, 1 / 3, 0.5, 0.8, 0.9, 0.9999])
@pytest.mark.parametrize("theta", [0.5, 1.5, 3.0, 10.0, None])
def test_refused_pairs_move_only_theta_onto_the_admissible_end(tau, theta):
    params = {"tau_k": tau} if theta is None else {"tau_k": tau, "theta7": theta}
    try:
        CopulaBB7(**params)
        return                                       # accepted: nothing to repair
    except CopulaParameterError:
        pass
    out = CopulaBB7.constructible_params(params)
    assert out is not params and out["tau_k"] == tau
    cop = CopulaBB7(**out)                           # must build
    assert 1.0 <= out["theta7"] <= CopulaBB7.theta_max(tau)
    if theta is not None and theta < 1.0:
        assert out["theta7"] == 1.0                  # pulled up to the Clayton end
    else:
        # BB7's upper end is an admissible double, so the repair sits *on* it,
        # without BB1's TAU_PAD_REL pull-in.
        assert out["theta7"] == CopulaBB7.theta_max(tau)
        assert cop.delta7 > 0.0


def test_constrain_params_projects_into_the_admissible_interval():
    for tau, theta in ((0.2, 5.0), (0.5, 0.1), (0.9, 100.0), (0.4, 1.2)):
        out = CopulaBB7.constrain_params({"tau_k": tau, "theta7": theta})
        assert 1.0 <= out["theta7"] <= CopulaBB7.theta_max(tau)
        CopulaBB7(**out)
    # a missing θ is the default, and the input dict is never mutated
    src = {"tau_k": 0.2}
    assert CopulaBB7.constrain_params(src)["theta7"] == 1.0
    assert src == {"tau_k": 0.2}


def test_a_theta_one_ulp_below_theta_max_still_builds():
    """BB6's ``_TAU_JOE_FLOOR`` regression, in BB7's own coordinates: a θ one
    or two ulps below ``theta_max`` needs a δ that the Brent bracket cannot
    separate from 0, and is answered with ``_DELTA_FLOOR`` — Joe to machine
    precision. Such pairs are what ``constructible_params`` produces, so they
    must build."""
    assert 0.0 < _DELTA_FLOOR < 1e-12
    assert 2.0 ** (-1.0 / _DELTA_FLOOR) == 0.0       # λ_L is Joe's 0 there
    rng = np.random.default_rng(27182)
    taus = np.concatenate([[1e-9, 0.1, 1 / 3, 0.5, 0.9, 0.9999], rng.uniform(0, 1, 60)])
    n_floor = 0
    for tau in taus:
        tau = float(tau)
        t_max = CopulaBB7.theta_max(tau)
        cop = CopulaBB7(tau_k=tau, theta7=t_max)     # the end itself must build
        assert cop.delta7 > 0.0
        assert cop.tau_of() == pytest.approx(tau, rel=0.0, abs=1e-12)
        n_floor += cop.delta7 <= 2.0 * _DELTA_FLOOR
        # τ_Joe is flat to the last bit over several consecutive doubles once θ
        # is large, and its series evaluation is not monotone there, so a θ
        # just *below* theta_max may still be refused — what must hold is that
        # the repair hook always produces something that builds.
        for theta in (float(np.nextafter(t_max, 1.0)),
                      float(np.nextafter(np.nextafter(t_max, 1.0), 1.0))):
            if theta < 1.0:
                continue
            out = CopulaBB7.constructible_params({"tau_k": tau, "theta7": theta})
            back = CopulaBB7(**out)                  # must not raise
            assert back.delta7 > 0.0
    assert n_floor > 10          # the _DELTA_FLOOR band is reached, not exotic


def test_bb7_is_not_treated_as_a_delta_family_by_the_bb1_guard():
    """``test_multistart_joint_constraints`` asserts that every family without
    a ``delta`` parameter returns an arbitrary dict untouched — BB7 must, and
    does, because its default θ = 1 is admissible at every τ."""
    params = {"tau_k": 0.3, "df": 4.0}
    assert CopulaBB7.constructible_params(params) is params


# --------------------------------------------------------------------------
# Kendall's τ — the Beta closed form and its removable singularity
# --------------------------------------------------------------------------

# 120-digit mpmath: 1 + 4·mp.quad(φ/φ′) over [10⁻⁴⁰, ½, 1 − 10⁻³⁰] with
# φ(t) = [1 − (1 − t)^θ]^{−δ} − 1 and φ′ = −δθ(1 − t)^{θ−1} a^{−δ−1} — BB7's own
# generator, not the Beta identity under test. φ′ is written analytically
# rather than taken from ``mp.diff`` (which underflows to 0 at the tanh-sinh
# nodes closest to the ends, dividing by zero) and was itself checked against
# ``mp.diff`` at benign points: 3.5·10⁻¹²⁰ relative. The two truncations are
# bounded by the behaviour of φ/φ′ itself — ``−t/δ`` at 0 and ``−(1 − t)/θ``
# at 1 — hence below 10⁻⁸⁰ and 10⁻⁶⁰.
#
# Measured agreement with the closed form on this table: 0 to 4·10⁻⁵⁸
# relative (θ ≤ 3) and 3.6·10⁻²⁹ at θ = 9, in every case the *quadrature's*
# limit and not the closed form's. θ = 2 is the removable singularity itself;
# θ = 1.9999999 and 2.0000001 straddle it.
#
# Large θ is deliberately **not** in this table: there the quadrature needs
# a working precision above θ·log₁₀(1/(1 − t)) — a = 1 − ū^θ rounds to exactly
# 1 below it, and φ to exactly 0, silently — and at 60 digits the θ = 200 rows
# came out 2·10⁻³ high. That region is pinned instead by two pieces of exact
# algebra, ``τ(θ, 1) = 1 − 2/(θ + 2)`` and ``τ(1, δ) = δ/(δ + 2)``, which need
# no quadrature at all (the two tests below). A reference is a measurement too.
_TAU_MPMATH = [
    (1.0, 0.01, 0.004975124378109453),
    (1.0, 0.3, 0.13043478260869565),
    (1.0, 1.0, 0.3333333333333333),
    (1.0, 4.0, 0.6666666666666666),
    (1.2, 0.01, 0.1065271582612787),
    (1.2, 0.3, 0.20767356458175116),
    (1.2, 1.0, 0.375),
    (1.2, 4.0, 0.665059205500382),
    (1.9999999, 0.01, 0.35707828461387523),
    (1.9999999, 0.3, 0.4091481640570784),
    (1.9999999, 1.0, 0.4999999874999997),
    (1.9999999, 4.0, 0.6791666638159723),
    (2.0, 0.01, 0.3570783066151538),
    (2.0, 0.3, 0.4091481824483252),
    (2.0, 1.0, 0.5),
    (2.0, 4.0, 0.6791666666666667),
    (2.0000001, 0.01, 0.3570783286164307),
    (2.0000001, 0.3, 0.40914820083957074),
    (2.0000001, 1.0, 0.5000000124999997),
    (2.0000001, 4.0, 0.6791666695173612),
    (3.0, 0.01, 0.5190669430313229),
    (3.0, 0.3, 0.5479561550348269),
    (3.0, 1.0, 0.6),
    (3.0, 4.0, 0.711038961038961),
    (9.0, 0.01, 0.8047403460124568),
    (9.0, 0.3, 0.8093832330362418),
    (9.0, 1.0, 0.8181818181818182),
    (9.0, 4.0, 0.8394654347467415),
]


@pytest.mark.parametrize("theta,delta,tau", _TAU_MPMATH)
def test_tau_closed_form_matches_the_mpmath_quadrature(theta, delta, tau):
    """(★) against the Genest–MacKay integral of BB7's own generator.

    The quadrature's own accuracy, not the closed form's, sets the tolerance.
    """
    assert _bb7_tau_from_theta(theta, delta) == pytest.approx(tau, rel=1e-13, abs=1e-15)


@pytest.mark.parametrize("delta", [0.001, 0.01, 0.3, 1.0, 2.0, 4.0, 10.0])
def test_theta_one_is_claytons_tau_exactly(delta):
    """τ(1, δ) = δ/(δ + 2) — Clayton's own closed form, bit for bit."""
    assert _bb7_tau_from_theta(1.0, delta) == delta / (delta + 2.0)


@pytest.mark.parametrize("theta", [1.0, 1.2, 2.0, 3.7, 8.0, 50.0, 200.0, 1e6])
def test_delta_one_collapses_to_one_minus_two_over_theta_plus_two(theta):
    """A second algebraic identity the Beta form must reproduce, and the one
    the 60-digit quadrature failed at θ ≥ 50: at δ = 1,
    ``e^L = 2/(1 + 2/θ)`` and (★) telescopes to ``τ = 1 − 2/(θ + 2)``."""
    assert _bb7_tau_from_theta(theta, 1.0) == pytest.approx(
        1.0 - 2.0 / (theta + 2.0), rel=1e-14, abs=1e-15)


@pytest.mark.parametrize("delta", [0.01, 0.3, 1.0, 4.0, 10.0])
def test_the_removable_singularity_at_theta_two(delta):
    """θ = 2 is 0/0 in (★): both ``2 − θ`` and the bracket vanish. The limit
    is ``1 − [ψ(δ + 2) − ψ(2)]/δ``, and the implementation must be continuous
    across it — not merely finite *at* it."""
    from scipy.special import digamma
    limit = 1.0 - (digamma(2.0 + delta) - digamma(2.0)) / delta
    assert _bb7_tau_from_theta(2.0, delta) == pytest.approx(limit, rel=1e-14)
    for eps in (1e-3, 1e-6, 1e-9, 1e-12, 1e-15):
        for theta in (2.0 - eps, 2.0 + eps):
            assert _bb7_tau_from_theta(theta, delta) == pytest.approx(
                limit, rel=eps + 2e-14)


@pytest.mark.parametrize("theta", [1.2, 2.0, 3.0, 7.0, 40.0])
def test_delta_to_zero_is_joes_tau(theta):
    """The second, weaker 0/0 of (★): δ → 0 is Joe at the same θ."""
    target = _joe_tau_from_theta(theta)
    for delta in (1e-6, 1e-9, 1e-12):
        assert _bb7_tau_from_theta(theta, delta) == pytest.approx(
            target, rel=0.0, abs=2.0 * delta + 1e-15)


@pytest.mark.parametrize("tau,theta", _CASES)
def test_tau_round_trip(tau, theta):
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    assert cop.tau_of() == pytest.approx(tau, rel=0.0, abs=1e-12)
    assert cop.params["tau_k"] == tau                 # stored, not re-derived


def test_every_tau_is_reachable_at_theta_one_and_none_above_theta_joe():
    """The reachable-τ statement: (0, 1) at θ = 1, and (τ_Joe(θ), 1) at θ."""
    for tau in (1e-9, 1e-4, 0.01, 0.3, 0.5, 0.9, 0.999):
        assert CopulaBB7(tau_k=tau).tau_of() == pytest.approx(tau, abs=1e-12)
    for theta in (1.5, 2.0, 5.0, 10.0):
        ceiling = _joe_tau_from_theta(theta)
        with pytest.raises(CopulaParameterError):
            CopulaBB7(tau_k=ceiling * (1.0 - 1e-6), theta7=theta)
        CopulaBB7(tau_k=min(ceiling + 1e-6, 0.999), theta7=theta)


# --------------------------------------------------------------------------
# Sub-model limits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [0.1, 0.35, 0.5, 0.75, 0.95])
def test_theta_one_is_clayton_to_machine_precision(tau):
    b7, cl = CopulaBB7(tau_k=tau, theta7=1.0), CopulaClayton(tau_k=tau)
    assert b7.theta == 1.0
    assert b7.delta7 == 2.0 * tau / (1.0 - tau)      # Clayton's closed form, bit for bit
    np.testing.assert_allclose(b7.cdf_array(_UV), cl.cdf_array(_UV),
                               rtol=0.0, atol=1e-15)
    a, b = b7.logpdf_array(_UV), cl.logpdf_array(_UV)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() == a.size
    assert np.max(np.abs(a - b)) < 1e-11
    assert b7.tail_dependence() == pytest.approx(cl.tail_dependence(), abs=1e-14)


@pytest.mark.parametrize("tau", [0.1, 0.35, 0.5, 0.75, 0.95])
def test_theta_at_its_maximum_is_joe(tau):
    """BB7's Joe end is the *upper* end of its θ range, where the recovered
    δ → 0. It is not attained exactly (δ = 0 is not a BB7), so what must hold
    is agreement at the ``_DELTA_FLOOR`` stand-in, not equality."""
    theta = CopulaBB7.theta_max(tau)
    b7, joe = CopulaBB7(tau_k=tau, theta7=theta), CopulaJoe(tau_k=tau)
    assert b7.theta == pytest.approx(joe.theta, rel=1e-14)
    assert b7.delta7 <= 2.0 * _DELTA_FLOOR
    assert b7.tail_dependence()[0] == 0.0             # λ_L = 2^{−10¹⁴}, Joe's 0
    np.testing.assert_allclose(b7.cdf_array(_UV), joe.cdf_array(_UV),
                               rtol=0.0, atol=1e-13)
    a, b = b7.logpdf_array(_UV), joe.logpdf_array(_UV)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() == a.size
    assert np.max(np.abs(a - b)) < 1e-9


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_the_joe_limit_is_first_order_in_delta(tau):
    """Approaching the Joe end from inside: the error falls linearly in the
    recovered δ. This is BB7's genuine limit statement — BB6's δ = 1 ≡ Joe is
    an equality, BB7's is not."""
    joe = CopulaJoe(tau_k=tau)
    prev = None
    for delta in (1e-6, 1e-9, 1e-12):
        b7 = CopulaBB7(tau_k=tau, theta7=_theta_at_delta(tau, delta))
        err = float(np.max(np.abs(b7.cdf_array(_UV) - joe.cdf_array(_UV))))
        assert err < 5.0 * delta + 1e-13
        if prev is not None:
            assert err < max(prev * 1e-2, 1e-13)
        prev = err


def test_theta_one_and_delta_to_zero_is_independence():
    cop = CopulaBB7(tau_k=1e-12)                      # θ = 1, δ = 2·10⁻¹²
    uv = np.array([[0.2, 0.8], [0.5, 0.5], [0.01, 0.99]])
    assert np.max(np.abs(cop.logpdf_array(uv))) < 1e-10
    assert cop.cdf([0.3, 0.7]) == pytest.approx(0.21, abs=1e-11)


# --------------------------------------------------------------------------
# C, h, ln c against the adaptive-precision mpmath ground truth
# --------------------------------------------------------------------------

# (τ, θ, u, v, C, h(v|u), ln c). C from φ^{-1}(φ(u) + φ(v)) alone, at the δ the
# constructor recovers; h and c its first and mixed central differences at a
# step 10⁻²⁰ of the distance to the edge; the working precision raised over
# (120 … 15360) digits until two successive ones agree to 12 digits,
# **starting** above 1.2·θ·log₁₀(1/ū) + 60 digits and never accepting an exact
# 0 below 1920 — below that a = 1 − ū^θ is exactly 1 and C comes out as
# exactly 1, and a difference quotient of 10⁻²⁴² comes out as exactly 0, at
# every precision tried, so the ladder "converges" on a wrong value (it did,
# twice, until both guards were added). The precision each row converged at is
# in the comment.
_POINT_MPMATH = [
    (0.3, 1.2, 0.3, 0.7, 0.25890014427080404, 0.7825602344747655, -0.09206857751970132),      # δ = 0.642612812, 240
    (0.3, 1.2, 0.7, 0.3, 0.25890014427080404, 0.18161881851935624, -0.09206857751970132),     # exchangeable, 240
    (0.4, 1.0, 1e-12, 1e-06, 9.999999925000002e-13, 0.9999999825000004, -3.7578723531008893),  # θ = 1 (Clayton), δ = 4/3, 240
    (0.7, 5.0, 0.999999, 0.5, 0.5, 1.5231302459163904e-23, -50.39859283633642),               # near the Joe end, δ = 0.55, 240
    (0.9, 3.0, 1e-12, 0.999999999999, 1e-12, 1.0, -1236.7186819572776),                       # δ = 44.71428231, 1920
    (0.9, 3.0, 0.9, 0.95, 0.8961678456314517, 0.9224922465077589, 1.4444943185644974),        # 240
    (0.95, 4.0, 1e-06, 1e-06, 9.974746811239448e-07, 0.49873733867276276, 18.043939945404958),  # δ = 274.1327626, 240
    (0.2, 1.0, 0.01, 0.99, 0.009989931978245265, 0.9984901769195805, -1.8840590997880746),    # 240
    (1e-4, 1.0, 1e-12, 0.999999999999, 9.999999999990056e-13, 0.9999999999990054, -0.005326756898203617),  # 240
    (0.99, 2.0, 0.999999, 0.999999, 0.9999985857864377, 0.7071067808693056, 12.775789788740642),  # δ = 597.1971691, 240
    (0.1, 1.05, 0.5, 0.5, 0.2736946087706551, 0.4908284313697481, 0.03286239331183712),       # 240
    (0.6, 2.0, 1e-12, 1e-12, 7.362712735729538e-13, 0.36813563678642836, 27.1215320730501),   # δ = 2.264027846, 240
]


@pytest.mark.parametrize("tau,theta,u,v,C,h,logc", _POINT_MPMATH)
def test_kernel_matches_mpmath(tau, theta, u, v, C, h, logc):
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    assert cop.cdf([u, v]) == pytest.approx(C, rel=1e-12)
    assert cop.conditional_cdf(v, u) == pytest.approx(h, rel=1e-11)
    assert float(cop.logpdf_array(np.array([[u, v]]))[0]) == pytest.approx(
        logc, rel=1e-12, abs=1e-12)
    assert cop.pdf([u, v]) == pytest.approx(math.exp(logc), rel=1e-11)


@pytest.mark.parametrize("tau,theta", _CASES)
def test_no_nan_on_the_edge_grid_and_pdf_is_exp_of_logpdf(tau, theta):
    """RB-8/RB-9: the whole [10⁻¹², 1 − 10⁻¹²] grid evaluates, and the linear
    density is exactly ``exp`` of the log one (no ``EPS`` floor)."""
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    logc = cop.logpdf_array(_UV)
    assert not np.any(np.isnan(logc))
    ok = np.isfinite(logc) & (logc > -700.0) & (logc < 700.0)
    np.testing.assert_allclose(cop.pdf_array(_UV)[ok], np.exp(logc[ok]), rtol=1e-12)
    scalar = np.array([cop.pdf([u, v]) for u, v in _UV])
    np.testing.assert_allclose(scalar[ok], np.exp(logc[ok]), rtol=1e-9)
    C = cop.cdf_array(_UV)
    assert np.all((C >= 0.0) & (C <= 1.0)) and not np.any(np.isnan(C))


def test_log_x_minus_one_asymptotic_branch_is_continuous():
    """``ln(a^{−δ} − 1) = ln δ + ln(−ln a)`` once ``δ(−ln a)`` underflows:
    without it ``logpdf_array`` is −∞ on the whole u = 1 − 10⁻¹² edge as soon
    as θ ≳ 20, because ``lx + ln1mexp(lx)`` is ``0 + (−∞)`` there."""
    for de in (0.01, 1.0, 7.0):
        for lp in (-690.0, -699.999, -700.0, -700.001, -710.0, -5000.0):
            got = float(_bb7_log_x_minus_one(np.array([lp]), de)[0])
            assert got == pytest.approx(math.log(de) + lp, rel=1e-12)
        # ordinary middle, against the definition
        for lp in (-5.0, 0.0, 3.0):
            lx = de * math.exp(lp)
            assert float(_bb7_log_x_minus_one(np.array([lp]), de)[0]) == pytest.approx(
                math.log(math.expm1(lx)), rel=1e-13)


def test_log_k_stays_finite_on_the_upper_corner():
    """``ln K = ln(T − 1) − ln δ`` once ``ln T`` itself underflows to 0.0 —
    the (1 − 10⁻¹², 1 − 10⁻¹²) corner at τ = 0.95, θ = 4."""
    cop = CopulaBB7(tau_k=0.95, theta7=4.0)
    ka = np.array([math.log1p(-(1 - 1e-12))])
    _, _, lt, lk, _ = _bb7_terms_k(ka, ka, cop.theta, cop.delta7)
    assert np.all(np.isfinite(lk)) and np.all(lk < 0.0)
    assert np.all(np.isfinite(lt)) and np.all(lt >= 0.0)
    # and the branch itself: ln(T − 1) − ln δ is ln((T − 1)/δ), the first term
    # of ln(1 − e^{−lnT/δ}) when ln T is (T − 1) to rounding
    for ltm1 in (-699.9, -700.1):
        assert ltm1 - math.log(3.0) == pytest.approx(
            math.log(math.exp(ltm1) / 3.0), rel=1e-12)


# --------------------------------------------------------------------------
# Tail dependence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,theta", _CASES)
def test_tail_dependence_matches_the_diagonal_limits(tau, theta):
    """λ_L = 2^{−1/δ} and λ_U = 2 − 2^{1/θ}, against the diagonal ratios of C.

    λ_L is taken on the **kernel**, which is not clipped at ε: the public
    ``cdf`` is, and the approach of ``C(u, u)/u`` to its limit is only
    algebraic — at δ = 0.55 the ratio is still 3 % high at u = 10⁻⁹ — so
    10⁻²⁵⁰ is needed to see it (BB6's round measured the same for its own
    lower tail).
    """
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    lam_L, lam_U = cop.tail_dependence()
    assert lam_L == pytest.approx(2.0 ** (-1.0 / cop.delta7), rel=1e-15)
    assert lam_U == pytest.approx(2.0 - 2.0 ** (1.0 / cop.theta), rel=1e-15)
    u = 1.0 - 1e-9
    assert (1.0 - 2.0 * u + cop.cdf([u, u])) / (1.0 - u) == pytest.approx(lam_U, abs=2e-4)
    ratios = []
    for k in (9, 30, 100, 250):
        u = 10.0 ** (-k)
        ka = np.array([math.log1p(-u)])
        ratios.append(float(_bb7_cdf_k(ka, ka, cop.theta, cop.delta7)[0][0]) / u)
    assert ratios[-1] == pytest.approx(lam_L, rel=1e-9)
    # ... and getting there: each step is at least as close as the last
    errs = [abs(r - lam_L) for r in ratios]
    assert all(b <= a + 1e-12 for a, b in zip(errs, errs[1:])), errs


def test_each_tail_depends_on_one_parameter_only():
    """The analytic expression of the two-way nest: λ_U moves with θ alone and
    λ_L with δ alone, which is why Patton (2006) uses them as coordinates."""
    a = CopulaBB7(tau_k=0.5, theta7=2.0)
    b = CopulaBB7(tau_k=0.7, theta7=2.0)            # same θ, different δ
    assert a.tail_dependence()[1] == pytest.approx(b.tail_dependence()[1], rel=1e-15)
    assert a.tail_dependence()[0] != pytest.approx(b.tail_dependence()[0], rel=1e-3)
    # θ = 1 is Clayton: no upper tail at all, whatever τ
    for tau in (0.3, 0.6, 0.9):
        assert CopulaBB7(tau_k=tau, theta7=1.0).tail_dependence()[1] == 0.0


# --------------------------------------------------------------------------
# h, inv_h, sampling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,theta", _CASES)
def test_h_inv_h_round_trip_and_saturation(tau, theta):
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    rng = np.random.default_rng(_seed("inv_h", tau, theta))
    u = rng.uniform(1e-6, 1 - 1e-6, 400)
    w = rng.uniform(1e-6, 1 - 1e-6, 400)
    v = cop.inv_h_array(w, u)
    back = np.array([cop.conditional_cdf(float(b), float(a)) for b, a in zip(v, u)])
    assert np.max(np.abs(back - w)) < 1e-9
    assert cop.inv_h(float(w[0]), float(u[0])) == pytest.approx(float(v[0]), rel=1e-12)
    for w_extreme in (0.0, 1.0):
        v_e = cop.inv_h(w_extreme, 0.5)
        assert EPS <= v_e <= 1.0 - EPS
        assert cop.conditional_cdf(v_e, 0.5) == pytest.approx(
            min(max(w_extreme, EPS), 1.0 - EPS), abs=1e-9)


@pytest.mark.parametrize("tau,theta", _CASES)
def test_h_is_a_cdf_in_v(tau, theta):
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    grid = np.linspace(1e-6, 1 - 1e-6, 200)
    h = np.array([cop.conditional_cdf(float(v), 0.37) for v in grid])
    assert np.all(np.diff(h) >= -1e-12)
    assert 0.0 <= h[0] and h[-1] <= 1.0


@pytest.mark.parametrize("tau,theta", _CASES)
def test_sample_margins_and_monte_carlo_tau(tau, theta):
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    s = cop.sample(n=20000, seed=_seed("sample", tau, theta))
    # 16 KS tests run here (8 cases × 2 margins), so the per-test level is
    # 0.001, not 0.01: at 0.01 one of the sixteen is expected to trip every
    # few runs on a *correct* sampler, and one did.
    assert kstest(s[:, 0], "uniform").pvalue > 0.001
    assert kstest(s[:, 1], "uniform").pvalue > 0.001
    # 20 000 pairs: sd of τ̂ is ≤ 0.006 over the whole range, so 0.02 is > 3 sd.
    assert kendalltau(s[:, 0], s[:, 1]).statistic == pytest.approx(tau, abs=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("tau,theta", [(0.3, 1.2), (0.7, 1.0), (0.9, 3.0)])
def test_monte_carlo_tau_agrees_with_the_closed_form(tau, theta):
    """(★) against simulation: 40 000 pairs × 5 seeds, |t| < 3 on the mean."""
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    vals = [kendalltau(*cop.sample(n=40000, seed=s).T).statistic
            for s in (111, 222, 333, 444, 555)]
    m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    assert abs(m - cop.tau_of()) < 3.0 * sd / math.sqrt(len(vals)) + 1e-3


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_fit_tau_method_is_itau_with_a_profile_mle(caplog):
    """FR-12: ``'tau'`` inverts Kendall's τ and fits θ by MLE at that τ.

    It used to log a warning and run the joint MLE (reported as ``'mle'``).
    """
    cop = CopulaBB7(tau_k=0.6, theta7=2.0)
    data = cop.sample(n=600, seed=4343)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.archimedean.bb7"):
        res = CopulaBB7.fit(data, method="tau")
    assert res.copula.theta >= 1.0
    assert res.method == "tau" and res.converged
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert res.tau_k == kendalltau(data[:, 0], data[:, 1])[0]      # itau: τ̂ itself
    # theta7 is the maximum of the likelihood at that τ (a profile MLE) …
    for h in (-1e-3, 1e-3):
        try:
            cop = CopulaBB7(tau_k=res.tau_k, theta7=res.copula.theta7 + h)
        except CopulaParameterError:
            continue
        assert float(np.sum(cop.logpdf_array(res.uv))) <= res.log_likelihood + 1e-9
    # … and not the joint MLE, which it used to be.
    mle = CopulaBB7.fit(data, method="mle")
    assert mle.log_likelihood >= res.log_likelihood and mle.tau_k != res.tau_k


@pytest.mark.parametrize("tau,theta,tol_tau,tol_theta", [
    (0.3, 1.2, 0.05, 0.40),
    (0.5, 1.5, 0.04, 0.55),
    (0.7, 2.5, 0.03, 0.90),
])
def test_fit_recovers_tau_and_theta(tau, theta, tol_tau, tol_theta):
    """n = 3000, one replicate per point: both parameters must come back.

    The tolerances are "did not stop at the start value" guards, not precision
    claims — θ is the weakly identified one.
    """
    cop = CopulaBB7(tau_k=tau, theta7=theta)
    data = cop.sample(n=3000, seed=_seed("fit", tau, theta))
    res = CopulaBB7.fit(data, method="mle")
    assert res.converged
    assert res.tau_k == pytest.approx(tau, abs=tol_tau)
    assert res.copula.theta == pytest.approx(theta, abs=tol_theta)


def test_fit_reaches_the_clayton_and_joe_boundaries():
    """On data from a sub-model the joint fit must go to that boundary, not
    stop in the interior: θ̂ → 1 on Clayton data, δ̂ → 0 on Joe data.

    Thresholds are "went to the right corner" guards, not precision claims.
    """
    clayton_data = CopulaClayton(tau_k=0.6).sample(n=3000, seed=515152)
    r = CopulaBB7.fit(clayton_data, method="mle")
    assert r.copula.theta < 1.15 and r.tau_k == pytest.approx(0.6, abs=0.05)

    joe_data = CopulaJoe(tau_k=0.6).sample(n=3000, seed=626263)
    r = CopulaBB7.fit(joe_data, method="mle")
    assert r.copula.delta7 < 0.35 and r.tau_k == pytest.approx(0.6, abs=0.05)


def test_fit_best_includes_bb7():
    from pmcprg.copulas import CopulaVirt
    data = CopulaBB7(tau_k=0.6, theta7=2.0).sample(n=800, seed=717172)
    res = CopulaVirt.fit_best(data, families=[CopulaBB7, CopulaJoe, CopulaClayton],
                              method="mle")
    assert [r.copula.__class__.__name__ for r in res] and not res.failures
    assert any(r.copula.__class__ is CopulaBB7 for r in res)


def test_robust_machinery_refuses_bb7_the_way_it_refuses_every_two_parameter_family():
    """``mle_tau_discrepancy_test`` and ``dpd_fit`` (FR-7) are **one-parameter**
    estimators — both refuse any family with a second parameter, by design and
    by their own module docstrings, because Kendall's τ does not identify it.
    BB7 must get that same documented ``NotImplementedError`` and not a crash:
    the check here is that BB7 is registered well enough for the refusal to be
    the *reasoned* one, exactly as for BB1, BB6 and Student.
    """
    from pmcprg.copulas import mle_tau_discrepancy_test
    from pmcprg.copulas._robust import dpd_fit
    uv = CopulaBB7(tau_k=0.6, theta7=2.0).sample(n=400, seed=818182)
    with pytest.raises(NotImplementedError, match="more than one parameter"):
        mle_tau_discrepancy_test(CopulaBB7, uv)
    with pytest.raises(NotImplementedError, match="one-parameter families only"):
        dpd_fit(CopulaBB7, uv, alpha=0.1)
    # ... and the message names BB7's own parameters, i.e. the registry entry
    # is what the refusal read.
    with pytest.raises(NotImplementedError, match="theta7"):
        dpd_fit(CopulaBB7, uv, alpha=0.1)


# --------------------------------------------------------------------------
# Sub-model likelihood-ratio tests (FR-4 machinery)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub,note_key", [(CopulaClayton, "theta7 = 1"),
                                          (CopulaJoe, "theta7 -> theta_Joe(tau)")])
def test_submodel_lr_test_accepts_bb7s_nestings(sub, note_key):
    from pmcprg.copulas import submodel_lr_test
    uv = sub(tau_k=0.6).sample(n=1500, seed=838384)
    t = submodel_lr_test(CopulaBB7, sub, uv)
    assert t.family == "CopulaBB7" and t.submodel == sub.__name__
    assert t.boundary and t.null_distribution == "0.5*chi2(0) + 0.5*chi2(1)"
    assert note_key in t.boundary_note
    assert set(t.full_params) == {"tau_k", "theta7"}
    assert t.statistic >= 0.0 and 0.0 <= t.p_value <= 1.0
    # data *from* the sub-model: the LR must not reject at 1 %
    assert t.p_value > 0.01


def test_submodel_lr_test_rejects_on_genuine_bb7_data():
    from pmcprg.copulas import submodel_lr_test
    uv = CopulaBB7(tau_k=0.75, theta7=3.0).sample(n=3000, seed=949495)
    for sub in (CopulaClayton, CopulaJoe):
        t = submodel_lr_test(CopulaBB7, sub, uv)
        assert t.statistic > 0.0


# --------------------------------------------------------------------------
# Multistart / ICE integration
# --------------------------------------------------------------------------

MODELS = "pmcprg/pmc/models"


def _model(blocks, fixture="pmc_gauss_k2.toml"):
    from pmcprg.pmc.model import PMCModel
    raw = PMCModel(f"{MODELS}/{fixture}").raw
    for blk, new in zip(raw["copulas"], blocks):
        blk.pop("tau", None)
        blk.update(new)
    return PMCModel.from_dict(raw)


@pytest.mark.parametrize("tau,theta", [(0.5, 1.5), (0.2, 1.0), (0.9, 5.0)])
@pytest.mark.parametrize("jitter", [0.1, 0.25, 0.5])
def test_jittered_starts_always_build(tau, theta, jitter):
    from pmcprg.pmc._estim_common import perturb_initial_model
    model = _model([{"name": "BB7", "tau": tau, "theta7": theta}] * 4)
    for seed in range(30):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=jitter)
        for blk in out.copula_blocks():
            params = CopulaBB7.constructible_params(
                {"tau_k": blk["tau"], "theta7": blk.get("theta7", _THETA_DEFAULT)})
            CopulaBB7(**params)


@pytest.mark.parametrize("mode", ["random", "sweep"])
@pytest.mark.parametrize("cands", [["Gauss", "BB7"], ["BB7", "Joe", "Clayton"]])
def test_family_starts_with_bb7_candidates_build(mode, cands):
    from pmcprg.pmc._estim_common import build_multistart_inits
    from pmcprg.pmc.model import PMCModel
    model = PMCModel(f"{MODELS}/pmc_gauss_k2.toml")
    for seed in range(4):
        cfg = {"n_starts": 1 + len(cands) ** 2 + 4, "multistart_seed": seed,
               "multistart_jitter": 0.25, "multistart_families": mode,
               "candidates": cands}
        for m, _ in build_multistart_inits(model, cfg, label="ICE"):
            for blk in m.copula_blocks():
                if blk["name"] == "BB7":
                    CopulaBB7(**CopulaBB7.constructible_params(
                        {"tau_k": blk["tau"],
                         "theta7": blk.get("theta7", _THETA_DEFAULT)}))


def test_ice_m_step_fits_theta7():
    """The ICE M-step must move θ off its start value (RB-3's failure mode)."""
    from pmcprg.pmc.ice import _fit_copula_params, _resolve_candidate
    uv = CopulaBB7(tau_k=0.8, theta7=3.0).sample(n=1500, seed=13)
    entry, cls = _resolve_candidate("BB7")
    assert cls is CopulaBB7
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(len(uv)))
    assert set(p) == {"tau_k", "theta7"}
    assert 1.8 < p["theta7"] < 5.0, p


def test_bb7_cases_are_covered_by_the_reference_file():
    """The decimal reference grid of ``test_copula_limits`` holds BB7."""
    import test_copula_limits as tcl
    keys = {k for k in tcl._file_table() if k[0] == "BB7"}
    assert len(keys) == len(tcl._TAIL_TAUS["BB7"])
