"""BB6 copula (FR-9) — generator, sub-models, τ, tails, joint constraint, fit.

Complements ``test_copula_limits.py`` (which adds BB6 to its high-precision
``decimal`` reference grid at four τ, its δ carried in the file's ``delta``
slot, two of the four cases being the Joe and Gumbel sub-models exactly) with
the checks specific to this family:

* the **closed form** ``τ(θ, δ) = 1 − (1 − τ_Joe(θ))/δ`` — the outer-power
  identity BB6 shares with BB1 — against a 60-digit ``mpmath`` Genest–MacKay
  quadrature ``1 + 4∫φ/φ′`` of BB6's *own* generator, and against a simulated
  Kendall's τ;
* C, h and ln c against an adaptive-precision ``mpmath`` ground truth (C from
  the generator alone, h and c as its finite differences at a step 10⁻²⁰ of
  the distance to the edge, the precision raised from 120 to 3840 digits until
  two successive ones agree to 12 digits — the recipe of
  ``test_copula_limits``, applied off-line and recorded below);
* δ = 1 ≡ Joe and θ = 1 ≡ Gumbel–Hougaard, to machine precision;
* λ_U = 2 − 2^{1/(θδ)} and λ_L = 0;
* the joint (τ, δ) constraint δ ≤ 1/(1 − τ), its repair hooks and the
  multistart draws — mirroring ``test_multistart_joint_constraints.py`` for
  BB1 — including the fact that BB6's upper end is *attained* (θ = 1), unlike
  BB1's;
* that ``delta6`` takes **its own** branch of ``_two_parameter_spec`` and not
  BB1's ``delta`` branch, which would fit BB6 with BB1's τ map;
* recovery of both τ and δ by ``fit``, and the ``method='tau'`` fallback.

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

from pmcprg.copulas import CopulaBB1, CopulaBB6, CopulaEnum, CopulaGH, CopulaJoe
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.copulas.archimedean.bb6 import (
    _DELTA_DEFAULT,
    _TAU_JOE_FLOOR,
    _bb6_cdf_k,
    _bb6_log_g,
    _bb6_logp,
)
from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_G = np.array([1e-12, 1e-6, 0.01, 0.3, 0.5, 0.9, 1 - 1e-6, 1 - 1e-12])
_UV = np.array([(u, v) for u in _G for v in _G])

# (τ, δ) points used throughout: interior, both sub-model ends, and the two
# corners of the (τ, δ) box the registry allows.
_CASES = [(0.2, 1.0), (0.3, 1.2), (0.4, 1.6), (0.5, 2.0),
          (0.7, 1.0), (0.7, 2.5), (0.9, 3.0), (0.9, 10.0)]


def _seed(*parts: object) -> int:
    """A reproducible seed from ``parts`` — ``zlib.crc32``, never ``hash()``."""
    return zlib.crc32(repr(parts).encode())


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    entry = CopulaEnum.BB6
    assert entry.value.SHORT_NAME == "BB6"
    assert entry.value.CLASS_NAME == "CopulaBB6"
    assert entry.klass is CopulaBB6
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "delta6"]
    assert entry.value.TAU_MIN_MAX == [EPS, 1.0]
    assert entry in CopulaEnum.available()
    assert CopulaBB6.n_params == 2
    assert CopulaEnum.from_short_name("BB6") is entry


def test_delta6_bounds_are_registered_and_do_not_reuse_bb1s():
    from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
    assert EXTRA_PARAM_BOUNDS_BY_PARAM["delta6"] == (1.0, 10.0, 1.0)
    assert EXTRA_PARAM_BOUNDS["CopulaBB6"] == {"delta6": (1.0, 10.0, 1.0)}
    # BB1's own entry is untouched, and BB6 does not appear under "delta".
    assert EXTRA_PARAM_BOUNDS["CopulaBB1"] == {"delta": (1.0, 10.0, 1.5)}
    assert "delta" not in CopulaEnum.BB6.value.PARAMETERS_SET_NAME


def test_two_parameter_spec_takes_its_own_branch_not_bb1s():
    """The silent-bug guard: ``_two_parameter_spec`` dispatches on the
    parameter *name*, so a BB6 registered under ``delta`` would be optimised
    with BB1's ``τ = 1 − 2/(δ(θ + 2))`` map. BB6's branch must produce its own
    kwargs key and its own τ map."""
    from pmcprg.copulas._fit import _two_parameter_spec

    p0, stages, params_of = _two_parameter_spec(CopulaBB6, CopulaEnum.BB6, 0.6)
    assert set(params_of(p0)) == {"tau_k", "delta6"}
    assert len(stages) == 2
    # Box = the admissible set: s = ln(θ − 1) and ln δ, every point buildable.
    (s_lo, s_hi), (ld_lo, ld_hi) = stages[0][2]
    assert s_lo == math.log(1e-10) and s_hi > 0.0
    assert ld_lo == 0.0 and ld_hi == pytest.approx(math.log(10.0))
    rng = np.random.default_rng(20260917)
    for _ in range(200):
        p = (rng.uniform(s_lo, s_hi), rng.uniform(1.0, 10.0))
        kw = params_of(p)
        cop = CopulaBB6(**kw)                       # must not raise
        assert 0.0 < kw["tau_k"] < 1.0
        # ... and the τ it returns is BB6's own (★), not BB1's map.
        theta = 1.0 + math.exp(min(max(p[0], s_lo), s_hi))
        assert kw["tau_k"] == pytest.approx(
            1.0 - (1.0 - _joe_tau_from_theta(theta)) / kw["delta6"], rel=1e-12)
        assert cop.theta >= 1.0

    # BB1's branch is unchanged and keyed on its own name.
    q0, _, pof1 = _two_parameter_spec(CopulaBB1, CopulaEnum.BB1, 0.6)
    assert set(pof1(q0)) == {"tau_k", "delta"}


# --------------------------------------------------------------------------
# Parameters, the joint (τ, δ) constraint
# --------------------------------------------------------------------------

def test_default_delta6_is_the_joe_member():
    """A block without δ must build at *every* registered τ — BB1's default
    1.5 is inadmissible below τ = 1/3, BB6's default 1.0 never is."""
    assert _DELTA_DEFAULT == 1.0
    for tau in (EPS, 1e-9, 1e-3, 0.1, 1 / 3, 0.5, 0.9, 1 - 1e-6):
        cop = CopulaBB6(tau_k=tau)
        assert cop.delta6 == 1.0
        assert cop.theta >= 1.0
        assert cop.params["delta6"] == 1.0


@pytest.mark.parametrize("tau", [1e-9, 1e-3, 0.1, 1 / 3, 0.5, 0.75, 0.9, 0.99])
def test_delta_max_is_the_largest_admissible_double(tau):
    """``delta_max`` must agree with the constructor to the last bit, and the
    end itself must be *attained* (θ = 1, the Gumbel member) — unlike BB1's."""
    d = CopulaBB6.delta_max(tau)
    cop = CopulaBB6(tau_k=tau, delta6=d)            # must not raise
    assert cop.theta == 1.0                         # exactly the Gumbel member
    assert d == pytest.approx(1.0 / (1.0 - tau), rel=1e-15)
    with pytest.raises(CopulaParameterError):
        CopulaBB6(tau_k=tau, delta6=float(np.nextafter(d, 2.0 * d)))


@pytest.mark.parametrize("tau,delta", [(0.2, 2.0), (0.5, 3.0), (0.1, 1.5), (0.9, 11.0)])
def test_refused_pairs(tau, delta):
    with pytest.raises(CopulaParameterError, match="tau >= 1 - 1/delta6"):
        CopulaBB6(tau_k=tau, delta6=delta)


@pytest.mark.parametrize("bad", [0.5, 0.0, -1.0, float("nan")])
def test_delta6_below_one_is_refused(bad):
    with pytest.raises(CopulaParameterError):
        CopulaBB6(tau_k=0.5, delta6=bad)


def test_accepted_pairs_are_returned_untouched():
    """``constructible_params`` is the identity — same object — on every pair
    the constructor accepts, and only then (the hook's contract)."""
    rng = np.random.default_rng(20260901)
    taus = np.concatenate([[EPS, 1e-9, 0.1, 1 / 3, 0.5, 0.9, 0.9999],
                           rng.uniform(0, 1, 40)])
    n_accepted = 0
    for tau in taus:
        hi = CopulaBB6.delta_max(float(tau))
        for delta in [1.0, 1.5, 3.0, 10.0, hi, float(np.nextafter(hi, 0.0)),
                      float(np.nextafter(hi, 2.0 * hi)), float(rng.uniform(1.0, 10.0))]:
            params = {"tau_k": float(tau), "delta6": float(delta)}
            try:
                CopulaBB6(**params)
                refused = False
            except CopulaParameterError:
                refused = True
            out = CopulaBB6.constructible_params(params)
            assert (out is params) == (not refused)
            n_accepted += out is params
    assert n_accepted > 100


@pytest.mark.parametrize("tau", [EPS, 1e-9, 1e-6, 0.05, 0.2, 1 / 3, 0.5, 0.8, 0.9, 0.9999])
@pytest.mark.parametrize("delta", [0.5, 1.5, 3.0, 10.0, None])
def test_refused_pairs_move_only_delta_onto_the_admissible_end(tau, delta):
    params = {"tau_k": tau} if delta is None else {"tau_k": tau, "delta6": delta}
    try:
        CopulaBB6(**params)
        return                                       # accepted: nothing to repair
    except CopulaParameterError:
        pass
    out = CopulaBB6.constructible_params(params)
    assert out is not params and out["tau_k"] == tau
    cop = CopulaBB6(**out)                           # must build
    assert 1.0 <= out["delta6"] <= CopulaBB6.delta_max(tau)
    if delta is not None and delta < 1.0:
        assert out["delta6"] == 1.0                  # pulled up to the Joe end
    else:
        # BB6's upper end is admissible, so the repair sits *on* it (θ = 1),
        # without BB1's TAU_PAD_REL pull-in.
        assert out["delta6"] == CopulaBB6.delta_max(tau)
        assert cop.theta == 1.0


def test_constrain_params_projects_into_the_admissible_interval():
    for tau, delta in ((0.2, 5.0), (0.5, 0.1), (0.9, 100.0), (0.4, 1.2)):
        out = CopulaBB6.constrain_params({"tau_k": tau, "delta6": delta})
        assert 1.0 <= out["delta6"] <= CopulaBB6.delta_max(tau)
        CopulaBB6(**out)
    # a missing δ is the default, and the input dict is never mutated
    src = {"tau_k": 0.2}
    assert CopulaBB6.constrain_params(src)["delta6"] == 1.0
    assert src == {"tau_k": 0.2}


def test_a_delta_one_ulp_below_delta_max_still_builds():
    """The regression this family's own round found: ``τ_Joe = 1 − δ(1 − τ)``
    lands on exactly ``ε/2`` for a δ one ulp below ``delta_max``, and Joe's
    Brent bracket ``[τ/(1 − τ), (1 + τ)/(1 − τ)]`` has no sign change there —
    ``1 + τ/(1 − τ)`` rounds *up* to 1 + ε, whose τ already exceeds the
    target — so the constructor used to raise ``ValueError`` (not even a
    ``CopulaParameterError``) on 37 of the 376 pairs the hook test draws, and
    on 5 of the 9 jittered-multistart configurations.
    """
    assert 0.0 < _TAU_JOE_FLOOR < 2e-16
    # no theta > 1 is representable below the floor
    assert _joe_tau_from_theta(float(np.nextafter(1.0, 2.0))) == _TAU_JOE_FLOOR
    rng = np.random.default_rng(31415)
    taus = np.concatenate([[1e-9, 0.1, 1 / 3, 0.5, 0.9, 0.9999], rng.uniform(0, 1, 60)])
    for tau in taus:
        d_max = CopulaBB6.delta_max(float(tau))
        for delta in (d_max, float(np.nextafter(d_max, 0.0)),
                      float(np.nextafter(np.nextafter(d_max, 0.0), 0.0))):
            if delta < 1.0:
                continue
            cop = CopulaBB6(tau_k=float(tau), delta6=delta)   # must not raise
            assert cop.theta >= 1.0
            # the snap moves τ by less than one ulp of τ
            assert cop.tau_of() == pytest.approx(float(tau), rel=4e-16, abs=1e-15)


def test_bb6_is_not_treated_as_a_delta_family_by_the_bb1_guard():
    """``test_multistart_joint_constraints`` asserts that every family without
    a ``delta`` parameter returns an arbitrary dict untouched — BB6 must."""
    params = {"tau_k": 0.3, "df": 4.0}
    assert CopulaBB6.constructible_params(params) is params


# --------------------------------------------------------------------------
# Kendall's τ — the outer-power closed form (★)
# --------------------------------------------------------------------------

# 60-digit mpmath: 1 + 4·mp.quad(φ/φ′) with φ(t) = (−ln(1 − (1 − t)^θ))^δ and
# φ′ a high-precision numerical derivative of φ itself — BB6's own generator,
# not the identity (★) under test. τ(θ, 1) is Joe's own τ and τ(1, δ) = 1 − 1/δ
# is Gumbel's, so the table also pins both sub-models.
#
# The mesh is graded towards t = 1 (split points 1 − 10⁻ᵏ, k = 1…14), where
# φ/φ′ concentrates once θ is large: on the plain [0, ¼, ½, ¾, 1] split the
# rows at θ = 50 came out 8·10⁻⁵ too high, and it was the *quadrature* that
# was wrong, not (★) — the graded rule is stable to all 17 digits from k = 6
# to k = 14, and lands on the closed form. Recorded here as a reminder that a
# reference is a measurement too.
_TAU_MPMATH = [
    # (theta, delta, tau)
    (1.0, 1.0, 0.0),
    (1.0, 1.4, 0.2857142857142857),
    (1.0, 2.6, 0.6153846153846154),
    (1.0, 5.0, 0.8),
    (1.0000001, 1.0, 5.797362290384311e-08),
    (1.0000001, 1.4, 0.28571432712401634),
    (1.0000001, 2.6, 0.6153846376821627),
    (1.0000001, 5.0, 0.8000000115947246),
    (1.2, 1.0, 0.10254687721263903),
    (1.2, 1.4, 0.358962055151885),
    (1.2, 2.6, 0.6548257220048611),
    (1.2, 5.0, 0.8205093754425278),
    (2.0, 1.0, 0.35506593315177354),
    (2.0, 1.4, 0.539332809394124),
    (2.0, 2.6, 0.7519484358276052),
    (2.0, 5.0, 0.8710131866303548),
    (3.7, 1.0, 0.5893378986240857),
    (3.7, 1.4, 0.7066699275886327),
    (3.7, 2.6, 0.8420530379323407),
    (3.7, 5.0, 0.9178675797248171),
    (8.0, 1.0, 0.7832540438417558),
    (8.0, 1.4, 0.8451814598869685),
    (8.0, 2.6, 0.9166361707083677),
    (8.0, 5.0, 0.9566508087683512),
    (50.0, 1.0, 0.9609975327493626),
    (50.0, 1.4, 0.9721410948209733),
    (50.0, 2.6, 0.9849990510574471),
    (50.0, 5.0, 0.9921995065498725),
    (200.0, 1.0, 0.9900639414851804),
    (200.0, 1.4, 0.9929028153465574),
    (200.0, 2.6, 0.9961784390327617),
    (200.0, 5.0, 0.9980127882970361),
]


@pytest.mark.parametrize("theta,delta,tau", _TAU_MPMATH)
def test_tau_closed_form_matches_the_mpmath_quadrature(theta, delta, tau):
    """(★) ``τ = 1 − (1 − τ_Joe(θ))/δ`` against the Genest–MacKay integral.

    The quadrature's own accuracy, not the identity's, sets the tolerance:
    the graded-mesh rule holds 1e-13 relative over the whole table.
    """
    got = 1.0 - (1.0 - _joe_tau_from_theta(theta)) / delta
    assert got == pytest.approx(tau, rel=1e-13, abs=1e-15)


@pytest.mark.parametrize("delta", [1.0, 1.5, 2.0, 4.0, 10.0])
def test_theta_one_is_gumbels_tau(delta):
    """τ(1, δ) = 1 − 1/δ exactly — the Gumbel sub-model's own closed form."""
    assert _joe_tau_from_theta(1.0) == 0.0
    assert 1.0 - (1.0 - _joe_tau_from_theta(1.0)) / delta == 1.0 - 1.0 / delta


@pytest.mark.parametrize("tau,delta", _CASES)
def test_tau_round_trip(tau, delta):
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    assert cop.tau_of() == pytest.approx(tau, rel=0.0, abs=1e-12)
    assert cop.params["tau_k"] == tau                 # stored, not re-derived


def test_tau_is_increasing_in_theta_and_in_delta():
    thetas = [1.0, 1.0 + 1e-9, 1.01, 1.3, 2.0, 5.0, 30.0, 1e3, 1e6]
    deltas = [1.0, 1.2, 2.0, 3.5, 7.0, 10.0]
    tab = np.array([[1.0 - (1.0 - _joe_tau_from_theta(t)) / d for d in deltas]
                    for t in thetas])
    assert np.all(np.diff(tab, axis=0) > 0.0)         # strictly ↑ in θ
    assert np.all(np.diff(tab, axis=1) > 0.0)         # strictly ↑ in δ


def test_every_tau_is_reachable_at_delta_one_and_none_below_one_minus_one_over_delta():
    """The reachable-τ statement: (0, 1) at δ = 1, and [1 − 1/δ, 1) at δ."""
    for tau in (1e-9, 1e-4, 0.01, 0.3, 0.5, 0.9, 0.999):
        assert CopulaBB6(tau_k=tau).tau_of() == pytest.approx(tau, abs=1e-12)
    for delta in (1.5, 2.0, 5.0, 10.0):
        floor = 1.0 - 1.0 / delta
        with pytest.raises(CopulaParameterError):
            CopulaBB6(tau_k=floor * (1.0 - 1e-6), delta6=delta)
        CopulaBB6(tau_k=min(floor * (1.0 + 1e-9) + 1e-12, 0.999), delta6=delta)


# --------------------------------------------------------------------------
# Sub-model limits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [1e-6, 0.05, 0.3, 0.7, 0.95])
def test_delta_one_is_joe_to_machine_precision(tau):
    b6, joe = CopulaBB6(tau_k=tau, delta6=1.0), CopulaJoe(tau_k=tau)
    assert b6.theta == joe.theta                       # the very same θ
    np.testing.assert_allclose(b6.cdf_array(_UV), joe.cdf_array(_UV),
                               rtol=0.0, atol=1e-15)
    a, b = b6.logpdf_array(_UV), joe.logpdf_array(_UV)
    assert np.all(np.isfinite(a) == np.isfinite(b))
    ok = np.isfinite(a)
    assert np.max(np.abs(a[ok] - b[ok])) < 1e-11
    assert b6.tail_dependence() == pytest.approx(joe.tail_dependence(), abs=1e-14)


@pytest.mark.parametrize("tau", [0.05, 0.25, 0.5, 0.75, 0.9])
def test_theta_one_is_gumbel_to_machine_precision(tau):
    delta = CopulaBB6.delta_max(tau)
    b6, gh = CopulaBB6(tau_k=tau, delta6=delta), CopulaGH(tau_k=tau)
    assert b6.theta == 1.0
    assert b6.delta6 == pytest.approx(gh.theta, rel=1e-15)
    np.testing.assert_allclose(b6.cdf_array(_UV), gh.cdf_array(_UV),
                               rtol=0.0, atol=1e-15)
    a, b = b6.logpdf_array(_UV), gh.logpdf_array(_UV)
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() == a.size
    assert np.max(np.abs(a - b)) < 1e-11
    assert b6.tail_dependence() == pytest.approx(gh.tail_dependence(), abs=1e-14)


def test_theta_and_delta_both_one_is_independence():
    cop = CopulaBB6(tau_k=0.0 + EPS, delta6=1.0)
    uv = np.array([[0.2, 0.8], [0.5, 0.5], [0.01, 0.99]])
    assert np.max(np.abs(cop.logpdf_array(uv))) < 1e-14
    assert cop.cdf([0.3, 0.7]) == pytest.approx(0.21, abs=1e-14)


# --------------------------------------------------------------------------
# C, h, ln c against the adaptive-precision mpmath ground truth
# --------------------------------------------------------------------------

# (τ, δ, u, v, C, h(v|u), ln c). C from φ^{-1}(φ(u) + φ(v)) alone; h and c its
# first and mixed central differences at a step 10⁻²⁰ of the distance to the
# edge; the working precision raised over (120, 240, 480, 960, 1920, 3840)
# digits until two successive ones agree to 12 digits (the precision each row
# converged at is in the comment).
_POINT_MPMATH = [
    (0.4, 1.6, 0.3, 0.7, 0.2734218089002736, 0.8690229944395305, -0.23118782650015338),          # θ = 1.0722803998, 240
    (0.4, 1.6, 0.7, 0.3, 0.2734218089002736, 0.16713468892528827, -0.23118782650015338),         # exchangeable, 240
    (0.5, 2.0, 1e-12, 1e-06, 3.8334705834656405e-14, 0.03428760325749981, 9.669677707414639),    # θ = 1 (Gumbel), 240
    (0.7, 1.0, 0.999999, 0.5, 0.5, 3.558205662864408e-26, -56.380834809859074),                  # δ = 1 (Joe), 240
    (0.9, 3.0, 1e-12, 0.999999999999, 1e-12, 1.0, -430.0154355266134),                           # θ = 5.4637565990, 960
    (0.9, 3.0, 0.9, 0.95, 0.8999999290229054, 0.9999890757607531, -5.632056556608785),           # 240
    (0.95, 10.0, 1e-06, 1e-06, 4.000120787332138e-07, 0.2143610459695798, 12.157067096740068),   # θ = 2.8562572120, 240
    (0.2, 1.0, 0.01, 0.99, 0.009987017949961239, 0.9986988770083557, -1.6720756735917761),       # 240
    (1e-4, 1.0, 1e-12, 0.999999999999, 9.999999999990048e-13, 0.9999999999990048, -0.004594187114106873),  # 240
    (0.99, 1.0, 0.999999, 0.999999, 0.9999989965057261, 0.5017471369354732, 17.719520655438547),  # θ = 198.71295874, 240
    (0.1, 1.05, 0.5, 0.5, 0.2737401234085181, 0.5180750479196844, 0.040147593497883084),          # 240
    (0.6, 1.5, 1e-12, 1e-12, 1.4273438013996523e-19, 1.1328835259303579e-07, 11.418312355194011),  # 240
]


@pytest.mark.parametrize("tau,delta,u,v,C,h,logc", _POINT_MPMATH)
def test_kernel_matches_mpmath(tau, delta, u, v, C, h, logc):
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    assert cop.cdf([u, v]) == pytest.approx(C, rel=1e-12)
    assert cop.conditional_cdf(v, u) == pytest.approx(h, rel=1e-11)
    assert float(cop.logpdf_array(np.array([[u, v]]))[0]) == pytest.approx(
        logc, rel=1e-12, abs=1e-12)
    assert cop.pdf([u, v]) == pytest.approx(math.exp(logc), rel=1e-11)


@pytest.mark.parametrize("tau,delta", _CASES)
def test_no_nan_on_the_edge_grid_and_pdf_is_exp_of_logpdf(tau, delta):
    """RB-8/RB-9: the whole [10⁻¹², 1 − 10⁻¹²] grid evaluates, and the linear
    density is exactly ``exp`` of the log one (no ``EPS`` floor)."""
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    logc = cop.logpdf_array(_UV)
    assert not np.any(np.isnan(logc))
    ok = np.isfinite(logc) & (logc > -700.0) & (logc < 700.0)
    np.testing.assert_allclose(cop.pdf_array(_UV)[ok], np.exp(logc[ok]), rtol=1e-12)
    scalar = np.array([cop.pdf([u, v]) for u, v in _UV])
    np.testing.assert_allclose(scalar[ok], np.exp(logc[ok]), rtol=1e-9)
    C = cop.cdf_array(_UV)
    assert np.all((C >= 0.0) & (C <= 1.0)) and not np.any(np.isnan(C))


def test_log_p_asymptotic_branch_is_continuous():
    """``ln p = −z`` beyond z = 700 must join the ``ln1mexp`` branch smoothly:
    without it, ``(1 − u)^θ`` underflows and ``ln p`` is −∞ where the density
    is finite (the whole u = 1 − 10⁻¹² edge at θ ≳ 2000)."""
    for z in (690.0, 699.999, 700.0, 700.001, 710.0, 1e4):
        lp = float(_bb6_logp(np.array([-z]), 1.0)[0])
        assert lp == pytest.approx(-z, rel=1e-12)
    # and the transition itself is smooth to the last bits
    lo = float(_bb6_logp(np.array([-699.9999999]), 1.0)[0])
    hi = float(_bb6_logp(np.array([-700.0000001]), 1.0)[0])
    assert abs(lo - hi) < 1e-6


def test_log_g_small_branch_is_continuous():
    """``ln G = ln(1 − e^{−S})`` must equal ``ln S − S/2`` for tiny S, where
    ``exp(ln S)`` underflows — ``ln S = −1068`` occurs at τ = 0.95, δ = 1."""
    # Either side of the ln S = −20 switch the two branches must agree: the
    # first neglected term of the series is S²/24 ≤ 2·10⁻¹⁹ there.
    for lS in (-30.0, -20.000001, -20.0, -19.999999, -19.0):
        got = float(_bb6_log_g(np.array([lS]))[0])
        assert got == pytest.approx(lS - 0.5 * math.exp(lS), abs=1e-15)
    # far below the switch, where exp(ln S) underflows to 0.0
    assert float(_bb6_log_g(np.array([-1068.3]))[0]) == pytest.approx(-1068.3, rel=1e-15)
    # far above it, where G = 1 to rounding
    assert float(_bb6_log_g(np.array([40.0]))[0]) == pytest.approx(0.0, abs=1e-15)
    # and the ordinary middle, against the definition
    for lS in (-5.0, 0.0, 2.0):
        assert float(_bb6_log_g(np.array([lS]))[0]) == pytest.approx(
            math.log(-math.expm1(-math.exp(lS))), rel=1e-14)


# --------------------------------------------------------------------------
# Tail dependence
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,delta", _CASES)
def test_tail_dependence_matches_the_diagonal_limits(tau, delta):
    """λ_U = 2 − 2^{1/(θδ)} and λ_L = 0, against the diagonal ratios of C."""
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    lam_L, lam_U = cop.tail_dependence()
    assert lam_L == 0.0
    assert lam_U == pytest.approx(2.0 - 2.0 ** (1.0 / (cop.theta * cop.delta6)), rel=1e-15)
    u = 1.0 - 1e-9
    num = 1.0 - 2.0 * u + cop.cdf([u, u])
    assert num / (1.0 - u) == pytest.approx(lam_U, rel=2e-5)
    # λ_L = lim C(u, u)/u: taken on the *kernel*, which is not clipped at EPS
    # — the public cdf is, and u ≥ 2·10⁻¹⁶ is nowhere near the limit. The
    # decay is only exp(−S(2^{1/δ} − 1)) with S ≈ θ·ln(1/u), i.e. a power of u
    # whose exponent can be small: at τ = 0.9, δ = 10 (θ = 1) the ratio is
    # still 0.226 at u = 10⁻⁹, which is why 10⁻²⁵⁰ is needed to see the 0.
    ratios = []
    for k in (2, 6, 12, 30, 100, 250):
        u = 10.0 ** (-k)
        ka = np.array([math.log1p(-u)])
        ratios.append(float(_bb6_cdf_k(ka, ka, cop.theta, cop.delta6)[0][0]) / u)
    assert all(b < a for a, b in zip(ratios, ratios[1:]))
    assert ratios[-1] < 1e-15


def test_lambda_u_depends_only_on_the_product_theta_delta():
    """A consequence of the closed form, and the reason λ_U alone cannot
    separate the two parameters."""
    a = CopulaBB6(tau_k=0.5, delta6=2.0)              # θ = 1, θδ = 2
    b = CopulaBB6(tau_k=1.0 - (1.0 - _joe_tau_from_theta(2.0)) / 1.0, delta6=1.0)
    assert a.theta * a.delta6 == pytest.approx(b.theta * b.delta6, rel=1e-14)
    assert a.tail_dependence()[1] == pytest.approx(b.tail_dependence()[1], rel=1e-14)


# --------------------------------------------------------------------------
# h, inv_h, sampling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,delta", _CASES)
def test_h_inv_h_round_trip_and_saturation(tau, delta):
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    rng = np.random.default_rng(_seed("inv_h", tau, delta))
    u = rng.uniform(1e-6, 1 - 1e-6, 400)
    w = rng.uniform(1e-6, 1 - 1e-6, 400)
    v = cop.inv_h_array(w, u)
    back = np.array([cop.conditional_cdf(float(b), float(a)) for b, a in zip(v, u)])
    assert np.max(np.abs(back - w)) < 1e-9
    assert cop.inv_h(float(w[0]), float(u[0])) == pytest.approx(float(v[0]), rel=1e-12)
    # The two extreme w: ``inv_h`` clips w into [ε, 1 − ε] first, so what must
    # hold is the round trip, not saturation at the grid ends — h(·|u) already
    # attains ε and 1 − ε strictly inside (0, 1) for a strongly dependent
    # copula, and any v it attains them at is a correct inverse.
    for w_extreme in (0.0, 1.0):
        v_e = cop.inv_h(w_extreme, 0.5)
        assert EPS <= v_e <= 1.0 - EPS
        assert cop.conditional_cdf(v_e, 0.5) == pytest.approx(
            min(max(w_extreme, EPS), 1.0 - EPS), abs=1e-9)


@pytest.mark.parametrize("tau,delta", _CASES)
def test_h_is_a_cdf_in_v(tau, delta):
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    grid = np.linspace(1e-6, 1 - 1e-6, 200)
    h = np.array([cop.conditional_cdf(float(v), 0.37) for v in grid])
    assert np.all(np.diff(h) >= -1e-12)
    assert 0.0 <= h[0] and h[-1] <= 1.0


@pytest.mark.parametrize("tau,delta", _CASES)
def test_sample_margins_and_monte_carlo_tau(tau, delta):
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    s = cop.sample(n=20000, seed=_seed("sample", tau, delta))
    assert kstest(s[:, 0], "uniform").pvalue > 0.01
    assert kstest(s[:, 1], "uniform").pvalue > 0.01
    # 20 000 pairs: sd of τ̂ is ≤ 0.006 over the whole range, so 0.02 is > 3 sd.
    assert kendalltau(s[:, 0], s[:, 1]).statistic == pytest.approx(tau, abs=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("tau,delta", [(0.3, 1.2), (0.7, 2.5), (0.9, 3.0)])
def test_monte_carlo_tau_agrees_with_the_closed_form(tau, delta):
    """(★) against simulation: 40 000 pairs × 5 seeds, |t| < 3 on the mean."""
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    vals = [kendalltau(*cop.sample(n=40000, seed=s).T).statistic
            for s in (101, 202, 303, 404, 505)]
    m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    assert abs(m - cop.tau_of()) < 3.0 * sd / math.sqrt(len(vals)) + 1e-3


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def test_fit_tau_method_falls_back_to_mle(caplog):
    cop = CopulaBB6(tau_k=0.6, delta6=1.8)
    data = cop.sample(n=600, seed=4242)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.archimedean.bb6"):
        res = CopulaBB6.fit(data, method="tau")
    assert res.method == "mle"
    assert any("delta6" in r.getMessage() for r in caplog.records)
    assert res.copula.delta6 >= 1.0


@pytest.mark.parametrize("tau,delta,tol_tau,tol_delta", [
    (0.3, 1.2, 0.05, 0.40),
    (0.5, 1.5, 0.04, 0.45),
    (0.7, 2.5, 0.03, 0.70),
])
def test_fit_recovers_tau_and_delta(tau, delta, tol_tau, tol_delta):
    """n = 3000, one replicate per point: both parameters must come back.

    The tolerances are ≈ 4 × the RMSEs measured over 25 replicates at n = 3000
    (τ: 0.005–0.011; δ: 0.06–0.13), i.e. this is a "did not stop at the start
    value" guard, not a precision claim — δ is the weakly identified one.
    """
    cop = CopulaBB6(tau_k=tau, delta6=delta)
    data = cop.sample(n=3000, seed=_seed("fit", tau, delta))
    res = CopulaBB6.fit(data, method="mle")
    assert res.converged
    assert res.tau_k == pytest.approx(tau, abs=tol_tau)
    assert res.copula.delta6 == pytest.approx(delta, abs=tol_delta)
    assert res.copula.delta6 != EXTRA_PARAM_BOUNDS_BY_PARAM["delta6"][2]   # left the start


def test_fit_reaches_the_joe_and_gumbel_boundaries():
    """On data from a sub-model the joint fit must go to that boundary, not
    stop in the interior: δ̂ → 1 on Joe data, θ̂ → 1 on Gumbel data.

    On Joe data at n = 3000 the measured RMSE of δ̂ is 0.057 with a +0.039
    bias (δ cannot go below its boundary 1), so the threshold is δ̂ < 1.3 —
    ≈ 2 RMSE above the bias, a "went to the right corner" guard rather than a
    precision claim. θ̂ on Gumbel data is far sharper: it lands on the floor.
    """
    joe_data = CopulaJoe(tau_k=0.6).sample(n=3000, seed=515151)
    r = CopulaBB6.fit(joe_data, method="mle")
    assert r.copula.delta6 < 1.3 and r.tau_k == pytest.approx(0.6, abs=0.05)

    gh_data = CopulaGH(tau_k=0.6).sample(n=3000, seed=626262)
    r = CopulaBB6.fit(gh_data, method="mle")
    assert r.copula.theta < 1.05 and r.tau_k == pytest.approx(0.6, abs=0.05)


def test_fit_best_includes_bb6():
    from pmcprg.copulas import CopulaVirt
    data = CopulaBB6(tau_k=0.6, delta6=1.8).sample(n=800, seed=717171)
    res = CopulaVirt.fit_best(data, families=[CopulaBB6, CopulaJoe, CopulaGH],
                              method="mle")
    assert [r.copula.__class__.__name__ for r in res] and not res.failures
    assert any(r.copula.__class__ is CopulaBB6 for r in res)


# --------------------------------------------------------------------------
# Sub-model likelihood-ratio tests (FR-4 machinery)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sub,note_key", [(CopulaJoe, "delta6 = 1"),
                                          (CopulaGH, "theta = 1")])
def test_submodel_lr_test_accepts_bb6s_nestings(sub, note_key):
    from pmcprg.copulas import submodel_lr_test
    uv = sub(tau_k=0.6).sample(n=1500, seed=838383)
    t = submodel_lr_test(CopulaBB6, sub, uv)
    assert t.family == "CopulaBB6" and t.submodel == sub.__name__
    assert t.boundary and t.null_distribution == "0.5*chi2(0) + 0.5*chi2(1)"
    assert note_key in t.boundary_note
    assert set(t.full_params) == {"tau_k", "delta6"}
    assert t.statistic >= 0.0 and 0.0 <= t.p_value <= 1.0
    # data *from* the sub-model: the LR must not reject at 1 %
    assert t.p_value > 0.01


def test_submodel_lr_test_rejects_on_genuine_bb6_data():
    from pmcprg.copulas import submodel_lr_test
    uv = CopulaBB6(tau_k=0.75, delta6=2.5).sample(n=3000, seed=949494)
    for sub in (CopulaJoe, CopulaGH):
        t = submodel_lr_test(CopulaBB6, sub, uv)
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


@pytest.mark.parametrize("tau,delta", [(0.5, 1.5), (0.2, 1.0), (0.9, 5.0)])
@pytest.mark.parametrize("jitter", [0.1, 0.25, 0.5])
def test_jittered_starts_always_build(tau, delta, jitter):
    from pmcprg.pmc._estim_common import perturb_initial_model
    model = _model([{"name": "BB6", "tau": tau, "delta6": delta}] * 4)
    for seed in range(30):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=jitter)
        for blk in out.copula_blocks():
            CopulaBB6(tau_k=blk["tau"], delta6=blk.get("delta6", _DELTA_DEFAULT))


@pytest.mark.parametrize("mode", ["random", "sweep"])
@pytest.mark.parametrize("cands", [["Gauss", "BB6"], ["BB6", "Joe", "GH"]])
def test_family_starts_with_bb6_candidates_build(mode, cands):
    from pmcprg.pmc._estim_common import build_multistart_inits
    from pmcprg.pmc.model import PMCModel
    model = PMCModel(f"{MODELS}/pmc_gauss_k2.toml")
    for seed in range(4):
        cfg = {"n_starts": 1 + len(cands) ** 2 + 4, "multistart_seed": seed,
               "multistart_jitter": 0.25, "multistart_families": mode,
               "candidates": cands}
        for m, _ in build_multistart_inits(model, cfg, label="ICE"):
            for blk in m.copula_blocks():
                if blk["name"] == "BB6":
                    CopulaBB6(tau_k=blk["tau"], delta6=blk.get("delta6", _DELTA_DEFAULT))


@pytest.mark.parametrize("candidates", [["BB6", "Gauss"], ["Gauss", "BB6"]])
def test_selection_placeholder_builds(candidates):
    from pmcprg.pmc.ice import (
        FIT_FAILED_KEY,
        _copula_candidate_fits,
        _copula_placeholder,
    )
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=50), rng.uniform(size=50)
    resolved, _ = _copula_candidate_fits(candidates, u, v, np.ones(50), "mle")
    blk = _copula_placeholder(candidates, resolved, "mle",
                              {"name": candidates[0], "tau": 0.0})
    assert blk[FIT_FAILED_KEY] is True
    stored = {k: val for k, val in blk.items() if k != FIT_FAILED_KEY}
    _model([stored])


def test_ice_m_step_fits_delta6():
    """The ICE M-step must move δ off its start value (RB-3's failure mode)."""
    from pmcprg.pmc.ice import _fit_copula_params, _resolve_candidate
    uv = CopulaBB6(tau_k=0.8, delta6=3.0).sample(n=1500, seed=11)
    entry, cls = _resolve_candidate("BB6")
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(len(uv)))
    assert set(p) == {"tau_k", "delta6"}
    assert 2.0 < p["delta6"] < 4.2, p


def test_bb6_cases_are_covered_by_the_reference_file():
    """The decimal reference grid of ``test_copula_limits`` holds BB6."""
    import test_copula_limits as tcl
    keys = {k for k in tcl._file_table() if k[0] == "BB6"}
    assert len(keys) == len(tcl._TAIL_TAUS["BB6"])
