"""The full three-parameter Tawn copula (FR-9, round 5) — model, general
reachable-τ cap, joint repair, n-parameter fit, ICE/multistart, and the
non-regression of every family that was fitted by the same optimiser before.

What is checked here, and why it is here rather than in
``test_copula_limits.py``
-------------------------------------------------------------------------
``test_copula_limits.py``'s reference file is keyed by ``(family, θ, δ)`` —
**one** slot for a second parameter, carrying BB1's δ, Tawn 1/2's ψ and
t-EV's ν. A three-parameter family does not fit that key, and widening it
would mean changing the file's format, i.e. rewriting all ~5 900 stored rows
(the audit's own rule: pre-existing entries stay byte-identical). Rather than
encode a composite value in a slot that ``_param_sort_key`` parses with
``float()``, ``CopulaTawn3`` is deliberately **kept out of that grid** and
validated here against its own ``mpmath`` ground truth, on the *same* corner
grid ``(1e-12 … 1 − 1e-12)²`` and with the same independence-from-the-code
discipline: the oracle is written from ``C = exp(−ℓ)``, never from the
module's log-space kernel, and is itself cross-checked against
``mpmath.diff`` of ``C`` (agreement ≤ 3.4·10⁻¹² — see
``test_ground_truth_oracle_is_itself_validated``).

Two properties make that a fair trade rather than a gap:

* the evaluation kernel is *literally* the one round 3 validated on that
  grid for types 1 and 2 (``_tawn_parts`` always took both weights), and
* the type-1/type-2 restrictions are reproduced **bit for bit** at
  ``psi_u = 1`` / ``psi_v = 1`` (``test_restrictions_are_bit_identical``), so
  every stored Tawn 1/2 row is also a Tawn 3 row at a ψ = 1 edge.

Every seed below is an integer literal or ``zlib.crc32`` of a repr — never
``hash()``, which is salted per process (fixed once in commit 5e56fda).
"""
from __future__ import annotations

import math
import platform
import sys
import zlib

import numpy as np
import pytest
from scipy.stats import kendalltau

from pmcprg.copulas import (CopulaBB1, CopulaEnum, CopulaGH, CopulaStudent,
                            CopulaTawn1, CopulaTawn2, CopulaTawn3, CopulaTEV)
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM, TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.copulas.extreme_value.tawn import _THETA_HI, _tau_of, _tawn_tau_quad
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

_G = (1e-12, 1e-6, 0.3, 0.5, 1 - 1e-6, 1 - 1e-12)
_UV = np.array([(u, v) for u in _G for v in _G])

# (τ, ψ_u, ψ_v) grid, both ψ = 1 edges and the registered box's lower end.
_TRIPLES = [
    (0.2, 1.0, 1.0),        # the Gumbel corner
    (0.3, 1.0, 0.35),       # the type-1 face
    (0.3, 0.35, 1.0),       # the type-2 face
    (0.6, 0.7, 1.0),
    (0.5, 0.9, 0.8),
    (0.7, 0.95, 0.9),
    (0.05, 0.5, 0.2),
    (0.95, 0.99, 0.98),
    (1e-4, 0.5, 0.5),
    (0.15, 0.2, 0.9),
    (0.005, 0.01, 1.0),
    (0.004, 0.01, 0.5),
]


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) % (2 ** 31)


def _cap(pu: float, pv: float) -> float:
    """The cap written the *other* way round from the module's reciprocal
    form — ψ_uψ_v/(ψ_u + ψ_v − ψ_uψ_v), the Marshall–Olkin τ."""
    return pu * pv / (pu + pv - pu * pv)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

def test_registered_in_copula_enum():
    entry = CopulaEnum.TAWN3
    assert entry.value.SHORT_NAME == "Tawn3"
    assert entry.klass is CopulaTawn3
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "psi_u", "psi_v"]
    assert entry.value.TAU_MIN_MAX == [EPS, 1.0]
    assert entry in CopulaEnum.available()
    assert CopulaTawn3.n_params == 3          # AIC/BIC must penalise three


def test_both_weight_bounds_are_registered():
    from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
    for key in ("psi_u", "psi_v"):
        assert EXTRA_PARAM_BOUNDS_BY_PARAM[key] == (0.01, 1.0, 1.0)
    # The class-keyed ICE view is derived from the registry: two extras, not one.
    assert EXTRA_PARAM_BOUNDS["CopulaTawn3"] == {"psi_u": (0.01, 1.0, 1.0),
                                                 "psi_v": (0.01, 1.0, 1.0)}
    # …and the two-parameter types keep exactly one.
    assert EXTRA_PARAM_BOUNDS["CopulaTawn1"] == {"psi": (0.01, 1.0, 1.0)}


def test_extra_copula_keys_picks_up_both_weights():
    """``_estim_common._EXTRA_COPULA_KEYS`` is derived from the registry, so a
    family change must drop *both* weights, not one."""
    from pmcprg.pmc._estim_common import _EXTRA_COPULA_KEYS
    assert {"psi_u", "psi_v"} <= _EXTRA_COPULA_KEYS


# --------------------------------------------------------------------------
# The general reachable-τ cap
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pu,pv", [(1.0, 1.0), (1.0, 0.4), (0.4, 1.0), (0.6, 0.8),
                                   (0.3, 0.3), (0.9, 0.2), (0.05, 0.9), (0.5, 0.5),
                                   (0.99, 0.99), (0.01, 0.01), (0.2, 0.7)])
def test_cap_is_the_marshall_olkin_tau(pu, pv):
    """τ_∞ = ψ_uψ_v/(ψ_u + ψ_v − ψ_uψ_v), approached from below as θ → ∞.

    The module's reciprocal form must agree with the product form, and the
    quadrature must converge to it like 1/θ (the residual shrinks by ≈ 10 from
    θ = 10⁵ to θ = 10⁶).
    """
    cap = CopulaTawn3.reachable_tau_cap(pu, pv)
    assert cap == pytest.approx(_cap(pu, pv), rel=1e-15)
    t5, t6 = _tawn_tau_quad(1e5, pu, pv), _tawn_tau_quad(1e6, pu, pv)
    assert t5 < t6 < cap
    assert (cap - t6) / cap < 1.1e-6
    assert (cap - t5) / (cap - t6) == pytest.approx(10.0, rel=0.05)


def test_cap_reduces_to_psi_when_one_weight_is_one():
    """The round-3 statement "τ < ψ" is the ψ_other = 1 case of the general one."""
    for psi in (0.01, 0.1, 0.35, 0.5, 0.9, 0.999, 1.0):
        assert CopulaTawn3.reachable_tau_cap(1.0, psi) == psi
        assert CopulaTawn3.reachable_tau_cap(psi, 1.0) == psi


def test_cap_does_not_underflow_for_tiny_weights():
    """The reciprocal form survives weights whose product would underflow."""
    tiny = 1e-200
    assert CopulaTawn3.reachable_tau_cap(tiny, tiny) == pytest.approx(0.5 * tiny, rel=1e-12)
    assert _cap(tiny, tiny) == 0.0            # the product form is the one that fails


@pytest.mark.parametrize("tau,pu,pv", _TRIPLES)
def test_constructor_accepts_below_the_cap_and_refuses_at_or_above(tau, pu, pv):
    cap = CopulaTawn3.reachable_tau_cap(pu, pv)
    assert tau < cap
    CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)          # builds
    if cap < 1.0:
        with pytest.raises(CopulaParameterError, match="not reachable"):
            CopulaTawn3(tau_k=cap, psi_u=pu, psi_v=pv)
        with pytest.raises(CopulaParameterError, match="not reachable"):
            CopulaTawn3(tau_k=min(cap * 1.05, 0.999), psi_u=pu, psi_v=pv)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan"), float("inf")])
def test_weights_outside_the_unit_interval_are_refused(bad):
    for kw in ({"psi_u": bad}, {"psi_v": bad}):
        with pytest.raises(CopulaParameterError, match=r"psi_[uv] must be in \(0, 1\]"):
            CopulaTawn3(tau_k=0.3, **kw)


def test_booleans_and_non_numbers_are_refused():
    # As for the two-parameter types, a *parseable* string is accepted
    # (``float("0.5")``); what must be refused is a bool — which ``float``
    # would silently turn into 1.0 — and anything ``float`` cannot read.
    for bad in (True, False, None, "abc", [0.5]):
        for key in ("psi_u", "psi_v"):
            with pytest.raises(CopulaParameterError, match="is not a number"):
                CopulaTawn3(tau_k=0.3, **{key: bad})


def test_missing_weights_default_to_the_gumbel_corner():
    cop = CopulaTawn3(tau_k=0.6)
    assert (cop.psi_u, cop.psi_v) == (1.0, 1.0)
    assert cop.params["psi_u"] == 1.0 and cop.params["psi_v"] == 1.0
    assert cop.theta == CopulaGH(tau_k=0.6).theta          # Gumbel's closed form


def test_reachable_tau_bounds_are_the_gumbel_corner_cap():
    lo, hi = CopulaTawn3.reachable_tau_bounds()
    assert lo == EPS
    assert hi == _tau_of(_THETA_HI, 1.0, 1.0) == CopulaTawn1.reachable_tau_bounds()[1]


# --------------------------------------------------------------------------
# The two-parameter restrictions, bit for bit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,psi", [(0.3, 0.35), (0.7, 0.9), (0.05, 0.5),
                                     (0.95, 0.97), (1e-8, 0.05), (0.5, 1.0)])
def test_restrictions_are_bit_identical(tau, psi):
    """ψ_u = 1 *is* type 1 and ψ_v = 1 *is* type 2 — same floating-point
    operations, so equality is exact, not approximate."""
    for cls2, kw in ((CopulaTawn1, {"psi_u": 1.0, "psi_v": psi}),
                     (CopulaTawn2, {"psi_u": psi, "psi_v": 1.0})):
        a, b = cls2(tau_k=tau, psi=psi), CopulaTawn3(tau_k=tau, **kw)
        assert a.theta == b.theta
        assert np.array_equal(a.logpdf_array(_UV), b.logpdf_array(_UV))
        assert np.array_equal(a.cdf_array(_UV), b.cdf_array(_UV))
        assert a.tail_dependence() == b.tail_dependence()
        for u, v in _UV:
            assert a.conditional_cdf(v, u) == b.conditional_cdf(v, u)
        assert a.params["tau_k"] == b.params["tau_k"]      # same θ-cap clamping


def test_tail_dependence_general_form_matches_the_round_3_expression():
    """The cancellation-free general λ_U must *evaluate* round 3's expression
    when the larger weight is 1 — ``x*1.0`` and ``x/1.0`` are exact."""
    for tau, psi in ((0.3, 0.35), (0.7, 0.9), (0.95, 0.97)):
        cop = CopulaTawn1(tau_k=tau, psi=psi)
        th = cop.theta
        old = float(psi - math.expm1(math.log1p(psi ** th) / th))
        assert cop.tail_dependence() == (0.0, old)


def test_gumbel_corner_matches_gumbels_own_tail_dependence():
    for tau in (0.2, 0.6, 0.9):
        t3, gh = CopulaTawn3(tau_k=tau), CopulaGH(tau_k=tau)
        assert t3.theta == gh.theta
        assert t3.tail_dependence()[1] == pytest.approx(2.0 - 2.0 ** (1.0 / t3.theta),
                                                        rel=1e-14)
        assert t3.tail_dependence()[0] == 0.0


@pytest.mark.parametrize("tau,pu,pv", [(0.3, 0.35, 0.9), (0.5, 0.9, 0.7), (0.15, 0.2, 0.95)])
def test_transposing_the_weights_transposes_the_copula(tau, pu, pv):
    """C(u, v; ψ_u, ψ_v) = C(v, u; ψ_v, ψ_u) — swapping u, v swaps w, z."""
    a = CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)
    b = CopulaTawn3(tau_k=tau, psi_u=pv, psi_v=pu)
    assert a.theta == pytest.approx(b.theta, rel=1e-12)
    np.testing.assert_allclose(a.cdf_array(_UV), b.cdf_array(_UV[:, ::-1]),
                               rtol=1e-13, atol=1e-300)
    np.testing.assert_allclose(a.logpdf_array(_UV), b.logpdf_array(_UV[:, ::-1]),
                               rtol=0.0, atol=1e-12)


def test_an_asymmetric_member_is_not_exchangeable_pointwise():
    cop = CopulaTawn3(tau_k=0.5, psi_u=0.95, psi_v=0.6)
    d = max(abs(cop.cdf([u, v]) - cop.cdf([v, u]))
            for u, v in ((0.2, 0.7), (0.3, 0.9), (0.45, 0.55)))
    assert d > 1e-3, "an asymmetric Tawn 3 must be visibly non-exchangeable"


# --------------------------------------------------------------------------
# The joint repair
# --------------------------------------------------------------------------

def test_accepted_triples_are_returned_untouched():
    rng = np.random.default_rng(20260917)
    n_accepted = 0
    for _ in range(300):
        tau = float(rng.uniform(0.0, 1.0))
        pu, pv = (float(x) for x in rng.uniform(0.01, 1.0, 2))
        params = {"tau_k": tau, "psi_u": pu, "psi_v": pv}
        out = CopulaTawn3.constructible_params(params)
        try:
            CopulaTawn3(**params)
            refused = False
        except CopulaParameterError:
            refused = True
        assert (out is params) == (not refused)       # the hook's test IS the constructor's
        n_accepted += out is params
    assert n_accepted > 50


def test_a_dict_without_the_weights_is_returned_untouched():
    """The whole-registry identity contract of
    ``test_multistart_joint_constraints.test_accepted_pairs_are_returned_untouched``."""
    params = {"tau_k": 0.3, "df": 4.0}
    assert CopulaTawn3.constructible_params(params) is params


@pytest.mark.parametrize("tau", [1e-9, 1e-4, 0.05, 0.2, 1 / 3, 0.5, 0.8, 0.95, 0.999])
@pytest.mark.parametrize("pu,pv", [(0.01, 0.01), (0.1, 0.9), (0.3, 0.3), (0.5, 0.05),
                                   (0.9, 0.9), (1.0, 0.2), (0.2, 1.0)])
def test_refused_triples_move_only_the_weights_and_always_build(tau, pu, pv):
    params = {"tau_k": tau, "psi_u": pu, "psi_v": pv}
    if CopulaTawn3.constructible_params(params) is params:
        return                                        # already admissible
    out = CopulaTawn3.constructible_params(params)
    assert out is not params and out["tau_k"] == tau
    CopulaTawn3(**out)                                # the point of the hook
    # Weights only ever move *towards* 1 (the Gumbel corner), never away.
    assert pu <= out["psi_u"] <= 1.0 and pv <= out["psi_v"] <= 1.0
    # …and land strictly inside the admissible set, with the usual padding.
    pad = max(TAU_PAD_REL * (1.0 - tau), TAU_PAD_ABS)
    cap = CopulaTawn3.reachable_tau_cap(out["psi_u"], out["psi_v"])
    assert cap - tau >= pad or (out["psi_u"], out["psi_v"]) == (1.0, 1.0)


def test_the_repair_keeps_the_direction_of_the_asymmetry():
    """Both weights travel to the Gumbel corner along the same ray, so the
    weaker weight stays the weaker one (unless the fallback to (1, 1) fires)."""
    out = CopulaTawn3.constructible_params({"tau_k": 0.5, "psi_u": 0.6, "psi_v": 0.2})
    assert out["psi_u"] > out["psi_v"]
    assert (1.0 - out["psi_u"]) / (1.0 - out["psi_v"]) == pytest.approx(
        (1.0 - 0.6) / (1.0 - 0.2), rel=1e-12)


@pytest.mark.parametrize("tau", [1e-6, 0.05, 0.3, 0.6, 0.9, 0.999])
@pytest.mark.parametrize("psi", [0.01, 0.1, 0.5, 0.9, 0.999])
def test_repair_reduces_bit_for_bit_to_the_two_parameter_rule(tau, psi):
    """With one weight at 1 the face P = 0 of the quadratic collapses to the
    round-3 rule ψ → (1 + τ)/2 — and is *evaluated* as that expression."""
    expected = CopulaTawn1.repaired_psi(tau)
    assert CopulaTawn3.repaired_psis(tau, 1.0, psi) == (1.0, expected)
    assert CopulaTawn3.repaired_psis(tau, psi, 1.0) == (expected, 1.0)


def test_both_weights_at_one_need_no_repair():
    assert CopulaTawn3.repaired_psis(0.999, 1.0, 1.0) == (1.0, 1.0)


@pytest.mark.parametrize("tau", [1.0, 1.5, 0.0, -0.3, float("nan")])
def test_a_tau_no_weight_can_repair_is_returned_unchanged(tau):
    params = {"tau_k": tau, "psi_u": 0.2, "psi_v": 0.2}
    assert CopulaTawn3.constructible_params(params) is params


def test_constrain_params_clips_both_weights_then_repairs():
    out = CopulaTawn3.constrain_params({"tau_k": 0.5, "psi_u": 5.0, "psi_v": -2.0})
    lo, hi, _ = EXTRA_PARAM_BOUNDS_BY_PARAM["psi_u"]
    assert lo <= out["psi_u"] <= hi and lo <= out["psi_v"] <= hi
    CopulaTawn3(**out)


def test_a_tau_beyond_the_theta_cap_stores_the_tau_it_realises():
    """RB-10, as Galambos: θ is capped at 10⁶, so the copula reports the τ that
    θ actually gives, not the one asked for."""
    tau = 1.0 - 1e-9
    cop = CopulaTawn3(tau_k=tau, psi_u=1.0, psi_v=1.0)
    assert cop.theta == _THETA_HI
    assert cop.params["tau_k"] < tau
    assert cop.params["tau_k"] == pytest.approx(1.0 - 1e-6, rel=1e-6)


# --------------------------------------------------------------------------
# The n-parameter fitting machinery
# --------------------------------------------------------------------------

def test_spec_takes_the_multi_extra_branch_and_every_point_builds():
    from pmcprg.copulas._fit import _two_parameter_spec
    p0, stages, params_of = _two_parameter_spec(CopulaTawn3, CopulaEnum.TAWN3, 0.4)
    assert len(p0) == 3
    assert len(stages) == 2 and stages[0] is stages[1]        # a restart, as types 1/2
    box = stages[0][2]
    assert len(box) == 3
    (s_lo, s_hi), pu_b, pv_b = box
    assert pu_b == pv_b == (0.01, 1.0)
    assert s_lo == pytest.approx(math.log(1e-6)) and s_hi > math.log(9e3)
    assert p0[1] == p0[2] == pytest.approx(0.7)               # (1 + τ_start)/2
    # The box *is* the admissible set: no corner of it is refused.
    for s in (s_lo, 0.0, s_hi):
        for pu in (0.01, 0.37, 1.0):
            for pv in (0.01, 0.37, 1.0):
                params = params_of((s, pu, pv))
                cop = CopulaTawn3(**params)
                assert cop.params["tau_k"] == params["tau_k"]      # no clamping
                assert cop.theta == pytest.approx(1.0 + math.exp(s), rel=1e-12)


def test_round_trip_through_to_x_and_from_x():
    from pmcprg.copulas._fit import _two_parameter_spec
    _, stages, _ = _two_parameter_spec(CopulaTawn3, CopulaEnum.TAWN3, 0.4)
    to_x, from_x, _ = stages[0]
    p = (-1.5, 0.42, 0.91)
    assert tuple(from_x(to_x(p))) == pytest.approx(p, rel=1e-15)


def test_one_extra_families_still_take_the_two_parameter_branch():
    """The dispatcher must not divert a one-extra family into the new code."""
    from pmcprg.copulas._fit import _two_parameter_spec
    for cls, entry in ((CopulaBB1, CopulaEnum.BB1), (CopulaStudent, CopulaEnum.STUDENT),
                       (CopulaTawn1, CopulaEnum.TAWN1), (CopulaTawn2, CopulaEnum.TAWN2),
                       (CopulaTEV, CopulaEnum.TEV)):
        p0, stages, _ = _two_parameter_spec(cls, entry, 0.4)
        assert len(p0) == 2
        assert all(len(box) == 2 for _, _, box in stages)


def test_a_family_with_no_extra_parameter_is_still_a_programming_error():
    from pmcprg.copulas._fit import _two_parameter_spec
    with pytest.raises(ValueError, match="at least one extra parameter"):
        _two_parameter_spec(CopulaGH, CopulaEnum.GH, 0.4)


def test_the_generic_n_extra_fallback_builds_a_box_of_the_right_shape():
    """``_multi_extra_spec``'s last branch — ``p = (τ, x₁, …, xₙ)`` in the
    registered boxes — is what a *future* family with several extras that need
    no reparametrisation would use. Exercised here through a stand-in with two
    arbitrary registered extras, so it is not untested dead code.
    """
    from pmcprg.copulas._fit import _multi_extra_spec

    class _Stub:
        @classmethod
        def constrain_params(cls, params):
            return dict(params)

    extras = ["delta", "df"]
    lo, hi = 0.01, 0.98
    p0, stages, params_of = _multi_extra_spec(_Stub, CopulaEnum.BB1, extras,
                                              lo, hi, 0.4, 0.0)
    assert len(p0) == 3 and p0[0] == 0.4
    assert len(stages) == 1
    tau_b, delta_b, df_b = stages[0][2]
    assert tau_b == (lo, hi)
    assert delta_b == EXTRA_PARAM_BOUNDS_BY_PARAM["delta"][:2]
    assert df_b == EXTRA_PARAM_BOUNDS_BY_PARAM["df"][:2]
    # p0 starts at each box's registered init, and values are clipped into it.
    assert p0[1:] == (EXTRA_PARAM_BOUNDS_BY_PARAM["delta"][2],
                      EXTRA_PARAM_BOUNDS_BY_PARAM["df"][2])
    assert params_of((0.5, 2.0, 7.0)) == {"tau_k": 0.5, "delta": 2.0, "df": 7.0}
    assert params_of((99.0, -5.0, 1e9)) == {"tau_k": hi, "delta": delta_b[0],
                                            "df": df_b[1]}
    to_x, from_x, _ = stages[0]
    assert tuple(from_x(to_x((0.3, 2.0, 7.0)))) == (0.3, 2.0, 7.0)


# Fitted parameters of every family that went through the joint optimiser
# *before* the n-parameter generalisation, as exact IEEE-754 doubles
# (``float.hex``), produced by the code at commit df4e662. Generalising
# ``_two_parameter_spec`` must not move a single bit of them: the new
# dispatch is an early return taken only when a family declares more than one
# extra, so for these the executed code is unchanged — this test is what
# proves it stayed that way. Data: ``cls(**kw).sample(n=1200, seed=seed)``.
_PRE_GENERALISATION_FITS = [
    ("CopulaBB1", dict(tau_k=0.5, delta=1.5), 11,
     {"delta": "0x1.8bdd52fe43c3ep+0", "tau_k": "0x1.0305c942ad680p-1"}),
    ("CopulaBB1", dict(tau_k=0.8, delta=3.0), 12,
     {"delta": "0x1.807e40287a7e8p+1", "tau_k": "0x1.9e251e8cf6840p-1"}),
    ("CopulaBB1", dict(tau_k=0.2, delta=1.1), 13,
     {"delta": "0x1.11eff3b72e78ep+0", "tau_k": "0x1.a37b39d6d9b20p-3"}),
    ("SurvivalBB1", dict(tau_k=0.6, delta=2.0), 14,
     {"delta": "0x1.08b79350a483bp+1", "tau_k": "0x1.323888a2435a3p-1"}),
    ("CopulaBB190", dict(tau_k=-0.5, delta=1.5), 15,
     {"delta": "0x1.91e743efe7f4dp+0", "tau_k": "-0x1.fdc69fd7272bcp-2"}),
    ("CopulaStudent", dict(tau_k=0.95, df=2.5), 21,
     {"df": "0x1.a937d69d917f0p+1", "tau_k": "0x1.e72b3ddb7d236p-1"}),
    ("CopulaStudent", dict(tau_k=-0.6, df=8.0), 22,
     {"df": "0x1.65b907816b516p+3", "tau_k": "-0x1.3a60c893de26dp-1"}),
    ("CopulaStudent", dict(tau_k=0.3, df=30.0), 23,
     {"df": "0x1.0491854853c9dp+6", "tau_k": "0x1.380e9a0424bddp-2"}),
    ("CopulaTawn1", dict(tau_k=0.4, psi=0.6), 31,
     {"psi": "0x1.3e9858e6fc0d2p-1", "tau_k": "0x1.9c3453a59cd9ap-2"}),
    ("CopulaTawn1", dict(tau_k=0.7, psi=0.95), 32,
     {"psi": "0x1.e4cf5f9dd0da0p-1", "tau_k": "0x1.635010b3942c8p-1"}),
    ("CopulaTawn2", dict(tau_k=0.5, psi=0.7), 33,
     {"psi": "0x1.639747f328e1ap-1", "tau_k": "0x1.fd23dbf7d9195p-2"}),
    ("CopulaTawn2", dict(tau_k=0.15, psi=0.3), 34,
     {"psi": "0x1.63c73ec35c8dbp-2", "tau_k": "0x1.3669de3a3c0c0p-3"}),
    ("CopulaTEV", dict(tau_k=0.5, nu=4.0), 41,
     {"nu": "0x1.34df03d5f3accp+2", "tau_k": "0x1.0ca23317e4906p-1"}),
    ("CopulaTEV", dict(tau_k=0.3, nu=1.0), 42,
     {"nu": "0x1.cd670819b3c88p-1", "tau_k": "0x1.34529b0e82accp-2"}),
    ("CopulaTEV", dict(tau_k=0.8, nu=50.0), 43,
     {"nu": "0x1.9000000000000p+6", "tau_k": "0x1.972106500bc34p-1"}),
]

#: The doubles above were written on macOS/arm64 and are compared bit for bit
#: only there. Elsewhere libm and NumPy's SIMD kernels round differently and
#: the optimiser stops elsewhere: ~10⁻⁸ relative on most parameters, but 2 %
#: on Student's ``df`` at seed 23, where the likelihood is nearly flat in ν
#: (measured on Linux aarch64). The invariant that survives is the *maximum*,
#: not the maximiser: evaluated on the same pseudo-observations, the golden
#: parameters' log-likelihood matches the fit's to ≤ 8·10⁻¹¹ for 13 of the 15
#: cases, 4.9·10⁻⁹ and 1.6·10⁻⁴ for Student at seeds 22 and 23 (the latter in
#: the Linux fit's favour: the macOS optimiser stopped short on the plateau).
_BIT_EXACT = sys.platform == "darwin" and platform.machine() == "arm64"


@pytest.mark.slow
@pytest.mark.parametrize("cls_name,kw,seed,expected", _PRE_GENERALISATION_FITS,
                         ids=[f"{n}-{s}" for n, _, s, _ in _PRE_GENERALISATION_FITS])
def test_existing_families_fit_to_the_same_bits(cls_name, kw, seed, expected):
    cls = next(e for e in CopulaEnum if e.value.CLASS_NAME == cls_name).klass
    uv = cls(**kw).sample(n=1200, seed=seed)
    got = cls.fit(uv, method="mle")
    assert got.converged
    if _BIT_EXACT:
        assert {k: float(v).hex() for k, v in got.copula.params.items()} == expected
    else:
        ref = cls(**{k: float.fromhex(v) for k, v in expected.items()})
        ll_ref = float(np.sum(ref.logpdf_array(got.uv)))
        assert got.log_likelihood == pytest.approx(ll_ref, abs=1e-3)


# --------------------------------------------------------------------------
# ``mpmath`` ground truth for pdf / cdf / h
# --------------------------------------------------------------------------

def _ell_mp(u, v, th, pu, pv, mp):
    w, z = -mp.log(u), -mp.log(v)
    n = ((pu * w) ** th + (pv * z) ** th) ** (1 / th)
    return (1 - pu) * w + (1 - pv) * z + n


def _oracle(u, v, th, pu, pv, mp):
    """(C, h, c) from the closed forms, written in ``mpmath`` straight from
    ``C = exp(−ℓ)`` — not from the module's log-space kernel."""
    u, v, th, pu, pv = (mp.mpf(x) for x in (u, v, th, pu, pv))
    w, z = -mp.log(u), -mp.log(v)
    x, y = pu * w, pv * z
    N = (x ** th + y ** th) ** (1 / th)
    p, q = (x / N) ** th, (y / N) ** th
    lw = (1 - pu) + pu * p ** (1 - 1 / th)
    lz = (1 - pv) + pv * q ** (1 - 1 / th)
    lwz = -(th - 1) * pu * pv * (p * q) ** (1 - 1 / th) / N
    C = mp.e ** (-((1 - pu) * w + (1 - pv) * z + N))
    return C, C * lw / u, C * (lw * lz - lwz) / (u * v)


@pytest.mark.slow
def test_ground_truth_oracle_is_itself_validated():
    """:func:`_oracle`'s closed forms vs ``mpmath.diff`` of C — i.e. vs no
    algebra at all. They must agree far below the tolerances the module is
    then held to (measured: ≤ 1.0·10⁻¹⁴ for C and h, ≤ 3.4·10⁻¹² for c, the
    numerical differentiation's own limit)."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    worst = 0.0
    for tau, pu, pv in _TRIPLES[:4]:
        th = CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv).theta

        def C(a, b, _t=th, _u=pu, _v=pv):
            return mp.e ** (-_ell_mp(a, b, _t, _u, _v, mp))

        for u, v in _UV:
            ref = _oracle(u, v, th, pu, pv, mp)
            num = (C(mp.mpf(u), mp.mpf(v)),
                   mp.diff(C, (mp.mpf(u), mp.mpf(v)), (1, 0)),
                   mp.diff(C, (mp.mpf(u), mp.mpf(v)), (1, 1)))
            for a, b in zip(num, ref):
                if b != 0:
                    worst = max(worst, float(abs(a - b) / abs(b)))
    assert worst < 1e-10, worst


@pytest.mark.slow
@pytest.mark.parametrize("tau,pu,pv", _TRIPLES)
def test_pdf_cdf_h_match_the_mpmath_ground_truth(tau, pu, pv):
    """Max relative error over the corner grid, measured on the whole
    (τ, ψ_u, ψ_v) grid: C ≤ 2.1·10⁻¹⁴, h ≤ 1.5·10⁻¹³, c ≤ 2.2·10⁻¹⁴ and
    |Δ ln c| ≤ 2.2·10⁻¹⁴ nat. The bounds below leave a decade of head-room."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    cop = CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)
    th = cop.theta
    got_C = cop.cdf_array(_UV)
    got_lc = cop.logpdf_array(_UV)
    for i, (u, v) in enumerate(_UV):
        ref_C, ref_h, ref_c = _oracle(u, v, th, pu, pv, mp)
        got_h = cop.conditional_cdf(v, u)
        assert abs(mp.mpf(float(got_C[i])) - ref_C) / abs(ref_C) < 1e-13
        assert abs(mp.mpf(got_h) - ref_h) / abs(ref_h) < 1e-12
        assert abs(mp.mpf(float(got_lc[i])) - mp.log(ref_c)) < 1e-13


# --------------------------------------------------------------------------
# τ, sampling and fitting
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tau,pu,pv", [(0.2, 1.0, 1.0), (0.3, 1.0, 0.35), (0.3, 0.35, 1.0),
                                       (0.5, 0.9, 0.8), (0.7, 0.95, 0.9), (0.05, 0.5, 0.2),
                                       (0.15, 0.2, 0.9), (0.45, 0.6, 0.95)])
def test_quadrature_tau_matches_simulated_kendall_tau(tau, pu, pv):
    """The τ the Pickands quadrature inverts must be the τ of the samples the
    h-inverse produces — the two are wholly independent code paths.
    Measured |Δ| ≤ 0.0021 at n = 200 000 (2·se ≈ 0.003)."""
    cop = CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)
    uv = cop.sample(n=200_000, seed=_seed("tauMC", tau, pu, pv))
    assert kendalltau(uv[:, 0], uv[:, 1]).statistic == pytest.approx(tau, abs=0.005)


@pytest.mark.slow
@pytest.mark.parametrize("tau,pu,pv,tol_tau,tol_psi", [
    (0.7, 0.95, 0.90, 0.03, 0.06),
    (0.5, 1.00, 0.70, 0.03, 0.08),
    (0.4, 0.80, 0.50, 0.03, 0.08),
])
def test_fit_recovers_all_three_parameters_where_they_are_identified(
        tau, pu, pv, tol_tau, tol_psi):
    """n = 3000, five replicates: the RMSE of the full study (40 replicates) is
    ≤ 0.010 on τ and ≤ 0.021 on each weight for τ ≥ 0.4 — see the module
    docstring of ``tawn.py`` for where identification breaks down."""
    truth = CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)
    for r in range(5):
        uv = truth.sample(n=3000, seed=_seed("fit", tau, pu, pv, r))
        res = CopulaTawn3.fit(uv, method="mle")
        assert res.converged
        assert res.tau_k == pytest.approx(truth.params["tau_k"], abs=tol_tau)
        assert res.copula.psi_u == pytest.approx(pu, abs=tol_psi)
        assert res.copula.psi_v == pytest.approx(pv, abs=tol_psi)


@pytest.mark.slow
def test_the_weights_are_weakly_identified_near_independence():
    """The counterpart of the test above: at τ = 0.05 the likelihood is flat in
    the weights, so the estimates scatter. This *documents* the limitation
    (module docstring, "Identification") rather than hiding it."""
    truth = CopulaTawn3(tau_k=0.05, psi_u=0.3, psi_v=0.6)
    est = []
    for r in range(12):
        uv = truth.sample(n=3000, seed=_seed("weak", r))
        res = CopulaTawn3.fit(uv, method="mle")
        est.append((res.copula.psi_u, res.copula.psi_v))
    e = np.asarray(est)
    spread = float(np.sqrt(((e - np.array([0.3, 0.6])) ** 2).mean(axis=0)).max())
    assert spread > 0.1, ("if this ever drops, the identification note in "
                          "tawn.py's docstring is too pessimistic and should be revised")


def test_fit_with_method_tau_is_itau_with_a_profile_mle(caplog):
    """FR-12: ``'tau'`` inverts Kendall's τ and fits both weights by MLE at that τ.

    It used to log "cannot identify psi" and run the joint MLE.
    """
    import logging

    from pmcprg.exceptions import CopulaParameterError
    uv = CopulaTawn3(tau_k=0.4, psi_u=0.8, psi_v=0.6).sample(n=600, seed=77)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.extreme_value.tawn"):
        res = CopulaTawn3.fit(uv, method="tau")
    assert res.method == "tau" and res.converged
    assert res.tau_k == kendalltau(uv[:, 0], uv[:, 1])[0]
    for key in ("psi_u", "psi_v"):
        for h in (-1e-3, 1e-3):
            p = dict(res.copula.params)
            p[key] += h
            try:
                cop = CopulaTawn3(**p)
            except CopulaParameterError:
                continue
            assert float(np.sum(cop.logpdf_array(res.uv))) <= res.log_likelihood + 1e-9
    mle = CopulaTawn3.fit(uv, method="mle")
    assert mle.log_likelihood >= res.log_likelihood and mle.tau_k != res.tau_k


@pytest.mark.slow
def test_fit_best_prefers_tawn3_on_doubly_asymmetric_data():
    """Data whose two weights differ must not be won by a restriction that can
    only bend one margin."""
    from pmcprg.copulas._base import CopulaVirt
    # τ = 0.45 is below the cap of this weight pair (0.5345), which is exactly
    # the point: a doubly asymmetric member lives well inside its own cap.
    uv = CopulaTawn3(tau_k=0.45, psi_u=0.95, psi_v=0.55).sample(n=4000, seed=909)
    res = CopulaVirt.fit_best(uv, families=[CopulaTawn3, CopulaTawn1, CopulaTawn2, CopulaGH],
                              method="mle")
    assert res[0].copula.__class__ is CopulaTawn3


# --------------------------------------------------------------------------
# Exchangeability (FR-10's test, used here as a property of the family)
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tau,pu,pv,expect_reject", [
    (0.5, 0.90, 0.90, False),      # symmetric weights: exchangeable
    (0.3, 1.00, 1.00, False),      # Gumbel
    (0.5, 1.00, 0.60, True),
    (0.5, 0.95, 0.55, True),
])
def test_exchangeability_sees_the_asymmetry(tau, pu, pv, expect_reject):
    """ψ_u = ψ_v makes the model exchangeable (the ℓ formula is then symmetric
    in w, z); ψ_u ≠ ψ_v breaks it. Rejection rates over 40 replicates at
    n = 500, B = 200: see the commit message."""
    from pmcprg.diagnostics.exchangeability import exchangeability_test
    cop = CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)
    rejects = 0
    for r in range(8):
        uv = cop.sample(n=500, seed=_seed("exch", tau, pu, pv, r))
        out = exchangeability_test(uv[:, 0], uv[:, 1], B=200,
                                   seed=_seed("exchB", tau, pu, pv, r),
                                   bootstrap="multiplier")
        rejects += bool(out.reject)
    if expect_reject:
        assert rejects >= 7, f"only {rejects}/8 rejections for psi_u != psi_v"
    else:
        assert rejects <= 1, f"{rejects}/8 rejections under exchangeability"


# --------------------------------------------------------------------------
# ICE: M-step, multistart jitter, family draws, selection placeholder
# --------------------------------------------------------------------------

MODELS = "pmcprg/pmc/models"


def _model(blocks, fixture="pmc_gauss_k2.toml"):
    from pmcprg.pmc.model import PMCModel
    raw = PMCModel(f"{MODELS}/{fixture}").raw
    for blk, new in zip(raw["copulas"], blocks):
        blk.pop("tau", None)
        blk.update(new)
    return PMCModel.from_dict(raw)


def _refused(tau, pu, pv) -> bool:
    try:
        CopulaTawn3(tau_k=tau, psi_u=pu, psi_v=pv)
    except CopulaParameterError:
        return True
    return False


@pytest.mark.slow
def test_ice_m_step_fits_all_three_parameters():
    """The weighted joint MLE behind the ICE M-step must return both weights —
    and must not leave them at their start value 1.0 (the RB-3 failure)."""
    from pmcprg.pmc.ice import _fit_copula_params, _resolve_candidate
    uv = CopulaTawn3(tau_k=0.45, psi_u=0.9, psi_v=0.55).sample(n=2500, seed=4242)
    entry, cls = _resolve_candidate("Tawn3")
    out = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(uv.shape[0]))
    assert set(out) >= {"tau_k", "psi_u", "psi_v"}
    assert out["psi_u"] != 1.0 and out["psi_v"] != 1.0
    assert out["psi_v"] == pytest.approx(0.55, abs=0.12)
    assert out["psi_u"] > out["psi_v"]
    CopulaTawn3(**{k: v for k, v in out.items() if k != "fit_failed"})


@pytest.mark.parametrize("jitter", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("tau,pu,pv", [(0.5, 0.9, 0.7), (0.16, 0.2, 0.9), (0.05, 0.3, 0.3)])
def test_multistart_jitter_never_builds_a_refused_triple(jitter, tau, pu, pv):
    from pmcprg.pmc._estim_common import perturb_initial_model
    model = _model([{"name": "Tawn3", "tau": tau, "psi_u": pu, "psi_v": pv}] * 4)
    for seed in range(60):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=jitter)
        for blk in out.copula_blocks():
            assert not _refused(blk["tau"], blk["psi_u"], blk["psi_v"])


def test_the_repair_is_what_makes_those_starts_build(monkeypatch):
    """The counterfactual, as ``test_multistart_joint_constraints`` does for
    BB1: with ``constructible_params`` neutralised, the very same draws raise —
    so the hook is doing the work, and this is not a vacuous test."""
    from pmcprg.pmc._estim_common import perturb_initial_model
    model = _model([{"name": "Tawn3", "tau": 0.16, "psi_u": 0.2, "psi_v": 0.9}] * 4)

    n_built = 0
    for seed in range(120):
        for blk in perturb_initial_model(model, np.random.default_rng(seed),
                                         jitter=0.25).copula_blocks():
            assert not _refused(blk["tau"], blk["psi_u"], blk["psi_v"])
            n_built += 1
    assert n_built == 480

    # Neutralise the hook — the identity every family without a joint
    # constraint already has — and replay the very same draws. The rebuild
    # itself then raises, which is precisely the failure the hook prevents.
    monkeypatch.setattr(CopulaTawn3, "constructible_params",
                        classmethod(lambda cls, params: params))
    n_raised = 0
    for seed in range(120):
        try:
            perturb_initial_model(model, np.random.default_rng(seed), jitter=0.25)
        except CopulaParameterError:
            n_raised += 1
    assert n_raised > 0, ("no draw of this model leaves the admissible set — pick a "
                          "start closer to the cap, or the test proves nothing")


@pytest.mark.parametrize("mode", ["random", "sweep"])
@pytest.mark.parametrize("cands", [["Gauss", "Tawn3"], ["Tawn3", "Tawn1", "BB1"]])
def test_family_starts_with_tawn3_candidates_build(mode, cands):
    from pmcprg.pmc._estim_common import build_multistart_inits
    from pmcprg.pmc.model import PMCModel
    for start in (PMCModel(f"{MODELS}/pmc_gauss_k2.toml"),
                  _model([{"name": "Tawn3", "tau": 0.5, "psi_u": 0.8, "psi_v": 0.9}] * 4)):
        for seed in range(3):
            cfg = {"n_starts": 1 + len(cands) ** 2 + 3, "multistart_seed": seed,
                   "multistart_jitter": 0.25, "multistart_families": mode,
                   "candidates": cands}
            for m, _ in build_multistart_inits(start, cfg, label="ICE"):
                for blk in m.copula_blocks():
                    if blk["name"] == "Tawn3":
                        assert not _refused(blk["tau"], blk.get("psi_u", 1.0),
                                            blk.get("psi_v", 1.0))


def test_a_family_change_to_tawn3_sets_both_weights_and_drops_the_others():
    from pmcprg.pmc._estim_common import _set_copula_family
    blk = {"i": 0, "j": 0, "name": "BB1", "tau": 0.5, "delta": 1.5}
    _set_copula_family(blk, CopulaEnum.TAWN3, 0.5)
    assert blk["name"] == "Tawn3"
    assert blk["psi_u"] == 1.0 and blk["psi_v"] == 1.0
    assert "delta" not in blk
    CopulaTawn3(tau_k=blk["tau"], psi_u=blk["psi_u"], psi_v=blk["psi_v"])


def test_a_family_change_away_from_tawn3_drops_both_weights():
    from pmcprg.pmc._estim_common import _set_copula_family
    blk = {"i": 0, "j": 0, "name": "Tawn3", "tau": 0.5, "psi_u": 0.8, "psi_v": 0.6}
    _set_copula_family(blk, CopulaEnum.CLAYTON, 0.5)
    assert "psi_u" not in blk and "psi_v" not in blk


def test_selection_placeholder_builds():
    """When no candidate has a finite score the M-step stores a placeholder; it
    must carry weights the constructor accepts at the placeholder's τ."""
    from pmcprg.pmc.ice import (FIT_FAILED_KEY, _copula_candidate_fits,
                                _copula_placeholder)
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=50), rng.uniform(size=50)
    candidates = ["Tawn3", "Gauss"]
    resolved, _ = _copula_candidate_fits(candidates, u, v, np.ones(50), "mle")
    blk = _copula_placeholder(candidates, resolved, "mle", {"name": "Tawn3", "tau": 0.0})
    assert blk[FIT_FAILED_KEY] is True
    stored = {k: val for k, val in blk.items() if k != FIT_FAILED_KEY}
    model = _model([stored])
    assert model.copula_blocks()[0]["name"] == "Tawn3"


def test_a_toml_round_trip_keeps_both_weights(tmp_path):
    from pmcprg.pmc.model import PMCModel
    model = _model([{"name": "Tawn3", "tau": 0.45, "psi_u": 0.85, "psi_v": 0.6}] * 4)
    path = tmp_path / "tawn3.toml"
    model.save(str(path))
    back = PMCModel(str(path)).copula_blocks()[0]
    assert (back["psi_u"], back["psi_v"]) == (0.85, 0.6)


# --------------------------------------------------------------------------
# Standard errors are deliberately not implemented for three parameters
# --------------------------------------------------------------------------

def test_standard_errors_raise_explicitly():
    uv = CopulaTawn3(tau_k=0.4, psi_u=0.8, psi_v=0.6).sample(n=200, seed=5)
    cop = CopulaTawn3(tau_k=0.4, psi_u=0.8, psi_v=0.6)
    with pytest.raises(NotImplementedError, match="3-parameter"):
        cop.standard_errors(uv, method="mle")
