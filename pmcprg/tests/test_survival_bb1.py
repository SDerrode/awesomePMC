"""Survival BB1 and its own 90°/270° rotations (audit FR-8, closing round).

FR-8's own list of families needing 90°/270° rotations named "BB1 de
survie" (survival BB1) separately from plain BB1: ``SurvivalClayton``,
``SurvivalGH`` and ``SurvivalJoe`` were already in
``pmcprg.copulas.archimedean.survival``, but ``SurvivalBB1`` itself did not
exist until this round, so its rotations could not either. This file mirrors
``test_rotated_bb1.py``'s depth for the three new families:

* the 180°-rotation identity Ŝ(u,v) = u+v−1+C(1−u,1−v) to machine precision
  (``SurvivalCopula``'s own module docstring), and its 90°/270° analogues
  (``rotated.py``'s own formulas, base = ``SurvivalBB1``);
* τ unchanged by the 180° rotation (unlike the 90°/270° case, which flips
  its sign) — checked against the registry, not assumed;
* ``delta`` pass-through and the joint ``constrain_params``/
  ``constructible_params`` delegation, needed for the first time by
  ``SurvivalCopula`` itself (audit G1, mirroring the fix ``RotatedCopula``
  needed for plain BB1);
* a genuine algebraic finding, checked numerically rather than left as an
  algebraic claim: ``SurvivalBB190`` and ``SurvivalBB1270`` are the *same
  copula* as ``CopulaBB1270``/``CopulaBB190`` (plain BB1's own rotations,
  swapped) at the same (τ, δ) — composing the 180° and 90° rotations in
  this order lands back on plain BB1's 270° rotation (``rotated.py``'s own
  section comment derives why: BB1 is exchangeable, so 180° ∘ 90° = 270°);
* fit recovery of τ and δ for all three families;
* the tail/corner-mass signature: BB1's two-sided asymmetry (λ_L, λ_U > 0)
  swaps role under the 180° rotation and is preserved in shape (still
  two-sided) — verified by simulation, not assumed to mirror plain BB1's
  numbers unchanged;
* exchangeability: ``SurvivalBB1`` is exchangeable (BB1 is, and the
  survival reflection (u,v) -> (1-u,1-v) preserves exchangeability); its
  90°/270° rotations are not, same as plain BB1's own rotations.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import CopulaBB1, CopulaBB190, CopulaBB1270, CopulaEnum
from pmcprg.copulas.archimedean.survival import SurvivalBB1
from pmcprg.copulas.archimedean.rotated import SurvivalBB190, SurvivalBB1270
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS


def _seed(tag: str) -> int:
    """Deterministic seed from a tag — never Python's salted ``hash()``
    (commit 5e56fda fixed an intermittent failure caused by that)."""
    return zlib.crc32(tag.encode())


SBB1 = CopulaEnum.SURVIVAL_BB1
SBB190 = CopulaEnum.SURVIVAL_BB190
SBB1270 = CopulaEnum.SURVIVAL_BB1270


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry,short,cls_name,klass", [
    (SBB1, "SBB1", "SurvivalBB1", SurvivalBB1),
    (SBB190, "SBB190", "SurvivalBB190", SurvivalBB190),
    (SBB1270, "SBB1270", "SurvivalBB1270", SurvivalBB1270),
])
def test_registered_in_copula_enum(entry, short, cls_name, klass):
    assert entry.value.SHORT_NAME == short
    assert entry.value.CLASS_NAME == cls_name
    assert entry.klass is klass
    assert entry.value.AVAILABLE
    assert entry in CopulaEnum.available()
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "delta"]


def test_tau_range_is_unchanged_under_180_degrees():
    """SurvivalBB1's range mirrors CopulaBB1's own [0+EPS, 1) bit for bit —
    unlike a 90°/270° rotation, a 180° rotation does not flip tau's sign."""
    assert SBB1.value.TAU_MIN_MAX == CopulaEnum.BB1.value.TAU_MIN_MAX


def test_90_270_tau_range_is_negative_only_and_matches_plain_bb1s():
    """SurvivalBB190/SurvivalBB1270 mirror CopulaBB190/CopulaBB1270's own
    registered range exactly (both [-1, -EPS], not [-1, -1/3] — verified
    against the registry, not assumed)."""
    for entry in (SBB190, SBB1270):
        lo, hi = entry.value.TAU_MIN_MAX
        assert lo == pytest.approx(-1.0)
        assert hi == pytest.approx(-EPS)
    assert SBB190.value.TAU_MIN_MAX == CopulaEnum.BB190.value.TAU_MIN_MAX
    assert SBB1270.value.TAU_MIN_MAX == CopulaEnum.BB1270.value.TAU_MIN_MAX


@pytest.mark.parametrize("cls", [SurvivalBB1])
@pytest.mark.parametrize("tau", [0.0, 1.0])
def test_bb1_singular_or_non_positive_tau_is_refused(cls, tau):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=tau, delta=1.5)


@pytest.mark.parametrize("cls", [SurvivalBB190, SurvivalBB1270])
@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0, -1.0])
def test_non_negative_or_singular_tau_is_refused(cls, tau):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=tau, delta=1.5)


# --------------------------------------------------------------------------
# 180° identity — machine precision
# --------------------------------------------------------------------------

_TAU_DELTA_POS = [(0.3, 1.2), (0.5, 1.5), (0.7, 2.5), (0.9, 5.0)]
_G = np.linspace(1e-3, 1 - 1e-3, 25)
_GRID = np.array([(u, v) for u in _G for v in _G])


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_POS)
def test_survival_bb1_cdf_matches_the_180_identity(tau, delta):
    """Ŝ(u,v) == u + v - 1 + C(1-u, 1-v) on a 25x25 grid, machine precision."""
    base = CopulaBB1(tau_k=tau, delta=delta)
    s = SurvivalBB1(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = s.cdf_array(_GRID)
    rhs = u + v - 1.0 + base.cdf_array(np.column_stack([1.0 - u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-12, f"max |S - (u+v-1+C(1-u,1-v))| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_POS)
def test_survival_bb1_pdf_matches_base_at_reflected_point(tau, delta):
    base = CopulaBB1(tau_k=tau, delta=delta)
    s = SurvivalBB1(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = s.pdf_array(_GRID)
    rhs = base.pdf_array(np.column_stack([1.0 - u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-9, f"max |s - c_base(1-u,1-v)| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_POS)
def test_survival_bb1_h_matches_one_minus_base_h_reflected(tau, delta):
    """h_hat(v|u) == 1 - h_base(1-v | 1-u) (module docstring of survival.py)."""
    base = CopulaBB1(tau_k=tau, delta=delta)
    s = SurvivalBB1(tau_k=tau, delta=delta)
    rng = np.random.default_rng(_seed("sbb1-h"))
    us = rng.uniform(1e-3, 1 - 1e-3, 100)
    vs = rng.uniform(1e-3, 1 - 1e-3, 100)
    got = np.array([s.conditional_cdf(v, u) for u, v in zip(us, vs)])
    ref = np.array([1.0 - base.conditional_cdf(1.0 - v, 1.0 - u) for u, v in zip(us, vs)])
    np.testing.assert_allclose(got, ref, atol=1e-9)


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_POS)
def test_survival_bb1_inv_h_round_trips(tau, delta):
    cop = SurvivalBB1(tau_k=tau, delta=delta)
    rng = np.random.default_rng(_seed("sbb1-invh"))
    for u, v in zip(rng.uniform(0.01, 0.99, 40), rng.uniform(0.01, 0.99, 40)):
        w = cop.conditional_cdf(v, u)
        v_back = cop.inv_h(w, u)
        w_back = cop.conditional_cdf(v_back, u)
        assert w_back == pytest.approx(w, abs=1e-6)


def test_tail_dependence_swaps_lower_and_upper():
    base = CopulaBB1(tau_k=0.6, delta=2.0)
    s = SurvivalBB1(tau_k=0.6, delta=2.0)
    lam_l_base, lam_u_base = base.tail_dependence()
    lam_l_s, lam_u_s = s.tail_dependence()
    assert lam_l_s == pytest.approx(lam_u_base)
    assert lam_u_s == pytest.approx(lam_l_base)


# --------------------------------------------------------------------------
# 90°/270° of the survival family — machine-precision identity, AND the
# coincidence with plain BB1's own rotations (a genuine finding, checked
# not assumed — see the module and rotated.py's own section docstring).
# --------------------------------------------------------------------------

_TAU_DELTA_ROT = [(-0.3, 1.2), (-0.6, 2.0), (-0.8, 3.0), (-0.9, 5.0)]


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_ROT)
def test_survival_bb190_cdf_matches_the_rotation_identity(tau, delta):
    """C90(u,v) == v - Shat_base(1-u, v) with Shat_base = SurvivalBB1(-tau, delta)."""
    base = SurvivalBB1(tau_k=-tau, delta=delta)
    rot = SurvivalBB190(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.cdf_array(_GRID)
    rhs = v - base.cdf_array(np.column_stack([1.0 - u, v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-10, f"max |C90 - (v - Shat(1-u,v))| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_ROT)
def test_survival_bb1270_cdf_matches_the_rotation_identity(tau, delta):
    base = SurvivalBB1(tau_k=-tau, delta=delta)
    rot = SurvivalBB1270(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.cdf_array(_GRID)
    rhs = u - base.cdf_array(np.column_stack([u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-10, f"max |C270 - (u - Shat(u,1-v))| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_ROT)
def test_survival_bb190_equals_plain_bb1270(tau, delta):
    """180 deg (survival) then 90 deg == plain BB1's own 270 deg rotation,
    at the same (tau, delta) -- the dihedral composition 180 o 90 = 270 for
    an exchangeable base (rotated.py's section comment derives this)."""
    a = SurvivalBB190(tau_k=tau, delta=delta)
    b = CopulaBB1270(tau_k=tau, delta=delta)
    pdf_err = np.max(np.abs(a.pdf_array(_GRID) - b.pdf_array(_GRID)))
    cdf_err = np.max(np.abs(a.cdf_array(_GRID) - b.cdf_array(_GRID)))
    assert pdf_err < 1e-8, f"pdf max diff {pdf_err:.3g}"
    assert cdf_err < 1e-8, f"cdf max diff {cdf_err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA_ROT)
def test_survival_bb1270_equals_plain_bb190(tau, delta):
    """180 deg then 270 deg == plain BB1's own 90 deg rotation."""
    a = SurvivalBB1270(tau_k=tau, delta=delta)
    b = CopulaBB190(tau_k=tau, delta=delta)
    pdf_err = np.max(np.abs(a.pdf_array(_GRID) - b.pdf_array(_GRID)))
    cdf_err = np.max(np.abs(a.cdf_array(_GRID) - b.cdf_array(_GRID)))
    assert pdf_err < 1e-8, f"pdf max diff {pdf_err:.3g}"
    assert cdf_err < 1e-8, f"cdf max diff {cdf_err:.3g}"


@pytest.mark.parametrize("cls", [SurvivalBB190, SurvivalBB1270])
@pytest.mark.parametrize("tau,delta", _TAU_DELTA_ROT)
def test_h_inv_h_round_trip(cls, tau, delta):
    cop = cls(tau_k=tau, delta=delta)
    rng = np.random.default_rng(_seed(f"sbb1-rot-invh-{cls.__name__}"))
    us = rng.uniform(0.01, 0.99, 40)
    vs = rng.uniform(0.01, 0.99, 40)
    for u, v in zip(us, vs):
        w = cop.conditional_cdf(v, u)
        v_back = cop.inv_h(w, u)
        w_back = cop.conditional_cdf(v_back, u)
        assert w_back == pytest.approx(w, abs=1e-6)


# --------------------------------------------------------------------------
# delta pass-through and joint-constraint delegation (audit G1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,delta", _TAU_DELTA_POS)
def test_delta_passes_through_survival_bb1_unchanged(tau, delta):
    s = SurvivalBB1(tau_k=tau, delta=delta)
    base = CopulaBB1(tau_k=tau, delta=delta)
    assert s.delta == pytest.approx(delta)
    assert s.theta == pytest.approx(base.theta, rel=1e-12)


@pytest.mark.parametrize("cls", [SurvivalBB190, SurvivalBB1270])
@pytest.mark.parametrize("tau,delta", _TAU_DELTA_ROT)
def test_delta_passes_through_the_rotations_unchanged(cls, tau, delta):
    rot = cls(tau_k=tau, delta=delta)
    base = SurvivalBB1(tau_k=-tau, delta=delta)
    assert rot.delta == pytest.approx(delta, rel=1e-12)
    assert rot.theta == pytest.approx(base.theta, rel=1e-12)


def test_survival_bb1_constructible_params_repairs_an_infeasible_delta():
    """SurvivalBB1 is the first base family SurvivalCopula wraps with a
    non-identity constrain_params/constructible_params — it must delegate
    to CopulaBB1's own hook, unchanged (no sign flip: module docstring of
    survival.py)."""
    params = {"tau_k": 0.01, "delta": 5.0}
    with pytest.raises(CopulaParameterError):
        SurvivalBB1(**params)
    repaired = SurvivalBB1.constructible_params(params)
    assert repaired["tau_k"] == params["tau_k"]
    assert 1.0 <= repaired["delta"] < 5.0
    cop = SurvivalBB1(**repaired)
    assert cop.delta == pytest.approx(repaired["delta"])
    assert repaired == CopulaBB1.constructible_params(params)


@pytest.mark.parametrize("cls", [SurvivalBB190, SurvivalBB1270])
def test_rotation_constructible_params_repairs_an_infeasible_delta(cls):
    params = {"tau_k": -0.01, "delta": 5.0}
    with pytest.raises(CopulaParameterError):
        cls(**params)
    repaired = cls.constructible_params(params)
    assert repaired["tau_k"] == params["tau_k"]
    assert 1.0 <= repaired["delta"] < 1.5
    cop = cls(**repaired)
    assert cop.delta == pytest.approx(repaired["delta"])


# --------------------------------------------------------------------------
# Exchangeability (audit FR-10-style check)
# --------------------------------------------------------------------------

def test_survival_bb1_is_exchangeable():
    """SurvivalBB1 preserves BB1's own exchangeability: the (u,v)->(1-u,1-v)
    reflection is itself symmetric under u<->v swap."""
    cop = SurvivalBB1(tau_k=0.6, delta=2.0)
    rng = np.random.default_rng(_seed("sbb1-exch"))
    uv = rng.uniform(0.02, 0.98, size=(30, 2))
    lhs = cop.pdf_array(uv)
    rhs = cop.pdf_array(uv[:, ::-1])
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


@pytest.mark.parametrize("cls", [SurvivalBB190, SurvivalBB1270])
def test_survival_bb1_rotations_are_not_exchangeable(cls):
    """Like plain BB190/BB1270, a 90°/270° rotation is not exchangeable —
    it reflects only one margin (module docstring of rotated.py)."""
    cop = cls(tau_k=-0.6, delta=2.0)
    rng = np.random.default_rng(_seed(f"sbb1-rot-exch-{cls.__name__}"))
    uv = rng.uniform(0.02, 0.98, size=(30, 2))
    lhs = cop.pdf_array(uv)
    rhs = cop.pdf_array(uv[:, ::-1])
    assert not np.allclose(lhs, rhs, rtol=1e-6)


# --------------------------------------------------------------------------
# Tail / corner-mass signature (measured by simulation, not assumed)
# --------------------------------------------------------------------------

def test_survival_bb1_has_two_sided_corner_mass_with_roles_swapped():
    """At (tau, delta) = (0.6, 2.0), plain BB1 has lambda_U > lambda_L
    (test below); the survival rotation swaps the roles, so its own
    lower-left corner (now carrying the base's upper-tail dependence)
    should be the larger one."""
    base = CopulaBB1(tau_k=0.6, delta=2.0)
    lam_l_base, lam_u_base = base.tail_dependence()
    assert lam_u_base > lam_l_base

    s = SurvivalBB1(tau_k=0.6, delta=2.0)
    uv = s.sample(n=40000, seed=_seed("sbb1-corners"))
    t = 0.05
    lower_left = np.mean((uv[:, 0] < t) & (uv[:, 1] < t))
    upper_right = np.mean((uv[:, 0] > 1.0 - t) & (uv[:, 1] > 1.0 - t))
    assert lower_left > 0.0 and upper_right > 0.0
    assert lower_left > upper_right, (
        f"expected the swapped (now dominant) lower-left corner to exceed "
        f"the upper-right one: {lower_left} vs {upper_right}")


def test_survival_bb1_rotations_populate_both_anti_diagonal_corners():
    """Both SurvivalBB190 and SurvivalBB1270 relocate SurvivalBB1's own
    two-sided mass to both anti-diagonal corners, same shape as plain
    BB190/BB1270's own signature (rotated.py's section comment)."""
    tau, delta = -0.6, 2.0
    t = 0.05
    s90 = SurvivalBB190(tau_k=tau, delta=delta).sample(n=40000, seed=_seed("sbb190-corners"))
    s270 = SurvivalBB1270(tau_k=tau, delta=delta).sample(n=40000, seed=_seed("sbb1270-corners"))

    def corners(uv):
        upper_left = np.mean((uv[:, 0] < t) & (uv[:, 1] > 1.0 - t))
        lower_right = np.mean((uv[:, 0] > 1.0 - t) & (uv[:, 1] < t))
        return upper_left, lower_right

    ul90, lr90 = corners(s90)
    ul270, lr270 = corners(s270)
    assert ul90 > 0.005 and lr90 > 0.005
    assert ul270 > 0.005 and lr270 > 0.005


# --------------------------------------------------------------------------
# fit — tau AND delta recovery, both methods
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau_true,delta_true", [(0.3, 1.2), (0.6, 2.0), (0.85, 4.0)])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_survival_bb1_fit_recovers_tau_and_delta(tau_true, delta_true, method):
    cop = SurvivalBB1(tau_k=tau_true, delta=delta_true)
    uv = cop.sample(n=3000, seed=_seed(f"sbb1-fit-{tau_true}-{delta_true}-{method}"))
    r = SurvivalBB1.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.06)
    assert r.tau_k > 0.0
    assert r.copula.params["tau_k"] == r.tau_k
    if method == "mle":
        assert r.copula.delta == pytest.approx(delta_true, rel=0.3)


@pytest.mark.parametrize("cls", [SurvivalBB190, SurvivalBB1270])
@pytest.mark.parametrize("tau_true,delta_true", [(-0.3, 1.2), (-0.6, 2.0), (-0.85, 4.0)])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_survival_bb1_rotations_fit_recovers_tau_and_delta(cls, tau_true, delta_true, method):
    cop = cls(tau_k=tau_true, delta=delta_true)
    uv = cop.sample(n=3000, seed=_seed(f"sbb1-rot-fit-{cls.__name__}-{tau_true}-{delta_true}-{method}"))
    r = cls.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.06)
    assert r.tau_k < 0.0
    assert r.copula.params["tau_k"] == r.tau_k
    if method == "mle":
        assert r.copula.delta == pytest.approx(delta_true, rel=0.3)


@pytest.mark.slow
@pytest.mark.parametrize("cls,tau,delta", [
    (SurvivalBB1, 0.5, 1.8),
    (SurvivalBB190, -0.5, 1.8),
    (SurvivalBB1270, -0.5, 1.8),
])
def test_sample_has_uniform_margins_and_the_right_tau(cls, tau, delta):
    cop = cls(tau_k=tau, delta=delta)
    uv = cop.sample(n=20000, seed=_seed(f"sbb1-margins-{cls.__name__}"))
    assert kstest(uv[:, 0], "uniform").pvalue > 0.001
    assert kstest(uv[:, 1], "uniform").pvalue > 0.001
    tau_hat = kendalltau(uv[:, 0], uv[:, 1]).statistic
    assert abs(tau_hat - tau) < 0.03, f"{cls.__name__}: tau_hat={tau_hat}, expected {tau}"


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_survival_bb1_family_is_covered_by_the_reference_file():
    from test_copula_limits import _TAIL_TAUS, _file_table, _required_param_keys, _match_file_key

    for short in ("SBB1", "SBB190", "SBB1270"):
        assert short in _TAIL_TAUS
    table = _file_table()
    need = {k for k in _required_param_keys() if k[0] in ("SBB1", "SBB190", "SBB1270")}
    assert need, "no SBB1/SBB190/SBB1270 cases registered in test_copula_limits.py"
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
