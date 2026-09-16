"""90°/270°-rotated BB1 copulas (audit FR-8, third round).

Mirrors ``test_rotated_gh.py``'s depth, plus the checks specific to BB1 as
the first *two-parameter* rotated family:

* ``pmcprg.copulas.archimedean.bb1`` needed its kernel interface added
  first — unlike GH/Joe, which already had it — so this file also stands in
  for a "did the refactor break anything" check by construction (every test
  here exercises ``CopulaBB1`` through the rotation wrapper's kernel calls).
* fit recovery of **both** τ and δ (τ alone never identified δ for BB1
  itself either; see ``test_ice_bb1_fit_leaves_the_start_value`` in
  ``test_copula_limits.py``);
* ``constructible_params``/``constrain_params`` delegation (audit G1) — BB1
  is the first base family whose joint constraint is not the identity, so
  the rotated variants need the sign-flipped delegation to inherit it;
* the tail-asymmetry signature is *derived from an actual sample*, not
  assumed: BB1 is a Joe-Clayton hybrid with generally BOTH λ_L, λ_U > 0
  (module docstring of ``bb1.py``), so a plain BB1 sample has residual mass
  in *both* diagonal corners — the 90°/270° rotations therefore populate
  *both* anti-diagonal corners too (not a single dominant one, as for the
  one-signed Clayton/GH/Joe rotations), in proportions that mirror across
  the two rotations.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import CopulaBB1, CopulaBB190, CopulaBB1270, CopulaEnum
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

BB190 = CopulaEnum.BB190
BB1270 = CopulaEnum.BB1270


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry,short,cls_name,klass", [
    (BB190, "BB190", "CopulaBB190", CopulaBB190),
    (BB1270, "BB1270", "CopulaBB1270", CopulaBB1270),
])
def test_registered_in_copula_enum(entry, short, cls_name, klass):
    assert entry.value.SHORT_NAME == short
    assert entry.value.CLASS_NAME == cls_name
    assert entry.klass is klass
    assert entry.value.AVAILABLE
    assert entry in CopulaEnum.available()
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k", "delta"]


@pytest.mark.parametrize("entry", [BB190, BB1270])
def test_tau_range_is_negative_only(entry):
    """Mirror of BB1's own [0+EPS, 1.0] one-sided range (audit FR-8)."""
    lo, hi = entry.value.TAU_MIN_MAX
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(-EPS)


@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_non_negative_tau_is_refused(cls, tau):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=tau)


@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
def test_tau_at_or_beyond_minus_one_is_refused(cls):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=-1.0)
    with pytest.raises(CopulaParameterError):
        cls(tau_k=-1.5)


# --------------------------------------------------------------------------
# Kernel interface (the actual FR-8 prerequisite this round added to BB1)
# --------------------------------------------------------------------------

def test_bb1_exposes_the_kernel_interface():
    for member in ("_kcoord", "_kcoord_reflected", "_k_logpdf", "_k_cdf", "_k_h", "_k_inv_h"):
        assert hasattr(CopulaBB1, member), f"CopulaBB1 is missing {member}"


def test_bb1_kernel_coordinate_matches_gumbels_not_joes():
    """BB1's generator is built from t directly (like Gumbel's (-ln t)^θ),
    not from its complement (like Joe's), so its kernel coordinate is
    ``log x``, not ``log(1 - x)`` (module docstring of ``bb1.py``)."""
    x = 0.37
    assert CopulaBB1._kcoord(x) == pytest.approx(np.log(x))
    assert CopulaBB1._kcoord_reflected(x) == pytest.approx(np.log1p(-x))


@pytest.mark.parametrize("tau,delta", [(0.5, 1.3), (0.7, 2.0), (0.85, 4.0)])
def test_bb1_public_api_unchanged_after_kernel_refactor(tau, delta):
    """The refactor rebuilds cdf/pdf/conditional_cdf from the kernel
    interface; this checks internal consistency directly (the existing
    ``test_palier1_b1_families.py``/``test_copula_limits.py`` BB1 cases
    already re-ran bit-identically against their pre-refactor references)."""
    cop = CopulaBB1(tau_k=tau, delta=delta)
    rng = np.random.default_rng(0)
    us = rng.uniform(1e-3, 1 - 1e-3, 30)
    vs = rng.uniform(1e-3, 1 - 1e-3, 30)
    for u, v in zip(us, vs):
        ka, kb = cop._kcoord(u), cop._kcoord(v)
        c_direct, _ = cop._k_cdf(ka, kb)
        assert cop.cdf([u, v]) == pytest.approx(float(c_direct), rel=1e-12)
        h_direct, _ = cop._k_h(kb, ka)
        assert cop.conditional_cdf(v, u) == pytest.approx(float(h_direct), rel=1e-12)
        logpdf_direct = cop._k_logpdf(ka, kb)
        assert np.log(cop.pdf([u, v])) == pytest.approx(float(logpdf_direct), rel=1e-9)


def test_bb1_k_inv_h_has_no_closed_form_but_round_trips():
    """No closed form (module docstring): Brent's method on kb = log v.
    Round-trips h(inv_h(w|u), u) == w to the solver's tolerance."""
    cop = CopulaBB1(tau_k=0.6, delta=2.0)
    rng = np.random.default_rng(1)
    for u, v in zip(rng.uniform(0.02, 0.98, 30), rng.uniform(0.02, 0.98, 30)):
        w = cop.conditional_cdf(v, u)
        v_back, one_minus_v_back = cop._k_inv_h(np.log(w), cop._kcoord(u))
        assert float(v_back) == pytest.approx(v, abs=1e-6)
        assert float(one_minus_v_back) == pytest.approx(1.0 - v, abs=1e-6)


# --------------------------------------------------------------------------
# VineCopula identity — the audit's primary correctness anchor
# --------------------------------------------------------------------------

# (tau, delta) pairs that are jointly admissible for the BASE BB1 at
# tau_base = -tau (delta*(1-tau_base) < 1 <=> delta < 1/(1-tau_base)).
_TAU_DELTA = [(-0.3, 1.2), (-0.6, 2.0), (-0.8, 3.0), (-0.9, 5.0)]
_G = np.linspace(1e-3, 1 - 1e-3, 25)
_GRID = np.array([(u, v) for u in _G for v in _G])


@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_bb190_cdf_matches_vinecopula_identity(tau, delta):
    """C90(u,v) == v − C_base(1−u, v) on a 25×25 grid, to machine precision."""
    base = CopulaBB1(tau_k=-tau, delta=delta)
    rot = CopulaBB190(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.cdf_array(_GRID)
    rhs = v - base.cdf_array(np.column_stack([1.0 - u, v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-12, f"max |C90 - (v - C(1-u,v))| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_bb1270_cdf_matches_vinecopula_identity(tau, delta):
    """C270(u,v) == u − C_base(u, 1−v) on a 25×25 grid, to machine precision."""
    base = CopulaBB1(tau_k=-tau, delta=delta)
    rot = CopulaBB1270(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.cdf_array(_GRID)
    rhs = u - base.cdf_array(np.column_stack([u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-12, f"max |C270 - (u - C(u,1-v))| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_bb190_pdf_matches_base_at_reflected_point(tau, delta):
    """c90(u,v) == c_base(1−u, v) — reuses the already-validated base BB1
    density instead of a fresh mpmath ground truth."""
    base = CopulaBB1(tau_k=-tau, delta=delta)
    rot = CopulaBB190(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.pdf_array(_GRID)
    rhs = base.pdf_array(np.column_stack([1.0 - u, v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-9, f"max |c90 - c_base(1-u,v)| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_bb1270_pdf_matches_base_at_reflected_point(tau, delta):
    """c270(u,v) == c_base(u, 1−v)."""
    base = CopulaBB1(tau_k=-tau, delta=delta)
    rot = CopulaBB1270(tau_k=tau, delta=delta)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.pdf_array(_GRID)
    rhs = base.pdf_array(np.column_stack([u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-9, f"max |c270 - c_base(u,1-v)| = {err:.3g}"


@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_bb190_h_matches_base_h_with_reflected_conditioning(tau, delta):
    """h90(v|u) == h_base(v | 1−u) — no ``1 −`` term (module docstring)."""
    base = CopulaBB1(tau_k=-tau, delta=delta)
    rot = CopulaBB190(tau_k=tau, delta=delta)
    rng = np.random.default_rng(0)
    us = rng.uniform(1e-3, 1 - 1e-3, 100)
    vs = rng.uniform(1e-3, 1 - 1e-3, 100)
    got = np.array([rot.conditional_cdf(v, u) for u, v in zip(us, vs)])
    ref = np.array([base.conditional_cdf(v, 1.0 - u) for u, v in zip(us, vs)])
    np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_bb1270_h_matches_one_minus_base_h_with_reflected_v(tau, delta):
    """h270(v|u) == 1 − h_base(1−v | u) — genuine ``1 −`` (module docstring)."""
    base = CopulaBB1(tau_k=-tau, delta=delta)
    rot = CopulaBB1270(tau_k=tau, delta=delta)
    rng = np.random.default_rng(1)
    us = rng.uniform(1e-3, 1 - 1e-3, 100)
    vs = rng.uniform(1e-3, 1 - 1e-3, 100)
    got = np.array([rot.conditional_cdf(v, u) for u, v in zip(us, vs)])
    ref = np.array([1.0 - base.conditional_cdf(1.0 - v, u) for u, v in zip(us, vs)])
    np.testing.assert_allclose(got, ref, atol=1e-10)


# --------------------------------------------------------------------------
# h / h⁻¹ round-trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_h_inv_h_round_trip(cls, tau, delta):
    cop = cls(tau_k=tau, delta=delta)
    rng = np.random.default_rng(2)
    us = rng.uniform(0.01, 0.99, 40)
    vs = rng.uniform(0.01, 0.99, 40)
    for u, v in zip(us, vs):
        w = cop.conditional_cdf(v, u)
        v_back = cop.inv_h(w, u)
        w_back = cop.conditional_cdf(v_back, u)
        assert w_back == pytest.approx(w, abs=1e-6)


# --------------------------------------------------------------------------
# τ, δ pass-through
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
@pytest.mark.parametrize("tau,delta", _TAU_DELTA)
def test_theta_delta_match_base_bb1_at_minus_tau(cls, tau, delta):
    """θ_rot == θ_BB1(−τ, δ), δ_rot == δ unchanged — a rotation reflects the
    support, not the shape parameters (module docstring of ``rotated.py``)."""
    rot = cls(tau_k=tau, delta=delta)
    base = CopulaBB1(tau_k=-tau, delta=delta)
    assert rot.theta == pytest.approx(base.theta, rel=1e-12)
    assert rot.delta == pytest.approx(delta, rel=1e-12)


# --------------------------------------------------------------------------
# constrain_params / constructible_params delegation (audit G1)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
def test_constructible_params_repairs_an_infeasible_delta(cls):
    """BB1 is the first base family whose joint constraint is not the
    identity: near the rotated family's own independence end (τ_k close to
    0⁻), the default δ = 1.5 is inadmissible (the base needs
    δ < 1/(1 − τ_base) ≈ 1), exactly as it is for BB1 itself
    (``CopulaBB1.constructible_params``'s own docstring)."""
    params = {"tau_k": -0.01, "delta": 5.0}
    with pytest.raises(CopulaParameterError):
        cls(**params)
    repaired = cls.constructible_params(params)
    assert repaired["tau_k"] == params["tau_k"]   # only delta moves
    assert 1.0 <= repaired["delta"] < 1.5
    cop = cls(**repaired)                          # must not raise
    assert cop.delta == pytest.approx(repaired["delta"])


@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
def test_constructible_params_is_identity_on_an_already_admissible_point(cls):
    params = {"tau_k": -0.6, "delta": 2.0}
    assert cls.constructible_params(params) == params


@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
def test_constrain_params_delegates_to_base_bb1_with_tau_negated(cls):
    """``constrain_params`` (the box-projection hook used by the joint MLE,
    ``pmcprg.copulas._fit._fit_two_parameter_mle``) round-trips through the
    base family's own hook, τ negated both ways."""
    params = {"tau_k": -0.01, "delta": 5.0}
    out = cls.constrain_params(params)
    base_equiv = dict(out)
    base_equiv["tau_k"] = -base_equiv["tau_k"]
    assert base_equiv == CopulaBB1.constrain_params(
        {"tau_k": -params["tau_k"], "delta": params["delta"]})


def test_one_parameter_rotations_are_unaffected_by_the_delegation():
    """Clayton/GH/Joe's rotations never needed this override — their base
    ``constrain_params``/``constructible_params`` are :class:`CopulaVirt`'s
    identity — so the round-trip through :class:`RotatedCopula`'s new
    generic delegation must still return the same dict unchanged."""
    from pmcprg.copulas.archimedean.rotated import CopulaClayton90
    params = {"tau_k": -0.4}
    assert CopulaClayton90.constrain_params(params) == params
    assert CopulaClayton90.constructible_params(params) == params


# --------------------------------------------------------------------------
# Sampling: negative τ, tail-asymmetry signature (measured, not assumed)
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tau,delta", [(-0.2, 1.1), (-0.5, 1.8), (-0.8, 3.0)])
def test_sample_has_uniform_margins_and_the_right_negative_tau(tau, delta):
    for cls in (CopulaBB190, CopulaBB1270):
        cop = cls(tau_k=tau, delta=delta)
        uv = cop.sample(n=20000, seed=42)
        assert kstest(uv[:, 0], "uniform").pvalue > 0.001
        assert kstest(uv[:, 1], "uniform").pvalue > 0.001
        tau_hat = kendalltau(uv[:, 0], uv[:, 1]).statistic
        assert abs(tau_hat - tau) < 0.03, f"{cls.__name__}: tau_hat={tau_hat}, expected {tau}"


def test_base_bb1_has_two_sided_corner_mass_unlike_one_signed_families():
    """The premise the rest of this section relies on: at (τ, δ) where both
    λ_L, λ_U > 0, a plain BB1 sample has mass in *both* diagonal corners —
    unlike Clayton (pure lower) or GH/Joe (pure upper)."""
    base = CopulaBB1(tau_k=0.6, delta=1.5)
    lam_L, lam_U = base.tail_dependence()
    assert lam_L > 0.0 and lam_U > 0.0
    uv = base.sample(n=40000, seed=7)
    t = 0.05
    lower_left = np.mean((uv[:, 0] < t) & (uv[:, 1] < t))
    upper_right = np.mean((uv[:, 0] > 1.0 - t) & (uv[:, 1] > 1.0 - t))
    assert lower_left > 0.0 and upper_right > 0.0
    # lambda_L > lambda_U at this (tau, delta) -> the lower-left corner should
    # carry more mass than the upper-right one.
    assert lower_left > upper_right


def test_90_and_270_both_populate_two_corners_mirrored_across_each_other():
    """The 90°/270° reflections of BB1's own two-sided asymmetry (section
    docstring of ``rotated.py``): BB190 relocates the base's lower-left mass
    to the lower-right corner and its upper-right mass to the upper-left
    corner; BB1270 does the mirror assignment (upper-left <- lower-left,
    lower-right <- upper-right). At (τ, δ) = (0.6, 1.5), λ_L > λ_U for the
    base (previous test), so BB190's lower-right corner should dominate its
    upper-left, while BB1270's upper-left should dominate its lower-right —
    the two rotations disagree on which corner dominates, unlike Clayton/GH/
    Joe's rotations, which agree with each other's *opposite* single corner."""
    tau, delta = -0.6, 1.5
    t = 0.05
    s90 = CopulaBB190(tau_k=tau, delta=delta).sample(n=40000, seed=7)
    s270 = CopulaBB1270(tau_k=tau, delta=delta).sample(n=40000, seed=7)

    def corners(uv):
        upper_left = np.mean((uv[:, 0] < t) & (uv[:, 1] > 1.0 - t))
        lower_right = np.mean((uv[:, 0] > 1.0 - t) & (uv[:, 1] < t))
        return upper_left, lower_right

    ul90, lr90 = corners(s90)
    ul270, lr270 = corners(s270)

    # Both corners are populated for both rotations (two-sided asymmetry) —
    # neither is anywhere close to zero, unlike the one-signed families.
    assert ul90 > 0.005 and lr90 > 0.005
    assert ul270 > 0.005 and lr270 > 0.005

    # BB190: lower-right (from the base's lower-left, the dominant corner at
    # this (tau, delta)) should dominate its own upper-left.
    assert lr90 > ul90, f"BB190 corners: upper-left={ul90}, lower-right={lr90}"
    # BB1270: upper-left (also from the base's lower-left) should dominate.
    assert ul270 > lr270, f"BB1270 corners: upper-left={ul270}, lower-right={lr270}"

    # And the two rotations are not the same copula.
    assert not np.allclose(np.sort(s90[:, 1]), np.sort(s270[:, 1]))


# --------------------------------------------------------------------------
# fit — the first two-parameter rotated family (tau AND delta recovery)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaBB190, CopulaBB1270])
@pytest.mark.parametrize("tau_true,delta_true", [(-0.2, 1.1), (-0.5, 1.8), (-0.85, 4.0)])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_recovers_negative_tau_and_delta_from_simulated_data(cls, tau_true, delta_true, method):
    """Both free parameters must come back close to their true values — δ
    was never exercised by the Clayton/GH/Joe rotations (one parameter
    only), so this is the first check of its kind for a rotated family."""
    cop = cls(tau_k=tau_true, delta=delta_true)
    uv = cop.sample(n=3000, seed=11)
    r = cls.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.06)
    assert r.tau_k < 0.0
    assert r.copula.params["tau_k"] == r.tau_k
    assert r.copula.delta == pytest.approx(delta_true, rel=0.25)


def test_fit_best_includes_bb1_rotations_in_other_method():
    """BB1 (and its rotations) always fit by MLE — τ alone cannot identify
    δ — so a ``method='tau'`` ``fit_best`` reports them in ``other_method``,
    not ranked, exactly like plain ``CopulaBB1`` (``FitBestResults``'s own
    docstring)."""
    from pmcprg.copulas._base import CopulaVirt
    cop = CopulaBB190(tau_k=-0.6, delta=2.0)
    uv = cop.sample(n=800, seed=3)
    results = CopulaVirt.fit_best(uv, families=[CopulaBB190, CopulaBB1270], method="tau")
    assert len(results) == 0
    assert {r.copula.__class__ for r in results.other_method} == {CopulaBB190, CopulaBB1270}


def test_fit_best_picks_the_correctly_rotated_family_at_low_delta():
    """At δ close to 1 BB1 is nearly one-sided (Clayton-like), so the two
    rotations are as separable as Clayton90/270 — the correctly-rotated
    family should win clearly. (At higher δ, both diagonal tails are
    comparable and 90°/270° become harder to tell apart — a real, reported
    property of this family, not a test flake; not asserted here.)"""
    from pmcprg.copulas._base import CopulaVirt
    cop = CopulaBB190(tau_k=-0.8, delta=1.1)
    uv = cop.sample(n=2000, seed=3)
    results = CopulaVirt.fit_best(uv, families=[CopulaBB190, CopulaBB1270], method="mle")
    assert len(results) == 2
    assert results[0].copula.__class__ is CopulaBB190


# --------------------------------------------------------------------------
# Reachability / registry sanity (same battery as every other family)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry", [BB190, BB1270])
def test_reachable_tau_bounds_is_none(entry):
    assert entry.klass.reachable_tau_bounds() is None


@pytest.mark.parametrize("entry", [BB190, BB1270])
def test_registry_bounds_are_constructible_or_refused_explicitly(entry):
    """RB-6, same battery as ``test_copula_limits.py``'s generic check.

    Unlike the one-parameter rotations, the registered τ bounds alone are
    not enough to build BB1 at the default δ near the independence end, so
    ``constructible_params`` (not a bare ``entry.klass(tau_k=tau)``) is the
    right way to reach a buildable point here."""
    from pmcprg.exceptions import CopulaError
    lo, hi = entry.value.TAU_MIN_MAX
    for tau in (lo, hi):
        params = entry.klass.constructible_params({"tau_k": tau})
        try:
            cop = entry.klass(**params)
        except CopulaError:
            continue
        vals = cop.logpdf_array(np.array([[0.3, 0.6], [0.5, 0.5]]))
        assert not np.any(np.isnan(vals)), f"{entry.value.SHORT_NAME} at τ={tau}: NaN log-density"


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_bb1_rotations_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _TAIL_TAUS, _file_table, _required_param_keys, _match_file_key,
    )
    for short in ("BB190", "BB1270"):
        assert short in _TAIL_TAUS
    table = _file_table()
    need = {k for k in _required_param_keys() if k[0] in ("BB190", "BB1270")}
    assert need, "no BB190/BB1270 cases registered in test_copula_limits.py"
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
