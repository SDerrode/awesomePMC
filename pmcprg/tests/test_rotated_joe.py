"""90°/270°-rotated Joe copulas (audit FR-8, second round).

Mirrors ``test_rotated_clayton.py``'s depth for the Joe base family: Joe
already exposed the kernel interface (used by ``SurvivalJoe`` in
``survival.py``), so ``CopulaJoe90``/``CopulaJoe270`` are the same two-line
``RotatedCopula90``/``RotatedCopula270`` subclasses as Clayton's, registered
as ``CopulaEnum.JOE90``/``JOE270``.

Complements ``test_copula_limits.py`` (which adds Joe90/Joe270 to its own
grid and high-precision reference file, derived from the *already validated*
base Joe's own decimal references reflected through
``C90(u,v) = v − C(1−u,v)`` / ``C270(u,v) = u − C(u,1−v)``) with checks
specific to the rotation mechanism itself:

* the VineCopula identity ``C_rot(u,v) == C_base(reflected)`` to machine
  precision (the primary correctness anchor the audit names);
* the two rotations give the *same* τ(θ) sign flip but are demonstrably
  *different* copulas (opposite tail asymmetry);
* h / h⁻¹ round-trips, fit recovery of a negative τ̂.

Sampling's corner-asymmetry check *is* repeated here (unlike some of the
other batteries, which run once on GH and are skipped on Joe as redundant):
Joe's own upper-tail dependence is markedly sharper than GH's at the same τ
(plain Joe at τ=0.6 puts ≈0.083 of its mass at the (1,1) corner vs. ≈0.030 at
(0,0), a bigger split than GH's ≈0.072/0.047), so it gives a cleaner,
less threshold-sensitive demonstration of the swap — closer to the Clayton
pilot's own 2.7× corner-mass ratio at the pilot's own threshold (0.1) and τ
(−0.6), whereas GH's weaker asymmetry needed a tighter corner threshold to
show the same ratio (see ``test_rotated_gh.py``).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import CopulaJoe, CopulaJoe90, CopulaJoe270, CopulaEnum
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS

JOE90 = CopulaEnum.JOE90
JOE270 = CopulaEnum.JOE270


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry,short,cls_name,klass", [
    (JOE90, "Joe90", "CopulaJoe90", CopulaJoe90),
    (JOE270, "Joe270", "CopulaJoe270", CopulaJoe270),
])
def test_registered_in_copula_enum(entry, short, cls_name, klass):
    assert entry.value.SHORT_NAME == short
    assert entry.value.CLASS_NAME == cls_name
    assert entry.klass is klass
    assert entry.value.AVAILABLE
    assert entry in CopulaEnum.available()


@pytest.mark.parametrize("entry", [JOE90, JOE270])
def test_tau_range_is_negative_only(entry):
    """Mirror of Joe's own [0+EPS, 1.0] one-sided range (audit FR-8)."""
    lo, hi = entry.value.TAU_MIN_MAX
    assert lo == pytest.approx(-1.0)
    assert hi == pytest.approx(-EPS)


@pytest.mark.parametrize("cls", [CopulaJoe90, CopulaJoe270])
@pytest.mark.parametrize("tau", [0.0, 0.5, 1.0])
def test_non_negative_tau_is_refused(cls, tau):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=tau)


@pytest.mark.parametrize("cls", [CopulaJoe90, CopulaJoe270])
def test_tau_at_or_beyond_minus_one_is_refused(cls):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=-1.0)
    with pytest.raises(CopulaParameterError):
        cls(tau_k=-1.5)


# --------------------------------------------------------------------------
# VineCopula identity — the audit's primary correctness anchor
# --------------------------------------------------------------------------

_TAUS = [-1e-4, -0.3, -0.6, -0.95]
_G = np.linspace(1e-3, 1 - 1e-3, 25)
_GRID = np.array([(u, v) for u in _G for v in _G])


@pytest.mark.parametrize("tau", _TAUS)
def test_joe90_cdf_matches_vinecopula_identity(tau):
    """C90(u,v) == v − C_base(1−u, v) on a 25×25 grid, to machine precision."""
    base = CopulaJoe(tau_k=-tau)
    rot = CopulaJoe90(tau_k=tau)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.cdf_array(_GRID)
    rhs = v - base.cdf_array(np.column_stack([1.0 - u, v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-12, f"max |C90 - (v - C(1-u,v))| = {err:.3g}"


@pytest.mark.parametrize("tau", _TAUS)
def test_joe270_cdf_matches_vinecopula_identity(tau):
    """C270(u,v) == u − C_base(u, 1−v) on a 25×25 grid, to machine precision."""
    base = CopulaJoe(tau_k=-tau)
    rot = CopulaJoe270(tau_k=tau)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.cdf_array(_GRID)
    rhs = u - base.cdf_array(np.column_stack([u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-12, f"max |C270 - (u - C(u,1-v))| = {err:.3g}"


@pytest.mark.parametrize("tau", _TAUS)
def test_joe90_pdf_matches_base_at_reflected_point(tau):
    """c90(u,v) == c_base(1−u, v), the stronger, cheaper check the task asks
    for: reuse the already-validated base Joe density instead of a fresh
    mpmath ground truth."""
    base = CopulaJoe(tau_k=-tau)
    rot = CopulaJoe90(tau_k=tau)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.pdf_array(_GRID)
    rhs = base.pdf_array(np.column_stack([1.0 - u, v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 3e-9, f"max |c90 - c_base(1-u,v)| = {err:.3g}"


@pytest.mark.parametrize("tau", _TAUS)
def test_joe270_pdf_matches_base_at_reflected_point(tau):
    """c270(u,v) == c_base(u, 1−v)."""
    base = CopulaJoe(tau_k=-tau)
    rot = CopulaJoe270(tau_k=tau)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lhs = rot.pdf_array(_GRID)
    rhs = base.pdf_array(np.column_stack([u, 1.0 - v]))
    err = np.max(np.abs(lhs - rhs))
    assert err < 3e-9, f"max |c270 - c_base(u,1-v)| = {err:.3g}"


@pytest.mark.parametrize("tau", _TAUS)
def test_joe90_h_matches_base_h_with_reflected_conditioning(tau):
    """h90(v|u) == h_base(v | 1−u) — no ``1 −`` term (module docstring)."""
    base = CopulaJoe(tau_k=-tau)
    rot = CopulaJoe90(tau_k=tau)
    rng = np.random.default_rng(0)
    us = rng.uniform(1e-3, 1 - 1e-3, 100)
    vs = rng.uniform(1e-3, 1 - 1e-3, 100)
    got = np.array([rot.conditional_cdf(v, u) for u, v in zip(us, vs)])
    ref = np.array([base.conditional_cdf(v, 1.0 - u) for u, v in zip(us, vs)])
    np.testing.assert_allclose(got, ref, atol=1e-10)


@pytest.mark.parametrize("tau", _TAUS)
def test_joe270_h_matches_one_minus_base_h_with_reflected_v(tau):
    """h270(v|u) == 1 − h_base(1−v | u) — genuine ``1 −`` (module docstring)."""
    base = CopulaJoe(tau_k=-tau)
    rot = CopulaJoe270(tau_k=tau)
    rng = np.random.default_rng(1)
    us = rng.uniform(1e-3, 1 - 1e-3, 100)
    vs = rng.uniform(1e-3, 1 - 1e-3, 100)
    got = np.array([rot.conditional_cdf(v, u) for u, v in zip(us, vs)])
    ref = np.array([1.0 - base.conditional_cdf(1.0 - v, u) for u, v in zip(us, vs)])
    np.testing.assert_allclose(got, ref, atol=1e-10)


# --------------------------------------------------------------------------
# h / h⁻¹ round-trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaJoe90, CopulaJoe270])
@pytest.mark.parametrize("tau", _TAUS)
def test_h_inv_h_round_trip(cls, tau):
    cop = cls(tau_k=tau)
    rng = np.random.default_rng(2)
    us = rng.uniform(0.01, 0.99, 40)
    vs = rng.uniform(0.01, 0.99, 40)
    for u, v in zip(us, vs):
        w = cop.conditional_cdf(v, u)
        v_back = cop.inv_h(w, u)
        w_back = cop.conditional_cdf(v_back, u)
        assert w_back == pytest.approx(w, abs=1e-6)


# --------------------------------------------------------------------------
# τ sign relationship
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaJoe90, CopulaJoe270])
@pytest.mark.parametrize("tau", [-1e-4, -0.3, -0.6, -0.95])
def test_theta_matches_base_joe_at_minus_tau(cls, tau):
    """θ_rot == θ_Joe(−τ) — the rotation only negates τ, θ itself is the
    base family's own map (module docstring: 'a rotation reflects the
    copula's support, not its shape parameters')."""
    rot = cls(tau_k=tau)
    base = CopulaJoe(tau_k=-tau)
    assert rot.theta == pytest.approx(base.theta, rel=1e-12)


# --------------------------------------------------------------------------
# Sampling: negative τ, and the two rotations are demonstrably different
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("tau", [-0.2, -0.5, -0.8])
def test_sample_has_uniform_margins_and_the_right_negative_tau(tau):
    for cls in (CopulaJoe90, CopulaJoe270):
        cop = cls(tau_k=tau)
        uv = cop.sample(n=20000, seed=42)
        assert kstest(uv[:, 0], "uniform").pvalue > 0.001
        assert kstest(uv[:, 1], "uniform").pvalue > 0.001
        tau_hat = kendalltau(uv[:, 0], uv[:, 1]).statistic
        assert abs(tau_hat - tau) < 0.03, f"{cls.__name__}: tau_hat={tau_hat}, expected {tau}"


def test_90_and_270_are_different_copulas_with_swapped_tail_asymmetry():
    """Same τ (both negative), but not the same joint law — and the *sign*
    of the swap is the mirror image of the Clayton pilot's, because Joe's
    own tail asymmetry is upper (mass at (1,1)), not lower (mass at (0,0))
    like Clayton's: reflecting (U,V)~Joe (concentrated near (1,1)) through
    (1−U,V) pushes that mass to (0,1), the upper-left corner, so Joe90
    concentrates *there*, not near the lower-right like Clayton90; Joe270
    is the mirror image, near the lower-right (u→1, v→0). A plain Joe
    sample at τ=0.6 has corner mass ≈0.083 at (1,1) vs. ≈0.030 at (0,0) —
    a sharper split than GH's, at the same τ (module docstring), which is
    why this check (module docstring) uses Joe rather than GH."""
    tau = -0.6
    s90 = CopulaJoe90(tau_k=tau).sample(n=40000, seed=7)
    s270 = CopulaJoe270(tau_k=tau).sample(n=40000, seed=7)

    def corner_masses(uv):
        upper_left = np.mean((uv[:, 0] < 0.1) & (uv[:, 1] > 0.9))
        lower_right = np.mean((uv[:, 0] > 0.9) & (uv[:, 1] < 0.1))
        return upper_left, lower_right

    ul90, lr90 = corner_masses(s90)
    ul270, lr270 = corner_masses(s270)

    # Joe90: upper-left corner should dominate (Joe's upper-tail dependence
    # reflected through the first margin only) — opposite of Clayton90.
    assert ul90 > 2.0 * lr90, f"Joe90 corners: upper-left={ul90}, lower-right={lr90}"
    # Joe270: lower-right corner should dominate (mirror image of 90°).
    assert lr270 > 2.0 * ul270, f"Joe270 corners: upper-left={ul270}, lower-right={lr270}"
    # And the two rotations should not coincide.
    assert not np.allclose(np.sort(s90[:, 1]), np.sort(s270[:, 1]))


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [CopulaJoe90, CopulaJoe270])
@pytest.mark.parametrize("tau_true", [-0.2, -0.5, -0.8])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_recovers_negative_tau_from_simulated_data(cls, tau_true, method):
    cop = cls(tau_k=tau_true)
    uv = cop.sample(n=3000, seed=11)
    r = cls.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.06)
    assert r.tau_k < 0.0
    assert r.copula.params["tau_k"] == r.tau_k


def test_fit_best_includes_joe_rotations():
    from pmcprg.copulas._base import CopulaVirt
    cop = CopulaJoe90(tau_k=-0.6)
    uv = cop.sample(n=800, seed=3)
    results = CopulaVirt.fit_best(uv, families=[CopulaJoe90, CopulaJoe270], method="tau")
    assert len(results) == 2
    # The correctly-rotated family should fit strictly better (lower AIC).
    assert results[0].copula.__class__ is CopulaJoe90


# --------------------------------------------------------------------------
# Reachability / registry sanity (same battery as every other family)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("entry", [JOE90, JOE270])
def test_reachable_tau_bounds_is_none(entry):
    """No parameter-map cap narrower than the registered range (unlike
    Frank/Plackett/Galambos) — every registered τ is reachable."""
    assert entry.klass.reachable_tau_bounds() is None


@pytest.mark.parametrize("entry", [JOE90, JOE270])
def test_registry_bounds_are_constructible_or_refused_explicitly(entry):
    """RB-6, same battery as ``test_copula_limits.py``'s generic check."""
    from pmcprg.exceptions import CopulaError
    lo, hi = entry.value.TAU_MIN_MAX
    for tau in (lo, hi):
        try:
            cop = entry.klass(tau_k=tau)
        except CopulaError:
            continue
        vals = cop.logpdf_array(np.array([[0.3, 0.6], [0.5, 0.5]]))
        assert not np.any(np.isnan(vals)), f"{entry.value.SHORT_NAME} at τ={tau}: NaN log-density"


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_joe_rotations_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _TAIL_TAUS, _INDEP_TAUS, _file_table, _required_param_keys, _match_file_key,
    )
    for short in ("Joe90", "Joe270"):
        assert short in _TAIL_TAUS
        assert short in _INDEP_TAUS
    table = _file_table()
    need = {k for k in _required_param_keys() if k[0] in ("Joe90", "Joe270")}
    assert need, "no Joe90/Joe270 cases registered in test_copula_limits.py"
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
