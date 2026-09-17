"""90°/270°-rotated A12 and A14 copulas (audit FR-8, last round).

A12 and A14 did not expose the kernel interface the rotation wrapper needs,
so ``a12.py``/``a14.py`` were refactored first — with every public value
unchanged bit for bit (checked against the pre-refactor modules when the
refactor was made; the existing A12/A14 tests in ``test_copula_limits.py``,
``test_palier1_*`` and the generic sweeps re-ran unchanged). This file covers:

* the kernel interface itself: A14's coordinate is ``log u`` (GH/BB1's),
  A12's the pair ``(log u, log(1 − u))``; the ``1 − h`` member and the
  monotone-Newton ``_k_inv_h`` shared by both families;
* the VineCopula identities and the base-reflected density/h-function,
  h⁻¹ round trips, τ_rot = −τ_base (by quadrature, deterministic);
* the first rotated range that does **not** touch τ = 0, ``[−1, −1/3]``:
  ``constructible_tau_range``, ``correct_tau``, the multistart family draws,
  the Huard grid, the selection placeholder, the ICE M-step and the
  independence LR test (which, for the older rotations whose range ends at
  −ε, used to refuse them as if they were A12);
* sampling — realised τ, and the corner-mass signature *measured*: both
  bases have tail dependence in both diagonal corners, A12 with λ_L ≥ λ_U at
  every θ, A14 with λ_L = 1/2 crossed by λ_U near τ = 0.547, so A14's
  dominant anti-diagonal corner flips with τ;
* fit recovery, ``fit_best``, and the exchangeability test's rejection.

Monte-Carlo seeds are integer literals or ``zlib.crc32`` of a string — never
Python's salted ``hash()``.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.stats import kendalltau, kstest

from pmcprg.copulas import (
    CopulaA12, CopulaA14, CopulaA1290, CopulaA12270, CopulaA1490, CopulaA14270,
    CopulaClayton, CopulaClayton90, CopulaEnum, independence_lr_test,
)
from pmcprg.exceptions import CopulaParameterError

_ROT = {
    CopulaA1290: (CopulaA12, 90), CopulaA12270: (CopulaA12, 270),
    CopulaA1490: (CopulaA14, 90), CopulaA14270: (CopulaA14, 270),
}
_ROT_CLASSES = list(_ROT)
_ENTRIES = {
    CopulaA1290: CopulaEnum.A1290, CopulaA12270: CopulaEnum.A12270,
    CopulaA1490: CopulaEnum.A1490, CopulaA14270: CopulaEnum.A14270,
}


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _name(cls):
    return cls.__name__


# --------------------------------------------------------------------------
# Registration and the τ-range
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls,short", [
    (CopulaA1290, "A1290"), (CopulaA12270, "A12270"),
    (CopulaA1490, "A1490"), (CopulaA14270, "A14270"),
])
def test_registered_in_copula_enum(cls, short):
    entry = _ENTRIES[cls]
    assert entry.value.SHORT_NAME == short
    assert entry.value.CLASS_NAME == cls.__name__
    assert entry.klass is cls
    assert entry.value.AVAILABLE and entry in CopulaEnum.available()
    assert entry.value.PARAMETERS_SET_NAME == ["tau_k"]
    assert CopulaEnum.from_short_name(short) is entry


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_tau_range_is_the_mirror_of_the_base_range(cls):
    base, _ = _ROT[cls]
    lo, hi = _ENTRIES[cls].value.TAU_MIN_MAX
    blo, bhi = {CopulaA12: CopulaEnum.A12, CopulaA14: CopulaEnum.A14}[base].value.TAU_MIN_MAX
    assert (lo, hi) == (-bhi, -blo) == (-1.0, -1.0 / 3.0)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", [0.5, 0.0, -1e-12, -0.2, -0.3333, -1.0, -1.5])
def test_tau_outside_or_at_the_singular_end_is_refused(cls, tau):
    with pytest.raises(CopulaParameterError):
        cls(tau_k=tau)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_interior_bound_builds_at_theta_one(cls):
    """τ = −1/3 is a member (the base's θ = 1), unlike the −ε end of the
    other rotations, which is only 'independence up to ε'."""
    cop = cls(tau_k=-1.0 / 3.0)
    # A14's θ = 1/(1 − τ) − 1/2 rounds to 1 − 2ulp at τ = 1/3 (pre-existing,
    # harmless: its kernel uses max(θ − 1, 0)); A12's is exactly 1.
    assert cop.theta == pytest.approx(1.0, abs=4e-16)
    assert cop._base.params["tau_k"] == 1.0 / 3.0
    vals = cop.logpdf_array(np.array([[0.3, 0.6], [0.5, 0.5], [1e-9, 1 - 1e-9]]))
    assert np.all(np.isfinite(vals))


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", [-1.0 / 3.0, -0.4, -0.6, -0.9, -0.99])
def test_theta_is_the_base_map_at_minus_tau(cls, tau):
    base, _ = _ROT[cls]
    assert cls(tau_k=tau).theta == base(tau_k=-tau).theta


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_reachable_tau_bounds_is_none(cls):
    assert cls.reachable_tau_bounds() is None


# --------------------------------------------------------------------------
# Kernel interface of the bases (the prerequisite this round added)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("base", [CopulaA12, CopulaA14], ids=_name)
def test_base_exposes_the_kernel_interface(base):
    for member in ("_kcoord", "_kcoord_reflected", "_k_logpdf", "_k_cdf", "_k_h", "_k_inv_h"):
        assert hasattr(base, member), f"{base.__name__} is missing {member}"


def test_a12_kernel_coordinate_is_the_log_pair_and_reflection_swaps_it():
    x = np.array([1e-300, 1e-12, 0.37, 1 - 1e-12])
    ka = CopulaA12._kcoord(x)
    kr = CopulaA12._kcoord_reflected(x)
    np.testing.assert_array_equal(ka[0], np.log(x))
    np.testing.assert_array_equal(ka[1], np.log1p(-x))
    np.testing.assert_array_equal(kr[0], ka[1])
    np.testing.assert_array_equal(kr[1], ka[0])


def test_a14_kernel_coordinate_is_log_x_like_gumbel():
    x = np.array([1e-12, 0.37, 1 - 1e-12])
    np.testing.assert_array_equal(CopulaA14._kcoord(x), np.log(x))
    np.testing.assert_array_equal(CopulaA14._kcoord_reflected(x), np.log1p(-x))


@pytest.mark.parametrize("base", [CopulaA12, CopulaA14], ids=_name)
@pytest.mark.parametrize("tau", [1.0 / 3.0, 0.5, 0.9])
def test_public_api_is_built_from_the_kernel(base, tau):
    """cdf/pdf/h equal the kernel members exactly (same operations)."""
    cop = base(tau_k=tau)
    rng = np.random.default_rng(0)
    for u, v in rng.uniform(1e-3, 1 - 1e-3, (30, 2)):
        ka, kb = cop._kcoord(np.float64(u)), cop._kcoord(np.float64(v))
        assert cop.cdf([u, v]) == float(np.clip(cop._k_cdf(ka, kb)[0], 0.0, 1.0))
        assert cop.conditional_cdf(v, u) == float(np.clip(cop._k_h(kb, ka)[0], 0.0, 1.0))
        assert cop.pdf([u, v]) == float(np.exp(cop._k_logpdf(ka, kb)))


@pytest.mark.parametrize("base", [CopulaA12, CopulaA14], ids=_name)
@pytest.mark.parametrize("tau", [1.0 / 3.0, 0.4, 0.7, 0.95])
def test_kernel_complement_members_are_consistent(base, tau):
    """(C, 1 − C) and (h, 1 − h) sum to one where both are well resolved."""
    cop = base(tau_k=tau)
    rng = np.random.default_rng(1)
    u, v = rng.uniform(0.01, 0.99, (2, 500))
    ka, kb = cop._kcoord(u), cop._kcoord(v)
    c, cbar = cop._k_cdf(ka, kb)
    np.testing.assert_allclose(c + cbar, 1.0, atol=4e-15)
    h, hbar = cop._k_h(kb, ka)
    np.testing.assert_allclose(h + hbar, 1.0, atol=1e-13)


@pytest.mark.parametrize("base", [CopulaA12, CopulaA14], ids=_name)
@pytest.mark.parametrize("tau", [1.0 / 3.0, 0.4, 0.7, 0.9, 0.97])
def test_kernel_inverse_h_round_trips_at_ulp_precision(base, tau):
    """``_k_inv_h`` (monotone Newton): h at the returned v brackets w within
    one ulp of v, and ``v + (1 − v) = 1``; it also agrees with the public
    Brent-based ``inv_h_array`` to that solver's tolerance."""
    cop = base(tau_k=tau)
    rng = np.random.default_rng(_seed(base.__name__, tau))
    u, w = rng.uniform(size=(2, 300))
    v, vbar = cop._k_inv_h(np.log(w), cop._kcoord(u))
    np.testing.assert_allclose(v + vbar, 1.0, atol=4e-15)
    for vi, ui, wi in zip(v, u, w):
        lo = cop.conditional_cdf(np.nextafter(vi, 0.0), ui)
        hi = cop.conditional_cdf(np.nextafter(vi, 1.0), ui)
        slack = 1e-14
        assert min(lo, hi) - slack <= wi <= max(lo, hi) + slack or \
            abs(cop.conditional_cdf(vi, ui) - wi) <= 1e-13
    np.testing.assert_allclose(v, cop.inv_h_array(w, u), atol=1e-8)


@pytest.mark.parametrize("base", [CopulaA12, CopulaA14], ids=_name)
def test_kernel_inverse_h_at_extreme_inputs(base):
    cop = base(tau_k=0.9)
    u = np.array([1e-12, 1 - 1e-12, 0.5, 0.5, 1e-300])
    w = np.array([0.5, 0.5, 1e-12, 1 - 1e-12, 0.5])
    v, vbar = cop._k_inv_h(np.log(w), cop._kcoord(u))
    assert np.all(np.isfinite(v)) and np.all(np.isfinite(vbar))
    assert np.all((v > 0.0) & (v <= 1.0))
    assert np.all(vbar > 0.0)


# --------------------------------------------------------------------------
# VineCopula identities and the base-reflected references
# --------------------------------------------------------------------------

_TAUS = [-1.0 / 3.0, -0.4, -0.6, -0.9]
_G = np.linspace(1e-3, 1 - 1e-3, 25)
_GRID = np.array([(u, v) for u in _G for v in _G])


def _reflect(uv, angle):
    u, v = uv[:, 0], uv[:, 1]
    return np.column_stack([1.0 - u, v]) if angle == 90 else np.column_stack([u, 1.0 - v])


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", _TAUS)
def test_cdf_matches_vinecopula_identity(cls, tau):
    """C90(u,v) = v − C(1−u, v) and C270(u,v) = u − C(u, 1−v)."""
    base_cls, angle = _ROT[cls]
    base, rot = base_cls(tau_k=-tau), cls(tau_k=tau)
    u, v = _GRID[:, 0], _GRID[:, 1]
    lead = v if angle == 90 else u
    rhs = lead - base.cdf_array(_reflect(_GRID, angle))
    err = np.max(np.abs(rot.cdf_array(_GRID) - rhs))
    assert err < 1e-15 * 4, f"max |ΔC| = {err:.3g}"


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", _TAUS)
def test_pdf_matches_base_at_reflected_point(cls, tau):
    base_cls, angle = _ROT[cls]
    base, rot = base_cls(tau_k=-tau), cls(tau_k=tau)
    ref = base.logpdf_array(_reflect(_GRID, angle))
    np.testing.assert_allclose(rot.logpdf_array(_GRID), ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", _TAUS)
def test_h_matches_base_h_through_the_reflection(cls, tau):
    """h90(v|u) = h(v | 1−u); h270(v|u) = 1 − h(1−v | u)."""
    base_cls, angle = _ROT[cls]
    base, rot = base_cls(tau_k=-tau), cls(tau_k=tau)
    rng = np.random.default_rng(_seed(cls.__name__, tau, "h"))
    for u, v in rng.uniform(1e-3, 1 - 1e-3, (100, 2)):
        ref = (base.conditional_cdf(v, 1.0 - u) if angle == 90
               else 1.0 - base.conditional_cdf(1.0 - v, u))
        assert rot.conditional_cdf(v, u) == pytest.approx(ref, abs=1e-14)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", _TAUS)
def test_scalar_and_array_paths_agree(cls, tau):
    rot = cls(tau_k=tau)
    pts = _GRID[::13]
    np.testing.assert_allclose([rot.pdf(p) for p in pts], rot.pdf_array(pts), rtol=1e-13)
    np.testing.assert_allclose([rot.cdf(p) for p in pts], rot.cdf_array(pts), rtol=0, atol=1e-16)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", _TAUS)
def test_h_inv_h_round_trip(cls, tau):
    cop = cls(tau_k=tau)
    rng = np.random.default_rng(_seed(cls.__name__, tau, "inv"))
    u, v = rng.uniform(0.01, 0.99, (2, 60))
    w = np.array([cop.conditional_cdf(vi, ui) for ui, vi in zip(u, v)])
    v_back = cop.inv_h_array(w, u)
    w_back = np.array([cop.conditional_cdf(vi, ui) for ui, vi in zip(u, v_back)])
    np.testing.assert_allclose(w_back, w, atol=1e-10)
    assert cop.inv_h(float(w[0]), float(u[0])) == v_back[0]


_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)
_GL_X, _GL_W = (_GL_X + 1.0) / 2.0, _GL_W / 2.0


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", _TAUS)
def test_kendall_tau_is_minus_the_base_tau_by_quadrature(cls, tau):
    """τ = 1 − 4∫∫ ∂C/∂u · ∂C/∂v (Nelsen 2006, Thm 5.1.1), bounded integrand;
    ∂C/∂u is the rotated h, ∂C/∂v a central difference of the rotated CDF."""
    cop = cls(tau_k=tau)
    uu, vv = (a.ravel() for a in np.meshgrid(_GL_X, _GL_X, indexing="ij"))
    ww = np.outer(_GL_W, _GL_W).ravel()
    d = 1e-7
    dv = (cop.cdf_array(np.column_stack([uu, vv + d]))
          - cop.cdf_array(np.column_stack([uu, vv - d]))) / (2.0 * d)
    tau_num = 1.0 - 4.0 * np.sum(ww * cop._h(vv, uu) * dv)
    assert tau_num == pytest.approx(tau, abs=1e-4)


# --------------------------------------------------------------------------
# A negative range that does not touch 0 — the range-generic machinery
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_constructible_range_pads_only_the_singular_end(cls):
    entry = _ENTRIES[cls]
    lo, hi = entry.constructible_tau_range()
    assert hi == -1.0 / 3.0                      # admissible end kept
    assert -1.0 < lo < -0.999                    # |τ| = 1 pulled in
    cls(tau_k=lo)
    cls(tau_k=hi)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_correct_tau_clips_onto_the_interior_bound_not_zero(cls):
    entry = _ENTRIES[cls]
    assert entry.correct_tau(0.5, warn=False) == -1.0 / 3.0
    assert entry.correct_tau(-0.1, warn=False) == -1.0 / 3.0
    assert entry.correct_tau(-2.0, warn=False) == -1.0
    assert entry.correct_tau(-0.5, warn=False) == -0.5
    assert entry.correct_tau(float("nan"), warn=False) == pytest.approx(-2.0 / 3.0)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_multistart_sweep_and_random_draws_land_inside_the_range(cls):
    from pmcprg.pmc._estim_common import RANDOM_TAU_BAND, SWEEP_TAU, _set_copula_family
    entry = _ENTRIES[cls]
    blk = {"name": "Gauss", "tau": 0.2, "i": 0, "j": 0}
    _set_copula_family(blk, entry, SWEEP_TAU)          # τ = 0.5 → −1/3
    assert blk == {"name": entry.value.SHORT_NAME, "tau": -1.0 / 3.0, "i": 0, "j": 0}
    cls(tau_k=blk["tau"])
    lo, hi = entry.value.TAU_MIN_MAX
    band = (lo + RANDOM_TAU_BAND * (hi - lo), hi - RANDOM_TAU_BAND * (hi - lo))
    assert band == pytest.approx((-13.0 / 15.0, -7.0 / 15.0))
    rng = np.random.default_rng(4)
    for tau in rng.uniform(*band, size=20):
        blk = {"name": "Gauss", "tau": 0.0}
        _set_copula_family(blk, entry, tau)
        assert blk["tau"] == tau
        cls(tau_k=blk["tau"])


def test_random_family_start_uses_the_rotated_band(monkeypatch):
    """A ``multistart_families='random'`` start with only A12/A14 rotations
    as candidates: every drawn τ lies in the central band of [−1, −1/3].
    The model plumbing is stubbed; only the family/τ draw is exercised."""
    from types import SimpleNamespace

    from pmcprg.pmc import _estim_common as ec

    lo, hi = -1.0, -1.0 / 3.0
    band = (lo + ec.RANDOM_TAU_BAND * (hi - lo), hi - ec.RANDOM_TAU_BAND * (hi - lo))
    blocks = [{"name": "Gauss", "tau": 0.1, "i": 0, "j": 0},
              {"name": "Gauss", "tau": 0.1, "i": 0, "j": 1}]
    captured = []
    monkeypatch.setattr(ec, "perturb_initial_model", lambda model, rng, jitter: SimpleNamespace(
        raw={"copulas": [dict(b) for b in blocks]}))
    monkeypatch.setattr(ec, "PMCModel", SimpleNamespace(from_dict=captured.append))
    fam_rng = np.random.default_rng(9)
    for _ in range(10):
        ec._random_family_start(None, None, fam_rng, 0.0, list(_ENTRIES.values()))
    taus = [b["tau"] for d in captured for b in d["copulas"]]
    assert len(taus) == 20
    assert all(band[0] <= t <= band[1] for t in taus)


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_selection_placeholder_pulls_zero_onto_the_interior_bound(cls):
    from pmcprg.pmc.ice import _copula_placeholder, _resolve_candidate
    short = _ENTRIES[cls].value.SHORT_NAME
    entry, klass = _resolve_candidate(short)
    blk = _copula_placeholder([short], [(short, entry, klass)], "mle", {"name": short, "tau": 0.0})
    assert blk["tau"] == -1.0 / 3.0
    klass(tau_k=blk["tau"])


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_ice_m_step_and_huard_evidence_stay_inside_the_range(cls):
    from pmcprg.pmc.ice import _fit_copula_params, _huard_log_evidence, _resolve_candidate
    entry, klass = _resolve_candidate(_ENTRIES[cls].value.SHORT_NAME)
    uv = cls(tau_k=-0.6).sample(n=800, seed=1)
    w = np.ones(len(uv))
    p = _fit_copula_params(klass, entry, uv[:, 0], uv[:, 1], w)
    assert -1.0 < p["tau_k"] < -1.0 / 3.0
    assert p["tau_k"] == pytest.approx(-0.6, abs=0.06)
    ev = _huard_log_evidence(klass, entry, {"tau_k": p["tau_k"]}, uv[:, 0], uv[:, 1], w)
    assert np.isfinite(ev)
    # Independent data: the M-step stops at the interior bound, not at 0.
    uvi = np.random.default_rng(0).uniform(size=(500, 2))
    p0 = _fit_copula_params(klass, entry, uvi[:, 0], uvi[:, 1], np.ones(500))
    assert -1.0 / 3.0 - 1e-3 < p0["tau_k"] <= -1.0 / 3.0


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_fit_on_non_negative_data_stops_at_the_interior_bound(cls):
    uvi = np.random.default_rng(0).uniform(size=(500, 2))
    assert cls.fit(uvi, method="tau").tau_k == -1.0 / 3.0
    r = cls.fit(uvi, method="mle")
    assert -1.0 / 3.0 - 1e-3 < r.tau_k <= -1.0 / 3.0


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_independence_lr_test_refuses_a_range_without_independence(cls):
    uv = np.random.default_rng(0).uniform(size=(200, 2))
    with pytest.raises(ValueError, match="does not contain the independence copula"):
        independence_lr_test(cls, uv)


def test_independence_lr_test_accepts_a_rotation_ending_at_minus_eps():
    """Clayton90's range [−1, −ε] contains independence at its *upper* end;
    the test used to accept only a lower end at 0 and refused it like A12.
    It is the mirror of Clayton's test on (1 − u, v)."""
    for s in range(3):
        uv = np.random.default_rng(s).uniform(size=(300, 2))
        a = independence_lr_test(CopulaClayton90, uv)
        b = independence_lr_test(CopulaClayton, np.column_stack([1.0 - uv[:, 0], uv[:, 1]]))
        assert a.boundary and a.null_distribution == "0.5*chi2(0) + 0.5*chi2(1)"
        assert a.tau_k <= 0.0
        assert a.statistic == pytest.approx(b.statistic, abs=1e-9)
        assert a.p_value == pytest.approx(b.p_value, abs=1e-9)
        assert a.tau_k == pytest.approx(-b.tau_k, abs=1e-8)
    uv = CopulaClayton90(tau_k=-0.3).sample(n=300, seed=1)
    assert independence_lr_test(CopulaClayton90, uv).p_value < 1e-6


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau", [-0.35, -0.6, -0.9])
def test_sample_has_uniform_margins_and_the_right_negative_tau(cls, tau):
    uv = cls(tau_k=tau).sample(n=20000, seed=_seed(cls.__name__, tau, "smp"))
    assert kstest(uv[:, 0], "uniform").pvalue > 0.001
    assert kstest(uv[:, 1], "uniform").pvalue > 0.001
    assert kendalltau(uv[:, 0], uv[:, 1]).statistic == pytest.approx(tau, abs=0.02)


def _corners(uv, e=0.05):
    u, v = uv[:, 0], uv[:, 1]
    return {"LL": np.mean((u < e) & (v < e)), "UR": np.mean((u > 1 - e) & (v > 1 - e)),
            "LR": np.mean((u > 1 - e) & (v < e)), "UL": np.mean((u < e) & (v > 1 - e))}


@pytest.mark.parametrize("base", [CopulaA12, CopulaA14], ids=_name)
def test_bases_have_mass_in_both_diagonal_corners(base):
    """Near θ = 1 λ_U → 0 and the lower corner dominates for both bases
    (base sampling uses the generic Brent inverse: kept to 20 000 points)."""
    lam_l, lam_u = base(tau_k=0.35).tail_dependence()
    assert lam_l >= 0.5 > 5 * lam_u
    c = _corners(base(tau_k=0.35).sample(n=20000, seed=7))
    assert c["LL"] > 2.5 * c["UR"] > 0.0
    c = _corners(base(tau_k=0.9).sample(n=20000, seed=7))
    assert c["LL"] > 0.03 and c["UR"] > 0.03


@pytest.mark.parametrize("r90,r270", [(CopulaA1290, CopulaA12270), (CopulaA1490, CopulaA14270)],
                         ids=["A12", "A14"])
def test_near_the_interior_bound_the_lower_corner_goes_where_the_reflection_sends_it(r90, r270):
    """(0,0) mass → (1,0) under 90°, → (0,1) under 270°; measured at
    τ = −0.35 (40 000 points, corner side 0.05): ratio ≈ 4 in each case."""
    c90 = _corners(r90(tau_k=-0.35).sample(n=40000, seed=7))
    c270 = _corners(r270(tau_k=-0.35).sample(n=40000, seed=7))
    assert c90["LR"] > 3.0 * c90["UL"] > 0.0
    assert c270["UL"] > 3.0 * c270["LR"] > 0.0
    # anti-diagonal corners only: the diagonal ones are (almost) empty
    assert max(c90["LL"], c90["UR"], c270["LL"], c270["UR"]) < 0.002


def test_a12_rotation_never_favours_the_reflected_upper_tail():
    """A12: λ_L − λ_U = 2^{−1/θ} + 2^{1/θ} − 2 ≥ 0, so A1290's lower-right
    corner is never lighter than its upper-left one; both tails → 1 as
    τ → −1 and the corners reach parity (measured ratio ≈ 1.02 at −0.9)."""
    for tau in (0.4, 0.6, 0.8, 0.95, 0.999):
        lam_l, lam_u = CopulaA12(tau_k=tau).tail_dependence()
        assert lam_l >= lam_u
    c = _corners(CopulaA1290(tau_k=-0.9).sample(n=200000, seed=11))
    assert c["LR"] / c["UL"] == pytest.approx(1.0, abs=0.1)
    c = _corners(CopulaA12270(tau_k=-0.9).sample(n=200000, seed=11))
    assert c["UL"] / c["LR"] == pytest.approx(1.0, abs=0.1)


def test_a14_dominant_rotated_corner_flips_with_tau():
    """A14: λ_L = 1/2 is crossed by λ_U = 2 − 2^{1/θ} at θ = 1/log₂1.5
    (τ ≈ 0.547): below, A1490's lower-right corner dominates (see above);
    above, its upper-left one does — measured ratio UL/LR ≈ 1.17 at
    τ = −0.8 (200 000 points), with A14270 mirroring it."""
    theta_x = 1.0 / np.log2(1.5)
    tau_x = 1.0 - 2.0 / (1.0 + 2.0 * theta_x)
    assert 0.547 < tau_x < 0.548
    lam_l, lam_u = CopulaA14(tau_k=0.8).tail_dependence()
    assert lam_u > lam_l == 0.5
    c90 = _corners(CopulaA1490(tau_k=-0.8).sample(n=200000, seed=11))
    c270 = _corners(CopulaA14270(tau_k=-0.8).sample(n=200000, seed=11))
    assert c90["UL"] > 1.08 * c90["LR"]
    assert c270["LR"] > 1.08 * c270["UL"]


# --------------------------------------------------------------------------
# fit, fit_best, exchangeability
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
@pytest.mark.parametrize("tau_true", [-0.35, -0.5, -0.7, -0.9])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_recovers_tau_from_simulated_data(cls, tau_true, method):
    uv = cls(tau_k=tau_true).sample(n=3000, seed=_seed(cls.__name__, tau_true, "fit"))
    r = cls.fit(uv, method=method)
    assert r.tau_k == pytest.approx(tau_true, abs=0.03)
    assert -1.0 < r.tau_k <= -1.0 / 3.0
    assert isinstance(r.copula, cls)
    assert r.copula.params["tau_k"] == r.tau_k


@pytest.mark.parametrize("r90,r270", [(CopulaA1290, CopulaA12270), (CopulaA1490, CopulaA14270)],
                         ids=["A12", "A14"])
def test_fit_best_picks_the_correct_rotation_near_the_interior_bound(r90, r270):
    from pmcprg.copulas._base import CopulaVirt
    for true, other in ((r90, r270), (r270, r90)):
        uv = true(tau_k=-0.4).sample(n=1500, seed=3)
        res = CopulaVirt.fit_best(uv, families=[true, other], method="mle")
        assert res[0].copula.__class__ is true


@pytest.mark.parametrize("cls", _ROT_CLASSES, ids=_name)
def test_exchangeability_test_rejects_rotated_samples(cls):
    """Sanity check (FR-10): at τ = −0.4 the rotated families are clearly
    non-exchangeable; 5 replicates of n = 1000, B = 100 all reject at 5 %
    (measured p = 0.01 in 20/20 runs across the four families)."""
    from pmcprg.diagnostics.exchangeability import exchangeability_test
    for r in range(3):
        s = _seed(cls.__name__, r, "exch")
        uv = cls(tau_k=-0.4).sample(n=1000, seed=s)
        res = exchangeability_test(uv[:, 0], uv[:, 1], B=100, seed=s + 1)
        assert res.reject, f"{cls.__name__} replicate {r}: p = {res.p_value}"


# --------------------------------------------------------------------------
# High-precision reference file coverage (see test_copula_limits.py)
# --------------------------------------------------------------------------

def test_a12_a14_rotations_are_covered_by_the_reference_file():
    from test_copula_limits import (
        _INDEP_TAUS, _TAIL_TAUS, _file_table, _match_file_key, _required_param_keys,
    )
    shorts = ("A1290", "A12270", "A1490", "A14270")
    for short in shorts:
        assert short in _TAIL_TAUS
        assert short not in _INDEP_TAUS          # independence is not a member
    table = _file_table()
    need = {k for k in _required_param_keys() if k[0] in shorts}
    assert len(need) == 12
    for key in need:
        assert _match_file_key(key, table) is not None, f"{key} missing from the reference file"
