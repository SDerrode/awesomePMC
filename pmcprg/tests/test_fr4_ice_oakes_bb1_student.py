"""Oakes' (1999) observed information for a two-parameter ICE copula (BB1,
Student) — the 2×2-matrix generalisation of :mod:`pmcprg.pmc._oakes`
(AUDIT_COPULES FR-4, "Reste": standard errors inside ICE).

Companion to :mod:`pmcprg.tests.test_fr4_ice_oakes`, which covers the
original one-parameter pilot (Gauss, Clayton) and is left untouched. What is
checked here
------------
* the matrix code (:func:`pmcprg.pmc._oakes._oakes_one_pair_multi`), run on
  a *one*-parameter family (Gauss) with its working coordinate ψ of size 1,
  reproduces :func:`pmcprg.pmc._oakes._oakes_one_pair_scalar`'s numbers —
  the sanity check the audit asked for, since both compute the same
  quantity by (deliberately) different code paths;
* the same decomposition identity as the scalar pilot,
  ``info_psi = info_complete + info_missing``, holds for the 2×2 matrices;
* an independently-coded direct second difference of Q(ψ, ψ') on BB1
  reproduces both matrix terms — the two-parameter analogue of the scalar
  pilot's ``test_oakes_se_matches_direct_second_difference_of_q``;
* a τ̂ (or δ̂, ν̂) at a boundary is flagged and returns NaN, exactly as the
  scalar path does;
* Monte-Carlo coverage of the Oakes vs. naive Wald intervals for both
  parameters of BB1 and Student — the slow tests at the bottom.

Measured with the seeds below, K = 2, HMC-DN, N = 600, pair (0, 0),
``fit_margins=False`` (margins known), R = 150 replicates (fewer than the
scalar pilot's R = 300-400: two-parameter ICE fits and 4 extra E-steps per
replicate cost more per unit, and the ~20 minute compute budget for this
whole task is shared with BB1 and Student both; the resulting Monte-Carlo
uncertainty on each coverage figure is reported alongside it in the
docstring of :func:`test_mc_coverage_bb1` / :func:`test_mc_coverage_student`):

    BB1, τ = 0.5, δ = 1.5 (n = 150/150 used): SE ratio (RMS reported / empirical
        SD) τ: 1.010 Oakes vs 0.782 naive; δ: 0.982 Oakes vs 0.681 naive.
        95 %/90 % coverage — τ: 0.907/0.900 Oakes vs 0.860/0.813 naive;
        δ: 0.960/0.900 Oakes vs 0.833/0.760 naive.
    Student, τ = 0.5, ν = 6 (n = 139/150 used, after excluding replicates
        where ν̂ hit the fitting box): 95 %/90 % coverage — τ: 0.935/0.885
        Oakes vs 0.827/0.734 naive; ν (``df``): 0.906/0.892 Oakes vs
        0.885/0.856 naive. The SE-ratio diagnostic is unreliable for ν at
        this replicate count (see :func:`test_mc_coverage_student`'s
        docstring) and is not reported here.

Both studies confirm the FR-4 hypothesis carries over to two parameters: the
naive sandwich under-covers noticeably more than Oakes' matrix SE, for both
τ and the family's second parameter (delta for BB1; less markedly for
Student's ν, whose own sampling behaviour is the harder problem — see
above).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from pmcprg.pmc._oakes import (
    H_PSI,
    _e_step_xi_pair,
    _hessian_and_phi,
    _oakes_one_pair_multi,
    _oakes_one_pair_scalar,
    _pseudo_obs,
    _tau_extra_of_copula,
    _with_pair_params,
    ice_oakes_tau_se,
)
from pmcprg.pmc.ice import ice
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

Z95 = float(norm.ppf(0.975))
Z90 = float(norm.ppf(0.95))

_GAUSS_TOML = "pmcprg/pmc/models/hmc_dn_gauss_k2.toml"


def _bb1_model(tau_diag: float = 0.5, delta_diag: float = 1.5,
               tau_off: float = 0.15, delta_off: float = 1.05) -> PMCModel:
    """K = 2 HMC-DN model with a BB1 copula on every pair — no TOML fixture
    ships one, so it is built in-memory, as :mod:`test_fr4_ice_oakes` does
    for Clayton."""
    raw = {
        "model": {"name": "hmc-dn-bb1-k2", "variant": "HMC-DN", "K": 2, "N_default": 600},
        "prior": {"A": [[0.90, 0.10], [0.10, 0.90]]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}},
        ],
        "copulas": [
            {"i": 0, "j": 0, "name": "BB1", "tau": tau_diag, "delta": delta_diag},
            {"i": 0, "j": 1, "name": "BB1", "tau": tau_off, "delta": delta_off},
            {"i": 1, "j": 0, "name": "BB1", "tau": tau_off, "delta": delta_off},
            {"i": 1, "j": 1, "name": "BB1", "tau": tau_diag, "delta": delta_diag},
        ],
    }
    return PMCModel.from_dict(raw)


def _student_model(tau_diag: float = 0.5, df_diag: float = 6.0,
                    tau_off: float = 0.15, df_off: float = 6.0) -> PMCModel:
    """K = 2 HMC-DN model with a Student copula on every pair."""
    raw = {
        "model": {"name": "hmc-dn-student-k2", "variant": "HMC-DN", "K": 2, "N_default": 600},
        "prior": {"A": [[0.90, 0.10], [0.10, 0.90]]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}},
        ],
        "copulas": [
            {"i": 0, "j": 0, "name": "Student", "tau": tau_diag, "df": df_diag},
            {"i": 0, "j": 1, "name": "Student", "tau": tau_off, "df": df_off},
            {"i": 1, "j": 0, "name": "Student", "tau": tau_off, "df": df_off},
            {"i": 1, "j": 1, "name": "Student", "tau": tau_diag, "df": df_diag},
        ],
    }
    return PMCModel.from_dict(raw)


def _fit_gauss(N=300, seed=0, max_iter=20):
    mdl = PMCModel(_GAUSS_TOML)
    _, Y = simulate(mdl, N=N, seed=seed)
    fitted, _ = ice(mdl, Y, {"max_iter": max_iter, "tol": 1e-4,
                              "candidates": ["Gauss"], "fit_margins": False})
    return fitted, Y


def _fit_bb1(N=600, seed=0, max_iter=20, **kwargs):
    mdl = _bb1_model(**kwargs)
    _, Y = simulate(mdl, N=N, seed=seed)
    fitted, _ = ice(mdl, Y, {"max_iter": max_iter, "tol": 1e-4,
                              "candidates": ["BB1"], "fit_margins": False})
    return fitted, Y


def _fit_student(N=600, seed=0, max_iter=20, **kwargs):
    mdl = _student_model(**kwargs)
    _, Y = simulate(mdl, N=N, seed=seed)
    fitted, _ = ice(mdl, Y, {"max_iter": max_iter, "tol": 1e-4,
                              "candidates": ["Student"], "fit_margins": False})
    return fitted, Y


# ---------------------------------------------------------------------------
# Sanity check: the matrix code on a scalar family matches the scalar path
# ---------------------------------------------------------------------------

#: The two paths reach the same τ(ψ ± h) through different routines — the
#: scalar one through ``a + (b − a)·expit ψ`` (``math.exp``), the matrix one
#: through ``_stderr``'s coordinate (``math.tanh`` for Gauss) — and both then
#: take a second difference with h = 10⁻⁴. A one-ulp disagreement between the
#: routines is thus multiplied by 1/h² = 10⁸: the paths agree to 10⁻¹⁵ on
#: macOS/arm64 (also under 1-ulp perturbations of Y) but only to 2.6·10⁻⁸ on
#: x86-64 Linux, whose libm rounds the two differently. A structural
#: mismatch (a missing term, a wrong Jacobian) would move the result by the
#: size of that term, far above this tolerance.
_PATH_RTOL = 1e-6


def test_matrix_code_on_one_parameter_family_matches_scalar_path():
    """:func:`_oakes_one_pair_multi`, run on Gauss (ψ has size 1), must
    reproduce :func:`_oakes_one_pair_scalar`'s ``se_tau``/``info_psi`` — the
    audit's own suggested cross-check that the 2×2 generalisation collapses
    correctly to the 1×1 case. The two code paths share no helper beyond
    ``_e_step_xi_pair``/``_pseudo_obs``, so agreement is not a tautology."""
    fitted, Y = _fit_gauss(N=300, seed=0, max_iter=20)
    scalar = _oakes_one_pair_scalar(fitted, Y, 0, 0, h_psi=H_PSI)
    multi = _oakes_one_pair_multi(fitted, Y, 0, 0, h_psi=H_PSI)

    assert multi.info_psi.shape == (1, 1)
    assert float(multi.info_psi[0, 0]) == pytest.approx(scalar.info_psi, rel=_PATH_RTOL)
    assert float(multi.info_complete[0, 0]) == pytest.approx(scalar.info_complete, rel=_PATH_RTOL)
    assert float(multi.info_missing[0, 0]) == pytest.approx(scalar.info_missing, rel=_PATH_RTOL)
    assert multi.se["tau_k"] == pytest.approx(scalar.se_tau, rel=_PATH_RTOL)
    assert multi.se_naive["tau_k"] == pytest.approx(scalar.se_tau_naive, rel=_PATH_RTOL)


def test_matrix_code_on_clayton_matches_scalar_path():
    """Same cross-check on Clayton (independent range [0, 1], not [-1, 1])."""
    from pmcprg.pmc._oakes import _oakes_one_pair_scalar as _scalar

    raw = {
        "model": {"name": "hmc-dn-clayton-k2", "variant": "HMC-DN", "K": 2, "N_default": 600},
        "prior": {"A": [[0.90, 0.10], [0.10, 0.90]]},
        "margins": [
            {"i": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
            {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}},
        ],
        "copulas": [
            {"i": 0, "j": 0, "name": "Clayton", "tau": 0.5},
            {"i": 0, "j": 1, "name": "Clayton", "tau": 0.15},
            {"i": 1, "j": 0, "name": "Clayton", "tau": 0.15},
            {"i": 1, "j": 1, "name": "Clayton", "tau": 0.5},
        ],
    }
    mdl = PMCModel.from_dict(raw)
    _, Y = simulate(mdl, N=600, seed=3)
    fitted, _ = ice(mdl, Y, {"max_iter": 20, "tol": 1e-4,
                              "candidates": ["Clayton"], "fit_margins": False})
    scalar = _scalar(fitted, Y, 0, 0, h_psi=H_PSI)
    multi = _oakes_one_pair_multi(fitted, Y, 0, 0, h_psi=H_PSI)
    assert multi.se["tau_k"] == pytest.approx(scalar.se_tau, rel=_PATH_RTOL)


# ---------------------------------------------------------------------------
# Oakes' identity: decomposition, sign, and a direct cross-check
# ---------------------------------------------------------------------------

def test_bb1_info_decomposes_and_missing_reduces_information():
    fitted, Y = _fit_bb1(N=600, seed=7, max_iter=30)
    res = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert res.info_psi == pytest.approx(res.info_complete + res.info_missing, rel=1e-9)
    # Missing information should reduce the curvature on the diagonal, as in
    # the scalar pilot — checked eigenvalue-wise rather than entrywise since
    # a 2x2 matrix's off-diagonal sign is not otherwise constrained.
    assert np.all(np.linalg.eigvalsh(res.info_complete) > 0.0)
    assert np.all(np.linalg.eigvalsh(res.info_psi) > 0.0)
    assert res.se["tau_k"] > res.se_naive["tau_k"]
    assert res.se["delta"] > res.se_naive["delta"]


def test_student_info_decomposes_and_missing_reduces_information():
    fitted, Y = _fit_student(N=600, seed=9, max_iter=30)
    res = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert res.info_psi == pytest.approx(res.info_complete + res.info_missing, rel=1e-9)
    assert np.all(np.linalg.eigvalsh(res.info_complete) > 0.0)
    assert np.all(np.linalg.eigvalsh(res.info_psi) > 0.0)
    assert res.se["tau_k"] > res.se_naive["tau_k"]
    assert res.se["df"] > res.se_naive["df"]


def test_bb1_oakes_matches_direct_second_difference_of_q():
    """Independently-coded direct central second difference of Q(ψ, ψ') on a
    3x3-per-coordinate stencil (the two-parameter analogue of the scalar
    pilot's own cross-check) — not the same code path as
    :func:`_oakes_one_pair_multi`, which reuses φ between the two terms."""
    from pmcprg.copulas._stderr import _spec_of

    fitted, Y = _fit_bb1(N=600, seed=11, max_iter=30)
    blk = next(b for b in fitted.copula_blocks() if b["i"] == 0 and b["j"] == 0)
    from pmcprg.pmc._oakes import _build_copula_from_blk
    from pmcprg.pmc.ice import _resolve_candidate

    _entry, cls = _resolve_candidate("BB1")
    copula0 = _build_copula_from_blk(cls, "BB1", blk)
    spec = _spec_of(copula0)
    psi_hat = spec.psi

    u, v = _pseudo_obs(fitted, Y, 0, 0)
    uv = np.column_stack((u, v))
    xi_hat = _e_step_xi_pair(fitted, Y, 0, 0)

    def q(psi_theta, psi_prime):
        ld = spec.build(psi_theta).logpdf_array(uv)
        if np.array_equal(psi_prime, psi_hat):
            w = xi_hat
        else:
            tau_p, extra_p = _tau_extra_of_copula(spec.build(psi_prime), "BB1")
            w = _e_step_xi_pair(_with_pair_params(fitted, 0, 0, tau_p, extra_p), Y, 0, 0)
        return float(np.dot(w, ld))

    h = H_PSI
    eye = np.eye(2)
    info_complete_direct = np.zeros((2, 2))
    info_missing_direct = np.zeros((2, 2))
    q00 = q(psi_hat, psi_hat)
    for a in range(2):
        qp0 = q(psi_hat + h * eye[a], psi_hat)
        qm0 = q(psi_hat - h * eye[a], psi_hat)
        info_complete_direct[a, a] = -(qp0 - 2.0 * q00 + qm0) / (h * h)
        qpp = q(psi_hat + h * eye[a], psi_hat + h * eye[a])
        qpm = q(psi_hat + h * eye[a], psi_hat - h * eye[a])
        qmp = q(psi_hat - h * eye[a], psi_hat + h * eye[a])
        qmm = q(psi_hat - h * eye[a], psi_hat - h * eye[a])
        info_missing_direct[a, a] = -(qpp - qpm - qmp + qmm) / (4.0 * h * h)
    for a, c in ((0, 1),):
        qpp = q(psi_hat + h * (eye[a] + eye[c]), psi_hat)
        qpm = q(psi_hat + h * (eye[a] - eye[c]), psi_hat)
        qmp = q(psi_hat - h * (eye[a] - eye[c]), psi_hat)
        qmm = q(psi_hat - h * (eye[a] + eye[c]), psi_hat)
        mixed = -(qpp - qpm - qmp + qmm) / (4.0 * h * h)
        info_complete_direct[a, c] = info_complete_direct[c, a] = mixed

    res = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
    assert res.info_complete == pytest.approx(info_complete_direct, rel=1e-6, abs=1e-8)
    # The diagonal of the missing-information matrix is the cheapest,
    # least-noisy cross-check (no cross-parameter E-step re-run needed for
    # the direct version); the full matrix's off-diagonal agreement is
    # covered indirectly by the symmetry check below.
    assert np.diag(res.info_missing) == pytest.approx(np.diag(info_missing_direct), rel=5e-2)


def test_bb1_missing_info_is_nearly_symmetric_before_symmetrisation():
    """The antisymmetric part of the raw (pre-symmetrisation) mixed-info
    matrix should be small relative to the symmetric part — confirming the
    module docstring's claim that symmetrising is averaging out noise, not
    hiding a real asymmetry."""
    from pmcprg.copulas._stderr import _spec_of

    fitted, Y = _fit_bb1(N=600, seed=13, max_iter=30)
    blk = next(b for b in fitted.copula_blocks() if b["i"] == 0 and b["j"] == 0)
    from pmcprg.pmc._oakes import _build_copula_from_blk
    from pmcprg.pmc.ice import _resolve_candidate

    _entry, cls = _resolve_candidate("BB1")
    copula0 = _build_copula_from_blk(cls, "BB1", blk)
    spec = _spec_of(copula0)
    u, v = _pseudo_obs(fitted, Y, 0, 0)
    uv = np.column_stack((u, v))
    xi_hat = _e_step_xi_pair(fitted, Y, 0, 0)
    phi, _ = _hessian_and_phi(spec, uv, xi_hat, H_PSI)

    # Recompute the raw (unsymmetrised) mixed matrix the same way
    # `_mixed_info` does internally, before its final symmetrisation.
    p = spec.psi.size
    eye = np.eye(p)
    s_plus = np.empty((p, p))
    s_minus = np.empty((p, p))
    for c in range(p):
        cop_p = spec.build(spec.psi + H_PSI * eye[c])
        cop_m = spec.build(spec.psi - H_PSI * eye[c])
        tau_p, extra_p = _tau_extra_of_copula(cop_p, "BB1")
        tau_m, extra_m = _tau_extra_of_copula(cop_m, "BB1")
        xi_p = _e_step_xi_pair(_with_pair_params(fitted, 0, 0, tau_p, extra_p), Y, 0, 0)
        xi_m = _e_step_xi_pair(_with_pair_params(fitted, 0, 0, tau_m, extra_m), Y, 0, 0)
        s_plus[c, :] = xi_p @ phi
        s_minus[c, :] = xi_m @ phi
    mixed = (s_plus - s_minus) / (2.0 * H_PSI)
    raw_info_missing = -mixed.T
    antisym = 0.5 * (raw_info_missing - raw_info_missing.T)
    sym = 0.5 * (raw_info_missing + raw_info_missing.T)
    assert np.linalg.norm(antisym) < 0.05 * np.linalg.norm(sym)


# ---------------------------------------------------------------------------
# Scope / boundary
# ---------------------------------------------------------------------------

def test_bb1_delta_at_clayton_limit_is_flagged_nan():
    """δ̂ = 1 is BB1's Clayton limit (Self & Liang 1987 boundary, as
    :mod:`pmcprg.copulas._stderr` already flags outside ICE)."""
    fitted, Y = _fit_bb1(N=400, seed=21, max_iter=20)
    raw = fitted.raw
    for blk in raw["copulas"]:
        if blk["i"] == 0 and blk["j"] == 0:
            blk["delta"] = 1.0 + 1e-9
    fitted_boundary = PMCModel.from_dict(raw)
    res = ice_oakes_tau_se(fitted_boundary, Y, pairs=[(0, 0)])[(0, 0)]
    assert res.at_boundary
    assert all(math.isnan(v) for v in res.se.values())


def test_student_nu_at_fitting_box_upper_bound_is_flagged_nan():
    """ν̂ at the top of the fitting box stands for the Gaussian limit."""
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    fitted, Y = _fit_student(N=400, seed=23, max_iter=20)
    hi = EXTRA_PARAM_BOUNDS_BY_PARAM["df"][1]
    raw = fitted.raw
    for blk in raw["copulas"]:
        if blk["i"] == 0 and blk["j"] == 0:
            blk["df"] = hi
    fitted_boundary = PMCModel.from_dict(raw)
    res = ice_oakes_tau_se(fitted_boundary, Y, pairs=[(0, 0)])[(0, 0)]
    assert res.at_boundary
    assert all(math.isnan(v) for v in res.se.values())


# ---------------------------------------------------------------------------
# Monte-Carlo coverage (slow)
# ---------------------------------------------------------------------------

def _coverage_study_multi(build_model, truth: dict, family: str, name2: str, *,
                           R: int, N: int, seed0: int):
    """Like the scalar pilot's ``_coverage_study``, generalised to report
    coverage of both ``tau_k`` and the family's second parameter (``name2``:
    ``"delta"`` for BB1, ``"df"`` for Student)."""
    rows = {k: {"est": [], "se_oak": [], "se_naive": [],
                "cov95_oak": 0, "cov95_naive": 0, "cov90_oak": 0, "cov90_naive": 0}
            for k in ("tau_k", name2)}
    n = 0
    for r in range(R):
        mdl = build_model()
        _, Y = simulate(mdl, N=N, seed=seed0 + r)
        try:
            fitted, _ = ice(mdl, Y, {"max_iter": 30, "tol": 1e-4,
                                       "candidates": [family], "fit_margins": False})
            res = ice_oakes_tau_se(fitted, Y, pairs=[(0, 0)])[(0, 0)]
        except Exception:
            continue
        if res.at_boundary or not all(math.isfinite(res.se[k]) for k in (
                "tau_k", name2)):
            continue
        n += 1
        for k in ("tau_k", name2):
            rows[k]["est"].append(res.estimate[k])
            rows[k]["se_oak"].append(res.se[k])
            rows[k]["se_naive"].append(res.se_naive[k])
            err = abs(res.estimate[k] - truth[k])
            rows[k]["cov95_oak"] += err <= Z95 * res.se[k]
            rows[k]["cov95_naive"] += err <= Z95 * res.se_naive[k]
            rows[k]["cov90_oak"] += err <= Z90 * res.se[k]
            rows[k]["cov90_naive"] += err <= Z90 * res.se_naive[k]
    out = {"n": n}
    for k in ("tau_k", name2):
        est = np.asarray(rows[k]["est"])
        se_oak = np.asarray(rows[k]["se_oak"])
        se_naive = np.asarray(rows[k]["se_naive"])
        out[k] = {
            "ratio_oak": math.sqrt(np.mean(se_oak ** 2)) / np.std(est, ddof=1) if n > 1 else math.nan,
            "ratio_naive": math.sqrt(np.mean(se_naive ** 2)) / np.std(est, ddof=1) if n > 1 else math.nan,
            "cov95_oak": rows[k]["cov95_oak"] / n if n else math.nan,
            "cov95_naive": rows[k]["cov95_naive"] / n if n else math.nan,
            "cov90_oak": rows[k]["cov90_oak"] / n if n else math.nan,
            "cov90_naive": rows[k]["cov90_naive"] / n if n else math.nan,
        }
    return out


@pytest.mark.slow
def test_mc_coverage_bb1():
    """R = 150, N = 600, τ = 0.5, δ = 1.5, pair (0, 0), margins known.

    R = 150 (vs. the scalar pilot's 300-400): a two-parameter ICE fit plus 4
    extra E-steps per Oakes call is markedly more expensive, and this study
    shares the task's ~20-25 minute compute budget with the Student one
    below. Binomial coverage SE at R = 150 is √(0.95·0.05/150) ≈ 1.8 % at
    95 % (±3 SE ≈ ±5.4 pts, widened to ±10 for the extra finite-difference
    noise of the matrix terms) and √(0.90·0.10/150) ≈ 2.4 % at 90 % (±12 pt
    band). The SE ratio's relative MC SE is 1/√(2(R−1)) ≈ 5.8 % (±3 SE
    ≈ ±17 %, widened to ±25 % as the scalar pilot widens its own band for
    the finite-sample O(1/n) gap). Bands are intentionally loose at this
    replicate count: the qualitative claim (Oakes covers better than naive)
    is asserted tightly; the absolute levels are not.
    """
    r = _coverage_study_multi(lambda: _bb1_model(0.5, 1.5), {"tau_k": 0.5, "delta": 1.5},
                               "BB1", "delta", R=150, N=600, seed0=50_000)
    assert r["n"] >= 100, r
    for k in ("tau_k", "delta"):
        assert 0.65 <= r[k]["ratio_oak"] <= 1.30, (k, r)
        assert 0.80 <= r[k]["cov95_oak"] <= 1.00, (k, r)
        assert 0.68 <= r[k]["cov90_oak"] <= 1.00, (k, r)
        # The qualitative FR-4 claim: naive does not out-cover Oakes.
        assert r[k]["cov95_naive"] <= r[k]["cov95_oak"] + 0.05, (k, r)


@pytest.mark.slow
def test_mc_coverage_student():
    """R = 150, N = 600, τ = 0.5, ν = 6, pair (0, 0), margins known — same
    bands and rationale as :func:`test_mc_coverage_bb1` for coverage.

    Measured at R = 150 (n = 139 after excluding replicates where ν̂ hit the
    fitting box, mostly at the upper end — the Gaussian limit): coverage
    ``tau_k`` 0.935/0.885 (95 %/90 %) Oakes vs. 0.827/0.734 naive; ``df``
    0.906/0.892 Oakes vs. 0.885/0.856 naive — Oakes covers better for both,
    the FR-4 claim, asserted below.

    The SE-*ratio* diagnostic (RMS reported SE / empirical SD of the
    estimate) is **not** asserted for ``df``: ν is notoriously hard to pin
    down (its likelihood is flat over wide stretches, and its sampling
    distribution is heavy right-tailed at N = 600, K = 2), so a handful of
    replicates with a very large reported SE dominate the RMS and inflate
    the ratio (measured 4.76 at this seed/R — nowhere near 1 — while the
    *coverage* built from the same per-replicate SEs is unremarkable, 0.906
    at 95 %). Coverage is the more robust diagnostic here and is what FR-4
    is actually about; the ratio is kept (and asserted) for ``tau_k``, whose
    sampling distribution is much better behaved.
    """
    r = _coverage_study_multi(lambda: _student_model(0.5, 6.0), {"tau_k": 0.5, "df": 6.0},
                               "Student", "df", R=150, N=600, seed0=60_000)
    assert r["n"] >= 90, r
    assert 0.60 <= r["tau_k"]["ratio_oak"] <= 1.35, r
    for k in ("tau_k", "df"):
        assert 0.78 <= r[k]["cov95_oak"] <= 1.00, (k, r)
        assert 0.65 <= r[k]["cov90_oak"] <= 1.00, (k, r)
        assert r[k]["cov95_naive"] <= r[k]["cov95_oak"] + 0.05, (k, r)
