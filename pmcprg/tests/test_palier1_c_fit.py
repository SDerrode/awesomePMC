"""Estimation contracts of AUDIT_COPULES palier 1 not covered by test_copula_limits.py.

RB-3 / RB-8 — the joint (τ, extra) fits of ICE leave their start value, and a
failure is flagged and logged; RB-7 — zero weights and non-finite
log-densities in every objective; RB-6 — ``fit(method='tau')`` on comonotone
data; RB-10 — ``fit_best`` ranks like with like and reports failures.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from pmcprg.copulas import CopulaBB1, CopulaClayton, CopulaGaussian, CopulaStudent
from pmcprg.copulas._base import CopulaEnum, CopulaVirt
from pmcprg.copulas._fit import FitBestResults, _weighted_log_density_sum
from pmcprg.pmc.ice import (
    FIT_FAILED_KEY,
    _copula_kwargs,
    _fit_copula_params,
    _resolve_candidate,
    _select_and_fit_copula,
    _weighted_log_likelihood,
    _weighted_mle_tau,
)


def _ll(cop, uv, w=None):
    return _weighted_log_density_sum(cop.logpdf_array(uv), w)


# --------------------------------------------------------------------------
# RB-3 / RB-8 — two-parameter ICE fits
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tau,delta,seed,delta_range", [
    # Measured with the former (τ, δ) L-BFGS-B, ftol 1e-7 on a Σw-normalised
    # objective and δ clamped to [1, δ_max(τ)]: δ = 1.5000 (the start value)
    # in both cases, 4.6 nat and 46.9 nat below the fit below.
    (0.9, 2.0, 3, (1.9, 2.7)),
    (0.5, 1.0, 4, (1.0, 1.1)),
])
def test_ice_bb1_joint_fit_does_not_stall_at_the_start_value(tau, delta, seed, delta_range):
    entry, cls = _resolve_candidate("BB1")
    uv = CopulaBB1(tau_k=tau, delta=delta).sample(n=500, seed=seed)
    w = np.ones(len(uv))
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], w)
    assert FIT_FAILED_KEY not in p, p
    assert delta_range[0] <= p["delta"] <= delta_range[1], p
    fitted = _ll(CopulaBB1(**p), uv)
    start = _ll(CopulaBB1(tau_k=p["tau_k"], delta=1.5), uv) if p["tau_k"] > 1 / 3 else -math.inf
    assert fitted > start + 3.0


def test_ice_bb1_fit_reaches_the_gumbel_boundary():
    """Gumbel data: the BB1 optimum is θ → 0 with δ = 1/(1 − τ). The fit must
    get there (θ at its floor), not stop in the flat valley of log θ."""
    from pmcprg.copulas import CopulaGH
    entry, cls = _resolve_candidate("BB1")
    uv = CopulaGH(tau_k=0.6).sample(n=1500, seed=11)
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(len(uv)))
    cop = CopulaBB1(**p)
    assert cop.theta < 1e-4, p
    assert p["delta"] == pytest.approx(1.0 / (1.0 - p["tau_k"]), rel=1e-3)


@pytest.mark.parametrize("df_true,seed,df_range", [(10.0, 2, (6.0, 20.0)), (2.5, 7, (2.0, 3.5))])
def test_ice_student_fit_moves_off_the_start_value_with_weights(df_true, seed, df_range):
    """τ = 0.95 with ξ-like weights, some exactly zero (the former fit returned ν ≈ 4.0000)."""
    entry, cls = _resolve_candidate("Student")
    uv = CopulaStudent(tau_k=0.95, df=df_true).sample(n=800, seed=seed)
    w = np.random.default_rng(seed).uniform(0.2, 1.0, len(uv))
    w[::10] = 0.0
    p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], w)
    assert FIT_FAILED_KEY not in p, p
    assert df_range[0] < p["df"] < df_range[1], p
    assert abs(p["df"] - 4.0) > 0.05


def test_student_df_lower_bound_is_admissible():
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
    lo, hi, init = EXTRA_PARAM_BOUNDS_BY_PARAM["df"]
    CopulaStudent(tau_k=0.3, df=lo)          # must not raise
    assert lo > 2.0


def test_ice_fit_failure_is_flagged_and_logged(monkeypatch, caplog):
    """A family whose likelihood is nowhere finite: the returned params carry
    FIT_FAILED_KEY, a WARNING names the family, and the key never reaches a
    constructor or a stored block."""
    monkeypatch.setattr(CopulaStudent, "logpdf_array",
                        lambda self, uv: np.full(len(uv), np.nan))
    entry, cls = _resolve_candidate("Student")
    uv = CopulaGaussian(tau_k=0.4).sample(n=200, seed=1)
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.ice"):
        p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(200))
    assert p[FIT_FAILED_KEY] is True
    assert any("CopulaStudent" in r.message and r.levelno == logging.WARNING
               for r in caplog.records)
    CopulaStudent(**_copula_kwargs(p))       # the parameters are still constructible
    # The selection scores accept the flagged dict.
    assert _weighted_log_likelihood(cls, p, uv[:, 0], uv[:, 1], np.ones(200)) == -math.inf
    blk = _select_and_fit_copula(["Student"], uv[:, 0], uv[:, 1], np.ones(200))
    assert blk.get(FIT_FAILED_KEY) is True


def test_one_parameter_fit_failure_is_flagged(monkeypatch, caplog):
    monkeypatch.setattr(CopulaClayton, "logpdf_array",
                        lambda self, uv: np.full(len(uv), -np.inf))
    entry, cls = _resolve_candidate("Clayton")
    uv = CopulaGaussian(tau_k=0.4).sample(n=100, seed=2)
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.ice"):
        p = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], np.ones(100))
    assert p[FIT_FAILED_KEY] is True
    assert any("Kendall" in r.message for r in caplog.records)


# --------------------------------------------------------------------------
# RB-7 — zero weights, non-finite log-densities
# --------------------------------------------------------------------------

def test_weighted_log_density_sum_conventions():
    ld = np.array([-np.inf, np.nan, np.inf, -0.5, 1.0])
    assert _weighted_log_density_sum(ld, [0, 0, 0, 2, 1]) == pytest.approx(0.0)
    assert _weighted_log_density_sum(ld, [1e-300, 0, 0, 2, 1]) == -math.inf
    assert _weighted_log_density_sum(ld, [0, 1, 0, 2, 1]) == -math.inf   # NaN ⇒ −∞, never NaN
    assert _weighted_log_density_sum(ld, [0, 0, 1, 2, 1]) == -math.inf   # +∞ is a kernel failure
    assert _weighted_log_density_sum(np.array([-1.0, -2.0])) == pytest.approx(-3.0)
    assert _weighted_log_density_sum(np.array([-1.0, np.nan])) == -math.inf
    with pytest.raises(ValueError):
        _weighted_log_density_sum(np.array([-1.0, -2.0]), [1.0, -1.0])


def _broken_corner_logpdf(self, uv):
    out = CopulaGaussian.logpdf_array(self, uv)
    out[0], out[1] = -np.inf, np.nan
    return out


# Gaussian copula whose log-density is −∞ / NaN at the first two points. The
# class keeps the registered name: CopulaVirt.__init__ looks the family up by it.
_GaussWithBrokenCorner = type("CopulaGaussian", (CopulaGaussian,),
                              {"logpdf_array": _broken_corner_logpdf})


def test_weighted_mle_tau_ignores_broken_points_of_zero_weight():
    """0·(−∞) and 0·NaN used to make every objective value NaN, so Brent's
    comparisons were all False and τ̂ was arbitrary."""
    entry = CopulaEnum.GAUSSIAN
    uv = CopulaGaussian(tau_k=0.4).sample(n=400, seed=3)
    w = np.ones(400)
    w[:2] = 0.0
    tau_broken = _weighted_mle_tau(_GaussWithBrokenCorner, entry, uv[:, 0], uv[:, 1], w)
    tau_clean = _weighted_mle_tau(CopulaGaussian, entry, uv[2:, 0], uv[2:, 1], w[2:])
    assert tau_broken == pytest.approx(tau_clean, abs=2e-6)


def test_base_mle_objective_maps_nan_to_the_penalty(monkeypatch, caplog):
    monkeypatch.setattr(CopulaClayton, "logpdf_array",
                        lambda self, uv: np.full(len(uv), np.nan))
    data = CopulaGaussian(tau_k=0.4).sample(n=60, seed=4)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._base"):
        r = CopulaClayton.fit(data, method="mle")
    assert r.converged is False
    assert r.log_likelihood == -math.inf          # formerly NaN, which breaks sorting on AIC
    assert any("not finite" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------
# RB-6 — fit(method='tau') on comonotone / countermonotone data
# --------------------------------------------------------------------------

# Joe and SJoe are left out: τ̂ = −1 clips to their independence end ε, where
# the family kernel has no θ yet (RB-2, covered by test_copula_limits.py).
_ONE_PARAM = [e for e in CopulaEnum
              if e.value.AVAILABLE and e.value.PARAMETERS_SET_NAME == ["tau_k"]
              and e.value.TAU_MIN_MAX[1] - e.value.TAU_MIN_MAX[0] > 0
              and e.value.SHORT_NAME not in ("Joe", "SJoe")]


@pytest.mark.parametrize("entry", _ONE_PARAM, ids=lambda e: e.value.SHORT_NAME)
@pytest.mark.parametrize("sign", [1, -1], ids=["comonotone", "countermonotone"])
def test_tau_fit_on_monotone_data_is_constructible(entry, sign):
    """τ̂ = ±1 is clipped into the constructible range, never onto |τ| = 1:
    formerly LinAlgError (Gaussian), ZeroDivisionError at the registered
    bound, θ ≈ 1e16 with LL = −∞ (Clayton)."""
    x = np.linspace(0.0, 1.0, 200)
    data = np.column_stack((x, sign * x))
    r = entry.klass.fit(data, method="tau")
    lo, hi = entry.constructible_tau_range()
    assert lo <= r.tau_k <= hi
    assert abs(r.tau_k) < 1.0
    assert not math.isnan(r.log_likelihood)
    if sign == 1 and entry.value.TAU_MIN_MAX[1] == 1.0:
        assert r.tau_k == pytest.approx(1.0 - 1e-4 * (1.0 - entry.value.TAU_MIN_MAX[0]))


@pytest.mark.parametrize("short", ["Gauss", "GH", "SGH", "A12", "A14", "Frank", "Plackett"])
def test_tau_fit_on_comonotone_data_has_a_finite_likelihood(short):
    """Families whose log-density kernel is exact at τ = 1 − 1e-4 (Clayton,
    Joe and BB1 kernels are being reworked separately, FR-1)."""
    x = np.linspace(0.0, 1.0, 200)
    r = CopulaEnum.from_short_name(short).klass.fit(np.column_stack((x, x)), method="tau")
    assert math.isfinite(r.log_likelihood), r


# --------------------------------------------------------------------------
# RB-10 — fit_best
# --------------------------------------------------------------------------

class _Failing(CopulaVirt):
    @classmethod
    def fit(cls, data, method="tau"):
        raise RuntimeError("deliberate failure")


def test_fit_best_ranks_like_with_like_and_reports_failures(caplog):
    """Every fit is ranked with the method asked for; failures are reported.

    Since FR-12 Student answers ``'tau'`` by itau (ν by MLE at the τ̂ of
    Kendall) and is ranked with the other τ-inversions; before, it fitted by
    joint MLE and went to ``other_method``.
    """
    data = CopulaStudent(tau_k=0.5, df=3.0).sample(n=120, seed=5)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._base"):
        res = CopulaVirt.fit_best(data, families=[CopulaGaussian, CopulaStudent,
                                                  CopulaClayton, _Failing], method="tau")
    assert isinstance(res, FitBestResults) and isinstance(res, list)
    assert [r.method for r in res] == ["tau", "tau", "tau"]
    assert {type(r.copula).__name__ for r in res} == {"CopulaGaussian", "CopulaStudent",
                                                      "CopulaClayton"}
    assert [r.aic for r in res] == sorted(r.aic for r in res)
    assert res.other_method == [] and res.criterion == "aic"
    assert res.failures == [("_Failing", "RuntimeError: deliberate failure")]
    assert any("_Failing" in rec.message for rec in caplog.records)
    assert not any("other_method" in rec.message for rec in caplog.records)

    res_mle = CopulaVirt.fit_best(data, families=[CopulaGaussian, CopulaStudent], method="mle")
    assert {type(r.copula).__name__ for r in res_mle} == {"CopulaGaussian", "CopulaStudent"}
    assert res_mle.other_method == [] and res_mle.failures == []
