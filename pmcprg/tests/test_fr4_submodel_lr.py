"""Likelihood-ratio tests of a two-parameter family's one-parameter sub-model
(AUDIT_COPULES FR-4, "tests RV des sous-modeles des familles a deux
parametres").

What is checked here
---------------------
* the nesting each implemented pair claims, numerically, independently of
  the family docstrings ``pmcprg.copulas._stderr`` cites: BB1 at delta = 1
  (resp. theta -> 0 with delta = 1/(1 - tau)) reproduces Clayton's (resp.
  Gumbel-Hougaard's) log-density; Tawn type 1/2 at psi = 1 reproduces
  Gumbel-Hougaard's log-density; Student at a large df approaches Gaussian's
  log-density, with the gap shrinking as df grows (the nu -> infinity limit
  has no exact finite-df equality, unlike the other two nestings);
* ``submodel_lr_test`` refuses a pair it does not implement;
* the statistic is always >= 0 and the p-value is the stated
  half-chi-square-mixture (Self & Liang 1987) function of it;
* a Monte-Carlo *level* study under each sub-model's own null (empirical
  rejection near nominal alpha at 5% and 10%, hundreds of replicates);
* a Monte-Carlo *power* study away from the sub-model (rejection rate much
  higher than alpha).

Student -> Gauss is handled like the other two: nu = 100 (the fitting box's
upper end, ``EXTRA_PARAM_BOUNDS_BY_PARAM['df']``) stands in for nu ->
infinity exactly as it already does for ``standard_errors``'s boundary flag
(``pmcprg.copulas._stderr`` module docstring) — this is not a materially
different treatment, just the same finite-box convention reused, so all
three nestings get the identical boundary mixture null and the identical
Monte-Carlo protocol below.

Monte-Carlo seeds are integer literals or ``zlib.crc32`` of a string — never
Python's salted ``hash()`` (an intermittent failure fixed in 5e56fda).
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.stats import chi2, rankdata

from pmcprg.copulas import (
    CopulaBB1,
    CopulaClayton,
    CopulaGaussian,
    CopulaGH,
    CopulaStudent,
    CopulaTawn1,
    CopulaTawn2,
    SubmodelLRTest,
    submodel_lr_test,
)


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _pseudo(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    return np.column_stack([rankdata(data[:, 0]) / (n + 1), rankdata(data[:, 1]) / (n + 1)])


# ---------------------------------------------------------------------------
# The nesting itself, verified numerically (not transcribed from a docstring
# or a commit message)
# ---------------------------------------------------------------------------

def test_bb1_at_delta_1_is_clayton():
    tau = 0.4
    uv = np.random.default_rng(_seed("bb1-clayton-grid")).uniform(0.01, 0.99, size=(50, 2))
    bb1 = CopulaBB1(tau_k=tau, delta=1.0)
    clay = CopulaClayton(tau_k=tau)
    assert np.max(np.abs(bb1.logpdf_array(uv) - clay.logpdf_array(uv))) < 1e-8


def test_bb1_at_theta_to_0_approaches_gumbel():
    """delta = 1/(1 - tau) - correction so tau is admissible at small theta.

    The gap between BB1's and Gumbel's log-density shrinks with theta —
    checked at three shrinking values, not asserted at theta = 0 exactly
    (inadmissible: BB1 requires theta > 0).
    """
    tau = 0.4
    uv = np.random.default_rng(_seed("bb1-gumbel-grid")).uniform(0.01, 0.99, size=(50, 2))
    gh = CopulaGH(tau_k=tau)
    gaps = []
    for theta in (1e-2, 1e-4, 1e-6):
        delta = 2.0 / ((1.0 - tau) * (theta + 2.0))
        bb1 = CopulaBB1(tau_k=tau, delta=delta)
        gaps.append(float(np.max(np.abs(bb1.logpdf_array(uv) - gh.logpdf_array(uv)))))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 1e-4


@pytest.mark.parametrize("tawn_cls", [CopulaTawn1, CopulaTawn2])
def test_tawn_at_psi_1_is_gumbel(tawn_cls):
    tau = 0.4
    uv = np.random.default_rng(_seed("tawn-gumbel-grid", tawn_cls.__name__)).uniform(
        0.01, 0.99, size=(50, 2))
    tawn = tawn_cls(tau_k=tau, psi=1.0)
    gh = CopulaGH(tau_k=tau)
    assert np.max(np.abs(tawn.logpdf_array(uv) - gh.logpdf_array(uv))) < 1e-10


def test_student_at_large_df_approaches_gaussian():
    tau = 0.4
    uv = np.random.default_rng(_seed("student-gaussian-grid")).uniform(0.01, 0.99, size=(50, 2))
    ga = CopulaGaussian(tau_k=tau)
    gaps = []
    for df in (30.0, 100.0, 1000.0):
        st = CopulaStudent(tau_k=tau, df=df)
        gaps.append(float(np.max(np.abs(st.logpdf_array(uv) - ga.logpdf_array(uv)))))
    assert gaps[0] > gaps[1] > gaps[2]


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------

def test_submodel_lr_test_refuses_an_unimplemented_pair():
    uv = np.random.default_rng(0).uniform(0.01, 0.99, size=(20, 2))
    with pytest.raises(ValueError, match="not an implemented sub-model"):
        submodel_lr_test(CopulaBB1, CopulaGaussian, uv)


_EXTRA_PARAM_NAME = {
    CopulaBB1: "delta",
    CopulaStudent: "df",
    CopulaTawn1: "psi",
    CopulaTawn2: "psi",
}


@pytest.mark.parametrize("full_cls,sub_cls", [
    (CopulaBB1, CopulaClayton),
    (CopulaBB1, CopulaGH),
    (CopulaStudent, CopulaGaussian),
    (CopulaTawn1, CopulaGH),
    (CopulaTawn2, CopulaGH),
])
def test_submodel_lr_test_result_shape(full_cls, sub_cls):
    tau = 0.4
    sub = sub_cls(tau_k=tau)
    data = sub.sample(400, seed=_seed("shape", full_cls.__name__, sub_cls.__name__))
    uv = _pseudo(data)
    t = submodel_lr_test(full_cls, sub_cls, uv)
    assert isinstance(t, SubmodelLRTest)
    assert t.family == full_cls.__name__
    assert t.submodel == sub_cls.__name__
    assert t.statistic >= 0.0
    assert t.null_distribution == "0.5*chi2(0) + 0.5*chi2(1)"
    assert t.boundary is True
    assert t.boundary_note
    assert set(t.full_params) == {"tau_k", _EXTRA_PARAM_NAME[full_cls]}
    assert set(t.sub_params) == {"tau_k"}
    assert t.log_likelihood_full >= t.log_likelihood_sub - 1e-6
    assert t.n_obs == 400
    assert t.n_eff == 400.0
    expected_p = 1.0 if t.statistic <= 0.0 else 0.5 * float(chi2.sf(t.statistic, 1))
    assert t.p_value == pytest.approx(expected_p)


def test_submodel_lr_test_accepts_an_instance_and_weights():
    tau = 0.4
    clay = CopulaClayton(tau_k=tau)
    data = clay.sample(300, seed=_seed("instance-weights"))
    uv = _pseudo(data)
    w = np.ones(300)
    t_inst = submodel_lr_test(CopulaBB1(tau_k=0.6, delta=2.0), clay, uv)
    t_cls = submodel_lr_test(CopulaBB1, CopulaClayton, uv)
    assert t_inst.statistic == pytest.approx(t_cls.statistic)
    t_w = submodel_lr_test(CopulaBB1, CopulaClayton, uv, weights=w)
    assert t_w.statistic == pytest.approx(t_cls.statistic)
    assert t_w.n_eff == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Monte-Carlo level and power (fast: n <= 500, hundreds of replicates, each
# call a few ms — see the report for measured timings)
# ---------------------------------------------------------------------------

_MC_CASES = {
    "bb1-clayton": dict(full=CopulaBB1, sub=CopulaClayton, tau=0.4,
                        power_kwargs=dict(delta=3.0), power_tau=0.75),
    "bb1-gumbel": dict(full=CopulaBB1, sub=CopulaGH, tau=0.4,
                       power_kwargs=dict(delta=1.2), power_tau=0.5),
    "student-gauss": dict(full=CopulaStudent, sub=CopulaGaussian, tau=0.4,
                          power_kwargs=dict(df=5.0), power_tau=0.4),
    "tawn1-gumbel": dict(full=CopulaTawn1, sub=CopulaGH, tau=0.4,
                        power_kwargs=dict(psi=0.5), power_tau=0.4),
}
_N_MC = 400
_N_OBS = 400


def _rejection_rate(full_cls, sub_cls, gen_copula, n_obs, n_rep, seed_tag, alpha):
    rej = np.zeros(len(alpha))
    for i in range(n_rep):
        data = gen_copula.sample(n_obs, seed=_seed(seed_tag, i))
        uv = _pseudo(data)
        t = submodel_lr_test(full_cls, sub_cls, uv)
        for j, a in enumerate(alpha):
            rej[j] += t.p_value <= a
    return rej / n_rep


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(_MC_CASES))
def test_submodel_lr_level_near_nominal(case):
    """Empirical rejection under H0 (data simulated from the sub-model itself)."""
    info = _MC_CASES[case]
    gen = info["sub"](tau_k=info["tau"])
    alpha = np.array([0.05, 0.10])
    rate = _rejection_rate(info["full"], info["sub"], gen, _N_OBS, _N_MC, f"level-{case}", alpha)
    # Binomial MC noise at n = 400: sd(0.05) ~= 0.011, sd(0.10) ~= 0.015 — a
    # tolerance of 0.05 absolute is ~4-5 sd, generous but not vacuous.
    assert abs(rate[0] - 0.05) < 0.05, f"{case}: 5% rejection rate {rate[0]:.3f}"
    assert abs(rate[1] - 0.10) < 0.06, f"{case}: 10% rejection rate {rate[1]:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(_MC_CASES))
def test_submodel_lr_power_away_from_submodel(case):
    """Empirical rejection with data simulated from inside the full family,
    away from the sub-model — must clear the level by a wide margin."""
    info = _MC_CASES[case]
    gen = info["full"](tau_k=info["power_tau"], **info["power_kwargs"])
    alpha = np.array([0.05])
    rate = _rejection_rate(info["full"], info["sub"], gen, _N_OBS, _N_MC, f"power-{case}", alpha)
    assert rate[0] > 0.5, f"{case}: power at 5% only {rate[0]:.3f}"
