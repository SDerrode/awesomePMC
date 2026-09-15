"""Standard errors of copula estimates outside ICE (AUDIT_COPULES FR-4) — fast checks.

What is pinned here, against references that share no code with
:mod:`pmcprg.copulas._stderr`:

* the finite-difference score, Hessian and mixed u/v derivatives against the
  closed-form derivatives of the Gaussian copula log-density;
* the Gaussian closed forms: n·Var(ρ̂) → (1 − ρ²)² for the rank-based
  pseudo-likelihood estimator (Klaassen & Wellner 1997, Bernoulli 3(1),
  55–77, doi:10.2307/3318652; the pseudo-MLE is asymptotically equivalent to the normal-scores
  correlation, Genest, Ghoudi & Rivest 1995), and (1 − ρ²)²/(1 + ρ²) for the
  MLE of ρ with known standard normal margins (inverse Fisher information);
* the τ route at independence: Kendall's exact null variance
  2(2n + 1)/(9n(n − 1)) (Kendall 1938);
* frequency weights: integer weights = repeated rows, a common factor on the
  weights = the same factor on n_eff, zero weights = dropped rows;
* the boundary flag and the independence LR test (Self & Liang 1987);
* a small Monte-Carlo coverage check (the long ones are in
  ``test_fr4_standard_errors_mc.py``, marked slow);
* the opt-in contract: ``fit`` and ``FitResult`` are untouched.

Tolerances are set from measured spreads, stated at each test.
"""
from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import pytest
from scipy.stats import chi2, norm, rankdata

import pmcprg.copulas._stderr as S
from pmcprg.copulas import (
    CopulaA12,
    CopulaBB1,
    CopulaClayton,
    CopulaFrank,
    CopulaGaussian,
    CopulaPlackett,
    CopulaProduct,
    CopulaStudent,
    FitResult,
    StandardErrors,
    independence_lr_test,
    standard_errors,
)
from pmcprg.numerics import EPS

Z95 = float(norm.ppf(0.975))


def _pseudo(data: np.ndarray) -> np.ndarray:
    n = data.shape[0]
    return np.column_stack([rankdata(data[:, 0]) / (n + 1), rankdata(data[:, 1]) / (n + 1)])


def _tau_of_rho(rho: float) -> float:
    return float(2.0 / math.pi * math.asin(rho))


# ---------------------------------------------------------------------------
# Finite differences against the closed-form Gaussian derivatives
# ---------------------------------------------------------------------------

def test_fd_derivatives_match_gaussian_closed_form():
    """Score, Hessian, ∂φ/∂u and ∂φ/∂v in ψ = logit((τ + 1)/2), analytically.

    log c = −½ log(1 − ρ²) − (ρ²(x² + y²) − 2ρxy) / (2(1 − ρ²)), x = Φ⁻¹(u):
    ∂ρ log c = ρ/(1 − ρ²) + (xy(1 + ρ²) − ρ(x² + y²))/(1 − ρ²)², ρ = sin(πτ/2).
    The central differences have O(10⁻⁸) truncation and rounding errors, so
    10⁻⁵ relative leaves a wide margin.
    """
    rho = 0.5
    cop = CopulaGaussian(tau_k=_tau_of_rho(rho))
    uv = np.random.default_rng(0).uniform(0.002, 0.998, size=(300, 2))
    spec = S._spec_of(cop)
    score, hess, dphi_u, dphi_v = S._derivatives(spec, uv, ranks=True)

    tau = cop.params["tau_k"]
    rho = cop.theta
    x, y = norm.ppf(uv[:, 0]), norm.ppf(uv[:, 1])
    one = 1.0 - rho * rho
    s_rho = rho / one + (x * y * (1 + rho * rho) - rho * (x * x + y * y)) / one ** 2
    h_rho = ((1 + rho * rho) / one ** 2
             + (2 * rho * x * y - (x * x + y * y)) / one ** 2
             + (x * y * (1 + rho * rho) - rho * (x * x + y * y)) * 4 * rho / one ** 3)
    ds_dx = (y * (1 + rho * rho) - 2 * rho * x) / one ** 2
    ds_dy = (x * (1 + rho * rho) - 2 * rho * y) / one ** 2
    t1 = 0.5 * (1 - tau * tau)                       # dτ/dψ
    t2 = -tau * t1                                   # d²τ/dψ²
    c, s = math.cos(0.5 * math.pi * tau), math.sin(0.5 * math.pi * tau)
    r1 = 0.5 * math.pi * c * t1
    r2 = -(0.5 * math.pi) ** 2 * s * t1 ** 2 + 0.5 * math.pi * c * t2

    np.testing.assert_allclose(score[:, 0], s_rho * r1, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(hess[:, 0, 0], h_rho * r1 ** 2 + s_rho * r2, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dphi_u[:, 0], ds_dx / norm.pdf(x) * r1, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(dphi_v[:, 0], ds_dy / norm.pdf(y) * r1, rtol=1e-5, atol=1e-6)


def test_bb1_and_student_jacobians_match_finite_differences():
    """The analytic delta-method Jacobians agree with differences of the build map."""
    for cop, read in [
        (CopulaBB1(tau_k=0.45, delta=1.7),
         lambda c: [c.params["tau_k"], c.delta, c.theta]),
        (CopulaStudent(tau_k=-0.3, df=6.5),
         lambda c: [c.params["tau_k"], c.df, c.theta]),
    ]:
        spec = S._spec_of(cop)
        h = 1e-6
        for a in range(2):
            e = np.zeros(2)
            e[a] = h
            fd = (np.array(read(spec.build(spec.psi + e)))
                  - np.array(read(spec.build(spec.psi - e)))) / (2 * h)
            np.testing.assert_allclose(spec.jac_psi[:, a], fd, rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rho, n, rel", [(0.8, 20_000, 0.06), (0.5, 100_000, 0.025)])
def test_gaussian_pseudo_likelihood_variance_closed_forms(rho, n, rel):
    """n·Var(ρ̂): (1 − ρ²)² with ranks, (1 − ρ²)²/(1 + ρ²) with known margins.

    Ratio of estimate to closed form, measured over 20 seeds: SD 0.016 (ranks)
    and 0.019 (known margins) at ρ = 0.8, n = 20 000; SD 0.006 and 0.007 at
    ρ = 0.5, n = 100 000; means within 0.6 % of 1. The tolerances are about
    3.5 SD. What they exclude: dropping the rank corrections W₁, W₂ (ratio
    0.61 at ρ = 0.8, 0.80 at ρ = 0.5) or flipping the sign of one of them
    (Cov{W₁(U), W₂(V)} = ρ⁴/(2(1 − ρ²)²) ≠ 0: ratio 0.70 and 0.92).
    """
    data = CopulaGaussian(tau_k=_tau_of_rho(rho)).sample(n, seed=0)
    fit = CopulaGaussian.fit(data, method="mle")
    rho = fit.copula.theta
    with_ranks = fit.standard_errors()
    known = fit.standard_errors(ranks=False)
    assert n * with_ranks.se["theta"] ** 2 == pytest.approx((1 - rho ** 2) ** 2, rel=rel)
    assert n * known.se["theta"] ** 2 == pytest.approx((1 - rho ** 2) ** 2 / (1 + rho ** 2),
                                                        rel=rel)


def test_tau_route_matches_kendall_null_variance_at_independence():
    """Mean of the estimated Var(τ̂) over 200 independent samples, n = 150.

    Kendall's exact variance under independence is 2(2n + 1)/(9n(n − 1)).
    Measured over five blocks of 200 samples: ratios 0.991–1.001; ±3 %.
    """
    n = 150
    est = []
    for r in range(200):
        x = np.random.default_rng(r).uniform(size=(n, 2))
        est.append(CopulaGaussian.fit(x, method="tau").standard_errors().se["tau_k"] ** 2)
    assert np.mean(est) == pytest.approx(2 * (2 * n + 1) / (9 * n * (n - 1)), rel=0.03)


def test_tau_route_variance_is_family_independent_and_delta_mapped():
    """Var(τ̂) does not depend on the family; SE(θ) = |dθ/dτ|·SE(τ)."""
    data = CopulaClayton(tau_k=0.35).sample(400, seed=3)
    se_c = CopulaClayton.fit(data, method="tau").standard_errors()
    se_g = CopulaGaussian.fit(data, method="tau").standard_errors()
    assert se_c.se["tau_k"] == pytest.approx(se_g.se["tau_k"], rel=1e-12)
    tau = se_c.estimate["tau_k"]
    assert se_c.se["theta"] == pytest.approx(2.0 / (1.0 - tau) ** 2 * se_c.se["tau_k"], rel=1e-6)
    tau = se_g.estimate["tau_k"]
    assert se_g.se["theta"] == pytest.approx(
        0.5 * math.pi * math.cos(0.5 * math.pi * tau) * se_g.se["tau_k"], rel=1e-6)


def test_plackett_uses_log_theta_and_agrees_with_frank_scale():
    """Plackett (θ set directly) gives a τ-SE of the same size as Frank's on the same data.

    Both are symmetric, tail-independent families; on Frank data at τ = 0.3,
    n = 1000, their pseudo-likelihood SEs of τ differ by a few percent.
    """
    data = CopulaFrank(tau_k=0.3).sample(1000, seed=5)
    se_p = CopulaPlackett.fit(data, method="mle").standard_errors()
    se_f = CopulaFrank.fit(data, method="mle").standard_errors()
    assert np.isfinite(se_p.se["theta"]) and se_p.se["theta"] > 0
    assert se_p.se["tau_k"] == pytest.approx(se_f.se["tau_k"], rel=0.15)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_WEIGHT_CASES = [
    (CopulaClayton(tau_k=0.4), "mle", True),
    (CopulaClayton(tau_k=0.4), "mle", False),
    (CopulaFrank(tau_k=-0.3), "tau", True),
    (CopulaBB1(tau_k=0.5, delta=1.4), "mle", True),
    (CopulaStudent(tau_k=0.5, df=6.0), "mle", True),
]


@pytest.mark.parametrize("cop, method, ranks", _WEIGHT_CASES,
                         ids=[f"{c.__class__.__name__}-{m}-{r}" for c, m, r in _WEIGHT_CASES])
def test_integer_weights_equal_repeated_rows(cop, method, ranks):
    uv = _pseudo(cop.sample(150, seed=11))
    w = np.random.default_rng(12).integers(0, 4, size=uv.shape[0])
    weighted = standard_errors(cop, uv, w, method, ranks=ranks)
    repeated = standard_errors(cop, np.repeat(uv, w, axis=0), None, method, ranks=ranks)
    assert weighted.n_eff == repeated.n_eff == float(w.sum())
    assert weighted.n_obs == int(np.count_nonzero(w))
    for name in weighted.names:
        assert weighted.se[name] == pytest.approx(repeated.se[name], rel=1e-9)
    np.testing.assert_allclose(weighted.cov, repeated.cov, rtol=1e-9, atol=1e-15)


def test_common_weight_factor_only_scales_n_eff():
    cop = CopulaGaussian(tau_k=0.3)
    uv = _pseudo(cop.sample(200, seed=2))
    w = np.random.default_rng(3).uniform(0.2, 1.0, size=200)
    for method in ("mle", "tau"):
        a = cop.standard_errors(uv, w, method)
        b = cop.standard_errors(uv, 3.0 * w, method)
        assert b.n_eff == pytest.approx(3.0 * a.n_eff, rel=1e-12)
        assert b.se["tau_k"] == pytest.approx(a.se["tau_k"] / math.sqrt(3.0), rel=1e-9)


def test_zero_weights_drop_rows():
    cop = CopulaClayton(tau_k=0.3)
    uv = _pseudo(cop.sample(120, seed=4))
    w = np.ones(120)
    w[::3] = 0.0
    a = cop.standard_errors(uv, w)
    b = cop.standard_errors(uv[w > 0])
    assert a.se["tau_k"] == pytest.approx(b.se["tau_k"], rel=1e-10)


# ---------------------------------------------------------------------------
# API and the opt-in contract
# ---------------------------------------------------------------------------

def test_fit_result_method_matches_copula_method_and_free_function():
    data = CopulaFrank(tau_k=0.4).sample(300, seed=6)
    for method in ("tau", "mle"):
        fit = CopulaFrank.fit(data, method=method)
        a = fit.standard_errors()
        b = fit.copula.standard_errors(fit.uv, method=method)
        c = standard_errors(fit.copula, fit.uv, None, method)
        assert isinstance(a, StandardErrors) and a.method == method
        assert a.se == b.se == c.se
        assert a.names == ("tau_k", "theta")
        assert a.n_obs == 300 and a.n_eff == 300.0
    # Student and BB1 always fit by MLE; the result follows fit.method.
    fit = CopulaStudent.fit(CopulaStudent(tau_k=0.4, df=5.0).sample(400, seed=7), method="mle")
    s = fit.standard_errors()
    assert s.method == "mle" and s.names == ("tau_k", "df", "rho")
    assert s.cov.shape == (3, 3)
    assert np.linalg.matrix_rank(s.cov, tol=1e-12 * np.abs(s.cov).max()) == 2
    np.testing.assert_allclose(s.cov_of("df", "tau_k"), s.cov[np.ix_([1, 0], [1, 0])])


def test_fit_and_fit_result_are_untouched():
    """Opt-in: no new FitResult field, and computing SEs changes nothing on the fit."""
    assert [f.name for f in dataclasses.fields(FitResult)] == [
        "copula", "method", "tau_k", "log_likelihood", "n_obs", "uv", "converged"]
    data = CopulaClayton(tau_k=0.3).sample(200, seed=8)
    fit = CopulaClayton.fit(data, method="mle")
    before = (repr(fit), fit.uv.copy(), dict(fit.copula.params), fit.copula.theta)
    fit.standard_errors()
    fit.copula.standard_errors(fit.uv, method="tau")
    assert repr(fit) == before[0]
    np.testing.assert_array_equal(fit.uv, before[1])
    assert fit.copula.params == before[2] and fit.copula.theta == before[3]
    again = CopulaClayton.fit(data, method="mle")
    assert repr(again) == before[0]


def test_interior_ci_is_the_wald_interval():
    data = CopulaGaussian(tau_k=0.4).sample(500, seed=9)
    s = CopulaGaussian.fit(data, method="mle").standard_errors()
    assert not s.at_boundary and s.boundary == ()
    lo, hi = s.ci()
    assert lo == pytest.approx(s.estimate["tau_k"] - Z95 * s.se["tau_k"], rel=1e-12)
    assert hi == pytest.approx(s.estimate["tau_k"] + Z95 * s.se["tau_k"], rel=1e-12)
    lo90, hi90 = s.ci(0.90, "theta")
    assert hi90 - lo90 == pytest.approx(2 * norm.ppf(0.95) * s.se["theta"], rel=1e-12)
    with pytest.raises(ValueError):
        s.ci(1.5)
    with pytest.raises(KeyError):
        s.ci(0.95, "delta")


@pytest.mark.parametrize("call, exc", [
    (lambda uv: CopulaProduct(tau_k=0.0).standard_errors(uv), ValueError),
    (lambda uv: CopulaStudent(tau_k=0.3, df=5.0).standard_errors(uv, method="tau"), ValueError),
    (lambda uv: CopulaClayton(tau_k=0.3).standard_errors(uv, method="ifm"), ValueError),
    (lambda uv: CopulaClayton(tau_k=0.3).standard_errors(np.vstack([uv, [[0.0, 0.5]]])), ValueError),
    (lambda uv: CopulaClayton(tau_k=0.3).standard_errors(uv, -np.ones(len(uv))), ValueError),
    (lambda uv: CopulaClayton(tau_k=0.3).standard_errors(uv, np.ones(3)), ValueError),
    (lambda uv: CopulaClayton(tau_k=0.3).standard_errors(uv[:3]), ValueError),
], ids=["product", "student-tau", "method", "uv-range", "neg-weights", "weights-len", "too-few"])
def test_invalid_requests_raise(call, exc):
    uv = _pseudo(CopulaClayton(tau_k=0.3).sample(50, seed=1))
    with pytest.raises(exc):
        call(uv)


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------

def _negatively_dependent(n=400, seed=3):
    return CopulaFrank(tau_k=-0.3).sample(n, seed=seed)


def test_clayton_mle_at_the_independence_end_is_flagged():
    fit = CopulaClayton.fit(_negatively_dependent(), method="mle")
    s = fit.standard_errors()
    assert s.at_boundary
    assert "½χ²₀ + ½χ²₁" in s.boundary[0]
    assert all(math.isnan(v) for v in s.se.values())
    assert all(math.isnan(v) for v in s.ci())


def test_clayton_tau_clipped_estimate_is_flagged_but_keeps_kendall_se():
    data = _negatively_dependent()
    s = CopulaClayton.fit(data, method="tau").standard_errors()
    assert s.estimate["tau_k"] == EPS and s.at_boundary
    assert all(math.isnan(v) for v in s.ci())
    # The SE is that of the unclipped Kendall's τ̂ — the same for every family.
    ref = CopulaGaussian.fit(data, method="tau").standard_errors()
    assert s.se["tau_k"] == pytest.approx(ref.se["tau_k"], rel=1e-12)
    assert s.se["theta"] == pytest.approx(2.0 * s.se["tau_k"], rel=1e-5)   # dθ/dτ = 2 at τ = 0


def test_bb1_delta_one_is_flagged():
    uv = _pseudo(CopulaClayton(tau_k=0.4).sample(300, seed=0))
    s = CopulaBB1(tau_k=0.4, delta=1.0).standard_errors(uv)
    assert s.at_boundary and any("Clayton limit" in b for b in s.boundary)
    assert all(math.isnan(v) for v in s.se.values())
    # On Clayton data the fitted δ̂ sits on δ = 1 for some samples and not for
    # others (seeds 0–5 give both); the flag follows δ̂ − 1 ≤ 10⁻⁴ exactly.
    seen = set()
    for seed in range(6):
        fit = CopulaBB1.fit(CopulaClayton(tau_k=0.4).sample(500, seed=seed))
        s = fit.standard_errors()
        on_bound = fit.copula.delta - 1.0 <= 1e-4
        assert s.at_boundary == on_bound
        assert np.isfinite(s.se["delta"]) == (not on_bound)
        seen.add(on_bound)
    assert seen == {True, False}


def test_student_df_at_its_upper_bound_is_flagged():
    uv = _pseudo(CopulaGaussian(tau_k=0.3).sample(300, seed=1))
    s = CopulaStudent(tau_k=0.3, df=100.0).standard_errors(uv)
    assert s.at_boundary and "Gaussian limit" in s.boundary[0]
    inner = CopulaStudent(tau_k=0.3, df=6.0).standard_errors(
        _pseudo(CopulaStudent(tau_k=0.3, df=6.0).sample(300, seed=1)))
    assert not inner.at_boundary


@pytest.mark.parametrize("sign", [1, -1])
def test_frank_fit_on_its_reachable_bound_is_flagged_without_crossing_it(sign, caplog):
    """G2: data more dependent than Frank reaches (θ ≤ 700) put τ̂ on ±0.994299.
    The central steps of dθ/dτ and of the ψ-sandwich crossed the bound, where θ
    is capped: the clamping WARNING at every evaluation, a derivative of half
    its value (one side flat) and a sandwich at a constrained stop. The
    estimate is flagged, the 'mle' SEs are NaN and dθ/dτ is one-sided inside."""
    from pmcprg.copulas.archimedean.frank import (
        _FRANK_TAU_MAX,
        find_theta_frank,
        kendall_tau_frank,
    )

    data = CopulaGaussian(tau_k=sign * 0.998).sample(400, seed=11)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        fit = CopulaFrank.fit(data, method="mle")
        s_mle = fit.standard_errors()
        s_tau = standard_errors(fit.copula, fit.uv, method="tau")
    assert [r.getMessage() for r in caplog.records] == []
    assert fit.tau_k == sign * _FRANK_TAU_MAX
    for s in (s_mle, s_tau):
        assert s.at_boundary and "end of the τ CopulaFrank reaches" in s.boundary[-1]
        assert all(math.isnan(v) for v in s.ci())
    assert all(math.isnan(v) for v in s_mle.se.values())
    # dθ/dτ at θ = 700 from Frank's own τ(θ) map, inside: 1 / τ'(θ).
    h = 1e-3
    dtheta_dtau = 2.0 * h / (kendall_tau_frank(700.0) - kendall_tau_frank(700.0 - 2.0 * h))
    assert s_tau.se["theta"] / s_tau.se["tau_k"] == pytest.approx(dtheta_dtau, rel=1e-3)
    assert abs(find_theta_frank(sign * _FRANK_TAU_MAX)) == 700.0

    # Just inside, where no step crosses: not flagged, as before.
    inner = standard_errors(CopulaFrank(tau_k=sign * 0.99429), fit.uv, method="tau")
    assert not inner.at_boundary


def test_plackett_estimate_on_a_table_end_is_flagged(caplog):
    """G2: τ̂ on a table end is where the fit stopped at θ = 10^{±6}."""
    lo, hi = CopulaPlackett.reachable_tau_bounds()
    uv = _pseudo(CopulaGaussian(tau_k=0.998).sample(400, seed=11))
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        fit = CopulaPlackett.fit(uv, method="mle")
        s = fit.standard_errors()
    assert [r.getMessage() for r in caplog.records] == []
    assert fit.tau_k == hi
    assert s.at_boundary and "end of the τ CopulaPlackett reaches" in s.boundary[-1]
    assert all(math.isnan(v) for v in s.se.values())
    s_lo = standard_errors(CopulaPlackett(tau_k=lo), uv, method="tau")
    assert s_lo.at_boundary and np.isfinite(s_lo.se["tau_k"])
    assert not standard_errors(CopulaPlackett(tau_k=0.99352), uv, method="tau").at_boundary


# ---------------------------------------------------------------------------
# Independence LR test
# ---------------------------------------------------------------------------

def test_lr_test_on_a_boundary_family_uses_the_mixture():
    data = CopulaClayton(tau_k=0.3).sample(300, seed=10)
    fit = CopulaClayton.fit(data, method="mle")
    t = independence_lr_test(CopulaClayton, fit.uv)
    assert t.boundary and t.null_distribution == "0.5*chi2(0) + 0.5*chi2(1)"
    # Same optimiser and objective as fit(method='mle').
    assert t.statistic == pytest.approx(2.0 * fit.log_likelihood, rel=1e-6)
    assert t.p_value == pytest.approx(0.5 * chi2.sf(t.statistic, 1), rel=1e-12)
    assert t.p_value < 1e-10
    # τ̂ on the independence end: LR = 0 and p = 1.
    t0 = independence_lr_test(CopulaClayton(tau_k=0.2), _pseudo(_negatively_dependent()))
    assert t0.statistic == 0.0 and t0.p_value == 1.0


def test_lr_test_on_an_interior_family_uses_chi2_1():
    uv = _pseudo(_negatively_dependent())
    t = independence_lr_test(CopulaFrank, uv)
    assert not t.boundary and t.null_distribution == "chi2(1)"
    assert t.tau_k < 0
    assert t.p_value == pytest.approx(chi2.sf(t.statistic, 1), rel=1e-12)
    w = np.random.default_rng(0).integers(0, 3, size=uv.shape[0])
    a = independence_lr_test(CopulaFrank, uv, w)
    b = independence_lr_test(CopulaFrank, np.repeat(uv, w, axis=0))
    assert a.statistic == pytest.approx(b.statistic, rel=1e-6)
    assert a.n_eff == b.n_eff


@pytest.mark.parametrize("family", [CopulaA12, CopulaStudent, CopulaProduct])
def test_lr_test_refuses_families_without_a_one_parameter_independence(family):
    with pytest.raises(ValueError):
        independence_lr_test(family, _pseudo(_negatively_dependent(100)))


# ---------------------------------------------------------------------------
# Fast Monte-Carlo coverage (the long ones: test_fr4_standard_errors_mc.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, method", [(CopulaGaussian, "mle"), (CopulaClayton, "tau")])
def test_fast_monte_carlo_sd_ratio_and_coverage(cls, method):
    """R = 300 replicates, n = 300.

    The empirical SD of τ̂ over R replicates has a relative standard error of
    1/√(2(R − 1)) = 4.1 %; ±12 % is 3 SE, widened to ±17 % for the O(1/n)
    finite-sample gap of an asymptotic formula (measured: ≤ 4 % at n = 400).
    Coverage of the 95 % Wald interval: binomial SE √(0.95·0.05/R) = 1.3 %,
    band ±3 SE.
    """
    R, n, tau = 300, 300, 0.4
    est, se, covered = [], [], 0
    for r in range(R):
        fit = cls.fit(cls(tau_k=tau).sample(n, seed=50_000 + r), method=method)
        s = fit.standard_errors()
        est.append(s.estimate["tau_k"])
        se.append(s.se["tau_k"])
        lo, hi = s.ci()
        covered += lo <= tau <= hi
    ratio = math.sqrt(np.mean(np.square(se))) / np.std(est, ddof=1)
    assert 0.83 <= ratio <= 1.17, ratio
    assert 0.912 <= covered / R <= 0.988, covered / R
