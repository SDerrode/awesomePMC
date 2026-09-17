"""Robust options outside ICE (AUDIT_COPULES FR-7 b and c).

Two deliverables are pinned here, each against references that share no code
with the module under test:

**(b) the MLE-vs-τ discrepancy diagnostic**
(:func:`pmcprg.copulas._stderr.mle_tau_discrepancy_test`)

* the two marginal variances it builds its contrast from are *exactly* the
  ones :func:`standard_errors` reports for ``method='mle'`` and
  ``method='tau'`` — so the diagnostic cannot silently describe different
  estimators;
* the τ̂ it uses equals :func:`scipy.stats.kendalltau` to machine precision on
  distinct pseudo-observations (the U-statistic form, not the V-statistic one,
  which is off by O(1/n) — the same order as the difference being tested);
* the Hausman "difference of variances" shortcut is shown to be *negative* on
  a real sample, which is why the joint influence-function construction is
  used instead;
* level under correct specification and power under contamination, both by
  Monte Carlo (slow).

**(c) the weighted density-power-divergence estimator**
(:mod:`pmcprg.copulas._robust`)

* ``∫∫ c^{1+α}`` against three exact references: 1 for the independence
  copula (which also checks that the quadrature weights integrate to 1), the
  Gaussian closed form derived in the module docstring, and FGM's
  ``1 + θ²/9`` at α = 1;
* the α = 0 objective reproduces :meth:`CopulaVirt.fit(method='mle')`;
* the α = 0 variance reproduces the Gaussian known-margin inverse Fisher
  information ``(4/π²)(1 − ρ²)/(1 + ρ²)``;
* the variance is NaN exactly where ``∫∫ c^{1+2α}`` stops converging;
* the clean-data efficiency loss and the contamination breakdown, by Monte
  Carlo (slow).

Monte-Carlo seeds are integer literals or ``zlib.crc32`` of a string — never
``hash()``, which is salted per process.

Measured with the seeds below.

*(b) level* (R = 500, rejection at 5 %/10 %, binomial SE 0.97 % at 5 %):
at n = 2000, Gaussian 5.2-6.0 %/10.2-12.2 %, Clayton 3.8-5.2 %/9.0-10.8 %,
Frank 3.4-4.4 %/7.4-9.2 %, Gumbel 4.4-9.0 %/11.4-15.4 %, Joe 4.0-4.8 %/
8.6-10.8 % over τ ∈ {0.2, 0.4, 0.6}; ``sd(Z)`` 0.94-1.08. At n = 500 the
statistic carries an O(n^{-1/2}) mean offset (E[Z] ≈ 0.45 for the Gaussian,
0.15-0.25 elsewhere, halving from n = 500 to n = 2000) which makes the test
mildly liberal for the Gaussian (7.2-8.0 % at nominal 5 %) and conservative
for Frank (2.2-3.4 %, ``sd(Z)`` 0.87-0.93).

*(b) power* (n = 1000, R = 500, τ = 0.4, rejection at 5 % for ε = 0/1/2/5/
10 % of **discordant corner pairs**): Gaussian 6.2/98.2/100/100/100 %, Frank
5.6/83.0/100/100/0.2 %, Clayton 6.0/5.4/5.4/18.2/92.8 %. With ε drawn from a
strongly **opposite** copula (Gaussian τ = −0.8) instead: Gaussian
7.4/5.4/6.0/5.4/10.0 %, Clayton 5.0/5.6/8.4/20.8/67.8 %, Frank
5.0/8.8/19.4/76.6/100 %. Read
:func:`test_discrepancy_power_under_contamination` for what the two weak
cases mean — the diagnostic is a specification contrast, and it is blind
wherever the contamination moves both estimators the same way.

*(c) clean data* (n = 1000, R = 200, τ = 0.4): RMS ``|τ̂_α − τ̂_MLE|`` and the
SD ratio ``sd(τ̂_α)/sd(τ̂_MLE)`` — α = 0.05: 5.9e-4 to 8.9e-4, 1.000-1.003;
α = 0.1: 1.2e-3 to 1.8e-3, 1.003-1.006; α = 0.25: 2.8e-3 to 5.2e-3,
1.018-1.051; α = 0.5: 5.3e-3 to 1.5e-2, 1.066-1.365 (Gaussian, Clayton,
Frank). The estimator's own SD is 0.0147-0.0165, so α ≤ 0.1 costs under
1 % of efficiency and moves τ̂ by under a tenth of its own standard error.

*(c) contamination* (n = 1000, R = 200, τ = 0.5, bias of τ̂ at ε = 0/1/5/
10 %): MLE +0.002/−0.075/−0.288/−0.457 against DPD α = 0.5
−0.003/+0.001/−0.002/−0.047 (Gaussian), and −0.001/−0.033/−0.160/−0.349
against −0.001/+0.001/+0.003/−0.022 (Clayton). The full table, the
intermediate α and Kendall's τ̂ for comparison, are in
:func:`test_dpd_breakdown_across_contamination_fractions`.

*(c) α selection* (n = 1000, R = 100): mean α̂ 0.12-0.15 on clean data and
0.25-0.50 at 5 % contamination, with the selected τ̂ biased by at most 0.034
where the MLE is biased by 0.12-0.29
(:func:`test_alpha_selection_moves_with_the_contamination`).
"""
from __future__ import annotations

import logging
import math
import zlib

import numpy as np
import pytest
from scipy.stats import kendalltau, rankdata

import pmcprg.copulas._robust as ROB
from pmcprg.copulas import (
    CopulaBB1,
    CopulaClayton,
    CopulaFGM,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH,
    CopulaJoe,
    CopulaProduct,
    CopulaStudent,
    DPDAlphaSelection,
    DPDFit,
    MleTauDiscrepancyTest,
    dpd_fit,
    dpd_objective,
    integral_c_power,
    mle_tau_discrepancy_test,
    select_alpha,
    standard_errors,
)
from pmcprg.copulas._base import CopulaEnum


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _pseudo(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    return np.column_stack([rankdata(x[:, 0]) / (n + 1), rankdata(x[:, 1]) / (n + 1)])


def _contaminate(x: np.ndarray, eps: float, rng) -> np.ndarray:
    """Replace a fraction ``eps`` of the pairs by strongly **discordant** corner pairs.

    Half go to the (u → 1, v → 0) corner and half to (u → 0, v → 1): points a
    positively dependent copula gives a very small density, so the
    unbounded-influence pseudo-likelihood score reacts to them and Kendall's
    τ̂ only shifts by O(eps).
    """
    n = x.shape[0]
    k = int(round(eps * n))
    if k == 0:
        return x
    idx = rng.choice(n, size=k, replace=False)
    out = x.copy()
    half = k // 2
    out[idx[:half], 0] = rng.uniform(0.980, 0.9995, size=half)
    out[idx[:half], 1] = rng.uniform(0.0005, 0.020, size=half)
    out[idx[half:], 0] = rng.uniform(0.0005, 0.020, size=k - half)
    out[idx[half:], 1] = rng.uniform(0.980, 0.9995, size=k - half)
    return out


def _entry(cls):
    return next(e for e in CopulaEnum if e.value.CLASS_NAME == cls.__name__)


# ===========================================================================
# (b) MLE-vs-tau discrepancy diagnostic
# ===========================================================================

@pytest.mark.parametrize("cls", [CopulaGaussian, CopulaClayton, CopulaFrank, CopulaGH])
def test_marginal_variances_are_the_standard_errors_of_the_two_methods(cls):
    """``se_tau_mle`` and ``se_tau_tau`` come from the same influence functions
    the contrast uses, so they must equal :func:`standard_errors` exactly."""
    data = cls(tau_k=0.4).sample(1500, seed=_seed("marginal", cls.__name__))
    fit = cls.fit(data, method="mle")
    t = mle_tau_discrepancy_test(cls, fit.uv)
    assert t.tau_mle == pytest.approx(fit.tau_k, abs=1e-8)
    mle = standard_errors(fit.copula, fit.uv, method="mle")
    inv = standard_errors(fit.copula, fit.uv, method="tau")
    assert t.se_tau_mle == pytest.approx(mle.se["tau_k"], rel=1e-12)
    assert t.se_tau_tau == pytest.approx(inv.se["tau_k"], rel=1e-12)


def test_tau_estimate_equals_scipy_kendalltau():
    """The U-statistic form written through C_n, not the V-statistic one."""
    for seed in (0, 1, 2):
        uv = _pseudo(CopulaGaussian(tau_k=0.4).sample(400, seed=seed))
        t = mle_tau_discrepancy_test(CopulaGaussian, uv)
        assert t.tau_tau == pytest.approx(float(kendalltau(uv[:, 0], uv[:, 1])[0]), abs=1e-14)


def test_variance_of_the_difference_is_the_joint_one_not_a_hausman_shortcut():
    """``Var(A − B) = Var A + Var B − 2 Cov``, and the shortcut can go negative.

    The classical Hausman variance ``Var(slow) − Var(fast)`` needs the
    efficient estimator under H0; the rank-based pseudo-MLE is not
    semiparametrically efficient in general (Genest & Werker 2002), and on
    ``CopulaFrank(tau_k=0.4)``, n = 600, seed 0 the shortcut is
    **−1.36e-06** while the joint construction gives **+4.10e-06**.
    """
    uv = _pseudo(CopulaFrank(tau_k=0.4).sample(600, seed=0))
    t = mle_tau_discrepancy_test(CopulaFrank, uv)
    va, vb, vd = t.se_tau_mle ** 2, t.se_tau_tau ** 2, t.se_difference ** 2
    assert vd == pytest.approx(va + vb - 2.0 * t.correlation * math.sqrt(va * vb), rel=1e-10)
    assert 0.0 < t.correlation < 1.0
    assert abs(t.se_tau_tau - t.se_tau_mle) < t.se_difference < t.se_tau_tau + t.se_tau_mle
    assert vb - va < 0.0 < vd            # the shortcut is not even a variance here


def test_discrepancy_result_shape_and_theta_reporting():
    data = CopulaClayton(tau_k=0.45).sample(800, seed=_seed("shape"))
    t = mle_tau_discrepancy_test(CopulaClayton, _pseudo(data))
    assert isinstance(t, MleTauDiscrepancyTest)
    assert t.family == "CopulaClayton" and t.n_obs == 800 and t.n_eff == 800.0
    assert not t.at_boundary and t.boundary == ()
    assert t.difference == pytest.approx(t.tau_mle - t.tau_tau, abs=1e-15)
    assert t.statistic == pytest.approx(t.difference / t.se_difference, rel=1e-12)
    # theta is reported for information: it is the family's own map of each tau.
    assert t.theta_mle == pytest.approx(CopulaClayton(tau_k=t.tau_mle).theta, rel=1e-12)
    assert t.theta_tau == pytest.approx(CopulaClayton(tau_k=t.tau_tau).theta, rel=1e-12)
    assert 0.0 <= t.p_value <= 1.0
    assert "CopulaClayton" in repr(t)


def test_discrepancy_accepts_an_instance_and_frequency_weights():
    """An instance is accepted like a class, and the MLE side matches repeated rows.

    ``tau_tau`` deliberately does **not**: compressing repeated rows into
    frequency weights removes the within-row tied pairs the expanded sample
    would count (``_tau_influence``'s docstring), an O(Σw²/(Σw)²) difference.
    """
    uv = _pseudo(CopulaGaussian(tau_k=0.4).sample(200, seed=_seed("weights")))
    w = np.random.default_rng(3).integers(1, 4, size=uv.shape[0]).astype(float)
    a = mle_tau_discrepancy_test(CopulaGaussian(tau_k=0.1), uv, w)
    b = mle_tau_discrepancy_test(CopulaGaussian, np.repeat(uv, w.astype(int), axis=0))
    assert a.n_eff == b.n_eff == float(w.sum())
    assert a.tau_mle == pytest.approx(b.tau_mle, abs=1e-8)
    assert a.se_tau_mle == pytest.approx(b.se_tau_mle, rel=1e-6)


@pytest.mark.parametrize("family", [CopulaStudent, CopulaBB1])
def test_discrepancy_refuses_two_parameter_families(family):
    uv = _pseudo(CopulaGaussian(tau_k=0.3).sample(200, seed=_seed("refuse")))
    with pytest.raises(NotImplementedError, match="more than one parameter"):
        mle_tau_discrepancy_test(family, uv)


def test_discrepancy_refuses_a_family_without_a_free_parameter():
    uv = _pseudo(CopulaGaussian(tau_k=0.3).sample(200, seed=_seed("product")))
    with pytest.raises(ValueError, match="no free parameter"):
        mle_tau_discrepancy_test(CopulaProduct, uv)


def test_discrepancy_is_nan_at_a_boundary(caplog):
    """Clayton on *negatively* dependent data stops at its independence end.

    The pseudo-MLE is then a constrained stop, not an interior optimum, and
    is not asymptotically normal (Self & Liang 1987) — so the contrast has no
    reference distribution, exactly as :func:`standard_errors` reports no
    Wald interval there.
    """
    x = CopulaGaussian(tau_k=-0.35).sample(500, seed=_seed("boundary"))
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        t = mle_tau_discrepancy_test(CopulaClayton, _pseudo(x))
    assert t.at_boundary and t.boundary
    assert math.isnan(t.statistic) and math.isnan(t.p_value)
    assert math.isnan(t.se_difference) and math.isnan(t.correlation)
    # The tau side is still reported: its estimator is regular at the boundary.
    assert np.isfinite(t.tau_tau) and np.isfinite(t.se_tau_tau)


def test_discrepancy_fires_on_a_contaminated_sample():
    """5 % discordant corner pairs on Gaussian data at τ = 0.5, n = 1000.

    Measured on this seed: τ̂_MLE = 0.208, τ̂_τ = 0.366, Z = −11.9,
    p = 1.4e-32 — while the clean sample gives τ̂_MLE = 0.5095, τ̂_τ = 0.5092,
    Z = 0.06, p = 0.95.
    """
    s = _seed("contaminated-fires")
    x = CopulaGaussian(tau_k=0.5).sample(1000, seed=s)
    clean = mle_tau_discrepancy_test(CopulaGaussian, _pseudo(x))
    dirty = mle_tau_discrepancy_test(
        CopulaGaussian, _pseudo(_contaminate(x, 0.05, np.random.default_rng(s + 1))))
    assert clean.p_value > 0.05
    assert dirty.p_value < 1e-6
    assert dirty.tau_mle < dirty.tau_tau - 0.05


# ===========================================================================
# (c) Density power divergence — exact references for the integral
# ===========================================================================

def test_quadrature_weights_integrate_to_one_and_the_product_copula_gives_one():
    """``∫∫ 1 du dv = 1``: the weights (with the logit Jacobian) must sum to 1."""
    for design in (ROB._QUAD_DEFAULT, ROB._QUAD_REFINED):
        _, log_w = ROB._quad_grid(*design)
        assert float(np.exp(log_w).sum()) == pytest.approx(1.0, abs=1e-13)
    prod = CopulaProduct(tau_k=0.0)
    for alpha in (0.0, 0.1, 0.5, 1.0):
        assert integral_c_power(prod, alpha) == pytest.approx(1.0, abs=1e-13)


@pytest.mark.parametrize("tau, tol", [(0.2, 1e-12), (0.5, 3e-8), (0.7, 1e-5), (0.9, 2e-3)])
def test_gaussian_integral_closed_form_matches_the_quadrature(tau, tol):
    """``I(ρ, α) = (1 − ρ²)^{−α/2}(1 − α²ρ²)^{−1/2}``, derived in the module docstring.

    The closed form is the reference; ``closed_form=False`` forces the
    quadrature. Measured relative errors over α = 0.1/0.25/0.5/0.75: at most
    4.6e-14 at τ = 0.2, 7.9e-9 at τ = 0.5, 2.4e-6 at τ = 0.7 and 4.6e-4 at
    τ = 0.9 (the worst is always α = 0.75). The tolerances are about 3× those
    — tight enough that halving the node count or dropping the outermost
    panel fails them.
    """
    cop = CopulaGaussian(tau_k=tau)
    for alpha in (0.1, 0.25, 0.5, 0.75):
        exact = ROB._gaussian_integral_closed_form(cop.theta, alpha)
        assert integral_c_power(cop, alpha) == exact          # the closed form is used
        quad = integral_c_power(cop, alpha, closed_form=False)
        assert quad == pytest.approx(exact, rel=tol), (tau, alpha, quad, exact)


@pytest.mark.parametrize("tau", [0.1, 0.2, -0.15])
def test_fgm_integral_at_alpha_one_matches_its_closed_form(tau):
    """``∫∫(1 + θ(1−2u)(1−2v))² = 1 + θ²/9`` — a second exact reference, on a
    family with no tail dependence, where α = 1 is still admissible."""
    cop = CopulaFGM(tau_k=tau)
    theta = cop.theta
    assert integral_c_power(cop, 1.0) == pytest.approx(1.0 + theta * theta / 9.0, rel=1e-12)


def test_integral_rel_error_reports_the_node_density_limit():
    """The two designs agree to 1e-8 at τ = 0.4 and part company at τ = 0.85."""
    easy = dpd_fit(CopulaClayton, _pseudo(CopulaClayton(tau_k=0.4).sample(
        400, seed=_seed("relerr"))), 0.25)
    assert easy.integral_rel_error < 1e-6
    a = integral_c_power(CopulaClayton(tau_k=0.85), 0.5)
    b = integral_c_power(CopulaClayton(tau_k=0.85), 0.5, refined=True)
    assert abs(a / b - 1.0) > 1e-2


# ===========================================================================
# (c) The estimator
# ===========================================================================

@pytest.mark.parametrize("cls", [CopulaGaussian, CopulaClayton, CopulaFrank, CopulaJoe])
def test_alpha_zero_reproduces_the_pseudo_mle(cls):
    """α = 0 is the negative mean pseudo-log-likelihood: the same minimiser."""
    data = cls(tau_k=0.45).sample(800, seed=_seed("alpha0", cls.__name__))
    fit = cls.fit(data, method="mle")
    d = dpd_fit(cls, fit.uv, 0.0)
    assert d.tau_k == pytest.approx(fit.tau_k, abs=1e-7)
    assert d.integral == 1.0 and d.integral_rel_error == 0.0
    assert d.objective == pytest.approx(-fit.log_likelihood / fit.n_obs, rel=1e-10)
    assert d.converged and isinstance(d, DPDFit)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0])
def test_objective_on_the_independence_copula_is_minus_one_over_alpha(alpha):
    """``c ≡ 1`` gives ``I = 1`` and ``mean c^α = 1``, so ``H = 1 − (1 + 1/α)``."""
    uv = _pseudo(CopulaGaussian(tau_k=0.3).sample(100, seed=_seed("objective")))
    got = dpd_objective(CopulaProduct(tau_k=0.0), uv, alpha)
    assert got == pytest.approx(-1.0 / alpha, abs=1e-12)


@pytest.mark.parametrize("tau", [0.2, 0.4, 0.6])
def test_dpd_variance_at_alpha_zero_is_the_gaussian_inverse_fisher_information(tau):
    """Known margins: ``n Var(ρ̂) → (1 − ρ²)²/(1 + ρ²)`` (the inverse Fisher
    information of the Gaussian copula), so in τ coordinates, with
    ``dτ/dρ = (2/π)/√(1 − ρ²)``, ``V_0 = (4/π²)(1 − ρ²)/(1 + ρ²)``.

    That closed form tests the ``J_α`` integral and the finite-difference
    score together, against a quantity this module never computes. Measured
    relative error 3.6e-8 to 4.0e-8 — the truncation error of the central
    difference in τ, not of the quadrature.
    """
    rho = math.sin(0.5 * math.pi * tau)
    closed = 4.0 / math.pi ** 2 * (1.0 - rho * rho) / (1.0 + rho * rho)
    v, rel = ROB._dpd_variance(CopulaGaussian, _entry(CopulaGaussian), tau, 0.0)
    assert rel == 0.0
    assert v == pytest.approx(closed, rel=1e-6)


def test_variance_is_nan_exactly_where_the_c_power_integral_stops_converging(caplog):
    """``K_α`` needs ``∫∫c^{1+2α}``: at α = 0.5 that is ``∫∫c²``, divergent for
    a family with tail dependence and finite for one without.

    The estimate itself is unaffected — only its asymptotic variance.
    """
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        clay = ROB._dpd_variance(CopulaClayton, _entry(CopulaClayton), 0.4, 0.5)
    assert math.isnan(clay[0]) and clay[1] > 0.3
    assert any("does not converge" in r.getMessage() for r in caplog.records)
    frank = ROB._dpd_variance(CopulaFrank, _entry(CopulaFrank), 0.4, 0.5)
    assert np.isfinite(frank[0]) and frank[1] < 1e-6
    # ...and at alpha = 0.25 the Clayton variance exists again.
    ok = ROB._dpd_variance(CopulaClayton, _entry(CopulaClayton), 0.4, 0.25)
    assert np.isfinite(ok[0]) and ok[1] < 1e-2
    # The estimate is returned in every case.
    uv = _pseudo(CopulaClayton(tau_k=0.4).sample(400, seed=_seed("nanvar")))
    f = dpd_fit(CopulaClayton, uv, 0.5)
    assert np.isfinite(f.tau_k) and math.isnan(f.variance) and math.isnan(f.se_hint)


@pytest.mark.parametrize("family", [CopulaStudent, CopulaBB1])
def test_dpd_fit_refuses_two_parameter_families(family):
    uv = _pseudo(CopulaGaussian(tau_k=0.3).sample(200, seed=_seed("dpd-refuse")))
    with pytest.raises(NotImplementedError, match="one-parameter families only"):
        dpd_fit(family, uv, 0.25)


@pytest.mark.parametrize("alpha", [-0.1, 1.5])
def test_dpd_fit_refuses_an_alpha_outside_the_supported_range(alpha):
    uv = _pseudo(CopulaGaussian(tau_k=0.3).sample(200, seed=_seed("dpd-alpha")))
    with pytest.raises(ValueError, match="alpha"):
        dpd_fit(CopulaGaussian, uv, alpha)


def test_dpd_fit_with_frequency_weights_matches_repeated_rows():
    uv = _pseudo(CopulaGaussian(tau_k=0.4).sample(200, seed=_seed("dpd-weights")))
    w = np.random.default_rng(11).integers(1, 4, size=uv.shape[0]).astype(float)
    a = dpd_fit(CopulaGaussian, uv, 0.25, w)
    b = dpd_fit(CopulaGaussian, np.repeat(uv, w.astype(int), axis=0), 0.25)
    assert a.n_eff == b.n_eff == float(w.sum())
    assert a.tau_k == pytest.approx(b.tau_k, abs=1e-8)
    assert a.objective == pytest.approx(b.objective, rel=1e-10)


def test_dpd_resists_a_contaminated_sample_the_mle_follows():
    """One seed, n = 1000, τ = 0.5, 5 % discordant corner pairs.

    Measured: τ̂ = 0.2107 at α = 0 against 0.5149 at α = 0.5; the clean
    sample gives 0.5076 and 0.4977.
    """
    s = _seed("dpd-contaminated")
    x = CopulaGaussian(tau_k=0.5).sample(1000, seed=s)
    dirty = _pseudo(_contaminate(x, 0.05, np.random.default_rng(s + 1)))
    mle = dpd_fit(CopulaGaussian, dirty, 0.0).tau_k
    rob = dpd_fit(CopulaGaussian, dirty, 0.5).tau_k
    assert abs(mle - 0.5) > 0.15
    assert abs(rob - 0.5) < 0.05


def test_select_alpha_result_shape():
    uv = _pseudo(CopulaFrank(tau_k=0.4).sample(600, seed=_seed("select-shape")))
    sel = select_alpha(CopulaFrank, uv)
    assert isinstance(sel, DPDAlphaSelection)
    assert sel.pilot_alpha == 0.5 and sel.pilot_tau == sel.fits[0.5].tau_k
    assert sel.alpha in sel.alphas and sel.fit is sel.fits[sel.alpha]
    assert len(sel.alphas) == len(sel.tau_k) == len(sel.criterion) == len(sel.variance)
    assert sel.criterion[sel.alphas.index(sel.alpha)] == min(sel.criterion)
    # The criterion at the pilot is exactly its own variance term.
    i = sel.alphas.index(0.5)
    assert sel.criterion[i] == pytest.approx(sel.variance[i] / sel.n_eff, rel=1e-12)


def test_public_api_exports_the_new_names():
    import pmcprg.copulas as C

    for name in ("MleTauDiscrepancyTest", "mle_tau_discrepancy_test", "DPDFit",
                 "DPDAlphaSelection", "DPD_ALPHAS", "dpd_fit", "dpd_objective",
                 "integral_c_power", "select_alpha"):
        assert name in C.__all__ and hasattr(C, name)


def test_the_robust_module_is_not_wired_into_the_fitting_path():
    """FR-7 (c) is additive: ``fit``, ``_fit`` and the ICE M-step are untouched."""
    import inspect

    import pmcprg.copulas._base as B
    import pmcprg.copulas._fit as F

    for mod in (B, F):
        assert "_robust" not in inspect.getsource(mod)
    assert "DPD" not in inspect.getsource(F)


# ===========================================================================
# Monte Carlo (slow)
# ===========================================================================

def _level(cls, tau, n, R, tag):
    p, z = [], []
    for r in range(R):
        x = cls(tau_k=tau).sample(n, seed=_seed(tag, cls.__name__, tau, n, r))
        t = mle_tau_discrepancy_test(cls, _pseudo(x))
        assert not t.at_boundary
        p.append(t.p_value)
        z.append(t.statistic)
    return np.asarray(p), np.asarray(z)


@pytest.mark.slow
@pytest.mark.parametrize("cls, tau", [(CopulaGaussian, 0.4), (CopulaClayton, 0.4),
                                      (CopulaFrank, 0.6), (CopulaJoe, 0.2)])
def test_discrepancy_level_near_nominal(cls, tau):
    """n = 2000, R = 400, rejection at 5 % and 10 %.

    Binomial SE at R = 400: 1.09 % at 5 %, 1.50 % at 10 %; the bands below are
    roughly ±3 SE widened for the finite-sample mean offset the module
    docstring of this file reports (E[Z] ≈ 0.06-0.26 at n = 2000). Full grid
    measured over six families × three τ × two n, R = 500: 3.4-9.0 % at
    nominal 5 %.
    """
    p, z = _level(cls, tau, 2000, 400, "mc-level")
    assert 0.02 <= float(np.mean(p <= 0.05)) <= 0.095, float(np.mean(p <= 0.05))
    assert 0.05 <= float(np.mean(p <= 0.10)) <= 0.165, float(np.mean(p <= 0.10))
    assert 0.85 <= float(z.std(ddof=1)) <= 1.15, float(z.std(ddof=1))


@pytest.mark.slow
@pytest.mark.parametrize("cls", [CopulaGaussian, CopulaFrank])
def test_discrepancy_power_under_contamination(cls):
    """n = 1000, R = 200, τ = 0.4, discordant corner pairs.

    Measured at R = 500, rejection at the 5 % level, ε = 0/1/2/5/10 %:
    Gaussian 6.2/98.2/100/100/100 %, Frank 5.6/83.0/100/100/**0.2** %,
    Clayton 6.0/5.4/5.4/18.2/92.8 %. Three things this campaign says, and
    none of them is hidden here:

    * against **discordant corner outliers** the contrast is very powerful
      for the Gaussian and Frank families — 1 % of contaminating pairs is
      already enough;
    * it is much weaker for **Clayton**, whose pseudo-likelihood is dominated
      by the lower tail and barely notices upper-corner pairs, so both
      estimators move together and there is nothing to contrast;
    * Frank's power is **not monotone**: at ε = 10 % the contaminated sample
      is genuinely close to a Frank copula at a smaller τ, both estimators
      agree on that τ, and the test returns to (below) nominal. This is a
      *specification* contrast, not an outlier counter.

    A second contamination, 1-10 % of draws from a strongly opposite
    Gaussian copula (τ = −0.8), leaves the Gaussian family at nominal
    throughout (5.4-10.0 %) — a mixture of two Gaussian copulas is still
    nearly a Gaussian copula — while Clayton reaches 67.8 % and Frank 100 %
    at ε = 10 %. Only the corner case is asserted below.
    """
    rates = {}
    for eps in (0.0, 0.01, 0.02, 0.05):
        p = []
        for r in range(200):
            s = _seed("mc-power", cls.__name__, eps, r)
            x = _contaminate(cls(tau_k=0.4).sample(1000, seed=s), eps,
                             np.random.default_rng(s + 1))
            t = mle_tau_discrepancy_test(cls, _pseudo(x))
            if not t.at_boundary:
                p.append(t.p_value)
        rates[eps] = float(np.mean(np.asarray(p) <= 0.05))
    assert rates[0.0] <= 0.12, rates
    assert rates[0.01] >= 0.60, rates
    assert rates[0.02] >= 0.90, rates
    assert rates[0.05] >= 0.99, rates
    assert rates[0.0] < rates[0.01] < rates[0.05] + 1e-12, rates


@pytest.mark.slow
@pytest.mark.parametrize("cls", [CopulaGaussian, CopulaClayton, CopulaFrank])
def test_dpd_reproduces_the_mle_on_clean_data(cls):
    """n = 1000, R = 100, τ = 0.4: the gap to τ̂_MLE and the efficiency loss
    both grow smoothly with α, and α = 0.05 costs essentially nothing.

    Bands from the R = 200 campaign in this file's docstring, widened for
    R = 100.
    """
    est = {a: [] for a in (0.0, 0.05, 0.1, 0.25, 0.5)}
    for r in range(100):
        uv = _pseudo(cls(tau_k=0.4).sample(1000, seed=_seed("mc-clean", cls.__name__, r)))
        for a in est:
            est[a].append(dpd_fit(cls, uv, a).tau_k)
    base = np.asarray(est[0.0])
    gaps = {a: float(np.sqrt(np.mean((np.asarray(v) - base) ** 2))) for a, v in est.items()}
    sds = {a: float(np.std(np.asarray(v), ddof=1)) for a, v in est.items()}
    assert gaps[0.0] == 0.0
    assert gaps[0.05] < 1.5e-3 < gaps[0.25] * 2.0
    assert gaps[0.05] < gaps[0.1] < gaps[0.25] < gaps[0.5]
    assert 0.99 <= sds[0.05] / sds[0.0] <= 1.02
    assert 0.99 <= sds[0.1] / sds[0.0] <= 1.03
    assert 1.0 <= sds[0.5] / sds[0.0] <= 1.60
    for a, v in est.items():
        assert abs(float(np.mean(v)) - 0.4) < 0.01, (a, float(np.mean(v)))


@pytest.mark.slow
@pytest.mark.parametrize("cls", [CopulaGaussian, CopulaFrank])
def test_dpd_breakdown_across_contamination_fractions(cls):
    """n = 1000, R = 100, τ = 0.5, discordant corner pairs; bias of τ̂.

    Measured at R = 200, bias of τ̂ (RMSE in the campaign log):

    ==============  ====  ========  ========  ========  ========  ===========
    family          eps   MLE       DPD 0.1   DPD 0.25  DPD 0.5   τ-inversion
    ==============  ====  ========  ========  ========  ========  ===========
    Gaussian        0.00  +0.0019   +0.0016   +0.0008   −0.0028   +0.0002
    Gaussian        0.01  −0.0747   −0.0204   +0.0047   +0.0005   −0.0275
    Gaussian        0.05  −0.2882   −0.2285   −0.0241   −0.0017   −0.1428
    Gaussian        0.10  −0.4574   −0.4503   −0.4323   −0.0465   −0.2735
    Clayton         0.00  −0.0005   −0.0004   −0.0002   −0.0013   −0.0018
    Clayton         0.01  −0.0330   −0.0117   +0.0016   +0.0011   −0.0293
    Clayton         0.05  −0.1595   −0.0881   −0.0078   +0.0029   −0.1379
    Clayton         0.10  −0.3492   −0.3096   −0.1468   −0.0222   −0.2754
    Frank           0.00  +0.0006   +0.0008   +0.0010   +0.0011   +0.0003
    Frank           0.01  −0.0198   −0.0138   −0.0074   −0.0030   −0.0267
    Frank           0.05  −0.1202   −0.0967   −0.0644   −0.0330   −0.1413
    Frank           0.10  −0.2702   −0.2460   −0.2000   −0.1120   −0.2734
    ==============  ====  ========  ========  ========  ========  ===========

    So the bias shrinks monotonically in α at every contamination fraction,
    α = 0.25 holds to 5 % for the Gaussian and Clayton copulas and breaks at
    10 %, and α = 0.5 still holds at 10 % (bias 0.02-0.05 there, against
    0.35-0.46 for the MLE). Frank is the hardest of the three because its
    density has thin tails: even α = 0.5 carries −0.11 at ε = 10 %.
    Kendall's τ̂ is **not** immune either (−0.27 at ε = 10 %) — contamination
    changes the population τ, and downweighting, not ranking, is what
    protects the estimate. That is the point of FR-7 (c) next to FR-7 (b).
    """
    bias = {}
    for eps in (0.0, 0.01, 0.05, 0.10):
        est = {a: [] for a in (0.0, 0.25, 0.5)}
        for r in range(100):
            s = _seed("mc-break", cls.__name__, eps, r)
            x = _contaminate(cls(tau_k=0.5).sample(1000, seed=s), eps,
                             np.random.default_rng(s + 1))
            uv = _pseudo(x)
            for a in est:
                est[a].append(dpd_fit(cls, uv, a).tau_k)
        bias[eps] = {a: float(np.mean(v)) - 0.5 for a, v in est.items()}
    for a in (0.0, 0.25, 0.5):
        assert abs(bias[0.0][a]) < 0.01, bias
    for eps in (0.01, 0.05, 0.10):
        assert abs(bias[eps][0.5]) < abs(bias[eps][0.25]) < abs(bias[eps][0.0]), (eps, bias)
    assert abs(bias[0.01][0.0]) < abs(bias[0.05][0.0]) < abs(bias[0.10][0.0]), bias
    assert abs(bias[0.05][0.5]) * 3.0 < abs(bias[0.05][0.0]), bias


@pytest.mark.slow
@pytest.mark.parametrize("cls", [CopulaGaussian, CopulaFrank])
def test_alpha_selection_moves_with_the_contamination(cls):
    """n = 1000, R = 60, τ = 0.5: α̂ is small on clean data and larger at 5 %
    contamination, and the selected estimate stays close to the truth.

    Measured at R = 100 (mean α̂ / bias of the selected τ̂), clean then 5 %
    contaminated: Gaussian 0.13/−0.0005 then 0.37/−0.0082; Frank
    0.12/−0.0039 then 0.50/−0.0338; Clayton 0.15/+0.0015 then 0.25/−0.0084.
    Clayton is capped at α̂ = 0.25 under contamination because α ≥ 0.5 has no
    finite DPD variance for a tail-dependent family (module docstring of
    :mod:`pmcprg.copulas._robust`, "Choosing α"), and the criterion is then
    ``+inf`` there — its selected bias is nonetheless 19× smaller than the
    MLE's −0.1595 at the same contamination. The mean, not the median, is
    asserted: the clean-data distribution of α̂ straddles the grid (Gaussian
    27/11/21/41 % on 0/0.05/0.1/0.25), so its median is not stable at R = 60.
    """
    out = {}
    for eps in (0.0, 0.05):
        chosen, taus = [], []
        for r in range(60):
            s = _seed("mc-select", cls.__name__, eps, r)
            x = _contaminate(cls(tau_k=0.5).sample(1000, seed=s), eps,
                             np.random.default_rng(s + 1))
            sel = select_alpha(cls, _pseudo(x))
            chosen.append(sel.alpha)
            taus.append(sel.fit.tau_k)
        out[eps] = (float(np.mean(chosen)), float(np.mean(taus)) - 0.5)
    assert out[0.0][0] <= 0.20, out
    assert out[0.05][0] >= out[0.0][0] + 0.15, out
    assert abs(out[0.0][1]) < 0.01, out
    assert abs(out[0.05][1]) < 0.06, out
