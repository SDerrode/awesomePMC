"""Unit tests for the copula / bivariate **fitting** layer (audit T-4).

Targets the previously under-covered surface:

* ``pmcprg.copulas._fit.FitResult`` — the information criteria (``aic`` / ``bic`` /
  ``aicc`` / ``hqc``), the parametric-bootstrap GoF test, the non-parametric
  bootstrap τ-CI, and the K-fold CV log-likelihood.
* ``pmcprg.copulas._bivariate_fit`` via ``BivariateLaw.fit`` / ``fit_best``.
* the pure helpers ``_empirical_copula`` / ``_cvm_statistic`` /
  ``_empirical_tail_dep``.

Small bootstrap counts keep the module fast (not marked ``slow``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as ss

from pmcprg.copulas import BivariateLaw, CopulaClayton, CopulaGaussian
from pmcprg.copulas._fit import (
    FitResult,
    GoFResult,
    _cvm_statistic,
    _empirical_copula,
    _empirical_tail_dep,
)


@pytest.fixture
def clayton_samples() -> np.ndarray:
    return CopulaClayton(tau_k=0.5).sample(n=400, seed=0)


# ---------------------------------------------------------------------------
# FitResult — information criteria
# ---------------------------------------------------------------------------

def test_fit_information_criteria_finite(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="mle")
    assert r.n_params == 1
    assert np.isfinite(r.log_likelihood)
    for crit in (r.aic, r.bic, r.aicc, r.hqc):
        assert np.isfinite(crit)
    # Small-sample correction is non-negative → aicc ≥ aic; the data IS Clayton
    # so the fitted log-likelihood should be positive.
    assert r.aicc >= r.aic
    assert r.log_likelihood > 0.0


def test_aic_bic_formulas(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="mle")
    k, n, ll = r.n_params, r.n_obs, r.log_likelihood
    assert r.aic == pytest.approx(2.0 * k - 2.0 * ll)
    assert r.bic == pytest.approx(k * np.log(n) - 2.0 * ll)


@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_recovers_tau(clayton_samples, method):
    r = CopulaClayton.fit(clayton_samples, method=method)
    assert abs(r.tau_k - 0.5) < 0.1


def test_aicc_nan_when_too_few_obs():
    """`aicc` is undefined (n − k − 1 ≤ 0) for k=1 and n ≤ 2.

    ``fit`` itself requires n ≥ 4, so we build a degenerate ``FitResult``
    directly to exercise the property's guard branch.
    """
    r = FitResult(
        copula         = CopulaClayton(tau_k=0.4),
        method         = "tau",
        tau_k          = 0.4,
        log_likelihood = 1.0,
        n_obs          = 2,
        uv             = np.zeros((2, 2)),
    )
    assert r.n_params == 1
    assert np.isnan(r.aicc)


# ---------------------------------------------------------------------------
# FitResult — bootstrap CI / GoF / CV
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_point_estimate(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="tau")
    lo, hi = r.bootstrap_ci(B=40, seed=0)
    assert lo <= hi
    assert lo <= r.tau_k <= hi


def test_bootstrap_ci_reproducible(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="tau")
    assert r.bootstrap_ci(B=40, seed=7) == r.bootstrap_ci(B=40, seed=7)


def test_gof_test_structure_and_reproducible(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="mle")
    g = r.gof_test(B=30, seed=0)
    assert isinstance(g, GoFResult)
    assert g.statistic >= 0.0
    assert 0.0 <= g.p_value <= 1.0
    assert g.n_valid_bootstrap > 0
    # Reproducible at a fixed seed.
    g2 = r.gof_test(B=30, seed=0)
    assert g2.p_value == pytest.approx(g.p_value)


def test_gof_test_does_not_reject_correct_family(clayton_samples):
    """A GoF test of the *correct* family should not reject at α=0.05."""
    r = CopulaClayton.fit(clayton_samples, method="mle")
    assert r.gof_test(B=40, seed=0).p_value > 0.05


def test_cv_loglik_finite(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="mle")
    assert np.isfinite(r.cv_loglik(K=4, seed=0))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_empirical_copula_in_unit_interval():
    rng = np.random.default_rng(0)
    uv  = rng.uniform(size=(200, 2))
    Cn  = _empirical_copula(uv, uv)
    assert Cn.shape == (200,)
    assert np.all((Cn >= 0.0) & (Cn <= 1.0))


def test_cvm_statistic_nonnegative(clayton_samples):
    r = CopulaClayton.fit(clayton_samples, method="mle")
    assert _cvm_statistic(r.uv, r.copula) >= 0.0


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_empirical_tail_dep_in_unit_interval(clayton_samples, side):
    u_grid = np.linspace(0.05, 0.45, 6)
    lam = _empirical_tail_dep(clayton_samples, u_grid, side)
    assert lam.shape == u_grid.shape
    assert np.all((lam >= 0.0) & (lam <= 1.0))


# ---------------------------------------------------------------------------
# BivariateLaw.fit / fit_best (BivariateFitResult)
# ---------------------------------------------------------------------------

def _gaussian_sklar_data(tau: float, n: int, seed: int) -> np.ndarray:
    law = BivariateLaw(
        copula       = CopulaGaussian(tau_k=tau),
        left_margin  = (ss.norm, 0.0, 1.0),
        right_margin = (ss.norm, 0.0, 1.0),
    )
    law.set_seed(seed)
    return law.sample(n)


def test_bivariate_fit_recovers_tau_and_scores():
    data = _gaussian_sklar_data(tau=0.4, n=600, seed=0)
    res  = BivariateLaw.fit(
        data, CopulaGaussian, ss.norm, ss.norm, copula_method="mle",
    )
    assert abs(res.copula_fit.tau_k - 0.4) < 0.12
    for crit in (res.aic, res.bic, res.aicc, res.hqc):
        assert np.isfinite(crit)
    assert np.isfinite(res.log_likelihood)


def test_bivariate_fit_input_validation():
    good = np.random.default_rng(0).uniform(size=(20, 2))
    with pytest.raises(ValueError, match=r"shape"):
        BivariateLaw.fit(np.zeros((10, 3)), CopulaGaussian, ss.norm, ss.norm)
    with pytest.raises(ValueError, match=r"4 observations"):
        BivariateLaw.fit(np.zeros((3, 2)), CopulaGaussian, ss.norm, ss.norm)
    with pytest.raises(ValueError, match=r"scipy continuous"):
        BivariateLaw.fit(good, CopulaGaussian, "not-a-dist", ss.norm)


def test_bivariate_fit_best_sorted_by_aic():
    data = _gaussian_sklar_data(tau=0.5, n=500, seed=1)
    results = BivariateLaw.fit_best(
        data, ss.norm, ss.norm,
        copula_families=[CopulaGaussian, CopulaClayton],
        copula_method="mle",
    )
    assert len(results) == 2
    aics = [r.aic for r in results]
    assert aics == sorted(aics)            # ascending AIC (best first)
