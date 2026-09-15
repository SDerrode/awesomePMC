"""
test_student.py — dedicated coverage for the Student-t copula (audit T-4).

``student.py`` was the least-covered copula module (49 %): its 2-D MLE fit
(the only ``fit`` override in the family registry), the tail-dependence
formula and the parameter validation had no dedicated tests. These tests pin
the *documented behaviours*: ρ = sin(πτ/2), the ν > 2 domain, the symmetric
tail dependence λ(ν) and its Gaussian limit, the no-closed-form CDF contract,
and the (ρ, ν) MLE — including the ``method='tau'`` fallback warning.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from pmcprg.copulas import CopulaStudent
from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
from pmcprg.exceptions import CopulaParameterError
from scipy.stats import t as _t


# ---------------------------------------------------------------------------
# Construction / parameter domain
# ---------------------------------------------------------------------------

def test_student_default_df_and_rho_link():
    """Default ν = 4; ρ = sin(π·τ/2); df is persisted into params."""
    cop = CopulaStudent(tau_k=0.5)
    assert cop.df == 4.0
    assert cop.params["df"] == 4.0
    assert cop.theta == pytest.approx(math.sin(math.pi * 0.5 / 2.0))


@pytest.mark.parametrize("df", [2.0, 1.0, -3.0])
def test_student_df_at_most_two_rejected(df):
    """The Student copula requires ν > 2 (finite variance)."""
    with pytest.raises(CopulaParameterError, match="df must be > 2"):
        CopulaStudent(tau_k=0.3, df=df)


def test_student_df_bounds_single_source():
    """The ICE-driven fit bounds for ν come from the shared registry
    (audit A-2) and respect the ν > 2 domain."""
    lo, hi, default = EXTRA_PARAM_BOUNDS_BY_PARAM["df"]
    assert lo >= 2.0
    assert lo < default < hi


# ---------------------------------------------------------------------------
# PDF / CDF / h-function
# ---------------------------------------------------------------------------

def test_student_cdf_has_no_closed_form():
    """Documented contract: the CDF raises (bivariate-t integral)."""
    with pytest.raises(NotImplementedError):
        CopulaStudent(tau_k=0.3).cdf([0.5, 0.5])


def test_student_pdf_finite_positive():
    cop = CopulaStudent(tau_k=0.4, df=5.0)
    for uv in ([0.3, 0.7], [0.5, 0.5], [0.01, 0.99]):
        p = cop.pdf(uv)
        assert math.isfinite(p) and p > 0.0


def test_student_conditional_cdf_properties():
    """h(v|u) is a CDF in v: within [0,1] and non-decreasing; near
    independence (τ→0, large ν) it approaches v."""
    cop = CopulaStudent(tau_k=0.4, df=5.0)
    grid = np.linspace(0.05, 0.95, 19)
    h = np.array([cop.conditional_cdf(v, 0.3) for v in grid])
    assert np.all((h >= 0.0) & (h <= 1.0))
    assert np.all(np.diff(h) >= 0.0)

    near_indep = CopulaStudent(tau_k=1e-9, df=200.0)
    for v in (0.2, 0.5, 0.8):
        assert near_indep.conditional_cdf(v, 0.5) == pytest.approx(v, abs=5e-3)


# ---------------------------------------------------------------------------
# Tail dependence
# ---------------------------------------------------------------------------

def test_student_tail_dependence_formula_and_df_monotonicity():
    """λ_L = λ_U = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) — checked against an
    independent evaluation; heavier tails (smaller ν) → larger λ."""
    cop4 = CopulaStudent(tau_k=0.5, df=4.0)
    lam_l, lam_u = cop4.tail_dependence()
    assert lam_l == lam_u                    # symmetric by construction

    rho = cop4.theta
    expected = 2.0 * _t.cdf(
        -math.sqrt(5.0 * (1.0 - rho) / (1.0 + rho)), df=5.0
    )
    assert lam_l == pytest.approx(expected, rel=1e-12)

    lam8, _ = CopulaStudent(tau_k=0.5, df=8.0).tail_dependence()
    assert 0.0 < lam8 < lam_l                # lighter tails at larger ν

    # Gaussian limit: ν → ∞ kills the tail dependence.
    lam_big, _ = CopulaStudent(tau_k=0.5, df=500.0).tail_dependence()
    assert lam_big < 0.01


# ---------------------------------------------------------------------------
# 2-D MLE fit
# ---------------------------------------------------------------------------

def test_student_fit_rejects_bad_input():
    with pytest.raises(ValueError, match="shape"):
        CopulaStudent.fit(np.zeros((10, 3)))
    with pytest.raises(ValueError, match="at least 8"):
        CopulaStudent.fit(np.zeros((5, 2)))


def test_student_fit_tau_method_warns_and_uses_mle(caplog):
    """``method='tau'`` is under-determined for (ρ, ν): it must warn and
    fall back to MLE, still returning a valid FitResult."""
    data = CopulaStudent(tau_k=0.4, df=4.0).sample(60, seed=3)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.elliptical.student"):
        r = CopulaStudent.fit(data, method="tau")
    assert any("under-determined" in rec.message for rec in caplog.records)
    assert r.method == "mle"
    assert isinstance(r.copula, CopulaStudent)
    assert math.isfinite(r.log_likelihood)
    assert r.n_obs == 60
    assert -1.0 < r.tau_k < 1.0


@pytest.mark.slow
def test_student_fit_recovers_tau_and_df():
    """MLE on data simulated from a known (τ=0.5, ν=4) Student copula
    recovers τ within ±0.1 and a plausible ν."""
    data = CopulaStudent(tau_k=0.5, df=4.0).sample(300, seed=7)
    r = CopulaStudent.fit(data, method="mle")
    assert abs(r.tau_k - 0.5) < 0.1
    assert 2.0 < r.copula.df < 10.0
    assert r.copula.params["df"] == pytest.approx(r.copula.df)
    assert math.isfinite(r.log_likelihood)
