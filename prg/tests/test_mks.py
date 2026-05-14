"""Tests for :mod:`prg.diagnostics.mks` — multivariate Kolmogorov-Smirnov test.

Coverage:
* Input validation (alpha range, sample dim mismatch, empty arrays, …).
* Calibration under H0 (Type-I error roughly ≤ nominal α).
* Power under H1 (Type-II error small on a contrast setting).
* Univariate sanity vs scipy's :func:`scipy.stats.ks_1samp` / ``ks_2samp``.
* 1- vs 2-sample dispatcher (:func:`mks_test`).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as _ss

from prg.diagnostics import MKSResult, mks_1samp, mks_2samp, mks_test


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_alpha_out_of_range_raises():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(100, 2))
    cdf = lambda t: float(_ss.norm.cdf(t[0]) * _ss.norm.cdf(t[1]))
    with pytest.raises(ValueError, match=r"alpha must be in"):
        mks_1samp(x, cdf, alpha=1.5)
    with pytest.raises(ValueError, match=r"alpha must be in"):
        mks_1samp(x, cdf, alpha=-0.1)


def test_alpha_wrong_type_raises():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(100, 2))
    cdf = lambda t: float(_ss.norm.cdf(t[0]) * _ss.norm.cdf(t[1]))
    with pytest.raises(TypeError, match="alpha"):
        mks_1samp(x, cdf, alpha="0.05")


def test_two_sample_dim_mismatch_raises():
    rng = np.random.default_rng(0)
    a   = rng.standard_normal(size=(80, 2))
    b   = rng.standard_normal(size=(80, 3))
    with pytest.raises(ValueError, match=r"features"):
        mks_2samp(a, b)


def test_empty_sample_raises():
    cdf = lambda t: float(_ss.norm.cdf(t[0]))
    with pytest.raises(ValueError, match=r"empty"):
        mks_1samp(np.empty((0, 1)), cdf)


def test_dispatcher_requires_exactly_one_of_other_or_cdf():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(80, 2))
    with pytest.raises(ValueError, match=r"exactly one"):
        mks_test(x)
    with pytest.raises(ValueError, match=r"exactly one"):
        mks_test(x, other=x, cdf=lambda t: 0.5)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

def test_returns_mks_result_with_expected_fields():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(120, 2))
    cdf = lambda t: float(_ss.norm.cdf(t[0]) * _ss.norm.cdf(t[1]))
    res = mks_1samp(x, cdf)
    assert isinstance(res, MKSResult)
    assert res.dim == 2
    assert res.n_samples_x == 120
    assert res.n_samples_y is None
    assert 0.0 <= res.statistic <= 1.0
    assert res.critical_value > 0.0
    assert isinstance(res.reject, bool)


# ---------------------------------------------------------------------------
# Calibration under H0 — Type-I error roughly ≤ nominal α
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_type_one_error_under_H0_one_sample():
    """Under H0 (true distribution), the rejection rate over many trials
    must not exceed the nominal α (finite-sample Naaman bound is
    conservative — so the empirical rate is typically much smaller)."""
    alpha   = 0.05
    n_trial = 80
    N       = 200
    cdf2d   = lambda t: float(_ss.norm.cdf(t[0]) * _ss.norm.cdf(t[1]))

    rng = np.random.default_rng(42)
    rejects = 0
    for _ in range(n_trial):
        x = rng.standard_normal(size=(N, 2))
        if mks_1samp(x, cdf2d, alpha=alpha).reject:
            rejects += 1
    rate = rejects / n_trial
    # The Naaman bound is conservative — empirical rejection rate should
    # be well below α.
    assert rate <= alpha + 0.03, (
        f"Type-I error {rate:.3f} exceeds α+0.03={alpha + 0.03:.3f}"
    )


@pytest.mark.slow
def test_type_one_error_under_H0_two_sample():
    alpha   = 0.05
    n_trial = 80
    N       = 200

    rng = np.random.default_rng(42)
    rejects = 0
    for _ in range(n_trial):
        a = rng.standard_normal(size=(N, 2))
        b = rng.standard_normal(size=(N, 2))
        if mks_2samp(a, b, alpha=alpha).reject:
            rejects += 1
    rate = rejects / n_trial
    assert rate <= alpha + 0.03, (
        f"Type-I error {rate:.3f} exceeds α+0.03={alpha + 0.03:.3f}"
    )


# ---------------------------------------------------------------------------
# Power under H1 — Type-II error small on a contrast setting
# ---------------------------------------------------------------------------

def test_power_two_sample_shifted_mean():
    """Two 2-D Gaussians shifted by +2 in both coords are clearly
    different — H0 must be rejected almost surely."""
    rng = np.random.default_rng(0)
    a   = rng.standard_normal(size=(300, 2))
    b   = rng.standard_normal(size=(300, 2)) + 2.0
    res = mks_2samp(a, b, alpha=0.05)
    assert res.reject
    # The statistic should be much larger than the critical value.
    assert res.statistic > 2 * res.critical_value


def test_power_one_sample_wrong_distribution():
    """A sample from N(2, 1) tested against the N(0, 1) CDF must reject."""
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=400) + 2.0
    res = mks_1samp(x, lambda t: float(_ss.norm.cdf(t[0])), alpha=0.05)
    assert res.reject


# ---------------------------------------------------------------------------
# Univariate sanity — the test should agree with scipy on order of magnitude
# ---------------------------------------------------------------------------

def test_univariate_statistic_close_to_scipy_one_sample():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=500)
    res = mks_1samp(x, lambda t: float(_ss.norm.cdf(t[0])))
    ref = _ss.ks_1samp(x, _ss.norm.cdf).statistic
    # Order-of-magnitude agreement (the multivariate extension uses a
    # slightly different corner grid, so exact equality is not expected).
    assert abs(res.statistic - ref) < 0.05


def test_univariate_statistic_close_to_scipy_two_sample():
    rng = np.random.default_rng(0)
    a   = rng.standard_normal(size=500)
    b   = rng.standard_normal(size=500)
    res = mks_2samp(a, b)
    ref = _ss.ks_2samp(a, b).statistic
    assert abs(res.statistic - ref) < 0.06


# ---------------------------------------------------------------------------
# 3-D smoke test
# ---------------------------------------------------------------------------

def test_3d_runs_and_returns_dim_three():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(200, 3))
    y   = rng.standard_normal(size=(200, 3))
    res = mks_2samp(x, y)
    assert res.dim == 3
    assert not res.reject


# ---------------------------------------------------------------------------
# Dispatcher routes to 1-/2-sample appropriately
# ---------------------------------------------------------------------------

def test_dispatcher_routes_one_sample():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(200, 2))
    cdf = lambda t: float(_ss.norm.cdf(t[0]) * _ss.norm.cdf(t[1]))
    res = mks_test(x, cdf=cdf)
    assert res.n_samples_y is None
    assert res.dim == 2


def test_dispatcher_routes_two_sample():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(200, 2))
    y   = rng.standard_normal(size=(180, 2))
    res = mks_test(x, other=y)
    assert res.n_samples_y == 180
    assert res.dim == 2


# ---------------------------------------------------------------------------
# Asymptotic critical value is tighter than the finite-sample one
# ---------------------------------------------------------------------------

def test_asymptotic_critical_value_is_smaller():
    rng = np.random.default_rng(0)
    x   = rng.standard_normal(size=(300, 2))
    cdf = lambda t: float(_ss.norm.cdf(t[0]) * _ss.norm.cdf(t[1]))
    fin = mks_1samp(x, cdf, asymptotic=False)
    asy = mks_1samp(x, cdf, asymptotic=True)
    assert asy.critical_value < fin.critical_value
    assert asy.statistic == fin.statistic   # statistic is independent of bound
