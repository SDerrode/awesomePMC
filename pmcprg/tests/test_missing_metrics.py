"""Metrics of ``pmcprg.missing.metrics`` against hand-computed and scipy references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from pmcprg.missing.metrics import (
    ErrorRates, crps_from_samples, crps_gaussian, error_rate_split,
    interval_coverage, mae, mutual_information, pearson, rmse,
)

T = np.array([1.0, 2.0, 3.0, 4.0])
H = np.array([1.0, 3.0, 3.0, 6.0])
M = np.array([False, True, True, True])


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def test_rmse_mae_by_hand():
    # errors on the mask: 1, 0, 2
    assert rmse(T, H, M) == pytest.approx(np.sqrt(5 / 3))
    assert mae(T, H, M) == pytest.approx(1.0)
    assert rmse(T, H, M.astype(int)) == pytest.approx(np.sqrt(5 / 3))  # 0/1 mask


def test_no_cap_on_large_errors():
    assert rmse([0.0, 0.0], [1e3, 1e3], [True, True]) == pytest.approx(1e3)


def test_metrics_are_pooled_over_series():
    t = np.array([[1.0, 10.0], [2.0, 20.0]])
    h = np.array([[2.0, 10.0], [2.0, 23.0]])
    m = np.array([[True, False], [True, True]])
    assert mae(t, h, m) == pytest.approx((1 + 0 + 3) / 3)


def test_nan_handling():
    t = np.array([1.0, np.nan, 3.0, 4.0])
    h = np.array([np.nan, 5.0, 3.5, 4.0])
    m = np.array([False, True, True, True])
    # NaN truth at a masked position is dropped; NaN prediction outside the
    # mask is ignored.
    assert mae(t, h, m) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="Y_hat has 1 non-finite value"):
        mae(t, np.array([1.0, 1.0, np.nan, 4.0]), m)
    with pytest.raises(ValueError, match="finite ground truth"):
        rmse(t, h, [False, True, False, False])


def test_refusals():
    with pytest.raises(ValueError, match="selects no position"):
        rmse(T, H, np.zeros(4, bool))
    with pytest.raises(ValueError, match="mask is required"):
        rmse(T, H, None)
    with pytest.raises(ValueError, match=r"Y_hat has shape \(3,\), expected \(4,\)"):
        rmse(T, H[:3], M)
    with pytest.raises(ValueError, match=r"mask has shape \(4, 1\)"):
        rmse(T, H, M[:, None])
    with pytest.raises(ValueError, match="boolean"):
        rmse(T, H, np.array([0, 2, 1, 1]))
    with pytest.raises(ValueError, match="numeric"):
        rmse(T, ["a", "b", "c", "d"], M)
    with pytest.raises(ValueError, match="empty"):
        rmse([], [], [])


def test_mutual_information_matches_sklearn_on_imputegap_binning():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    t = rng.normal(size=500)
    h = t + rng.normal(scale=0.7, size=500)
    m = rng.random(500) < 0.6
    ref = sk.mutual_info_score(
        np.digitize(t[m], np.histogram_bin_edges(t[m], bins=10)),
        np.digitize(h[m], np.histogram_bin_edges(h[m], bins=10)),
    )
    assert mutual_information(t, h, m) == pytest.approx(ref, rel=1e-12)


def test_mutual_information_by_hand():
    # Perfect imputation: MI = entropy of the labels. Four values, one per
    # bin (edges 0, .3, .6, .9, 1.2, 1.5, … , 3 with bins=10) — the maximum
    # 3 falls on the last edge, ImputeGAP's eleventh label: 4 distinct labels.
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert mutual_information(t, t, np.ones(4, bool)) == pytest.approx(np.log(4))
    # Two equiprobable labels each side, independent → 0.
    a = np.array([0.0, 0.0, 1.0, 1.0])
    b = np.array([0.0, 1.0, 0.0, 1.0])
    assert mutual_information(a, b, np.ones(4, bool), bins=2) == pytest.approx(0.0)
    assert mutual_information(a, a, np.ones(4, bool), bins=2) == pytest.approx(np.log(2))
    with pytest.raises(ValueError, match="bins"):
        mutual_information(a, b, np.ones(4, bool), bins=0)


def test_pearson_matches_scipy_and_is_nan_when_constant():
    rng = np.random.default_rng(1)
    t, h = rng.normal(size=50), rng.normal(size=50)
    m = rng.random(50) < 0.5
    assert pearson(t, h, m) == pytest.approx(stats.pearsonr(t[m], h[m])[0])
    assert np.isnan(pearson(t, np.full(50, 2.0), m))
    assert np.isnan(pearson(t, h, np.eye(1, 50, 3, dtype=bool)[0]))   # one point


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------


def _crps_brute(y, x):
    x = np.asarray(x, float)
    M_ = len(x)
    return (np.mean(np.abs(x - y))
            - np.abs(x[:, None] - x[None, :]).sum() / (2 * M_ * (M_ - 1)))


def test_crps_from_samples_by_hand_and_brute_force():
    assert crps_from_samples([1.0], [[0.0, 2.0]]) == pytest.approx(0.0)
    assert crps_from_samples([3.0], [[0.0, 2.0]]) == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    y = rng.normal(size=6)
    s = rng.normal(size=(6, 37))
    ref = np.mean([_crps_brute(y[i], s[i]) for i in range(6)])
    assert crps_from_samples(y, s) == pytest.approx(ref, rel=1e-12)


def test_crps_gaussian_closed_form_against_integral():
    """CRPS = ∫ (F(x) − 1{x ≥ y})² dx, integrated numerically."""
    mu, sd, y = 0.3, 1.7, 1.1
    F = stats.norm(mu, sd).cdf
    ref = (integrate.quad(lambda x: F(x) ** 2, -np.inf, y)[0]
           + integrate.quad(lambda x: (1 - F(x)) ** 2, y, np.inf)[0])
    assert crps_gaussian([y], [mu], [sd]) == pytest.approx(ref, rel=1e-8)
    # z = 0: sd (2 φ(0) − 1/√π)
    assert crps_gaussian(0.0, 0.0, 2.0) == pytest.approx(
        2.0 * (2 * stats.norm.pdf(0) - 1 / np.sqrt(np.pi)))


def test_crps_gaussian_agrees_with_sample_crps():
    rng = np.random.default_rng(3)
    y = np.array([-1.0, 0.0, 0.4, 2.5])
    mu = np.array([0.0, 0.5, 0.4, 1.0])
    sd = np.array([1.0, 0.3, 2.0, 0.8])
    draws = mu[:, None] + sd[:, None] * rng.standard_normal((4, 40_000))
    assert crps_from_samples(y, draws) == pytest.approx(
        crps_gaussian(y, mu, sd), rel=5e-3)


def test_probabilistic_masks_scalars_and_refusals():
    y = np.array([[0.0, 1.0], [np.nan, 3.0]])
    m = np.array([[True, False], [True, True]])
    # evaluated: (0,0) and (1,1); NaN truth dropped
    expected = np.mean([crps_gaussian(0.0, 0.0, 1.0), crps_gaussian(3.0, 0.0, 1.0)])
    assert crps_gaussian(y, 0.0, 1.0, mask=m) == pytest.approx(expected)
    s = np.zeros((2, 2, 5))
    s[..., :] = np.arange(5.0)
    assert crps_from_samples(y, s, mask=m) == pytest.approx(
        np.mean([_crps_brute(0.0, np.arange(5.0)), _crps_brute(3.0, np.arange(5.0))]))
    with pytest.raises(ValueError, match="at least 2 draws"):
        crps_from_samples([1.0], [[1.0]])
    with pytest.raises(ValueError, match=r"expected \(2, 2\) \+ \(M,\)"):
        crps_from_samples(y, np.zeros((2, 5)))
    with pytest.raises(ValueError, match="non-finite draws at 1"):
        crps_from_samples([1.0, 2.0], [[0.0, np.nan], [1.0, 2.0]])
    with pytest.raises(ValueError, match="sd must be > 0"):
        crps_gaussian([1.0], [0.0], [0.0])
    with pytest.raises(ValueError, match=r"mean has shape \(3,\), expected \(1,\) \(the shape of Y_true\) or a scalar"):
        crps_gaussian([1.0], np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="selects no position"):
        crps_gaussian([1.0], 0.0, 1.0, mask=[False])


def test_interval_coverage():
    y = np.array([0.0, 1.0, 2.0, 3.0, np.nan])
    lo = np.array([-1.0, 1.0, 2.5, 0.0, 0.0])
    hi = np.array([1.0, 1.0, 3.0, 2.0, 0.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(2 / 4)   # closed bounds
    assert interval_coverage(y, lo, hi, mask=[False, False, True, True, True]) == 0.0
    assert interval_coverage(y, -10, 10) == 1.0
    with pytest.raises(ValueError, match="lo > hi at 3 evaluated"):
        interval_coverage(y, hi, lo)


# ---------------------------------------------------------------------------
# Classification split
# ---------------------------------------------------------------------------


def test_error_rate_split_by_hand():
    xt = np.array([0, 0, 1, 1, 0, 1])
    xh = np.array([0, 1, 1, 0, 0, 1])
    m = np.array([True, True, False, False, False, True])
    r = error_rate_split(xt, xh, m, align=False)
    assert isinstance(r, ErrorRates)
    assert r.missing == pytest.approx(1 / 3)
    assert r.observed == pytest.approx(1 / 3)
    assert r.overall == pytest.approx(2 / 6)
    assert (r.n_missing, r.n_observed) == (3, 3)


def test_error_rate_split_aligns_labels_like_error_rate():
    from pmcprg.pmc.inference import error_rate

    rng = np.random.default_rng(4)
    xt = rng.integers(0, 3, size=300)
    xh = np.where(rng.random(300) < 0.8, xt, rng.integers(0, 3, size=300))
    xh = np.array([2, 0, 1])[xh]                        # permuted labels
    m = rng.random(300) < 0.3
    r = error_rate_split(xt, xh, m)
    assert r.overall == pytest.approx(error_rate(xt, xh))
    wrong = xt != np.array([1, 2, 0])[xh]               # inverse permutation
    assert r.missing == pytest.approx(wrong[m].mean())
    assert r.observed == pytest.approx(wrong[~m].mean())
    assert error_rate_split(xt, xh, m, align=False).overall > 0.5


def test_error_rate_split_multichannel_mask_and_edges():
    xt = np.array([0, 1, 1, 0])
    xh = np.array([0, 1, 0, 0])
    m2 = np.array([[False, False], [True, False], [False, True], [False, False]])
    r = error_rate_split(xt, xh, m2, align=False)       # rows 1, 2 missing
    assert (r.n_missing, r.missing, r.observed) == (2, 0.5, 0.0)
    allm = error_rate_split(xt, xh, np.ones(4, bool), align=False)
    assert np.isnan(allm.observed) and allm.missing == allm.overall == 0.25
    with pytest.raises(ValueError, match="no missing position"):
        error_rate_split(xt, xh, np.zeros(4, bool))
    with pytest.raises(ValueError, match="same shape"):
        error_rate_split(xt, xh[:3], np.ones(4, bool))
    with pytest.raises(ValueError, match="integer labels"):
        error_rate_split(np.array([0.0, np.nan, 1.0, 1.0]), xh, np.ones(4, bool))
    with pytest.raises(ValueError, match=r"mask has shape \(3,\)"):
        error_rate_split(xt, xh, np.ones(3, bool))
