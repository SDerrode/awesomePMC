"""Tests for pmcprg.diagnostics.radial_symmetry (audit FR-10).

Fast, small-N/small-B versions of the size and power studies reported in the
FR-10 commit — enough replicates to see the qualitative behaviour (symmetric
families near nominal, asymmetric ones rejected more often), not enough to
pin the level to two digits; the full study lives in the commit message.
"""
import numpy as np
import pytest

from pmcprg.copulas import (
    CopulaClayton,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH,
    CopulaProduct,
)
from pmcprg.diagnostics.radial_symmetry import (
    RadialSymmetryResult,
    radial_symmetry_statistic,
    radial_symmetry_test,
)


def test_result_fields_and_types():
    cop = CopulaGaussian(tau_k=0.5)
    uv = cop.sample(80, seed=1)
    res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=40, seed=2)
    assert isinstance(res, RadialSymmetryResult)
    assert res.n == 80
    assert res.B == 40
    assert 0.0 <= res.p_value <= 1.0
    assert isinstance(res.reject, bool)
    assert res.statistic >= 0.0


def test_independence_copula_never_rejects():
    cop = CopulaProduct(tau_k=0.0)
    for seed in range(5):
        uv = cop.sample(150, seed=seed)
        res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=100, seed=seed + 1000)
        assert not res.reject, f"independence copula rejected at seed={seed}: {res}"


def test_statistic_is_nonnegative_and_finite():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=60), rng.uniform(size=60)
    stat = radial_symmetry_statistic(u, v)
    assert np.isfinite(stat)
    assert stat >= 0.0


def test_statistic_grows_under_stronger_asymmetric_dependence():
    # More Clayton-like (one-sided) dependence should push the statistic up,
    # on average, relative to independence — a coarse monotonicity check on
    # the formula (averaged over replicates to beat sampling noise) rather
    # than a precise power claim (see the study below).
    rng = np.random.default_rng(0)
    indep_stats = [
        radial_symmetry_statistic(rng.uniform(size=200), rng.uniform(size=200))
        for _ in range(20)
    ]
    clayton = CopulaClayton(tau_k=0.7)
    clayton_stats = [
        radial_symmetry_statistic(*clayton.sample(200, seed=1000 + r).T)
        for r in range(20)
    ]
    assert np.mean(clayton_stats) > np.mean(indep_stats)


def test_degenerate_samples_do_not_crash():
    for n in (0, 1, 2, 3):
        x = np.arange(n, dtype=float)
        res = radial_symmetry_test(x, x, B=10, seed=0)
        assert np.isnan(res.statistic)
        assert np.isnan(res.p_value)
        assert res.reject is False


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        radial_symmetry_test(np.zeros(5), np.zeros(6))


def test_weights_are_rejected_explicitly():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(NotImplementedError):
        radial_symmetry_test(x, x, weights=np.ones(20))


def test_invalid_alpha_raises():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(ValueError):
        radial_symmetry_test(x, x, alpha=1.5)


# ---------------------------------------------------------------------------
# Small size / power studies — qualitative, not a precise level/power claim.
# ---------------------------------------------------------------------------

def _rejection_rate(family_cls, tau, n, n_reps, B, alpha, seed0):
    cop = family_cls(tau_k=tau)
    n_reject = 0
    n_ok = 0
    for r in range(n_reps):
        uv = cop.sample(n, seed=seed0 * 1000 + r)
        res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=B, seed=seed0 * 1000 + r + 1,
                                    alpha=alpha)
        if np.isfinite(res.p_value):
            n_ok += 1
            n_reject += int(res.reject)
    return n_reject / n_ok if n_ok else float("nan")


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.5),
    (CopulaFrank, 0.5),
])
def test_size_study_symmetric_families_near_nominal(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=150, n_reps=60, B=60, alpha=0.10,
                            seed0=hash((family_cls.__name__, tau)) % 1000)
    # 60 replicates at nominal 10 %: binomial sd ~= 0.039, so a generous band.
    assert rate < 0.30, f"{family_cls.__name__} tau={tau}: over-rejected at {rate:.3f}"


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaClayton, 0.6),
    (CopulaGH, 0.6),
])
def test_power_study_asymmetric_families_reject_more_often(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=200, n_reps=60, B=60, alpha=0.10,
                            seed0=hash((family_cls.__name__, tau, "power")) % 1000)
    assert rate > 0.30, f"{family_cls.__name__} tau={tau}: power too low at {rate:.3f}"
