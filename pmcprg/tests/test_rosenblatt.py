"""Tests for pmcprg.diagnostics.rosenblatt (audit FR-10, Rosenblatt-transform GoF).

Fast, small-N/small-B versions of the size and power studies reported in the
FR-10 commit — enough replicates to see the qualitative behaviour (correct
size regardless of the true family's own symmetry, higher rejection when the
fitted family is wrong), not enough to pin the level to two digits; the full
study lives in the commit message, following the same convention as
``test_radial_symmetry.py``.
"""
import zlib

import numpy as np
import pytest

from pmcprg.copulas import (
    CopulaClayton,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH,
    CopulaProduct,
)
from pmcprg.diagnostics.rosenblatt import (
    RosenblattGoFResult,
    rosenblatt_gof_test,
    rosenblatt_statistic,
    rosenblatt_transform,
)


def _stable_seed(*parts) -> int:
    """A seed derived from ``parts``, stable across interpreter runs.

    Python's ``hash()`` of a str (or a tuple containing one) is salted at
    process start (``PYTHONHASHSEED``), so a test seeded from it draws a
    different Monte-Carlo sample every run — a size/power study run close to
    its threshold then fails intermittently for no code reason. CRC32 has no
    such salt.
    """
    return zlib.crc32(repr(parts).encode()) % 1000


def test_result_fields_and_types():
    cop = CopulaGaussian(tau_k=0.5)
    uv = cop.sample(80, seed=1)
    res = rosenblatt_gof_test(uv[:, 0], uv[:, 1], CopulaGaussian, B=40, seed=2)
    assert isinstance(res, RosenblattGoFResult)
    assert res.n == 80
    assert res.B == 40
    assert res.family == "CopulaGaussian"
    assert 0.0 <= res.p_value <= 1.0
    assert isinstance(res.reject, bool)
    assert res.statistic >= 0.0
    assert res.n_valid <= res.B


def test_transform_of_independence_copula_is_identity_in_u():
    # For the independence copula h(v|u) = v, so the Rosenblatt transform
    # leaves the pseudo-observations unchanged: (U, V) = (u, v).
    cop = CopulaProduct(tau_k=0.0)
    rng = np.random.default_rng(0)
    u, v = rng.uniform(0.05, 0.95, 50), rng.uniform(0.05, 0.95, 50)
    U, V = rosenblatt_transform(u, v, cop)
    assert U == pytest.approx(u)
    assert V == pytest.approx(v, abs=1e-6)


def test_independence_copula_rejection_rate_near_nominal():
    # A single replicate can reject at nominal alpha by chance (this *is* the
    # correctly-specified case) — check the rate over enough replicates
    # instead of demanding zero rejections.
    cop = CopulaProduct(tau_k=0.0)
    n_reject = 0
    n_reps = 40
    for seed in range(n_reps):
        uv = cop.sample(150, seed=seed)
        res = rosenblatt_gof_test(uv[:, 0], uv[:, 1], CopulaProduct, B=60, seed=seed + 1000)
        n_reject += int(res.reject)
    rate = n_reject / n_reps
    assert rate < 0.25, f"independence copula over-rejected at rate={rate:.3f}"


def test_statistic_is_nonnegative_and_finite():
    rng = np.random.default_rng(0)
    U, V = rng.uniform(size=60), rng.uniform(size=60)
    stat = rosenblatt_statistic(U, V)
    assert np.isfinite(stat)
    assert stat >= 0.0


def test_statistic_grows_under_misspecification():
    # Fitting Gauss to Clayton-generated data should push S_n up, on average,
    # relative to fitting Gauss to genuinely Gaussian data — a coarse
    # monotonicity check on the formula (averaged to beat sampling noise)
    # rather than a precise power claim (see the study below).
    rng_seed = 0
    n = 200
    gauss_stats = []
    clayton_stats = []
    for r in range(20):
        cop_g = CopulaGaussian(tau_k=0.5)
        uv_g = cop_g.sample(n, seed=rng_seed + r)
        fit_g = CopulaGaussian.fit(uv_g)
        from pmcprg.diagnostics.rosenblatt import _pseudo_obs
        u, v = _pseudo_obs(uv_g[:, 0], uv_g[:, 1])
        U, V = rosenblatt_transform(u, v, fit_g.copula)
        gauss_stats.append(rosenblatt_statistic(U, V))

        cop_c = CopulaClayton(tau_k=0.5)
        uv_c = cop_c.sample(n, seed=rng_seed + r)
        fit_c = CopulaGaussian.fit(uv_c)   # wrong family on purpose
        u, v = _pseudo_obs(uv_c[:, 0], uv_c[:, 1])
        U, V = rosenblatt_transform(u, v, fit_c.copula)
        clayton_stats.append(rosenblatt_statistic(U, V))
    assert np.mean(clayton_stats) > np.mean(gauss_stats)


def test_degenerate_samples_do_not_crash():
    for n in (0, 1, 2, 3):
        x = np.arange(n, dtype=float)
        res = rosenblatt_gof_test(x, x, CopulaGaussian, B=10, seed=0)
        assert np.isnan(res.statistic)
        assert np.isnan(res.p_value)
        assert res.reject is False
        assert res.n_valid == 0


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        rosenblatt_gof_test(np.zeros(5), np.zeros(6), CopulaGaussian)


def test_weights_are_rejected_explicitly():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(NotImplementedError):
        rosenblatt_gof_test(x, x, CopulaGaussian, weights=np.ones(20))


def test_invalid_alpha_raises():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(ValueError):
        rosenblatt_gof_test(x, x, CopulaGaussian, alpha=1.5)


def test_invalid_method_raises():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(ValueError):
        rosenblatt_gof_test(x, x, CopulaGaussian, method="bogus")


# ---------------------------------------------------------------------------
# Small size / power studies — qualitative, not a precise level/power claim.
# ---------------------------------------------------------------------------

def _rejection_rate(true_cls, tau, fit_cls, n, n_reps, B, alpha, seed0):
    cop = true_cls(tau_k=tau)
    n_reject = 0
    n_ok = 0
    for r in range(n_reps):
        uv = cop.sample(n, seed=seed0 * 1000 + r)
        res = rosenblatt_gof_test(uv[:, 0], uv[:, 1], fit_cls, B=B,
                                   seed=seed0 * 1000 + r + 1, alpha=alpha)
        if np.isfinite(res.p_value):
            n_ok += 1
            n_reject += int(res.reject)
    return n_reject / n_ok if n_ok else float("nan")


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.5),
    (CopulaClayton, 0.5),
    (CopulaFrank, 0.5),
])
def test_size_study_true_family_near_nominal(family_cls, tau):
    # Size should be close to nominal regardless of whether the true family
    # is radially symmetric (Gauss, Frank) or not (Clayton) — unlike the
    # radial-symmetry test, this one is not targeted at symmetry.
    rate = _rejection_rate(family_cls, tau, family_cls, n=150, n_reps=60, B=60,
                            alpha=0.10, seed0=_stable_seed(family_cls.__name__, tau))
    # 60 replicates at nominal 10 %: binomial sd ~= 0.039, so a generous band.
    assert rate < 0.35, f"{family_cls.__name__} tau={tau}: over-rejected at {rate:.3f}"


@pytest.mark.parametrize("true_cls,tau,fit_cls,n", [
    (CopulaClayton, 0.6, CopulaGaussian, 200),
    (CopulaGH, 0.6, CopulaGaussian, 300),
])
def test_power_study_wrong_family_rejects_more_often(true_cls, tau, fit_cls, n):
    rate = _rejection_rate(true_cls, tau, fit_cls, n=n, n_reps=60, B=60,
                            alpha=0.10,
                            seed0=_stable_seed(true_cls.__name__, tau, fit_cls.__name__))
    assert rate > 0.30, (
        f"true={true_cls.__name__} tau={tau} fit={fit_cls.__name__} n={n}: "
        f"power too low at {rate:.3f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.3), (CopulaGaussian, 0.6),
    (CopulaClayton, 0.3), (CopulaClayton, 0.6),
    (CopulaFrank, 0.3), (CopulaFrank, 0.6),
])
def test_full_size_study(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, family_cls, n=200, n_reps=200, B=150,
                            alpha=0.05,
                            seed0=_stable_seed(family_cls.__name__, tau, "full-size"))
    assert 0.0 <= rate <= 0.15, f"{family_cls.__name__} tau={tau}: rate={rate:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("true_cls,tau,fit_cls,n", [
    (CopulaClayton, 0.2, CopulaGaussian, 100), (CopulaClayton, 0.4, CopulaGaussian, 100),
    (CopulaClayton, 0.6, CopulaGaussian, 100),
    (CopulaClayton, 0.2, CopulaGaussian, 300), (CopulaClayton, 0.4, CopulaGaussian, 300),
    (CopulaClayton, 0.6, CopulaGaussian, 300),
    (CopulaGH, 0.2, CopulaGaussian, 100), (CopulaGH, 0.4, CopulaGaussian, 100),
    (CopulaGH, 0.6, CopulaGaussian, 100),
])
def test_full_power_study(true_cls, tau, fit_cls, n):
    rate = _rejection_rate(true_cls, tau, fit_cls, n=n, n_reps=200, B=150, alpha=0.05,
                            seed0=_stable_seed(true_cls.__name__, tau, fit_cls.__name__, n, "full"))
    assert 0.0 <= rate <= 1.0, (
        f"true={true_cls.__name__} tau={tau} fit={fit_cls.__name__} N={n}: rate={rate:.3f}"
    )
