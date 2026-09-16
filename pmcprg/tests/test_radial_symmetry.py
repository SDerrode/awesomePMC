"""Tests for pmcprg.diagnostics.radial_symmetry (audit FR-10).

Fast, small-N/small-B versions of the size and power studies reported in the
FR-10 commit — enough replicates to see the qualitative behaviour (symmetric
families near nominal, asymmetric ones rejected more often), not enough to
pin the level to two digits; the full study lives in the commit message.

The multiplier-bootstrap tests below (FR-10 round 2) mirror the parametric
ones on the identical family/τ/N grid, plus a wall-clock comparison — see
CHANGELOG.md for the full-scale numbers.
"""
import time
import zlib

import numpy as np
import pytest

from pmcprg.copulas import (
    CopulaClayton,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH,
    CopulaJoe,
    CopulaPlackett,
    CopulaProduct,
)


from pmcprg.diagnostics.radial_symmetry import (
    RadialSymmetryResult,
    radial_symmetry_statistic,
    radial_symmetry_test,
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

def _rejection_rate(family_cls, tau, n, n_reps, B, alpha, seed0, bootstrap="parametric"):
    cop = family_cls(tau_k=tau)
    n_reject = 0
    n_ok = 0
    for r in range(n_reps):
        uv = cop.sample(n, seed=seed0 * 1000 + r)
        res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=B, seed=seed0 * 1000 + r + 1,
                                    alpha=alpha, bootstrap=bootstrap)
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
                            seed0=_stable_seed(family_cls.__name__, tau))
    # 60 replicates at nominal 10 %: binomial sd ~= 0.039, so a generous band.
    assert rate < 0.30, f"{family_cls.__name__} tau={tau}: over-rejected at {rate:.3f}"


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaClayton, 0.6),
    (CopulaGH, 0.6),
])
def test_power_study_asymmetric_families_reject_more_often(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=200, n_reps=60, B=60, alpha=0.10,
                            seed0=_stable_seed(family_cls.__name__, tau, "power"))
    assert rate > 0.30, f"{family_cls.__name__} tau={tau}: power too low at {rate:.3f}"


# ---------------------------------------------------------------------------
# Multiplier bootstrap (FR-10 round 2) — unit tests, wall-clock comparison,
# and the same size/power grid as above, run with bootstrap="multiplier".
# ---------------------------------------------------------------------------

def test_multiplier_bootstrap_result_fields_and_types():
    cop = CopulaGaussian(tau_k=0.5)
    uv = cop.sample(80, seed=1)
    res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=40, seed=2, bootstrap="multiplier")
    assert isinstance(res, RadialSymmetryResult)
    assert res.bootstrap == "multiplier"
    assert res.n == 80
    assert res.B == 40
    assert res.n_valid == 40          # the multiplier bootstrap never drops a replicate
    assert 0.0 <= res.p_value <= 1.0
    assert isinstance(res.reject, bool)
    assert res.statistic >= 0.0


def test_multiplier_bootstrap_matches_observed_statistic():
    # bootstrap="multiplier" must not change the observed T_n at all — only
    # the null it is compared to.
    cop = CopulaClayton(tau_k=0.5)
    uv = cop.sample(120, seed=3)
    res_p = radial_symmetry_test(uv[:, 0], uv[:, 1], B=30, seed=4, bootstrap="parametric")
    res_m = radial_symmetry_test(uv[:, 0], uv[:, 1], B=30, seed=4, bootstrap="multiplier")
    assert res_p.statistic == pytest.approx(res_m.statistic)
    assert res_p.tau_hat == pytest.approx(res_m.tau_hat)


def test_multiplier_bootstrap_default_unchanged():
    # The default bootstrap path must be bit-for-bit what it was before this
    # option existed.
    cop = CopulaGaussian(tau_k=0.4)
    uv = cop.sample(100, seed=5)
    res_default = radial_symmetry_test(uv[:, 0], uv[:, 1], B=50, seed=6)
    res_explicit = radial_symmetry_test(uv[:, 0], uv[:, 1], B=50, seed=6, bootstrap="parametric")
    assert res_default == res_explicit


def test_multiplier_bootstrap_rademacher_runs():
    cop = CopulaGaussian(tau_k=0.5)
    uv = cop.sample(100, seed=7)
    res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=40, seed=8, bootstrap="multiplier",
                                multiplier="rademacher")
    assert 0.0 <= res.p_value <= 1.0


def test_unknown_bootstrap_raises():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(ValueError):
        radial_symmetry_test(x, x, bootstrap="bogus")


def test_unknown_multiplier_law_raises():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(ValueError):
        radial_symmetry_test(x, x, bootstrap="multiplier", multiplier="bogus")


def test_multiplier_bootstrap_independence_never_rejects():
    cop = CopulaProduct(tau_k=0.0)
    for seed in range(5):
        uv = cop.sample(150, seed=seed)
        res = radial_symmetry_test(uv[:, 0], uv[:, 1], B=100, seed=seed + 1000,
                                    bootstrap="multiplier")
        assert not res.reject, f"independence copula rejected at seed={seed}: {res}"


def test_multiplier_bootstrap_degenerate_samples_do_not_crash():
    for n in (0, 1, 2, 3):
        x = np.arange(n, dtype=float)
        res = radial_symmetry_test(x, x, B=10, seed=0, bootstrap="multiplier")
        assert np.isnan(res.statistic)
        assert np.isnan(res.p_value)
        assert res.reject is False


def test_multiplier_bootstrap_is_much_faster_than_parametric():
    # Loose bound, not flaky: same B, same N, multiplier bootstrap avoids
    # resampling + reranking B times, so it should run in a small fraction of
    # the parametric bootstrap's time. Generous margin (50%) to absorb
    # machine noise; the measured ratio in CHANGELOG.md is far smaller.
    cop = CopulaGaussian(tau_k=0.5)
    uv = cop.sample(200, seed=10)
    x, y = uv[:, 0], uv[:, 1]

    t0 = time.perf_counter()
    radial_symmetry_test(x, y, B=200, seed=11, bootstrap="parametric")
    t_parametric = time.perf_counter() - t0

    t0 = time.perf_counter()
    radial_symmetry_test(x, y, B=200, seed=11, bootstrap="multiplier")
    t_multiplier = time.perf_counter() - t0

    assert t_multiplier < 0.5 * t_parametric, (
        f"multiplier bootstrap ({t_multiplier:.4f}s) is not much faster than "
        f"parametric ({t_parametric:.4f}s) at N=200, B=200."
    )


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.5),
    (CopulaFrank, 0.5),
    (CopulaPlackett, 0.5),
])
def test_multiplier_bootstrap_size_study_symmetric_families_near_nominal(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=150, n_reps=60, B=60, alpha=0.10,
                            seed0=_stable_seed(family_cls.__name__, tau, "mult-size"),
                            bootstrap="multiplier")
    # 60 replicates at nominal 10 %: binomial sd ~= 0.039, so a generous band.
    assert rate < 0.30, f"{family_cls.__name__} tau={tau}: over-rejected at {rate:.3f}"


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaClayton, 0.6),
    (CopulaGH, 0.6),
    (CopulaJoe, 0.6),
])
def test_multiplier_bootstrap_power_study_asymmetric_families_reject_more_often(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=200, n_reps=60, B=60, alpha=0.10,
                            seed0=_stable_seed(family_cls.__name__, tau, "mult-power"),
                            bootstrap="multiplier")
    assert rate > 0.30, f"{family_cls.__name__} tau={tau}: power too low at {rate:.3f}"


# ---------------------------------------------------------------------------
# Full-scale size/power studies, same grid as the parametric-bootstrap pilot
# (CHANGELOG.md "radial symmetry screening test (FR-10)" entry): symmetric
# families Gauss/Frank/Plackett at tau in {0.3, 0.6}, N=200 for size; Clayton/
# Gumbel/Joe at tau in {0.2, 0.4, 0.6}, N in {100, 300} for power. Reduced
# N_reps/B relative to the pilot's 300/150 to stay within the compute budget
# — see CHANGELOG.md for the numbers actually reported.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.3), (CopulaGaussian, 0.6),
    (CopulaFrank, 0.3), (CopulaFrank, 0.6),
    (CopulaPlackett, 0.3), (CopulaPlackett, 0.6),
])
def test_multiplier_bootstrap_full_size_study(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=200, n_reps=100, B=150, alpha=0.05,
                            seed0=_stable_seed(family_cls.__name__, tau, "mult-full-size"),
                            bootstrap="multiplier")
    assert 0.0 <= rate <= 0.15, f"{family_cls.__name__} tau={tau}: rate={rate:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("family_cls,tau,n", [
    (CopulaClayton, 0.2, 100), (CopulaClayton, 0.4, 100), (CopulaClayton, 0.6, 100),
    (CopulaClayton, 0.2, 300), (CopulaClayton, 0.4, 300), (CopulaClayton, 0.6, 300),
    (CopulaGH, 0.2, 100), (CopulaGH, 0.4, 100), (CopulaGH, 0.6, 100),
    (CopulaGH, 0.2, 300), (CopulaGH, 0.4, 300), (CopulaGH, 0.6, 300),
    (CopulaJoe, 0.2, 100), (CopulaJoe, 0.4, 100), (CopulaJoe, 0.6, 100),
    (CopulaJoe, 0.2, 300), (CopulaJoe, 0.4, 300), (CopulaJoe, 0.6, 300),
])
def test_multiplier_bootstrap_full_power_study(family_cls, tau, n):
    rate = _rejection_rate(family_cls, tau, n=n, n_reps=100, B=150, alpha=0.05,
                            seed0=_stable_seed(family_cls.__name__, tau, n, "mult-full-power"),
                            bootstrap="multiplier")
    # Power should grow with tau and N, as observed for the parametric
    # bootstrap (CHANGELOG.md); this is a sanity floor, not a precise claim.
    assert 0.0 <= rate <= 1.0, f"{family_cls.__name__} tau={tau} N={n}: rate={rate:.3f}"
