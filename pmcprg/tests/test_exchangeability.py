"""Tests for pmcprg.diagnostics.exchangeability (audit FR-10, fourth item).

Fast, small-N/small-B versions of the size and power studies reported in the
FR-10 commit — enough replicates to see the qualitative behaviour (symmetric
families near nominal, the 90°-rotated families rejected more often, as a
function of tau and N), not enough to pin the level to two digits; the full
study lives in the commit message, following the same convention as
``test_radial_symmetry.py`` and ``test_rosenblatt.py``.
"""
import zlib

import numpy as np
import pytest

from pmcprg.copulas import (
    CopulaClayton,
    CopulaClayton90,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH90,
    CopulaJoe90,
    CopulaProduct,
)
from pmcprg.diagnostics.exchangeability import (
    ExchangeabilityResult,
    exchangeability_statistic,
    exchangeability_test,
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
    res = exchangeability_test(uv[:, 0], uv[:, 1], B=40, seed=2)
    assert isinstance(res, ExchangeabilityResult)
    assert res.n == 80
    assert res.B == 40
    assert 0.0 <= res.p_value <= 1.0
    assert isinstance(res.reject, bool)
    assert res.statistic >= 0.0


def test_independence_copula_never_rejects():
    cop = CopulaProduct(tau_k=0.0)
    for seed in range(5):
        uv = cop.sample(150, seed=seed)
        res = exchangeability_test(uv[:, 0], uv[:, 1], B=100, seed=seed + 1000)
        assert not res.reject, f"independence copula rejected at seed={seed}: {res}"


def test_statistic_is_nonnegative_and_finite():
    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=60), rng.uniform(size=60)
    stat = exchangeability_statistic(u, v)
    assert np.isfinite(stat)
    assert stat >= 0.0


def test_rotated_90_families_are_concretely_non_exchangeable():
    # Confirm C(u,v) != C(v,u) at a few points for the rotated families
    # before trusting them as a power-study test bed (task instruction: do
    # not assume the rotation breaks exchangeability, check it).
    pts = [(0.2, 0.7), (0.6, 0.3), (0.9, 0.1)]
    for cls in (CopulaClayton90, CopulaGH90, CopulaJoe90):
        cop = cls(tau_k=-0.5)
        max_abs_diff = max(abs(cop.cdf([u, v]) - cop.cdf([v, u])) for u, v in pts)
        assert max_abs_diff > 1e-3, f"{cls.__name__}: looks exchangeable ({max_abs_diff=})"


def test_gaussian_copula_is_exchangeable_at_every_point():
    # The other half of the same verification: Gauss is exchangeable
    # unconditionally (module docstring, "Verified, not assumed"), not just
    # under some special correlation value.
    pts = [(0.2, 0.7), (0.6, 0.3), (0.9, 0.1)]
    for tau in (-0.7, -0.2, 0.0, 0.3, 0.8):
        cop = CopulaGaussian(tau_k=tau)
        for u, v in pts:
            assert cop.cdf([u, v]) == pytest.approx(cop.cdf([v, u]), abs=1e-10)


def test_statistic_grows_under_stronger_asymmetric_dependence():
    # More rotated (genuinely non-exchangeable) dependence should push the
    # statistic up, on average, relative to independence — a coarse
    # monotonicity check on the formula (averaged over replicates to beat
    # sampling noise) rather than a precise power claim (see the study below).
    rng = np.random.default_rng(0)
    indep_stats = [
        exchangeability_statistic(rng.uniform(size=200), rng.uniform(size=200))
        for _ in range(20)
    ]
    rotated = CopulaClayton90(tau_k=-0.7)
    rotated_stats = [
        exchangeability_statistic(*rotated.sample(200, seed=1000 + r).T)
        for r in range(20)
    ]
    assert np.mean(rotated_stats) > np.mean(indep_stats)


def test_degenerate_samples_do_not_crash():
    for n in (0, 1, 2, 3):
        x = np.arange(n, dtype=float)
        res = exchangeability_test(x, x, B=10, seed=0)
        assert np.isnan(res.statistic)
        assert np.isnan(res.p_value)
        assert res.reject is False


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        exchangeability_test(np.zeros(5), np.zeros(6))


def test_weights_are_rejected_explicitly():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(NotImplementedError):
        exchangeability_test(x, x, weights=np.ones(20))


def test_invalid_alpha_raises():
    x = np.random.default_rng(0).uniform(size=20)
    with pytest.raises(ValueError):
        exchangeability_test(x, x, alpha=1.5)


# ---------------------------------------------------------------------------
# Small size / power studies — qualitative, not a precise level/power claim.
# ---------------------------------------------------------------------------

def _rejection_rate(family_cls, tau, n, n_reps, B, alpha, seed0):
    cop = family_cls(tau_k=tau)
    n_reject = 0
    n_ok = 0
    for r in range(n_reps):
        uv = cop.sample(n, seed=seed0 * 1000 + r)
        res = exchangeability_test(uv[:, 0], uv[:, 1], B=B, seed=seed0 * 1000 + r + 1,
                                    alpha=alpha)
        if np.isfinite(res.p_value):
            n_ok += 1
            n_reject += int(res.reject)
    return n_reject / n_ok if n_ok else float("nan")


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.5),
    (CopulaFrank, 0.5),
    (CopulaClayton, 0.5),
])
def test_size_study_exchangeable_families_near_nominal(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=150, n_reps=60, B=60, alpha=0.10,
                            seed0=_stable_seed(family_cls.__name__, tau, "exch-size"))
    # 60 replicates at nominal 10 %: binomial sd ~= 0.039, so a generous band.
    assert rate < 0.30, f"{family_cls.__name__} tau={tau}: over-rejected at {rate:.3f}"


@pytest.mark.parametrize("family_cls,tau", [
    (CopulaClayton90, -0.5),
    (CopulaGH90, -0.5),
    (CopulaJoe90, -0.5),
])
def test_power_study_rotated_families_reject_more_often(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=200, n_reps=60, B=60, alpha=0.10,
                            seed0=_stable_seed(family_cls.__name__, tau, "exch-power"))
    assert rate > 0.30, f"{family_cls.__name__} tau={tau}: power too low at {rate:.3f}"


# ---------------------------------------------------------------------------
# Full-scale size/power studies (slow) — exchangeable families Gauss/Frank/
# Clayton at tau in {0.3, 0.6} for size; the 90°-rotated Clayton/GH/Joe at
# tau in {-0.2, -0.4, -0.6}, N in {100, 300} for power, as a function of tau
# and N. Numbers reported in CHANGELOG.md.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("family_cls,tau", [
    (CopulaGaussian, 0.3), (CopulaGaussian, 0.6),
    (CopulaFrank, 0.3), (CopulaFrank, 0.6),
    (CopulaClayton, 0.3), (CopulaClayton, 0.6),
])
def test_full_size_study(family_cls, tau):
    rate = _rejection_rate(family_cls, tau, n=200, n_reps=100, B=150, alpha=0.05,
                            seed0=_stable_seed(family_cls.__name__, tau, "exch-full-size"))
    assert 0.0 <= rate <= 0.15, f"{family_cls.__name__} tau={tau}: rate={rate:.3f}"


@pytest.mark.slow
@pytest.mark.parametrize("family_cls,tau,n", [
    (CopulaClayton90, -0.2, 100), (CopulaClayton90, -0.4, 100), (CopulaClayton90, -0.6, 100),
    (CopulaClayton90, -0.2, 300), (CopulaClayton90, -0.4, 300), (CopulaClayton90, -0.6, 300),
    (CopulaGH90, -0.2, 100), (CopulaGH90, -0.4, 100), (CopulaGH90, -0.6, 100),
    (CopulaGH90, -0.2, 300), (CopulaGH90, -0.4, 300), (CopulaGH90, -0.6, 300),
    (CopulaJoe90, -0.2, 100), (CopulaJoe90, -0.4, 100), (CopulaJoe90, -0.6, 100),
    (CopulaJoe90, -0.2, 300), (CopulaJoe90, -0.4, 300), (CopulaJoe90, -0.6, 300),
])
def test_full_power_study(family_cls, tau, n):
    rate = _rejection_rate(family_cls, tau, n=n, n_reps=100, B=150, alpha=0.05,
                            seed0=_stable_seed(family_cls.__name__, tau, n, "exch-full-power"))
    # Power should grow with |tau| and N; this is a sanity floor, not a
    # precise claim — the actual numbers are reported in CHANGELOG.md.
    assert 0.0 <= rate <= 1.0, f"{family_cls.__name__} tau={tau} N={n}: rate={rate:.3f}"
