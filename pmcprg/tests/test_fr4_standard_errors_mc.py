"""Monte-Carlo calibration of the FR-4 standard errors and LR test (slow).

For each family, R samples of size n are drawn from a known copula with fixed
seeds, fitted with ``fit(method=...)``, and :meth:`FitResult.standard_errors`
is computed on every fit. Two things are compared with their targets:

* ``ratio`` — root-mean-square reported SE over the empirical SD of the
  estimates. The SD over R replicates has a relative standard error
  1/√(2(R − 1)) (normal approximation): 3.5 % at R = 400, 5.0 % at R = 200.
  The bands are ±3 of those, plus a few percent for the O(1/n) gap between an
  asymptotic formula and its finite-sample target (≤ 5 % in every
  measurement made while writing these tests).
* ``coverage`` of the 95 % Wald interval — binomial SE √(0.95·0.05/R):
  1.1 % at R = 400, 1.5 % at R = 200; bands ±3 SE.

Student's ν̂ is right-skewed at n = 1000 (measured SD ratio 1.07, coverage
0.90 over 300 replicates at ν = 5): its bands are wider and stated there.

The LR test of independence is checked for its level under H₀ on rank
pseudo-observations: ½χ²₀ + ½χ²₁ for Clayton (independence on the boundary;
about half of the statistics are 0), χ²₁ for Frank (interior).

Measured with the seeds below (ratio / coverage, τ first):
Gauss 1.017/0.935 (mle), 1.012/0.945 (tau); Clayton 0.990/0.950, 0.994/0.932;
Gumbel 1.018/0.950, 1.024/0.953; Frank 1.035/0.935, 1.030/0.938; Plackett
1.030/0.948 (mle); BB1 τ 1.098/0.950, δ 0.985/0.960, θ 0.983/0.950; Student
τ 0.935/0.935, ν 1.018/0.945, ρ 0.933/0.935.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import rankdata

from pmcprg.copulas import (
    CopulaBB1,
    CopulaClayton,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH,
    CopulaPlackett,
    CopulaStudent,
    independence_lr_test,
)

pytestmark = pytest.mark.slow


def _plackett_sample(cop, n: int, seed: int) -> np.ndarray:
    """Plackett sample by the closed-form root of h(v | u) = t.

    ``CopulaPlackett.sample`` inverts h by Brent's method point by point
    (≈ 1 ms per point); this is the same inverse in closed form, checked
    against the package's h-function in :func:`test_plackett_sampler_inverts_h`.
    """
    th = cop.theta
    rng = np.random.default_rng(seed)
    u, t = rng.uniform(size=n), rng.uniform(size=n)
    a = t * (1.0 - t)
    b = th + a * (th - 1.0) ** 2
    c = 2.0 * a * (u * th * th + 1.0 - u) + th * (1.0 - 2.0 * a)
    d = math.sqrt(th) * np.sqrt(th + 4.0 * a * u * (1.0 - u) * (th - 1.0) ** 2)
    return np.column_stack([u, (c - (1.0 - 2.0 * t) * d) / (2.0 * b)])


def test_plackett_sampler_inverts_h():
    cop = CopulaPlackett(tau_k=0.4)
    uv = _plackett_sample(cop, 200, seed=0)
    rng = np.random.default_rng(0)
    rng.uniform(size=200)                       # the u draws
    t = rng.uniform(size=200)
    h = np.array([cop.conditional_cdf(float(v), float(u)) for u, v in uv])
    np.testing.assert_allclose(h, t, atol=1e-12)


def _monte_carlo(cls, params: dict, method: str, n: int, R: int, seed0: int):
    truth_cop = cls(**params)
    sample = ((lambda seed: _plackett_sample(truth_cop, n, seed)) if cls is CopulaPlackett
              else (lambda seed: cls(**params).sample(n, seed=seed)))
    truth = {"tau_k": params["tau_k"], "theta": getattr(truth_cop, "theta", math.nan),
             "rho": getattr(truth_cop, "theta", math.nan),
             "df": getattr(truth_cop, "df", math.nan), "delta": getattr(truth_cop, "delta", math.nan)}
    est, se, cover, boundary = {}, {}, {}, 0
    for r in range(R):
        fit = cls.fit(sample(seed0 + r), method=method)
        s = fit.standard_errors()
        boundary += s.at_boundary
        for name in s.names:
            est.setdefault(name, []).append(s.estimate[name])
            se.setdefault(name, []).append(s.se[name])
            lo, hi = s.ci(0.95, name)
            cover.setdefault(name, []).append(lo <= truth[name] <= hi)
    out = {}
    for name in est:
        e, s = np.asarray(est[name]), np.asarray(se[name])
        out[name] = (math.sqrt(np.mean(s * s)) / np.std(e, ddof=1), float(np.mean(cover[name])))
    return out, boundary


ONE_PARAMETER = [
    (CopulaGaussian, "mle"), (CopulaGaussian, "tau"),
    (CopulaClayton, "mle"), (CopulaClayton, "tau"),
    (CopulaGH, "mle"), (CopulaGH, "tau"),
    (CopulaFrank, "mle"), (CopulaFrank, "tau"),
    (CopulaPlackett, "mle"),
]


@pytest.mark.parametrize("cls, method", ONE_PARAMETER,
                         ids=[f"{c.__name__}-{m}" for c, m in ONE_PARAMETER])
def test_one_parameter_sd_ratio_and_coverage(cls, method):
    """n = R = 400 at τ = 0.4: ratio ∈ [0.85, 1.15], coverage ∈ [0.917, 0.983]."""
    res, boundary = _monte_carlo(cls, {"tau_k": 0.4}, method, n=400, R=400, seed0=1000)
    assert boundary == 0
    for name, (ratio, coverage) in res.items():
        assert 0.85 <= ratio <= 1.15, (name, ratio)
        assert 0.917 <= coverage <= 0.983, (name, coverage)


def test_bb1_sd_ratio_and_coverage():
    """n = 1000, R = 200 at (τ, δ) = (0.5, 1.5), θ = 2/3: ratio ∈ [0.82, 1.20],
    coverage ∈ [0.905, 0.995] for τ, δ and θ."""
    res, boundary = _monte_carlo(CopulaBB1, {"tau_k": 0.5, "delta": 1.5}, "mle",
                                 n=1000, R=200, seed0=2000)
    assert boundary == 0
    for name, (ratio, coverage) in res.items():
        assert 0.82 <= ratio <= 1.20, (name, ratio)
        assert 0.905 <= coverage <= 0.995, (name, coverage)


def test_student_sd_ratio_and_coverage():
    """n = 1000, R = 200 at (τ, ν) = (0.5, 5).

    τ and ρ: ratio ∈ [0.82, 1.20], coverage ∈ [0.905, 0.995]. ν: the
    estimator is right-skewed at this n and the Wald interval symmetric, so
    under-coverage of a few percent is the expected finite-sample behaviour,
    not a defect of the SE: ratio ∈ [0.8, 1.35], coverage ∈ [0.85, 0.99].
    """
    res, boundary = _monte_carlo(CopulaStudent, {"tau_k": 0.5, "df": 5.0}, "mle",
                                 n=1000, R=200, seed0=3000)
    assert boundary <= 2          # ν̂ may reach its box bound 100 on a rare sample
    for name in ("tau_k", "rho"):
        ratio, coverage = res[name]
        assert 0.82 <= ratio <= 1.20, (name, ratio)
        assert 0.905 <= coverage <= 0.995, (name, coverage)
    ratio, coverage = res["df"]
    assert 0.8 <= ratio <= 1.35, ratio
    assert 0.85 <= coverage <= 0.99, coverage


@pytest.mark.parametrize("family, boundary", [(CopulaClayton, True), (CopulaFrank, False)])
def test_independence_lr_test_level(family, boundary):
    """R = 1000 independent samples, n = 200, rank pseudo-observations.

    Rejection rate at 5 %: binomial SE 0.69 % → band [0.029, 0.071]. For the
    boundary family, the share of zero statistics estimates the χ²₀ weight ½
    (SE 1.6 %); the band [0.45, 0.60] allows for the τ-pad of the optimiser,
    which turns a few tiny positive estimates into LR ≈ 0 (measured: 0.53).
    """
    n, R = 200, 1000
    p, zero = [], 0
    for r in range(R):
        x = np.random.default_rng(70_000 + r).uniform(size=(n, 2))
        uv = np.column_stack([rankdata(x[:, 0]) / (n + 1), rankdata(x[:, 1]) / (n + 1)])
        t = independence_lr_test(family, uv)
        assert t.boundary == boundary
        p.append(t.p_value)
        zero += t.statistic < 1e-3
    assert 0.029 <= np.mean(np.asarray(p) <= 0.05) <= 0.071
    if boundary:
        assert 0.45 <= zero / R <= 0.60
    else:
        assert zero / R <= 0.06
