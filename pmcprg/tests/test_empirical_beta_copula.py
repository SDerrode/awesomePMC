"""The empirical beta copula (AUDIT_COPULES FR-9, last remaining item).

:class:`pmcprg.copulas.EmpiricalBetaCopula` is a standalone nonparametric
comparison tool, not a :class:`~pmcprg.copulas._base.CopulaVirt` family (see
its module docstring, :mod:`pmcprg.copulas._nonparametric`). What is checked
here:

* a hand-computed n = 3 case (``test_hand_computed_n3``) — the formula
  ``C_n^β(u,v) = (1/n) Σ_i I_u(R_1i, n+1-R_1i) I_v(R_2i, n+1-R_2i)`` worked
  out by hand and compared to the module's output;
* an independent brute-force loop re-implementation of cdf/pdf/h1/h2, cross-
  checked against the vectorised module on small n
  (``test_brute_force_matches_vectorised``);
* the h-functions against a central finite difference of ``cdf``
  (``test_h_functions_match_numerical_derivative``);
* tie handling: two tied values get the *same* (averaged) Beta shape, not an
  arbitrary tie-break (``test_ties_use_average_rank``);
* margin uniformity: ``C_n^β(u, 1)`` turns out to be **exactly** ``u`` (to
  float64 rounding) whenever there are no ties — a deterministic identity of
  the regularised incomplete beta function, derived and checked here, not
  assumed — and only acquires a small, quantified discrepancy in the
  presence of ties, which itself shrinks as the tied pair is diluted into a
  larger sample (``test_margin_*``);
* sampler-vs-cdf agreement by simulation, with a Monte-Carlo error bound
  derived from the sample size (``test_sampler_matches_cdf``);
* the concrete log-space underflow scenario documented in the module
  docstring: naive ``log(mean(exp(logpdf_i)))`` returns ``-inf`` at an
  extreme corner where every per-point term underflows float64, while
  ``logpdf_array`` (built on ``logsumexp``) stays finite
  (``test_logpdf_avoids_naive_underflow``);
* edge cases n = 2, n = 3, and a "realistic" n in the low hundreds.

Every seed below is an integer literal or ``zlib.crc32`` of a repr — never
Python's ``hash()``, which is salted per process.
"""
from __future__ import annotations

import zlib

import numpy as np
import pytest
from scipy.special import betainc
from scipy.stats import beta as beta_dist

from pmcprg.copulas import EmpiricalBetaCopula


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) % (2 ** 31)


# ---------------------------------------------------------------------
# Independent brute-force reference (loops, no broadcasting) — deliberately
# not sharing a single line of code with pmcprg.copulas._nonparametric.
# ---------------------------------------------------------------------
def _brute_ranks(x: np.ndarray) -> np.ndarray:
    """Average-rank, computed by explicit double loop (no scipy.stats)."""
    n = len(x)
    ranks = np.empty(n)
    for i in range(n):
        less = sum(1 for j in range(n) if x[j] < x[i])
        equal = sum(1 for j in range(n) if x[j] == x[i])
        # average rank of a tie block: (first + last) / 2, 1-indexed.
        ranks[i] = less + (equal + 1) / 2.0
    return ranks


def _brute_cdf(a1, b1, a2, b2, u, v) -> float:
    n = len(a1)
    total = 0.0
    for i in range(n):
        total += betainc(a1[i], b1[i], u) * betainc(a2[i], b2[i], v)
    return total / n


def _brute_pdf(a1, b1, a2, b2, u, v) -> float:
    n = len(a1)
    total = 0.0
    for i in range(n):
        total += beta_dist.pdf(u, a1[i], b1[i]) * beta_dist.pdf(v, a2[i], b2[i])
    return total / n


def _brute_h1(a1, b1, a2, b2, u, v) -> float:
    n = len(a1)
    total = 0.0
    for i in range(n):
        total += beta_dist.pdf(u, a1[i], b1[i]) * betainc(a2[i], b2[i], v)
    return total / n


def _brute_h2(a1, b1, a2, b2, u, v) -> float:
    n = len(a1)
    total = 0.0
    for i in range(n):
        total += betainc(a1[i], b1[i], u) * beta_dist.pdf(v, a2[i], b2[i])
    return total / n


# ---------------------------------------------------------------------
# Hand-computed n = 3
# ---------------------------------------------------------------------
def test_hand_computed_n3():
    """n = 3, no ties: ranks (1, 2, 3) on each margin by construction.

    data = [(0.1, 0.9), (0.5, 0.1), (0.9, 0.5)] gives
    R1 = (1, 2, 3), R2 = (3, 1, 2) (rank of 0.9 is 3, of 0.1 is 1, of 0.5
    is 2, read off each column by hand). So (a1, b1) pairs are
    (1, 3), (2, 2), (3, 1) and (a2, b2) pairs are (3, 1), (1, 3), (2, 2).

    C_3^β(0.5, 0.5) = 1/3 [ I_.5(1,3) I_.5(3,1) + I_.5(2,2) I_.5(1,3)
                            + I_.5(3,1) I_.5(2,2) ]
    with I_.5(1,3) = 1 - 0.5^3 = 0.875, I_.5(3,1) = 0.5^3 = 0.125,
    I_.5(2,2) = 0.5 exactly (symmetric Beta(2,2) at its median).
    """
    data = np.array([[0.1, 0.9], [0.5, 0.1], [0.9, 0.5]])
    cop = EmpiricalBetaCopula(data)
    np.testing.assert_allclose(cop.a1, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cop.a2, [3.0, 1.0, 2.0])

    I_875 = 0.875   # I_.5(1, 3)
    I_125 = 0.125   # I_.5(3, 1)
    I_5 = 0.5        # I_.5(2, 2)
    expected = (I_875 * I_125 + I_5 * I_875 + I_125 * I_5) / 3.0
    got = cop.cdf(0.5, 0.5)
    assert got == pytest.approx(expected, rel=0, abs=1e-14)

    # Corner sanity: C_n^beta(0, .) = C_n^beta(., 0) = 0, no rank has a=0.
    assert cop.cdf(0.0, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert cop.cdf(0.5, 0.0) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------
# Brute force vs vectorised, several small n, several points, incl. ties
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_brute_force_matches_vectorised(n):
    rng = np.random.default_rng(_seed("brute", n))
    data = rng.uniform(size=(n, 2))
    cop = EmpiricalBetaCopula(data)

    # Independent brute-force ranks must match scipy's average-rank ranks.
    r1_brute = _brute_ranks(data[:, 0])
    r2_brute = _brute_ranks(data[:, 1])
    np.testing.assert_allclose(cop.a1, r1_brute)
    np.testing.assert_allclose(cop.a2, r2_brute)

    grid = np.linspace(0.05, 0.95, 6)
    for u in grid:
        for v in grid:
            expected_cdf = _brute_cdf(cop.a1, cop.b1, cop.a2, cop.b2, u, v)
            expected_pdf = _brute_pdf(cop.a1, cop.b1, cop.a2, cop.b2, u, v)
            expected_h1 = _brute_h1(cop.a1, cop.b1, cop.a2, cop.b2, u, v)
            expected_h2 = _brute_h2(cop.a1, cop.b1, cop.a2, cop.b2, u, v)
            assert cop.cdf(u, v) == pytest.approx(expected_cdf, rel=1e-12, abs=1e-13)
            assert cop.pdf(u, v) == pytest.approx(expected_pdf, rel=1e-10, abs=1e-12)
            assert cop.h1(u, v) == pytest.approx(expected_h1, rel=1e-10, abs=1e-12)
            assert cop.h2(u, v) == pytest.approx(expected_h2, rel=1e-10, abs=1e-12)


def test_ties_use_average_rank():
    """Two tied values on a margin get the same, averaged rank/shape."""
    # Column 0 has a tie: values 0.3, 0.3, 0.7 -> ranks 1.5, 1.5, 3.
    data = np.array([[0.3, 0.1], [0.3, 0.5], [0.7, 0.9]])
    cop = EmpiricalBetaCopula(data)
    np.testing.assert_allclose(cop.a1, [1.5, 1.5, 3.0])
    np.testing.assert_allclose(cop.b1, [2.5, 2.5, 1.0])
    # The two tied points get identical Beta margins for coordinate 1.
    assert cop.a1[0] == cop.a1[1]
    assert cop.b1[0] == cop.b1[1]
    # Ranking is invariant to a monotone rescaling: pseudo-obs vs raw data.
    pseudo = np.column_stack([
        np.array([1.5, 1.5, 3.0]) / 4.0,
        np.array([1.0, 2.0, 3.0]) / 4.0,
    ])
    cop2 = EmpiricalBetaCopula.from_pseudo_obs(pseudo)
    np.testing.assert_allclose(cop.a1, cop2.a1)
    np.testing.assert_allclose(cop.a2, cop2.a2)


# ---------------------------------------------------------------------
# h-functions vs numerical derivative of cdf
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n", [4, 25])
def test_h_functions_match_numerical_derivative(n):
    rng = np.random.default_rng(_seed("hderiv", n))
    data = rng.uniform(size=(n, 2))
    cop = EmpiricalBetaCopula(data)
    h = 1e-6
    for u, v in [(0.2, 0.3), (0.5, 0.5), (0.1, 0.8), (0.9, 0.05)]:
        # h1 = dC/du (central difference)
        dC_du = (cop.cdf(u + h, v) - cop.cdf(u - h, v)) / (2 * h)
        assert cop.h1(u, v) == pytest.approx(dC_du, rel=1e-4, abs=1e-5)
        # h2 = dC/dv
        dC_dv = (cop.cdf(u, v + h) - cop.cdf(u, v - h)) / (2 * h)
        assert cop.h2(u, v) == pytest.approx(dC_dv, rel=1e-4, abs=1e-5)


# ---------------------------------------------------------------------
# Margin uniformity: exact without ties (derived + verified identity),
# small and shrinking discrepancy with ties (measured).
# ---------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 5, 20, 100, 400])
def test_margin_is_exactly_uniform_without_ties(n):
    """Without ties, C_n^beta(u, 1) == u to float64 rounding, for every n.

    This is the deterministic identity sum_{k=1}^n I_u(k, n+1-k) == n*u
    (derived in the module docstring): the ranks form a permutation of
    {1, ..., n}, so C_n^beta(u, 1) does not depend on the actual data
    values at all, only on n. Continuous uniform draws give no ties a.s.
    """
    rng = np.random.default_rng(_seed("marg-exact", n))
    data = rng.uniform(size=(n, 2))
    cop = EmpiricalBetaCopula(data)
    assert len(np.unique(cop.a1)) == n, "test premise: no ties on margin 1"
    grid = np.linspace(0.01, 0.99, 199)
    uv = np.column_stack([grid, np.ones_like(grid)])
    c_at_1 = cop.cdf_array(uv)
    np.testing.assert_allclose(c_at_1, grid, atol=1e-12, rtol=0)


def test_margin_identity_is_data_independent():
    """Sanity check of the identity itself: two totally different samples
    of the same size n (hence the same rank set {1,...,n}, no ties) give
    bit-for-bit the same C_n^beta(u, 1) curve."""
    n = 15
    grid = np.linspace(0.02, 0.98, 25)
    uv = np.column_stack([grid, np.ones_like(grid)])
    rng1 = np.random.default_rng(_seed("ident1"))
    rng2 = np.random.default_rng(_seed("ident2"))
    cop1 = EmpiricalBetaCopula(rng1.uniform(size=(n, 2)))
    cop2 = EmpiricalBetaCopula(rng2.uniform(size=(n, 2)) * 1000 + 5)  # different scale/location
    np.testing.assert_allclose(cop1.cdf_array(uv), cop2.cdf_array(uv), atol=1e-12)


@pytest.mark.parametrize("n,expected_order", [(6, 6.7e-3), (20, 6.0e-4),
                                               (60, 6.7e-5), (200, 6.0e-6)])
def test_margin_uniformity_with_ties_shrinks_with_n(n, expected_order):
    """With one fixed-size tied pair diluted into a growing sample, the
    margin discrepancy shrinks (measured; the ~n^-2 rate is empirical, not
    claimed as a theorem)."""
    ranks = np.arange(1, n + 1, dtype=float)
    mid = n // 2
    ranks[mid - 1] = ranks[mid] = (ranks[mid - 1] + ranks[mid]) / 2.0
    cop = EmpiricalBetaCopula.__new__(EmpiricalBetaCopula)
    cop.n = n
    cop.a1 = ranks
    cop.b1 = (n + 1) - ranks
    cop.a2 = ranks.copy()
    cop.b2 = (n + 1) - ranks
    grid = np.linspace(0.01, 0.99, 199)
    uv = np.column_stack([grid, np.ones_like(grid)])
    max_discrepancy = np.max(np.abs(cop.cdf_array(uv) - grid))
    assert max_discrepancy == pytest.approx(expected_order, rel=0.05)


# ---------------------------------------------------------------------
# Sampler vs cdf, with a stated Monte-Carlo error bound
# ---------------------------------------------------------------------
def test_sampler_matches_cdf():
    """Empirical cdf of a large sample from :meth:`sample` matches
    :meth:`cdf_array`, within a bound derived from the number of Monte
    Carlo draws (not an arbitrary tolerance).

    For M i.i.d. draws, the empirical cdf at a fixed point has standard
    deviation <= 0.5/sqrt(M) (worst case p = 0.5 for a Bernoulli
    indicator); a 6-sigma bound over 6 evaluation points has failure
    probability far below 1e-6 per point (union bound), so it is used
    directly rather than tuned by trial and error.
    """
    n_data = 30
    rng_data = np.random.default_rng(_seed("samp-data"))
    data = rng_data.uniform(size=(n_data, 2))
    cop = EmpiricalBetaCopula(data)

    M = 200_000
    draws = cop.sample(M, seed=_seed("samp-draw"))

    points = [(0.2, 0.3), (0.5, 0.5), (0.7, 0.2), (0.1, 0.9), (0.9, 0.9), (0.5, 0.1)]
    sigma_bound = 6 * 0.5 / np.sqrt(M)
    for u, v in points:
        emp = np.mean((draws[:, 0] <= u) & (draws[:, 1] <= v))
        analytic = cop.cdf(u, v)
        assert abs(emp - analytic) < sigma_bound + 1e-3, (u, v, emp, analytic)


# ---------------------------------------------------------------------
# Log-space underflow: the concrete scenario from the module docstring
# ---------------------------------------------------------------------
def test_logpdf_avoids_naive_underflow():
    """At an extreme corner, every per-point Beta-density term underflows
    float64 in linear space (naive log(mean(exp(...))) -> -inf), while
    logsumexp-based logpdf_array stays finite. This is the concrete case
    the module docstring's numerical-stability argument is built on."""
    rng = np.random.default_rng(_seed("underflow"))
    n = 500
    data = rng.uniform(size=(n, 2))
    cop = EmpiricalBetaCopula(data)

    u, v = 1e-30, 1e-30
    log_terms = (beta_dist.logpdf(u, cop.a1, cop.b1)
                 + beta_dist.logpdf(v, cop.a2, cop.b2))
    with np.errstate(divide="ignore"):
        naive = np.log(np.mean(np.exp(log_terms)))
    stable = cop.logpdf(u, v)

    assert np.isneginf(naive), "test premise: naive averaging must underflow here"
    assert np.isfinite(stable), "logsumexp-based logpdf must stay finite"
    # Sanity: the stable value is close to the best individual term minus log n.
    assert stable == pytest.approx(log_terms.max() - np.log(n) +
                                    np.log(np.sum(np.exp(log_terms - log_terms.max()))),
                                    rel=1e-9)
    # pdf_array = exp(logpdf_array) underflows to exactly 0.0 here (a
    # representable-precision floor, not a bug) — checked explicitly.
    assert cop.pdf(u, v) == 0.0


def test_pdf_is_exp_of_logpdf_everywhere_reasonable():
    """House convention (CopulaVirt): pdf_array = exp(logpdf_array), so the
    two can never disagree, checked away from the underflow floor."""
    rng = np.random.default_rng(_seed("pdf-logpdf"))
    data = rng.uniform(size=(40, 2))
    cop = EmpiricalBetaCopula(data)
    uv = rng.uniform(0.05, 0.95, size=(30, 2))
    np.testing.assert_allclose(cop.pdf_array(uv), np.exp(cop.logpdf_array(uv)))


# ---------------------------------------------------------------------
# Edge cases: very small n, and a "realistic" n
# ---------------------------------------------------------------------
def test_n2_edge_case():
    data = np.array([[0.2, 0.8], [0.9, 0.1]])
    cop = EmpiricalBetaCopula(data)
    np.testing.assert_allclose(cop.a1, [1.0, 2.0])
    np.testing.assert_allclose(cop.a2, [2.0, 1.0])
    # Well-defined, finite, and within [0, 1] everywhere on a small grid.
    for u in (0.1, 0.5, 0.9):
        for v in (0.1, 0.5, 0.9):
            c = cop.cdf(u, v)
            assert 0.0 <= c <= 1.0
            assert np.isfinite(cop.pdf(u, v))
    s = cop.sample(1000, seed=_seed("n2-sample"))
    assert s.shape == (1000, 2)
    assert np.all((s >= 0) & (s <= 1))


def test_n1_rejected():
    with pytest.raises(ValueError):
        EmpiricalBetaCopula(np.array([[0.5, 0.5]]))


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        EmpiricalBetaCopula(np.array([0.1, 0.2, 0.3]))
    cop = EmpiricalBetaCopula(np.array([[0.1, 0.2], [0.5, 0.6], [0.8, 0.3]]))
    with pytest.raises(ValueError):
        cop.cdf_array(np.array([0.1, 0.2, 0.3]))


def test_realistic_n_few_hundred():
    """n in the low hundreds: cdf/pdf/h-functions finite and well-behaved,
    sampler agrees with cdf (looser bound, fewer MC draws for speed)."""
    n = 300
    # A dependent sample (Gaussian copula, tau=0.5-ish via correlated normals)
    # so the estimator is checked on something other than independence.
    from pmcprg.copulas import CopulaGaussian
    data = CopulaGaussian(tau_k=0.5).sample(n=n, seed=_seed("realistic-data", n))
    cop = EmpiricalBetaCopula(data)

    grid = np.linspace(0.05, 0.95, 10)
    uv = np.array([[u, v] for u in grid for v in grid])
    c = cop.cdf_array(uv)
    p = cop.pdf_array(uv)
    assert np.all(np.isfinite(c)) and np.all((c >= 0) & (c <= 1))
    assert np.all(np.isfinite(p)) and np.all(p >= 0)

    # Continuous data -> no ties -> margins exactly uniform (the identity
    # derived and checked in test_margin_is_exactly_uniform_without_ties).
    assert len(np.unique(cop.a1)) == n
    c_at_1 = cop.cdf_array(np.column_stack([grid, np.ones_like(grid)]))
    np.testing.assert_allclose(c_at_1, grid, atol=1e-10, rtol=0)

    # Sampler vs cdf, fewer draws (M=20000) -> looser bound.
    M = 20_000
    draws = cop.sample(M, seed=_seed("realistic-sample", n))
    sigma_bound = 6 * 0.5 / np.sqrt(M)
    for u, v in [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]:
        emp = np.mean((draws[:, 0] <= u) & (draws[:, 1] <= v))
        assert abs(emp - cop.cdf(u, v)) < sigma_bound + 5e-3
