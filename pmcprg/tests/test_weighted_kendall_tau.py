"""``pmcprg.pmc.ice._weighted_kendall_tau`` — O(N log N) version against the N×N one.

The product-weight Kendall coefficient

    τ = Σ_{i<j} w_i w_j sign(u_i − u_j) sign(v_i − v_j) / Σ_{i<j} w_i w_j

used to be computed on N×N arrays (≈ 290 MB each and ~98 % of an ICE run at
N = 6000). ``_reference_tau`` below is a verbatim copy of that code: the merge
sort must reproduce it to rounding, zero and negative weights and the
degenerate-input contract included.

Since FR-12 the coefficient is the weighted **τ-b** when points of positive
weight tie — the denominator is √(Σ w_i w_j sgn(Δu)² · Σ w_i w_j sgn(Δv)²),
as in pyvinecopulib and VineCopula — and stays the τ-a above, by the very
same code, when none do. ``_reference_tau_b`` is the N×N oracle of that
definition, and ``_former_tau_a`` a copy of the O(N log N) code before the
change: without ties the function must still return its value bit for bit
(it is the start value of ICE's joint MLE).
"""
from __future__ import annotations

import math
import time
import tracemalloc

import numpy as np
import pytest
from scipy.stats import kendalltau

from pmcprg.numerics import MIN_POSITIVE
from pmcprg.pmc.ice import (
    _run_start_index,
    _run_starts,
    _weight_before,
    _weighted_increasing_pairs,
    _weighted_kendall_tau,
)

TOL = 1e-12


def _reference_tau(u, v, weights) -> float:
    """The former O(N²) implementation (body verbatim from 05c2878), the oracle."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    n = u.size
    if n < 2:
        return 0.0

    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= MIN_POSITIVE:
        return 0.0

    # Pairwise sign products restricted to the strict upper triangle
    # (avoid double counting and the diagonal).
    du   = np.sign(u[:, None] - u[None, :])
    dv   = np.sign(v[:, None] - v[None, :])
    sgn  = du * dv                            # (n, n) ∈ {-1, 0, +1}
    W    = w[:, None] * w[None, :]            # (n, n) outer product
    triu = np.triu_indices(n, k=1)

    den = float(W[triu].sum())
    if den <= MIN_POSITIVE:
        return 0.0
    num = float((W * sgn)[triu].sum())
    return float(np.clip(num / den, -1.0, 1.0))


def _reference_tau_b(u, v, weights) -> float:
    """The definition since FR-12, on N×N arrays: τ-a without tied pairs of positive weight, τ-b with them."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    n = u.size
    if n < 2:
        return 0.0
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= MIN_POSITIVE:
        return 0.0
    du = np.sign(u[:, None] - u[None, :])
    dv = np.sign(v[:, None] - v[None, :])
    W = w[:, None] * w[None, :]
    triu = np.triu_indices(n, k=1)
    den = float(W[triu].sum())
    if den <= MIN_POSITIVE:
        return 0.0
    num = float((W * du * dv)[triu].sum())
    tied_u = float((W * (du == 0))[triu].sum())
    tied_v = float((W * (dv == 0))[triu].sum())
    if tied_u == 0.0 and tied_v == 0.0:
        return float(np.clip(num / den, -1.0, 1.0))
    a_u = float((W * du * du)[triu].sum())
    a_v = float((W * dv * dv)[triu].sum())
    if not (a_u > MIN_POSITIVE and a_v > MIN_POSITIVE):
        return 0.0
    return float(np.clip(num / (math.sqrt(a_u) * math.sqrt(a_v)), -1.0, 1.0))


def _former_tau_a(u, v, weights) -> float:
    """``_weighted_kendall_tau`` before FR-12 (body verbatim from 5ca00ef): the τ-a in O(N log N)."""
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    n = u.size
    if n < 2:
        return 0.0

    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= MIN_POSITIVE:
        return 0.0

    den = float(np.dot(w[1:], np.cumsum(w[:-1])))
    if den <= MIN_POSITIVE:
        return 0.0
    if np.isnan(u).any() or np.isnan(v).any():
        return float("nan")

    by_vu = np.lexsort((u, v))
    v_new = _run_starts(v[by_vu])
    vu_new = v_new | _run_starts(u[by_vu])
    rv = np.empty(n, dtype=np.int64)
    rv[by_vu] = np.cumsum(v_new) - 1
    w_vu = w[by_vu]
    before = _weight_before(w_vu)
    e = float(np.dot(w_vu, before[_run_start_index(vu_new)]
                     - before[_run_start_index(v_new)]))

    by_u = np.lexsort((-rv, u))
    w_u = w[by_u]
    a = float(np.dot(w_u, _weight_before(w_u)[_run_start_index(_run_starts(u[by_u]))]))
    p = _weighted_increasing_pairs(rv[by_u], w_u)
    return float(np.clip((2.0 * p - a + e) / den, -1.0, 1.0))


def _assert_agrees(u, v, w, tol=TOL):
    """The fast coefficient against the N×N oracle of its definition (τ-b on ties)."""
    new = _weighted_kendall_tau(u, v, w)
    ref = _reference_tau_b(u, v, w)
    assert isinstance(new, float)
    assert abs(new - ref) <= tol, f"new={new!r} ref={ref!r} Δ={new - ref:.3e}"
    return new, ref


# --------------------------------------------------------------------------
# Agreement with the N×N reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 5, 17, 64, 257, 1000, 3000])
@pytest.mark.parametrize("seed", [0, 1])
def test_continuous_data_positive_weights(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.6], [0.6, 1.0]], size=n)
    w = rng.random(n)
    _assert_agrees(z[:, 0], z[:, 1], w)
    # Without ties the definitions agree and the value is the former code's, bit for bit.
    assert _reference_tau(z[:, 0], z[:, 1], w) == pytest.approx(
        _reference_tau_b(z[:, 0], z[:, 1], w), abs=TOL)
    assert _weighted_kendall_tau(z[:, 0], z[:, 1], w) == _former_tau_a(z[:, 0], z[:, 1], w)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ties_on_zero_weights_only_keep_the_tau_a_code(seed):
    """A tie that involves a point of weight zero carries no pair weight: τ-a branch, bit for bit."""
    rng = np.random.default_rng(seed)
    n = 500
    u, v, w = rng.random(n), rng.random(n), rng.random(n)
    w[:40] = 0.0
    u[:20] = u[20:40]              # ties among zero-weight points
    v[40:60] = v[:20]              # ties between a zero- and a positive-weight point
    assert _weighted_kendall_tau(u, v, w) == _former_tau_a(u, v, w)
    _assert_agrees(u, v, w)


def test_ties_give_tau_b_not_tau_a():
    """With tied pairs of positive weight τ-b and τ-a differ; the function returns τ-b."""
    rng = np.random.default_rng(12)
    u = rng.integers(0, 6, size=400).astype(float)
    v = np.round(u + rng.normal(0.0, 1.5, size=400))
    w = rng.random(400)
    new, ref_b = _assert_agrees(u, v, w)
    tau_a = _reference_tau(u, v, w)
    assert new - tau_a > 1e-2                # a definition, not rounding
    assert abs(_former_tau_a(u, v, w) - tau_a) <= TOL


def test_unit_weights_with_ties_match_scipy_tau_b():
    rng = np.random.default_rng(13)
    u = rng.integers(0, 9, size=1500).astype(float)
    v = np.round(0.5 * u + rng.normal(0.0, 2.0, size=1500))
    assert _weighted_kendall_tau(u, v, np.ones(1500)) == pytest.approx(
        kendalltau(u, v)[0], abs=TOL)


@pytest.mark.parametrize("n", [2, 10, 200, 3000])
@pytest.mark.parametrize("levels", [1, 2, 3, 7])
def test_heavy_ties_in_both_columns(n, levels):
    rng = np.random.default_rng(100 * levels + n)
    u = rng.integers(0, levels, size=n).astype(float)
    v = (u + rng.integers(0, levels, size=n)) % levels
    w = rng.random(n)
    _assert_agrees(u, v, w)


@pytest.mark.parametrize("n", [50, 2000])
def test_ties_in_one_column_only(n):
    rng = np.random.default_rng(n)
    cont = rng.random(n)
    tied = np.round(cont + 0.3 * rng.random(n), 1)
    w = rng.exponential(size=n)
    _assert_agrees(tied, cont, w)
    _assert_agrees(cont, tied, w)


def test_signed_zero_is_a_tie():
    u = np.array([0.0, -0.0, 1.0, -1.0, 0.0])
    v = np.array([-0.0, 0.0, 0.5, 0.5, 2.0])
    _assert_agrees(u, v, np.array([1.0, 2.0, 0.5, 1.5, 1.0]))
    _assert_agrees(v, u, np.array([1.0, 2.0, 0.5, 1.5, 1.0]))


@pytest.mark.parametrize("shape", ["u", "v", "both"])
def test_all_equal_columns_give_zero(shape):
    rng = np.random.default_rng(3)
    n = 500
    u = np.full(n, 0.25) if shape in ("u", "both") else rng.random(n)
    v = np.full(n, 0.75) if shape in ("v", "both") else rng.random(n)
    new, ref = _assert_agrees(u, v, rng.random(n), tol=0.0)
    assert new == ref == 0.0


def test_zero_weights_are_ignored():
    rng = np.random.default_rng(4)
    n = 1500
    u, v, w = rng.random(n), rng.random(n), rng.random(n)
    w[rng.random(n) < 0.7] = 0.0
    new, _ = _assert_agrees(u, v, w)
    keep = w > 0
    assert new == pytest.approx(
        _reference_tau_b(u[keep], v[keep], w[keep]), abs=TOL,
    )


@pytest.mark.parametrize("w", [
    np.zeros(6),                                   # Σw = 0
    np.array([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),      # one weighted point: no pair
    np.array([np.nan, 1, 1, 1, 1, 1]),             # non-finite Σw
    np.array([np.inf, 1, 1, 1, 1, 1]),
    np.array([-1.0, -2, 0.5, 0.1, 0.1, 0.1]),      # Σw < 0
])
def test_degenerate_weights_give_zero(w):
    u = np.array([0.1, 0.5, 0.3, 0.9, 0.7, 0.2])
    v = np.array([0.2, 0.4, 0.1, 0.8, 0.9, 0.3])
    new, ref = _assert_agrees(u, v, w, tol=0.0)
    assert new == ref == 0.0


@pytest.mark.parametrize("n", [0, 1])
def test_fewer_than_two_points_give_zero(n):
    assert _weighted_kendall_tau(np.zeros(n), np.zeros(n), np.ones(n)) == 0.0
    assert _reference_tau_b(np.zeros(n), np.zeros(n), np.ones(n)) == 0.0


def test_two_points():
    for u, v, expected in [((0.1, 0.2), (0.3, 0.4), 1.0),
                           ((0.1, 0.2), (0.4, 0.3), -1.0),
                           ((0.1, 0.1), (0.4, 0.3), 0.0),
                           ((0.1, 0.2), (0.3, 0.3), 0.0)]:
        new, _ = _assert_agrees(np.array(u), np.array(v), np.array([0.3, 2.0]), tol=0.0)
        assert new == expected


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mixed_sign_weights_accepted_like_the_reference(seed):
    # The reference only rejects Σw ≤ 0 and Σ_{i<j} w_i w_j ≤ 0; with mixed
    # signs the ratio can leave [−1, 1] and is clipped the same way. (ICE's
    # weights are posteriors and the public fit refuses negative weights:
    # this pins the internal function's behaviour, τ-b as the data tie.)
    rng = np.random.default_rng(seed)
    n = 800
    u = np.round(rng.random(n), 2)
    v = np.round(u + 0.5 * rng.random(n), 2)
    w = rng.random(n) + 0.2
    w[rng.random(n) < 0.3] *= -1.0
    _assert_agrees(u, v, w)


def test_mixed_sign_weights_with_nonpositive_pair_weight_give_zero():
    # Σw = 1 > 0 but Σ_{i<j} w_i w_j = ((Σw)² − Σw²)/2 = (1 − 5)/2 < 0.
    u, v = np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.6, 0.3])
    w = np.array([2.0, -1.0, 0.0])
    new, ref = _assert_agrees(u, v, w, tol=0.0)
    assert new == ref == 0.0


def test_concentrated_weight_keeps_the_pair_weight():
    # ((Σw)² − Σw²)/2 would round to 0 here and return 0.0; the prefix-sum
    # denominator keeps Σ_{i<j} w_i w_j ≈ 1e-20, like the reference.
    u = np.array([0.1, 0.4, 0.3, 0.8])
    v = np.array([0.2, 0.5, 0.1, 0.9])
    w = np.array([1.0, 1e-20, 0.0, 0.0])
    new, ref = _assert_agrees(u, v, w, tol=0.0)
    assert new == ref == 1.0


@pytest.mark.parametrize("n", [400, 3000])
def test_perfect_concordance_and_discordance(n):
    rng = np.random.default_rng(n)
    u = rng.random(n)
    w = rng.random(n)
    assert _weighted_kendall_tau(u, 2.0 * u + 1.0, w) == pytest.approx(1.0, abs=TOL)
    assert _weighted_kendall_tau(u, -u, w) == pytest.approx(-1.0, abs=TOL)
    _assert_agrees(u, -u, w)


def test_unit_weights_without_ties_match_scipy():
    rng = np.random.default_rng(7)
    z = rng.multivariate_normal([0.0, 0.0], [[1.0, -0.4], [-0.4, 1.0]], size=2000)
    assert _weighted_kendall_tau(z[:, 0], z[:, 1], np.ones(2000)) == pytest.approx(
        kendalltau(z[:, 0], z[:, 1])[0], abs=TOL,
    )


def test_input_shapes_and_dtypes():
    rng = np.random.default_rng(8)
    u = rng.integers(0, 20, size=(300, 1))           # column, integer dtype
    v = rng.integers(0, 20, size=300).tolist()        # plain list
    w = rng.random((1, 300))
    _assert_agrees(u, v, w)


def test_nan_observation_gives_nan_like_the_reference():
    u = np.array([0.1, np.nan, 0.3, 0.4])
    v = np.array([0.2, 0.1, 0.5, 0.3])
    w = np.ones(4)
    assert np.isnan(_weighted_kendall_tau(u, v, w))
    assert np.isnan(_reference_tau(u, v, w))
    assert np.isnan(_reference_tau_b(u, v, w))
    assert np.isnan(_weighted_kendall_tau(v, u, w))


@pytest.mark.filterwarnings("ignore:invalid value encountered in subtract:RuntimeWarning")
def test_single_infinity_is_an_extreme_value():
    # (The reference computes inf − inf on its diagonal, hence the warning.)
    u = np.array([0.1, np.inf, 0.3, -np.inf, 0.2])
    v = np.array([0.2, 0.9, 0.5, 0.0, 0.1])
    _assert_agrees(u, v, np.array([1.0, 0.5, 2.0, 1.0, 0.3]))


def test_weight_scale_invariance():
    rng = np.random.default_rng(9)
    u, v, w = rng.random(700), rng.random(700), rng.random(700)
    base = _weighted_kendall_tau(u, v, w)
    assert _weighted_kendall_tau(u, v, 1e-150 * w) == pytest.approx(base, abs=TOL)
    assert _weighted_kendall_tau(u, v, 1e120 * w) == pytest.approx(base, abs=TOL)


# --------------------------------------------------------------------------
# The merge-sort kernel on its own
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 8, 9, 100, 1025])
def test_weighted_increasing_pairs_brute_force(n):
    rng = np.random.default_rng(n)
    r = rng.integers(-5, 6, size=n)
    w = rng.normal(size=n)
    brute = sum(w[a] * w[b] for a in range(n) for b in range(a + 1, n) if r[a] < r[b])
    assert _weighted_increasing_pairs(r, w) == pytest.approx(brute, abs=1e-9)


# --------------------------------------------------------------------------
# Complexity
# --------------------------------------------------------------------------

def _large_input(n=20_000):
    rng = np.random.default_rng(10)
    u = rng.random(n)
    v = np.round(u + rng.random(n), 3)               # many ties in v
    return u, v, rng.random(n)


def test_large_n_memory_is_linear():
    # Deterministic: one N×N float array at N = 20 000 is 3.2 GB; the merge
    # sort peaks at a few hundred kB per N-vector.
    u, v, w = _large_input()
    tracemalloc.start()
    try:
        _weighted_kendall_tau(u, v, w)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * 2**20, f"peak {peak / 2**20:.1f} MiB"


def test_large_n_is_fast():
    # ~20 ms on a loaded 10-core laptop (NumPy 2.4). The bound leaves a
    # factor ≳ 40 and takes the best of three calls, so machine load cannot
    # make it flake; the N×N version could not even allocate its arrays.
    u, v, w = _large_input()
    best = np.inf
    for _ in range(3):
        t0 = time.perf_counter()
        tau = _weighted_kendall_tau(u, v, w)
        best = min(best, time.perf_counter() - t0)
    assert -1.0 <= tau <= 1.0
    assert best < 1.0, f"N=20000 took {best:.3f} s (best of 3)"
