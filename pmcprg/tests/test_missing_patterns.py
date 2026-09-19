"""Missingness patterns of ``pmcprg.missing.patterns`` (GenGap geometry).

Properties checked: exact counts and rates, block sizes, alignment across
channels, protected offset, determinism with a seed, disjoint / overlap
geometry, blackout rows, handling of values already missing, refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

from pmcprg.missing import patterns as pt
from pmcprg.missing.patterns import (
    aligned, blackout, disjoint, distribution, gaussian, mask_from_nan, mcar,
    overlap, scattered, state_dependent, state_markov,
)


def _data(N=200, d=4, seed=0):
    Y = np.random.default_rng(seed).normal(size=(N, d))
    return Y if d > 1 else Y[:, 0]


def _runs(col_mask):
    """(start, stop) of the maximal runs of True in a 1-D mask."""
    edges = np.diff(np.concatenate(([0], col_mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


# Every pattern, called with settings valid for N=200, d=4.
_ALL = [
    ("mcar",         lambda Y, **k: mcar(Y, 0.2, block_size=5, seed=1, **k)),
    ("aligned",      lambda Y, **k: aligned(Y, 0.2, **k)),
    ("scattered",    lambda Y, **k: scattered(Y, 0.2, seed=1, **k)),
    ("blackout",     lambda Y, **k: blackout(Y, 0.2, **k)),
    ("disjoint",     lambda Y, **k: disjoint(Y, 0.2, **k)),
    ("overlap",      lambda Y, **k: overlap(Y, 0.2, shift=0.05, **k)),
    ("gaussian",     lambda Y, **k: gaussian(Y, 0.2, seed=1, **k)),
    ("distribution", lambda Y, **k: distribution(
        Y, np.ones(200 - pt._offset_count(200, k.get("offset", 0.1))), 0.2,
        seed=1, **k)),
]


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,fn", _ALL, ids=[n for n, _ in _ALL])
@pytest.mark.parametrize("d", [1, 4])
def test_contract_shape_nan_and_input_untouched(name, fn, d):
    Y = _data(d=d)
    Y0 = Y.copy()
    Ym, m = fn(Y)
    assert Ym.shape == Y.shape and m.shape == Y.shape
    assert m.dtype == bool and m.any()
    np.testing.assert_array_equal(np.isnan(Ym), m)
    np.testing.assert_array_equal(Ym[~m], Y[~m])
    np.testing.assert_array_equal(Y, Y0)            # input not modified


@pytest.mark.parametrize("name,fn", _ALL, ids=[n for n, _ in _ALL])
@pytest.mark.parametrize("offset,P", [(0.1, 20), (0.25, 50), (7, 7), (0, 0)])
def test_offset_is_never_masked(name, fn, offset, P):
    Y = _data()
    _, m = fn(Y, offset=offset)
    assert not m[:P].any()
    if P:
        assert m[P:].any()


@pytest.mark.parametrize("name,fn", _ALL, ids=[n for n, _ in _ALL])
def test_each_contaminated_series_loses_W_values(name, fn):
    """W = floor(N * rate_series) = 40, on the full length N (not N - P)."""
    _, m = fn(_data())
    # mcar: B * block_size = 8 * 5; disjoint/overlap: all 4 blocks fit.
    assert list(m.sum(axis=0)) == [40, 40, 40, 40]


def test_offset_rules():
    assert pt._offset_count(200, 0.1) == 20
    assert pt._offset_count(101, 0.1) == 11          # ceil, as GenGap
    assert pt._offset_count(100, 0.07) == 7          # not 8: round-off absorbed
    assert pt._offset_count(200, 1) == 1             # >= 1 is a count
    assert pt._offset_count(200, 30.0) == 30
    with pytest.raises(ValueError, match="integer"):
        pt._offset_count(200, 1.5)
    with pytest.raises(ValueError, match=">= 0"):
        pt._offset_count(200, -0.1)
    with pytest.raises(ValueError, match="nothing is left"):
        pt._offset_count(50, 50)


def test_rate_rules_and_refusals():
    Y = _data(N=100, d=1)
    assert pt._n_missing(100, 0, 0.29) == 29         # GenGap's int() gives 28
    with pytest.raises(ValueError, match=r"\(0, 1\] \(a fraction, not a percentage\)"):
        aligned(Y, 20)
    with pytest.raises(ValueError, match=r"rate_series must lie in \(0, 1\]"):
        aligned(Y, 0.0)
    with pytest.raises(ValueError, match=r"rate_series must lie in \(0, 1\]"):
        aligned(Y, float("nan"))
    with pytest.raises(ValueError, match="= 0 values"):
        aligned(Y, 0.005)
    with pytest.raises(ValueError, match=r"offset \(10\) plus the missing values \(95\)"):
        aligned(Y, 0.95)
    with pytest.raises(ValueError, match="rate_dataset"):
        aligned(_data(), 0.1, rate_dataset=0)
    with pytest.raises(ValueError, match="rate_dataset"):
        aligned(Y, 0.1, rate_dataset=1.5)            # validated even when 1-D
    with pytest.raises(ValueError, match="1-D"):
        aligned(np.zeros((5, 5, 2)), 0.1)
    with pytest.raises(ValueError, match="empty"):
        aligned(np.zeros(0), 0.1)
    with pytest.raises(ValueError, match="numeric"):
        aligned(["a", "b"], 0.5)


def test_mask_from_nan():
    Y = np.array([[1.0, np.nan], [np.nan, np.nan], [3.0, 4.0]])
    np.testing.assert_array_equal(
        mask_from_nan(Y), [[False, True], [True, True], [False, False]])
    np.testing.assert_array_equal(mask_from_nan([1.0, np.nan]), [False, True])
    Ym, m = mcar(_data(d=3), 0.3, block_size=4, seed=2)
    np.testing.assert_array_equal(mask_from_nan(Ym), m)


# ---------------------------------------------------------------------------
# Random patterns — determinism
# ---------------------------------------------------------------------------


_RANDOM = [
    ("mcar",      lambda Y, s: mcar(Y, 0.2, block_size=5, rate_dataset=0.5, seed=s)),
    ("scattered", lambda Y, s: scattered(Y, 0.2, seed=s)),
    ("gaussian",  lambda Y, s: gaussian(Y, 0.2, seed=s)),
    ("distribution", lambda Y, s: distribution(Y, np.ones(180), 0.2, seed=s)),
]


@pytest.mark.parametrize("name,fn", _RANDOM, ids=[n for n, _ in _RANDOM])
def test_same_seed_same_mask_different_seed_different_mask(name, fn):
    Y = _data()
    _, m1 = fn(Y, 123)
    _, m2 = fn(Y, 123)
    _, m3 = fn(Y, 124)
    np.testing.assert_array_equal(m1, m2)
    assert not np.array_equal(m1, m3)
    _, m4 = fn(Y, np.random.default_rng(123))       # a Generator is accepted
    np.testing.assert_array_equal(m1, m4)


def test_univariate_ignores_rate_dataset():
    y = _data(d=1)
    _, a = mcar(y, 0.2, block_size=5, rate_dataset=0.01, seed=9)
    _, b = mcar(y, 0.2, block_size=5, rate_dataset=1.0, seed=9)
    np.testing.assert_array_equal(a, b)
    assert a.sum() == 40
    for fn in (aligned, lambda v, r, **k: scattered(v, r, seed=3, **k),
               lambda v, r, **k: gaussian(v, r, seed=3, **k)):
        _, lo = fn(y, 0.2, rate_dataset=0.01)
        _, hi = fn(y, 0.2, rate_dataset=1.0)
        np.testing.assert_array_equal(lo, hi)


# ---------------------------------------------------------------------------
# mcar
# ---------------------------------------------------------------------------


def test_mcar_count_is_whole_blocks():
    """W = 46, B = floor(46 / 10) = 4 → exactly 40 values, as GenGap."""
    _, m = mcar(_data(d=1), 0.23, block_size=10, seed=0)
    assert m.sum() == 40


@pytest.mark.parametrize("seed", range(8))
def test_mcar_gaps_are_made_of_whole_blocks(seed):
    """Blocks never overlap and merge whole: each gap (runs touching the two
    ends of [P, N) are one gap, by wrap-around) is a multiple of block_size."""
    N, P, bs = 120, 12, 6
    _, m = mcar(_data(N=N, d=3), 0.45, block_size=bs, offset=P, seed=seed)
    for j in range(3):
        runs = _runs(m[:, j])
        lengths = [b - a for a, b in runs]
        if len(runs) > 1 and runs[0][0] == P and runs[-1][1] == N:
            lengths = [lengths[0] + lengths[-1]] + lengths[1:-1]
        assert all(L % bs == 0 for L in lengths), lengths
        assert m[:, j].sum() == (int(N * 0.45) // bs) * bs


def test_mcar_wraps_and_skips_to_fill_the_whole_range():
    """rate 1, offset 0: 3 blocks of 10 on N = 30 must cover everything —
    only possible with the wrap-around and the skip-forward rules."""
    for seed in range(20):
        _, m = mcar(np.arange(30.0), 1.0, block_size=10, offset=0, seed=seed)
        assert m.all()


def test_mcar_selects_random_series():
    Y = _data(d=6)
    seen = set()
    for seed in range(10):
        _, m = mcar(Y, 0.2, block_size=5, rate_dataset=0.5, seed=seed)
        cols = np.flatnonzero(m.any(axis=0))
        assert len(cols) == 3                          # ceil(6 * 0.5)
        seen.add(tuple(cols))
    assert len(seen) > 1


def test_mcar_refusals():
    y = _data(d=1)
    with pytest.raises(ValueError, match="no whole block fits"):
        mcar(y, 0.02, block_size=10)                   # W = 4 < 10
    for bad in (0, -1, 2.5, True, "10"):
        with pytest.raises(ValueError, match="block_size"):
            mcar(y, 0.2, block_size=bad)


def test_mcar_with_values_already_missing():
    y = _data(d=1)
    y[50:90] = np.nan
    Ym, m = mcar(y, 0.3, block_size=5, seed=4)
    assert m.sum() == 60                               # exact count, new values
    assert not m[50:90].any()                          # never flagged
    assert np.isnan(Ym[50:90]).all()                   # still missing
    np.testing.assert_array_equal(np.isnan(Ym), m | np.isnan(y))

    z = _data(d=1)
    z[30:190] = np.nan                                 # 20 observed after P
    with pytest.raises(ValueError, match="observed values after the offset"):
        mcar(z, 0.2, block_size=5)


# ---------------------------------------------------------------------------
# aligned / scattered / blackout
# ---------------------------------------------------------------------------


def test_aligned_block_is_synchronised_on_the_first_series():
    _, m = aligned(_data(d=4), 0.15, rate_dataset=0.5)
    expected = np.zeros((200, 4), dtype=bool)
    expected[20:50, :2] = True                         # ceil(4 * 0.5) = 2 first
    np.testing.assert_array_equal(m, expected)


def test_scattered_one_block_per_series_with_its_own_start():
    N, W, P = 200, 30, 20
    starts = []
    for seed in range(6):
        _, m = scattered(_data(d=4), 0.15, rate_dataset=0.75, seed=seed)
        assert not m[:, 3].any()                       # ceil(4 * 0.75) = 3 first
        for j in range(3):
            (runs,) = [_runs(m[:, j])]
            assert len(runs) == 1
            a, b = runs[0]
            assert b - a == W and P <= a <= N - W
            starts.append(a)
    assert len(set(starts)) > 3


def test_blackout_masks_all_channels_on_the_same_rows():
    Ym, m = blackout(_data(d=5), 0.1, offset=0.3)
    rows = np.flatnonzero(m.any(axis=1))
    np.testing.assert_array_equal(rows, np.arange(60, 80))
    assert m[rows].all() and np.isnan(Ym[rows]).all()
    y = _data(d=1)
    np.testing.assert_array_equal(blackout(y, 0.1)[1], aligned(y, 0.1)[1])


def test_block_patterns_flag_only_observed_values():
    Y = _data(d=2)
    Y[25:30, 0] = np.nan
    Ym, m = blackout(Y, 0.1)                           # rows 20..39
    assert m[:, 0].sum() == 15 and m[:, 1].sum() == 20
    assert not m[25:30, 0].any() and np.isnan(Ym[25:30, 0]).all()


# ---------------------------------------------------------------------------
# disjoint / overlap
# ---------------------------------------------------------------------------


def test_disjoint_tiles_blocks_series_after_series():
    _, m = disjoint(_data(N=100, d=4), 0.3)            # P = 10, W = 30
    assert _runs(m[:, 0]) == [(10, 40)]
    assert _runs(m[:, 1]) == [(40, 70)]
    assert _runs(m[:, 2]) == [(70, 100)]
    assert not m[:, 3].any()                           # no room left
    assert m.sum(axis=1).max() == 1                    # never two series at once


def test_disjoint_truncates_at_limit():
    _, m = disjoint(_data(N=100, d=4), 0.3, limit=0.8)
    assert _runs(m[:, 2]) == [(70, 80)]
    assert not m[80:].any() and not m[:, 3].any()
    _, m = disjoint(_data(N=100, d=3), 0.2, limit=1.0)  # all series fit
    assert [_runs(m[:, j]) for j in range(3)] == [[(10, 30)], [(30, 50)], [(50, 70)]]
    with pytest.raises(ValueError, match="before the first block"):
        disjoint(_data(N=100, d=4), 0.3, limit=0.35)
    with pytest.raises(ValueError, match="limit"):
        disjoint(_data(N=100, d=4), 0.3, limit=0)


def test_overlap_blocks_share_shift_timestamps():
    _, m = overlap(_data(N=100, d=4), 0.2, shift=0.05)  # P 10, W 20, sh 5
    assert [_runs(m[:, j]) for j in range(4)] == [
        [(10, 30)], [(25, 45)], [(40, 60)], [(55, 75)]]
    for j in range(3):
        assert (m[:, j] & m[:, j + 1]).sum() == 5
    np.testing.assert_array_equal(
        overlap(_data(N=100, d=4), 0.2, shift=0)[1],
        disjoint(_data(N=100, d=4), 0.2)[1])
    _, m0 = overlap(_data(N=100, d=2), 0.2, shift=0.05, offset=0)
    assert _runs(m0[:, 1]) == [(15, 35)]               # GenGap's shift<=offset not needed


def test_overlap_refusals():
    with pytest.raises(ValueError, match="blocks would not advance"):
        overlap(_data(N=100, d=3), 0.1, shift=0.1)
    with pytest.raises(ValueError, match=r"shift must lie in \[0, 1\)"):
        overlap(_data(N=100, d=3), 0.1, shift=-0.1)


def test_disjoint_univariate_is_one_block():
    _, m = disjoint(_data(d=1), 0.2)
    assert _runs(m) == [(20, 60)]
    _, m = overlap(_data(d=1), 0.2)
    assert _runs(m) == [(20, 60)]


# ---------------------------------------------------------------------------
# gaussian / distribution
# ---------------------------------------------------------------------------


def test_gaussian_position_centre_and_values_centre():
    N, P = 2000, 200
    y = np.full(N, 0.8)
    pos = [np.flatnonzero(gaussian(y, 0.05, std_dev=0.05, seed=s)[1]).mean()
           for s in range(5)]
    assert abs(np.mean(pos) - (P + N) / 2) < 15        # centre 1100
    val = [np.flatnonzero(gaussian(y, 0.05, std_dev=0.05, selected_mean="values",
                                   seed=s)[1]).mean() for s in range(5)]
    assert abs(np.mean(val) - (P + 0.8 * (N - P))) < 15  # centre 1640
    big = np.full(N, 7.0)                              # mean clipped to 1 → N
    _, m = gaussian(big, 0.05, std_dev=0.05, selected_mean="values", seed=0)
    assert np.flatnonzero(m).min() > 1500


def test_gaussian_draws_isolated_positions_exactly():
    _, m = gaussian(_data(d=4), 0.3, rate_dataset=0.5, seed=0)
    assert list(m.sum(axis=0)) == [60, 60, 0, 0]


def test_gaussian_refusals():
    y = _data(d=1)
    with pytest.raises(ValueError, match="selected_mean"):
        gaussian(y, 0.1, selected_mean="median")
    with pytest.raises(ValueError, match="std_dev must be > 0"):
        gaussian(y, 0.1, std_dev=0)
    with pytest.raises(ValueError, match="positive probability"):
        gaussian(y, 0.5, std_dev=1e-6)                 # pdf underflows


def test_gaussian_skips_values_already_missing():
    y = _data(N=400, d=1)
    y[150:250] = np.nan
    Ym, m = gaussian(y, 0.2, seed=1)
    assert m.sum() == 80 and not m[150:250].any()


def test_distribution_follows_the_given_weights():
    N, P = 200, 20
    probs = np.zeros(N - P)
    probs[100:140] = 3.0                               # positions 120..159, unnormalised
    _, m = distribution(_data(d=1), probs, 0.1, seed=0)
    idx = np.flatnonzero(m)
    assert len(idx) == 20 and idx.min() >= 120 and idx.max() < 160

    per_series = np.zeros((N - P, 2))
    per_series[:40, 0] = 1                             # series 0 → 20..59
    per_series[-40:, 1] = 1                            # series 1 → 160..199
    _, m2 = distribution(_data(d=2), per_series, 0.1, seed=0)
    assert np.flatnonzero(m2[:, 0]).max() < 60
    assert np.flatnonzero(m2[:, 1]).min() >= 160


def test_distribution_refusals():
    Y = _data(d=2)
    with pytest.raises(ValueError, match=r"shape \(180,\) or \(180, 2\)"):
        distribution(Y, np.ones(200), 0.1)
    with pytest.raises(ValueError, match="non-negative"):
        distribution(Y, -np.ones(180), 0.1)
    few = np.zeros(180)
    few[:5] = 1
    with pytest.raises(ValueError, match="only 5 observed positions"):
        distribution(Y, few, 0.1)
    with pytest.raises(ValueError, match=r"shape \(180,\) — one entry"):
        distribution(Y[:, 0], np.ones((180, 2)), 0.1)


# ---------------------------------------------------------------------------
# State-dependent masks (P6): state_dependent, state_markov
# ---------------------------------------------------------------------------

def _blocks(N, length=50):
    """State path alternating 0 / 1 by blocks of ``length``."""
    return (np.arange(N) // length) % 2


_STATE_GENS = [
    ("state_dependent", lambda Y, X, seed=3: state_dependent(Y, X, [0.1, 0.4], seed=seed)),
    ("state_markov", lambda Y, X, seed=3: state_markov(Y, X, [0.05, 0.2], [0.5, 0.9], seed=seed)),
]


@pytest.mark.parametrize("name,fn", _STATE_GENS, ids=[n for n, _ in _STATE_GENS])
@pytest.mark.parametrize("d", [1, 3])
def test_state_masks_contract(name, fn, d):
    Y = _data(N=400, d=d)
    Y0 = Y.copy()
    X = _blocks(400)
    Ym, m = fn(Y, X)
    assert Ym.shape == Y.shape and m.shape == Y.shape and m.dtype == bool and m.any()
    np.testing.assert_array_equal(np.isnan(Ym), m)
    np.testing.assert_array_equal(Ym[~m], Y[~m])
    np.testing.assert_array_equal(Y, Y0)
    if d > 1:                                          # whole rows
        assert np.all(m.all(axis=1) == m.any(axis=1))
    _, m2 = fn(Y, X)
    _, m3 = fn(Y, X, seed=4)
    np.testing.assert_array_equal(m, m2)
    assert not np.array_equal(m, m3)


@pytest.mark.parametrize("name,fn", _STATE_GENS, ids=[n for n, _ in _STATE_GENS])
def test_state_masks_skip_values_already_missing(name, fn):
    Y = _data(N=400, d=1)
    X = _blocks(400)
    _, m_ref = fn(Y, X)
    Y[100:140] = np.nan
    Ym, m = fn(Y, X)
    assert not m[100:140].any() and np.isnan(Ym[100:140]).all()
    # the draw does not depend on them: same rows outside
    np.testing.assert_array_equal(m[:100], m_ref[:100])
    np.testing.assert_array_equal(m[140:], m_ref[140:])


def test_state_dependent_rates_per_state():
    N = 40_000
    X = _blocks(N, 37)
    rates = np.array([0.05, 0.4])
    _, m = state_dependent(np.zeros(N), X, rates, seed=11)
    for i in range(2):
        n = int((X == i).sum())
        f = m[X == i].mean()
        assert abs(f - rates[i]) < 4.5 * np.sqrt(rates[i] * (1 - rates[i]) / n)
    _, m = state_dependent(np.zeros(N), X, [0.0, 1.0], seed=11)
    np.testing.assert_array_equal(m, X == 1)


def test_state_markov_onset_persistence_and_initial_law():
    N = 60_000
    X = _blocks(N, 400)
    a, b = np.array([0.02, 0.1]), np.array([0.6, 0.9])
    _, m = state_markov(np.zeros(N), X, a, b, seed=5)
    prev, cur, x = m[:-1], m[1:], X[1:]
    for i in range(2):
        for p_true, sel in ((a[i], ~prev & (x == i)), (b[i], prev & (x == i))):
            n = int(sel.sum())
            assert abs(cur[sel].mean() - p_true) < 4.5 * np.sqrt(p_true * (1 - p_true) / n)
    # m_0 ~ Bernoulli(a / (1 − b + a)) given x_0: 4000 independent first rows
    s = a / (1 - b + a)
    for i in range(2):
        first = np.array([state_markov(np.zeros(1), [i], a, b, seed=k)[1][0] for k in range(4000)])
        assert abs(first.mean() - s[i]) < 4.5 * np.sqrt(s[i] * (1 - s[i]) / 4000)
    # a = 0 never starts a burst; b = 0 ends every burst after one row
    _, m = state_markov(np.zeros(2000), np.zeros(2000, int), [0.0, 0.5], [0.3, 0.5], seed=1)
    assert not m.any()
    _, m = state_markov(np.zeros(2000), np.zeros(2000, int), [0.5, 0.5], [0.0, 0.5], seed=1)
    assert not np.any(m[1:] & m[:-1]) and m.any()


def test_state_mask_refusals():
    Y = np.zeros(10)
    X = np.zeros(10, int)
    with pytest.raises(ValueError, match="state path"):
        state_dependent(Y, X[:9], [0.1, 0.2])
    with pytest.raises(ValueError, match="integer states"):
        state_dependent(Y, X + 0.5, [0.1, 0.2])
    with pytest.raises(ValueError, match="negative state"):
        state_dependent(Y, X - 1, [0.1, 0.2])
    with pytest.raises(ValueError, match="only 2 entries"):
        state_dependent(Y, X + 2, [0.1, 0.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        state_dependent(Y, X, [0.1, 1.2])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        state_dependent(Y, X, [np.nan, 0.2])
    with pytest.raises(ValueError, match="1-D"):
        state_dependent(Y, X, [[0.1, 0.2]])
    with pytest.raises(ValueError, match="one entry per state each"):
        state_markov(Y, X, [0.1, 0.2], [0.5])
    with pytest.raises(ValueError, match="0/0"):
        state_markov(Y, X, [0.0, 0.2], [1.0, 0.5])
    # integer-valued floats are states
    _, m = state_dependent(Y, X.astype(float), [1.0, 0.0])
    assert m.all()
