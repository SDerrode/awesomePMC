"""pmcprg.missing.patterns — artificial missingness patterns (GenGap geometry).

Every pattern takes an observation array ``Y`` — ``(N,)`` for one series or
``(N, d)`` for ``d`` series stored as *columns* — and returns
``(Y_masked, mask)``:

* ``Y_masked`` — a float copy of ``Y`` with NaN at every removed position;
* ``mask`` — a boolean array of ``Y``'s shape, True where *this call*
  removed a value.

The geometry follows ImputeGAP's ``GenGap`` (``imputegap/recovery/
contamination.py``; Khayati et al., arXiv:2503.15250; the scenarios of
*Mind the Gap*, PVLDB 13(5):768–782, 2020, doi:10.14778/3377369.3377383).
ImputeGAP is not a dependency, and masks are not bit-identical to its own:
it seeds numpy's legacy global generator with 42, these functions draw from
``numpy.random.default_rng(seed)``.

Shared rules
------------
Protected offset ``P``
    The leading ``P`` values of every series are never removed.
    ``offset < 1`` is a fraction: ``P = ceil(N * offset)`` (GenGap's
    ``_compute_offset``); ``offset >= 1`` is a count and must be an integer.
    Default 0.1, as in GenGap.
Values removed per series ``W``
    ``W = floor(N * rate_series)`` — a fraction of the *whole* length ``N``,
    not of ``N - P``, as in GenGap. ``rate_series`` must lie in ``(0, 1]``
    (a fraction, not a percentage) and ``W`` must be at least 1;
    ``P + W > N`` is refused, GenGap's own check. Unlike GenGap, the floor
    and the ceiling above absorb float round-off (``100 * 0.29`` gives
    ``W = 29``, where GenGap's ``int`` gives 28).
Series contaminated ``ceil(d * rate_dataset)``
    ``rate_dataset`` in ``(0, 1]``; default 1.0 here (all series), where
    GenGap defaults to 0.2. ``mcar`` picks the series at random, as GenGap;
    ``aligned``, ``scattered``, ``gaussian`` and ``distribution`` take the
    *first* ones, as GenGap. For a univariate ``(N,)`` input ``rate_dataset``
    is validated but otherwise ignored: the single series is always
    contaminated — GenGap's ``ceil(1 * rate_dataset)`` is 1 for any
    ``rate_dataset`` in ``(0, 1]``, so this is its behaviour, not a new rule.
Values already missing in ``Y``
    They stay NaN in ``Y_masked`` but are never flagged in ``mask`` — the
    mask marks positions whose ground truth is known, which is what the
    metrics of :mod:`pmcprg.missing.metrics` score. Random patterns draw among
    the observed positions only, so their count stays exact (``mcar`` skips
    them forward, GenGap's own rule); fixed-geometry patterns (``aligned``,
    ``scattered``, ``blackout``, ``disjoint``, ``overlap``) keep their block
    and flag only its observed part.
Errors
    Impossible settings raise ``ValueError`` naming the quantity at fault.
    GenGap sometimes corrects silently instead (``mcar`` shrinks
    ``block_size``); these functions never do.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy.stats import norm

__all__ = [
    "mask_from_nan",
    "mcar",
    "aligned",
    "scattered",
    "blackout",
    "disjoint",
    "overlap",
    "gaussian",
    "distribution",
]

Seed = Union[int, np.random.Generator, np.random.SeedSequence, None]

# Slack that lets floor/ceil absorb binary round-off (100 * 0.29 = 28.999…).
_ROUND_TOL = 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mask_from_nan(Y) -> np.ndarray:
    """Boolean mask of the missing (NaN) entries of ``Y``, same shape.

    For a multichannel ``(N, d)`` array the mask is per entry; a row is
    missing for inference as soon as one of its components is
    (``mask.any(axis=1)``).
    """
    try:
        arr = np.asarray(Y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Y must be numeric: {exc}") from exc
    return np.isnan(arr)


def _as_columns(Y) -> tuple[np.ndarray, bool]:
    """``(N, d)`` float copy of ``Y`` and whether the input was 1-D."""
    try:
        arr = np.array(Y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Y must be numeric: {exc}") from exc
    if arr.ndim == 1:
        arr = arr[:, None]
        univariate = True
    elif arr.ndim == 2:
        univariate = False
    else:
        raise ValueError(
            f"Y must be 1-D (N,) or 2-D (N, d) with series as columns; "
            f"got shape {arr.shape}."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Y is empty (shape {arr.shape}).")
    return arr, univariate


def _fraction(name: str, value, *, allow_zero: bool = False) -> float:
    """Validate a fraction in (0, 1] (or [0, 1) when ``allow_zero``)."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}.") from exc
    if allow_zero:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"{name} must lie in [0, 1), got {value!r}.")
    elif not (0.0 < v <= 1.0):
        hint = " (a fraction, not a percentage)" if v > 1.0 else ""
        raise ValueError(f"{name} must lie in (0, 1]{hint}, got {value!r}.")
    return v


def _offset_count(N: int, offset) -> int:
    """Number of protected leading values ``P`` (fraction < 1, count >= 1)."""
    try:
        v = float(offset)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"offset must be a number, got {offset!r}.") from exc
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"offset must be >= 0, got {offset!r}.")
    if v < 1.0:
        P = math.ceil(N * v - _ROUND_TOL)
    else:
        if not v.is_integer():
            raise ValueError(
                f"offset >= 1 is a count of protected values and must be an "
                f"integer, got {offset!r} (use a fraction < 1 otherwise)."
            )
        P = int(v)
    P = max(P, 0)
    if P >= N:
        raise ValueError(
            f"offset protects {P} leading values, but the series has only "
            f"N={N}: nothing is left to contaminate."
        )
    return P


def _n_missing(N: int, P: int, rate_series) -> int:
    """Values removed per series ``W = floor(N * rate_series)``, validated."""
    rate = _fraction("rate_series", rate_series)
    W = math.floor(N * rate + _ROUND_TOL)
    if W < 1:
        raise ValueError(
            f"rate_series={rate_series!r} removes floor({N} * {rate_series!r}) "
            f"= 0 values from a series of length N={N}; raise the rate."
        )
    if P + W > N:
        raise ValueError(
            f"The protected offset ({P}) plus the missing values ({W}) exceed "
            f"the series length N={N}; lower rate_series or offset."
        )
    return W


def _positive_int(name: str, value) -> int:
    ok = (not isinstance(value, bool)
          and isinstance(value, (int, np.integer, float, np.floating))
          and math.isfinite(value) and float(value).is_integer() and value >= 1)
    if not ok:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}.")
    return int(value)


def _n_series(d: int, rate_dataset) -> int:
    rate = _fraction("rate_dataset", rate_dataset)
    return max(1, math.ceil(d * rate - _ROUND_TOL))


def _finish(arr: np.ndarray, mask: np.ndarray, univariate: bool):
    out = arr.copy()
    out[mask] = np.nan
    if univariate:
        return out[:, 0], mask[:, 0]
    return out, mask


# ---------------------------------------------------------------------------
# Random blocks
# ---------------------------------------------------------------------------


def mcar(Y, rate_series, *, block_size: int = 10, rate_dataset=1.0,
         offset=0.1, seed: Seed = None):
    """Missing completely at random, in blocks of ``block_size`` values.

    GenGap's ``mcar``: ``ceil(d * rate_dataset)`` series chosen at random
    (without replacement); in each, ``B = floor(W / block_size)`` block starts
    drawn without replacement in ``[P, N)``, and every block removes exactly
    ``block_size`` positions from its start onward. A block running past the
    end wraps around to ``P``; a position already missing (an earlier block
    or a NaN of ``Y``) is skipped forward. Blocks therefore never overlap —
    they may merge into longer gaps — and each contaminated series loses
    exactly ``B * block_size`` values, which can be less than ``W`` (GenGap's
    count too).

    Differences from GenGap: when ``W < block_size`` (``B = 0``) GenGap
    prints a correction and shrinks ``block_size`` to ``W // 2``; this
    function raises. When a series has fewer than ``B * block_size`` observed
    positions after the offset, GenGap skips it with a message; this function
    raises.

    Parameters
    ----------
    Y : array (N,) or (N, d)
    rate_series : float in (0, 1] — fraction of each series' length N removed.
    block_size : int >= 1, default 10.
    rate_dataset : float in (0, 1], default 1.0 — share of series contaminated
        (ignored for a 1-D ``Y``, see the module docstring).
    offset : float, default 0.1 — protected leading fraction (< 1) or count.
    seed : int, Generator, SeedSequence or None — ``default_rng(seed)``.

    Returns
    -------
    (Y_masked, mask) — see the module docstring.
    """
    arr, univariate = _as_columns(Y)
    N, d = arr.shape
    block_size = _positive_int("block_size", block_size)
    P = _offset_count(N, offset)
    W = _n_missing(N, P, rate_series)
    n_sel = _n_series(d, rate_dataset)
    B = W // block_size
    if B < 1:
        raise ValueError(
            f"block_size={block_size} exceeds the {W} values to remove "
            f"(N={N}, rate_series={rate_series!r}): no whole block fits. "
            f"Lower block_size or raise rate_series."
        )
    rng = np.random.default_rng(seed)
    cols = [0] if d == 1 else rng.choice(d, size=n_sel, replace=False)

    missing = np.isnan(arr)
    mask = np.zeros_like(missing)
    for j in cols:
        taken = missing[:, j].copy()   # already-missing or removed positions
        available = int((~taken[P:]).sum())
        if available < B * block_size:
            raise ValueError(
                f"Series {j} has {available} observed values after the offset, "
                f"fewer than the {B * block_size} to remove."
            )
        starts = rng.choice(np.arange(P, N), size=B, replace=False)
        for start in starts:
            for jump in range(block_size):
                pos = int(start) + jump
                if pos >= N:
                    pos = P + (pos - N)
                while taken[pos]:
                    pos += 1
                    if pos >= N:
                        pos = P + (pos - N)
                taken[pos] = True
                mask[pos, j] = True
    return _finish(arr, mask, univariate)


# ---------------------------------------------------------------------------
# Single contiguous block per series
# ---------------------------------------------------------------------------


def aligned(Y, rate_series, *, rate_dataset=1.0, offset=0.1):
    """One block ``[P, P + W)`` in each of the first ``ceil(d * rate_dataset)`` series.

    GenGap's ``aligned``: the gaps are synchronised across the contaminated
    series and start right after the protected offset. Deterministic (no
    seed, as in GenGap).
    """
    arr, univariate = _as_columns(Y)
    N, d = arr.shape
    P = _offset_count(N, offset)
    W = _n_missing(N, P, rate_series)
    n_sel = _n_series(d, rate_dataset)
    block = np.zeros_like(arr, dtype=bool)
    block[P:P + W, :n_sel] = True
    return _finish(arr, block & ~np.isnan(arr), univariate)


def scattered(Y, rate_series, *, rate_dataset=1.0, offset=0.1,
              seed: Seed = None):
    """One block of ``W`` values per series, each at its own random start.

    GenGap's ``scattered``: for each of the first ``ceil(d * rate_dataset)``
    series, the start is uniform on ``[P, N - W]`` (inclusive), independently
    across series.
    """
    arr, univariate = _as_columns(Y)
    N, d = arr.shape
    P = _offset_count(N, offset)
    W = _n_missing(N, P, rate_series)
    n_sel = _n_series(d, rate_dataset)
    rng = np.random.default_rng(seed)
    block = np.zeros_like(arr, dtype=bool)
    for j in range(n_sel):
        start = P + int(rng.integers(0, N - W - P + 1))
        block[start:start + W, j] = True
    return _finish(arr, block & ~np.isnan(arr), univariate)


def blackout(Y, rate_series, *, offset=0.1):
    """Every series loses the same block ``[P, P + W)`` — all channels at once.

    GenGap's ``blackout`` is ``aligned`` with ``rate_dataset = 1``; so is this
    one. For a multichannel observation every row of the block is entirely
    missing. For a univariate series it coincides with :func:`aligned`.
    """
    return aligned(Y, rate_series, rate_dataset=1.0, offset=offset)


def _staircase(Y, rate_series, *, shift, limit, offset):
    """Blocks of W values, series after series, each shifted back by ``sh``."""
    arr, univariate = _as_columns(Y)
    N, d = arr.shape
    P = _offset_count(N, offset)
    W = _n_missing(N, P, rate_series)
    lim = _fraction("limit", limit)
    end = math.floor(N * lim + _ROUND_TOL)   # positions < end may be removed
    if P + W > end:
        raise ValueError(
            f"limit={limit!r} stops contamination at position {end}, before "
            f"the first block [{P}, {P + W}) fits; raise limit or lower "
            f"rate_series/offset."
        )
    sh = math.floor(N * _fraction("shift", shift, allow_zero=True) + _ROUND_TOL)
    if sh >= W:
        raise ValueError(
            f"shift={shift!r} moves each block back by {sh} values, not less "
            f"than the block length W={W}: blocks would not advance."
        )
    block = np.zeros_like(arr, dtype=bool)
    start = P
    for j in range(d):
        stop = min(start + W, end)
        block[start:stop, j] = True
        if stop >= end:
            break
        start = stop - sh
    return _finish(arr, block & ~np.isnan(arr), univariate)


def disjoint(Y, rate_series, *, limit=1.0, offset=0.1):
    """Consecutive, non-overlapping blocks: series ``j`` loses ``[P + jW, P + (j+1)W)``.

    GenGap's ``disjoint``: the blocks tile the time axis from the offset on,
    one series after the other, until the series run out or the
    contamination reaches ``floor(N * limit)``; the block that reaches it is
    truncated there and the later series are left intact. ``rate_dataset``
    is not a parameter — the number of contaminated series follows from the
    geometry (all ``d`` if they fit).

    GenGap stops *after* writing index ``int(N * limit) - 1``; the rule here
    (positions ``< floor(N * limit)``) is the same. A ``limit`` too small for
    the first block is refused (GenGap would leave a one-value gap). A
    univariate series gets the single block ``[P, P + W)``.
    """
    return _staircase(Y, rate_series, shift=0.0, limit=limit, offset=offset)


def overlap(Y, rate_series, *, shift=0.05, limit=1.0, offset=0.1):
    """Consecutive blocks that overlap: each starts ``floor(N * shift)`` before the previous end.

    GenGap's ``overlap``: series ``j`` loses
    ``[P + j (W - sh), P + j (W - sh) + W)`` with ``sh = floor(N * shift)``,
    truncated at ``floor(N * limit)`` as in :func:`disjoint`; ``shift = 0``
    is :func:`disjoint`. Consecutive series therefore share ``sh``
    missing timestamps.

    Differences from GenGap: ``sh >= W`` is refused (blocks would stall or
    run backwards into the protected zone). GenGap's extra check
    ``int(N * shift) > int(N * offset)`` is *not* enforced: blocks move back
    relative to the previous block only, never below ``P``, so it guards
    nothing and would forbid ``offset = 0``.
    """
    return _staircase(Y, rate_series, shift=shift, limit=limit, offset=offset)


# ---------------------------------------------------------------------------
# Position distributions
# ---------------------------------------------------------------------------


def _draw_positions(rng, candidates: np.ndarray, weights: np.ndarray, W: int,
                    j: int, what: str) -> np.ndarray:
    positive = int((weights > 0).sum())
    if positive < W:
        raise ValueError(
            f"Series {j}: only {positive} observed positions have a positive "
            f"probability under the {what}, fewer than the {W} to remove."
        )
    p = weights / weights.sum()
    return rng.choice(candidates, size=W, replace=False, p=p)


def gaussian(Y, rate_series, *, rate_dataset=1.0, std_dev=0.2,
             selected_mean: str = "position", offset=0.1, seed: Seed = None):
    """Isolated missing values whose positions follow a Gaussian bump.

    GenGap's ``gaussian``: in each of the first ``ceil(d * rate_dataset)``
    series, ``W`` distinct positions are drawn without replacement from
    ``[P, N)`` with probability proportional to
    ``norm.pdf(n, loc=center, scale=std_dev * (N - P))``.

    ``selected_mean="position"`` (GenGap's default) centres the bump in the
    middle of the unprotected range, ``center = (P + N) / 2``.
    ``selected_mean="values"`` uses the series' mean value ``m``, clipped to
    ``[-1, 1]``: ``center = P + m (N - P)``. That rule assumes a normalised
    series (z-score or min-max, ImputeGAP's loaders); on raw data the centre
    is pinned to an end of the range. The mean ignores NaN here (GenGap's
    ``np.mean`` would propagate one).
    """
    arr, univariate = _as_columns(Y)
    N, d = arr.shape
    P = _offset_count(N, offset)
    W = _n_missing(N, P, rate_series)
    n_sel = _n_series(d, rate_dataset)
    try:
        sd = float(std_dev)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"std_dev must be a number, got {std_dev!r}.") from exc
    if not (math.isfinite(sd) and sd > 0):
        raise ValueError(f"std_dev must be > 0, got {std_dev!r}.")
    if selected_mean not in ("position", "values"):
        raise ValueError(
            f"selected_mean must be 'position' or 'values', got {selected_mean!r}."
        )
    rng = np.random.default_rng(seed)
    missing = np.isnan(arr)
    mask = np.zeros_like(missing)
    positions = np.arange(P, N)
    for j in range(n_sel):
        if selected_mean == "position":
            center = (P + N) / 2
        else:
            col = arr[:, j]
            m = float(np.nanmean(col)) if (~missing[:, j]).any() else 0.0
            m = max(min(m, 1.0), -1.0)
            center = P + m * (N - P)
        cand = positions[~missing[P:, j]]
        weights = norm.pdf(cand, loc=center, scale=sd * (N - P))
        chosen = _draw_positions(rng, cand, weights, W, j,
                                 f"Gaussian (std_dev={std_dev!r})")
        mask[chosen, j] = True
    return _finish(arr, mask, univariate)


def distribution(Y, probabilities, rate_series, *, rate_dataset=1.0,
                 offset=0.1, seed: Seed = None):
    """Isolated missing values drawn from a user-supplied position distribution.

    GenGap's ``distribution``: in each of the first ``ceil(d * rate_dataset)``
    series, ``W`` distinct positions of ``[P, N)`` are drawn without
    replacement with the given probabilities.

    ``probabilities`` covers the unprotected positions ``P, …, N-1``: shape
    ``(N - P,)`` — shared by every series — or ``(N - P, d)``, one column per
    series (GenGap takes ``(d, N - P)``, one *row* per series, because its
    matrix is transposed). Entries must be finite and non-negative; each
    column is renormalised to sum 1 (GenGap requires the caller to have done
    it). At least ``W`` observed positions need a positive probability.
    """
    arr, univariate = _as_columns(Y)
    N, d = arr.shape
    P = _offset_count(N, offset)
    W = _n_missing(N, P, rate_series)
    n_sel = _n_series(d, rate_dataset)
    try:
        probs = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"probabilities must be numeric: {exc}") from exc
    shapes = [(N - P,)] + ([] if univariate else [(N - P, d)])
    if probs.shape not in shapes:
        raise ValueError(
            f"probabilities must have shape "
            f"{' or '.join(str(s) for s in shapes)} — one entry per "
            f"unprotected position (N={N}, offset P={P}); got {probs.shape}."
        )
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probabilities must be finite and non-negative.")
    if probs.ndim == 1:
        probs = np.repeat(probs[:, None], d, axis=1)
    rng = np.random.default_rng(seed)
    missing = np.isnan(arr)
    mask = np.zeros_like(missing)
    positions = np.arange(P, N)
    for j in range(n_sel):
        keep = ~missing[P:, j]
        chosen = _draw_positions(rng, positions[keep], probs[keep, j], W, j,
                                 "given distribution")
        mask[chosen, j] = True
    return _finish(arr, mask, univariate)
