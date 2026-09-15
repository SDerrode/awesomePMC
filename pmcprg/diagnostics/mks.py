"""
mks.py — multivariate extension of the Kolmogorov-Smirnov goodness-of-fit
test.

Public API
----------
mks_1samp(sample, cdf, *, alpha=0.05, asymptotic=False) -> MKSResult
    One-sample test: H0 = "sample is drawn from the distribution with CDF
    `cdf`".  Reject when ``result.reject`` is True.

mks_2samp(sample_a, sample_b, *, alpha=0.05, asymptotic=False) -> MKSResult
    Two-sample test: H0 = "the two samples share the same distribution".

mks_test(sample, other=None, cdf=None, alpha=0.05, asymptotic=False)
    Convenience dispatcher — calls :func:`mks_1samp` or :func:`mks_2samp`
    depending on whether ``other`` or ``cdf`` is provided.

Algorithm
---------
The test extends the classical 1-D Kolmogorov-Smirnov statistic to
``d > 1`` dimensions via Naaman's construction (see references). For each
coordinate axis ``h`` we sort the sample by that column, then form a
sequence of "lower-orthant" corner points whose CDF values give an
exhaustive grid of comparison locations. The statistic is the maximum
absolute deviation between the empirical multivariate CDF and the
reference (or second-sample empirical) CDF over those grid points.

The companion finite-sample critical value is derived by a union-bound
argument and is therefore *conservative*. The asymptotic critical value
is tighter but only valid for large sample sizes.

References
----------
- M. Naaman, "On the tight constant in the multivariate Dvoretzky-
  Kiefer-Wolfowitz inequality", *Statistics & Probability Letters*,
  Vol. 173 (2021), 109088.
  doi:10.1016/j.spl.2021.109088
- Source-of-truth Python implementation:
  https://github.com/o-laurent/multivariate-ks-test

Origin
------
Ported (rewritten in awesomePMC style) from the companion project
``markovchain_todelete`` (``pmcprg/mks_test/``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


__all__ = [
    "MKSResult",
    "mks_1samp",
    "mks_2samp",
    "mks_test",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MKSResult:
    """Typed result of a multivariate Kolmogorov-Smirnov test.

    Fields
    ------
    statistic       : float — the KS statistic (max absolute deviation).
    critical_value  : float — the critical value at significance level
                      :attr:`alpha`. Computed under the finite-sample
                      bound (default) or the asymptotic approximation
                      (``asymptotic=True``).
    alpha           : float — the significance level used.
    reject          : bool  — True if ``statistic > critical_value``
                      (H0 rejected at level α).
    n_samples_x     : int   — size of the (first) sample.
    n_samples_y     : int | None — size of the second sample (None for
                      the one-sample test).
    dim             : int   — observation dimensionality.
    asymptotic      : bool  — whether the critical value uses the
                      asymptotic approximation.
    """
    statistic:      float
    critical_value: float
    alpha:          float
    reject:         bool
    n_samples_x:    int
    n_samples_y:    int | None
    dim:            int
    asymptotic:     bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_alpha(alpha: float) -> float:
    """Validate the significance level."""
    if not isinstance(alpha, (float, int)):
        raise TypeError(f"alpha must be a float, got {type(alpha).__name__}.")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    return alpha


def _as_2d(sample: np.ndarray, name: str) -> np.ndarray:
    """Coerce ``sample`` to a 2-D ``(N, d)`` float array; raise on bad input."""
    arr = np.asarray(sample, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 1-D or 2-D, got {arr.ndim}-D with shape {arr.shape}."
        )
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _mecdf_batch(sample: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Vectorised multivariate empirical CDF at every row of ``points``.

    Parameters
    ----------
    sample : (N, d) — observations.
    points : (M, d) — query points.

    Returns
    -------
    np.ndarray of shape (M,) — F̂_emp evaluated at each query point.

    The intermediate boolean tensor has shape (M, N, d); peak memory is
    therefore ``M * N * d`` bytes (≈ N²·d for the MKS corner grid). For
    typical MKS workloads (N ≲ a few thousand, d ≲ 10) this is well under
    100 MB; callers needing bigger samples can chunk ``points``.
    """
    sample = np.asarray(sample, dtype=float)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != sample.shape[1]:
        raise ValueError(
            "_mecdf_batch: sample/points dims must agree; got "
            f"{sample.shape} vs {points.shape}."
        )
    le = np.all(sample[None, :, :] <= points[:, None, :], axis=2)   # (M, N)
    return le.mean(axis=1)


def _eval_cdf_grid(cdf: Callable[[np.ndarray], float],
                   points: np.ndarray) -> np.ndarray:
    """Evaluate a user-supplied scalar ``cdf`` callable at every row of ``points``.

    The callable signature is one-point-in / scalar-out (per the public API),
    so we cannot vectorise the call itself — but we can hoist the Python-level
    loop out of the statistic-computation routine to keep that path tidy.
    """
    return np.fromiter(
        (float(cdf(points[i])) for i in range(points.shape[0])),
        dtype=float,
        count=points.shape[0],
    )


def _corner_grid(sample: np.ndarray) -> np.ndarray:
    """Build the (N, d, d) corner-grid used by both 1- and 2-sample tests.

    For each dimension ``h``:
      1. Sort ``sample`` by column ``h`` descending.
      2. For each column ``i``, compute the running max from the bottom
         up; store it in ``z[:, i, h]``.

    The resulting ``z[j, :, h]`` is an "upper-left orthant corner" obtained
    by elementwise-max of all rows whose column-``h`` value is ≤ the
    sorted threshold at position ``j``.

    Returns
    -------
    z : np.ndarray, shape ``(N, d, d)``.
    """
    N, dim = sample.shape
    z = np.empty((N, dim, dim))
    for h in range(dim):
        # Sort descending by column h.
        order = np.argsort(sample[:, h])[::-1]
        sorted_x = sample[order]
        # Running maximum from the bottom up, per column.
        # cummax_from_bottom[j, i] = max(sorted_x[j:, i])
        flipped = sorted_x[::-1]                          # (N, d), ascending
        cummax = np.maximum.accumulate(flipped, axis=0)   # (N, d)
        running = cummax[::-1]                            # back to descending
        z[:, :, h] = running
    return z


# ---------------------------------------------------------------------------
# Critical values
# ---------------------------------------------------------------------------

def _critical_1samp(
    n: int, dim: int, alpha: float, *, asymptotic: bool,
) -> float:
    """Finite-sample (default) or asymptotic critical value for 1-sample MKS."""
    if asymptotic:
        return math.sqrt(-math.log(alpha / (2 * dim)) * (0.5 / n))
    return math.sqrt(
        -math.log(alpha / (2 * (n + 1) * dim)) * (0.5 / n)
    )


def _critical_2samp(
    n_x: int, n_y: int, dim: int, alpha: float, *, asymptotic: bool,
) -> float:
    """Finite-sample (default) or asymptotic critical value for 2-sample MKS."""
    if asymptotic:
        return (
            math.sqrt(-math.log(alpha / (4 * dim)) * (0.5 / n_x))
            + math.sqrt(-math.log(alpha / (4 * dim)) * (0.5 / n_y))
        )
    return (
        math.sqrt(-math.log(alpha / (2 * (n_x + 1) * dim)) * (0.5 / n_x))
        + math.sqrt(-math.log(alpha / (2 * (n_y + 1) * dim)) * (0.5 / n_y))
    )


# ---------------------------------------------------------------------------
# Public API — one-sample
# ---------------------------------------------------------------------------

def mks_1samp(
    sample: np.ndarray,
    cdf: Callable[[np.ndarray], float],
    *,
    alpha: float = 0.05,
    asymptotic: bool = False,
) -> MKSResult:
    """One-sample multivariate KS test.

    Test H0: ``sample`` is drawn from the distribution whose CDF is
    ``cdf``.

    Parameters
    ----------
    sample     : np.ndarray, shape ``(N,)`` or ``(N, d)`` — observations.
    cdf        : callable — must accept a 1-D array of length ``d`` and
                  return a scalar in [0, 1] (the joint CDF evaluated at
                  the point).
    alpha      : float — significance level, in (0, 1). Default 0.05.
    asymptotic : bool — use the asymptotic critical value (tighter, but
                  only valid for large ``N``). Default False (finite-sample
                  Naaman bound).

    Returns
    -------
    MKSResult — see :class:`MKSResult`.
    """
    alpha   = _check_alpha(alpha)
    sample  = _as_2d(sample, "sample")
    N, dim  = sample.shape

    z = _corner_grid(sample)

    # Maximum |F̂_emp(z) − F_cdf(z)| over the corner grid, restricted by
    # the Naaman "tightness" indicator (round(N · F̂) == N − i).
    diff      = np.zeros((N, dim))
    idx_gate  = (N - np.arange(N))                          # (N,)
    for h in range(dim):
        corners_h = z[:, :, h]                              # (N, d)
        f_emp_h   = _mecdf_batch(sample, corners_h)         # (N,)
        f_ref_h   = _eval_cdf_grid(cdf, corners_h)          # (N,)
        indicator = (np.rint(N * f_emp_h).astype(np.int64) == idx_gate)
        diff[:, h] = np.abs(f_emp_h - f_ref_h) * indicator

    # h=0 column also considers the deviation at the data points themselves.
    f_emp_xi   = _mecdf_batch(sample, sample)
    f_ref_xi   = _eval_cdf_grid(cdf, sample)
    diff[:, 0] = np.maximum(diff[:, 0], np.abs(f_emp_xi - f_ref_xi))

    stat   = float(diff.max())
    crit   = _critical_1samp(N, dim, alpha, asymptotic=asymptotic)
    reject = stat > crit

    logger.debug(
        "mks_1samp: N=%d d=%d  stat=%.4f  crit=%.4f  α=%.3f  reject=%s",
        N, dim, stat, crit, alpha, reject,
    )
    return MKSResult(
        statistic      = stat,
        critical_value = crit,
        alpha          = alpha,
        reject         = bool(reject),
        n_samples_x    = N,
        n_samples_y    = None,
        dim            = dim,
        asymptotic     = bool(asymptotic),
    )


# ---------------------------------------------------------------------------
# Public API — two-sample
# ---------------------------------------------------------------------------

def mks_2samp(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    *,
    alpha: float = 0.05,
    asymptotic: bool = False,
) -> MKSResult:
    """Two-sample multivariate KS test.

    Test H0: ``sample_a`` and ``sample_b`` share the same distribution.

    Parameters
    ----------
    sample_a, sample_b : np.ndarray, shape ``(N_a, d)`` and ``(N_b, d)``.
        Must have the same trailing dimension. 1-D arrays are reshaped
        to ``(N, 1)`` automatically.
    alpha, asymptotic   : see :func:`mks_1samp`.

    Returns
    -------
    MKSResult — see :class:`MKSResult`.
    """
    alpha    = _check_alpha(alpha)
    sample_a = _as_2d(sample_a, "sample_a")
    sample_b = _as_2d(sample_b, "sample_b")
    N_a, dim = sample_a.shape
    N_b, d_b = sample_b.shape
    if dim != d_b:
        raise ValueError(
            f"sample_a has d={dim} features but sample_b has d={d_b}."
        )

    z = _corner_grid(sample_a)

    diff     = np.zeros((N_a, dim))
    idx_gate = (N_a - np.arange(N_a))                                # (N_a,)
    for h in range(dim):
        corners_h = z[:, :, h]                                       # (N_a, d)
        f_emp_a_h = _mecdf_batch(sample_a, corners_h)                # (N_a,)
        f_emp_b_h = _mecdf_batch(sample_b, corners_h)                # (N_a,)
        indicator = (np.rint(N_a * f_emp_a_h).astype(np.int64) == idx_gate)
        diff[:, h] = np.abs(f_emp_a_h - f_emp_b_h) * indicator

    # h=0 column also considers the deviation at the data points of sample_a.
    f_emp_a_xi = _mecdf_batch(sample_a, sample_a)
    f_emp_b_xi = _mecdf_batch(sample_b, sample_a)
    diff[:, 0] = np.maximum(diff[:, 0], np.abs(f_emp_a_xi - f_emp_b_xi))

    stat   = float(diff.max())
    crit   = _critical_2samp(N_a, N_b, dim, alpha, asymptotic=asymptotic)
    reject = stat > crit

    logger.debug(
        "mks_2samp: N_a=%d  N_b=%d  d=%d  stat=%.4f  crit=%.4f  α=%.3f  reject=%s",
        N_a, N_b, dim, stat, crit, alpha, reject,
    )
    return MKSResult(
        statistic      = stat,
        critical_value = crit,
        alpha          = alpha,
        reject         = bool(reject),
        n_samples_x    = N_a,
        n_samples_y    = N_b,
        dim            = dim,
        asymptotic     = bool(asymptotic),
    )


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def mks_test(
    sample: np.ndarray,
    other: np.ndarray | None = None,
    cdf: Callable[[np.ndarray], float] | None = None,
    *,
    alpha: float = 0.05,
    asymptotic: bool = False,
) -> MKSResult:
    """Dispatch to :func:`mks_1samp` or :func:`mks_2samp`.

    Exactly one of ``other`` / ``cdf`` must be provided.

    Parameters
    ----------
    sample, other, cdf : see :func:`mks_1samp` and :func:`mks_2samp`.
    alpha, asymptotic  : significance level / critical-value style.

    Returns
    -------
    MKSResult.
    """
    if (other is None) == (cdf is None):
        raise ValueError(
            "mks_test: pass exactly one of `other` (two-sample) or `cdf` "
            "(one-sample)."
        )
    if cdf is not None:
        return mks_1samp(sample, cdf, alpha=alpha, asymptotic=asymptotic)
    return mks_2samp(sample, other, alpha=alpha, asymptotic=asymptotic)
