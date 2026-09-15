"""pmcprg.missing.metrics — scoring imputations and classifications on missing positions.

Point metrics (ImputeGAP's ``recovery/evaluation.py`` set): :func:`rmse`,
:func:`mae`, :func:`mutual_information`, :func:`pearson`. Probabilistic
metrics: :func:`crps_from_samples`, :func:`crps_gaussian`,
:func:`interval_coverage`. Classification: :func:`error_rate_split`.

Evaluated positions
-------------------
Point metrics take the ``mask`` returned by a pattern of
:mod:`pmcprg.missing.patterns` (True = removed) and score those positions only,
pooled over every series. Probabilistic metrics take ``mask`` as an optional
keyword: ``None`` scores every position, which suits predictions already
restricted to the gaps.

NaN handling
------------
A position whose ground truth is not finite (NaN: a real gap, truth
unknown) is dropped from the evaluated set. A non-finite *prediction* at an
evaluated position raises ``ValueError`` — skipping it would flatter an
incomplete imputation. Predictions may hold anything outside the evaluated
set. A mask selecting nothing, or nothing with a finite ground truth, raises.

Differences from ImputeGAP
--------------------------
RMSE and MAE are not capped (ImputeGAP clips them at 100). Pearson returns
NaN for a constant input, as ImputeGAP does. The mutual information uses
ImputeGAP's exact discretisation — ``np.digitize`` on
``np.histogram_bin_edges(values, bins=10)`` of each array separately — so
the maximum of each array falls in an eleventh label, and scores are
comparable with ImputeGAP's (natural log, as ``sklearn.metrics.
mutual_info_score``, which is not needed here).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import norm, pearsonr

__all__ = [
    "rmse",
    "mae",
    "mutual_information",
    "pearson",
    "crps_from_samples",
    "crps_gaussian",
    "interval_coverage",
    "error_rate_split",
    "ErrorRates",
]


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def _float_array(name: str, value) -> np.ndarray:
    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric: {exc}") from exc


def _bool_mask(mask, shape: tuple, *, of: str = "Y_true") -> np.ndarray:
    m = np.asarray(mask)
    if m.shape != shape:
        raise ValueError(
            f"mask has shape {m.shape}, expected {shape} (the shape of {of})."
        )
    if m.dtype != bool:
        if not (np.issubdtype(m.dtype, np.number) and np.isin(m, (0, 1)).all()):
            raise ValueError(f"mask must be boolean (or 0/1), got dtype {m.dtype}.")
        m = m.astype(bool)
    return m


def _evaluated(Y_true, mask, *, required: bool):
    """Ground truth as float and the boolean index of the evaluated positions."""
    t = _float_array("Y_true", Y_true)
    if t.size == 0:
        raise ValueError("Y_true is empty.")
    if mask is None:
        if required:
            raise ValueError("mask is required (True = missing position to score).")
        sel = np.ones(t.shape, dtype=bool)
    else:
        sel = _bool_mask(mask, t.shape)
        if not sel.any():
            raise ValueError("mask selects no position: nothing to score.")
    sel = sel & np.isfinite(t)
    if not sel.any():
        raise ValueError(
            "No evaluated position has a finite ground truth: nothing to score."
        )
    return t, sel


def _prediction(name: str, value, shape: tuple, sel: np.ndarray, *,
                scalar_ok: bool = False) -> np.ndarray:
    """Values of a prediction at the evaluated positions, checked finite."""
    p = _float_array(name, value)
    if scalar_ok and p.ndim == 0:
        p = np.broadcast_to(p, shape)
    if p.shape != shape:
        raise ValueError(
            f"{name} has shape {p.shape}, expected {shape} (the shape of Y_true)"
            + (" or a scalar." if scalar_ok else ".")
        )
    vals = p[sel]
    bad = int((~np.isfinite(vals)).sum())
    if bad:
        raise ValueError(
            f"{name} has {bad} non-finite value(s) at evaluated positions."
        )
    return vals


def _pair(Y_true, Y_hat, mask):
    t, sel = _evaluated(Y_true, mask, required=True)
    return t[sel], _prediction("Y_hat", Y_hat, t.shape, sel)


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------


def rmse(Y_true, Y_hat, mask) -> float:
    """Root mean squared error over the masked positions (no cap)."""
    t, h = _pair(Y_true, Y_hat, mask)
    return float(np.sqrt(np.mean((t - h) ** 2)))


def mae(Y_true, Y_hat, mask) -> float:
    """Mean absolute error over the masked positions (no cap)."""
    t, h = _pair(Y_true, Y_hat, mask)
    return float(np.mean(np.abs(t - h)))


def _discretise(values: np.ndarray, bins: int) -> np.ndarray:
    return np.digitize(values, bins=np.histogram_bin_edges(values, bins=bins))


def mutual_information(Y_true, Y_hat, mask, *, bins: int = 10) -> float:
    """Mutual information (nats) between truth and imputation on the masked positions.

    Each array is discretised on its own ``bins`` equal-width bins
    (ImputeGAP's rule, see the module docstring), then
    ``I = sum p(a, b) log(p(a, b) / (p(a) p(b)))`` over the joint labels.
    """
    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError(f"bins must be an integer >= 1, got {bins!r}.")
    t, h = _pair(Y_true, Y_hat, mask)
    _, a = np.unique(_discretise(t, bins), return_inverse=True)
    _, b = np.unique(_discretise(h, bins), return_inverse=True)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a.ravel(), b.ravel()), 1.0)
    joint /= joint.sum()
    outer = np.outer(joint.sum(axis=1), joint.sum(axis=0))
    nz = joint > 0
    mi = float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))
    return max(mi, 0.0)


def pearson(Y_true, Y_hat, mask) -> float:
    """Pearson correlation on the masked positions; NaN when either side is constant."""
    t, h = _pair(Y_true, Y_hat, mask)
    if t.size < 2 or np.ptp(t) == 0 or np.ptp(h) == 0:
        return float("nan")
    return float(pearsonr(t, h)[0])


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------


def crps_from_samples(Y_true, samples, *, mask=None) -> float:
    """Mean CRPS estimated from predictive draws — unbiased energy form.

    For each evaluated position with truth ``y`` and draws ``x_1 … x_M``
    (``M >= 2``)::

        CRPS = (1/M) sum_i |x_i - y|  -  1/(2 M (M-1)) sum_{i != j} |x_i - x_j|

    the "fair" estimator of ``E|X - y| - E|X - X'| / 2``, computed in
    ``O(M log M)`` by sorting. It is unbiased for the CRPS of the predictive
    law, so a single position can score slightly below zero.

    ``samples`` has shape ``Y_true.shape + (M,)`` — the draws on the last
    axis. Returns the mean over evaluated positions.
    """
    t, sel = _evaluated(Y_true, mask, required=False)
    s = _float_array("samples", samples)
    if s.ndim != t.ndim + 1 or s.shape[:-1] != t.shape:
        raise ValueError(
            f"samples has shape {s.shape}, expected {t.shape} + (M,) — "
            f"the draws on the last axis."
        )
    M = s.shape[-1]
    if M < 2:
        raise ValueError(f"crps_from_samples needs at least 2 draws, got M={M}.")
    x = s[sel]                                   # (k, M)
    bad = int((~np.isfinite(x)).any(axis=1).sum())
    if bad:
        raise ValueError(
            f"samples has non-finite draws at {bad} evaluated position(s)."
        )
    y = t[sel]
    term1 = np.mean(np.abs(x - y[:, None]), axis=1)
    weights = 2.0 * np.arange(1, M + 1) - M - 1  # sum_{i<j} x_(j) - x_(i)
    term2 = np.sort(x, axis=1) @ weights / (M * (M - 1))
    return float(np.mean(term1 - term2))


def crps_gaussian(Y_true, mean, sd, *, mask=None) -> float:
    """Mean CRPS of Gaussian predictive laws ``N(mean, sd**2)``, closed form.

    ``CRPS = sd [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)]`` with
    ``z = (y - mean) / sd`` (Gneiting & Raftery 2007, JASA 102:359–378,
    doi:10.1198/016214506000001437). ``mean`` and ``sd`` have the shape of
    ``Y_true`` or are scalars; ``sd`` must be > 0 at evaluated positions.
    """
    t, sel = _evaluated(Y_true, mask, required=False)
    mu = _prediction("mean", mean, t.shape, sel, scalar_ok=True)
    sigma = _prediction("sd", sd, t.shape, sel, scalar_ok=True)
    if np.any(sigma <= 0):
        raise ValueError("sd must be > 0 at every evaluated position.")
    z = (t[sel] - mu) / sigma
    crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z)
                    - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def interval_coverage(Y_true, lo, hi, *, mask=None) -> float:
    """Share of evaluated positions whose truth lies in the closed interval ``[lo, hi]``.

    ``lo`` and ``hi`` have the shape of ``Y_true`` or are scalars;
    ``lo > hi`` at an evaluated position raises.
    """
    t, sel = _evaluated(Y_true, mask, required=False)
    a = _prediction("lo", lo, t.shape, sel, scalar_ok=True)
    b = _prediction("hi", hi, t.shape, sel, scalar_ok=True)
    if np.any(a > b):
        raise ValueError(
            f"lo > hi at {int((a > b).sum())} evaluated position(s)."
        )
    y = t[sel]
    return float(np.mean((a <= y) & (y <= b)))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class ErrorRates(NamedTuple):
    """Classification error split by missingness (rates in [0, 1])."""

    missing: float     #: error on positions whose observation is missing
    observed: float    #: error on observed positions (NaN if there are none)
    overall: float     #: error on all positions
    n_missing: int
    n_observed: int


def _labels(name: str, value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D label array, got shape {arr.shape}.")
    try:
        f = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must hold integer labels: {exc}") from exc
    if not np.all(np.isfinite(f)) or np.any(f != np.round(f)) or np.any(f < 0):
        raise ValueError(f"{name} must hold non-negative integer labels (no NaN).")
    return f.astype(int)


def error_rate_split(X_true, X_hat, mask, *, align: bool = True) -> ErrorRates:
    """Error rate on the missing positions, on the observed ones and overall.

    Parameters
    ----------
    X_true, X_hat : (N,) non-negative integer labels.
    mask : (N,) or (N, d) bool — True where the observation is missing. A
        multichannel row counts as missing as soon as one component is (the
        rule of the forward-backward with gaps).
    align : bool, default True — relabel ``X_hat`` by the permutation that
        minimises the *overall* error (Hungarian on the confusion matrix, as
        :func:`pmcprg.pmc.inference.error_rate`), then split. ``overall`` then
        equals ``error_rate(X_true, X_hat)``. One permutation serves both
        subsets: labels are a property of the model, not of the gaps.
        ``align=False`` compares labels as given.

    Returns
    -------
    ErrorRates(missing, observed, overall, n_missing, n_observed) —
    ``observed`` is NaN when every position is missing; a mask with no
    missing position raises.
    """
    xt = _labels("X_true", X_true)
    xh = _labels("X_hat", X_hat)
    if xt.shape != xh.shape:
        raise ValueError(f"X_true {xt.shape} and X_hat {xh.shape} must have the same shape.")
    N = xt.size
    m = np.asarray(mask)
    if m.ndim == 2 and m.shape[0] == N:
        m = _bool_mask(m, m.shape).any(axis=1)
    else:
        m = _bool_mask(m, (N,), of="X_true, or (N, d)")
    if not m.any():
        raise ValueError("mask selects no missing position: nothing to split.")

    if align:
        from scipy.optimize import linear_sum_assignment

        K = int(max(xt.max(), xh.max())) + 1
        C = np.zeros((K, K), dtype=int)
        np.add.at(C, (xt, xh), 1)
        rows, cols = linear_sum_assignment(-C)
        relabel = np.full(K, -1, dtype=int)
        relabel[cols] = rows
        xh = relabel[xh]

    wrong = xt != xh
    n_missing = int(m.sum())
    n_observed = N - n_missing
    return ErrorRates(
        missing=float(wrong[m].mean()),
        observed=float(wrong[~m].mean()) if n_observed else float("nan"),
        overall=float(wrong.mean()),
        n_missing=n_missing,
        n_observed=n_observed,
    )
