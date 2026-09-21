"""
pmcprg.copulas._weighted — the weighted copula engine (audit AUDIT_COPULES FR-12).

Weighted Kendall's τ, weighted pseudo-observations, weighted maximum
likelihood and weighted inversion of Kendall's τ, and weighted family
selection. One engine for two callers:

* the copula step of the ICE/SEM M-step (:mod:`pmcprg.pmc.ice`), which weights
  the pseudo-observations of the pair of states (i, j) by the posteriors
  ξ_n(i, j). ICE imports every function below under the name it had there;
* the public :meth:`pmcprg.copulas.CopulaVirt.fit` and
  :meth:`~pmcprg.copulas.CopulaVirt.fit_best` with ``weights=``.

The ICE functions (``_weighted_kendall_tau`` … ``_copula_placeholder``,
``_weighted_ecdf``) were moved here from ``pmcprg/pmc/ice.py`` without change,
so ICE's fits are bit-identical; the FR-11 wave-3 parity tests
(``pmcprg/tests/test_parity_weighted.py``) validate them against
pyvinecopulib and VineCopula, and the public API with them.

Contents
--------
* weighted Kendall's τ — :func:`_weighted_kendall_tau` (τ-b on ties, the
  former τ-a without), public :func:`weighted_kendall_tau`;
* weighted MLE — :func:`_weighted_mle_fit` (status and counts), ICE's
  :func:`_fit_copula_params` (a dict flagged ``FIT_FAILED_KEY`` on failure);
* weighted selection — :data:`_SCORE_FN`, :func:`_select_and_fit_copula`;
* weighted pseudo-observations — :func:`_weighted_ecdf`,
  :func:`weighted_pseudo_obs` (FR-7 a's convention);
* itau — :func:`_itau_fit`, :func:`_weighted_itau_fit` and the profile MLE of
  the other parameters, :func:`_profile_extras` (used unweighted by
  ``CopulaVirt.fit(method='tau')`` too);
* the public weighted fit — :func:`fit_weighted`, behind
  ``CopulaVirt.fit(..., weights=…)``.

Weights are validated by :func:`pmcprg.copulas._fit.validate_weights`, the
package's one check.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from pmcprg.copulas._base import (
    TAU_PAD_ABS as _BASE_TAU_PAD_ABS,
    TAU_PAD_REL as _BASE_TAU_PAD_REL,
    CopulaEnum,
    padded_tau_range,
)
from pmcprg.copulas._fit import (
    MLE_FAIL_PENALTY,
    _cvm_statistic,
    _fit_two_parameter_mle,
    _weighted_log_density_sum,
    validate_weights,
)
from pmcprg.numerics import EPS, MIN_POSITIVE, ONE_MINUS_EPS

logger = logging.getLogger(__name__)


# Failure sentinels for the bounded optimisers / weighted-MLE objectives,
# centralised so their magnitudes stay consistent (audit Q-8). They must be
# finite (so a *minimised* neg-objective can still rank a failed evaluation as
# "worst") and large enough to dominate any genuine (neg-)log-likelihood.
_OPT_FAIL_PENALTY: float = MLE_FAIL_PENALTY   # minimised neg-objective on failure (exception, NaN, ±∞)
_LOGLIK_FAIL:      float = -1e12   # returned by a log-likelihood on failure
_LOGPDF_FLOOR:     float = -1e8    # per-element floor for non-finite log-densities

# Key added to the dict returned by :func:`_fit_copula_params` — only when the
# fit failed — so that callers can see it (audit RB-3/FR-2). It is not a copula
# parameter: build copulas from :func:`_copula_kwargs`, which drops it.
FIT_FAILED_KEY: str = "fit_failed"

# Numerical-regularisation constants, centralised so the values stay consistent
# instead of being copy-pasted with subtly different floors (audit Q-14). The
# τ padding is defined once, in ``pmcprg.copulas._base``, for both standalone
# and ICE-driven fitting.
_TAU_PAD_REL:   float = _BASE_TAU_PAD_REL   # τ-bound padding, relative to the τ-span
_TAU_PAD_ABS:   float = _BASE_TAU_PAD_ABS   # τ-bound padding, absolute floor
_HUARD_N_GRID:  int   = 50     # coarse τ-grid size for the Huard Bayesian evidence
# Adaptive refinement of that grid around the likelihood peak (audit K-6): the
# peak narrows like 1/√n_eff while a fixed grid does not, and a 50-point grid
# on (−1, 1) mis-integrated the Gauss evidence by −0.28…+0.21 nat at
# n_eff = 1500 depending on where τ̂ fell between two nodes — a
# family-dependent artefact of the size of the prior-width effects the
# criteria variants are meant to isolate. The grid is refined around the
# maximum until the two neighbouring nodes lie within _HUARD_RESOLVE_NAT of
# it (step ≲ one posterior standard deviation, where the trapezoid rule on a
# peaked smooth integrand is spectrally accurate).
_HUARD_RESOLVE_NAT:    float = 0.5   # peak resolved when neighbours are this close
_HUARD_REFINE_HALF:    int   = 3     # refine ± this many local steps around the max
_HUARD_REFINE_POINTS:  int   = 61    # nodes in the refinement window (step/10)
_HUARD_MAX_REFINE:     int   = 4     # rounds; each divides the local step by 10


def _pad_tau_bounds(tau_min: float, tau_max: float) -> tuple[float, float]:
    """Return ``(lo, hi)`` — the τ-range padded away from its boundaries.

    The bounded optimisers must never evaluate a copula exactly at ``τ_min`` /
    ``τ_max`` (degenerate densities). Single source for the padding used by
    :func:`_weighted_mle_tau` and :func:`_fit_copula_params`.

    A **degenerate** range (``τ_max == τ_min``, i.e. the Product copula, whose
    only admissible τ is 0) is returned untouched: padding it inward yields the
    inverted interval ``(+pad, −pad)``, which every bounded optimiser rejects —
    so independence could never be selected, not even on exactly independent
    data.

    Delegates to :func:`pmcprg.copulas._base.padded_tau_range`, which
    ``CopulaVirt.fit(method='mle')`` uses too.
    """
    return padded_tau_range(tau_min, tau_max)


# Bounds and initial values for the *non*-τ parameters of multi-parameter
# copula families, keyed by ``CopulaVirt`` subclass name → ``{param:
# (lower, upper, init)}``. Consumed here and by the GUI copula dialog
# (``pmcprg.pmc.gui.dialogs``).
#
# Derived (audit A-2) from the single source of truth — the param-keyed
# ``EXTRA_PARAM_BOUNDS_BY_PARAM`` in :mod:`pmcprg.copulas._base` — plus each
# family's declared parameter set, so the per-param values live in exactly one
# place and standalone/ICE fitting cannot drift. Today this yields
# ``{"CopulaBB1": {"delta": …}, "CopulaStudent": {"df": …}}``.
def _build_extra_param_bounds() -> dict[str, dict[str, tuple[float, float, float]]]:
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for entry in CopulaEnum:
        cls_name = entry.value.CLASS_NAME
        extras = {
            p: EXTRA_PARAM_BOUNDS_BY_PARAM[p]
            for p in entry.value.PARAMETERS_SET_NAME
            if p in EXTRA_PARAM_BOUNDS_BY_PARAM
        }
        if extras:
            out[cls_name] = extras
    return out


EXTRA_PARAM_BOUNDS: dict[str, dict[str, tuple[float, float, float]]] = (
    _build_extra_param_bounds()
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_kendall_tau(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Kendall's τ on pseudo-observations (u, v): τ-a without ties, τ-b with them.

    Definition (weighted version of the Mann/Kendall coefficient)::

                     Σ_{i<j} w_i w_j · sign(u_i − u_j) · sign(v_i − v_j)
        τ  =  ───────────────────────────────────────────────────────────────
              √( Σ_{i<j} w_i w_j sign(u_i − u_j)² · Σ_{i<j} w_i w_j sign(v_i − v_j)² )

    — the **weighted τ-b**, the coefficient of pyvinecopulib (``wdm``) and of
    VineCopula (``TauMatrix``, ``BiCopEst``'s ``emptau``; audit FR-12). Each
    pair (i, j) is weighted by ``w_i w_j``, the natural choice when ``w`` are
    posterior memberships of the pair ``(i, j)`` of states. It is *not* the
    rank-position-weighted coefficient of ``scipy.stats.weightedtau``
    (weights that decay with the rank position), which answers a different
    question. With unit weights it is the τ-b of ``scipy.stats.kendalltau``.

    **Without ties** — no two points of positive weights share a u or a v,
    the case of continuous margins and so of ICE — both sums under the root
    are ``Σ_{i<j} w_i w_j`` and τ-b is the weighted τ-a ``num / Σ_{i<j} w_i
    w_j``. That case takes its own branch, the τ-a code as it always was, so
    its value is unchanged to the last bit. **With ties** a tied pair is
    neither concordant nor discordant, and τ-b divides by the weight of the
    pairs that are *not* tied in each margin instead of by every pair: on
    the 8 × 8 grid of the FR-11 parity data it is 4.6·10⁻² to 5.1·10⁻²
    larger than τ-a, and it is the coefficient whose inversion is itau for a
    tied sample (``test_parity_weighted.py`` compares it with both
    references). Ties that involve a point of zero weight carry no pair
    weight and leave the τ-a branch.

    Reference: Kendall, M. G. (1938). A new measure of rank correlation.
    *Biometrika* 30(1–2), 81–93; τ-b: Kendall, M. G. (1945). The treatment
    of ties in ranking problems. *Biometrika* 33(3), 239–251.

    Implementation: O(N log N) time, O(N) memory — Knight's (1966)
    merge-sort algorithm, weighted. The pairs strictly ordered in u (weight
    ``A``) are strictly concordant (``P``), strictly discordant (``Q``) or
    tied in v (``E``); pairs tied in u are in none of them. So

        num = P − Q = 2P − A + E,

    and a tie contributes nothing — exactly ``sign(0) = 0``. Sorted by u,
    then by v *descending*, a pair tied in u is never increasing in v, so
    ``P`` is the weight of the increasing pairs of that sequence of v-ranks
    (:func:`_weighted_increasing_pairs`). ``A`` and ``E`` are prefix sums
    over runs of equal values (sorted by u; by v then u), and the
    denominator is ``Σ_j w_j · Σ_{i<j} w_i`` — equal to
    ``((Σw)² − Σw²)/2`` without its cancellation when the weight sits on a
    few points. An all-equal column gives an exact 0. This replaces an N×N
    vectorisation (≈ 290 MB per array and ~98 % of the ICE run time at
    N = 6000); the two agree to rounding
    (``pmcprg/tests/test_weighted_kendall_tau.py``).

    Reference for the algorithm: Knight, W. R. (1966). A computer method for
    calculating Kendall's tau with ungrouped data. *Journal of the American
    Statistical Association* 61(314), 436–439.

    The τ-b denominators are the weights ``A_u`` and ``A_v`` of the pairs
    strictly ordered in u and in v — the same prefix sums as ``A``, taken
    over the runs of equal u and of equal v — rather than ``Σ w_i w_j`` minus
    the tied pairs, which would cancel; the root is taken as ``√A_u · √A_v``
    so that neither underflows with tiny weights.

    Returns ``0.0`` for degenerate inputs (N < 2, zero or non-finite total
    weight, zero pair weight, all-equal observations, …) so the caller can
    safely use the result as an optimiser initial value. A NaN in ``u`` or
    ``v`` gives NaN, as the N×N version did.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    n = u.size
    if n < 2:
        return 0.0

    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= MIN_POSITIVE:
        return 0.0

    # Σ_{i<j} w_i w_j, as Σ_j w_j · (w_0 + … + w_{j−1}).
    den = float(np.dot(w[1:], np.cumsum(w[:-1])))
    if den <= MIN_POSITIVE:
        return 0.0
    if np.isnan(u).any() or np.isnan(v).any():
        return float("nan")

    # Sorted by v then u: dense v-ranks, and E = Σ_b w_b · (weight earlier in
    # b's run of equal v, outside its run of equal (v, u)).
    by_vu = np.lexsort((u, v))
    v_new = _run_starts(v[by_vu])             # −0.0 == 0.0: one run
    vu_new = v_new | _run_starts(u[by_vu])
    rv = np.empty(n, dtype=np.int64)
    rv[by_vu] = np.cumsum(v_new) - 1
    w_vu = w[by_vu]
    before = _weight_before(w_vu)
    e = float(np.dot(w_vu, before[_run_start_index(vu_new)]
                     - before[_run_start_index(v_new)]))

    # Sorted by u then v descending: A = Σ_b w_b · (weight of earlier u-runs).
    by_u = np.lexsort((-rv, u))
    w_u = w[by_u]
    u_new = _run_starts(u[by_u])
    before_u = _weight_before(w_u)
    run_u = _run_start_index(u_new)
    a = float(np.dot(w_u, before_u[run_u]))
    p = _weighted_increasing_pairs(rv[by_u], w_u)
    if not (u_new.all() and v_new.all()):
        # Ties. Weight of the pairs tied in u, in v (weight earlier in the run).
        run_v = _run_start_index(v_new)
        tied_u = float(np.dot(w_u, before_u - before_u[run_u]))
        tied_v = float(np.dot(w_vu, before - before[run_v]))
        if tied_u != 0.0 or tied_v != 0.0:
            # τ-b: over √(A_u · A_v), the pairs strictly ordered in u and in v.
            a_v = float(np.dot(w_vu, before[run_v]))
            if not (a > MIN_POSITIVE and a_v > MIN_POSITIVE):
                return 0.0                    # a column tied throughout
            return float(np.clip((2.0 * p - a + e) / (math.sqrt(a) * math.sqrt(a_v)),
                                 -1.0, 1.0))
    return float(np.clip((2.0 * p - a + e) / den, -1.0, 1.0))


def _run_starts(x_sorted: np.ndarray) -> np.ndarray:
    """Boolean mask, True where a run of equal values of ``x_sorted`` begins."""
    starts = np.empty(x_sorted.size, dtype=bool)
    starts[:1] = True
    np.not_equal(x_sorted[1:], x_sorted[:-1], out=starts[1:])
    return starts


def _run_start_index(starts: np.ndarray) -> np.ndarray:
    """For each element, the index where its run begins."""
    idx = np.where(starts, np.arange(starts.size), 0)
    return np.maximum.accumulate(idx)


def _weight_before(w: np.ndarray) -> np.ndarray:
    """``before[k] = w[0] + … + w[k−1]`` (``before[0] = 0``)."""
    before = np.empty(w.size)
    before[:1] = 0.0
    np.cumsum(w[:-1], out=before[1:])
    return before


def _weighted_increasing_pairs(r: np.ndarray, w: np.ndarray) -> float:
    """``Σ_{a<b, r[a] < r[b]} w[a] w[b]`` for an integer sequence ``r``.

    Bottom-up merge sort, one vectorised pass per level (⌈log₂ N⌉ levels).
    At the level of half-width ``s`` every pair ``a < b`` whose positions
    first share a block of width ``2s`` has ``a`` in the left half and ``b``
    in the right one; each half is already sorted by ``r`` (previous
    level). A stable sort of the block on ``(r, right before left)`` then
    puts, before each right element, exactly the left elements of strictly
    smaller ``r``: their weight is a prefix sum inside the block. The sort
    only merges two sorted runs per block, which NumPy's stable sort
    (timsort on int64) does in linear time, hence O(N log N) overall. ``r``
    should be ranks (a range of order N) so the block keys fit in int64.
    """
    r = np.asarray(r, dtype=np.int64).ravel()
    w = np.asarray(w, dtype=float).ravel()
    n = r.size
    if n < 2:
        return 0.0
    r = r - r.min()
    span = 2 * int(r.max()) + 2               # > 2·r + 1: blocks never overlap
    pos = np.arange(n, dtype=np.int64)
    before = np.empty(n + 1)
    before[0] = 0.0
    total = 0.0
    s = 1
    while s < n:
        block = pos // (2 * s)
        right = (pos // s) & 1                # 1 ⇔ right half of its block
        key = block * span + 2 * r + (1 - right)
        order = np.argsort(key, kind="stable")
        # A block keeps its positions [2s·b, 2s·(b+1)) after the sort.
        r, w, right = r[order], w[order], right[order]
        w_right = w * right                   # exact: w·1 = w, w·0 = 0
        np.cumsum(w - w_right, out=before[1:])  # left weights only
        # Right element at k: left weight of its block sorted before it.
        total += float(np.dot(w_right, before[1:] - before[block * (2 * s)]))
        s *= 2
    return total


def _resolve_candidate(short_name: str):
    """Return (CopulaEnum_entry, class) for a SHORT_NAME. Raises if not found."""
    entry = CopulaEnum.from_short_name(short_name)
    if entry is None:
        available = [c.value.SHORT_NAME for c in CopulaEnum.available()]
        raise ValueError(
            f"Candidate copula {short_name!r} not found. "
            f"Available SHORT_NAMEs: {available}"
        )
    return entry, entry.klass


def _copula_kwargs(params: dict) -> dict:
    """Constructor keyword arguments from a fitted-parameter dict.

    Drops :data:`FIT_FAILED_KEY`, which :func:`_fit_copula_params` adds when a
    fit failed and which no copula accepts.
    """
    return {k: v for k, v in params.items() if k != FIT_FAILED_KEY}


def _weighted_mle_tau(
    cls,
    entry,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    tau_init: float | None = None,
    *,
    return_status: bool = False,
):
    """
    Weighted MLE estimate of τ for copula class `cls` on pseudo-observations (u, v).

    Returns the τ̂ that maximises  Σ_n w_n · log c_{τ}(u_n, v_n) — or, with
    ``return_status=True``, ``(τ̂, reason)`` where ``reason`` is ``None`` on
    success and a message when the search failed.

    The objective follows :func:`pmcprg.copulas._fit._weighted_log_density_sum`:
    points of zero weight are ignored whatever their log-density, and a
    non-finite log-density at a positive weight — like an exception — gives
    the failure penalty (audit RB-7). Before, ``np.dot`` turned 0·(−∞) into
    NaN, on which every comparison of Brent's method is False.

    ``tau_init`` is an optional data-aware starting guess (e.g. weighted
    Kendall's τ) used as the fallback if the bounded optimiser fails. The
    ``minimize_scalar(method="bounded")`` Brent search is bracketed and
    therefore does not consume an initial value, but it is still useful
    as a safety-net so we don't fall back to a generic Pearson moment.

    :func:`_weighted_mle_tau_search` is the same search with its iteration
    and evaluation counts.
    """
    tau_hat, reason, _, _ = _weighted_mle_tau_search(cls, entry, u, v, weights, tau_init)
    return (tau_hat, reason) if return_status else tau_hat


def _weighted_mle_tau_search(cls, entry, u, v, weights, tau_init=None):
    """:func:`_weighted_mle_tau` — ``(τ̂, reason, n_iter, n_eval)``, the search's counts added."""
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    lo, hi = _pad_tau_bounds(tau_min, tau_max)

    w  = np.asarray(weights, dtype=float)
    w  = w / (w.sum() + MIN_POSITIVE)
    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))

    def _neg_wll(tau: float) -> float:
        try:
            # Beyond the family's reachable |τ| (Frank) the density is that of
            # the bound: evaluate there, silently (F1).
            cop = cls(tau_k=entry.reachable_tau(float(tau)))
            ll  = _weighted_log_density_sum(cop.logpdf_array(uv), w)
        except Exception as exc:
            logger.debug(
                "_neg_wll: copula eval failed at tau=%.6f — %s", tau, exc,
                exc_info=True,
            )
            return _OPT_FAIL_PENALTY
        return -ll if np.isfinite(ll) else _OPT_FAIL_PENALTY

    reason = None
    n_iter = n_eval = 0
    try:
        res = minimize_scalar(_neg_wll, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-6})
        n_iter, n_eval = int(getattr(res, "nit", 0)), int(getattr(res, "nfev", 0))
        tau_hat = float(np.clip(res.x, lo, hi))
        if not res.fun < _OPT_FAIL_PENALTY:
            reason = "the likelihood is not finite at any τ the search evaluated"
    except Exception as exc:
        reason = f"bounded search raised {type(exc).__name__}: {exc}"
        tau_hat = None

    if reason is not None:
        # Prefer the supplied data-aware estimate (weighted Kendall) when
        # available; otherwise compute it on the fly. Kendall's τ is the
        # natural concordance measure for copulas and is far closer to the
        # true MLE than the previous Pearson-based fallback.
        if tau_init is None or not np.isfinite(tau_init):
            tau_init = _weighted_kendall_tau(u, v, weights)
        logger.warning(
            "Weighted MLE of τ failed for %s (%s; Σw=%.4g) — using the "
            "weighted Kendall estimate τ=%.6g.",
            cls.__name__, reason, float(np.sum(weights)), float(np.clip(tau_init, lo, hi)),
        )
        tau_hat = float(np.clip(tau_init, lo, hi))
    tau_hat = entry.reachable_tau(tau_hat)     # the τ the copula uses (F1)
    return tau_hat, reason, n_iter, n_eval


def _weighted_log_likelihood(
    cls,
    params: dict,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> float:
    """**Total** weighted log-likelihood ``Σ_n w_n · log c(u_n, v_n)``.

    Deliberately *not* divided by ``Σw``. Ranking candidates on likelihood
    alone is invariant to that choice (``Σw`` is fixed within a block), but
    the penalised criteria are not: AIC's ``2k`` and BIC's ``k·log n`` are
    calibrated against a *sum*. Comparing them against a per-observation mean
    over-penalises by a factor of ``Σw`` — the threshold a 2-parameter family
    had to clear became ~1 nat *per observation* instead of ~1/n, which no
    real family ever clears, so Student and BB1 were effectively unselectable.

    Points of zero weight are ignored whatever their log-density (0·(−∞) used
    to give NaN); a non-finite log-density at a positive weight gives ``−∞``
    (audit RB-7, :func:`pmcprg.copulas._fit._weighted_log_density_sum`). An
    exception while building or evaluating the copula still gives
    ``_LOGLIK_FAIL``.
    """
    try:
        cop = cls(**_copula_kwargs(params))
        uv  = np.column_stack((np.asarray(u, dtype=float),
                               np.asarray(v, dtype=float)))
        return _weighted_log_density_sum(cop.logpdf_array(uv), weights)
    except Exception as exc:
        logger.debug(
            "_weighted_log_likelihood: failed for params=%s — %s",
            params, exc, exc_info=True,
        )
        return _LOGLIK_FAIL


@dataclass
class WeightedFit:
    """One weighted fit of the engine: parameters and the optimiser's status.

    ``params`` — constructor keyword arguments (``tau_k`` and the extra
    parameters), always constructible. ``converged`` — the optimiser's
    verdict (``True`` for a moment estimate); ``message`` — what it said;
    ``n_iter`` / ``n_eval`` — its iterations and likelihood evaluations (0
    for a moment estimate). ``tau_start`` — the weighted τ the joint MLE of a
    multi-parameter family started from (``None`` otherwise).
    """
    params: dict
    converged: bool
    message: str
    n_iter: int = 0
    n_eval: int = 0
    tau_start: float | None = None


def _weighted_mle_fit(cls, entry, u, v, weights) -> WeightedFit:
    """Weighted MLE for **all** free parameters of `cls` — the engine of :func:`_fit_copula_params`.

    The same computation as ICE's M-step, returned with the optimiser's
    status and counts instead of ICE's ``FIT_FAILED_KEY`` flag.
    """
    # Key by the registry's CLASS_NAME (the contract source) rather than
    # cls.__name__ — they coincide, but EXTRA_PARAM_BOUNDS is built from
    # CLASS_NAME, so use the same source consistently (audit A-9).
    extras = EXTRA_PARAM_BOUNDS.get(entry.value.CLASS_NAME, {})

    # Data-aware initial value: weighted Kendall's τ, clipped to the
    # family's valid τ range. For one-parameter families the bounded Brent
    # search ignores it — it is only the fallback when that search fails —
    # so τ̂ does not depend on it. For two-parameter families it is the
    # start of the joint optimiser, whose result depends on it at the
    # optimiser's tolerance: in an ICE replay of Exp. 3, rounding-level
    # changes of τ_emp moved the fitted Student parameters and, through the
    # posteriors, every τ̂ of those runs by < 1e-6, no family decision.
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    lo, hi  = _pad_tau_bounds(tau_min, tau_max)
    tau_emp = float(np.clip(_weighted_kendall_tau(u, v, weights), lo, hi))

    # 1-D case — fall back to the existing dedicated routine, supplying
    # the data-aware estimate as a robust fallback / hint.
    if not extras:
        tau_hat, reason, n_iter, n_eval = _weighted_mle_tau_search(
            cls, entry, u, v, weights, tau_init=tau_emp,
        )
        return WeightedFit(
            {"tau_k": float(tau_hat)}, converged=reason is None,
            message=(reason if reason is not None else
                     "bounded Brent search on τ converged (xatol 1e-6)"),
            n_iter=n_iter, n_eval=n_eval,
        )

    # Two-parameter case — joint optimisation started at (τ_emp, default extra).
    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))
    fit = _fit_two_parameter_mle(cls, entry, uv, np.asarray(weights, dtype=float), tau_emp)
    return WeightedFit(dict(fit.params), converged=fit.converged, message=fit.message,
                       n_iter=fit.n_iter, n_eval=fit.n_eval, tau_start=tau_emp)


def _fit_copula_params(
    cls,
    entry,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """Weighted MLE for **all** free parameters of `cls`.

    For 1-parameter families this is just τ̂ (delegated to ``_weighted_mle_tau``).
    For two-parameter families (BB1, Student) it runs the joint optimiser
    :func:`pmcprg.copulas._fit._fit_two_parameter_mle` — the same one as
    ``CopulaVirt.fit(method='mle')`` — with the bounds of
    ``EXTRA_PARAM_BOUNDS``, in coordinates whose box is the admissible set
    (Student ``(atanh τ, 1/ν)``; BB1 ``(log θ, log δ)`` then ``(θ, log δ)``)
    and with scale-free
    stopping on the total weighted log-likelihood (audit RB-3, RB-8). The
    former (τ, extra) L-BFGS-B with ``ftol = 1e-7`` on a Σw-normalised
    objective stopped at its start value: Student ν = 3.9998 for a true 2.5
    at τ = 0.95, BB1 δ back at 1.5 after an "ABNORMAL" stop at iteration 0.

    The computation is :func:`_weighted_mle_fit`'s, which the public
    ``CopulaVirt.fit(method='mle', weights=…)`` also runs.

    Returns
    -------
    dict — keyword arguments suitable for ``cls(**dict)``, always containing
    ``tau_k`` and any extra parameters declared in the registry. **When the
    fit failed** (optimiser not converged, or no finite likelihood) the dict
    also carries ``FIT_FAILED_KEY: True`` and the parameters are the best
    point found; a WARNING names the family, the reason and the values.
    Build copulas from :func:`_copula_kwargs` to drop that key. (The public
    fit reports the same status as ``FitResult.converged`` / ``failed`` /
    ``message``.)
    """
    wf = _weighted_mle_fit(cls, entry, u, v, weights)
    out = dict(wf.params)
    if not wf.converged:
        if wf.tau_start is not None:
            logger.warning(
                "ICE: joint MLE of %s did not converge (%s; Σw=%.4g, start τ=%.4g) "
                "— keeping the best point found %s.",
                cls.__name__, wf.message, float(np.sum(weights)), wf.tau_start, wf.params,
            )
        out[FIT_FAILED_KEY] = True
    return out


# ---------------------------------------------------------------------------
# Selection criteria — score functions for the candidate copula loop
# ---------------------------------------------------------------------------
#
# Each criterion returns a *score* expressed so that "higher is better"
# (so the same ``arg max`` loop in :func:`_select_and_fit_copula` works
# regardless of the criterion). Convention:
#
#   * ``mle``   → weighted log-likelihood (default; what was used pre-v0.5)
#   * ``aic``   → −AIC = 2·logL − 2k
#   * ``bic``   → −BIC = 2·logL − k·log n
#   * ``huard`` → log of Huard et al. (2006) Bayesian evidence (CSDA-2013 Eq. 20)
#   * ``cvm``   → −CvM statistic (smaller CvM ⇒ better fit ⇒ higher score)
#   * ``xvcic`` → cross-validated out-of-fold weighted log-likelihood
#
# Provenance (audit K-16):
#   * ``huard`` is DerrodePieczynski_CSDA2013 Eq. 20 (Derrode & Pieczynski
#     2013, CSDA) — the criterion the paper actually used for copula
#     selection.
#   * ``mle``   is the "PLM" rule of DerrodePieczynski_SP2016 Example 3.3,
#     Eq. (11) (Derrode & Pieczynski 2016, Signal Processing);
#     DerrodePieczynski_SP2016 reports Huard's criterion "less efficient
#     than PLM" in its experiments.
#   * ``aic``, ``bic``, ``cvm``, ``xvcic``, ``huard_common``, ``huard_global``
#     are package additions with no counterpart in either paper.
#
# All functions share the signature
#     score(cls, entry, params, u, v, weights) -> float
# where ``params`` is the dict produced by :func:`_fit_copula_params` for the
# candidate (so we factor out the τ̂-fitting once per candidate).

SELECTION_CRITERIA = ("mle", "aic", "bic", "huard", "huard_common", "huard_global", "cvm", "xvcic")
DEFAULT_SELECTION_CRITERION = "mle"

# --- xv-CIC (cross-validated criterion) ------------------------------------
_XVCIC_N_FOLDS: int   = 5      # never leave-one-out: the cost is ×N
_XVCIC_PURGE:   int   = 1      # pairs n and n+1 share Y_{n+1} — must be ≥ 1
_XVCIC_MIN_W:   float = 20.0   # below this Σw the CV estimate is pure noise


def _effective_n(weights: np.ndarray) -> float:
    """Effective sample size ``Σw`` of a ξ-weighted block.

    A pair ``(i, j)`` carrying a tenth of the posterior mass represents ~N/10
    observations, not N — which is what the sequence length ``len(u)`` would
    claim.
    """
    return float(np.sum(np.asarray(weights, dtype=float)))


def _score_mle(cls, entry, params, u, v, weights) -> float:
    """Weighted log-likelihood — the v0.4 baseline."""
    return _weighted_log_likelihood(cls, params, u, v, weights)


def _score_aic(cls, entry, params, u, v, weights) -> float:
    """``−AIC`` = 2·log L − 2 k.  Penalises parameter count."""
    ll = _weighted_log_likelihood(cls, params, u, v, weights)
    k  = getattr(cls, "n_params", 1)
    return 2.0 * ll - 2.0 * k


def _score_bic(cls, entry, params, u, v, weights) -> float:
    """``−BIC`` = 2·log L − k·log n_eff.  Stronger penalty than AIC for large n.

    ``n_eff`` is the block's effective sample size ``Σξ`` (see
    :func:`_effective_n`), not the sequence length: the log-likelihood being
    penalised only accumulates over the mass actually assigned to this pair.
    """
    ll = _weighted_log_likelihood(cls, params, u, v, weights)
    k  = getattr(cls, "n_params", 1)
    return 2.0 * ll - k * np.log(max(_effective_n(weights), 1.0))


def _weighted_cvm_statistic(uv: np.ndarray, weights: np.ndarray, cop) -> float:
    """ξ-weighted Cramér–von Mises statistic on pseudo-observations.

    Thin delegator to :func:`pmcprg.copulas._fit._cvm_statistic` with
    ``weights=`` — the weighted empirical copula and the weighted outer
    L²-sum are defined next to their unweighted originals so the two
    formulas cannot drift apart (audit Q-20). Weighting keeps the ``cvm``
    selection criterion consistent with the ξ-weighted M-step (every other
    criterion — mle/aic/bic/huard — already uses the pair-posterior
    weights); an unweighted call would let observations that barely belong
    to pair ``(i, j)`` drive family selection (audit N-6). ``O(N²)``.
    """
    return _cvm_statistic(
        np.asarray(uv, dtype=float), cop, np.asarray(weights, dtype=float),
    )


def _score_cvm(cls, entry, params, u, v, weights) -> float:
    """``−`` (ξ-weighted) Cramér–von Mises statistic on the pseudos.

    The CvM statistic measures the L²-distance between the empirical and
    fitted copula CDFs and does not depend on the family's parameter count;
    it is therefore a non-likelihood-based criterion that complements
    :func:`_score_mle`. The empirical copula is **weighted** by the ICE
    pair-posteriors so the criterion is consistent with the weighted M-step
    (see :func:`_weighted_cvm_statistic`).

    The unweighted original is the ``S_n`` of Genest, Rémillard & Beaudoin
    (2009, §2). The ξ-weighted form used here is **package-specific**: the
    empirical copula is the weighted one, the pseudo-observations are the
    parametric ``F_i(Y)`` rather than ranks (so the ``n/(n+1)`` rescaling of
    the rank form does not apply), and there is no factor ``n`` — a
    per-pair, per-family score, not a test statistic with a tabulated law.

    Reference: Genest, C., Rémillard, B. & Beaudoin, D. (2009).
    Goodness-of-fit tests for copulas: A review and a power study.
    *Insurance: Mathematics and Economics* 44(2), 199–213.
    """
    try:
        cop = cls(**_copula_kwargs(params))
        # The CvM statistic has no closed form for some families that lack
        # a CDF (e.g. Student, which raises NotImplementedError); fall back to a
        # low (worst) score on any failure here.
        cop.cdf([0.5, 0.5])
    except Exception:
        return -np.inf
    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))
    try:
        S = _weighted_cvm_statistic(uv, weights, cop)
    except Exception:
        return -np.inf
    return -S


def _huard_log_evidence(
    cls,
    entry,
    params,    # fixes multi-parameter extras (Student df, BB1 δ) below
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    n_grid: int = _HUARD_N_GRID,
    tau_range: tuple[float, float] | None = None,
) -> float:
    """Huard et al. (2006) Bayesian model evidence — CSDA-2013 Eq. (20).

    The criterion picks the family that maximises the integrated likelihood
    over τ ∈ [τ_min, τ_max] under a uniform prior::

        s(r) = (1/(τ_max − τ_min)) ∫ ∏_n c_r(u_n, v_n; τ) dτ

    We compute this in log space by the trapezoid rule on a τ-grid that
    starts uniform (``n_grid`` nodes) and is refined around the likelihood
    maximum until it resolves the peak (see ``_HUARD_RESOLVE_NAT``), then
    normalised by the prior width: ``log s(r) = log ∫ L_r dτ − log(τ_max −
    τ_min)``. A fixed grid could not follow the peak as it narrows with
    ``n_eff`` and gave a family-dependent error of ±0.25 nat at
    ``n_eff = 1500`` (audit K-6); measured against ``scipy.integrate.quad``
    the adaptive scheme is exact to < 0.01 nat there.

    Notes
    -----
    * Degenerate ranges (Product copula has ``τ_max == τ_min``) are handled
      by returning the single-point log-likelihood.
    * For multi-parameter families (Student `df`, BB1 `δ`), we fix the extra
      parameter at the MLE-fitted value passed via ``params`` and integrate
      over τ only — a pragmatic choice that matches CSDA-2013's intent (the
      paper considers single-parameter copulas; Student `df` is taken as
      known).
    * ``tau_range`` overrides the family's own τ-range. Passing the *same*
      range for every candidate removes the differential prior width, which
      is what the ``huard_common`` criterion does: with a uniform prior on
      each family's own range, a narrow-support family concentrates more
      prior mass and is rewarded for it, so a Huard-versus-likelihood
      comparison mixes "Bayesian averaging helps" with "this family's
      support happens to be narrow and to contain the truth". Integrating
      every candidate over a common support separates the two.
    """
    tau_min, tau_max = tau_range if tau_range is not None else entry.value.TAU_MIN_MAX
    span = tau_max - tau_min
    # Huard-specific padding: the τ-span is clamped to ≥ 1 so a wide-range
    # family is not over-padded, and the boundary is only trimmed when the span
    # is comfortably larger than the pad (degenerate Product-copula range → no
    # trim). Shares the relative/absolute constants with _pad_tau_bounds (Q-14).
    pad  = max(_TAU_PAD_REL * max(span, 1.0), _TAU_PAD_ABS)

    lo = tau_min + pad
    hi = tau_max - pad if span > 2 * pad else tau_max

    # Single-point case (Product copula).
    if hi <= lo:
        return _weighted_log_likelihood(cls, params, u, v, weights)

    extras = {k: v_ for k, v_ in _copula_kwargs(params).items() if k != "tau_k"}

    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))
    # Total (un-normalised) weighted log-likelihood at each grid τ: the Occam
    # factor that makes this an *evidence* only exists on that scale. Averaging
    # the log-density first flattens L(τ) so the marginalisation gives back
    # roughly the maximum, and the criterion stops penalising flexibility.
    w  = np.asarray(weights, dtype=float)

    def log_lik(tau: float) -> float:
        try:
            cop = cls(tau_k=entry.reachable_tau(float(tau)), **extras)   # same density (F1)
            log_pdfs = cop.logpdf_array(uv)
            log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, _LOGPDF_FLOOR)
            return float(np.dot(w, log_pdfs))
        except Exception as exc:                             # pragma: no cover
            logger.debug("Huard: cls=%s τ=%.4f failed (%s)", cls.__name__, tau, exc)
            return -np.inf

    taus = np.linspace(lo, hi, n_grid)
    vals = np.array([log_lik(t) for t in taus])
    if taus.size == 1:
        return float(vals[0])

    # Refine around the maximum until the grid resolves the peak (audit K-6).
    # A neighbour that failed (−inf) counts as unresolved and triggers a
    # refinement too, so a failure next to the peak cannot hide it.
    for _ in range(_HUARD_MAX_REFINE):
        finite = np.isfinite(vals)
        if not finite.any():
            break
        k = int(np.argmax(np.where(finite, vals, -np.inf)))
        n = taus.size
        drop_l = vals[k] - vals[k - 1] if k > 0 else 0.0
        drop_r = vals[k] - vals[k + 1] if k < n - 1 else 0.0
        if max(drop_l, drop_r) <= _HUARD_RESOLVE_NAT:
            break
        step = min(taus[k] - taus[k - 1] if k > 0 else np.inf,
                   taus[k + 1] - taus[k] if k < n - 1 else np.inf)
        a = max(lo, taus[k] - _HUARD_REFINE_HALF * step)
        b = min(hi, taus[k] + _HUARD_REFINE_HALF * step)
        fine = np.linspace(a, b, _HUARD_REFINE_POINTS)
        new  = fine[np.min(np.abs(fine[:, None] - taus[None, :]), axis=1) > 1e-12]
        if new.size == 0:
            break
        taus = np.concatenate([taus, new])
        vals = np.concatenate([vals, [log_lik(t) for t in new]])
        order = np.argsort(taus)
        taus, vals = taus[order], vals[order]

    # Trapezoid rule in log space, normalised by the *prior width* — not by the
    # surviving-node count — so a family whose likelihood fails at some nodes
    # (→ -inf, treated as exp = 0) is not silently re-normalised onto a
    # sub-interval and given inflated evidence (audit N-7).
    dw = np.empty_like(taus)
    dw[0]    = 0.5 * (taus[1] - taus[0])
    dw[-1]   = 0.5 * (taus[-1] - taus[-2])
    dw[1:-1] = 0.5 * (taus[2:] - taus[:-2])
    finite = np.isfinite(vals)
    if not finite.any():
        return -np.inf
    m = vals[finite].max()
    return float(m + np.log(np.sum(np.exp(vals[finite] - m) * dw[finite]))
                 - np.log(hi - lo))


def _score_huard(cls, entry, params, u, v, weights) -> float:
    """Convenience wrapper: dispatch-compatible signature for Huard.

    Forwards to :func:`_huard_log_evidence` with the default τ-grid
    (50 points). Kept separate so the math primitive
    :func:`_huard_log_evidence` accepts ``n_grid`` for tests / advanced
    use, while the dispatch table stays homogeneous.
    """
    return _huard_log_evidence(cls, entry, params, u, v, weights)


def _score_huard_common(cls, entry, params, u, v, weights, *, tau_range=None) -> float:
    """Huard evidence integrated over a τ-support **shared** by all candidates.

    Same estimator as :func:`_score_huard`, but the uniform prior is placed
    on one common range instead of each family's own. Its purpose is
    diagnostic: with per-family supports, a narrow-support family
    concentrates more prior mass and is rewarded for it, so comparing Huard
    against a likelihood criterion conflates "integrating over τ is robust
    on sparse cells" with "this family's declared range happens to be narrow
    and to contain the truth". Running both variants separates the two.

    ``tau_range`` is supplied by :func:`_select_and_fit_copula`, which knows
    the candidate set; falling back to the family's own range makes this
    identical to ``huard``.
    """
    return _huard_log_evidence(
        cls, entry, params, u, v, weights, tau_range=tau_range,
    )


def _score_huard_global(cls, entry, params, u, v, weights, *, prior_width=None) -> float:
    """Huard evidence under a prior of common *mass* rather than common support.

    Eq.~(20) puts a uniform prior on each family's own τ-range Λ_r, so its
    log-evidence is ``log ∫_{Λ_r} L_r dτ − log Λ_r``. Placing instead a single
    un-normalised uniform prior on a global range Λ_g, truncated to each
    family's domain, gives ``log ∫_{Λ_r} L_r dτ − log Λ_g``. The two therefore
    differ by an exact, data-independent constant per family::

        score_global(r) = score_Eq20(r) − log(Λ_g / Λ_r)

    so Eq.~(20) hands every candidate a free bonus of ``−log Λ_r``: +1.80 nat
    to a family declaring ``[0, 0.165]``, 0 to one declaring ``[0, 1]``, −0.69
    to one declaring ``[−1, 1]``. On a sparse block those constants are the
    same order as the log-likelihood differences between neighbouring
    families, which is enough to decide the selection.

    Unlike :func:`_score_huard_common`, which integrates every candidate over
    the *intersection* of the declared ranges, this variant leaves each
    family's integration domain intact — no family is dragged into a region
    where it cannot compete — and equalises only the prior normalisation.
    That makes it the cleaner of the two manipulations.
    """
    base = _huard_log_evidence(cls, entry, params, u, v, weights)
    if prior_width is None:
        return base
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    own = max(float(tau_max) - float(tau_min), MIN_POSITIVE)
    return base + float(np.log(own / max(float(prior_width), MIN_POSITIVE)))


def _contiguous_weight_balanced_folds(
    weights: np.ndarray, n_folds: int,
) -> list[tuple[int, int]]:
    """Split ``[0, n)`` into contiguous blocks of roughly equal ``Σw``.

    Contiguous — not random — because the pseudo-observations form a Markov
    chain: a random split would put a point's neighbours in the training set
    and turn cross-validation into interpolation. Balanced on ``Σw`` rather
    than on point counts because the weights are ξ pair-posteriors, so a long
    stretch of near-zero weight carries almost no information. Deterministic
    and ``O(n)``; returns ``[]`` when the weights are degenerate.
    """
    w = np.asarray(weights, dtype=float)
    n = w.size
    cw = np.cumsum(w)
    total = float(cw[-1]) if n else 0.0
    if not np.isfinite(total) or total <= 0.0:
        return []
    targets = total * np.arange(1, n_folds) / n_folds
    cuts = np.unique(np.clip(np.searchsorted(cw, targets, side="left") + 1, 1, n - 1))
    bounds = np.concatenate(([0], cuts, [n]))
    return [(int(bounds[i]), int(bounds[i + 1]))
            for i in range(bounds.size - 1) if bounds[i + 1] > bounds[i]]


def _score_xvcic(
    cls, entry, params, u, v, weights,
    n_folds: int = _XVCIC_N_FOLDS, purge: int = _XVCIC_PURGE,
) -> float:
    """Cross-validated out-of-fold weighted log-likelihood.

    What it is, plainly: an **exact purged K-fold cross-validated
    log-likelihood** — the family is refitted on each training part and
    scored on the held-out part. It is *not* the analytic first-order
    xv-CIC formula of Grønneberg & Hjort (2014), which approximates
    leave-one-out cross-validation by an AIC-like penalty; the name ``xvcic``
    is kept for the criterion's motivation, not its formula.

    Motivation — AIC and BIC add a *fixed* penalty (``2k``, ``k·log n_eff``)
    derived for a genuine likelihood with known margins. Here the margins are
    plug-in estimates and the weights are posterior probabilities, so that
    derivation does not apply: the correct penalty is neither known in closed
    form nor family-independent (Grønneberg & Hjort 2014 show the analogous
    breakdown under rank-based pseudo-likelihood: their abstract states that
    the CIC "cannot exist for copula models with densities that grow very
    fast near the edge of the unit cube"). Cross-validation sidesteps the
    derivation entirely: it *measures* out-of-sample fit, so the effective
    penalty adapts to the family.

    Estimator — the folds are contiguous and ``Σw``-balanced (see
    :func:`_contiguous_weight_balanced_folds`), and ``purge`` points are
    dropped from the training set on each side of the held-out block because
    consecutive pairs share an observation (pair *n* is ``(Y_n, Y_{n+1})``).
    Each fold refits the family on its training part and scores the held-out
    part; the result is rescaled to the block's full weight so that a fold
    lost to a failed fit does not depress the score. Fully deterministic.

    Caveat — the pseudo-observations ``u = F_i(Y_n)`` come from margins fitted
    on *all* the data, so the folds are not fully independent: this validates
    the copula stage conditionally on the margins, not the two-stage procedure
    as a whole. It is therefore an approximation of the two-stage criterion of
    Ko & Hjort rather than a rank-based CIC, and it does not carry the
    Grønneberg–Hjort guarantee.

    Degrades gracefully to the plain weighted log-likelihood when the block is
    too small for the split to mean anything.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = u.size
    total_w = float(w.sum())
    if not np.isfinite(total_w) or total_w <= MIN_POSITIVE:
        return -np.inf
    if total_w < _XVCIC_MIN_W or n < 4 * n_folds:
        return _weighted_log_likelihood(cls, params, u, v, w)

    folds = _contiguous_weight_balanced_folds(w, n_folds)
    if len(folds) < 2:
        return _weighted_log_likelihood(cls, params, u, v, w)

    uv = np.column_stack((u, v))
    num = den = 0.0
    for a, b in folds:
        train = np.ones(n, dtype=bool)
        train[max(0, a - purge):min(n, b + purge)] = False
        if not train.any():
            continue
        w_train = w[train]
        if float(w_train.sum()) <= MIN_POSITIVE:
            continue
        try:
            fold_params = _fit_copula_params(
                cls, entry, u[train], v[train], w_train,
            )
            log_pdfs = cls(**_copula_kwargs(fold_params)).logpdf_array(uv[a:b])
        except Exception as exc:                             # pragma: no cover
            logger.debug("xvcic: cls=%s fold [%d,%d) failed (%s)",
                         cls.__name__, a, b, exc)
            continue
        log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, _LOGPDF_FLOOR)
        num += float(np.dot(w[a:b], log_pdfs))
        den += float(w[a:b].sum())

    if den <= MIN_POSITIVE:
        return -np.inf
    # Rescale to the full block weight: comparable to the other likelihood
    # criteria, and unaffected by a fold that had to be skipped.
    return num * (total_w / den)


_SCORE_FN: dict[str, Callable[..., float]] = {
    "mle":   _score_mle,
    "aic":   _score_aic,
    "bic":   _score_bic,
    "cvm":   _score_cvm,
    "huard": _score_huard,
    "huard_common": _score_huard_common,
    "huard_global": _score_huard_global,
    "xvcic": _score_xvcic,
}


def _select_and_fit_copula(
    candidates: list[str],
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    criterion: str = DEFAULT_SELECTION_CRITERION,
) -> dict:
    """Select the best copula family from ``candidates`` and return a TOML block.

    Parameters
    ----------
    candidates : list of SHORT_NAMEs (``["Gauss", "Clayton", …]``).
    u, v       : pseudo-observation columns (length N each).
    weights    : per-observation weights (e.g. ICE pair-posteriors ξ).
    criterion  : key in :data:`SELECTION_CRITERIA`. ``"mle"`` is the v0.4
                 baseline (max weighted log-lik). ``"huard"`` is the
                 CSDA-2013 Bayesian evidence; AIC / BIC penalise the
                 parameter count; ``"cvm"`` uses the Cramér–von Mises
                 statistic on the empirical-vs-fitted copula CDF.

    The τ̂ that ends up in the returned block is always the maximum-likelihood
    estimate (``_fit_copula_params``); only the *family* picked changes with
    the criterion.

    Returns
    -------
    A TOML-compatible block ``{"name": ..., "tau": ..., …}``. When the
    selected family's fit failed, the block also carries
    ``FIT_FAILED_KEY: True`` (see :func:`_fit_copula_params`); the M-step
    removes it before storing the block and logs the pair of states instead.
    """
    resolved, fits = _copula_candidate_fits(candidates, u, v, weights, criterion)

    best_score = -np.inf
    best_block: dict = {"name": candidates[0], "tau": 0.0}

    for short, params, score in fits:
        if score > best_score:
            best_score = score
            best_block = _copula_block(short, params)

    if not best_score > -np.inf:
        best_block = _copula_placeholder(candidates, resolved, criterion, best_block)

    return best_block


def _copula_candidate_fits(
    candidates: list[str],
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    criterion: str,
) -> tuple[list[tuple[str, object, type]], list[tuple[str, dict, float]]]:
    """Fit every candidate family and score it — the loop of :func:`_select_and_fit_copula`.

    Returns ``(resolved, fits)``: the resolvable candidates ``(short, entry,
    cls)`` and, in the same order, ``(short, params, score)`` with ``params``
    from :func:`_fit_copula_params` (possibly flagged ``FIT_FAILED_KEY``) and
    ``score`` the ``criterion`` score (higher is better). Shared with the
    multiple-imputation M-step of ICE with missing observations, which pools
    these scores over the completed series.
    """
    if criterion not in _SCORE_FN:
        raise ValueError(
            f"Unknown selection_criterion {criterion!r}. "
            f"Valid: {sorted(_SCORE_FN)}"
        )
    score_fn = _SCORE_FN[criterion]

    resolved: list[tuple[str, object, type]] = []
    for short in candidates:
        try:
            entry, cls = _resolve_candidate(short)
        except ValueError as exc:
            logger.warning("Skipping candidate %r: %s", short, exc)
            continue
        resolved.append((short, entry, cls))

    # ``huard_common`` integrates every candidate over one shared τ-support —
    # the intersection of the declared ranges — so no family is rewarded
    # merely for declaring a narrow one. Only this criterion needs to know
    # the candidate set, hence the keyword rather than a wider signature.
    score_kw: dict = {}
    if resolved and criterion in ("huard_common", "huard_global"):
        los, his = zip(*(e.value.TAU_MIN_MAX for _, e, _ in resolved))
        if criterion == "huard_common":
            score_kw["tau_range"] = (max(los), min(his))
        else:
            # Common prior *mass*: the widest declared range among candidates.
            score_kw["prior_width"] = max(hi - lo for lo, hi in zip(los, his))

    fits: list[tuple[str, dict, float]] = []
    for short, entry, cls in resolved:
        params = _fit_copula_params(cls, entry, u, v, weights)
        score  = score_fn(cls, entry, params, u, v, weights, **score_kw)
        logger.debug(
            "  candidate=%s  params=%s  criterion=%s  score=%.4f",
            short, params, criterion, score,
        )
        fits.append((short, params, score))
    return resolved, fits


def _copula_block(short: str, params: dict) -> dict:
    """TOML copula block ``{"name", "tau", extras…}`` from fitted parameters."""
    # An average of fitted τ (multiple imputation) may pass a reachable bound
    # by a few ulps: store the τ the copula uses (F1).
    entry = CopulaEnum.from_short_name(short)
    tau = params["tau_k"] if entry is None else entry.reachable_tau(params["tau_k"])
    block = {"name": short, "tau": tau}
    for k, val in params.items():
        if k != "tau_k":
            block[k] = val
    return block


def _copula_placeholder(candidates, resolved, criterion, best_block: dict) -> dict:
    """The block returned when no candidate produced a usable score.

    It is a placeholder (first candidate, τ = 0 pulled into its range), not a
    fit — flagged like a failed fit instead of returned as if it were one.
    A family with jointly constrained parameters gets extras the constructor
    accepts at that τ (BB1 refuses its initial δ = 1.5 below τ = 1/3); for
    every other family the block is unchanged.
    """
    if resolved:
        _, entry0, _ = resolved[0]
        best_block["tau"] = float(np.clip(0.0, *entry0.constructible_tau_range()))
        own = EXTRA_PARAM_BOUNDS.get(entry0.value.CLASS_NAME, {})
        params = {"tau_k": best_block["tau"],
                  **{k: best_block.get(k, init) for k, (_, _, init) in own.items()}}
        repaired = entry0.klass.constructible_params(params)
        if repaired is not params:
            best_block.update((k, val) for k, val in repaired.items() if k != "tau_k")
    best_block[FIT_FAILED_KEY] = True
    logger.warning(
        "Copula selection: no candidate among %s has a finite %s score — "
        "returning the placeholder %s.", candidates, criterion,
        {k: v for k, v in best_block.items() if k != FIT_FAILED_KEY},
    )
    return best_block


# ---------------------------------------------------------------------------
# Weighted pseudo-observations — the FR-7(a) weighted empirical CDF (FR-12)
# ---------------------------------------------------------------------------

def _weighted_ecdf(y_sample: np.ndarray, w_sample: np.ndarray,
                   y_eval: np.ndarray) -> np.ndarray:
    """``F̂(y) = Σ_k w_k 1{y_k ≤ y} / (Σ_k w_k + 1)`` at every ``y`` of ``y_eval``.

    ``≤``: a value tied with sample points gets the weight of all of them (the
    upper end of the tie block), so tied observations share one F̂ value.
    With ``w_k ≥ 0`` the result lies in ``[0, Σw/(Σw + 1)]`` — never 1. An
    empty sample gives 0 everywhere. O((m + n) log m) for m sample and n
    evaluation points (one sort, one binary search per point).

    One function for the posterior-weighted empirical margins of ICE's
    ``copula_margins = "empirical"`` (audit FR-7 a,
    :func:`pmcprg.pmc.ice._empirical_margin_cdfs`, which derives the ``+ 1``)
    and the weighted pseudo-observations of the public weighted fit
    (:func:`weighted_pseudo_obs`, FR-12). Moved here from ``ice.py``
    unchanged.
    """
    y_sample = np.asarray(y_sample, dtype=float)
    y_eval = np.asarray(y_eval, dtype=float)
    if y_sample.size == 0:
        return np.zeros(y_eval.shape)
    order = np.argsort(y_sample, kind="stable")
    ys = y_sample[order]
    cw = np.cumsum(np.asarray(w_sample, dtype=float)[order])
    # Number of sample points ≤ y (side="right" puts ties below y).
    idx = np.searchsorted(ys, y_eval, side="right")
    num = np.where(idx > 0, cw[np.maximum(idx - 1, 0)], 0.0)
    return num / (cw[-1] + 1.0)


# Below this Σw the ``+ 1`` of the weighted ECDF compresses the pseudo-
# observations by more than 9 % (Σw/(Σw + 1) < 0.91): a WARNING says so.
_SMALL_WEIGHT_SUM: float = 10.0


def weighted_pseudo_obs(data, weights) -> np.ndarray:
    """Weighted pseudo-observations of an ``(n, 2)`` sample: each column through its weighted ECDF.

    ``û_ij = Σ_k w_k 1{x_kj ≤ x_ij} / (Σ_k w_k + 1)`` — the convention of
    ICE's posterior-weighted empirical margins (audit FR-7 a,
    :func:`_weighted_ecdf`) — clipped to ``[EPS, 1 − EPS]`` as those are.

    * Unit weights give ``rank/(n + 1)``, the rank pseudo-observations of
      the unweighted fit, exactly when there are no ties (ties share the
      **upper** end of their block here, the mid-rank in the unweighted fit).
    * ``{0, 1}`` weights give the rank pseudo-observations of the kept
      subset, since a zero weight adds nothing to any F̂.
    * The ``+ 1`` is on the scale of the weights: they are **frequency
      weights** (an integer weight counts its observation that many times,
      the package's convention, as for ``n_eff = Σw`` and the BIC), so
      multiplying every weight by c ≠ 1 moves the pseudo-observations by
      O(1/Σw) (measured: 4.7·10⁻³ for ×10³ at Σw = 210). Pass
      pseudo-observations with ``pseudo_obs=True`` for a fit that does not
      depend on the weights' scale. Weights summing to less than 10 put
      every pseudo-observation below Σw/(Σw + 1) < 0.91 and log a WARNING —
      probability weights, summing to 1, put them all below ½.

    Raises ``ValueError`` for data that are not ``(n, 2)`` and finite, and
    for invalid weights (:func:`validate_weights`).
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"data must be shape (n, 2), got {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must be finite to be rank-transformed with weights.")
    w = validate_weights(weights, x.shape[0])
    sum_w = float(w.sum())
    if sum_w < _SMALL_WEIGHT_SUM:
        logger.warning(
            "weighted pseudo-observations: the weights sum to %.4g, so every "
            "pseudo-observation is below Σw/(Σw + 1) = %.4g. Weights are frequency weights "
            "(one unit counts one observation): if these are probability or normalised "
            "weights, rescale them (e.g. to mean 1) or pass pseudo-observations with "
            "pseudo_obs=True.", sum_w, sum_w / (sum_w + 1.0))
    out = np.column_stack([_weighted_ecdf(x[:, j], w, x[:, j]) for j in (0, 1)])
    np.clip(out, EPS, ONE_MINUS_EPS, out=out)
    return out


# ---------------------------------------------------------------------------
# Inversion of Kendall's τ (itau) — weighted or not (FR-12)
# ---------------------------------------------------------------------------
#
# One-parameter families: the family's τ ↦ θ map applied to Kendall's τ̂ (the
# copulas are parametrised by τ, so τ̂ clipped into the constructible range
# *is* the estimate). Multi-parameter families: τ̂ the same way, then the
# other parameters by (weighted) maximum likelihood with τ held at τ̂ — the
# profile likelihood of VineCopula's ``BiCopEst(method = "itau")`` for
# Student's ν (Dißmann et al. 2013, §4) and of pyvinecopulib's itau.

_PROFILE_XATOL: float = 1e-6            # Brent's tolerance in the search coordinate
_PROFILE_BISECT: int = 60               # bisection steps to the edge of the admissible set
_RECIPROCAL_EXTRAS: tuple[str, ...] = ("df", "nu")   # searched in 1/ν, like the joint MLE


def _extra_names(entry) -> list[str]:
    """The registered non-τ parameters of a family, in the registry's order."""
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    return [p for p in entry.value.PARAMETERS_SET_NAME
            if p != "tau_k" and p in EXTRA_PARAM_BOUNDS_BY_PARAM]


def _builds(cls, params: dict) -> bool:
    try:
        cls(**params)
    except Exception:
        return False
    return True


def _admissible_interval(cls, tau: float, name: str, lo: float, hi: float,
                         start: float) -> tuple[float, float]:
    """The ``x ∈ [lo, hi]`` at which ``cls`` builds at ``(τ, x)``, ends pulled in when refused.

    Every family's constraint on one extra parameter at fixed τ is one-sided
    (BB1 δ < 1/(1 − τ), BB6 δ ≤ 1/(1 − τ), BB7 θ < θ_Joe(τ), Tawn ψ > τ, …),
    so the admissible set is an interval containing ``start``. A refused box
    end is replaced by the edge found by bisection with the constructor's
    own test, pulled inside by the padding of
    :func:`pmcprg.copulas._base.constructible_tau_range` — the edge is often
    a singular limit (Tawn ψ → τ is the Marshall–Olkin copula).
    """
    if not _builds(cls, {"tau_k": tau, name: float(start)}):
        return float(lo), float(hi)              # no admissible anchor: the penalties decide
    ends = []
    for end in (lo, hi):
        if _builds(cls, {"tau_k": tau, name: float(end)}):
            ends.append((float(end), False))
            continue
        good, bad = float(start), float(end)
        for _ in range(_PROFILE_BISECT):
            mid = 0.5 * (good + bad)
            if mid in (good, bad):
                break
            if _builds(cls, {"tau_k": tau, name: mid}):
                good = mid
            else:
                bad = mid
        ends.append((good, True))
    (a, a_cut), (b, b_cut) = ends
    pad = max(_TAU_PAD_REL * (b - a), _TAU_PAD_ABS)
    if a_cut and b - a > 2.0 * pad:
        a += pad
    if b_cut and b - a > 2.0 * pad:
        b -= pad
    return a, b


def _profile_extras(cls, entry, uv: np.ndarray, weights, tau: float) -> WeightedFit:
    """(Weighted) MLE of the extra parameters of ``cls`` with τ held at ``tau``.

    The objective is ``Σ wᵢ log c(uᵢ, vᵢ; τ, extras)`` with the weights
    normalised to sum 1 (unit weights when ``weights`` is None) — the argmax
    of the total, and scale-free. One extra: bounded Brent on its admissible
    interval at ``tau`` (:func:`_admissible_interval`), in 1/ν for Student's
    ``df`` and t-EV's ``nu`` (the likelihood is flat in ν for large ν, and
    ν → ∞ is the end 1/ν_max of the box), ``xatol`` 10⁻⁶. Several extras
    (the three-parameter Tawn): Nelder–Mead in their registered boxes, an
    inadmissible point scoring the failure penalty. Returns the parameters —
    constructible — with the optimiser's status and counts.
    """
    from scipy.optimize import minimize

    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    uv = np.asarray(uv, dtype=float)
    w = np.ones(uv.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    w = w / (w.sum() + MIN_POSITIVE)
    names = _extra_names(entry)
    boxes = [EXTRA_PARAM_BOUNDS_BY_PARAM[n] for n in names]
    start = cls.constructible_params({"tau_k": tau, **{n: init for n, (_, _, init) in
                                                         zip(names, boxes)}})
    x0 = [float(start[n]) for n in names]

    def neg_ll(xs) -> float:
        try:
            cop = cls(tau_k=tau, **{n: float(x) for n, x in zip(names, xs)})
            ll = _weighted_log_density_sum(cop.logpdf_array(uv), w)
        except Exception:
            return _OPT_FAIL_PENALTY
        return -ll if np.isfinite(ll) else _OPT_FAIL_PENALTY

    if len(names) == 1:
        (name,), ((lo, hi, _),) = names, boxes
        a, b = _admissible_interval(cls, tau, name, float(lo), float(hi), x0[0])
        recip = name in _RECIPROCAL_EXTRAS
        s_lo, s_hi = sorted((1.0 / a, 1.0 / b)) if recip else (a, b)

        def x_of(s: float) -> float:
            s = float(np.clip(s, s_lo, s_hi))
            return 1.0 / s if recip else s

        if not s_hi > s_lo:
            x, ok, msg, nit, nfev = a, True, f"{name} has a single admissible value at τ", 0, 0
        else:
            res = minimize_scalar(lambda s: neg_ll([x_of(s)]), bounds=(s_lo, s_hi),
                                  method="bounded", options={"xatol": _PROFILE_XATOL})
            x = x_of(res.x)
            ok = bool(res.success) and bool(res.fun < _OPT_FAIL_PENALTY)
            msg = (f"profile MLE of {name} at τ = {tau:.6g}: bounded Brent in "
                   f"{'1/' + name if recip else name}, {res.message}")
            nit, nfev = int(getattr(res, "nit", 0)), int(getattr(res, "nfev", 0))
        xs = [x]
    else:
        box = [(float(lo), float(hi)) for lo, hi, _ in boxes]
        # An explicit initial simplex inside the box. scipy's own (x0 plus 5 %
        # per coordinate) is clipped onto a bound when x0 sits on or next to
        # it, and before scipy 1.11 that collapses the simplex: from Tawn3's
        # start (ψ_u, ψ_v) = (1, 1) the search never left that corner under
        # scipy 1.10 (log-likelihood 176.1 against 214.6 at the profile
        # optimum; minimum-versions CI job). Each vertex steps 10 % of its
        # coordinate's range towards the interior.
        lo_b = np.array([b[0] for b in box])
        hi_b = np.array([b[1] for b in box])
        span = hi_b - lo_b
        v0 = np.clip(np.asarray(x0, dtype=float), lo_b + 0.1 * span, hi_b - 0.1 * span)
        simplex = [v0]
        for k in range(v0.size):
            v = v0.copy()
            v[k] += 0.1 * span[k] if v0[k] + 0.1 * span[k] <= hi_b[k] else -0.1 * span[k]
            simplex.append(v)
        res = minimize(neg_ll, v0, method="Nelder-Mead", bounds=box,
                       options={"xatol": _PROFILE_XATOL, "fatol": 1e-12, "maxiter": 4000,
                                "initial_simplex": np.array(simplex)})
        xs = [float(np.clip(x, lo, hi)) for x, (lo, hi) in zip(res.x, box)]
        ok = bool(res.success) and bool(res.fun < _OPT_FAIL_PENALTY)
        msg = f"profile MLE of {', '.join(names)} at τ = {tau:.6g}: Nelder–Mead, {res.message}"
        nit, nfev = int(getattr(res, "nit", 0)), int(getattr(res, "nfev", 0))
    params = {"tau_k": tau, **{n: float(x) for n, x in zip(names, xs)}}
    if not _builds(cls, params):                 # the optimiser's point refused: keep the start
        params, ok = {"tau_k": tau, **{n: float(start[n]) for n in names}}, False
        msg += "; its point is not admissible, the start value is kept"
    if not neg_ll([params[n] for n in names]) < _OPT_FAIL_PENALTY:
        ok = False
        msg += "; the likelihood is not finite at any evaluated value"
    return WeightedFit(params, converged=ok, message=msg, n_iter=nit, n_eval=nfev)


def _itau_fit(cls, entry, uv: np.ndarray, weights, tau_hat: float) -> WeightedFit:
    """itau from Kendall's τ̂ (weighted or not): τ̂ into the constructible range, then the profile.

    τ̂ is clipped into :meth:`CopulaEnum.constructible_tau_range` — the rule
    of ``fit(method='tau')`` since RB-6: a singular end |τ| = 1 is replaced
    by the padded value next to it, an admissible end is kept, and a τ̂
    beyond the reachable |τ| (Frank) goes to that bound. A one-parameter
    family is then fitted; a multi-parameter one gets its other parameters
    from :func:`_profile_extras` at that τ.
    """
    lo, hi = entry.constructible_tau_range()
    tau_k = float(np.clip(tau_hat, lo, hi))
    if tau_k != tau_hat:
        logger.info("%s itau: τ̂ = %.6g outside [%.6g, %.6g] — clipped to %.6g.",
                    cls.__name__, tau_hat, lo, hi, tau_k)
    if not _extra_names(entry):
        return WeightedFit({"tau_k": tau_k}, converged=True,
                           message="inversion of Kendall's τ (moment estimate)")
    return _profile_extras(cls, entry, uv, weights, tau_k)


def _weighted_itau_fit(cls, entry, u, v, weights) -> WeightedFit:
    """Weighted itau: the weighted τ (τ-b on ties) inverted, then the weighted profile MLE."""
    uv = np.column_stack((np.asarray(u, dtype=float), np.asarray(v, dtype=float)))
    return _itau_fit(cls, entry, uv, weights, _weighted_kendall_tau(u, v, weights))


def weighted_kendall_tau(u, v, weights=None) -> float:
    """Weighted Kendall's τ of two samples — the coefficient the weighted fit inverts.

    The weighted τ-b of pyvinecopulib and VineCopula, which is the weighted
    τ-a when no two points of positive weight tie (the engine's
    ``_weighted_kendall_tau``, O(n log n)); ``weights=None`` gives the τ-b of
    ``scipy.stats.kendalltau``. Only ranks matter: ``u``, ``v`` may be raw
    data or pseudo-observations. Weights are validated
    (:func:`validate_weights`); ``u`` and ``v`` must have one value per
    weight.
    """
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if u.shape != v.shape:
        raise ValueError(f"u and v must have the same length, got {u.size} and {v.size}.")
    w = np.ones(u.size) if weights is None else validate_weights(weights, u.size)
    return _weighted_kendall_tau(u, v, w)


# ---------------------------------------------------------------------------
# The public weighted fit — CopulaVirt.fit(..., weights=…) (FR-12)
# ---------------------------------------------------------------------------

def fit_weighted(cls, data, method: str, weights, pseudo_obs: bool):
    """``cls.fit(data, method, weights=weights, pseudo_obs=pseudo_obs)`` — see :meth:`CopulaVirt.fit`.

    1. The weights are validated (:func:`validate_weights`) and the rows of
       weight zero dropped: they carry no information for any quantity below.
    2. If every remaining weight is 1, the fit **is** the unweighted fit of
       those rows, and is computed by it (``cls.fit(rows, method,
       pseudo_obs=…)``), bit for bit: unit weights are no weights, and
       ``{0, 1}`` weights are the fit on the kept subset.
    3. Otherwise the weighted engine that ICE's M-step uses: pseudo-
       observations by the weighted ECDF (:func:`weighted_pseudo_obs`) unless
       ``pseudo_obs``; ``'mle'`` — :func:`_weighted_mle_fit`, the fit of
       :func:`_fit_copula_params`; ``'tau'`` — :func:`_weighted_itau_fit`.
       The reported log-likelihood is ``Σ wᵢ log c(ûᵢ, v̂ᵢ)`` (the one the
       ICE selection scores use).
    """
    from pmcprg.copulas._base import _pseudo_observations, _registry_entry
    from pmcprg.copulas._fit import FitResult

    x = np.asarray(data, dtype=float)
    w = validate_weights(weights, x.shape[0])
    keep = w > 0.0
    if not keep.all():
        x, w = x[keep], w[keep]
    if np.all(w == 1.0):
        return cls.fit(x, method=method, pseudo_obs=pseudo_obs)
    if method not in ("tau", "mle"):
        raise ValueError(f"method must be 'tau' or 'mle', got {method!r}.")
    n = x.shape[0]
    if n < 4:
        raise ValueError(f"At least 4 observations of positive weight required, got {n}.")
    entry = _registry_entry(cls)
    uv = _pseudo_observations(x, True) if pseudo_obs else weighted_pseudo_obs(x, w)
    u, v = uv[:, 0], uv[:, 1]
    if method == "mle":
        wf = _weighted_mle_fit(cls, entry, u, v, w)
    else:
        if np.ptp(u) == 0.0 or np.ptp(v) == 0.0:
            raise ValueError(f"{cls.__name__}.fit(method='tau', weights=…): Kendall's τ is "
                             "undefined on these data (a constant column?).")
        wf = _weighted_itau_fit(cls, entry, u, v, w)
    params = _copula_kwargs(wf.params)
    if not wf.converged:
        logger.warning("%s.fit(method=%r, weights=…): %s; returning the best point found %s.",
                       cls.__name__, method, wf.message, params)
    copula = cls(**params)
    return FitResult(
        copula=copula, method=method, tau_k=float(params["tau_k"]),
        log_likelihood=_weighted_log_likelihood(cls, params, u, v, w), n_obs=n, uv=uv,
        converged=wf.converged, message=wf.message, n_iter=wf.n_iter, n_eval=wf.n_eval,
        weights=w,
    )
