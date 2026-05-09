"""
ice.py — Iterative Conditional Estimation (ICE) for unsupervised PMC/HMC fitting.

Public API
----------
ice(model, Y, ice_cfg=None) -> (PMCModel, IceTrace)
    Fit a PMCModel to the observation sequence Y using ICE.
    Returns the fitted model and an :class:`IceTrace` capturing per-iteration
    diagnostics (log-lik, τ, family, prior matrix, margin params; plus the
    losing runs when multistart is enabled). Use ``trace.log_liks`` for the
    plain log-likelihood history.

Algorithm (ICE for SR-PMC)
--------------------------
Given an initial model θ^(0) and observations Y = y_{1:N}:

Iterate until convergence:

  E-step:
    Compute forward-backward quantities:
      α̂_n(j), β̂_n(j)          — normalized forward/backward variables
      γ_n(j)   = P(X_n=j | Y)  — marginal posteriors
      ξ_n(i,j) = P(X_n=i, X_{n+1}=j | Y)  — joint posteriors

  M-step (Conditional estimation):
    1. Prior p̂[i,j] = (1/N-1) Σ_n ξ_n(i,j)

    2. For each pair (i,j) — copula selection + τ estimation:
       a. Collect pseudo-observations (u_n, v_n) = (F_{ij}(y_n), F_{ji}(y_{n+1}))
          weighted by ξ_n(i,j)  for n = 1, …, N-1.
       b. Select best copula family from 'candidates' by weighted log-likelihood.
       c. Estimate τ_{ij} by weighted MLE on τ ∈ [τ_min, τ_max].

    3. (Optional) Margin re-estimation for Gaussian margins:
       μ̂_{ij} = Σ_n w_n y_n;   σ̂_{ij} = sqrt(Σ_n w_n (y_n - μ̂)²)
       where w_n = ξ_n(i,j) / Σ_n ξ_n(i,j).

ICE configuration (TOML [ice] section or dict)
-----------------------------------------------
  fit_margins : bool   (default False) — re-estimate margin parameters.
  max_iter    : int    (default 50)    — maximum EM iterations.
  tol         : float  (default 1e-4)  — relative log-likelihood convergence threshold.
  candidates  : list[str]              — SHORT_NAMEs of candidate copula families.
                 default: all 1-parameter available families except Product.
"""

import copy as _copy
import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from prg.copulas._base import CopulaEnum
from prg.pmc.inference import backward, forward, precompute_weights, smooth
from prg.pmc.model import PMCModel, Variant, _MULTIVARIATE_DISTS as _MULTIVARIATE_DIST_NAMES
from prg.numerics import EPS, ONE_MINUS_EPS, MIN_POSITIVE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IceTrace — captured per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class IceTrace:
    """Per-iteration ICE history, captured for diagnostic visualisations.

    Snapshot semantics
    ------------------
    All time-indexed arrays/lists have length ``T`` = number of completed
    E-steps. The snapshot at index ``t`` reflects the model state *as the
    log-likelihood ``log_liks[t]`` was computed* — i.e. just before the
    M-step that produces the model used for ``log_liks[t + 1]``::

           E-step ──→  M-step  ──→  E-step ──→ …
              ▲              ▲
              │              │
        snapshot[t]     snapshot[t+1]
        log_liks[t]     log_liks[t+1]

    For T = 0 (no completed iteration), the array fields are empty arrays
    of shape ``(0, K, K)`` — never ``None``.

    Fields
    ------
    log_liks       : ``list[float]`` — log-likelihood trace.
    tau_history    : ``np.ndarray`` shape ``(T, K, K)``, dtype float —
                     τ_K of the copula at pair ``(i, j)`` for each iteration.
                     ``NaN`` if the variant has no copula at that pair (e.g.
                     HMC-IN) or the candidate selection failed.
    family_history : ``list[list[list[str]]]`` — same shape ``(T, K, K)``,
                     each entry is the SHORT_NAME of the selected family
                     (or ``""`` if no copula).
    p_history      : ``np.ndarray`` shape ``(T, K, K)`` — joint prior
                     ``p[i,j] = P(X_n=i, X_{n+1}=j)`` (computed as
                     ``π·A`` for HMC variants).
    margin_history : ``list[list[dict]]`` — at index ``t`` a list of margin
                     blocks ``{"i": ..., "j": ..., "dist": ..., "params": {...}}``
                     in declaration order, mirroring ``model.margin_blocks()``.
    multistart_runs: ``list[IceTrace]`` — when ``n_starts > 1``, the traces
                     from the *non-best* runs (the chosen run is the trace
                     itself). Empty list when single-start.
    run_tag        : str — label of this run (``"unperturbed"``,
                     ``"perturbed-3"``, …) when multistart was used.
    candidates     : ``list[str]`` — copula candidate SHORT_NAMEs that ICE
                     iterated over, captured for the family-ribbon legend.
    """
    log_liks:        list[float]              = field(default_factory=list)
    # Always an ndarray (possibly shape (0, K, K)); the ``field(default=…)``
    # rule prevents mutable defaults so we use a default_factory here too.
    tau_history:     np.ndarray               = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=float)
    )
    family_history:  list[list[list[str]]]    = field(default_factory=list)
    p_history:       np.ndarray               = field(
        default_factory=lambda: np.empty((0, 0, 0), dtype=float)
    )
    margin_history:  list[list[dict]]         = field(default_factory=list)
    multistart_runs: list["IceTrace"]         = field(default_factory=list)
    run_tag:         str                      = ""
    candidates:      list[str]                = field(default_factory=list)

    @property
    def n_iters(self) -> int:
        return len(self.log_liks)

    def __len__(self) -> int:
        return self.n_iters


@dataclass
class IceResult:
    """Bundle returned by GUI/diagnostics layers around an ICE run.

    Carries everything the View-selector needs to draw the 12 ICE views
    (the trace alone is not sufficient — view ``J`` needs the *initial*
    model, view ``C`` and ``H`` need the observation sequence ``Y``).

    Fields
    ------
    initial_model : :class:`PMCModel` — the model handed to ``ice()``.
    fitted_model  : :class:`PMCModel` — the best-run model returned by ICE.
    Y             : ``np.ndarray`` — observations ICE was fitted to.
    trace         : :class:`IceTrace` — per-iteration history.
    """
    initial_model: PMCModel
    fitted_model:  PMCModel
    Y:             np.ndarray
    trace:         IceTrace


# ---------------------------------------------------------------------------
# Per-iteration snapshot helpers
# ---------------------------------------------------------------------------

def _snapshot_tau_family(model: PMCModel) -> tuple[np.ndarray, list[list[str]]]:
    """Return ``(tau_kk, family_kk)`` for the current model.

    ``tau_kk`` is a ``(K, K)`` float array (NaN where no copula is declared);
    ``family_kk`` is a ``K × K`` list-of-lists of SHORT_NAME strings (``""``
    where no copula).
    """
    K = model.K
    tau = np.full((K, K), np.nan, dtype=float)
    fam = [["" for _ in range(K)] for _ in range(K)]
    if not model.variant.uses_copula:
        return tau, fam
    for blk in model.copula_blocks():
        i = int(blk["i"]); j = int(blk["j"])
        tau[i, j] = float(blk.get("tau", np.nan))
        fam[i][j] = str(blk.get("name", ""))
    return tau, fam


def _snapshot_prior_p(model: PMCModel) -> np.ndarray:
    """Joint prior ``p[i, j]`` regardless of variant.

    For HMC variants the model stores a row-stochastic ``A``; the joint is
    recovered as ``p[i, j] = π_i · A[i, j]`` (always exists since the
    stationary distribution is computed at construction time).
    """
    if model.variant.has_markov_prior:
        pi = np.asarray(model.stationary_pi, dtype=float)
        A  = np.asarray(model.transition_A,   dtype=float)
        return pi[:, None] * A
    return np.asarray(model.prior_p, dtype=float).copy()


def _snapshot_margins(model: PMCModel) -> list[dict]:
    """Deep-copy of the ``margins`` blocks in declaration order."""
    return _copy.deepcopy(list(model.margin_blocks()))

# Default candidate SHORT_NAMEs (1-parameter families, no Product)
_DEFAULT_CANDIDATES = ["Gauss", "GH", "Clayton", "Frank", "Joe"]


# Bounds and initial values for the *non*-τ parameters of multi-parameter
# copula families. Single source of truth — also consumed by the GUI
# copula dialog (``prg.pmc.gui.dialogs``). Keyed by ``CopulaVirt`` subclass
# name; each entry maps a parameter name to ``(lower, upper, init)``.
EXTRA_PARAM_BOUNDS: dict[str, dict[str, tuple[float, float, float]]] = {
    "CopulaBB1":     {"delta": (1.0,   10.0,  1.5)},
    "CopulaStudent": {"df":    (2.0,  100.0,  4.0)},
}
# Backward-compatibility shim — the underscore-prefixed name was used
# internally before ``EXTRA_PARAM_BOUNDS`` was promoted to public API.
_EXTRA_PARAM_BOUNDS = EXTRA_PARAM_BOUNDS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _weighted_kendall_tau(
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted Kendall's τ on pseudo-observations (u, v).

    Definition (weighted version of the Mann/Kendall coefficient)::

        τ = ( Σ_{i<j} w_i w_j · sign(u_i − u_j) · sign(v_i − v_j) )
            ────────────────────────────────────────────────────────
            ( Σ_{i<j} w_i w_j )

    Implementation is fully vectorised (O(N²) memory, O(N²) ops). For
    typical ICE workloads (N ≲ a few thousand) this costs a handful of
    milliseconds — negligible compared to the per-iteration MLE.

    Returns ``0.0`` for degenerate inputs (zero total weight, all-equal
    observations, …) so the caller can safely use the result as an
    optimiser initial value.
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


def _joint_posteriors(
    alpha_hat: np.ndarray,
    W: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute  ξ_n(i,j) = P(X_n=i, X_{n+1}=j | Y)  for n = 0, …, N-2.

    Vectorised expression:
        ξ_n(i,j) ∝ α̂_n(i) · W[n,i,j] · β̂_{n+1}(j)
    using broadcasting on the (N-1, K, K) axes.

    Returns
    -------
    xi : (N-1, K, K)
    """
    # alpha_hat[:-1, :, None] : (N-1, K, 1)
    # beta_hat[1:,  None, :]  : (N-1, 1, K)
    xi = alpha_hat[:-1, :, None] * W * beta_hat[1:, None, :]

    # Per-step normalisation; rows that summed to zero stay zero (no division)
    sums = xi.sum(axis=(1, 2), keepdims=True)
    np.divide(xi, sums, out=xi, where=sums > 0)
    return xi


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


def _weighted_mle_tau(
    cls,
    entry,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    tau_init: float | None = None,
) -> float:
    """
    Weighted MLE estimate of τ for copula class `cls` on pseudo-observations (u, v).

    Returns the τ̂ that maximises  Σ_n w_n · log c_{τ}(u_n, v_n).

    ``tau_init`` is an optional data-aware starting guess (e.g. weighted
    Kendall's τ) used as the fallback if the bounded optimiser fails. The
    ``minimize_scalar(method="bounded")`` Brent search is bracketed and
    therefore does not consume an initial value, but it is still useful
    as a safety-net so we don't fall back to a generic Pearson moment.
    """
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    pad = max(1e-4 * (tau_max - tau_min), 1e-8)
    lo, hi = tau_min + pad, tau_max - pad

    w  = weights / (weights.sum() + MIN_POSITIVE)
    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))

    def _neg_wll(tau: float) -> float:
        try:
            cop = cls(tau_k=float(tau))
            log_pdfs = cop.logpdf_array(uv)
            return -float(np.dot(w, log_pdfs))
        except Exception as exc:
            logger.debug(
                "_neg_wll: copula eval failed at tau=%.6f — %s", tau, exc,
                exc_info=True,
            )
            return 1e12

    try:
        res = minimize_scalar(_neg_wll, bounds=(lo, hi), method="bounded",
                              options={"xatol": 1e-6})
        return float(np.clip(res.x, lo, hi))
    except Exception as exc:
        logger.debug(
            "_weighted_mle_tau failed (%s); falling back to data-aware Kendall.",
            exc,
        )
        # Prefer the supplied data-aware estimate (weighted Kendall) when
        # available; otherwise compute it on the fly. Kendall's τ is the
        # natural concordance measure for copulas and is far closer to the
        # true MLE than the previous Pearson-based fallback.
        if tau_init is None or not np.isfinite(tau_init):
            tau_init = _weighted_kendall_tau(u, v, weights)
        return float(np.clip(tau_init, lo, hi))


def _weighted_log_likelihood(
    cls,
    params: dict,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Weighted log-likelihood of the copula evaluated at the given parameter dict."""
    try:
        cop = cls(**params)
        w   = weights / (weights.sum() + MIN_POSITIVE)
        uv  = np.column_stack((np.asarray(u, dtype=float),
                               np.asarray(v, dtype=float)))
        return float(np.dot(w, cop.logpdf_array(uv)))
    except Exception as exc:
        logger.debug(
            "_weighted_log_likelihood: failed for params=%s — %s",
            params, exc, exc_info=True,
        )
        return -1e12


def _fit_copula_params(
    cls,
    entry,
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """Weighted MLE for **all** free parameters of `cls`.

    For 1-parameter families this is just τ̂ (delegated to ``_weighted_mle_tau``).
    For multi-parameter families (BB1, Student) it runs a joint L-BFGS-B
    optimisation over (τ, *extras) using the bounds in ``_EXTRA_PARAM_BOUNDS``.

    Returns
    -------
    dict — keyword arguments suitable for ``cls(**dict)``.  Always contains
    ``tau_k`` and any extra parameters declared in the registry.
    """
    extras = _EXTRA_PARAM_BOUNDS.get(cls.__name__, {})

    # Data-aware initial value: weighted Kendall's τ, clipped to the
    # family's valid τ range. This is the natural concordance measure
    # for copulas; for many one-parameter families θ̂(τ_emp) is already
    # within ~10⁻² of the MLE, so the optimiser typically converges in
    # 1–2 fewer iterations and the fallback is far more robust than a
    # generic mid-range guess.
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    pad     = max(1e-4 * (tau_max - tau_min), 1e-8)
    lo, hi  = tau_min + pad, tau_max - pad
    tau_emp = float(np.clip(_weighted_kendall_tau(u, v, weights), lo, hi))

    # 1-D case — fall back to the existing dedicated routine, supplying
    # the data-aware estimate as a robust fallback / hint.
    if not extras:
        tau_hat = _weighted_mle_tau(cls, entry, u, v, weights, tau_init=tau_emp)
        return {"tau_k": float(tau_hat)}

    # n-D case — joint optimisation, started at (τ_emp, default extras).
    bounds_list: list[tuple[float, float]] = [(lo, hi)]
    init: list[float]      = [tau_emp]
    extra_names: list[str] = []
    for name, (xlo, xhi, x0) in extras.items():
        bounds_list.append((xlo, xhi))
        init.append(x0)
        extra_names.append(name)

    w  = weights / (weights.sum() + MIN_POSITIVE)
    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))

    def _neg_ll(x: np.ndarray) -> float:
        kw: dict = {"tau_k": float(x[0])}
        for k, name in enumerate(extra_names):
            kw[name] = float(x[k + 1])
        try:
            cop = cls(**kw)
            return -float(np.dot(w, cop.logpdf_array(uv)))
        except Exception as exc:
            logger.debug(
                "_fit_copula_params: %s failed at %s — %s",
                cls.__name__, kw, exc, exc_info=True,
            )
            return 1e12

    try:
        res = minimize(
            _neg_ll, init, method="L-BFGS-B", bounds=bounds_list,
            options={"maxiter": 80, "ftol": 1e-7},
        )
        out = {"tau_k": float(np.clip(res.x[0], *bounds_list[0]))}
        for k, name in enumerate(extra_names):
            out[name] = float(np.clip(res.x[k + 1], *bounds_list[k + 1]))
        return out
    except Exception as exc:
        logger.debug(
            "Multi-parameter MLE failed for %s (%s); using initial values.",
            cls.__name__, exc,
        )
        out = {"tau_k": init[0]}
        for k, name in enumerate(extra_names):
            out[name] = init[k + 1]
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
#
# All functions share the signature
#     score(cls, entry, params, u, v, weights) -> float
# where ``params`` is the dict produced by :func:`_fit_copula_params` for the
# candidate (so we factor out the τ̂-fitting once per candidate).

SELECTION_CRITERIA = ("mle", "aic", "bic", "huard", "cvm")
DEFAULT_SELECTION_CRITERION = "mle"


def _score_mle(cls, entry, params, u, v, weights) -> float:
    """Weighted log-likelihood — the v0.4 baseline."""
    return _weighted_log_likelihood(cls, params, u, v, weights)


def _score_aic(cls, entry, params, u, v, weights) -> float:
    """``−AIC`` = 2·log L − 2 k.  Penalises parameter count."""
    ll = _weighted_log_likelihood(cls, params, u, v, weights)
    k  = getattr(cls, "n_params", 1)
    return 2.0 * ll - 2.0 * k


def _score_bic(cls, entry, params, u, v, weights) -> float:
    """``−BIC`` = 2·log L − k·log n.  Stronger penalty than AIC for large n."""
    ll = _weighted_log_likelihood(cls, params, u, v, weights)
    k  = getattr(cls, "n_params", 1)
    n  = len(u)
    return 2.0 * ll - k * np.log(max(n, 1))


def _score_cvm(cls, entry, params, u, v, weights) -> float:
    """``−`` Cramér–von Mises statistic on the (un-weighted) pseudos.

    The CvM statistic measures the L²-distance between the empirical and
    fitted copula CDFs and does not depend on the family's parameter count;
    it is therefore a non-likelihood-based criterion that complements
    :func:`_score_mle`.

    Importing ``_cvm_statistic`` lazily avoids an unconditional dependency
    on the GoF helper from :mod:`prg.copulas._fit`.
    """
    from prg.copulas._fit import _cvm_statistic
    try:
        cop = cls(**params)
        # The CvM statistic has no closed form for some families that lack
        # a CDF (e.g. Student); fall back to a low (worst) score.
        cop.cdf([0.5, 0.5])
    except (NotImplementedError, Exception):
        return -np.inf
    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))
    try:
        S = float(_cvm_statistic(uv, cop))
    except Exception:
        return -np.inf
    return -S


def _huard_log_evidence(
    cls,
    entry,
    params,    # noqa: ARG001  (signature uniformity)
    u: np.ndarray,
    v: np.ndarray,
    weights: np.ndarray,
    n_grid: int = 50,
) -> float:
    """Huard et al. (2006) Bayesian model evidence — CSDA-2013 Eq. (20).

    The criterion picks the family that maximises the integrated likelihood
    over τ ∈ [τ_min, τ_max] under a uniform prior::

        s(r) = (1/(τ_max − τ_min)) ∫ ∏_n c_r(u_n, v_n; τ) dτ

    We compute this in log space on a uniform τ-grid: ``log s(r) =
    logsumexp_k log L_r(τ_k) − log n_grid``, where the implicit factor
    ``(τ_max − τ_min)/n_grid`` cancels between candidates if their τ-ranges
    coincide (otherwise it correctly down-weights the wider-range family).

    Notes
    -----
    * Degenerate ranges (Product copula has ``τ_max == τ_min``) are handled
      by returning the single-point log-likelihood.
    * For multi-parameter families (Student `df`, BB1 `δ`), we fix the extra
      parameter at the MLE-fitted value passed via ``params`` and integrate
      over τ only — a pragmatic choice that matches CSDA-2013's intent (the
      paper considers single-parameter copulas; Student `df` is taken as
      known).
    """
    tau_min, tau_max = entry.value.TAU_MIN_MAX
    span = tau_max - tau_min
    pad  = max(1e-4 * max(span, 1.0), 1e-9)

    # Pad away from the boundary (Brent-style) to avoid degenerate evaluations.
    lo = tau_min + pad
    hi = tau_max - pad if span > 2 * pad else tau_max

    # Single-point case (Product copula).
    if hi <= lo:
        return _weighted_log_likelihood(cls, params, u, v, weights)

    taus = np.linspace(lo, hi, n_grid)
    log_evidence = np.full(n_grid, -np.inf, dtype=float)

    extras = {k: v_ for k, v_ in params.items() if k != "tau_k"}

    uv = np.column_stack((np.asarray(u, dtype=float),
                          np.asarray(v, dtype=float)))
    w  = np.asarray(weights, dtype=float)
    w  = w / (w.sum() + MIN_POSITIVE)

    for k, tau in enumerate(taus):
        try:
            cop = cls(tau_k=float(tau), **extras)
            log_pdfs = cop.logpdf_array(uv)
            log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, -1e6)
            log_evidence[k] = float(np.dot(w, log_pdfs))
        except Exception as exc:                             # pragma: no cover
            logger.debug("Huard: cls=%s τ=%.4f failed (%s)", cls.__name__, tau, exc)

    # log-mean-exp over the τ grid (uniform prior on [lo, hi]).
    finite = log_evidence[np.isfinite(log_evidence)]
    if finite.size == 0:
        return -np.inf
    a = finite.max()
    return float(a + np.log(np.exp(finite - a).mean()))


def _score_huard(cls, entry, params, u, v, weights) -> float:
    """Convenience wrapper: dispatch-compatible signature for Huard.

    Forwards to :func:`_huard_log_evidence` with the default τ-grid
    (50 points). Kept separate so the math primitive
    :func:`_huard_log_evidence` accepts ``n_grid`` for tests / advanced
    use, while the dispatch table stays homogeneous.
    """
    return _huard_log_evidence(cls, entry, params, u, v, weights)


_SCORE_FN: dict[str, callable] = {
    "mle":   _score_mle,
    "aic":   _score_aic,
    "bic":   _score_bic,
    "cvm":   _score_cvm,
    "huard": _score_huard,
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
    A TOML-compatible block ``{"name": ..., "tau": ..., …}``.
    """
    if criterion not in _SCORE_FN:
        raise ValueError(
            f"Unknown selection_criterion {criterion!r}. "
            f"Valid: {sorted(_SCORE_FN)}"
        )
    score_fn = _SCORE_FN[criterion]

    best_score = -np.inf
    best_block: dict = {"name": candidates[0], "tau": 0.0}

    for short in candidates:
        try:
            entry, cls = _resolve_candidate(short)
        except ValueError as exc:
            logger.warning("Skipping candidate %r: %s", short, exc)
            continue

        params = _fit_copula_params(cls, entry, u, v, weights)
        score  = score_fn(cls, entry, params, u, v, weights)
        logger.debug(
            "  candidate=%s  params=%s  criterion=%s  score=%.4f",
            short, params, criterion, score,
        )

        if score > best_score:
            best_score = score
            best_block = {"name": short, "tau": params["tau_k"]}
            for k, val in params.items():
                if k != "tau_k":
                    best_block[k] = val

    return best_block


# ---------------------------------------------------------------------------
# Weighted MLE for an arbitrary scipy.stats margin
# ---------------------------------------------------------------------------
#
# Default per-parameter bounds for the numerical optimiser. Any name not
# listed defaults to ``(-inf, +inf)``. Convention: scipy.stats spells the
# scale parameter "scale" and shape parameters use family-specific names
# ("a", "b" for beta; "df" for t; "s" for lognorm; "a" for gamma; …).
_DEFAULT_PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "loc":   (-np.inf, np.inf),
    "scale": (1e-8,    np.inf),
    # Shape-parameter conventions (scipy.stats):
    "df":    (1.0,     np.inf),     # Student-t:    df > 0
    "a":     (1e-8,    np.inf),     # gamma, beta:  shape > 0
    "b":     (1e-8,    np.inf),     # beta:         second shape > 0
    "s":     (1e-8,    np.inf),     # lognorm:      σ > 0
    "c":     (1e-8,    np.inf),     # weibull, etc.
    "alpha": (1e-8,    np.inf),
    "beta":  (1e-8,    np.inf),
}


def _fit_margin_weighted(blk: dict, y: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted maximum-likelihood update for a single ``[[margins]]`` block.

    Strategy:

    * **Closed form** for Gaussian (``dist="norm"``) — the only scalar family
      with a one-shot weighted MLE.  μ̂ = Σ w_i y_i, σ̂² = Σ w_i (y_i − μ̂)².

    * **Closed form** for multivariate Gaussian (``dist="multivariate_normal"``)
      — μ̂ = Σ w_i y_i,  Σ̂ = Σ w_i (y_i − μ̂)(y_i − μ̂)ᵀ. ``y`` is expected
      to be 2D ``(N, d)`` in this branch; a small diagonal jitter is added
      to ``Σ̂`` to keep it positive definite if a state is starved.

    * **Numerical fit** for every other ``scipy.stats`` continuous family
      via ``scipy.optimize.minimize`` (L-BFGS-B) on the negative weighted
      log-likelihood. Bounds come from :data:`_DEFAULT_PARAM_BOUNDS`.

    Importantly, the **family is preserved**: only the ``params`` are
    updated. (The previous ``_fit_gaussian_margin_weighted`` silently
    replaced any margin with a Gaussian, which was a footgun.)

    On any optimisation failure the original block is returned unchanged
    and a warning is logged.

    Returns a *new* dict ``{"params": {...}}`` suitable for
    ``blk.update(...)``.
    """
    dist_name   = blk.get("dist", "norm")
    init_params = dict(blk.get("params", {}))
    w_norm      = weights / (weights.sum() + MIN_POSITIVE)

    # ── Fast path: scalar Gaussian — closed form ──────────────────────
    if dist_name == "norm":
        mu   = float(np.dot(w_norm, y))
        sig2 = float(np.dot(w_norm, (y - mu) ** 2))
        sig  = float(np.sqrt(max(sig2, 1e-8)))
        return {"params": {"loc": mu, "scale": sig}}

    # ── Multivariate Gaussian — closed form on (N, d) data ────────────
    if dist_name == "multivariate_normal":
        return _fit_multivariate_gaussian_weighted(y, w_norm)

    # ── Numerical path for any other scipy.stats family ───────────────
    return _fit_margin_weighted_numerical(dist_name, init_params, y, w_norm)


def _fit_multivariate_gaussian_weighted(
    y: np.ndarray,
    w_norm: np.ndarray,
) -> dict:
    """Closed-form weighted MLE for the multivariate Gaussian distribution.

    Parameters
    ----------
    y      : np.ndarray, shape (N, d) — observations.
    w_norm : np.ndarray, shape (N,)  — non-negative weights summing to 1.

    Returns
    -------
    dict with key ``"params"`` containing ``{"mean": [...], "cov": [[...]]}``
    serialisable to TOML (lists, not arrays).

    Notes
    -----
    Adds a small diagonal jitter (``1e-8 * I``) to the empirical covariance
    so a starved state (very small Σ w) does not yield a singular matrix
    that would break the next E-step. This is the standard regulariser for
    Gaussian HMM ICE / EM.
    """
    Y = np.asarray(y, dtype=float)
    if Y.ndim != 2:
        raise ValueError(
            f"_fit_multivariate_gaussian_weighted expects 2D y of shape "
            f"(N, d); got {Y.shape}."
        )
    N, d = Y.shape

    # Weighted mean: μ_k = Σ_n w_n y_n
    mu = (w_norm[:, None] * Y).sum(axis=0)            # (d,)

    # Weighted covariance:  Σ = Σ_n w_n (y_n − μ)(y_n − μ)ᵀ
    centered = Y - mu                                  # (N, d)
    cov = (w_norm[:, None, None] * centered[:, :, None] * centered[:, None, :]).sum(axis=0)
    # Diagonal jitter for positive-definiteness under starved states.
    cov += 1e-8 * np.eye(d)

    return {
        "params": {
            "mean": [float(x) for x in mu],
            "cov":  [[float(x) for x in row] for row in cov],
        }
    }


def _fit_margin_weighted_numerical(
    dist_name:    str,
    init_params:  dict,
    y:            np.ndarray,
    w_norm:       np.ndarray,
) -> dict:
    """Generic numerical weighted MLE via L-BFGS-B on neg log-likelihood."""
    import scipy.stats as _ss
    from scipy.optimize import minimize

    try:
        dist_cls = getattr(_ss, dist_name)
    except AttributeError:
        logger.warning(
            "fit_margins: unknown scipy.stats family %r; keeping original params.",
            dist_name,
        )
        return {}

    if not init_params:
        logger.warning(
            "fit_margins: no initial params for %r; cannot run weighted MLE.",
            dist_name,
        )
        return {}

    # Preserve insertion order so the bounds and x0 vectors match.
    param_names = list(init_params.keys())
    x0          = np.array([init_params[k] for k in param_names], dtype=float)
    bounds      = [_DEFAULT_PARAM_BOUNDS.get(k, (-np.inf, np.inf))
                   for k in param_names]

    def _neg_loglik(theta: np.ndarray) -> float:
        kwargs = {k: float(theta[i]) for i, k in enumerate(param_names)}
        try:
            log_pdfs = dist_cls.logpdf(np.asarray(y, dtype=float), **kwargs)
            log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, -1e8)
            return -float(np.dot(w_norm, log_pdfs))
        except Exception as exc:
            logger.debug(
                "fit_margins[%s]: logpdf failed at %s — %s",
                dist_name, kwargs, exc,
            )
            return 1e15

    try:
        res = minimize(
            _neg_loglik, x0,
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-7},
        )
        if not np.all(np.isfinite(res.x)):
            raise ValueError("optimiser returned non-finite parameters")
        new_params = {k: float(res.x[i]) for i, k in enumerate(param_names)}
        return {"params": new_params}
    except Exception as exc:
        logger.warning(
            "fit_margins: numerical MLE failed for %r (%s); keeping initial params.",
            dist_name, exc,
        )
        return {}


# Back-compat alias — previously-public name; preserves the old return shape
# (with a redundant "dist" key) so any external caller still works.
def _fit_gaussian_margin_weighted(y: np.ndarray, weights: np.ndarray) -> dict:
    """Deprecated shim — call :func:`_fit_margin_weighted` instead."""
    out = _fit_margin_weighted({"dist": "norm"}, y, weights)
    return {"dist": "norm", **out}


# ---------------------------------------------------------------------------
# Generalized ICE — automatic margin family selection (SP-2016 GICE)
# ---------------------------------------------------------------------------
#
# When a [[margins]] block declares a ``candidates`` list, the M-step fits
# each candidate by weighted MLE and picks the winner via a configurable
# decision rule. This is the GICE extension of Derrode & Pieczynski (2016),
# §3, without the Pearson moment-matching shortcut — we always use a
# numerical L-BFGS-B fit.
#
# Decision rules (SP-2016 §3, ex. 3.1):
#   * "mle"        — max weighted log-likelihood (PLM equivalent for margins)
#   * "kolmogorov" — min sup |F_k(y) − F_n(y)| (paper's Example 3.1)
#   * "aic"        — −AIC = 2 logL − 2k
#   * "bic"        — −BIC = 2 logL − k log n

MARGIN_SELECTION_RULES = ("mle", "kolmogorov", "aic", "bic")
DEFAULT_MARGIN_SELECTION_RULE = "mle"

# Reasonable initial guesses for scipy.stats families that don't appear in
# the candidate's incoming `params`. Used by ``_select_margin_family`` so the
# user only has to declare the candidate dist names — the M-step picks
# data-aware starting points by itself.
_INIT_PARAM_HEURISTICS = {
    "norm":      lambda mu, sd: {"loc": mu, "scale": max(sd, 1e-3)},
    "gamma":     lambda mu, sd: {"a": max((mu / sd) ** 2, 0.5),
                                 "loc": mu - max((mu / sd) ** 2, 0.5) * sd,
                                 "scale": max(sd, 1e-3)},
    "invgamma":  lambda mu, sd: {"a": max(2.5, (mu / sd) ** 2 + 2),
                                 "loc": mu, "scale": max(sd, 1e-3)},
    "betaprime": lambda mu, sd: {"a": 2.0, "b": 5.0,
                                 "loc": mu, "scale": max(sd, 1e-3)},
    "lognorm":   lambda mu, sd: {"s": 0.5, "loc": min(mu - 3 * sd, mu),
                                 "scale": max(sd, 1e-3)},
    "expon":     lambda mu, sd: {"loc": mu - sd, "scale": max(sd, 1e-3)},
    "weibull_min": lambda mu, sd: {"c": 1.5, "loc": mu - sd,
                                    "scale": max(sd, 1e-3)},
    "beta":      lambda mu, sd: {"a": 2.0, "b": 5.0,
                                 "loc": mu - sd, "scale": max(sd, 1e-3)},
}

# Public re-exports for the GUI and downstream tooling. ``GICE_KNOWN_FAMILIES``
# is the canonical list of scipy.stats family names that ship with a
# data-aware init heuristic (other families still work — we fall back to
# ``scipy.stats.<dist>.fit`` — but these are the ones we recommend in the
# Edit-Margin dialog). ``SP2016_DEFAULT_CANDIDATES`` is the four-family set
# used in Derrode-Pieczynski SP 2016 §3 / Example 3.1.
GICE_KNOWN_FAMILIES        = tuple(_INIT_PARAM_HEURISTICS.keys())
SP2016_DEFAULT_CANDIDATES  = ("norm", "gamma", "invgamma", "betaprime")


def _data_aware_init_params(dist_name: str, y: np.ndarray, w: np.ndarray) -> dict:
    """Return an initial parameter dict for ``dist_name`` from weighted moments.

    Used as the L-BFGS-B starting point when the user gave only a candidate
    name without explicit ``init_params``. Falls back to scipy's first-pass
    MLE (``dist.fit(y)``) for families not in the heuristics table.
    """
    mu = float(np.dot(w, y))
    sd = float(np.sqrt(max(np.dot(w, (y - mu) ** 2), 1e-9)))
    if dist_name in _INIT_PARAM_HEURISTICS:
        return _INIT_PARAM_HEURISTICS[dist_name](mu, sd)
    # Generic fallback: scipy's unweighted fit for a starting point.
    import scipy.stats as _ss
    try:
        dist_cls = getattr(_ss, dist_name)
        fit_args = dist_cls.fit(y)
        # scipy fit() returns shape params... + (loc, scale)
        names = (list(getattr(dist_cls, "shapes", "") or "").split(",")
                 if dist_cls.shapes else [])
        names = [n.strip() for n in names if n.strip()]
        names += ["loc", "scale"]
        return {n: float(v) for n, v in zip(names, fit_args)}
    except Exception:
        return {"loc": mu, "scale": max(sd, 1e-3)}


def _weighted_log_likelihood_margin(
    dist_name: str, params: dict, y: np.ndarray, w: np.ndarray,
) -> float:
    """Weighted log-likelihood of a univariate scipy.stats family on ``y``."""
    import scipy.stats as _ss
    try:
        dist_cls = getattr(_ss, dist_name)
        log_pdfs = dist_cls.logpdf(y, **params)
        log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, -1e8)
        return float(np.dot(w, log_pdfs))
    except Exception:
        return -np.inf


def _kolmogorov_distance(
    dist_name: str, params: dict, y: np.ndarray, w: np.ndarray,
) -> float:
    """Weighted Kolmogorov–Smirnov distance to the empirical CDF.

    SP-2016 Example 3.1: ``D₁(y) = sup_y |F_k(y) − F_n(y)|`` evaluated at
    the data points ``y_n``. Using the *weighted* empirical CDF
    ``F_n(y) = Σ_{n: y_n ≤ y} w_n / Σ_n w_n`` makes the distance
    consistent with the ξ-weighted M-step.
    """
    import scipy.stats as _ss
    try:
        dist_cls = getattr(_ss, dist_name)
        cdf_th = dist_cls.cdf(y, **params)
    except Exception:
        return np.inf
    if not np.all(np.isfinite(cdf_th)):
        return np.inf
    # Weighted empirical CDF (right-continuous step function evaluated at y).
    w_norm = w / (w.sum() + MIN_POSITIVE)
    order  = np.argsort(y)
    cdf_emp_sorted = np.cumsum(w_norm[order])
    # Map each y[n] back to its CDF value (use the *right* limit of the step).
    cdf_emp = np.empty_like(cdf_emp_sorted)
    cdf_emp[order] = cdf_emp_sorted
    return float(np.max(np.abs(cdf_th - cdf_emp)))


def _select_margin_family(
    blk:    dict,
    y:      np.ndarray,
    weights: np.ndarray,
    rule:   str = DEFAULT_MARGIN_SELECTION_RULE,
) -> dict:
    """GICE M-step for one [[margins]] block: family + parameter selection.

    Reads ``blk["candidates"]`` (defaults to ``[blk["dist"]]`` for back-compat)
    and:

    1. For each candidate, runs a weighted MLE (numerical L-BFGS-B) starting
       from a data-aware moment-based heuristic (or from
       ``blk["init_params"]`` if the user supplied one).
    2. Applies the decision rule to pick the winner family.
    3. Returns ``{"dist": ..., "params": ...}`` ready for ``blk.update(...)``.

    Falls back to :func:`_fit_margin_weighted` (single-family numerical MLE)
    if no ``candidates`` field is present, preserving v0.5 behaviour.
    """
    cands_raw = blk.get("candidates", None)
    if not cands_raw:
        # No candidate set → keep v0.5 behaviour.
        return _fit_margin_weighted(blk, y, weights)

    # GICE is undefined for multivariate margins (the candidate vocabulary
    # is implicitly a list of *univariate* scipy.stats families). Skip it
    # cleanly — the M-step still updates the (mean, cov) of the
    # multivariate Gaussian via the closed-form path.
    if blk.get("dist") in _MULTIVARIATE_DIST_NAMES:
        logger.info(
            "GICE: 'candidates' ignored for multivariate margin (i=%s, "
            "dist=%s); only mean/cov updated.",
            blk.get("i"), blk.get("dist"),
        )
        return _fit_margin_weighted(blk, y, weights)

    if rule not in MARGIN_SELECTION_RULES:
        raise ValueError(
            f"Unknown margin_selection_rule {rule!r}. "
            f"Valid: {sorted(MARGIN_SELECTION_RULES)}"
        )

    w_norm = weights / (weights.sum() + MIN_POSITIVE)
    n      = float(np.sum(weights > 0)) or float(len(y))   # effective n

    fits: list[dict] = []
    for cand in cands_raw:
        # Use the user-specified init for the *current* dist if available;
        # otherwise derive a data-aware init from the weighted moments.
        if cand == blk.get("dist") and blk.get("params"):
            init = dict(blk["params"])
        else:
            init = _data_aware_init_params(cand, y, w_norm)
        result = _fit_margin_weighted_numerical(cand, init, y, w_norm)
        if not result.get("params"):
            logger.debug("GICE: candidate %r failed weighted MLE; skipping.", cand)
            continue
        params  = result["params"]
        log_lik = _weighted_log_likelihood_margin(cand, params, y, w_norm)
        n_params = len(params)
        fits.append({
            "dist":     cand,
            "params":   params,
            "log_lik":  log_lik,
            "n_params": n_params,
        })

    if not fits:
        logger.warning(
            "GICE: every candidate failed for block (i=%s, j=%s); "
            "keeping previous family.",
            blk.get("i"), blk.get("j"),
        )
        return {}

    # Higher = better (for any rule). Convention is consistent with the
    # copula-selection dispatcher.
    if rule == "mle":
        scores = [f["log_lik"] for f in fits]
    elif rule == "aic":
        scores = [2.0 * f["log_lik"] - 2.0 * f["n_params"] for f in fits]
    elif rule == "bic":
        scores = [2.0 * f["log_lik"] - f["n_params"] * np.log(max(n, 1.0))
                  for f in fits]
    elif rule == "kolmogorov":
        scores = [-_kolmogorov_distance(f["dist"], f["params"], y, w_norm)
                  for f in fits]
    else:                                                       # pragma: no cover
        raise AssertionError(f"unreachable rule {rule!r}")

    best_idx = int(np.argmax(scores))
    best     = fits[best_idx]
    logger.debug(
        "GICE margin (i=%s,j=%s) rule=%s: candidates=%s scores=%s → %s",
        blk.get("i"), blk.get("j"), rule,
        [f["dist"] for f in fits],
        [round(s, 3) for s in scores],
        best["dist"],
    )
    return {"dist": best["dist"], "params": best["params"]}


# ---------------------------------------------------------------------------
# ICE configuration parser
# ---------------------------------------------------------------------------

def _parse_ice_cfg(model: PMCModel, ice_cfg: dict | None) -> dict:
    """Merge TOML [ice] section with defaults.

    Recognised keys
    ---------------
    * ``fit_margins``      (bool, default False) — re-estimate margin params.
    * ``max_iter``         (int,  default 50)    — max EM iterations per start.
    * ``tol``              (float,default 1e-4)  — relative LL convergence.
    * ``candidates``       (list[str])           — copula SHORT_NAMEs.
    * ``patience``         (int,  default 3)     — consecutive LL regressions
                                                  before early stop.
    * ``n_starts``         (int,  default 1)     — number of independent ICE
                                                  runs from perturbed initial
                                                  models; the run with the
                                                  highest final log-likelihood
                                                  wins.
    * ``multistart_seed``  (int,  default 0)     — RNG seed for the
                                                  perturbations.
    * ``multistart_jitter``(float,default 0.10)  — relative perturbation
                                                  amplitude (10 % of each
                                                  parameter's natural scale).
    * ``selection_criterion``(str, default "mle")— rule for picking the
                                                  best candidate copula at
                                                  each M-step. One of
                                                  :data:`SELECTION_CRITERIA`
                                                  (``"mle"`` / ``"aic"`` /
                                                  ``"bic"`` / ``"huard"`` /
                                                  ``"cvm"``).
    * ``margin_selection_rule``(str, default "mle") — when a margin block
                                                  declares a ``candidates``
                                                  list (GICE — Derrode &
                                                  Pieczynski, SP 2016 §3),
                                                  the M-step picks the
                                                  winner family by this
                                                  rule. One of
                                                  :data:`MARGIN_SELECTION_RULES`
                                                  (``"mle"`` /
                                                  ``"kolmogorov"`` /
                                                  ``"aic"`` / ``"bic"``).
    """
    defaults: dict = {
        "fit_margins":            False,
        "max_iter":               50,
        "tol":                    1e-4,
        "candidates":             _DEFAULT_CANDIDATES,
        "patience":               3,
        "n_starts":               1,
        "multistart_seed":        0,
        "multistart_jitter":      0.10,
        "selection_criterion":    DEFAULT_SELECTION_CRITERION,
        "margin_selection_rule":  DEFAULT_MARGIN_SELECTION_RULE,
    }
    toml_ice = model.ice_config()
    cfg = {**defaults, **toml_ice}
    if ice_cfg:
        cfg.update(ice_cfg)
    return cfg


# ---------------------------------------------------------------------------
# Multistart: random perturbation of an initial model
# ---------------------------------------------------------------------------

def _perturb_initial_model(
    model: PMCModel,
    rng:   np.random.Generator,
    jitter: float = 0.10,
) -> PMCModel:
    """Return a *new* PMCModel obtained by perturbing the parameters of
    ``model`` with random noise of relative amplitude ``jitter``.

    The perturbations are designed to stay inside each parameter's valid
    domain so the rebuilt model is always a legal starting point:

    * **Prior** ``p`` (or ``A``): multiplied by ``exp(jitter · N(0,1))`` and
      renormalised. Symmetry is preserved for SR-PMC variants.
    * **Margin params**: ``loc`` is shifted by ``jitter · |scale|``; ``scale``
      and other strictly-positive params are scaled by ``exp(jitter · N(0,1))``.
    * **Copula τ**: shifted by ``jitter · (τ_max − τ_min)``, clipped via
      :meth:`CopulaEnum.correct_tau`.

    The perturbation amplitude is intentionally modest (10 % by default) so
    that the multistart explores nearby basins without throwing ICE at a
    completely different region of parameter space.
    """
    raw = model.raw  # already a deep copy

    # --- prior ---------------------------------------------------------
    prior = raw.get("prior", {})
    if "p" in prior:
        p = np.asarray(prior["p"], dtype=float)
        noise = np.exp(jitter * rng.standard_normal(p.shape))
        p = p * noise
        # Preserve SR-PMC symmetry of the joint: p[i,j] = p[j,i].
        p = 0.5 * (p + p.T)
        p = np.clip(p, 0.0, None)
        p /= p.sum()
        prior["p"] = p.tolist()
    elif "A" in prior:
        A = np.asarray(prior["A"], dtype=float)
        noise = np.exp(jitter * rng.standard_normal(A.shape))
        A = A * noise
        A = np.clip(A, 0.0, None)
        A /= A.sum(axis=1, keepdims=True)
        prior["A"] = A.tolist()

    # --- margins -------------------------------------------------------
    for blk in raw.get("margins", []):
        params = blk.get("params", {})
        if not params:
            continue
        scale = float(params.get("scale", 1.0))
        for k, v in list(params.items()):
            if k == "loc":
                params[k] = float(v) + jitter * abs(scale) * float(rng.standard_normal())
            elif k in ("scale", "df", "a", "b", "s", "c", "alpha", "beta"):
                # Strictly-positive params: log-normal multiplicative noise.
                params[k] = float(v) * float(np.exp(jitter * rng.standard_normal()))

    # --- copulas -------------------------------------------------------
    for blk in raw.get("copulas", []):
        short = blk.get("name")
        entry = CopulaEnum.from_short_name(short) if short else None
        if entry is None:
            continue
        tau_min, tau_max = entry.value.TAU_MIN_MAX
        span = tau_max - tau_min
        tau  = float(blk.get("tau", 0.5 * (tau_min + tau_max)))
        tau += jitter * span * float(rng.standard_normal())
        # Use correct_tau to clip + log if out-of-range.
        blk["tau"] = entry.correct_tau(tau)
        # Perturb extra parameters of multi-parameter copulas
        # (currently: BB1.delta, Student.df).
        cls_name = entry.value.CLASS_NAME
        for ekey, (xlo, xhi, _) in _EXTRA_PARAM_BOUNDS.get(cls_name, {}).items():
            if ekey in blk:
                v = float(blk[ekey]) * float(np.exp(jitter * rng.standard_normal()))
                blk[ekey] = float(np.clip(v, xlo, xhi))

    return PMCModel.from_dict(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ice(
    model: PMCModel,
    Y: np.ndarray,
    ice_cfg: dict | None = None,
    progress_cb=None,
) -> tuple[PMCModel, IceTrace]:
    """Iterative Conditional Estimation (ICE) for unsupervised PMC/HMC fitting.

    With ``n_starts > 1`` (config), runs ICE several times from random
    perturbations of ``model`` and keeps the run with the highest final
    log-likelihood — this hardens the fit against multimodal log-likelihoods
    (a real concern for archimedean copula mixtures and HMC variants with
    many similar margins). The first start is always the unperturbed model.

    Parameters
    ----------
    model        : PMCModel — initial parameter values (starting point).
    Y            : np.ndarray, shape (N,) — observation sequence.
    ice_cfg      : dict, optional — override ICE settings from TOML [ice] section.
                    See :func:`_parse_ice_cfg` for the recognised keys.
    progress_cb  : callable, optional — invoked as
                    ``progress_cb(it, max_iter, log_lik, run_tag)`` after every
                    E-step. Designed for the GUI's progress bar; safe to leave
                    as ``None`` for headless / scripted use.

    Returns
    -------
    fitted_model : PMCModel — model with updated parameters (the best run
                              when multistart is enabled).
    trace        : IceTrace — captured per-iteration diagnostics. Use
                    ``trace.log_liks`` for the plain log-likelihood history.
                    When multistart was active, ``trace.multistart_runs``
                    holds the traces of the non-best runs.
    """
    cfg       = _parse_ice_cfg(model, ice_cfg)
    n_starts  = max(1, int(cfg.get("n_starts", 1)))

    if n_starts == 1:
        return _ice_single_run(model, Y, cfg, progress_cb=progress_cb)

    rng = np.random.default_rng(int(cfg.get("multistart_seed", 0)))
    jitter = float(cfg.get("multistart_jitter", 0.10))

    # Mute the per-run "ICE: variant=…" header except for the first run, to
    # avoid log spam — instead we emit a single summary at the end.
    runs: list[tuple[float, PMCModel, IceTrace]] = []

    logger.info("ICE multistart: n_starts=%d  jitter=%.2f", n_starts, jitter)

    for s in range(n_starts):
        if s == 0:
            init_mdl = model
            tag = "unperturbed"
        else:
            init_mdl = _perturb_initial_model(model, rng, jitter=jitter)
            tag = f"perturbed-{s}"
        try:
            fitted_s, trace_s = _ice_single_run(
                init_mdl, Y, cfg, run_tag=tag, progress_cb=progress_cb,
            )
            final_ll = trace_s.log_liks[-1] if trace_s.log_liks else -np.inf
        except Exception as exc:                          # pragma: no cover
            logger.warning("ICE multistart: run %d (%s) failed: %s", s, tag, exc)
            continue
        logger.info(
            "ICE multistart: run %d (%s) → final log-lik = %.4f  (iters=%d)",
            s, tag, final_ll, len(trace_s.log_liks),
        )
        runs.append((final_ll, fitted_s, trace_s))

    if not runs:                                          # pragma: no cover
        # All runs failed — fall back to a single unperturbed run so callers
        # always get a model back (matches the original semantics).
        return _ice_single_run(model, Y, cfg, progress_cb=progress_cb)

    best_idx, (best_ll, best_mdl, best_trace) = max(
        enumerate(runs), key=lambda kv: kv[1][0],
    )
    logger.info(
        "ICE multistart: best run = #%d  (final log-lik = %.4f)",
        best_idx, best_ll,
    )
    # Keep the losing-run traces for the GUI's multistart-comparison view.
    best_trace.multistart_runs = [
        trace for k, (_, _, trace) in enumerate(runs) if k != best_idx
    ]
    return best_mdl, best_trace


def ice_image(
    model: PMCModel,
    img: np.ndarray,
    ice_cfg: dict | None = None,
    progress_cb=None,
) -> tuple[PMCModel, IceTrace]:
    """ICE on a 2D image — linearises along the gilbert path then calls :func:`ice`.

    Parameters
    ----------
    model    : PMCModel — initial model (starting point for ICE).
    img      : np.ndarray
        Shape ``(H, W)`` for grayscale (when ``model.d == 1``) or
        ``(H, W, d)`` for multi-channel (when ``model.d > 1``).
    ice_cfg  : dict, optional — same as :func:`ice`.
    progress_cb : callable, optional — same as :func:`ice`.

    Returns
    -------
    fitted_model : PMCModel
    trace        : IceTrace
    """
    from prg.pmc.peano import image_to_signal

    if img.ndim not in (2, 3):
        raise ValueError(
            f"ice_image expects 2D (H, W) or 3D (H, W, d); got shape {img.shape}."
        )
    img_d = 1 if img.ndim == 2 else img.shape[2]
    if img_d != model.d:
        raise ValueError(
            f"Image channels ({img_d}) do not match model.d = {model.d}."
        )
    Y = image_to_signal(img)
    return ice(model, Y, ice_cfg=ice_cfg, progress_cb=progress_cb)


def _ice_single_run(
    model: PMCModel,
    Y: np.ndarray,
    cfg: dict,
    run_tag: str = "",
    progress_cb=None,
) -> tuple[PMCModel, IceTrace]:
    """Single ICE run — extracted from :func:`ice` to support multistart.

    ``cfg`` must already be the merged dict returned by :func:`_parse_ice_cfg`.
    ``run_tag`` is an optional label included in log messages (used by the
    multistart driver to disambiguate concurrent runs in the log).
    """
    max_iter  = int(cfg["max_iter"])
    tol       = float(cfg["tol"])
    candidates = list(cfg["candidates"])
    fit_margins = bool(cfg["fit_margins"])
    # Patience: stop early if LL regresses on this many consecutive iterations.
    patience  = int(cfg.get("patience", 3))
    selection_criterion = str(cfg.get(
        "selection_criterion", DEFAULT_SELECTION_CRITERION,
    ))
    if selection_criterion not in _SCORE_FN:
        raise ValueError(
            f"Unknown selection_criterion {selection_criterion!r}. "
            f"Valid: {sorted(_SCORE_FN)}"
        )
    margin_selection_rule = str(cfg.get(
        "margin_selection_rule", DEFAULT_MARGIN_SELECTION_RULE,
    ))
    if margin_selection_rule not in MARGIN_SELECTION_RULES:
        raise ValueError(
            f"Unknown margin_selection_rule {margin_selection_rule!r}. "
            f"Valid: {sorted(MARGIN_SELECTION_RULES)}"
        )
    log_prefix = f"ICE[{run_tag}]" if run_tag else "ICE"

    var = model.variant
    K   = model.K
    N   = len(Y)

    log_liks: list[float] = []
    # Per-iteration trace buffers (captured BEFORE the M-step at iter t, so
    # they correspond to the model that produced ``log_liks[t]``).
    tau_buf:    list[np.ndarray]        = []
    fam_buf:    list[list[list[str]]]   = []
    p_buf:      list[np.ndarray]        = []
    margin_buf: list[list[dict]]        = []

    regress_streak  = 0   # consecutive iterations with LL decrease

    # Public ``raw`` already returns a deep copy — safe to mutate.
    raw = model.raw

    # Build a mutable model reference updated each iteration
    current = PMCModel.from_dict(raw)

    logger.info(
        "%s: variant=%s  K=%d  N=%d  max_iter=%d  candidates=%s",
        log_prefix, var.value, K, N, max_iter, candidates,
    )

    for it in range(max_iter):
        # ── E-step ────────────────────────────────────────────────────────
        W, f_pdf = precompute_weights(current, Y)
        alpha_hat, log_lik = forward(current, Y, W=W, f_pdf=f_pdf)
        beta_hat  = backward(current, Y, W=W)
        gamma     = smooth(alpha_hat, beta_hat)       # (N, K) marginal posteriors
        xi        = _joint_posteriors(alpha_hat, W, beta_hat)  # (N-1, K, K)

        log_liks.append(log_lik)
        # Capture the snapshot of ``current`` corresponding to this LL value.
        tau_kk, fam_kk = _snapshot_tau_family(current)
        tau_buf.append(tau_kk)
        fam_buf.append(fam_kk)
        p_buf.append(_snapshot_prior_p(current))
        margin_buf.append(_snapshot_margins(current))

        logger.info("%s iter %d: log-lik = %.4f", log_prefix, it, log_lik)
        if progress_cb is not None:
            try:
                progress_cb(it, max_iter, float(log_lik), run_tag)
            except Exception as exc:                       # pragma: no cover
                # A faulty callback must not abort the optimisation.
                logger.debug("ICE progress_cb raised %s; ignoring.", exc)

        # Convergence / regression check (after first iteration)
        if it > 0:
            diff_signed = log_liks[-1] - log_liks[-2]   # >0 if improving
            ref         = abs(log_liks[-2]) + 1e-10

            # Regression: LL went down. ICE is not strictly EM, so isolated
            # decreases are tolerable, but persistent regression signals a bug
            # or a degenerate optimum.
            if diff_signed < -tol * ref:
                regress_streak += 1
                logger.warning(
                    "%s iter %d: log-lik regressed by %.4e (streak=%d/%d).",
                    log_prefix, it, -diff_signed, regress_streak, patience,
                )
                if regress_streak >= patience:
                    logger.warning(
                        "%s: %d consecutive regressions — stopping early at iter %d.",
                        log_prefix, regress_streak, it,
                    )
                    break
            else:
                regress_streak = 0

            # Convergence on absolute relative change
            if abs(diff_signed) / ref < tol:
                logger.info(
                    "%s converged at iteration %d (Δ/|LL| = %.2e).",
                    log_prefix, it, abs(diff_signed) / ref,
                )
                break

        # ── M-step ───────────────────────────────────────────────────────

        # 1. Update prior distribution.
        # The raw average of ξ_n is asymmetric by O(1/√N) finite-sample noise;
        # under SR-PMC (the framework this package commits to) the true joint
        # ``p[i,j]`` is *exactly* symmetric. We therefore symmetrise ξ̄
        # before further processing — this both eliminates noise in p_hat
        # and prevents the SR-symmetry warning in PMCModel from firing
        # at every ICE iteration on PMC variants.
        p_hat = xi.sum(axis=0) / (N - 1)
        p_hat = 0.5 * (p_hat + p_hat.T)            # SR-PMC: enforce p[i,j] = p[j,i]
        p_hat = np.clip(p_hat, 0.0, None)
        p_hat /= p_hat.sum()

        if var in (Variant.HMC_IN, Variant.HMC_IN2, Variant.HMC_DN):
            # Convert joint → row-stochastic A
            pi_hat = p_hat.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                A_hat = np.where(pi_hat[:, None] > 0, p_hat / pi_hat[:, None], 1.0 / K)
            # Normalize rows
            row_s = A_hat.sum(axis=1, keepdims=True)
            A_hat = np.where(row_s > 0, A_hat / row_s, 1.0 / K)
            raw["prior"]["A"] = A_hat.tolist()
        else:
            raw["prior"]["p"] = p_hat.tolist()

        # 2. Update margins (optional). Under SR-PMC there are exactly K
        #    state-indexed densities, regardless of variant; the family
        #    declared in each block is preserved, only the params are
        #    re-estimated by weighted MLE.
        #
        #    The weight on Y[k] for state i is the *full* posterior of being
        #    in state i at time k:
        #
        #        w_i[k] = γ_k(i) = P(X_k = i | Y)
        #
        #    Equivalently in terms of joint posteriors:
        #
        #        w_i[k] = Σ_j ξ[k, i, j]      (k = 0…N-2; "first-view" sum)
        #               = Σ_j ξ[k-1, j, i]    (k = 1…N-1; "second-view" sum)
        #               = γ_k(i)              (forward-backward marginal)
        #
        #    Pooling over j is the SR-PMC consistent estimator: K MLE per
        #    M-step instead of K² (one per (i, j) block) — eliminating the
        #    redundant K-fold replication of each f_i.
        #
        #    Reference: SR-PMC reversibility — Derrode-Pieczynski (CSDA 2013),
        #    Eq. (14).
        if fit_margins:
            margins_raw = raw.get("margins", [])
            for blk in margins_raw:
                i_idx = int(blk["i"])
                w_n   = gamma[:, i_idx]            # P(X_n=i | Y)
                # GICE: if the block declares a ``candidates`` list, also
                # select the family at this M-step (SP-2016 §3); otherwise
                # the helper falls through to the v0.5 single-family fit.
                blk.update(_select_margin_family(
                    blk, Y, w_n, rule=margin_selection_rule,
                ))

        # 3. Update copulas (only for variants that use copulas)
        if var.uses_copula:
            copulas_raw = raw.get("copulas", [])

            # Pre-compute marginal CDFs once per state (K cdf vectors,
            # not K²) — under SR-PMC, F_{ij} = F_i regardless of j.
            f_cdf_state = np.zeros((N, K))
            for kk in range(K):
                f_cdf_state[:, kk] = current.margin(kk).cdf_vec(Y)
            np.clip(f_cdf_state, EPS, ONE_MINUS_EPS, out=f_cdf_state)

            for blk in copulas_raw:
                ii = int(blk["i"])
                jj = int(blk["j"])

                # Weighted pseudo-observations
                weights_pair = xi[:, ii, jj]          # shape (N-1,)
                total_w = weights_pair.sum()
                if total_w < 1e-12:
                    logger.debug("  (i=%d,j=%d): negligible weight, skip.", ii, jj)
                    continue

                # Pseudo-obs of pair (i, j): u = F_i(Y_n), v = F_j(Y_{n+1}).
                u_arr = f_cdf_state[: N - 1, ii]
                v_arr = f_cdf_state[1:,      jj]

                logger.debug("  fitting copula (i=%d, j=%d)  Σw=%.4f", ii, jj, total_w)
                best_blk = _select_and_fit_copula(
                    candidates, u_arr, v_arr, weights_pair,
                    criterion=selection_criterion,
                )
                blk.update(best_blk)

        # Rebuild the model with updated raw dict
        current = PMCModel.from_dict(raw)

    # Pack the per-iteration buffers into an :class:`IceTrace`.
    trace = IceTrace(
        log_liks       = log_liks,
        tau_history    = (np.stack(tau_buf, axis=0) if tau_buf
                          else np.empty((0, K, K), dtype=float)),
        family_history = fam_buf,
        p_history      = (np.stack(p_buf, axis=0) if p_buf
                          else np.empty((0, K, K), dtype=float)),
        margin_history = margin_buf,
        run_tag        = run_tag,
        candidates     = list(candidates),
    )
    return current, trace


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from prg.pmc.logging_setup import configure as _cfg_log
    _cfg_log(level=logging.INFO)

    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate
    from prg.pmc.inference import error_rate, classify

    models_dir = pathlib.Path(__file__).parent / "models"

    # Test on a subset of models (those with copulas are more interesting)
    test_files = [
        "hmc_in_gauss_k2.toml",
        "pmc_gauss_k2.toml",
    ]

    for name in test_files:
        toml_file = models_dir / name
        if not toml_file.exists():
            continue
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        mdl = PMCModel(toml_file)

        # Simulate a sequence
        X_ref, Y = simulate(mdl, N=1000, seed=42)

        # Build a perturbed initial model (shift means by +0.3, τ by −0.1)
        raw_init = mdl.raw
        for blk in raw_init.get("margins", []):
            p = blk.get("params", {})
            if "loc" in p:
                p["loc"] = p["loc"] + 0.3
        for blk in raw_init.get("copulas", []):
            blk["tau"] = max(-0.9, blk.get("tau", 0.0) - 0.1)
        init_mdl = PMCModel.from_dict(raw_init)

        # Run ICE
        fitted, trace = ice(
            init_mdl,
            Y,
            ice_cfg={"max_iter": 10, "candidates": ["Gauss", "Clayton", "GH"]},
        )
        print(f"  log-lik history: {[f'{ll:.2f}' for ll in trace.log_liks]}")
        if trace.tau_history.size:
            print(f"  τ_final = {trace.tau_history[-1].round(3).tolist()}")

        # Evaluate on fitted model
        X_hat, _, _ = classify(fitted, Y)
        er = error_rate(X_ref, X_hat)
        print(f"  error rate (fitted): {er:.4f}  ({er*100:.1f} %)")

        print(f"  Fitted prior p =\n{fitted.prior_p.round(4)}")
        if fitted.variant.uses_copula:
            for ii in range(fitted.K):
                for jj in range(fitted.K):
                    cop = fitted.copula(ii, jj)
                    print(f"    copula({ii},{jj}): {cop.copula_enum.value.SHORT_NAME}"
                          f"  τ={cop.params['tau_k']:.4f}")
