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
          weighted by ξ_n(i,j)  for n = 1, …, N-1  (F_{ij} = F_i for state
          margins). With ``copula_margins = "empirical"`` the F are the
          posterior-weighted empirical margins of :func:`_empirical_margin_cdfs`
          (a weighted rank pseudo-likelihood; the E-step keeps the model's F).
       b. Select best copula family from 'candidates' by weighted log-likelihood.
       c. Estimate τ_{ij} by weighted MLE on τ ∈ [τ_min, τ_max].

    3. (Optional, ``fit_margins``) Margin re-estimation by weighted MLE, the
       declared family kept (or selected among ``candidates`` — GICE):
       * state margins f_i : y_n with weight γ_n(i);
       * pair margins f_ij (general PMC, DerrodePieczynski_CSDA2013
         Eqs. 12–14) — "dual view": y_n with weight ξ_n(i,j) and
         y_{n+1} with weight ξ_n(j,i), since the right margin of the
         pair (j,i) is f_ij. For a Gaussian margin,
         μ̂_{ij} = Σ w y / Σ w and σ̂²_{ij} = Σ w (y − μ̂)² / Σ w
         over that sample. An ICE-style estimator, not an exact EM
         M-step (see ``_pair_margin_sample``).

Relation to the papers' ICE (DerrodePieczynski_CSDA2013 §4.2, DerrodePieczynski_SP2016 §3)
-----------------------------------------------
This is a *responsibility-weighted* variant of ICE, not the scheme of
Derrode & Pieczynski (2013, §4.2, Eqs. 21–24) and (2016, §3 steps (b)–(c),
Remark 3.1). There, only the prior p_ij is updated through the conditional
expectation of Eq. 22; the copula (and margin) parameters, whose conditional
expectation is not computable, are estimated by the complete-data estimator
applied to **one** posterior draw x^(q) ~ p(x | y, θ^q) — "replacing x_{1:N}
by x^q_{1:N} (L = 1)" — i.e. on the hard sub-samples y^{ij}(x^(q)). Here every
ξ_n(i,j) enters as a weight instead of a 0/1 membership, which makes the
M-step deterministic and EM-like (for known margins it *is* EM's M-step for
the copula parameters). :func:`pmcprg.pmc.sem.sem` is the hard-draw estimator
and therefore the closest thing in the package to the papers' ICE with
L = 1 — except that it also takes p_ij from the draw rather than from the
expectation. Neither reproduces the papers' scheme to the letter; the
reproduction report (``report/csda2013_reproduction.tex``, §Exp. 3) says
which one it ran. (Audit K-4.)

ICE configuration (TOML [ice] section or dict)
-----------------------------------------------
  fit_margins : bool   (default False) — re-estimate margin parameters.
  max_iter    : int    (default 50)    — maximum EM iterations.
  tol         : float  (default 1e-4)  — relative log-likelihood convergence threshold.
  candidates  : list[str]              — SHORT_NAMEs of candidate copula families.
                 default: all 1-parameter available families except Product.
  n_starts, multistart_seed, multistart_jitter, multistart_workers,
  multistart_families — multistart (see :func:`_parse_ice_cfg` for the full list).
  return_best_iterate : bool (default False) — return the iterate with the
                 highest log-likelihood instead of the last one.
  missing_strategy, missing_draws, missing_seed, gap_nodes — missing
  observations (NaN rows of Y): see :func:`ice`, section "Missing observations".
  copula_margins : str (default "parametric") — "empirical" computes the
                 copula step's pseudo-observations from posterior-weighted
                 empirical margins instead of the model's F (AUDIT_COPULES
                 FR-7 a; see :func:`_parse_ice_cfg` and
                 :func:`_empirical_margin_cdfs`).
  missingness : str (default "model") — the missingness mechanism: the
                 model's, held fixed; "ignorable"; or "state" /
                 "state-markov", estimated (see :func:`ice`, section
                 "Missingness mechanism").

References
----------
ICE itself (the references DerrodePieczynski_SP2016 gives for the method):

* Pieczynski, W. (1992). Statistical image segmentation. *Machine Graphics
  and Vision* 1(1/2), 261–268.
* Delignon, Y., Marzouki, A. & Pieczynski, W. (1997). Estimation of
  generalized mixtures and its application in image segmentation. *IEEE
  Trans. Image Processing* 6(10), 1364–1375.
* Giordana, N. & Pieczynski, W. (1997). Estimation of generalized
  multisensor hidden Markov chains and unsupervised image segmentation.
  *IEEE Trans. PAMI* 19(5), 465–475.

The two papers this package reproduces:

* DerrodePieczynski_CSDA2013 — Derrode, S. & Pieczynski, W. (2013).
  Unsupervised data classification using pairwise Markov chains with
  automatic copulas selection. *CSDA* 63, 81–98.
* DerrodePieczynski_SP2016 — Derrode, S. & Pieczynski, W. (2016).
  Unsupervised classification using hidden Markov chain with unknown
  noise copulas and margins. *Signal Processing* 128, 8–17.
"""

import copy as _copy
import logging
from dataclasses import dataclass, field

import numpy as np

from pmcprg.copulas._base import CopulaEnum
from pmcprg.copulas._fit import validate_weights
# The weighted copula engine — weighted Kendall's τ, weighted MLE, weighted
# family selection — lives in :mod:`pmcprg.copulas._weighted` (audit FR-12),
# shared with the public ``CopulaVirt.fit(weights=…)``. Its names are
# imported here under the names they had in this module, which the M-step,
# the GUI, the diagnostics and the tests use.
from pmcprg.copulas._weighted import (  # noqa: F401  (re-exported names)
    _build_extra_param_bounds,
    _contiguous_weight_balanced_folds,
    _copula_block,
    _copula_candidate_fits,
    _copula_kwargs,
    _copula_placeholder,
    DEFAULT_SELECTION_CRITERION,
    _effective_n,
    EXTRA_PARAM_BOUNDS,
    _fit_copula_params,
    FIT_FAILED_KEY,
    _huard_log_evidence,
    _HUARD_MAX_REFINE,
    _HUARD_N_GRID,
    _HUARD_REFINE_HALF,
    _HUARD_REFINE_POINTS,
    _HUARD_RESOLVE_NAT,
    _LOGLIK_FAIL,
    _LOGPDF_FLOOR,
    _OPT_FAIL_PENALTY,
    _pad_tau_bounds,
    _resolve_candidate,
    _run_start_index,
    _run_starts,
    _score_aic,
    _score_bic,
    _score_cvm,
    _SCORE_FN,
    _score_huard,
    _score_huard_common,
    _score_huard_global,
    _score_mle,
    _score_xvcic,
    _select_and_fit_copula,
    SELECTION_CRITERIA,
    _TAU_PAD_ABS,
    _TAU_PAD_REL,
    _weight_before,
    _weighted_cvm_statistic,
    _weighted_ecdf,
    _weighted_increasing_pairs,
    _weighted_kendall_tau,
    _weighted_log_likelihood,
    _weighted_mle_tau,
    _XVCIC_MIN_W,
    _XVCIC_N_FOLDS,
    _XVCIC_PURGE,
)
from pmcprg.pmc.inference import (
    backward,
    forward,
    joint_posteriors,
    precompute_weights,
    smooth,
)
from pmcprg.pmc.model import PMCModel, Variant, MULTIVARIATE_DISTS as _MULTIVARIATE_DIST_NAMES
from pmcprg.numerics import EPS, ONE_MINUS_EPS, MIN_POSITIVE

logger = logging.getLogger(__name__)


__all__ = [
    # canonical public API
    "IceTrace",
    "IceResult",
    "ice",
    "ice_image",
    # public configuration surface (consumed by GUI / tests)
    "EXTRA_PARAM_BOUNDS",
    "FIT_FAILED_KEY",
    "INIT_STRATEGIES",
    "SELECTION_CRITERIA",
    "DEFAULT_SELECTION_CRITERION",
    "MARGIN_SELECTION_RULES",
    "DEFAULT_MARGIN_SELECTION_RULE",
    "GICE_KNOWN_FAMILIES",
    "SP2016_DEFAULT_CANDIDATES",
    "MISSING_STRATEGIES",
    "DEFAULT_MISSING_STRATEGY",
    "DEFAULT_MISSING_DRAWS",
    "COPULA_MARGIN_MODES",
    "DEFAULT_COPULA_MARGINS",
    "MISSINGNESS_MODES",
    "DEFAULT_MISSINGNESS",
]


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
    best_iter      : int — index of the highest ``log_liks`` value (first
                     one on ties, NaN ranked lowest); ``-1`` when the trace
                     is empty. Set whatever ``return_best_iterate``.
    returned_iter  : int — iterate q of the returned parameter set θ^q, so
                     that ``log_liks[returned_iter]`` is its log-likelihood
                     when ``returned_iter < len(log_liks)``. With
                     ``return_best_iterate`` it equals ``best_iter``. Without
                     it, ``len(log_liks) - 1`` when ICE stopped early
                     (convergence, ``patience``) and ``len(log_liks)`` when
                     the run used all ``max_iter`` iterations: the model of
                     the last M-step was returned without being evaluated
                     (always the case for SEM). ``-1`` when not recorded.
    degenerate     : ``list[DegenerateFinding] | None`` — degenerate states
                     of the returned model
                     (:func:`pmcprg.pmc._estim_common.degenerate_states`),
                     ``[]`` when none; ``None`` when not checked (the traces
                     of ``multistart_runs``, whose models are not kept).
    missingness_history : ``list[dict | None]`` — length ``T``: at index
                     ``t`` the ``[missingness]`` table of the iterate
                     (``{"mechanism": "state", "rates": [...]}`` or
                     ``{"mechanism": "state-markov", "onset": [...],
                     "persistence": [...]}``, :mod:`pmcprg.pmc.missingness`),
                     ``None`` when it is ignorable. Constant unless the
                     ``missingness`` config key estimates the mechanism.
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
    best_iter:       int                      = -1
    returned_iter:   int                      = -1
    degenerate:      list | None              = None
    missingness_history: list[dict | None]    = field(default_factory=list)

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


# Numerical-regularisation constants of the margin and k-means steps (audit
# Q-14); those of the copula step moved with it to pmcprg.copulas._weighted.
_VAR_FLOOR:     float = 1e-8   # floor on a fitted variance / σ²
_COV_JITTER:    float = 1e-8   # diagonal jitter added to a weighted covariance
_KMEANS_N_INIT: int   = 10     # n_init passed to sklearn KMeans


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
        sig  = float(np.sqrt(max(sig2, _VAR_FLOOR)))
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
    cov += _COV_JITTER * np.eye(d)

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
            log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, _LOGPDF_FLOOR)
            return -float(np.dot(w_norm, log_pdfs))
        except Exception as exc:
            logger.debug(
                "fit_margins[%s]: logpdf failed at %s — %s",
                dist_name, kwargs, exc,
            )
            return _OPT_FAIL_PENALTY

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
#
# Provenance (audit K-16): "kolmogorov" is DerrodePieczynski_SP2016
# Example 3.1, Eq. (10) (Derrode & Pieczynski 2016, Signal Processing) —
# the rule the paper used for margin selection. The margin "mle", "aic"
# and "bic" rules are package additions (the paper's PLM rule,
# DerrodePieczynski_SP2016 Example 3.3 Eq. (11), is stated for copula
# selection).

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
    sd = float(np.sqrt(max(np.dot(w, (y - mu) ** 2), _VAR_FLOOR)))
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
        log_pdfs = np.where(np.isfinite(log_pdfs), log_pdfs, _LOGPDF_FLOOR)
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
    # Weighted empirical CDF, right-continuous:  F_n(y) = Σ_{m: y_m ≤ y} w_m.
    # A plain cumsum over argsort gives *tied* points distinct partial sums
    # (e.g. y = [1, 1, 1, 2] → 0.25, 0.50, 0.75 instead of 0.75, 0.75, 0.75),
    # which fabricates spurious KS deviations on quantised/discrete data
    # (image gray levels, sensor counts). We broadcast each tied run's final
    # cumulative weight to all its members so identical y share one F_n value
    # (audit N-4).
    w_norm   = w / (w.sum() + MIN_POSITIVE)
    order    = np.argsort(y, kind="mergesort")        # stable
    y_sorted = y[order]
    cum      = np.cumsum(w_norm[order])
    # Index of the last element of each tied run (in sorted order).
    last_in_run    = np.searchsorted(y_sorted, y_sorted, side="right") - 1
    cdf_emp_sorted = cum[last_in_run]
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

    fits, scores = _margin_candidate_fits(blk, y, weights, rule)

    if not fits:
        logger.warning(
            "GICE: every candidate failed for block (i=%s, j=%s); "
            "keeping previous family.",
            blk.get("i"), blk.get("j"),
        )
        return {}

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


def _margin_candidate_fits(
    blk: dict,
    y: np.ndarray,
    weights: np.ndarray,
    rule: str,
) -> tuple[list[dict], list[float]]:
    """Fit and score every candidate family of a univariate GICE margin block.

    The loop of :func:`_select_margin_family` (the block must declare
    ``candidates``). Returns ``(fits, scores)``: one dict ``{"dist",
    "params", "log_lik", "n_params"}`` per candidate whose weighted MLE
    succeeded, in declaration order, and the matching ``rule`` scores (higher
    is better). Shared with the multiple-imputation M-step of ICE with missing
    observations, which pools the scores over the completed series.
    """
    if rule not in MARGIN_SELECTION_RULES:
        raise ValueError(
            f"Unknown margin_selection_rule {rule!r}. "
            f"Valid: {sorted(MARGIN_SELECTION_RULES)}"
        )
    cands_raw = blk["candidates"]

    # Normalised weights drive the *fits* and the KS distance (which needs a
    # proper empirical CDF); the penalised scores below need the total
    # log-likelihood and the effective sample size Σw instead — same
    # mean-versus-sum mismatch as on the copula side.
    w_norm = weights / (weights.sum() + MIN_POSITIVE)
    n_eff  = float(np.sum(weights)) or float(len(y))

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
        log_lik = _weighted_log_likelihood_margin(cand, params, y, weights)
        n_params = len(params)
        fits.append({
            "dist":     cand,
            "params":   params,
            "log_lik":  log_lik,
            "n_params": n_params,
        })

    if not fits:
        return fits, []

    # Higher = better (for any rule). Convention is consistent with the
    # copula-selection dispatcher.
    if rule == "mle":
        scores = [f["log_lik"] for f in fits]
    elif rule == "aic":
        scores = [2.0 * f["log_lik"] - 2.0 * f["n_params"] for f in fits]
    elif rule == "bic":
        scores = [2.0 * f["log_lik"] - f["n_params"] * np.log(max(n_eff, 1.0))
                  for f in fits]
    elif rule == "kolmogorov":
        scores = [-_kolmogorov_distance(f["dist"], f["params"], y, w_norm)
                  for f in fits]
    else:                                                       # pragma: no cover
        raise AssertionError(f"unreachable rule {rule!r}")
    return fits, scores


# ---------------------------------------------------------------------------
# ICE configuration parser
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# M-step — extracted for reuse by both the ICE iteration loop and the K-means
# warm-start path (see ``_warmstart_from_kmeans``).
# ---------------------------------------------------------------------------

# A pair margin whose dual-view weight Σw falls below this is left unchanged
# by the M-step (same threshold as the copula blocks): with a one-hot ξ (SEM
# draw, K-means warm-start) a pair (i, j) that never occurs has Σw = 0, and
# the closed-form Gaussian update would collapse to N(0, 1e-4).
_PAIR_MARGIN_MIN_WEIGHT: float = 1e-12


def _state_margin_sample(
    Y: np.ndarray,
    gamma: np.ndarray,
    i: int,
    obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sample of the state margin f_i: every ``y_n`` with weight ``γ_n(i)``.

    ``obs`` (N,) bool — observed rows (missing observations, strategy
    ``"available"``): only the observed ``y_n`` are kept. The one sample both
    the parametric margin update of :func:`_m_step` and the empirical copula
    margins of :func:`_empirical_margin_cdfs` are built from.
    """
    w_n = gamma[:, i]                      # P(X_n=i | Y)
    y_fit = Y
    if obs is not None:                    # missing rows: observed y_n only
        y_fit, w_n = Y[obs], w_n[obs]
    return y_fit, w_n


def _pair_margin_sample_observed(
    Y: np.ndarray,
    xi: np.ndarray,
    i: int,
    j: int,
    obs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`_pair_margin_sample`, restricted to the observed endpoints when
    ``obs`` (N,) bool is given (missing observations, strategy
    ``"available"``) — each kept endpoint with its own ξ weight."""
    y_ij, w_ij = _pair_margin_sample(Y, xi, i, j)
    if obs is not None:
        keep = np.concatenate((obs[:-1], obs[1:]))
        y_ij, w_ij = y_ij[keep], w_ij[keep]
    return y_ij, w_ij


def _pair_margin_sample(
    Y: np.ndarray,
    xi: np.ndarray,
    i: int,
    j: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dual-view weighted sample of the pair margin f_ij (general PMC).

    In an SR-PMC the pair ``(y_n, y_{n+1})`` given
    ``(x_n, x_{n+1}) = (i, j)`` has density ``f_ij(y_n) f_ji(y_{n+1})
    c_ij(F_ij(y_n), F_ji(y_{n+1}))`` (DerrodePieczynski_CSDA2013 Eq. 12):
    ``y_n`` is the *left* observation of the pair ``(x_n, x_{n+1})`` and
    ``y_{n+1}`` its *right* observation, and the right margin of the pair
    ``(j, i)`` is ``f_ij``. The margin f_ij is therefore fitted by
    weighted maximum likelihood on

        { y_n     with weight ½ ξ_n(i, j) }  ∪  { y_{n+1} with weight ½ ξ_n(j, i) },
        n = 0 … N−2,

    ``ξ`` being the pair posteriors (ICE) or one-hot draws (SEM, K-means).

    The factor ½ does not change any fit (the weighted MLEs normalise the
    weights); it makes each observation count once in total over the K²
    blocks — ``Σ_ij w = N − 1``, as ``Σ_i γ = N`` for state margins — so the
    effective sample size ``Σw`` that the ``aic`` / ``bic`` GICE rules
    penalise is not doubled by the two views.

    Why this is ICE, not EM
    -----------------------
    It is the conditional (ICE-style) estimator: the complete-data estimator
    of f_ij on the pairs, with the unknown pair memberships replaced by their
    posterior weights. It is **not** an exact EM M-step. With pair margins
    the transition of the complete chain is (DerrodePieczynski_CSDA2013 Eqs. 13–14)

        p(x_{n+1} = j, y_{n+1} | x_n = i, y_n)
            = p_ij f_ij(y_n) / Σ_k p_ik f_ik(y_n) · f_ji(y_{n+1}) c_ij(·, ·),

    so the complete-data log-likelihood contains ``−log Σ_k p_ik f_ik(y_n)``,
    which couples the K margins f_ik of a row and does not separate into one
    weighted log-likelihood per block. The dual view maximises instead the
    pairwise (composite) likelihood ``Σ_n Σ_ij ξ_n(i, j) [log f_ij(y_n) +
    log f_ji(y_{n+1})]``, the copula term being fitted afterwards on the
    pseudo-observations — the two-stage scheme the package applies to state
    margins too. Pooled over j (f_ij = f_i) it gives back the γ-weighted
    state update, up to the weight of the two end points.

    Returns
    -------
    y : ``(2(N−1),)`` or ``(2(N−1), d)`` — ``Y[:-1]`` followed by ``Y[1:]``.
    w : ``(2(N−1),)`` — the matching weights.
    """
    y = np.concatenate((Y[:-1], Y[1:]), axis=0)
    w = 0.5 * np.concatenate((xi[:, i, j], xi[:, j, i]))
    return y, w


def _m_step_pair_margins(
    margins_raw: list[dict],
    Y: np.ndarray,
    xi: np.ndarray,
    margin_selection_rule: str,
    obs: np.ndarray | None = None,
) -> None:
    """Margin part of the M-step for pair margins f_ij — updates blocks in place.

    Every K²-format block ``{"i", "j", ...}`` is re-estimated on its own
    dual-view sample (:func:`_pair_margin_sample`,
    DerrodePieczynski_CSDA2013 Eqs. 12–14), so the K² densities stay
    distinct, and GICE (a ``candidates`` list) selects the family per pair,
    not per state. A block with negligible weight is kept.

    ``obs`` (N,) bool — observed rows (missing observations, strategy
    ``"available"``): the dual-view sample keeps only the observed endpoints,
    each with its ξ weight.
    """
    for blk in margins_raw:
        i_idx, j_idx = int(blk["i"]), int(blk["j"])
        y_ij, w_ij = _pair_margin_sample_observed(Y, xi, i_idx, j_idx, obs)
        total_w = float(w_ij.sum())
        if not total_w >= _PAIR_MARGIN_MIN_WEIGHT:
            logger.debug(
                "  margin (i=%d, j=%d): negligible weight Σw=%.3g, kept.",
                i_idx, j_idx, total_w,
            )
            continue
        blk.update(_select_margin_family(
            blk, y_ij, w_ij, rule=margin_selection_rule,
        ))


def _m_step(
    raw: dict,
    current: PMCModel,
    Y: np.ndarray,
    xi: np.ndarray,
    gamma: np.ndarray,
    *,
    fit_margins: bool,
    candidates: list[str],
    selection_criterion: str,
    margin_selection_rule: str,
    obs: np.ndarray | None = None,
    copula_margins: str = "parametric",
) -> None:
    """ICE M-step — update ``raw`` in place from posterior weights.

    Parameters
    ----------
    raw   : mutable ``dict`` — the model's raw TOML representation; this
            function mutates ``raw["prior"]``, ``raw["margins"]``, and
            ``raw["copulas"]`` (depending on the variant and ``fit_margins``).
    current : :class:`PMCModel` — the model snapshot whose margins are used
            as marginal CDFs for the copula pseudo-observations. Must be
            consistent with the *current* iteration's parameters (i.e. the
            model that produced ``xi`` / ``gamma`` via forward-backward).
    Y     : ``np.ndarray`` shape ``(N,)`` or ``(N, d)`` — observation sequence.
    xi    : ``np.ndarray`` shape ``(N-1, K, K)`` — joint posteriors
            ``ξ_n(i, j) = P(X_n=i, X_{n+1}=j | Y)``.
    gamma : ``np.ndarray`` shape ``(N, K)`` — marginal posteriors
            ``γ_n(i) = P(X_n=i | Y)``.
    fit_margins, candidates, selection_criterion, margin_selection_rule :
            forwarded from the ICE config; see :func:`_parse_ice_cfg`.
    obs   : ``np.ndarray`` shape ``(N,)`` bool, optional — observed rows of a
            ``Y`` with missing observations (NaN rows), ``None`` for complete
            data (the historical code path, unchanged). With ``obs`` —
            strategy ``"available"`` of ICE, and the k-means warm start —
            ``xi`` and ``gamma`` are the exact posteriors given the observed
            data at every n, and: the prior uses ξ over all n; state margins
            are fitted on the observed ``y_n`` with weight ``γ_n(i)``; pair
            margins on the observed endpoints of their dual view; copulas on
            the pairs whose two endpoints are observed, weight ``ξ_n(i, j)``.
    copula_margins : ``"parametric"`` (default — the historical code path,
            unchanged) or ``"empirical"``: the margins the copula
            pseudo-observations of step 3 are computed with
            (:func:`_empirical_margin_cdfs`; AUDIT_COPULES FR-7 a). Steps 1
            and 2 do not depend on it.

    Behaviour
    ---------
    1. **Prior**: ``p̂[i,j]`` from ξ̄ (symmetrised under SR-PMC, then either
       stored as joint ``p`` or converted to row-stochastic ``A`` for HMC
       variants).
    2. **Margins** (only when ``fit_margins`` is truthy): weighted MLE /
       GICE family selection via :func:`_select_margin_family`, per block:

       * state margins f_i — weights ``γ_n(i)`` on ``y_n``;
       * pair margins f_ij (general PMC, DerrodePieczynski_CSDA2013
         Eqs. 12–14) — the dual-view sample of
         :func:`_pair_margin_sample`: ``y_n`` with weight
         ``ξ_n(i, j)`` and ``y_{n+1}`` with weight ``ξ_n(j, i)``.
    3. **Copulas** (only for variants that use them): per-pair (i, j)
       weighted family selection + τ fit via :func:`_select_and_fit_copula`.
       Pseudo-observations ``(F_ij(y_n), F_ji(y_{n+1}))`` with weight
       ``ξ_n(i, j)`` are computed from ``current``'s margins
       (``(F_i(y_n), F_j(y_{n+1}))`` for state margins) — or, with
       ``copula_margins="empirical"``, from the weighted empirical
       counterparts of those margins built on the samples of step 2 with the
       same posterior weights (:func:`_empirical_margin_cdfs`), whether or
       not ``fit_margins`` is on. The family scores and the τ fit both see
       these pseudo-observations.

    Both margin updates are ICE-style conditional estimators, not exact EM
    M-steps once the model is a general PMC: see :func:`_pair_margin_sample`.
    """
    _check_copula_margins(copula_margins)
    var = current.variant

    # 1. Update prior distribution.
    _m_step_prior(raw, current, xi)

    # 2. Update margins (optional). The family declared in each block is
    #    preserved (unless GICE selects another), only the params are
    #    re-estimated by weighted MLE. Two margin structures
    #    (``PMCModel.margin_structure``):
    #
    #    * "state" — K densities f_i. The weight on Y[k] for state i is the
    #      *full* posterior of being in state i at time k:
    #
    #        w_i[k] = γ_k(i) = P(X_k = i | Y)
    #               = Σ_j ξ[k, i, j]      (k = 0…N-2; "first-view" sum)
    #               = Σ_j ξ[k-1, j, i]    (k = 1…N-1; "second-view" sum)
    #
    #      i.e. the pair dual view below pooled over j (up to the
    #      two end points): K MLE per M-step. By the Proposition of
    #      DerrodePieczynski_CSDA2013 §2.1 this is the structure
    #      where X is Markov.
    #
    #    * "pair" — K² densities f_ij (general PMC,
    #      DerrodePieczynski_CSDA2013 Eqs. 12–14): the dual-view
    #      weighted MLE of :func:`_pair_margin_sample`, which does
    #      not tie f_ij to f_ik. (Weighting every block by γ_n(i),
    #      as the state branch does, would give f_ij = f_ik after
    #      one M-step.)
    if fit_margins:
        margins_raw = raw.get("margins", [])
        if current.margin_structure == "pair":
            _m_step_pair_margins(margins_raw, Y, xi, margin_selection_rule, obs=obs)
        else:
            for blk in margins_raw:
                i_idx = int(blk["i"])
                # P(X_n=i | Y) on y_n — observed rows only when obs is given.
                y_fit, w_n = _state_margin_sample(Y, gamma, i_idx, obs)
                # GICE: if the block declares a ``candidates`` list, also
                # select the family at this M-step (SP-2016 §3); otherwise
                # the helper falls through to the v0.5 single-family fit.
                blk.update(_select_margin_family(
                    blk, y_fit, w_n, rule=margin_selection_rule,
                ))

    # 3. Update copulas (only for variants that use copulas)
    if var.uses_copula:
        copulas_raw = raw.get("copulas", [])

        # Marginal CDFs of the *current* model (the one that produced ξ).
        # Pseudo-observations of the copula c_ij (DerrodePieczynski_CSDA2013 Eq. 12):
        #     u_n = F_ij(Y_n),  v_n = F_ji(Y_{n+1}),  weight ξ_n(i, j).
        both = None
        if obs is not None:
            # Missing rows: CDFs on a finite placeholder, then only the pairs
            # whose two endpoints are observed are kept.
            both = np.nonzero(obs[:-1] & obs[1:])[0]
            Y = np.array(Y, dtype=float, copy=True)
            Y[~obs] = 0.0
        if copula_margins == "parametric":
            f_cdf = _margin_cdfs(current, Y)
        else:
            # FR-7 a: weighted empirical margins, same samples and weights as
            # the margin update of step 2 (observed rows only when obs is set).
            f_cdf = _empirical_margin_cdfs(current, Y, xi, gamma, obs=obs)

        for blk in copulas_raw:
            ii = int(blk["i"])
            jj = int(blk["j"])

            # Weighted pseudo-observations
            weights_pair = xi[:, ii, jj]          # shape (N-1,)
            if both is not None:
                weights_pair = weights_pair[both]
            total_w = weights_pair.sum()
            if total_w < 1e-12:
                logger.debug("  (i=%d,j=%d): negligible weight, skip.", ii, jj)
                continue

            u_arr, v_arr = _pair_pseudo_obs(current, f_cdf, ii, jj)
            if both is not None:
                u_arr, v_arr = u_arr[both], v_arr[both]

            # The package's one weight check (FR-12), shared with the public
            # weighted fit; the same array comes back, so nothing changes.
            weights_pair = validate_weights(weights_pair, u_arr.shape[0])
            logger.debug("  fitting copula (i=%d, j=%d)  Σw=%.4f", ii, jj, total_w)
            best_blk = _select_and_fit_copula(
                candidates, u_arr, v_arr, weights_pair,
                criterion=selection_criterion,
            )
            # A failed fit is reported, not stored: the flag is not a copula
            # parameter and would outlive the next, successful, M-step.
            if best_blk.pop(FIT_FAILED_KEY, False):
                logger.warning(
                    "ICE M-step, pair (i=%d, j=%d): the selected %s fit did "
                    "not converge (Σw=%.4g) — its parameters are the best "
                    "point found: %s.",
                    ii, jj, best_blk.get("name"), float(total_w),
                    {k: val for k, val in best_blk.items() if k != "name"},
                )
            blk.update(best_blk)


def _m_step_prior(raw: dict, current: PMCModel, xi: np.ndarray) -> None:
    """Prior part of the M-step: ``p̂[i, j]`` from ξ̄ — updates ``raw["prior"]``.

    The raw average of ξ_n is asymmetric by O(1/√N) finite-sample noise;
    under SR-PMC (the framework this package commits to) the true joint
    ``p[i,j]`` is *exactly* symmetric. We therefore symmetrise ξ̄
    before further processing — this both eliminates noise in p_hat
    and prevents the SR-symmetry warning in PMCModel from firing
    at every ICE iteration on PMC variants. Stored as the joint ``p``, or
    converted to the row-stochastic ``A`` for the HMC variants.
    """
    var = current.variant
    K   = current.K
    N   = xi.shape[0] + 1
    p_hat = xi.sum(axis=0) / max(N - 1, 1)
    p_hat = 0.5 * (p_hat + p_hat.T)            # SR-PMC: enforce p[i,j] = p[j,i]
    p_hat = np.clip(p_hat, 0.0, None)
    total = p_hat.sum()
    if total > MIN_POSITIVE:
        p_hat /= total
    else:
        p_hat = np.full((K, K), 1.0 / (K * K))

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


def _margin_cdfs(current: PMCModel, Y: np.ndarray) -> np.ndarray:
    """Marginal CDFs of ``current`` at ``Y``, clipped to [EPS, 1 − EPS].

    Shape ``(N, K, K)`` with ``F[n, i, j] = F_ij(Y[n])`` for pair margins,
    ``(N, K)`` with ``F[n, k] = F_k(Y[n])`` for state margins (K CDF vectors,
    not K² — F_ij = F_i whatever j).
    """
    N, K = len(Y), current.K
    if current.margin_structure == "pair":
        f_cdf_pair = np.empty((N, K, K))
        for ii in range(K):
            for jj in range(K):
                f_cdf_pair[:, ii, jj] = current.margin(ii, jj).cdf_vec(Y)
        np.clip(f_cdf_pair, EPS, ONE_MINUS_EPS, out=f_cdf_pair)
        return f_cdf_pair
    f_cdf_state = np.zeros((N, K))
    for kk in range(K):
        f_cdf_state[:, kk] = current.margin(kk).cdf_vec(Y)
    np.clip(f_cdf_state, EPS, ONE_MINUS_EPS, out=f_cdf_state)
    return f_cdf_state


def _pair_pseudo_obs(current: PMCModel, f_cdf: np.ndarray, ii: int, jj: int):
    """Pseudo-observations ``(u_n, v_n)``, n = 0 … N−2, of the copula c_ij.

    ``u = F_ij(Y_n), v = F_ji(Y_{n+1})`` for pair margins, ``u = F_i(Y_n),
    v = F_j(Y_{n+1})`` for state margins; ``f_cdf`` from :func:`_margin_cdfs`.
    """
    N = f_cdf.shape[0]
    if current.margin_structure == "pair":
        return f_cdf[: N - 1, ii, jj], f_cdf[1:, jj, ii]
    return f_cdf[: N - 1, ii], f_cdf[1:, jj]


# ---------------------------------------------------------------------------
# Empirical copula margins — config key ``copula_margins`` (AUDIT_COPULES FR-7 a)
# ---------------------------------------------------------------------------

#: Values of the ICE / SEM ``copula_margins`` config key: the margins the
#: copula step's pseudo-observations are computed with. ``"parametric"`` is
#: the historical behaviour (the model's own F_i / F_ij); ``"empirical"`` the
#: posterior-weighted empirical margins of :func:`_empirical_margin_cdfs`.
COPULA_MARGIN_MODES: tuple[str, ...] = ("parametric", "empirical")

#: Default ``copula_margins``: the parametric margins, efficient when they are
#: validated (Genest & Werker 2002) and the only mode the standard errors of
#: :mod:`pmcprg.pmc._oakes`, :mod:`pmcprg.pmc._godambe` and
#: :mod:`pmcprg.pmc._lystig_hughes` are derived for.
DEFAULT_COPULA_MARGINS: str = "parametric"

_COPULA_MARGINS_KEY: str = "copula_margins"


def _check_copula_margins(value) -> str:
    """Validate the ``copula_margins`` config key (one of :data:`COPULA_MARGIN_MODES`)."""
    if not isinstance(value, str) or value not in COPULA_MARGIN_MODES:
        raise ValueError(
            f"Unknown copula_margins {value!r}. Valid: {list(COPULA_MARGIN_MODES)}"
        )
    return value


def _empirical_margin_cdfs(
    current: PMCModel,
    Y: np.ndarray,
    xi: np.ndarray,
    gamma: np.ndarray,
    obs: np.ndarray | None = None,
) -> np.ndarray:
    """Posterior-weighted empirical margins at ``Y`` — :func:`_margin_cdfs`'s
    nonparametric counterpart for the copula step (AUDIT_COPULES FR-7 a).

    Same shape and layout as :func:`_margin_cdfs` (``(N, K)`` for state
    margins, ``(N, K, K)`` with ``F[n, i, j] = F̂_ij(Y[n])`` for pair margins),
    so :func:`_pair_pseudo_obs` builds the copula pseudo-observations from it
    unchanged; clipped to [EPS, 1 − EPS] like the parametric ones.

    What is estimated, and with which weights (derivation)
    -------------------------------------------------------
    Each F̂ is the nonparametric estimate of *the margin the parametric M-step
    fits*, on *the sample and weights it fits it on*. That M-step maximises a
    weighted log-likelihood ``Σ_k w_k log f(y_k)`` over a parametric family;
    over all distributions the maximiser of the same objective (the weighted
    nonparametric MLE) puts mass ``w_k / Σw`` on each ``y_k`` — the weighted
    empirical distribution. So:

    * **state margins** f_i (HMC-DN; PMC with ``margin_structure="state"``):
      the parametric update fits f_i on ``{y_n, weight γ_n(i)}``
      (:func:`_state_margin_sample`), hence ::

          F̂_i(y) = Σ_n γ_n(i) 1{y_n ≤ y} / (Σ_n γ_n(i) + 1).

    * **pair margins** f_ij (general PMC, DerrodePieczynski_CSDA2013 Eq. 12):
      the pair density
      ``f_ij(y_n) f_ji(y_{n+1}) c_ij(F_ij(y_n), F_ji(y_{n+1}))`` given
      ``(x_n, x_{n+1}) = (i, j)`` says that f_ij is the law of the *left*
      observation ``y_n`` of a pair in state (i, j) **and** of the *right*
      observation ``y_{n+1}`` of a pair in state (j, i) — the two places f_ij
      enters the complete-data likelihood. The parametric update therefore
      fits f_ij on the dual-view sample of :func:`_pair_margin_sample`, and
      so does this function ::

          F̂_ij(y) = [Σ_n ½ξ_n(i, j) 1{y_n ≤ y} + Σ_n ½ξ_n(j, i) 1{y_{n+1} ≤ y}]
                    / (S_ij + 1),     S_ij = ½ Σ_n [ξ_n(i, j) + ξ_n(j, i)].

      The copula c_ij then gets ``u_n = F̂_ij(y_n)`` and ``v_n = F̂_ji(y_{n+1})``,
      where F̂_ji's sample contains ``y_{n+1}`` with weight ½ξ_n(i, j): the
      right observation of the pair (i, j) *is* a draw of f_ji. The weights
      are ξ, not γ — weighting f_ij by γ_n(i) would estimate the pooled f_i
      for every j (the reason the parametric pair update does not use γ
      either). Pooled over j, the dual-view weight of an interior y_n is
      ½[Σ_j ξ_n(i, j) + Σ_j ξ_{n−1}(j, i)] = γ_n(i): the state formula again,
      up to the two end points.

    With missing observations (``obs``, strategy ``"available"``) the samples
    keep the observed values only, exactly as the parametric update does.

    The ``+ 1``
    -----------
    The weighted empirical CDF equals 1 at the sample maximum, where most
    copula log-densities are infinite. As in the rank pseudo-likelihood,
    whose margins are ``rank/(n + 1)`` (Genest, Ghoudi & Rivest 1995), the
    denominator is ``Σw + 1``, so ``F̂ ≤ Σw/(Σw + 1) < 1``; and every
    pseudo-observation with a positive copula weight ξ_n(i, j) is itself in
    its margin's sample with weight ≥ ½ξ_n(i, j) (pair margins; γ_n(i) ≥
    ξ_n(i, j) for state margins), so ``F̂ > 0`` there. The ``1`` is on the
    scale of the parametric update's own sample: ``Σ_i Σ_n γ_n(i) = N`` and,
    thanks to its factor ½, ``Σ_ij S_ij = N − 1`` — each observation counts
    once over all the blocks. For one-hot weights (an SEM draw, the k-means
    labels) F̂_i is exactly the classical rescaled empirical CDF of the
    sub-sample, ``#{n : x_n = i, y_n ≤ y} / (n_i + 1)``; a pair margin's
    one-hot dual-view weights are ½ or 1 per point, and its ``+ 1`` stays on
    that same half-weight scale. Ties: ``≤`` (see :func:`_weighted_ecdf`).

    Why the E-step keeps the parametric margins
    --------------------------------------------
    Forward–backward needs transition *densities*
    ``f_j(y_{n+1}) c_ij(F_i(y_n), F_j(y_{n+1}))``, which integrate to one in
    ``y_{n+1}`` only if ``F_j`` is the CDF of ``f_j``: a step function has no
    density, and ``c(F̂, F̂)`` next to a parametric ``f`` is not a proper
    kernel. So only the copula step changes: it becomes the second step of
    Chen & Fan's (2006) two-step semiparametric estimator for copula-based
    Markov models — nonparametric margins first, then the copula parameter
    by pseudo-likelihood on rescaled empirical CDFs — weighted by posteriors
    from the working parametric model. When the parametric margins are right,
    F̂ and F estimate the same distribution and the two modes the same τ, the
    parametric one more efficiently (Genest & Werker 2002). When a margin is
    contaminated or misspecified, **only the copula step is protected**: γ,
    ξ, and the margins fitted with ``fit_margins``, still come from the
    parametric model.

    References
    ----------
    * Chen, X. & Fan, Y. (2006). Estimation of copula-based semiparametric
      time series models. *J. Econometrics* 130(2), 307–335.
      doi:10.1016/j.jeconom.2005.03.004
    * Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). A semiparametric
      estimation procedure of dependence parameters in multivariate families
      of distributions. *Biometrika* 82(3), 543–552. doi:10.1093/biomet/82.3.543
    * Genest, C. & Werker, B. J. M. (2002). Conditions for the asymptotic
      semiparametric efficiency of an omnibus estimator of dependence
      parameters in copula models. In *Distributions with Given Marginals
      and Statistical Modelling*. doi:10.1007/978-94-017-0061-0_12
    * Kim, G., Silvapulle, M. J. & Silvapulle, P. (2007). Comparison of
      semiparametric and parametric methods for estimating copulas. *CSDA*.
      doi:10.1016/j.csda.2006.10.009
    """
    y_eval = np.asarray(Y, dtype=float).reshape(len(Y))   # copula variants: d = 1
    N, K = len(y_eval), current.K
    if current.margin_structure == "pair":
        f_pair = np.empty((N, K, K))
        for ii in range(K):
            for jj in range(K):
                y_s, w_s = _pair_margin_sample_observed(y_eval, xi, ii, jj, obs)
                f_pair[:, ii, jj] = _weighted_ecdf(y_s, w_s, y_eval)
        np.clip(f_pair, EPS, ONE_MINUS_EPS, out=f_pair)
        return f_pair
    f_state = np.empty((N, K))
    for kk in range(K):
        y_s, w_s = _state_margin_sample(y_eval, gamma, kk, obs)
        f_state[:, kk] = _weighted_ecdf(y_s, w_s, y_eval)
    np.clip(f_state, EPS, ONE_MINUS_EPS, out=f_state)
    return f_state


def _record_copula_margins(raw: dict, value: str, *, section: str) -> None:
    """Make the model's own config tables resolve to the ``copula_margins`` used.

    ``section`` is ``"ice"`` (ICE reads ``[ice]``) or ``"sem"`` (SEM reads
    ``[ice]`` then ``[sem]``). The key is written into ``raw[section]`` only
    when the tables would otherwise resolve to another value — never on the
    default path with the key absent, so that path's models are unchanged.
    This is what lets the standard-error modules refuse a fit made with
    empirical copula margins (:func:`_refuse_nonparametric_copula_margins`).
    """
    tables = [raw.get("ice", {})] + ([raw.get("sem", {})] if section == "sem" else [])
    resolved = DEFAULT_COPULA_MARGINS
    for tbl in tables:
        resolved = tbl.get(_COPULA_MARGINS_KEY, resolved)
    if resolved != value:
        raw.setdefault(section, {})[_COPULA_MARGINS_KEY] = value


def _recorded_copula_margins(model: PMCModel) -> str:
    """``copula_margins`` a model's own tables declare: ``[ice]``, else ``[sem]``
    (over ``[ice]``) if that one is not the default, else the default."""
    ice_val = model.ice_config().get(_COPULA_MARGINS_KEY, DEFAULT_COPULA_MARGINS)
    if ice_val != DEFAULT_COPULA_MARGINS:
        return ice_val
    sem_tbl = {**model.ice_config(), **model.sem_config()}
    return sem_tbl.get(_COPULA_MARGINS_KEY, DEFAULT_COPULA_MARGINS)


def _refuse_nonparametric_copula_margins(model: PMCModel, who: str) -> None:
    """Raise if ``model`` was fitted with ``copula_margins`` other than parametric.

    The standard errors of :mod:`pmcprg.pmc._oakes`, :mod:`pmcprg.pmc._godambe`
    and :mod:`pmcprg.pmc._lystig_hughes` are derived for the parametric-margin
    ICE fixed point (a stationary point of the expected complete-data
    likelihood with the model's own F). With empirical copula margins the
    copula parameter solves a rank-based estimating equation instead, whose
    asymptotic variance carries an extra term from the estimated margins
    (Chen & Fan 2006) that none of them computes — their number would be
    the SE of another estimator. ICE and SEM record the option in the fitted
    model's ``[ice]`` / ``[sem]`` table (:func:`_record_copula_margins`),
    which is what is read here.
    """
    mode = _recorded_copula_margins(model)
    if mode != DEFAULT_COPULA_MARGINS:
        raise NotImplementedError(
            f"{who}: the model declares copula_margins = {mode!r} in its "
            "[ice]/[sem] table — its copulas were fitted on empirical "
            "margins (AUDIT_COPULES FR-7 a). These standard errors assume "
            "the parametric margins; the Chen–Fan-type correction that "
            "empirical margins need is not implemented."
        )


# ---------------------------------------------------------------------------
# Missing observations (NaN rows of Y) — E-step, completions, M-steps
# ---------------------------------------------------------------------------
#
# A Y without missing rows never reaches this section: ICE, SEM and the
# k-means warm start test ``missing_mask(Y).any()`` first and run their
# historical code otherwise (bit-identical results,
# ``pmcprg/tests/test_estim_complete_data_identity.py``).

#: Values of the ICE ``missing_strategy`` config key (docstring of :func:`ice`,
#: section "Missing observations").
MISSING_STRATEGIES: tuple[str, ...] = ("available", "impute")

#: Default ``missing_strategy``. Measured on simulated data (N = 3000, MCAR
#: blocks of 10 at 10/20/40 %, ``fit_margins = True``, 30 replications, 12 for
#: ``sp2016_gice_k2``), against SEM, from a start near the truth:
#:
#: * with the package defaults (``tol = 1e-4``, ICE stopping after 8–10
#:   iterations), ``"available"`` has a lower RMSE than ``"impute"`` for the
#:   prior, the margin means and standard deviations and τ on every fixture
#:   and rate (exceptions: two 10 % cells within 1 %, and the heavy-tailed
#:   margin standard deviation of the GICE fixture at 40 %), and stays near
#:   its complete-data level — ``pmc_gauss_k2`` at 40 %: prior 0.027 ("impute" 0.044, SEM
#:   0.038; complete data 0.030), margin means 0.099 (0.140, 0.131; 0.089), τ
#:   0.040 (0.045, 0.054; 0.034). "impute" moves more slowly away from its
#:   start (its completions come from the current model) and its Monte-Carlo
#:   noise decreases the log-likelihood (17–43 % of the runs on the state
#:   fixtures, against none). On the GICE fixture at 40 % it is clearly worse
#:   (prior RMSE 0.042 vs 0.022, τ 0.069 vs 0.040), with family switches that
#:   drop the log-likelihood by hundreds of nats;
#: * iterated to convergence from the truth (20 replications of the two PMC
#:   fixtures, ``tol = 1e-9``; "impute": 60 iterations), "impute" lowers the RMSE of the prior and margin means by up
#:   to 25 % (``pmc_gauss_k2`` at 40 %: prior 0.029 vs 0.034, margin means
#:   0.109 vs 0.129; ``pmc_pair_gauss_k2``: 0.061 vs 0.081, 0.364 vs 0.391;
#:   τ and margin standard deviations within ±6 %) — the information of the
#:   pairs across a gap edge — at 1.3–3.5× the time;
#: * "available" is deterministic and 3–4× (grid variants) to 8–19× (GICE)
#:   faster than "impute" with 5 draws.
#:
#: The exception is family selection. On ``sp2016_gice_k2`` the GICE margin
#: families are recovered in 39 % of the fits with "available" against 57 %
#: with "impute" and 61 % on complete data (96, 96 and 36 margin blocks
#: pooled over the two studies), and on CSDA-2013 Exp. 3 setting 1 (sweep
#: multistart, 6 seeds) both recover the diagonal copula families in every
#: run, "impute" the off-diagonal ones a little more often (0.83–0.92 vs
#: 0.79–0.83). Prefer ``missing_strategy = "impute"`` when margin families
#: are selected and its cost is acceptable.
DEFAULT_MISSING_STRATEGY: str = "available"
DEFAULT_MISSING_DRAWS: int = 5

#: How ``"impute"`` picks a family from the D completed series: ``"pooled"``
#: — the candidate with the highest criterion summed over the D series (the
#: same criterion ``selection_criterion`` / ``margin_selection_rule`` as on
#: complete data: a Monte-Carlo estimate of its conditional expectation);
#: ``"majority"`` — the family selected on most series, ties broken by the
#: pooled criterion. Private switch; see :func:`_combine_choice`.
#:
#: Measured (6 seeds per cell, 20 and 40 % missing): on CSDA-2013 Exp. 3
#: setting 1 (sweep multistart) both recover the diagonal families in every
#: run, majority the off-diagonal GH pairs slightly more often (0.92–0.96 vs
#: 0.83–0.92); on ``sp2016_gice_k2`` pooled recovers the margin families
#: more often (0.75 vs 0.67–0.70), copulas mixed (0.83 vs 0.75–0.85), and one
#: majority run failed (a fitted support excluded an observation). The two
#: rules chose different families in 7 of 24 paired runs. No rule is better
#: overall; "pooled" is kept as the direct extension of the complete-data
#: criterion.
_IMPUTE_FAMILY_RULE: str = "pooled"

#: How a missing y drawn on the quadrature grid of the grid variants is
#: decoded: ``"nodes"`` — the node y_g itself (the discrete law whose moments
#: are the quadrature moments of the posterior); ``"jitter"`` — uniform within
#: the node's Gauss–Legendre cell in the variable s of the grid, mapped
#: through y = G_ref^{-1}(ψ(s)). Private switch kept for the measurements.
#:
#: Measured at the true model (N = 3000, MCAR blocks of 10 at 40 %, 60
#: replications × 8 draws; paired difference between the hard-label M-step
#: on the completed series and on the true complete data, whose expectation
#: is 0 for an exact sampler): with ``"nodes"`` no τ differs from 0 by more
#: than 1.4 standard errors (``pmc_gauss_k2``, ``pmc_pair_gauss_k2``) and
#: 256 nodes change every τ by < 4e-4; ``"jitter"`` biases τ downwards (−0.0020
#: ± 0.0006 and −0.0032 ± 0.0008 on the Clayton pairs of ``pmc_pair_gauss_k2``,
#: −0.0018 ± 0.0008 on ``pmc_gauss_k2``) — it adds within-cell noise to the
#: dependence. Hence ``"nodes"`` with the default 64 nodes.
_GAP_DRAW: str = "nodes"


def _check_missing_cfg(cfg: dict) -> None:
    """Validate the missing-data keys of a merged ICE/SEM config."""
    strategy = cfg.get("missing_strategy", DEFAULT_MISSING_STRATEGY)
    if strategy not in MISSING_STRATEGIES:
        raise ValueError(
            f"Unknown missing_strategy {strategy!r}. Valid: {list(MISSING_STRATEGIES)}"
        )
    draws = cfg.get("missing_draws", DEFAULT_MISSING_DRAWS)
    if isinstance(draws, bool) or int(draws) != draws or int(draws) < 1:
        raise ValueError(f"missing_draws must be an integer ≥ 1, got {draws!r}.")
    nodes = cfg.get("gap_nodes")
    if nodes is not None and (isinstance(nodes, bool) or int(nodes) != nodes or int(nodes) < 2):
        raise ValueError(f"gap_nodes must be an integer ≥ 2, got {nodes!r}.")


def _missing_rows_or_none(Y: np.ndarray) -> np.ndarray | None:
    """Mask (N,) of the missing rows of Y, or ``None`` when Y is complete."""
    from pmcprg.pmc.gaps import missing_mask
    miss = missing_mask(Y)
    return miss if miss.any() else None


# ---------------------------------------------------------------------------
# Missingness mechanism — config key ``missingness`` (P6)
# ---------------------------------------------------------------------------

#: Values of the ICE / SEM ``missingness`` config key: what the estimators do
#: with the missingness mechanism of the model (:mod:`pmcprg.pmc.missingness`).
#:
#: * ``"model"``        — carry ``model.missingness`` unchanged (ignorable when
#:   the model has none): its parameters are held fixed. The historical
#:   behaviour.
#: * ``"ignorable"``    — drop any mechanism: every iterate, and the returned
#:   model, has ``missingness is None`` (observed-data likelihood p(y_obs)).
#: * ``"state"`` / ``"state-markov"`` — estimate that mechanism with the other
#:   parameters (:func:`pmcprg.pmc.missingness.estimate_mechanism`).
MISSINGNESS_MODES: tuple[str, ...] = ("model", "ignorable", "state", "state-markov")

#: Default ``missingness``: the model's mechanism, held fixed.
DEFAULT_MISSINGNESS: str = "model"

#: The values of ``missingness`` whose mechanism the estimators estimate.
_ESTIMATED_MECHANISMS: tuple[str, ...] = ("state", "state-markov")


def _check_missingness_mode(value) -> str:
    """Validate the ``missingness`` config key (one of :data:`MISSINGNESS_MODES`)."""
    if not isinstance(value, str) or value not in MISSINGNESS_MODES:
        raise ValueError(
            f"Unknown missingness {value!r}. Valid: {list(MISSINGNESS_MODES)}"
        )
    return value


def _effective_missingness(mode: str, miss: np.ndarray | None, label: str) -> str:
    """``mode``, or ``"ignorable"`` (WARNING) when it asks to estimate a mechanism
    from a Y without missing rows.

    The mask of a complete Y is m = 0 everywhere: the maximum-likelihood rates
    are π̂_i = 0 (``"state"``), and â_i = 0 with b̂_i unidentified
    (``"state-markov"``) — mechanisms whose evidence factors on m = 0 are all
    1, i.e. the ignorable model. The estimators fit that model and say so.
    """
    mode = _check_missingness_mode(mode)
    if mode in _ESTIMATED_MECHANISMS and miss is None:
        logger.warning(
            "%s: missingness=%r but Y has no missing rows — the mask m = 0 "
            "identifies no missing rate (maximum likelihood: every rate 0, "
            "i.e. ignorable). Fitting with missingness='ignorable'; the "
            "returned model has no [missingness] table.", label, mode,
        )
        return "ignorable"
    return mode


def _missingness_start(model: PMCModel, mode: str, miss: np.ndarray):
    """Starting mechanism of an estimated ``missingness`` mode.

    ``model.missingness`` when it is of that kind; a ``"state"`` start π for
    ``"state-markov"`` is its nested equivalent a = b = π. Otherwise the
    state-independent maximum-likelihood mechanism of the mask
    (:func:`pmcprg.pmc.missingness.common_mechanism`): its evidence factors do
    not depend on the state, so the first E-step gives the posteriors of the
    ignorable model, and the first M-step estimates the mechanism from them.
    """
    from pmcprg.pmc.missingness import StateMarkovMissingness, common_mechanism
    mech = model.missingness
    if mech is not None and mech.mechanism == mode:
        return mech
    if mode == "state-markov" and mech is not None and mech.mechanism == "state":
        return StateMarkovMissingness(onset=mech.rates, persistence=mech.rates)
    return common_mechanism(mode, miss, model.K)


def _set_start_missingness(raw: dict, model: PMCModel, mode: str,
                           miss: np.ndarray | None) -> None:
    """Put the starting mechanism of ``mode`` in ``raw`` (no-op for ``"model"``)."""
    if mode == "ignorable":
        raw.pop("missingness", None)
    elif mode in _ESTIMATED_MECHANISMS:
        raw["missingness"] = _missingness_start(model, mode, miss).to_table()


def _m_step_missingness(raw: dict, mode: str, gamma: np.ndarray, miss: np.ndarray) -> None:
    """Mechanism part of the M-step (estimated ``missingness`` modes): updates
    ``raw["missingness"]`` from the state posteriors ``gamma`` given (y_obs, m)
    — :func:`pmcprg.pmc.missingness.estimate_mechanism`."""
    from pmcprg.pmc.missingness import estimate_mechanism
    raw["missingness"] = estimate_mechanism(mode, gamma, miss).to_table()


def _snapshot_missingness(model: PMCModel) -> dict | None:
    """The ``[missingness]`` table of ``model`` (``None``: ignorable)."""
    mech = model.missingness
    return None if mech is None else mech.to_table()


@dataclass
class _GapEStep:
    """E-step of a Y with missing rows.

    log_lik : log p(y_obs | θ) — the observed-data log-likelihood.
    gamma, xi : exact posteriors P(x_n | y_obs), P(x_n, x_{n+1} | y_obs) at
        every n (``posterior=True`` only).
    X_draws : (S, N) int — joint posterior draws of the state path.
    Y_draws : (S, N) or (S, N, d) — Y with its missing rows drawn jointly
        with ``X_draws`` (completed series).
    """
    log_lik: float
    gamma:   np.ndarray | None = None
    xi:      np.ndarray | None = None
    X_draws: np.ndarray | None = None
    Y_draws: np.ndarray | None = None


def _grid_cell_draw(grid, g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Missing values drawn on quadrature nodes ``g`` — decoded per :data:`_GAP_DRAW`."""
    if _GAP_DRAW == "nodes":
        return grid.nodes[g]
    edges = np.concatenate(([0.0], np.cumsum(grid.ws)))
    edges[-1] = 1.0
    s = edges[g] + rng.random(g.shape) * (edges[g + 1] - edges[g])
    return grid.y_of_s(s)


def _gap_e_step(
    model: PMCModel,
    Y: np.ndarray,
    miss: np.ndarray,
    *,
    gap_nodes: int | None,
    posterior: bool,
    n_draws: int,
    rng: np.random.Generator | None,
) -> _GapEStep:
    """Observed-data log-likelihood, posteriors and joint completions of a NaN-bearing Y.

    Uses the missing-data inference of :mod:`pmcprg.pmc.gaps`: the exact
    K-state shortcut for HMC-IN, HMC-IN2 and PMC-IN with state margins, the
    augmented quadrature chain otherwise. ``posterior`` runs forward and
    backward (γ, ξ); otherwise the forward pass only. ``n_draws`` joint
    forward-filter backward-sample draws of (x_{1:N}, y_mis) are taken from
    the same forward messages: for the shortcut variants the states, then
    each missing y_n ~ f_{x_n}; for the grid variants the augmented path, the
    missing y then sitting on the quadrature nodes (decoded by
    :func:`_grid_cell_draw`).
    """
    from pmcprg.pmc import gaps

    trans = alphas = grid = None
    log_space = False
    out = _GapEStep(log_lik=float("nan"))
    if posterior:
        post, run = gaps._posterior(model, Y, gap_nodes, True)
        out.log_lik, out.gamma, out.xi = float(post.log_lik), post.gamma, post.xi
        if post.method == "grid":
            chain, alphas, _ = run
            trans, log_space, grid = chain.trans, chain.log, post.grid
        else:
            trans, alphas = run[0], list(post.alpha_hat)
    elif gaps.needs_grid(model):
        grid = gaps.reference_grid(model, gap_nodes)
        chain, alphas, ll, _ = gaps._run_chain(model, np.asarray(Y, dtype=float), miss, grid,
                                               backward=False)
        trans, log_space, out.log_lik = chain.trans, chain.log, float(ll)
    else:
        W, f_pdf = precompute_weights(model, Y)
        alpha_hat, ll = forward(model, Y, W=W, f_pdf=f_pdf)
        trans, alphas, out.log_lik = W, list(alpha_hat), float(ll)

    if n_draws:
        U = gaps._ffbs(trans, alphas, log_space, rng, int(n_draws))
        idx = np.nonzero(miss)[0]
        Yd = np.repeat(np.asarray(Y, dtype=float)[None], int(n_draws), axis=0)
        if grid is not None:
            X = np.where(miss[None, :], U // grid.G, U).astype(int)
            Yd[:, idx] = _grid_cell_draw(grid, U[:, idx] % grid.G, rng)
        else:
            X = U.astype(int)
            Yd[:, idx] = gaps._draw_state_margins(model, X[:, idx], rng)
        out.X_draws, out.Y_draws = X, Yd
    return out


def _average_params(params_list: list[dict]) -> dict:
    """Arithmetic mean, key by key, of parameter dicts (scalars or nested lists)."""
    out = {}
    for k in params_list[0]:
        if k == FIT_FAILED_KEY:
            continue
        vals = np.asarray([p[k] for p in params_list], dtype=float)
        mean = vals.mean(axis=0)
        out[k] = mean.tolist() if mean.ndim else float(mean)
    return out


def _combine_choice(scores: np.ndarray) -> int:
    """Index of the candidate chosen from a (D, C) score table (−inf: unusable).

    ``"pooled"`` (:data:`_IMPUTE_FAMILY_RULE`): arg max of the column sums
    over the candidates usable in every series. ``"majority"``: the candidate
    selected on most series, ties broken by the pooled sum. Returns −1 when
    no candidate is usable in every series. First index wins ties, as the
    strict ``>`` of the complete-data selection loops.
    """
    usable = np.all(np.isfinite(scores), axis=0)
    if not usable.any():
        return -1
    pooled = np.where(usable, np.where(usable, scores, 0.0).sum(axis=0), -np.inf)
    if _IMPUTE_FAMILY_RULE == "pooled":
        return int(np.argmax(pooled))
    masked = np.where(usable[None, :], scores, -np.inf)
    votes = np.bincount(np.argmax(masked, axis=1), minlength=scores.shape[1])
    tied = votes == votes.max()
    return int(np.argmax(np.where(tied, pooled, -np.inf)))


def _combine_margin_fits(blk: dict, samples: list[tuple[np.ndarray, np.ndarray]],
                         rule: str) -> dict:
    """Margin update of ``"impute"``: fit on each completed series, then average.

    ``samples`` holds one weighted sample ``(y, w)`` per completed series.
    Without a GICE candidate list (or for a multivariate margin) the declared
    family is fitted on each and the parameters are averaged. With
    candidates, each family is fitted on each series; the family is chosen by
    :func:`_combine_choice` on the ``rule`` scores and its parameters are
    averaged over the series.
    """
    if not samples:
        return {}
    if not blk.get("candidates") or blk.get("dist") in _MULTIVARIATE_DIST_NAMES:
        outs = [_fit_margin_weighted(blk, y, w) for y, w in samples]
        good = [o["params"] for o in outs if o.get("params")]
        return {"params": _average_params(good)} if good else {}

    names = list(dict.fromkeys(blk["candidates"]))
    table = np.full((len(samples), len(names)), -np.inf)
    params: list[dict[str, dict]] = []
    for d, (y, w) in enumerate(samples):
        fits, scores = _margin_candidate_fits(blk, y, w, rule)
        params.append({})
        for f, s in zip(fits, scores):
            c = names.index(f["dist"])
            table[d, c] = s
            params[d][f["dist"]] = f["params"]
    c = _combine_choice(table)
    if c < 0:
        logger.warning(
            "GICE (missing data, impute): no candidate fitted on every completed "
            "series for block (i=%s, j=%s); keeping previous family.",
            blk.get("i"), blk.get("j"),
        )
        return {}
    dist = names[c]
    return {"dist": dist, "params": _average_params([p[dist] for p in params])}


def _m_step_impute(
    raw: dict,
    current: PMCModel,
    Y_draws: np.ndarray,
    xi: np.ndarray,
    *,
    fit_margins: bool,
    candidates: list[str],
    selection_criterion: str,
    margin_selection_rule: str,
    copula_margins: str = "parametric",
    miss: np.ndarray | None = None,
) -> None:
    """M-step of ICE strategy ``"impute"`` — multiple imputation, updates ``raw`` in place.

    ``Y_draws`` (D, N[, d]) are D completed series drawn jointly with the
    states from P(x, y_mis | y_obs, θ^q); ``xi`` the exact pair posteriors
    given the observed data. ``miss`` is the mask of the observed series: a
    completed series keeps it, so with a non-ignorable
    ``current.missingness`` its complete-data E-step is P(x | y^(d), m),
    with the missingness factors of that mask (not of the completed series,
    which has none missing).

    * Prior: from ``xi`` (its conditional expectation — no Monte Carlo).
    * On each completed series y^(d): the complete-data E-step under θ^q
      (forward–backward, γ^(d), ξ^(d) = P(x | y^(d))), then the complete-data
      ICE estimator of every margin and copula block (weighted MLE, family
      scores per candidate).
    * Parameters: the D estimates are averaged, as DerrodePieczynski_CSDA2013
      Eq. 24 averages the estimates of L posterior draws: τ (and a
      two-parameter family's extra
      parameter) on its natural scale, margin parameters on theirs (``loc``,
      ``scale``, shapes; mean vector and covariance matrix of a multivariate
      normal). The family is chosen once from the D score tables
      (:func:`_combine_choice`) and the average is over that family's D fits.
    * ``copula_margins="empirical"``: the pseudo-observations of completed
      series d come from the weighted empirical margins of that series with
      its own posteriors γ^(d), ξ^(d) (:func:`_empirical_margin_cdfs`) — the
      complete-data rule of :func:`_m_step`, applied per completed series.
    """
    _check_copula_margins(copula_margins)
    D = int(Y_draws.shape[0])
    _m_step_prior(raw, current, xi)
    var = current.variant

    from pmcprg.pmc import inference as _inf
    ev = None if miss is None else _inf._evidence(current, miss)
    posts = []
    for Yd in Y_draws:
        if ev is None:
            W, f_pdf = precompute_weights(current, Yd)
            a, _ = forward(current, Yd, W=W, f_pdf=f_pdf)
            b = backward(current, Yd, W=W)
        else:
            W, f_pdf = _inf._weights(current, Yd, ev)
            a, _ = _inf._forward(current, Yd, W, f_pdf, ev=ev)
            b = _inf._backward(current, Yd, W, ev=ev)
        posts.append((smooth(a, b), joint_posteriors(a, W, b)))

    if fit_margins:
        for blk in raw.get("margins", []):
            i_idx = int(blk["i"])
            samples = []
            for Yd, (gamma_d, xi_d) in zip(Y_draws, posts):
                if current.margin_structure == "pair":
                    y_ij, w_ij = _pair_margin_sample(Yd, xi_d, i_idx, int(blk["j"]))
                    if not float(w_ij.sum()) >= _PAIR_MARGIN_MIN_WEIGHT:
                        continue
                    samples.append((y_ij, w_ij))
                else:
                    samples.append((Yd, gamma_d[:, i_idx]))
            blk.update(_combine_margin_fits(blk, samples, margin_selection_rule))

    if not var.uses_copula:
        return
    if copula_margins == "parametric":
        cdfs = [_margin_cdfs(current, Yd) for Yd in Y_draws]
    else:
        cdfs = [_empirical_margin_cdfs(current, Yd, xi_d, gamma_d)
                for Yd, (gamma_d, xi_d) in zip(Y_draws, posts)]
    for blk in raw.get("copulas", []):
        ii, jj = int(blk["i"]), int(blk["j"])
        resolved = None
        tables, fits_d, total_w = [], [], 0.0
        for d in range(D):
            w = posts[d][1][:, ii, jj]
            if float(w.sum()) < 1e-12:
                continue
            total_w += float(w.sum()) / D
            u_arr, v_arr = _pair_pseudo_obs(current, cdfs[d], ii, jj)
            w = validate_weights(w, u_arr.shape[0])       # the one weight check (FR-12)
            resolved, fits = _copula_candidate_fits(
                candidates, u_arr, v_arr, w, selection_criterion,
            )
            tables.append([s for _, _, s in fits])
            fits_d.append(fits)
        if not fits_d:
            logger.debug("  (i=%d,j=%d): negligible weight, skip.", ii, jj)
            continue
        c = _combine_choice(np.asarray(tables, dtype=float))
        if c < 0:
            best_blk = _copula_placeholder(candidates, resolved, selection_criterion,
                                           {"name": candidates[0], "tau": 0.0})
        else:
            chosen = [fits[c][1] for fits in fits_d]
            best_blk = _copula_block(fits_d[0][c][0], _average_params(chosen))
            if any(p.get(FIT_FAILED_KEY) for p in chosen):
                best_blk[FIT_FAILED_KEY] = True
        if best_blk.pop(FIT_FAILED_KEY, False):
            logger.warning(
                "ICE M-step (missing data, impute), pair (i=%d, j=%d): the %s fit "
                "did not converge on every completed series (mean Σw=%.4g) — "
                "averaged parameters: %s.",
                ii, jj, best_blk.get("name"), total_w,
                {k: val for k, val in best_blk.items() if k != "name"},
            )
        blk.update(best_blk)


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
    * ``return_best_iterate``(bool, default False) — return the parameter set
                                                  with the highest
                                                  ``trace.log_liks`` value
                                                  instead of the last one.
                                                  ICE is not monotone: after
                                                  ``patience`` regressions it
                                                  stops *below* its best
                                                  iterate (by up to 50 nats
                                                  on the real series of
                                                  ``report/real_series``).
                                                  ``log_liks[q]`` is computed
                                                  in the E-step with θ^q,
                                                  before the M-step that
                                                  gives θ^(q+1); the returned
                                                  model is θ^q for
                                                  q = ``trace.best_iter``. A
                                                  run that uses all
                                                  ``max_iter`` iterations
                                                  evaluates the model of its
                                                  last M-step once more (one
                                                  E-step, no M-step), so the
                                                  trace then has
                                                  ``max_iter + 1`` entries
                                                  and every parameter set
                                                  produced is a candidate.
                                                  Multistart ranks each start
                                                  by the log-likelihood of
                                                  the model it returns.
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
    * ``multistart_workers``(int, default 1)     — run the multistart fits in
                                                  this many parallel worker
                                                  *processes*. Results are
                                                  identical to the sequential
                                                  run; per-iteration progress
                                                  reporting is unavailable
                                                  while parallel.
    * ``multistart_families``(str, default "none") — copula families of
                                                  the starts after the first
                                                  (variants with copulas).
                                                  ``"none"``: parameter jitter
                                                  only. ``"random"``: every
                                                  pair (i, j) also redrawn
                                                  among ``candidates``, τ on
                                                  the central band of the
                                                  family's range.
                                                  ``"sweep"``: every
                                                  combination of candidate
                                                  families on the diagonal
                                                  pairs (i, i), τ = 0.5 —
                                                  1 + |C|^K starts, truncated
                                                  (WARNING) or filled with
                                                  ``"random"`` starts to
                                                  ``n_starts``. ICE and SEM
                                                  can end on different copula
                                                  families from different
                                                  starts; jitter alone keeps
                                                  the families. See
                                                  :func:`pmcprg.pmc._estim_common.build_multistart_inits`.
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
    * ``init``             (str,  default "model") — initial parameter
                                                  values for ICE. One of:
                                                  ``"model"`` (use the
                                                  parameters declared in the
                                                  input :class:`PMCModel`)
                                                  or ``"kmeans"`` (cluster
                                                  ``Y`` with K-means, then
                                                  derive a hard-labelled
                                                  warm-start model via a
                                                  single supervised-style
                                                  M-step). The ``"kmeans"``
                                                  option requires
                                                  ``scikit-learn``
                                                  (``pip install
                                                  awesomepmc[ml]``).
    * ``kmeans_seed``      (int,  default 0)     — RNG seed for the K-means
                                                  initialisation (only used
                                                  when ``init == "kmeans"``).
    * ``copula_margins``   (str,  default ``"parametric"``) — margins of the
                                                  copula step's pseudo-
                                                  observations, one of
                                                  :data:`COPULA_MARGIN_MODES`.
                                                  ``"parametric"``: the
                                                  model's F_i / F_ij (the
                                                  historical behaviour).
                                                  ``"empirical"``: their
                                                  posterior-weighted
                                                  empirical counterparts,
                                                  F̂_i(y) = Σ_n γ_n(i)
                                                  1{y_n ≤ y} / (Σ_n γ_n(i)
                                                  + 1) for state margins and
                                                  the ξ-weighted dual-view
                                                  sample for pair margins
                                                  (derivation, ties and the
                                                  ``+1``:
                                                  :func:`_empirical_margin_cdfs`)
                                                  — a weighted rank pseudo-
                                                  likelihood (Chen & Fan
                                                  2006) against the extreme
                                                  pseudo-observations of
                                                  AUDIT_COPULES RB-5 / FR-7 a.
                                                  Only the copula step
                                                  changes (τ fit *and* family
                                                  scores, the k-means warm
                                                  start included); the
                                                  E-step, the prior and the
                                                  ``fit_margins`` update keep
                                                  the parametric margins, so
                                                  only the copula is
                                                  protected. Its costs,
                                                  measured: see
                                                  ``pmcprg/tests/test_fr7a_empirical_copula_margins.py``.
                                                  The fixed point is no
                                                  longer a stationary point
                                                  of ``trace.log_liks``' (the
                                                  parametric model's)
                                                  likelihood, and the
                                                  standard errors of
                                                  :mod:`pmcprg.pmc._oakes`,
                                                  :mod:`pmcprg.pmc._godambe`
                                                  and
                                                  :mod:`pmcprg.pmc._lystig_hughes`
                                                  do not apply (they need a
                                                  Chen–Fan correction that is
                                                  not implemented): a non-
                                                  default value is recorded
                                                  in the fitted model's
                                                  ``[ice]`` table and those
                                                  functions refuse such a
                                                  model. Kept out of
                                                  :func:`~pmcprg.pmc._estim_common.ice_estim_defaults`
                                                  (the GUI-widget contract):
                                                  no widget yet — set it in
                                                  the TOML ``[ice]`` table,
                                                  which a GUI round trip
                                                  preserves, or in
                                                  ``ice_cfg``.

    Missing observations (read only when Y has NaN rows; see :func:`ice`):

    * ``missing_strategy`` (str,  default ``"available"``) — ``"available"``
                                                  or ``"impute"``.
    * ``missing_draws``    (int,  default 5)     — completed series per
                                                  iteration (``"impute"``).
    * ``missing_seed``     (int,  default 0)     — RNG seed of those draws;
                                                  start s of a multistart
                                                  uses ``missing_seed + s``.
    * ``gap_nodes``        (int,  default 64)    — quadrature nodes of the
                                                  grid variants
                                                  (:mod:`pmcprg.pmc.gaps`).

    Missingness mechanism (:mod:`pmcprg.pmc.missingness`; see :func:`ice`,
    section "Missingness mechanism"):

    * ``missingness``      (str,  default ``"model"``) — one of
                                                  :data:`MISSINGNESS_MODES`.
                                                  ``"model"``: the model's
                                                  mechanism held fixed (the
                                                  historical behaviour);
                                                  ``"ignorable"``: dropped;
                                                  ``"state"`` /
                                                  ``"state-markov"``:
                                                  estimated. Kept out of
                                                  :func:`~pmcprg.pmc._estim_common.ice_estim_defaults`
                                                  (the GUI-widget contract),
                                                  like ``copula_margins``.
    """
    # Defaults come from a single source of truth so that ICE, SEM and the
    # GUI cannot silently drift apart (audit Q-9). Local import avoids a
    # circular import — _estim_common itself re-exports symbols defined
    # later in this module.
    from pmcprg.pmc._estim_common import (
        check_multistart_families,
        check_return_best_iterate,
        copula_margin_defaults,
        ice_estim_defaults,
        ice_missing_defaults,
        missingness_defaults,
    )
    defaults: dict = {**ice_estim_defaults(), **ice_missing_defaults(),
                      **copula_margin_defaults(), **missingness_defaults()}
    toml_ice = model.ice_config()
    cfg = {**defaults, **toml_ice}
    if ice_cfg:
        cfg.update(ice_cfg)
    check_multistart_families(cfg["multistart_families"])
    check_return_best_iterate(cfg["return_best_iterate"])
    _check_missing_cfg(cfg)
    _check_copula_margins(cfg["copula_margins"])
    _check_missingness_mode(cfg["missingness"])
    return cfg


# ---------------------------------------------------------------------------
# Valid values for the ``init`` config key (and a helper guard).
# ---------------------------------------------------------------------------

INIT_STRATEGIES: tuple[str, ...] = ("model", "kmeans")


def _check_init_strategy(value: str) -> str:
    """Normalise & validate the ``init`` config key."""
    if value not in INIT_STRATEGIES:
        raise ValueError(
            f"Unknown init strategy {value!r}. Valid: {list(INIT_STRATEGIES)}"
        )
    return value


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
      :meth:`CopulaEnum.correct_tau` and then into
      :meth:`CopulaEnum.constructible_tau_range` — a registered bound at
      |τ| = 1 is refused by the constructor (RB-6), a τ beyond Frank's
      reachable |τ| would not be used. The clipping is logged at DEBUG only:
      the shifted value is a draw, not an input to report.
    * **Copula extras** (BB1 ``delta``, Student ``df``): scaled by
      ``exp(jitter · N(0,1))`` and clipped into ``EXTRA_PARAM_BOUNDS``. A
      pair (τ, δ) that BB1 refuses (δ ≥ 1/(1 − τ)) has δ moved just inside
      the admissible interval by :meth:`CopulaVirt.constructible_params`;
      every pair the constructor accepts is kept as drawn.

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
        # Clip with correct_tau, then keep off a singular endpoint (|τ| = 1),
        # where the copula cannot be built (RB-6), and inside the reachable
        # |τ| (Frank). The jittered τ is no user input: log at DEBUG (F2).
        blk["tau"] = float(np.clip(entry.correct_tau(tau, warn=False),
                                   *entry.constructible_tau_range()))
        # Perturb extra parameters of multi-parameter copulas
        # (currently: BB1.delta, Student.df).
        cls_name = entry.value.CLASS_NAME
        extra_bounds = EXTRA_PARAM_BOUNDS.get(cls_name, {})
        for ekey, (xlo, xhi, _) in extra_bounds.items():
            if ekey in blk:
                v = float(blk[ekey]) * float(np.exp(jitter * rng.standard_normal()))
                blk[ekey] = float(np.clip(v, xlo, xhi))
        # τ and the extras were each drawn in their own box; a jointly
        # constrained family (BB1: δ < 1/(1 − τ)) may refuse the pair. Only
        # then are the extras moved onto the admissible set (G1), without an
        # RNG draw, so every other start is unchanged.
        params = {"tau_k": blk["tau"], **{k: blk[k] for k in extra_bounds if k in blk}}
        blk.update((k, val) for k, val in entry.klass.constructible_params(params).items()
                   if k != "tau_k")

    return PMCModel.from_dict(raw)


# ---------------------------------------------------------------------------
# K-means warm-start (alternative to model-based initialisation)
# ---------------------------------------------------------------------------

def _kmeans_label_assignment(
    Y: np.ndarray,
    K: int,
    *,
    random_state: int,
) -> np.ndarray:
    """Cluster observations ``Y`` into ``K`` groups with k-means++.

    Returns a hard label per observation, shape ``(N,)``, dtype ``int``.

    Univariate ``Y`` is reshaped to ``(N, 1)`` automatically. Requires
    ``scikit-learn`` — installed via ``pip install awesomepmc[ml]``.
    Raises :class:`ImportError` with an explicit install hint otherwise.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:                                     # pragma: no cover
        raise ImportError(
            "ICE init='kmeans' requires scikit-learn. "
            "Install with: pip install 'awesomepmc[ml]'"
        ) from exc

    Y_2d = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    km = KMeans(n_clusters=int(K), n_init=_KMEANS_N_INIT, random_state=int(random_state))
    km.fit(Y_2d)
    return km.labels_.astype(int)


def _align_kmeans_labels(
    labels: np.ndarray,
    Y: np.ndarray,
    model: PMCModel,
) -> np.ndarray:
    """Renumber K-means clusters so that cluster ``c`` becomes the state whose
    declared law fits it best.

    K-means numbers its clusters arbitrarily, while the states of ``model``
    carry fixed margins, copulas and (with ``fit_margins = False``) known
    parameters: without renumbering, about one start in two estimates the
    prior and the copulas on labels permuted with respect to the margins, and
    ICE then starts far from any sensible basin. The assignment maximises
    Σ_n log g_{k(c_n)}(y_n) over bijections clusters → states (Hungarian
    algorithm), where g_k is the law of y_n given x_n = k under ``model``:
    f_k for state margins, the mixture Σ_j (p_kj / p_k) f_kj for pair
    margins (DerrodePieczynski_CSDA2013 Eq. 12). Declared margins
    identical across states leave every assignment equal and the numbering
    unchanged.
    """
    from scipy.optimize import linear_sum_assignment

    K = model.K
    N = len(Y)
    loglik = np.empty((N, K), dtype=float)
    if model.margin_structure == "pair":
        p = model.prior_p
        for k in range(K):
            w = p[k] / max(float(p[k].sum()), MIN_POSITIVE)
            dens = sum(w[j] * np.asarray(model.margin(k, j).pdf_vec(Y), dtype=float)
                       for j in range(K))
            loglik[:, k] = np.log(np.maximum(dens, MIN_POSITIVE))
    else:
        for k in range(K):
            dens = np.asarray(model.margin(k).pdf_vec(Y), dtype=float)
            loglik[:, k] = np.log(np.maximum(dens, MIN_POSITIVE))
    # cost[c, k] = − Σ_{n in cluster c} log g_k(y_n)
    cost = np.zeros((K, K), dtype=float)
    for c in range(K):
        members = labels == c
        if members.any():
            cost[c] = -loglik[members].sum(axis=0)
    rows, cols = linear_sum_assignment(cost)
    mapping = np.empty(K, dtype=int)
    mapping[rows] = cols
    if not np.array_equal(mapping, np.arange(K)):
        logger.info(
            "ICE init='kmeans': clusters renumbered to match the declared "
            "state laws (cluster → state %s).", mapping.tolist(),
        )
    return mapping[labels]


def _warmstart_from_kmeans(
    model: PMCModel,
    Y: np.ndarray,
    *,
    random_state: int,
    fit_margins: bool,
    candidates: list[str],
    selection_criterion: str,
    margin_selection_rule: str,
    copula_margins: str = "parametric",
) -> PMCModel:
    """Build a starting PMCModel from a hard K-means clustering of ``Y``.

    The variant, K, margin distribution families, and copula candidate
    declarations of the input ``model`` are preserved. Its *values* —
    the prior matrix and, when applicable, margin parameters and copula
    τ — are replaced by their hard-labelled estimates derived from a
    single supervised-style M-step on the K-means clustering of ``Y``.
    The margin structure is preserved too: a pair model keeps its K²
    blocks, each fitted on its dual-view sub-sample of the labelled pairs
    (:func:`_pair_margin_sample`, DerrodePieczynski_CSDA2013 Eqs. 12–14);
    a pair that never occurs in the labelling keeps its declared margin.

    Parameters
    ----------
    model        : :class:`PMCModel` — initial model; defines variant, K,
                    margin families, copula candidates.
    Y            : ``np.ndarray`` shape ``(N,)`` or ``(N, d)`` — observations.
    random_state : ``int`` — RNG seed forwarded to ``sklearn.cluster.KMeans``.
    fit_margins, candidates, selection_criterion, margin_selection_rule,
    copula_margins :
                    same semantics as in :func:`_parse_ice_cfg` (with
                    ``copula_margins="empirical"`` the copulas are fitted on
                    the empirical margins of the k-means sub-samples).

    Returns
    -------
    PMCModel — the warm-started model. Safe to feed directly into the
        regular ICE loop / multistart.

    Missing observations (NaN rows of ``Y``): K-means clusters the observed
    rows only and the renumbering uses them only. Missing positions get no
    label — they are skipped, not filled from their neighbours: the one-hot
    γ is zero there and ξ is zero on every pair with a missing endpoint, and
    the M-step runs with the observation mask (:func:`_m_step` ``obs``). The
    prior is therefore estimated on the pairs with both endpoints observed,
    state margins on the observed values, pair margins on the observed
    endpoints of those pairs, copulas on those pairs. A complete ``Y`` takes
    the historical path unchanged.
    """
    miss = _missing_rows_or_none(Y)
    if miss is not None:
        return _warmstart_from_kmeans_missing(
            model, Y, miss, random_state=random_state, fit_margins=fit_margins,
            candidates=candidates, selection_criterion=selection_criterion,
            margin_selection_rule=margin_selection_rule, copula_margins=copula_margins,
        )
    K = model.K
    N = len(Y)
    labels = _kmeans_label_assignment(Y, K, random_state=random_state)
    labels = _align_kmeans_labels(labels, Y, model)

    # One-hot encode the hard labels as ξ̄ / γ posteriors.
    gamma = np.zeros((N, K), dtype=float)
    gamma[np.arange(N), labels] = 1.0
    xi = np.zeros((N - 1, K, K), dtype=float)
    xi[np.arange(N - 1), labels[:-1], labels[1:]] = 1.0

    # Reuse the M-step on these hard posteriors. ``raw`` returned by
    # ``model.raw`` is already a deep copy → safe to mutate in place.
    raw = model.raw
    current = PMCModel.from_dict(raw)
    _m_step(
        raw, current, Y, xi, gamma,
        fit_margins=fit_margins,
        candidates=candidates,
        selection_criterion=selection_criterion,
        margin_selection_rule=margin_selection_rule,
        copula_margins=copula_margins,
    )
    return PMCModel.from_dict(raw)


def _warmstart_from_kmeans_missing(
    model: PMCModel,
    Y: np.ndarray,
    miss: np.ndarray,
    *,
    random_state: int,
    fit_margins: bool,
    candidates: list[str],
    selection_criterion: str,
    margin_selection_rule: str,
    copula_margins: str = "parametric",
) -> PMCModel:
    """:func:`_warmstart_from_kmeans` for a Y with missing rows (see its docstring)."""
    K = model.K
    N = len(Y)
    obs = ~miss
    n_obs = int(obs.sum())
    if n_obs < K:
        raise ValueError(
            f"init='kmeans' needs at least K={K} observed values; Y has {n_obs} "
            f"observed out of {N}."
        )
    Y_obs = Y[obs]
    labels_obs = _kmeans_label_assignment(Y_obs, K, random_state=random_state)
    labels_obs = _align_kmeans_labels(labels_obs, Y_obs, model)
    labels = np.zeros(N, dtype=int)
    labels[obs] = labels_obs

    gamma = np.zeros((N, K), dtype=float)
    gamma[np.nonzero(obs)[0], labels_obs] = 1.0
    both = np.nonzero(obs[:-1] & obs[1:])[0]
    xi = np.zeros((N - 1, K, K), dtype=float)
    xi[both, labels[both], labels[both + 1]] = 1.0
    logger.info(
        "ICE init='kmeans' with missing observations: %d of %d rows clustered; "
        "%d of %d pairs with both endpoints observed enter the warm-start M-step.",
        n_obs, N, both.size, N - 1,
    )

    raw = model.raw
    current = PMCModel.from_dict(raw)
    _m_step(
        raw, current, Y, xi, gamma,
        fit_margins=fit_margins,
        candidates=candidates,
        selection_criterion=selection_criterion,
        margin_selection_rule=margin_selection_rule,
        obs=obs,
        copula_margins=copula_margins,
    )
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
    Parameter jitter keeps the copula families of ``model``; with
    ``multistart_families = "random"`` or ``"sweep"`` the starts also change
    them, which is what leaves a wrong-family basin (see
    :func:`_parse_ice_cfg`).

    Parameters
    ----------
    model        : PMCModel — initial parameter values (starting point).
    Y            : np.ndarray, shape (N,) or (N, d) — observation sequence;
                   non-finite rows are missing observations (below).
    ice_cfg      : dict, optional — override ICE settings from TOML [ice] section.
                    See :func:`_parse_ice_cfg` for the recognised keys.
    progress_cb  : callable, optional — invoked as
                    ``progress_cb(it, max_iter, log_lik, run_tag)`` after every
                    E-step. Designed for the GUI's progress bar; safe to leave
                    as ``None`` for headless / scripted use.

    Returns
    -------
    fitted_model : PMCModel — model with updated parameters (the best run
                              when multistart is enabled; its best iterate
                              with ``return_best_iterate``).
    trace        : IceTrace — captured per-iteration diagnostics. Use
                    ``trace.log_liks`` for the plain log-likelihood history.
                    When multistart was active, ``trace.multistart_runs``
                    holds the traces of the non-best runs.
                    ``trace.best_iter`` / ``trace.returned_iter`` locate the
                    best and the returned iterates.

    Degenerate states
    -----------------
    The returned model is checked by
    :func:`pmcprg.pmc._estim_common.degenerate_states` (a state of
    stationary weight below 0.5 %, a margin standard deviation below 1 % of
    that of the observed data, a copula τ within 1e-3 of ±1): the findings
    are stored in ``trace.degenerate`` and, when there are any, logged as one
    WARNING line. On such a fit the likelihood may be unbounded — a variance
    collapsing on a data atom gains nats without limit — and should not be
    compared with others. Nothing in the estimate changes.

    Missing observations
    --------------------
    A row of ``Y`` with a non-finite value is missing (ignorable missingness,
    MCAR/MAR — :mod:`pmcprg.pmc.gaps`). A ``Y`` without missing rows runs the
    historical algorithm unchanged. Otherwise every E-step uses the
    missing-data inference of :mod:`pmcprg.pmc.gaps` (exact K-state shortcut
    for HMC-IN, HMC-IN2 and PMC-IN with state margins, quadrature grid of
    ``gap_nodes`` nodes otherwise), ``trace.log_liks`` is the observed-data
    log-likelihood log p(y_obs), and the M-step follows ``missing_strategy``:

    * ``"available"`` (default; deterministic) — γ, ξ exact given y_obs at
      every n (``gap_posterior``); prior from ξ over all n; state margins on
      the observed y_n with weight γ_n(i); pair margins on the observed
      endpoints of their dual view (same ξ weights); copulas on the pairs
      whose two endpoints are observed, weight ξ_n(i, j). See :func:`_m_step`.
    * ``"impute"`` — multiple imputation: ``missing_draws`` completions
      (x, y_mis) ~ P(· | y_obs, θ^q) per iteration (seed ``missing_seed``),
      the complete-data ICE estimator on each completed series,
      parameter estimates averaged (DerrodePieczynski_CSDA2013
      Eq. 24 with L = ``missing_draws``), prior from the exact ξ.
      See :func:`_m_step_impute`.

    ``init = "kmeans"`` clusters the observed rows only
    (:func:`_warmstart_from_kmeans`); GICE margin selection runs on the
    samples above (observed values, or each completed series). Why
    ``"available"`` is the default (bias/RMSE against the missing rate) is
    documented at :data:`DEFAULT_MISSING_STRATEGY`.

    Missingness mechanism
    ---------------------
    The config key ``missingness`` (:data:`MISSINGNESS_MODES`, also in the
    TOML ``[ice]`` table) says what ICE does with the law of the mask m
    (:mod:`pmcprg.pmc.missingness`, section "Estimation"):

    * ``"model"`` (default) — ``model.missingness`` is carried unchanged to
      every iterate and to the returned model, its parameters held fixed:
      ignorable when the model has none (the text above), otherwise every
      E-step is given (y_obs, m) — ``trace.log_liks`` is log p(y_obs, m), γ
      and ξ are P(· | y_obs, m), the completions of ``"impute"`` are drawn
      given (y_obs, m) and the complete-data E-step of each completed series
      keeps the observed mask. A Y without missing rows still has a mask
      (m = 0), which the E-step uses too. The M-step formulas are those
      above: the factors of p(m | x) do not involve the other parameters.
    * ``"ignorable"`` — any mechanism is dropped: every iterate and the
      returned model have ``missingness is None``.
    * ``"state"`` / ``"state-markov"`` — the mechanism of that kind is
      estimated with the other parameters. E-steps as for ``"model"``; the
      M-step adds :func:`pmcprg.pmc.missingness.estimate_mechanism` on the
      exact γ given (y_obs, m) — for ``"impute"`` too, as the prior takes the
      exact ξ. Start: the model's mechanism of that kind if it has one,
      otherwise the state-independent MLE of the mask, whose first E-step is
      the ignorable one. Each M-step adds one pseudo-observation at the
      pooled rate (the boundary guard: a rate at 0 or 1 would be
      absorbing). The returned model carries the estimate. A Y without
      missing rows identifies no rate (MLE 0: the ignorable model): WARNING
      and ``"ignorable"``.

    ``trace.missingness_history`` records the ``[missingness]`` table of
    every iterate (``None`` when ignorable), like the other parameters.
    Testing a state-dependent mechanism against a state-independent one:
    :func:`pmcprg.pmc.missingness_lr.missingness_lr_test`.
    """
    cfg = _parse_ice_cfg(model, ice_cfg)
    miss = _missing_rows_or_none(Y)
    if miss is not None:
        logger.info(
            "ICE: %d of %d observations missing (%.1f %%) — observed-data "
            "likelihood, missing_strategy=%r.",
            int(miss.sum()), miss.size, 100.0 * miss.mean(), cfg["missing_strategy"],
        )
    cfg["missingness"] = _effective_missingness(cfg["missingness"], miss, "ICE")

    # Optional K-means warm-start: replace the user's initial parameters
    # by a single-shot supervised-style M-step on the K-means clustering
    # of Y. The variant, K, margin families and copula candidates are
    # preserved. Multistart (if enabled) then perturbs *this* warm-started
    # model. See ``_warmstart_from_kmeans`` for details.
    init_strategy = _check_init_strategy(str(cfg["init"]))
    if init_strategy == "kmeans":
        logger.info(
            "ICE init='kmeans': clustering Y (N=%d, K=%d, seed=%d) "
            "before estimation.",
            len(Y), model.K, int(cfg["kmeans_seed"]),
        )
        model = _warmstart_from_kmeans(
            model, Y,
            random_state=int(cfg["kmeans_seed"]),
            fit_margins=bool(cfg["fit_margins"]),
            candidates=list(cfg["candidates"]),
            selection_criterion=str(cfg["selection_criterion"]),
            margin_selection_rule=str(cfg["margin_selection_rule"]),
            copula_margins=str(cfg["copula_margins"]),
        )

    # Multistart driver is shared with SEM (see _estim_common.run_multistart).
    # Local import avoids a circular import (_estim_common imports from ice).
    from pmcprg.pmc._estim_common import report_degenerate_states, run_multistart

    def _single(init_mdl, run_tag, start_index):
        # ICE is deterministic on complete data — the start index is unused;
        # with missing_strategy="impute" it offsets the draws' seed.
        return _ice_single_run(init_mdl, Y, cfg, run_tag=run_tag, progress_cb=progress_cb,
                               seed_offset=start_index)

    fitted, trace = run_multistart(
        _single, model, cfg, label="ICE", worker=(_multistart_worker, Y),
    )
    # Diagnostic only: reads the returned model, changes nothing in it.
    report_degenerate_states(fitted, trace, Y, label="ICE")
    return fitted, trace


def _multistart_worker(raw: dict, Y, cfg: dict, run_tag: str, start_index: int):
    """Top-level (picklable) multistart worker for ``multistart_workers > 1``.

    Rebuilds the model from its raw dict inside the subprocess and runs one
    ICE fit. ``progress_cb`` cannot cross process boundaries; ``start_index``
    only offsets the seed of the ``"impute"`` draws (missing observations).
    """
    return _ice_single_run(PMCModel.from_dict(raw), Y, cfg, run_tag=run_tag,
                           seed_offset=start_index)


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
    from pmcprg.pmc._estim_common import linearise_image

    Y = linearise_image(model, img, what="ice_image")
    return ice(model, Y, ice_cfg=ice_cfg, progress_cb=progress_cb)


def _ice_single_run(
    model: PMCModel,
    Y: np.ndarray,
    cfg: dict,
    run_tag: str = "",
    progress_cb=None,
    *,
    seed_offset: int = 0,
) -> tuple[PMCModel, IceTrace]:
    """Single ICE run — extracted from :func:`ice` to support multistart.

    ``cfg`` must already be the merged dict returned by :func:`_parse_ice_cfg`.
    ``run_tag`` is an optional label included in log messages (used by the
    multistart driver to disambiguate concurrent runs in the log).
    ``seed_offset`` is added to ``missing_seed`` (strategy ``"impute"`` on a
    Y with missing rows), so each multistart run draws its own completions.

    Iterate indexing: ``log_liks[q]`` is computed in the E-step of iteration
    q with θ^q (θ^0 = the start), before the M-step that gives θ^(q+1). The
    loop stops *after* an E-step (convergence, ``patience``: the model
    returned is θ^(T-1), T = ``len(log_liks)``) or after the M-step of
    iteration ``max_iter - 1`` (θ^max_iter, never evaluated). With
    ``return_best_iterate`` the second case evaluates θ^max_iter once more
    (``log_liks[max_iter]``, snapshot, no M-step, no progress callback), and
    the model returned is θ^q for the first q maximising ``log_liks``. It is
    kept by reference: a :class:`PMCModel` is never mutated once built (the
    M-step writes ``raw``, and ``current`` is rebuilt from a deep copy), so
    holding the best one costs one model, not one per iteration.
    """
    return_best = bool(cfg.get("return_best_iterate", False))
    miss = _missing_rows_or_none(Y)
    strategy = str(cfg.get("missing_strategy", DEFAULT_MISSING_STRATEGY))
    n_draws = int(cfg.get("missing_draws", DEFAULT_MISSING_DRAWS))
    gap_nodes = cfg.get("gap_nodes")
    rng_missing = (np.random.default_rng(int(cfg.get("missing_seed", 0)) + int(seed_offset))
                   if miss is not None and strategy == "impute" else None)
    max_iter  = int(cfg["max_iter"])
    tol       = float(cfg["tol"])
    candidates = list(cfg["candidates"])
    fit_margins = bool(cfg["fit_margins"])
    # Patience: stop early if LL regresses on this many consecutive iterations.
    patience  = int(cfg["patience"])
    selection_criterion = str(cfg["selection_criterion"])
    if selection_criterion not in _SCORE_FN:
        raise ValueError(
            f"Unknown selection_criterion {selection_criterion!r}. "
            f"Valid: {sorted(_SCORE_FN)}"
        )
    margin_selection_rule = str(cfg["margin_selection_rule"])
    if margin_selection_rule not in MARGIN_SELECTION_RULES:
        raise ValueError(
            f"Unknown margin_selection_rule {margin_selection_rule!r}. "
            f"Valid: {sorted(MARGIN_SELECTION_RULES)}"
        )
    copula_margins = _check_copula_margins(
        cfg.get("copula_margins", DEFAULT_COPULA_MARGINS))
    # ``ice`` has already warned when an estimated mechanism meets a complete Y.
    mech_mode = _check_missingness_mode(cfg.get("missingness", DEFAULT_MISSINGNESS))
    if mech_mode in _ESTIMATED_MECHANISMS and miss is None:
        mech_mode = "ignorable"
    estimate_mech = mech_mode in _ESTIMATED_MECHANISMS
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
    mech_buf:   list[dict | None]       = []

    regress_streak  = 0   # consecutive iterations with LL decrease
    stopped_early   = False
    best = _BestIterate()  # used only with return_best_iterate

    # Public ``raw`` already returns a deep copy — safe to mutate.
    raw = model.raw
    # Every iterate (hence the returned model) records a non-default
    # copula_margins in its [ice] table; a no-op on the default path.
    _record_copula_margins(raw, copula_margins, section="ice")
    # Starting mechanism of missingness = "ignorable" / "state" /
    # "state-markov"; a no-op on the default path ("model").
    _set_start_missingness(raw, model, mech_mode, miss)

    # Build a mutable model reference updated each iteration
    current = PMCModel.from_dict(raw)

    logger.info(
        "%s: variant=%s  K=%d  N=%d  max_iter=%d  candidates=%s",
        log_prefix, var.value, K, N, max_iter, candidates,
    )
    if copula_margins != DEFAULT_COPULA_MARGINS and var.uses_copula:
        logger.info("%s: copula step on %s margins (FR-7 a).", log_prefix, copula_margins)
    if estimate_mech:
        logger.info("%s: estimating the %r missingness mechanism, start %s.",
                    log_prefix, mech_mode, _snapshot_missingness(current))

    for it in range(max_iter + 1 if return_best else max_iter):
        # ── E-step ────────────────────────────────────────────────────────
        if it == max_iter:
            # return_best_iterate: evaluate θ^max_iter, the model of the last
            # M-step (forward pass only; no draw, no M-step follows).
            log_lik = _evaluate_log_lik(current, Y, miss, gap_nodes)
        elif miss is None:
            W, f_pdf = precompute_weights(current, Y)
            alpha_hat, log_lik = forward(current, Y, W=W, f_pdf=f_pdf)
            beta_hat  = backward(current, Y, W=W)
            gamma     = smooth(alpha_hat, beta_hat)       # (N, K) marginal posteriors
            xi        = joint_posteriors(alpha_hat, W, beta_hat)  # (N-1, K, K)
        else:
            # Missing rows: exact posteriors given y_obs, observed-data
            # log-likelihood, and the completions of strategy "impute".
            gap = _gap_e_step(
                current, Y, miss, gap_nodes=gap_nodes, posterior=True,
                n_draws=n_draws if strategy == "impute" else 0, rng=rng_missing,
            )
            log_lik, gamma, xi = gap.log_lik, gap.gamma, gap.xi

        log_liks.append(log_lik)
        # Capture the snapshot of ``current`` corresponding to this LL value.
        tau_kk, fam_kk = _snapshot_tau_family(current)
        tau_buf.append(tau_kk)
        fam_buf.append(fam_kk)
        p_buf.append(_snapshot_prior_p(current))
        margin_buf.append(_snapshot_margins(current))
        mech_buf.append(_snapshot_missingness(current))
        if return_best:
            best.offer(it, log_lik, current)
        if it == max_iter:
            logger.info("%s model of the last M-step (iterate %d): log-lik = %.4f",
                        log_prefix, it, log_lik)
            break

        logger.info("%s iter %d: log-lik = %.4f", log_prefix, it, log_lik)
        if estimate_mech:
            logger.debug("%s iter %d: missingness %s", log_prefix, it, mech_buf[-1])
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
                    stopped_early = True
                    break
            else:
                regress_streak = 0

            # Convergence on absolute relative change
            if abs(diff_signed) / ref < tol:
                logger.info(
                    "%s converged at iteration %d (Δ/|LL| = %.2e).",
                    log_prefix, it, abs(diff_signed) / ref,
                )
                stopped_early = True
                break

        # ── M-step ───────────────────────────────────────────────────────
        if miss is not None and strategy == "impute":
            _m_step_impute(
                raw, current, gap.Y_draws, xi,
                fit_margins=fit_margins,
                candidates=candidates,
                selection_criterion=selection_criterion,
                margin_selection_rule=margin_selection_rule,
                copula_margins=copula_margins,
                miss=miss,
            )
        else:
            _m_step(
                raw, current, Y, xi, gamma,
                fit_margins=fit_margins,
                candidates=candidates,
                selection_criterion=selection_criterion,
                margin_selection_rule=margin_selection_rule,
                obs=None if miss is None else ~miss,
                copula_margins=copula_margins,
            )
        if estimate_mech:
            # γ given (y_obs, m) — exact for both strategies ("impute" too:
            # like the prior, the mechanism takes the conditional expectation).
            _m_step_missingness(raw, mech_mode, gamma, miss)
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
        best_iter      = _best_iter(log_liks),
        missingness_history = mech_buf,
    )
    if return_best and best.model is not None:
        trace.returned_iter = best.iter
        if best.iter < len(log_liks) - 1:
            logger.info(
                "%s: returning the best iterate %d (log-lik %.4f) instead of the "
                "last one %d (log-lik %.4f).",
                log_prefix, best.iter, log_liks[best.iter],
                len(log_liks) - 1, log_liks[-1],
            )
        return best.model, trace
    trace.returned_iter = len(log_liks) - 1 if stopped_early else len(log_liks)
    return current, trace


def _ll_rank(log_lik: float) -> float:
    """Ranking key of a log-likelihood: NaN ranks below every number."""
    v = float(log_lik)
    return -np.inf if np.isnan(v) else v


def _best_iter(log_liks) -> int:
    """First index of the highest log-likelihood (NaN lowest); -1 if empty."""
    best, best_key = -1, -np.inf
    for q, ll in enumerate(log_liks):
        if best < 0 or _ll_rank(ll) > best_key:
            best, best_key = q, _ll_rank(ll)
    return best


class _BestIterate:
    """Running argmax of the log-likelihood trace and the model θ^q it belongs to.

    Same tie and NaN rule as :func:`_best_iter` (the first maximum wins), so
    ``iter`` always equals ``_best_iter(log_liks)``. Only a reference to the
    model is stored, replaced on improvement (models are never mutated).
    """

    __slots__ = ("iter", "key", "model")

    def __init__(self) -> None:
        self.iter, self.key, self.model = -1, -np.inf, None

    def offer(self, q: int, log_lik: float, model) -> None:
        if self.model is None or _ll_rank(log_lik) > self.key:
            self.iter, self.key, self.model = q, _ll_rank(log_lik), model


def _evaluate_log_lik(model: PMCModel, Y: np.ndarray, miss: np.ndarray | None,
                      gap_nodes) -> float:
    """Log-likelihood of ``model`` — the forward pass of the estimators' E-step.

    Complete data: :func:`forward`; missing rows: the observed-data
    log-likelihood of :func:`_gap_e_step` (forward pass only, no draw — the
    same value as its posterior pass). Used for the final iterate of
    ``return_best_iterate`` (ICE and SEM).
    """
    if miss is None:
        W, f_pdf = precompute_weights(model, Y)
        return forward(model, Y, W=W, f_pdf=f_pdf)[1]
    return _gap_e_step(model, Y, miss, gap_nodes=gap_nodes, posterior=False,
                       n_draws=0, rng=None).log_lik


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from pmcprg.pmc.logging_setup import configure as _cfg_log
    _cfg_log(level=logging.INFO)

    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate
    from pmcprg.pmc.inference import error_rate, classify

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
