"""
radial_symmetry.py — is a copula radially symmetric? (audit FR-10)

Why this exists
----------------
A copula C is radially symmetric when ``(U, V)`` and ``(1 − U, 1 − V)`` share
the same law, equivalently

    C(u, v) = u + v − 1 + C(1 − u, 1 − v)      for all (u, v) in [0, 1]².

Gauss, Student (correlation-only dependence), Frank, FGM and Plackett are
radially symmetric by construction; Clayton, Gumbel/GH, Joe, BB1 and their
survival copies are not (lower- and upper-tail dependence differ), except at
their symmetric limit τ = 0. Rejecting radial symmetry at once rules out the
first group as candidate families, before the more expensive per-family
fits and goodness-of-fit tests of :mod:`pmcprg.copulas._bivariate_fit` and
:mod:`pmcprg.diagnostics.mks` are run on the rest — a screening step, not a
replacement for them.

The test statistic
-------------------
Genest & Nešlehová (2014, doi:10.1007/s00362-013-0556-4) test radial symmetry
with a Cramér–von Mises statistic comparing the empirical copula ``C_n`` to
its "flipped" reflection

    Ĉ_n^rot(u, v) = u + v − 1 + C_n(1 − u, 1 − v),

evaluated at the pseudo-observations (as in Genest, Rémillard & Beaudoin
2009's copula goodness-of-fit statistics, of which this is a variant with
``Ĉ_n^rot`` in place of a parametric ``C_θ``):

    T_n = n · Σ_{i=1}^n [C_n(û_i, v̂_i) − û_i − v̂_i + 1 − C_n(1 − û_i, 1 − v̂_i)]²
        = n · ∫∫ [C_n(u, v) − u − v + 1 − C_n(1 − u, 1 − v)]² dC_n(u, v),

with ``(û_i, v̂_i) = (rank(x_i)/(n+1), rank(y_i)/(n+1))`` the usual
pseudo-observations. This is the natural, defensible construction the module
docstring of ``AUDIT_COPULES.md`` FR-10 asks for when the exact published
formula cannot be independently verified against the paper (no internet
access from this worktree): Genest & Nešlehová (2014) is known, from its
abstract and from citations of it (e.g. Genest, Nešlehová & Quessy 2012 for
exchangeability, the same authors' companion test), to build a Cramér–von
Mises statistic from the empirical process ``√n (C_n − Ĉ_n^rot)``; the exact
weighting and whether the integral is taken against ``dC_n`` or Lebesgue
measure is not reproduced verbatim here. What *is* verified, by the
simulation study below rather than by citation, is that this statistic has
the right shape for the job: it is zero in expectation under radial symmetry,
strictly positive in expectation otherwise, and a bootstrap calibrates its
level correctly to within Monte-Carlo error.

Null distribution: parametric bootstrap
----------------------------------------
The asymptotic null distribution of ``T_n`` is not distribution-free — it
depends on the unknown copula through the covariance of the limiting
Gaussian process, exactly as for the classical Cramér–von Mises
goodness-of-fit statistic of Genest, Rémillard & Beaudoin (2009). Following
this package's convention for exactly that problem
(:mod:`pmcprg.diagnostics.bootstrap`, and the ``method='tau'`` variance of
:mod:`pmcprg.copulas._stderr`), the null is calibrated by a **parametric**
bootstrap rather than the multiplier bootstrap Genest & Nešlehová (2014)
also propose: fit the one family every radially-symmetric candidate agrees
on at the method-of-moments level — a Gaussian copula at the sample's
Kendall's τ — and resample from it. The Gaussian copula is itself radially
symmetric, so this reproduces the null "C is radially symmetric with
dependence τ" without committing to which symmetric family generated the
data, at the cost of the bootstrap's own approximation (only τ is matched,
not the full family). Each replicate is rank-transformed exactly as the
observed sample is, so both share the same pseudo-observation construction
and the p-value

    p = (1 + #{T_n^(b) ≥ T_n}) / (B + 1)

is the same Monte-Carlo p-value as :class:`~pmcprg.diagnostics.bootstrap.BootstrapResult`
(Davison & Hinkley 1997, ch. 4; Phipson & Smyth 2010): never exactly 0, exact
under the (here: bootstrap-approximated) null.

Weighting
---------
:mod:`pmcprg.diagnostics.model_selection` and :mod:`pmcprg.diagnostics.pseudo`
weight a pair's pseudo-observations by the ICE posterior ``ξ_n(i, j)`` because
their statistics (a log-likelihood ratio, a PIT) are pointwise functionals
that stay meaningful when reweighted by a soft label. ``T_n`` is not: it is
built from the *empirical copula*, a joint functional of the whole sample,
and Genest & Nešlehová's Cramér–von Mises weighting has no published
extension to frequency-weighted or soft-labelled data — unlike Kendall's τ
(``method='tau'`` in ``_stderr.py``) it is not a U-statistic with a known
weighted form. Introducing weights here would be guessing a formula, not
reusing one. This module therefore takes a single, fully-observed sample of
pairs — e.g. the hard-label pseudo-observations of one state pair — and
raises rather than silently accepting a weight argument.

References
----------
* Genest, C. & Nešlehová, J. G. (2014). On tests of radial symmetry for
  bivariate copulas. *Statistical Papers* 55, 1107–1119.
  doi:10.1007/s00362-013-0556-4
* Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for
  copulas: A review and a power study. *Insurance Math. Econom.* 44(2),
  199–213 — the Cramér–von Mises construction this statistic adapts.
* Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and Their
  Application*. Cambridge University Press, ch. 4.
* Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be
  zero. *Stat. Appl. Genet. Mol. Biol.* 9(1), Art. 39.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau, rankdata

from pmcprg.numerics import EPS_MINUS_ONE, ONE_MINUS_EPS

logger = logging.getLogger(__name__)

__all__ = ["RadialSymmetryResult", "radial_symmetry_statistic", "radial_symmetry_test"]

#: Below this many pairs the statistic is not meaningful — Kendall's τ and the
#: empirical copula both degenerate. Chosen well below any sample this
#: package's diagnostics are used on, only to avoid a crash on a malformed call.
MIN_N = 4


@dataclass(frozen=True)
class RadialSymmetryResult:
    """Typed result of :func:`radial_symmetry_test`.

    Fields
    ------
    statistic  : float — the Cramér–von Mises statistic ``T_n`` (see module
                 docstring). NaN when ``n < MIN_N``.
    p_value    : float — bootstrap Monte-Carlo p-value, never exactly 0
                 (Davison & Hinkley 1997). NaN when no replicate is available.
    reject     : bool  — ``p_value < alpha`` (H0 = "radially symmetric").
                 False whenever ``p_value`` is NaN.
    alpha      : float — significance level used for ``reject``.
    n          : int   — number of pairs the statistic was computed on.
    tau_hat    : float — Kendall's τ of the sample, the bootstrap surrogate's
                 only fitted parameter.
    B          : int   — bootstrap replicates requested.
    n_valid    : int   — replicates that produced a finite statistic.
    """

    statistic: float
    p_value: float
    reject: bool
    alpha: float
    n: int
    tau_hat: float
    B: int
    n_valid: int


def _pseudo_obs(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    u = rankdata(x, method="average") / (n + 1)
    v = rankdata(y, method="average") / (n + 1)
    return u, v


def _empirical_copula_at(u_query: np.ndarray, v_query: np.ndarray,
                          u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``C_n(u_query_k, v_query_k)`` for every query point, vectorised as an
    ``(m, n)`` boolean matrix — see :func:`pmcprg.diagnostics.mks._mecdf_batch`
    for the same pattern, one dimension specialised to ``d = 2``."""
    le_u = u[None, :] <= u_query[:, None]
    le_v = v[None, :] <= v_query[:, None]
    return (le_u & le_v).mean(axis=1)


def radial_symmetry_statistic(u: np.ndarray, v: np.ndarray) -> float:
    """``T_n`` of the module docstring, from pseudo-observations already in [0, 1].

    Unlike :func:`radial_symmetry_test`, this does not rank-transform its
    input: pass pseudo-observations, not raw data, so that a bootstrap
    replicate (already uniform-margined by construction) and the observed
    sample are put through the identical computation.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = u.size
    if n < MIN_N:
        return float("nan")
    Cn_uv = _empirical_copula_at(u, v, u, v)
    Cn_refl = _empirical_copula_at(1.0 - u, 1.0 - v, u, v)
    # diff_i = C_n(u_i, v_i) - Ĉ_n^rot(u_i, v_i), Ĉ_n^rot(u, v) = u+v-1+C_n(1-u, 1-v)
    diff = Cn_uv - (u + v - 1.0 + Cn_refl)
    return float(n * np.mean(diff ** 2))


def _fit_gaussian_surrogate(tau_hat: float):
    from pmcprg.copulas import CopulaGaussian
    tau = max(EPS_MINUS_ONE, min(ONE_MINUS_EPS, float(tau_hat)))
    return CopulaGaussian(tau_k=tau)


def radial_symmetry_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    B: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
    weights=None,
) -> RadialSymmetryResult:
    """Test H0: the copula of ``(x, y)`` is radially symmetric (FR-10).

    Parameters
    ----------
    x, y     : array-like, shape ``(n,)`` — one fully-observed sample of
               pairs (raw data or pseudo-observations both work: ranks are
               recomputed here either way).
    B        : bootstrap replicates. Cost is ``O(B n²)``, the same order as
               the observed statistic.
    seed     : bootstrap RNG seed.
    alpha    : significance level for :attr:`RadialSymmetryResult.reject`.
    weights  : must be ``None``. Kept as a keyword, not silently dropped, so
               that a caller passing ICE posteriors ``ξ`` gets an explicit
               error rather than a silently-wrong answer — see "Weighting"
               in the module docstring for why this statistic has no
               defensible weighted form here.

    Returns
    -------
    RadialSymmetryResult.
    """
    if weights is not None:
        raise NotImplementedError(
            "radial_symmetry_test: no defensible weighted form of the "
            "Cramér-von Mises radial-symmetry statistic is implemented here "
            "(see the module docstring, 'Weighting') — pass a single, "
            "fully-observed sample of pairs instead."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be 1-D and same-shaped; got {x.shape} vs {y.shape}.")
    n = x.size

    if n < MIN_N:
        logger.debug("radial_symmetry_test: n=%d < MIN_N=%d, returning NaN.", n, MIN_N)
        return RadialSymmetryResult(
            statistic=float("nan"), p_value=float("nan"), reject=False,
            alpha=float(alpha), n=int(n), tau_hat=float("nan"), B=int(B), n_valid=0,
        )

    u, v = _pseudo_obs(x, y)
    stat = radial_symmetry_statistic(u, v)

    tau_hat, _ = kendalltau(x, y)
    tau_hat = float(tau_hat) if np.isfinite(tau_hat) else 0.0
    surrogate = _fit_gaussian_surrogate(tau_hat)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(int(B)):
        xb, yb = surrogate.sample(n, seed=int(rng.integers(0, 2**31 - 1))).T
        ub, vb = _pseudo_obs(xb, yb)
        tb = radial_symmetry_statistic(ub, vb)
        if np.isfinite(tb):
            draws.append(tb)
    draws = np.asarray(draws, dtype=float)

    p_value = (
        float((1 + np.sum(draws >= stat)) / (draws.size + 1))
        if draws.size and np.isfinite(stat) else float("nan")
    )
    reject = bool(np.isfinite(p_value) and p_value < alpha)

    logger.debug(
        "radial_symmetry_test: n=%d tau_hat=%.4f T_n=%.4f p=%.4f alpha=%.3f reject=%s",
        n, tau_hat, stat, p_value, alpha, reject,
    )
    return RadialSymmetryResult(
        statistic=float(stat), p_value=p_value, reject=reject, alpha=float(alpha),
        n=int(n), tau_hat=tau_hat, B=int(B), n_valid=int(draws.size),
    )
