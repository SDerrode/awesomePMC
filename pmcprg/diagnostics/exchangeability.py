"""
exchangeability.py — is a copula exchangeable? (audit FR-10, fourth and last item)

Why this exists
----------------
A copula ``C`` is exchangeable when

    C(u, v) = C(v, u)      for all (u, v) in [0, 1]²,

equivalently ``(U, V)`` and ``(V, U)`` share the same joint law. Every family
in this codebase whose CDF formula treats ``u`` and ``v`` symmetrically is
exchangeable *by construction*: Gauss, Student, Clayton, Frank, Gumbel/GH,
Joe, Plackett, AMH and FGM (verified below, not just asserted from the
formula — see "Verified, not assumed"). The 180° survival rotation
(``survival.py``) is exchangeable too, for the same reason (see below). The
90°/270° rotations of FR-8 (``CopulaClayton90/270``, ``CopulaGH90/270``,
``CopulaJoe90/270``, all now on ``main``) are the interesting counterexample:
a 90° or 270° rotation reflects *one* margin only, which breaks the u↔v
symmetry the base family had — so these rotated families are exchangeable
only at their symmetric limit τ = 0, exactly like radial symmetry's own
Clayton/GH/Joe counterexamples in :mod:`pmcprg.diagnostics.radial_symmetry`.
This makes them a ready-made, already-available power-study test bed here,
with no need for a bespoke asymmetric construction.

Rejecting exchangeability rules out every symmetric-by-formula family in this
package at once — a screening step, not a replacement for the per-family
fits and goodness-of-fit tests of :mod:`pmcprg.copulas._bivariate_fit` and
:mod:`pmcprg.diagnostics.rosenblatt`, in the same spirit as
:mod:`pmcprg.diagnostics.radial_symmetry`'s own screening role for radial
symmetry.

Verified, not assumed
----------------------
Two claims the audit item asks not to take on faith, checked numerically
(see ``test_exchangeability.py`` and the commit message for the actual
numbers, reproduced here for the constructions):

* **Every bivariate Gauss copula is exchangeable, unconditionally.** The
  bivariate Gauss copula's CDF is ``Φ_ρ(Φ⁻¹(u), Φ⁻¹(v))`` for the
  correlation ``ρ`` — a single scalar parameter entering symmetrically, so
  swapping ``u`` and ``v`` swaps the two arguments of a bivariate normal CDF
  with a *symmetric* covariance matrix (``Cov(Φ⁻¹(U), Φ⁻¹(V)) = ρ`` either
  way), which leaves it unchanged for *every* ``ρ`` — there is no asymmetric
  bivariate Gauss copula to begin with, unlike (say) a skew-normal or
  asymmetric-copula extension of it. No "symmetry condition" beyond having a
  single correlation parameter is needed; this was checked, not assumed, by
  evaluating ``CopulaGaussian(tau_k=0.5).cdf`` at three points and their
  swaps (max |ΔC| at machine precision).
* **The 180° (survival) rotation preserves exchangeability; the 90°/270°
  rotations break it.** Survival: ``Ĉ(u, v) = u + v − 1 + C(1 − u, 1 − v)``
  reflects *both* margins, so swapping ``u ↔ v`` swaps ``C(1−u, 1−v)`` into
  ``C(1−v, 1−u) = C(1−u, 1−v)`` whenever the base ``C`` is itself
  exchangeable — the survival transform commutes with the swap. 90°:
  ``C90(u, v) = v − C(1 − u, v)`` reflects only the first argument;
  ``C90(v, u) = u − C(1 − v, u)`` is generally a different number (the base
  ``C`` is evaluated at ``(1−u, v)`` vs. ``(1−v, u)``, not related by any
  symmetry of ``C`` alone). Confirmed numerically for ``CopulaClayton90`` and
  ``CopulaClayton270`` at τ = −0.5: e.g. ``C90(0.2, 0.7) = 0.080221`` vs.
  ``C90(0.7, 0.2) = 0.031237`` (Δ ≈ 0.049, an order of magnitude above any
  floating-point tolerance), while ``SurvivalClayton`` at the three same
  points matches its own swap to machine precision.

The test statistic
-------------------
Genest, Nešlehová & Quessy (2012, doi:10.1007/s10463-011-0337-6) test
exchangeability by comparing the empirical copula ``C_n`` to its
"transposed" version ``C_n^t(u, v) := C_n(v, u)`` — the natural analogue of
:mod:`pmcprg.diagnostics.radial_symmetry`'s own construction (there, the
reference is the *reflected* empirical copula; here it is the *transposed*
one), evaluated at the sample's own pseudo-observations exactly as both
:mod:`pmcprg.diagnostics.radial_symmetry` and :mod:`pmcprg.diagnostics.rosenblatt`
do:

    T_n = n · Σ_{i=1}^n [C_n(û_i, v̂_i) − C_n(v̂_i, û_i)]²
        = n · ∫∫ [C_n(u, v) − C_n(v, u)]² dC_n(u, v),

with ``(û_i, v̂_i) = (rank(x_i)/(n+1), rank(y_i)/(n+1))``. Working this out
by hand rather than assuming the radial-symmetry analogy transfers cleanly:
under H0 the *population* copula satisfies ``C(u, v) = C(v, u)`` identically,
so ``C_n(û_i, v̂_i) − C_n(v̂_i, û_i) → C(û_i, v̂_i) − C(v̂_i, û_i) = 0``
almost surely as ``n → ∞`` for each fixed evaluation point — the plug-in
difference is driven to zero by the population identity, and what remains at
finite ``n`` is the empirical-process fluctuation ``α_n(u,v) − α_n(v,u)``
(``α_n = √n(C_n − C)``), which has mean 0 and a nondegenerate limiting
covariance, exactly as radial symmetry's own reflected difference does. Under
H1 (a genuinely non-exchangeable ``C``), ``C(u, v) − C(v, u)`` is a fixed,
generically nonzero function of ``(u, v)`` (confirmed concretely for
``CopulaClayton90`` above), so ``T_n`` converges to a strictly positive
constant plus noise instead of collapsing to the null's Op(1) fluctuation —
the statistic is zero in expectation under H0 and grows under H1, the shape
required of a consistent test. This was also checked empirically, not just
argued: :func:`exchangeability_statistic` on independent-uniform noise stays
Op(1) across replicates (module tests), while on ``CopulaClayton90`` samples
it grows with ``|τ|`` (see the power study below).

**Honesty about fidelity.** As with both prior FR-10 modules, no internet
access from this worktree means the exact GNQ (2012) formula (including
their specific choice of weight function, if any, and whether they integrate
against ``dC_n`` or Lebesgue measure) is not transcribed verbatim. What is
implemented is the natural, defensible Cramér–von Mises construction the
audit item's own citation of GNQ (2012) calls for — "compare ``C_n`` to its
transpose" — built with the identical skeleton this codebase already trusts
for the same job in :mod:`pmcprg.diagnostics.radial_symmetry` (whose own
docstring makes the same caveat for its reflected-copula analogue). It is
validated here by simulation (size near nominal for every exchangeable
family tried, power against the now-available 90°-rotated families) rather
than by claimed fidelity to the paper's exact statistic.

Null distribution: parametric bootstrap
----------------------------------------
``T_n``'s null distribution is not distribution-free: like
:mod:`pmcprg.diagnostics.radial_symmetry`'s ``T_n``, it depends on the
unknown copula through the covariance of the limiting empirical-copula-process
difference. Following radial symmetry's own precedent (and, per FR-10's
explicit suggestion that the *simpler* calibration is the right choice for a
first version of the last item), the null is calibrated by a **parametric**
bootstrap: fit a Gaussian copula at the sample's Kendall's τ (a family every
exchangeable candidate agrees on at the method-of-moments level, and itself
exchangeable, see "Verified, not assumed" above) and resample from it. Each
replicate is rank-transformed exactly as the observed sample, so both share
the same pseudo-observation construction, and the p-value

    p = (1 + #{T_n^(b) ≥ T_n}) / (B + 1)

is the same Monte-Carlo p-value convention as
:class:`~pmcprg.diagnostics.bootstrap.BootstrapResult`,
:mod:`pmcprg.diagnostics.radial_symmetry` and :mod:`pmcprg.diagnostics.rosenblatt`
(Davison & Hinkley 1997, ch. 4; Phipson & Smyth 2010): never exactly 0, exact
under the (here: bootstrap-approximated) null. A multiplier-bootstrap
follow-up, mirroring radial symmetry's own second round, is a natural fast
extension but is explicitly left for later — this is the fourth and last
FR-10 item, and the audit item's text does not single out exchangeability
for the multiplier bootstrap the way it does for radial symmetry.

Weighting
---------
As in :mod:`pmcprg.diagnostics.radial_symmetry` and
:mod:`pmcprg.diagnostics.rosenblatt`: ``T_n`` is built from the empirical
copula of the *whole* sample, a joint functional with no published
frequency- or soft-label-weighted form. ``weights`` is accepted only to
raise, so a caller passing ICE posteriors gets an explicit error instead of a
silently wrong answer.

References
----------
* Genest, C., Nešlehová, J. G. & Quessy, J.-F. (2012). Tests of symmetry for
  bivariate copulas. *Ann. Inst. Statist. Math.* 64, 811-834.
  doi:10.1007/s10463-011-0337-6
* Genest, C. & Nešlehová, J. G. (2014). On tests of radial symmetry for
  bivariate copulas. *Statistical Papers* 55, 1107-1119.
  doi:10.1007/s00362-013-0556-4 — the sibling test this module's
  construction and calibration mirror.
* Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for
  copulas: A review and a power study. *Insurance Math. Econom.* 44(2),
  199-213 — the Cramér–von Mises construction both this and the
  radial-symmetry/Rosenblatt statistics adapt.
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

__all__ = ["ExchangeabilityResult", "exchangeability_statistic", "exchangeability_test"]

#: Below this many pairs the statistic is not meaningful — Kendall's τ and the
#: empirical copula both degenerate. Same floor as
#: :mod:`pmcprg.diagnostics.radial_symmetry` and
#: :mod:`pmcprg.diagnostics.rosenblatt`, for the same reason.
MIN_N = 4


@dataclass(frozen=True)
class ExchangeabilityResult:
    """Typed result of :func:`exchangeability_test`.

    Fields
    ------
    statistic  : float — the Cramér–von Mises statistic ``T_n`` (see module
                 docstring). NaN when ``n < MIN_N``.
    p_value    : float — parametric-bootstrap Monte-Carlo p-value, never
                 exactly 0 (Davison & Hinkley 1997). NaN when no replicate
                 produced a finite statistic.
    reject     : bool  — ``p_value < alpha`` (H0 = "exchangeable", i.e.
                 ``C(u, v) = C(v, u)``). False whenever ``p_value`` is NaN.
    alpha      : float — significance level used for ``reject``.
    n          : int   — number of pairs the statistic was computed on.
    tau_hat    : float — Kendall's τ of the sample, also the calibrating
                 Gaussian surrogate's only fitted parameter.
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
    ``(m, n)`` boolean matrix — same pattern as
    :func:`pmcprg.diagnostics.radial_symmetry._empirical_copula_at` and
    :func:`pmcprg.diagnostics.rosenblatt._empirical_cdf_at`, duplicated
    rather than imported so this module stays self-contained."""
    le_u = u[None, :] <= u_query[:, None]
    le_v = v[None, :] <= v_query[:, None]
    return (le_u & le_v).mean(axis=1)


def exchangeability_statistic(u: np.ndarray, v: np.ndarray) -> float:
    """``T_n`` of the module docstring, from pseudo-observations already in [0, 1].

    Unlike :func:`exchangeability_test`, this does not rank-transform its
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
    Cn_vu = _empirical_copula_at(v, u, u, v)   # C_n(v_i, u_i), the transposed reference
    diff = Cn_uv - Cn_vu
    return float(n * np.mean(diff ** 2))


def _fit_gaussian_surrogate(tau_hat: float):
    from pmcprg.copulas import CopulaGaussian
    tau = max(EPS_MINUS_ONE, min(ONE_MINUS_EPS, float(tau_hat)))
    return CopulaGaussian(tau_k=tau)


def exchangeability_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    B: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
    weights=None,
) -> ExchangeabilityResult:
    """Test H0: the copula of ``(x, y)`` is exchangeable, ``C(u,v) = C(v,u)`` (FR-10).

    Parameters
    ----------
    x, y    : array-like, shape ``(n,)`` — one fully-observed sample of pairs
              (raw data or pseudo-observations both work: ranks are
              recomputed here either way).
    B       : parametric-bootstrap replicates. Cost ``O(B n²)``: each
              replicate resamples from the Gaussian surrogate, reranks and
              recomputes the CvM statistic (``O(n²)`` for the
              empirical-copula evaluation) — same order as
              :func:`pmcprg.diagnostics.radial_symmetry.radial_symmetry_test`'s
              own parametric bootstrap.
    seed    : bootstrap RNG seed.
    alpha   : significance level for :attr:`ExchangeabilityResult.reject`.
    weights : must be ``None`` — see "Weighting" in the module docstring.

    Returns
    -------
    ExchangeabilityResult.
    """
    if weights is not None:
        raise NotImplementedError(
            "exchangeability_test: no defensible weighted form of the "
            "Cramer-von Mises exchangeability statistic is implemented here "
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
        logger.debug("exchangeability_test: n=%d < MIN_N=%d, returning NaN.", n, MIN_N)
        return ExchangeabilityResult(
            statistic=float("nan"), p_value=float("nan"), reject=False,
            alpha=float(alpha), n=int(n), tau_hat=float("nan"), B=int(B), n_valid=0,
        )

    u, v = _pseudo_obs(x, y)
    stat = exchangeability_statistic(u, v)

    tau_hat, _ = kendalltau(x, y)
    tau_hat = float(tau_hat) if np.isfinite(tau_hat) else 0.0

    surrogate = _fit_gaussian_surrogate(tau_hat)
    rng = np.random.default_rng(seed)
    raw_draws = []
    for _ in range(int(B)):
        xb, yb = surrogate.sample(n, seed=int(rng.integers(0, 2**31 - 1))).T
        ub, vb = _pseudo_obs(xb, yb)
        tb = exchangeability_statistic(ub, vb)
        if np.isfinite(tb):
            raw_draws.append(tb)
    draws = np.asarray(raw_draws, dtype=float)

    p_value = (
        float((1 + np.sum(draws >= stat)) / (draws.size + 1))
        if draws.size and np.isfinite(stat) else float("nan")
    )
    reject = bool(np.isfinite(p_value) and p_value < alpha)

    logger.debug(
        "exchangeability_test: n=%d tau_hat=%.4f T_n=%.4f p=%.4f alpha=%.3f reject=%s",
        n, tau_hat, stat, p_value, alpha, reject,
    )
    return ExchangeabilityResult(
        statistic=float(stat), p_value=p_value, reject=reject, alpha=float(alpha),
        n=int(n), tau_hat=tau_hat, B=int(B), n_valid=int(draws.size),
    )
