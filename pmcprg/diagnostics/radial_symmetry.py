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

Null distribution: multiplier bootstrap (FR-10 round 2)
--------------------------------------------------------
FR-10 actually asks for the **multiplier bootstrap** of Kojadinovic & Yan
(2011, doi:10.1007/s11222-009-9142-y; Kojadinovic, Yan & Holmes 2011,
doi:10.5705/ss.2011.037a) instead of the parametric bootstrap above: same
asymptotics for the null of ``T_n``, without refitting or resampling a
surrogate copula ``B`` times. ``bootstrap="multiplier"`` implements it.

The derivation, worked out here (no published formula for *this* statistic
to transcribe — see the caveat below): under H0 the deterministic term
``u + v − 1`` cancels between ``T_n``'s two empirical-copula evaluations, so

    √n D_n(u, v) := √n [C_n(u, v) − u − v + 1 − C_n(1 − u, 1 − v)]
                  = α_n(u, v) − α_n(1 − u, 1 − v) + o_P(1),

where ``α_n = √n(C_n − C)`` is the empirical copula process. Its standard
multiplier-CLT linearisation (Rémillard & Scaillet 2009; Kojadinovic & Yan
2011; the same construction underlies the ``copula`` R package's
``gofCopula(sim = "mult")`` and, in this codebase, the score corrections
``Ẇ₁, Ẇ₂`` of :func:`pmcprg.copulas._stderr.standard_errors` — a plug-in
empirical partial derivative reweighted by an i.i.d. term per observation)
replaces ``α_n(u, v)`` by

    α_n^ξ(u, v) = n^{-1/2} Σ_{i=1}^n ξ_i [1{û_i ≤ u, v̂_i ≤ v} − C_n(u, v)
                  − Ċ_1(u, v)(1{û_i ≤ u} − u) − Ċ_2(u, v)(1{v̂_i ≤ v} − v)],

with ``ξ_1, …, ξ_n`` i.i.d., mean 0, variance 1 (independent of the data,
resampled fresh per replicate — standard normal by default, ``±1``
Rademacher optionally), and ``Ċ_1, Ċ_2`` the partial derivatives of ``C_n``
in each argument, estimated by a central finite difference with a one-sided
correction at the boundary (:func:`_empirical_copula_partials`; bandwidth
``h = min(0.5, n^{-1/2})``, the same order Rémillard & Scaillet and
Kojadinovic & Yan use for this plug-in). One bootstrap replicate is

    T_n^ξ = mean_i [α_n^ξ(û_i, v̂_i) − α_n^ξ(1 − û_i, 1 − v̂_i)]².

Because ``C_n``, its ranks and ``Ċ_1, Ċ_2`` are computed **once** from the
observed sample, a replicate is only a fresh draw of ``ξ`` and a
matrix–vector product — no resampling, no rank recomputation, no refit. All
``B`` replicates are produced by one ``(2n × n) @ (n × B)`` matrix product,
which is what makes this the fast alternative FR-10 asks for (see the
speed-up measured in ``CHANGELOG.md`` and ``test_radial_symmetry.py``).

**Honesty about fidelity.** This is a derivation from the general multiplier
CLT for the empirical copula process, applied by hand to *this* statistic's
particular reflected-difference form — not a transcription of a published
formula for the radial-symmetry statistic specifically (Kojadinovic & Yan
2011 give the general recipe for a CvM statistic built from ``C_n`` against
a fixed or parametrically-estimated reference; here the "reference" is the
sample's own reflection, which needed working out). It is offered with the
same posture as the parametric-bootstrap statistic itself: the shape of the
construction is principled and directly parallels machinery this codebase
already trusts (``_stderr.py``'s ``Ẇ₁, Ẇ₂``), but it is validated here by
simulation (size and power, matched against the parametric bootstrap's own
numbers on the same family/τ/N grid) rather than by claimed fidelity to a
specific paper's exact statistic.

References (multiplier bootstrap)
----------------------------------
* Kojadinovic, I. & Yan, J. (2011). A goodness-of-fit test for multivariate
  multiplicative models with unspecified marginals. *Stat. Comput.* 21,
  17–30. doi:10.1007/s11222-009-9142-y
* Kojadinovic, I., Yan, J. & Holmes, M. (2011). Fast large-sample
  goodness-of-fit tests for copulas. *Statist. Sinica* 21, 841–871.
  doi:10.5705/ss.2011.037a
* Rémillard, B. & Scaillet, O. (2009). Testing for equality between two
  copulas. *J. Multivariate Anal.* 100(3), 377–386 — the multiplier-CLT
  linearisation of the empirical copula process this module reuses.

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
import math
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

#: Bootstrap calibration methods accepted by :func:`radial_symmetry_test`.
BOOTSTRAP_METHODS = ("parametric", "multiplier")
#: Multiplier laws accepted for ``bootstrap="multiplier"`` — both i.i.d.,
#: mean 0, variance 1, as the multiplier CLT requires.
MULTIPLIER_LAWS = ("normal", "rademacher")


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
    tau_hat    : float — Kendall's τ of the sample. For ``bootstrap='parametric'``
                 it is also the calibrating Gaussian surrogate's only fitted
                 parameter; for ``bootstrap='multiplier'`` it is reported for
                 diagnostics only (the multiplier bootstrap fits nothing).
    B          : int   — bootstrap replicates requested.
    n_valid    : int   — replicates that produced a finite statistic.
    bootstrap  : str   — ``'parametric'`` (default) or ``'multiplier'``, the
                 calibration method actually used — see the module docstring.
    """

    statistic: float
    p_value: float
    reject: bool
    alpha: float
    n: int
    tau_hat: float
    B: int
    n_valid: int
    bootstrap: str = "parametric"


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


# ---------------------------------------------------------------------------
# Multiplier bootstrap (FR-10 round 2) — see the module docstring for the
# derivation. Everything below operates on pseudo-observations already in
# (0, 1); ``u, v`` is always the observed sample the linearisation is built
# from, ``uq, vq`` the (possibly different) points it is evaluated at.
# ---------------------------------------------------------------------------

def _empirical_copula_partials(
    u: np.ndarray, v: np.ndarray, uq: np.ndarray, vq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Plug-in estimates of ``∂C_n/∂u`` and ``∂C_n/∂v`` at ``(uq, vq)``.

    Central finite difference with bandwidth ``h = min(0.5, n^{-1/2})``
    (Rémillard & Scaillet 2009; Kojadinovic & Yan 2011 use the same order for
    this plug-in), one-sided within ``h`` of the boundary by construction:
    clipping the shifted argument to ``[0, 1]`` and dividing by the *actual*
    (possibly shrunk) step reduces to a one-sided difference there without a
    separate branch, exactly the trick :mod:`pmcprg.copulas._stderr` uses for
    its own central differences near a boundary.
    """
    n = u.size
    h = min(0.5, 1.0 / math.sqrt(n))
    u_hi, u_lo = np.minimum(uq + h, 1.0), np.maximum(uq - h, 0.0)
    v_hi, v_lo = np.minimum(vq + h, 1.0), np.maximum(vq - h, 0.0)
    dC_du = (_empirical_copula_at(u_hi, vq, u, v)
             - _empirical_copula_at(u_lo, vq, u, v)) / (u_hi - u_lo)
    dC_dv = (_empirical_copula_at(uq, v_hi, u, v)
             - _empirical_copula_at(uq, v_lo, u, v)) / (v_hi - v_lo)
    return dC_du, dC_dv


def _multiplier_kernel(
    u: np.ndarray, v: np.ndarray, uq: np.ndarray, vq: np.ndarray,
) -> np.ndarray:
    """``K`` such that ``α_n^ξ(uq, vq) = K @ ξ / √n`` (module docstring).

    Returns an ``(m, n)`` matrix, ``m = uq.size``. Built once per observed
    sample; every bootstrap replicate then only needs a fresh ``ξ`` and one
    matrix–vector product.
    """
    Cn_q = _empirical_copula_at(uq, vq, u, v)                     # (m,)
    dC_du, dC_dv = _empirical_copula_partials(u, v, uq, vq)       # (m,)
    le_u = (u[None, :] <= uq[:, None]).astype(float)              # (m, n)
    le_v = (v[None, :] <= vq[:, None]).astype(float)              # (m, n)
    indicator = le_u * le_v
    K = (indicator - Cn_q[:, None]
         - dC_du[:, None] * (le_u - uq[:, None])
         - dC_dv[:, None] * (le_v - vq[:, None]))
    return K


def _draw_multipliers(n: int, B: int, rng: np.random.Generator, law: str) -> np.ndarray:
    """``(n, B)`` i.i.d. mean-0, variance-1 multipliers."""
    if law == "normal":
        return rng.standard_normal(size=(n, B))
    if law == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(n, B))
    raise ValueError(f"multiplier must be one of {MULTIPLIER_LAWS}, got {law!r}.")


def _multiplier_bootstrap_draws(
    u: np.ndarray, v: np.ndarray, *, B: int, seed: int, multiplier: str,
) -> np.ndarray:
    """``T_n^ξ`` for ``B`` multiplier replicates, vectorised over all of them at once.

    Cost: one ``(2n × n)`` kernel built once (``O(n²)``, the same order as the
    observed statistic itself), then one ``(2n × n) @ (n × B)`` matrix
    product for all ``B`` replicates together — no resampling, no rank
    recomputation, no refit (contrast :func:`radial_symmetry_test`'s
    parametric-bootstrap loop, which pays for all three, B times).
    """
    n = u.size
    uq = np.concatenate([u, 1.0 - u])
    vq = np.concatenate([v, 1.0 - v])
    K = _multiplier_kernel(u, v, uq, vq)               # (2n, n)
    rng = np.random.default_rng(seed)
    xi = _draw_multipliers(n, B, rng, multiplier)      # (n, B)
    alpha = (K @ xi) / math.sqrt(n)                    # (2n, B)
    diff = alpha[:n, :] - alpha[n:, :]                 # (n, B): √n D_n^ξ(û_i, v̂_i)
    return np.mean(diff ** 2, axis=0)                  # (B,): T_n^ξ


def radial_symmetry_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    B: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
    weights=None,
    bootstrap: str = "parametric",
    multiplier: str = "normal",
) -> RadialSymmetryResult:
    """Test H0: the copula of ``(x, y)`` is radially symmetric (FR-10).

    Parameters
    ----------
    x, y       : array-like, shape ``(n,)`` — one fully-observed sample of
                 pairs (raw data or pseudo-observations both work: ranks are
                 recomputed here either way).
    B          : bootstrap replicates. ``bootstrap='parametric'`` costs
                 ``O(B n²)`` with a per-replicate refit-free resample and
                 rerank; ``bootstrap='multiplier'`` costs the same ``O(n²)``
                 once plus one ``O(n² B)`` matrix product for *all* replicates
                 together — see the module docstring for why this is
                 dramatically faster in practice.
    seed       : bootstrap RNG seed.
    alpha      : significance level for :attr:`RadialSymmetryResult.reject`.
    weights    : must be ``None``. Kept as a keyword, not silently dropped, so
                 that a caller passing ICE posteriors ``ξ`` gets an explicit
                 error rather than a silently-wrong answer — see "Weighting"
                 in the module docstring for why this statistic has no
                 defensible weighted form here.
    bootstrap  : ``'parametric'`` (default, unchanged from FR-10 round 1) or
                 ``'multiplier'`` (FR-10 round 2, see the module docstring).
    multiplier : ``'normal'`` (default) or ``'rademacher'`` — the i.i.d.
                 mean-0, variance-1 law of the multiplier bootstrap's ``ξ_i``.
                 Ignored when ``bootstrap='parametric'``.

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
    if bootstrap not in BOOTSTRAP_METHODS:
        raise ValueError(f"bootstrap must be one of {BOOTSTRAP_METHODS}, got {bootstrap!r}.")
    if bootstrap == "multiplier" and multiplier not in MULTIPLIER_LAWS:
        raise ValueError(f"multiplier must be one of {MULTIPLIER_LAWS}, got {multiplier!r}.")

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
            bootstrap=bootstrap,
        )

    u, v = _pseudo_obs(x, y)
    stat = radial_symmetry_statistic(u, v)

    tau_hat, _ = kendalltau(x, y)
    tau_hat = float(tau_hat) if np.isfinite(tau_hat) else 0.0

    if bootstrap == "multiplier":
        draws = _multiplier_bootstrap_draws(u, v, B=int(B), seed=seed, multiplier=multiplier)
        draws = draws[np.isfinite(draws)]
    else:
        surrogate = _fit_gaussian_surrogate(tau_hat)
        rng = np.random.default_rng(seed)
        raw_draws = []
        for _ in range(int(B)):
            xb, yb = surrogate.sample(n, seed=int(rng.integers(0, 2**31 - 1))).T
            ub, vb = _pseudo_obs(xb, yb)
            tb = radial_symmetry_statistic(ub, vb)
            if np.isfinite(tb):
                raw_draws.append(tb)
        draws = np.asarray(raw_draws, dtype=float)

    p_value = (
        float((1 + np.sum(draws >= stat)) / (draws.size + 1))
        if draws.size and np.isfinite(stat) else float("nan")
    )
    reject = bool(np.isfinite(p_value) and p_value < alpha)

    logger.debug(
        "radial_symmetry_test: n=%d tau_hat=%.4f T_n=%.4f p=%.4f alpha=%.3f reject=%s "
        "bootstrap=%s",
        n, tau_hat, stat, p_value, alpha, reject, bootstrap,
    )
    return RadialSymmetryResult(
        statistic=float(stat), p_value=p_value, reject=reject, alpha=float(alpha),
        n=int(n), tau_hat=tau_hat, B=int(B), n_valid=int(draws.size), bootstrap=bootstrap,
    )
