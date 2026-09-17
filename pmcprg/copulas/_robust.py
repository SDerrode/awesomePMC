"""
pmcprg.copulas._robust — density-power-divergence estimation of a copula
parameter, outside ICE (audit AUDIT_COPULES FR-7 c).

Standalone and additive: nothing here is called by :meth:`CopulaVirt.fit`,
:mod:`pmcprg.copulas._fit` or the ICE M-step. It offers a second estimator,
next to the pseudo-MLE and the τ-inversion of :mod:`pmcprg.copulas._stderr`,
that downweights outlying pairs instead of letting them dominate the score.

The objective
-------------
Basu, Harris, Hjort & Jones (1998) measure the discrepancy between the true
density g and a model f_θ by the **density power divergence** with tuning
constant α ≥ 0::

    d_α(g, f_θ) = ∫ { f_θ^{1+α} − (1 + 1/α) g f_θ^α + (1/α) g^{1+α} }

The last term does not involve θ, and ``∫ g f_θ^α`` is ``E_g[f_θ(X)^α]``, so
minimising ``d_α`` over θ is minimising its empirical version. For a copula
density ``c_θ`` on the unit square, with (weighted) pseudo-observations
``(û_i, v̂_i)`` and normalised weights ``w̄_i = w_i / Σw``, that is

.. math::

    H_n(θ; α) = \\underbrace{\\int_0^1\\!\\!\\int_0^1 c_θ(u,v)^{1+α}\\,du\\,dv}_{
                 I(θ, α)}
              - \\Bigl(1 + \\tfrac1α\\Bigr) \\sum_i \\bar w_i\\, c_θ(û_i, v̂_i)^α ,
    \\qquad α > 0,

and, at ``α = 0``, the limit ``H_n(θ; 0) = −Σ_i w̄_i log c_θ(û_i, v̂_i)`` — the
negative mean pseudo-log-likelihood, so **α = 0 is exactly the pseudo-MLE**
this package already fits (verified numerically, not only by the limit
argument: ``test_fr7_robust_options.py``). Writing ``c^α = 1 + α log c +
O(α²)`` and ``∫c^{1+α} = 1 + α ∫ c log c + O(α²)`` gives
``H_n(θ; α) = −α·[mean log c] + O(α²)`` up to θ-free terms, which is the same
argument.

What α buys: a point where the model density is small contributes ``c^α → 0``
instead of ``log c → −∞``. The estimating equation
``Σ w̄_i c^α u_θ − ∫ c^{1+α} u_θ = 0`` (``u_θ = ∂ log c/∂θ``) weights each
observation's score by ``c(û_i, v̂_i)^α``, so a pair the model finds very
unlikely is downweighted rather than allowed to pull θ̂ towards it. α = 0
recovers the unbounded-influence MLE; larger α trades efficiency for
robustness.

The integral ``I(θ, α)`` — the crux
-----------------------------------
``I(θ, α) = ∫∫ c_θ^{1+α}`` has to be evaluated at every trial θ, and it is
the only part of the objective that is not a finite sum. It is **1** for
every θ at α = 0 and grows with both α and the strength of dependence.

*Closed form, where the family allows it.* For the Gaussian copula, writing
``c = f_ρ/(φφ)`` with ``f_ρ`` the bivariate normal density and substituting
``u = Φ(x)``, ``v = Φ(y)``::

    I = ∫∫ f_ρ^{1+α} (φ(x)φ(y))^{-α} dx dy
      = (2π)^{-1} |Σ|^{-(1+α)/2} ∫ exp(−½ z'A z) dz ,   A = (1+α)Σ⁻¹ − αI ,

and with ``|Σ| = 1 − ρ²`` one gets ``|A| = (1 − α²ρ²)/(1 − ρ²)``, hence

.. math::  I(ρ, α) = (1 - ρ^2)^{-α/2}\\,(1 - α^2ρ^2)^{-1/2},
           \\qquad α|ρ| < 1 .

This is derived here, not quoted, and it is what the quadrature below is
measured against — an exact reference at every (ρ, α), which no
self-consistency check could provide. :func:`integral_c_power` uses it for
:class:`CopulaGaussian` (and for the Student copula's Gaussian member only in
the sense that Student is not supported at all here — see "Limits").

*Quadrature, otherwise.* The substitution ``s = logit u``, ``t = logit v``
maps the unit square to ℝ², with ``du dv = u(1−u)·v(1−v) ds dt``. Copula
densities blow up at the corners but the Jacobian vanishes there, and for
every family in this package the transformed integrand decays exponentially
in ``|s|, |t|``: for Clayton on the lower diagonal ``c ~ C/r`` with
``r = u = v``, so the integrand behaves like ``r^{1−α} = e^{(1−α)s}``. A
**composite Gauss–Legendre** rule with geometrically widening panels
(``|s|`` edges at 0, ¼, ½, 1, 2, 4, 8, 16, 32 and the representable limit
``logit(1 − 2⁻⁵²) = 36.04``, 12 nodes per panel — 216 nodes per axis, 46 656
grid points) resolves both the O(1) scale near the middle and the
exponential tails. Everything is summed in log space with a
maximum-subtraction, so a corner where ``c^{1+α}`` overflows in linear space
does not poison the total.

*Measured accuracy.* Relative error, against the Gaussian closed form for
the Gaussian copula and against a refined design (edges from ⅛, 18 nodes per
panel — 360 nodes per axis) for the others
(``pmcprg/tests/test_fr7_robust_options.py``):

============================  =======  =======  =======  =======
family, τ                      α=0.1    α=0.25   α=0.5    α=0.75
============================  =======  =======  =======  =======
Gaussian τ = 0.5              5.1e-15  9.6e-14  2.1e-11  7.9e-09
Gaussian τ = 0.7              1.3e-14  8.8e-13  1.3e-09  2.4e-06
Gaussian τ = 0.9              7.5e-08  6.4e-07  1.7e-05  4.6e-04
Frank/Plackett/AMH/FGM        <1e-15   <1e-15   <1e-15   <1e-15
Clayton  τ = 0.4              2.7e-09  2.6e-08  6.1e-06  7.3e-04
Gumbel   τ = 0.4              3.0e-09  2.2e-08  2.8e-06  3.1e-04
Gumbel   τ = 0.7              8.1e-06  3.9e-05  5.5e-04  1.0e-02
Clayton  τ = 0.7              1.6e-04  5.6e-04  4.8e-03  4.8e-02
Clayton  τ = 0.85             6.9e-03  1.5e-02  5.6e-02  2.0e-01
============================  =======  =======  =======  =======

Two honest limits follow, and both are reported rather than hidden:

* accuracy degrades with **strong tail dependence**: at τ = 0.85 the Clayton
  density concentrates in a band of width ≈ 1/θ = 0.09 around the diagonal in
  (s, t), which 216 nodes per axis do not resolve — the measured relative
  error is 6.9e-3 (α = 0.1) to 5.6e-2 (α = 0.5). Every :class:`DPDFit`
  therefore carries ``integral_rel_error``, the discrepancy between the
  default and the refined design at the fitted parameter, so a caller can
  see when the quadrature, not the data, is the binding constraint;
* accuracy degrades as **α → 1**: ``∫∫ c^{1+α}`` diverges (or nearly does)
  for a tail-dependent family as α → 1 — with ``c ~ C/r`` on the diagonal the
  corner contribution is ``∫ r^{-α} dr``. :data:`DPD_ALPHAS` therefore stops
  at 0.75 and :func:`dpd_fit` refuses α > 1. The same argument applied to
  ``K_α``'s ``∫∫ c^{1+2α}`` halves the usable α range for the *variance*:
  see "Choosing α" below.

Choosing α from the data
------------------------
:func:`select_alpha` minimises the empirical mean-squared-error criterion of
Warwick & Jones (2005), as used and extended by Ghosh & Basu (2015)::

    Ĥ(α) = (τ̂_α − τ̂_P)²  +  V̂_α / n_eff

— a squared-bias proxy, the distance from a **pilot** estimate, plus the
estimated asymptotic variance of τ̂_α. ``V̂_α = K_α / J_α²`` is the
Basu et al. (1998) sandwich under the model, evaluated on the same quadrature
grid::

    J_α = ∫ u_τ² c^{1+α},   ξ_α = ∫ u_τ c^{1+α},   K_α = ∫ u_τ² c^{1+2α} − ξ_α²,

with ``u_τ = ∂ log c/∂τ`` by a central difference. At α = 0 this is the
inverse Fisher information, as it must be. On clean data every τ̂_α is close
to the pilot, so the variance term decides and a small α is chosen; under
contamination the small-α estimates are far from the robust pilot and the
bias term decides.

``K_α`` integrates ``c^{1+2α}``, so it exists over **half** the α range the
objective does. For a family with tail dependence (``c ~ C/r`` on a corner
diagonal, so the corner contributes ``∫ r^{-2α} dr``) it already diverges at
α = 0.5 — and a quadrature then returns whatever its truncation dictates.
:func:`_dpd_variance` tests for this by recomputing ``V̂_α`` with ``|logit u|``
capped at 24 instead of 36.04 and reporting NaN when the two differ by more
than 1 %. Measured (n = 1000, τ = 0.4, relative change in brackets):
Clayton/Gumbel/Joe/A12 give a variance at α ≤ 0.25 (≤ 1.2e-4) and NaN from
α = 0.5 (0.39-0.50); Frank and Plackett agree to 4e-10 over the whole grid;
the Gaussian copula is finite to α = 0.5 (2.1e-4) and NaN at α = 0.75
(4.7e-2), where ``∫ u_τ² c^{2.5}`` is finite but so flat that its value would
be set by the truncation. The criterion is therefore ``+inf`` at those α:
**for a tail-dependent family α̂ is in practice chosen from {0, 0.05, 0.1,
0.25}**, and this is a limitation of the variance term, not of the estimator
— ``τ̂_α`` at α = 0.5 is computed and returned as usual.

Two honest caveats. **(1)** Ghosh & Basu (2015) is not reachable offline, so
what is implemented is the Warwick–Jones criterion with an explicitly
documented pilot rather than a word-for-word reconstruction of their
pilot-free refinement — the same posture the FR-9 and FR-10 rounds took for
unreachable papers. The pilot is ``τ̂`` at ``pilot_alpha`` (default 0.5, the
usual robust default of the DPD literature); the criterion is therefore not
pilot-free, and a different pilot can move α̂ by one grid step. **(2)**
``V̂_α`` is the **known-margin** DPD variance. Rank-based pseudo-observations
add the margin-estimation terms that
:func:`pmcprg.copulas._stderr.standard_errors` applies to the pseudo-MLE
sandwich; they are not included here. Since the criterion only *ranks* α
values and the missing correction inflates every ``V̂_α`` in the same
direction, the ranking is affected much less than the level — but the
variance term is not an honest standard error, and is not reported as one.

Limits
------
* **One-parameter families only.** The DPD fit is a 1-D bounded Brent search
  over the family's padded τ-range, the same search
  :meth:`CopulaVirt.fit(method='mle')` runs. Student and BB1 (and the other
  two-parameter families) raise :class:`NotImplementedError`: their joint
  optimiser lives in :mod:`pmcprg.copulas._fit`, which this module
  deliberately does not touch.
* The observations are assumed **i.i.d.**, as everywhere outside FR-5.
* ``weights`` are frequency weights, the package's convention (audit K-4),
  and only enter the finite sum: the integral does not depend on them.

References
----------
* Basu, A., Harris, I. R., Hjort, N. L. & Jones, M. C. (1998). Robust and
  efficient estimation by minimising a density power divergence.
  *Biometrika* 85(3), 549–559. doi:10.1093/biomet/85.3.549
* Ghosh, A. & Basu, A. (2015). Robust estimation for non-homogeneous data and
  the selection of the optimal tuning parameter: the density power divergence
  approach. *J. Appl. Stat.* 42(9), 2056–2072.
  doi:10.1080/02664763.2015.1016901
* Warwick, J. & Jones, M. C. (2005). Choosing a robustness tuning parameter.
  *J. Stat. Comput. Simul.* 75(7), 581–588. doi:10.1080/00949650412331299120
* Croux, C. & Dehon, C. (2010). Influence functions of the Spearman and
  Kendall correlation measures. *Stat. Methods Appl.* 19(4), 497–515.
  doi:10.1007/s10260-010-0142-z — the companion diagnostic of FR-7 (b),
  :func:`pmcprg.copulas._stderr.mle_tau_discrepancy_test`.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DPDAlphaSelection",
    "DPDFit",
    "DPD_ALPHAS",
    "dpd_fit",
    "dpd_objective",
    "integral_c_power",
    "select_alpha",
]

# Default α grid of :func:`select_alpha`. It stops at 0.75: ∫∫ c^{1+α} is
# divergent or nearly so at α = 1 for a tail-dependent family (module
# docstring, "The integral").
DPD_ALPHAS: tuple[float, ...] = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75)

# Largest α :func:`dpd_fit` accepts.
_ALPHA_MAX: float = 1.0

# Composite Gauss-Legendre design in logit coordinates: panel edges
# 0, inner, inner·ratio, ... up to the representable limit, mirrored to
# negative s, with ``nodes`` Gauss-Legendre points per panel.
# ``(inner, ratio, nodes)`` — default and refined (the second is only used to
# measure the first: ``DPDFit.integral_rel_error``).
_QUAD_DEFAULT: tuple[float, float, int] = (0.25, 2.0, 12)
_QUAD_REFINED: tuple[float, float, int] = (0.125, 2.0, 18)

# Objective value returned where the density cannot be evaluated — the same
# order of magnitude as the likelihood fits' own penalty, so a bounded
# optimiser leaves such a region rather than stalling on it.
_FAIL_PENALTY: float = 1e12

# Divergence test of the asymptotic DPD variance (see :func:`_dpd_variance`):
# the cap on |logit u| the test substitutes for the representable 36.04, and
# the largest relative change in V̂ at which V̂ is still reported.
_QUAD_TRUNCATED_LIMIT: float = 24.0
_VARIANCE_REL_TOL: float = 1e-2


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DPDFit:
    """Minimum-density-power-divergence estimate of a one-parameter family.

    Fields
    ------
    family     : class name.
    alpha      : the tuning constant used (0 = pseudo-MLE).
    copula     : the fitted instance (``cls(tau_k=tau_k)``).
    tau_k      : τ̂_α.
    theta      : the family's native parameter at τ̂_α.
    objective  : ``H_n(τ̂_α; α)`` (module docstring) — the minimised value;
                 comparable across τ at fixed α, **not** across α.
    integral   : ``I(τ̂_α, α) = ∫∫ c^{1+α}`` at the estimate; 1.0 at α = 0.
    integral_rel_error : ``|I_refined − I_default| / I_refined`` at the
                 estimate — the quadrature's own error estimate, 0.0 at α = 0
                 (where the integral is exactly 1) and 0.0 for the Gaussian
                 copula (closed form). Compare it with ``se_hint`` before
                 trusting the last digits of ``tau_k``.
    variance   : ``V̂_α = K_α/J_α²``, the known-margin asymptotic variance of
                 ``√n_eff (τ̂_α − τ)`` under the model (module docstring,
                 "Choosing α"); **NaN** when the two quadrature designs
                 disagree, which is what a divergent ``∫∫c^{1+2α}`` looks like
                 (:func:`_dpd_variance`) — the estimate ``tau_k`` itself is
                 unaffected.
    variance_rel_error : the relative disagreement behind that decision.
    se_hint    : ``sqrt(variance / n_eff)`` — a *hint*, not a standard error:
                 the rank-margin correction is not included (module docstring).
    converged  : the bounded Brent search reported success and the objective
                 is finite at the estimate.
    n_obs, n_eff : positive-weight observations and Σ w.
    """

    family: str
    alpha: float
    copula: object
    tau_k: float
    theta: float
    objective: float
    integral: float
    integral_rel_error: float
    variance: float
    variance_rel_error: float
    se_hint: float
    converged: bool
    n_obs: int
    n_eff: float

    def __repr__(self) -> str:
        return (f"DPDFit({self.family}, alpha={self.alpha:g}, tau_k={self.tau_k:.4g}, "
                f"theta={self.theta:.4g}, n_eff={self.n_eff:g}"
                f"{'' if self.converged else ', not converged'})")


@dataclass(frozen=True)
class DPDAlphaSelection:
    """Data-driven choice of the DPD tuning constant (module docstring).

    Fields
    ------
    family      : class name.
    alpha       : the selected α — the minimiser of ``criterion``.
    fit         : the :class:`DPDFit` at ``alpha``.
    alphas      : the grid searched, in the order of the arrays below.
    tau_k       : τ̂_α on the grid.
    variance    : V̂_α on the grid.
    criterion   : ``(τ̂_α − τ̂_P)² + V̂_α/n_eff`` on the grid.
    pilot_alpha : the α of the pilot estimate τ̂_P.
    pilot_tau   : τ̂_P itself.
    fits        : every :class:`DPDFit` computed, keyed by α.
    n_obs, n_eff: positive-weight observations and Σ w.
    """

    family: str
    alpha: float
    fit: DPDFit
    alphas: tuple
    tau_k: tuple
    variance: tuple
    criterion: tuple
    pilot_alpha: float
    pilot_tau: float
    fits: dict
    n_obs: int
    n_eff: float

    def __repr__(self) -> str:
        return (f"DPDAlphaSelection({self.family}, alpha={self.alpha:g}, "
                f"tau_k={self.fit.tau_k:.4g}, pilot_alpha={self.pilot_alpha:g}, "
                f"n_eff={self.n_eff:g})")


# ---------------------------------------------------------------------------
# Quadrature grid in logit coordinates
# ---------------------------------------------------------------------------

def _representable_logit_limit() -> float:
    """``logit(1 − 2⁻⁵²) = 36.0436…`` — the widest useful range in double precision."""
    from pmcprg.numerics import ONE_MINUS_EPS

    return float(math.log(ONE_MINUS_EPS) - math.log1p(-ONE_MINUS_EPS))


@lru_cache(maxsize=16)
def _quad_grid(inner: float, ratio: float, n_per: int,
               limit: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(u, log_w)`` of the composite Gauss-Legendre rule (module docstring).

    ``u`` are the nodes in (0, 1) and ``log_w`` the logarithm of the
    quadrature weight *including* the Jacobian ``u(1 − u)`` of ``s = logit u``.
    Both are computed from ``s`` so that ``log u(1 − u) = −softplus(s) −
    softplus(−s)`` stays exact where ``u`` itself rounds to 0 or 1. ``limit``
    caps ``|s|``; the default is the widest double precision represents.
    """
    from scipy.special import expit, roots_legendre

    from pmcprg.numerics import EPS, ONE_MINUS_EPS

    limit = _representable_logit_limit() if limit is None else float(limit)
    edges = [0.0, inner]
    while edges[-1] < limit:
        edges.append(min(edges[-1] * ratio, limit))
    edges = [-e for e in reversed(edges[1:])] + edges

    x, gw = roots_legendre(n_per)
    s_all, w_all = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        half = 0.5 * (hi - lo)
        s_all.append(0.5 * (hi + lo) + half * x)
        w_all.append(half * gw)
    s = np.concatenate(s_all)
    gw = np.concatenate(w_all)
    # log[u(1−u)] = −log(1 + e^{s}) − log(1 + e^{−s}), exact for every s.
    log_jac = -np.logaddexp(0.0, s) - np.logaddexp(0.0, -s)
    u = np.clip(expit(s), EPS, ONE_MINUS_EPS)
    return u, np.log(gw) + log_jac


def _grid_log_density(copula, u: np.ndarray) -> np.ndarray:
    """``log c`` on the tensor grid ``u × u``, shape (m, m)."""
    m = u.size
    uu, vv = np.meshgrid(u, u, indexing="ij")
    ld = np.asarray(copula.logpdf_array(np.column_stack([uu.ravel(), vv.ravel()])), dtype=float)
    return ld.reshape(m, m)


def _grid_log_weight(log_w: np.ndarray) -> np.ndarray:
    return log_w[:, None] + log_w[None, :]


def _weighted_grid_sum(exponent: np.ndarray, factor: np.ndarray | None = None) -> float:
    """``Σ factor · exp(exponent)``, stabilised by subtracting the maximum.

    ``exponent`` holds ``(1 + α) log c + log w`` on the grid; ``factor`` an
    optional (signed) multiplier such as the score or its square. Non-finite
    entries of ``exponent`` are dropped: ``−∞`` is a density of exactly 0 and
    contributes nothing, and a NaN is a kernel failure at that node (logged by
    the caller), not a value to propagate through the whole integral.
    """
    good = np.isfinite(exponent)
    if not np.any(good):
        return 0.0
    e = exponent[good]
    top = float(e.max())
    vals = np.exp(e - top)
    if factor is not None:
        vals = vals * factor[good]
    return float(math.exp(top) * float(vals.sum()))


def _gaussian_integral_closed_form(rho: float, alpha: float) -> float:
    """``(1 − ρ²)^{−α/2} (1 − α²ρ²)^{−1/2}`` — module docstring, "Closed form"."""
    r2 = float(rho) * float(rho)
    ar2 = alpha * alpha * r2
    if not r2 < 1.0 or not ar2 < 1.0:
        return float("nan")
    return float((1.0 - r2) ** (-0.5 * alpha) / math.sqrt(1.0 - ar2))


def integral_c_power(copula, alpha: float, *, refined: bool = False,
                     closed_form: bool = True) -> float:
    """``I(θ, α) = ∫∫ c_θ(u, v)^{1+α} du dv`` over the unit square.

    Exactly 1.0 at ``alpha = 0``. For :class:`CopulaGaussian` the closed form
    derived in the module docstring is used unless ``closed_form=False``
    (which is how the tests measure the quadrature against it). Every other
    family goes through the composite Gauss-Legendre rule; ``refined=True``
    selects the finer design whose discrepancy with the default one is
    :attr:`DPDFit.integral_rel_error`.
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}.")
    if alpha == 0.0:
        return 1.0
    if closed_form and type(copula).__name__ == "CopulaGaussian":
        val = _gaussian_integral_closed_form(float(copula.theta), alpha)
        if np.isfinite(val):
            return val
    inner, ratio, n_per = _QUAD_REFINED if refined else _QUAD_DEFAULT
    u, log_w = _quad_grid(inner, ratio, n_per)
    ld = _grid_log_density(copula, u)
    if np.any(np.isnan(ld)):
        logger.warning("%s: the log-density is NaN at %d of %d quadrature nodes; those nodes "
                       "are dropped from the integral of c^(1+alpha).",
                       type(copula).__name__, int(np.isnan(ld).sum()), ld.size)
    return _weighted_grid_sum((1.0 + alpha) * ld + _grid_log_weight(log_w))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _prepare(uv, weights):
    """``(uv, w̄, n_obs, n_eff)`` with zero-weight rows dropped; ``w̄`` sums to 1."""
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (n, 2), got {uv.shape}.")
    if not np.all(np.isfinite(uv)) or np.any(uv <= 0.0) or np.any(uv >= 1.0):
        raise ValueError("pseudo-observations must lie strictly inside (0, 1).")
    if weights is None:
        w = np.ones(uv.shape[0])
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != uv.shape[0]:
            raise ValueError(f"weights ({w.shape[0]}) and uv ({uv.shape[0]}) differ in length.")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        keep = w > 0.0
        uv, w = uv[keep], w[keep]
    if uv.shape[0] < 4:
        raise ValueError(f"at least 4 positive-weight observations are required, "
                         f"got {uv.shape[0]}.")
    n_eff = float(w.sum())
    if not n_eff > 0.0:
        raise ValueError("weights must not all be zero.")
    return uv, w / n_eff, int(uv.shape[0]), n_eff


def dpd_objective(copula, uv, alpha: float, weights=None) -> float:
    """``H_n(θ; α)`` of the module docstring — the quantity :func:`dpd_fit` minimises.

    ``alpha = 0`` gives ``−Σ w̄_i log c``, the negative *mean* weighted
    pseudo-log-likelihood (so it is the pseudo-MLE objective up to the
    positive factor Σw, which does not move the minimiser).

    Returns ``+inf`` when α = 0 and the log-density is not finite at a
    positive-weight observation — the convention of
    :func:`pmcprg.copulas._fit._weighted_log_density_sum`. For α > 0 a density
    of exactly 0 is finite in this objective (it contributes ``0^α = 0``),
    which is precisely the robustness α buys.
    """
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError(f"alpha must be >= 0, got {alpha!r}.")
    uv, wbar, _, _ = _prepare(uv, weights)
    ld = np.asarray(copula.logpdf_array(uv), dtype=float)
    if alpha == 0.0:
        if not np.all(np.isfinite(ld)):
            return float("inf")
        return float(-np.dot(wbar, ld))
    with np.errstate(over="ignore"):
        ca = np.exp(alpha * ld)
    ca = np.where(np.isnan(ca), 0.0, ca)
    integ = integral_c_power(copula, alpha)
    return float(integ - (1.0 + 1.0 / alpha) * float(np.dot(wbar, ca)))


# ---------------------------------------------------------------------------
# Asymptotic variance under the model (Basu et al. 1998)
# ---------------------------------------------------------------------------

def _tau_score_step(cls, entry, tau: float) -> float:
    """Central-difference step in τ that stays inside the family's usable range."""
    from pmcprg.copulas._base import padded_tau_range

    lo, hi = padded_tau_range(*(float(t) for t in entry.value.TAU_MIN_MAX))
    bounds = cls.reachable_tau_bounds()
    if bounds is not None:
        lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
    d = min(tau - lo, hi - tau)
    return 1e-4 * min(d, 1.0) if d > 0.0 else 0.0


def _variance_on_grid(cls, tau: float, alpha: float, h: float,
                      limit: float | None = None) -> float:
    """``K_α/J_α²`` on one quadrature design (``1/J_0`` at α = 0)."""
    inner, ratio, n_per = _QUAD_DEFAULT
    u, log_w = _quad_grid(inner, ratio, n_per, limit)
    try:
        ld = _grid_log_density(cls(tau_k=float(tau)), u)
        ld_p = _grid_log_density(cls(tau_k=float(tau + h)), u)
        ld_m = _grid_log_density(cls(tau_k=float(tau - h)), u)
    except Exception as exc:                                  # pragma: no cover - guard
        logger.warning("%s: the DPD variance could not be evaluated at tau=%.6g (%s).",
                       cls.__name__, tau, exc)
        return float("nan")
    score = (ld_p - ld_m) / (2.0 * h)
    score = np.where(np.isfinite(score), score, 0.0)
    gw = _grid_log_weight(log_w)
    j = _weighted_grid_sum((1.0 + alpha) * ld + gw, score * score)
    if not j > 0.0:
        return float("nan")
    if alpha == 0.0:
        return float(1.0 / j)
    xi = _weighted_grid_sum((1.0 + alpha) * ld + gw, score)
    k = _weighted_grid_sum((1.0 + 2.0 * alpha) * ld + gw, score * score) - xi * xi
    if not k > 0.0:
        return float("nan")
    return float(k / (j * j))


def _dpd_variance(cls, entry, tau: float, alpha: float) -> tuple[float, float]:
    """``(V̂_α, rel)`` — the variance of the module docstring, and a divergence test.

    ``K_α`` needs ``∫∫ c^{1+2α}``, which converges over **half** the α range
    the objective's own ``∫∫ c^{1+α}`` does. For a family with tail dependence
    (``c ~ C/r`` along a corner diagonal, so the corner contributes
    ``∫ r^{-2α} dr``) that integral already diverges at ``α = 0.5``: the
    Basu et al. (1998) asymptotic variance simply does not exist there, and
    any number a quadrature returns is an artefact of where the grid stops.
    The estimate ``τ̂_α`` itself is unaffected — only its variance is.

    The test is therefore a **range** test, not a node-density one: ``V̂_α`` is
    recomputed with ``|logit u|`` capped at :data:`_QUAD_TRUNCATED_LIMIT`
    instead of the representable 36.04. A convergent integral barely notices
    (the omitted tail is exponentially small); a divergent one is dominated by
    exactly the region removed. ``rel`` is the relative difference, and a
    ``rel`` above :data:`_VARIANCE_REL_TOL` makes the variance NaN, with a
    WARNING. Node density is covered separately by
    :attr:`DPDFit.integral_rel_error`.
    """
    h = _tau_score_step(cls, entry, tau)
    if not h > 0.0:
        return float("nan"), float("nan")
    full = _variance_on_grid(cls, tau, alpha, h)
    if alpha == 0.0:
        return full, 0.0
    cut = _variance_on_grid(cls, tau, alpha, h, limit=_QUAD_TRUNCATED_LIMIT)
    if not (np.isfinite(full) and np.isfinite(cut)) or full <= 0.0:
        return float("nan"), float("nan")
    rel = abs(full - cut) / full
    if rel > _VARIANCE_REL_TOL:
        logger.warning(
            "%s: at alpha=%g, capping |logit u| at %g instead of %.4g changes the DPD "
            "variance from %.6g to %.6g (relative difference %.3g): the integral of "
            "c^(1+2*alpha) behind K_alpha does not converge for this family at this alpha, "
            "so the asymptotic variance is reported as NaN.", cls.__name__, alpha,
            _QUAD_TRUNCATED_LIMIT, _representable_logit_limit(), full, cut, rel)
        return float("nan"), float(rel)
    return float(full), float(rel)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _one_parameter_entry(family):
    from pmcprg.copulas._base import CopulaEnum

    cls = family if isinstance(family, type) else type(family)
    entry = None
    for e in CopulaEnum:
        if e.value.CLASS_NAME == cls.__name__:
            entry = e
            break
    if entry is None:
        raise ValueError(f"{cls.__name__!r} is not registered in CopulaEnum.")
    if len(entry.value.PARAMETERS_SET_NAME) != 1:
        raise NotImplementedError(
            f"the density-power-divergence estimator is implemented for one-parameter "
            f"families only; {cls.__name__} has "
            f"{entry.value.PARAMETERS_SET_NAME} (module docstring, 'Limits').")
    a, b = (float(t) for t in entry.value.TAU_MIN_MAX)
    if b - a < 1e-8:
        raise ValueError(f"{cls.__name__} has no free parameter.")
    return cls, entry


def dpd_fit(family, uv, alpha: float, weights=None) -> DPDFit:
    """Minimum-DPD estimate of a one-parameter family's τ (module docstring).

    The objective :func:`dpd_objective` is minimised by the same bounded Brent
    search over the padded τ-range that :meth:`CopulaVirt.fit(method='mle')`
    uses, so ``alpha = 0`` reproduces the pseudo-MLE to the optimiser's
    tolerance.

    Parameters
    ----------
    family  : a one-parameter ``CopulaVirt`` subclass (or an instance).
    uv      : (n, 2) pseudo-observations in (0, 1)².
    alpha   : tuning constant in [0, 1]; 0 is the pseudo-MLE. Above ~0.75 the
              integral ``∫∫c^{1+α}`` becomes hard for tail-dependent families
              (module docstring); above 1 the call is refused.
    weights : optional (n,) non-negative frequency weights.

    Returns
    -------
    DPDFit
    """
    from scipy.optimize import minimize_scalar

    from pmcprg.copulas._base import padded_tau_range

    alpha = float(alpha)
    if not 0.0 <= alpha <= _ALPHA_MAX:
        raise ValueError(f"alpha must lie in [0, {_ALPHA_MAX:g}], got {alpha!r} — "
                         "the integral of c^(1+alpha) diverges for a tail-dependent "
                         "family as alpha approaches 1 (module docstring).")
    cls, entry = _one_parameter_entry(family)
    uv, wbar, n_obs, n_eff = _prepare(uv, weights)
    lo, hi = padded_tau_range(*(float(t) for t in entry.value.TAU_MIN_MAX))

    def objective(tau: float) -> float:
        try:
            cop = cls(tau_k=entry.reachable_tau(float(tau)))
            val = dpd_objective(cop, uv, alpha, wbar)
        except Exception:
            return _FAIL_PENALTY
        return val if np.isfinite(val) else _FAIL_PENALTY

    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
    tau_hat = entry.reachable_tau(float(res.x))
    best = objective(tau_hat)
    converged = bool(getattr(res, "success", True)) and best < _FAIL_PENALTY
    if not converged:
        logger.warning("%s: the DPD objective (alpha=%g) is not finite at any tau the search "
                       "evaluated; returning tau = %.6g.", cls.__name__, alpha, tau_hat)
    cop = cls(tau_k=tau_hat)
    integ = integral_c_power(cop, alpha)
    if alpha == 0.0 or type(cop).__name__ == "CopulaGaussian":
        rel_err = 0.0
    else:
        fine = integral_c_power(cop, alpha, refined=True)
        rel_err = abs(fine - integ) / abs(fine) if fine != 0.0 else float("nan")
    var, var_rel = _dpd_variance(cls, entry, tau_hat, alpha)
    return DPDFit(
        family=cls.__name__, alpha=alpha, copula=cop, tau_k=float(tau_hat),
        theta=float(getattr(cop, "theta", math.nan)), objective=float(best),
        integral=float(integ), integral_rel_error=float(rel_err), variance=float(var),
        variance_rel_error=float(var_rel),
        se_hint=float(math.sqrt(var / n_eff)) if np.isfinite(var) and var > 0.0 else math.nan,
        converged=converged, n_obs=n_obs, n_eff=n_eff,
    )


def select_alpha(family, uv, weights=None, *, alphas=DPD_ALPHAS,
                 pilot_alpha: float = 0.5) -> DPDAlphaSelection:
    """Data-driven choice of α by the empirical-MSE criterion (module docstring).

    ``Ĥ(α) = (τ̂_α − τ̂_P)² + V̂_α/n_eff`` is minimised over ``alphas``, with
    the pilot ``τ̂_P`` the estimate at ``pilot_alpha``. Read the module
    docstring's two caveats before quoting α̂: the criterion is Warwick &
    Jones (2005) with an explicit pilot rather than a reconstruction of
    Ghosh & Basu (2015)'s pilot-free variant, and ``V̂_α`` ignores the
    rank-margin correction.

    Parameters
    ----------
    family      : a one-parameter ``CopulaVirt`` subclass (or an instance).
    uv          : (n, 2) pseudo-observations in (0, 1)².
    weights     : optional (n,) non-negative frequency weights.
    alphas      : the grid to search; ``pilot_alpha`` is added to it if absent.
    pilot_alpha : α of the pilot estimate (default 0.5).

    Returns
    -------
    DPDAlphaSelection
    """
    cls, _ = _one_parameter_entry(family)
    grid = sorted({float(a) for a in alphas} | {float(pilot_alpha)})
    fits = {a: dpd_fit(cls, uv, a, weights) for a in grid}
    pilot = fits[float(pilot_alpha)]
    n_eff = pilot.n_eff
    crit = []
    for a in grid:
        f = fits[a]
        v = f.variance
        if not np.isfinite(v):
            crit.append(float("inf"))
        else:
            crit.append((f.tau_k - pilot.tau_k) ** 2 + v / n_eff)
    if not np.any(np.isfinite(crit)):
        logger.warning("%s: the alpha-selection criterion is not finite anywhere on the grid "
                       "%s; falling back on the pilot alpha %g.", cls.__name__, grid,
                       pilot_alpha)
        best = float(pilot_alpha)
    else:
        best = grid[int(np.argmin(crit))]
    return DPDAlphaSelection(
        family=cls.__name__, alpha=float(best), fit=fits[best], alphas=tuple(grid),
        tau_k=tuple(fits[a].tau_k for a in grid),
        variance=tuple(fits[a].variance for a in grid), criterion=tuple(crit),
        pilot_alpha=float(pilot_alpha), pilot_tau=float(pilot.tau_k), fits=fits,
        n_obs=pilot.n_obs, n_eff=n_eff,
    )
