"""
pmcprg.copulas._stderr — standard errors of copula parameter estimates, outside ICE
(audit AUDIT_COPULES FR-4).

What is computed
----------------
:func:`standard_errors` returns the asymptotic covariance of an estimate
computed on **pseudo-observations** ``(û_i, v̂_i)`` with optional observation
weights ``w_i``, for the two estimators of :meth:`CopulaVirt.fit`:

``method='mle'`` — maximum pseudo-likelihood (Genest, Ghoudi & Rivest 1995).
    θ̂ solves Σ w_i φ(û_i, v̂_i; θ) = 0 with φ = ∂ log c / ∂θ. With rank-based
    pseudo-observations (``ranks=True``) the margins are estimated, and the
    score acquires one correction per margin (Genest, Ghoudi & Rivest 1995)::

        Σ = B⁻¹ Ω B⁻ᵀ / n_eff,      B = −E[∂φ/∂θᵀ],
        Ω = Var{φ(U, V) + W₁(U) + W₂(V)},
        W₁(x) = E[∂φ/∂u (U, V) · 1{x ≤ U}],   W₂(y) = E[∂φ/∂v (U, V) · 1{y ≤ V}],

    each expectation replaced by its weighted empirical version over the
    pseudo-observations: ``Ŵ₁(û_i) = Σ_j w_j ∂φ/∂u(û_j, v̂_j) 1{û_i ≤ û_j} / Σw``
    (O(n log n) by suffix sums), ``B̂ = −Σ w_i ∂φ/∂θᵀ(û_i, v̂_i) / Σw`` (the
    observed information, which stays valid under misspecification), and
    ``Ω̂`` the weighted covariance of ``φ + Ŵ₁ + Ŵ₂`` (divisor Σw).
    ``ranks=False`` drops W₁, W₂: the classical sandwich for margins that are
    known — or treated as known, as with parametric margins (whose own
    estimation error then needs the IFM Godambe matrix, not provided here).

``method='tau'`` — inversion of Kendall's τ.
    √n (τ̂ − τ) → N(0, 16 · Var{2C(U, V) − U − V}) (Hoeffding's projection of
    the U-statistic; Genest & Favre 2007; Kojadinovic & Yan 2010),
    estimated with the (weighted) empirical copula C_n at the pseudo-
    observations, then carried to θ by the delta method, dθ/dτ. At
    independence the estimate targets Kendall's exact null variance
    2(2n + 1) / (9n(n − 1)) (Kendall 1938), to O(n⁻²).

Derivatives
-----------
No family has analytic derivatives of its log-density. φ, ∂φ/∂θᵀ and the
mixed derivatives ∂φ/∂u, ∂φ/∂v are **central finite differences** of
``logpdf_array`` in an unconstrained working coordinate ψ — one in which
every step stays admissible — and ∂/∂u is taken in logit(u), so the step in u
is proportional to u(1 − u) and never leaves (0, 1):

* one-parameter families: ψ = logit of τ on its registered range [a, b]
  (2·atanh τ on (−1, 1), logit τ on (0, 1) — i.e. ≈ log θ near independence);
  Plackett, whose θ(τ) is a piecewise-linear table interpolation with no
  usable second derivative, in ψ = log θ with θ set directly;
* Student (τ, ν): ψ = (atanh τ, log(ν − 2));
* BB1 (θ, δ): ψ = (log θ, log(δ − 1)).

Steps are 10⁻⁴ in ψ and in logit u (truncation O(10⁻⁸), rounding
O(10⁻⁸)·|log c| for the second differences). The Hessian in ψ is freed of
its chain-rule gradient term, so the sandwich is the one in the coordinates
the fit optimised (τ; θ for Plackett; (τ, ν); (θ, δ)) even where the
optimiser stopped short of the root of the score. The covariance in ψ is carried
to the reported quantities — τ, the native parameters, and ν or δ — by the
delta method (analytic Jacobians; dθ/dτ of one-parameter families by a
central difference of the family's own τ → θ map).

Weights
-------
``weights`` are **frequency weights**: integer weights give exactly the
result of the data with each row repeated ``w_i`` times, and
``n_eff = Σ w`` (the package's convention, audit K-4). For fractional
weights ≤ 1 — posterior memberships — this n_eff overstates the variance
compared with the Kish size (Σw)²/Σw², and neither accounts for the
uncertainty of the memberships themselves (the missing information of the
latent states, FR-4 inside ICE: Oakes 1999). Zero-weight rows are dropped.

Boundaries
----------
At the edge of the parameter space the estimator is not asymptotically
normal: when the truth is on the boundary, half of its mass sits on the
boundary itself (Self & Liang 1987). A result is flagged
``at_boundary`` when τ̂ lies within two optimiser pads (``TAU_PAD_REL``) of an
end of its registered range — the independence end of Clayton, Gumbel,
Joe, their survival versions and the Cubic Section, the Fréchet bounds |τ| → 1,
the ends of FGM and AMH, τ = 1/3 for A12/A14 — when τ̂ sits on the end of
the τ its family reaches below the registered range (Frank ±0.994299 at
θ = ±700, Plackett ±0.9935 at θ = 10^{±6}; for Frank, when a difference step
would cross it — dθ/dτ is then one-sided inside), when BB1 has δ̂ = 1 (Clayton
limit) or θ̂ at its floor (Gumbel limit), or when Student's ν̂ reaches an end
of its fitting box (the upper end stands for the Gaussian limit ν → ∞).
:meth:`StandardErrors.ci` then returns NaN instead of a Wald interval, and
the ``'mle'`` standard errors are NaN too: the curvature of the likelihood at
a constrained stop is not the information of an interior optimum. The
``'tau'`` standard errors are kept — they are those of the *unclipped*
Kendall's τ̂, whose limit is regular; only the clipped estimate is not.
Testing a parameter *at* such a boundary by likelihood ratio uses the
mixture ½χ²₀ + ½χ²₁: :func:`independence_lr_test` does it for the
one-parameter families.

Two-parameter sub-model tests
------------------------------
:func:`submodel_lr_test` is the analogous likelihood-ratio test of a
**two**-parameter family's one-parameter sub-model — H0: the extra
parameter sits at the sub-model's value, H1: the full two-parameter family
— for the nestings this package registers: BB1 ⊃ {Clayton, Gumbel},
Student ⊃ Gauss, Tawn (types 1 and 2) ⊃ Gumbel, BB6 ⊃ {Joe, Gumbel} and
BB7 ⊃ {Clayton, Joe} and BB8 ⊃ Joe.
Each was verified against the family's own module docstring, not assumed:

* **BB1 → Clayton** (δ = 1) and **BB1 → Gumbel** (θ → 0, i.e. the floor
  ``CopulaBB1._THETA_FLOOR``): ``pmcprg.copulas.archimedean.bb1``'s module
  docstring states both limits explicitly (and documents a K-10 correction
  of an earlier, crossed statement of the θ/δ ↔ tail-dependence limits — a
  reminder that these nestings are exactly the kind of claim to re-derive,
  not transcribe). Both δ = 1 and θ = 0 are the *lower ends* of BB1's
  registered ranges δ ≥ 1 and θ > 0: **boundary** sub-models.
* **Student → Gauss** (ν → ∞): ``pmcprg.copulas.elliptical.student``'s
  docstring states it directly, and ν → ∞ is not reachable by a numerical
  optimiser — the joint fit already treats ν's fitting box
  (``EXTRA_PARAM_BOUNDS_BY_PARAM['df'] = (2.001, 100.0, 4.0)``) as standing
  in for it, exactly as this module's own boundary flag for standard errors
  does (module docstring above). The test below fits the Gaussian sub-model
  directly (closed form, not a limit of the Student log-likelihood) and
  compares it with BB1-style two-parameter MLE, treating ν = 100 (the box's
  upper end) as the boundary the ν̂ → ∞ limit sits at: **boundary**
  sub-model.
* **Tawn → Gumbel** (ψ = 1): ``pmcprg.copulas.extreme_value.tawn``'s
  docstring states ψ_u = ψ_v = 1 → Gumbel–Hougaard, and this package's Tawn
  types 1/2 fix the *other* weight at 1 and free ψ ∈ (0, 1] — so ψ = 1 is
  the *upper end* of the registered range, not an interior point:
  **boundary** sub-model. (Re-verified numerically, not just by citation of
  the FR-9 commit message that introduced Tawn: ``CopulaTawn1(psi=1.0)``
  reproduces ``CopulaGH``'s log-density to machine precision at matched τ —
  see ``pmcprg/tests/test_fr4_submodel_lr.py``.)
* **BB7 → Clayton** (θ = 1) and **BB7 → Joe** (δ → 0), FR-9's BB7 round:
  ``pmcprg.copulas.archimedean.bb7``'s module docstring derives both by hand
  from the generator ``φ = [1 − (1 − t)^θ]^{−δ} − 1`` — θ = 1 leaves
  ``t^{−δ} − 1``, Clayton's own generator, and ``φ/δ → −ln(1 − (1 − t)^θ)``,
  Joe's, as δ → 0 — and checks each at 120 digits (4.8·10⁻¹²² and an error
  exactly linear in δ). BB7 stores **θ**, not δ (τ is not monotone in θ at
  fixed δ — that module's "Parametrisation"), so in its own coordinates
  θ = 1 is the *attained* lower end of θ ≥ 1 and the Joe limit is the
  **open** upper end θ → θ_Joe(τ), where the recovered δ → 0; the
  constructor's floor δ = 10⁻¹⁴ stands in for it as Student's ν = 100 does
  for ν → ∞ (λ_L = 2^{−10¹⁴} is already 0 to machine precision there). Both
  are **boundary** sub-models.
* **BB6 → Joe** (δ = 1) and **BB6 → Gumbel** (θ = 1), FR-9's BB6 round:
  ``pmcprg.copulas.archimedean.bb6``'s module docstring derives both by
  hand from the generator ``φ = (−ln(1 − (1 − t)^θ))^δ`` — δ = 1 leaves
  Joe's generator, θ = 1 leaves ``(−ln t)^δ`` — and checks each numerically
  to 60 digits. Both are the *lower ends* of BB6's admissible θ ≥ 1, δ ≥ 1:
  **boundary** sub-models, and unlike Student's ν → ∞ both are attained, so
  no fitting-box end has to stand in for them (the joint fit's θ floor
  1 + 10⁻¹⁰ is only the resolution at which it reaches θ = 1).
* **BB8 → Joe** (δ = 1), FR-9's BB8 round, and **only** that one:
  ``pmcprg.copulas.archimedean.bb8``'s module docstring shows that
  ``(1 − δ)^θ = 0`` at δ = 1 makes the generator's denominator 1 and leaves
  Joe's generator *exactly* (no limit), δ = 1 being the **upper end** of
  δ ∈ (0, 1] — a boundary sub-model like Tawn's ψ = 1. BB8's other edge,
  θ = 1, gives ``φ = −ln t``: the independence copula for every δ, which has
  no free parameter to profile and belongs to
  :func:`independence_lr_test`. And despite the family's "Joe–Frank"
  nickname, Frank is *not* a sub-model at all — only the joint limit
  θ → ∞, δ → 0 at fixed θδ, which no parameter value realises (checked
  numerically in that module, not inferred from the name).

All these sub-models are therefore boundary cases of the *same* kind
:func:`independence_lr_test` already handles for one parameter: the
constrained (2-D) MLE, under H0, sits on the boundary of the admissible
extra-parameter box about half the time, so ``LR → ½χ²₀ + ½χ²₁`` (Self &
Liang 1987) — not the plain ``χ²₁`` that would apply were the sub-model
interior to the two-parameter family's range. :func:`submodel_lr_test`
therefore always reports the mixture for these three pairs (the boundary
is a fact of the parametrisation, not of the data, so it is not
recomputed per call — unlike the one-parameter ``independence_lr_test``,
whose H0 can be either interior or boundary depending on family).

MLE-vs-τ discrepancy diagnostic (FR-7 b)
----------------------------------------
:func:`mle_tau_discrepancy_test` is a **Hausman-style specification test**
(Hausman 1978) built on the two estimators above. Kendall's τ̂ has a bounded
influence function — one observation can move it by O(1/n) at most — while
the log-density score φ that defines the pseudo-MLE does not (Croux & Dehon
2010). Under correct specification both estimators are consistent for the
same τ, so their difference is O_p(n^{-1/2}) and centred at 0; under
contamination or misspecification the unbounded-influence estimator moves
and the rank-based one does not, so the difference grows.

The difference is taken on the **τ scale**, the one coordinate every
one-parameter family shares (and the one the τ-inversion estimator is
defined in). A statistic formed on the native θ scale is the same to first
order: both numerator and denominator carry the same dθ/dτ factor, which
cancels in the ratio, so ``θ̂`` values are reported for information only.

``Cov(τ̂_MLE, τ̂_τ)`` is **not** the Hausman "difference of variances".
That shortcut needs the first estimator to be efficient under H0, and the
rank-based pseudo-MLE is not semiparametrically efficient in general
(Genest & Werker 2002) — the shortcut can and does return negative
variances here. Instead both estimators are written as sums of their own
**influence functions on the same sample**, and the variance of the
difference is the (weighted) empirical variance of the difference of those
influence functions::

    a_i = (dτ/dψ) · B⁻¹ [φ_i + Ŵ₁(û_i) + Ŵ₂(v̂_i)]     (pseudo-MLE, above)
    b_i = 4 (ẑ_i − z̄),   ẑ_i = 2 C_n(û_i, v̂_i) − û_i − v̂_i   (Kendall's τ̂)
    Var(τ̂_MLE − τ̂_τ) = Var_w(a − b) / Σw

``b`` is the influence function behind the ``'tau'`` variance of this
module: ``Var(τ̂) = 16 Var{2C(U, V) − U − V}/n``. It is the influence
function of the V-statistic ``4 · mean(C_n) − 1``: for
``V_n = n⁻² ΣΣ 1{û_j ≤ û_i, v̂_j ≤ v̂_i}`` the delta method gives
``IF(x) = C(u, v) + P(U ≥ u, V ≥ v) − 2 E C = z(x) − E z``, and τ̂ = 4V_n − 1
multiplies it by 4 — hence 16 in the variance, not the 4 a naive plug-in
would give. Both marginal variances this construction reproduces are
exactly the ones :func:`standard_errors` reports for the two methods, so
the diagnostic cannot silently disagree with them; what it adds is the
cross term, which is large (measured correlation 0.97-0.99 at τ = 0.4, see
``pmcprg/tests/test_fr7_robust_options.py``). A bootstrap would also be
defensible; the influence-function route was taken because it is O(n log n),
deterministic, and reuses the estimators' own asymptotics rather than
resampling pseudo-observations whose ranks are not independent.

Weighted density-power-divergence estimation (FR-7 c) lives in
:mod:`pmcprg.copulas._robust`, which does not touch this module.

The observations are assumed **i.i.d.** Serially dependent pairs
(consecutive states of a Markov chain share y_n) need FR-5.

References
----------
* Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). A semiparametric estimation
  procedure of dependence parameters in multivariate families of
  distributions. *Biometrika* 82(3), 543–552. doi:10.1093/biomet/82.3.543
* Genest, C. & Favre, A.-C. (2007). Everything you always wanted to know about
  copula modeling but were afraid to ask. *J. Hydrol. Eng.* 12(4), 347–368.
  doi:10.1061/(ASCE)1084-0699(2007)12:4(347)
* Kojadinovic, I. & Yan, J. (2010). Comparison of three semiparametric methods
  for estimating dependence parameters in copula models. *Insurance Math.
  Econom.* 47(1), 52–63. doi:10.1016/j.insmatheco.2010.03.008
* Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. *J. Amer. Statist. Assoc.* 82(398), 605–610.
  doi:10.1080/01621459.1987.10478472
* Kendall, M. G. (1938). A new measure of rank correlation. *Biometrika*
  30(1/2), 81–93. doi:10.1093/biomet/30.1-2.81
* Klaassen, C. A. J. & Wellner, J. A. (1997). Efficient estimation in the
  bivariate normal copula model: normal margins are least favourable.
  *Bernoulli* 3(1), 55–77. doi:10.2307/3318652 — n·Var(ρ̂) → (1 − ρ²)² for the rank-based
  estimator, the closed form the tests check.
* Hausman, J. A. (1978). Specification tests in econometrics.
  *Econometrica* 46(6), 1251–1271. doi:10.2307/1913827 — the
  consistent-vs-consistent-and-efficient contrast :func:`mle_tau_discrepancy_test`
  follows in form (but not in its variance estimate: see above).
* Croux, C. & Dehon, C. (2010). Influence functions of the Spearman and
  Kendall correlation measures. *Stat. Methods Appl.* 19(4), 497–515.
  doi:10.1007/s10260-010-0142-z — Kendall's τ has a bounded influence
  function; the log-density score does not.
* Genest, C. & Werker, B. J. M. (2002). Conditions for the asymptotic
  semiparametric efficiency of an omnibus estimator of dependence parameters
  in copula models. In *Distributions with Given Marginals and Statistical
  Modelling*, 103–112. doi:10.1007/978-94-017-0061-0_12 — why the
  Hausman difference-of-variances shortcut is not available here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "IndependenceLRTest",
    "MleTauDiscrepancyTest",
    "StandardErrors",
    "SubmodelLRTest",
    "independence_lr_test",
    "mle_tau_discrepancy_test",
    "standard_errors",
    "submodel_lr_test",
]

# Central-difference steps in the working coordinate ψ and in logit(u).
_H_PSI: float = 1e-4
_H_Z: float = 1e-4
# τ̂ within this many optimiser pads of a range end is "at the boundary".
_BOUNDARY_PADS: float = 2.0
# Relative tolerance for an extra parameter sitting on its bound.
_BOUNDARY_REL: float = 1e-4
# Smallest number of positive-weight observations accepted.
_MIN_OBS: int = 4


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StandardErrors:
    """Asymptotic standard errors of a copula estimate — see the module docstring.

    Fields
    ------
    family      : class name of the copula.
    method      : ``'mle'`` (pseudo-likelihood sandwich) or ``'tau'`` (τ inversion).
    ranks       : whether the estimated-margin correction (W terms) was applied
                  (always ``True`` for ``'tau'``, whose variance already is the
                  rank statistic's).
    names       : reported quantities, in the order of ``cov``:
                  ``('tau_k', 'theta')`` for one-parameter families (``theta``
                  is the family's native parameter — ρ for the Gaussian),
                  ``('tau_k', 'df', 'rho')`` for Student,
                  ``('tau_k', 'delta', 'theta')`` for BB1.
    estimate    : ``{name: value}`` at which the variance was evaluated.
    se          : ``{name: standard error}``; NaN where it could not be computed,
                  and for ``'mle'`` at a boundary (module docstring).
    cov         : covariance matrix of ``names`` (rank = number of free
                  parameters; the native parameters are deterministic
                  functions of the free ones).
    at_boundary : the estimate sits on the boundary of the parameter space
                  (see ``boundary``); Wald intervals are not reported.
    boundary    : one sentence per boundary condition met (empty if interior).
    n_obs       : number of positive-weight observations.
    n_eff       : Σ w — the sample size the variance refers to.
    """

    family: str
    method: str
    ranks: bool
    names: tuple
    estimate: dict
    se: dict
    cov: np.ndarray
    at_boundary: bool
    boundary: tuple
    n_obs: int
    n_eff: float

    def ci(self, level: float = 0.95, name: str = "tau_k") -> tuple[float, float]:
        """Wald interval ``estimate ± z_{(1+level)/2} · se`` for ``name``.

        ``(nan, nan)`` when the estimate is at a boundary of the parameter
        space — the estimator is then not asymptotically normal (Self & Liang
        1987) — or when the standard error is not finite. The interval is on
        the natural scale and is not clipped to the parameter space.
        """
        from scipy.stats import norm

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}.")
        if name not in self.se:
            raise KeyError(f"{name!r} is not one of {self.names}.")
        se = self.se[name]
        if self.at_boundary or not np.isfinite(se):
            return float("nan"), float("nan")
        z = float(norm.ppf(0.5 + 0.5 * level))
        est = self.estimate[name]
        return est - z * se, est + z * se

    def cov_of(self, *names: str) -> np.ndarray:
        """Covariance sub-matrix of the requested quantities, in the given order."""
        idx = [self.names.index(n) for n in names]
        return self.cov[np.ix_(idx, idx)]

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={self.estimate[k]:.4g}±{self.se[k]:.3g}" for k in self.names)
        flag = ", at_boundary" if self.at_boundary else ""
        return (f"StandardErrors({self.family}, method={self.method!r}, {parts}, "
                f"n_eff={self.n_eff:g}{flag})")


@dataclass(frozen=True)
class IndependenceLRTest:
    """Likelihood-ratio test of independence in a one-parameter family.

    Fields
    ------
    family            : class name.
    statistic         : ``2 (ℓ(τ̂) − ℓ(Π))`` with ``ℓ = Σ w log c`` and
                        ``ℓ(Π) = 0``; ≥ 0.
    p_value           : from ``null_distribution``.
    null_distribution : ``'chi2(1)'`` when independence (τ = 0) is interior to
                        the family, ``'0.5*chi2(0) + 0.5*chi2(1)'`` when it is
                        an end of the parameter space (Self & Liang 1987).
    boundary          : independence is on the boundary of the family.
    tau_k             : the constrained (weighted) pseudo-maximum-likelihood τ̂.
    log_likelihood    : ``Σ w log c`` at τ̂.
    n_obs, n_eff      : positive-weight observations and Σ w.
    """

    family: str
    statistic: float
    p_value: float
    null_distribution: str
    boundary: bool
    tau_k: float
    log_likelihood: float
    n_obs: int
    n_eff: float


@dataclass(frozen=True)
class MleTauDiscrepancyTest:
    """Hausman-style contrast of the pseudo-MLE and the τ-inversion estimate.

    H0: both estimators are consistent for the same τ — what correct
    specification of the family and uncontaminated data imply. See the module
    docstring ("MLE-vs-τ discrepancy diagnostic") for the construction of
    ``se_difference``, which is *not* a difference of the two variances.

    Fields
    ------
    family         : class name of the one-parameter family.
    statistic      : ``(τ̂_MLE − τ̂_τ) / SE(τ̂_MLE − τ̂_τ)``; N(0, 1) under H0.
    p_value        : two-sided, ``2 Φ(−|statistic|)``.
    difference     : ``τ̂_MLE − τ̂_τ``.
    se_difference  : its standard error, from the joint influence functions.
    tau_mle, tau_tau       : the two estimates of τ.
    theta_mle, theta_tau   : the family's native parameter at each — reported
                             for information; the test is on the τ scale, which
                             is equivalent to first order (module docstring).
    se_tau_mle, se_tau_tau : the two marginal standard errors of τ̂, identical
                             to what :func:`standard_errors` returns for
                             ``method='mle'`` and ``method='tau'``.
    correlation    : estimated ``Corr(τ̂_MLE, τ̂_τ)`` from the same influence
                     functions — typically 0.95-0.99, which is exactly why the
                     cross term may not be dropped.
    at_boundary    : the pseudo-MLE sits on a boundary of the parameter space.
                     ``statistic``, ``p_value``, ``se_difference`` and
                     ``correlation`` are then NaN: neither the sandwich nor the
                     normal limit applies there (Self & Liang 1987), the same
                     convention :func:`standard_errors` follows.
    boundary       : one sentence per boundary condition met (empty if interior).
    n_obs, n_eff   : positive-weight observations and Σ w.
    """

    family: str
    statistic: float
    p_value: float
    difference: float
    se_difference: float
    tau_mle: float
    tau_tau: float
    theta_mle: float
    theta_tau: float
    se_tau_mle: float
    se_tau_tau: float
    correlation: float
    at_boundary: bool
    boundary: tuple
    n_obs: int
    n_eff: float

    def __repr__(self) -> str:
        flag = ", at_boundary" if self.at_boundary else ""
        return (f"MleTauDiscrepancyTest({self.family}, tau_mle={self.tau_mle:.4g}, "
                f"tau_tau={self.tau_tau:.4g}, z={self.statistic:.3g}, "
                f"p={self.p_value:.3g}, n_eff={self.n_eff:g}{flag})")


@dataclass(frozen=True)
class SubmodelLRTest:
    """Likelihood-ratio test of a one-parameter sub-model of a two-parameter family.

    H0: the extra parameter sits at the sub-model's value (module docstring
    lists the three implemented nestings and why each is a boundary case);
    H1: the full two-parameter family.

    Fields
    ------
    family             : class name of the two-parameter family (H1).
    submodel           : class name of the one-parameter sub-model (H0).
    statistic          : ``2 (ℓ_full − ℓ_sub) ≥ 0``, ``ℓ = Σ w log c`` at each
                        model's own (weighted) pseudo-MLE.
    p_value            : from ``null_distribution``.
    null_distribution  : ``'0.5*chi2(0) + 0.5*chi2(1)'`` for every pair this
                        module implements — the sub-model sits at a boundary
                        of the full family's admissible extra-parameter range
                        in all three cases (Self & Liang 1987).
    boundary           : always ``True`` here (see ``null_distribution``).
    boundary_note      : one sentence identifying which bound the sub-model
                        is (module docstring).
    full_params        : the full family's fitted ``{'tau_k': ..., extra: ...}``.
    sub_params         : the sub-model's fitted ``{'tau_k': ...}``.
    log_likelihood_full, log_likelihood_sub : ``Σ w log c`` at each fit.
    n_obs, n_eff       : positive-weight observations and Σ w.
    """

    family: str
    submodel: str
    statistic: float
    p_value: float
    null_distribution: str
    boundary: bool
    boundary_note: str
    full_params: dict
    sub_params: dict
    log_likelihood_full: float
    log_likelihood_sub: float
    n_obs: int
    n_eff: float


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _prepare(uv, weights) -> tuple[np.ndarray, np.ndarray, bool]:
    """``(uv, w, weighted)`` with zero-weight rows dropped; ``w`` is ones when unweighted."""
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (n, 2), got {uv.shape}.")
    if not np.all(np.isfinite(uv)) or np.any(uv <= 0.0) or np.any(uv >= 1.0):
        raise ValueError("pseudo-observations must lie strictly inside (0, 1).")
    if weights is None:
        w = np.ones(uv.shape[0])
        weighted = False
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape[0] != uv.shape[0]:
            raise ValueError(f"weights ({w.shape[0]}) and uv ({uv.shape[0]}) differ in length.")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        keep = w > 0.0
        uv, w = uv[keep], w[keep]
        weighted = True
    if uv.shape[0] < _MIN_OBS:
        raise ValueError(f"at least {_MIN_OBS} positive-weight observations are required, "
                         f"got {uv.shape[0]}.")
    return uv, w, weighted


def _registry_entry(cls):
    from pmcprg.copulas._base import CopulaEnum

    for entry in CopulaEnum:
        if entry.value.CLASS_NAME == cls.__name__:
            return entry
    raise ValueError(f"{cls.__name__!r} is not registered in CopulaEnum.")


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------

def _central_step(x: float, lo: float, hi: float) -> float:
    """Step of the central difference of :func:`_derivative_in_range` at x; 0 at an end."""
    d = min(x - lo, hi - x)
    return 1e-4 * min(d, 1.0) if d > 0.0 else 0.0


def _derivative_in_range(f: Callable[[float], float], x: float, lo: float, hi: float) -> float:
    """df/dx at x ∈ [lo, hi] without evaluating f outside the interval.

    Central difference with a step of 10⁻⁴ of the distance to the nearer end
    (capped at 10⁻⁴); at an end, the one-sided three-point formula with a step
    of 10⁻⁶ of the interval.
    """
    h = _central_step(x, lo, hi)
    if h > 0.0:
        return (f(x + h) - f(x - h)) / (2.0 * h)
    h = 1e-6 * min(hi - lo, 1.0)
    if x - lo <= 0.0:
        return (-3.0 * f(x) + 4.0 * f(x + h) - f(x + 2.0 * h)) / (2.0 * h)
    return (3.0 * f(x) - 4.0 * f(x - h) + f(x - 2.0 * h)) / (2.0 * h)


def _logit(x):
    return np.log(x) - np.log1p(-x)


def _expit(z):
    from scipy.special import expit

    return expit(z)


# ---------------------------------------------------------------------------
# Per-family working coordinates
# ---------------------------------------------------------------------------

@dataclass
class _Spec:
    """Working coordinate ψ of a family at an estimate, and its Jacobian.

    Each ψ_c is a function of one *natural* coordinate p_c = g_c(ψ_c) only —
    τ (one-parameter families), θ (Plackett), (τ, ν) (Student), (θ, δ)
    (BB1), the coordinates the fits optimise in. ``curv`` holds g''_c/g'_c.
    """
    cls: type
    names: tuple            # reported quantities
    estimate: dict
    psi: np.ndarray         # ψ̂ (may hold ±inf on an exact boundary)
    build: Callable         # ψ → copula instance
    jac_psi: np.ndarray     # (m, p) d(names)/dψ at ψ̂
    dq_dtau: np.ndarray | None   # (m,) d(names)/dτ — one-parameter families only
    boundary: tuple
    curv: np.ndarray        # (p,) g''/g' of the natural-coordinate maps


def _tau_boundary(copula, tau: float, independence_note: bool) -> list[str]:
    from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL

    a, b = copula.tau_min, copula.tau_max
    pad = max(TAU_PAD_REL * (b - a), TAU_PAD_ABS)
    out = []
    for end, dist in ((a, tau - a), (b, b - tau)):
        if dist <= _BOUNDARY_PADS * pad:
            if abs(end) <= 1e-12 and independence_note:
                out.append(
                    f"τ̂ = {tau:.3g} at the independence end of the range [{a:.3g}, {b:.3g}]: "
                    "a likelihood-ratio test of independence has the null distribution "
                    "½χ²₀ + ½χ²₁ (Self & Liang 1987)."
                )
            else:
                out.append(f"τ̂ = {tau:.6g} at the end {end:.6g} of the range [{a:.6g}, {b:.6g}].")
    return out


def _reachable_note(cls, tau: float, bound: float) -> str:
    return (f"τ̂ = {tau:.6g} at {bound:.6g}, the end of the τ {cls.__name__} reaches: its "
            "parameter is capped there and the likelihood is flat beyond, so the estimate "
            "is a constrained stop, not an interior optimum.")


def _one_parameter_spec(copula) -> _Spec:
    from pmcprg.copulas.explicit.plackett import CopulaPlackett, _plackett_tau_from_theta

    cls = type(copula)
    a, b = float(copula.tau_min), float(copula.tau_max)
    tau = float(copula.params["tau_k"])
    theta = float(copula.theta)
    names = ("tau_k", "theta")
    estimate = {"tau_k": tau, "theta": theta}
    boundary = _tau_boundary(copula, tau, independence_note=True)
    # A family whose parameter is capped below its registered τ-range (Frank,
    # Plackett) stores a fit beyond the cap at the reachable bound (F1, G2).
    reach = cls.reachable_tau_bounds()

    if isinstance(copula, CopulaPlackett):
        # θ is set directly below: no difference step crosses the cap, but τ̂
        # at a table end is where the fit stopped at θ = 10^{±6}.
        if reach is not None and not reach[0] < tau < reach[1]:
            boundary.append(_reachable_note(cls, tau, reach[1] if tau > 0.0 else reach[0]))
        # θ(τ) is a table interpolation (piecewise linear in log θ): take the
        # derivatives in log θ, setting θ directly, and τ'(θ) from the
        # quadrature that defines τ(θ).
        log_theta = math.log(theta)

        def build(p, _tau=tau):
            cop = cls(tau_k=_tau)
            cop.theta = float(math.exp(p[0]))
            return cop

        dtau_dlogtheta = _derivative_in_range(
            lambda lt: _plackett_tau_from_theta(math.exp(lt)), log_theta, -np.inf, np.inf)
        dq_dtau = np.array([1.0, theta / dtau_dlogtheta]) if dtau_dlogtheta != 0.0 else None
        jac = np.array([[dtau_dlogtheta], [theta]])
        return _Spec(cls, names, estimate, np.array([log_theta]), build, jac, dq_dtau,
                     tuple(boundary), np.array([1.0]))

    span = b - a
    if a < tau < b:
        psi = math.log(tau - a) - math.log(b - tau)
        dtau_dpsi = (tau - a) * (b - tau) / span
    else:
        psi = -math.inf if tau <= a else math.inf
        dtau_dpsi = 0.0

    def build(p, _a=a, _span=span):
        return cls(tau_k=float(_a + _span * _expit(p[0])))

    lo_d, hi_d = a, b
    if reach is not None:
        # Beyond the reachable bound θ is capped: a central step across it
        # sees a flat θ(τ) and likelihood (one-sided values, and the family's
        # clamping WARNING at every evaluation). When a stencil — τ ± h of
        # dθ/dτ, or ψ̂ ± h of the sandwich — would cross, τ̂ is on the bound:
        # flagged, and dθ/dτ is taken inside [max(a, lo), min(b, hi)].
        h = _central_step(tau, a, b)
        stencil = [tau - h, tau + h]
        if math.isfinite(psi):
            stencil += [float(a + span * _expit(psi + sgn * _H_PSI)) for sgn in (-1.0, 1.0)]
        if min(stencil) < reach[0] or max(stencil) > reach[1]:
            boundary.append(_reachable_note(cls, tau, reach[1] if tau > 0.0 else reach[0]))
            lo_d, hi_d = max(a, reach[0]), min(b, reach[1])
    dtheta_dtau = _derivative_in_range(lambda t: float(cls(tau_k=float(t)).theta), tau, lo_d, hi_d)
    dq_dtau = np.array([1.0, dtheta_dtau])
    jac = (dq_dtau * dtau_dpsi).reshape(2, 1)
    curv = np.array([(a + b - 2.0 * tau) / span])      # τ''/τ' of τ = a + span·expit(ψ)
    return _Spec(cls, names, estimate, np.array([psi]), build, jac, dq_dtau, tuple(boundary),
                 curv)


def _student_spec(copula) -> _Spec:
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    cls = type(copula)
    tau = float(copula.params["tau_k"])
    nu = float(copula.df)
    rho = float(copula.theta)
    names = ("tau_k", "df", "rho")
    estimate = {"tau_k": tau, "df": nu, "rho": rho}
    psi = np.array([math.atanh(tau) if abs(tau) < 1.0 else math.copysign(math.inf, tau),
                    math.log(nu - 2.0) if nu > 2.0 else -math.inf])

    def build(p):
        return cls(tau_k=float(math.tanh(p[0])), df=float(2.0 + math.exp(p[1])))

    jac = np.array([
        [1.0 - tau * tau, 0.0],
        [0.0, nu - 2.0],
        [0.5 * math.pi * math.cos(0.5 * math.pi * tau) * (1.0 - tau * tau), 0.0],
    ])
    boundary = _tau_boundary(copula, tau, independence_note=False)
    lo, hi, _ = EXTRA_PARAM_BOUNDS_BY_PARAM["df"]
    if nu >= hi * (1.0 - _BOUNDARY_REL):
        boundary.append(
            f"ν̂ = {nu:.6g} at the upper bound {hi:g} of the fitting box, which stands for the "
            "Gaussian limit ν → ∞ (η = 1/ν = 0 on the boundary): a likelihood-ratio test of the "
            "Gaussian copula has the null distribution ½χ²₀ + ½χ²₁ (Self & Liang 1987)."
        )
    if nu <= lo * (1.0 + _BOUNDARY_REL):
        boundary.append(f"ν̂ = {nu:.6g} at the lower bound {lo:g} of the fitting box (ν > 2).")
    # τ = tanh ψ₁: τ''/τ' = −2τ;  ν = 2 + exp ψ₂: 1.
    return _Spec(cls, names, estimate, psi, build, jac, None, tuple(boundary),
                 np.array([-2.0 * tau, 1.0]))


def _bb1_spec(copula) -> _Spec:
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    cls = type(copula)
    tau = float(copula.params["tau_k"])
    delta = float(copula.delta)
    theta = float(copula.theta)
    names = ("tau_k", "delta", "theta")
    estimate = {"tau_k": tau, "delta": delta, "theta": theta}
    psi = np.array([math.log(theta) if theta > 0.0 else -math.inf,
                    math.log(delta - 1.0) if delta > 1.0 else -math.inf])

    def build(p):
        th = float(math.exp(p[0]))
        de = float(1.0 + math.exp(p[1]))
        # τ = (δθ + 2(δ − 1)) / (δ(θ + 2)): 1 − 2/(δ(θ + 2)) without cancellation.
        return cls(tau_k=(de * th + 2.0 * (de - 1.0)) / (de * (th + 2.0)), delta=de)

    dtau_dtheta = 2.0 / (delta * (theta + 2.0) ** 2)
    dtau_ddelta = 2.0 / (delta * delta * (theta + 2.0))
    jac = np.array([
        [theta * dtau_dtheta, (delta - 1.0) * dtau_ddelta],
        [0.0, delta - 1.0],
        [theta, 0.0],
    ])
    boundary = _tau_boundary(copula, tau, independence_note=False)
    if delta - 1.0 <= _BOUNDARY_REL:
        boundary.append(
            f"δ̂ = {delta:.6g} at δ = 1, the Clayton limit of BB1: a likelihood-ratio test of "
            "the Clayton copula has the null distribution ½χ²₀ + ½χ²₁ (Self & Liang 1987)."
        )
    theta_floor = float(getattr(cls, "_THETA_FLOOR", 1e-6))
    if theta <= 10.0 * theta_floor:
        boundary.append(f"θ̂ = {theta:.3g} at its floor {theta_floor:g}, the Gumbel limit "
                        "θ → 0 of BB1.")
    d_hi = EXTRA_PARAM_BOUNDS_BY_PARAM["delta"][1]
    if abs(delta - d_hi) <= _BOUNDARY_REL * d_hi:
        boundary.append(f"δ̂ = {delta:.6g} at the upper bound {d_hi:g} of the ICE fitting box.")
    # θ = exp ψ₁ and δ = 1 + exp ψ₂: g''/g' = 1 for both.
    return _Spec(cls, names, estimate, psi, build, jac, None, tuple(boundary),
                 np.array([1.0, 1.0]))


def _spec_of(copula) -> _Spec:
    from pmcprg.copulas.archimedean.bb1 import CopulaBB1
    from pmcprg.copulas.elliptical.student import CopulaStudent

    if copula.tau_max - copula.tau_min < 1e-8:
        raise ValueError(f"{type(copula).__name__} has no free parameter.")
    if isinstance(copula, CopulaStudent):
        return _student_spec(copula)
    if isinstance(copula, CopulaBB1):
        return _bb1_spec(copula)
    if int(getattr(copula, "n_params", 1)) != 1:
        raise NotImplementedError(
            f"standard errors are not implemented for the {copula.n_params}-parameter "
            f"family {type(copula).__name__}.")
    return _one_parameter_spec(copula)


# ---------------------------------------------------------------------------
# Estimated-margin corrections
# ---------------------------------------------------------------------------

def _upper_weighted_sums(x: np.ndarray, vals: np.ndarray, w: np.ndarray) -> np.ndarray:
    """``S_i = Σ_j w_j vals_j 1{x_i ≤ x_j}`` for every i, in O(n log n).

    ``vals`` has shape (n, p); ties count (≤), so repeated rows and integer
    weights give the same sums.
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    wv = (w[:, None] * vals)[order]
    suffix = np.cumsum(wv[::-1], axis=0)[::-1]
    suffix = np.vstack([suffix, np.zeros((1, vals.shape[1]))])
    return suffix[np.searchsorted(xs, x, side="left")]


# ---------------------------------------------------------------------------
# Covariance in ψ: pseudo-likelihood sandwich
# ---------------------------------------------------------------------------

def _derivatives(spec: _Spec, uv: np.ndarray, ranks: bool):
    """Per-observation score (n, p), Hessian (n, p, p) and, with ``ranks``,
    ∂φ/∂u and ∂φ/∂v (n, p) — central differences in ψ and logit(u)."""
    p = spec.psi.size
    h, k = _H_PSI, _H_Z
    eye = np.eye(p)

    def ll(psi, pts=uv):
        return np.asarray(spec.build(psi).logpdf_array(pts), dtype=float)

    l0 = ll(spec.psi)
    lp = [ll(spec.psi + h * eye[a]) for a in range(p)]
    lm = [ll(spec.psi - h * eye[a]) for a in range(p)]
    n = uv.shape[0]
    score = np.empty((n, p))
    hess = np.empty((n, p, p))
    for a in range(p):
        score[:, a] = (lp[a] - lm[a]) / (2.0 * h)
        hess[:, a, a] = (lp[a] - 2.0 * l0 + lm[a]) / (h * h)
        for c in range(a + 1, p):
            mixed = (ll(spec.psi + h * (eye[a] + eye[c])) - ll(spec.psi + h * (eye[a] - eye[c]))
                     - ll(spec.psi - h * (eye[a] - eye[c])) + ll(spec.psi - h * (eye[a] + eye[c])))
            hess[:, a, c] = hess[:, c, a] = mixed / (4.0 * h * h)
    if not ranks:
        return score, hess, None, None

    margin_derivs = []
    for col in (0, 1):
        x = uv[:, col]
        z = _logit(x)
        up, dn = uv.copy(), uv.copy()
        up[:, col] = _expit(z + k)
        dn[:, col] = _expit(z - k)
        dphi = np.empty((n, p))
        for a in range(p):
            plus, minus = spec.psi + h * eye[a], spec.psi - h * eye[a]
            d2 = (ll(plus, up) - ll(plus, dn) - ll(minus, up) + ll(minus, dn)) / (4.0 * h * k)
            dphi[:, a] = d2 / (x * (1.0 - x))          # ∂/∂u = (∂/∂z) / (du/dz)
        margin_derivs.append(dphi)
    return score, hess, margin_derivs[0], margin_derivs[1]


def _score_and_information(spec: _Spec, uv: np.ndarray, w: np.ndarray,
                           ranks: bool, what: str) -> tuple[np.ndarray, np.ndarray] | None:
    """``(M, B)`` of the pseudo-likelihood sandwich in ψ, or ``None``.

    ``M`` is the (n, p) per-observation estimating function ``φ + Ŵ₁ + Ŵ₂``
    (``φ`` alone when ``ranks`` is False) and ``B`` the (p, p) observed
    information with its chain-rule gradient term removed. ``B⁻¹ M_i`` is
    then the influence function of ψ̂ — used both by
    :func:`_sandwich_cov_psi` and by :func:`mle_tau_discrepancy_test`, so the
    two can never describe different estimators. ``None`` (with a WARNING
    naming ``what``) when the estimate is on an exact boundary, when a
    derivative is not finite, or when the information is not positive
    definite.
    """
    if not np.all(np.isfinite(spec.psi)):
        logger.warning("%s %s: the estimate is on an exact boundary (ψ = ±∞); "
                       "no pseudo-likelihood variance is computed.", spec.cls.__name__, what)
        return None
    score, hess, dphi_u, dphi_v = _derivatives(spec, uv, ranks)
    arrays = [score, hess] + ([dphi_u, dphi_v] if ranks else [])
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        logger.warning("%s %s: non-finite log-density derivatives at some "
                       "observations; returning NaN.", spec.cls.__name__, what)
        return None

    sw = float(w.sum())
    B = -np.einsum("i,iab->ab", w, hess) / sw
    B = 0.5 * (B + B.T)
    # Remove the gradient term of the chain rule, ∂ℓ/∂p_c · g''_c: the Hessian
    # in ψ is then G·(Hessian in the natural coordinates p)·G, G = diag g'.
    # It vanishes at an exact root of the score, not at an optimiser's stop;
    # measured after fit(method='mle') (n = 2000, Clayton/GH/Gauss, τ̂ from
    # 0.002 to 0.5) it changes Var(τ̂) by at most 0.04 %.
    B = B + np.diag((w @ score) / sw * spec.curv)
    M = score.copy()
    if ranks:
        M += _upper_weighted_sums(uv[:, 0], dphi_u, w) / sw
        M += _upper_weighted_sums(uv[:, 1], dphi_v, w) / sw
    eig = np.linalg.eigvalsh(B)
    if not np.all(eig > 0.0):
        logger.warning("%s %s: the pseudo-likelihood is not locally concave at "
                       "the estimate (information eigenvalues %s); returning NaN.",
                       spec.cls.__name__, what, eig)
        return None
    return M, B


def _sandwich_cov_psi(spec: _Spec, uv: np.ndarray, w: np.ndarray, ranks: bool) -> np.ndarray:
    p = spec.psi.size
    got = _score_and_information(spec, uv, w, ranks, "standard errors")
    if got is None:
        return np.full((p, p), np.nan)
    M, B = got
    sw = float(w.sum())
    Mc = M - (w @ M) / sw
    Omega = (Mc * w[:, None]).T @ Mc / sw
    Binv = np.linalg.inv(B)
    return Binv @ Omega @ Binv.T / sw


def _tau_influence(uv: np.ndarray, w: np.ndarray, weighted: bool) -> tuple[np.ndarray, float]:
    """``(4 (ẑ − z̄), τ̂)`` — Kendall's τ̂ and its influence function, from C_n.

    ``ẑ_i = 2 C_n(û_i, v̂_i) − û_i − v̂_i``, with the weighted empirical copula
    when weights are given. The influence function ``4(ẑ − z̄)`` is the one
    behind this module's ``'tau'`` variance ``16 Var{2C − U − V}/n`` (module
    docstring, "MLE-vs-τ discrepancy diagnostic").

    τ̂ itself is the **U-statistic** form written through the same C_n. With
    ``S = Σw`` and ``Q = Σw²``, ``S² mean_w(C_n) = Σ_{i,j} w_i w_j
    1{û_j ≤ û_i, v̂_j ≤ v̂_i}`` counts every ordered pair including ``i = j``,
    so removing the Q diagonal terms and normalising by ``S² − Q`` gives

        τ̂ = 4 (S² mean_w(C_n) − Q) / (S² − Q) − 1 ,

    which for unit weights and distinct pseudo-observations is **exactly**
    ``scipy.stats.kendalltau`` (pinned in ``test_fr7_robust_options.py``), and
    not the V-statistic ``4 mean(C_n) − 1``, which differs by O(1/n) — an
    offset of the same order as the discrepancy being tested at moderate n.
    With frequency weights the within-row pairs a repeated row would
    contribute are excluded along with the diagonal, an O(Q/S²) difference
    from literally expanding the rows.
    """
    from pmcprg.copulas._fit import _empirical_copula

    cn = _empirical_copula(uv, uv, w if weighted else None)
    z = 2.0 * cn - uv[:, 0] - uv[:, 1]
    sw = float(w.sum())
    sq = float(w @ w)
    zbar = float(w @ z) / sw
    denom = sw * sw - sq
    if denom <= 0.0:
        return 4.0 * (z - zbar), float("nan")
    tau_hat = 4.0 * (sw * float(w @ cn) - sq) / denom - 1.0
    return 4.0 * (z - zbar), float(tau_hat)


def _tau_variance(uv: np.ndarray, w: np.ndarray, weighted: bool) -> float:
    """Var(τ̂) ≈ 16 · Var_w{2 C_n(û, v̂) − û − v̂} / Σw."""
    b, _ = _tau_influence(uv, w, weighted)
    sw = float(w.sum())
    return float(w @ (b * b)) / sw / sw


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def standard_errors(copula, uv, weights=None, method: str = "mle", *,
                    ranks: bool = True) -> StandardErrors:
    """Asymptotic standard errors of the parameters of ``copula`` estimated on ``uv``.

    Parameters
    ----------
    copula  : a fitted copula instance — its parameters are the estimate.
    uv      : (n, 2) pseudo-observations in (0, 1)², **the ones the estimate was
              computed from** (``FitResult.uv`` after :meth:`CopulaVirt.fit`).
    weights : optional (n,) non-negative frequency weights (module docstring).
    method  : ``'mle'`` — pseudo-likelihood sandwich (Genest, Ghoudi & Rivest
              1995); ``'tau'`` — variance of the inversion of Kendall's τ
              (Genest & Favre 2007; Kojadinovic & Yan 2010). ``'tau'`` needs a
              one-parameter family: τ does not identify Student's ν or BB1's δ.
    ranks   : ``'mle'`` only. ``True`` (default) when ``uv`` are rank-based
              pseudo-observations, whose estimated margins add the W terms;
              ``False`` for margins taken as known.

    Returns
    -------
    StandardErrors

    Raises
    ------
    ValueError          for a family without a free parameter, an unknown
                        method, ``'tau'`` on a two-parameter family, or invalid
                        ``uv`` / ``weights``.
    NotImplementedError for a multi-parameter family other than Student and BB1.
    """
    if method not in ("mle", "tau"):
        raise ValueError(f"method must be 'mle' or 'tau', got {method!r}.")
    uv, w, weighted = _prepare(uv, weights)
    spec = _spec_of(copula)

    if method == "mle":
        if spec.boundary:
            # Not asymptotically normal there, and the ψ-curvature at a
            # constrained stop describes the box, not the likelihood.
            m = len(spec.names)
            cov = np.full((m, m), np.nan)
        else:
            cov_psi = _sandwich_cov_psi(spec, uv, w, ranks)
            cov = spec.jac_psi @ cov_psi @ spec.jac_psi.T
    else:
        if spec.dq_dtau is None:
            raise ValueError(
                f"method='tau' is not available for {spec.cls.__name__}: Kendall's τ does not "
                f"identify its second parameter (and its fit is always 'mle').")
        ranks = True
        cov = np.outer(spec.dq_dtau, spec.dq_dtau) * _tau_variance(uv, w, weighted)

    se = {name: float(math.sqrt(cov[i, i])) if cov[i, i] >= 0.0 else float("nan")
          for i, name in enumerate(spec.names)}
    return StandardErrors(
        family=spec.cls.__name__, method=method, ranks=bool(ranks), names=spec.names,
        estimate=dict(spec.estimate), se=se, cov=cov,
        at_boundary=bool(spec.boundary), boundary=spec.boundary,
        n_obs=int(uv.shape[0]), n_eff=float(w.sum()),
    )


def independence_lr_test(family, uv, weights=None) -> IndependenceLRTest:
    """Likelihood-ratio test of independence within a one-parameter family.

    The (weighted) pseudo-likelihood ``ℓ(τ) = Σ w log c(û, v̂; τ)`` is maximised
    over the padded τ-range, as :meth:`CopulaVirt.fit` does, and compared with
    the independence copula, ``ℓ(Π) = 0``: ``LR = 2 max(ℓ(τ̂), 0)``.

    * Independence **inside** the family (Gaussian, Frank, FGM, AMH,
      Plackett): LR → χ²₁.
    * Independence at an **end** of the parameter space (Clayton, Gumbel, Joe,
      their survival versions, Cubic Section — τ ≥ 0 only — and the
      90°/270° rotations of FR-8 whose range ends at τ = −ε, e.g.
      ``CopulaClayton90`` — τ ≤ 0 only): the restricted
      maximiser sits on the boundary about half of the time under H₀, and
      LR → ½χ²₀ + ½χ²₁ (Self & Liang 1987): ``p = ½ P(χ²₁ ≥ LR)`` for
      LR > 0, ``p = 1`` for LR = 0.

    On rank-based pseudo-observations the W corrections of the pseudo-
    likelihood score vanish under independence (∫ φ(u, v) dv = 0 for every u
    when c ≡ 1), so these reference distributions hold unchanged (Genest,
    Ghoudi & Rivest 1995). Weights are frequency weights; with fractional
    weights ≤ 1 the statistic is stochastically smaller than its reference
    and the test conservative.

    Parameters
    ----------
    family  : a one-parameter CopulaVirt subclass (or an instance of one).
    uv      : (n, 2) pseudo-observations in (0, 1)².
    weights : optional (n,) non-negative weights.

    Raises
    ------
    ValueError when the family does not contain independence (A12, A14 and
    their rotations, whose ranges stop at ±1/3), has
    no free parameter (Product) or has two parameters.
    """
    from scipy.optimize import minimize_scalar
    from scipy.stats import chi2

    from pmcprg.copulas._base import padded_tau_range
    from pmcprg.copulas._fit import MLE_FAIL_PENALTY, _weighted_log_density_sum

    cls = family if isinstance(family, type) else type(family)
    entry = _registry_entry(cls)
    if len(entry.value.PARAMETERS_SET_NAME) != 1:
        raise ValueError(f"{cls.__name__} has more than one parameter; the independence LR "
                         "test is implemented for one-parameter families.")
    a, b = (float(t) for t in entry.value.TAU_MIN_MAX)
    if b - a < 1e-8:
        raise ValueError(f"{cls.__name__} has no free parameter.")
    # ``end``: the independence end of a one-sided range. The negative
    # mirror (FR-8 rotations, range [−1, −ε]) used to fall through to the
    # "does not contain independence" error below, as if it were A12's.
    end = None
    if a < 0.0 < b:
        boundary = False
    elif 0.0 <= a <= 1e-12:
        boundary, end = True, "lo"
    elif -1e-12 <= b <= 0.0:
        boundary, end = True, "hi"
    else:
        raise ValueError(f"{cls.__name__} does not contain the independence copula "
                         f"(τ-range [{a:.4g}, {b:.4g}]).")
    uv, w, weighted = _prepare(uv, weights)

    def loglik(tau: float) -> float:
        try:
            # Beyond the reachable |τ| (Frank) the density is that of the bound.
            ld = cls(tau_k=entry.reachable_tau(float(tau))).logpdf_array(uv)
        except Exception:
            return -math.inf
        return _weighted_log_density_sum(ld, w if weighted else None)

    def neg(tau: float) -> float:
        ll = loglik(tau)
        return -ll if np.isfinite(ll) else MLE_FAIL_PENALTY

    lo, hi = padded_tau_range(a, b)
    res = minimize_scalar(neg, bounds=(lo, hi), method="bounded")
    tau_hat, ll_hat = entry.reachable_tau(float(res.x)), loglik(float(res.x))
    if boundary:
        tau_end = lo if end == "lo" else hi
        ll_end = loglik(tau_end)
        if not np.isfinite(ll_hat) or (np.isfinite(ll_end) and ll_end > ll_hat):
            tau_hat, ll_hat = tau_end, ll_end
    if not np.isfinite(ll_hat):
        raise ValueError(f"{cls.__name__}: the pseudo-likelihood is not finite at any τ "
                         "the search evaluated.")

    stat = max(2.0 * ll_hat, 0.0)
    if boundary:
        null = "0.5*chi2(0) + 0.5*chi2(1)"
        p_value = 1.0 if stat <= 0.0 else 0.5 * float(chi2.sf(stat, 1))
    else:
        null = "chi2(1)"
        p_value = float(chi2.sf(stat, 1))
    return IndependenceLRTest(
        family=cls.__name__, statistic=float(stat), p_value=float(p_value),
        null_distribution=null, boundary=boundary, tau_k=tau_hat,
        log_likelihood=float(ll_hat), n_obs=int(uv.shape[0]), n_eff=float(w.sum()),
    )


# ---------------------------------------------------------------------------
# Two-parameter sub-model LR test (module docstring, "Two-parameter
# sub-model tests")
# ---------------------------------------------------------------------------

# (full family CLASS_NAME, sub-model CLASS_NAME) -> where the sub-model sits.
# Every entry here is a *boundary* of the full family's registered
# extra-parameter range (verified against each family's own module
# docstring — module docstring above); none is interior, so there is no
# branch for a plain chi2(1) null below. Adding a genuinely interior
# nesting later would need one.
_SUBMODEL_NESTING: dict[tuple[str, str], str] = {
    ("CopulaBB1", "CopulaClayton"): (
        "delta = 1 is the lower end of BB1's registered delta >= 1 range "
        "(the Clayton limit; pmcprg.copulas.archimedean.bb1 module docstring)."
    ),
    ("CopulaBB1", "CopulaGH"): (
        "theta -> 0 is the lower end of BB1's registered theta > 0 range "
        "(the floor CopulaBB1._THETA_FLOOR stands in for it in the joint fit; "
        "the Gumbel-Hougaard limit; pmcprg.copulas.archimedean.bb1 module "
        "docstring)."
    ),
    ("CopulaStudent", "CopulaGaussian"): (
        "nu -> infinity is not reachable by the optimiser; the upper end of "
        "the fitting box EXTRA_PARAM_BOUNDS_BY_PARAM['df'] = (2.001, 100.0) "
        "stands in for it (the Gaussian limit; "
        "pmcprg.copulas.elliptical.student module docstring)."
    ),
    ("CopulaTawn1", "CopulaGH"): (
        "psi = 1 is the upper end of Tawn's registered psi in (0, 1] range "
        "(the Gumbel-Hougaard limit; pmcprg.copulas.extreme_value.tawn module "
        "docstring)."
    ),
    ("CopulaTawn2", "CopulaGH"): (
        "psi = 1 is the upper end of Tawn's registered psi in (0, 1] range "
        "(the Gumbel-Hougaard limit; pmcprg.copulas.extreme_value.tawn module "
        "docstring)."
    ),
    # FR-9, BB6 round. Both of BB6's sub-models are boundaries of its
    # admissible set {theta >= 1, delta >= 1} — the set the joint fit's own
    # (ln(theta - 1), ln delta) box is — so both keep the one-sided null.
    ("CopulaBB6", "CopulaJoe"): (
        "delta6 = 1 is the lower end of BB6's registered delta6 >= 1 range "
        "(the Joe limit; pmcprg.copulas.archimedean.bb6 module docstring)."
    ),
    ("CopulaBB6", "CopulaGH"): (
        "theta = 1 is the lower end of BB6's registered theta >= 1 range "
        "(the floor theta = 1 + 1e-10 of the joint fit's ln(theta - 1) box "
        "stands in for it; the Gumbel-Hougaard limit; "
        "pmcprg.copulas.archimedean.bb6 module docstring)."
    ),
    # FR-9, BB7 round. Both of BB7's sub-models are boundaries of its
    # admissible set {theta >= 1, delta > 0} too, but of two different kinds:
    # theta = 1 is attained (Clayton), delta = 0 is an *open* end (Joe is a
    # limit no parameter realises), which the fitting box's lower end stands
    # in for exactly as Student's nu = 100 stands in for nu -> infinity.
    # Either way the constrained MLE sits on the boundary, so the null stays
    # one-sided.
    ("CopulaBB7", "CopulaClayton"): (
        "theta7 = 1 is the lower end of BB7's registered theta7 >= 1 range "
        "(the floor theta = 1 + 1e-10 of the joint fit's ln(theta - 1) box "
        "stands in for it; the Clayton limit; "
        "pmcprg.copulas.archimedean.bb7 module docstring)."
    ),
    ("CopulaBB7", "CopulaJoe"): (
        "theta7 -> theta_Joe(tau) is the open upper end of BB7's admissible "
        "theta7 range, where its recovered delta -> 0; the constructor's own "
        "floor delta = 1e-14, at which lambda_L = 2^-1e14 is 0 to machine "
        "precision, stands in for it (the Joe limit; "
        "pmcprg.copulas.archimedean.bb7 module docstring)."
    ),
    # FR-9, BB8 round. BB8 has exactly ONE two-parameter nesting, not two:
    # delta8 = 1 leaves Joe's generator outright, while theta = 1 leaves
    # -ln(t) — the *independence* copula, whatever delta8 is — which has no
    # free parameter to profile and is independence_lr_test's business, not
    # this function's. The "Joe-Frank" nickname notwithstanding, Frank is not
    # a sub-model either: it is the joint limit theta -> infinity with
    # theta*delta8 fixed, a path no single parameter value sits on
    # (pmcprg.copulas.archimedean.bb8 module docstring, verified numerically
    # there and in test_bb8.py, not taken from the name).
    ("CopulaBB8", "CopulaJoe"): (
        "delta8 = 1 is the upper end of BB8's registered delta8 in (0, 1] "
        "range (the Joe limit, exact and not asymptotic: (1 - delta8)^theta "
        "= 0 leaves Joe's own generator; "
        "pmcprg.copulas.archimedean.bb8 module docstring)."
    ),
    # FR-9, Tawn-3 round. The first nesting where the *sub-model itself* has
    # a free extra parameter (Tawn1/2's own psi), not just tau: fitting it
    # needs the joint MLE (_fit_two_parameter_mle), not the one-parameter
    # profile every entry above uses — see submodel_lr_test's dispatch on
    # len(PARAMETERS_SET_NAME). psi_u = 1 and psi_v = 1 are each the upper
    # end of Tawn3's registered psi_* in (0, 1] range, so the null is the
    # usual one-sided one (pmcprg.copulas.extreme_value.tawn module
    # docstring, "Sub-models": "psi_u = 1 -> type 1 at psi = psi_v; psi_v = 1
    # -> type 2 at psi = psi_u").
    ("CopulaTawn3", "CopulaTawn1"): (
        "psi_u = 1 is the upper end of Tawn3's registered psi_u in (0, 1] "
        "range (Tawn type 1 at psi = psi_v; "
        "pmcprg.copulas.extreme_value.tawn module docstring)."
    ),
    ("CopulaTawn3", "CopulaTawn2"): (
        "psi_v = 1 is the upper end of Tawn3's registered psi_v in (0, 1] "
        "range (Tawn type 2 at psi = psi_u; "
        "pmcprg.copulas.extreme_value.tawn module docstring)."
    ),
}


def _fit_one_parameter_profile(cls, uv: np.ndarray, weights) -> tuple[float, float]:
    """``(τ̂, ℓ(τ̂))`` maximising ``Σ w log c`` over a one-parameter family.

    Same Brent search, over the padded τ-range, as the inline profile
    :func:`independence_lr_test` runs against independence — reused here
    against a sub-model's own free parameter instead.
    """
    from scipy.optimize import minimize_scalar

    from pmcprg.copulas._base import padded_tau_range
    from pmcprg.copulas._fit import MLE_FAIL_PENALTY, _weighted_log_density_sum

    entry = _registry_entry(cls)
    a, b = (float(t) for t in entry.value.TAU_MIN_MAX)
    lo, hi = padded_tau_range(a, b)

    def loglik(tau: float) -> float:
        try:
            ld = cls(tau_k=entry.reachable_tau(float(tau))).logpdf_array(uv)
        except Exception:
            return -math.inf
        return _weighted_log_density_sum(ld, weights)

    def neg(tau: float) -> float:
        ll = loglik(tau)
        return -ll if np.isfinite(ll) else MLE_FAIL_PENALTY

    res = minimize_scalar(neg, bounds=(lo, hi), method="bounded")
    tau_hat = entry.reachable_tau(float(res.x))
    ll_hat = loglik(tau_hat)
    if not np.isfinite(ll_hat):
        raise ValueError(f"{cls.__name__}: the pseudo-likelihood is not finite at any τ "
                         "the search evaluated.")
    return tau_hat, ll_hat


def submodel_lr_test(full_family, sub_family, uv, weights=None) -> SubmodelLRTest:
    """Likelihood-ratio test of a family's sub-model, one boundary parameter away.

    ``H0``: ``sub_family`` (the extra parameter at its sub-model value);
    ``H1``: ``full_family``. The (weighted) pseudo-likelihoods
    ``ℓ = Σ w log c(û, v̂; ·)`` are maximised separately over each model —
    the full family by the same joint MLE as
    :meth:`CopulaVirt.fit(method='mle')` on a multi-parameter family
    (:func:`pmcprg.copulas._fit._fit_two_parameter_mle`, which despite its
    name fits any number of extras) — and compared:
    ``LR = 2 max(ℓ_full − ℓ_sub, 0)``. The sub-model is fitted the same way,
    dispatched on how many parameters it has: the 1-D profile of
    :func:`independence_lr_test` when it is a plain one-parameter family
    (every nesting below except the last two), the same joint MLE as the
    full model when it is itself a two-parameter family — Tawn 1/2's own
    ``psi`` (Tawn-3 round, FR-9): profiling τ alone there would leave ``psi``
    at the constructor's default instead of its own MLE, understating
    ``ℓ_sub`` and biasing the statistic upward.

    Implemented nestings (module docstring, "Two-parameter sub-model tests",
    verifies each against the family's own module docstring): BB1 → Clayton,
    BB1 → Gumbel (``CopulaGH``), Student → Gauss (``CopulaGaussian``), Tawn
    type 1 or 2 → Gumbel (``CopulaGH``), BB6 → Joe, BB6 → Gumbel
    (``CopulaGH``), BB7 → Clayton, BB7 → Joe, BB8 → Joe, Tawn 3 → Tawn type 1
    or 2. Every one is a **boundary** of the full family's admissible
    extra-parameter range, so ``LR → ½χ²₀ + ½χ²₁`` (Self & Liang 1987),
    unlike the plain ``χ²₁`` an interior sub-model would give.

    Parameters
    ----------
    full_family : the multi-parameter ``CopulaVirt`` subclass (or an
                  instance) — H1.
    sub_family  : the ``CopulaVirt`` subclass (or an instance) nested inside
                  it — H0, one or two parameters. Must be one of the
                  nestings ``full_family`` registers above.
    uv          : (n, 2) pseudo-observations in (0, 1)².
    weights     : optional (n,) non-negative frequency weights.

    Raises
    ------
    ValueError for a ``(full_family, sub_family)`` pair this module does not
    implement, for a ``sub_family`` with more than two parameters, or when
    either pseudo-likelihood is not finite at any evaluated parameter value.
    """
    from scipy.stats import chi2, kendalltau

    from pmcprg.copulas._fit import _fit_two_parameter_mle

    full_cls = full_family if isinstance(full_family, type) else type(full_family)
    sub_cls = sub_family if isinstance(sub_family, type) else type(sub_family)
    key = (full_cls.__name__, sub_cls.__name__)
    if key not in _SUBMODEL_NESTING:
        raise ValueError(
            f"{sub_cls.__name__} is not an implemented sub-model of {full_cls.__name__}; "
            f"the pairs this module implements are {sorted(_SUBMODEL_NESTING)}.")
    note = _SUBMODEL_NESTING[key]

    uv, w, weighted = _prepare(uv, weights)
    w_arg = w if weighted else None

    full_entry = _registry_entry(full_cls)
    sub_entry = _registry_entry(sub_cls)
    tau0, _ = kendalltau(uv[:, 0], uv[:, 1])
    tau_start = float(tau0) if np.isfinite(tau0) else 0.0
    pfit = _fit_two_parameter_mle(full_cls, full_entry, uv, w_arg, tau_start)
    if not pfit.converged:
        logger.warning("%s vs %s submodel LR test: the full model's joint MLE did not "
                       "converge (%s); using the best point found %s.",
                       full_cls.__name__, sub_cls.__name__, pfit.message, pfit.params)
    ll_full = float(pfit.log_likelihood)
    if not np.isfinite(ll_full):
        raise ValueError(f"{full_cls.__name__}: the pseudo-likelihood is not finite at any "
                         "evaluated (θ, δ)/(τ, extra) point.")

    # The sub-model itself can be a *two*-parameter family (Tawn1/2's own
    # psi, FR-9's Tawn-3 round): a one-parameter profile over tau alone
    # would silently leave psi at whatever default the constructor picks,
    # understating ll_sub and biasing the statistic upward. Dispatch on the
    # registry the same way _two_parameter_spec/_fit_two_parameter_mle do —
    # by how many parameters the sub-model actually has — and reuse the
    # identical joint MLE the full model above is fitted with, not a
    # bespoke optimiser.
    n_sub_params = len(sub_entry.value.PARAMETERS_SET_NAME)
    if n_sub_params == 1:
        tau_sub, ll_sub = _fit_one_parameter_profile(sub_cls, uv, w_arg)
        sub_params = {"tau_k": float(tau_sub)}
    elif n_sub_params == 2:
        sfit = _fit_two_parameter_mle(sub_cls, sub_entry, uv, w_arg, tau_start)
        if not sfit.converged:
            logger.warning("%s vs %s submodel LR test: the sub-model's joint MLE did not "
                           "converge (%s); using the best point found %s.",
                           full_cls.__name__, sub_cls.__name__, sfit.message, sfit.params)
        ll_sub = float(sfit.log_likelihood)
        sub_params = dict(sfit.params)
    else:
        raise ValueError(f"{sub_cls.__name__} has {n_sub_params} parameters; "
                         "submodel_lr_test only fits a one- or two-parameter sub-model.")
    if not np.isfinite(ll_sub):
        raise ValueError(f"{sub_cls.__name__}: the pseudo-likelihood is not finite at any "
                         "evaluated point.")

    stat = max(2.0 * (ll_full - ll_sub), 0.0)
    p_value = 1.0 if stat <= 0.0 else 0.5 * float(chi2.sf(stat, 1))

    return SubmodelLRTest(
        family=full_cls.__name__, submodel=sub_cls.__name__,
        statistic=float(stat), p_value=float(p_value),
        null_distribution="0.5*chi2(0) + 0.5*chi2(1)", boundary=True, boundary_note=note,
        full_params=dict(pfit.params), sub_params=sub_params,
        log_likelihood_full=ll_full, log_likelihood_sub=float(ll_sub),
        n_obs=int(uv.shape[0]), n_eff=float(w.sum()),
    )


# ---------------------------------------------------------------------------
# MLE-vs-τ discrepancy diagnostic (module docstring, "MLE-vs-τ discrepancy
# diagnostic"; audit AUDIT_COPULES FR-7 b)
# ---------------------------------------------------------------------------

def _theta_at(cls, entry, tau: float) -> float:
    """The family's native parameter at ``tau``, or NaN when τ is outside its range."""
    lo, hi = entry.constructible_tau_range()
    try:
        return float(cls(tau_k=float(min(max(tau, lo), hi))).theta)
    except Exception:
        return float("nan")


def mle_tau_discrepancy_test(family, uv, weights=None) -> MleTauDiscrepancyTest:
    """Hausman-style test that the pseudo-MLE and Kendall's τ̂ agree (FR-7 b).

    ``H0``: the family is correctly specified and the sample uncontaminated,
    so both estimators are consistent for the same τ and
    ``τ̂_MLE − τ̂_τ = O_p(n^{-1/2})`` is centred at 0. Kendall's τ̂ has a
    bounded influence function; the pseudo-likelihood score does not (Croux &
    Dehon 2010), so a few contaminating pairs move ``τ̂_MLE`` and leave
    ``τ̂_τ`` alone, and the standardised difference grows.

    Both estimates are recomputed here from ``uv`` — the pseudo-MLE by the
    same bounded Brent search over the padded τ-range as
    :meth:`CopulaVirt.fit(method='mle')`, Kendall's τ̂ through the (weighted)
    empirical copula in the U-statistic form of :func:`_tau_influence`, which
    with unit weights and distinct pseudo-observations equals
    :func:`scipy.stats.kendalltau` exactly. The standard error of their
    **difference** comes from the two
    influence functions evaluated on the same sample, *not* from a difference
    of variances: see the module docstring, "MLE-vs-τ discrepancy
    diagnostic", for why the Hausman shortcut is not available here and what
    is computed instead.

    Parameters
    ----------
    family  : a one-parameter ``CopulaVirt`` subclass (or an instance of one).
    uv      : (n, 2) pseudo-observations in (0, 1)² — ``FitResult.uv``.
    weights : optional (n,) non-negative frequency weights (module docstring,
              "Weights").

    Returns
    -------
    MleTauDiscrepancyTest — ``statistic`` is N(0, 1) under H0, ``p_value`` is
    two-sided. At a boundary of the parameter space the statistic, its
    standard error, the correlation and the p-value are NaN (``at_boundary``
    set, ``boundary`` saying why): the pseudo-MLE is not asymptotically
    normal there, so the contrast has no reference distribution.

    Raises
    ------
    ValueError          for a family with no free parameter, an unusable
                        ``uv``/``weights``, or a pseudo-likelihood that is not
                        finite anywhere the search looked.
    NotImplementedError for a two-parameter family: Kendall's τ does not
                        identify its second parameter, exactly as for
                        :func:`standard_errors` with ``method='tau'``.
    """
    from scipy.stats import norm

    cls = family if isinstance(family, type) else type(family)
    entry = _registry_entry(cls)
    if len(entry.value.PARAMETERS_SET_NAME) != 1:
        raise NotImplementedError(
            f"{cls.__name__} has more than one parameter: Kendall's τ does not identify "
            f"its second parameter, so there is no τ-inversion estimate to contrast the "
            f"pseudo-MLE with.")
    a, b = (float(t) for t in entry.value.TAU_MIN_MAX)
    if b - a < 1e-8:
        raise ValueError(f"{cls.__name__} has no free parameter.")

    uv, w, weighted = _prepare(uv, weights)
    sw = float(w.sum())
    w_arg = w if weighted else None

    tau_mle, _ = _fit_one_parameter_profile(cls, uv, w_arg)
    b_inf, tau_hat = _tau_influence(uv, w, weighted)
    if not np.isfinite(tau_hat):
        raise ValueError(f"{cls.__name__}: Kendall's τ is undefined on these "
                         "pseudo-observations.")

    spec = _spec_of(cls(tau_k=float(tau_mle)))
    theta_mle = float(spec.estimate["theta"])
    theta_tau = _theta_at(cls, entry, tau_hat)
    diff = float(tau_mle) - float(tau_hat)
    nan = float("nan")
    common = dict(
        family=cls.__name__, difference=diff, tau_mle=float(tau_mle), tau_tau=float(tau_hat),
        theta_mle=theta_mle, theta_tau=theta_tau,
        se_tau_tau=float(math.sqrt(float(w @ (b_inf * b_inf)) / sw / sw)),
        boundary=spec.boundary, n_obs=int(uv.shape[0]), n_eff=sw,
    )
    if spec.boundary:
        return MleTauDiscrepancyTest(statistic=nan, p_value=nan, se_difference=nan,
                                     se_tau_mle=nan, correlation=nan, at_boundary=True,
                                     **common)

    got = _score_and_information(spec, uv, w, True, "MLE-vs-tau discrepancy")
    if got is None:
        return MleTauDiscrepancyTest(statistic=nan, p_value=nan, se_difference=nan,
                                     se_tau_mle=nan, correlation=nan, at_boundary=False,
                                     **common)
    M, B = got
    # Influence function of τ̂_MLE: B⁻¹ φ in ψ, carried to τ by dτ/dψ.
    a_inf = float(spec.jac_psi[0, 0]) * M[:, 0] / float(B[0, 0])
    a_inf = a_inf - float(w @ a_inf) / sw
    d = a_inf - b_inf
    var_d = float(w @ (d * d)) / sw / sw
    var_a = float(w @ (a_inf * a_inf)) / sw / sw
    var_b = float(w @ (b_inf * b_inf)) / sw / sw
    cov_ab = float(w @ (a_inf * b_inf)) / sw / sw
    denom = math.sqrt(var_a * var_b)
    corr = cov_ab / denom if denom > 0.0 else nan
    if not (var_d > 0.0) or not np.isfinite(var_d):
        logger.warning("%s MLE-vs-tau discrepancy: the variance of the difference is not "
                       "positive (%r); returning NaN.", cls.__name__, var_d)
        return MleTauDiscrepancyTest(statistic=nan, p_value=nan, se_difference=nan,
                                     se_tau_mle=float(math.sqrt(var_a)), correlation=corr,
                                     at_boundary=False, **common)
    se_d = math.sqrt(var_d)
    z = diff / se_d
    return MleTauDiscrepancyTest(
        statistic=float(z), p_value=float(2.0 * norm.sf(abs(z))), se_difference=float(se_d),
        se_tau_mle=float(math.sqrt(var_a)), correlation=float(corr), at_boundary=False,
        **common)
