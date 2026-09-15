"""
model_selection.py — is the choice of a copula family statistically significant?

Why this exists
---------------
Every selection rule of the package (``fit_best``'s AIC ranking, the ICE
criteria ``mle`` / ``aic`` / ``bic`` / ``huard`` / ``cvm`` / ``xvcic``) returns
*one* family, even when the runner-up is behind by a tenth of a nat. This
module says whether the gap is larger than its sampling noise, and **declares
a tie** when it is not (audit FR-6).

Two tests for non-nested model comparison are provided, both on (weighted)
pseudo-observations ``(u_n, v_n)`` with weights ``w_n``:

* :func:`vuong_test` — Vuong (1989). The pointwise log-density differences
  ``m_n = log c_A(u_n, v_n; θ̂_A) − log c_B(u_n, v_n; θ̂_B)`` at each family's
  fitted parameters have mean zero when both families are equally close (in
  Kullback–Leibler divergence) to the truth. The statistic is

      Z = √n_eff · (m̄_w − K / n_eff) / σ̂,     m̄_w = Σ w_n m_n / Σ w_n,

  with ``K`` an optional parameter-count correction and ``σ̂²`` a
  heteroskedasticity-and-autocorrelation-consistent (HAC) long-run variance
  of ``m`` (below). ``Z → N(0, 1)`` under the null; ``|Z| > z_{1−α/2}`` favours
  the family the sign points to, otherwise the two are **tied** at level α.
* :func:`clarke_test` — Clarke's (2007) distribution-free sign test: under
  the null the *median* of ``m_n`` is zero, so the (weighted) share of
  positive ``m_n`` is 1/2. Exact binomial for i.i.d. frequency weights, normal
  approximation with a HAC variance otherwise. Clarke (2007) argues it is
  more efficient than Vuong's when the ``m_n`` are strongly peaked; read the
  caveat below on estimated parameters before using it.

On top of the pairwise tests:

* :func:`comparison_matrix` — every pair of a candidate set;
* :func:`confidence_set` — the families **not significantly worse** than the
  best one (the reference family is kept by construction);
* :func:`fit_best_comparison` / :func:`fit_best_confidence_set` — the same
  from the result of :meth:`pmcprg.copulas.CopulaVirt.fit_best`, also exposed as
  ``FitBestResults.compare()`` and ``FitBestResults.confidence_set()``;
* :func:`ice_pair_comparisons` — for each pair of states ``(i, j)`` of a
  fitted chain, the comparison of the ICE candidates on the ξ-weighted
  pseudo-observations ``(F_ij(y_n), F_ji(y_{n+1}))``, ``w_n = ξ_n(i, j)``
  (A16 Eq. 12), with the family the ICE criterion selects and its runner-up.

Nothing here changes a default or a selection decision: these functions are
read-only diagnostics, opt-in.

Weights and effective sample size
---------------------------------
The package's effective sample size of a ξ-weighted block is ``n_eff = Σ w``
(``pmcprg.pmc.ice._effective_n``, audit K-4/K-5), and so is the default here
(``n_eff="sum"``): weights are treated as **frequency weights** — integer
weights give exactly the test on the data repeated that many times. The long-
run variance is then that of the series ``√w_n · (m_n − m̄_w)``:

    σ̂² = [Σ_{|h| ≤ L} k(h) Σ_n √(w_n w_{n+h}) d_n d_{n+h}] / Σ w,   d_n = m_n − m̄_w,

with Bartlett weights ``k(h) = 1 − |h|/(L + 1)``. For 0/1 weights (a hard
labelling) this is exactly the Newey–West variance of the labelled subsample
kept at its time positions, and at lag 0 ``Σ w d²`` is the variance of the
hard-label sum ``Σ 1_n m_n`` because ``E[1_n²] = E[1_n] = ξ_n`` — the
posterior mass is the expected count.

``n_eff="kish"`` uses ``(Σw)²/Σw²`` and the series ``w_n · d_n`` instead (the
variance of ``Σ w_n m_n`` for *fixed* weights). For weights in ``[0, 1]``,
``Σw ≤ Kish``, so the default is the more conservative of the two: it
declares more ties. With uniform weights drawn independently of the data
(an exact null, N = 1000, 1000 replicates) Kish rejects 5.0 % of the time and
``Σw`` 2.3 %. Neither accounts for the randomness of the posterior weights
themselves, nor for their dependence on the data in ICE.

Serial dependence (HAC)
-----------------------
Consecutive pairs ``(y_n, y_{n+1})`` and ``(y_{n+1}, y_{n+2})`` share
``y_{n+1}``, and the hidden chain adds longer-range dependence, so the
``m_n`` are not independent (audit FR-5). The variance is a Newey & West
(1987) Bartlett-kernel estimator, which is positive semi-definite by
construction. ``bandwidth="auto"`` (the default) is the Newey & West (1994)
plug-in for the Bartlett kernel, without prewhitening, computed on the
weighted series — ``L = ⌊γ̂ T^{1/3}⌋``, ``γ̂ = 1.1447 (ŝ₁/ŝ₀)^{2/3}`` from
``⌊4 (T/100)^{2/9}⌋`` pilot autocovariances — and **at least 1**, since
consecutive pairs share an observation by construction. ``bandwidth=0``
gives the i.i.d. variance of the original papers; any integer fixes ``L``.
Zero-weight points stay in the series as zeros, so lags are measured in time.
Measured on an exact null with strongly autocorrelated ``m_n`` (a copula
Markov chain switching between Clayton and survival Clayton with a hidden
regime of persistence 0.9, T = 2000, 1000 replicates): the i.i.d. variance
rejects 21.5 % (Vuong) and 14.4 % (Clarke) of the time at 5 %, the automatic
HAC 6.7 % and 5.5 % (median L = 18).

Parameter-count corrections
---------------------------
``correction="none"`` — the plain likelihood ratio; ``"akaike"`` —
``K = k_A − k_B``, consistent with AIC (Vuong 1989); ``"schwarz"`` —
``K = (k_A − k_B)/2 · log n_eff``, consistent with BIC. The corrected Vuong
numerator ``Σ w m − K`` is half the difference of the package's own AIC/BIC
(``ice._score_aic``/``_score_bic``, ``FitResult.aic``/``bic``), so the sign of
the statistic agrees with the ranking by the same criterion. Clarke (2007)
spreads the same ``K`` evenly over the observations, ``m_n − K / n_eff``,
before taking signs.

What the tests do not account for (caveats)
-------------------------------------------
The rates quoted below were measured with fixed seeds, at α = 5 %, on an
exact non-nested null unless stated otherwise — Clayton vs survival Clayton,
both fitted by maximum likelihood, on Gaussian-copula data (τ = 0.5), which is
radially symmetric so the two families are equally close to the truth (1000
replicates). ``pmcprg/tests/test_model_selection_mc.py`` re-runs the designs and
bounds the rates.

* **Estimated pseudo-observations.** The ``(u, v)`` are not observed: they
  are ranks (``fit_best``) or come from estimated parametric margins (ICE
  with ``fit_margins``). Chen & Fan (2006) study exactly this setting —
  pseudo-likelihood ratio tests between copula-based models under
  misspecification — and show that the estimation of the margins adds a term
  to the asymptotic variance of the statistic. **That term is ignored here**:
  the variance is the one Vuong derived for observed data. Measured: Vuong
  rejects 4.8 % (i.i.d. variance) / 5.1 % (HAC) of the time with the true
  uniform margins (N = 500), but 8.7–11.3 % on rank pseudo-observations
  (N = 500 and 2000; the standard deviation of Z is 1.18 and 1.14 instead of
  1). A "significant" Vuong decision on
  rank pseudo-observations near the 5 % boundary is therefore weaker than
  its p-value says; a tie is not affected in that direction. ICE with fixed
  margins (``fit_margins = False``) has no margin-estimation effect.
* **Estimated copula parameters — Vuong.** Vuong's variance ignores the
  estimation of ``θ̂`` because, at the maximum-likelihood pseudo-true value,
  the score has mean zero. That argument fails for ``fit(method='tau')``
  (τ-inversion is not the likelihood maximiser), whose estimation noise is
  then ignored too.
* **Estimated copula parameters — Clarke.** The share of positive ``m_n`` is
  not smooth in ``θ̂``: when the two fitted densities are close, ``m_n``
  concentrates near 0 and the ``O(n^{-1/2})`` estimation error of ``θ̂``
  moves the share by as much as its own sampling noise. Clarke's binomial
  law ignores it. Measured on the null above: **66–72 % rejections at a
  nominal 5 %** (N = 500–2000), even with the true margins, while the same
  test at the true parameters rejects 3.8 %. Use Clarke's test for
  families that are far apart, or as a complement; do not rely on it to
  declare that two close families differ.
* **Posterior weights.** ``ξ`` is treated as known; the ICE estimation of the
  chain (``p``, the margins) is not propagated.
* **Overlapping or nested families.** When both fitted families describe
  the data equally (Gaussian vs Student with ν̂ at its upper bound, a family
  vs itself), ``ω² ≈ 0`` and the normal limit degenerates; Vuong's (1989)
  procedure for overlapping models first tests ``ω² = 0``, which needs the
  scores and Hessians of both models and is **not implemented**. Without a
  correction the statistic stays near N(0, 1) or below, so the test declares
  a tie (measured: 5.0 % rejections, Gaussian vs Student both fitted on
  Gaussian data, N = 1000, 200 replicates, ν̂ = 100 in 59 % of them).
  **With a correction**, the numerator carries the deterministic
  ``−K/n_eff`` while ``σ̂ → 0``, so the smaller family is declared
  significantly better (63.5 % of the time with ``"akaike"``, 86 % with
  ``"schwarz"`` in the same design): this is the consistency of the
  corrected procedure for nested models (Vuong 1989) — it means "same
  fit, fewer parameters", not "better fit". Read the uncorrected test for
  the evidence in the data. An exactly zero variance is reported as a tie.
* **Multiplicity.** :func:`confidence_set` makes ``m − 1`` comparisons with
  the best family. ``adjust="holm"`` or ``"bonferroni"`` controls the
  family-wise error of the exclusions; ``adjust=None`` (default) does not.
  Neither is the model confidence set of Hansen, Lunde & Nason (2011).
* **Clarke's null is a median null.** With a tiny parameter perturbation of
  the same family, the mean of ``m`` is ~0 but its median need not be, and
  the sign test may reject where Vuong does not.

Conventions of the results
--------------------------
``statistic > 0`` favours ``A`` (the first argument); ``p_value`` is two-
sided; ``decision`` is ``"A"``, ``"B"``, ``"tie"`` (not significant at
``alpha``) or ``"undetermined"`` (both likelihoods are ``−∞`` somewhere, or a
log-density is NaN). A log-density of ``−∞`` at a point of positive weight
for one family only makes the other family strictly preferred (statistic
``±∞``, p-value 0), following :func:`pmcprg.copulas._fit._weighted_log_density_sum`
(audit RB-7); points of zero weight are ignored whatever their log-density.
The results are antisymmetric: ``test(A, B).statistic == −test(B, A).statistic``.

Related software: the R packages GJRM and copBasic expose Vuong and Clarke
tests for copula selection (audit FR-6).

References
----------
* Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
  non-nested hypotheses. *Econometrica* 57(2), 307–333. doi:10.2307/1912557
* Clarke, K. A. (2007). A simple distribution-free test for nonnested model
  selection. *Political Analysis* 15(3), 347–363. doi:10.1093/pan/mpm004
* Chen, X. & Fan, Y. (2006). Estimation and model selection of
  semiparametric copula-based multivariate dynamic models under copula
  misspecification. *Journal of Econometrics* 135(1–2), 125–154.
  doi:10.1016/j.jeconom.2005.07.027
* Newey, W. K. & West, K. D. (1987). A simple, positive semi-definite,
  heteroskedasticity and autocorrelation consistent covariance matrix.
  *Econometrica* 55(3), 703–708. doi:10.2307/1913610
* Newey, W. K. & West, K. D. (1994). Automatic lag selection in covariance
  matrix estimation. *Review of Economic Studies* 61(4), 631–653.
  doi:10.2307/2297912
* Hansen, P. R., Lunde, A. & Nason, J. M. (2011). The model confidence set.
  *Econometrica* 79(2), 453–497. doi:10.3982/ECTA5771
* Derrode, S. & Pieczynski, W. (2013). Unsupervised data classification using
  pairwise Markov chains with automatic copulas selection. *Comput. Statist.
  Data Anal.* 63, 81–98. doi:10.1016/j.csda.2013.01.027 (A16) — Eq. 12, the
  pseudo-observations of the pair copulas.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import stats

__all__ = [
    "CORRECTIONS",
    "N_EFF_KINDS",
    "TESTS",
    "ComparisonMatrix",
    "ComparisonResult",
    "ConfidenceSet",
    "PairComparison",
    "clarke_test",
    "comparison_matrix",
    "confidence_set",
    "fit_best_comparison",
    "fit_best_confidence_set",
    "hac_variance",
    "ice_pair_comparisons",
    "newey_west_bandwidth",
    "vuong_test",
]

#: Parameter-count corrections (Vuong 1989; Clarke 2007).
CORRECTIONS: tuple[str, ...] = ("none", "akaike", "schwarz")
#: Effective-sample-size conventions (see the module docstring).
N_EFF_KINDS: tuple[str, ...] = ("sum", "kish")
#: Available tests.
TESTS: tuple[str, ...] = ("vuong", "clarke")

#: The correction consistent with each ICE selection criterion. The
#: non-likelihood criteria have no parameter-count counterpart.
CRITERION_CORRECTION: dict[str, str] = {"mle": "none", "aic": "akaike", "bic": "schwarz"}

# A long-run standard deviation below this fraction of the log-density scale
# is floating-point noise: the two fitted densities are the same function.
_DEGENERATE_SD_REL = 1e-12

# ICE skips the copula of a pair whose total posterior weight is below this
# (``pmcprg.pmc.ice._m_step``); the diagnostic does the same.
_ICE_MIN_PAIR_WEIGHT = 1e-12


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of one pairwise test of family ``A`` against family ``B``.

    Attributes
    ----------
    test       : ``"vuong"`` or ``"clarke"``.
    names      : ``(name_A, name_B)``.
    statistic  : standardised statistic, ``> 0`` favours ``A``. For Clarke's
                 exact test, ``(B − N/2)/√(N/4)`` with ``B`` the count.
    p_value    : two-sided p-value.
    alpha      : level of the decision.
    decision   : ``"A"``, ``"B"``, ``"tie"`` or ``"undetermined"``.
    method     : ``"normal"``, ``"exact"`` (binomial), ``"infinite"`` (one
                 likelihood is ``−∞``), ``"degenerate"`` (zero variance or
                 no informative point), ``"undetermined"``.
    estimate   : Vuong — weighted mean ``m̄_w`` of the *uncorrected*
                 differences (nat per unit weight); Clarke — weighted share of
                 positive corrected differences.
    penalty    : total correction ``K`` subtracted from ``Σ w m``.
    sd         : long-run standard deviation ``σ̂`` (NaN when undefined).
    n_eff      : effective sample size (``Σw`` or Kish).
    sum_weights: ``Σ w`` over the points that entered the test.
    bandwidth  : HAC lag truncation ``L`` actually used.
    correction, n_eff_kind : the options used.
    count, n_trials : Clarke only — weighted count of positive differences and
                 weighted number of non-zero differences.
    """
    test:        str
    names:       tuple[str, str]
    statistic:   float
    p_value:     float
    alpha:       float
    decision:    str
    method:      str
    estimate:    float
    penalty:     float
    sd:          float
    n_eff:       float
    sum_weights: float
    bandwidth:   int
    correction:  str
    n_eff_kind:  str
    count:       float = math.nan
    n_trials:    float = math.nan

    @property
    def tie(self) -> bool:
        """``True`` when the test does not separate the two families."""
        return self.decision == "tie"

    @property
    def preferred(self) -> str | None:
        """Name of the significantly preferred family, ``None`` otherwise."""
        if self.decision == "A":
            return self.names[0]
        if self.decision == "B":
            return self.names[1]
        return None

    def swapped(self) -> "ComparisonResult":
        """The same comparison read as ``B`` against ``A``."""
        swap = {"A": "B", "B": "A"}
        est = self.estimate
        if self.test == "vuong":
            est = -est
        elif np.isfinite(est):
            est = 1.0 - est
        count = self.count
        if np.isfinite(self.n_trials) and np.isfinite(count):
            count = self.n_trials - count
        return replace(
            self,
            names=(self.names[1], self.names[0]),
            statistic=-self.statistic,
            decision=swap.get(self.decision, self.decision),
            estimate=est,
            penalty=-self.penalty,
            count=count,
        )

    def __str__(self) -> str:
        a, b = self.names
        verdict = {"A": f"{a} preferred", "B": f"{b} preferred",
                   "tie": "tie", "undetermined": "undetermined"}[self.decision]
        return (f"{self.test} {a} vs {b}: statistic={self.statistic:+.3f}  "
                f"p={self.p_value:.4g}  → {verdict} at α={self.alpha:g} "
                f"(n_eff={self.n_eff:.1f}, L={self.bandwidth}, "
                f"correction={self.correction})")


@dataclass
class ComparisonMatrix:
    """All pairwise comparisons of a candidate set.

    ``statistic[a, b]`` is the statistic of ``names[a]`` against ``names[b]``
    (antisymmetric, 0 on the diagonal); ``p_value`` is symmetric (1 on the
    diagonal); ``decision[a][b]`` is the preferred family's name, ``"tie"`` or
    ``"undetermined"`` (``"—"`` on the diagonal). ``results[(A, B)]`` holds
    the :class:`ComparisonResult` for every ordered pair ``A ≠ B``.
    """
    names:     list[str]
    statistic: np.ndarray
    p_value:   np.ndarray
    decision:  list[list[str]]
    results:   dict[tuple[str, str], ComparisonResult]
    test:      str
    alpha:     float
    correction: str

    def result(self, a: str, b: str) -> ComparisonResult:
        return self.results[(a, b)]

    def ties(self) -> list[tuple[str, str]]:
        """Unordered pairs that the test does not separate."""
        out = []
        for x in range(len(self.names)):
            for y in range(x + 1, len(self.names)):
                if self.decision[x][y] == "tie":
                    out.append((self.names[x], self.names[y]))
        return out

    def __str__(self) -> str:
        w = max([len(n) for n in self.names] + [8])
        head = " " * w + "".join(f"{n:>{w + 2}}" for n in self.names)
        lines = [f"{self.test} statistics (row vs column, > 0 favours the row), "
                 f"α={self.alpha:g}, correction={self.correction}", head]
        for a, na in enumerate(self.names):
            cells = []
            for b in range(len(self.names)):
                if a == b:
                    cells.append(f"{'—':>{w + 2}}")
                else:
                    mark = {"tie": " ", "undetermined": "?"}.get(self.decision[a][b], "*")
                    cells.append(f"{self.statistic[a, b]:>{w + 1}.2f}{mark}")
            lines.append(f"{na:<{w}}" + "".join(cells))
        lines.append("* significant at α, ? undetermined")
        return "\n".join(lines)


@dataclass
class ConfidenceSet:
    """Families not significantly worse than a reference family.

    Attributes
    ----------
    best       : the reference family (kept by construction).
    members    : retained families, reference first, then by decreasing score.
    excluded   : families significantly worse than ``best``.
    scores     : the score used for the ordering (corrected weighted
                 log-likelihood, or the caller's scores).
    comparisons: ``{name: test(best, name)}`` for every other family.
    p_adjusted : the p-values after ``adjust`` (equal to the raw ones when
                 ``adjust`` is ``None``).
    """
    best:        str
    members:     list[str]
    excluded:    list[str]
    scores:      dict[str, float]
    comparisons: dict[str, ComparisonResult]
    p_adjusted:  dict[str, float]
    alpha:       float
    adjust:      str | None
    test:        str
    correction:  str

    def __contains__(self, name: str) -> bool:
        return name in self.members

    def __len__(self) -> int:
        return len(self.members)

    @property
    def is_tie(self) -> bool:
        """``True`` when at least one other family is retained with ``best``."""
        return len(self.members) > 1


@dataclass
class PairComparison:
    """Family comparison for one pair of states ``(i, j)`` of a fitted chain.

    Attributes
    ----------
    pair          : ``(i, j)``.
    model_family  : the family the fitted model carries for the pair.
    selected      : the family the ICE criterion selects on this weighted
                    sample (what the next M-step would keep; equal to
                    ``model_family`` at a converged run).
    runner_up     : second family by the criterion (``None`` with a single
                    usable candidate).
    criterion     : the ICE selection criterion used for the ranking.
    scores        : criterion score of every candidate (higher is better).
    log_likelihood: ``Σ ξ log c`` of every candidate at its fitted parameters.
    params        : fitted parameters of every candidate.
    n_params      : parameter count of every candidate.
    sum_weights   : ``Σ_n ξ_n(i, j)``.
    matrix        : :class:`ComparisonMatrix` over the usable candidates.
    vs_runner_up  : ``test(selected, runner_up)``.
    confidence_set: families not significantly worse than ``selected``.
    failed        : candidates whose fitted copula could not be evaluated.
    log_densities : ``{family: log c(u_n, v_n)}`` at the fitted parameters, on
                    the ``N − 1`` transitions — to run another test without
                    refitting, e.g. ``clarke_test(ld[a], ld[b], weights)``.
    weights       : ``ξ_n(i, j)``, shape ``(N − 1,)``.
    """
    pair:           tuple[int, int]
    model_family:   str
    selected:       str
    runner_up:      str | None
    criterion:      str
    scores:         dict[str, float]
    log_likelihood: dict[str, float]
    params:         dict[str, dict]
    n_params:       dict[str, int]
    sum_weights:    float
    matrix:         ComparisonMatrix | None
    vs_runner_up:   ComparisonResult | None
    confidence_set: ConfidenceSet | None
    failed:         list[str] = field(default_factory=list)
    log_densities:  dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    weights:        np.ndarray | None = field(default=None, repr=False)

    @property
    def tied_with_runner_up(self) -> bool:
        """``True`` when the selected family is not significantly better than
        the runner-up (``False`` when there is no runner-up)."""
        return self.vs_runner_up is not None and self.vs_runner_up.decision != "A"


# ---------------------------------------------------------------------------
# HAC variance
# ---------------------------------------------------------------------------

def _autocov_sums(z: np.ndarray, max_lag: int) -> np.ndarray:
    """``[Σ_t z_t z_{t+h} for h = 0 … max_lag]`` (sums, not means)."""
    T = z.size
    out = np.zeros(max_lag + 1)
    out[0] = float(np.dot(z, z))
    for h in range(1, min(max_lag, T - 1) + 1):
        out[h] = float(np.dot(z[:-h], z[h:]))
    return out


def newey_west_bandwidth(z: np.ndarray) -> int:
    """Newey & West (1994) automatic lag truncation for the Bartlett kernel.

    ``L = ⌊γ̂ T^{1/3}⌋`` with ``γ̂ = 1.1447 (ŝ₁/ŝ₀)^{2/3}``,
    ``ŝ₀ = σ̂₀ + 2 Σ_{j=1}^{n} σ̂_j``, ``ŝ₁ = 2 Σ_{j=1}^{n} j σ̂_j``, from the
    ``n = ⌊4 (T/100)^{2/9}⌋`` pilot autocovariances of ``z`` (no prewhitening;
    the same rule as ``sandwich::bwNeweyWest`` in R without it). Returns 0
    for a series too short or without variance; capped at ``T − 1``.

    ``z`` is used as given — centre it first.
    """
    z = np.asarray(z, dtype=float).ravel()
    T = z.size
    if T < 3:
        return 0
    n_pilot = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    n_pilot = max(1, min(n_pilot, T - 1))
    sig = _autocov_sums(z, n_pilot) / T
    j = np.arange(1, n_pilot + 1)
    s0 = sig[0] + 2.0 * float(np.sum(sig[1:]))
    s1 = 2.0 * float(np.sum(j * sig[1:]))
    if not (np.isfinite(s0) and np.isfinite(s1)) or s0 <= 0.0:
        return 0
    gamma = 1.1447 * ((s1 / s0) ** 2) ** (1.0 / 3.0)
    L = int(math.floor(gamma * T ** (1.0 / 3.0)))
    return int(max(0, min(L, T - 1)))


def hac_variance(z: np.ndarray, bandwidth: int) -> float:
    """Newey–West (1987) estimate of ``Var(Σ_t z_t)`` for a centred series.

    ``Σ_{|h| ≤ L} (1 − |h|/(L + 1)) Σ_t z_t z_{t+h}`` — non-negative for any
    ``z`` (Bartlett kernel). ``bandwidth = 0`` gives ``Σ z_t²``, the i.i.d.
    variance.
    """
    z = np.asarray(z, dtype=float).ravel()
    L = int(bandwidth)
    if L < 0:
        raise ValueError(f"bandwidth must be ≥ 0, got {bandwidth!r}.")
    L = min(L, max(z.size - 1, 0))
    g = _autocov_sums(z, L)
    if L == 0:
        return float(g[0])
    k = 1.0 - np.arange(1, L + 1) / (L + 1.0)
    return float(g[0] + 2.0 * np.dot(k, g[1:]))


def _resolve_bandwidth(bandwidth, z_for_selection: np.ndarray) -> int:
    if isinstance(bandwidth, str):
        if bandwidth != "auto":
            raise ValueError(f"bandwidth must be a non-negative int or 'auto', got {bandwidth!r}.")
        return max(1, newey_west_bandwidth(z_for_selection))
    if isinstance(bandwidth, (bool, np.bool_)) or not float(bandwidth).is_integer():
        raise ValueError(f"bandwidth must be a non-negative int or 'auto', got {bandwidth!r}.")
    L = int(bandwidth)
    if L < 0:
        raise ValueError(f"bandwidth must be ≥ 0, got {bandwidth!r}.")
    return L


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------

def _check_options(correction: str, n_eff: str, alpha: float) -> None:
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}.")
    if n_eff not in N_EFF_KINDS:
        raise ValueError(f"n_eff must be one of {N_EFF_KINDS}, got {n_eff!r}.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")


def _prepare(logc_a, logc_b, weights):
    """Validated arrays and the status of the non-finite values.

    Returns ``(la, lb, w, pos, status)`` where ``status`` is ``None`` (every
    positive-weight log-density finite), ``"A"`` / ``"B"`` (only the other
    family has a ``−∞``), or ``"undetermined"``.
    """
    la = np.asarray(logc_a, dtype=float).ravel()
    lb = np.asarray(logc_b, dtype=float).ravel()
    if la.shape != lb.shape:
        raise ValueError(f"log-densities must have the same length, got {la.shape} and {lb.shape}.")
    if la.size == 0:
        raise ValueError("empty log-density arrays.")
    if weights is None:
        w = np.ones_like(la)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != la.shape:
            raise ValueError(f"weights {w.shape} and log-densities {la.shape} must have the same length.")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative.")
    pos = w > 0.0
    if not np.any(pos):
        raise ValueError("all weights are zero.")
    a, b = la[pos], lb[pos]
    fin_a, fin_b = np.isfinite(a), np.isfinite(b)
    status = None
    if not (np.all(fin_a) and np.all(fin_b)):
        a_minf = np.isneginf(a)
        b_minf = np.isneginf(b)
        bad = (~fin_a & ~a_minf) | (~fin_b & ~b_minf)     # NaN or +inf
        if np.any(bad) or np.any(a_minf & b_minf) or (np.any(a_minf) and np.any(b_minf)):
            status = "undetermined"
        elif np.any(b_minf):
            status = "A"
        else:
            status = "B"
    return la, lb, w, pos, status


def _n_eff(w: np.ndarray, kind: str) -> float:
    s = float(np.sum(w))
    if kind == "sum":
        return s
    s2 = float(np.dot(w, w))
    return s * s / s2 if s2 > 0.0 else 0.0


def _penalty(k_a: float, k_b: float, correction: str, n_eff: float) -> float:
    if correction == "none":
        return 0.0
    if correction == "akaike":
        return float(k_a - k_b)
    return 0.5 * float(k_a - k_b) * math.log(max(n_eff, 1.0))


def _long_run_variance(d: np.ndarray, w: np.ndarray, kind: str, bandwidth,
                       d_select: np.ndarray | None = None) -> tuple[float, int]:
    """``(σ̂², L)`` of the per-unit-weight long-run variance of ``d``.

    ``d`` is zero at zero-weight points. ``d_select`` (default ``d``) is the
    series the automatic bandwidth is chosen on.
    """
    if kind == "sum":
        scale = np.sqrt(w)
        denom = float(np.sum(w))
    else:
        scale = w
        denom = float(np.dot(w, w))
    z = scale * d
    zs = z if d_select is None else scale * d_select
    L = _resolve_bandwidth(bandwidth, zs)
    return hac_variance(z, L) / denom, L


def _decide(statistic: float, p_value: float, alpha: float) -> str:
    if not np.isfinite(p_value):
        return "undetermined"
    if p_value <= alpha and statistic != 0.0:
        return "A" if statistic > 0.0 else "B"
    return "tie"


def _infinite_result(test, names, status, alpha, correction, n_eff_kind, w, pos) -> ComparisonResult:
    sw = float(np.sum(w[pos]))
    if status == "undetermined":
        return ComparisonResult(
            test=test, names=names, statistic=math.nan, p_value=math.nan,
            alpha=alpha, decision="undetermined", method="undetermined",
            estimate=math.nan, penalty=math.nan, sd=math.nan,
            n_eff=_n_eff(w, n_eff_kind), sum_weights=sw, bandwidth=0,
            correction=correction, n_eff_kind=n_eff_kind,
        )
    stat = math.inf if status == "A" else -math.inf
    return ComparisonResult(
        test=test, names=names, statistic=stat, p_value=0.0, alpha=alpha,
        decision=status, method="infinite", estimate=stat, penalty=math.nan,
        sd=math.nan, n_eff=_n_eff(w, n_eff_kind), sum_weights=sw, bandwidth=0,
        correction=correction, n_eff_kind=n_eff_kind,
    )


# ---------------------------------------------------------------------------
# Vuong
# ---------------------------------------------------------------------------

def vuong_test(
    logc_a,
    logc_b,
    weights=None,
    *,
    k_a: float = 1,
    k_b: float = 1,
    correction: str = "none",
    alpha: float = 0.05,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    names: tuple[str, str] = ("A", "B"),
) -> ComparisonResult:
    """Weighted Vuong (1989) test of family ``A`` against family ``B``.

    Parameters
    ----------
    logc_a, logc_b : pointwise log-densities ``log c_A(u_n, v_n; θ̂_A)`` and
                     ``log c_B(u_n, v_n; θ̂_B)`` at each family's fitted
                     parameters, shape ``(N,)``.
    weights        : ``w_n ≥ 0`` (ICE: ``ξ_n(i, j)``); ``None`` for unit weights.
    k_a, k_b       : parameter counts, used by the correction.
    correction     : ``"none"``, ``"akaike"`` or ``"schwarz"``.
    alpha          : level of the three-way decision.
    bandwidth      : HAC lag truncation ``L`` (int ≥ 0) or ``"auto"``
                     (Newey–West 1994, at least 1).
    n_eff          : ``"sum"`` (``Σw``, the package convention) or ``"kish"``.
    names          : labels of the two families in the result.

    Returns
    -------
    ComparisonResult — ``Z = √n_eff (m̄_w − K/n_eff) / σ̂``, two-sided normal
    p-value, decision ``"A"`` / ``"B"`` / ``"tie"``. See the module docstring
    for the variance, the corrections and the caveats (margins estimated,
    Chen & Fan 2006; overlapping families).
    """
    _check_options(correction, n_eff, alpha)
    names = (str(names[0]), str(names[1]))
    la, lb, w, pos, status = _prepare(logc_a, logc_b, weights)
    if status is not None:
        return _infinite_result("vuong", names, status, alpha, correction, n_eff, w, pos)

    m = np.zeros_like(la)
    m[pos] = la[pos] - lb[pos]
    sw = float(np.sum(w))
    mean = float(np.dot(w, m)) / sw
    ne = _n_eff(w, n_eff)
    pen = _penalty(k_a, k_b, correction, ne)
    d = np.zeros_like(m)
    d[pos] = m[pos] - mean
    var, L = _long_run_variance(d, w, n_eff, bandwidth)
    sd = math.sqrt(max(var, 0.0))
    scale = max(1.0, float(np.max(np.abs(la[pos]))), float(np.max(np.abs(lb[pos]))))
    common = dict(test="vuong", names=names, alpha=alpha, estimate=mean,
                  penalty=pen, sd=sd, n_eff=ne, sum_weights=sw, bandwidth=L,
                  correction=correction, n_eff_kind=n_eff)
    if not sd > _DEGENERATE_SD_REL * scale:
        # ω̂² = 0: the two fitted densities coincide on the sample — the
        # models are observationally equivalent there (Vuong 1989).
        return ComparisonResult(statistic=0.0, p_value=1.0, decision="tie",
                                method="degenerate", **common)
    stat = math.sqrt(ne) * (mean - pen / ne) / sd
    p = float(2.0 * stats.norm.sf(abs(stat)))
    return ComparisonResult(statistic=float(stat), p_value=p,
                            decision=_decide(stat, p, alpha), method="normal",
                            **common)


# ---------------------------------------------------------------------------
# Clarke
# ---------------------------------------------------------------------------

def clarke_test(
    logc_a,
    logc_b,
    weights=None,
    *,
    k_a: float = 1,
    k_b: float = 1,
    correction: str = "none",
    alpha: float = 0.05,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    method: str = "auto",
    names: tuple[str, str] = ("A", "B"),
) -> ComparisonResult:
    """Weighted distribution-free sign test of Clarke (2007).

    ``H₀: median(m_n) = 0``, with the corrected differences
    ``m_n − K / n_eff`` (Clarke 2007). Points whose corrected
    difference is exactly zero carry no sign and are dropped, as in the
    classical sign test.

    Parameters
    ----------
    As :func:`vuong_test`, plus

    method : ``"exact"`` — binomial ``B ~ Bin(N, 1/2)`` on the weighted count
             ``B = Σ w 1{m > 0}`` of ``N = Σ w`` trials; valid only with
             integer (frequency) weights, ``bandwidth=0`` and ``n_eff="sum"``.
             ``"normal"`` — ``Z = √n_eff (p̂ − 1/2) / σ̂``, ``p̂ = B/N``, with
             the HAC variance of ``1{m > 0} − 1/2`` (centred at the null value,
             so that ``bandwidth=0`` gives ``σ̂² = 1/4``, the binomial
             variance; the automatic bandwidth is chosen on the series centred
             at ``p̂``). ``"auto"`` — ``"exact"`` when its conditions hold,
             ``"normal"`` otherwise.

    The binomial law assumes independent signs: with serially dependent pairs
    (ICE) use the default HAC.
    """
    _check_options(correction, n_eff, alpha)
    if method not in ("auto", "exact", "normal"):
        raise ValueError(f"method must be 'auto', 'exact' or 'normal', got {method!r}.")
    names = (str(names[0]), str(names[1]))
    la, lb, w, pos, status = _prepare(logc_a, logc_b, weights)
    if status is not None:
        return _infinite_result("clarke", names, status, alpha, correction, n_eff, w, pos)

    ne_all = _n_eff(w, n_eff)
    pen = _penalty(k_a, k_b, correction, ne_all)
    mc = np.zeros_like(la)
    mc[pos] = (la[pos] - lb[pos]) - pen / ne_all
    informative = pos & (mc != 0.0)
    wi = np.where(informative, w, 0.0)
    n_trials = float(np.sum(wi))
    positive = informative & (mc > 0.0)
    count = float(np.sum(w[positive]))
    common = dict(test="clarke", names=names, alpha=alpha, penalty=pen,
                  sum_weights=float(np.sum(w)), correction=correction,
                  n_eff_kind=n_eff, count=count, n_trials=n_trials)
    if n_trials <= 0.0:
        return ComparisonResult(statistic=0.0, p_value=1.0, decision="tie",
                                method="degenerate", estimate=math.nan,
                                sd=math.nan, n_eff=0.0, bandwidth=0, **common)
    share = count / n_trials
    integer_w = bool(np.all(wi == np.round(wi)))
    L_fixed = None if isinstance(bandwidth, str) else _resolve_bandwidth(bandwidth, mc)
    exact_ok = integer_w and L_fixed == 0 and n_eff == "sum"
    if method == "exact" and not exact_ok:
        raise ValueError("clarke_test(method='exact') needs integer weights, "
                         "bandwidth=0 and n_eff='sum'.")
    if method == "exact" or (method == "auto" and exact_ok):
        N = int(round(n_trials))
        B = int(round(count))
        p = float(stats.binomtest(B, N, 0.5).pvalue)
        stat = (B - 0.5 * N) / math.sqrt(0.25 * N)
        return ComparisonResult(statistic=float(stat), p_value=p,
                                decision=_decide(stat, p, alpha), method="exact",
                                estimate=share, sd=0.5, n_eff=float(N),
                                bandwidth=0, **common)

    s = positive.astype(float)
    d_null = np.where(informative, s - 0.5, 0.0)
    d_sel = np.where(informative, s - share, 0.0)
    var, L = _long_run_variance(d_null, wi, n_eff, bandwidth, d_select=d_sel)
    ne = _n_eff(wi, n_eff)
    sd = math.sqrt(max(var, 0.0))
    if not sd > 0.0:
        return ComparisonResult(statistic=0.0, p_value=1.0, decision="tie",
                                method="degenerate", estimate=share, sd=sd,
                                n_eff=ne, bandwidth=L, **common)
    stat = math.sqrt(ne) * (share - 0.5) / sd
    p = float(2.0 * stats.norm.sf(abs(stat)))
    return ComparisonResult(statistic=float(stat), p_value=p,
                            decision=_decide(stat, p, alpha), method="normal",
                            estimate=share, sd=sd, n_eff=ne, bandwidth=L, **common)


_TEST_FN = {"vuong": vuong_test, "clarke": clarke_test}


def _test_fn(test: str):
    try:
        return _TEST_FN[test]
    except KeyError:
        raise ValueError(f"test must be one of {TESTS}, got {test!r}.") from None


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------

def _as_mapping(log_densities, n_params):
    if not isinstance(log_densities, Mapping):
        raise TypeError("log_densities must be a mapping {name: log-density array}.")
    names = [str(k) for k in log_densities]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate family names: {names}.")
    arrays = {str(k): np.asarray(v, dtype=float).ravel() for k, v in log_densities.items()}
    if n_params is None:
        k = {n: 1 for n in names}
    else:
        missing = [n for n in names if n not in n_params]
        if missing:
            raise ValueError(f"n_params is missing {missing}.")
        k = {n: n_params[n] for n in names}
    return names, arrays, k


def comparison_matrix(
    log_densities: Mapping[str, np.ndarray],
    weights=None,
    *,
    n_params: Mapping[str, float] | None = None,
    test: str = "vuong",
    correction: str = "none",
    alpha: float = 0.05,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    **test_kwargs,
) -> ComparisonMatrix:
    """Pairwise tests between all families of a candidate set.

    Parameters
    ----------
    log_densities : ``{name: log c_name(u_n, v_n; θ̂_name)}``, every array on
                    the same ``N`` points, in the order of the output.
    weights       : shared weights ``w_n`` (``None``: unit).
    n_params      : ``{name: k}`` for the corrections (default 1 each).
    test          : ``"vuong"`` or ``"clarke"``.
    other options : forwarded to the test.

    Each unordered pair is tested once; the lower triangle is the swapped
    result, so the matrix is exactly antisymmetric.
    """
    fn = _test_fn(test)
    names, arrays, k = _as_mapping(log_densities, n_params)
    M = len(names)
    statistic = np.zeros((M, M))
    p_value = np.ones((M, M))
    decision = [["—"] * M for _ in range(M)]
    results: dict[tuple[str, str], ComparisonResult] = {}
    for a in range(M):
        for b in range(a + 1, M):
            na, nb = names[a], names[b]
            r = fn(arrays[na], arrays[nb], weights, k_a=k[na], k_b=k[nb],
                   correction=correction, alpha=alpha, bandwidth=bandwidth,
                   n_eff=n_eff, names=(na, nb), **test_kwargs)
            rs = r.swapped()
            results[(na, nb)] = r
            results[(nb, na)] = rs
            statistic[a, b], statistic[b, a] = r.statistic, rs.statistic
            p_value[a, b] = p_value[b, a] = r.p_value
            label = r.preferred if r.preferred is not None else r.decision
            decision[a][b] = decision[b][a] = label
    return ComparisonMatrix(names=names, statistic=statistic, p_value=p_value,
                            decision=decision, results=results, test=test,
                            alpha=alpha, correction=correction)


def _corrected_scores(arrays, weights, k, correction, n_eff):
    """``Σ w log c − k_pen`` per family, with the penalty of ``correction``
    in log-likelihood units (AIC/2, BIC/2 up to sign)."""
    out = {}
    for name, ld in arrays.items():
        w = np.ones_like(ld) if weights is None else np.asarray(weights, dtype=float).ravel()
        pos = w > 0.0
        vals = ld[pos]
        if not np.all(np.isfinite(vals)):
            out[name] = -math.inf
            continue
        ll = float(np.dot(w[pos], vals))
        out[name] = ll - _penalty(k[name], 0.0, correction, _n_eff(w, n_eff))
    return out


def _adjust_p(p: dict[str, float], method: str | None) -> dict[str, float]:
    if method is None:
        return dict(p)
    names = list(p)
    m = len(names)
    if m == 0:
        return {}
    vals = np.array([p[n] if np.isfinite(p[n]) else 1.0 for n in names])
    if method == "bonferroni":
        adj = np.minimum(1.0, vals * m)
    elif method == "holm":
        order = np.argsort(vals, kind="stable")
        adj_sorted = np.minimum(1.0, np.maximum.accumulate(
            vals[order] * (m - np.arange(m))))
        adj = np.empty(m)
        adj[order] = adj_sorted
    else:
        raise ValueError(f"adjust must be None, 'holm' or 'bonferroni', got {method!r}.")
    return {n: float(a) for n, a in zip(names, adj)}


def confidence_set(
    log_densities: Mapping[str, np.ndarray],
    weights=None,
    *,
    n_params: Mapping[str, float] | None = None,
    best: str | None = None,
    scores: Mapping[str, float] | None = None,
    test: str = "vuong",
    correction: str = "none",
    alpha: float = 0.05,
    adjust: str | None = None,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    **test_kwargs,
) -> ConfidenceSet:
    """Families not significantly worse than the best one.

    The reference ``best`` is, by default, the family with the highest
    ``scores`` — or, without scores, the highest corrected weighted
    log-likelihood ``Σ w log c − K`` (the ranking of AIC for
    ``correction="akaike"``, of BIC for ``"schwarz"``). Every other family
    ``F`` is tested against it; ``F`` is **excluded** only when the test
    significantly prefers ``best`` (``p ≤ alpha`` after ``adjust`` and a
    statistic of the sign favouring ``best``). The rest are retained: they
    are statistically tied with the best, or better than it under the test
    (possible when ``best`` comes from a non-likelihood score).

    ``adjust`` (``None``, ``"holm"``, ``"bonferroni"``) corrects the ``m − 1``
    p-values for multiplicity. This is a set of pairwise comparisons with one
    reference, not the model confidence set of Hansen, Lunde & Nason (2011).
    """
    fn = _test_fn(test)
    names, arrays, k = _as_mapping(log_densities, n_params)
    if not names:
        raise ValueError("empty candidate set.")
    if scores is None:
        sc = _corrected_scores(arrays, weights, k, correction, n_eff)
    else:
        sc = {n: float(scores[n]) for n in names}
    if best is None:
        # First maximum in the given order, as the ICE selection loop.
        best = names[0]
        for n in names:
            if sc[n] > sc[best]:
                best = n
    elif best not in arrays:
        raise ValueError(f"best={best!r} is not among {names}.")
    comps: dict[str, ComparisonResult] = {}
    for n in names:
        if n == best:
            continue
        comps[n] = fn(arrays[best], arrays[n], weights, k_a=k[best], k_b=k[n],
                      correction=correction, alpha=alpha, bandwidth=bandwidth,
                      n_eff=n_eff, names=(best, n), **test_kwargs)
    p_adj = _adjust_p({n: r.p_value for n, r in comps.items()}, adjust)
    excluded = [n for n, r in comps.items()
                if r.decision != "undetermined" and r.statistic > 0.0
                and p_adj[n] <= alpha]
    others = sorted((n for n in comps if n not in excluded),
                    key=lambda n: (-sc[n] if np.isfinite(sc[n]) else math.inf, names.index(n)))
    excluded.sort(key=lambda n: (-sc[n] if np.isfinite(sc[n]) else math.inf, names.index(n)))
    return ConfidenceSet(best=best, members=[best] + others, excluded=excluded,
                         scores=sc, comparisons=comps, p_adjusted=p_adj,
                         alpha=alpha, adjust=adjust, test=test, correction=correction)


# ---------------------------------------------------------------------------
# fit_best hook
# ---------------------------------------------------------------------------

def _fit_best_log_densities(results) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Pointwise log-densities of the ranked fits of a ``FitBestResults``."""
    fits = list(results)
    if not fits:
        raise ValueError("no ranked fit to compare.")
    uv0 = np.asarray(fits[0].uv, dtype=float)
    logc: dict[str, np.ndarray] = {}
    k: dict[str, int] = {}
    for r in fits:
        uv = np.asarray(r.uv, dtype=float)
        if uv.shape != uv0.shape or not np.array_equal(uv, uv0):
            raise ValueError("the fits were not obtained on the same pseudo-observations.")
        name = r.copula.copula_enum.value.SHORT_NAME
        if name in logc:
            raise ValueError(f"family {name!r} appears twice in the results.")
        logc[name] = np.asarray(r.copula.logpdf_array(uv), dtype=float)
        k[name] = int(r.n_params)
    return logc, k


def fit_best_comparison(
    results,
    test: str = "vuong",
    *,
    correction: str = "akaike",
    alpha: float = 0.05,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    **test_kwargs,
) -> ComparisonMatrix:
    """:func:`comparison_matrix` of the ranked fits of ``CopulaVirt.fit_best``.

    The families are in the AIC order of ``results``; ``other_method`` fits
    (Student and BB1 under ``method='tau'``) are left out, as from the
    ranking. The default ``correction="akaike"`` makes the sign of each
    statistic agree with the AIC ranking. The data are the rank pseudo-
    observations of the fit, with unit weights — if they are serially
    dependent keep the HAC default, otherwise ``bandwidth=0`` gives the
    i.i.d. variance of Vuong (1989).
    """
    logc, k = _fit_best_log_densities(results)
    return comparison_matrix(logc, None, n_params=k, test=test,
                             correction=correction, alpha=alpha,
                             bandwidth=bandwidth, n_eff=n_eff, **test_kwargs)


def fit_best_confidence_set(
    results,
    test: str = "vuong",
    *,
    correction: str = "akaike",
    alpha: float = 0.05,
    adjust: str | None = None,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    **test_kwargs,
) -> ConfidenceSet:
    """Families of ``CopulaVirt.fit_best`` not significantly worse than the best.

    With the default ``correction="akaike"`` the reference is the AIC winner
    ``results[0]`` (up to exact AIC ties). See :func:`confidence_set`.
    """
    logc, k = _fit_best_log_densities(results)
    return confidence_set(logc, None, n_params=k, test=test,
                          correction=correction, alpha=alpha, adjust=adjust,
                          bandwidth=bandwidth, n_eff=n_eff, **test_kwargs)


# ---------------------------------------------------------------------------
# ICE hook
# ---------------------------------------------------------------------------

def ice_pair_comparisons(
    model,
    Y,
    xi=None,
    *,
    candidates: Sequence[str] | None = None,
    criterion: str | None = None,
    test: str = "vuong",
    correction: str | None = None,
    alpha: float = 0.05,
    adjust: str | None = None,
    bandwidth: int | str = "auto",
    n_eff: str = "sum",
    pairs: Sequence[tuple[int, int]] | None = None,
    **test_kwargs,
) -> dict[tuple[int, int], PairComparison]:
    """Is the copula family ICE selected for each pair of states significant?

    For every pair ``(i, j)`` this rebuilds the weighted sample the ICE
    M-step fits the copula ``c_ij`` on — ``(F_ij(y_n), F_ji(y_{n+1}))`` with
    weight ``ξ_n(i, j)`` (A16 Eq. 12), the CDFs clipped as in
    ``pmcprg.pmc.ice._m_step`` — refits every candidate with the M-step's own
    weighted MLE (``ice._fit_copula_params``), scores it with the ICE
    criterion (``ice._SCORE_FN``), and compares the candidates with
    :func:`vuong_test` or :func:`clarke_test`.

    Parameters
    ----------
    model      : fitted :class:`~pmcprg.pmc.model.PMCModel` (a copula variant).
    Y          : observations the model was fitted to, shape ``(N,)``.
    xi         : pair posteriors ``(N−1, K, K)``; ``None`` runs the E-step
                 (forward–backward) of ``model`` on ``Y``, as ICE does.
    candidates : copula SHORT_NAMEs; ``None`` uses the ICE configuration of
                 the model (``[ice]`` section, else the package default).
    criterion  : ICE selection criterion ranking the candidates; ``None``
                 uses the model's ICE configuration.
    test       : ``"vuong"`` or ``"clarke"``.
    correction : ``None`` takes the one consistent with ``criterion``
                 (``mle`` → ``"none"``, ``aic`` → ``"akaike"``, ``bic`` →
                 ``"schwarz"``, other criteria → ``"none"``).
    alpha, adjust, bandwidth, n_eff : see :func:`confidence_set`.
    pairs      : restrict to these ``(i, j)``; default every pair whose total
                 weight is not negligible (ICE skips the others too).

    Returns
    -------
    ``{(i, j): PairComparison}``. ``tied_with_runner_up`` answers the
    question "is the selected family significantly better than the next
    one?". Nothing in ``model`` is modified.

    Notes
    -----
    ``selected`` is the criterion's choice on *this* sample; it equals the
    model's family when ICE stopped at convergence (the returned model then
    produced ``ξ``), and may differ after ``max_iter`` iterations. The test
    ignores the estimation of the margins and of ``ξ`` (Chen & Fan 2006;
    see the module docstring).
    """
    import importlib

    from pmcprg.diagnostics.pseudo import copula_pseudo_obs, margin_cdfs
    from pmcprg.numerics import EPS
    from pmcprg.pmc.inference import (
        backward, forward, joint_posteriors, precompute_weights,
    )

    # ``pmcprg.pmc`` re-exports the *function* ``ice``, which shadows the module
    # in ``from pmcprg.pmc import ice``.
    _ice = importlib.import_module("pmcprg.pmc.ice")

    _test_fn(test)
    if not model.variant.uses_copula:
        raise ValueError(f"variant {model.variant.value} has no copula to compare.")
    cfg = _ice._parse_ice_cfg(model, None)
    if candidates is None:
        candidates = list(cfg["candidates"])
    candidates = [str(c) for c in candidates]
    if criterion is None:
        criterion = str(cfg["selection_criterion"])
    if criterion not in _ice._SCORE_FN:
        raise ValueError(f"Unknown criterion {criterion!r}. Valid: {sorted(_ice._SCORE_FN)}")
    if correction is None:
        correction = CRITERION_CORRECTION.get(criterion, "none")
    _check_options(correction, n_eff, alpha)

    Y = np.asarray(Y)
    K = int(model.K)
    N = len(Y)
    if xi is None:
        W, f_pdf = precompute_weights(model, Y)
        alpha_hat, _ = forward(model, Y, W=W, f_pdf=f_pdf)
        beta_hat = backward(model, Y, W=W)
        xi = joint_posteriors(alpha_hat, W, beta_hat)
    xi = np.asarray(xi, dtype=float)
    if xi.shape != (N - 1, K, K):
        raise ValueError(f"xi must have shape {(N - 1, K, K)}, got {xi.shape}.")

    # Same clipping as the ICE M-step: [EPS, 1 − EPS].
    F = margin_cdfs(model, Y, clip=EPS)

    resolved = []
    for short in candidates:
        try:
            entry, cls = _ice._resolve_candidate(short)
        except ValueError:
            continue
        resolved.append((short, entry, cls))
    # Score keywords exactly as ``ice._select_and_fit_copula`` builds them.
    score_kw: dict = {}
    if resolved and criterion in ("huard_common", "huard_global"):
        los, his = zip(*(e.value.TAU_MIN_MAX for _, e, _ in resolved))
        if criterion == "huard_common":
            score_kw["tau_range"] = (max(los), min(his))
        else:
            score_kw["prior_width"] = max(hi - lo for lo, hi in zip(los, his))

    if pairs is None:
        pairs = [(i, j) for i in range(K) for j in range(K)]
    out: dict[tuple[int, int], PairComparison] = {}
    for (i, j) in pairs:
        i, j = int(i), int(j)
        uv, w = copula_pseudo_obs(F, xi, i, j)
        sw = float(np.sum(w))
        if sw < _ICE_MIN_PAIR_WEIGHT:
            continue
        u, v = uv[:, 0], uv[:, 1]
        scores: dict[str, float] = {}
        lls: dict[str, float] = {}
        params_d: dict[str, dict] = {}
        kk: dict[str, int] = {}
        logc: dict[str, np.ndarray] = {}
        failed: list[str] = []
        selected, best_score = None, -np.inf
        score_fn = _ice._SCORE_FN[criterion]
        for short, entry, cls in resolved:
            params = _ice._fit_copula_params(cls, entry, u, v, w)
            score = float(score_fn(cls, entry, params, u, v, w, **score_kw))
            kwargs = _ice._copula_kwargs(params)
            params_d[short] = kwargs
            kk[short] = int(getattr(cls, "n_params", 1))
            scores[short] = score
            try:
                ld = np.asarray(cls(**kwargs).logpdf_array(uv), dtype=float)
            except Exception:
                failed.append(short)
                lls[short] = -math.inf
            else:
                logc[short] = ld
                pos = w > 0.0
                lls[short] = (float(np.dot(w[pos], ld[pos]))
                              if np.all(np.isfinite(ld[pos])) else -math.inf)
            if score > best_score:
                best_score, selected = score, short
        cop = model.copula(i, j)
        model_family = cop.copula_enum.value.SHORT_NAME
        if selected is None:
            selected = model_family if model_family in logc else (next(iter(logc), None) or "")
        order = sorted((s for s in scores if s in logc and s != selected),
                       key=lambda s: -scores[s] if np.isfinite(scores[s]) else math.inf)
        runner_up = order[0] if order else None
        matrix = cset = vs = None
        if selected in logc and len(logc) >= 2:
            matrix = comparison_matrix(logc, w, n_params={s: kk[s] for s in logc},
                                       test=test, correction=correction,
                                       alpha=alpha, bandwidth=bandwidth,
                                       n_eff=n_eff, **test_kwargs)
            cset = confidence_set(logc, w, n_params={s: kk[s] for s in logc},
                                  best=selected, scores={s: scores[s] for s in logc},
                                  test=test, correction=correction, alpha=alpha,
                                  adjust=adjust, bandwidth=bandwidth,
                                  n_eff=n_eff, **test_kwargs)
            if runner_up is not None:
                vs = matrix.result(selected, runner_up)
        out[(i, j)] = PairComparison(
            pair=(i, j), model_family=model_family, selected=selected,
            runner_up=runner_up, criterion=criterion, scores=scores,
            log_likelihood=lls, params=params_d, n_params=kk, sum_weights=sw,
            matrix=matrix, vs_runner_up=vs, confidence_set=cset, failed=failed,
            log_densities=logc, weights=np.asarray(w, dtype=float),
        )
    return out
