"""
outliers.py — erroneous observations (spikes, sensor faults) in PMC/HMC series.

Public API
----------
predictive_pit(model, Y, *, gap_nodes=64) -> PredictivePIT
    One-step-ahead probability integral transform PIT_n = P(Y_n ≤ y_n | y_1:n−1)
    of every observed row, its two-sided p-value and normal score.
flag_outliers(model, Y, *, alpha=1e-3, sequential=True, correction=None,
              gap_nodes=64) -> OutlierFlags
    Rows whose p-value falls below the (corrected) level; with ``sequential``
    a flagged row is treated as missing by the filter of the rows after it.
pit_checks(pit, *, lags=10, exclude=None) -> PitChecks
    Uniformity (Kolmogorov–Smirnov) and independence (Ljung–Box on the
    normal scores and on their squares) of a PIT sequence.
robust_estimate(model, Y, estim_cfg=None, *, algorithm="ice", flag_cfg=None,
                max_rounds=10, initial_mask=None, progress_cb=None) -> RobustFit
    Fit, flag, mask the flagged rows (NaN), refit, until the mask stops
    changing: a trimmed ICE/SEM whose trimming set is chosen by the model's
    own predictive test.

Scalar observations only (``model.d == 1``): the PIT of a vector needs a
Rosenblatt ordering of its components, not implemented here.

The predictive CDF
------------------
Z = (X, Y) is a Markov chain (:mod:`pmcprg.pmc.inference`) whose transition
factorises as q(j, y' | i, y) = T_ij(y) κ_ij(y' | y): T_ij(y) =
P(x_n = j | x_{n−1} = i, y_{n−1} = y) and κ_ij the density of y_n given
(x_{n−1}, x_n, y_{n−1}) = (i, j, y). With the forward filter α̂_{n−1}(i) =
P(x_{n−1} = i | y_1:n−1),

    P(Y_n ≤ y | y_1:n−1) = Σ_i α̂_{n−1}(i) Σ_j T_ij(y_{n−1}) K_ij(y | y_{n−1}),

K_ij the CDF of κ_ij. Per variant, with the weights of ``precompute_weights``
(``W[n, i, j] = T_ij(y_n) κ_ij(y_{n+1} | y_n)``):

=================  ===============================  ==============================
variant            T_ij(y)                          K_ij(y' | y)
=================  ===============================  ==============================
HMC-IN, HMC-IN2    A_ij                             F_j(y')
HMC-DN             A_ij                             h_ij(F_j(y') | F_i(y))
PMC-IN, state      p_ij / Σ_k p_ik                  F_j(y')
PMC-IN, pair       p_ij f_ij(y) / D_i(y)            F_ji(y')
PMC, state         p_ij / Σ_k p_ik                  h_ij(F_j(y') | F_i(y))
PMC, pair          p_ij f_ij(y) / D_i(y)            h_ij(F_ji(y') | F_ij(y))
=================  ===============================  ==============================

with D_i(y) = Σ_k p_ik f_ik(y) — the division by D of the PMC weights
(DerrodePieczynski_CSDA2013 Eq. 13; for state margins f_i(y) cancels) — and
h_ij(v | u) = ∂C_ij(u, v)/∂u = ``model.copula(i, j).conditional_cdf(v, u)``
the h-function of the copula c_ij(u, v) of the weights, whose first argument
belongs to y_{n−1}: ∫_{−∞}^{y} f_ji(y') c_ij(u, F_ji(y')) dy' =
h_ij(F_ji(y) | u). Copula arguments are clipped to [EPS, 1 − EPS] as in
``precompute_weights``. T_ij is computed in log space (a pair-margin D_i
that underflows keeps its ratio).

* **Row 1** — the law of y_1, the initial message of the forward pass:
  Σ_j (Σ_i p_ij) F_j(y) for state margins (π_j F_j for a stationary HMC),
  Σ_{i,j} p_ij F_ij(y) for pair margins.
* **After missing rows** — at a missing n−1 the filter of the grid variants
  is the augmented message α̃_{n−1}(i, g) = P(x_{n−1} = i, y_{n−1} ∈ node g
  | past) of :mod:`pmcprg.pmc.gaps` (block-renormalised quadrature, G =
  ``gap_nodes``), and

      P(Y_n ≤ y | past) = Σ_{i,g} α̃_{n−1}(i, g) Σ_j T_ij(y_g) K_ij(y | y_g),

  exact up to the quadrature of α̃ (T and K are evaluated exactly at the
  nodes). For HMC-IN, HMC-IN2 and PMC-IN with state margins T and K do not
  depend on y_{n−1}: the K-state message of the exact shortcut suffices.
* **Consistency** — the derivative of the PIT in y is the forward normaliser
  C_n of the same pass, the predictive density p(y_n | past) whose logarithms
  sum to the log-likelihood (``log_pred``, ``log_lik``): the PIT is the CDF of
  the law the likelihood uses, not an approximation of it.
* **Upper tail** — 1 − PIT is summed from the survival functions (the
  margins' own ``sf`` without a copula; 1 − h with one), so p-values in the
  upper tail are not lost to cancellation for the margin-only variants; for
  the copula variants 1 − h resolves p-values down to ~1e-15.
* **Empty states** — a state with an all-zero row of p (ICE and SEM leave
  one when a state loses every observation, :mod:`pmcprg.pmc.inference`)
  has no transition law, T_i· = 0/0; its message is 0 and its T is taken as
  0 (log −∞): the results are those of the model without the state (8.9e-16
  measured). As NaN, 0 · NaN made every PIT after row 1 NaN and nothing was
  flagged.
* **Non-ignorable missingness** (``model.missingness``) — the filter is given
  (y_obs, m) as in :mod:`pmcprg.pmc.gaps`, and so is the PIT: with e_n(j) the
  factor of the observed mask value m_n = 0, P(Y_n ≤ y | past, m_1:n) =
  Σ_ij α̂(i) T_ij K_ij(y) e_n(j) / Σ_ij α̂(i) T_ij e_n(j). A row gated by
  :func:`flag_outliers` keeps m_n = 0: it was recorded, only its value is
  distrusted.

The filter runs in log space step by step (normalised messages), so no
underflow of an extreme observation needs the linear/log switch of the
batch passes; its messages equal those of ``forward`` / ``gap_posterior``
on the same missing rows to rounding (``test_outliers.py``).

Calibration under the model
---------------------------
By Rosenblatt (1952) the PITs of the observed rows are iid U(0, 1) under the
true model (for any mask independent of the values), and the normal scores
z_n = Φ⁻¹(PIT_n) iid N(0, 1) (Diebold, Gunther & Tay 1998; Berkowitz 2001).
:func:`pit_checks` tests both: Kolmogorov–Smirnov on the PITs, Ljung–Box
(1978) on z and on z² (McLeod & Li 1983) over the observed rows in order. A
misspecified model shows up there before any single p-value is extreme.

Flagging and multiplicity
-------------------------
p_n = 2 min(PIT_n, 1 − PIT_n), flagged when below the threshold. N rows are
N tests: at level α a correct model flags about αN clean rows per series (3
per 3000 at α = 1e-3), and at least one with probability 1 − (1 − α)^N (95 %
at N = 3000). ``correction``:

* ``None`` — per-row level α (a trimming rule, not a test of the series);
* ``"bonferroni"`` — α/m for m tested rows: family-wise error ≤ α;
* ``"bh"`` — Benjamini & Hochberg (1995) step-up, threshold α k*/m with
  k* = max{k : p_(k) ≤ α k/m}: false discovery rate ≤ α.

Under the model the p-values are independent (Rosenblatt, above), the case in
which both corrections hold exactly. Under contamination a spike moves the
filter of the rows after it, which is why the gating below matters.

Sequential gating (innovation gating)
-------------------------------------
With ``sequential=True`` row n is tested with the filter of the rows before
it, and a flagged row is treated as MISSING from then on: its value is
integrated out of the filter (the gap machinery above) instead of entering
the prediction of y_{n+1} through the y_n → y_{n+1} dependence and the state
filter — the validation gate of Kalman tracking (Bar-Shalom, Li &
Kirubarajan 2001), for additive outliers in the sense of Fox (1972). Without
it (``sequential=False``: every row enters the filter) a copula of strong
positive dependence predicts y_{n+1} near the spike, and the clean
neighbour gets a tiny p-value too; for HMC-IN the spike only moves the state
filter. Gating has a cost under the model: a legitimate extreme value of a
strongly dependent series, once gated, makes its successor surprising too.
Measured in ``report/erroneous_data`` (α = 1e-3, N = 2000, 30 series per
cell): on clean ``pmc_gauss_k2`` series the false-flag rate is 1.27 per
1000 with gating and 0.97 without (0.98 for both on ``hmc_in_gauss_k2``);
with 5 % of +6 sd spikes it is 1.26 with gating and 2.26 without, and the
false flags on the row right after a spike drop from 11 to 1.
Without gating, a run of erroneous readings is judged conditionally on its
own first value: under a strongly dependent copula the second reading of a
plateau is *not* surprising given the first. Measured on the Intel Lab
battery failures (``report/erroneous_data/intel_lab``): the non-sequential
recall stops near 0.5 because every missed row follows another failing
reading, whereas gating flags them all. This is the model's answer, not a
numerical artefact, so use ``sequential=True`` for detection. One
approximation does enter at the extreme: copula arguments are clipped to
[EPS, 1 − EPS], so two consecutive readings both beyond F⁻¹(1 − EPS) are
evaluated at the clipped corner (Gauss, τ = 0.99, 50 after 50 on N(0, 1)
margins: PIT 0.525, where the exact copula gives about 0.65).
With ``"bh"`` the threshold depends on all the p-values, which depend
on the gating: the gated filter is rerun from the Bonferroni threshold with
the BH threshold of its p-values until the threshold repeats (at most
``_BH_MAX_RUNS`` runs, WARNING otherwise).

Robust estimation (flag and mask)
---------------------------------
:func:`robust_estimate` alternates, from the fit on Y (or on Y with an
``initial_mask`` set to NaN):

1. flag the ORIGINAL Y with the current fit (:func:`flag_outliers`);
2. set the flagged rows to NaN and refit, from the initial model and config.

It stops when a round flags exactly the rows its fit had masked (a fixed
point) or after ``max_rounds`` refits. Design:

* **Masking is trimming.** With ICE's ``missing_strategy = "available"`` a
  NaN row gets zero weight in the M-step — its y_n in the margins, the pairs
  (n−1, n) and (n, n+1) that touch it in the copulas — and its value is
  integrated out in the E-step: a trimmed likelihood (Neykov, Filzmoser,
  Dimova & Neytchev 2007) whose trimmed set is chosen by the model's own
  predictive test instead of a fixed fraction.
* **Why iterate.** A contaminated fit inflates the variances and weakens the
  dependence, which makes the spikes look less extreme (the masking effect of
  robust statistics): the first flags are too few. Refitting on the masked
  series sharpens the model, which flags more. The alternation is the
  concentration step of FAST-LTS and of the trimmed likelihood (Rousseeuw &
  Van Driessen 2006; Neykov et al. 2007): fix the model, choose the trimmed
  set; fix the set, refit.
* **Flags recomputed on the original Y at every round**, not accumulated: a
  clean row flagged by an early, contaminated fit is released later.
* **Every refit restarts from the initial model** (and config: ``init =
  "kmeans"`` clusters the rows left observed), not from the previous fit. A
  contaminated fit sits in its own basin, and ICE started there stays near
  it. Measured in ``report/erroneous_data`` on ``pmc_gauss_k2`` (N = 2000,
  1 % of spikes of 4, 6 and 8 sd, 30 replicates each, ICE started at the
  true model): refits started from the previous fit ended with the two
  regimes merged (both fitted means within 0.75 of 0, for −1 and +1) in 70
  of 90 runs; refits from the initial model ended within 0.25 of both means
  in all 90, with a clean-row classification error of 11.7 % (11.5 % for
  ICE on the clean series). On ``hmc_in_gauss_k2`` the two agree (one
  failure of the warm restarts in 90). The state labels are those of the
  initial model in every fit.
* **Breakdown.** When the spikes are frequent enough to form a regime of
  their own, the first fit gives them a state; under that model they are
  not outliers, nothing is flagged and the loop stops at once. Measured
  (same study): at 5 % of +6 and +8 sd spikes ``hmc_in_gauss_k2`` fits a
  state of mean 8.5 and 11.3 (the spikes' level) in 60 of 60 runs, and
  ``pmc_gauss_k2`` breaks down at 5 % for every k (90 of 90). At 1 %
  ``pmc_gauss_k2``'s raw fit already gives the spikes a state (a
  persistence A_11 < 0.7 in 88 of 90 runs), but that fit still flags some
  of them and the loop recovers. A model-free pre-screen passed as
  ``initial_mask`` (the robust estimate's starting set, as the random
  subsets of FAST-LTS) moves the first fit out of that basin; its rows are
  re-tested by the model from the first round on, so a clean row it masked
  is released. A Hampel filter (4 MADs, window of 11) restored oracle-level
  estimates at 5 % of 6 and 8 sd in all 120 runs of both models, but not at
  5 % of 4 sd on ``pmc_gauss_k2`` (3 of 30: it catches 68 % of those
  spikes). An explicit contamination component in the model is the
  principled remedy (not implemented).
* **Stopping.** No objective is guaranteed to be monotone (the trimmed set
  changes size), hence ``max_rounds``; a mask that still changes is reported
  (``converged = False``, WARNING). From a contaminated first fit the
  trimmed set grows by a few rows per round: 2 to 6 fits on
  ``pmc_gauss_k2`` at 1 %, up to 11 on ``hmc_in_gauss_k2`` at 5 % of 4 sd,
  where 1 run of 30 still changed after 10 refits — hence the default of
  10.
* **Cost of trimming clean rows.** At α = 1e-3 about 0.1 % of the clean rows
  are trimmed, the tails of the margins lose a little weight and their
  scales shrink slightly; no consistency correction is applied. Measured on
  the clean series of the study: 1.0 (HMC-IN) and 1.3 (PMC) rows trimmed
  per 1000, fitted σ biased by −0.008 and −0.020 (−0.002 and −0.009 for ICE
  on the same series), clean-row classification error 6.12 % and 11.64 %
  (6.10 % and 11.48 %).
* A non-ignorable missingness mechanism is refused: the masked rows would be
  read as missing by that mechanism.

References
----------
* Rosenblatt, M. (1952). Remarks on a multivariate transformation. *Ann.
  Math. Statist.* 23(3), 470–472.
* Diebold, F. X., Gunther, T. A. & Tay, A. S. (1998). Evaluating density
  forecasts with applications to financial risk management. *Int. Econ.
  Rev.* 39(4), 863–883.
* Berkowitz, J. (2001). Testing density forecasts, with applications to risk
  management. *J. Bus. Econ. Statist.* 19(4), 465–474.
* Ljung, G. M. & Box, G. E. P. (1978). On a measure of lack of fit in time
  series models. *Biometrika* 65(2), 297–303.
* McLeod, A. I. & Li, W. K. (1983). Diagnostic checking ARMA time series
  models using squared-residual autocorrelations. *J. Time Series Anal.*
  4(4), 269–273.
* Benjamini, Y. & Hochberg, Y. (1995). Controlling the false discovery rate:
  a practical and powerful approach to multiple testing. *J. R. Statist.
  Soc. B* 57(1), 289–300.
* Fox, A. J. (1972). Outliers in time series. *J. R. Statist. Soc. B* 34(3),
  350–363.
* Bar-Shalom, Y., Li, X. R. & Kirubarajan, T. (2001). *Estimation with
  Applications to Tracking and Navigation*. Wiley — validation gates.
* Neykov, N., Filzmoser, P., Dimova, R. & Neytchev, P. (2007). Robust fitting
  of mixtures using the trimmed likelihood estimator. *Comput. Stat. Data
  Anal.* 52(1), 299–308.
* Rousseeuw, P. J. & Van Driessen, K. (2006). Computing LTS regression for
  large data sets. *Data Min. Knowl. Discov.* 12(1), 29–45.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as _ss
from scipy.special import ndtri

from pmcprg.exceptions import IncompatibleObservationError
from pmcprg.numerics import MIN_POSITIVE
from pmcprg.pmc import inference as _inf
from pmcprg.pmc.gaps import (
    _frozen,
    _initial,
    _log_kernel,
    _margin_eval,
    _normalise_blocks,
    _x_transition,
    missing_mask,
    needs_grid,
    reference_grid,
)
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = [
    "CORRECTIONS",
    "DEFAULT_ALPHA",
    "PredictivePIT",
    "OutlierFlags",
    "PitChecks",
    "RobustFit",
    "predictive_pit",
    "flag_outliers",
    "pit_checks",
    "robust_estimate",
]

#: Default per-row level of :func:`flag_outliers`.
DEFAULT_ALPHA = 1e-3
#: Values of the ``correction`` argument of :func:`flag_outliers`.
CORRECTIONS: tuple = (None, "bonferroni", "bh")
#: Maximum number of gated-filter runs of the ``"bh"`` fixed point.
_BH_MAX_RUNS = 10


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PredictivePIT:
    """One-step-ahead predictive PIT of a series (module docstring).

    Attributes
    ----------
    pit       : (N,) PIT_n = P(Y_n ≤ y_n | past observed rows); NaN at a
                missing row.
    pvalue    : (N,) two-sided p-value 2 min(PIT_n, 1 − PIT_n), capped at 1;
                NaN at a missing row.
    z         : (N,) normal score Φ⁻¹(PIT_n), from the smaller tail (finite:
                a tail probability is floored at the smallest normal float).
    log_pred  : (N,) log p(y_n | past), the log predictive density; NaN at a
                missing row.
    log_lik   : float — Σ of the log normalisers of the pass: the
                observed-data log-likelihood of the rows the filter used.
    alpha_hat : (N, K) — P(x_n = i | rows up to n used by the filter).
    miss      : (N,) bool — missing rows of Y (non-finite values).
    method    : ``"exact"`` (K-state filter) or ``"grid"`` (quadrature grid
                for the missing rows).
    """

    pit: np.ndarray
    pvalue: np.ndarray
    z: np.ndarray
    log_pred: np.ndarray
    log_lik: float
    alpha_hat: np.ndarray
    miss: np.ndarray
    method: str


@dataclass
class OutlierFlags:
    """Rows flagged by :func:`flag_outliers` and the statistics used.

    Attributes
    ----------
    flagged    : (N,) bool — flagged rows (never a missing row).
    pvalue     : (N,) two-sided p-values the decision used (with
                 ``sequential``, each from the filter gated by the flags
                 before it).
    pit, z     : (N,) the matching PITs and normal scores.
    log_pred   : (N,) log predictive densities of the rows that entered the
                 filter; NaN at missing and flagged rows when ``sequential``.
    threshold  : float — the per-row threshold: flagged ⇔ p < threshold.
    alpha      : float — the requested level (per row, family-wise or FDR).
    correction : ``None``, ``"bonferroni"`` or ``"bh"``.
    sequential : bool — whether flagged rows were gated out of the filter.
    n_tests    : int — number of tested (observed) rows.
    log_lik    : float — log-likelihood of the rows that entered the filter
                 (flagged rows integrated out when ``sequential``).
    alpha_hat  : (N, K) — the filter P(x_n = i | rows up to n it used).
    miss       : (N,) bool — missing rows of Y.
    method     : ``"exact"`` or ``"grid"``.
    """

    flagged: np.ndarray
    pvalue: np.ndarray
    pit: np.ndarray
    z: np.ndarray
    log_pred: np.ndarray
    threshold: float
    alpha: float
    correction: str | None
    sequential: bool
    n_tests: int
    log_lik: float
    alpha_hat: np.ndarray
    miss: np.ndarray
    method: str

    @property
    def index(self) -> np.ndarray:
        """Positions of the flagged rows."""
        return np.nonzero(self.flagged)[0]

    @property
    def n_flagged(self) -> int:
        return int(self.flagged.sum())


@dataclass
class PitChecks:
    """Calibration checks of a PIT sequence (:func:`pit_checks`).

    Attributes
    ----------
    n           : number of PITs used (observed, not excluded).
    ks_stat, ks_pvalue   : Kolmogorov–Smirnov test of U(0, 1) on the PITs.
    lb_stat, lb_pvalue   : Ljung–Box Q on the normal scores, ``lags`` lags.
    lb2_stat, lb2_pvalue : Ljung–Box Q on the squared normal scores
                           (McLeod–Li).
    lags        : number of lags.
    z_mean, z_sd: mean and standard deviation of the normal scores (0 and 1
                  under the model).
    """

    n: int
    ks_stat: float
    ks_pvalue: float
    lb_stat: float
    lb_pvalue: float
    lb2_stat: float
    lb2_pvalue: float
    lags: int
    z_mean: float
    z_sd: float


@dataclass
class RobustFit:
    """Result of :func:`robust_estimate`.

    Attributes
    ----------
    model     : PMCModel — the last fit.
    trace     : IceTrace or SemTrace of the last fit.
    mask      : (N,) bool — rows set to NaN for the last fit.
    flags     : OutlierFlags — the flags of the last fit on the original Y
                (``flags.flagged == mask`` when ``converged``).
    masks     : list of (N,) bool — the mask of every fit, ``masks[0]`` the
                ``initial_mask`` (empty by default: the fit on Y as given).
    traces    : list — the trace of every fit.
    converged : bool — the last flags equal the last mask.
    """

    model: PMCModel
    trace: object
    mask: np.ndarray
    flags: OutlierFlags
    masks: list = field(default_factory=list)
    traces: list = field(default_factory=list)
    converged: bool = False

    @property
    def n_fits(self) -> int:
        return len(self.traces)


# ---------------------------------------------------------------------------
# Checks and small helpers
# ---------------------------------------------------------------------------

def _check_scalar(model: PMCModel, Y) -> np.ndarray:
    d = getattr(model, "d", 1)
    if d != 1:
        raise NotImplementedError(
            f"predictive PIT / outlier flags are implemented for scalar "
            f"observations only (model.d = {d}): the PIT of a vector needs a "
            f"Rosenblatt ordering of its components."
        )
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 2 and Y.shape[1] == 1:
        Y = Y[:, 0]
    if Y.ndim != 1:
        raise ValueError(f"Y must be a 1-D series for a scalar model, got shape {Y.shape}.")
    if len(Y) < 1:
        raise ValueError("Y must hold at least one row.")
    return Y


def _margin_cdf_sf(model: PMCModel, y: np.ndarray):
    """F[:, i, j] = F_ij(y) and S = 1 − F from the margins' ``sf`` (M, K, K).

    State margins are broadcast like :func:`pmcprg.pmc.gaps._margin_eval`:
    F[:, k, :] = F_k(y). Not clipped (these are CDF values, not copula
    arguments).
    """
    K, M = model.K, len(y)
    F = np.empty((M, K, K))
    S = np.empty((M, K, K))
    with np.errstate(all="ignore"):
        if model.margin_structure == "pair":
            for i in range(K):
                for j in range(K):
                    fr = _frozen(model.margin(i, j))
                    F[:, i, j] = fr.cdf(y)
                    S[:, i, j] = fr.sf(y)
        else:
            for k in range(K):
                fr = _frozen(model.margin(k))
                F[:, k, :] = np.asarray(fr.cdf(y), dtype=float)[:, None]
                S[:, k, :] = np.asarray(fr.sf(y), dtype=float)[:, None]
    return F, S


def _log_x_transition(model: PMCModel, lf: np.ndarray) -> np.ndarray:
    """log T_ij(y) = log P(x_n = j | x_{n−1} = i, y_{n−1} = y) at the rows of lf.

    :func:`pmcprg.pmc.gaps._x_transition` with its NaN set to −∞: the row
    of an empty state (all-zero row of p, 0/0) and, for pair margins, a row
    whose densities p_ik f_ik(y) all vanish. The filter gives such a state
    probability 0, and 0 · NaN would turn every PIT after it into NaN
    (module docstring, "Empty states").
    """
    T = np.array(_x_transition(model, lf, log=True))
    T[np.isnan(T)] = -np.inf
    return T


def _h(cop, v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """h(v | u) = ``cop.conditional_cdf(v, u)`` elementwise (broadcast)."""
    v, u = np.broadcast_arrays(np.asarray(v, dtype=float), np.asarray(u, dtype=float))
    out = np.fromiter((cop.conditional_cdf(float(a), float(b))
                       for a, b in zip(v.ravel(), u.ravel())), dtype=float, count=v.size)
    return np.clip(out, 0.0, 1.0).reshape(v.shape)


def _pvalue(c: float, s: float) -> float:
    return float(min(1.0, 2.0 * min(c, s)))


def _normal_score(c: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Φ⁻¹(PIT) from the smaller tail: Φ⁻¹(c) if c ≤ s else −Φ⁻¹(s)."""
    c = np.asarray(c, dtype=float)
    s = np.asarray(s, dtype=float)
    with np.errstate(invalid="ignore"):
        lo = ndtri(np.maximum(c, MIN_POSITIVE))
        hi = -ndtri(np.maximum(s, MIN_POSITIVE))
        return np.where(np.isnan(c) | np.isnan(s), np.nan, np.where(c <= s, lo, hi))


# ---------------------------------------------------------------------------
# The step-by-step predictive filter
# ---------------------------------------------------------------------------

class _Grid:
    """Quadrature-grid pieces shared by the steps of one filter (lazy)."""

    def __init__(self, model: PMCModel, gap_nodes):
        self.grid = grid = reference_grid(model, gap_nodes)
        self.G = grid.G
        self.lw = np.log(grid.omega)
        self.lf, self.F = _margin_eval(model, grid.nodes, log=True)          # (G, K, K)
        self.logT = _log_x_transition(model, self.lf)                        # (G, K, K)
        self._Q = None
        self.model = model

    def Q(self) -> np.ndarray:
        """log Q[(i, g), (j, g')] — missing → missing, as ``gaps._Chain``."""
        if self._Q is None:
            model, G, K = self.model, self.G, self.model.K
            rep, til = np.repeat(np.arange(G), G), np.tile(np.arange(G), G)
            ker = _log_kernel(model, self.lf[rep], _take(self.F, rep),
                              self.lf[til], _take(self.F, til)).reshape(G, G, K, K)
            Q = ker.transpose(2, 0, 3, 1) + self.lw[None, None, None, :]      # (K, G, K, G)
            Q, _ = _normalise_blocks(Q, self.logT.transpose(1, 0, 2), log=True)
            self._Q = Q.reshape(K * G, K * G)
        return self._Q


def _take(F, idx):
    return None if F is None else F[idx]


class _Filter:
    """Forward filter of a scalar series with optional innovation gating.

    Precomputes what does not depend on the gating (margins at every row,
    observed → observed log weights, the conditional CDFs of every observed
    pair); :meth:`run` then walks the series once, testing each observed row
    before deciding whether it enters the filter.
    """

    def __init__(self, model: PMCModel, Y, gap_nodes):
        Y = _check_scalar(model, Y)
        self.model = model
        self.gap_nodes = gap_nodes
        self.N = N = len(Y)
        self.K = K = model.K
        self.Y = Y
        self.miss = miss = missing_mask(Y)
        self.grid_needed = needs_grid(model)
        self.uses_cop = model.variant.uses_copula
        obs = ~miss
        fill = float(np.median(Y[obs])) if obs.any() else 0.0
        Yf = np.where(miss, fill, Y)
        self.lf, self.Fc = _margin_eval(model, Yf, log=True)                # (N, K, K)
        self.Fm, self.Sm = _margin_cdf_sf(model, Yf)
        ev = _inf._evidence(model, miss)
        with np.errstate(divide="ignore"):
            self.lev = None if ev is None else np.log(ev)
        self._grid = None
        self.cops = ([[model.copula(i, j) for j in range(K)] for i in range(K)]
                     if self.uses_cop else None)
        if N > 1:
            self.logW = _log_kernel(model, self.lf[:-1], _take(self.Fc, slice(0, -1)),
                                    self.lf[1:], _take(self.Fc, slice(1, None)))
            self.logT = _log_x_transition(model, self.lf[:-1])
            # K_ij(y_n | y_{n−1}) and its complement for every observed pair.
            self.Kc = np.full((N - 1, K, K), np.nan)
            self.Ks = np.full((N - 1, K, K), np.nan)
            if not self.uses_cop:
                # F_ji(y_n): independent of y_{n−1}, needed after a gap too.
                self.Kc[:] = self.Fm[1:].transpose(0, 2, 1)
                self.Ks[:] = self.Sm[1:].transpose(0, 2, 1)
            else:
                pairs = np.nonzero(obs[:-1] & obs[1:])[0]
                if pairs.size:
                    for i in range(K):
                        for j in range(K):
                            h = _h(self.cops[i][j], self.Fc[pairs + 1, j, i], self.Fc[pairs, i, j])
                            self.Kc[pairs, i, j] = h
                            self.Ks[pairs, i, j] = 1.0 - h

    # ---- pieces ----------------------------------------------------------

    @property
    def method(self) -> str:
        return "grid" if self._grid is not None else "exact"

    def grid(self) -> _Grid:
        if self._grid is None:
            self._grid = _Grid(self.model, self.gap_nodes)
        return self._grid

    def _ev(self, n: int, aug: bool):
        """log e_n in the layout of a destination (augmented if ``aug``)."""
        if self.lev is None:
            return 0.0
        return np.repeat(self.lev[n], self.grid().G) if aug else self.lev[n]

    def _e(self, n: int) -> np.ndarray:
        return np.ones(self.K) if self.lev is None else np.exp(self.lev[n])

    def _init_observed(self) -> np.ndarray:
        la = _initial(self.model, self.lf[:1], log=True)[0]
        return la + (0.0 if self.lev is None else self.lev[0])

    def _init_missing(self) -> tuple[np.ndarray, bool]:
        model, K = self.model, self.K
        prior = _initial(model, np.zeros((1, K, K)), log=True)                # (1, K)
        if not self.grid_needed:
            return prior[0] + (0.0 if self.lev is None else self.lev[0]), False
        g = self.grid()
        mu = _initial(model, g.lf, log=True).T + g.lw[None, :]               # (K, G)
        init, _ = _normalise_blocks(mu[None], prior, log=True)
        return init[0] + self._ev(0, True), True

    def _first_cdf(self) -> tuple[float, float, float]:
        """(P(Y_1 ≤ y_1), P(Y_1 > y_1), log P(m_1)) — the law of y_1."""
        model = self.model
        p = model.prior_p
        e = self._e(0)
        if model.margin_structure == "pair":
            w = p * e[:, None]                                                 # p_ij e_0(i)
            c, s = float((w * self.Fm[0]).sum()), float((w * self.Sm[0]).sum())
        else:
            w = p.sum(axis=0) * e                                              # Σ_i p_ij e_0(j)
            c = float(w @ self.Fm[0, :, 0])
            s = float(w @ self.Sm[0, :, 0])
        den = float(w.sum())
        return c / den, s / den, float(np.log(den))

    def _cdf(self, n: int, la: np.ndarray, aug: bool) -> tuple[float, float, float]:
        """(PIT_n, 1 − PIT_n, log P(m_n = 0 | past)) from the log message ``la``."""
        K = self.K
        e = self._e(n)
        a = np.exp(la)
        if not aug:
            T = np.exp(self.logT[n - 1])
            Tc, Ts = T * self.Kc[n - 1], T * self.Ks[n - 1]
            den = float(a @ T @ e)
            c, s = float(a @ Tc @ e), float(a @ Ts @ e)
        else:
            g = self.grid()
            G = g.G
            a = a.reshape(K, G)
            T = np.exp(g.logT)                                                 # (G, K, K)
            if self.uses_cop:
                Kc = np.empty((G, K, K))
                for i in range(K):
                    for j in range(K):
                        Kc[:, i, j] = _h(self.cops[i][j], self.Fc[n, j, i], g.F[:, i, j])
                Ks = 1.0 - Kc
            else:
                Kc = np.broadcast_to(self.Fm[n].T, (G, K, K))
                Ks = np.broadcast_to(self.Sm[n].T, (G, K, K))
            den = float(np.einsum("ig,gij,j->", a, T, e))
            c = float(np.einsum("ig,gij,j->", a, T * Kc, e))
            s = float(np.einsum("ig,gij,j->", a, T * Ks, e))
        with np.errstate(divide="ignore"):
            return c / den, s / den, float(np.log(den))

    def _L_observed(self, n: int, aug: bool) -> np.ndarray:
        """log transition into an observed y_n from the message at n−1."""
        if not aug:
            return self.logW[n - 1] + (0.0 if self.lev is None else self.lev[n][None, :])
        g, K = self.grid(), self.K
        G = g.G
        rep = np.zeros(G, dtype=int) + n
        ker = _log_kernel(self.model, g.lf, g.F, self.lf[rep], _take(self.Fc, rep))  # (G, K, K)
        X = ker.transpose(1, 0, 2).reshape(K * G, K)
        return X + (0.0 if self.lev is None else self.lev[n][None, :])

    def _L_missing(self, n: int, aug: bool) -> tuple[np.ndarray, bool]:
        """log transition into a missing (or gated) y_n; returns (L, augmented)."""
        if not self.grid_needed:
            return self.logT[n - 1] + (0.0 if self.lev is None else self.lev[n][None, :]), False
        g = self.grid()
        G = g.G
        if aug:
            return g.Q() + self._ev(n, True), True
        rep = np.zeros(G, dtype=int) + (n - 1)
        ker = _log_kernel(self.model, self.lf[rep], _take(self.Fc, rep), g.lf, g.F)  # (G, K, K)
        ker = ker.transpose(1, 2, 0)[None] + g.lw                             # (1, K, K, G)
        E, _ = _normalise_blocks(ker, _log_x_transition(self.model, self.lf[n - 1:n]), log=True)
        return E[0] + self._ev(n, True), True

    def _marginal(self, la: np.ndarray, aug: bool) -> np.ndarray:
        a = np.exp(la)
        return a.reshape(self.K, -1).sum(axis=1) if aug else a

    # ---- the pass ----------------------------------------------------------

    def run(self, threshold: float | None = None) -> dict:
        """One pass; rows with p-value < ``threshold`` are gated (None: none)."""
        N, K = self.N, self.K
        cdf = np.full(N, np.nan)
        sf = np.full(N, np.nan)
        log_pred = np.full(N, np.nan)
        gated = np.zeros(N, dtype=bool)
        alpha_hat = np.empty((N, K))

        def gate(c, s) -> bool:
            return threshold is not None and _pvalue(c, s) < threshold

        lden = 0.0
        if self.miss[0]:
            la, aug = self._init_missing()
        else:
            c, s, lden = self._first_cdf()
            cdf[0], sf[0] = c, s
            if gate(c, s):
                gated[0] = True
                la, aug = self._init_missing()
            else:
                la, aug = self._init_observed(), False
        lc = float(_inf._lse(la))
        if not np.isfinite(lc):
            raise IncompatibleObservationError(
                f"Predictive filter: Y[0]={self.Y[0]!r} has zero density under every "
                f"state (log C_1 = {lc!r})."
            )
        log_lik = lc
        if not (self.miss[0] or gated[0]):
            log_pred[0] = lc - lden
        la = la - lc
        alpha_hat[0] = self._marginal(la, aug)

        for n in range(1, N):
            observed = False
            if self.miss[n]:
                L, aug_n = self._L_missing(n, aug)
            else:
                c, s, lden = self._cdf(n, la, aug)
                cdf[n], sf[n] = c, s
                if gate(c, s):
                    gated[n] = True
                    L, aug_n = self._L_missing(n, aug)
                else:
                    L, aug_n = self._L_observed(n, aug), False
                    observed = True
            r = _inf._lse(la[:, None] + L, axis=0)
            lc = float(_inf._lse(r))
            if not np.isfinite(lc):
                raise IncompatibleObservationError(
                    f"Predictive filter: Y[{n}]={self.Y[n]!r} has zero density under "
                    f"every state reachable from step {n - 1} (log C = {lc!r}). "
                    f"flag_outliers(..., sequential=True) gates such a row out."
                )
            log_lik += lc
            if observed:
                log_pred[n] = lc - lden
            la = r - lc
            aug = aug_n
            alpha_hat[n] = self._marginal(la, aug)

        return dict(cdf=cdf, sf=sf, log_pred=log_pred, gated=gated,
                    log_lik=float(log_lik), alpha_hat=alpha_hat)


def _pvalues(cdf: np.ndarray, sf: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.minimum(1.0, 2.0 * np.minimum(cdf, sf))


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def predictive_pit(
    model: PMCModel,
    Y,
    *,
    gap_nodes: int | None = None,
) -> PredictivePIT:
    """One-step-ahead predictive PIT, p-values and normal scores of Y.

    PIT_n = P(Y_n ≤ y_n | y_1:n−1) from the forward filter and the
    conditional CDFs of the transition (module docstring, per variant);
    missing rows before n are integrated out (exact shortcut, or the
    quadrature grid of :mod:`pmcprg.pmc.gaps`), missing rows get NaN. Every
    observed row enters the filter: to keep a flagged row out of the
    prediction of the next ones, use :func:`flag_outliers`.

    Parameters
    ----------
    model     : PMCModel — known parameters, scalar observations (d = 1).
    Y         : (N,) observations, NaN (any non-finite value) where missing.
    gap_nodes : quadrature nodes G for the missing rows of the grid variants
                (default 64).

    Returns
    -------
    PredictivePIT
    """
    filt = _Filter(model, Y, gap_nodes)
    out = filt.run(None)
    return PredictivePIT(
        pit=out["cdf"], pvalue=_pvalues(out["cdf"], out["sf"]),
        z=_normal_score(out["cdf"], out["sf"]), log_pred=out["log_pred"],
        log_lik=out["log_lik"], alpha_hat=out["alpha_hat"], miss=filt.miss,
        method=filt.method,
    )


def _bh_threshold(p: np.ndarray, alpha: float) -> float:
    """Benjamini–Hochberg threshold t with (p < t) ⇔ rejected; 0 if none."""
    p = np.sort(np.asarray(p, dtype=float))
    m = p.size
    if m == 0:
        return 0.0
    ok = np.nonzero(p <= alpha * np.arange(1, m + 1) / m)[0]
    if ok.size == 0:
        return 0.0
    return float(np.nextafter(alpha * (ok[-1] + 1) / m, np.inf))


def _check_flag_args(alpha, correction) -> float:
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}.")
    if correction not in CORRECTIONS:
        raise ValueError(f"Unknown correction {correction!r}. Valid: {list(CORRECTIONS)}.")
    return alpha


def flag_outliers(
    model: PMCModel,
    Y,
    *,
    alpha: float = DEFAULT_ALPHA,
    sequential: bool = True,
    correction: str | None = None,
    gap_nodes: int | None = None,
) -> OutlierFlags:
    """Flag the observed rows whose predictive p-value is too small.

    Parameters
    ----------
    model      : PMCModel — known parameters, scalar observations (d = 1).
    Y          : (N,) observations, NaN where missing (never flagged).
    alpha      : level in (0, 1) (default 1e-3): per row without
                 ``correction``, family-wise with ``"bonferroni"``, false
                 discovery rate with ``"bh"``.
    sequential : gate a flagged row out of the filter of the rows after it
                 (default True; module docstring, "Sequential gating").
                 False: every row enters the filter, p-values are those of
                 :func:`predictive_pit`.
    correction : ``None`` (default), ``"bonferroni"`` or ``"bh"`` — module
                 docstring, "Flagging and multiplicity".
    gap_nodes  : quadrature nodes G of the grid variants (default 64).

    Returns
    -------
    OutlierFlags — ``flagged ⇔ pvalue < threshold`` at every observed row.
    """
    alpha = _check_flag_args(alpha, correction)
    filt = _Filter(model, Y, gap_nodes)
    obs = ~filt.miss
    m = int(obs.sum())
    if correction == "bonferroni":
        thr = alpha / max(m, 1)
    elif correction == "bh":
        thr = float(np.nextafter(alpha / max(m, 1), np.inf))       # BH with k = 1
    else:
        thr = alpha

    if not sequential:
        out = filt.run(None)
        p = _pvalues(out["cdf"], out["sf"])
        if correction == "bh":
            thr = _bh_threshold(p[obs], alpha)
        with np.errstate(invalid="ignore"):
            flagged = obs & (p < thr)
    elif correction == "bh":
        for run in range(_BH_MAX_RUNS):
            out = filt.run(thr)
            p = _pvalues(out["cdf"], out["sf"])
            new = _bh_threshold(p[obs], alpha)
            if new == thr:
                break
            if run == _BH_MAX_RUNS - 1:
                logger.warning(
                    "flag_outliers: the Benjamini–Hochberg threshold of the gated "
                    "filter did not settle in %d runs (last %.3g, next %.3g); the flags "
                    "of the last run are returned.", _BH_MAX_RUNS, thr, new,
                )
                break
            thr = new
        flagged = out["gated"].copy()
    else:
        out = filt.run(thr)
        p = _pvalues(out["cdf"], out["sf"])
        flagged = out["gated"].copy()

    return OutlierFlags(
        flagged=flagged, pvalue=p, pit=out["cdf"], z=_normal_score(out["cdf"], out["sf"]),
        log_pred=out["log_pred"], threshold=float(thr), alpha=alpha, correction=correction,
        sequential=bool(sequential), n_tests=m, log_lik=out["log_lik"],
        alpha_hat=out["alpha_hat"], miss=filt.miss, method=filt.method,
    )


def _ljung_box(x: np.ndarray, lags: int) -> tuple[float, float]:
    n = x.size
    xc = x - x.mean()
    den = float(xc @ xc)
    if n <= lags + 1 or den <= 0.0:
        return float("nan"), float("nan")
    r = np.array([float(xc[:-k] @ xc[k:]) / den for k in range(1, lags + 1)])
    q = n * (n + 2.0) * float(np.sum(r * r / (n - np.arange(1, lags + 1))))
    return q, float(_ss.chi2.sf(q, lags))


def pit_checks(pit, *, lags: int = 10, exclude=None) -> PitChecks:
    """Uniformity and independence checks of a PIT sequence.

    Parameters
    ----------
    pit     : a :class:`PredictivePIT` or :class:`OutlierFlags`, or an (N,)
              array of PITs (NaN rows are skipped).
    lags    : number of lags of the Ljung–Box statistics (default 10).
    exclude : optional (N,) bool mask of rows to leave out (e.g. the flagged
              rows).

    Returns
    -------
    PitChecks — KS test of U(0, 1) on the PITs; Ljung–Box on the normal
    scores and on their squares, over the kept rows in order (the PITs of
    the observed rows of a correct model are iid, whatever the missing rows
    between them).
    """
    if hasattr(pit, "pit"):
        u = np.asarray(pit.pit, dtype=float)
        z = np.asarray(pit.z, dtype=float)
    else:
        u = np.asarray(pit, dtype=float)
        z = _normal_score(u, 1.0 - u)
    keep = np.isfinite(u)
    if exclude is not None:
        keep &= ~np.asarray(exclude, dtype=bool)
    u, z = u[keep], z[keep]
    lags = int(lags)
    if lags < 1:
        raise ValueError(f"lags must be ≥ 1, got {lags}.")
    if u.size < 2:
        raise ValueError("pit_checks needs at least two PITs.")
    ks = _ss.kstest(u, "uniform")
    lb, lbp = _ljung_box(z, lags)
    lb2, lb2p = _ljung_box(z * z, lags)
    return PitChecks(n=int(u.size), ks_stat=float(ks.statistic), ks_pvalue=float(ks.pvalue),
                     lb_stat=lb, lb_pvalue=lbp, lb2_stat=lb2, lb2_pvalue=lb2p, lags=lags,
                     z_mean=float(z.mean()), z_sd=float(z.std(ddof=1)))


# ---------------------------------------------------------------------------
# Robust estimation: flag, mask, refit
# ---------------------------------------------------------------------------

_FLAG_KEYS = ("alpha", "sequential", "correction", "gap_nodes")


def _refuse_nonignorable(model: PMCModel, cfg: dict, algorithm: str) -> None:
    table = model.ice_config() if algorithm == "ice" else model.sem_config()
    mode = {**table, **cfg}.get("missingness", "model")
    if mode in ("state", "state-markov") or (mode == "model" and model.missingness is not None):
        raise NotImplementedError(
            "robust_estimate masks the flagged rows as missing values; with a "
            "non-ignorable missingness mechanism (model.missingness or the config "
            "key missingness) they would be read as missing by that mechanism. "
            "Use missingness = 'ignorable', or flag with flag_outliers only."
        )


def robust_estimate(
    model: PMCModel,
    Y,
    estim_cfg: dict | None = None,
    *,
    algorithm: str = "ice",
    flag_cfg: dict | None = None,
    max_rounds: int = 10,
    initial_mask=None,
    progress_cb=None,
) -> RobustFit:
    """Flag-and-mask estimation: fit, flag, set the flags to NaN, refit.

    The design (masking as trimming, iteration against the masking effect,
    flags recomputed on the original Y, restarts from the initial model,
    breakdown, stopping rule) is in the module docstring, "Robust
    estimation".

    Parameters
    ----------
    model        : PMCModel — starting model of every fit (d = 1).
    Y            : (N,) observations, NaN where missing.
    estim_cfg    : ICE / SEM config dict, as for :func:`pmcprg.pmc.ice` /
                   :func:`pmcprg.pmc.sem`, used unchanged by every fit.
    algorithm    : ``"ice"`` (default) or ``"sem"``.
    flag_cfg     : keyword arguments of :func:`flag_outliers` (``alpha``,
                   ``sequential``, ``correction``, ``gap_nodes``); by default
                   its defaults, with the ``gap_nodes`` of ``estim_cfg``.
    max_rounds   : maximum number of refits after the first fit (default 10).
    initial_mask : optional (N,) bool — rows set to NaN for the first fit
                   (a pre-screen such as the Hampel filter of
                   ``report/erroneous_data/run_study.py``, sensor-level
                   flags); re-tested by the model from the first round on.
                   Needed when the spikes are frequent enough to form a
                   state of the first fit (module docstring, "Breakdown").
    progress_cb  : passed to every fit.

    Returns
    -------
    RobustFit
    """
    from pmcprg.pmc.ice import ice
    from pmcprg.pmc.sem import sem

    if algorithm not in ("ice", "sem"):
        raise ValueError(f"algorithm must be 'ice' or 'sem', got {algorithm!r}.")
    max_rounds = int(max_rounds)
    if max_rounds < 0:
        raise ValueError(f"max_rounds must be ≥ 0, got {max_rounds}.")
    Y = _check_scalar(model, Y)
    cfg = dict(estim_cfg or {})
    _refuse_nonignorable(model, cfg, algorithm)
    fkw = dict(flag_cfg or {})
    unknown = set(fkw) - set(_FLAG_KEYS)
    if unknown:
        raise ValueError(f"Unknown flag_cfg keys {sorted(unknown)}. Valid: {list(_FLAG_KEYS)}.")
    fkw.setdefault("gap_nodes", cfg.get("gap_nodes"))
    _check_flag_args(fkw.get("alpha", DEFAULT_ALPHA), fkw.get("correction"))
    fit = ice if algorithm == "ice" else sem
    miss = missing_mask(Y)
    if initial_mask is None:
        mask = np.zeros(len(Y), dtype=bool)
    else:
        mask = np.asarray(initial_mask, dtype=bool)
        if mask.shape != Y.shape:
            raise ValueError(f"initial_mask must have shape {Y.shape}, got {mask.shape}.")
        mask = mask & ~miss

    masks, traces = [], []
    while True:
        Ym = Y.copy()
        Ym[mask] = np.nan
        if masks:
            logger.info("robust_estimate: refit %d with %d masked row(s).",
                        len(masks), int(mask.sum()))
        fitted, trace = fit(model, Ym, cfg, progress_cb)
        masks.append(mask.copy())
        traces.append(trace)
        flags = flag_outliers(fitted, Y, **fkw)
        if np.array_equal(flags.flagged, mask) or len(masks) > max_rounds:
            break
        mask = flags.flagged.copy()
    converged = bool(np.array_equal(flags.flagged, mask))
    if not converged:
        logger.warning(
            "robust_estimate: the flagged rows still change after %d refit(s) "
            "(%d masked, %d flagged by the last fit).",
            len(traces) - 1, int(mask.sum()), flags.n_flagged,
        )
    return RobustFit(model=fitted, trace=trace, mask=mask, flags=flags, masks=masks,
                     traces=traces, converged=converged)
