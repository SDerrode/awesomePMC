"""
gaps.py — supervised inference with missing observations (NaN) for PMC/HMC models.

Public API
----------
gap_posterior(model, Y, *, gap_nodes=64) -> GapPosterior
    Filtering/smoothing quantities of a NaN-bearing sequence: log-likelihood of
    the observed data, α̂, β̂, γ, ξ and, for the grid variants, the posterior of
    (x_n, y_n) on the quadrature nodes at every missing position.
impute(model, Y, *, gap_nodes=64, quantiles=(0.05, 0.5, 0.95), n_samples=0, rng=None)
    -> Imputation
    Posterior mean, standard deviation and quantiles of every missing y_n given
    all the observed data; optionally joint FFBS draws of the missing values.
forecast(model, Y, h, *, gap_nodes=64, quantiles=(0.05, 0.5, 0.95)) -> Forecast
    h-step predictive law of (x_{N+k}, y_{N+k}), k = 1..h, given the observed
    part of Y (Y may itself contain NaN).
missing_mask(Y), needs_grid(model), reference_grid(model, gap_nodes)
    Helpers (mask of missing rows, whether a model needs the quadrature grid,
    the quadrature grid itself).

The NaN-aware entry points of :mod:`pmcprg.pmc.inference` (``classify``,
``forward``, ``backward``, ``precompute_weights``, ``sample_posterior``)
dispatch here; a Y without missing values never reaches this module.

Missing observations
--------------------
A row of Y is *missing* when it holds a non-finite value (NaN, ±inf):
``miss = ~np.isfinite(Y)`` for d = 1; for d > 1 a row with any non-finite
component is missing as a whole (partially observed rows are not supported in
this version — their finite components are ignored). The mask is m_n = 1 iff
row n is missing.

Missingness mechanisms
----------------------
The law of the mask belongs to the model (``model.missingness``, TOML table
``[missingness]``, :mod:`pmcprg.pmc.missingness`):

* **Ignorable** (MCAR/MAR — ``model.missingness is None``, the default): the
  mask carries no information beyond y_obs, and the quantities below are
  those of the observed-data likelihood p(y_obs) = ∫ p(y) dy_miss.
* **State-dependent, non-ignorable** (``"state"``, ``"state-markov"``):
  given the states the mask is independent of Y and p(m | x) = Π_n e_n(x_n),
  with e_n(i) = π_i^{m_n} (1 − π_i)^{1 − m_n} for ``"state"`` (independent
  masks, P(m_n = 1 | x_n = i) = π_i) and e_n(i) = p(m_n | m_{n-1}, x_n = i)
  for ``"state-markov"`` (a two-state Markov mask whose onset and
  persistence probabilities depend on the current state — bursts of gaps).
  Then

      p(y_obs, m) = Σ_x ∫ p(x, y) Π_n e_n(x_n) dy_miss,

  a selection model for data missing not at random (Little & Rubin 2019):
  the mask itself is evidence on the states (e.g. sensor dropouts more
  frequent during some activities). e_n depends on x_n only, so it
  multiplies the message at position n componentwise — the initial message
  by e_0, the columns of the transition into n + 1 by e_{n+1}, e_n(i) for
  every node g of an augmented state (i, g). It is a likelihood factor, not
  part of the transition kernel: it is applied after the Tauchen–Hussey
  block renormalisation below (and in ``precompute_weights``' W and f_pdf,
  after the transition, for the exact shortcut). log p(y_obs, m) =
  Σ_n log C_n as before, and γ, ξ, MPM, FFBS draws and imputations are given
  (y_obs, m). Given x, y_miss does not depend on m: the laws of the missing
  values given the states are unchanged, only the state posteriors move. A
  complete Y has the mask m = 0 and still gets its factors (1 − π_i for
  ``"state"``). ``forecast`` gives no factor to its h appended rows, whose
  mask is unknown (Σ_m p(m | x) = 1).

Z = (X, Y) is a Markov chain with transition q(j, y' | i, y) = W[n, i, j] for
y = y_n, y' = y_{n+1} (see :mod:`pmcprg.pmc.inference`) and initial density
μ(i, y) = α_1(i) evaluated at y = y_1.

1. **Exact shortcut** (HMC-IN, HMC-IN2, PMC-IN with state margins). The
   transition q(j, y' | i, y) = A_ij f_j(y') does not depend on y, and
   ∫ f_j(y') dy' = 1: a missing y_{n+1} contributes the factor 1 in place of
   f_j(y_{n+1}), and a missing y_1 gives α_1(j) = Σ_i p_ij. This is exact and
   stays a K-state recursion, so :func:`pmcprg.pmc.inference.precompute_weights`
   returns the marginalised weights (f_pdf row = 1 at a missing n) and every
   K-state function of :mod:`pmcprg.pmc.inference` is exact on them. Given the
   states, the missing y_n are independent with law f_{x_n}: imputation and
   forecasting use the exact mixtures Σ_j P(x_n = j | y_obs) f_j.

2. **Quadrature grid** (HMC-DN, PMC, PMC-IN with pair margins). The
   transition depends on the value of y_n through the copula and/or the pair
   margins, so a missing y_n is integrated out on an augmented state (i, g)
   over quadrature nodes y_g.

   * Reference law g_ref(y) = Σ_i μ(i, y) = Σ_{i,j} p_ij f_ij(y) (pair margins)
     or Σ_j (Σ_i p_ij) f_j(y) (state margins; π_j f_j for the HMC variants) —
     the law of y_1, the stationary law of y_n for a stationary chain.
   * Nodes y_g = G_ref^{-1}(t_g), t_g = ψ(s_g), with (s_g, ws_g) the G-point
     Gauss–Legendre rule on (0, 1) and ψ(s) = s² / (s² + (1 − s)²) an
     endpoint transform; the weights w_g = ws_g ψ'(s_g) integrate over t, and
     ω_g = w_g / g_ref(y_g), so ∫ h(y) dy ≈ Σ_g ω_g h(y_g). Without ψ (plain
     Gauss–Legendre in t = G_ref(y)) the integrands, a conditional density
     over g_ref, behave like t^β at t → 0, 1 (e.g. β = 2ρ²/(1 − ρ²) for a
     Gaussian copula ρ between N(0, 1) margins): an endpoint singularity that
     makes the rule converge only algebraically — 1e-4 errors at G = 64 for
     ρ = 0.5 — whereas ψ restores fast convergence (1e-7 at G = 64, see the
     accuracy tests). G_ref^{-1} is a vectorised bisection on the mixture CDF
     (on the survival function in the upper half), bracketed by the component
     quantiles. Default G = 64 (``gap_nodes``). This *reference grid* serves
     every missing position whose laws it resolves; the others get a local
     grid (below). Each missing position n has its own grid, nodes y_{n,g}
     and weights ω_{n,g}.
   * Transitions of the augmented chain (augmented index u = i·G + g):

       observed n → missing n+1  E[i, (j, g)]        = q(j, y_{n+1,g} | i, y_n) ω_{n+1,g}
       missing n → missing n+1   Q[(i, g), (j, g')]  = q(j, y_{n+1,g'} | i, y_{n,g}) ω_{n+1,g'}
       missing n → observed n+1  X[(i, g), j]        = q(j, y_{n+1} | i, y_{n,g})
       observed → observed       W[n, i, j]          (unchanged)
       missing y_1               α̃_1(i, g)           = μ(i, y_{1,g}) ω_{1,g}

     Q between two reference grids does not depend on the data and is built
     once per call; a step touching a local grid gets its own Q.
   * Renormalisation (Tauchen & Hussey 1991). The quadrature does not
     integrate the transition density exactly, but two of its integrals are
     known: Σ_g of the block (i, y) → (j, ·) must be P(x_{n+1} = j | x_n = i,
     y_n = y) — A_ij, or p_ij f_ij(y) / Σ_k p_ik f_ik(y) with pair margins —
     and the block (·, g) of α̃_1 must be P(x_1 = i). Every block of E, Q and
     α̃_1 is rescaled to that mass (then every row to 1, a no-op unless a
     block vanished). A trailing gap then contributes exactly log C = 0, a
     sequence with no observed value has log-likelihood 0, and for state
     margins (X Markov) the state marginals across a gap are exact.
     Measured on the references of the test-suite, this also lowers the
     likelihood and moment errors 2–5× against plain row renormalisation.
   * Forward and backward run the Devijver/Rabiner-scaled recursions of
     :mod:`pmcprg.pmc.inference` on this chain of variable width (K at an
     observed position, K·G at a missing one); log p(y_obs) = Σ_n log C_n.
     γ_n(i) = Σ_g α̃_n(i, g) β̃_n(i, g) at a missing n, ξ_n by summing the
     augmented pair posterior over the nodes. Copula arguments are clipped to
     [EPS, 1 − EPS] as in ``precompute_weights``.
   * Log space. When a linear step underflows (a normaliser not finite or
     below MIN_POSITIVE) or a kernel value overflows (e.g. a copula density
     beyond 1.8e308), the whole pass is redone in log space from the exact
     log-densities (``logpdf_array`` of the copulas).
   * Laws of the missing values. Moments of y_n are the quadratures
     Σ_g mass_g y_g^k of the node posterior. Quantiles need the CDF between
     the nodes: the density of y_n at any y is interpolated from the messages
     of the neighbouring positions through the exact kernels (Nyström
     interpolation, :func:`_nystrom_density`) on the same grid with four
     times more nodes, and the CDF of its Legendre interpolant is inverted
     (on a local grid, panel by panel: the CDF at a panel boundary is the
     mass of the panels below). (Interpolating the G node masses directly
     loses half of the polynomial degree: quantile errors 1e-4 where the
     moments are at 1e-6.) ``forecast`` uses the normalised forward messages
     of a trailing gap of length h. The Nyström density propagates the
     discrete message of the previous position through the exact kernel, not
     through the chain's renormalised rows: a row whose kernel the grid
     misses has a block factor of up to 1e30, which the former interpolation
     applied off the nodes — the P3 failure of ``report/forecasting`` (2-step
     forecast quantiles 13.4 / 15.8 / 16.7 for a law of mean 4.26, sd 1.42;
     now 3.04 / 4.00 / 10.47 at every G ≥ 128, within 4e-3 at G = 32).
     Safety net: where the interpolated CDF is not monotone or leaves the
     Markov–Stieltjes bracket [Σ_{g'<g} m_g', Σ_{g'≤g} m_g'] of the node
     masses by more than 1e-6, the piecewise-linear CDF of the cumulative
     masses at the cell edges is inverted instead (first-order accurate) and
     a WARNING is logged. The ``density`` of :class:`Imputation` and
     :class:`Forecast` is given at the reference nodes (mass / ω there, the
     normalised Nyström density for a position on a local grid);
     ``grid_nodes`` / ``grid_mass`` hold each position's own discrete law.

Local grids
-----------
The reference grid places its nodes in the quantiles of the *stationary* law.
Under strong serial dependence — copulas at τ ≈ 0.99, as for a sensor sampled
every 30 s — the law of a missing y given its neighbours is far narrower than
the node spacing and the quadrature fails silently: a Gaussian AR(1) at ρ =
0.9999 written as a PMC gets a 2-step forecast sd of 0.0000 for 0.0200, 843
of 3000 rows flagged by :func:`pmcprg.pmc.outliers.flag_outliers` for 2, and
the log-likelihood of Intel Lab mote 48 moves by 13 600 nats between G = 64
and 512 (``report/erroneous_data/intel_lab``). Every missing position whose
laws the reference grid does not resolve gets its own grid instead:

* **Proposal.** Each run of missing rows [a, b] is summarised by Gaussian
  pieces (m, s): forward pieces from y_{a−1} (one per transition i → j: the
  exact conditional law of y_a, through the copula's h-inverse
  y = F_ji^{-1}(h_ij^{-1}(t | F_ij(y_{a−1}))), then propagated along the gap
  by a Gaussian-sum filter whose classes keep the paths that stayed in their
  state apart from the others), backward pieces from y_{b+1} (the law of y_b
  given (i → j, y_{b+1}) under μ(i, y_b) q(j, y_{b+1} | i, y_b): the
  transposed copula's h-inverse), and their products — the *bridges*, where a
  y pinned by both neighbours sits even across a jump of 20 innovations.
  Location Q(1/2), core scale (Q(Φ(1)) − Q(Φ(−1)))/2 and a tail extent from
  Q(Φ(±2)), Q(Φ(±3)) (skewed or heavy-tailed laws, Clayton and Gumbel in
  their dependent tail). Weights: the one-observation state law of the
  neighbour times the transition; bridges ½, forward and backward pieces ¼
  each; near-identical pieces merged, six kept.
* **Grid.** A composite rule on disjoint panels: every narrow piece owns a
  *core* panel m ± 8 e s (e the tail extent), cut where cores meet and given
  to the piece whose density dominates there, with Gauss–Legendre in u for
  y = m + 2s·sinh(u) (32 nodes on m ± 8s: 1e-14 for N(m, s²), 5e-10 for
  N(m + 1.5s, (0.7s)²), exact for a constant — narrow and flat integrands
  alike); the rest of the support of g_ref is covered by *background* panels,
  ψ-Gauss–Legendre in its probability scale restricted to them. 35 % of the
  nodes go to the background (the broad paths: state switches under weak
  copulas, the prior in a leading gap), the rest to the cores ∝ √(weight).
  Disjoint panels, not a single mixture proposal: a mixture puts steps into
  the transformed integrand where a component's mass ends (1e-3 errors with
  a g_ref defensive component), and a multiple-importance rule leaks each
  narrow piece's tails into the coarse rules (5.6e-4); a panel only ever
  integrates a function that is smooth on it.
* **Selection.** A position keeps the reference grid unless a piece of weight
  ≥ 1e-4 has fewer than 4 reference nodes within one scale, the reference
  grid misintegrates the pieces and the transition kernels into the position
  (their mass and second moment, :func:`_proposal_error`) by more than 1e-6,
  the local grid does 3 times better, and it is not worse on the integrals
  known in closed form next to an observed row (:func:`_neighbour_error`: the
  entry masses T_ij(y_{a−1}), the exit masses ∫ μ(i, y) q(k, y_{b+1} | i, y)
  dy, which see support boundaries and skewed laws the Gaussian pieces
  miss) nor on the prior at a missing y_1 (within 1e-2). Weak and moderate
  dependence keep the reference grid everywhere, bit for bit.
* **Exactness kept.** The block renormalisation applies unchanged: a trailing
  gap still contributes log C = 0 and the state marginals of state margins
  stay exact; the log-space fallback rebuilds the same grids. The filter of
  :mod:`pmcprg.pmc.outliers` builds the grids of a run as it enters it
  (:func:`_run_grids`), the same as :func:`_gap_grids` on the same missing
  rows: its PITs and log-likelihood equal ``gap_posterior``'s.
* **Measured** (exact references; ``test_gaps_local_grids.py``): AR(1) at
  ρ = 0.998, 0.999, 0.9999, every gap pattern, G = 64: log-likelihood
  2.4e-7–5.0e-7, means, sds and quantiles 3e-8–9e-7 of the conditional sd
  (reference grid: 0.24–35 nats, sds off by 41–180 %); forecasts to 2e-9;
  the PIT after a gap to 3e-12 (0.85 before). Pair-margin PMCs at τ = 0.9–0.99
  against brute force (the ten cases of the test): 7e-8–3.8e-2 at G = 64
  and 2e-13–2.4e-4 at G = 128, where the reference grid gave 5.7e-3–5.6 and
  2.6e-5–1.0; on every case measured, a local grid is never worse than the
  reference grid it replaces. Intel mote 48: 65 641.05 / 65 640.28 /
  65 640.245 / 65 640.245 nats at G = 64 / 128 / 256 / 512 (reference grid:
  51 798 / 57 574 / 63 096 / 65 364). The grid still has G nodes: a gap
  longer than ~20 rows at ρ = 0.9999 (the law of the middle rows broadens as
  √L while the kernel stays narrow) or six separated narrow pieces need a
  larger G — the diagnostic below says so.
* **Cost** (forward pass, N = 3000, 10 % single gaps, Gaussian copulas,
  G = 64): τ = 0.6, no candidate: 38 against 34 ms (K = 2), 64 against
  52 ms (K = 3); τ = 0.9, candidates screened, few or none switched: 101
  against 36 ms, 152 against 53 ms; τ = 0.99, every gap local: 112 against
  34 ms, 175 against 52 ms. Intel mote 48 (3 592 missing rows, 491 of them
  inside longer gaps): 3.1 s against 0.6 s. The posterior pass adds the
  same backward pass as before.

Convergence diagnostic
----------------------
:func:`_quadrature_report` checks, after each pass, integrals every grid must
reproduce and whose exact values are known — the raw block masses before the
renormalisation, weighted by the posterior mass through them, and the exit
masses of the local grids — and sums their relative errors over the missing
positions (``GapPosterior.quad_error``). The rows inside a leading gap (which
carries the prior forward) and from the background panels of a local grid
(which carry a broad law) are left out: the renormalisation conserves that
mass wherever it relocates it. A WARNING ("Missing-data quadrature not
converged … increase gap_nodes") is logged above ``QUAD_WARN`` · √R = 1e-3 ·
√R, R the number of runs of missing rows (:func:`quad_warn_limit`): the
report adds unsigned errors over the runs, whose likelihood errors partly
cancel. It is a screen, not an error bound. Measured:

* 144 AR(1) runs with the local grids (ρ = 0.5–0.9999, G = 32, 64, 128,
  three variants, 10 runs of missing rows each): no warning; none is off by
  more than 1e-2, 6 (at G = 32) by 1.1e-3–1.2e-3.
* The same runs on the reference grid alone: 54 of the 66 runs off by more
  than 1e-2 would be flagged; the 12 missed (ρ = 0.995 at G = 64, ρ = 0.999
  at G = 128) are laws whose masses the grid still integrates to 1e-4 while
  their second moments are 2–5 % off.
* One gap of 50 rows at ρ = 0.9999, G = 64: 3.5e-2, flagged (1.1e-2 nats
  off); 9.8e-8 at G = 128 (6e-8 off). Jumps of 5–40 innovations across a
  missing row: ≤ 2.6e-4, quiet (≤ 2.6e-4 off).
* Randomised 3-state pair-margin models at G = 64: 3 of the 4 runs off by
  more than 1e-2 flagged (the one missed, 0.14 nats, keeps the reference
  grid on six separated pieces per gap and converges at G = 256).
* Intel mote 48 (3 090 runs; limit 0.056): 14 at G = 64 (0.81 nats off),
  0.16 at G = 128 (0.04 off), 0.039 at G = 256 (< 1e-3 off, quiet).

Returned α̂ and β̂ (K-state view of the augmented chain)
-------------------------------------------------------
``alpha_hat[n, i] = P(x_n = i | y_obs ∩ 1:n)`` at every n (missing or not).
``beta_hat[n]`` is the normalised backward message at an observed n; at a
missing n there is no K-state backward message independent of the forward
(y_n couples x_{n-1} and x_{n+1}), so it is defined as
``beta_hat[n, i] ∝ Σ_g α̃_n(i, g) β̃_n(i, g) / Σ_g α̃_n(i, g)`` — the
likelihood of the future observations given x_n = i and the past ones. With
this definition ``smooth(alpha_hat, beta_hat)`` is exact at every n. No K×K
tensor W makes ``joint_posteriors(alpha_hat, W, beta_hat)`` exact for the grid
variants (the posterior of X given y_obs is not Markov across a gap): use
``gap_posterior(model, Y).xi``.

References
----------
* Tauchen, G. & Hussey, R. (1991). Quadrature-based methods for obtaining
  approximate solutions to nonlinear asset pricing models. *Econometrica*
  59(2), 371–396 — row-renormalised quadrature discretisation of a Markov
  kernel.
* Kitagawa, G. (1987). Non-Gaussian state-space modeling of nonstationary
  time series. *J. Amer. Statist. Assoc.* 82(400), 1032–1041 — numerical
  integration of filtering densities on a grid.
* Nyström, E. J. (1930). Über die praktische Auflösung von Integralgleichungen
  mit Anwendungen auf Randwertaufgaben. *Acta Math.* 54, 185–204 — the
  interpolation of a quadrature solution through its kernel.
* Alspach, D. L. & Sorenson, H. W. (1972). Nonlinear Bayesian estimation
  using Gaussian sum approximations. *IEEE Trans. Automat. Control* 17(4),
  439–448 — the Gaussian-sum propagation of the local proposals.
* Davis, P. J. & Rabinowitz, P. (1984). *Methods of Numerical Integration*,
  2nd ed., Academic Press — composite Gauss–Legendre rules and variable
  transformations (the sinh map of the core panels).
* Little, R. J. A. & Rubin, D. B. (2019). *Statistical Analysis with Missing
  Data*, 3rd ed., Wiley — ignorable missingness, observed-data likelihood;
  selection models for data missing not at random (the ``[missingness]``
  mechanisms).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, replace

import numpy as np
from numpy.polynomial import legendre as _leg
from scipy import stats as _ss
from scipy.special import ndtr as _sp_ndtr

from pmcprg.exceptions import IncompatibleObservationError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, MIN_POSITIVE
from pmcprg.pmc import inference as _inf
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_GAP_NODES",
    "DEFAULT_QUANTILES",
    "QuadratureGrid",
    "GapPosterior",
    "Imputation",
    "Forecast",
    "missing_mask",
    "needs_grid",
    "reference_grid",
    "gap_posterior",
    "impute",
    "forecast",
]

#: Default number of Gauss–Legendre nodes of the quadrature grid.
DEFAULT_GAP_NODES = 64
#: Default quantile levels reported by :func:`impute` and :func:`forecast`.
DEFAULT_QUANTILES = (0.05, 0.5, 0.95)


# ---------------------------------------------------------------------------
# Masks and dispatch helpers
# ---------------------------------------------------------------------------

def missing_mask(Y) -> np.ndarray:
    """Boolean mask (N,) of the missing rows of Y (any non-finite component)."""
    fin = np.isfinite(np.asarray(Y, dtype=float))
    if fin.ndim > 1:
        fin = fin.reshape(fin.shape[0], -1).all(axis=1)
    return ~fin


def needs_grid(model: PMCModel) -> bool:
    """True when a missing y must be integrated on the quadrature grid.

    The transition depends on the value of y_n for HMC-DN and PMC (copula) and
    for PMC-IN with pair margins (A_ij(y_n) ∝ p_ij f_ij(y_n)); HMC-IN, HMC-IN2
    and PMC-IN with state margins admit the exact shortcut.
    """
    return bool(model.variant.uses_copula or model.margin_structure == "pair")


def _resolve_nodes(gap_nodes) -> int:
    G = DEFAULT_GAP_NODES if gap_nodes is None else int(gap_nodes)
    if G < 2:
        raise ValueError(f"gap_nodes must be an integer ≥ 2, got {gap_nodes!r}.")
    return G


def _check_quantiles(quantiles) -> tuple[float, ...]:
    qs = tuple(float(q) for q in quantiles)
    if any(not (0.0 < q < 1.0) for q in qs):
        raise ValueError(f"quantiles must lie in (0, 1); got {qs}.")
    return qs


def _check_grid_supported(model: PMCModel) -> None:
    if getattr(model, "d", 1) != 1:
        raise NotImplementedError(
            f"Missing observations for variant {model.variant.value} with "
            f"{model.margin_structure} margins need the quadrature grid, which "
            f"is implemented for scalar observations only (model.d = {model.d})."
        )


# ---------------------------------------------------------------------------
# Mixtures of scalar laws: CDF and quantile by vectorised bisection
# ---------------------------------------------------------------------------

def _frozen(margin):
    fr = getattr(margin, "_frozen", None)
    if fr is None:
        raise TypeError(f"margin {margin!r} exposes no frozen scipy.stats law.")
    return fr


def _mixture_ppf(weights: np.ndarray, dists: list, q: np.ndarray,
                 qc: np.ndarray | None = None) -> np.ndarray:
    """Quantiles of the mixtures Σ_c weights[m, c] · dists[c].

    ``weights`` (M, C) rows summing to 1, ``q`` (M, Q) levels in (0, 1) and
    optionally ``qc`` = 1 − q computed without cancellation (upper tail).
    Bracket [min_c F_c^{-1}(q), max_c F_c^{-1}(q)] over the components of
    positive weight (the mixture CDF is ≤ q below every component quantile and
    ≥ q above them), then bisection to relative width 1e-13 — on the CDF for
    q ≤ 1/2 and on the survival function against ``qc`` above, so that levels
    within 1e-16 of 1 keep their precision.
    """
    weights = np.asarray(weights, dtype=float)
    q = np.asarray(q, dtype=float)
    qc = (1.0 - q) if qc is None else np.asarray(qc, dtype=float)
    upper = q > 0.5
    q = np.clip(q, MIN_POSITIVE, 1.0)
    qc = np.clip(qc, MIN_POSITIVE, 1.0)
    pos = weights > 0.0
    lo = np.full(q.shape, np.inf)
    hi = np.full(q.shape, -np.inf)
    for c, dist in enumerate(dists):
        with np.errstate(all="ignore"):
            yc = np.where(upper, dist.isf(qc), dist.ppf(q))
        active = pos[:, c][:, None]
        lo = np.where(active & (yc < lo), yc, lo)
        hi = np.where(active & (yc > hi), yc, hi)
    target = np.where(upper, qc, q)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        F = np.zeros(q.shape)
        for c, dist in enumerate(dists):
            F += weights[:, c][:, None] * np.where(upper, dist.sf(mid), dist.cdf(mid))
        left = np.where(upper, F > target, F < target)      # the root lies above mid
        lo = np.where(left, mid, lo)
        hi = np.where(left, hi, mid)
        if np.all(hi - lo <= 1e-13 * np.maximum(1.0, np.abs(mid))):
            break
    return 0.5 * (lo + hi)


def _ref_ppf(weights, dists, t: np.ndarray, tc: np.ndarray) -> np.ndarray:
    """Quantiles of g_ref = Σ_c weights[c] dists[c] at levels t (tc = 1 − t), for local grids.

    Safeguarded Newton inside the bracket of the component quantiles (on the
    survival function for t > 1/2), each element stopping on its own — its
    value does not depend on the other elements of the batch, so the filter
    of :mod:`pmcprg.pmc.outliers` and the batch passes build the same grids.
    """
    t = np.asarray(t, dtype=float)
    tc = np.asarray(tc, dtype=float)
    upper = t > 0.5
    target = np.where(upper, np.maximum(tc, MIN_POSITIVE), np.maximum(t, MIN_POSITIVE))
    lo = np.full(t.shape, np.inf)
    hi = np.full(t.shape, -np.inf)
    for w_, dist in zip(weights, dists):
        if w_ <= 0.0:
            continue
        with np.errstate(all="ignore"):
            yc = np.where(upper, dist.isf(target), dist.ppf(target))
        lo = np.minimum(lo, yc)
        hi = np.maximum(hi, yc)
    y = 0.5 * (lo + hi)
    todo = np.nonzero(hi > lo)[0]
    for _ in range(200):
        if not todo.size:
            break
        yy, up, tg = y[todo], upper[todo], target[todo]
        F = np.zeros(yy.shape)
        f = np.zeros(yy.shape)
        for w_, dist in zip(weights, dists):
            with np.errstate(all="ignore"):
                F += w_ * np.where(up, dist.sf(yy), dist.cdf(yy))
                f += w_ * dist.pdf(yy)
        diff = F - tg
        above = np.where(up, diff > 0.0, diff < 0.0)          # the root lies above yy
        l_ = np.where(above, yy, lo[todo])
        h_ = np.where(above, hi[todo], yy)
        with np.errstate(all="ignore"):
            yn = np.where(up, yy + diff / f, yy - diff / f)
        yn = np.where(np.isfinite(yn) & (yn > l_) & (yn < h_), yn, 0.5 * (l_ + h_))
        conv = (np.abs(diff) <= 1e-14 * tg) | (h_ - l_ <= 1e-13 * np.maximum(1.0, np.abs(yy)))
        y[todo] = np.where(conv, yy, yn)
        lo[todo], hi[todo] = l_, h_
        todo = todo[~conv]
    return y


# ---------------------------------------------------------------------------
# Reference law and quadrature grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Panels:
    """Panel structure of a local grid (module docstring, "Local grids").

    The real line is cut into disjoint panels, each carrying a Gauss–Legendre
    rule. A core panel (``loc`` finite) is the interval y = loc + sc·sinh(u),
    u ∈ [t_lo, t_hi]; a background panel (``loc`` NaN) is the probability
    range [t_lo, t_hi] of g_ref with the endpoint transform ψ (``tc_lo``,
    ``tc_hi`` the complements, kept without cancellation). ``sizes`` holds
    the nodes of every panel and ``panel`` (G,) the panel of every node,
    nodes sorted by value.
    """

    loc: np.ndarray
    sc: np.ndarray
    t_lo: np.ndarray
    t_hi: np.ndarray
    tc_lo: np.ndarray
    tc_hi: np.ndarray
    sizes: np.ndarray
    panel: np.ndarray


@dataclass(frozen=True)
class QuadratureGrid:
    """Quadrature rule of a missing y (module docstring).

    The reference grid is a single ψ–Gauss–Legendre rule in the quantiles of
    g_ref. A *local* grid (``mix`` not None) is a composite rule on disjoint
    panels (:class:`_Panels`; module docstring, "Local grids"); its per-node
    ``s``, ``ws``, ``t``, ``w`` are those of the panel of the node.

    Attributes
    ----------
    s       : (G,) Gauss–Legendre nodes on (0, 1) (per panel).
    ws      : (G,) Gauss–Legendre weights on (0, 1) (per panel).
    power   : exponent p of the endpoint transform t = ψ_p(s).
    t       : (G,) the nodes in the probability scale of their law.
    w       : (G,) ws_g ψ_p'(s_g) (× the panel's t-width for a local grid)
              — weights of ∫ · dt.
    nodes   : (G,) the node values y_g, increasing.
    ref_pdf : (G,) the density of the law of the node's panel at y_g —
              g_ref(y_g) for the reference grid.
    omega   : (G,) ω_g = w_g / ref_pdf_g — ∫ h(y) dy ≈ Σ_g ω_g h(y_g).
    weights : (C,) mixture weights of the reference law g_ref.
    dists   : list of the C frozen component laws of g_ref.
    mix     : None for the reference grid, the :class:`_Panels` of a local
              grid.
    exit_error : for a local grid next to an observed row after it, its exit
              check (:func:`_neighbour_error`), kept for the diagnostic.
    """

    s: np.ndarray
    ws: np.ndarray
    power: int
    t: np.ndarray
    w: np.ndarray
    nodes: np.ndarray
    ref_pdf: np.ndarray
    omega: np.ndarray
    weights: np.ndarray
    dists: list
    mix: _Panels | None = None
    exit_error: float | None = None

    @property
    def G(self) -> int:
        return int(self.s.size)

    @property
    def local(self) -> bool:
        """True for a local grid (module docstring, "Local grids")."""
        return self.mix is not None

    def y_of_s(self, s: np.ndarray, panel: np.ndarray | None = None) -> np.ndarray:
        """The point of GL variable s: R^{-1}(ψ_p(s)) (upper tail without cancellation).

        R is g_ref's CDF for the reference grid; for a local grid, the
        restricted CDF of each ``panel`` (same shape as ``s``).
        """
        s = np.asarray(s, dtype=float)
        if self.mix is None:
            t, tc, _ = _psi(s.reshape(1, -1), self.power)
            return _mixture_ppf(self.weights[None, :], self.dists, t, tc).reshape(s.shape)
        k = np.asarray(panel, dtype=int).ravel()
        y, _, _ = _panel_points(self.mix, k, s.ravel(), self.power, self.weights, self.dists)
        return y.reshape(s.shape)


def _psi(s: np.ndarray, p: int):
    """Endpoint transform t = s^p / (s^p + (1-s)^p): (t, 1 − t, dt/ds).

    p = 1 is the identity. For p ≥ 2, ψ' vanishes to order p − 1 at both ends,
    which turns an integrand behaving like t^β at t → 0 or 1 (an algebraic
    endpoint singularity, typical of a conditional density divided by g_ref)
    into one behaving like s^{p(β+1)−1}, restoring fast Gauss–Legendre
    convergence.
    """
    s = np.asarray(s, dtype=float)
    if p == 1:
        return s, 1.0 - s, np.ones_like(s)
    a, b = s ** p, (1.0 - s) ** p
    den = a + b
    dt = p * (s * (1.0 - s)) ** (p - 1) / (den * den)
    return a / den, b / den, dt


def _reference_mixture(model: PMCModel) -> tuple[np.ndarray, list]:
    """Weights and frozen components of g_ref = Σ_i μ(i, ·)."""
    p = model.prior_p
    K = model.K
    if model.margin_structure == "pair":
        pairs = [(p[i, j], model.margin(i, j)) for i in range(K) for j in range(K)]
    else:
        col = p.sum(axis=0)
        pairs = [(col[j], model.margin(j)) for j in range(K)]
    w = np.array([max(float(c[0]), 0.0) for c in pairs])
    keep = w > 0.0
    if not keep.any():
        raise ValueError("The prior gives zero probability to every state.")
    dists = [_frozen(mg) for (_, mg), k in zip(pairs, keep) if k]
    return w[keep] / w[keep].sum(), dists


#: Exponent p of the endpoint transform ψ_p of the Gauss–Legendre nodes (see
#: :func:`_psi`). p = 2 measured best overall on the Gaussian AR(1) reference
#: (pmcprg/tests/test_gaps.py): p = 1 converges only algebraically for weak or
#: moderate dependence (|ρ| ≲ 0.7), p = 3 under-resolves the interior for
#: ρ ≈ 0.99. Private switch kept for accuracy studies.
_T_POWER = 2


def reference_grid(model: PMCModel, gap_nodes: int | None = None) -> QuadratureGrid:
    """The G-node quadrature grid of the reference law g_ref of ``model``."""
    _check_grid_supported(model)
    G = _resolve_nodes(gap_nodes)
    x, W = _leg.leggauss(G)
    s = 0.5 * (x + 1.0)
    ws = 0.5 * W
    t, tc, dt = _psi(s, _T_POWER)
    w = ws * dt
    weights, dists = _reference_mixture(model)
    nodes = _mixture_ppf(weights[None, :], dists, t[None, :], tc[None, :])[0]
    ref_pdf = np.zeros(G)
    for c, dist in enumerate(dists):
        ref_pdf += weights[c] * dist.pdf(nodes)
    if not np.all(np.isfinite(ref_pdf) & (ref_pdf > 0.0)):
        raise ValueError(
            "reference_grid: the reference density g_ref vanishes at a "
            "quadrature node; the margins' supports leave a gap the grid "
            "cannot represent."
        )
    return QuadratureGrid(s=s, ws=ws, power=_T_POWER, t=t, w=w, nodes=nodes,
                          ref_pdf=ref_pdf, omega=w / ref_pdf, weights=weights,
                          dists=dists)


# ---------------------------------------------------------------------------
# Local grids: proposals adapted to the conditional laws around each gap
# ---------------------------------------------------------------------------

#: Private switch kept for accuracy studies: ``False`` puts every missing
#: position on the reference grid (the quadrature before local grids).
_LOCAL_GRIDS = True
#: Shares of the bridge, forward and backward groups of a local proposal
#: when both neighbours of the gap are observed (module docstring).
_W_GROUPS = (0.5, 0.25, 0.25)
#: Gaussian components kept per proposal, after merging.
_MAX_COMPONENTS = 6
#: Two components merge when their locations differ by < _MERGE_LOC times
#: the smaller scale and their scales by a factor < _MERGE_SCALE.
_MERGE_LOC = 0.25
_MERGE_SCALE = 1.25
#: A position keeps the reference grid when every proposal component of
#: weight ≥ _RESOLVE_MIN_WEIGHT has at least _RESOLVE_NODES reference nodes
#: within one scale of its location (module docstring, "Local grids").
_RESOLVE_NODES = 4
_RESOLVE_MIN_WEIGHT = 1e-4
#: Positions with a component spanning fewer reference nodes than this within
#: one scale are candidates for a local grid (:func:`_local_grid_list`).
_CANDIDATE_NODES = 4
#: A candidate switches to its local grid when the reference grid's error on
#: the proposal exceeds _SWITCH_TOL and the local grid's is _SWITCH_GAIN
#: times smaller (:func:`_local_grid_list`).
_SWITCH_TOL = 1e-6
#: A local grid of a missing y_1 must integrate the prior law to this
#: relative accuracy (:func:`_local_grid_list`).
_PRIOR_GUARD = 1e-2
_SWITCH_GAIN = 3.0
#: Share of the G nodes of a local grid on the background (g_ref) panels,
#: and the half-width of a core panel, in scales of its Gaussian.
_BACKGROUND_NODES = 0.35
_CORE_WIDTH = 8.0
#: Scale c of the sinh map y = m + c·s·sinh(u) of a core panel, in scales s
#: of its Gaussian: plain Gauss–Legendre in u integrates both a Gaussian
#: of scale s and a flat integrand accurately (32 nodes on m ± 8s: 1e-14
#: for N(m, s²), 5e-10 for N(m + 1.5s, 0.7²s²), exact for a constant).
_SINH_SCALE = 2.0
#: Adjacent core pieces share one panel when their Gaussians' scales differ
#: by a factor ≤ _NEST_RATIO and the panel spans at most _MERGE_SPAN scales
#: of the narrower; a narrower core inside a wider one, or a chain of cores
#: strung along the line, gets several panels.
_NEST_RATIO = 4.0
_MERGE_SPAN = 3.0 * _CORE_WIDTH
#: Fewest nodes of a core panel and of a background panel.
_MIN_CORE_NODES = 8
_MIN_BACKGROUND_NODES = 4
#: Standard normal levels of the location / scale summaries of a law:
#: m = Q(1/2), core scale s = (Q(Φ(1)) − Q(Φ(−1))) / 2 and tail extent
#: e = max_k (Q(Φ(k)) − Q(Φ(−k))) / (2k s) ≥ 1, k = 1, 2, 3: a skewed or
#: heavy-tailed law (a Clayton or Gumbel conditional law in its dependent
#: tail) keeps its narrow core scale — what the grid must resolve — and gets
#: a core panel wide enough for its tails; e = 1 for a Gaussian law.
_LEVELS = np.array([float(_ss.norm.cdf(k)) for k in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)])
_MID = 3
_SQRT_2PI = float(np.sqrt(2.0 * np.pi))

_GL_CACHE: dict = {}


def _gl(n: int):
    """Gauss–Legendre rule of n nodes on (0, 1): (s, ws), cached."""
    r = _GL_CACHE.get(n)
    if r is None:
        x, W = _leg.leggauss(n)
        r = _GL_CACHE[n] = (0.5 * (x + 1.0), 0.5 * W)
    return r


def _ref_pdf(weights, dists, y: np.ndarray) -> np.ndarray:
    """g_ref(y), elementwise."""
    out = np.zeros(np.shape(y))
    for wr, dist in zip(weights, dists):
        with np.errstate(all="ignore"):
            out = out + wr * dist.pdf(y)
    return out


def _ref_cdf_sf(weights, dists, y: np.ndarray):
    """G_ref(y) and 1 − G_ref(y) (from the survival functions), elementwise."""
    c = np.zeros(np.shape(y))
    sf = np.zeros(np.shape(y))
    for wr, dist in zip(weights, dists):
        with np.errstate(all="ignore"):
            c = c + wr * dist.cdf(y)
            sf = sf + wr * dist.sf(y)
    return c, sf


def _panel_points(pan: _Panels, k: np.ndarray, s: np.ndarray, power: int, weights, dists):
    """Points of GL variable ``s`` in panels ``k``: (y, pdf, dw) with ω = ws · dw / pdf.

    Core panel: u = t_lo + (t_hi − t_lo) s, y = loc + sc sinh(u), dw = t_hi −
    t_lo and pdf = 1 / (sc cosh u). Background panel: t = t_lo + Δ ψ(s) (its
    complement from ψ's, without cancellation), y = G_ref^{-1}(t), pdf =
    g_ref(y) and dw = Δ ψ'(s).
    """
    tlo, thi, tclo, tchi = pan.t_lo[k], pan.t_hi[k], pan.tc_lo[k], pan.tc_hi[k]
    loc, sc = pan.loc[k], pan.sc[k]
    core = np.isfinite(loc)
    y = np.empty(s.shape)
    pdf = np.empty(s.shape)
    dw = np.empty(s.shape)
    if core.any():
        du = thi[core] - tlo[core]
        u = tlo[core] + du * s[core]
        y[core] = loc[core] + sc[core] * np.sinh(u)
        pdf[core] = 1.0 / (sc[core] * np.cosh(u))
        dw[core] = du
    bg = ~core
    if bg.any():
        psi, psic, dpsi = _psi(s[bg], power)
        D = np.where(tlo[bg] <= 0.5, thi[bg] - tlo[bg], tclo[bg] - tchi[bg])
        t = tlo[bg] + D * psi
        tc = tchi[bg] + D * psic
        yy = _ref_ppf(weights, dists, t, tc)
        y[bg] = yy
        pdf[bg] = _ref_pdf(weights, dists, yy)
        dw[bg] = D * dpsi
    return y, pdf, dw


def _margin_law(model: PMCModel, i: int, j: int):
    """Frozen law of f_ij, the margin of y_n in the pair (x_n, x_{n+1}) = (i, j)."""
    return _frozen(model.margin(i, j) if model.margin_structure == "pair" else model.margin(i))


def _cond_ppf(model: PMCModel, i: int, j: int, y, t, *, reverse: bool = False) -> np.ndarray:
    """Quantiles (M, L) of the one-step conditional laws of the transition.

    Forward: y_{n+1} given (x_n, x_{n+1}, y_n) = (i, j, y), of density
    f_ji(y') c_ij(F_ij(y), F_ji(y')) — y' = F_ji^{-1}(h_ij^{-1}(t | F_ij(y))).
    ``reverse``: y_n given (x_n, x_{n+1}, y_{n+1}) = (i, j, y) under
    μ(i, y_n) q(j, y_{n+1} | i, y_n), of density f_ij(y_n) c_ij(F_ij(y_n),
    F_ji(y)) — the h-inverse of the transposed copula. Without a copula
    (PMC-IN) the law is the margin itself.
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if reverse:
        unk, cond = _margin_law(model, i, j), _margin_law(model, j, i)
    else:
        unk, cond = _margin_law(model, j, i), _margin_law(model, i, j)
    M, L = y.size, t.size
    with np.errstate(all="ignore"):
        if not model.variant.uses_copula:
            return np.broadcast_to(unk.ppf(t), (M, L)).copy()
        u = np.clip(cond.cdf(y), EPS, ONE_MINUS_EPS)
        cop = model.copula(i, j)
        if reverse:
            cop = cop.transposed()
        w = np.ascontiguousarray(np.broadcast_to(t[None, :], (M, L)).ravel())
        uu = np.ascontiguousarray(np.broadcast_to(u[:, None], (M, L)).ravel())
        v = np.asarray(cop.inv_h_array(w, uu), dtype=float).reshape(M, L)
        return unk.ppf(v)


def _summaries(q: np.ndarray, *, extent: bool = False):
    """Location Q(1/2) and core scale (Q(Φ(1)) − Q(Φ(−1))) / 2 from quantiles at
    _LEVELS; with ``extent`` also the tail extent (the comment of _LEVELS)."""
    loc = q[..., _MID]
    sc = np.maximum(0.5 * (q[..., _MID + 1] - q[..., _MID - 1]),
                    64.0 * EPS * np.maximum(1.0, np.abs(loc)))
    if not extent:
        return loc, sc
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wide = np.nanmax(np.stack([(q[..., _MID + k] - q[..., _MID - k]) / (2.0 * k)
                                   for k in (2, 3)]), axis=0)
    return loc, sc, np.where(np.isfinite(wide), np.maximum(wide / sc, 1.0), 1.0)


def _entry_components(model: PMCModel, y: np.ndarray, *, reverse: bool):
    """Components of the law of a missing y next to the observed value y.

    Forward (y = y_{a−1}, the position after it): one per (i, j), the exact
    conditional law of the transition i → j from y, weight a_i T_ij(y) with
    a_i ∝ μ(i, y) the state law given y alone. ``reverse`` (y = y_{b+1}, the
    position before it): one per (i, j) with i the state at the missing
    position, the law of y_b given (i → j, y), weight ∝ p_ij f_ji(y).
    Returns (lam, loc, sc) (P, K²), the state at the missing position and
    the stay flag (i == j) of every column, and the tail extents (P, K²)
    (:data:`_LEVELS`).
    """
    K = model.K
    y = np.asarray(y, dtype=float)
    P = y.size
    f, _ = _margin_eval(model, y, log=False)
    lam = np.zeros((P, K * K))
    loc = np.zeros((P, K * K))
    sc = np.ones((P, K * K))
    ex = np.ones((P, K * K))
    state = np.empty(K * K, dtype=int)
    stay = np.empty(K * K, dtype=bool)
    with np.errstate(all="ignore"):
        if reverse:
            wgt = model.prior_p[None, :, :] * f.transpose(0, 2, 1)             # p_ij f_ji(y)
        else:
            a = _initial(model, f, log=False)
            a = a / np.where(a.sum(axis=1, keepdims=True) > 0.0, a.sum(axis=1, keepdims=True), 1.0)
            wgt = a[:, :, None] * _x_transition(model, f, log=False)
    for i in range(K):
        for j in range(K):
            c = i * K + j
            state[c] = i if reverse else j
            stay[c] = i == j
            w = wgt[:, i, j]
            if not np.any(w > 0.0):
                continue
            m, s, e = _summaries(_cond_ppf(model, i, j, y, _LEVELS, reverse=reverse), extent=True)
            ok = np.isfinite(w) & (w > 0.0) & np.isfinite(m) & np.isfinite(s)
            lam[:, c] = np.where(ok, w, 0.0)
            loc[:, c] = np.where(ok, m, 0.0)
            sc[:, c] = np.where(ok, s, 1.0)
            ex[:, c] = np.where(ok, e, 1.0)
    return lam, loc, sc, state, stay, ex


def _push(model: PMCModel, lam, loc, sc, state, stay, *, reverse: bool):
    """One step of the Gaussian-sum propagation of the components.

    Each component (state a, location m, scale s) goes through every
    transition a → k (forward) or k ← a (``reverse``): location Q(1/2 | m),
    scale² s_ak(m)² + (dQ(1/2 | y)/dy)² s² (the slope by central differences
    at m ± s). The results are merged by moments into 2K classes (k, stay):
    ``stay`` keeps the components that never left their state since the
    anchor — the narrow ones under strongly dependent diagonal copulas — apart
    from every other path. Returns (lam, loc, sc) (P, 2K), state, stay and
    the one-step scale s_ak(m) of each class (weighted mean): the width of
    the transition kernel into the position, which its grid must resolve too;
    last, the (weighted mean) tail extents of the one-step laws.
    """
    K = model.K
    P, C = lam.shape
    Wt = np.zeros((P, 2, K))
    S1 = np.zeros((P, 2, K))
    S2 = np.zeros((P, 2, K))
    SK = np.zeros((P, 2, K))
    SE = np.zeros((P, 2, K))
    if reverse:
        p = model.prior_p
        Rv = p / np.where(p.sum(axis=0, keepdims=True) > 0.0, p.sum(axis=0, keepdims=True), 1.0)
    for c in range(C):
        a = int(state[c])
        rows = np.nonzero(lam[:, c] > 0.0)[0]
        if rows.size == 0:
            continue
        n = rows.size
        m, s = loc[rows, c], sc[rows, c]
        ys = np.concatenate([m, m - s, m + s])
        if not reverse:
            f, _ = _margin_eval(model, m, log=False)
            Tm = _x_transition(model, f, log=False)
        for k in range(K):
            wt = np.full(n, Rv[k, a]) if reverse else Tm[:, a, k]
            if not np.any(wt > 0.0):
                continue
            i_, j_ = (k, a) if reverse else (a, k)
            q = _cond_ppf(model, i_, j_, ys, _LEVELS, reverse=reverse)
            l1, s1, e1 = _summaries(q[:n], extent=True)
            with np.errstate(all="ignore"):
                slope = (q[2 * n:, _MID] - q[n:2 * n, _MID]) / (2.0 * s)
                s2 = np.sqrt(s1 * s1 + slope * slope * s * s)
            ww = lam[rows, c] * wt
            ok = np.isfinite(l1) & np.isfinite(s2) & np.isfinite(ww) & (ww > 0.0)
            cls = 1 if (stay[c] and a == k) else 0
            r = rows[ok]
            Wt[r, cls, k] += ww[ok]
            S1[r, cls, k] += ww[ok] * l1[ok]
            S2[r, cls, k] += ww[ok] * (s2[ok] ** 2 + l1[ok] ** 2)
            SK[r, cls, k] += ww[ok] * s1[ok]
            SE[r, cls, k] += ww[ok] * e1[ok]
    pos = Wt > 0.0
    Wd = np.where(pos, Wt, 1.0)
    m = np.where(pos, S1 / Wd, 0.0)
    v = np.where(pos, S2 / Wd - m * m, 1.0)
    s = np.sqrt(np.maximum(v, (64.0 * EPS * np.maximum(1.0, np.abs(m))) ** 2))
    ks = np.where(pos, SK / Wd, 1.0)
    es = np.where(pos, SE / Wd, 1.0)
    state2 = np.tile(np.arange(K), 2)
    stay2 = np.repeat([False, True], K)
    return (Wt.reshape(P, 2 * K), m.reshape(P, 2 * K), s.reshape(P, 2 * K), state2, stay2,
            ks.reshape(P, 2 * K), es.reshape(P, 2 * K))


def _bridges(F, B):
    """Products of the forward and backward components of a common state.

    F, B = (lam, loc, sc, state, extent) with per-position state arrays (P, C).
    The product of N(m_F, s_F²) and N(m_B, s_B²) — the law of a y pinned by
    its two neighbours — weighted by λ_F λ_B N(m_F − m_B; 0, s_F² + s_B²).
    Returns (lam, loc, sc, extent) (P, C_F·C_B), lam normalised per position
    (the larger of the two extents).
    """
    lF, mF, sF, stF, eF = F
    lB, mB, sB, stB, eB = B
    vF, vB = (sF * sF)[:, :, None], (sB * sB)[:, None, :]
    v = vF + vB
    same = (stF[:, :, None] == stB[:, None, :]) & (lF[:, :, None] > 0.0) & (lB[:, None, :] > 0.0)
    with np.errstate(all="ignore"):
        d = mF[:, :, None] - mB[:, None, :]
        lw = (np.log(lF)[:, :, None] + np.log(lB)[:, None, :]
              - 0.5 * d * d / v - 0.5 * np.log(v))
        lw = np.where(same, lw, -np.inf)
        mx = lw.max(axis=(1, 2), keepdims=True)
        w = np.where(np.isfinite(mx), np.exp(lw - np.where(np.isfinite(mx), mx, 0.0)), 0.0)
        loc = (mF[:, :, None] * vB + mB[:, None, :] * vF) / v
        sc = np.sqrt(vF * vB / v)
    P = lF.shape[0]
    w = w.reshape(P, -1)
    tot = w.sum(axis=1, keepdims=True)
    w = w / np.where(tot > 0.0, tot, 1.0)
    ex = np.maximum(eF[:, :, None], eB[:, None, :]).reshape(P, -1)
    return (w, np.where(w > 0.0, loc.reshape(P, -1), 0.0), np.where(w > 0.0, sc.reshape(P, -1), 1.0),
            np.where(w > 0.0, ex, 1.0))


def _gap_proposals(model: PMCModel, yL, yR, L, *, fwd0=None):
    """Local proposals of the positions of P runs of missing rows (module docstring).

    ``yL``, ``yR`` (P,) the observed neighbours (NaN: none), ``L`` (P,) the
    run lengths; ``fwd0`` optional forward components (lam, loc, sc, state,
    stay) at the position before each run (a continued gap, see
    :mod:`pmcprg.pmc.outliers`) used instead of ``yL``.

    Returns (lam, loc, sc, fwd, kern): the Gaussian components (ΣL, C) of
    every position in run order (λ summing to 1, or to 0 for a position with
    no observed neighbour), ``fwd`` the forward components of every position
    (to continue a run), ``kern`` the (weight, location, one-step scale)
    of the transition kernels into each position (:func:`_push`) and ``ex``
    the tail extents of the components (:data:`_LEVELS`).
    """
    K = model.K
    yL = np.asarray(yL, dtype=float)
    yR = np.asarray(yR, dtype=float)
    L = np.asarray(L, dtype=int)
    P = L.size
    off = np.concatenate([[0], np.cumsum(L)])
    Mt = int(off[-1])
    CF = K * K
    Fl, Fm, Fs = np.zeros((Mt, CF)), np.zeros((Mt, CF)), np.ones((Mt, CF))
    Fst = np.zeros((Mt, CF), dtype=int)
    Fsy = np.zeros((Mt, CF), dtype=bool)
    Fk = np.ones((Mt, CF))
    Fe = np.ones((Mt, CF))
    Bl, Bm, Bs = np.zeros((Mt, CF)), np.zeros((Mt, CF)), np.ones((Mt, CF))
    Bst = np.zeros((Mt, CF), dtype=int)
    Bk = np.ones((Mt, CF))
    Be = np.ones((Mt, CF))

    def put(dst, rows, vals, width):
        dst[0][rows, :width], dst[1][rows, :width], dst[2][rows, :width] = vals[0], vals[1], vals[2]

    # forward components, depth by depth
    hasF = np.isfinite(yL) if fwd0 is None else np.ones(P, dtype=bool)
    cur, runs = None, np.nonzero(hasF & (L >= 1))[0]
    if runs.size:
        if fwd0 is None:
            lam, loc, sc, st, sy, es = _entry_components(model, yL[runs], reverse=False)
            ks = sc
        else:
            lam, loc, sc, st, sy, ks, es = _push(
                model, *[np.asarray(x)[runs] if np.ndim(x) == 2 else x for x in fwd0], reverse=False)
        cur = (lam, loc, sc, st, sy, ks, es)
    d = 1
    while runs.size:
        lam, loc, sc, st, sy, ks, es = cur
        rows = off[runs] + d - 1
        put((Fl, Fm, Fs), rows, (lam, loc, sc), lam.shape[1])
        Fst[rows, :lam.shape[1]] = st
        Fsy[rows, :lam.shape[1]] = sy
        Fk[rows, :lam.shape[1]] = ks
        Fe[rows, :lam.shape[1]] = es
        keep = L[runs] > d
        runs = runs[keep]
        if not runs.size:
            break
        cur = _push(model, lam[keep], loc[keep], sc[keep], st, sy, reverse=False)
        d += 1
    # backward components, depth by depth from the right neighbour
    runs = np.nonzero(np.isfinite(yR) & (L >= 1))[0]
    if runs.size:
        lam, loc, sc, st, sy, es = _entry_components(model, yR[runs], reverse=True)
        ks = sc
        d = 1
        while True:
            rows = off[runs + 1] - d
            put((Bl, Bm, Bs), rows, (lam, loc, sc), lam.shape[1])
            Bst[rows, :lam.shape[1]] = st
            Bk[rows, :lam.shape[1]] = ks
            Be[rows, :lam.shape[1]] = es
            keep = L[runs] > d
            runs = runs[keep]
            if not runs.size:
                break
            lam, loc, sc, st, sy, ks, es = _push(model, lam[keep], loc[keep], sc[keep], st, sy,
                                                 reverse=True)
            d += 1

    def norm(x):
        tot = x.sum(axis=1, keepdims=True)
        return x / np.where(tot > 0.0, tot, 1.0), tot[:, 0] > 0.0

    Fl, hF = norm(Fl)
    Bl, hB = norm(Bl)
    Pl, Pm, Ps, Pe = _bridges((Fl, Fm, Fs, Fst, Fe), (Bl, Bm, Bs, Bst, Be))
    hP = Pl.sum(axis=1) > 0.0
    share = np.array(_W_GROUPS, dtype=float)[None, :] * np.stack([hP, hF, hB], axis=1)
    tot = share.sum(axis=1, keepdims=True)
    share = share / np.where(tot > 0.0, tot, 1.0)
    lam = np.concatenate([Pl * share[:, :1], Fl * share[:, 1:2], Bl * share[:, 2:]], axis=1)
    loc = np.concatenate([Pm, Fm, Bm], axis=1)
    sc = np.concatenate([Ps, Fs, Bs], axis=1)
    ex = np.concatenate([Pe, Fe, Be], axis=1)
    lam, loc, sc, ex = _merge_components(lam, loc, sc, ex)
    tot = lam.sum(axis=1, keepdims=True)
    lam = np.where(tot > 0.0, lam / np.where(tot > 0.0, tot, 1.0), 0.0)
    # kernel test pieces: the one-step laws into the position (its grid must
    # resolve each of them, not only their mixture)
    kt = np.concatenate([Fl * share[:, 1:2] + 0.0, Bl * share[:, 2:]], axis=1)
    kt = kt / np.where(kt.sum(axis=1, keepdims=True) > 0.0, kt.sum(axis=1, keepdims=True), 1.0)
    kern = (kt, np.concatenate([Fm, Bm], axis=1), np.concatenate([Fk, Bk], axis=1))
    return lam, loc, sc, (Fl, Fm, Fs, Fst, Fsy), kern, ex


def _merge_components(lam, loc, sc, ex):
    """Merge near-identical components, then keep the _MAX_COMPONENTS heaviest.

    Components in decreasing weight; each merges (by moments) into the
    heaviest earlier one whose location is within _MERGE_LOC of the smaller
    scale and whose scale is within a factor _MERGE_SCALE.
    """
    order = np.argsort(-lam, axis=1, kind="stable")
    lam = np.take_along_axis(lam, order, axis=1).copy()
    loc = np.take_along_axis(loc, order, axis=1).copy()
    sc = np.take_along_axis(sc, order, axis=1).copy()
    ex = np.take_along_axis(ex, order, axis=1).copy()
    C = lam.shape[1]
    for c in range(1, C):
        live = lam[:, c] > 0.0
        if not live.any():
            break
        lo, so = loc[:, :c], sc[:, :c]
        smin = np.minimum(so, sc[:, c:c + 1])
        close = ((lam[:, :c] > 0.0) & (np.abs(lo - loc[:, c:c + 1]) <= _MERGE_LOC * smin)
                 & (np.maximum(so, sc[:, c:c + 1]) <= _MERGE_SCALE * smin))
        rows = np.nonzero(live & close.any(axis=1))[0]
        if not rows.size:
            continue
        tgt = np.argmax(close[rows], axis=1)
        w1, w2 = lam[rows, tgt], lam[rows, c]
        m1, m2 = loc[rows, tgt], loc[rows, c]
        v1, v2 = sc[rows, tgt] ** 2, sc[rows, c] ** 2
        w = w1 + w2
        m = (w1 * m1 + w2 * m2) / w
        v = (w1 * (v1 + m1 * m1) + w2 * (v2 + m2 * m2)) / w - m * m
        lam[rows, tgt], loc[rows, tgt] = w, m
        sc[rows, tgt] = np.sqrt(np.maximum(v, np.minimum(v1, v2)))
        ex[rows, tgt] = np.maximum(ex[rows, tgt], ex[rows, c])
        lam[rows, c] = 0.0
    order = np.argsort(-lam, axis=1, kind="stable")[:, :_MAX_COMPONENTS]
    return (np.take_along_axis(lam, order, axis=1), np.take_along_axis(loc, order, axis=1),
            np.take_along_axis(sc, order, axis=1), np.take_along_axis(ex, order, axis=1))


def _split(total: int, score: np.ndarray, floor: int) -> np.ndarray:
    """Integers ≥ 1 summing to ``total``: ``floor`` each (at most half of the
    total between them), the rest ∝ ``score`` by largest remainders."""
    n = score.size
    base = np.full(n, max(1, min(floor, total // (2 * n))))
    extra = (total - base.sum()) * score / score.sum() if score.sum() > 0 else np.zeros(n)
    out = base + np.floor(extra).astype(int)
    rem = total - out.sum()
    if rem > 0:
        out[np.argsort(-(extra - np.floor(extra)), kind="stable")[:rem]] += 1
    return out


def _panel_layout(lam, loc, sc, core, G, cdf_at, support=(-np.inf, np.inf), ex=None):
    """Panels of one local grid (module docstring, "Local grids").

    Core panels: the intervals m_c ± κ e_c s_c of the ``core`` components cut
    at each other's ends, every piece on the component whose density
    λ_c N(y; m_c, s_c²) dominates at its middle; adjacent pieces merge while
    their scales stay within _NEST_RATIO and the panel within _MERGE_SPAN
    scales.
    Background panels (g_ref): the rest of the line. Cores are clipped to
    the ``support`` of g_ref. ``cdf_at(y)`` gives (G_ref(y), 1 − G_ref(y)).
    Returns the :class:`_Panels` fields as lists (sizes summing to G), or
    None when there is no core.
    """
    ex = np.ones_like(sc) if ex is None else ex
    c = np.nonzero(core)[0]
    lo = np.maximum(loc[c] - _CORE_WIDTH * ex[c] * sc[c], support[0])
    hi = np.minimum(loc[c] + _CORE_WIDTH * ex[c] * sc[c], support[1])
    c, lo, hi = c[lo < hi], lo[lo < hi], hi[lo < hi]
    if not c.size:
        return None
    pts = np.unique(np.concatenate([lo, hi]))
    pieces = []                                 # (a, b, driver or −1)
    for a, b in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (a + b)
        cov = c[(lo <= mid) & (mid <= hi)]
        # the component that dominates the mixture density there drives the piece
        z = (mid - loc[cov]) / sc[cov]
        drv = int(cov[np.argmax(np.log(lam[cov]) - 0.5 * z * z - np.log(sc[cov]))]) if cov.size else -1
        if pieces and pieces[-1][2] == drv:
            pieces[-1] = (pieces[-1][0], b, drv)
        else:
            pieces.append((a, b, drv))
    pieces = [(-np.inf, pts[0], -1)] + pieces + [(pts[-1], np.inf, -1)]
    merged = []                                 # (a, b, driver, smallest, largest scale)
    for a, b, d in pieces:
        sd = sc[d] * ex[d] if d >= 0 else np.nan
        if merged and merged[-1][2] == -1 and d == -1:
            merged[-1] = (merged[-1][0], b, -1, np.nan, np.nan)
        elif (merged and merged[-1][2] >= 0 and d >= 0
              and max(merged[-1][4], sd) <= _NEST_RATIO * min(merged[-1][3], sd)
              and b - merged[-1][0] <= _MERGE_SPAN * min(merged[-1][3], sd)):
            a0, _, d0, lo0, hi0 = merged[-1]
            merged[-1] = (a0, b, d0 if sc[d0] * ex[d0] <= sd else d, min(lo0, sd), max(hi0, sd))
        else:
            merged.append((a, b, d, sd, sd))
    merged = [(a, b, d) for a, b, d, _, _ in merged]
    rows = []
    for a, b, d in merged:
        if d >= 0:
            m, c_ = loc[d], _SINH_SCALE * sc[d]
            ua, ub = np.arcsinh((a - m) / c_), np.arcsinh((b - m) / c_)
            # the share of the component's mass on the panel, for the allocation
            w_ = _sp_ndtr((b - m) / sc[d]) - _sp_ndtr((a - m) / sc[d])
            rows.append((m, c_, ua, ub, w_, 0.0, lam[d]))
        else:
            ca, sa = cdf_at(a)
            cb, sb = cdf_at(b)
            rows.append((np.nan, np.nan, ca, cb, sa, sb, 0.0))
    if len(rows) > G // 2:
        return None
    rows = np.array(rows, dtype=float)
    bg = ~np.isfinite(rows[:, 0])
    width = np.where(bg, np.where(rows[:, 2] <= 0.5, rows[:, 3] - rows[:, 2], rows[:, 4] - rows[:, 5]),
                     rows[:, 4])
    keep = ~bg | (width > 1e-15)
    rows, width, bg = rows[keep], width[keep], bg[keep]
    sizes = np.zeros(len(rows), dtype=int)
    n_bg = int(round(_BACKGROUND_NODES * G)) if bg.any() else 0
    if bg.any():
        sizes[bg] = _split(n_bg, np.sqrt(width[bg]), _MIN_BACKGROUND_NODES)
    if (~bg).any():
        sizes[~bg] = _split(G - n_bg, np.sqrt(rows[~bg, 6] * np.maximum(width[~bg], 0.0)),
                            _MIN_CORE_NODES)
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4], rows[:, 5], sizes


def _single_core_layouts(lam, loc, sc, core, G, cdf, sf, sup, ex) -> dict:
    """:func:`_panel_layout` of the positions whose core intervals form one
    cluster of scales within _NEST_RATIO — one core panel between two
    background panels — computed for all of them at once (the common case).

    ``cdf``, ``sf``: g_ref at the clipped core ends, in the order of
    :func:`_panel_grids` (all lower ends, then all upper ends). Returns
    {position: layout}; the others go through :func:`_panel_layout`.
    """
    P, C = lam.shape
    lo = np.where(core, np.maximum(loc - _CORE_WIDTH * ex * sc, sup[0]), np.inf)
    hi = np.where(core, np.minimum(loc + _CORE_WIDTH * ex * sc, sup[1]), -np.inf)
    nc = core.sum()
    cl = np.full((P, C), np.nan)
    ch = np.full((P, C), np.nan)
    cs = np.full((P, C), np.nan)
    cl[core], ch[core] = cdf[:nc], cdf[nc:]
    sl = np.full((P, C), np.nan)
    sh = np.full((P, C), np.nan)
    sl[core], sh[core] = sf[:nc], sf[nc:]
    del cs
    ok = core.any(axis=1) & np.all(~core | (lo < hi), axis=1)
    # one cluster: sorted by lower end, every lower end below the running upper end
    order = np.argsort(np.where(core, lo, np.inf), axis=1, kind="stable")
    los = np.take_along_axis(lo, order, axis=1)
    his = np.take_along_axis(hi, order, axis=1)
    run = np.maximum.accumulate(np.where(np.isfinite(his), his, -np.inf), axis=1)
    gapped = (los[:, 1:] > run[:, :-1]) & np.isfinite(los[:, 1:])
    ok &= ~gapped.any(axis=1)
    smin = np.where(core, ex * sc, np.inf).min(axis=1)
    smax = np.where(core, ex * sc, -np.inf).max(axis=1)
    ok &= smax <= _NEST_RATIO * smin
    ok &= np.where(core, hi, -np.inf).max(axis=1) - np.where(core, lo, np.inf).min(axis=1) \
        <= _MERGE_SPAN * smin
    a = lo.min(axis=1)
    b = hi.max(axis=1)
    ia, ib = np.argmin(lo, axis=1), np.argmax(hi, axis=1)
    r = np.arange(P)
    ca, sa = cl[r, ia], sl[r, ia]
    cb, sb = ch[r, ib], sh[r, ib]
    wl = ca                                                        # g_ref mass below a
    wu = np.where(cb <= 0.5, 1.0 - cb, sb)                         # above b
    ok &= (wl > 1e-15) & (wu > 1e-15)
    out = {}
    idx = np.nonzero(ok)[0]
    if not idx.size:
        return out
    d = np.argmin(np.where(core, sc, np.inf), axis=1)
    n_bg = int(round(_BACKGROUND_NODES * G))
    for p in idx:
        m, c_ = loc[p, d[p]], _SINH_SCALE * sc[p, d[p]]
        ua, ub = np.arcsinh((a[p] - m) / c_), np.arcsinh((b[p] - m) / c_)
        w_ = _sp_ndtr((b[p] - m) / sc[p, d[p]]) - _sp_ndtr((a[p] - m) / sc[p, d[p]])
        bgs = _split(n_bg, np.sqrt(np.array([wl[p], wu[p]])), _MIN_BACKGROUND_NODES)
        out[int(p)] = (np.array([np.nan, m, np.nan]), np.array([np.nan, c_, np.nan]),
                       np.array([0.0, ua, cb[p]]), np.array([ca[p], ub, 1.0]),
                       np.array([1.0, w_, sb[p]]), np.array([sa[p], 0.0, 0.0]),
                       np.array([bgs[0], G - n_bg, bgs[1]]))
    return out


def _panel_grids(ref: QuadratureGrid, lam, loc, sc, G: int, ex=None) -> list:
    """The local grids (G nodes each) of P proposals (module docstring, "Local grids")."""
    P = lam.shape[0]
    ex = np.ones_like(sc) if ex is None else ex
    cnt = (np.searchsorted(ref.nodes, loc + sc, side="right")
           - np.searchsorted(ref.nodes, loc - sc, side="left"))
    core = (lam >= _RESOLVE_MIN_WEIGHT) & (cnt < _RESOLVE_NODES / _BACKGROUND_NODES)
    sup = _support(ref)
    # g_ref CDF at every core end, in one call
    hw = _CORE_WIDTH * ex * sc
    ends = np.clip(np.concatenate([(loc - hw)[core], (loc + hw)[core]]), *sup)
    cdf, sf = _ref_cdf_sf(ref.weights, ref.dists, ends)
    table = dict(zip(ends.tolist(), zip(cdf.tolist(), sf.tolist())))
    table[-np.inf], table[np.inf] = (0.0, 1.0), (1.0, 0.0)

    def cdf_at(y):
        return table[float(y)]

    fast = _single_core_layouts(lam, loc, sc, core, G, cdf, sf, sup, ex)
    layouts = [fast[p] if p in fast
               else _panel_layout(lam[p], loc[p], sc[p], core[p], G, cdf_at, sup, ex[p])
               for p in range(P)]
    good = [L is not None for L in layouts]
    if not all(good):
        sub = [p for p in range(P) if good[p]]
        it = iter(_panel_grids(ref, lam[sub], loc[sub], sc[sub], G, ex[sub]) if sub else [])
        return [next(it) if good[p] else None for p in range(P)]
    npan = np.array([len(L[6]) for L in layouts])
    cat = [np.concatenate([L[f] for L in layouts]) for f in range(7)]
    pan_all = _Panels(loc=cat[0], sc=cat[1], t_lo=cat[2], t_hi=cat[3], tc_lo=cat[4], tc_hi=cat[5],
                      sizes=cat[6].astype(int), panel=np.empty(0, dtype=int))
    sizes = pan_all.sizes
    k = np.repeat(np.arange(sizes.size), sizes)                        # panel of every node
    j = np.arange(k.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)  # index within the panel
    s_ = np.empty(k.size)
    ws = np.empty(k.size)
    for m in np.unique(sizes):
        sel = sizes[k] == m
        sm, wm = _gl(int(m))
        s_[sel], ws[sel] = sm[j[sel]], wm[j[sel]]
    y, pdf, dw = _panel_points(pan_all, k, s_, ref.power, ref.weights, ref.dists)
    with np.errstate(all="ignore"):
        omega = ws * dw / pdf
    out = []
    p0 = np.concatenate([[0], np.cumsum(npan)])
    for p in range(P):
        kp = slice(p0[p], p0[p + 1])
        sel = slice(int(sizes[:p0[p]].sum()), int(sizes[:p0[p + 1]].sum()))
        yy, oo = y[sel], omega[sel]
        ok = (np.all(np.isfinite(yy)) and np.all(np.isfinite(oo)) and np.all(oo >= 0.0)
              and np.all(np.diff(yy) >= 0.0) and yy.size == G)
        if not ok:
            out.append(None)
            continue
        pan = _Panels(loc=pan_all.loc[kp], sc=pan_all.sc[kp], t_lo=pan_all.t_lo[kp],
                      t_hi=pan_all.t_hi[kp], tc_lo=pan_all.tc_lo[kp], tc_hi=pan_all.tc_hi[kp],
                      sizes=sizes[kp], panel=k[sel] - p0[p])
        out.append(QuadratureGrid(s=s_[sel], ws=ws[sel], power=ref.power, t=s_[sel],
                                  w=ws[sel] * dw[sel], nodes=yy, ref_pdf=pdf[sel], omega=oo,
                                  weights=ref.weights, dists=ref.dists, mix=pan))
    return out


def _support(ref: QuadratureGrid) -> tuple[float, float]:
    """Support (lo, hi) of g_ref."""
    lo, hi = np.inf, -np.inf
    for d in ref.dists:
        a, b = d.support()
        lo, hi = min(lo, float(a)), max(hi, float(b))
    return lo, hi


def _proposal_error(nodes: np.ndarray, omega: np.ndarray, lam, loc, sc,
                    support=(-np.inf, np.inf)) -> np.ndarray:
    """Quadrature error (P,) of a grid on its proposal's components.

    Σ_c λ_c (|∫ N_c − I_0| + |∫ N_c (y − m_c)² / s_c² − I_2|) / I_0, the
    integrals by the rule (``nodes``, ``omega``) of each position and I_0,
    I_2 their exact values on the ``support`` of g_ref: how well the grid
    integrates the Gaussian pieces that approximate the law of the missing y
    (module docstring, "Local grids"). Components with less than 1e-3 of
    their mass on the support are left out.
    """
    with np.errstate(all="ignore"):
        za, zb = (support[0] - loc) / sc, (support[1] - loc) / sc
        I0 = _sp_ndtr(zb) - _sp_ndtr(za)
        pa = np.where(np.isfinite(za), za * np.exp(-0.5 * za * za), 0.0) / _SQRT_2PI
        pb = np.where(np.isfinite(zb), zb * np.exp(-0.5 * zb * zb), 0.0) / _SQRT_2PI
        I2 = I0 - pb + pa
        z = (nodes[:, None, :] - loc[:, :, None]) / sc[:, :, None]            # (P, C, G)
        f = np.exp(-0.5 * z * z) / (sc[:, :, None] * _SQRT_2PI) * omega[:, None, :]
        e = (np.abs(f.sum(axis=2) - I0) + np.abs((f * z * z).sum(axis=2) - I2)) / I0
    e = np.where(np.isfinite(e), e, 2.0)
    return (np.where((lam > 0.0) & (I0 > 1e-3), lam, 0.0) * e).sum(axis=1)


def _neighbour_error(model: PMCModel, nodes: np.ndarray, omega: np.ndarray,
                     yl: np.ndarray, yr: np.ndarray, *, split: bool = False, first=None):
    """Relative error (P,) of grids on the integrals of their gap they must reproduce.

    Entry, where the row before the missing y is observed (``yl`` finite):
    Σ_g ω_g q(j, y_g | i, y_l) against its exact value T_ij(y_l), weighted by
    a_i T_ij (a_i ∝ μ(i, y_l)) — the raw block masses of the Tauchen–Hussey
    renormalisation. Exit, where the row after is observed (``yr`` finite):
    Σ_g ω_g μ(i, y_g) q(k, y_r | i, y_g) against ∫ μ(i, y) q(k, y_r | i, y) dy
    = p_ik f_ki(y_r) (pair margins; π_i T_ik f_k(y_r) for state margins — the
    copula density integrates to 1 in its first argument), weighted by these
    exact masses: the exit kernel is as narrow as the transition. Both use
    the exact kernels, so they see what the Gaussian proposal does not
    (support boundaries, skewed conditional laws). A non-finite value counts
    as an error 1. ``first`` (P,) marks a missing y_1: Σ_g ω_g μ(i, y_g)
    against P(x_1 = i), counted with the entry part. Returns the sum (P,),
    or with ``split`` the entry and exit parts.
    """
    K = model.K
    P, G = nodes.shape
    parts = {"entry": np.zeros(P), "exit": np.zeros(P)}
    fN, FN = _margin_eval(model, nodes.ravel(), log=False)

    def take(F, idx):
        return None if F is None else F[idx]

    with np.errstate(all="ignore"):
        if first is not None and np.any(first):
            rows = np.nonzero(first)[0]
            iN = (rows[:, None] * G + np.arange(G)[None, :]).ravel()
            mu = _initial(model, fN[iN], log=False).reshape(rows.size, G, K)
            got = (mu * omega[rows][:, :, None]).sum(axis=1)                   # (n, K)
            exact = _initial(model, np.ones((1, K, K)), log=False)[0]
            ok = exact > 0.0
            rel = np.where(ok, np.abs(got / np.where(ok, exact, 1.0) - 1.0), 0.0)
            rel = np.where(np.isfinite(rel), rel, 1.0)
            parts["entry"][rows] += (np.where(ok, exact, 0.0) * rel).sum(axis=1) / exact[ok].sum()
        for side, y in (("entry", yl), ("exit", yr)):
            rows = np.nonzero(np.isfinite(y))[0]
            if not rows.size:
                continue
            fo, Fo = _margin_eval(model, y[rows], log=False)
            rp = np.repeat(np.arange(rows.size), G)
            iN = (rows[:, None] * G + np.arange(G)[None, :]).ravel()
            om = omega[rows][:, :, None, None]
            if side == "entry":
                ker, _ = _kernel(model, fo[rp], take(Fo, rp), fN[iN], take(FN, iN))
                got = (ker.reshape(rows.size, G, K, K) * om).sum(axis=1)       # (n, K, K)
                exact = _x_transition(model, fo, log=False)
                a = _initial(model, fo, log=False)
                wgt = a[:, :, None] * exact
            else:
                ker, _ = _kernel(model, fN[iN], take(FN, iN), fo[rp], take(Fo, rp))
                mu = _initial(model, fN[iN], log=False)                          # (nG, K)
                got = ((ker * mu[:, :, None]).reshape(rows.size, G, K, K) * om).sum(axis=1)
                fki = fo.transpose(0, 2, 1)                                     # f_ki(y_r) at [i, k]
                if model.margin_structure == "pair":
                    exact = model.prior_p[None, :, :] * fki
                else:
                    pi = _initial(model, np.ones((1, K, K)), log=False)[0]
                    exact = pi[None, :, None] * _x_transition(model, fo, log=False) * fki
                wgt = exact
            ok = np.isfinite(exact) & (exact > 0.0)
            rel = np.where(ok, np.abs(got / np.where(ok, exact, 1.0) - 1.0), 0.0)
            rel = np.where(np.isfinite(rel), rel, 1.0)
            wgt = np.where(ok & np.isfinite(wgt), wgt, 0.0)
            tot = wgt.sum(axis=(1, 2))
            parts[side][rows] += np.where(
                tot > 0.0, (wgt * rel).sum(axis=(1, 2)) / np.where(tot > 0.0, tot, 1.0), 0.0)
    if split:
        return parts["entry"], parts["exit"]
    return parts["entry"] + parts["exit"]


def _grid_error(model, ref, nodes, omega, lam, loc, sc, yl, yr, kern=None, exact=True,
                first=None):
    """(proxy, (entry, exit)) errors (P,) of grids: :func:`_proposal_error` on
    the Gaussian proxies of the law of y and on the transition kernels into
    the position (``kern``: weights, locations, one-step scales),
    :func:`_neighbour_error` on the exact entry / exit integrals (0 where the
    gap has no observed neighbour, or without ``exact``)."""
    first = np.zeros(nodes.shape[0], bool) if first is None else np.asarray(first, dtype=bool)
    near = np.isfinite(yl) | np.isfinite(yr) | first
    ex = (np.zeros(nodes.shape[0]), np.zeros(nodes.shape[0]))
    if exact and near.any():
        e_in, e_out = _neighbour_error(model, nodes[near], omega[near], yl[near], yr[near],
                                       split=True, first=first[near])
        ex[0][near], ex[1][near] = e_in, e_out
    sup = _support(ref)
    px = _proposal_error(nodes, omega, lam, loc, sc, sup)
    if kern is not None:
        px = px + _proposal_error(nodes, omega, *kern, sup)
    return px, ex


def _local_grid_list(model: PMCModel, ref: QuadratureGrid, lam, loc, sc, yl=None, yr=None,
                     kern=None, first=None, ex=None) -> list:
    """Grids of the proposals: the reference grid unless a local grid does better.

    Candidates are the positions with a component of weight ≥
    _RESOLVE_MIN_WEIGHT that has fewer than _CANDIDATE_NODES reference nodes
    within one scale of its location. A candidate gets its local grid
    (:func:`_panel_grids`) when the reference grid's error on the Gaussian
    proxies of the law of y (:func:`_proposal_error`, sensitive to the
    resolution of the posterior) exceeds _SWITCH_TOL, the local grid's is
    _SWITCH_GAIN times smaller, and the local grid is not worse on the exact
    entry / exit integrals next to an observed row ``yl`` / ``yr``
    (:func:`_neighbour_error`, which sees support boundaries and skewed laws
    the proxies miss). The reference grid is kept wherever it resolves the
    laws of the gap — weak and moderate dependence — with results bit for bit
    unchanged.
    """
    Mt = lam.shape[0]
    yl = np.full(Mt, np.nan) if yl is None else np.asarray(yl, dtype=float)
    yr = np.full(Mt, np.nan) if yr is None else np.asarray(yr, dtype=float)
    first = np.zeros(Mt, bool) if first is None else np.asarray(first, dtype=bool)
    out = [ref] * Mt
    heavy = lam >= _RESOLVE_MIN_WEIGHT
    cnt = (np.searchsorted(ref.nodes, loc + sc, side="right")
           - np.searchsorted(ref.nodes, loc - sc, side="left"))
    cand = np.any(heavy & (cnt < _CANDIDATE_NODES), axis=1)
    if kern is not None:
        kl, km, kss = kern
        kc = (np.searchsorted(ref.nodes, km + kss, side="right")
              - np.searchsorted(ref.nodes, km - kss, side="left"))
        cand |= np.any((kl >= _RESOLVE_MIN_WEIGHT) & (kc < _CANDIDATE_NODES), axis=1)
    idx = np.nonzero(cand)[0]
    if not idx.size:
        return out

    def ksel(rows):
        return None if kern is None else tuple(x[rows] for x in kern)

    p_ref, _ = _grid_error(model, ref, np.broadcast_to(ref.nodes, (idx.size, ref.G)),
                           np.broadcast_to(ref.omega, (idx.size, ref.G)),
                           lam[idx], loc[idx], sc[idx], yl[idx], yr[idx], ksel(idx), exact=False)
    keep = p_ref > _SWITCH_TOL
    idx, p_ref = idx[keep], p_ref[keep]
    if not idx.size:
        return out
    loc_g = _panel_grids(ref, lam[idx], loc[idx], sc[idx], ref.G,
                         None if ex is None else ex[idx])
    ok = np.array([g is not None for g in loc_g], dtype=bool)
    if not ok.any():
        return out
    sel = idx[ok]
    gl = [g for g in loc_g if g is not None]
    gn, go = np.array([g.nodes for g in gl]), np.array([g.omega for g in gl])
    p_loc, (xi_loc, xo_loc) = _grid_error(model, ref, gn, go, lam[sel], loc[sel], sc[sel],
                                          yl[sel], yr[sel], ksel(sel))
    x_loc = xi_loc + xo_loc
    better = _SWITCH_GAIN * p_loc < p_ref[ok]
    # a missing y_1: the grid must carry the prior (the reference grid does, by
    # construction), to within _PRIOR_GUARD — a gross-failure guard
    fs = first[sel]
    if fs.any():
        nf = int(fs.sum())
        e_pr, _ = _neighbour_error(model, gn[fs], go[fs], np.full(nf, np.nan), np.full(nf, np.nan),
                                   split=True, first=np.ones(nf, bool))
        better[np.nonzero(fs)[0][e_pr > _PRIOR_GUARD]] = False
    # the reference grid's exact check only where the local grid's is not clean
    dirty = np.nonzero(better & (x_loc > _SWITCH_TOL))[0]
    x_ref = np.zeros(sel.size)
    if dirty.size:
        r = sel[dirty]
        e_in, e_out = _neighbour_error(model, np.broadcast_to(ref.nodes, (r.size, ref.G)),
                                       np.broadcast_to(ref.omega, (r.size, ref.G)), yl[r], yr[r],
                                       split=True)
        x_ref[dirty] = e_in + e_out
    for q, (k, g) in enumerate(zip(sel, gl)):
        if better[q] and x_loc[q] <= max(x_ref[q], _SWITCH_TOL):
            out[k] = replace(g, exit_error=float(xo_loc[q])) if np.isfinite(yr[k]) else g
    return out


def _fine_grid(ref: QuadratureGrid, ref_fine: QuadratureGrid, grid: QuadratureGrid,
               factor: int) -> QuadratureGrid:
    """The same panels with ``factor`` times more nodes each (Nyström grid)."""
    if grid.mix is None:
        return ref_fine
    pan = grid.mix
    sizes = factor * pan.sizes
    k = np.repeat(np.arange(sizes.size), sizes)
    j = np.arange(k.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    s_ = np.empty(k.size)
    ws = np.empty(k.size)
    for m in np.unique(sizes):
        sel = sizes[k] == m
        sm, wm = _gl(int(m))
        s_[sel], ws[sel] = sm[j[sel]], wm[j[sel]]
    y, pdf, dw = _panel_points(pan, k, s_, ref.power, ref.weights, ref.dists)
    with np.errstate(all="ignore"):
        omega = ws * dw / pdf
    fpan = _Panels(loc=pan.loc, sc=pan.sc, t_lo=pan.t_lo, t_hi=pan.t_hi, tc_lo=pan.tc_lo,
                   tc_hi=pan.tc_hi, sizes=sizes, panel=k)
    return QuadratureGrid(s=s_, ws=ws, power=ref.power, t=s_, w=ws * dw,
                          nodes=y, ref_pdf=pdf, omega=omega, weights=ref.weights,
                          dists=ref.dists, mix=fpan)


def _runs(miss: np.ndarray):
    """(starts, ends) of the maximal runs of True in ``miss`` (inclusive ends)."""
    m = np.concatenate([[False], np.asarray(miss, dtype=bool), [False]])
    dm = np.diff(m.astype(int))
    return np.nonzero(dm == 1)[0], np.nonzero(dm == -1)[0] - 1


def _run_grids(model: PMCModel, ref: QuadratureGrid, yL: float, yR: float, L: int,
               fwd0=None, start: int = -1) -> tuple[list, list]:
    """Grids of one run of L missing rows, and the forward components of each.

    The step-by-step filter of :mod:`pmcprg.pmc.outliers` builds the grids of
    a run as it enters it: ``yL`` / ``yR`` its observed neighbours (NaN:
    none), ``fwd0`` the forward components of the position before it when
    the run continues a gap (a gated row), ``start`` the row of its first
    position. On the same run it gives the grids of :func:`_gap_grids`.
    """
    if not (_LOCAL_GRIDS and model.variant.uses_copula):
        return [ref] * L, [None] * L
    f0 = None if fwd0 is None else (fwd0[0][None], fwd0[1][None], fwd0[2][None], fwd0[3], fwd0[4])
    lam, loc, sc, (Fl, Fm, Fs, Fst, Fsy), kern, ex = _gap_proposals(
        model, np.array([yL]), np.array([yR]), np.array([L]), fwd0=f0)
    yl, yr = np.full(L, np.nan), np.full(L, np.nan)
    yl[0] = yL if fwd0 is None else np.nan
    yr[-1] = yR
    first = np.zeros(L, bool)
    first[0] = start == 0
    grids = _local_grid_list(model, ref, lam, loc, sc, yl, yr, kern, first, ex)
    return grids, [(Fl[k], Fm[k], Fs[k], Fst[k], Fsy[k]) for k in range(L)]


def _gap_grids(model: PMCModel, Y: np.ndarray, miss: np.ndarray, ref: QuadratureGrid) -> dict:
    """{n: grid} for every missing n (the reference grid where it suffices)."""
    idx = np.nonzero(miss)[0]
    if not (_LOCAL_GRIDS and model.variant.uses_copula) or not idx.size:
        return {int(n): ref for n in idx}
    Y = np.asarray(Y, dtype=float)
    N = len(Y)
    a, b = _runs(miss)
    yL = np.where(a > 0, Y[np.maximum(a - 1, 0)], np.nan)
    yR = np.where(b < N - 1, Y[np.minimum(b + 1, N - 1)], np.nan)
    lam, loc, sc, _, kern, ex = _gap_proposals(model, yL, yR, b - a + 1)
    pos = np.concatenate([np.arange(s, e + 1) for s, e in zip(a, b)])
    yl = np.where(np.isin(pos, a) & (pos > 0), Y[np.maximum(pos - 1, 0)], np.nan)
    yr = np.where(np.isin(pos, b) & (pos < N - 1), Y[np.minimum(pos + 1, N - 1)], np.nan)
    grids = _local_grid_list(model, ref, lam, loc, sc, yl, yr, kern, pos == 0, ex)
    return {int(n): g for n, g in zip(pos, grids)}


# ---------------------------------------------------------------------------
# Transition kernel q(j, y' | i, y) on arbitrary pairs — linear and log
# ---------------------------------------------------------------------------

def _margin_eval(model: PMCModel, y: np.ndarray, *, log: bool):
    """f[:, i, j] = f_ij(y) (log f if ``log``), F[:, i, j] = F_ij(y) clipped.

    Shapes (M, K, K); for state margins both are broadcast from the K
    per-state evaluations, as in ``precompute_weights``.
    """
    K = model.K
    M = len(y)
    uses_cop = model.variant.uses_copula
    f = np.empty((M, K, K))
    F = np.empty((M, K, K)) if uses_cop else None
    if model.margin_structure == "pair":
        for ii in range(K):
            for jj in range(K):
                mg = model.margin(ii, jj)
                f[:, ii, jj] = (_inf._margin_logpdf_vec(mg, y) if log else mg.pdf_vec(y))
                if uses_cop:
                    F[:, ii, jj] = mg.cdf_vec(y)
    else:
        for kk in range(K):
            mg = model.margin(kk)
            f[:, kk, :] = (_inf._margin_logpdf_vec(mg, y) if log else mg.pdf_vec(y))[:, None]
            if uses_cop:
                F[:, kk, :] = np.asarray(mg.cdf_vec(y))[:, None]
    if F is not None:
        np.clip(F, EPS, ONE_MINUS_EPS, out=F)
    return f, F


def _kernel(model: PMCModel, fL, FL, fR, FR) -> tuple[np.ndarray, bool]:
    """q(j, y_R | i, y_L) on (M,) pairs — the formulas of ``precompute_weights``.

    Returns ``(Wk, overflow)`` with Wk (M, K, K) cleaned like
    ``precompute_weights`` (non-finite → 0, negatives → 0) and ``overflow``
    True when a raw value was +inf or NaN (the caller then switches to log
    space).
    """
    var = model.variant
    K = model.K
    f_next_T = fR.transpose(0, 2, 1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if var.has_markov_prior:
            Wk = model.transition_A[None, :, :] * f_next_T
        else:
            p = model.prior_p
            joint = p[None, :, :] * fL * f_next_T
            D = np.einsum("ij,nij->ni", p, fL)
            Wk = joint / np.maximum(D, MIN_POSITIVE)[:, :, None]
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((FL[:, ii, jj], FR[:, jj, ii]))
                    Wk[:, ii, jj] *= model.copula(ii, jj).pdf_array(uv)
    bad = ~np.isfinite(Wk)
    overflow = bool(bad.any())
    if overflow:
        Wk[bad] = 0.0
    np.clip(Wk, 0.0, None, out=Wk)
    return Wk, overflow


def _log_kernel(model: PMCModel, lfL, FL, lfR, FR) -> np.ndarray:
    """log q(j, y_R | i, y_L) on (M,) pairs — the formulas of ``_log_transition_weights``."""
    var = model.variant
    K = model.K
    lf_next_T = lfR.transpose(0, 2, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = model.prior_p
        log_p = np.log(p)
        if var.has_markov_prior:
            L = np.log(model.transition_A)[None, :, :] + lf_next_T
        elif model.margin_structure == "pair":
            left = log_p[None, :, :] + lfL
            log_d = _inf._lse(left, axis=2)
            L = left - log_d[:, :, None] + lf_next_T
        else:
            log_kernel = log_p - np.log(p.sum(axis=1, keepdims=True))
            L = log_kernel[None, :, :] + lf_next_T
        if var.uses_copula:
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((FL[:, ii, jj], FR[:, jj, ii]))
                    L[:, ii, jj] += model.copula(ii, jj).logpdf_array(uv)
    L[np.isnan(L) | (L == np.inf)] = -np.inf
    return L


def _initial(model: PMCModel, f: np.ndarray, *, log: bool) -> np.ndarray:
    """μ(i, y) = α_1(i) at the rows of f (M, K, K) — the ``forward`` initialisation."""
    p = model.prior_p
    if not log:
        if model.margin_structure == "pair":
            return np.einsum("ij,nij->ni", p, f)
        return np.einsum("ij,nji->nj", p, f)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.log(p)
        if model.margin_structure == "pair":
            out = _inf._lse(log_p[None, :, :] + f, axis=2)
        else:
            out = _inf._lse(log_p, axis=0)[None, :] + f[:, :, 0]
    out[np.isnan(out) | (out == np.inf)] = -np.inf
    return out


#: Renormalisation of the quadrature rows of E, Q and α̃_1 (module docstring):
#: ``"block"`` (production) rescales every block (i, g) → (j, ·) to the exact
#: transition probability P(x_{n+1} = j | x_n = i, y_n), then every row to 1;
#: ``"row"`` only rescales rows; ``None`` keeps the raw quadrature. Private
#: switch kept for accuracy studies.
_RENORMALISE = "block"


def _x_transition(model: PMCModel, f: np.ndarray, *, log: bool) -> np.ndarray:
    """P(x_{n+1} = j | x_n = i, y_n) at the rows of f (M, K, K) — log if ``log``.

    A_ij for the HMC variants and for state margins (p_ij / Σ_k p_ik); for
    pair margins p_ij f_ij(y) / Σ_k p_ik f_ik(y) (DerrodePieczynski_CSDA2013
    Eq. 13), with the linear-space guard of ``precompute_weights``.
    """
    M, K = f.shape[0], model.K
    p = model.prior_p
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        if model.variant.has_markov_prior or model.margin_structure != "pair":
            A = model.transition_A if model.variant.has_markov_prior else p / p.sum(axis=1, keepdims=True)
            T = np.broadcast_to(np.log(A) if log else A, (M, K, K))
        elif log:
            left = np.log(p)[None, :, :] + f
            T = left - _inf._lse(left, axis=2)[:, :, None]
        else:
            joint = p[None, :, :] * f
            T = joint / np.maximum(joint.sum(axis=2), MIN_POSITIVE)[:, :, None]
    return T


def _normalise_blocks(B: np.ndarray, target: np.ndarray | None, *, log: bool,
                      with_mass: bool = False):
    """Rescale the quadrature tensor B (..., K, G) to its exact block masses.

    ``B[..., j, g]`` holds the weights of the destination (j, y_g) and
    ``target[..., j]`` the exact mass Σ_g of each block (a transition
    probability). Blocks are rescaled to ``target`` (``"block"`` mode), then the
    flattened rows (..., K·G) to sum to 1 (``"block"`` and ``"row"``); a block
    whose quadrature mass vanished keeps 0 and its mass goes to the other
    blocks through the row step.

    Returns ``(rows, factors)``: the (..., K·G) rows and the (..., K) factors
    by which each block was multiplied, in linear scale. ``with_mass`` adds
    the raw block masses Σ_g B (log Σ_g exp B in log space) — the quadrature
    of a known integral, whose distance to ``target`` is the convergence
    diagnostic of :func:`_quadrature_report`.
    """
    shape = B.shape[:-2] + (B.shape[-2] * B.shape[-1],)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        raw = (_inf._lse(B, axis=-1) if log else B.sum(axis=-1)) if with_mass else None
    if _RENORMALISE is None:
        out = (B.reshape(shape), np.ones(B.shape[:-1]))
        return out + (raw,) if with_mass else out
    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        if log:
            fac = np.zeros(B.shape[:-1])
            if _RENORMALISE == "block" and target is not None:
                sb = _inf._lse(B, axis=-1)
                ok = np.isfinite(sb) & np.isfinite(target)
                fac = np.where(ok, np.where(ok, target, 0.0) - np.where(ok, sb, 0.0), -np.inf)
                B = np.where(ok[..., None], B + np.where(ok, fac, 0.0)[..., None], -np.inf)
            R = B.reshape(shape)
            sr = _inf._lse(R, axis=-1)
            fin = np.isfinite(sr)
            R = np.where(fin[..., None], R - np.where(fin, sr, 0.0)[..., None], R)
            fac = np.exp(fac - np.where(fin, sr, 0.0)[..., None])
            out = (R, np.where(np.isfinite(fac), fac, 0.0))
            return out + (raw,) if with_mass else out
        fac = np.ones(B.shape[:-1])
        if _RENORMALISE == "block" and target is not None:
            sb = B.sum(axis=-1)
            ok = np.isfinite(sb) & (sb > 0.0)
            fac = np.where(ok, target / np.where(ok, sb, 1.0), 0.0)
            B = B * fac[..., None]
        R = B.reshape(shape)
        sr = R.sum(axis=-1, keepdims=True)
        ok = np.isfinite(sr) & (sr > 0.0)
        R = np.divide(R, np.where(ok, sr, 1.0), out=np.zeros_like(R), where=ok)
        fac = fac / np.where(ok, sr, 1.0)
        out = (R, np.broadcast_to(fac, B.shape[:-1]).copy())
        return out + (raw,) if with_mass else out


# ---------------------------------------------------------------------------
# The augmented chain
# ---------------------------------------------------------------------------

class _Chain:
    """Transitions of the chain on the augmented state (module docstring).

    ``trans[n]`` maps position n to n+1 and has shape S_n × S_{n+1}, with
    S_n = K at an observed n and K·G at a missing one (index i·G + g); in log
    space when ``log``. ``init`` has shape (S_0,).

    ``ev`` (N, K): the missingness evidence factors e_n(i) (module
    docstring), by default those of ``model.missingness`` on ``miss``;
    ``None`` applies none. They multiply ``init`` and the columns of every
    ``trans[n]`` (e_{n+1}(i) repeated over the nodes g at a missing n+1)
    after the block renormalisation; ``self.ev`` keeps the factors for
    :func:`_nystrom_density`.

    ``grids`` {n: QuadratureGrid}: the grid of every missing n (by default
    :func:`_gap_grids`: the reference grid ``grid`` or a local one).
    ``block_dev`` {n: (raw, target)}: the raw quadrature masses of the blocks
    into n and their exact values, for :func:`_quadrature_report`, which
    sets ``quad_error``.
    """

    def __init__(self, model: PMCModel, Y: np.ndarray, miss: np.ndarray,
                 grid: QuadratureGrid, *, log: bool, ev=_inf._FROM_MODEL, grids=None):
        self.model = model
        self.K = K = model.K
        self.G = G = grid.G
        self.N = N = len(Y)
        self.miss = miss
        self.log = log
        self.grid = grid
        self.overflow = False
        Y = np.asarray(Y, dtype=float)
        self.Y = Y
        # Grid of every missing position: the reference grid, or a local one.
        self.grids = _gap_grids(model, Y, miss, grid) if grids is None else grids
        #: Raw quadrature masses of the blocks against their exact values
        #: (entry, inner and initial blocks; see :func:`_quadrature_report`).
        self.block_dev = {}
        self.quad_error = None
        Yf = np.where(miss, grid.nodes[G // 2], Y)
        # Raw weights: the evidence factors are applied below, uniformly.
        if log:
            Wobs, _ = _inf._log_transition_weights(model, Yf, ev=None)
        else:
            Wobs, _ = _inf._weights(model, Yf, None)
        self.Wobs = Wobs

        fN, FN = _margin_eval(model, grid.nodes, log=log)

        def kernel(fl, Fl, fr, Fr):
            if log:
                return _log_kernel(model, fl, Fl, fr, Fr)
            Wk, over = _kernel(model, fl, Fl, fr, Fr)
            self.overflow |= over
            return Wk

        def take(F, idx):
            return None if F is None else F[idx]

        def node_eval(g):
            return (fN, FN) if g is grid else _margin_eval(model, g.nodes, log=log)

        def weigh(ker, om):                              # × ω of the destination nodes
            return (ker + np.log(om)) if log else ker * om

        def is_ref(n):
            return self.grids[int(n)] is grid

        def between(A, B):
            """Q[(i, g), (j, g')] from the nodes of A to those of B, block-normalised."""
            fA, FA = node_eval(A)
            fB, FB = node_eval(B)
            rep, til = np.repeat(np.arange(G), G), np.tile(np.arange(G), G)
            ker = kernel(fA[rep], take(FA, rep), fB[til], take(FB, til)).reshape(G, G, K, K)
            Qb = weigh(ker.transpose(2, 0, 3, 1), B.omega[None, None, None, :])  # (K, G, K, G)
            T = _x_transition(model, fA, log=log).transpose(1, 0, 2)               # (K, G, K)
            Qb, _, raw = _normalise_blocks(Qb, T, log=log, with_mass=True)
            return Qb.reshape(K * G, K * G), (raw, T)

        m0, m1 = miss[:-1], miss[1:]
        inner = np.nonzero(m0 & m1)[0]
        entries = np.nonzero(~m0 & m1)[0]
        exits = np.nonzero(m0 & ~m1)[0]

        Q = None
        Qn = {}
        ref_inner = np.array([is_ref(n) and is_ref(n + 1) for n in inner], dtype=bool)
        if ref_inner.any():
            Q, dev = between(grid, grid)
            for n in inner[ref_inner]:
                self.block_dev[int(n) + 1] = dev
        for n in inner[~ref_inner]:
            Qn[int(n)], self.block_dev[int(n) + 1] = between(self.grids[int(n)], self.grids[int(n) + 1])

        E = {}
        ref_e = np.array([is_ref(n + 1) for n in entries], dtype=bool)
        for sel in (entries[ref_e], entries[~ref_e]):
            if not sel.size:
                continue
            fo, Fo = _margin_eval(model, Y[sel], log=log)
            e_rep = np.repeat(np.arange(sel.size), G)
            if is_ref(sel[0] + 1):
                g_til = np.tile(np.arange(G), sel.size)
                fr, Fr, om = fN[g_til], take(FN, g_til), grid.omega
            else:
                gs = [self.grids[int(n) + 1] for n in sel]
                fr, Fr = _margin_eval(model, np.concatenate([g.nodes for g in gs]), log=log)
                om = np.stack([g.omega for g in gs])[:, None, None, :]
            ker = kernel(fo[e_rep], take(Fo, e_rep), fr, Fr)
            ker = weigh(ker.reshape(sel.size, G, K, K).transpose(0, 2, 3, 1), om)  # (n_e, K, K, G)
            T = _x_transition(model, fo, log=log)
            ker, _, raw = _normalise_blocks(ker, T, log=log, with_mass=True)
            for e, n in enumerate(sel):
                E[int(n)] = ker[e]
                self.block_dev[int(n) + 1] = (raw[e], T[e])

        X = {}
        ref_x = np.array([is_ref(n) for n in exits], dtype=bool)
        for sel in (exits[ref_x], exits[~ref_x]):
            if not sel.size:
                continue
            fo, Fo = _margin_eval(model, Y[sel + 1], log=log)
            x_rep = np.repeat(np.arange(sel.size), G)
            if is_ref(sel[0]):
                g_til = np.tile(np.arange(G), sel.size)
                fl, Fl = fN[g_til], take(FN, g_til)
            else:
                fl, Fl = _margin_eval(model, np.concatenate([self.grids[int(n)].nodes for n in sel]),
                                      log=log)
            ker = kernel(fl, Fl, fo[x_rep], take(Fo, x_rep))
            ker = ker.reshape(sel.size, G, K, K).transpose(0, 2, 1, 3)     # (n_x, K, G, K)
            ker = ker.reshape(sel.size, K * G, K)
            for x, n in enumerate(sel):
                X[int(n)] = ker[x]

        if miss[0]:
            g0 = self.grids[0]
            f0, _ = node_eval(g0)
            mu = _initial(model, f0, log=log).T                  # (K, G)
            if not log and not np.all(np.isfinite(mu)):
                self.overflow = True
                mu = np.where(np.isfinite(mu), mu, 0.0)
            mu = weigh(mu, g0.omega[None, :])
            prior = _initial(model, np.zeros((1, K, K)) if log else np.ones((1, K, K)), log=log)
            init, _, raw = _normalise_blocks(mu[None], prior, log=log, with_mass=True)
            self.init = init[0]
            self.block_dev[0] = (raw[0], prior[0])
        else:
            f0, _ = _margin_eval(model, Y[:1], log=log)
            self.init = _initial(model, f0, log=log)[0]

        trans = []
        for n in range(N - 1):
            if not m0[n] and not m1[n]:
                trans.append(Wobs[n])
            elif not m0[n]:
                trans.append(E[n])
            elif m1[n]:
                trans.append(Qn.get(n, Q))
            else:
                trans.append(X[n])
        self.trans = trans

        if ev is _inf._FROM_MODEL:
            ev = _inf._evidence(model, miss)
        self.ev = ev
        if ev is not None:
            # Missingness evidence: a likelihood factor of the destination
            # state, after the block renormalisation (module docstring).
            with np.errstate(divide="ignore"):
                fac = np.log(ev) if log else np.asarray(ev, dtype=float)

            def at(n):                       # (S_n,) factor, augmented layout
                return np.repeat(fac[n], G) if miss[n] else fac[n]

            def scale(T, v):
                return (T + v) if log else (T * v)

            self.init = scale(self.init, at(0))
            q_scaled = {}                    # Q is shared: one copy per factor
            for n in range(N - 1):
                v = at(n + 1)
                if Q is not None and trans[n] is Q:
                    key = v.tobytes()
                    if key not in q_scaled:
                        q_scaled[key] = scale(Q, v[None, :])
                    trans[n] = q_scaled[key]
                else:
                    trans[n] = scale(trans[n], v[None, :])

    def size(self, n: int) -> int:
        return self.K * self.G if self.miss[n] else self.K

    def nodes_at(self, positions) -> np.ndarray:
        """(P, G) quadrature nodes of the missing ``positions``."""
        return np.array([self.grids[int(n)].nodes for n in positions]).reshape(-1, self.G)


def _forward_chain(chain: _Chain):
    """Scaled forward pass; ``None`` when a linear step underflows.

    Returns ``(alphas, log_lik)`` with ``alphas[n]`` the normalised message in
    *linear* scale (S_n,).
    """
    N = chain.N
    alphas = [None] * N
    if chain.log:
        la = chain.init
        lc = float(_inf._lse(la))
        if not np.isfinite(lc):
            raise IncompatibleObservationError(
                f"Forward pass (missing data): Y[0] has zero density under every "
                f"state even in log space (log C_1 = {lc!r})."
            )
        ll = lc
        la = la - lc
        alphas[0] = np.exp(la)
        for n in range(N - 1):
            r = _inf._lse(la[:, None] + chain.trans[n], axis=0)
            lc = float(_inf._lse(r))
            if not np.isfinite(lc):
                raise IncompatibleObservationError(
                    f"Forward pass (missing data): Y[{n + 1}] has zero density "
                    f"under every state reachable from step {n} (log C = {lc!r}), "
                    f"even in log space. The observation sequence is incompatible "
                    f"with the model."
                )
            ll += lc
            la = r - lc
            alphas[n + 1] = np.exp(la)
        return alphas, float(ll)

    a = chain.init
    C = float(a.sum())
    if not (np.isfinite(C) and C >= MIN_POSITIVE):
        return None
    ll = np.log(C)
    alphas[0] = a / C
    for n in range(N - 1):
        raw = alphas[n] @ chain.trans[n]
        C = float(raw.sum())
        if not (np.isfinite(C) and C >= MIN_POSITIVE):
            return None
        ll += np.log(C)
        alphas[n + 1] = raw / C
    return alphas, float(ll)


def _backward_chain(chain: _Chain):
    """Scaled backward pass (own normaliser per step); ``None`` on linear underflow."""
    N = chain.N
    betas = [None] * N
    S = chain.size(N - 1)
    if chain.log:
        lb = np.full(S, -np.log(S))
        betas[N - 1] = np.exp(lb)
        n_void = 0
        for n in range(N - 2, -1, -1):
            r = _inf._lse(chain.trans[n] + lb[None, :], axis=1)
            ld = float(_inf._lse(r))
            if np.isfinite(ld):
                lb = r - ld
            else:
                n_void += 1
                s = chain.size(n)
                lb = np.full(s, -np.log(s))
            betas[n] = np.exp(lb)
        if n_void:
            logger.warning(
                "Backward (missing data, log space): %d step(s) with zero weight "
                "under every state — β̂ reset to uniform there.", n_void,
            )
        return betas
    b = np.full(S, 1.0 / S)
    betas[N - 1] = b
    for n in range(N - 2, -1, -1):
        raw = chain.trans[n] @ betas[n + 1]
        D = float(raw.sum())
        if not (np.isfinite(D) and D >= MIN_POSITIVE):
            return None
        betas[n] = raw / D
    return betas


def _run_chain(model, Y, miss, grid, *, backward: bool, ev=_inf._FROM_MODEL):
    """Build the chain and run forward (and backward), linear first, log on failure.

    ``ev``: missingness evidence factors (:class:`_Chain`), by default those
    of ``model.missingness`` on ``miss``.
    """
    if ev is _inf._FROM_MODEL:
        ev = _inf._evidence(model, miss)
    grids = _gap_grids(model, Y, miss, grid)
    chain = _Chain(model, Y, miss, grid, log=False, ev=ev, grids=grids)
    fw = bw = None
    if not chain.overflow:
        fw = _forward_chain(chain)
        if fw is not None and backward:
            bw = _backward_chain(chain)
    if chain.overflow or fw is None or (backward and bw is None):
        logger.warning(
            "Missing-data pass: %s in linear space; recomputing in log space.",
            "a kernel value overflows" if chain.overflow else "a step underflows",
        )
        chain = _Chain(model, Y, miss, grid, log=True, ev=ev, grids=grids)
        fw = _forward_chain(chain)
        bw = _backward_chain(chain) if backward else None
    chain.quad_error = _quadrature_report(chain, fw[0], bw)
    limit = quad_warn_limit(miss)
    if chain.quad_error > limit:
        logger.warning(
            "Missing-data quadrature not converged: relative error %.1e > %.1e on the integrals "
            "the grid must reproduce exactly (%d missing rows, gap_nodes = %d). Results that "
            "involve the missing rows may be off; increase gap_nodes.",
            chain.quad_error, limit, int(miss.sum()), chain.G,
        )
    return chain, fw[0], fw[1], bw


#: Threshold of the quadrature diagnostic (:func:`_quadrature_report`) for
#: one run of missing rows; the WARNING threshold is QUAD_WARN · √(number of
#: runs) (:func:`quad_warn_limit`).
QUAD_WARN = 1e-3


def quad_warn_limit(miss: np.ndarray) -> float:
    """The WARNING threshold of ``quad_error`` for the mask ``miss``: QUAD_WARN · √R.

    R is the number of runs of missing rows. The report sums unsigned
    relative errors over the runs, while the likelihood errors of separate
    runs partly cancel: measured on Intel Lab mote 48 (3 090 runs), the report
    is 14, 0.16 and 0.039 at G = 64, 128 and 256 for log-likelihood errors of
    0.81, 0.04 and < 1e-3 nats.
    """
    return QUAD_WARN * math.sqrt(max(1, int(_runs(miss)[0].size)))


def _quadrature_report(chain: _Chain, alphas, betas=None) -> float:
    """Convergence diagnostic of the quadrature of a pass: its largest relative error.

    Every missing position is checked on integrals its grid must reproduce
    and whose exact value is known:

    * the raw block masses before the Tauchen–Hussey renormalisation
      (Σ_g ω_g q(j, y_g | u) against P(x_n = j | u) for every source u —
      the observed row before a gap, a node of the previous missing
      position, or the prior for a missing y_1), weighted by the posterior
      mass that goes through them (α̂ × transition × β̂ summed over the
      nodes of the block; a backward pass is run when the caller has none).
      Inside a leading gap, which carries the prior forward, and from the
      background panels of a local grid, which carry a broad law, the rows
      are not counted: the renormalisation conserves that mass wherever it
      relocates it;
    * at the last missing row before an observed y_{n+1}, on a local grid,
      the exit integrals Σ_g ω_g μ(i, y_g) q(k, y_{n+1} | i, y_g) against
      their closed form (:func:`_neighbour_error`): the exit kernel is as
      narrow as the transition, and the renormalisation does not see it (a
      position kept on the reference grid passed the selection's resolution
      test, :func:`_local_grid_list`).

    Each check is a weighted mean of relative errors; the report is the
    largest over the positions (0 with no missing row). The renormalisation
    makes the masses exact, so this does not measure the error of the
    results; it measures how well the grid resolves the laws it integrates.
    Measured (module docstring, "Convergence diagnostic"): above QUAD_WARN
    the results are off, below it they are not, on the cases of the
    test-suite.
    """
    K, G = chain.K, chain.G
    miss = chain.miss
    worst = 0.0
    if betas is None and chain.block_dev:
        betas = _backward_chain(chain)
    # a leading gap carries the prior forward: the renormalisation conserves
    # it wherever its mass goes, and its exit is checked below
    obs = np.nonzero(~miss)[0]
    lead = int(obs[0]) if obs.size else chain.N
    with np.errstate(all="ignore"):
        for n, (raw, tgt) in chain.block_dev.items():
            if 0 < n < lead:
                continue
            if chain.log:
                ok = np.isfinite(raw) & np.isfinite(tgt)
                rel = np.where(ok, np.abs(np.expm1(np.where(ok, raw - tgt, 0.0))), 0.0)
                T = np.where(np.isfinite(tgt), np.exp(tgt), 0.0)
                bad = np.isfinite(tgt) & ~np.isfinite(raw)
            else:
                ok = np.isfinite(raw) & (tgt > 0.0)
                rel = np.where(ok, np.abs(raw / np.where(ok, tgt, 1.0) - 1.0), 0.0)
                T = np.where(np.isfinite(tgt), tgt, 0.0)
                bad = (tgt > 0.0) & ~(np.isfinite(raw) & (raw > 0.0))
            rel = np.where(bad, 1.0, rel)
            if betas is None:                            # forward weights only
                if n == 0 and miss[0]:
                    w = T                                                # (K,)
                else:
                    a = alphas[n - 1]
                    a = a.reshape(K, G) if miss[n - 1] else a
                    w = a[..., None] * T
            elif n == 0 and miss[0]:
                w = (alphas[0] * betas[0]).reshape(K, G).sum(axis=1)
            else:
                Tn = chain.trans[n - 1]
                if chain.log:
                    L = np.log(alphas[n - 1])[:, None] + Tn + np.log(betas[n])[None, :]
                    mx = np.max(L)
                    J = np.exp(L - mx) if np.isfinite(mx) else np.zeros_like(L)
                else:
                    J = alphas[n - 1][:, None] * Tn * betas[n][None, :]
                w = J.reshape(J.shape[0], K, G).sum(axis=2)              # (S_{n-1}, K)
                w = w.reshape(raw.shape)
                src = chain.grids[n - 1] if miss[n - 1] else None
                if src is not None and src.local:
                    # sources on background panels carry a broad law, which
                    # the renormalised rows keep however sparse the nodes
                    bgn = ~np.isfinite(src.mix.loc[src.mix.panel])      # (G,)
                    w = np.where(bgn[None, :, None], 0.0, w)
            tot = w.sum()
            if np.isfinite(tot) and tot > 0.0:
                worst += float((w * rel).sum() / tot)
        exits = [int(n) for n in np.nonzero(miss[:-1] & ~miss[1:])[0] if chain.grids[int(n)].local]
        cached = [chain.grids[n].exit_error for n in exits]
        worst += float(sum(c for c in cached if c is not None))
        todo = np.array([n for n, c in zip(exits, cached) if c is None], dtype=int)
        if todo.size:
            nodes = np.array([chain.grids[int(n)].nodes for n in todo])
            omega = np.array([chain.grids[int(n)].omega for n in todo])
            e = _neighbour_error(chain.model, nodes, omega, np.full(todo.size, np.nan),
                                 chain.Y[todo + 1])
            worst += float(np.sum(e))
    return worst


# ---------------------------------------------------------------------------
# Posterior summaries of the augmented chain
# ---------------------------------------------------------------------------

@dataclass
class GapPosterior:
    """Posterior quantities of a sequence with missing observations.

    Attributes
    ----------
    miss       : (N,) bool — missing rows.
    log_lik    : float — log p(y_obs), the observed-data log-likelihood
                 (log p(y_obs, m) with a non-ignorable ``model.missingness``,
                 every posterior below being then given the mask m too).
    alpha_hat  : (N, K) — P(x_n = i | observations up to n).
    beta_hat   : (N, K) — normalised backward messages (module docstring for
                 their definition at a missing n).
    gamma      : (N, K) — P(x_n = i | y_obs).
    xi         : (N-1, K, K) or None — P(x_n = i, x_{n+1} = j | y_obs).
    method     : ``"exact"`` (K-state shortcut) or ``"grid"``.
    grid       : QuadratureGrid or None (exact shortcut) — the reference grid.
    node_post  : (M, K, G) or None — P(x_n = i, y_n ∈ node g | y_obs) at the
                 M missing positions ``np.nonzero(miss)[0]`` (grid variants).
    nodes      : (M, G) or None — the quadrature nodes y_g of each missing
                 position: the reference nodes, or those of its local grid
                 (module docstring, "Local grids").
    quad_error : float or None — the quadrature diagnostic of
                 :func:`_quadrature_report` (grid variants): the largest
                 relative error of the quadrature on the integrals it must
                 reproduce exactly, weighted by the posterior. Above
                 ``QUAD_WARN`` (1e-3) a WARNING is logged.
    """

    miss: np.ndarray
    log_lik: float
    alpha_hat: np.ndarray
    beta_hat: np.ndarray
    gamma: np.ndarray
    xi: np.ndarray | None
    method: str
    grid: QuadratureGrid | None = None
    node_post: np.ndarray | None = None
    nodes: np.ndarray | None = None
    quad_error: float | None = None

    @property
    def index(self) -> np.ndarray:
        """Positions of the missing rows."""
        return np.nonzero(self.miss)[0]


def _marginal_alpha(chain: _Chain, alphas) -> np.ndarray:
    K, G = chain.K, chain.G
    out = np.empty((chain.N, K))
    for n in range(chain.N):
        a = alphas[n]
        out[n] = a.reshape(K, G).sum(axis=1) if chain.miss[n] else a
    return out


def _chain_posterior(chain: _Chain, alphas, betas, *, want_xi: bool) -> dict:
    K, G, N, miss = chain.K, chain.G, chain.N, chain.miss
    alpha_hat = _marginal_alpha(chain, alphas)
    beta_hat = np.empty((N, K))
    gamma = np.empty((N, K))
    idx = np.nonzero(miss)[0]
    node_post = np.empty((idx.size, K, G))
    obs = np.nonzero(~miss)[0]
    if obs.size:
        A = np.array([alphas[n] for n in obs])
        B = np.array([betas[n] for n in obs])
        beta_hat[obs] = B
        gamma[obs] = _inf.smooth(A, B)
    for m, n in enumerate(idx):
        a = alphas[n].reshape(K, G)
        b = betas[n].reshape(K, G)
        joint = a * b
        tot = joint.sum()
        if np.isfinite(tot) and tot > 0.0:
            node_post[m] = joint / tot
        else:
            node_post[m] = a / max(a.sum(), MIN_POSITIVE)
        gamma[n] = node_post[m].sum(axis=1)
        am = a.sum(axis=1)
        bm = np.where(am > 0.0, joint.sum(axis=1) / np.where(am > 0.0, am, 1.0), b.mean(axis=1))
        s = bm.sum()
        beta_hat[n] = bm / s if (np.isfinite(s) and s > 0.0) else np.full(K, 1.0 / K)

    xi = None
    if want_xi and N > 1:
        xi = np.empty((N - 1, K, K))
        with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
            for n in range(N - 1):
                T = chain.trans[n]
                if chain.log:
                    la = np.log(alphas[n])[:, None]
                    lb = np.log(betas[n + 1])[None, :]
                    L = la + T + lb
                    mx = np.max(L)
                    J = np.exp(L - mx) if np.isfinite(mx) else np.zeros_like(L)
                else:
                    J = alphas[n][:, None] * T * betas[n + 1][None, :]
                if miss[n]:
                    J = J.reshape(K, G, -1).sum(axis=1)
                if miss[n + 1]:
                    J = J.reshape(K, K, G).sum(axis=2)
                s = J.sum()
                if np.isfinite(s) and s > 0.0:
                    xi[n] = J / s
                else:
                    xi[n] = gamma[n][:, None] * gamma[n + 1][None, :]
    return dict(alpha_hat=alpha_hat, beta_hat=beta_hat, gamma=gamma, xi=xi,
                node_post=node_post, nodes=chain.nodes_at(idx))


def _check_length(Y) -> None:
    if len(Y) < 2:
        raise ValueError(
            f"Inference with missing observations requires N ≥ 2 rows, got N={len(Y)}."
        )


def gap_posterior(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = None,
    xi: bool = True,
) -> GapPosterior:
    """Exact (up to quadrature) posterior quantities of a NaN-bearing Y.

    Works for every variant and for a Y without missing values too (then
    identical to the functions of :mod:`pmcprg.pmc.inference`). The exact
    shortcut variants use ``precompute_weights`` / ``forward`` / ``backward``
    / ``smooth`` / ``joint_posteriors`` on the marginalised weights; the grid
    variants run the augmented chain (module docstring). With a
    non-ignorable ``model.missingness`` every quantity is given (y_obs, m),
    the missingness factors included (module docstring, "Missingness
    mechanisms").

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) observations, NaN (any non-finite value) where missing.
    gap_nodes : number G of quadrature nodes (default 64; grid variants only).
    xi        : also compute the pairwise posteriors ξ (default True).
    """
    return _posterior(model, Y, gap_nodes, xi)[0]


def _posterior(model, Y, gap_nodes, xi, *, ev=_inf._FROM_MODEL):
    """:func:`gap_posterior` plus the run it came from.

    Returns ``(GapPosterior, run)`` with ``run = (W, None, None)`` for the
    exact shortcut (the marginalised weights, missingness factors included)
    and ``(chain, alphas, betas)`` for the grid. ``ev``: evidence factors,
    by default those of ``model.missingness`` on the mask of Y.
    """
    Y = np.asarray(Y, dtype=float)
    _check_length(Y)
    miss = missing_mask(Y)
    if ev is _inf._FROM_MODEL:
        ev = _inf._evidence(model, miss)
    if not needs_grid(model) or not miss.any():
        W, f_pdf = _inf._weights(model, Y, ev)
        a, ll = _inf._forward(model, Y, W, f_pdf, ev=ev)
        b = _inf._backward(model, Y, W, ev=ev)
        g = _inf.smooth(a, b)
        x = _inf.joint_posteriors(a, W, b) if xi else None
        return (GapPosterior(miss=miss, log_lik=float(ll), alpha_hat=a, beta_hat=b,
                             gamma=g, xi=x, method="exact"), (W, None, None))
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, ll, betas = _run_chain(model, Y, miss, grid, backward=True, ev=ev)
    post = _chain_posterior(chain, alphas, betas, want_xi=xi)
    return (GapPosterior(miss=miss, log_lik=float(ll), method="grid", grid=grid,
                         quad_error=chain.quad_error, **post),
            (chain, alphas, betas))


# ---- entry points used by pmcprg.pmc.inference --------------------------------

def _grid_forward(model, Y, miss, gap_nodes, *, ev=_inf._FROM_MODEL):
    _check_length(Y)
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, ll, _ = _run_chain(model, np.asarray(Y, dtype=float), miss, grid,
                                      backward=False, ev=ev)
    return _marginal_alpha(chain, alphas), ll


def _grid_backward(model, Y, miss, gap_nodes, *, ev=_inf._FROM_MODEL):
    return _posterior(model, Y, gap_nodes, False, ev=ev)[0].beta_hat


def _grid_sample(model, Y, miss, rng, gap_nodes, return_y):
    Y = np.asarray(Y, dtype=float)
    _check_length(Y)
    _check_grid_supported(model)
    grid = reference_grid(model, gap_nodes)
    chain, alphas, _, _ = _run_chain(model, Y, miss, grid, backward=False)
    U = _ffbs(chain.trans, alphas, chain.log, rng, 1)[0]
    X, Yc = _decode_path(U, miss, chain, Y)
    return (X, Yc) if return_y else X


# ---------------------------------------------------------------------------
# Forward-filter backward-sample on a chain of variable width (vectorised)
# ---------------------------------------------------------------------------

def _draw_rows(w: np.ndarray, rng: np.random.Generator, fallback: np.ndarray) -> np.ndarray:
    """One index per row of w (S, M), with probability ∝ the row."""
    S, M = w.shape
    tot = w.sum(axis=1)
    bad = ~(np.isfinite(tot) & (tot > 0.0))
    if bad.any():
        fb = np.asarray(fallback, dtype=float)
        s = fb.sum()
        fb = fb / s if (np.isfinite(s) and s > 0.0) else np.full(M, 1.0 / M)
        w = w.copy()
        w[bad] = fb
    cum = np.cumsum(w, axis=1)
    u = rng.random(S) * cum[:, -1]
    idx = np.sum(cum <= u[:, None], axis=1)
    over = idx >= M
    if over.any():
        last_pos = M - 1 - np.argmax(w[over, ::-1] > 0.0, axis=1)
        idx[over] = last_pos
    return idx


def _ffbs(trans, alphas, log: bool, rng: np.random.Generator, S: int) -> np.ndarray:
    """S joint draws of the (augmented) path; (S, N) indices."""
    N = len(alphas)
    U = np.empty((S, N), dtype=int)
    last = np.broadcast_to(alphas[N - 1], (S, alphas[N - 1].size))
    U[:, N - 1] = _draw_rows(np.array(last), rng, alphas[N - 1])
    with np.errstate(divide="ignore", under="ignore", over="ignore", invalid="ignore"):
        for n in range(N - 2, -1, -1):
            T = trans[n][:, U[:, n + 1]].T                 # (S, S_n)
            if log:
                L = np.log(alphas[n])[None, :] + T
                mx = np.max(L, axis=1, keepdims=True)
                w = np.where(np.isfinite(mx), np.exp(L - np.where(np.isfinite(mx), mx, 0.0)), 0.0)
            else:
                w = alphas[n][None, :] * T
            U[:, n] = _draw_rows(w, rng, alphas[n])
    return U


def _node_draws(chain: _Chain, U: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The node values (…, M) of augmented indices ``U`` (…, M) at the missing ``idx``."""
    nodes = chain.nodes_at(idx)                                      # (M, G)
    return nodes[np.arange(len(idx)), U % chain.G]


def _decode_path(U: np.ndarray, miss: np.ndarray, chain: _Chain, Y: np.ndarray):
    """Augmented indices → (states, Y with the missing rows set to the drawn nodes)."""
    X = np.where(miss, U // chain.G, U).astype(int)
    Yc = np.array(Y, dtype=float, copy=True)
    idx = np.nonzero(miss)[0]
    Yc[idx] = _node_draws(chain, U[idx], idx)
    return X, Yc


# ---------------------------------------------------------------------------
# Nyström interpolation of the posterior density of a missing observation
# ---------------------------------------------------------------------------

#: Size of the fine grid of the Nyström interpolation, in multiples of G.
_NYSTROM_FACTOR = 4


def _nystrom_density(chain: _Chain, alphas, betas, positions: np.ndarray,
                     points: np.ndarray) -> np.ndarray:
    """Unnormalised posterior density of y_n at ``points`` (P, M), for the missing ``positions``.

    The density of y_n = y given the observations follows from the messages
    of the neighbouring positions through the exact kernels, at any y:

        π_n(y) ∝ Σ_i F_n(i, y) B_n(i, y),
        F_n(i, y) = Σ_u α̃_{n-1}(u) q(i, y | u),
        B_n(i, y) = Σ_v T_{(i, y) → v} β̃_{n+1}(v),

    with u, v the (augmented) states at n − 1 and n + 1 on their own grids,
    and T_{(i, y) → v} the chain's transition out of (i, y), built like a row
    of Q (block-normalised to P(x_{n+1} | x_n = i, y_n = y)) or the exit
    density q(j, y_{n+1} | i, y) when y_{n+1} is observed (Nyström 1930).
    F_n propagates the discrete message of n − 1 through the exact kernel:
    ∫ F_n(i, y) dy = Σ_u α̃_{n−1}(u) P(x_n = i | u) exactly, wherever the
    kernel puts its mass. (Until this version F_n also carried the block
    factors of the chain's renormalisation — exact at the nodes, but a factor
    of up to 1e30 off the nodes for a row whose kernel the grid misses, which
    made the quantiles of problem P3 in ``report/forecasting``.)
    ``betas`` None means B ≡ 1 (a trailing gap, for :func:`forecast`).
    With missingness factors (``chain.ev``) F_n is multiplied by e_n(i) and
    β̃_{n+1}(v) by e_{n+1}(v), as the chain's messages are.

    Rows that are not finite and positive somewhere are returned as NaN (the
    caller then uses the coarse node masses).
    """
    model, K, G = chain.model, chain.K, chain.G
    miss, N, Y = chain.miss, chain.N, chain.Y
    positions = np.asarray(positions, dtype=int)
    points = np.asarray(points, dtype=float)
    P, M = points.shape
    ev = chain.ev
    fP, FP = _margin_eval(model, points.ravel(), log=False)
    fP = fP.reshape(P, M, K, K)
    FP = None if FP is None else FP.reshape(P, M, K, K)
    cache = {}

    def node_margins(g):
        key = id(g)
        if key not in cache:
            cache[key] = _margin_eval(model, g.nodes, log=False)
        return cache[key]

    def take(F, idx):
        return None if F is None else F[idx]

    def kern(fl, Fl, fr, Fr):
        return _kernel(model, fl, Fl, fr, Fr)[0]

    with np.errstate(divide="ignore", invalid="ignore", under="ignore", over="ignore"):
        F = np.zeros((P, K, M))
        B = np.ones((P, K, M))
        first = positions == 0
        prev_miss = np.zeros(P, bool)
        prev_miss[~first] = miss[positions[~first] - 1]
        last = positions == N - 1
        next_miss = np.zeros(P, bool)
        next_miss[~last] = miss[positions[~last] + 1]

        # ---- incoming side ------------------------------------------------
        for p in np.nonzero(first)[0]:
            F[p] = _initial(model, fP[p], log=False).T                          # μ(i, y)
        sel = np.nonzero(~first & ~prev_miss)[0]
        if sel.size:
            n_prev = positions[sel] - 1
            fo, Fo = _margin_eval(model, Y[n_prev], log=False)
            rep_ = np.repeat(np.arange(sel.size), M)
            ker = kern(fo[rep_], take(Fo, rep_), fP[sel].reshape(-1, K, K),
                       take(FP, sel).reshape(-1, K, K) if FP is not None else None)
            ker = ker.reshape(sel.size, M, K, K)                                   # [p, m, h, i]
            a = np.array([alphas[int(n)] for n in n_prev])                          # (P', K)
            F[sel] = np.einsum("ph,pmhi->pim", a, ker)
        for p in np.nonzero(~first & prev_miss)[0]:
            n = int(positions[p])
            A = chain.grids[n - 1]
            fA, FA = node_margins(A)
            rep_, til = np.repeat(np.arange(G), M), np.tile(np.arange(M), G)
            ker = kern(fA[rep_], take(FA, rep_), fP[p][til], take(FP, p)[til] if FP is not None else None)
            C = ker.reshape(G, M, K, K).transpose(2, 0, 3, 1).reshape(K * G, K * M)
            F[p] = (alphas[n - 1] @ C).reshape(K, M)
        if ev is not None:
            F *= ev[positions][:, :, None]                     # e_n(i) of the transition into n

        # ---- outgoing side ------------------------------------------------
        if betas is not None:
            sel = np.nonzero(~last & ~next_miss)[0]
            if sel.size:
                n_next = positions[sel] + 1
                fo, Fo = _margin_eval(model, Y[n_next], log=False)
                rep_ = np.repeat(np.arange(sel.size), M)
                ker = kern(fP[sel].reshape(-1, K, K),
                           take(FP, sel).reshape(-1, K, K) if FP is not None else None,
                           fo[rep_], take(Fo, rep_)).reshape(sel.size, M, K, K)
                b = np.array([betas[int(n)] for n in n_next])                     # (P', K)
                if ev is not None:
                    b = b * ev[n_next]                          # e_{n+1}(j), as in the chain
                B[sel] = np.einsum("pmij,pj->pim", ker, b)
            for p in np.nonzero(~last & next_miss)[0]:
                n = int(positions[p])
                Bg = chain.grids[n + 1]
                fB, FB = node_margins(Bg)
                rep_, til = np.repeat(np.arange(M), G), np.tile(np.arange(G), M)
                ker = kern(fP[p][rep_], take(FP, p)[rep_] if FP is not None else None,
                           fB[til], take(FB, til)).reshape(M, G, K, K)
                T = ker.transpose(2, 0, 3, 1) * Bg.omega                          # (K, M, K, G)
                T, _ = _normalise_blocks(T, _x_transition(model, fP[p], log=False).transpose(1, 0, 2),
                                         log=False)
                b = betas[n + 1]
                if ev is not None:
                    b = b * np.repeat(ev[n + 1], G)             # e_{n+1}(j), per node
                B[p] = (T.reshape(K * M, K * G) @ b).reshape(K, M)
            # the chain rescales β̃ by its own sum at every step: any positive
            # constant per position cancels in the normalisation.

        dens = (F * B).sum(axis=1)
    bad = ~(np.all(np.isfinite(dens), axis=1) & (dens.sum(axis=1) > 0.0))
    dens[bad] = np.nan
    return dens


def _fine_grids(chain: _Chain, positions: np.ndarray) -> list:
    """The Nyström grid (_NYSTROM_FACTOR × G nodes) of every missing position."""
    ref = chain.grid
    ref_fine = reference_grid(chain.model, _NYSTROM_FACTOR * ref.G)
    return [_fine_grid(ref, ref_fine, chain.grids[int(n)], _NYSTROM_FACTOR) for n in positions]


# ---------------------------------------------------------------------------
# Laws on the grid: moments and quantiles
# ---------------------------------------------------------------------------

#: Tolerance of the quantile safety net on the CDF of a law on the grid: the
#: interpolated CDF must be non-decreasing and stay within this distance of
#: the Markov–Stieltjes bracket [Σ_{g'<g} m_g', Σ_{g'≤g} m_g'] at every node.
_CDF_TOL = 1e-6


def _legendre_cdf(masses: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Coefficients (n+1, R) of the CDF H(s) = ½ Σ_k c_k P_k(2s − 1) of rules.

    ``masses`` (R, n) the masses of R rules of n Gauss–Legendre nodes ``s``
    (n,): the Legendre interpolant of the density h(s) = m_g / ws_g,
    integrated from 0 (spectrally accurate for a smooth h).
    """
    n = s.size
    V = _leg.legvander(2.0 * s - 1.0, n - 1)                         # (n, n)
    c = (masses @ V) * (2.0 * np.arange(n) + 1.0)                   # (R, n)
    return _leg.legint(c, lbnd=-1, axis=1).T                        # (n+1, R)


def _cdf_ok(cint: np.ndarray, masses: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Safety net (R,) of :func:`_legendre_cdf`: monotone, and within _CDF_TOL
    of the Markov–Stieltjes bracket of the masses at the nodes."""
    n = s.size
    dense = np.linspace(0.0, 1.0, 4 * n + 1)
    H = 0.5 * _leg.legval(2.0 * dense - 1.0, cint)                   # (R, 4n+1)
    mono = np.all(np.diff(H, axis=1) >= -_CDF_TOL, axis=1)
    Hn = 0.5 * _leg.legval(2.0 * s - 1.0, cint)                      # (R, n)
    cum = np.cumsum(masses, axis=1)
    inside = np.all((Hn >= cum - masses - _CDF_TOL) & (Hn <= cum + _CDF_TOL), axis=1)
    return mono & inside & np.all(np.isfinite(H), axis=1)


def _monotone_ppf(masses: np.ndarray, ws: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Fallback quantiles in s (Q,) of a rule: the piecewise-linear CDF through
    the cumulative masses at the Gauss–Legendre cell edges Σ_{g'≤g} ws_g',
    inverted — monotone whatever the masses (first-order accurate)."""
    edges = np.concatenate([[0.0], np.cumsum(ws)])
    edges[-1] = 1.0
    cum = np.concatenate([[0.0], np.cumsum(np.maximum(masses, 0.0))])
    cum = cum / cum[-1] if cum[-1] > 0 else np.linspace(0.0, 1.0, cum.size)
    keep = np.concatenate([[True], np.diff(cum) > 0.0])      # strictly increasing knots
    return np.clip(np.interp(np.clip(levels, 0.0, 1.0), cum[keep], edges[keep]), 0.0, 1.0)


def _grid_summary(mass: np.ndarray, nodes: np.ndarray, qs: tuple[float, ...],
                  fine_mass: np.ndarray | None, grids: list, fine: list | None):
    """Mean, sd and quantiles of the laws with node masses ``mass`` (P, G).

    Moments are the quadratures Σ_g mass_g y_g^k on each position's nodes.
    Quantiles invert the CDF of the Legendre interpolant, in the Gauss–
    Legendre variable s of each rule, of the density h(s) = mass_g / ws_g —
    spectrally accurate for a smooth h — then map s to y; with ``fine_mass``
    (the Nyström masses on the grids ``fine``; NaN rows: not available) that
    finer representation is used. A local grid is a composite rule: the CDF
    at a panel boundary is the sum of the masses of the panels below, the
    interpolant is used inside a panel. Safety net (:func:`_cdf_ok`): where
    the interpolated CDF is not monotone or leaves the Markov–Stieltjes
    bracket of the masses by more than _CDF_TOL, the monotone PCHIP CDF of
    the cumulative masses is inverted instead (first-order accurate) and a
    WARNING is logged.
    """
    mass = np.asarray(mass, dtype=float)
    tot = mass.sum(axis=1, keepdims=True)
    mass = mass / np.where(tot > 0.0, tot, 1.0)
    y = np.asarray(nodes, dtype=float)
    if all(not g.local for g in grids):
        y = y[0] if y.size else np.zeros(mass.shape[1])       # the reference nodes, shared
        mean = mass @ y
        var = mass @ (y * y) - mean * mean
    else:
        mean = (mass * y).sum(axis=1)
        var = (mass * y * y).sum(axis=1) - mean * mean
    sd = np.sqrt(np.maximum(var, 0.0))
    P = mass.shape[0]
    if not qs:
        return mean, sd, np.empty((P, 0))
    levels = np.array(qs, dtype=float)
    use_fine = np.zeros(P, bool) if fine_mass is None else np.all(np.isfinite(fine_mass), axis=1)
    qv = np.empty((P, levels.size))
    n_fallback = 0
    # ---- single-rule grids (the reference grid): vectorised --------------
    for fine_rows in (True, False):
        rows = [p for p in range(P) if use_fine[p] == fine_rows and not grids[p].local]
        if not rows:
            continue
        g = fine[rows[0]] if fine_rows else grids[rows[0]]
        m = np.asarray(fine_mass[rows] if fine_rows else mass[rows], dtype=float)
        m = m / m.sum(axis=1, keepdims=True)
        cint = _legendre_cdf(m, g.s)
        target = levels[:, None] * np.ones((1, len(rows)))           # (Q, R)
        lo = np.zeros_like(target)
        hi = np.ones_like(target)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            H = 0.5 * _leg.legval(2.0 * mid - 1.0, cint, tensor=False)
            below = H < target
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        s_q = 0.5 * (lo + hi)                                          # (Q, R)
        ok = _cdf_ok(cint, m, g.s)
        for r in np.nonzero(~ok)[0]:
            s_q[:, r] = _monotone_ppf(m[r], g.ws, levels)
            n_fallback += 1
        qv[rows] = g.y_of_s(s_q).T
    # ---- composite (local) grids ------------------------------------------
    for p in range(P):
        if not grids[p].local:
            continue
        g = fine[p] if use_fine[p] else grids[p]
        m = np.asarray(fine_mass[p] if use_fine[p] else mass[p], dtype=float)
        m = m / m.sum()
        pan = g.mix
        edges = np.concatenate([[0], np.cumsum(pan.sizes)])
        pm = np.array([m[edges[k]:edges[k + 1]].sum() for k in range(pan.sizes.size)])
        cum = np.concatenate([[0.0], np.cumsum(pm)])
        out_s = np.empty(levels.size)
        out_k = np.empty(levels.size, dtype=int)
        for a, q in enumerate(levels):
            k = int(min(np.searchsorted(cum[1:], q, side="left"), pan.sizes.size - 1))
            sl = slice(edges[k], edges[k + 1])
            mk, sk = m[sl], g.s[sl]
            r = q - cum[k]
            if pm[k] <= 0.0:
                out_s[a], out_k[a] = 0.5, k
                continue
            cint = _legendre_cdf(mk[None, :], sk)[:, 0]
            if _cdf_ok(cint[:, None], mk[None, :], sk)[0]:
                lo_, hi_ = 0.0, 1.0
                for _ in range(60):
                    mid = 0.5 * (lo_ + hi_)
                    if 0.5 * _leg.legval(2.0 * mid - 1.0, cint) < r:
                        lo_ = mid
                    else:
                        hi_ = mid
                out_s[a] = 0.5 * (lo_ + hi_)
            else:
                out_s[a] = _monotone_ppf(mk / pm[k], g.ws[sl], np.array([r / pm[k]]))[0]
                n_fallback += 1
            out_k[a] = k
        qv[p] = g.y_of_s(out_s, out_k)
    if n_fallback:
        logger.warning(
            "Quantiles of the missing values: the interpolated CDF of %d law(s) is not "
            "monotone or leaves the bracket of its node masses by more than %g; used the "
            "monotone CDF of the node masses there (first-order accurate).",
            n_fallback, _CDF_TOL,
        )
    return mean, sd, qv


def _grid_laws(chain: _Chain, alphas, betas, positions: np.ndarray, mass: np.ndarray, qs):
    """Mean, sd, quantiles and reference-node density of the laws of y at ``positions``.

    ``mass`` (P, G) the node masses on each position's grid. Quantiles use the
    Nyström masses on the fine grids (:func:`_nystrom_density`), the coarse
    masses where those are not available. The density is returned at the
    reference nodes: mass / ω on the reference grid (a position on it), the
    normalised Nyström density on a local grid.
    """
    G = chain.G
    ref = chain.grid
    grids = [chain.grids[int(n)] for n in positions]
    nodes = chain.nodes_at(positions)
    loc_rows = np.array([g.local for g in grids], dtype=bool)
    fine = fine_mass = None
    if qs or loc_rows.any():
        fine = _fine_grids(chain, positions)
        pts = np.array([g.nodes for g in fine])
        dens = _nystrom_density(chain, alphas, betas, positions,
                                np.concatenate([pts, np.broadcast_to(ref.nodes, (len(grids), G))],
                                               axis=1))
        Mf = pts.shape[1]
        om = np.array([g.omega for g in fine])
        with np.errstate(invalid="ignore"):
            fm = dens[:, :Mf] * om
            Z = fm.sum(axis=1, keepdims=True)
            fine_mass = fm / Z
            dref = dens[:, Mf:] / Z
    mean, sd, qv = _grid_summary(mass, nodes, qs, fine_mass if qs else None, grids, fine)
    density = mass / ref.omega[None, :]
    if loc_rows.any():
        dl = dref[loc_rows]
        density[loc_rows] = np.where(np.isfinite(dl), dl, np.nan)
    return mean, sd, qv, density


def _state_dists(model: PMCModel) -> list:
    return [_frozen(model.margin(k)) for k in range(model.K)]


def _mixture_summary(model: PMCModel, weights: np.ndarray, qs: tuple[float, ...]):
    """Mean, sd, quantiles of Σ_k weights[m, k] f_k (state margins, exact).

    d = 1: shapes (M,), (M,), (M, Q). d > 1 (multivariate normal margins):
    (M, d), (M, d) and per-component quantiles (M, Q, d).
    """
    K = model.K
    d = getattr(model, "d", 1)
    weights = np.asarray(weights, dtype=float)
    M = weights.shape[0]
    if d == 1:
        dists = _state_dists(model)
        mu = np.array([float(dd.mean()) for dd in dists])
        v = np.array([float(dd.var()) for dd in dists])
        mean = weights @ mu
        sd = np.sqrt(np.maximum(weights @ (v + mu * mu) - mean * mean, 0.0))
        qv = (_mixture_ppf(weights, dists, np.broadcast_to(np.array(qs), (M, len(qs))))
              if qs else np.empty((M, 0)))
        return mean, sd, qv
    mus = np.array([model.margin(k).params["mean"] for k in range(K)], dtype=float)       # (K, d)
    vs = np.array([np.diag(np.asarray(model.margin(k).params["cov"], dtype=float))
                   for k in range(K)])                                                     # (K, d)
    mean = weights @ mus
    sd = np.sqrt(np.maximum(weights @ (vs + mus * mus) - mean * mean, 0.0))
    qv = np.empty((M, len(qs), d))
    for c in range(d):
        dists = [_ss.norm(loc=mus[k, c], scale=np.sqrt(vs[k, c])) for k in range(K)]
        if qs:
            qv[:, :, c] = _mixture_ppf(weights, dists, np.broadcast_to(np.array(qs), (M, len(qs))))
    return mean, sd, qv


def _mixture_density(model: PMCModel, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    dists = _state_dists(model)
    dens = np.stack([dd.pdf(y) for dd in dists], axis=0)          # (K, G)
    return np.asarray(weights, dtype=float) @ dens


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------

@dataclass
class Imputation:
    """Posterior law of the missing observations given all the observed ones.

    Attributes
    ----------
    index           : (M,) positions of the missing rows.
    mean, sd        : (M,) — or (M, d) — posterior mean and standard deviation
                      of y_n given y_obs.
    quantiles       : the requested levels.
    quantile_values : (M, Q) — or (M, Q, d), per component — posterior quantiles.
    gamma           : (M, K) — P(x_n = i | y_obs) at the missing positions.
    log_lik         : float — log p(y_obs).
    Y_mean          : Y with every missing row replaced by its posterior mean.
    method          : ``"exact"`` or ``"grid"``.
    nodes           : (G,) or None — the reference nodes y_g (d = 1).
    density         : (M, G) or None — posterior density of y_n at ``nodes``
                      (the exact mixture density for the shortcut variants;
                      mass / ω on the reference grid, the Nyström density on
                      a local grid, see :func:`_grid_laws`).
    grid_nodes      : (M, G) or None — the quadrature nodes of each missing
                      position (grid variants: reference or local grid).
    grid_mass       : (M, G) or None — the posterior masses on ``grid_nodes``,
                      the discrete law whose moments are ``mean`` and ``sd``.
    x_samples       : (S, N) int or None — FFBS draws of the whole state path.
    y_samples       : (S, M) — or (S, M, d) — or None: the matching draws of the
                      missing observations (exact continuous draws for the
                      shortcut variants; drawn on the quadrature nodes — the
                      discrete law whose moments are ``mean`` and ``sd`` — for
                      the grid variants).
    """

    index: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    quantiles: tuple
    quantile_values: np.ndarray
    gamma: np.ndarray
    log_lik: float
    Y_mean: np.ndarray
    method: str
    nodes: np.ndarray | None = None
    density: np.ndarray | None = None
    x_samples: np.ndarray | None = None
    y_samples: np.ndarray | None = None
    grid_nodes: np.ndarray | None = None
    grid_mass: np.ndarray | None = None


def impute(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = DEFAULT_GAP_NODES,
    quantiles=DEFAULT_QUANTILES,
    n_samples: int = 0,
    rng=None,
) -> Imputation:
    """Posterior law of every missing y_n given all the observed data.

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) with NaN (any non-finite value) at missing rows.
    gap_nodes : quadrature nodes G for the grid variants (default 64).
    quantiles : levels in (0, 1) of the reported posterior quantiles.
    n_samples : number S of joint FFBS draws of (x_{1:N}, y_miss) (default 0).
    rng       : seed or :class:`numpy.random.Generator` for the draws.

    Returns
    -------
    Imputation — see its docstring. A Y without missing rows gives empty
    per-position arrays.
    """
    Y = np.asarray(Y, dtype=float)
    qs = _check_quantiles(quantiles)
    n_samples = int(n_samples)
    if n_samples < 0:
        raise ValueError(f"n_samples must be ≥ 0, got {n_samples}.")
    post, run = _posterior(model, Y, gap_nodes, False)
    idx = post.index
    d = getattr(model, "d", 1)
    nodes = density = grid_nodes = grid_mass = None
    if post.method == "grid":
        chain, alphas, betas = run
        mass = post.node_post.sum(axis=1)                              # (M, G)
        mean, sd, qv, density = _grid_laws(chain, alphas, betas, idx, mass, qs)
        nodes, grid_nodes, grid_mass = post.grid.nodes, post.nodes, mass
    elif idx.size == 0:
        # Complete data (any variant): nothing to impute.
        tail = (d,) if d > 1 else ()
        mean, sd = np.empty((0,) + tail), np.empty((0,) + tail)
        qv = np.empty((0, len(qs)) + tail)
    else:
        w = post.gamma[idx]
        mean, sd, qv = _mixture_summary(model, w, qs)
        if d == 1:
            nodes = reference_grid(model, gap_nodes).nodes
            density = _mixture_density(model, w, nodes)
    Y_mean = np.array(Y, copy=True)
    Y_mean[idx] = mean

    x_s = y_s = None
    if n_samples > 0:
        gen = np.random.default_rng(rng)
        miss = post.miss
        if post.method == "grid":
            chain, alphas, _ = run
            U = _ffbs(chain.trans, alphas, chain.log, gen, n_samples)
            x_s = np.where(miss[None, :], U // chain.G, U).astype(int)
            y_s = _node_draws(chain, U[:, idx], idx)
        else:
            W = run[0]
            trans = [W[n] for n in range(len(Y) - 1)]
            alphas = list(post.alpha_hat)
            x_s = _ffbs(trans, alphas, False, gen, n_samples)
            y_s = _draw_state_margins(model, x_s[:, idx], gen)
    return Imputation(index=idx, mean=mean, sd=sd, quantiles=qs, quantile_values=qv,
                      gamma=post.gamma[idx], log_lik=post.log_lik, Y_mean=Y_mean,
                      method=post.method, nodes=nodes, density=density,
                      x_samples=x_s, y_samples=y_s, grid_nodes=grid_nodes, grid_mass=grid_mass)


def _draw_state_margins(model: PMCModel, states: np.ndarray, rng) -> np.ndarray:
    """y ~ f_{state} elementwise for integer ``states`` of any shape."""
    d = getattr(model, "d", 1)
    out = np.empty(states.shape + ((d,) if d > 1 else ()))
    for k in range(model.K):
        sel = states == k
        cnt = int(sel.sum())
        if cnt:
            draws = np.asarray(model.margin(k).rvs(cnt, rng), dtype=float)
            out[sel] = draws.reshape((cnt, d) if d > 1 else (cnt,))
    return out


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

@dataclass
class Forecast:
    """h-step predictive law of (x_{N+k}, y_{N+k}) given the observed part of Y.

    Attributes
    ----------
    h               : horizon.
    state_probs     : (h, K) — P(x_{N+k} = j | y_obs), k = 1..h.
    mean, sd        : (h,) — or (h, d) — predictive mean and standard deviation.
    quantiles       : the requested levels.
    quantile_values : (h, Q) — or (h, Q, d) — predictive quantiles.
    log_lik         : float — log p(y_obs) of the conditioning sequence.
    method          : ``"exact"`` or ``"grid"``.
    nodes           : (G,) or None — the reference nodes y_g (d = 1).
    density         : (h, G) or None — predictive density at ``nodes``.
    grid_nodes      : (h, G) or None — the quadrature nodes of each horizon
                      (grid variants: reference or local grid).
    grid_mass       : (h, G) or None — the predictive masses on ``grid_nodes``.
    """

    h: int
    state_probs: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    quantiles: tuple
    quantile_values: np.ndarray
    log_lik: float
    method: str
    nodes: np.ndarray | None = None
    density: np.ndarray | None = None
    grid_nodes: np.ndarray | None = None
    grid_mass: np.ndarray | None = None


def forecast(
    model: PMCModel,
    Y: np.ndarray,
    h: int,
    *,
    gap_nodes: int | None = DEFAULT_GAP_NODES,
    quantiles=DEFAULT_QUANTILES,
) -> Forecast:
    """Predictive laws of the next h observations, a trailing gap of length h.

    Runs the forward filter on Y followed by h missing rows: the normalised
    filter at N + k is P(x_{N+k}, y_{N+k} | y_obs) (no backward message is
    needed for a trailing gap). Y may contain missing rows.

    With a non-ignorable ``model.missingness`` the laws are given
    (y_obs, m_{1:N}): the N rows of Y carry their missingness factors, the h
    appended rows none — their mask is unknown, and summing p(m_n | m_{n-1},
    x_n) over it gives 1.

    Parameters
    ----------
    model     : PMCModel — known parameters.
    Y         : (N,) or (N, d) conditioning observations (N ≥ 1).
    h         : horizon ≥ 1.
    gap_nodes : quadrature nodes G for the grid variants (default 64).
    quantiles : levels in (0, 1) of the predictive quantiles.
    """
    Y = np.asarray(Y, dtype=float)
    qs = _check_quantiles(quantiles)
    h = int(h)
    if h < 1:
        raise ValueError(f"forecast horizon h must be ≥ 1, got {h}.")
    if len(Y) < 1:
        raise ValueError("forecast needs at least one row of Y (it may be NaN).")
    tail = np.full((h,) + Y.shape[1:], np.nan)
    Yx = np.concatenate([Y, tail], axis=0)
    N = len(Y)
    miss = missing_mask(Yx)
    # Missingness factors of the N conditioning rows only: the mask of the h
    # future rows is unknown, and Σ_m p(m_n | m_{n-1}, x_n) = 1.
    ev = _inf._evidence(model, miss[:N])
    if ev is not None:
        ev = np.concatenate([ev, np.ones((h, model.K))], axis=0)
    d = getattr(model, "d", 1)
    nodes = density = grid_nodes = grid_mass = None
    if needs_grid(model):
        _check_grid_supported(model)
        grid = reference_grid(model, gap_nodes)
        chain, alphas, ll, _ = _run_chain(model, Yx, miss, grid, backward=False, ev=ev)
        K, G = chain.K, chain.G
        joint = np.array([alphas[N + k].reshape(K, G) for k in range(h)])   # (h, K, G)
        joint /= joint.sum(axis=(1, 2), keepdims=True)
        state_probs = joint.sum(axis=2)
        mass = joint.sum(axis=1)
        pos = np.arange(N, N + h)
        mean, sd, qv, density = _grid_laws(chain, alphas, None, pos, mass, qs)
        method = "grid"
        nodes, grid_nodes, grid_mass = grid.nodes, chain.nodes_at(pos), mass
    else:
        W, f_pdf = _inf._weights(model, Yx, ev)
        a, ll = _inf._forward(model, Yx, W, f_pdf, ev=ev)
        state_probs = a[N:]
        state_probs = state_probs / state_probs.sum(axis=1, keepdims=True)
        mean, sd, qv = _mixture_summary(model, state_probs, qs)
        method = "exact"
        if d == 1:
            nodes = reference_grid(model, gap_nodes).nodes
            density = _mixture_density(model, state_probs, nodes)
    return Forecast(h=h, state_probs=state_probs, mean=mean, sd=sd, quantiles=qs,
                    quantile_values=qv, log_lik=float(ll), method=method,
                    nodes=nodes, density=density, grid_nodes=grid_nodes, grid_mass=grid_mass)
