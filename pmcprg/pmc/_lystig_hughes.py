"""
pmcprg.pmc._lystig_hughes — Lystig & Hughes' (2002) exact observed information
of the prior, margin and copula parameters of a model fitted by ICE
(AUDIT_COPULES FR-4, "Reste": standard errors inside ICE, third of the three
ranked options after :mod:`pmcprg.pmc._oakes` and :mod:`pmcprg.pmc._godambe`).

Scope
-----
K-state models whose transition weight factorises as
``W[n, i, j] = A_ij · f_j(y_{n+1}) · c_ij(F_i(y_n), F_j(y_{n+1}))`` — HMC-DN,
and PMC with state margins (for which :func:`~pmcprg.pmc.inference.precompute_weights`'
``p_ij f_i(y_n) / Σ_k p_ik f_i(y_n)`` is exactly ``A_ij``, the y_n-dependence
cancelling for *every* value of the margin parameters, which is what lets the
margin extension below treat both variants with one formula). Validated at
K = 2. Complete data only — **gaps are out of scope** (:mod:`pmcprg.pmc.gaps`;
see "What extending this would need").

Two axes were added to the original pilot, which handled one-parameter
copulas with the margins held fixed:

* **copula families** — the one-parameter Gauss and Clayton (the pilot), and
  now the two-parameter **BB1** and **Student**, whose pair contributes two
  coordinates to θ instead of one. Tawn and t-EV are *not* covered: they are
  two-parameter families, but :func:`pmcprg.copulas._stderr._spec_of` has no
  ``_Spec`` for them (it raises ``NotImplementedError`` for any
  ``n_params != 1`` family other than Student and BB1), i.e. they have no
  registered working coordinate ψ and no analytic Jacobian to (τ, ·) — the
  two things this module reuses rather than reinvents. Registering them is a
  :mod:`pmcprg.copulas._stderr` job, not this one's.
* **margins** — held fixed (``fit_margins=False``, the pilot, still the
  default) or, with ``fit_margins=True``, **re-estimated inside ICE**: the
  ``2K`` coordinates of the state-indexed **Gaussian** margins join θ. That
  is deliberately the same case :mod:`pmcprg.pmc._godambe` piloted, so the
  two methods are comparable on one fixture. Non-Gaussian state margins and
  **pair margins** stay out of scope (see the end of this docstring).

What it computes, and why it is worth having next to Oakes and Godambe
-----------------------------------------------------------------------
The **full** observed information matrix ``I(θ) = −∂²ℓ(θ)/∂θ∂θᵀ`` of the
observed-data log-likelihood ℓ(θ) = log p(y_{1:N}; θ) — the value
:func:`~pmcprg.pmc.inference.forward` returns — over the joint parameter
vector θ = (prior coordinates, [margin coordinates,] every free pair's
copula ψ). Inverting it gives the standard error of each pair's τ that
accounts for the estimation of the prior, of the margins *and* of the other
pairs' copula parameters. :mod:`pmcprg.pmc._oakes` computes exactly the
diagonal entry (or, for a two-parameter family, the pair's own 2×2 block)
with every other parameter held at its fitted value; ``1/I[k,k] ≤ (I⁻¹)[k,k]``
(Schur complement), so the joint SE is never smaller than Oakes' partial one.
:mod:`pmcprg.pmc._godambe` models one channel Oakes ignores (the margins'
re-estimation) with an IFM sandwich, but not the latent-state channel; the
matrix here carries **every** channel at once, exactly.

An important caveat, measured and not assumed: ``I(θ̂)⁻¹`` is the asymptotic
variance of the **maximum-likelihood** estimator. ICE with ``fit_margins=True``
is *not* an EM algorithm for this model — its margin M-step is the γ-weighted
Gaussian MLE (:func:`pmcprg.pmc.ice._fit_margin_weighted`), which maximises
the emission part of Q while ignoring Q's copula term ``Σ ξ log c(F_i(y_n),
F_j(y_{n+1}))``, itself a function of (μ, σ). The ICE fit is therefore a
two-stage (IFM-flavoured) estimator whose sampling variance exceeds the
MLE's, and no exact information matrix can reproduce it. "ICE is not the
MLE", below, gives the diagnostic that measures the gap; the validation in
:mod:`pmcprg.tests.test_fr4_ice_lystig_hughes` reports what it costs in
coverage.

Parametrisation (the minimal free coordinates of the ICE M-step)
-----------------------------------------------------------------
θ is laid out as ``(η …, [margins …,] ψ …)``:

**Prior (η).** :func:`pmcprg.pmc.ice._m_step_prior` estimates the joint
``p_ij`` as the **symmetrised** ξ̄ (SR-PMC: p_ij = p_ji), normalised to 1;
HMC variants store ``A_ij = p_ij / π_i`` with ``π_i = Σ_j p_ij`` (the
stationary law of A, because p is symmetric). The free parameters are
therefore the ``K(K+1)/2`` distinct entries of a symmetric ``p`` summing to
1, i.e. ``K(K+1)/2 − 1`` coordinates — 2 at K = 2, not K² − 1 = 3.
Unconstrained coordinates: list the unordered pairs ``s = (i ≤ j)`` in row
order, ``q_s = p_ii`` on the diagonal and ``q_s = p_ij + p_ji = 2 p_ij`` off
it (so Σ_s q_s = 1), and take the **softmax** with the first pair (0, 0) as
reference::

    q = softmax(0, η_1, …, η_{S−1}),   η_s = log(q_s / q_00),
    p_ij = q_s · (1 if i = j else 1/2),  π_i = Σ_j p_ij,  A_ij = p_ij / π_i.

Names: ``"eta_ij"`` for the pair (i, j), i ≤ j, (i, j) ≠ (0, 0).

**Margins** (``fit_margins=True`` only). Two coordinates per state,
``(μ_k, log σ_k)`` — the same working coordinates :mod:`pmcprg.pmc._godambe`
uses, chosen so a difference step never leaves σ > 0. Names ``"mu_k"``,
``"log_sigma_k"``; the reported :attr:`LHInformation.margin_se` is in the
natural units (μ, σ), by the delta method dσ/d(log σ) = σ.

**Copulas (ψ).** One block per pair, of size 1 or 2:

* one-parameter families — ``"psi_ij"``, the working coordinate of
  :mod:`pmcprg.pmc._oakes`, ``ψ = log(τ − a) − log(b − τ)`` on the family's
  registered range [a, b] (2·atanh τ for Gauss, logit τ for Clayton). A pair
  whose τ̂ sits within Oakes' boundary pad of [a, b] is **held fixed**
  (excluded from θ, its SE is NaN — Self & Liang 1987), so that one
  non-regular coordinate does not spoil the others.
* two-parameter families — ``"psi_ij_0"``, ``"psi_ij_1"``, the two
  coordinates of :func:`pmcprg.copulas._stderr._spec_of`'s ψ: ``(log θ,
  log(δ − 1))`` for BB1, ``(atanh τ, log(ν − 2))`` for Student. The same
  ``_Spec`` supplies the map ψ → copula (``build``) and the analytic
  Jacobian ``jac_psi`` from ψ to the reported quantities (τ, δ, θ / τ, ν, ρ)
  — reused, not reimplemented, exactly as :mod:`pmcprg.pmc._oakes`'s own
  two-parameter path does. A pair whose ``_spec_of`` reports any boundary
  condition (τ̂ at a range end, δ̂ = 1, ν̂ at the fitting box) is held fixed,
  the same convention the one-parameter path applies to τ̂.

A pair held fixed still contributes to the *margin* derivatives (its copula
is part of W), it simply has no coordinate of its own in θ.

The forward recursion and its derivatives, in scaled form
----------------------------------------------------------
The package's forward pass (Devijver 1985) is, with row vectors,

    a_1 = π ∘ f(y_1),                   C_1 = 1ᵀa_1,   α̂_1 = a_1 / C_1,
    a_{n+1} = α̂_n W_n,                  C_{n+1} = 1ᵀa_{n+1},
    α̂_{n+1} = a_{n+1} / C_{n+1},        ℓ = Σ_n log C_n.

Lystig & Hughes (2002) differentiate the *unscaled* recursion, which
underflows; here the scaled one is differentiated directly. Write a
subscript r (or r, s) for ∂/∂θ_r (∂²/∂θ_r∂θ_s). Differentiating
``a_{n+1} = α̂_n W_n``::

    a_r  = α̂_r W + α̂ W_r
    a_rs = α̂_rs W + α̂_r W_s + α̂_s W_r + α̂ W_rs

then ``C = 1ᵀa``, ``C_r = 1ᵀa_r``, ``C_rs = 1ᵀa_rs``, and from ``a = C α̂``::

    α̂_r  = (a_r − C_r α̂) / C
    α̂_rs = (a_rs − C_s α̂_r − C_r α̂_s − C_rs α̂) / C

(with the *new* α̂, α̂_r on the right) and the log-likelihood increments

    ∂_r log C  = C_r / C,
    ∂²_rs log C = C_rs / C − C_r C_s / C².

Every quantity is O(1) at every step — α̂ and its derivatives are
derivatives of a probability vector, and a, a_r, a_rs are of the order of
one step's weights — so the recursion needs no log-space and no further
rescaling: the scaling constant C carries the whole scale, exactly as in
the value recursion. **The recursion itself is unchanged by both
extensions**: θ merely grows, and every new coordinate enters through the
per-step weight derivatives ``W_r``, ``W_rs`` below (and, for a margin
coordinate, through the initial step).

Initial step
-------------
``a_1 = π ∘ f(y_1)``. The prior coordinates move π (``a_{1,r} = π_r ∘ f(y_1)``,
``a_{1,rs} = π_rs ∘ f(y_1)``). With ``fit_margins=True`` a margin coordinate
of state k also moves ``f_k(y_1)``: writing ``m_k = log f_k`` and
``η = (μ_k, log σ_k)``,

    a_{1,r}[k]  = π_k f_k(y_1) · ∂_r m_k(y_1),
    a_{1,rs}[k] = π_k f_k(y_1) · (∂_r m_k ∂_s m_k + ∂²_rs m_k)   (same k),

zero for any other state and for two coordinates of two different states;
the prior × margin cross term is ``π_{k,t} f_k(y_1) ∂_r m_k(y_1)``, and the
margin × ψ cross term is zero (the copula does not enter step 1).

Weight derivatives
-------------------
``W_r = W · ∂_r log W`` and ``W_rs = W · (∂_r log W ∂_s log W + ∂²_rs log W)``,
with ``log W[n, i, j] = log A_ij + log f_j(y_{n+1}) + log c_ij(u_n, v_n)``,
``u_n = F_i(y_n)``, ``v_n = F_j(y_{n+1})``:

* **prior coordinates** — analytic: ``∂ log p_ij/∂η_t = 1{s(ij) = t} − q_t``,
  ``∂² log p_ij/∂η_t∂η_u = −q_t (1{t = u} − q_u)``, ``π_i`` and its
  derivatives summed from ``p_i·``, and ``log A_ij = log p_ij − log π_i``;
* **copula ψ_ij** — only W[·, i, j] moves. One-parameter families: ``∂ log
  c/∂ψ`` and ``∂² log c/∂ψ²`` are **central finite differences** of
  ``logpdf_array`` per observation, step ``H_PSI = 1e-4``
  (:mod:`pmcprg.copulas._stderr`/:mod:`pmcprg.pmc._oakes` convention:
  truncation O(10⁻⁸), rounding O(10⁻⁸)·|log c|) — the pilot's own three
  lines, kept verbatim. Two-parameter families: the same stencil per
  coordinate plus the standard **4-point mixed central difference** for the
  within-pair cross term ψ_0 × ψ_1, obtained from
  :func:`pmcprg.pmc._oakes._logc_derivatives` (the per-observation half of
  the function Oakes' own two-parameter path already uses for its "term 1").
  Cross terms between two *different* pairs' ψ are exactly zero (each pair's
  copula appears in one entry of W);
* **margin coordinate r = (k, c)** — this is what the extension adds, and it
  enters through **two** paths, as FR-4's note anticipated:

  1. the **emission** ``log f_j(y_{n+1})``, for j = k and every i — analytic
     for a Gaussian margin: with ``z = (y − μ_k)/σ_k``,
     ``∂m/∂μ = z/σ``, ``∂m/∂log σ = z² − 1``, ``∂²m/∂μ² = −1/σ²``,
     ``∂²m/∂μ∂log σ = −2z/σ``, ``∂²m/∂(log σ)² = −2z²``;
  2. the **copula's arguments**: state k's own CDF is the pseudo-observation
     ``u_n = F_k(y_n)`` of every pair (k, j) and the ``v_n = F_k(y_{n+1})``
     of every pair (i, k), so ``∂ log c_ij/∂r = ∂ log c/∂u · ∂F_k(y_n)/∂r``
     (+ the ∂v term when j = k; **both**, on a diagonal pair i = j = k).
     Rather than finite-differencing ∂log c/∂u and ∂F/∂r separately and
     multiplying, the composition is differenced **as a whole**: the margin
     coordinate is perturbed by ±``H_ETA``, the perturbed CDF column is
     recomputed, and ``logpdf_array`` is evaluated there. One central
     difference gives ∂/∂r and the matching second difference ∂²/∂r²; a
     4-point mixed difference gives the r × r′ and the r × ψ cross terms
     (divisor ``4·h_eta·h_psi`` for the latter). This is exact up to the
     same O(h²) truncation as every other derivative in the module, needs no
     chain-rule bookkeeping through ∂u/∂η, and — unlike differencing ∂log
     c/∂u — never has to choose a step in u near 0 or 1.

  Mixed prior × margin and prior × ψ second derivatives of log W are exactly
  zero (log A is additive and free of both); margin × ψ is *not* zero — it
  is the term that makes the matrix genuinely joint.

Cost
----
One :func:`~pmcprg.pmc.inference.precompute_weights` call, one ``_margin_cdfs``
call, and one O(N · P² · K²) forward sweep (P = number of free parameters) —
no E-step re-run at all. The ``logpdf_array`` count is 3 per free
one-parameter pair, 9 per free two-parameter pair, and — with
``fit_margins=True`` — a further O(K² · (2K)²) for the margin block's
gradient, diagonal and cross terms (about 130 at K = 2 with one-parameter
copulas). Measured at K = 2, N = 600 (ms per call):

    Gauss,   margins fixed  (P = 6)  : LH 17 · Oakes 4 pairs 47 · 1 pair 12
    Gauss,   fit_margins    (P = 10) : LH 31 · Godambe 4 pairs 51 · 1 pair 10
    BB1,     margins fixed  (P = 10) : LH 20 · Oakes multi 4 pairs 52 · 1 pair 23
    BB1,     fit_margins    (P = 14) : LH 32
    Student, margins fixed  (P = 10) : LH 31 · Oakes multi 1 pair 44

Every LH figure is for the **whole** joint matrix, which neither of the
other two returns: per pair covered, it is the cheapest of the three except
against a single Godambe pair.

ICE is not the MLE
-------------------
The ICE τ update is an exact EM step (weighted MLE of Σ ξ log c), but the
prior update p̂ = sym(ξ̄) ignores the initial-state term log π_{x_1} and the
dependence of A on π, the margin update (``fit_margins=True``) ignores the
copula's dependence on (μ, σ) entirely (see the caveat above), and ICE stops
at a tolerance: the score ``grad`` at an ICE fit is not zero.
:attr:`LHInformation.newton_step` = I⁻¹·grad is the one-step distance to the
MLE, in working coordinates — the diagnostic of how far ICE stopped from the
MLE, to be compared with the SE. With margins fixed it is a small fraction of
an SE (every component under 0.5 SE on the package's K = 2 fixture); with
margins re-estimated, the median over 300 replicates is 0.51 SE in the
margin coordinates against 0.21 SE in ψ_00 (0.40 vs 0.12 when the states are
well separated), and the median score 4.6 against 1.7 — the quantitative
statement that ICE's margin M-step is not a Q-maximiser. The information is
evaluated at the ICE fit (not at a re-optimised MLE), which is what the SEs
reported for the ICE estimate should use. Empirically that costs about 13 %
of *conservative* error in the reported SE on the overlapping-states fixture
and nothing measurable on the well-separated one
(:mod:`pmcprg.tests.test_fr4_ice_lystig_hughes`).

What extending this would need
-------------------------------
* **Other margin families** — only ``dist == "norm"`` state margins are
  covered. The emission derivatives above are the family's analytic ∂log f;
  another family needs its own (or a finite difference of ``pdf_vec``). The
  copula path (2) needs nothing new: it perturbs the margin block and
  recomputes the CDF, whatever the family.
* **Pair margins** (general PMC, ``margin_structure == "pair"``) — the
  prior derivative of log W then depends on y_n through the normaliser
  ``Σ_k p_ik f_ik(y_n)``, which no longer cancels: ``∂ log W/∂η`` gains a
  y-dependent term, and a margin coordinate additionally moves that
  normaliser. Still per-step analytic, but it is a different formula, not a
  re-parametrisation of this one.
* **Tawn / t-EV** — need a ``_Spec`` in :mod:`pmcprg.copulas._stderr`
  (working coordinates and analytic Jacobian); once registered there they
  would flow through this module's two-parameter path unchanged.
* **Gaps** (:mod:`pmcprg.pmc.gaps`) — **out of scope**, on both axes. The
  recursion would have to run on the augmented (state, grid-node) chain,
  whose transition matrices are (K·G)², and whose quadrature nodes are
  themselves functions of the margin parameters when those are free — so a
  margin coordinate would move the grid, not only the weights. Missing rows
  are refused with a ``ValueError``, as in the pilot.

References
----------
* Lystig, T. C. & Hughes, J. P. (2002). Exact computation of the observed
  information matrix for hidden Markov models. *J. Comput. Graph. Statist.*
  11(3), 678-689. doi:10.1198/106186002402
* Devijver, P. A. (1985). Baum's forward-backward algorithm revisited.
  *Pattern Recognition Letters* 3(6), 369-373.
* Oakes, D. (1999). Direct calculation of the information matrix via the EM
  algorithm. *J. R. Statist. Soc. B* 61(2), 479-482.
* Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method
  for copula-based models. *J. Multivariate Anal.* 94(2), 401-419.
* Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. *J. Amer. Statist. Assoc.* 82(398), 605-610.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.copulas._stderr import _spec_of
from pmcprg.numerics import EPS, MIN_POSITIVE, ONE_MINUS_EPS
from pmcprg.pmc._oakes import (
    _build_copula_from_blk,
    _logc_derivatives,
    _psi_of_tau,
    _tau_extra_of_copula,
    _tau_of_psi,
)
from pmcprg.pmc.ice import _margin_cdfs, _pair_pseudo_obs, _resolve_candidate
from pmcprg.pmc.inference import precompute_weights
from pmcprg.pmc.model import PMCModel, Variant

logger = logging.getLogger(__name__)

__all__ = ["LHInformation", "ice_lh_information", "lh_loglik_derivatives"]

#: Central-difference step in the copula working coordinate ψ — same value as
#: :data:`pmcprg.pmc._oakes.H_PSI` and :data:`pmcprg.copulas._stderr._H_PSI`.
H_PSI: float = 1e-4

#: Central-difference step in a margin working coordinate (μ, log σ) — same
#: value and role as :data:`pmcprg.pmc._godambe.H_ETA`.
H_ETA: float = 1e-4

#: One-parameter copula families — the original pilot's scalar code path,
#: kept verbatim (its numbers are bit-identical to the pilot's).
_ONE_PARAM_FAMILIES: tuple[str, ...] = ("Gauss", "Clayton")

#: Two-parameter copula families — two θ coordinates per pair, via
#: :func:`pmcprg.copulas._stderr._spec_of` (module docstring). Tawn and t-EV
#: are excluded: ``_spec_of`` has no ``_Spec`` for them.
_TWO_PARAM_FAMILIES: tuple[str, ...] = ("BB1", "Student")

#: Every copula family this module supports (audit FR-4 scope).
SUPPORTED_FAMILIES: tuple[str, ...] = _ONE_PARAM_FAMILIES + _TWO_PARAM_FAMILIES

#: Margin working coordinates of a Gaussian state margin, in θ order.
MARGIN_COMPONENTS: tuple[str, ...] = ("mu", "log_sigma")

#: Boundary pad, in optimiser pads — :data:`pmcprg.pmc._oakes._BOUNDARY_PADS`.
_BOUNDARY_PADS: float = 2.0

#: Largest tolerated asymmetry |p_ij − p_ji| of the model's joint prior.
_SYM_TOL: float = 1e-6

#: No perturbation of a margin block, in the ``(Δμ, Δlog σ)`` sign convention
#: of :func:`_cdf_column`.
_NO_SHIFT: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class LHInformation:
    """Lystig–Hughes observed information of an ICE fit (module docstring).

    Fields
    ------
    names       : θ coordinates, in matrix order — ``"eta_ij"`` (prior,
                  softmax coordinates), then, with ``fit_margins``,
                  ``"mu_k"``/``"log_sigma_k"`` per state, then the free
                  copula pairs' ``"psi_ij"`` (one-parameter families) or
                  ``"psi_ij_0"``/``"psi_ij_1"`` (BB1, Student).
    theta       : θ at the model handed in.
    log_lik     : ℓ(θ), equal to :func:`~pmcprg.pmc.inference.forward`'s value.
    grad        : ∂ℓ/∂θ (exact recursion; copula-density and margin/copula
                  chain-rule derivatives by central differences).
    hessian     : ∂²ℓ/∂θ∂θᵀ, same accuracy.
    info        : ``−hessian``, the observed information.
    cov         : ``info⁻¹`` (NaN if ``info`` is not positive-definite).
    newton_step : ``info⁻¹ · grad`` — one Newton step from the ICE fit
                  towards the MLE, in working coordinates (NaN as ``cov``).
    tau_hat     : ``{(i, j): τ̂}`` for every copula pair.
    se_tau      : ``{(i, j): SE}`` of τ̂ from the full ``cov`` (delta method).
    se_tau_partial : ``{(i, j): SE}`` of τ̂ from the pair's **own** diagonal
                  block of ``info`` alone — the other parameters held fixed;
                  equals Oakes' SE (:func:`pmcprg.pmc._oakes.ice_oakes_tau_se`)
                  up to its finite-difference error, for one- and
                  two-parameter families alike.
    fixed_pairs : pairs held fixed at a boundary estimate (SE NaN).
    fit_margins : whether the margin coordinates are part of θ.
    pair_names  : ``{(i, j): names}`` — the quantities reported for that pair
                  (``("tau_k",)`` for a one-parameter family; ``_spec_of``'s
                  ``("tau_k", "delta", "theta")`` / ``("tau_k", "df", "rho")``
                  for BB1 / Student).
    estimate    : ``{(i, j): {name: value}}`` at the ICE fit.
    se          : ``{(i, j): {name: SE}}`` from the full ``cov``.
    se_partial  : ``{(i, j): {name: SE}}`` from the pair's own block alone.
    margin_names: ``("mu_0", "sigma_0", …)`` — empty unless ``fit_margins``.
    margin_se   : ``{name: SE}`` of each of ``margin_names``, in natural
                  units (σ, not log σ), from the same joint ``cov``.
    """

    names: tuple[str, ...]
    theta: np.ndarray
    log_lik: float
    grad: np.ndarray
    hessian: np.ndarray
    info: np.ndarray
    cov: np.ndarray
    newton_step: np.ndarray
    tau_hat: dict
    se_tau: dict
    se_tau_partial: dict
    fixed_pairs: tuple
    fit_margins: bool = False
    pair_names: dict = field(default_factory=dict)
    estimate: dict = field(default_factory=dict)
    se: dict = field(default_factory=dict)
    se_partial: dict = field(default_factory=dict)
    margin_names: tuple[str, ...] = ()
    margin_se: dict = field(default_factory=dict)

    def ci(self, pair: tuple[int, int], level: float = 0.95,
           name: str = "tau_k") -> tuple[float, float]:
        """Wald interval for pair ``pair``'s ``name``, from :attr:`se`.

        ``name`` defaults to ``"tau_k"``, the only quantity a one-parameter
        family reports; BB1 also reports ``"delta"``/``"theta"`` and Student
        ``"df"``/``"rho"`` (:attr:`pair_names`).
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}.")
        per_pair = self.se.get(pair, {})
        if name not in per_pair:
            raise KeyError(f"{name!r} is not one of {self.pair_names.get(pair, ())}.")
        se = per_pair[name]
        if not np.isfinite(se):
            return float("nan"), float("nan")
        z = float(norm.ppf(0.5 + 0.5 * level))
        est = self.estimate[pair][name]
        return est - z * se, est + z * se


# ---------------------------------------------------------------------------
# Prior parametrisation
# ---------------------------------------------------------------------------

def _unique_pairs(K: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(K) for j in range(i, K)]


def _eta_of_model(model: PMCModel) -> np.ndarray:
    p = np.asarray(model.prior_p, dtype=float)
    asym = float(np.max(np.abs(p - p.T)))
    if asym > _SYM_TOL:
        raise ValueError(
            f"ice_lh_information: the joint prior p is not symmetric "
            f"(max |p_ij − p_ji| = {asym:.3g}); the ICE M-step's "
            "parametrisation (module docstring) needs SR-PMC symmetry."
        )
    p = 0.5 * (p + p.T)
    q = np.array([p[i, j] if i == j else 2.0 * p[i, j] for i, j in _unique_pairs(model.K)])
    if np.any(q <= 0.0):
        raise ValueError("ice_lh_information: a prior probability is 0 — no interior η.")
    return np.log(q[1:] / q[0])


def _prior_from_eta(eta: np.ndarray, K: int):
    """``(p, π, dlogA, d2logA, dπ, d2π)`` at η — see the module docstring.

    Shapes: p (K, K); π (K,); dlogA (T, K, K); d2logA (T, T, K, K);
    dπ (T, K); d2π (T, T, K) with T = len(η) (derivatives of π itself,
    not of log π — the initial step multiplies π).
    """
    pairs = _unique_pairs(K)
    S = len(pairs)
    z = np.concatenate(([0.0], np.asarray(eta, dtype=float)))
    z = z - z.max()
    q = np.exp(z)
    q /= q.sum()
    T = S - 1
    p = np.empty((K, K))
    # dlogp[t, i, j], d2logp[t, u, i, j]
    dlogp = np.empty((T, K, K))
    d2logp = np.empty((T, T, K, K))
    qt = q[1:]
    d2_common = -(np.diag(qt) - np.outer(qt, qt))      # −∂q_t/∂η_u
    for s, (i, j) in enumerate(pairs):
        val = q[s] if i == j else 0.5 * q[s]
        ind = np.zeros(T)
        if s > 0:
            ind[s - 1] = 1.0
        for (a, b) in {(i, j), (j, i)}:
            p[a, b] = val
            dlogp[:, a, b] = ind - qt
            d2logp[:, :, a, b] = d2_common
    pi = p.sum(axis=1)
    dp = p[None, :, :] * dlogp                           # (T, K, K)
    d2p = p[None, None, :, :] * (dlogp[:, None] * dlogp[None, :] + d2logp)
    dpi = dp.sum(axis=2)                                 # (T, K)
    d2pi = d2p.sum(axis=3)                               # (T, T, K)
    dlogpi = dpi / pi[None, :]
    d2logpi = d2pi / pi[None, None, :] - dlogpi[:, None, :] * dlogpi[None, :, :]
    dlogA = dlogp - dlogpi[:, :, None]
    d2logA = d2logp - d2logpi[:, :, :, None]
    return p, pi, dlogA, d2logA, dpi, d2pi


def _check_scope(model: PMCModel) -> None:
    if model.variant not in (Variant.HMC_DN, Variant.PMC):
        raise NotImplementedError(
            f"ice_lh_information: variant {model.variant.value} is out of the "
            "FR-4 Lystig–Hughes scope (HMC-DN, PMC with state margins)."
        )
    if model.margin_structure != "state":
        raise NotImplementedError(
            "ice_lh_information: pair margins are out of scope "
            "(module docstring, 'What extending this would need')."
        )


# ---------------------------------------------------------------------------
# Copula blocks: working coordinates, one or two per pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Pair:
    """One ``[[copulas]]`` block's working coordinates and reported quantities.

    ``one_param`` selects the code path: the pilot's verbatim scalar stencil
    (Gauss, Clayton) or :func:`pmcprg.pmc._oakes._logc_derivatives` on
    :func:`pmcprg.copulas._stderr._spec_of`'s ψ (BB1, Student).
    """

    pair: tuple[int, int]
    family: str
    cls: type
    one_param: bool
    tau: float
    tau_min: float          # one-parameter families only (NaN otherwise)
    tau_max: float
    psi: np.ndarray         # working coordinate; empty when ``fixed``
    build: Callable         # ψ vector → copula instance
    copula0: object         # the copula as the raw block stores it
    spec: object            # ``_Spec`` (two-parameter families) or None
    names: tuple
    estimate: dict
    jac_psi: np.ndarray     # (m, p) d(names)/dψ
    fixed: bool
    boundary: tuple


def _pair_info(model: PMCModel) -> list[_Pair]:
    """One :class:`_Pair` per copula block, in block order (i, j)."""
    out: list[_Pair] = []
    blocks = sorted(model.copula_blocks(), key=lambda b: (int(b["i"]), int(b["j"])))
    for blk in blocks:
        family = str(blk["name"])
        pair = (int(blk["i"]), int(blk["j"]))
        if family not in SUPPORTED_FAMILIES:
            raise NotImplementedError(
                f"ice_lh_information: family {family!r} for pair {pair} is not "
                f"one of {SUPPORTED_FAMILIES} (audit FR-4 Lystig–Hughes scope; "
                "Tawn and t-EV have no _spec_of working coordinate — see the "
                "module docstring)."
            )
        entry, cls = _resolve_candidate(family)
        tau = float(blk["tau"])
        copula0 = _build_copula_from_blk(cls, family, blk)
        if family in _ONE_PARAM_FAMILIES:
            a, b = (float(x) for x in entry.value.TAU_MIN_MAX)
            pad = _BOUNDARY_PADS * max(TAU_PAD_REL * (b - a), TAU_PAD_ABS)
            fixed = not (a + pad < tau < b - pad)
            psi = np.empty(0) if fixed else np.array([_psi_of_tau(tau, a, b)])

            def build(vec, _cls=cls, _a=a, _b=b):
                return _cls(tau_k=_tau_of_psi(float(vec[0]), _a, _b))

            dtau_dpsi = (tau - a) * (b - tau) / (b - a)
            out.append(_Pair(
                pair, family, cls, True, tau, a, b, psi, build, copula0, None,
                ("tau_k",), {"tau_k": tau}, np.array([[dtau_dpsi]]), fixed, (),
            ))
        else:
            spec = _spec_of(copula0)
            psi = np.asarray(spec.psi, dtype=float)
            fixed = bool(spec.boundary) or not bool(np.all(np.isfinite(psi)))
            out.append(_Pair(
                pair, family, cls, False, tau, math.nan, math.nan,
                np.empty(0) if fixed else psi, spec.build, copula0, spec,
                tuple(spec.names), dict(spec.estimate),
                np.asarray(spec.jac_psi, dtype=float), fixed, tuple(spec.boundary),
            ))
    return out


def _theta_layout(model: PMCModel, fit_margins: bool):
    """``(names, pairs, margin_index, psi_index)`` — θ's coordinate layout.

    ``margin_index[(k, c)]`` and ``psi_index[(i, j)]`` give the θ index (or
    list of indices) of each block; ``margin_index`` is empty unless
    ``fit_margins``.
    """
    names = [f"eta_{i}{j}" for i, j in _unique_pairs(model.K)[1:]]
    margin_index: dict[tuple[int, int], int] = {}
    if fit_margins:
        for k in range(model.K):
            for c, comp in enumerate(MARGIN_COMPONENTS):
                margin_index[(k, c)] = len(names)
                names.append(f"{comp}_{k}")
    pairs = _pair_info(model)
    psi_index: dict[tuple[int, int], list[int]] = {}
    for pr in pairs:
        if pr.fixed:
            continue
        start = len(names)
        ii, jj = pr.pair
        if pr.one_param:
            names.append(f"psi_{ii}{jj}")
        else:
            names += [f"psi_{ii}{jj}_{c}" for c in range(pr.psi.size)]
        psi_index[pr.pair] = list(range(start, len(names)))
    return names, pairs, margin_index, psi_index


# ---------------------------------------------------------------------------
# Gaussian state margins: parameters, log-density derivatives, perturbed CDFs
# ---------------------------------------------------------------------------

def _gaussian_margin_params(model: PMCModel) -> list[tuple[float, float]]:
    """``[(μ_k, σ_k)]`` per state, refusing a non-Gaussian margin block."""
    blocks = {int(m["i"]): m for m in model.margin_blocks()}
    out = []
    for k in range(model.K):
        blk = blocks.get(k)
        if blk is None or blk.get("dist") != "norm":
            raise NotImplementedError(
                f"ice_lh_information: fit_margins=True needs a Gaussian "
                f"('dist' == 'norm') margin for every state; state {k} has "
                f"{None if blk is None else blk.get('dist')!r} (audit FR-4 "
                "scope, the case pmcprg.pmc._godambe piloted)."
            )
        out.append((float(blk["params"]["loc"]), float(blk["params"]["scale"])))
    return out


def _log_f_derivatives(y: np.ndarray, mu: float, sigma: float):
    """``(d1 (2, n), d2 (2, 2, n))`` of ``log f(y; μ, σ)`` in (μ, log σ)."""
    z = (y - mu) / sigma
    d1 = np.empty((2, y.size))
    d1[0] = z / sigma
    d1[1] = z * z - 1.0
    d2 = np.empty((2, 2, y.size))
    d2[0, 0] = -1.0 / (sigma * sigma)
    d2[0, 1] = d2[1, 0] = -2.0 * z / sigma
    d2[1, 1] = -2.0 * z * z
    return d1, d2


def _cdf_column(Y: np.ndarray, mu: float, sigma: float,
                shift: tuple[int, int], h_eta: float) -> np.ndarray:
    """``F_k(Y)`` with the margin shifted by ``shift`` steps of ``h_eta`` in
    (μ, log σ), clipped exactly as :func:`pmcprg.pmc.ice._margin_cdfs` does.

    ``shift == (0, 0)`` reproduces the unperturbed column bit for bit.
    """
    m = mu + shift[0] * h_eta
    s = sigma * math.exp(shift[1] * h_eta)
    return np.clip(norm.cdf(Y, loc=m, scale=s), EPS, ONE_MINUS_EPS)


def _unit_shift(comp: int, sign: int) -> tuple[int, int]:
    return (sign, 0) if comp == 0 else (0, sign)


def _combine_shift(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _add_margin_derivatives(
    model: PMCModel,
    Y: np.ndarray,
    pairs: list[_Pair],
    margin_index: dict,
    psi_index: dict,
    dlogW: np.ndarray,
    d2logW: np.ndarray,
    *,
    h_psi: float,
    h_eta: float,
):
    """Add the margin coordinates' contributions to ``∂log W`` / ``∂²log W``.

    Both paths of the module docstring's "margin coordinate r = (k, c)":
    the analytic emission derivatives of ``log f_k(y_{n+1})``, and the
    copula's dependence on state k's CDF, differenced as a whole by
    recomputing the perturbed pseudo-observations.

    Returns ``(g1, g2)`` — ``∂ log f_k(y_1)/∂r`` (K, 2) and its second
    derivative (K, 2, 2) — which the recursion's initial step needs.
    """
    K = model.K
    params = _gaussian_margin_params(model)
    d1 = np.empty((K, 2, Y.size))
    d2 = np.empty((K, 2, 2, Y.size))
    for k, (mu, sigma) in enumerate(params):
        d1[k], d2[k] = _log_f_derivatives(Y, mu, sigma)

    # ── path 1: the emission log f_j(y_{n+1}), j = k, every i ────────────
    for k in range(K):
        for c in range(2):
            r = margin_index[(k, c)]
            dlogW[:, r, :, k] += d1[k, c, 1:][:, None]
            for c2 in range(2):
                s = margin_index[(k, c2)]
                d2logW[:, r, s, :, k] += d2[k, c, c2, 1:][:, None]

    # ── path 2: the copula's pseudo-observations ─────────────────────────
    # Every shift of a single margin block that any stencil below needs.
    cols: dict[tuple[int, tuple[int, int]], np.ndarray] = {}
    for k, (mu, sigma) in enumerate(params):
        for s0 in (-1, 0, 1):
            for s1 in (-1, 0, 1):
                cols[(k, (s0, s1))] = _cdf_column(Y, mu, sigma, (s0, s1), h_eta)

    coords = sorted(margin_index)                    # [(k, c), …] in θ order
    for pr in pairs:
        ii, jj = pr.pair
        touching = [(k, c) for (k, c) in coords if k in (ii, jj)]
        if not touching:
            continue
        cache: dict = {}

        def value(cfg: dict, cop, key, _ii=ii, _jj=jj, _cache=cache):
            """``log c_ij`` at the margin configuration ``cfg`` (state →
            shift) and the copula ``cop`` (identified by ``key``)."""
            sh = (cfg.get(_ii, _NO_SHIFT), cfg.get(_jj, _NO_SHIFT), key)
            if sh not in _cache:
                uv = np.column_stack((cols[(_ii, sh[0])][:-1], cols[(_jj, sh[1])][1:]))
                _cache[sh] = np.asarray(cop.logpdf_array(uv), dtype=float)
            return _cache[sh]

        cop0 = pr.copula0
        l0 = value({}, cop0, None)
        psi_cols = psi_index.get(pr.pair, [])
        psi_hat = pr.psi

        # gradient and the pure-margin diagonal
        for (k, c) in touching:
            r = margin_index[(k, c)]
            lp = value({k: _unit_shift(c, +1)}, cop0, None)
            lm = value({k: _unit_shift(c, -1)}, cop0, None)
            dlogW[:, r, ii, jj] += (lp - lm) / (2.0 * h_eta)
            d2logW[:, r, r, ii, jj] += (lp - 2.0 * l0 + lm) / (h_eta * h_eta)

        # margin × margin cross terms (4-point mixed central difference)
        for a, (k, c) in enumerate(touching):
            for (k2, c2) in touching[a + 1:]:
                r, s = margin_index[(k, c)], margin_index[(k2, c2)]
                acc = np.zeros_like(l0)
                for sr in (+1, -1):
                    for ss in (+1, -1):
                        if k == k2:
                            cfg = {k: _combine_shift(_unit_shift(c, sr),
                                                     _unit_shift(c2, ss))}
                        else:
                            cfg = {k: _unit_shift(c, sr), k2: _unit_shift(c2, ss)}
                        acc += (sr * ss) * value(cfg, cop0, None)
                acc /= 4.0 * h_eta * h_eta
                d2logW[:, r, s, ii, jj] += acc
                d2logW[:, s, r, ii, jj] += acc

        # margin × ψ cross terms — the genuinely joint block
        for a, ra in enumerate(psi_cols):
            cop_shift = {}
            for sa in (+1, -1):
                vec = psi_hat.copy()
                vec[a] += sa * h_psi
                cop_shift[sa] = pr.build(vec)
            for (k, c) in touching:
                r = margin_index[(k, c)]
                acc = np.zeros_like(l0)
                for sr in (+1, -1):
                    for sa in (+1, -1):
                        acc += (sr * sa) * value({k: _unit_shift(c, sr)},
                                                 cop_shift[sa], (a, sa))
                acc /= 4.0 * h_eta * h_psi
                d2logW[:, r, ra, ii, jj] += acc
                d2logW[:, ra, r, ii, jj] += acc

    return d1[:, :, 0].copy(), d2[:, :, :, 0].copy()


# ---------------------------------------------------------------------------
# θ → model
# ---------------------------------------------------------------------------

def model_from_theta(model: PMCModel, theta: np.ndarray, *,
                     fit_margins: bool = False) -> PMCModel:
    """A copy of ``model`` with θ (in :func:`ice_lh_information`'s order) set.

    Pairs held fixed at a boundary keep their parameters. Used by the
    finite-difference checks; the prior is written as ``A`` (HMC-DN) or ``p``
    (PMC), the margins as ``loc``/``scale``, and a two-parameter copula's
    block gets both ``tau`` and its extra key (``delta``, ``df``).
    """
    _check_scope(model)
    names, pairs, margin_index, psi_index = _theta_layout(model, fit_margins)
    K = model.K
    T = len(_unique_pairs(K)) - 1
    theta = np.asarray(theta, dtype=float)
    if theta.size != len(names):
        raise ValueError(
            f"model_from_theta: θ has {theta.size} coordinates, the layout "
            f"needs {len(names)} ({names})."
        )
    p, pi, *_ = _prior_from_eta(theta[:T], K)
    raw = model.raw
    if model.variant.has_markov_prior:
        raw["prior"] = {k: v for k, v in raw["prior"].items() if k != "p"}
        raw["prior"]["A"] = (p / pi[:, None]).tolist()
    else:
        raw["prior"] = {k: v for k, v in raw["prior"].items() if k != "A"}
        raw["prior"]["p"] = p.tolist()
    if fit_margins:
        blocks = {int(m["i"]): m for m in raw.get("margins", [])}
        for k in range(K):
            blk = blocks[k]
            blk["params"] = {
                "loc": float(theta[margin_index[(k, 0)]]),
                "scale": float(math.exp(theta[margin_index[(k, 1)]])),
            }
    new_params = {}
    for pr in pairs:
        if pr.fixed:
            continue
        vec = theta[psi_index[pr.pair]]
        cop = pr.build(vec)
        new_params[pr.pair] = _tau_extra_of_copula(cop, pr.family)
    for blk in raw["copulas"]:
        pair = (int(blk["i"]), int(blk["j"]))
        if pair in new_params:
            tau, extra = new_params[pair]
            blk["tau"] = float(tau)
            for key, val in extra.items():
                blk[key] = float(val)
    return PMCModel.from_dict(raw)


def _theta_of_model(model: PMCModel, pairs: list[_Pair], margin_index: dict,
                    psi_index: dict, fit_margins: bool, n: int) -> np.ndarray:
    theta = np.empty(n)
    eta = _eta_of_model(model)
    theta[: eta.size] = eta
    if fit_margins:
        for k, (mu, sigma) in enumerate(_gaussian_margin_params(model)):
            theta[margin_index[(k, 0)]] = mu
            theta[margin_index[(k, 1)]] = math.log(sigma)
    for pr in pairs:
        if not pr.fixed:
            theta[psi_index[pr.pair]] = pr.psi
    return theta


# ---------------------------------------------------------------------------
# The scaled derivative recursion
# ---------------------------------------------------------------------------

def lh_loglik_derivatives(
    model: PMCModel, Y: np.ndarray, *, h_psi: float = H_PSI,
    fit_margins: bool = False, h_eta: float = H_ETA,
) -> tuple[float, np.ndarray, np.ndarray, tuple[str, ...]]:
    """``(ℓ, ∂ℓ/∂θ, ∂²ℓ/∂θ∂θᵀ, names)`` at ``model`` — the recursion of the
    module docstring. θ as in :func:`ice_lh_information`."""
    _check_scope(model)
    Y = np.asarray(Y, dtype=float)
    if np.any(~np.isfinite(Y)):
        raise ValueError(
            "ice_lh_information: Y has missing (non-finite) rows — gaps are "
            "out of the Lystig–Hughes scope (module docstring)."
        )
    K = model.K
    names, pairs, margin_index, psi_index = _theta_layout(model, fit_margins)
    free = [pr for pr in pairs if not pr.fixed]
    T = len(_unique_pairs(K)) - 1
    P = len(names)
    eta = _eta_of_model(model)
    _, pi, dlogA, d2logA, dpi, d2pi = _prior_from_eta(eta, K)

    W, f_pdf = precompute_weights(model, Y)
    Nm1 = W.shape[0]

    dlogW = np.zeros((Nm1, P, K, K))
    d2logW = np.zeros((Nm1, P, P, K, K))
    dlogW[:, :T] = dlogA[None]
    d2logW[:, :T, :T] = d2logA[None]

    f_cdf = _margin_cdfs(model, Y)
    for pr in free:
        ii, jj = pr.pair
        u, v = _pair_pseudo_obs(model, f_cdf, ii, jj)
        uv = np.column_stack((u, v))
        idx = psi_index[pr.pair]
        if pr.one_param:
            # The pilot's stencil, verbatim: same three evaluations, same
            # expressions, so a one-parameter margins-fixed model reproduces
            # the pre-extension numbers bit for bit.
            a, b, tau, cls = pr.tau_min, pr.tau_max, pr.tau, pr.cls
            psi = _psi_of_tau(tau, a, b)
            l0 = np.asarray(cls(tau_k=tau).logpdf_array(uv), dtype=float)
            lp = np.asarray(cls(tau_k=_tau_of_psi(psi + h_psi, a, b)).logpdf_array(uv), dtype=float)
            lm = np.asarray(cls(tau_k=_tau_of_psi(psi - h_psi, a, b)).logpdf_array(uv), dtype=float)
            r = idx[0]
            dlogW[:, r, ii, jj] = (lp - lm) / (2.0 * h_psi)
            d2logW[:, r, r, ii, jj] = (lp - 2.0 * l0 + lm) / (h_psi * h_psi)
        else:
            phi, curv = _logc_derivatives(pr.spec, uv, h_psi)
            for a_, ra in enumerate(idx):
                dlogW[:, ra, ii, jj] = phi[:, a_]
                for c_, rc in enumerate(idx):
                    d2logW[:, ra, rc, ii, jj] = curv[:, a_, c_]

    if fit_margins:
        g1, g2 = _add_margin_derivatives(
            model, Y, pairs, margin_index, psi_index, dlogW, d2logW,
            h_psi=h_psi, h_eta=h_eta,
        )
    else:
        g1 = g2 = None

    live = W > 0.0
    Wb = W[:, None, :, :]
    dW = np.where(live[:, None], Wb * dlogW, 0.0)
    d2W = np.where(live[:, None, None],
                   Wb[:, None] * (dlogW[:, :, None] * dlogW[:, None, :] + d2logW), 0.0)
    if not (np.all(np.isfinite(dW)) and np.all(np.isfinite(d2W))):
        raise FloatingPointError(
            "ice_lh_information: non-finite copula or margin log-density "
            "derivative at a positive transition weight."
        )

    # Initial step: a_1 = π ∘ f(y_1) (state margins: f_pdf[0, j, ·] = f_j(y_1)).
    f1 = f_pdf[0, :, 0]
    a = pi * f1
    a_r = np.zeros((P, K))
    a_rs = np.zeros((P, P, K))
    a_r[:T] = dpi * f1[None]
    a_rs[:T, :T] = d2pi * f1[None, None]
    if fit_margins:
        for k in range(K):
            base = pi[k] * f1[k]
            for c in range(2):
                r = margin_index[(k, c)]
                a_r[r, k] = base * g1[k, c]
                a_rs[:T, r, k] = dpi[:, k] * f1[k] * g1[k, c]
                a_rs[r, :T, k] = a_rs[:T, r, k]
                for c2 in range(2):
                    s = margin_index[(k, c2)]
                    a_rs[r, s, k] = base * (g1[k, c] * g1[k, c2] + g2[k, c, c2])

    ll = 0.0
    g = np.zeros(P)
    H = np.zeros((P, P))
    n = 0
    while True:
        # Normalise step n: (a, a_r, a_rs) → (α̂, α̂_r, α̂_rs), accumulate ℓ.
        C = float(a.sum())
        if not (np.isfinite(C) and C >= MIN_POSITIVE):
            raise FloatingPointError(
                f"ice_lh_information: normaliser C={C!r} underflows at step {n} "
                "— the linear-scale derivative recursion cannot proceed."
            )
        C_r = a_r.sum(axis=1)
        C_rs = a_rs.sum(axis=2)
        ll += math.log(C)
        g += C_r / C
        H += C_rs / C - np.outer(C_r, C_r) / (C * C)
        al = a / C
        al_r = (a_r - C_r[:, None] * al[None]) / C
        al_rs = (a_rs - C_r[None, :, None] * al_r[:, None, :]
                 - C_r[:, None, None] * al_r[None, :, :]
                 - C_rs[:, :, None] * al[None, None, :]) / C
        if n == Nm1:
            break
        # Propagate through W_n and its derivatives.
        Wn, dWn, d2Wn = W[n], dW[n], d2W[n]
        cross = np.einsum("ri,sij->rsj", al_r, dWn)
        a = al @ Wn
        a_r = al_r @ Wn + np.einsum("i,rij->rj", al, dWn)
        a_rs = (al_rs @ Wn + cross + cross.transpose(1, 0, 2)
                + np.einsum("i,rsij->rsj", al, d2Wn))
        n += 1
    H = 0.5 * (H + H.T)
    return ll, g, H, tuple(names)


def ice_lh_information(
    model: PMCModel, Y: np.ndarray, *, h_psi: float = H_PSI,
    fit_margins: bool = False, h_eta: float = H_ETA,
) -> LHInformation:
    """Lystig & Hughes' (2002) full observed information of an ICE fit.

    Parameters
    ----------
    model : the fitted model (:func:`pmcprg.pmc.ice.ice`; HMC-DN or
            state-margin PMC; Gauss, Clayton, BB1 or Student copulas;
            symmetric prior).
    Y     : the complete observation sequence it was fitted on.
    h_psi : central-difference step in a copula working coordinate ψ.
    fit_margins : ``True`` when the ICE run re-estimated the margins, so that
            their ``2K`` coordinates (Gaussian state margins only) join θ —
            the case :mod:`pmcprg.pmc._godambe` piloted. ``False`` (the
            default) reproduces the original margins-fixed pilot exactly.
    h_eta : central-difference step in a margin working coordinate (μ, log σ).

    Returns
    -------
    :class:`LHInformation`.

    Raises
    ------
    NotImplementedError : variant, margin structure or copula family out of
        scope, or ``fit_margins=True`` with a non-Gaussian margin (module
        docstring).
    ValueError : missing rows, or a non-symmetric / degenerate prior.
    """
    if not model.variant.uses_copula:
        raise ValueError(f"Variant {model.variant.value} does not use copulas.")
    _check_scope(model)
    ll, g, H, names = lh_loglik_derivatives(
        model, Y, h_psi=h_psi, fit_margins=fit_margins, h_eta=h_eta,
    )
    K = model.K
    _, pairs, margin_index, psi_index = _theta_layout(model, fit_margins)
    free = [pr for pr in pairs if not pr.fixed]
    P = len(names)
    theta = _theta_of_model(model, pairs, margin_index, psi_index, fit_margins, P)
    info_m = -H
    try:
        eig = np.linalg.eigvalsh(info_m)
        pos_def = bool(np.all(eig > 0.0))
    except np.linalg.LinAlgError:
        pos_def = False
        eig = None
    if pos_def:
        cov = np.linalg.inv(info_m)
        cov = 0.5 * (cov + cov.T)
        step = cov @ g
    else:
        logger.warning(
            "ice_lh_information: observed information is not positive-definite "
            "(eigenvalues %s) — returning NaN standard errors.", eig,
        )
        cov = np.full((P, P), math.nan)
        step = np.full(P, math.nan)

    tau_hat = {pr.pair: pr.tau for pr in pairs}
    se_full = {pr.pair: math.nan for pr in pairs}
    se_part = {pr.pair: math.nan for pr in pairs}
    pair_names = {pr.pair: pr.names for pr in pairs}
    estimate = {pr.pair: dict(pr.estimate) for pr in pairs}
    se_rep = {pr.pair: {nm: math.nan for nm in pr.names} for pr in pairs}
    se_rep_part = {pr.pair: {nm: math.nan for nm in pr.names} for pr in pairs}
    for pr in free:
        idx = psi_index[pr.pair]
        if pr.one_param:
            # The pilot's scalar formulas, verbatim (bit-identical SEs).
            r = idx[0]
            a, b, tau = pr.tau_min, pr.tau_max, pr.tau
            dtau = (tau - a) * (b - tau) / (b - a)
            if pos_def and cov[r, r] > 0.0:
                se_full[pr.pair] = abs(dtau) * math.sqrt(cov[r, r])
            if info_m[r, r] > 0.0:
                se_part[pr.pair] = abs(dtau) / math.sqrt(info_m[r, r])
            se_rep[pr.pair]["tau_k"] = se_full[pr.pair]
            se_rep_part[pr.pair]["tau_k"] = se_part[pr.pair]
            continue
        # Two-parameter family: the pair's own block, carried to the reported
        # quantities by _spec_of's analytic Jacobian (as _oakes does).
        block = info_m[np.ix_(idx, idx)]
        if pos_def:
            cov_rep = pr.jac_psi @ cov[np.ix_(idx, idx)] @ pr.jac_psi.T
            for m, nm in enumerate(pr.names):
                if cov_rep[m, m] >= 0.0:
                    se_rep[pr.pair][nm] = float(math.sqrt(cov_rep[m, m]))
        if bool(np.all(np.linalg.eigvalsh(block) > 0.0)):
            cov_part = pr.jac_psi @ np.linalg.inv(block) @ pr.jac_psi.T
            for m, nm in enumerate(pr.names):
                if cov_part[m, m] >= 0.0:
                    se_rep_part[pr.pair][nm] = float(math.sqrt(cov_part[m, m]))
        se_full[pr.pair] = se_rep[pr.pair]["tau_k"]
        se_part[pr.pair] = se_rep_part[pr.pair]["tau_k"]

    margin_names: tuple[str, ...] = ()
    margin_se: dict = {}
    if fit_margins:
        margin_names = tuple(x for k in range(K) for x in (f"mu_{k}", f"sigma_{k}"))
        sigmas = [s for _, s in _gaussian_margin_params(model)]
        for k in range(K):
            for c, nm in enumerate((f"mu_{k}", f"sigma_{k}")):
                r = margin_index[(k, c)]
                var = cov[r, r] if pos_def else math.nan
                scale = 1.0 if c == 0 else sigmas[k]
                margin_se[nm] = (scale * math.sqrt(var)
                                 if np.isfinite(var) and var >= 0.0 else math.nan)

    fixed = tuple(pr.pair for pr in pairs if pr.fixed)
    if fixed:
        logger.warning(
            "ice_lh_information: pair(s) %s at a boundary of their parameter "
            "range are held fixed (Self & Liang 1987); their SE is NaN.", fixed,
        )
    return LHInformation(
        names=names, theta=theta, log_lik=ll, grad=g, hessian=H, info=info_m,
        cov=cov, newton_step=step, tau_hat=tau_hat, se_tau=se_full,
        se_tau_partial=se_part, fixed_pairs=fixed, fit_margins=fit_margins,
        pair_names=pair_names, estimate=estimate, se=se_rep,
        se_partial=se_rep_part, margin_names=margin_names, margin_se=margin_se,
    )
