"""
pmcprg.copulas._fit — fitting, goodness-of-fit, and diagnostic helpers.

Contents
--------
_weighted_log_density_sum
    Σ w_i log c_i with the conventions shared by every likelihood objective
    of the package (zero weights ignored, non-finite ⇒ −∞; audit RB-7).
_eval_log_likelihood, _empirical_copula, _cvm_statistic, _empirical_tail_dep
    Pseudo-observation utilities (used by both `CopulaVirt.fit` and
    `FitResult.gof_test`).
_fit_two_parameter_mle, ParameterFit
    Joint (weighted) MLE of a multi-parameter family — the single optimiser
    behind ``CopulaVirt.fit(method='mle')`` and the ICE M-step (RB-3, RB-8).
    The name is historical: the driver was always written for a parameter
    vector of arbitrary length, and FR-9 round 5 (the three-parameter Tawn)
    used it unchanged, adding only a coordinate branch in
    ``_multi_extra_spec``.
GoFResult
    Cramér-von Mises GoF test result.
FitResult
    `CopulaVirt.fit()` return value: τ̂, log-likelihood, AIC/BIC/AICc/HQC,
    GoF test, bootstrap CI, K-fold CV, and diagnostic plots.
FitBestResults
    `CopulaVirt.fit_best()` return value: the ranked fits, plus the fits that
    are not comparable and the families whose fit failed (RB-10).

These were originally in ``_base.py``; extracted to keep the base module focused
on the copula class hierarchy.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from pmcprg.numerics   import MIN_POSITIVE
from pmcprg.plot_style import DEFAULT_FONT_SIZE as FONT_SIZE

if TYPE_CHECKING:
    from pmcprg.copulas._base import CopulaVirt

logger = logging.getLogger(__name__)


# Value returned by a *minimised* negative log-likelihood when the copula
# cannot be evaluated (exception, NaN, ±∞). Finite, so a bounded optimiser
# can still rank the point as the worst; larger than any genuine
# |Σ w log c|. Shared with the ICE objectives (``pmcprg.pmc.ice``).
MLE_FAIL_PENALTY: float = 1e12


# ---------------------------------------------------------------------------
# Pseudo-observation utilities
# ---------------------------------------------------------------------------

def _weighted_log_density_sum(
    log_density: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """``Σ_i w_i · log c_i`` — the one place where 0·(−∞) is decided (RB-7).

    Conventions, shared by every likelihood objective of the package
    (``CopulaVirt.fit``, the ICE M-step, the ICE selection scores):

    * a point of **zero weight contributes nothing**, whatever its
      log-density. ``np.dot`` would evaluate ``0 · (−∞) = NaN`` and
      ``0 · NaN = NaN``, so a single corner point that the ICE posterior
      assigns to another pair of states used to turn the whole objective
      into NaN;
    * a **non-finite log-density at a point of positive weight** makes the
      total ``−∞``. ``−∞`` means the parameter cannot explain the point;
      NaN or ``+∞`` mean the kernel failed there. Neither may be averaged
      away, and all three rank the parameter value as impossible.

    ``weights=None`` means unit weights. Weights must be finite and ≥ 0.
    """
    ld = np.asarray(log_density, dtype=float).ravel()
    if weights is None:
        if not np.all(np.isfinite(ld)):
            return -math.inf
        return float(ld.sum())
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != ld.shape:
        raise ValueError(
            f"weights {w.shape} and log-densities {ld.shape} must have the same length."
        )
    if not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    pos = w > 0.0
    ld_pos = ld[pos]
    if not np.all(np.isfinite(ld_pos)):
        return -math.inf
    return float(np.dot(w[pos], ld_pos))


def _eval_log_likelihood(copula: "CopulaVirt", uv: np.ndarray) -> float:
    """∑ log c(û_i, v̂_i) on pseudo-observations.

    Uses the vectorised ``logpdf_array`` (50-100× faster than a Python loop
    over scalar ``pdf``). A non-finite log-density at any point gives ``−∞``
    — never NaN, which would break every ``sort`` on AIC downstream
    (see :func:`_weighted_log_density_sum`).
    """
    return _weighted_log_density_sum(copula.logpdf_array(np.asarray(uv, dtype=float)))


# ---------------------------------------------------------------------------
# Joint MLE of the multi-parameter families (Student, BB1, Tawn, t-EV)
# ---------------------------------------------------------------------------

@dataclass
class ParameterFit:
    """Outcome of :func:`_fit_two_parameter_mle`.

    ``params`` is always constructible (``cls(**params)``): on failure it is
    the best point the optimiser evaluated, the start value at worst.
    ``converged`` is ``False`` when the optimiser raised, hit its iteration
    limit, stopped on a line-search failure that a restart could not resolve,
    or never found a parameter value at which the likelihood is finite;
    ``message`` says which.
    """
    params:         dict
    log_likelihood: float
    converged:      bool
    message:        str
    n_iter:         int = 0
    n_eval:         int = 0


# L-BFGS-B settings for the joint fit (audit RB-3). The objective is the
# *total* weighted negative log-likelihood — not divided by Σw — and ``ftol``
# is the scale-free relative reduction (f_k − f_{k+1}) / max(|f_k|, |f_{k+1}|, 1):
# with the former ``ftol = 1e-7`` on a Σw-normalised objective (|f| < 1, so
# the ``max(…, 1)`` floor made the test absolute) the fit stopped after 3–5
# iterations next to its start value. ``gtol`` only matters for a tiny total
# weight, where the finite-difference gradient noise (≈ 1e-8 · Σ|w log c|)
# falls below it.
_MLE2_MAXITER: int   = 200
_MLE2_FTOL:    float = 1e-12
_MLE2_GTOL:    float = 1e-6
# A line-search failure whose restart improves f by less than this (relative)
# amount is a stop at the precision of the objective, not a failure.
_MLE2_STALL_REL: float = 1e-10


def _multi_extra_spec(cls, entry, extras: list[str], lo: float, hi: float,
                      t0: float, tau_min: float):
    """:func:`_two_parameter_spec` for a family with **two or more** extras.

    Same contract — ``(p0, stages, params_of)`` with ``p`` of length
    ``1 + len(extras)``. Nothing else had to change: the driver
    :func:`_fit_two_parameter_mle` never knew the dimension (it only calls
    ``to_x``/``from_x`` and hands ``box`` to L-BFGS-B), so growing the
    parameter vector is a change to this file alone, and only to the branch
    that builds the coordinates.

    * **Tawn 3** (``psi_u``, ``psi_v``; FR-9, round 5): ``p = (ln(θ − 1),
      ψ_u, ψ_v)``. τ is *jointly* constrained with the weights
      (``τ < 1/(1/ψ_u + 1/ψ_v − 1)``), but every (θ, ψ_u, ψ_v) with θ > 1 and
      the weights in (0, 1] **is** a Tawn copula — so, exactly as for types
      1/2, optimise in θ rather than τ and the box *is* the admissible set.
      τ comes back through the family's own quadrature, whose memo hands the
      constructor this very θ without a Brent re-inversion. The upper bound
      ln(θ − 1) ≤ ln(1/(1 − τ_hi) − 1) keeps τ(θ, ψ_u, ψ_v) ≤ τ(θ, 1, 1) =
      1 − 1/θ ≤ τ_hi (τ increases with each weight); the lower bound keeps
      τ ≳ 5·10⁻⁸ ≫ ε. The box is run twice for the same reason as types 1/2
      (a restart with a fresh curvature memory on a narrow curved ridge).
    * **any other set of extras**: ``p = (τ, x₁, …, xₙ)`` in the registered
      boxes, projected by ``cls.constrain_params`` — the n-dimensional form of
      the one-extra fallback at the end of :func:`_two_parameter_spec`.
    """
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM

    boxes = [EXTRA_PARAM_BOUNDS_BY_PARAM[name] for name in extras]

    def clip(val, bounds):
        return float(min(max(float(val), bounds[0]), bounds[1]))

    if tuple(extras) == ("psi_u", "psi_v"):
        from pmcprg.copulas.extreme_value.tawn import _tau_of
        s_b = (math.log(1e-6), math.log(max(1.0 / (1.0 - hi) - 1.0, 1e-5)))
        psi_b = [(xlo, xhi) for xlo, xhi, _ in boxes]
        # Start: θ₀ from Gumbel's closed form θ = 1/(1 − τ_start), and both
        # weights at (1 + τ_start)/2 — interior to the box and to the
        # admissible set, the same rule types 1/2 use for their single ψ.
        psi0 = [clip(0.5 * (1.0 + t0), b) for b in psi_b]
        p0 = (clip(math.log(max(1.0 / (1.0 - t0) - 1.0, 1e-6)), s_b), *psi0)

        def params_of(p) -> dict:
            s = clip(p[0], s_b)
            psis = [clip(v, b) for v, b in zip(p[1:], psi_b)]
            tau = _tau_of(1.0 + math.exp(s), *psis)
            return {"tau_k": max(tau, float(tau_min)),
                    **dict(zip(extras, psis))}

        box = [s_b, *psi_b]
        stage = (lambda p: [clip(p[0], s_b)] + [clip(v, b) for v, b in zip(p[1:], psi_b)],
                 lambda x: (clip(x[0], s_b),
                            *(clip(v, b) for v, b in zip(x[1:], psi_b))),
                 box)
        return p0, [stage, stage], params_of

    tau_b = (lo, hi)
    extra_b = [(xlo, xhi) for xlo, xhi, _ in boxes]
    p0 = (t0, *(clip(xinit, b) for (_, _, xinit), b in zip(boxes, extra_b)))

    def params_of(p) -> dict:
        vals = [clip(v, b) for v, b in zip(p[1:], extra_b)]
        return cls.constrain_params({"tau_k": clip(p[0], tau_b), **dict(zip(extras, vals))})

    stages = [(lambda p: [clip(p[0], tau_b)] + [clip(v, b) for v, b in zip(p[1:], extra_b)],
               lambda x: (clip(x[0], tau_b),
                          *(clip(v, b) for v, b in zip(x[1:], extra_b))),
               [tau_b, *extra_b])]
    return p0, stages, params_of


def _two_parameter_spec(cls, entry, tau_start: float):
    """Natural parameters and optimiser coordinates for the joint MLE.

    Returns ``(p0, stages, params_of)``: a start value of the family's natural
    parameter vector ``p``, a list of optimisation stages
    ``(to_x, from_x, box)``, and the map ``p → constructor kwargs``. Every
    stage's box is **the admissible set itself** — a bounded optimiser only
    handles boxes, and a box that must be projected onto the admissible set
    creates flat directions on which the optimiser stalls (audit RB-8).
    Stages run in order, each starting from the best point of the previous
    one.

    The name is historical: ``p`` has ``1 + (number of extras)`` components,
    two for every family until FR-9 round 5 added the three-parameter Tawn.
    This function dispatches on the *names* of the extras; a family with more
    than one is handed to :func:`_multi_extra_spec`.

    * Student (``df``): ``p = (τ, η = 1/ν)``, one stage in ``(atanh τ, η)``
      with ``η ∈ [1/ν_max, 1/ν_min]``. The likelihood is nearly flat in ν for
      large ν, and the Gaussian copula — the limit ν → ∞ (Demarta & McNeil
      2005) — sits at the boundary ``η = 0``; in η the start value ν = 4 is
      no longer on a plateau, and ``atanh`` evens out the τ-curvature, which
      grows without bound as |τ| → 1. Measured on Student(τ = 0.95, ν = 2.5),
      n = 1500: ν̂ = 2.479 in 9 iterations, against the start value 3.9996
      (6.9 nat lost) with the former (τ, ν) fit.
    * BB1 (``delta``): ``p = (θ, δ)``, ``θ ∈ [θ_floor, θ_max]``,
      ``δ ∈ [1, δ_max]``, mapped back by ``τ = 1 − 2/(δ(θ + 2))`` (Joe 1997,
      ch. 5) — the former (τ, δ) box had to clamp δ into ``[1, 1/(1 − τ)]``,
      a joint constraint no box expresses. Two stages: ``(log θ, log δ)``
      finds interior optima whatever the scale of θ; ``(θ, log δ)`` then
      reaches the Gumbel limit θ → θ_floor, which the logarithm puts at the
      end of a long flat valley (on Gumbel data, τ = 0.6, n = 1500, the first
      stage alone stopped 7·10⁻³ nat short of it). This ``τ = 1 − 2/(δ(θ+2))``
      map is BB1's own — built for a *positive* τ range. A rotated BB1
      (``CopulaBB190``/``CopulaBB1270``, audit FR-8, BB1 round) registers a
      *negative* range (only τ's sign flips under a 90°/270° rotation), so
      the θ/δ box is built on the shadow problem ``−τ`` and the fitted τ is
      negated back in ``params_of``: without this, the θ-bound built from
      the padded τ *ceiling* (``2/(1−τ_max) − 2``, meant to stay finite as
      τ_max → 1) degenerates on a range whose "ceiling" is ≈0 instead, and
      the optimiser can return a τ̂ outside the family's own registered
      range (found in practice fitting ``CopulaBB190``, not merely
      suspected: ``CopulaVirt.fit`` then failed to reconstruct the copula
      from that τ̂). ``CopulaBB190``/``CopulaBB1270`` additionally override
      ``fit`` to reuse BB1's own already-validated fit via the reflection
      identity instead of this routine directly; this sign-awareness is
      still needed here because ICE's M-step
      (``pmcprg.pmc.ice._fit_copula_params``) calls this function directly.
    * BB6 (``delta6``, FR-9): ``p = (ln(θ − 1), ln δ)``, a box that is the
      admissible set (every θ ≥ 1, δ ≥ 1 is a BB6 copula), mapped back by the
      family's **closed form** ``τ = 1 − (1 − τ_Joe(θ))/δ``; two stages, the
      second linear in θ − 1 to reach the Gumbel boundary θ = 1 (branch
      comment below). ``delta6`` is deliberately not BB1's ``delta``: this
      function dispatches on the parameter name, and the ``delta`` branch is
      BB1's, with BB1's own τ map
      (``pmcprg.copulas.archimedean.bb6``, "Parametrisation").
    * Tawn types 1/2 (``psi``, FR-9): ``p = (ln(θ − 1), ψ)``, a box that is
      the admissible set, mapped back by the family's τ(θ, ψ); two passes
      (branch comment below).
    * t-EV (``nu``, FR-9): ``p = (logit λ_U, 1/ν)``, a box every point of
      which is a t-EV copula, mapped back by the closed form s(λ_U, ν) and
      the family's τ(s, ν) (branch comment below). ``nu`` is deliberately
      not Student's ``df``: this function dispatches on the parameter name,
      and the ``df`` branch is Student's, with Student's bounds (ν > 2).
    * any other extra parameter: ``p = (τ, extra)`` in the registered boxes,
      projected by ``cls.constrain_params``.

    Bounds come from ``EXTRA_PARAM_BOUNDS_BY_PARAM`` and
    :func:`pmcprg.copulas._base.padded_tau_range` (single source of truth).
    """
    from pmcprg.copulas._base import EXTRA_PARAM_BOUNDS_BY_PARAM, padded_tau_range

    tau_min, tau_max = entry.value.TAU_MIN_MAX
    lo, hi = padded_tau_range(tau_min, tau_max)
    t0 = float(np.clip(tau_start if np.isfinite(tau_start) else 0.5 * (lo + hi), lo, hi))
    extras = [p for p in entry.value.PARAMETERS_SET_NAME if p != "tau_k"]
    if len(extras) > 1:
        # Two or more extras (FR-9, round 5: the three-parameter Tawn). Handled
        # in its own builder so that every branch below — the coordinates,
        # start values and stage lists of BB1, Student, Tawn 1/2 and t-EV —
        # stays exactly the code it was, down to the line.
        return _multi_extra_spec(cls, entry, extras, lo, hi, t0, tau_min)
    if len(extras) != 1:
        raise ValueError(
            f"{entry.value.CLASS_NAME}: expected at least one extra parameter, got {extras}."
        )
    name = extras[0]
    xlo, xhi, xinit = EXTRA_PARAM_BOUNDS_BY_PARAM[name]

    def clip(val, bounds):
        return float(min(max(float(val), bounds[0]), bounds[1]))

    if name == "df":
        tau_b, eta_b = (lo, hi), (1.0 / xhi, 1.0 / xlo)
        z_b = (math.atanh(lo), math.atanh(hi))
        p0 = (t0, clip(1.0 / xinit, eta_b))

        def params_of(p) -> dict:
            return {"tau_k": clip(p[0], tau_b), "df": 1.0 / clip(p[1], eta_b)}

        stages = [(
            lambda p: [math.atanh(clip(p[0], tau_b)), clip(p[1], eta_b)],
            lambda x: (clip(math.tanh(clip(x[0], z_b)), tau_b), clip(x[1], eta_b)),
            [z_b, eta_b],
        )]
        return p0, stages, params_of

    if name == "delta":
        # BB1's own (θ, δ) ↔ τ map is built for a positive τ range; a rotated
        # BB1 registers a negative one (module docstring above). Solve the
        # shadow problem on −τ (sign = −1) when the registered range is
        # entirely ≤ 0, and negate τ back in ``params_of`` — everything else
        # below is exactly BB1's own positive-τ derivation, just applied to
        # ``(lo_p, hi_p, t0_p) = (−hi, −lo, −t0)`` instead of ``(lo, hi, t0)``.
        sign = -1.0 if hi <= 0.0 else 1.0
        hi_p, t0_p = (-lo, -t0) if sign < 0.0 else (hi, t0)
        theta_floor = float(getattr(cls, "_THETA_FLOOR", 1e-6))
        # θ at the padded τ ceiling with δ = 1; a larger δ only raises τ, and
        # τ = 1 − 2/(δ(θ + 2)) stays < 1 on the whole box.
        theta_b = (theta_floor, max(2.0 / (1.0 - hi_p) - 2.0, 10.0 * theta_floor))
        delta_b = (xlo, xhi)
        # Start at (τ_start, δ_init) when that pair is admissible, else in the
        # middle of the admissible δ-interval [1, 1/(1 − τ_start)).
        delta0 = xinit
        if 2.0 / (delta0 * (1.0 - t0_p)) - 2.0 <= 10.0 * theta_floor:
            delta0 = 0.5 * (1.0 + 1.0 / (1.0 - t0_p))
        delta0 = clip(delta0, delta_b)
        p0 = (clip(2.0 / (delta0 * (1.0 - t0_p)) - 2.0, theta_b), delta0)

        def params_of(p) -> dict:
            theta, delta = clip(p[0], theta_b), clip(p[1], delta_b)
            # τ = (δθ + 2(δ − 1)) / (δ(θ + 2)): the value of 1 − 2/(δ(θ + 2))
            # without its cancellation near independence; negated back to the
            # registered sign when solving the shadow (rotated) problem.
            tau = (delta * theta + 2.0 * (delta - 1.0)) / (delta * (theta + 2.0))
            return {"tau_k": sign * tau, "delta": delta}

        log_theta_b = (math.log(theta_b[0]), math.log(theta_b[1]))
        log_delta_b = (math.log(delta_b[0]), math.log(delta_b[1]))
        stages = [
            (lambda p: [math.log(clip(p[0], theta_b)), math.log(clip(p[1], delta_b))],
             lambda x: (clip(math.exp(clip(x[0], log_theta_b)), theta_b),
                        clip(math.exp(clip(x[1], log_delta_b)), delta_b)),
             [log_theta_b, log_delta_b]),
            (lambda p: [clip(p[0], theta_b), math.log(clip(p[1], delta_b))],
             lambda x: (clip(x[0], theta_b),
                        clip(math.exp(clip(x[1], log_delta_b)), delta_b)),
             [theta_b, log_delta_b]),
        ]
        return p0, stages, params_of

    if name == "delta6":
        # BB6 (FR-9): (τ, δ) is jointly constrained (δ ≤ 1/(1 − τ)), but every
        # θ ≥ 1, δ ≥ 1 *is* a BB6 copula — so optimise in (ln(θ − 1), ln δ),
        # whose box is the admissible set, and map back by the family's own
        # closed form τ = 1 − (1 − τ_Joe(θ))/δ (bb6 module docstring, (★)).
        # ``delta6`` is deliberately not BB1's ``delta``: that branch carries
        # BB1's τ = 1 − 2/(δ(θ + 2)) map, which is a different family.
        # θ − 1 ≤ 2/(1 − τ_hi) bounds τ_Joe(θ) ≈ 1 − 2/θ, so τ stays below 1
        # on the whole box (BB1's own ceiling 2/(1 − τ_hi) − 2, same
        # reasoning: the padded τ ceiling, not the registered τ = 1);
        # ln(θ − 1) ≥ ln 1e-10 keeps τ(θ, 1) ≈ 6·10⁻¹¹ ≫ ε, and θ = 1 + 1e-10
        # is the stand-in for the Gumbel boundary θ = 1 (as BB1's θ floor
        # stands in for its own Gumbel limit θ → 0).
        from pmcprg.copulas.archimedean.joe import _joe_tau_from_theta
        s_b = (math.log(1e-10), math.log(max(2.0 / (1.0 - hi), 10.0)))
        delta_b = (xlo, xhi)
        log_delta_b = (math.log(delta_b[0]), math.log(delta_b[1]))
        # Start: δ₀ = √δ_max(τ_start), interior to the admissible [1, δ_max]
        # and splitting the "Joe versus Gumbel" share of τ evenly in log; θ₀
        # is then the θ that realises τ_start at δ₀.
        delta_max = 1.0 / (1.0 - t0) if t0 < 1.0 else xhi
        delta0 = clip(math.sqrt(max(delta_max, 1.0)), delta_b)
        tau_joe0 = min(max(1.0 - delta0 * (1.0 - t0), 0.0), 1.0 - 1e-12)
        theta0 = 1.0 / (1.0 - tau_joe0)      # Gumbel's map: within a factor 2 of Joe's
        p0 = (clip(math.log(max(theta0 - 1.0, 1e-10)), s_b), delta0)

        def params_of(p) -> dict:
            s, delta = clip(p[0], s_b), clip(p[1], delta_b)
            tau = 1.0 - (1.0 - _joe_tau_from_theta(1.0 + math.exp(s))) / delta
            return {"tau_k": min(max(tau, float(tau_min)), 1.0 - 1e-12), "delta6": delta}

        # Two stages, as BB1: (ln(θ − 1), ln δ) finds interior optima whatever
        # the scale of θ − 1; (θ − 1, ln δ) then reaches the Gumbel boundary
        # θ → 1, which the logarithm puts at the end of a long flat valley.
        exc_b = (math.exp(s_b[0]), math.exp(s_b[1]))
        stages = [
            (lambda p: [clip(p[0], s_b), math.log(clip(p[1], delta_b))],
             lambda x: (clip(x[0], s_b),
                        clip(math.exp(clip(x[1], log_delta_b)), delta_b)),
             [s_b, log_delta_b]),
            (lambda p: [clip(math.exp(clip(p[0], s_b)), exc_b), math.log(clip(p[1], delta_b))],
             lambda x: (clip(math.log(clip(x[0], exc_b)), s_b),
                        clip(math.exp(clip(x[1], log_delta_b)), delta_b)),
             [exc_b, log_delta_b]),
        ]
        return p0, stages, params_of

    if name == "psi":
        # Tawn types 1/2 (FR-9): (τ, ψ) is jointly constrained (τ < ψ), but
        # every (θ, ψ) with θ > 1, ψ ∈ (0, 1] is a Tawn copula — so optimise
        # in (ln(θ − 1), ψ), whose box is the admissible set, and map back by
        # the family's own τ(θ, ψ) quadrature (whose memo hands the
        # constructor this very θ back, without a Brent re-inversion).
        # θ ≤ 1/(1 − τ_hi) keeps τ(θ, ψ) ≤ τ(θ, 1) = 1 − 1/θ ≤ τ_hi (τ is
        # increasing in ψ); ln(θ − 1) ≥ ln 1e-6 keeps τ ≥ ≈ 5e-8 ≫ ε.
        from pmcprg.copulas.extreme_value.tawn import _tau_of
        s_b = (math.log(1e-6), math.log(max(1.0 / (1.0 - hi) - 1.0, 1e-5)))
        psi_b = (xlo, xhi)
        weights = cls._weights
        # Start: θ₀ from Gumbel's closed form θ = 1/(1 − τ_start), and
        # ψ₀ = (1 + τ_start)/2, interior to the admissible ψ-interval (τ, 1].
        psi0 = clip(0.5 * (1.0 + t0), psi_b)
        p0 = (clip(math.log(max(1.0 / (1.0 - t0) - 1.0, 1e-6)), s_b), psi0)

        def params_of(p) -> dict:
            s, psi = clip(p[0], s_b), clip(p[1], psi_b)
            tau = _tau_of(1.0 + math.exp(s), *weights(psi))
            return {"tau_k": max(tau, float(tau_min)), "psi": psi}

        # The same box twice: the second stage is a restart with a fresh
        # curvature memory. The likelihood is a narrow curved ridge in these
        # coordinates; from a poor start (τ_start far from the data's) the
        # first pass's line searches bounce off the (θ_max, ψ_min) corner and
        # can stop short — 3 of 50 contrived starts (5 data sets × 5 starts,
        # both types), once by 131 nat — while the restart reaches the same
        # optimum from all 50.
        stage = (lambda p: [clip(p[0], s_b), clip(p[1], psi_b)],
                 lambda x: (clip(x[0], s_b), clip(x[1], psi_b)),
                 [s_b, psi_b])
        return p0, [stage, stage], params_of

    if name == "nu":
        # t-EV (FR-9): every (τ, ν) is admissible, but the constructor inverts
        # τ(s, ν) by Brent's method — so optimise in (logit λ_U, 1/ν) instead:
        # s follows from λ_U = 2·T_{ν+1}(−√((ν+1)η)) in closed form, and the
        # memo of the τ(s, ν) quadrature hands the constructor this very s
        # back. λ_U/2 ≤ τ ≤ λ_U on the whole box (t_ev module docstring), so
        # λ_U ∈ [1e-6, τ_hi] keeps τ ∈ [5e-7, τ_hi]; 1/ν as for Student (the
        # Hüsler–Reiss limit ν → ∞ lies beyond the 1/ν_max side).
        from pmcprg.copulas.extreme_value.t_ev import _s_from_lambda, _tau_of

        def logit(p):
            return math.log(p) - math.log1p(-p)

        lam_b = (logit(1e-6), logit(hi))
        inv_b = (1.0 / xhi, 1.0 / xlo)
        # Start: λ₀ = τ_start (τ ≤ λ_U ≤ 2τ), ν₀ from the registry.
        p0 = (clip(logit(min(max(t0, 1e-6), hi)), lam_b), clip(1.0 / xinit, inv_b))

        def params_of(p) -> dict:
            y, e = clip(p[0], lam_b), clip(p[1], inv_b)
            nu = 1.0 / e
            tau = _tau_of(_s_from_lambda(1.0 / (1.0 + math.exp(-y)), nu), nu)
            return {"tau_k": max(tau, float(tau_min)), "nu": nu}

        stage = (lambda p: [clip(p[0], lam_b), clip(p[1], inv_b)],
                 lambda x: (clip(x[0], lam_b), clip(x[1], inv_b)),
                 [lam_b, inv_b])
        return p0, [stage, stage], params_of

    tau_b, extra_b = (lo, hi), (xlo, xhi)
    p0 = (t0, clip(xinit, extra_b))

    def params_of(p) -> dict:
        return cls.constrain_params({"tau_k": clip(p[0], tau_b), name: clip(p[1], extra_b)})

    stages = [(lambda p: [clip(p[0], tau_b), clip(p[1], extra_b)],
               lambda x: (clip(x[0], tau_b), clip(x[1], extra_b)),
               [tau_b, extra_b])]
    return p0, stages, params_of


def _fit_two_parameter_mle(
    cls,
    entry,
    uv: np.ndarray,
    weights: np.ndarray | None,
    tau_start: float,
) -> ParameterFit:
    """Maximise ``Σ w log c(u, v; τ, extras…)`` over a multi-parameter family.

    L-BFGS-B with a finite-difference gradient, run over the stages of
    :func:`_two_parameter_spec` on the **total** weighted negative
    log-likelihood, with the scale-free tolerances above. Any evaluation
    that raises or returns a non-finite value is mapped to
    :data:`MLE_FAIL_PENALTY` (RB-7). A line-search failure
    (``ABNORMAL_TERMINATION_IN_LNSRCH``) triggers one restart with a fresh
    curvature memory; if the restart cannot improve the objective the point
    is accepted as converged at the precision of the objective.

    ``weights=None`` means unit weights. On pseudo-observations built from
    ranks and unit weights this is the pseudo-maximum-likelihood estimator of
    Genest, Ghoudi & Rivest (1995).
    """
    from scipy.optimize import minimize

    uv = np.asarray(uv, dtype=float)
    p0, stages, params_of = _two_parameter_spec(cls, entry, tau_start)
    n_eval = [0]

    def neg_ll(p) -> float:
        n_eval[0] += 1
        try:
            ll = _weighted_log_density_sum(cls(**params_of(p)).logpdf_array(uv), weights)
        except Exception as exc:          # inadmissible or failing evaluation
            logger.debug("%s joint MLE: evaluation failed at %s — %s",
                         cls.__name__, p, exc)
            return MLE_FAIL_PENALTY
        return -ll if np.isfinite(ll) else MLE_FAIL_PENALTY

    options = {"maxiter": _MLE2_MAXITER, "ftol": _MLE2_FTOL, "gtol": _MLE2_GTOL}
    best_p, best_f = tuple(p0), neg_ll(p0)
    n_iter = 0
    converged = False
    messages: list[str] = []

    for k, (to_x, from_x, box) in enumerate(stages):
        f_before = best_f

        def g(x, _from_x=from_x):
            return neg_ll(_from_x(x))

        try:
            res = minimize(g, to_x(best_p), method="L-BFGS-B", bounds=box, options=options)
            n_iter += int(res.nit)
            if res.fun <= best_f:
                best_p, best_f = tuple(from_x(res.x)), float(res.fun)
            ok, msg = bool(res.success), str(res.message)
            if not ok and "ABNORMAL" in msg.upper():
                f_mid = best_f
                res2 = minimize(g, to_x(best_p), method="L-BFGS-B", bounds=box, options=options)
                n_iter += int(res2.nit)
                gain = f_mid - float(res2.fun)
                if res2.fun <= best_f:
                    best_p, best_f = tuple(from_x(res2.x)), float(res2.fun)
                if res2.success:
                    ok, msg = True, f"{msg}; restart: {res2.message}"
                elif (gain <= _MLE2_STALL_REL * max(1.0, abs(best_f))
                      and (res.nit + res2.nit > 0 or k > 0)):
                    # Moved, then could not improve: the precision of the
                    # objective. A first stage that never left its start
                    # value is the RB-3 failure itself and stays a failure.
                    ok = True
                    msg = (f"{msg}; restart gained {gain:.3g} — stopped at the "
                           f"precision of the objective")
                else:
                    msg = f"{msg}; restart: {res2.message}"
        except Exception as exc:
            ok, msg = False, f"optimiser raised {type(exc).__name__}: {exc}"
        messages.append(f"stage {k + 1}: {msg}")
        # A later stage that fails without improving the point does not undo
        # the convergence of an earlier one.
        stalled = (f_before - best_f) <= _MLE2_STALL_REL * max(1.0, abs(best_f))
        converged = ok or (converged and stalled)

    message = "; ".join(messages)
    if best_f >= MLE_FAIL_PENALTY:
        converged = False
        message = f"{message}; the likelihood is not finite at any evaluated parameter value"
    ll = -best_f if best_f < MLE_FAIL_PENALTY else -math.inf
    return ParameterFit(params=params_of(best_p), log_likelihood=ll, converged=converged,
                        message=message, n_iter=n_iter, n_eval=n_eval[0])


def _normalised_weights(weights: np.ndarray) -> np.ndarray:
    """Weights scaled to sum to 1 (guarded against an all-zero vector)."""
    w = np.asarray(weights, dtype=float)
    return w / (w.sum() + MIN_POSITIVE)


# Evaluation points processed per pass. The indicator matrix is (chunk, n),
# so peak memory is O(chunk·n) instead of O(n²) — and the matmul against the
# weights upcasts it to float64, which is where the O(n²) form really hurt:
# 68 MB per call at n = 2500, enough to kill a worker when several ICE
# processes hit it at once.
_EMPIRICAL_COPULA_CHUNK = 512


def _empirical_copula(
    uv: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """C_n(u,v) = Σ_i w_i 1{û_i ≤ u, v̂_i ≤ v}, evaluated at given points.

    ``weights=None`` is the classic uniform case (w_i = 1/n). Explicit
    weights are normalised to sum to 1 — the **weighted** empirical copula
    used by the ξ-weighted ICE ``cvm`` selection criterion (audit N-6/Q-20).

    Evaluated in row chunks: mathematically identical, but bounded in memory.
    """
    uv     = np.asarray(uv, dtype=float)
    points = np.asarray(points, dtype=float)
    m      = points.shape[0]
    w      = None if weights is None else _normalised_weights(weights)
    out    = np.empty(m, dtype=float)

    for start in range(0, m, _EMPIRICAL_COPULA_CHUNK):
        stop  = min(start + _EMPIRICAL_COPULA_CHUNK, m)
        leq_u = uv[:, 0][None, :] <= points[start:stop, 0][:, None]
        leq_v = uv[:, 1][None, :] <= points[start:stop, 1][:, None]
        ind   = leq_u & leq_v
        out[start:stop] = ind.mean(axis=1) if w is None else ind @ w
    return out


def _cvm_statistic(
    uv: np.ndarray,
    copula: "CopulaVirt",
    weights: np.ndarray | None = None,
) -> float:
    """Cramér-von Mises statistic.

    ``weights=None``: S_n = Σ (C_n(u_i,v_i) − C_θ(u_i,v_i))² — the classic
    unweighted form used by the GoF test and its bootstrap.

    With ``weights``: both the empirical copula and the outer L²-sum use the
    same normalised weights, S = Σ w_i (C_n^w − C_θ)² — the ξ-weighted form
    of the ICE ``cvm`` criterion. Defined here next to the unweighted
    original so the two formulas cannot drift apart (audit Q-20).

    Reference: Genest, C., Rémillard, B. & Beaudoin, D. (2009).
    Goodness-of-fit tests for copulas: A review and a power study.
    *Insurance: Mathematics and Economics* 44(2), 199–213, §2 — the
    statistic ``S_n`` on pseudo-observations ``R_i / (n + 1)``, which is
    what :meth:`CopulaVirt.fit` forms.
    """
    uv     = np.asarray(uv, dtype=float)
    Cn     = _empirical_copula(uv, uv, weights)
    Ctheta = copula.cdf_array(uv)
    if weights is None:
        return float(np.sum((Cn - Ctheta) ** 2))
    return float(np.dot(_normalised_weights(weights), (Cn - Ctheta) ** 2))


def _empirical_tail_dep(uv: np.ndarray, u_grid: np.ndarray, side: str) -> np.ndarray:
    """Non-parametric tail dependence estimator on a grid of thresholds.

    side='lower': λ̂_L(u) = C_n(u, u) / u
    side='upper': λ̂_U(u) = (1 − 2u + C_n(u, u)) / (1 − u)
    References: Joe (1997); Caillault, C. & Guégan, D. (2005). Empirical
    estimation of tail dependence using copulas: application to Asian
    markets. *Quantitative Finance* 5(5), 489–501.
    """
    out = np.empty(len(u_grid))
    for k, u in enumerate(u_grid):
        Cn_uu = float(np.mean((uv[:, 0] <= u) & (uv[:, 1] <= u)))
        if side == 'lower':
            out[k] = Cn_uu / u if u > 0.0 else 0.0
        else:
            out[k] = (1.0 - 2.0*u + Cn_uu) / (1.0 - u) if u < 1.0 else 0.0
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# GoFResult
# ---------------------------------------------------------------------------

@dataclass
class GoFResult:
    """Returned by :meth:`FitResult.gof_test` — Cramér-von Mises GoF test."""
    statistic:        float
    p_value:          float
    B:                int
    n_valid_bootstrap: int
    bootstrap_stats:  np.ndarray

    def __repr__(self) -> str:
        return (f'GoFResult(S_n={self.statistic:.4f}, p-value={self.p_value:.3f}, '
                f'B={self.n_valid_bootstrap}/{self.B})')


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Returned by :meth:`CopulaVirt.fit`.

    ``converged`` is ``False`` when the numerical optimiser behind a
    ``method='mle'`` fit did not converge (the copula is then the best point
    it found, and a WARNING was logged). Moment fits are always ``True``.
    """
    copula:         "CopulaVirt"
    method:         str
    tau_k:          float
    log_likelihood: float
    n_obs:          int
    uv:             np.ndarray   # pseudo-observations used for the fit, shape (n, 2)
    converged:      bool = True

    @property
    def n_params(self) -> int:
        return getattr(self.copula, 'n_params', 1)

    @property
    def aic(self) -> float:
        """Akaike Information Criterion: 2k − 2·loglik (smaller is better)."""
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        """Bayesian Information Criterion: k·log(n) − 2·loglik (smaller is better)."""
        return self.n_params * np.log(self.n_obs) - 2.0 * self.log_likelihood

    @property
    def aicc(self) -> float:
        """Corrected AIC for small samples: AIC + 2k(k+1)/(n−k−1)."""
        k, n = self.n_params, self.n_obs
        if n - k - 1 <= 0:
            return float('nan')
        return self.aic + 2.0 * k * (k + 1) / (n - k - 1)

    @property
    def hqc(self) -> float:
        """Hannan-Quinn: 2k·log(log n) − 2·loglik (asymptotically less biased than BIC)."""
        return 2.0 * self.n_params * np.log(np.log(self.n_obs)) - 2.0 * self.log_likelihood

    def __repr__(self) -> str:
        name = self.copula.copula_enum.value.LONG_NAME
        return (
            f'FitResult(copula={name!r}, method={self.method!r}, '
            f'tau_k={self.tau_k:.4f}, loglik={self.log_likelihood:.4f}, '
            f'AIC={self.aic:.2f}, BIC={self.bic:.2f}, n={self.n_obs})'
        )

    # ------------------------------------------------------------------
    # Goodness-of-fit (Cramér-von Mises, parametric bootstrap)
    # ------------------------------------------------------------------
    def gof_test(self, B: int = 100, seed: int | None = None) -> GoFResult:
        """Cramér-von Mises GoF test via parametric bootstrap.

        H₀: data was generated by the fitted copula family.
        Returns S_n and a bootstrap p-value over B replicates. The copula is
        **re-fitted on every replicate**, so this is the parametric bootstrap
        of Genest & Rémillard (2008, Ann. IHP 44(6):1096–1127) for the
        composite null, as reviewed by Genest, Rémillard & Beaudoin (2009,
        Insurance Math. Econom. 44(2):199–213). The p-value is
        ``(1 + #{S*_b ≥ S_n}) / (n_valid + 1)`` (Davison & Hinkley 1997, ch. 4) — never 0; GRB 2009 (App. A) write the plain proportion,
        which differs by at most 1/(B + 1).
        """
        try:
            self.copula.cdf([0.5, 0.5])
        except NotImplementedError:
            raise NotImplementedError(
                f'GoF test requires C(u,v); not implemented for '
                f'{self.copula.__class__.__name__}.'
            )
        rng = np.random.default_rng(seed)
        S_n = _cvm_statistic(self.uv, self.copula)

        cls   = self.copula.__class__
        stats = np.full(B, np.nan)
        # Log every ~10 % of progress (B/10 iterations) for long bootstraps.
        log_every = max(B // 10, 1)
        for b in range(B):
            sample = self.copula.sample(n=self.n_obs,
                                        seed=int(rng.integers(0, 2**31 - 1)))
            try:
                r_b      = cls.fit(sample, method=self.method)
                stats[b] = _cvm_statistic(r_b.uv, r_b.copula)
            except Exception as e:
                logger.debug('Bootstrap iter %d failed: %s', b, e)
            if (b + 1) % log_every == 0:
                logger.info('GoF bootstrap: %d / %d done', b + 1, B)

        valid    = ~np.isnan(stats)
        n_valid  = int(valid.sum())
        p_value  = (float((1 + np.sum(stats[valid] >= S_n)) / (n_valid + 1))
                    if n_valid > 0 else float('nan'))
        return GoFResult(statistic=S_n, p_value=p_value, B=B,
                         n_valid_bootstrap=n_valid,
                         bootstrap_stats=stats[valid])

    # ------------------------------------------------------------------
    # Bootstrap confidence interval on tau_k
    # ------------------------------------------------------------------
    def bootstrap_ci(self, B: int = 500, alpha: float = 0.05,
                     seed: int | None = None) -> tuple[float, float]:
        """Non-parametric bootstrap percentile CI for tau_k at level (1-alpha).

        Resamples the pseudo-observations with replacement B times, refits with
        the same method, and returns the (alpha/2, 1-alpha/2) quantiles.
        """
        rng = np.random.default_rng(seed)
        cls = self.copula.__class__
        tau_boots = np.full(B, np.nan)
        log_every = max(B // 10, 1)
        for b in range(B):
            idx    = rng.integers(0, self.n_obs, size=self.n_obs)
            sample = self.uv[idx]
            try:
                tau_boots[b] = cls.fit(sample, method=self.method).tau_k
            except Exception as e:
                logger.debug('Bootstrap iter %d failed: %s', b, e)
            if (b + 1) % log_every == 0:
                logger.info('Bootstrap τ-CI: %d / %d done', b + 1, B)
        valid = ~np.isnan(tau_boots)
        if valid.sum() < 10:
            raise RuntimeError(f'Bootstrap failed: only {valid.sum()}/{B} valid replicates.')
        lo = float(np.quantile(tau_boots[valid], alpha / 2.0))
        hi = float(np.quantile(tau_boots[valid], 1.0 - alpha / 2.0))
        return lo, hi

    # ------------------------------------------------------------------
    # Asymptotic standard errors (audit FR-4, outside ICE)
    # ------------------------------------------------------------------
    def standard_errors(self, *, ranks: bool = True):
        """Asymptotic standard errors of this fit — opt-in, nothing is cached.

        ``copula.standard_errors(self.uv, method=self.method, ranks=ranks)``:
        the pseudo-likelihood sandwich with the estimated-rank corrections for
        ``method='mle'`` (Genest, Ghoudi & Rivest 1995), the variance of the
        inversion of Kendall's τ for ``method='tau'`` (Genest & Favre 2007;
        Kojadinovic & Yan 2010). Returns a
        :class:`pmcprg.copulas._stderr.StandardErrors`; see
        :mod:`pmcprg.copulas._stderr`. Unlike :meth:`bootstrap_ci`, it is
        analytic (no refit) and reports SEs on τ and on the native parameters.
        """
        from pmcprg.copulas._stderr import standard_errors as _standard_errors
        return _standard_errors(self.copula, self.uv, None, self.method, ranks=ranks)

    # ------------------------------------------------------------------
    # K-fold cross-validation log-likelihood
    # ------------------------------------------------------------------
    def cv_loglik(self, K: int = 5, seed: int | None = None) -> float:
        """K-fold CV log-likelihood (held-out test loglik summed over folds).

        For each fold, the copula is refitted on the K−1 training folds (using
        the same `method`) and its log-density is summed over the held-out
        test fold's pseudo-observations. Higher is better.
        """
        if K < 2 or K > self.n_obs:
            raise ValueError(f'K must be in [2, n_obs], got {K} for n={self.n_obs}.')
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n_obs)
        folds = np.array_split(idx, K)
        cls = self.copula.__class__

        cv_ll = 0.0
        for k in range(K):
            test_idx  = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
            try:
                r_k    = cls.fit(self.uv[train_idx], method=self.method)
                cv_ll += _eval_log_likelihood(r_k.copula, self.uv[test_idx])
            except Exception as e:
                logger.warning('CV fold %d failed: %s', k, e)
                return float('nan')
        return float(cv_ll)

    # ------------------------------------------------------------------
    # Visual diagnostics
    # ------------------------------------------------------------------
    def plot_diagnostics(self, plot_dir: str, prefix: str = '') -> None:
        """6-panel diagnostic plot:
        (0,0) pseudo-observations + fitted PDF contours
        (0,1) PP plot  C_n vs C_θ at the data points
        (0,2) lower tail dependence  λ̂_L(u) vs fitted λ_L
        (1,0) empirical copula heatmap C_n(u,v)
        (1,1) residuals heatmap C_n − C_θ
        (1,2) upper tail dependence  λ̂_U(u) vs fitted λ_U
        """
        try:
            self.copula.cdf([0.5, 0.5])
            has_cdf = True
        except NotImplementedError:
            has_cdf = False

        name      = self.copula.copula_enum.value.LONG_NAME
        short     = self.copula.copula_enum.value.SHORT_NAME
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f'{name} — fit diagnostics  (τ̂={self.tau_k:.3f}, n={self.n_obs})',
            fontsize=FONT_SIZE + 1, y=0.995,
        )

        grid    = np.linspace(0.01, 0.99, 80)
        GX, GY  = np.meshgrid(grid, grid)
        Z_pdf   = np.vectorize(lambda u, v: self.copula.pdf([u, v]))(GX, GY)
        vmax    = float(np.nanpercentile(Z_pdf, 95))

        # (0,0) — pseudo-obs over fitted PDF contours
        ax = axes[0, 0]
        # Explicit levels capped at the 95th percentile, not `levels=10`:
        # an integer spreads 10 levels over the RAW data range, and a copula
        # density spans ~15 decades because it diverges at the corners, so
        # every interior cell fell in the first bin and the panel came out a
        # flat rectangle. extend="max" keeps the clipped peak painted.
        ax.contourf(GX, GY, Z_pdf, np.linspace(0, max(vmax, 1e-10), 10),
                    cmap='Blues', alpha=0.4,
                    vmin=0, vmax=max(vmax, 1e-10), extend='max')
        ax.scatter(self.uv[:, 0], self.uv[:, 1], s=4, alpha=0.55, c='black')
        ax.set_title('Pseudo-obs + fitted PDF')
        ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')

        if has_cdf:
            Cn_pts = _empirical_copula(self.uv, self.uv)
            Ct_pts = np.array([self.copula.cdf(list(row)) for row in self.uv])
            points = np.column_stack([GX.ravel(), GY.ravel()])
            Cn_g   = _empirical_copula(self.uv, points).reshape(GX.shape)
            Ct_g   = np.vectorize(lambda u, v: self.copula.cdf([u, v]))(GX, GY)

            # (0,1) — PP plot
            ax = axes[0, 1]
            ax.scatter(Cn_pts, Ct_pts, s=6, alpha=0.55)
            ax.plot([0, 1], [0, 1], 'r--', lw=1)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
            ax.set_xlabel('$C_n$ (empirical)'); ax.set_ylabel(r'$C_\theta$ (fitted)')
            ax.set_title('PP plot')

            # (1,0) — empirical copula
            ax = axes[1, 0]
            cs = ax.contourf(GX, GY, Cn_g, levels=11, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(cs, ax=ax)
            ax.set_title(r'$C_n(u,v)$ (empirical)')
            ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')

            # (1,1) — residuals
            ax    = axes[1, 1]
            diff  = Cn_g - Ct_g
            vmaxd = float(max(abs(diff.min()), abs(diff.max()), 1e-3))
            cs = ax.contourf(GX, GY, diff, levels=15, cmap='RdBu_r',
                             vmin=-vmaxd, vmax=vmaxd)
            fig.colorbar(cs, ax=ax)
            ax.set_title(rf'Residuals $C_n - C_\theta$  (max |Δ|={vmaxd:.3f})')
            ax.set_xlabel('u'); ax.set_ylabel('v'); ax.set_aspect('equal')
        else:
            for r, c in [(0, 1), (1, 0), (1, 1)]:
                axes[r, c].axis('off')
                axes[r, c].text(0.5, 0.5, f'CDF not available\nfor {name}',
                                ha='center', va='center', transform=axes[r, c].transAxes)

        # (0,2) — lower tail dependence
        u_low  = np.linspace(0.02, 0.30, 25)
        u_high = np.linspace(0.70, 0.98, 25)
        lL_emp = _empirical_tail_dep(self.uv, u_low,  'lower')
        lU_emp = _empirical_tail_dep(self.uv, u_high, 'upper')
        lL_th, lU_th = self.copula.tail_dependence()

        ax = axes[0, 2]
        ax.plot(u_low, lL_emp, 'o-', ms=4, label=r'Empirical $\hat\lambda_L(u)$')
        if not np.isnan(lL_th):
            ax.axhline(lL_th, color='red', ls='--', lw=1.2,
                       label=rf'Fitted $\lambda_L = {lL_th:.3f}$')
        ax.set_xlabel('u'); ax.set_ylabel(r'$\lambda_L$')
        ax.set_title('Lower tail dependence')
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)

        # (1,2) — upper tail dependence
        ax = axes[1, 2]
        ax.plot(u_high, lU_emp, 'o-', ms=4, label=r'Empirical $\hat\lambda_U(u)$')
        if not np.isnan(lU_th):
            ax.axhline(lU_th, color='red', ls='--', lw=1.2,
                       label=rf'Fitted $\lambda_U = {lU_th:.3f}$')
        ax.set_xlabel('u'); ax.set_ylabel(r'$\lambda_U$')
        ax.set_title('Upper tail dependence')
        ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{prefix}Diagnostics_{short}.png'),
                    bbox_inches='tight')
        plt.close()


# ---------------------------------------------------------------------------
# FitBestResults
# ---------------------------------------------------------------------------

class FitBestResults(list):
    """Returned by :meth:`CopulaVirt.fit_best` — a ``list[FitResult]`` sorted
    by AIC, carrying what the ranking left out (audit RB-10).

    Attributes
    ----------
    method : str
        The estimation method that was requested for every family.
    other_method : list[FitResult]
        Fits that the family could only produce with another method — Student
        and BB1 always fit by MLE, since τ alone does not identify their
        second parameter. Their log-likelihood is a *maximum*, which a
        τ-inversion log-likelihood is not, so they are reported here and
        **not ranked** with the others.
    failures : list[tuple[str, str]]
        ``(class name, "ExceptionType: message")`` for every family whose fit
        raised. Each is also logged at WARNING.
    """

    def __init__(self, ranked=(), *, method: str = "tau",
                 other_method=None, failures=None):
        super().__init__(ranked)
        self.method = method
        self.other_method: list = list(other_method or [])
        self.failures: list[tuple[str, str]] = list(failures or [])

    # Opt-in significance of the ranking (audit FR-6). Read-only: neither
    # method changes the ranking. Implemented in
    # :mod:`pmcprg.diagnostics.model_selection` (imported lazily).
    def compare(self, test: str = "vuong", **kwargs):
        """Pairwise Vuong / Clarke tests between the ranked fits.

        Returns a :class:`pmcprg.diagnostics.model_selection.ComparisonMatrix`;
        keyword arguments (``correction="akaike"``, ``alpha``, ``bandwidth``,
        ``n_eff``) as :func:`~pmcprg.diagnostics.model_selection.fit_best_comparison`.
        """
        from pmcprg.diagnostics.model_selection import fit_best_comparison
        return fit_best_comparison(self, test, **kwargs)

    def confidence_set(self, test: str = "vuong", **kwargs):
        """Families not significantly worse than the AIC winner — the ties.

        Returns a :class:`pmcprg.diagnostics.model_selection.ConfidenceSet`;
        keyword arguments (``correction="akaike"``, ``alpha``, ``adjust``,
        ``bandwidth``, ``n_eff``) as
        :func:`~pmcprg.diagnostics.model_selection.fit_best_confidence_set`.
        """
        from pmcprg.diagnostics.model_selection import fit_best_confidence_set
        return fit_best_confidence_set(self, test, **kwargs)
