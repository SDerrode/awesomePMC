"""
pmcprg.pmc._oakes — Oakes' (1999) observed information for a copula τ fitted
*inside* ICE (AUDIT_COPULES FR-4, "Reste": standard errors inside ICE).

Scope (pilot)
-------------
One-parameter copula families only, validated for **Gauss** and **Clayton**;
K = 2 state models; one pair (i, j) of :meth:`PMCModel.copula_blocks` at a
time. Margins are held fixed at the model handed in (``fit_margins=False``
in the ICE run that produced it) — extending this to margins re-estimated
inside ICE, to two-parameter families (BB1, Student), or to the prior /
other pairs' τ is future work; see the module docstring of
:mod:`pmcprg.pmc.ice` for how those enter the M-step.

Why the outside-ICE sandwich (:mod:`pmcprg.copulas._stderr`) is not enough
---------------------------------------------------------------------------
:func:`pmcprg.copulas._stderr.standard_errors` treats its ``weights`` as
*given*, fixed numbers. Applied naively to the converged ICE posteriors
ξ̂_n(i, j), it is the curvature of

    Q(θ | θ̂) = Σ_n ξ̂_n(i, j) · log c_θ(û_n, v̂_n)

at θ = θ̂ — the **complete-data** observed information, as if the state
sequence were known exactly and equal to its expectation. It ignores that
ξ̂_n(i, j) is itself an estimate — computed from a latent state sequence —
and so understates the true (observed-data) uncertainty of τ̂ (Louis 1982;
Oakes 1999, §1).

Oakes' identity
----------------
For an EM fit at θ̂, with Q(θ | θ′) the expected complete-data log-likelihood
under the E-step run at θ′ (Dempster, Laird & Rubin 1977), the *observed*
information at θ̂ is (Oakes 1999, Theorem)::

    I(θ̂) = − ∂²Q(θ | θ̂)/∂θ² |_{θ=θ̂}  −  ∂²Q(θ | θ′)/∂θ ∂θ′ |_{θ=θ′=θ̂}

The first term is the complete-data information the naive sandwich already
computes (no E-step re-run: ξ̂ is fixed, only the copula density moves).
The second is the "missing information" Louis' identity would otherwise
need Var[score | Y] for; Oakes' trick gets it from one more E-step, run at
a θ′ perturbed away from θ̂, with the copula density held at θ̂. Both
derivatives are **central finite differences**, taken exactly as
:mod:`pmcprg.copulas._stderr` documents: in the unconstrained working
coordinate ψ of the one-parameter families, ψ = log(τ − a) − log(b − τ) on
the registered range [a, b] — 2·atanh(τ) for Gauss (a, b) = (−1, 1),
log(τ/(1 − τ)) for Clayton (a, b) = (0, 1) — never in τ itself, so a step
never leaves the admissible range.

Concretely, for the pair (i, j), only its own τ is perturbed; every other
model parameter (prior, other pairs' τ, margins) is held at the ICE fit.
This is a **partial** Oakes computation — it profiles out the rest of the
parameter vector rather than inverting a joint Hessian — appropriate for a
pilot on one scalar parameter, not for a joint covariance of several.

Cost
----
One E-step (:func:`~pmcprg.pmc.inference.precompute_weights`,
:func:`~pmcprg.pmc.inference.forward`, :func:`~pmcprg.pmc.inference.backward`,
:func:`~pmcprg.pmc.inference.joint_posteriors`) at the fitted model — to get
ξ̂_n(i, j) if the caller has not already got it — plus two more at
τ′ = τ(ψ̂ ± h_ψ) for the mixed term. For K = 2 forward-backward is O(N),
so this is a handful of extra passes over the data, not an extra ICE run.

Step size and error
--------------------
``H_PSI = 1e-4`` (:data:`pmcprg.copulas._stderr._H_PSI`, same value): the
curvature (term 1) has truncation error O(h²) ≈ 10⁻⁸ and rounding error
O(ε_mach · |Q|) / h² which, for Q ~ n_eff · O(1) and double precision,
stays several orders below the O(1/√n_eff) sampling error of τ̂ itself for
any n_eff used here (tens to a few thousand). The mixed term (term 2) is a
first difference of a first difference (S(ψ′) is already ∂Q/∂θ at θ = θ̂);
its truncation error is also O(h²), and its rounding error is dominated by
the Monte-Carlo noise of the E-step re-run's ξ (deterministic given θ′, so
no extra randomness — the only floating-point noise is the forward-backward
normalisation, ~ε_mach relative).

References
----------
* Oakes, D. (1999). Direct calculation of the information matrix via the EM
  algorithm. *J. R. Statist. Soc. B* 61(2), 479-482.
  doi:10.1111/1467-9868.00188
* Louis, T. A. (1982). Finding the observed information matrix when using
  the EM algorithm. *J. R. Statist. Soc. B* 44(2), 226-233.
* Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977). Maximum likelihood
  from incomplete data via the EM algorithm. *J. R. Statist. Soc. B* 39(1),
  1-38.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.copulas._stderr import standard_errors as _naive_standard_errors
from pmcprg.pmc.ice import _margin_cdfs, _pair_pseudo_obs, _resolve_candidate
from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = ["OakesResult", "ice_oakes_tau_se"]

#: Central-difference step in the working coordinate ψ — same value as
#: :data:`pmcprg.copulas._stderr._H_PSI` (see the module docstring).
H_PSI: float = 1e-4

#: Families this pilot supports (audit FR-4 scope: one-parameter, validated).
SUPPORTED_FAMILIES: tuple[str, ...] = ("Gauss", "Clayton")

#: τ̂ within this many optimiser pads (:data:`pmcprg.copulas._base.TAU_PAD_REL`)
#: of a range end is "at the boundary" — same convention as
#: :mod:`pmcprg.copulas._stderr`'s ``_BOUNDARY_PADS``.
_BOUNDARY_PADS: float = 2.0


@dataclass(frozen=True)
class OakesResult:
    """Oakes' observed-information standard error of one ICE copula τ.

    Fields
    ------
    pair        : (i, j) of the copula block.
    family      : SHORT_NAME of the copula (``"Gauss"`` or ``"Clayton"``).
    tau_hat     : the ICE-fitted τ.
    se_tau      : Oakes' standard error of τ̂ (NaN if ``at_boundary`` or the
                  observed information is not positive).
    se_tau_naive: the naive :mod:`pmcprg.copulas._stderr` sandwich applied to
                  the final ICE weights ξ̂_n(i, j) as if they were fixed —
                  the quantity FR-4 flags as understating the uncertainty.
    info_complete : − ∂²Q(θ|θ̂)/∂θ² |_{θ̂} in ψ — the complete-data term.
    info_missing  : − ∂²Q(θ|θ')/∂θ∂θ' |_{θ̂} in ψ — Oakes' correction
                  (Louis' "missing information"; typically negative, since it
                  *reduces* the complete-data information down to the
                  observed one).
    info_psi    : ``info_complete + info_missing`` — the observed information
                  in ψ. ``se_tau`` is this converted to τ by the delta method.
    n_eff       : Σ_n ξ̂_n(i, j).
    at_boundary : τ̂ within :data:`_BOUNDARY_PADS` working-coordinate steps of
                  the registered range [a, b] — ψ̂ is not usable.
    """

    pair: tuple[int, int]
    family: str
    tau_hat: float
    se_tau: float
    se_tau_naive: float
    info_complete: float
    info_missing: float
    info_psi: float
    n_eff: float
    at_boundary: bool


def _psi_of_tau(tau: float, a: float, b: float) -> float:
    return math.log(tau - a) - math.log(b - tau)


def _tau_of_psi(psi: float, a: float, b: float) -> float:
    # a + (b - a) * expit(psi), guarded against overflow of exp(-psi).
    if psi >= 0.0:
        z = math.exp(-psi)
        return a + (b - a) / (1.0 + z)
    z = math.exp(psi)
    return a + (b - a) * z / (1.0 + z)


def _e_step_xi_pair(model: PMCModel, Y: np.ndarray, ii: int, jj: int) -> np.ndarray:
    """Run one complete E-step on ``model`` and return ξ_n(ii, jj), n = 0…N-2."""
    W, f_pdf = precompute_weights(model, Y)
    alpha_hat, _ = forward(model, Y, W=W, f_pdf=f_pdf)
    beta_hat = backward(model, Y, W=W)
    xi = joint_posteriors(alpha_hat, W, beta_hat)
    return xi[:, ii, jj]


def _with_pair_tau(model: PMCModel, ii: int, jj: int, tau: float) -> PMCModel:
    """Deep-copy ``model``'s raw dict with pair (ii, jj)'s copula τ replaced."""
    raw = model.raw
    found = False
    for blk in raw.get("copulas", []):
        if int(blk["i"]) == ii and int(blk["j"]) == jj:
            blk["tau"] = float(tau)
            found = True
            break
    if not found:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}) in this model.")
    return PMCModel.from_dict(raw)


def _oakes_one_pair(
    model: PMCModel,
    Y: np.ndarray,
    ii: int,
    jj: int,
    *,
    h_psi: float,
) -> OakesResult:
    blocks = {(int(b["i"]), int(b["j"])): b for b in model.copula_blocks()}
    if (ii, jj) not in blocks:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}).")
    blk = blocks[(ii, jj)]
    family = str(blk["name"])
    if family not in SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"ice_oakes_tau_se: family {family!r} for pair ({ii}, {jj}) is not "
            f"one of the validated one-parameter families {SUPPORTED_FAMILIES} "
            f"(audit FR-4 pilot scope)."
        )
    entry, cls = _resolve_candidate(family)
    a, b = (float(x) for x in entry.value.TAU_MIN_MAX)
    tau_hat = float(blk["tau"])

    pad = _BOUNDARY_PADS * max(TAU_PAD_REL * (b - a), TAU_PAD_ABS)
    at_boundary = not (a + pad < tau_hat < b - pad)
    if at_boundary:
        logger.warning(
            "ice_oakes_tau_se: pair (i=%d, j=%d) τ̂=%.6g is at the boundary of "
            "%s's range [%.4g, %.4g] — Oakes' SE is not defined there "
            "(Self & Liang 1987); returning NaN.", ii, jj, tau_hat, family, a, b,
        )
        naive = math.nan
        try:
            xi_hat = _e_step_xi_pair(model, Y, ii, jj)
            u_b, v_b = _pseudo_obs(model, Y, ii, jj)
            naive = _naive_standard_errors(cls(tau_k=tau_hat), np.column_stack((u_b, v_b)),
                                            weights=xi_hat, method="mle",
                                            ranks=False).se["tau_k"]
        except Exception:
            pass
        return OakesResult((ii, jj), family, tau_hat, math.nan, naive,
                            math.nan, math.nan, math.nan, math.nan, True)

    psi_hat = _psi_of_tau(tau_hat, a, b)
    u, v = _pseudo_obs(model, Y, ii, jj)
    uv = np.column_stack((u, v))

    xi_hat = _e_step_xi_pair(model, Y, ii, jj)
    n_eff = float(xi_hat.sum())

    # ── term 1: complete-data curvature, − ∂²Q(θ|θ̂)/∂θ² at fixed ξ̂ ──────
    logpdf_0 = cls(tau_k=tau_hat).logpdf_array(uv)
    logpdf_p = cls(tau_k=_tau_of_psi(psi_hat + h_psi, a, b)).logpdf_array(uv)
    logpdf_m = cls(tau_k=_tau_of_psi(psi_hat - h_psi, a, b)).logpdf_array(uv)
    curvature = (logpdf_p - 2.0 * logpdf_0 + logpdf_m) / (h_psi * h_psi)
    info_complete = -float(np.dot(xi_hat, curvature))

    # φ_n = ∂ log c_θ(û_n, v̂_n)/∂θ at θ = θ̂ (central difference, same stencil).
    phi = (logpdf_p - logpdf_m) / (2.0 * h_psi)

    # ── term 2: Oakes' correction, − ∂²Q(θ|θ')/∂θ∂θ' at θ=θ'=θ̂ ──────────
    # S(ψ') = Σ_n ξ_n(ψ') · φ_n  is ∂Q(θ|θ')/∂θ at θ=θ̂, θ'=ψ'; one more
    # E-step per perturbation gives ξ_n(ψ').
    model_p = _with_pair_tau(model, ii, jj, _tau_of_psi(psi_hat + h_psi, a, b))
    model_m = _with_pair_tau(model, ii, jj, _tau_of_psi(psi_hat - h_psi, a, b))
    xi_p = _e_step_xi_pair(model_p, Y, ii, jj)
    xi_m = _e_step_xi_pair(model_m, Y, ii, jj)
    s_p = float(np.dot(xi_p, phi))
    s_m = float(np.dot(xi_m, phi))
    mixed = (s_p - s_m) / (2.0 * h_psi)
    info_missing = -mixed

    info_psi = info_complete + info_missing
    dtau_dpsi = (tau_hat - a) * (b - tau_hat) / (b - a)
    if info_psi > 0.0:
        se_tau = abs(dtau_dpsi) / math.sqrt(info_psi)
    else:
        logger.warning(
            "ice_oakes_tau_se: pair (i=%d, j=%d) observed information %.4g "
            "is not positive (complete=%.4g, missing=%.4g) — returning NaN.",
            ii, jj, info_psi, info_complete, info_missing,
        )
        se_tau = math.nan

    naive = _naive_standard_errors(
        cls(tau_k=tau_hat), uv, weights=xi_hat, method="mle", ranks=False,
    ).se["tau_k"]

    return OakesResult(
        (ii, jj), family, tau_hat, se_tau, naive,
        info_complete, info_missing, info_psi, n_eff, False,
    )


def _pseudo_obs(model: PMCModel, Y: np.ndarray, ii: int, jj: int) -> tuple[np.ndarray, np.ndarray]:
    f_cdf = _margin_cdfs(model, Y)
    return _pair_pseudo_obs(model, f_cdf, ii, jj)


def ice_oakes_tau_se(
    model: PMCModel,
    Y: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
    *,
    h_psi: float = H_PSI,
) -> dict[tuple[int, int], OakesResult]:
    """Oakes' (1999) standard error of the copula τ of each pair in ``pairs``.

    Parameters
    ----------
    model : PMCModel — the fitted model returned by :func:`pmcprg.pmc.ice.ice`
            (margins held fixed at that fit; ``fit_margins=False`` in the ICE
            run this pilot was validated against — see the module docstring).
    Y     : the observation sequence the ICE run was fitted on. Must be
            complete (no missing rows) — the pilot does not cover
            :mod:`pmcprg.pmc.gaps`.
    pairs : which copula blocks to compute; ``None`` = every block whose
            family is in :data:`SUPPORTED_FAMILIES`.
    h_psi : central-difference step in the working coordinate ψ (see the
            module docstring for the truncation/rounding trade-off).

    Returns
    -------
    ``{(i, j): OakesResult}``.

    Raises
    ------
    NotImplementedError : a requested pair's family is not
        :data:`SUPPORTED_FAMILIES` (BB1, Student, Frank, … — the audit's
        cheapest-option pilot is one-parameter Gauss/Clayton only).
    """
    if not model.variant.uses_copula:
        raise ValueError(f"Variant {model.variant.value} does not use copulas.")
    Y = np.asarray(Y, dtype=float)
    if np.any(~np.isfinite(Y)):
        raise ValueError(
            "ice_oakes_tau_se: Y has missing (non-finite) rows — the Oakes "
            "pilot does not cover gap variants yet (see the module docstring)."
        )

    all_blocks = model.copula_blocks()
    if pairs is None:
        pairs = [(int(b["i"]), int(b["j"])) for b in all_blocks
                  if str(b["name"]) in SUPPORTED_FAMILIES]

    return {(ii, jj): _oakes_one_pair(model, Y, ii, jj, h_psi=h_psi) for ii, jj in pairs}
