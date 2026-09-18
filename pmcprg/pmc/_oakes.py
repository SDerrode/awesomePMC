"""
pmcprg.pmc._oakes — Oakes' (1999) observed information for a copula τ fitted
*inside* ICE (AUDIT_COPULES FR-4, "Reste": standard errors inside ICE).

Scope
-----
K = 2 state models; one pair (i, j) of :meth:`PMCModel.copula_blocks` at a
time. Margins are held fixed at the model handed in (``fit_margins=False``
in the ICE run that produced it) — extending this to margins re-estimated
inside ICE, or to the prior / other pairs' parameters, is future work; see
the module docstring of :mod:`pmcprg.pmc.ice` for how those enter the M-step.

Two families of copula are supported, in two code paths:

* **one-parameter** (Gauss, Clayton) — the original pilot, scalar
  throughout: a scalar ψ, a scalar curvature and mixed term, a scalar
  :class:`OakesResult`. This path is untouched by the generalisation below
  (same functions, same numbers, to the last bit).
* **two-parameter** (BB1, Student) — generalises every scalar quantity of
  the pilot to its 2×2 matrix equivalent (see "Generalisation to two
  parameters" below), returning an :class:`OakesResultMulti` that mirrors
  the ``names``/``estimate``/``se``/``cov`` convention of
  :class:`pmcprg.copulas._stderr.StandardErrors` — the "outside ICE"
  sibling this pilot already extends. It reuses that module's ``_Spec`` /
  ``_spec_of`` (the same working coordinate ψ — ``(log θ, log(δ − 1))`` for
  BB1, ``(atanh τ, log(ν − 2))`` for Student — and the same analytic
  Jacobian ``jac_psi`` from ψ to the reported quantities) rather than
  reimplementing that bookkeeping. The dispatch between the two paths is by
  family in :func:`ice_oakes_tau_se` / :func:`_oakes_one_pair`; callers only
  need to know that a result for a one-parameter family is an
  :class:`OakesResult` and for a two-parameter family an
  :class:`OakesResultMulti`.

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
τ′ = τ(ψ̂ ± h_ψ) for the mixed term (one-parameter families), or four more,
one pair per working coordinate, for two-parameter families (below). For
K = 2 forward-backward is O(N), so this is a handful of extra passes over
the data, not an extra ICE run.

Generalisation to two parameters (BB1, Student)
------------------------------------------------
Both scalar terms of Oakes' identity become 2×2 matrices in the working
coordinate ψ = (ψ₁, ψ₂) of the family (BB1: ``log θ, log(δ − 1)``; Student:
``atanh τ, log(ν − 2)`` — :func:`pmcprg.copulas._stderr._spec_of`):

* **term 1** (complete-data curvature) — the Hessian of
  ``Q(θ|θ̂) = Σ_n ξ̂_n(i,j) log c_ψ(û_n, v̂_n)`` in ψ at ψ̂, a 2×2 matrix
  whose diagonal is the same central second difference as the scalar case
  (one extra pair of ``logpdf_array`` evaluations per coordinate) and whose
  off-diagonal is the standard 4-point mixed central difference (one extra
  quadruple of evaluations) — no extra E-step, exactly as term 1 needs none
  in the scalar case (:func:`_hessian_and_phi`).
* **term 2** (Oakes' correction) — for two parameters the correction is the
  Jacobian of ``S(ψ') = Σ_n ξ_n(ψ') φ_n`` (now a 2-vector) with respect to
  ψ' (now 2-dimensional): a 2×2 matrix, ``M[a, c] = ∂S_a/∂ψ'_c``. Each
  *column* c needs one E-step at ψ' = ψ̂ ± h·e_c with the density held at
  ψ̂ (:func:`_mixed_info`) — 2 columns × 2 perturbations = 4 extra E-steps,
  vs. 2 for a one-parameter family, as the pilot's own report anticipated
  ("up to 4 extra E-step evaluations per pair instead of 2"). A smarter
  scheme (e.g. a one-sided difference, halving the E-step count at the cost
  of O(h) instead of O(h²) truncation error) was not needed: even 4 extra
  full forward-backward passes at K = 2, N in the hundreds, is a small
  fraction of one ICE iteration's cost, and the coverage study below shows
  no sign that the O(h²) central scheme needs improving. ``M`` is not
  symmetric by construction (unlike term 1, it is a directional derivative
  of a vector-valued map, not a Hessian) but Oakes' theorem implies the
  *combined* observed information ``info_complete + info_missing`` is
  symmetric; ``M`` is symmetrised, ``0.5·(M + Mᵀ)``, before use — the
  antisymmetric part measured in the validation below is consistently two
  to three orders of magnitude below the symmetric part, i.e. numerical
  noise, not a sign of a genuine asymmetry the averaging would hide.

The resulting 2×2 observed information in ψ is inverted (after an
eigenvalue check, as :mod:`pmcprg.copulas._stderr` does for its own
sandwich) and carried to the reported quantities — τ, δ, θ for BB1; τ, ν, ρ
for Student — by :attr:`pmcprg.copulas._stderr._Spec.jac_psi`, the same
analytic Jacobian the naive sandwich already uses. This is the one part of
the two-parameter path that is *not* reimplemented: reusing it is exactly
what keeps :mod:`pmcprg.pmc._oakes` a sibling of
:mod:`pmcprg.copulas._stderr` rather than a divergent copy.

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

Empirical copula margins (``copula_margins = "empirical"``, AUDIT_COPULES FR-7 a)
-----------------------------------------------------------------------------------
Everything here is derived for ICE's parametric copula step — pseudo-
observations built from the model's own F. With ICE's ``copula_margins =
"empirical"`` the copula parameter solves a posterior-weighted *rank*
pseudo-likelihood instead (:func:`pmcprg.pmc.ice._empirical_margin_cdfs`),
whose asymptotic variance carries an extra term from the estimated margins
(Chen & Fan 2006, doi:10.1016/j.jeconom.2005.03.004) that is not computed
here, so this would be the standard error of another estimator. ICE and SEM
record a non-default value in the fitted model's ``[ice]`` / ``[sem]`` table,
and :func:`ice_oakes_tau_se` raises ``NotImplementedError`` on such a model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.copulas._stderr import _spec_of
from pmcprg.copulas._stderr import standard_errors as _naive_standard_errors
from pmcprg.pmc.ice import (
    _margin_cdfs,
    _pair_pseudo_obs,
    _refuse_nonparametric_copula_margins,
    _resolve_candidate,
)
from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = ["OakesResult", "OakesResultMulti", "ice_oakes_tau_se"]

#: Central-difference step in the working coordinate ψ — same value as
#: :data:`pmcprg.copulas._stderr._H_PSI` (see the module docstring).
H_PSI: float = 1e-4

#: One-parameter families — the original pilot's scalar code path
#: (:func:`_oakes_one_pair_scalar`, :class:`OakesResult`), untouched.
_ONE_PARAM_FAMILIES: tuple[str, ...] = ("Gauss", "Clayton")

#: Two-parameter families — the 2×2-matrix generalisation
#: (:func:`_oakes_one_pair_multi`, :class:`OakesResultMulti`).
_TWO_PARAM_FAMILIES: tuple[str, ...] = ("BB1", "Student")

#: Every family this module supports (audit FR-4 scope).
SUPPORTED_FAMILIES: tuple[str, ...] = _ONE_PARAM_FAMILIES + _TWO_PARAM_FAMILIES

#: Extra (non-τ) copula-block keys of each two-parameter family, in the order
#: :func:`pmcprg.copulas._stderr._spec_of` reports them after ``tau_k``.
_EXTRA_PARAM_KEYS: dict[str, tuple[str, ...]] = {"BB1": ("delta",), "Student": ("df",)}

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


@dataclass(frozen=True)
class OakesResultMulti:
    """Oakes' observed information for a two-parameter ICE copula (BB1, Student).

    The 2×2-matrix generalisation of :class:`OakesResult` (see the module
    docstring, "Generalisation to two parameters"). Its shape mirrors
    :class:`pmcprg.copulas._stderr.StandardErrors` — ``names``/``estimate``/
    ``se``/``cov`` — rather than a bespoke layout, since this *is* that
    class's sibling computation, one level deeper (accounting for the
    ICE weights' own sampling uncertainty).

    Fields
    ------
    pair        : (i, j) of the copula block.
    family      : SHORT_NAME of the copula (``"BB1"`` or ``"Student"``).
    names       : reported quantities, in the order of ``estimate``/``se``/
                  ``cov`` — ``("tau_k", "delta", "theta")`` for BB1,
                  ``("tau_k", "df", "rho")`` for Student (same order and
                  names :attr:`pmcprg.copulas._stderr.StandardErrors.names`
                  uses for these families).
    estimate    : ``{name: value}`` at the ICE fit.
    se          : Oakes' standard error of each quantity (NaN at a boundary,
                  or where the observed information is not positive-definite).
    se_naive    : the naive :mod:`pmcprg.copulas._stderr` sandwich applied to
                  the final ICE weights ξ̂_n(i, j) as if they were fixed —
                  the quantity FR-4 flags as understating the uncertainty.
    cov         : Oakes' covariance matrix of ``names`` (NaN where ``se`` is).
    cov_naive   : the naive sandwich's covariance matrix of ``names``.
    info_complete : − ∂²Q(θ|θ̂)/∂ψ∂ψᵀ |_{θ̂} — the complete-data term, 2×2,
                  in the family's working coordinate ψ.
    info_missing  : − ∂²Q(θ|θ')/∂ψ∂ψ'ᵀ |_{θ̂} — Oakes' correction, 2×2, in ψ
                  (symmetrised; see the module docstring for why it need not
                  be symmetric before that).
    info_psi    : ``info_complete + info_missing`` — the observed information
                  in ψ, 2×2. ``cov`` is this inverted and carried to
                  ``names`` by the family's analytic Jacobian
                  (:attr:`pmcprg.copulas._stderr._Spec.jac_psi`).
    n_eff       : Σ_n ξ̂_n(i, j).
    at_boundary : the estimate is within the boundary tolerance of
                  :func:`pmcprg.copulas._stderr._spec_of` on any of its
                  coordinates — ψ̂ is not usable (Self & Liang 1987).
    boundary    : one sentence per boundary condition met (empty if interior)
                  — same convention and wording as
                  :attr:`pmcprg.copulas._stderr.StandardErrors.boundary`.
    """

    pair: tuple[int, int]
    family: str
    names: tuple
    estimate: dict
    se: dict
    se_naive: dict
    cov: np.ndarray
    cov_naive: np.ndarray
    info_complete: np.ndarray
    info_missing: np.ndarray
    info_psi: np.ndarray
    n_eff: float
    at_boundary: bool
    boundary: tuple

    def ci(self, level: float = 0.95, name: str = "tau_k") -> tuple[float, float]:
        """Wald interval for ``name`` — see :meth:`StandardErrors.ci`."""
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


def _with_pair_params(
    model: PMCModel, ii: int, jj: int, tau: float, extra: dict | None = None,
) -> PMCModel:
    """Deep-copy ``model``'s raw dict with pair (ii, jj)'s copula τ and any
    extra (non-τ) parameters (BB1's δ, Student's ν) replaced.

    Generalises :func:`_with_pair_tau` to two-parameter families; the
    one-parameter path keeps using :func:`_with_pair_tau` directly (bit-
    identical to the pilot — see the module docstring).
    """
    raw = model.raw
    found = False
    for blk in raw.get("copulas", []):
        if int(blk["i"]) == ii and int(blk["j"]) == jj:
            blk["tau"] = float(tau)
            for k, v in (extra or {}).items():
                blk[k] = float(v)
            found = True
            break
    if not found:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}) in this model.")
    return PMCModel.from_dict(raw)


def _tau_extra_of_copula(cop, family: str) -> tuple[float, dict]:
    """``(τ, {extra params})`` of a fitted copula instance, keyed as the raw
    ``[[copulas]]`` block would store them (:data:`_EXTRA_PARAM_KEYS`)."""
    tau = float(cop.params["tau_k"])
    if family == "BB1":
        return tau, {"delta": float(cop.delta)}
    if family == "Student":
        return tau, {"df": float(cop.df)}
    return tau, {}


def _build_copula_from_blk(cls, family: str, blk: dict):
    """A copula instance from a raw ``[[copulas]]`` block — τ plus any of
    :data:`_EXTRA_PARAM_KEYS` present."""
    kwargs = {"tau_k": float(blk["tau"])}
    for k in _EXTRA_PARAM_KEYS.get(family, ()):
        if k in blk:
            kwargs[k] = float(blk[k])
    return cls(**kwargs)


def _oakes_one_pair(
    model: PMCModel,
    Y: np.ndarray,
    ii: int,
    jj: int,
    *,
    h_psi: float,
) -> "OakesResult | OakesResultMulti":
    """Dispatch to the scalar (one-parameter) or matrix (two-parameter) path."""
    blocks = {(int(b["i"]), int(b["j"])): b for b in model.copula_blocks()}
    if (ii, jj) not in blocks:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}).")
    family = str(blocks[(ii, jj)]["name"])
    if family in _ONE_PARAM_FAMILIES:
        return _oakes_one_pair_scalar(model, Y, ii, jj, h_psi=h_psi)
    if family in _TWO_PARAM_FAMILIES:
        return _oakes_one_pair_multi(model, Y, ii, jj, h_psi=h_psi)
    raise NotImplementedError(
        f"ice_oakes_tau_se: family {family!r} for pair ({ii}, {jj}) is not "
        f"one of the supported families {SUPPORTED_FAMILIES} (audit FR-4 scope)."
    )


def _oakes_one_pair_scalar(
    model: PMCModel,
    Y: np.ndarray,
    ii: int,
    jj: int,
    *,
    h_psi: float,
) -> OakesResult:
    """One-parameter families (Gauss, Clayton) — the original pilot, unchanged."""
    blocks = {(int(b["i"]), int(b["j"])): b for b in model.copula_blocks()}
    if (ii, jj) not in blocks:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}).")
    blk = blocks[(ii, jj)]
    family = str(blk["name"])
    if family not in _ONE_PARAM_FAMILIES:
        raise NotImplementedError(
            f"_oakes_one_pair_scalar: family {family!r} for pair ({ii}, {jj}) is not "
            f"one of the one-parameter families {_ONE_PARAM_FAMILIES}."
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


# ---------------------------------------------------------------------------
# Two-parameter families (BB1, Student): the matrix generalisation
# ---------------------------------------------------------------------------

def _logc_derivatives(spec, uv: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation ``∂ log c/∂ψ`` (n, p) and ``∂² log c/∂ψ∂ψᵀ`` (n, p, p).

    ``p = spec.psi.size`` — 1 for a one-parameter family, 2 for BB1/Student
    (this function is generic in p; only the two-parameter path calls it in
    production, but the one-parameter case is exercised by the sanity check
    of :mod:`pmcprg.tests.test_fr4_ice_oakes_bb1_student` that this matrix
    code reproduces :func:`_oakes_one_pair_scalar`'s numbers). Central
    differences of ``spec.build(psi).logpdf_array(uv)``, same stencil (and,
    for a one-parameter family, same step ``h``) as the scalar path and as
    :func:`pmcprg.copulas._stderr._derivatives`, which this mirrors: one
    second difference per coordinate on the diagonal, the standard 4-point
    mixed central difference off it.

    Split out of :func:`_hessian_and_phi` (which weights the curvature by ξ
    and sums it, Oakes' "term 1") so that the *per-observation* curvature is
    available to :mod:`pmcprg.pmc._lystig_hughes`, whose recursion needs
    ``∂² log W[n]/∂ψ∂ψᵀ`` at every n rather than one ξ-weighted total. The
    arithmetic is unchanged, operation for operation.
    """
    psi = spec.psi
    p = psi.size
    eye = np.eye(p)

    def ll(pp: np.ndarray) -> np.ndarray:
        return np.asarray(spec.build(pp).logpdf_array(uv), dtype=float)

    l0 = ll(psi)
    lp = [ll(psi + h * eye[a]) for a in range(p)]
    lm = [ll(psi - h * eye[a]) for a in range(p)]
    n = uv.shape[0]
    phi = np.empty((n, p))
    curvature = np.empty((n, p, p))
    for a in range(p):
        phi[:, a] = (lp[a] - lm[a]) / (2.0 * h)
        curvature[:, a, a] = (lp[a] - 2.0 * l0 + lm[a]) / (h * h)
        for c in range(a + 1, p):
            mixed = (ll(psi + h * (eye[a] + eye[c])) - ll(psi + h * (eye[a] - eye[c]))
                     - ll(psi - h * (eye[a] - eye[c])) + ll(psi - h * (eye[a] + eye[c])))
            curvature[:, a, c] = curvature[:, c, a] = mixed / (4.0 * h * h)
    return phi, curvature


def _hessian_and_phi(spec, uv: np.ndarray, xi: np.ndarray, h: float) -> tuple[np.ndarray, np.ndarray]:
    """φ_n (n, p) and the ξ-weighted complete-data curvature (p, p) at ``spec.psi``.

    The derivatives themselves come from :func:`_logc_derivatives`; here they
    are dotted with ``xi`` (a *sum*, "term 1" of Oakes' identity) rather than
    averaged with weights ``w`` (a *mean*, :mod:`pmcprg.copulas._stderr`'s
    sandwich curvature ``B``), and without that module's ``ranks`` margin
    corrections (not needed here: this pilot holds margins fixed).
    """
    phi, curvature = _logc_derivatives(spec, uv, h)
    info_complete = -np.einsum("n,nab->ab", xi, curvature)
    info_complete = 0.5 * (info_complete + info_complete.T)
    return phi, info_complete


def _mixed_info(
    model: PMCModel, Y: np.ndarray, ii: int, jj: int, family: str, spec, phi: np.ndarray, h: float,
) -> np.ndarray:
    """Oakes' correction, − ∂²Q(θ|θ')/∂ψ∂ψ'ᵀ at θ = θ' = θ̂, as a (p, p) matrix.

    Column c is one central difference of ``S(ψ') = ξ(ψ')ᵀ φ`` (a p-vector)
    across ψ'_c ± h — one E-step per perturbation, ``2p`` in total (``p = 2``
    for BB1/Student: 4 extra E-steps, vs. 2 for a one-parameter family — see
    the module docstring). The result is not symmetric by construction
    (unlike ``info_complete``, it differentiates a vector map in one
    direction only); it is symmetrised before use, on the grounds given in
    the module docstring (Oakes' theorem: the sum with ``info_complete`` is
    symmetric; the antisymmetric part measured here is noise, not signal).
    """
    p = spec.psi.size
    eye = np.eye(p)
    s_plus = np.empty((p, p))   # s_plus[c, a] = S_a(ψ' = ψ̂ + h·e_c)
    s_minus = np.empty((p, p))
    for c in range(p):
        cop_p = spec.build(spec.psi + h * eye[c])
        cop_m = spec.build(spec.psi - h * eye[c])
        tau_p, extra_p = _tau_extra_of_copula(cop_p, family)
        tau_m, extra_m = _tau_extra_of_copula(cop_m, family)
        xi_p = _e_step_xi_pair(_with_pair_params(model, ii, jj, tau_p, extra_p), Y, ii, jj)
        xi_m = _e_step_xi_pair(_with_pair_params(model, ii, jj, tau_m, extra_m), Y, ii, jj)
        s_plus[c, :] = xi_p @ phi
        s_minus[c, :] = xi_m @ phi
    # mixed[c, a] = ∂S_a/∂ψ'_c  →  info_missing[a, c] = −mixed[c, a]
    mixed = (s_plus - s_minus) / (2.0 * h)
    info_missing = -mixed.T
    return 0.5 * (info_missing + info_missing.T)


def _oakes_one_pair_multi(
    model: PMCModel,
    Y: np.ndarray,
    ii: int,
    jj: int,
    *,
    h_psi: float,
) -> OakesResultMulti:
    """Two-parameter families (BB1, Student) — the 2×2-matrix generalisation.

    Generic in the number of working coordinates ``p`` (see
    :func:`_hessian_and_phi`'s docstring): production only calls this for
    BB1/Student (``p = 2``), but nothing here assumes ``p = 2`` specifically.
    """
    blocks = {(int(b["i"]), int(b["j"])): b for b in model.copula_blocks()}
    if (ii, jj) not in blocks:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}).")
    blk = blocks[(ii, jj)]
    family = str(blk["name"])
    _entry, cls = _resolve_candidate(family)
    copula0 = _build_copula_from_blk(cls, family, blk)
    spec = _spec_of(copula0)
    m = len(spec.names)

    u, v = _pseudo_obs(model, Y, ii, jj)
    uv = np.column_stack((u, v))
    xi_hat = _e_step_xi_pair(model, Y, ii, jj)
    n_eff = float(xi_hat.sum())

    naive = _naive_standard_errors(copula0, uv, weights=xi_hat, method="mle", ranks=False)

    if spec.boundary:
        logger.warning(
            "ice_oakes_tau_se: pair (i=%d, j=%d) %s estimate is at a boundary "
            "(%s) — Oakes' SE is not defined there (Self & Liang 1987); "
            "returning NaN.", ii, jj, family, "; ".join(spec.boundary),
        )
        nan_pp = np.full((spec.psi.size, spec.psi.size), math.nan)
        return OakesResultMulti(
            (ii, jj), family, spec.names, dict(spec.estimate),
            {k: math.nan for k in spec.names}, dict(naive.se),
            np.full((m, m), math.nan), naive.cov,
            nan_pp, nan_pp, nan_pp, n_eff, True, spec.boundary,
        )

    phi, info_complete = _hessian_and_phi(spec, uv, xi_hat, h_psi)
    info_missing = _mixed_info(model, Y, ii, jj, family, spec, phi, h_psi)
    info_psi = info_complete + info_missing

    eig = np.linalg.eigvalsh(info_psi)
    if np.all(eig > 0.0):
        cov_psi = np.linalg.inv(info_psi)
        cov = spec.jac_psi @ cov_psi @ spec.jac_psi.T
        se = {name: (float(math.sqrt(cov[k, k])) if cov[k, k] >= 0.0 else math.nan)
              for k, name in enumerate(spec.names)}
    else:
        logger.warning(
            "ice_oakes_tau_se: pair (i=%d, j=%d) %s observed information is not "
            "positive-definite (eigenvalues %s) — returning NaN.", ii, jj, family, eig,
        )
        cov = np.full((m, m), math.nan)
        se = {name: math.nan for name in spec.names}

    return OakesResultMulti(
        (ii, jj), family, spec.names, dict(spec.estimate), se, dict(naive.se),
        cov, naive.cov, info_complete, info_missing, info_psi, n_eff, False, (),
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
) -> "dict[tuple[int, int], OakesResult | OakesResultMulti]":
    """Oakes' (1999) standard error(s) of the copula parameter(s) of each pair.

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
    ``{(i, j): result}`` — an :class:`OakesResult` for a one-parameter pair
    (Gauss, Clayton) or an :class:`OakesResultMulti` for a two-parameter one
    (BB1, Student); see the module docstring for which is which.

    Raises
    ------
    NotImplementedError : a requested pair's family is not
        :data:`SUPPORTED_FAMILIES` (Frank, Joe, GH, … — every family other
        than Gauss, Clayton, BB1, Student is out of this audit's scope); or
        the model was fitted with ``copula_margins = "empirical"`` (module
        docstring, last section).
    """
    if not model.variant.uses_copula:
        raise ValueError(f"Variant {model.variant.value} does not use copulas.")
    _refuse_nonparametric_copula_margins(model, "ice_oakes_tau_se")
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
