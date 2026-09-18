"""
pmcprg.pmc._godambe — Godambe's (IFM) information for a copula τ fitted
*inside* ICE, alongside margins re-estimated by weighted MLE (AUDIT_COPULES
FR-4, "Reste": standard errors inside ICE, second of the three ranked
options after Oakes' identity — :mod:`pmcprg.pmc._oakes`).

Scope
-----
K = 2 state models, ``margin_structure == "state"`` (state-indexed Gaussian
margins, ``dist == "norm"``, re-estimated by weighted MLE every M-step —
:func:`pmcprg.pmc.ice._fit_margin_weighted`'s closed-form path), **Gauss**
copula only, one pair (i, j) of :meth:`PMCModel.copula_blocks` at a time.
This is deliberately the simplest well-defined case where the IFM
correction below is *needed*: with ``fit_margins=False`` (margins held
fixed at the model handed in) there is no margin re-estimation error to
account for, and :mod:`pmcprg.pmc._oakes` (or even the naive
:mod:`pmcprg.copulas._stderr` sandwich) already suffices — see that
module's own scope note, which is silent on margin re-estimation for the
same reason. Extending this module to Clayton/BB1/Student, to pair margins,
or to non-Gaussian state margins is future work; see "What would extending
this need" at the end of this docstring.

Why Oakes' identity, alone, is not enough here
-----------------------------------------------
:mod:`pmcprg.pmc._oakes` corrects the naive sandwich for exactly one source
of extra uncertainty: ξ̂_n(i, j) is itself an estimate from a latent state
sequence, not fixed data (Louis 1982; Oakes 1999). It does **not** correct
for a second, independent source that appears the moment ``fit_margins`` is
turned on: the margin parameters (μ_k, σ_k) that the copula's pseudo-
observations û_n = F_i(y_n), v̂_n = F_j(y_{n+1}) are built from are
*themselves* estimated, from the same data, by weighted MLE inside the same
M-step. Both stages' estimating equations are correlated (they share
γ_n(i), ξ_n(i, j) and the raw y_n) — exactly the situation Inference
Functions for Margins (IFM, Joe 2005, §2; Joe & Xu 1996) is built for: the
copula-parameter block of a naive two-block sandwich, applied as if the
margins were known, ignores a whole off-diagonal covariance term and is
neither the right variance nor (in general) conservative.

Joe's IFM / Godambe sandwich
-----------------------------
Write the joint parameter vector as ``(η, θ)`` — here η = the pair's own
margin parameters (μ_i, σ_i[, μ_j, σ_j] — one state block if the pair is
"diagonal", i = j, two if i ≠ j) and θ = τ, the copula parameter of the
pair. A two-stage M-estimator solves the joint estimating equation
``g(η, θ) = 0`` where ``g`` stacks every block's own score:

* the margin blocks' weighted log-likelihood score, one 2-vector per state
  k ∈ {i, j} used by the pair, weighted by the E-step's marginal posterior
  γ_n(k) — exactly the score whose root :func:`pmcprg.pmc.ice._fit_margin_weighted`
  finds in closed form for ``dist == "norm"`` (μ̂_k = Σ γ_n(k) y_n / Σ γ_n(k),
  σ̂_k² = Σ γ_n(k) (y_n − μ̂_k)² / Σ γ_n(k));
* the copula block's score φ_n = ∂ log c_θ(û_n, v̂_n)/∂θ, weighted by the
  E-step's joint posterior ξ_n(i, j) — the same object
  :mod:`pmcprg.copulas._stderr` and :mod:`pmcprg.pmc._oakes` already use.

Godambe's sandwich (Joe 2005, Eq. 2.3; the "sensitivity"/"variability"
decomposition of an estimating-equation-based, as opposed to likelihood-
based, M-estimator) is

    Avar(η̂, θ̂) = D⁻¹ M D⁻ᵀ,     D = E[∂g/∂(η, θ)ᵀ],     M = Var[g],

D estimated by minus the Jacobian of the *summed* estimating equation
(finite differences, or — for the Gaussian margin blocks — the elementary
closed form below) and M by the empirical (uncentred, since the summed
score is ≈ 0 at the joint fit — same convention :mod:`pmcprg.copulas._stderr`
uses for its own Ω) covariance of the per-observation score contributions.
``se_tau`` is the (θ, θ) entry of this joint covariance, in the working
coordinate ψ (:mod:`pmcprg.pmc._oakes`'s ψ = 2·atanh(τ) for Gauss), carried
to τ by the delta method exactly as Oakes' pilot does.

D is block lower-triangular
----------------------------
The margin log-density does not involve τ at all, so ∂(margin score)/∂τ is
*exactly* zero — the (η, θ) block of D is zero, not merely small:

    D = [ D_ηη   0    ]        D_ηη = −∂(margin score)/∂ηᵀ  (block-diagonal
        [ D_θη   D_θθ ]                per state, closed form, below)
                                D_θθ = −∂(Σ ξ_n φ_n)/∂θ       (finite
                                        difference — Oakes' "term 1", the
                                        complete-data curvature, unchanged)
                                D_θη = −∂(Σ ξ_n φ_n)/∂ηᵀ       (finite
                                        difference — **this is the IFM
                                        correction term itself**: how much
                                        the copula score moves when a margin
                                        parameter perturbs the pseudo-
                                        observations it is built from; zero
                                        would mean margins are exogenous, and
                                        the naive sandwich would already be
                                        correct)

so D⁻¹ is the standard block lower-triangular inverse and the (θ, θ) entry
of ``D⁻¹ M D⁻ᵀ`` is a scalar quadratic form in the row vector
``r = (−D_θθ⁻¹ D_θη D_ηη⁻¹, D_θθ⁻¹)``: ``Var(ψ̂) = r M rᵀ``. This is exactly
Joe's IFM correction (Joe 2005, Eq. 2.5, specialised to one nuisance block):
the naive sandwich is the ``D_θη = 0`` special case of the same formula.

Margin score, in closed form
-----------------------------
For state k with weighted Gaussian MLE (μ_k, σ_k) (:func:`pmcprg.pmc.ice._fit_margin_weighted`,
``dist == "norm"`` branch) and working coordinate ``(μ_k, log σ_k)`` (chosen,
as the module docstring of :mod:`pmcprg.pmc.ice` and its Godambe-neighbour
:mod:`pmcprg.pmc._oakes` do, so a perturbation never leaves σ > 0):

    score_μ (y_n)     = γ_n(k) · (y_n − μ_k) / σ_k²
    score_log σ (y_n) = γ_n(k) · [((y_n − μ_k)/σ_k)² − 1]

At the fitted (μ̂_k, σ̂_k) these are elementary to differentiate again (no
finite difference needed): with G_k = Σ_n γ_n(k),

    D_ηη[state k] = [ G_k / σ̂_k²        0    ]
                    [        0        2 G_k   ]

(the μ/log σ cross term is Σ_n γ_n(k) (y_n − μ̂_k) · (−2/σ̂_k²), which is
exactly −2/σ̂_k² times the μ score equation's own root, i.e. zero at μ̂_k —
the familiar orthogonality of a Gaussian MLE's mean and (log) scale). G_k
is summed over the **full** sample n = 0…N−1 — the actual estimating
equation the M-step's closed form solves — while the M-matrix's per-unit
contributions below use a per-transition alignment; see "A modelling
choice, and its limits".

A modelling choice, and its limits
------------------------------------
Godambe's ``M = Var[g]`` needs per-observation score contributions that sum
(to a good approximation) to the total score used for D. The copula score
is naturally indexed by transition n = 0…N−2 (ξ_n(i, j) couples y_n, y_{n+1});
the margin score is naturally indexed by time n = 0…N−1 (γ_n(k) is a
single-time posterior). To build the joint per-unit vector this module's M
needs, each transition n is assigned the margin contributions of *both*
endpoints it touches: γ_n(i)'s score at y_n (the pseudo-observation's "left"
half, u_n = F_i(y_n)) and γ_{n+1}(j)'s score at y_{n+1} (the "right" half,
v_n = F_j(y_{n+1})) — the same two posteriors and the same two observations
that build (û_n, v̂_n) in the first place. For a diagonal pair (i = j) both
halves land on the same 2 coordinates and are simply added. This drops
γ_0(i)'s and γ_{N-1}(j)'s "outer" contribution (the state margin's very
first / very last time point, whichever endpoint of the pair does not
already appear via some transition) from M only — not from D, which still
uses the full-sample G_k — a small, one-term-out-of-N approximation in the
same spirit as the package's documented i.i.d.-across-n convention
(:mod:`pmcprg.copulas._stderr`'s module docstring: "observations assumed
i.i.d.; serially dependent pairs need FR-5"). The coverage study below
finds no sign that this costs anything measurable at N in the hundreds.

Cost
----
One E-step (:func:`~pmcprg.pmc.inference.precompute_weights`,
:func:`~pmcprg.pmc.inference.forward`, :func:`~pmcprg.pmc.inference.backward`,
:func:`~pmcprg.pmc.inference.smooth`, :func:`~pmcprg.pmc.inference.joint_posteriors`)
at the fitted model, plus a handful of ``logpdf_array`` evaluations for
``D_θθ`` (Oakes' unchanged "term 1", no missing-information re-run needed:
Godambe gets its extra information from the margin cross term D_θη instead
of from re-running the E-step) and, for D_θη, one pair of ``_margin_cdfs`` /
``_pair_pseudo_obs`` recomputations per perturbed margin coordinate (2 or 4
of them) — no extra E-step at all, unlike Oakes' mixed term. This pilot is
cheaper than :mod:`pmcprg.pmc._oakes`'s two-parameter path.

What would extending this need
--------------------------------
* **Other copula families** (Clayton, BB1, Student, …): D_θθ generalises to
  a p × p matrix exactly as :mod:`pmcprg.pmc._oakes`'s two-parameter path
  already shows how to do (its ``_hessian_and_phi``); D_θη generalises to a
  p × (margin dim) matrix by the same finite-difference recipe used here,
  one column per working coordinate.
* **Pair margins** (``margin_structure == "pair"``, general PMC): the
  margin score is no longer the simple state-γ-weighted MLE — see
  :mod:`pmcprg.pmc.ice`'s "pair" branch (:func:`_pair_margin_sample`) — and
  D_ηη would need its own (still closed-form, for Gaussian pair margins)
  Hessian from that dual-view sample.
* **Non-Gaussian margins**: D_ηη would need a finite-difference Hessian of
  the family's weighted log-likelihood (no closed form in general —
  :func:`pmcprg.pmc.ice._fit_margin_weighted_numerical`'s own scope note),
  and D_θη's finite difference is unchanged in form.
* **GICE family selection**: out of scope regardless — the family switching
  between M-steps is not a smooth parameter to differentiate through.

References
----------
* Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method
  for copula-based models. *J. Multivariate Anal.* 94(2), 401-419.
  doi:10.1016/j.jmva.2004.06.003
* Joe, H. & Xu, J. J. (1996). The estimation method of inference functions
  for margins for multivariate models. Technical Report 166, Dept. of
  Statistics, University of British Columbia.
* Godambe, V. P. (1960). An optimum property of regular maximum likelihood
  estimation. *Ann. Math. Statist.* 31(4), 1208-1211.
* Oakes, D. (1999). Direct calculation of the information matrix via the EM
  algorithm. *J. R. Statist. Soc. B* 61(2), 479-482. doi:10.1111/1467-9868.00188
* Louis, T. A. (1982). Finding the observed information matrix when using
  the EM algorithm. *J. R. Statist. Soc. B* 44(2), 226-233.

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
and :func:`ice_godambe_tau_se` raises ``NotImplementedError`` on such a model.
The IFM correction above is the parametric
counterpart of that term: D_θη measures how the copula score moves with the
*parametric* margin parameters, which empirical copula margins no longer use.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.copulas._stderr import standard_errors as _naive_standard_errors
from pmcprg.pmc._oakes import _psi_of_tau, _tau_of_psi
from pmcprg.pmc.ice import (
    _margin_cdfs,
    _pair_pseudo_obs,
    _refuse_nonparametric_copula_margins,
    _resolve_candidate,
)
from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights, smooth
from pmcprg.pmc.model import PMCModel

logger = logging.getLogger(__name__)

__all__ = ["GodambeResult", "ice_godambe_tau_se"]

#: Central-difference step in the working coordinate ψ of τ — same value and
#: role as :data:`pmcprg.pmc._oakes.H_PSI` / :data:`pmcprg.copulas._stderr._H_PSI`.
H_PSI: float = 1e-4

#: Central-difference step in a margin's working coordinate (μ, log σ). Both
#: are O(1) quantities for data on a standardised-ish scale (as every model
#: this pilot was built and validated against is); an absolute rather than
#: relative step is the same simplification :mod:`pmcprg.copulas._stderr`
#: makes for its own ``_H_Z`` (logit(u) step).
H_ETA: float = 1e-4

#: This pilot's only supported copula family (audit FR-4 scope: the Godambe
#: option starts from Oakes' own first family).
SUPPORTED_FAMILIES: tuple[str, ...] = ("Gauss",)

#: τ̂ within this many optimiser pads of a range end is "at the boundary" —
#: same convention as :mod:`pmcprg.pmc._oakes._BOUNDARY_PADS`.
_BOUNDARY_PADS: float = 2.0


@dataclass(frozen=True)
class GodambeResult:
    """Godambe's (IFM) observed information for one ICE copula τ, accounting
    for the margins' own re-estimation (see the module docstring).

    Fields
    ------
    pair          : (i, j) of the copula block.
    family        : SHORT_NAME of the copula (``"Gauss"``, this pilot's only
                    supported family).
    tau_hat       : the ICE-fitted τ.
    se_tau        : Godambe's standard error of τ̂ (NaN if ``at_boundary`` or
                    the joint sensitivity/variability matrices are singular).
    se_tau_naive  : the naive :mod:`pmcprg.copulas._stderr` sandwich applied
                    to the final ICE weights ξ̂_n(i, j) *and* to the final
                    margin parameters, both treated as fixed — the quantity
                    this module's IFM correction is built to fix.
    margin_names  : the margin parameters entering η, in ``cov``'s order —
                    ``("mu_i", "sigma_i")`` for a diagonal pair (i = j),
                    ``("mu_i", "sigma_i", "mu_j", "sigma_j")`` otherwise.
    margin_se     : ``{name: standard error}`` of each of ``margin_names``,
                    a side product of the same joint covariance (not
                    separately validated — see the module docstring's scope
                    note).
    cov           : the joint covariance of ``margin_names`` and τ (last
                    row/column), in *reported* units (μ, σ — not log σ — and
                    τ — not ψ): ``D⁻¹ M D⁻ᵀ`` carried from the working
                    coordinates by the delta method.
    D             : the sensitivity matrix, in the working coordinates
                    (μ, log σ, …, ψ) — block lower-triangular by
                    construction (see the module docstring).
    M             : the variability matrix, in the same working coordinates.
    n_eff         : Σ_n ξ̂_n(i, j).
    at_boundary   : τ̂ within :data:`_BOUNDARY_PADS` working-coordinate steps
                    of Gauss's registered range [-1, 1] — ψ̂ is not usable.
    """

    pair: tuple[int, int]
    family: str
    tau_hat: float
    se_tau: float
    se_tau_naive: float
    margin_names: tuple[str, ...]
    margin_se: dict
    cov: np.ndarray
    D: np.ndarray
    M: np.ndarray
    n_eff: float
    at_boundary: bool

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        """Wald interval for τ — see :meth:`pmcprg.pmc._oakes.OakesResult.ci`."""
        from scipy.stats import norm

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}.")
        if self.at_boundary or not np.isfinite(self.se_tau):
            return float("nan"), float("nan")
        z = float(norm.ppf(0.5 + 0.5 * level))
        return self.tau_hat - z * self.se_tau, self.tau_hat + z * self.se_tau


def _e_step(model: PMCModel, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run one complete E-step; return ``(gamma, xi)``, shapes ``(N, K)`` and
    ``(N-1, K, K)``."""
    W, f_pdf = precompute_weights(model, Y)
    alpha_hat, _ = forward(model, Y, W=W, f_pdf=f_pdf)
    beta_hat = backward(model, Y, W=W)
    gamma = smooth(alpha_hat, beta_hat)
    xi = joint_posteriors(alpha_hat, W, beta_hat)
    return gamma, xi


def _with_margin_params(model: PMCModel, k: int, mu: float, sigma: float) -> PMCModel:
    """Deep-copy ``model``'s raw dict with state ``k``'s Gaussian margin
    replaced by ``(mu, sigma)`` — the margin analogue of
    :func:`pmcprg.pmc._oakes._with_pair_tau`."""
    raw = model.raw
    found = False
    for blk in raw.get("margins", []):
        if int(blk["i"]) == k:
            blk["params"] = {"loc": float(mu), "scale": float(sigma)}
            found = True
            break
    if not found:
        raise KeyError(f"No margin block for state i={k} in this model.")
    return PMCModel.from_dict(raw)


def _pseudo_obs(model: PMCModel, Y: np.ndarray, ii: int, jj: int) -> tuple[np.ndarray, np.ndarray]:
    f_cdf = _margin_cdfs(model, Y)
    return _pair_pseudo_obs(model, f_cdf, ii, jj)


def _phi_sum(cls, uv: np.ndarray, xi: np.ndarray, psi_hat: float, h_psi: float) -> tuple[float, np.ndarray]:
    """``(S, phi)`` — ``S = Σ_n xi_n · phi_n``, the copula score summed and
    weighted, and ``phi_n`` itself — both at ``psi_hat`` (central difference,
    same stencil as :mod:`pmcprg.pmc._oakes`)."""
    tau_p = _tau_of_psi(psi_hat + h_psi, -1.0, 1.0)
    tau_m = _tau_of_psi(psi_hat - h_psi, -1.0, 1.0)
    ld_p = cls(tau_k=tau_p).logpdf_array(uv)
    ld_m = cls(tau_k=tau_m).logpdf_array(uv)
    phi = (ld_p - ld_m) / (2.0 * h_psi)
    return float(np.dot(xi, phi)), phi


def _state_score_terms(y: np.ndarray, gamma_col: np.ndarray, mu: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-observation Gaussian weighted-MLE score contributions ``(score_mu,
    score_log_sigma)`` for state weight ``gamma_col`` at points ``y``."""
    z = (y - mu) / sigma
    return gamma_col * z / sigma, gamma_col * (z * z - 1.0)


def _godambe_one_pair(
    model: PMCModel,
    Y: np.ndarray,
    ii: int,
    jj: int,
    *,
    h_psi: float,
    h_eta: float,
) -> GodambeResult:
    if model.margin_structure != "state":
        raise NotImplementedError(
            "ice_godambe_tau_se: margin_structure must be 'state' (audit "
            f"FR-4 Godambe pilot scope); got {model.margin_structure!r}."
        )

    blocks = {(int(b["i"]), int(b["j"])): b for b in model.copula_blocks()}
    if (ii, jj) not in blocks:
        raise KeyError(f"No copula block for pair (i={ii}, j={jj}).")
    blk = blocks[(ii, jj)]
    family = str(blk["name"])
    if family not in SUPPORTED_FAMILIES:
        raise NotImplementedError(
            f"ice_godambe_tau_se: family {family!r} for pair ({ii}, {jj}) is "
            f"not one of the supported families {SUPPORTED_FAMILIES} (audit "
            "FR-4 Godambe pilot scope)."
        )
    entry, cls = _resolve_candidate(family)
    a, b = (float(x) for x in entry.value.TAU_MIN_MAX)
    tau_hat = float(blk["tau"])

    margin_blocks = {int(m["i"]): m for m in model.margin_blocks()}
    unique_states = sorted({ii, jj})
    for k in unique_states:
        if k not in margin_blocks or margin_blocks[k].get("dist") != "norm":
            raise NotImplementedError(
                f"ice_godambe_tau_se: state {k}'s margin must be a Gaussian "
                "('dist' == 'norm') for the audit FR-4 Godambe pilot; got "
                f"{margin_blocks.get(k, {}).get('dist')!r}."
            )

    Y = np.asarray(Y, dtype=float)
    dim = 2 * len(unique_states)
    margin_names: list[str] = []
    for k in unique_states:
        margin_names += [f"mu_{k}", f"sigma_{k}"]

    pad = _BOUNDARY_PADS * max(TAU_PAD_REL * (b - a), TAU_PAD_ABS)
    at_boundary = not (a + pad < tau_hat < b - pad)

    gamma, xi = _e_step(model, Y)
    xi_hat = xi[:, ii, jj]
    n_eff = float(xi_hat.sum())

    mu = {k: float(margin_blocks[k]["params"]["loc"]) for k in unique_states}
    sigma = {k: float(margin_blocks[k]["params"]["scale"]) for k in unique_states}

    u, v = _pseudo_obs(model, Y, ii, jj)
    uv = np.column_stack((u, v))

    naive = _naive_standard_errors(
        cls(tau_k=tau_hat), uv, weights=xi_hat, method="mle", ranks=False,
    ).se["tau_k"]

    nan_dim1 = dim + 1
    if at_boundary:
        logger.warning(
            "ice_godambe_tau_se: pair (i=%d, j=%d) τ̂=%.6g is at the boundary "
            "of %s's range [%.4g, %.4g] — Godambe's SE is not defined there "
            "(Self & Liang 1987); returning NaN.", ii, jj, tau_hat, family, a, b,
        )
        return GodambeResult(
            (ii, jj), family, tau_hat, math.nan, naive, tuple(margin_names),
            {name: math.nan for name in margin_names},
            np.full((nan_dim1, nan_dim1), math.nan),
            np.full((nan_dim1, nan_dim1), math.nan),
            np.full((nan_dim1, nan_dim1), math.nan),
            n_eff, True,
        )

    psi_hat = _psi_of_tau(tau_hat, a, b)

    # ---- D_ηη (closed form, block-diagonal per state) and M's margin part
    D = np.zeros((nan_dim1, nan_dim1))
    g_margin = np.zeros((Y.shape[0] - 1, dim))
    for idx, k in enumerate(unique_states):
        gk_full = float(gamma[:, k].sum())
        D[2 * idx, 2 * idx] = gk_full / (sigma[k] ** 2)
        D[2 * idx + 1, 2 * idx + 1] = 2.0 * gk_full
        if k == ii:
            s_mu, s_ls = _state_score_terms(Y[:-1], gamma[:-1, k], mu[k], sigma[k])
            g_margin[:, 2 * idx] += s_mu
            g_margin[:, 2 * idx + 1] += s_ls
        if k == jj:
            s_mu, s_ls = _state_score_terms(Y[1:], gamma[1:, k], mu[k], sigma[k])
            g_margin[:, 2 * idx] += s_mu
            g_margin[:, 2 * idx + 1] += s_ls

    # ---- D_θθ and phi_hat (Oakes' unchanged "term 1" / phi)
    S_hat, phi_hat = _phi_sum(cls, uv, xi_hat, psi_hat, h_psi)
    logpdf_0 = cls(tau_k=tau_hat).logpdf_array(uv)
    logpdf_p = cls(tau_k=_tau_of_psi(psi_hat + h_psi, a, b)).logpdf_array(uv)
    logpdf_m = cls(tau_k=_tau_of_psi(psi_hat - h_psi, a, b)).logpdf_array(uv)
    curvature = (logpdf_p - 2.0 * logpdf_0 + logpdf_m) / (h_psi * h_psi)
    D[dim, dim] = -float(np.dot(xi_hat, curvature))

    # ---- D_θη (finite difference — the IFM correction term)
    for idx, k in enumerate(unique_states):
        for comp, h in ((0, h_eta), (1, h_eta)):
            mu_p, sigma_p = mu[k], sigma[k]
            mu_m, sigma_m = mu[k], sigma[k]
            if comp == 0:
                mu_p, mu_m = mu[k] + h, mu[k] - h
            else:
                sigma_p = sigma[k] * math.exp(h)
                sigma_m = sigma[k] * math.exp(-h)
            model_p = _with_margin_params(model, k, mu_p, sigma_p)
            model_m = _with_margin_params(model, k, mu_m, sigma_m)
            u_p, v_p = _pseudo_obs(model_p, Y, ii, jj)
            u_m, v_m = _pseudo_obs(model_m, Y, ii, jj)
            S_p, _ = _phi_sum(cls, np.column_stack((u_p, v_p)), xi_hat, psi_hat, h_psi)
            S_m, _ = _phi_sum(cls, np.column_stack((u_m, v_m)), xi_hat, psi_hat, h_psi)
            D[dim, 2 * idx + comp] = -(S_p - S_m) / (2.0 * h)

    # ---- M (variability matrix): uncentred empirical covariance of the
    # per-transition joint score contributions (mean ≈ 0 at the joint fit).
    g_theta = xi_hat * phi_hat
    M = np.zeros((nan_dim1, nan_dim1))
    M[:dim, :dim] = g_margin.T @ g_margin
    M[:dim, dim] = g_margin.T @ g_theta
    M[dim, :dim] = M[:dim, dim]
    M[dim, dim] = float(np.dot(g_theta, g_theta))

    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        logger.warning(
            "ice_godambe_tau_se: pair (i=%d, j=%d) sensitivity matrix D is "
            "singular — returning NaN.", ii, jj,
        )
        return GodambeResult(
            (ii, jj), family, tau_hat, math.nan, naive, tuple(margin_names),
            {name: math.nan for name in margin_names}, M, D, M, n_eff, False,
        )

    cov_working = D_inv @ M @ D_inv.T
    cov_working = 0.5 * (cov_working + cov_working.T)

    var_psi = cov_working[dim, dim]
    dtau_dpsi = (tau_hat - a) * (b - tau_hat) / (b - a)
    if var_psi > 0.0:
        se_tau = abs(dtau_dpsi) * math.sqrt(var_psi)
    else:
        logger.warning(
            "ice_godambe_tau_se: pair (i=%d, j=%d) Godambe variance of ψ̂ "
            "(%.4g) is not positive — returning NaN.", ii, jj, var_psi,
        )
        se_tau = math.nan

    # Reported covariance / SEs, natural units: μ unchanged, σ = exp(log σ)
    # via the delta method (dσ/d(log σ) = σ), τ via dτ/dψ as above.
    jac = np.eye(nan_dim1)
    for idx, k in enumerate(unique_states):
        jac[2 * idx + 1, 2 * idx + 1] = sigma[k]
    jac[dim, dim] = dtau_dpsi
    cov = jac @ cov_working @ jac.T

    margin_se = {}
    for idx, name in enumerate(margin_names):
        val = cov[idx, idx]
        margin_se[name] = float(math.sqrt(val)) if val >= 0.0 else math.nan

    return GodambeResult(
        (ii, jj), family, tau_hat, se_tau, naive, tuple(margin_names),
        margin_se, cov, D, M, n_eff, False,
    )


def ice_godambe_tau_se(
    model: PMCModel,
    Y: np.ndarray,
    pairs: list[tuple[int, int]] | None = None,
    *,
    h_psi: float = H_PSI,
    h_eta: float = H_ETA,
) -> "dict[tuple[int, int], GodambeResult]":
    """Godambe's (IFM) standard error(s) of the copula parameter(s) of each
    pair, accounting for the margins' own re-estimation inside ICE.

    Parameters
    ----------
    model : PMCModel — the fitted model returned by :func:`pmcprg.pmc.ice.ice`
            with ``fit_margins=True`` (the case this module exists for; see
            the module docstring — with ``fit_margins=False`` there is no
            margin re-estimation error to correct for and
            :mod:`pmcprg.pmc._oakes` already suffices).
    Y     : the observation sequence the ICE run was fitted on. Must be
            complete (no missing rows) — the pilot does not cover
            :mod:`pmcprg.pmc.gaps`.
    pairs : which copula blocks to compute; ``None`` = every block whose
            family is in :data:`SUPPORTED_FAMILIES` (``"Gauss"`` only).
    h_psi : central-difference step in the copula's working coordinate ψ.
    h_eta : central-difference step in a margin's working coordinate
            (μ, log σ).

    Returns
    -------
    ``{(i, j): GodambeResult}``.

    Raises
    ------
    NotImplementedError : the model's margin structure is not ``"state"``,
        a requested pair's states' margins are not Gaussian, or a
        requested pair's family is not :data:`SUPPORTED_FAMILIES` (this
        pilot: Gauss only — every other family is out of this audit's scope);
        or the model was fitted with ``copula_margins = "empirical"`` (module
        docstring, last section).
    """
    if not model.variant.uses_copula:
        raise ValueError(f"Variant {model.variant.value} does not use copulas.")
    _refuse_nonparametric_copula_margins(model, "ice_godambe_tau_se")
    Y = np.asarray(Y, dtype=float)
    if np.any(~np.isfinite(Y)):
        raise ValueError(
            "ice_godambe_tau_se: Y has missing (non-finite) rows — the "
            "Godambe pilot does not cover gap variants yet (see the module "
            "docstring)."
        )

    all_blocks = model.copula_blocks()
    if pairs is None:
        pairs = [(int(b["i"]), int(b["j"])) for b in all_blocks
                  if str(b["name"]) in SUPPORTED_FAMILIES]

    return {(ii, jj): _godambe_one_pair(model, Y, ii, jj, h_psi=h_psi, h_eta=h_eta)
            for ii, jj in pairs}
