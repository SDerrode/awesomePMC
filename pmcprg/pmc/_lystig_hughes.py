"""
pmcprg.pmc._lystig_hughes — Lystig & Hughes' (2002) exact observed information
of the prior and copula parameters of a model fitted by ICE (AUDIT_COPULES
FR-4, "Reste": standard errors inside ICE, third of the three ranked options
after :mod:`pmcprg.pmc._oakes` and :mod:`pmcprg.pmc._godambe`).

Scope
-----
K-state models whose transition weight factorises as
``W[n, i, j] = A_ij · f_j(y_{n+1}) · c_ij(F_i(y_n), F_j(y_{n+1}))`` — HMC-DN,
and PMC with state margins (for which :func:`~pmcprg.pmc.inference.precompute_weights`'
``p_ij f_i(y_n) / Σ_k p_ik f_i(y_n)`` is exactly ``A_ij``). Validated at
K = 2. Margins are held **fixed** (``fit_margins=False``): their uncertainty
is :mod:`pmcprg.pmc._godambe`'s channel. Copula families: the one-parameter
Gauss and Clayton, as Oakes' pilot. Complete data only (no gaps).

What it computes, and why it is worth having next to Oakes
-----------------------------------------------------------
The **full** observed information matrix ``I(θ) = −∂²ℓ(θ)/∂θ∂θᵀ`` of the
observed-data log-likelihood ℓ(θ) = log p(y_{1:N}; θ) — the value
:func:`~pmcprg.pmc.inference.forward` returns — over the joint parameter
vector θ = (prior coordinates, every pair's copula ψ). Inverting it gives the
standard error of each pair's τ that accounts for the estimation of the
prior *and* of the other pairs' τ. :mod:`pmcprg.pmc._oakes` computes exactly
the diagonal entry ``I[ψ_ij, ψ_ij]`` (Oakes' identity holds at every θ, not
only at the MLE), i.e. the information with every other parameter held at
its fitted value; ``1/I[k,k] ≤ (I⁻¹)[k,k]`` (Schur complement), so the joint
SE is never smaller than Oakes' partial one.

Parametrisation (the minimal free coordinates of the ICE M-step)
-----------------------------------------------------------------
:func:`pmcprg.pmc.ice._m_step_prior` estimates the joint ``p_ij`` as the
**symmetrised** ξ̄ (SR-PMC: p_ij = p_ji), normalised to 1; HMC variants store
``A_ij = p_ij / π_i`` with ``π_i = Σ_j p_ij`` (the stationary law of A,
because p is symmetric). The free parameters are therefore the
``K(K+1)/2`` distinct entries of a symmetric ``p`` summing to 1, i.e.
``K(K+1)/2 − 1`` coordinates — 2 at K = 2, not K² − 1 = 3. Unconstrained
coordinates: list the unordered pairs ``s = (i ≤ j)`` in row order,
``q_s = p_ii`` on the diagonal and ``q_s = p_ij + p_ji = 2 p_ij`` off it
(so Σ_s q_s = 1), and take the **softmax** with the first pair (0, 0) as
reference::

    q = softmax(0, η_1, …, η_{S−1}),   η_s = log(q_s / q_00),
    p_ij = q_s · (1 if i = j else 1/2),  π_i = Σ_j p_ij,  A_ij = p_ij / π_i.

Names: ``"eta_ij"`` for the pair (i, j), i ≤ j, (i, j) ≠ (0, 0). Copula
blocks: ``"psi_ij"``, the working coordinate of :mod:`pmcprg.pmc._oakes`,
``ψ = log(τ − a) − log(b − τ)`` on the family's registered range [a, b]
(2·atanh τ for Gauss, logit τ for Clayton). A pair whose τ̂ sits within
Oakes' boundary pad of [a, b] is **held fixed** (excluded from θ, its SE is
NaN — Self & Liang 1987), so that one non-regular coordinate does not spoil
the others.

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
the value recursion. The initial step is the same with ``a_1 = π ∘ f(y_1)``,
``a_{1,r} = π_r ∘ f(y_1)``, ``a_{1,rs} = π_rs ∘ f(y_1)`` (only the prior
coordinates move π). The recursion is exact: its only approximation is in
the per-step weight derivatives below.

Weight derivatives
-------------------
``W_r = W · ∂_r log W`` and ``W_rs = W · (∂_r log W ∂_s log W + ∂²_rs log W)``,
with ``log W[n, i, j] = log A_ij + log f_j(y_{n+1}) + log c_ij(u_n, v_n)``:

* prior coordinates — analytic: ``∂ log p_ij/∂η_t = 1{s(ij) = t} − q_t``,
  ``∂² log p_ij/∂η_t∂η_u = −q_t (1{t = u} − q_u)``, ``π_i`` and its
  derivatives summed from ``p_i·``, and ``log A_ij = log p_ij − log π_i``;
* copula ψ_ij — only W[·, i, j] moves; ``∂ log c/∂ψ`` and ``∂² log c/∂ψ²``
  are **central finite differences** of ``logpdf_array`` per observation,
  step ``H_PSI = 1e-4`` (:mod:`pmcprg.copulas._stderr`/:mod:`pmcprg.pmc._oakes`
  convention: truncation O(10⁻⁸), rounding O(10⁻⁸)·|log c|);
* mixed prior/copula second derivatives of log W are exactly zero (additive).

Cost
----
One :func:`~pmcprg.pmc.inference.precompute_weights` call, 3 ``logpdf_array``
calls per free copula pair, and one O(N · P² · K²) forward sweep (P = number
of free parameters) — no E-step re-run at all. At K = 2, P = 6, N = 600 it is
a few tens of milliseconds, cheaper than Oakes' one-pair computation (which
re-runs forward-backward twice per pair) while returning the whole matrix.

ICE is not the MLE
-------------------
The ICE τ update is an exact EM step (weighted MLE of Σ ξ log c), but the
prior update p̂ = sym(ξ̄) ignores the initial-state term log π_{x_1} and the
dependence of A on π, and ICE stops at a tolerance: the score ``grad`` at an
ICE fit is not zero. :attr:`LHInformation.newton_step` = I⁻¹·grad is the
one-step distance to the MLE, in working coordinates — the diagnostic of how
far ICE stopped from the MLE, to be compared with the SE. The information is
evaluated at the ICE fit (not at a re-optimised MLE), which is what the SEs
reported for the ICE estimate should use.

What extending this would need
-------------------------------
* **Margins** — add their working coordinates to θ; ``∂ log W`` then gains
  ``∂ log f_j(y_{n+1})`` (analytic for Gaussian margins) *and* the chain
  rule through the pseudo-observations ``∂ log c/∂u · ∂F_i(y_n)/∂η`` (finite
  differences in u as :mod:`pmcprg.copulas._stderr` does), and ``a_1`` gains
  ``∂f(y_1)``; the recursion itself is unchanged.
* **Two-parameter copulas** — one θ coordinate per working coordinate (the
  ``_spec_of`` ψ of :mod:`pmcprg.copulas._stderr`), with the 4-point mixed
  difference for the within-pair cross term; recursion unchanged.
* **Pair margins** (general PMC) — the prior derivative of log W then
  depends on y_n through ``Σ_k p_ik f_ik(y_n)``; still per-step analytic.
* **Gaps** — the recursion runs on the augmented (state, grid-node) chain of
  :mod:`pmcprg.pmc.gaps`; the same scaled derivative recursion applies to
  that chain's (larger) transition matrices.

References
----------
* Lystig, T. C. & Hughes, J. P. (2002). Exact computation of the observed
  information matrix for hidden Markov models. *J. Comput. Graph. Statist.*
  11(3), 678-689. doi:10.1198/106186002402
* Devijver, P. A. (1985). Baum's forward-backward algorithm revisited.
  *Pattern Recognition Letters* 3(6), 369-373.
* Oakes, D. (1999). Direct calculation of the information matrix via the EM
  algorithm. *J. R. Statist. Soc. B* 61(2), 479-482.
* Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. *J. Amer. Statist. Assoc.* 82(398), 605-610.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.numerics import MIN_POSITIVE
from pmcprg.pmc._oakes import _psi_of_tau, _pseudo_obs, _tau_of_psi
from pmcprg.pmc.ice import _resolve_candidate
from pmcprg.pmc.inference import precompute_weights
from pmcprg.pmc.model import PMCModel, Variant

logger = logging.getLogger(__name__)

__all__ = ["LHInformation", "ice_lh_information", "lh_loglik_derivatives"]

#: Central-difference step in the copula working coordinate ψ — same value as
#: :data:`pmcprg.pmc._oakes.H_PSI` and :data:`pmcprg.copulas._stderr._H_PSI`.
H_PSI: float = 1e-4

#: Copula families of this pilot (one-parameter, as Oakes' pilot).
SUPPORTED_FAMILIES: tuple[str, ...] = ("Gauss", "Clayton")

#: Boundary pad, in optimiser pads — :data:`pmcprg.pmc._oakes._BOUNDARY_PADS`.
_BOUNDARY_PADS: float = 2.0

#: Largest tolerated asymmetry |p_ij − p_ji| of the model's joint prior.
_SYM_TOL: float = 1e-6


@dataclass(frozen=True)
class LHInformation:
    """Lystig–Hughes observed information of an ICE fit (module docstring).

    Fields
    ------
    names       : θ coordinates, in matrix order — ``"eta_ij"`` (prior,
                  softmax coordinates) then ``"psi_ij"`` (free copula pairs).
    theta       : θ at the model handed in.
    log_lik     : ℓ(θ), equal to :func:`~pmcprg.pmc.inference.forward`'s value.
    grad        : ∂ℓ/∂θ (exact recursion; copula-density derivatives by
                  central differences).
    hessian     : ∂²ℓ/∂θ∂θᵀ, same accuracy.
    info        : ``−hessian``, the observed information.
    cov         : ``info⁻¹`` (NaN if ``info`` is not positive-definite).
    newton_step : ``info⁻¹ · grad`` — one Newton step from the ICE fit
                  towards the MLE, in working coordinates (NaN as ``cov``).
    tau_hat     : ``{(i, j): τ̂}`` for every copula pair.
    se_tau      : ``{(i, j): SE}`` from the full ``cov`` (delta method).
    se_tau_partial : ``{(i, j): SE}`` from ``1 / info[k, k]`` alone — the
                  other parameters held fixed; equals Oakes' SE
                  (:func:`pmcprg.pmc._oakes.ice_oakes_tau_se`) up to its
                  finite-difference error.
    fixed_pairs : pairs held fixed at a boundary τ̂ (SE NaN).
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

    def ci(self, pair: tuple[int, int], level: float = 0.95) -> tuple[float, float]:
        """Wald interval for pair ``pair``'s τ, from :attr:`se_tau`."""
        from scipy.stats import norm

        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level!r}.")
        se = self.se_tau[pair]
        if not np.isfinite(se):
            return float("nan"), float("nan")
        z = float(norm.ppf(0.5 + 0.5 * level))
        return self.tau_hat[pair] - z * se, self.tau_hat[pair] + z * se


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
    """``(p, π, dlogA, d2logA, dlogπ, d2logπ)`` at η — see the module docstring.

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
            "FR-4 Lystig–Hughes pilot scope (HMC-DN, PMC with state margins)."
        )
    if model.margin_structure != "state":
        raise NotImplementedError(
            "ice_lh_information: pair margins are out of the pilot scope "
            "(module docstring, 'What extending this would need')."
        )


def _pair_info(model: PMCModel):
    """``[(pair, family, cls, a, b, τ̂, at_boundary)]`` in block order (i, j)."""
    out = []
    blocks = sorted(model.copula_blocks(), key=lambda b: (int(b["i"]), int(b["j"])))
    for blk in blocks:
        family = str(blk["name"])
        pair = (int(blk["i"]), int(blk["j"]))
        if family not in SUPPORTED_FAMILIES:
            raise NotImplementedError(
                f"ice_lh_information: family {family!r} for pair {pair} is not "
                f"one of {SUPPORTED_FAMILIES} (audit FR-4 Lystig–Hughes pilot)."
            )
        entry, cls = _resolve_candidate(family)
        a, b = (float(x) for x in entry.value.TAU_MIN_MAX)
        tau = float(blk["tau"])
        pad = _BOUNDARY_PADS * max(TAU_PAD_REL * (b - a), TAU_PAD_ABS)
        out.append((pair, family, cls, a, b, tau, not (a + pad < tau < b - pad)))
    return out


def _theta_names(model: PMCModel) -> tuple[list[str], list]:
    names = [f"eta_{i}{j}" for i, j in _unique_pairs(model.K)[1:]]
    free = [info for info in _pair_info(model) if not info[6]]
    names += [f"psi_{info[0][0]}{info[0][1]}" for info in free]
    return names, free


def model_from_theta(model: PMCModel, theta: np.ndarray) -> PMCModel:
    """A copy of ``model`` with θ (in :func:`ice_lh_information`'s order) set.

    Pairs held fixed at a boundary keep their τ. Used by the finite-difference
    checks; the prior is written as ``A`` (HMC-DN) or ``p`` (PMC).
    """
    _check_scope(model)
    _, free = _theta_names(model)
    K = model.K
    T = len(_unique_pairs(K)) - 1
    theta = np.asarray(theta, dtype=float)
    p, pi, *_ = _prior_from_eta(theta[:T], K)
    raw = model.raw
    if model.variant.has_markov_prior:
        raw["prior"] = {k: v for k, v in raw["prior"].items() if k != "p"}
        raw["prior"]["A"] = (p / pi[:, None]).tolist()
    else:
        raw["prior"] = {k: v for k, v in raw["prior"].items() if k != "A"}
        raw["prior"]["p"] = p.tolist()
    new_tau = {info[0]: _tau_of_psi(float(theta[T + k]), info[3], info[4])
               for k, info in enumerate(free)}
    for blk in raw["copulas"]:
        pair = (int(blk["i"]), int(blk["j"]))
        if pair in new_tau:
            blk["tau"] = float(new_tau[pair])
    return PMCModel.from_dict(raw)


# ---------------------------------------------------------------------------
# The scaled derivative recursion
# ---------------------------------------------------------------------------

def lh_loglik_derivatives(
    model: PMCModel, Y: np.ndarray, *, h_psi: float = H_PSI,
) -> tuple[float, np.ndarray, np.ndarray, tuple[str, ...]]:
    """``(ℓ, ∂ℓ/∂θ, ∂²ℓ/∂θ∂θᵀ, names)`` at ``model`` — the recursion of the
    module docstring. θ as in :func:`ice_lh_information`."""
    _check_scope(model)
    Y = np.asarray(Y, dtype=float)
    if np.any(~np.isfinite(Y)):
        raise ValueError(
            "ice_lh_information: Y has missing (non-finite) rows — gaps are "
            "out of the Lystig–Hughes pilot scope (module docstring)."
        )
    K = model.K
    names, free = _theta_names(model)
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
    for k, (pair, _family, cls, a, b, tau, _) in enumerate(free):
        ii, jj = pair
        u, v = _pseudo_obs(model, Y, ii, jj)
        uv = np.column_stack((u, v))
        psi = _psi_of_tau(tau, a, b)
        l0 = np.asarray(cls(tau_k=tau).logpdf_array(uv), dtype=float)
        lp = np.asarray(cls(tau_k=_tau_of_psi(psi + h_psi, a, b)).logpdf_array(uv), dtype=float)
        lm = np.asarray(cls(tau_k=_tau_of_psi(psi - h_psi, a, b)).logpdf_array(uv), dtype=float)
        r = T + k
        dlogW[:, r, ii, jj] = (lp - lm) / (2.0 * h_psi)
        d2logW[:, r, r, ii, jj] = (lp - 2.0 * l0 + lm) / (h_psi * h_psi)

    live = W > 0.0
    Wb = W[:, None, :, :]
    dW = np.where(live[:, None], Wb * dlogW, 0.0)
    d2W = np.where(live[:, None, None],
                   Wb[:, None] * (dlogW[:, :, None] * dlogW[:, None, :] + d2logW), 0.0)
    if not (np.all(np.isfinite(dW)) and np.all(np.isfinite(d2W))):
        raise FloatingPointError(
            "ice_lh_information: non-finite copula log-density derivative at a "
            "positive transition weight."
        )

    # Initial step: a_1 = π ∘ f(y_1) (state margins: f_pdf[0, j, ·] = f_j(y_1)).
    f1 = f_pdf[0, :, 0]
    a = pi * f1
    a_r = np.zeros((P, K))
    a_rs = np.zeros((P, P, K))
    a_r[:T] = dpi * f1[None]
    a_rs[:T, :T] = d2pi * f1[None, None]

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
) -> LHInformation:
    """Lystig & Hughes' (2002) full observed information of an ICE fit.

    Parameters
    ----------
    model : the fitted model (:func:`pmcprg.pmc.ice.ice`, ``fit_margins=False``;
            HMC-DN or state-margin PMC; Gauss/Clayton copulas; symmetric prior).
    Y     : the complete observation sequence it was fitted on.
    h_psi : central-difference step in the copula working coordinate ψ.

    Returns
    -------
    :class:`LHInformation`.

    Raises
    ------
    NotImplementedError : variant, margin structure or copula family out of
        the pilot scope (module docstring).
    ValueError : missing rows, or a non-symmetric / degenerate prior.
    """
    if not model.variant.uses_copula:
        raise ValueError(f"Variant {model.variant.value} does not use copulas.")
    _check_scope(model)
    ll, g, H, names = lh_loglik_derivatives(model, Y, h_psi=h_psi)
    K = model.K
    T = len(_unique_pairs(K)) - 1
    eta = _eta_of_model(model)
    pairs = _pair_info(model)
    free = [info for info in pairs if not info[6]]
    theta = np.concatenate((eta, [_psi_of_tau(info[5], info[3], info[4]) for info in free]))
    info_m = -H
    P = len(names)
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

    tau_hat = {info[0]: info[5] for info in pairs}
    se_full = {info[0]: math.nan for info in pairs}
    se_part = {info[0]: math.nan for info in pairs}
    for k, (pair, _f, _c, a, b, tau, _) in enumerate(free):
        r = T + k
        dtau = (tau - a) * (b - tau) / (b - a)
        if pos_def and cov[r, r] > 0.0:
            se_full[pair] = abs(dtau) * math.sqrt(cov[r, r])
        if info_m[r, r] > 0.0:
            se_part[pair] = abs(dtau) / math.sqrt(info_m[r, r])
    fixed = tuple(info[0] for info in pairs if info[6])
    if fixed:
        logger.warning(
            "ice_lh_information: pair(s) %s at a boundary of their τ range are "
            "held fixed (Self & Liang 1987); their SE is NaN.", fixed,
        )
    return LHInformation(
        names, theta, ll, g, H, info_m, cov, step,
        tau_hat, se_full, se_part, fixed,
    )
