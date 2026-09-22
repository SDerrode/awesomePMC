"""
inference.py — Baum-Welch forward-backward + MPM classification for PMC/HMC models.

Public API
----------
classify(model, Y) -> X_hat
    Supervised MPM classification of the observation sequence Y.

forward(model, Y)   -> (alpha_hat, log_lik)
backward(model, Y, W=None) -> beta_hat
smooth(alpha_hat, beta_hat) -> gamma
mpm(gamma) -> X_hat
precompute_weights(model, Y) -> (W, f_pdf)
sample_posterior(model, Y, rng) -> X_sample
    Forward-Filter Backward-Sample (FFBS) draw  X̃ ~ P(X | Y).

Model
-----
Z = (X, Y) is a pairwise Markov chain
(DerrodePieczynski_CSDA2013 §2.1, Eq. 4); forward–backward runs on its
transition p(z_{n+1} | z_n) = W[n, i, j]
(DerrodePieczynski_CSDA2013 Eqs. 16–18), with margins f_ij =
``model.margin(i, j)`` of the observation y_n given
(x_n, x_{n+1}) = (i, j) (DerrodePieczynski_CSDA2013 Eq. 12):

  HMC-IN, HMC-IN2 : W[n, i, j] = A[i,j] · f_j(y_{n+1})
  HMC-DN          : W[n, i, j] = A[i,j] · f_j(y_{n+1}) · c_ij(F_i(y_n), F_j(y_{n+1}))
  PMC-IN          : W[n, i, j] = p[i,j] f_ij(y_n) / Σ_k p[i,k] f_ik(y_n) · f_ji(y_{n+1})
  PMC             : the PMC-IN weight · c_ij(F_ij(y_n), F_ji(y_{n+1}))

the PMC-* weights being DerrodePieczynski_CSDA2013 Eq. 13 (transition of x)
times Eq. 14 (density of y_{n+1}). With pair margins
(``model.margin_structure == "pair"``, general PMC) the (N, K, K) tensor
f_pdf[n, i, j] = f_ij(y_n) holds K² distinct densities; with state margins
f_ij = f_i and it is broadcast from K vectors — the ratio
p[i,j] f_i(y_n) / Σ_k p[i,k] f_i(y_n) then no longer depends on y_n: X is
Markov (Proposition of DerrodePieczynski_CSDA2013 §2.1).

Algorithm
---------
Devijver (1985) normalization (Baum-Welch revisited):

  Initialization — α_1 = p(x_1, y_1)
    state margins : α_1(j) = Σ_i  p[i,j] · f_j(y_1)
    pair margins  : α_1(i) = Σ_j  p[i,j] · f_ij(y_1)     (DerrodePieczynski_CSDA2013 §3.1)
    C_1     = Σ_j  α_1(j)
    α̂_1(j) = α_1(j) / C_1

  Recursion  (n = 1, …, N-1)
    α_{n+1}(j) = Σ_i  α̂_n(i) · w(i, j, y_n, y_{n+1})
    C_{n+1}    = Σ_j  α_{n+1}(j)
    α̂_{n+1}(j)= α_{n+1}(j) / C_{n+1}

  Log-likelihood = Σ_{n=1}^{N} log C_n

  Backward (for MPM)
    β̂_N(j) = 1/K
    β_n(i)  = Σ_j  w(i, j, y_n, y_{n+1}) · β̂_{n+1}(j)
    D_n     = Σ_i  β_n(i)
    β̂_n(i) = β_n(i) / D_n

  Note on the backward scaling (audit K-13): β̂ is rescaled at each step by
  its **own** sum D_n — a Rabiner (1989, §V.A)-type scaling — rather than by
  the forward constants C_{n+1} as in Devijver (1985). The two differ only by
  a per-step positive factor, which is harmless here because ``smooth`` and
  ``joint_posteriors`` renormalise every row; the β̂ returned by ``backward``
  are therefore *not* the Devijver quantities P(y_{n+1:N} | x_n) / ∏ C_k,
  only proportional to them at each n.

  Posterior marginals
    γ_n(j) ∝  α̂_n(j) · β̂_n(j)

  MPM
    X̂_n = argmax_j  γ_n(j)

  Underflow (audit FR-2 / RB-9)
    Copula and margin densities are no longer floored at EPS, so a transition
    weight can legitimately be 0.0 in float64 — e.g. a Gaussian copula at
    τ = 0.95 gives log c ≈ −3000 on a discordant jump. When every weight
    reachable at a step underflows, the linear recursion gets C = 0: it used
    to floor C at MIN_POSITIVE, which *over*-states the likelihood (−708 nat
    instead of −3000) and zeroes α̂ for the rest of the sequence — one step
    destroyed every posterior before and after it. ``forward`` and
    ``backward`` now detect such a step (C or D not finite, or below
    MIN_POSITIVE) and redo the whole pass in log space from the exact log
    weights (:func:`_log_transition_weights`), which is exact wherever the
    log-densities are finite. A step whose weights are exactly zero in log
    space too (−∞: the observation is impossible under every reachable
    state) raises :class:`IncompatibleObservationError` in ``forward``, as
    Y[0] already did. The fast linear path is unchanged otherwise.

  Empty states
    A state i with π_i = 0 — an all-zero row and column of p. ICE leaves
    one when the state's density underflows to 0.0 at every row (γ and ξ
    are then exactly 0 there, and so is p̂ at the next M-step), SEM when a
    posterior draw never visits it; both report it (``degenerate_states``,
    π = 0). Such a state has α̂_n(i) = γ_n(i) = 0 at every n and changes
    nothing else: forward, backward, classify and FFBS give the results of
    the model without it, in linear and log space (to 8.9e-16 on α̂, γ and
    the PIT of :mod:`pmcprg.pmc.outliers`, equal log-likelihoods:
    ``test_empty_state_clip_corner.py``). Its row of log transition weights
    is −∞, not log 0 − log 0 = NaN.

Missing observations (NaN)
--------------------------
A row of Y holding a non-finite value (NaN, ±inf; for d > 1 any component)
is a missing observation, integrated out of the likelihood — the method is in
:mod:`pmcprg.pmc.gaps`. A Y without missing rows takes exactly the code path
above (bit-identical results). With missing rows:

* **Exact shortcut** — HMC-IN, HMC-IN2 and PMC-IN with state margins, whose
  transition A_ij f_j(y_{n+1}) does not depend on y_n. ``precompute_weights``
  returns the marginalised tensors (``f_pdf[n] = 1`` at a missing n, so
  ``W[n, i, j] = A_ij`` when y_{n+1} is missing and α_1(j) = Σ_i p_ij when
  y_1 is): ``forward``, ``backward``, ``smooth``, ``joint_posteriors``,
  ``sample_posterior``, ``classify`` and the log-space passes are exact on
  them, unchanged.
* **Quadrature grid** — HMC-DN, PMC, and PMC-IN with pair margins: the
  transition depends on the value of a missing y through the copula or the
  pair margins, so a gap is integrated on the augmented state (x_n, y_n ∈
  nodes). No K×K weight tensor represents this (the posterior of X is not
  Markov across a gap): ``precompute_weights`` raises ``ValueError`` and the
  functions that take a model — ``forward``, ``backward``, ``classify``,
  ``sample_posterior`` — dispatch to :mod:`pmcprg.pmc.gaps` (keyword
  ``gap_nodes``, default 64; passing ``W`` is refused). ``forward`` returns
  α̂_n(i) = P(x_n = i | observations up to n) at every n, ``backward`` a β̂
  with ``smooth(α̂, β̂)`` exact at every n (its definition at a missing n is in
  :mod:`pmcprg.pmc.gaps`); the pairwise posteriors ξ are
  ``pmcprg.pmc.gaps.gap_posterior(model, Y).xi``.

Non-ignorable missingness
-------------------------
When ``model.missingness`` is not None (``[missingness]`` table,
:mod:`pmcprg.pmc.missingness`) the mask m of Y — m_n = 1 at a missing row —
is evidence on the states, p(m | x) = Π_n e_n(x_n), and every function above
works with p(y_obs, m) (a complete Y has the mask m = 0). The factor e_n(i)
multiplies the message at position n componentwise: ``precompute_weights``
puts e_{n+1}(j) in the columns of W[n] and e_n(i) in f_pdf[n, i, :] (hence in
α_1), ``_log_transition_weights`` adds their logarithms, and the augmented
chain of :mod:`pmcprg.pmc.gaps` applies them after its block
renormalisation. With ``model.missingness is None`` nothing changes.

References
----------
* Devijver, P. A. (1985). Baum's forward-backward algorithm revisited.
  *Pattern Recognition Letters* 3(6), 369–373.
* Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected
  applications in speech recognition. *Proc. IEEE* 77(2), 257–286.
* Carter, C. K. & Kohn, R. (1994). On Gibbs sampling for state space
  models. *Biometrika* 81(3), 541–553; Frühwirth-Schnatter, S. (1994). Data
  augmentation and dynamic linear models. *J. Time Series Anal.* 15(2),
  183–202; Chib, S. (1996). Calculating posterior distributions and modal
  estimates in Markov mixture models. *J. Econometrics* 75(1), 79–97 —
  FFBS (:func:`sample_posterior`).
"""

import logging

import numpy as np

from pmcprg.exceptions    import IncompatibleObservationError
from pmcprg.pmc.model     import PMCModel, Variant
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, MIN_POSITIVE

logger = logging.getLogger(__name__)


__all__ = [
    "precompute_weights",
    "forward",
    "backward",
    "smooth",
    "joint_posteriors",
    "sample_posterior",
    "mpm",
    "classify",
    "classify_image",
    "error_rate",
]


# ---------------------------------------------------------------------------
# Missing observations — detection and dispatch (method in pmcprg.pmc.gaps)
# ---------------------------------------------------------------------------

def _missing_rows(Y) -> np.ndarray | None:
    """Mask (N,) of the rows of Y with a non-finite value, or ``None`` if none.

    ``None`` is the complete-data case: callers then run their unchanged code.
    """
    fin = np.isfinite(np.asarray(Y, dtype=float))
    if fin.all():
        return None
    if fin.ndim > 1:
        fin = fin.reshape(fin.shape[0], -1).all(axis=1)
    return ~fin


def _fill_missing(Y, miss: np.ndarray) -> np.ndarray:
    """Copy of Y with the missing rows set to 0.0 (overwritten by the caller)."""
    Yf = np.array(Y, dtype=float, copy=True)
    Yf[miss] = 0.0
    return Yf


def _grid_needed(model: PMCModel) -> bool:
    """Transition depends on y_n (copula or pair margins) — see :mod:`pmcprg.pmc.gaps`."""
    return bool(model.variant.uses_copula or model.margin_structure == "pair")


def _refuse_weights_with_gaps(model: PMCModel, what: str) -> None:
    raise ValueError(
        f"{what}: Y has missing observations and variant {model.variant.value} "
        f"({model.margin_structure} margins) links a missing y_n to its "
        f"neighbours (copula or pair margins): the gap is integrated on an "
        f"augmented state, which no (N-1, K, K) weight tensor represents. Call "
        f"forward / backward / classify / sample_posterior without W (they "
        f"handle NaN), or pmcprg.pmc.gaps.gap_posterior(model, Y) for α̂, β̂, γ, ξ."
    )


# ---------------------------------------------------------------------------
# Non-ignorable missingness — the evidence factors of the observed mask
# ---------------------------------------------------------------------------

#: Sentinel of the private ``ev`` arguments: the evidence factors of
#: ``model.missingness`` on the mask of ``Y`` (the default of every public
#: entry point). An explicit array (or ``None``, no factor) is passed by
#: :func:`pmcprg.pmc.gaps.forecast`, whose appended rows have no known mask,
#: and by ICE's ``"impute"`` strategy, whose completed series keep the mask of
#: the observed one.
_FROM_MODEL = object()


def _evidence(model: PMCModel, miss: np.ndarray) -> np.ndarray | None:
    """(N, K) factors e_n(i) = p(m_n | m_{n-1}, x_n = i) of the model's
    missingness mechanism on the boolean mask ``miss``; ``None`` when the
    mechanism is ignorable (``model.missingness is None``)."""
    mech = getattr(model, "missingness", None)
    return None if mech is None else mech.evidence(miss)


def _model_evidence(model: PMCModel, Y) -> np.ndarray | None:
    """:func:`_evidence` on the mask of the rows of ``Y`` (all observed if none)."""
    if getattr(model, "missingness", None) is None:
        return None
    miss = _missing_rows(Y)
    return _evidence(model, np.zeros(len(Y), bool) if miss is None else miss)


def _resolve_evidence(model: PMCModel, Y, ev):
    return _model_evidence(model, Y) if ev is _FROM_MODEL else ev


# ---------------------------------------------------------------------------
# Weight pre-computation
# ---------------------------------------------------------------------------

def precompute_weights(
    model: PMCModel,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute the transition tensor  W[n, i, j] = p(z_{n+1} | z_n)
    (module docstring) and the margin-PDF tensor  f_pdf[n, i, j] = f_ij(Y[n]).

    For state margins f_pdf[n, i, j] = f_i(Y[n]) is broadcast from K density
    evaluations; for pair margins (general PMC) the K² densities
    ``model.margin(i, j)`` are evaluated. In both cases the PMC-* weight is
    p[i,j] f_ij(y_n) / Σ_k p[i,k] f_ik(y_n) · f_ji(y_{n+1}) [· c_ij(F_ij(y_n),
    F_ji(y_{n+1}))] (DerrodePieczynski_CSDA2013 Eqs. 13–14); its relation to
    :meth:`PMCModel.weight` is documented there.

    Parameters
    ----------
    model : PMCModel
    Y     : np.ndarray, shape (N,) or (N, d)

    Returns
    -------
    W     : np.ndarray, shape (N-1, K, K)
    f_pdf : np.ndarray, shape (N,   K, K)

    Missing observations (non-finite rows of Y): for HMC-IN, HMC-IN2 and
    PMC-IN with state margins the returned tensors are marginalised exactly —
    ``f_pdf[n] = 1`` at a missing n (the integral of a density), whence
    ``W[n, i, j] = A_ij`` for a missing y_{n+1} — and every K-state function of
    this module is exact on them. For the other variants the gap needs the
    augmented state of :mod:`pmcprg.pmc.gaps` and ``ValueError`` is raised.

    Non-ignorable missingness (``model.missingness`` not None,
    :mod:`pmcprg.pmc.missingness`): the evidence factors e_n(i) = p(m_n |
    m_{n-1}, x_n = i) of the mask of Y live in the returned tensors,

        W[n, i, j]     ← W[n, i, j] · e_{n+1}(j),
        f_pdf[n, i, j] ← f_ij(y_n) · e_n(i)       (1 · e_n(i) at a missing n),

    so that W is the transition of (x, y, m) at the observed (y, m) and
    f_pdf[0] carries e_0 into the initialisation. Every function of this
    module that takes ``W`` / ``f_pdf`` — ``forward``, ``backward``,
    ``joint_posteriors``, ``sample_posterior`` — is then exact for
    p(y_obs, m) without knowing the mechanism, and a user call
    ``forward(model, Y, W=W, f_pdf=f_pdf)`` gives what ``forward(model, Y)``
    gives. A complete Y still has a mask (m_n = 0 everywhere) and gets the
    factors. With ``model.missingness is None`` the tensors are unchanged.
    """
    return _weights(model, Y, _model_evidence(model, Y))


def _weights(
    model: PMCModel,
    Y: np.ndarray,
    ev: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`precompute_weights` with explicit evidence factors ``ev`` (N, K)
    (``None``: none) — see there."""
    N   = len(Y)
    K   = model.K
    var = model.variant

    if N < 2:
        raise ValueError(
            f"precompute_weights requires N ≥ 2 observations, got N={N}. "
            f"With N < 2 there are no transitions and forward-backward is "
            f"undefined; use ``model.margin(...).pdf(Y[0])`` directly."
        )

    miss = _missing_rows(Y)
    if miss is not None:
        if _grid_needed(model):
            _refuse_weights_with_gaps(model, "precompute_weights")
        Y = _fill_missing(Y, miss)

    # ── Margin PDFs (and CDFs for copula variants) ────────────────────────
    # f_pdf[n, i, j] = f_ij(Y[n]), f_cdf[n, i, j] = F_ij(Y[n]).
    if model.margin_structure == "pair":
        # General PMC: K² distinct densities f_ij (DerrodePieczynski_CSDA2013 Eq. 12).
        f_pdf = np.empty((N, K, K))
        f_cdf = np.empty((N, K, K)) if var.uses_copula else None
        for ii in range(K):
            for jj in range(K):
                mg = model.margin(ii, jj)
                f_pdf[:, ii, jj] = mg.pdf_vec(Y)
                if f_cdf is not None:
                    f_cdf[:, ii, jj] = mg.cdf_vec(Y)
        if f_cdf is not None:
            np.clip(f_cdf, EPS, ONE_MINUS_EPS, out=f_cdf)
    else:
        # State margins: K densities f_i, evaluated once and broadcast into
        # the (N, K, K) shape (``f_pdf[n, i, j] = f_i(Y[n])`` for all j).
        f_pdf_state = np.zeros((N, K))
        f_cdf_state = np.zeros((N, K)) if var.uses_copula else None
        for kk in range(K):
            f_pdf_state[:, kk] = model.margin(kk).pdf_vec(Y)
            if f_cdf_state is not None:
                f_cdf_state[:, kk] = model.margin(kk).cdf_vec(Y)
        if f_cdf_state is not None:
            np.clip(f_cdf_state, EPS, ONE_MINUS_EPS, out=f_cdf_state)

        # ``broadcast_to`` returns a read-only view, so we copy to a writable
        # array (the caller stores it and may later mutate it).
        f_pdf = np.broadcast_to(f_pdf_state[:, :, None], (N, K, K)).copy()
        f_cdf = (np.broadcast_to(f_cdf_state[:, :, None], (N, K, K)).copy()
                 if f_cdf_state is not None else None)

    if miss is not None:
        # Exact shortcut: ∫ f(y) dy = 1 replaces the density of a missing row
        # (no copula here, and the PMC-IN ratio p_ij f_i / Σ_k p_ik f_i = A_ij).
        f_pdf[miss] = 1.0

    A = model.transition_A   # (K, K)
    p = model.prior_p        # (K, K)

    # ── Transition weights W ──────────────────────────────────────────────
    # W[n, i, j] = w(i, j, Y[n], Y[n+1])
    #
    # f_pdf[n+1].transpose(0,2,1)[i,j] = f_pdf[n+1, j, i] = f_{ji}(Y[n+1])
    # We build the "non-copula" prefix tensor in one vectorised expression,
    # then multiply by the copula PDF if needed (loop only over K² pairs).
    f_next_T = f_pdf[1:].transpose(0, 2, 1)   # (N-1, K, K) :  [n,i,j] = f_{ji}(Y[n+1])

    if var in (Variant.HMC_IN, Variant.HMC_IN2):
        # W[n, i, j] = A[i,j] · f_{ji}(Y[n+1])
        W = A[None, :, :] * f_next_T

    elif var == Variant.PMC_IN:
        # Joint pair density:
        #     joint[n, i, j] = p[i,j] · f_{ij}(Y[n]) · f_{ji}(Y[n+1])
        # The Devijver recursion ``α_{n+1}(j) = Σ_i α̂_n(i) · K(i,j; …)`` requires
        # the *conditional* transition kernel
        #     K(i, j; y_n, y_{n+1}) = joint[n, i, j] / D[n, i],
        # where  D[n, i] = p(X_n=i, Y_n=y_n) = Σ_{j'} p[i, j'] · f_{ij'}(Y[n]).
        # Without dividing by D, the recursion double-counts ``f_{ij}(Y[n])``
        # at each step and the resulting log-lik is wrong (off by an
        # additive Y-dependent term).
        joint = p[None, :, :] * f_pdf[:-1] * f_next_T
        D     = np.einsum("ij,nij->ni", p, f_pdf[:-1])     # (N-1, K)
        W     = joint / np.maximum(D, MIN_POSITIVE)[:, :, None]

    elif var in (Variant.HMC_DN, Variant.PMC):
        # Build the non-copula prefix, then multiply by c_{ij}(F_{ij}(Y[n]), F_{ji}(Y[n+1]))
        if var == Variant.HMC_DN:
            W = A[None, :, :] * f_next_T
        else:  # PMC — same divide-by-D[n,i] correction as PMC_IN above.
            joint = p[None, :, :] * f_pdf[:-1] * f_next_T
            D     = np.einsum("ij,nij->ni", p, f_pdf[:-1])
            W     = joint / np.maximum(D, MIN_POSITIVE)[:, :, None]

        # Vectorised copula PDF — one call per (i,j) pair returning an (N-1,) array.
        for ii in range(K):
            for jj in range(K):
                cop = model.copula(ii, jj)
                uv  = np.column_stack((f_cdf[: N - 1, ii, jj],
                                       f_cdf[1:,      jj, ii]))
                W[:, ii, jj] *= cop.pdf_array(uv)

    # A non-finite weight is a failed density evaluation, not a probability:
    # report it and give the transition no weight (log-space convention of
    # the likelihood objectives, RB-7). Densities are no longer floored at
    # EPS, so the kernels' failures reach this point instead of being hidden.
    bad = ~np.isfinite(W)
    if bad.any():
        pairs = sorted({(int(i), int(j)) for i, j in zip(*np.nonzero(bad.any(axis=0)))})
        logger.warning(
            "precompute_weights: %d non-finite transition weight(s) (pairs %s) — "
            "a margin or copula density could not be evaluated; set to 0.",
            int(bad.sum()), pairs,
        )
        W[bad] = 0.0

    # Guard against numerical zeros / negatives
    np.clip(W, 0.0, None, out=W)

    if ev is not None:
        # Missingness evidence (docstring of precompute_weights): a likelihood
        # factor of the destination state, applied after the transition.
        W *= ev[1:, None, :]
        f_pdf *= ev[:, :, None]

    logger.debug(
        "precompute_weights: variant=%s  N=%d  K=%d  W.min=%.3e  W.max=%.3e",
        var.value, N, K, W.min(), W.max(),
    )
    return W, f_pdf


# ---------------------------------------------------------------------------
# Log-space weights — the fallback of forward/backward on underflow
# ---------------------------------------------------------------------------

def _lse(a: np.ndarray, axis=None) -> np.ndarray:
    """``log Σ exp(a)`` along ``axis``; ``−∞`` for an all-``−∞`` slice."""
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out.reshape(()) if axis is None else np.squeeze(out, axis=axis)


def _margin_logpdf_vec(margin, Y: np.ndarray) -> np.ndarray:
    """log f(Y) for a model margin, exact where ``pdf_vec`` underflows."""
    frozen = getattr(margin, "_frozen", None)
    with np.errstate(divide="ignore", invalid="ignore", under="ignore"):
        if frozen is not None and hasattr(frozen, "logpdf"):
            return np.asarray(frozen.logpdf(np.asarray(Y, dtype=float)), dtype=float)
        return np.log(np.asarray(margin.pdf_vec(Y), dtype=float))


def _log_transition_weights(
    model: PMCModel,
    Y: np.ndarray,
    *,
    ev=_FROM_MODEL,
) -> tuple[np.ndarray, np.ndarray]:
    """``(log W, log α_1)`` — the exact logarithms of what :func:`precompute_weights`
    and the forward initialisation form in linear space.

    ``log W[n, i, j]`` has shape (N-1, K, K) and ``log α_1`` (unnormalised)
    shape (K,). Built from log margin densities and ``logpdf_array`` of the
    copulas, so a weight whose linear value underflows to 0.0 keeps its finite
    logarithm. Non-finite values other than −∞ (a failed evaluation) are
    mapped to −∞, like :func:`precompute_weights` maps them to 0.

    * State margins: for the PMC variants the factor ``f_i(y_n)`` that
      ``joint / D`` cancels in linear space is cancelled symbolically,
      ``W = p[i,j] / Σ_j' p[i,j'] · f_j(y_{n+1}) [· c_ij]``, and
      ``log α_1[j] = log Σ_i p[i,j] + log f_j(y_1)``.
    * Pair margins (general PMC): f_ij(y_n) depends on j, so nothing
      cancels — ``log W = log p[i,j] + log f_ij(y_n) − log Σ_k p[i,k] f_ik(y_n)
      + log f_ji(y_{n+1}) [+ log c_ij(F_ij(y_n), F_ji(y_{n+1}))]`` and
      ``log α_1[i] = log Σ_j p[i,j] f_ij(y_1)``
      (DerrodePieczynski_CSDA2013 Eqs. 13–14, §3.1).

    Missing rows of Y (exact-shortcut variants only, as in
    :func:`precompute_weights`): ``log f = 0`` there.

    Missingness evidence (``ev``, by default from ``model.missingness`` and
    the mask of Y): ``log e_{n+1}(j)`` is added to ``log W[n, :, j]`` and
    ``log e_0`` to ``log α_1`` — the logarithms of the factors of
    :func:`precompute_weights` (−∞ where e = 0).
    """
    N, K, var = len(Y), model.K, model.variant
    ev = _resolve_evidence(model, Y, ev)

    miss = _missing_rows(Y)
    if miss is not None:
        if _grid_needed(model):
            _refuse_weights_with_gaps(model, "_log_transition_weights")
        Y = _fill_missing(Y, miss)

    with np.errstate(divide="ignore", invalid="ignore"):
        if var.has_markov_prior:
            log_kernel = np.log(model.transition_A)                      # (K, K)
        else:
            p = model.prior_p
            rows = p.sum(axis=1, keepdims=True)
            # An empty state (all-zero row of p, e.g. emptied by ICE/SEM) has
            # no transition out of it: −∞, not log 0 − log 0 = NaN (and a
            # RuntimeWarning). Its forward probability is 0, so any value
            # works for the likelihood; −∞ keeps the pass NaN-free.
            log_kernel = np.where(rows > 0.0, np.log(p) - np.log(rows), -np.inf)
        log_p = np.log(model.prior_p)

    if model.margin_structure == "pair":
        # log f_ij(Y[n]), shape (N, K, K).
        log_f = np.empty((N, K, K))
        for ii in range(K):
            for jj in range(K):
                log_f[:, ii, jj] = _margin_logpdf_vec(model.margin(ii, jj), Y)
        with np.errstate(invalid="ignore"):
            left  = log_p[None, :, :] + log_f[:-1]                       # log p_ij f_ij(y_n)
            log_d = _lse(left, axis=2)                                   # log Σ_k p_ik f_ik(y_n)
            logW  = left - log_d[:, :, None] + log_f[1:].transpose(0, 2, 1)
        if var.uses_copula:
            f_cdf = np.empty((N, K, K))
            for ii in range(K):
                for jj in range(K):
                    f_cdf[:, ii, jj] = model.margin(ii, jj).cdf_vec(Y)
            np.clip(f_cdf, EPS, ONE_MINUS_EPS, out=f_cdf)
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((f_cdf[: N - 1, ii, jj], f_cdf[1:, jj, ii]))
                    logW[:, ii, jj] += model.copula(ii, jj).logpdf_array(uv)
        log_a1 = _lse(log_p + log_f[0], axis=1)                          # Σ_j p[i,j] f_ij(y_0)
    else:
        log_f = np.empty((N, K))
        for k in range(K):
            log_f[:, k] = _margin_logpdf_vec(model.margin(k), Y)
        if miss is not None:
            log_f[miss] = 0.0                                            # log ∫ f = 0

        logW = log_kernel[None, :, :] + log_f[1:, None, :]               # (N-1, K, K)

        if var.uses_copula:
            f_cdf = np.empty((N, K))
            for k in range(K):
                f_cdf[:, k] = model.margin(k).cdf_vec(Y)
            np.clip(f_cdf, EPS, ONE_MINUS_EPS, out=f_cdf)
            for ii in range(K):
                for jj in range(K):
                    uv = np.column_stack((f_cdf[: N - 1, ii], f_cdf[1:, jj]))
                    logW[:, ii, jj] += model.copula(ii, jj).logpdf_array(uv)

        log_a1 = _lse(log_p, axis=0) + log_f[0]                          # Σ_i p[i,j] f_j(y_0)

    for arr in (logW, log_a1):
        arr[np.isnan(arr) | (arr == np.inf)] = -np.inf
    if ev is not None:
        with np.errstate(divide="ignore"):
            log_ev = np.log(ev)
        logW = logW + log_ev[1:, None, :]
        log_a1 = log_a1 + log_ev[0]
    return logW, log_a1


def _forward_log_space(model: PMCModel, Y: np.ndarray, *,
                       ev=_FROM_MODEL) -> tuple[np.ndarray, float]:
    """The normalised forward pass of :func:`forward`, carried in log space."""
    N, K = len(Y), model.K
    logW, log_a = _log_transition_weights(model, Y, ev=ev)
    alpha_hat = np.empty((N, K))
    log_c = float(_lse(log_a))
    if not np.isfinite(log_c):
        y0_repr = np.array2string(np.asarray(Y[0]), precision=4, separator=", ")
        raise IncompatibleObservationError(
            f"Forward pass: marginal density of Y[0]={y0_repr} is zero under "
            f"every state (log C_1 = {log_c!r})."
        )
    log_lik = log_c
    la = log_a - log_c
    alpha_hat[0] = np.exp(la)
    for n in range(N - 1):
        r = _lse(la[:, None] + logW[n], axis=0)
        log_c = float(_lse(r))
        if not np.isfinite(log_c):
            y_repr = np.array2string(np.asarray(Y[n + 1]), precision=4, separator=", ")
            raise IncompatibleObservationError(
                f"Forward pass: Y[{n + 1}]={y_repr} has zero density under "
                f"every state reachable from step {n} (log C = {log_c!r}), "
                f"even in log space. The observation sequence is incompatible "
                f"with the model."
            )
        log_lik += log_c
        la = r - log_c
        alpha_hat[n + 1] = np.exp(la)
    return alpha_hat, float(log_lik)


def _backward_log_space(model: PMCModel, Y: np.ndarray, *, ev=_FROM_MODEL) -> np.ndarray:
    """The normalised backward pass of :func:`backward`, carried in log space."""
    N, K = len(Y), model.K
    logW, _ = _log_transition_weights(model, Y, ev=ev)
    beta_hat = np.empty((N, K))
    lb = np.full(K, -np.log(K))
    beta_hat[N - 1] = 1.0 / K
    n_void = 0
    for n in range(N - 2, -1, -1):
        r = _lse(logW[n] + lb[None, :], axis=1)
        log_d = float(_lse(r))
        if np.isfinite(log_d):
            lb = r - log_d
        else:
            n_void += 1
            lb = np.full(K, -np.log(K))
        beta_hat[n] = np.exp(lb)
    if n_void:
        logger.warning(
            "Backward (log space): %d step(s) with zero weight under every "
            "state even in log space — β̂ reset to uniform there.", n_void,
        )
    return beta_hat


# ---------------------------------------------------------------------------
# Forward pass  (Devijver 1985 normalization)
# ---------------------------------------------------------------------------

def forward(
    model: PMCModel,
    Y: np.ndarray,
    W: np.ndarray | None = None,
    f_pdf: np.ndarray | None = None,
    *,
    gap_nodes: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Normalized forward pass (Devijver 1985 / Baum-Welch).

    Parameters
    ----------
    model : PMCModel
    Y     : np.ndarray, shape (N,) or (N, d); non-finite rows are missing.
    W     : pre-computed weight tensor (N-1, K, K).  Computed if None.
    f_pdf : pre-computed margin-PDF tensor (N, K, K).  Computed with W if None.
    gap_nodes : quadrature nodes for the missing rows of the grid variants
            (default 64, see :mod:`pmcprg.pmc.gaps`); ignored otherwise.

    Returns
    -------
    alpha_hat : np.ndarray, shape (N, K)   — normalized forward variables
    log_lik   : float                       — log-likelihood  log p(y_{1:N})

    Underflow: when a step's normaliser C is not finite or below
    ``MIN_POSITIVE`` — every weight reachable from α̂_n underflowed — the
    pass is recomputed in log space from the model (see the module
    docstring); ``W`` must therefore be ``precompute_weights(model, Y)``.
    A step impossible even in log space raises
    :class:`IncompatibleObservationError`.

    Missing rows (module docstring): the exact-shortcut variants run this
    recursion on the marginalised ``precompute_weights``; the grid variants
    run the augmented chain of :mod:`pmcprg.pmc.gaps` (``W`` must then be None),
    ``alpha_hat[n, i]`` being P(x_n = i | observations up to n) at every n and
    ``log_lik`` the observed-data log-likelihood.

    Non-ignorable missingness (``model.missingness`` not None): ``alpha_hat``
    and ``log_lik`` are those of p(y_obs, m) — the mask up to n is part of
    the conditioning; the factors live in ``W`` and ``f_pdf``
    (:func:`precompute_weights`), which must then come from the same model.
    """
    return _forward(model, Y, W, f_pdf, gap_nodes=gap_nodes, ev=_FROM_MODEL)


def _forward(model, Y, W, f_pdf, *, gap_nodes=None, ev=_FROM_MODEL):
    """:func:`forward` with explicit evidence factors ``ev`` (``_FROM_MODEL``:
    those of the model on the mask of Y); ``W``/``f_pdf`` must carry them."""
    N = len(Y)
    K = model.K

    miss = _missing_rows(Y)
    if miss is not None and _grid_needed(model):
        if W is not None:
            _refuse_weights_with_gaps(model, "forward")
        from pmcprg.pmc import gaps
        return gaps._grid_forward(model, Y, miss, gap_nodes, ev=ev)

    ev = _resolve_evidence(model, Y, ev)
    if W is None:
        W, f_pdf = _weights(model, Y, ev)

    alpha_hat = np.zeros((N, K))
    log_lik   = 0.0
    p         = model.prior_p   # (K, K)

    # ── Initialization: α_1 = p(x_1, y_1) ────────────────────────────────
    if model.margin_structure == "pair":
        # α_1(i) = Σ_j p[i,j] · f_ij(Y[0]): y_1 follows the mixture of the
        # left margins of the pairs (x_1, j) (DerrodePieczynski_CSDA2013 §3.1).
        alpha_1 = np.einsum("ij,ij->i", p, f_pdf[0])
    else:
        # State margins: α_1(j) = Σ_i p[i,j] · f_j(Y[0]).
        alpha_1 = np.einsum("ij,ji->j", p, f_pdf[0])
    C_1     = float(alpha_1.sum())
    if not np.isfinite(C_1) or C_1 <= 0.0:
        # Genuine modelling error: Y[0] has zero density under every state.
        # Continuing with a synthetic floor would silently lie to the caller.
        # ``Y[0]`` is a scalar in the d=1 case but a vector for multivariate
        # observations — use ``np.array2string`` for a uniform short repr.
        y0_repr = np.array2string(np.asarray(Y[0]), precision=4, separator=", ")
        raise IncompatibleObservationError(
            f"Forward pass: marginal density of Y[0]={y0_repr} is zero or "
            f"non-finite under every state (C_1={C_1!r}). The observation "
            f"sequence is incompatible with the model — check margin "
            f"parameters or for outliers."
        )
    log_lik      += np.log(C_1)
    alpha_hat[0]  = alpha_1 / C_1

    # ── Recursion ─────────────────────────────────────────────────────────
    for n in range(N - 1):
        # α_{n+1}(j) = Σ_i α̂_n(i) · W[n, i, j]
        alpha_raw = alpha_hat[n] @ W[n]   # shape (K,)
        C = float(alpha_raw.sum())
        if not (np.isfinite(C) and C >= MIN_POSITIVE):
            # Every weight reachable from α̂_n underflowed. Flooring C (the
            # former behaviour) over-stated the likelihood and zeroed α̂ for
            # the rest of the sequence; the log-space pass is exact.
            logger.warning(
                "Forward: C=%.3e at step n=%d — the transition weights "
                "underflow; recomputing the pass in log space.",
                C, n + 1,
            )
            return _forward_log_space(model, Y, ev=ev)
        log_lik         += np.log(C)
        alpha_hat[n + 1] = alpha_raw / C

    return alpha_hat, float(log_lik)


# ---------------------------------------------------------------------------
# Backward pass  (normalised; see the module docstring for how the scaling
# differs from Devijver 1985)
# ---------------------------------------------------------------------------

def backward(
    model: PMCModel,
    Y: np.ndarray,
    W: np.ndarray | None = None,
    *,
    gap_nodes: int | None = None,
) -> np.ndarray:
    """
    Normalized backward pass.

    Each β̂_n is rescaled by its own sum (Rabiner 1989 §V.A-type scaling),
    not by the forward constants C_{n+1} as in Devijver (1985) — harmless
    because :func:`smooth` and :func:`joint_posteriors` renormalise
    (audit K-13).

    Parameters
    ----------
    model : PMCModel
    Y     : np.ndarray, shape (N,) or (N, d); non-finite rows are missing.
    W     : pre-computed weight tensor (N-1, K, K).  Computed if None.
    gap_nodes : quadrature nodes for the missing rows of the grid variants
            (default 64, see :mod:`pmcprg.pmc.gaps`); ignored otherwise.

    Returns
    -------
    beta_hat : np.ndarray, shape (N, K)

    Underflow: a step whose sum D is not finite or below ``MIN_POSITIVE``
    makes the pass restart in log space from the model, as in
    :func:`forward` (the former floor zeroed β̂ for every earlier step).

    Missing rows: exact-shortcut variants as in :func:`forward`. For the grid
    variants (``W`` must be None) β̂ at a missing n depends on the forward
    messages — ``β̂_n(i) ∝ Σ_g α̃_n(i, g) β̃_n(i, g) / Σ_g α̃_n(i, g)`` — so the
    forward pass is run too; ``smooth(forward(...)[0], backward(...))`` is
    then exact at every n (:mod:`pmcprg.pmc.gaps`).

    Non-ignorable missingness: β̂ carries the factors of the later masks, as
    ``W`` does (:func:`precompute_weights`).
    """
    return _backward(model, Y, W, gap_nodes=gap_nodes, ev=_FROM_MODEL)


def _backward(model, Y, W, *, gap_nodes=None, ev=_FROM_MODEL):
    """:func:`backward` with explicit evidence factors ``ev`` (see :func:`_forward`)."""
    N = len(Y)
    K = model.K

    miss = _missing_rows(Y)
    if miss is not None and _grid_needed(model):
        if W is not None:
            _refuse_weights_with_gaps(model, "backward")
        from pmcprg.pmc import gaps
        return gaps._grid_backward(model, Y, miss, gap_nodes, ev=ev)

    ev = _resolve_evidence(model, Y, ev)
    if W is None:
        W, _ = _weights(model, Y, ev)

    beta_hat = np.zeros((N, K))

    # Initialization: uniform at step N
    beta_hat[N - 1] = 1.0 / K

    # Recursion (backward)
    for n in range(N - 2, -1, -1):
        # β_n(i) = Σ_j W[n, i, j] · β̂_{n+1}(j)
        beta_raw = W[n] @ beta_hat[n + 1]   # shape (K,)
        D = float(beta_raw.sum())
        if not (np.isfinite(D) and D >= MIN_POSITIVE):
            logger.warning(
                "Backward: D=%.3e at step n=%d — the transition weights "
                "underflow; recomputing the pass in log space.", D, n,
            )
            return _backward_log_space(model, Y, ev=ev)
        beta_hat[n] = beta_raw / D

    return beta_hat


# ---------------------------------------------------------------------------
# Posterior marginals
# ---------------------------------------------------------------------------

def smooth(
    alpha_hat: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """
    Compute the posterior marginals  γ_n(j) = P(X_n=j | y_{1:N}).

    Parameters
    ----------
    alpha_hat : (N, K)
    beta_hat  : (N, K)

    Returns
    -------
    gamma : (N, K)  — each row sums to 1.
    """
    gamma = alpha_hat * beta_hat                   # element-wise
    row_sums = gamma.sum(axis=1, keepdims=True)
    # A degenerate row (≈ 0 everywhere — only in pathological / underflowed
    # sequences) carries no information. Fall back to a *uniform* posterior 1/K
    # rather than a zero row: zeros bias mpm's argmax (and any γ-weighted
    # downstream step, e.g. the M-step) toward state 0, whereas 1/K is the
    # correct "no information" prior and keeps every row a valid distribution
    # summing to 1 (audit N-8 — matches sample_posterior's degenerate fallback).
    K          = gamma.shape[1]
    degenerate = row_sums <= 0
    safe_sums  = np.where(degenerate, 1.0, row_sums)
    return np.where(degenerate, 1.0 / K, gamma / safe_sums)


def joint_posteriors(
    alpha_hat: np.ndarray,
    W: np.ndarray,
    beta_hat: np.ndarray,
) -> np.ndarray:
    """Pairwise posteriors  ξ_n(i, j) = P(X_n=i, X_{n+1}=j | Y), n = 0 … N-2.

    The pairwise companion of :func:`smooth` (which returns the marginal
    γ_n). Fully vectorised::

        ξ_n(i, j) ∝ α̂_n(i) · W[n, i, j] · β̂_{n+1}(j)

    with a per-step normalisation. A step whose un-normalised mass is zero —
    its transition weights underflowed to 0.0 — falls back to the product
    ``γ_n(i) · γ_{n+1}(j)`` of the smoothed marginals (:func:`smooth`), the
    coupling that keeps **both** marginal-consistency invariants
    γ_n(i) = Σ_j ξ_n(i, j) and γ_{n+1}(j) = Σ_i ξ_n(i, j) (audit N-11). The
    former uniform 1/K² kept them only while γ itself was uniform, which was
    the case when an underflowed step zeroed α̂ and β̂; since
    :func:`forward` and :func:`backward` recover from underflow in log space,
    γ around such a step is informative. With uniform γ the two coincide.

    Parameters
    ----------
    alpha_hat : (N, K) — normalised forward variables (:func:`forward`).
    W         : (N-1, K, K) — transition-weight tensor (:func:`precompute_weights`).
    beta_hat  : (N, K) — normalised backward variables (:func:`backward`).

    Returns
    -------
    xi : (N-1, K, K)
    """
    # alpha_hat[:-1, :, None] : (N-1, K, 1)
    # beta_hat[1:,  None, :]  : (N-1, 1, K)
    xi = alpha_hat[:-1, :, None] * W * beta_hat[1:, None, :]

    # Per-step normalisation; a zero-mass step falls back to γ_n ⊗ γ_{n+1}
    # (see docstring — keeps both marginals of ξ_n consistent with smooth).
    sums = xi.sum(axis=(1, 2), keepdims=True)
    void = ~(np.isfinite(sums) & (sums > 0)).reshape(-1)
    np.divide(xi, sums, out=xi, where=~void[:, None, None])
    if void.any():
        idx   = np.nonzero(void)[0]
        gamma = smooth(alpha_hat[np.concatenate((idx, idx + 1))],
                       beta_hat[np.concatenate((idx, idx + 1))])
        g_n, g_next = gamma[: idx.size], gamma[idx.size:]
        xi[idx] = g_n[:, :, None] * g_next[:, None, :]
    return xi


# ---------------------------------------------------------------------------
# Forward-Filter Backward-Sample  (FFBS — posterior draw of X | Y)
# ---------------------------------------------------------------------------

def sample_posterior(
    model: PMCModel,
    Y: np.ndarray,
    rng: np.random.Generator,
    *,
    W: np.ndarray | None = None,
    f_pdf: np.ndarray | None = None,
    alpha_hat: np.ndarray | None = None,
    gap_nodes: int | None = None,
    return_y: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Draw one realisation of the state sequence  X̃ ~ P(X | Y).

    Implements **Forward-Filter Backward-Sample (FFBS)** — the standard
    posterior sampler for state-space models (Carter & Kohn 1994;
    Frühwirth-Schnatter 1994; and, for the discrete-state case used here,
    Chib 1996). DerrodePieczynski_CSDA2013 Eq. 23 samples *forward* instead, along
    ``p(x_{n+1} | x_n, y_{1:N})`` — the same posterior law, drawn in the
    other direction.

    1. *Filter*: compute (or accept pre-computed) ``α̂_n(j) = P(X_n=j | Y_{1:n})``.
    2. *Sample* ``X̃_{N-1} ~ α̂_{N-1}`` (the last filtered posterior is the
       smoothed posterior because there is no future observation).
    3. *Backward-sample*, for ``n = N-2, …, 0``::

           P(X_n=i | X̃_{n+1}, Y_{1:N}) ∝ α̂_n(i) · W[n, i, X̃_{n+1}]

       Z = (X, Y) being Markov (DerrodePieczynski_CSDA2013 Eq. 4), ``X_n``
       and ``Y_{n+2:N}`` are independent given
       ``Z_{n+1} = (X_{n+1}, Y_{n+1})``, which makes this sampler exact for
       pair margins (general PMC, X alone not Markov) as well as for state
       margins.

    Used by SEM (Stochastic EM) to obtain the hard pseudo-labels on
    which the M-step is then run.

    Parameters
    ----------
    model     : PMCModel
    Y         : np.ndarray, shape (N,) or (N, d).
    rng       : :class:`numpy.random.Generator` — source of randomness.
    W, f_pdf, alpha_hat : optional pre-computed tensors (skip re-computation).
    gap_nodes : quadrature nodes for the missing rows of the grid variants
                (default 64); ignored otherwise.
    return_y  : also return Y with its missing rows drawn jointly with X.

    Returns
    -------
    X_sample  : np.ndarray, shape (N,), dtype int — one draw from P(X | Y).
    Y_sample  : (only with ``return_y``) copy of Y whose missing rows are drawn
                from P(y_miss | X̃, y_obs) — jointly with X̃.

    Missing rows (non-finite rows of Y): the draw is from P(X | y_obs). The
    exact-shortcut variants (HMC-IN, HMC-IN2, PMC-IN with state margins) run
    the FFBS below on the marginalised weights, then draw each missing
    y_n ~ f_{X̃_n}. The grid variants run FFBS on the augmented chain of
    :mod:`pmcprg.pmc.gaps` (``W``/``alpha_hat`` must be None); the missing y are
    then drawn on its quadrature nodes.

    Non-ignorable missingness (``model.missingness`` not None): the draw is
    from P(X | y_obs, m) — the factors live in ``W`` (:func:`precompute_weights`)
    and in the augmented chain; given X̃ the missing y are drawn as above
    (y_miss does not depend on m given the states).
    """
    N = len(Y)
    K = model.K
    if N < 2:
        raise ValueError(
            f"sample_posterior requires N ≥ 2 observations, got N={N}."
        )

    miss = _missing_rows(Y)
    if miss is not None and _grid_needed(model):
        if W is not None or alpha_hat is not None:
            _refuse_weights_with_gaps(model, "sample_posterior")
        from pmcprg.pmc import gaps
        return gaps._grid_sample(model, Y, miss, rng, gap_nodes, return_y)

    if W is None or f_pdf is None:
        W, f_pdf = precompute_weights(model, Y)
    if alpha_hat is None:
        alpha_hat, _ = forward(model, Y, W=W, f_pdf=f_pdf)

    X = np.empty(N, dtype=int)

    # ── Step 1: draw X[N-1] ~ α̂_{N-1} ────────────────────────────────────
    p_last = alpha_hat[N - 1].astype(float, copy=True)
    s = p_last.sum()
    if not np.isfinite(s) or s <= 0.0:
        p_last = np.full(K, 1.0 / K)
    else:
        p_last /= s
    X[N - 1] = int(rng.choice(K, p=p_last))

    # ── Step 2: backward sample ───────────────────────────────────────────
    # P(X_n = i | X_{n+1}, Y_{1:N}) ∝ α̂_n(i) · W[n, i, X_{n+1}].
    for n in range(N - 2, -1, -1):
        weights = alpha_hat[n] * W[n, :, X[n + 1]]
        s = float(weights.sum())
        if not np.isfinite(s) or s <= 0.0:
            # Degenerate step (numerical underflow): fall back to the
            # filtered marginal — still a valid draw from a proper
            # distribution, just not the exact backward kernel.
            logger.debug(
                "sample_posterior: degenerate backward step at n=%d "
                "(weight sum=%.3e); falling back to α̂_n.", n, s,
            )
            weights = alpha_hat[n].astype(float, copy=True)
            s = float(weights.sum())
            if not np.isfinite(s) or s <= 0.0:
                weights = np.full(K, 1.0 / K)
            else:
                weights = weights / s
        else:
            weights = weights / s
        X[n] = int(rng.choice(K, p=weights))

    if not return_y:
        return X
    Y_sample = np.array(Y, dtype=float, copy=True)
    if miss is not None:
        from pmcprg.pmc import gaps
        Y_sample[miss] = gaps._draw_state_margins(model, X[miss], rng)
    return X, Y_sample


# ---------------------------------------------------------------------------
# MPM classifier
# ---------------------------------------------------------------------------

def mpm(gamma: np.ndarray) -> np.ndarray:
    """
    Marginal Posterior Mode (MPM) classification.

    Parameters
    ----------
    gamma : (N, K) — posterior marginals (rows sum to 1).

    Returns
    -------
    X_hat : (N,) int — state sequence maximising the marginal posterior.
    """
    return np.argmax(gamma, axis=1).astype(int)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

def classify(
    model: PMCModel,
    Y: np.ndarray,
    *,
    gap_nodes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Supervised MPM classification of the observation sequence Y.

    Runs the full forward-backward-smooth-MPM pipeline.

    Parameters
    ----------
    model : PMCModel — fully specified model (parameters known).
    Y     : np.ndarray, shape (N,) or (N, d) — observed sequence; non-finite
            rows are missing observations, integrated out (module docstring).
    gap_nodes : quadrature nodes for the missing rows of the grid variants
            (default 64, see :mod:`pmcprg.pmc.gaps`); ignored otherwise.

    Returns
    -------
    X_hat     : np.ndarray, shape (N,) int — MPM class labels (every n,
                missing rows included).
    gamma     : np.ndarray, shape (N, K)   — posterior marginals P(x_n | y_obs).
    log_lik   : float                       — log p(y_obs | model), the
                observed-data log-likelihood (log p(y_{1:N}) without gaps);
                log p(y_obs, m | model) when ``model.missingness`` is not
                None, γ being then P(x_n | y_obs, m).
    """
    miss = _missing_rows(Y)
    if miss is not None and _grid_needed(model):
        from pmcprg.pmc import gaps
        post = gaps.gap_posterior(model, Y, gap_nodes=gap_nodes, xi=False)
        return mpm(post.gamma), post.gamma, post.log_lik

    W, f_pdf = precompute_weights(model, Y)

    alpha_hat, log_lik = forward(model, Y, W=W, f_pdf=f_pdf)
    beta_hat            = backward(model, Y, W=W)

    gamma = smooth(alpha_hat, beta_hat)
    X_hat = mpm(gamma)

    return X_hat, gamma, log_lik


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def classify_image(
    model: PMCModel,
    img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Supervised MPM classification of a 2D image (mono- or multi-channel).

    The image is linearised along the Generalized Hilbert ("gilbert") path,
    classified with :func:`classify`, then re-folded into 2D maps.

    Parameters
    ----------
    model : PMCModel — fully specified model (parameters known).
    img   : np.ndarray
        Shape ``(H, W)`` for grayscale (when ``model.d == 1``) or
        ``(H, W, d)`` for multi-channel (when ``model.d > 1``).

    Returns
    -------
    X_hat_2d : np.ndarray, shape (H, W), int — MPM class-label map.
    gamma_2d : np.ndarray, shape (H, W, K)   — per-pixel posterior marginals.
    log_lik  : float                          — log p(image | model).

    Notes
    -----
    The class-label map is always 2D (segmentation is per-pixel, scalar)
    regardless of the input dimensionality. Posterior marginals are over
    states, so their last axis is always ``K`` — independent of ``model.d``.
    """
    from pmcprg.pmc.peano import image_to_signal, signal_to_image

    if img.ndim not in (2, 3):
        raise ValueError(
            f"classify_image expects 2D (H, W) or 3D (H, W, d); "
            f"got shape {img.shape}."
        )
    img_d = 1 if img.ndim == 2 else img.shape[2]
    if img_d != model.d:
        raise ValueError(
            f"Image channels ({img_d}) do not match model.d = {model.d}. "
            f"Use load_grayscale for a d=1 model or load_color for d=3."
        )

    Y = image_to_signal(img)                 # (N,) if d=1, (N, d) if d>1
    X_hat, gamma, log_lik = classify(model, Y)

    H, W = img.shape[:2]
    X_hat_2d = signal_to_image(X_hat, (H, W))
    K        = gamma.shape[1]
    gamma_2d = np.empty((H, W, K), dtype=gamma.dtype)
    for k in range(K):
        gamma_2d[..., k] = signal_to_image(gamma[:, k], (H, W))

    return X_hat_2d, gamma_2d, log_lik


def error_rate(X_true: np.ndarray, X_hat: np.ndarray) -> float:
    """
    Classification error rate (fraction of misclassified labels), invariant
    under label permutation.

    For unsupervised classifiers, predicted class indices are arbitrary. We
    therefore find the optimal one-to-one relabeling π̂ of predicted labels
    that minimises ``mean(X_true != π̂(X_hat))`` and return that minimum.

    Implementation: build the K×K confusion matrix C[k, ℓ] = #{n : X_true=k, X_hat=ℓ}
    then solve the linear assignment problem (Kuhn-Munkres / Hungarian) on
    ``-C`` to maximise total agreement.

    Parameters
    ----------
    X_true : np.ndarray, shape (N,) — ground-truth labels in {0, …, K-1}.
    X_hat  : np.ndarray, shape (N,) — predicted labels (any integer encoding).

    Returns
    -------
    float — minimum error rate over all label permutations, in [0, 1].
    """
    from scipy.optimize import linear_sum_assignment

    X_true = np.asarray(X_true, dtype=int)
    X_hat  = np.asarray(X_hat,  dtype=int)
    if X_true.shape != X_hat.shape:
        raise ValueError(
            f"X_true {X_true.shape} and X_hat {X_hat.shape} must have same shape."
        )

    N = X_true.size
    K = int(max(X_true.max(), X_hat.max())) + 1

    # Confusion matrix (rectangular if X_hat uses fewer/more labels)
    C = np.zeros((K, K), dtype=int)
    for t, h in zip(X_true, X_hat):
        C[t, h] += 1

    # Diagnostics on the number of classes. Only counts matter: the error is
    # invariant under relabeling, so label *values* need not match — a
    # reference image coded {0, 255} against predictions {0, 1} is fine.
    n_true = int(np.unique(X_true).size)
    n_hat  = int(np.unique(X_hat).size)
    if n_hat < n_true:
        # X_hat collapsed onto fewer classes than the truth — typical symptom
        # of a degenerate ICE solution. The Hungarian still picks an optimal
        # partial relabeling.
        logger.warning(
            "error_rate: X_hat uses %d classes, the reference %d. The fitted "
            "model likely collapsed to a degenerate solution; the returned "
            "error rate is still the optimal-relabeling value (correct as a "
            "metric), but a fit worth trusting should use every class.",
            n_hat, n_true,
        )
    elif n_hat > n_true:
        # More predicted classes than true ones — e.g. ICE run with a K larger
        # than the number of true classes. The Hungarian leaves the extra
        # predicted classes unmatched (counted as errors).
        logger.warning(
            "error_rate: X_hat uses %d classes, the reference only %d; the "
            "extra classes count as errors in the returned rate (this is not "
            "a bug in the metric) — consider whether K was specified "
            "correctly.",
            n_hat, n_true,
        )

    # Hungarian on -C maximises agreement (= minimises misclassification)
    row_ind, col_ind = linear_sum_assignment(-C)
    correct = C[row_ind, col_ind].sum()

    return float(1.0 - correct / N)


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    models_dir = pathlib.Path(__file__).parent / "models"
    for toml_file in sorted(models_dir.glob("*.toml")):
        print(f"\n── {toml_file.name} ──")
        mdl   = PMCModel(toml_file)
        X_ref, Y = simulate(mdl, N=1000, seed=0)

        X_hat, gamma, ll = classify(mdl, Y)

        er = error_rate(X_ref, X_hat)
        print(f"  log-lik     = {ll:.4f}")
        print(f"  error rate  = {er:.4f}  ({er*100:.1f} %)")
        print(f"  gamma[0]    = {gamma[0].round(4)}")
