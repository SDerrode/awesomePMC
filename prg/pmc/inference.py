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

Algorithm
---------
Devijver normalization (Baum-Welch revisited):

  Initialization
    α_1(j) = Σ_i  p[i,j] · f_{ji}(y_1)
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

  Posterior marginals
    γ_n(j) ∝  α̂_n(j) · β̂_n(j)

  MPM
    X̂_n = argmax_j  γ_n(j)
"""

import logging

import numpy as np

from prg.exceptions    import IncompatibleObservationError
from prg.pmc.model     import PMCModel, Variant
from prg.numerics   import EPS, ONE_MINUS_EPS, MIN_POSITIVE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight pre-computation
# ---------------------------------------------------------------------------

def precompute_weights(
    model: PMCModel,
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-compute the weight tensor  W[n, i, j] = w(i, j, Y[n], Y[n+1])
    and the margin-PDF tensor  f_pdf[n, i, j] = f_{ij}(Y[n]).

    Parameters
    ----------
    model : PMCModel
    Y     : np.ndarray, shape (N,)

    Returns
    -------
    W     : np.ndarray, shape (N-1, K, K)
    f_pdf : np.ndarray, shape (N,   K, K)
    """
    N   = len(Y)
    K   = model.K
    var = model.variant

    if N < 2:
        raise ValueError(
            f"precompute_weights requires N ≥ 2 observations, got N={N}. "
            f"With N < 2 there are no transitions and forward-backward is "
            f"undefined; use ``model.margin(...).pdf(Y[0])`` directly."
        )

    # ── Margin PDFs (and CDFs for copula variants) ────────────────────────
    # Under SR-PMC there are K state-indexed densities, so we evaluate
    # K vectors and broadcast them into the (N, K, K) shape that the rest
    # of the pipeline expects (``f_pdf[n, i, j] = f_i(Y[n])`` for all j).
    f_pdf_state = np.zeros((N, K))
    f_cdf_state = np.zeros((N, K)) if var.uses_copula else None
    for kk in range(K):
        f_pdf_state[:, kk] = model.margin(kk).pdf_vec(Y)
        if f_cdf_state is not None:
            f_cdf_state[:, kk] = model.margin(kk).cdf_vec(Y)
    if f_cdf_state is not None:
        np.clip(f_cdf_state, EPS, ONE_MINUS_EPS, out=f_cdf_state)

    # Broadcast to the historic (N, K, K) shape: f_pdf[n, i, j] = f_i(Y[n]).
    # ``broadcast_to`` returns a read-only view, so we copy to a writable
    # array (the caller stores it and may later mutate it).
    f_pdf = np.broadcast_to(f_pdf_state[:, :, None], (N, K, K)).copy()
    f_cdf = (np.broadcast_to(f_cdf_state[:, :, None], (N, K, K)).copy()
             if f_cdf_state is not None else None)

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

    # Guard against numerical zeros / negatives
    np.clip(W, 0.0, None, out=W)

    logger.debug(
        "precompute_weights: variant=%s  N=%d  K=%d  W.min=%.3e  W.max=%.3e",
        var.value, N, K, W.min(), W.max(),
    )
    return W, f_pdf


# ---------------------------------------------------------------------------
# Forward pass  (Devijver normalization)
# ---------------------------------------------------------------------------

def forward(
    model: PMCModel,
    Y: np.ndarray,
    W: np.ndarray | None = None,
    f_pdf: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Normalized forward pass (Devijver / Baum-Welch).

    Parameters
    ----------
    model : PMCModel
    Y     : np.ndarray, shape (N,)
    W     : pre-computed weight tensor (N-1, K, K).  Computed if None.
    f_pdf : pre-computed margin-PDF tensor (N, K, K).  Computed with W if None.

    Returns
    -------
    alpha_hat : np.ndarray, shape (N, K)   — normalized forward variables
    log_lik   : float                       — log-likelihood  log p(y_{1:N})
    """
    N = len(Y)
    K = model.K

    if W is None:
        W, f_pdf = precompute_weights(model, Y)

    alpha_hat = np.zeros((N, K))
    log_lik   = 0.0
    p         = model.prior_p   # (K, K)

    # ── Initialization: α_1(j) = Σ_i p[i,j] · f_{ji}(Y[0]) ─────────────
    # In numpy: α_1 = einsum('ij,ji->j', p, f_pdf[0])
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
        if not np.isfinite(C) or C <= 0.0:
            # Underflow rather than modelling error — some isolated step has
            # vanishing weight. Log a warning and floor C to keep the chain
            # alive; the resulting log-lik is a lower bound but does not lie
            # about its sign.
            logger.warning(
                "Forward: C=%.3e at step n=%d (likely numerical underflow) — "
                "flooring to MIN_POSITIVE.",
                C, n + 1,
            )
            C = MIN_POSITIVE
        log_lik         += np.log(C)
        alpha_hat[n + 1] = alpha_raw / C

    return alpha_hat, float(log_lik)


# ---------------------------------------------------------------------------
# Backward pass  (Devijver normalization)
# ---------------------------------------------------------------------------

def backward(
    model: PMCModel,
    Y: np.ndarray,
    W: np.ndarray | None = None,
) -> np.ndarray:
    """
    Normalized backward pass.

    Parameters
    ----------
    model : PMCModel
    Y     : np.ndarray, shape (N,)
    W     : pre-computed weight tensor (N-1, K, K).  Computed if None.

    Returns
    -------
    beta_hat : np.ndarray, shape (N, K)
    """
    N = len(Y)
    K = model.K

    if W is None:
        W, _ = precompute_weights(model, Y)

    beta_hat = np.zeros((N, K))

    # Initialization: uniform at step N
    beta_hat[N - 1] = 1.0 / K

    # Recursion (backward)
    for n in range(N - 2, -1, -1):
        # β_n(i) = Σ_j W[n, i, j] · β̂_{n+1}(j)
        beta_raw = W[n] @ beta_hat[n + 1]   # shape (K,)
        D = float(beta_raw.sum())
        if not np.isfinite(D) or D <= 0.0:
            D = MIN_POSITIVE
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
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    return gamma / row_sums


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
) -> np.ndarray:
    """Draw one realisation of the state sequence  X̃ ~ P(X | Y).

    Implements **Forward-Filter Backward-Sample (FFBS)** — the standard
    posterior sampler for state-space models.

    1. *Filter*: compute (or accept pre-computed) ``α̂_n(j) = P(X_n=j | Y_{1:n})``.
    2. *Sample* ``X̃_{N-1} ~ α̂_{N-1}`` (the last filtered posterior is the
       smoothed posterior because there is no future observation).
    3. *Backward-sample*, for ``n = N-2, …, 0``::

           P(X_n=i | X̃_{n+1}, Y_{1:N}) ∝ α̂_n(i) · W[n, i, X̃_{n+1}]

       Conditional independence of ``X_n`` and ``Y_{n+2:N}`` given
       ``X_{n+1}`` is what makes this sampler exact.

    Used by SEM (Stochastic EM) to obtain the hard pseudo-labels on
    which the M-step is then run.

    Parameters
    ----------
    model     : PMCModel
    Y         : np.ndarray, shape (N,) or (N, d).
    rng       : :class:`numpy.random.Generator` — source of randomness.
    W, f_pdf, alpha_hat : optional pre-computed tensors (skip re-computation).

    Returns
    -------
    X_sample  : np.ndarray, shape (N,), dtype int — one draw from P(X | Y).
    """
    N = len(Y)
    K = model.K
    if N < 2:
        raise ValueError(
            f"sample_posterior requires N ≥ 2 observations, got N={N}."
        )

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

    return X


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
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Supervised MPM classification of the observation sequence Y.

    Runs the full forward-backward-smooth-MPM pipeline.

    Parameters
    ----------
    model : PMCModel — fully specified model (parameters known).
    Y     : np.ndarray, shape (N,) — observed sequence.

    Returns
    -------
    X_hat     : np.ndarray, shape (N,) int — MPM class labels.
    gamma     : np.ndarray, shape (N, K)   — posterior marginals.
    log_lik   : float                       — log p(y_{1:N} | model).
    """
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
    from prg.pmc.peano import image_to_signal, signal_to_image

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

    # Diagnostics on label-set mismatches.
    used_true = set(np.unique(X_true).tolist())
    used_hat  = set(np.unique(X_hat).tolist())
    if used_hat < used_true:
        # X_hat collapsed onto a subset of the true classes — typical
        # symptom of a degenerate ICE solution. The Hungarian still picks
        # an optimal partial relabeling.
        missing = sorted(used_true - used_hat)
        logger.warning(
            "error_rate: X_hat uses %d/%d classes (missing %s). "
            "The fitted model likely collapsed to a degenerate solution.",
            len(used_hat), len(used_true), missing,
        )
    elif not used_hat.issubset(used_true):
        # X_hat uses labels outside the true set — happens when the
        # supplied K is larger than the number of true classes (e.g.
        # ICE was run with a mis-specified K). Hungarian leaves the
        # extra X_hat columns unmatched.
        extra = sorted(used_hat - used_true)
        logger.warning(
            "error_rate: X_hat uses labels %s outside the true set "
            "{0,…,%d}. The reported error counts these as unmatched; "
            "consider whether K was specified correctly.",
            extra, max(used_true) if used_true else 0,
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

    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

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
