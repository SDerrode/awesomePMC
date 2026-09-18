"""
simulate.py — Sequence generator for all 5 PMC/HMC variants.

Public API
----------
simulate(model, N=None, seed=None) -> (X, Y)
    Generate a synthetic sequence (X_{1:N}, Y_{1:N}) from a PMCModel.

Algorithm
---------
State margins (``model.margin_structure == "state"``, f_ij = f_i) — the
hidden chain X is Markov (Proposition of DerrodePieczynski_CSDA2013 §2.1),
so for every variant:
  1. Sample the latent chain  X_0, X_1, ..., X_N  (X_0 is a virtual "warm-up" state,
     not returned).  The chain is driven by the row-stochastic matrix A[i,j].
  2. Sample observations  Y_1, ..., Y_N  according to the variant-specific conditional:

     HMC-IN  : Y_n | X_n = j                    ~  f_j
     HMC-IN2 : Y_n | X_{n-1}=i, X_n=j           ~  f_j
     HMC-DN  : Y_1 | X_0=i, X_1=j               ~  f_j  (marginal, no copula)
               Y_n | X_{n-1}=i, X_n=j, Y_{n-1}  ~  f_j(·) · c_{ij}(F_i(Y_{n-1}), F_j(·))
     PMC-IN  : identical to HMC-IN2 (different prior, same conditional)
     PMC     : identical to HMC-DN  (different prior, same conditional)

Pair margins (``model.margin_structure == "pair"``, general PMC, PMC and
PMC-IN only) — X is not Markov, and the pairs z_n = (x_n, y_n) are drawn
alternately, as in DerrodePieczynski_CSDA2013 §3.1 (Eqs. 13–14):

     x_1 ~ π,  π_i = Σ_j p[i,j];   y_1 | x_1 = i  ~  Σ_j A[i,j] f_ij
     x_{n+1} | x_n = i, y_n               ∝  p[i, x_{n+1}] · f_{i,x_{n+1}}(y_n)      (Eq. 13)
     y_{n+1} | x_n = i, x_{n+1} = j, y_n  ~  f_ji(·) · c_ij(F_ij(y_n), F_ji(·))     (Eq. 14)

PMC-IN drops the copula factor (y_{n+1} ~ f_ji). The mixture for y_1 is
drawn through an auxiliary state j ~ A[x_1, ·], and y_{n+1} by conditional
inversion of c_ij at u = F_ij(y_n), then F_ji^{-1}.

In both cases the right margin of the pair (i, j) is ``model.margin(j, i)``
= f_ji (the index inversion of DerrodePieczynski_CSDA2013 Eq. 12), which is
f_j for state margins.

Returns
-------
X : np.ndarray, shape (N,), dtype int
    Hidden state sequence  X_{1:N}  with values in {0, ..., K-1}.
Y : np.ndarray, shape (N,), dtype float
    Observed sequence  Y_{1:N}.
"""

import logging
import math

import numpy as np

from pmcprg.pmc.model import PMCModel
from pmcprg.numerics import EPS, ONE_MINUS_EPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _sample_copula_conditional(
    model: PMCModel,
    i: int,
    j: int,
    y_prev: float,
    rng: np.random.Generator,
) -> float:
    """
    Draw  y_next  from  p(y_next | y_prev, X_n=i, X_{n+1}=j).

    Uses the copula h-function (Rosenblatt inversion), with the left
    margin F_ij for y_prev and the right margin F_ji for y_next
    (DerrodePieczynski_CSDA2013 Eq. 14; F_i and F_j for state margins):

        u  = F_ij(y_prev)
        w  ~ Uniform(0, 1)
        v  = h_{ij}^{-1}(w | u)         (closed form when available, else Brent)
        y_next = F_ji^{-1}(v)

    This is conditional-inverse sampling — Rosenblatt, M. (1952). Remarks
    on a multivariate transformation. *Ann. Math. Statist.* 23(3),
    470–472; Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
    Springer, §2.9. DerrodePieczynski_CSDA2013 Appendix B draws the same
    conditional by rejection sampling instead — same law, different
    mechanism.
    """
    cop = model.copula(i, j)
    u   = float(np.clip(model.margin(i, j).cdf(y_prev), EPS, ONE_MINUS_EPS))
    w   = float(rng.uniform(EPS, ONE_MINUS_EPS))
    v   = cop.inv_h(w, u)
    return float(model.margin(j, i).ppf(v))


def _draw_index(weights, rng: np.random.Generator) -> int:
    """Index drawn with probability ∝ ``weights`` (finite, ≥ 0, positive sum).

    Inverse-CDF draw from one uniform — cheaper than ``rng.choice`` inside the
    per-step loop of :func:`_simulate_pair`.
    """
    total = 0.0
    for w in weights:
        total += w
    u = float(rng.random()) * total
    acc = 0.0
    last = 0
    for k, w in enumerate(weights):
        if w > 0.0:
            last = k
            acc += w
            if u < acc:
                return k
    return last                        # u·total rounded up to total


def _sample_margin(margin, rng: np.random.Generator, d: int):
    """One draw from ``margin`` — a float for d = 1, a (d,) array otherwise."""
    sample = margin.rvs(1, rng)
    if d > 1:
        # multivariate_normal.rvs(size=1) returns (1, d); strip the singleton.
        return np.asarray(sample, dtype=float).reshape(d)
    return float(np.asarray(sample).reshape(-1)[0])


def _simulate_pair(
    model: PMCModel,
    N: int,
    rng: np.random.Generator,
    pi: np.ndarray,
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """General PMC with pair margins f_ij — DerrodePieczynski_CSDA2013 §3.1, Eqs. 13–14.

    ``pi`` and ``A`` are the renormalised π_i = Σ_j p[i,j] and
    A[i,j] = p[i,j] / π_i of :func:`simulate`.
    """
    K = model.K
    d = getattr(model, "d", 1)
    p = model.prior_p
    uses_copula = model.variant.uses_copula
    margins = {(i, j): model.margin(i, j) for i in range(K) for j in range(K)}

    X = np.empty(N, dtype=int)
    Y = np.empty((N, d) if d > 1 else (N,), dtype=float)

    # ── n = 1: x_1 ~ π, y_1 ~ Σ_j A[x_1, j] f_{x_1 j} ────────────────────
    x = int(rng.choice(K, p=pi))
    j_aux = int(rng.choice(K, p=A[x, :]))
    X[0] = x
    Y[0] = _sample_margin(margins[(x, j_aux)], rng, d)

    for n in range(N - 1):
        i, y = int(X[n]), Y[n]
        # ── Eq. 13: x_{n+1} | (x_n = i, y_n) ∝ p[i, j] · f_ij(y_n) ───────
        w = [float(p[i, jj]) * margins[(i, jj)].pdf(y) for jj in range(K)]
        s = sum(w)                         # NaN / inf if any weight is
        if math.isfinite(s) and s > 0.0:
            j = _draw_index(w, rng)
        else:
            # y_n has zero (underflowed) or undefined density under every
            # f_ij with p[i, j] > 0 — fall back on the prior row.
            logger.debug("simulate: degenerate Eq. 13 weights %s at n=%d; "
                         "drawing x_{n+1} from A[%d].", w, n, i)
            j = int(rng.choice(K, p=A[i, :]))
        X[n + 1] = j
        # ── Eq. 14: y_{n+1} | (i, j, y_n) ~ f_ji(·) [· c_ij(F_ij(y_n), F_ji(·))]
        if uses_copula:
            Y[n + 1] = _sample_copula_conditional(model, i, j, float(y), rng)
        else:
            Y[n + 1] = _sample_margin(margins[(j, i)], rng, d)
    return X, Y


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate(
    model: PMCModel,
    N: int | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic sequence from a PMCModel.

    Parameters
    ----------
    model : PMCModel
        The model to simulate from.
    N : int, optional
        Sequence length.  Defaults to ``model.N_default``.
    seed : int, optional
        Seed for the random number generator (reproducibility).

    Returns
    -------
    X : np.ndarray, shape (N,), dtype int
        Hidden state sequence  X_{1:N}  ∈ {0, …, K-1}.
    Y : np.ndarray
        Observed sequence  Y_{1:N}. Shape ``(N,)`` for scalar models
        (``model.d == 1``); ``(N, d)`` when observations are vectors.
    """
    if N is None:
        N = model.N_default
    rng = np.random.default_rng(seed)
    K   = model.K
    d   = getattr(model, "d", 1)
    pi  = model.stationary_pi
    A   = model.transition_A

    # Defensive renormalisation. ``np.random.Generator.choice`` rejects any
    # probability vector whose sum drifts more than ~``sqrt(eps) ≈ 1.5e-8``
    # from 1, but :class:`PMCModel` accepts inputs up to ``atol=1e-6``
    # — and per-row drift can fall in that gap, especially when the prior
    # tab round-trips a fitted prior through 6-decimal display strings.
    # We rescale once here so the model's invariants stay readable while
    # the chain sampler always sees clean probabilities.
    pi_sum = pi.sum()
    if pi_sum > 0:
        pi = pi / pi_sum
    row_sums = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums > 0)

    if model.margin_structure == "pair":
        # General PMC (DerrodePieczynski_CSDA2013 Eqs. 13–14): X is not
        # Markov, pairs are drawn in turn.
        X_out, Y = _simulate_pair(model, N, rng, pi, A)
        logger.debug(
            "simulate: variant=%s (pair margins)  K=%d  N=%d  seed=%s",
            model.variant.value, K, N, seed,
        )
        return X_out, Y

    # ── 1. Generate latent chain X_0, X_1, ..., X_N ──────────────────────────
    # X_0 is a virtual "warm-up" state; X_1:N are returned.
    states = np.empty(N + 1, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for n in range(N):
        states[n + 1] = rng.choice(K, p=A[states[n], :])

    X_out = states[1:]   # shape (N,)

    # ── 2. Generate observations Y_1, ..., Y_N ───────────────────────────────
    # Allocate (N,) for scalar models, (N, d) for vector models.
    Y = np.empty((N, d) if d > 1 else (N,), dtype=float)
    v  = model.variant
    uses_copula = v.uses_copula

    for n in range(N):
        i = int(states[n])       # X_{n}   (previous latent state)
        j = int(states[n + 1])   # X_{n+1} (current  latent state)

        if uses_copula and n > 0:
            # Copula path is forbidden for d>1 (validated at model load),
            # so this branch is only reachable for scalar margins.
            Y[n] = _sample_copula_conditional(model, i, j, Y[n - 1], rng)
        else:
            # Marginal sample: with state margins the right margin of the
            # pair (i, j), f_ji, is f_j — independent of the previous state.
            sample = model.margin(j).rvs(1, rng)
            if d > 1:
                # multivariate_normal.rvs(size=1) returns (1, d); strip
                # the leading singleton.
                Y[n] = np.asarray(sample, dtype=float).reshape(d)
            else:
                Y[n] = float(np.asarray(sample).reshape(-1)[0])

    logger.debug(
        "simulate: variant=%s  K=%d  N=%d  seed=%s",
        v.value, K, N, seed,
    )
    return X_out, Y


# ---------------------------------------------------------------------------
# Quick smoke-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

    from pmcprg.pmc.model import PMCModel

    models_dir = pathlib.Path(__file__).parent / "models"
    for toml_file in sorted(models_dir.glob("*.toml")):
        print(f"\n── {toml_file.name} ──")
        mdl = PMCModel(toml_file)
        X, Y = simulate(mdl, N=500, seed=42)
        print(f"  X shape = {X.shape},  unique states = {np.unique(X)}")
        print(f"  Y shape = {Y.shape}")
        print(f"  Y mean  = {Y.mean():.4f},  Y std = {Y.std():.4f}")
        # Empirical class frequencies
        for k in range(mdl.K):
            freq = (X == k).mean()
            print(f"    P(X={k}) empirical = {freq:.3f}  (theoretical π={mdl.stationary_pi[k]:.3f})")
