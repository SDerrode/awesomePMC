"""
simulate.py — Sequence generator for all 5 PMC/HMC variants.

Public API
----------
simulate(model, N=None, seed=None) -> (X, Y)
    Generate a synthetic sequence (X_{1:N}, Y_{1:N}) from a PMCModel.

Algorithm
---------
For every variant:
  1. Sample the latent chain  X_0, X_1, ..., X_N  (X_0 is a virtual "warm-up" state,
     not returned).  The chain is driven by the row-stochastic matrix A[i,j].
  2. Sample observations  Y_1, ..., Y_N  according to the variant-specific conditional.
     Under the SR-PMC contract there are exactly K marginal densities
     (one per state), so f_{ij} = f_i regardless of the other index:

     HMC-IN  : Y_n | X_n = j                    ~  f_j
     HMC-IN2 : Y_n | X_{n-1}=i, X_n=j           ~  f_j
     HMC-DN  : Y_1 | X_0=i, X_1=j               ~  f_j  (marginal, no copula)
               Y_n | X_{n-1}=i, X_n=j, Y_{n-1}  ~  f_j(·) · c_{ij}(F_i(Y_{n-1}), F_j(·))
     PMC-IN  : identical to HMC-IN2 (different prior, same conditional)
     PMC     : identical to HMC-DN  (different prior, same conditional)

Historic notation in the package used "f_{ji}" for the right margin of pair
(i, j); under the SR-PMC factorisation this equals f_j.

Returns
-------
X : np.ndarray, shape (N,), dtype int
    Hidden state sequence  X_{1:N}  with values in {0, ..., K-1}.
Y : np.ndarray, shape (N,), dtype float
    Observed sequence  Y_{1:N}.
"""

import logging

import numpy as np

from prg.pmc.model import PMCModel
from prg.numerics import EPS, ONE_MINUS_EPS

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

    Uses the copula h-function (Rosenblatt inversion). Under the SR-PMC
    factorisation, both margins are state-indexed (F_i for y_prev,
    F_j for y_next):

        u  = F_i(y_prev)
        w  ~ Uniform(0, 1)
        v  = h_{ij}^{-1}(w | u)         (closed form when available, else Brent)
        y_next = F_j^{-1}(v)
    """
    cop = model.copula(i, j)
    u   = float(np.clip(model.margin(i).cdf(y_prev), EPS, ONE_MINUS_EPS))
    w   = float(rng.uniform(EPS, ONE_MINUS_EPS))
    v   = cop.inv_h(w, u)
    return float(model.margin(j).ppf(v))


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
    Y : np.ndarray, shape (N,), dtype float
        Observed sequence  Y_{1:N}.
    """
    if N is None:
        N = model.N_default
    rng = np.random.default_rng(seed)
    K   = model.K
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

    # ── 1. Generate latent chain X_0, X_1, ..., X_N ──────────────────────────
    # X_0 is a virtual "warm-up" state; X_1:N are returned.
    states = np.empty(N + 1, dtype=int)
    states[0] = rng.choice(K, p=pi)
    for n in range(N):
        states[n + 1] = rng.choice(K, p=A[states[n], :])

    X_out = states[1:]   # shape (N,)

    # ── 2. Generate observations Y_1, ..., Y_N ───────────────────────────────
    Y = np.empty(N, dtype=float)
    v  = model.variant
    uses_copula = v.uses_copula

    for n in range(N):
        i = int(states[n])       # X_{n}   (previous latent state)
        j = int(states[n + 1])   # X_{n+1} (current  latent state)

        if uses_copula and n > 0:
            # Copula-based conditional on the previous observation
            Y[n] = _sample_copula_conditional(model, i, j, Y[n - 1], rng)
        else:
            # Marginal sample: under SR-PMC the right-margin of pair
            # (i, j) is f_j — independent of the previous state.
            Y[n] = float(model.margin(j).rvs(1, rng)[0])

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

    from prg.pmc.model import PMCModel

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
