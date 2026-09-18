"""
test_forward_correctness.py — verifies that the forward log-likelihood
returned by ``pmcprg.pmc.inference.forward`` matches the true value computed
by exhaustive enumeration over all hidden-state sequences.

This is a *correctness net* for the Devijver normalization. Earlier versions
of the package had a silent bug on the PMC / PMC-IN variants where the
transition kernel double-counted ``f_{ij}(y_n)``; the discrepancy was
only visible against a brute-force reference.

Each parametrised case enumerates K^(N+1) hidden state sequences (K=2,
N=4-5 → 64-128 sequences), computes the joint p(X_{0:N}, Y_{1:N})
under the SR-PMC factorisation, then ``logsumexp``-marginalises over X.
The resulting reference must match ``forward()`` to floating-point
precision.

These tests are inexpensive (N is tiny) and run in < 1 s for the full
parametrised set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmcprg.numerics    import EPS, ONE_MINUS_EPS
from pmcprg.pmc         import PMCModel, simulate
from pmcprg.pmc.inference import forward


MODELS_DIR = Path(__file__).resolve().parents[1] / "pmc" / "models"
ALL_MODELS = sorted(MODELS_DIR.glob("*.toml"))


# ---------------------------------------------------------------------------
# Brute-force reference
# ---------------------------------------------------------------------------

def _brute_force_log_lik(mdl: PMCModel, Y: np.ndarray) -> float:
    """Compute log p(Y_{1:N}) by exhaustive state enumeration.

    State margins (X Markov) — the factorisation:
        p(X_{0:N}, Y_{1:N}) = π_{X_0} · ∏_n A[X_{n-1}, X_n]
                            · ∏_n f_{X_n, X_{n-1}}(Y_n)
                            · ∏_{n≥1, copula} c_{X_{n-1}, X_n}(F_{X_{n-1}, X_n}(Y_{n-1}),
                                                                F_{X_n, X_{n-1}}(Y_n))
    Pair margins (general PMC, X not Markov) — :func:`_brute_force_log_lik_pair`.
    """
    if mdl.margin_structure == "pair":
        return _brute_force_log_lik_pair(mdl, Y)
    K     = mdl.K
    pi    = mdl.stationary_pi
    A     = mdl.transition_A
    N     = len(Y)
    log_total = -np.inf

    for s in range(K ** (N + 1)):
        # decode integer s into a base-K state sequence of length N+1
        x = []
        z = s
        for _ in range(N + 1):
            x.append(z % K)
            z //= K
        x = np.asarray(x, dtype=int)

        # state sequence log-prob
        lp = np.log(max(pi[x[0]], 1e-300))
        for n in range(N):
            lp += np.log(max(A[x[n], x[n + 1]], 1e-300))

        # observation log-prob (with optional copula link to Y_{n-1})
        for n in range(N):
            i, j = int(x[n]), int(x[n + 1])
            lp += mdl.margin(j, i).logpdf(Y[n])
            if mdl.variant.uses_copula and n > 0:
                u = float(np.clip(mdl.cdf(i, j, Y[n - 1]), EPS, ONE_MINUS_EPS))
                v = float(np.clip(mdl.cdf(j, i, Y[n]),     EPS, ONE_MINUS_EPS))
                lp += np.log(max(mdl.copula(i, j).pdf([u, v]), EPS))

        log_total = np.logaddexp(log_total, lp)

    return float(log_total)


def _brute_force_log_lik_pair(mdl: PMCModel, Y: np.ndarray) -> float:
    """log p(Y_{1:N}) for pair margins f_ij, from DerrodePieczynski_CSDA2013 Eqs. 12–14:

        p(x_1, y_1)                   = Σ_j p[x_1, j] · f_{x_1 j}(y_1)
        p(x_{n+1} | x_n = i, y_n)     = p[i, x_{n+1}] f_{i x_{n+1}}(y_n) / Σ_k p[i, k] f_ik(y_n)
        p(y_{n+1} | i, j, y_n)        = f_ji(y_{n+1}) [· c_ij(F_ij(y_n), F_ji(y_{n+1}))]
    """
    K, N = mdl.K, len(Y)
    p = mdl.prior_p
    log_total = -np.inf
    for s in range(K ** N):
        x = [(s // K ** n) % K for n in range(N)]
        lp = np.log(sum(p[x[0], j] * mdl.pdf(x[0], j, Y[0]) for j in range(K)))
        for n in range(N - 1):
            i, j = x[n], x[n + 1]
            lp += np.log(p[i, j] * mdl.pdf(i, j, Y[n])
                         / sum(p[i, k] * mdl.pdf(i, k, Y[n]) for k in range(K)))
            lp += mdl.margin(j, i).logpdf(Y[n + 1])
            if mdl.variant.uses_copula:
                u = float(np.clip(mdl.cdf(i, j, Y[n]),     EPS, ONE_MINUS_EPS))
                v = float(np.clip(mdl.cdf(j, i, Y[n + 1]), EPS, ONE_MINUS_EPS))
                lp += np.log(mdl.copula(i, j).pdf([u, v]))
        log_total = np.logaddexp(log_total, lp)
    return float(log_total)


# ---------------------------------------------------------------------------
# Parametrised regression test
# ---------------------------------------------------------------------------

# Tight tolerance: the only sources of error are floating-point round-off
# in the forward sum and in the brute-force logsumexp. 1e-10 is comfortable
# for K=2, N≤6.
_FWD_BRUTE_TOL = 1e-10


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
@pytest.mark.parametrize("seed", [0, 1, 2], ids=lambda s: f"seed{s}")
def test_forward_matches_brute_force(toml_path: Path, seed: int):
    """``forward()`` must match exhaustive enumeration on a short sequence."""
    mdl = PMCModel(toml_path)
    N   = 5   # K^(N+1) = K^6 = 64 (K=2) — fast
    _, Y = simulate(mdl, N=N, seed=seed)

    _, ll_fwd = forward(mdl, Y)
    ll_ref    = _brute_force_log_lik(mdl, Y)

    assert abs(ll_fwd - ll_ref) < _FWD_BRUTE_TOL, (
        f"{toml_path.name}: forward log-lik {ll_fwd:.6f} disagrees with "
        f"brute force {ll_ref:.6f} (Δ = {ll_fwd - ll_ref:.2e}). "
        f"Likely cause: incorrect Devijver kernel. See "
        f"pmcprg/pmc/inference.py:precompute_weights."
    )


def test_pmc_in_reduces_to_hmc_in2():
    """When ``p[i,j] = π_i · A[i,j]`` and the margins are identical, the
    PMC-IN model is mathematically the same as the HMC-IN2 model. Their
    forward log-likelihoods on a common Y must therefore be exactly equal.

    We build the two equivalent models in-memory (not from the demo TOML
    files, whose margins were chosen differently for variety).
    """
    A    = [[0.9, 0.1], [0.1, 0.9]]
    pi   = [0.5, 0.5]                                    # stationary of A
    p_ij = [[pi[i] * A[i][j] for j in range(2)] for i in range(2)]

    margins = [
        {"i": 0, "j": 0, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
        {"i": 0, "j": 1, "dist": "norm", "params": {"loc": -1.0, "scale": 1.0}},
        {"i": 1, "j": 0, "dist": "norm", "params": {"loc":  1.0, "scale": 1.0}},
        {"i": 1, "j": 1, "dist": "norm", "params": {"loc":  1.0, "scale": 1.0}},
    ]

    # K² blocks are pair margins by default, which an HMC-* variant refuses
    # (DerrodePieczynski_CSDA2013 §2.1 Proposition): collapse them explicitly.
    # The PMC-IN model keeps them as (tied) pair margins, so the check also
    # crosses the two paths.
    mdl_hmc = PMCModel.from_dict({
        "model":   {"name": "hmc-in2-eq", "variant": "HMC-IN2", "K": 2,
                    "N_default": 200, "margin_structure": "state"},
        "prior":   {"A": A},
        "margins": margins,
    })
    mdl_pmc = PMCModel.from_dict({
        "model":   {"name": "pmc-in-eq", "variant": "PMC-IN", "K": 2,
                    "N_default": 200},
        "prior":   {"p": p_ij},
        "margins": margins,
    })

    _, Y = simulate(mdl_hmc, N=200, seed=0)
    _, ll_pmc = forward(mdl_pmc, Y)
    _, ll_hmc = forward(mdl_hmc, Y)
    assert abs(ll_pmc - ll_hmc) < 1e-10, (
        f"PMC-IN and HMC-IN2 should give identical log-lik when "
        f"p[i,j] = π_i · A[i,j] and margins coincide, got "
        f"{ll_pmc:.6f} vs {ll_hmc:.6f}"
    )
