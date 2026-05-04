"""
test_forward_backward_invariants.py — probability-conservation tests for the
forward / backward / smoothed posteriors of the PMC inference module.

The Devijver-normalised forward-backward algorithm produces three quantities
that are probability distributions — and therefore must sum to 1 (within
floating-point tolerance) over the appropriate axis:

* ``α̂_n(j)``   — filtered posterior  P(X_n = j | Y_{1:n}),  Σ_j α̂_n(j) = 1
* ``β̂_n(j)``   — Devijver-normalised backward variable, Σ_j β̂_n(j) = 1
* ``γ_n(j)``    — smoothed posterior  P(X_n = j | Y_{1:N}),  Σ_j γ_n(j) = 1
* ``ξ_n(i, j)`` — joint pair posterior P(X_n=i, X_{n+1}=j | Y_{1:N}),
                  Σ_{i,j} ξ_n(i, j) = 1

Marginal consistency:

* Σ_j ξ_n(i, j) = γ_n(i)         (marginalising out the next state)
* Σ_i ξ_n(i, j) = γ_{n+1}(j)     (marginalising out the current state)

These invariants caught the PMC kernel bug fixed in this release; they
serve as a regression net.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from prg.pmc          import PMCModel, simulate
from prg.pmc.inference import (
    backward,
    forward,
    precompute_weights,
    smooth,
)
from prg.pmc.ice      import _joint_posteriors


MODELS_DIR = Path(__file__).resolve().parents[1] / "pmc" / "models"
ALL_MODELS = sorted(MODELS_DIR.glob("*.toml"))


# Tolerances. Devijver normalisation is exact in exact arithmetic; in
# float64 we expect round-off of order ≈ N · eps ≈ 1e-13 for N=500.
_TOL_NORM    = 1e-10   # |Σ p − 1| ≤ TOL
_TOL_MARGIN  = 1e-10   # marginal consistency between ξ and γ


def _all_quantities(mdl: PMCModel, N: int = 500, seed: int = 7):
    """Run the full forward-backward pipeline and return α̂, β̂, γ, ξ."""
    _, Y = simulate(mdl, N=N, seed=seed)
    W, f_pdf = precompute_weights(mdl, Y)
    alpha, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
    beta     = backward(mdl, Y, W=W)
    gamma    = smooth(alpha, beta)
    xi       = _joint_posteriors(alpha, W, beta)
    return alpha, beta, gamma, xi


# ===========================================================================
# Per-step normalisation
# ===========================================================================

@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_alpha_sums_to_one(toml_path: Path):
    """Σ_j α̂_n(j) = 1 for every n ∈ {1, …, N}."""
    alpha, _, _, _ = _all_quantities(PMCModel(toml_path))

    sums = alpha.sum(axis=1)                          # shape (N,)
    assert sums.shape == (alpha.shape[0],)
    np.testing.assert_allclose(
        sums, 1.0, atol=_TOL_NORM,
        err_msg=(
            f"{toml_path.name}: α̂_n is the filtered posterior "
            f"P(X_n=j|Y_{{1:n}}) — must sum to 1 over j; "
            f"max deviation = {np.max(np.abs(sums - 1)):.3e}"
        ),
    )


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_beta_sums_to_one(toml_path: Path):
    """Σ_j β̂_n(j) = 1 for every n (Devijver normalisation)."""
    _, beta, _, _ = _all_quantities(PMCModel(toml_path))

    sums = beta.sum(axis=1)
    np.testing.assert_allclose(
        sums, 1.0, atol=_TOL_NORM,
        err_msg=(
            f"{toml_path.name}: β̂_n is Devijver-normalised — "
            f"must sum to 1; max deviation = {np.max(np.abs(sums - 1)):.3e}"
        ),
    )


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_gamma_sums_to_one(toml_path: Path):
    """Σ_j γ_n(j) = 1 — γ_n is the smoothed posterior P(X_n=j|Y_{1:N})."""
    _, _, gamma, _ = _all_quantities(PMCModel(toml_path))

    sums = gamma.sum(axis=1)
    assert (gamma >= 0).all() and (gamma <= 1).all(), (
        f"{toml_path.name}: γ_n must be a valid probability vector"
    )
    np.testing.assert_allclose(
        sums, 1.0, atol=_TOL_NORM,
        err_msg=(
            f"{toml_path.name}: γ_n must sum to 1; "
            f"max deviation = {np.max(np.abs(sums - 1)):.3e}"
        ),
    )


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_xi_sums_to_one(toml_path: Path):
    """Σ_{i,j} ξ_n(i, j) = 1 — ξ is the joint pair posterior."""
    _, _, _, xi = _all_quantities(PMCModel(toml_path))

    assert (xi >= 0).all() and (xi <= 1).all(), (
        f"{toml_path.name}: ξ_n(i,j) must be a valid probability"
    )
    sums = xi.sum(axis=(1, 2))                          # shape (N-1,)
    np.testing.assert_allclose(
        sums, 1.0, atol=_TOL_NORM,
        err_msg=(
            f"{toml_path.name}: ξ_n is the joint pair posterior "
            f"P(X_n=i, X_{{n+1}}=j | Y_{{1:N}}) — must sum to 1 over (i, j); "
            f"max deviation = {np.max(np.abs(sums - 1)):.3e}"
        ),
    )


# ===========================================================================
# Marginal consistency between ξ and γ
# ===========================================================================

@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_xi_marginalises_to_gamma_left(toml_path: Path):
    """Σ_j ξ_n(i, j) = γ_n(i)  for n = 0, …, N-2.

    The "current state" marginal of the joint pair posterior must equal
    the smoothed posterior at the same time step.
    """
    _, _, gamma, xi = _all_quantities(PMCModel(toml_path))

    xi_marg_left = xi.sum(axis=2)                       # (N-1, K) over j
    gamma_left   = gamma[:-1]                           # (N-1, K)

    np.testing.assert_allclose(
        xi_marg_left, gamma_left, atol=_TOL_MARGIN,
        err_msg=(
            f"{toml_path.name}: Σ_j ξ_n(i, j) should equal γ_n(i); "
            f"max deviation = {np.max(np.abs(xi_marg_left - gamma_left)):.3e}"
        ),
    )


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_xi_marginalises_to_gamma_right(toml_path: Path):
    """Σ_i ξ_n(i, j) = γ_{n+1}(j)  for n = 0, …, N-2.

    The "next state" marginal of the joint pair posterior must equal
    the smoothed posterior at the *next* time step.
    """
    _, _, gamma, xi = _all_quantities(PMCModel(toml_path))

    xi_marg_right = xi.sum(axis=1)                      # (N-1, K) over i
    gamma_right   = gamma[1:]                           # (N-1, K)

    np.testing.assert_allclose(
        xi_marg_right, gamma_right, atol=_TOL_MARGIN,
        err_msg=(
            f"{toml_path.name}: Σ_i ξ_n(i, j) should equal γ_{{n+1}}(j); "
            f"max deviation = {np.max(np.abs(xi_marg_right - gamma_right)):.3e}"
        ),
    )
