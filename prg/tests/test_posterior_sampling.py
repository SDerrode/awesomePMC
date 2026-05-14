"""Tests for :func:`prg.pmc.inference.sample_posterior` — Forward-Filter
Backward-Sample (FFBS) draw of X | Y, used by SEM (PR2)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from prg.pmc.inference import (
    backward,
    forward,
    precompute_weights,
    sample_posterior,
    smooth,
)
from prg.pmc.model    import PMCModel
from prg.pmc.simulate import simulate


MODELS    = pathlib.Path("prg/pmc/models")
HMC_IN_K2 = MODELS / "hmc_in_gauss_k2.toml"
HMC_IN_K3 = MODELS / "hmc_in_gauss_k3.toml"
PMC_K2    = MODELS / "pmc_gauss_k2.toml"


# ---------------------------------------------------------------------------
# Shape / dtype / basic sanity
# ---------------------------------------------------------------------------

def test_sample_posterior_returns_int_array_of_length_N():
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=500, seed=0)
    rng = np.random.default_rng(0)
    X = sample_posterior(mdl, Y, rng)
    assert X.shape == (500,)
    assert X.dtype.kind == "i"
    assert set(np.unique(X).tolist()).issubset({0, 1})


def test_sample_posterior_short_sequence_raises():
    """N < 2 has no transitions; sampling must raise rather than silently misbehave."""
    mdl = PMCModel(HMC_IN_K2)
    with pytest.raises(ValueError, match="requires N"):
        sample_posterior(mdl, np.array([0.5]), np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_sample_posterior_same_seed_same_draw():
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    X_a = sample_posterior(mdl, Y, np.random.default_rng(42))
    X_b = sample_posterior(mdl, Y, np.random.default_rng(42))
    np.testing.assert_array_equal(X_a, X_b)


def test_sample_posterior_different_seeds_differ():
    """Two independent FFBS draws on the same model + Y should not agree on
    every time-step (the posterior is non-degenerate everywhere)."""
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    X_a = sample_posterior(mdl, Y, np.random.default_rng(1))
    X_b = sample_posterior(mdl, Y, np.random.default_rng(2))
    assert not np.array_equal(X_a, X_b)


# ---------------------------------------------------------------------------
# Distributional check: empirical marginals from many draws ≈ smoothed γ
# ---------------------------------------------------------------------------

def test_empirical_marginal_matches_smoother():
    """Average many FFBS draws — per-time-step empirical state frequencies
    must converge to the smoothed posterior γ_n(j). This is the strongest
    correctness check for the sampler (Monte-Carlo estimate of γ)."""
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=200, seed=0)

    # Reference smoother
    W, f_pdf  = precompute_weights(mdl, Y)
    alpha_hat, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
    beta_hat  = backward(mdl, Y, W=W)
    gamma_ref = smooth(alpha_hat, beta_hat)            # (N, K)

    # Monte-Carlo via FFBS
    N, K = gamma_ref.shape
    counts = np.zeros((N, K), dtype=int)
    n_draws = 1500
    rng = np.random.default_rng(0)
    for _ in range(n_draws):
        X = sample_posterior(mdl, Y, rng, W=W, f_pdf=f_pdf, alpha_hat=alpha_hat)
        counts[np.arange(N), X] += 1
    gamma_emp = counts / n_draws

    # Per-time-step L1 distance must be small. With 1500 draws and K=2,
    # standard deviation on a single γ_n(j) estimate is √(p(1-p)/n_draws)
    # ≲ 0.013, so a 0.05 tolerance leaves a comfortable margin.
    max_diff = float(np.abs(gamma_emp - gamma_ref).max())
    assert max_diff < 0.06, f"max |γ_emp − γ_ref| = {max_diff:.4f} too large"


# ---------------------------------------------------------------------------
# Degenerate posterior: well-separated mixture → posterior concentrates on truth
# ---------------------------------------------------------------------------

def test_well_separated_posterior_recovers_truth():
    """With very well-separated states the posterior P(X|Y) is nearly
    deterministic. A single FFBS draw should then match the ground truth
    almost everywhere (after the Hungarian label alignment handled at
    evaluation time — here we evaluate only the agreement rate)."""
    mdl = PMCModel(HMC_IN_K2)
    # Pump up the separation by overriding the margin locs.
    raw = mdl.raw
    raw["margins"][0]["params"]["loc"] = -10.0
    raw["margins"][1]["params"]["loc"] = +10.0
    sharp_mdl = PMCModel.from_dict(raw)

    X_ref, Y = simulate(sharp_mdl, N=600, seed=0)
    X_draw   = sample_posterior(sharp_mdl, Y, np.random.default_rng(0))

    # Either X_draw == X_ref or X_draw is the flipped version (label
    # permutation invariance under the unsupervised setting).
    agree = max(float(np.mean(X_draw == X_ref)),
                float(np.mean(X_draw == (1 - X_ref))))
    assert agree > 0.99, f"agreement = {agree:.3f} below 0.99"


# ---------------------------------------------------------------------------
# Variants: PMC (with copulas) — smoke test
# ---------------------------------------------------------------------------

def test_sample_posterior_runs_on_pmc_variant():
    mdl = PMCModel(PMC_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    X = sample_posterior(mdl, Y, np.random.default_rng(0))
    assert X.shape == (400,)
    assert set(np.unique(X).tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Variants: K=3 — smoke test
# ---------------------------------------------------------------------------

def test_sample_posterior_runs_on_k3_variant():
    mdl = PMCModel(HMC_IN_K3)
    _, Y = simulate(mdl, N=400, seed=0)
    X = sample_posterior(mdl, Y, np.random.default_rng(0))
    assert X.shape == (400,)
    assert set(np.unique(X).tolist()).issubset({0, 1, 2})
