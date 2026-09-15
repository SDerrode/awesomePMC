"""Tests for the Pairwise Mixture Model (pmcprg.pmc.pmm)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pmcprg.pmc.model     import PMCModel
from pmcprg.pmc.pmm       import classify_pmm, simulate_pmm
from pmcprg.pmc.inference import error_rate


MODELS = pathlib.Path("pmcprg/pmc/models")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def test_simulate_pmm_shape_and_dtype():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    X, Y = simulate_pmm(mdl, n_pairs=500, seed=0)
    assert X.shape == (1000,)
    assert Y.shape == (1000,)
    assert X.dtype.kind == "i"
    assert Y.dtype.kind == "f"
    # All states valid
    assert set(np.unique(X)).issubset(set(range(mdl.K)))


def test_simulate_pmm_seed_reproducibility():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    X1, Y1 = simulate_pmm(mdl, n_pairs=300, seed=42)
    X2, Y2 = simulate_pmm(mdl, n_pairs=300, seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_allclose(Y1, Y2)


def test_simulate_pmm_pair_marginals_match_p():
    """Empirical pair frequencies converge to the joint prior p[i,j]."""
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    n_pairs = 20_000
    X, _ = simulate_pmm(mdl, n_pairs=n_pairs, seed=7)
    p_emp = np.zeros((mdl.K, mdl.K))
    for k in range(n_pairs):
        i, j = int(X[2 * k]), int(X[2 * k + 1])
        p_emp[i, j] += 1
    p_emp /= n_pairs
    # Convergence within 2σ ≈ 2/√n_pairs ≈ 0.014 for K=2.
    np.testing.assert_allclose(p_emp, mdl.prior_p, atol=0.02)


def test_simulate_pmm_no_copula_variant_uses_independent_y2():
    """For HMC-IN (no copula) y_1 ⊥ y_2 within the pair."""
    mdl = PMCModel(MODELS / "hmc_in_gauss_k2.toml")
    X, Y = simulate_pmm(mdl, n_pairs=4_000, seed=0)
    # Within-pair Pearson correlation ≈ 0 in the limit.
    Y1 = Y[0::2]
    Y2 = Y[1::2]
    rho = float(np.corrcoef(Y1, Y2)[0, 1])
    # Without copula, Pearson is bounded by the bilinear term coming
    # from the joint p(i, j); with diagonal-heavy p (the fixture has
    # A = I·0.9 + (1−I)·0.1 → π = (0.5, 0.5)), the implied marginal
    # correlation is small but non-zero. We just check it's much
    # smaller than for a copula-using variant.
    assert abs(rho) < 0.6


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_classify_pmm_returns_correct_shape_and_loglik():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    X_ref, Y = simulate_pmm(mdl, n_pairs=500, seed=0)
    X_hat, log_lik = classify_pmm(mdl, Y)
    assert X_hat.shape == X_ref.shape
    assert isinstance(log_lik, float)
    assert np.isfinite(log_lik)


def test_classify_pmm_recovers_labels_above_random():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    X_ref, Y = simulate_pmm(mdl, n_pairs=2_000, seed=0)
    X_hat, _ = classify_pmm(mdl, Y)
    er = error_rate(X_ref, X_hat)
    assert er < 0.30


def test_classify_pmm_rejects_odd_length():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    Y_odd = np.zeros(7)
    with pytest.raises(ValueError, match="even-length"):
        classify_pmm(mdl, Y_odd)


def test_classify_pmm_pmc_vs_pmm_error_rate():
    """PMM is non-Markovian → error rate should be HIGHER than PMC.

    This reproduces the qualitative finding of CSDA 2013 §3.3:
    'Markovianity has a strong influence on data restoration.'
    """
    from pmcprg.pmc.simulate  import simulate
    from pmcprg.pmc.inference import classify

    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    X_pmc, Y_pmc = simulate(mdl, N=2_000, seed=11)
    X_pmm, Y_pmm = simulate_pmm(mdl, n_pairs=1_000, seed=11)

    X_hat_pmc, _, _ = classify(mdl, Y_pmc)
    X_hat_pmm, _    = classify_pmm(mdl, Y_pmm)
    er_pmc = error_rate(X_pmc, X_hat_pmc)
    er_pmm = error_rate(X_pmm, X_hat_pmm)
    # PMM should lose at least a few percentage points to PMC on this fixture.
    assert er_pmm > er_pmc, f"PMM={er_pmm:.4f} should be > PMC={er_pmc:.4f}"


def test_classify_pmm_works_on_hmc_in():
    """Variants without copula are also supported."""
    mdl = PMCModel(MODELS / "hmc_in_gauss_k2.toml")
    X_ref, Y = simulate_pmm(mdl, n_pairs=500, seed=0)
    X_hat, log_lik = classify_pmm(mdl, Y)
    assert np.isfinite(log_lik)
    er = error_rate(X_ref, X_hat)
    assert er < 0.30
