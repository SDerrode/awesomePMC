"""Tests for the K-means warm-start option of ICE.

Covers the new ``init`` key parsed by :func:`prg.pmc.ice._parse_ice_cfg`:

* ``init = "model"`` (default) — behaviour preserved.
* ``init = "kmeans"``           — cluster Y with k-means++ then derive a
                                  warm-start :class:`PMCModel` via a
                                  single supervised-style M-step.

The K-means option requires ``scikit-learn`` (declared in the ``[ml]``
optional-dependency group); the whole module is skipped if it is not
available.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip(
    "sklearn",
    reason="ICE init='kmeans' requires scikit-learn (pip install copulasformm[ml]).",
)

from prg.pmc.ice import (
    INIT_STRATEGIES,
    _check_init_strategy,
    _kmeans_label_assignment,
    _parse_ice_cfg,
    _warmstart_from_kmeans,
    ice,
)
from prg.pmc.inference import classify, error_rate
from prg.pmc.model     import PMCModel
from prg.pmc.simulate  import simulate


MODELS = pathlib.Path("prg/pmc/models")
HMC_IN_K2 = MODELS / "hmc_in_gauss_k2.toml"
PMC_K2    = MODELS / "pmc_gauss_k2.toml"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_init_strategies_exposed():
    """Public constant is the source of truth for legal values."""
    assert tuple(sorted(INIT_STRATEGIES)) == ("kmeans", "model")


def test_default_init_is_model():
    """Default cfg preserves backward compatibility (init='model')."""
    mdl = PMCModel(HMC_IN_K2)
    cfg = _parse_ice_cfg(mdl, None)
    assert cfg["init"] == "model"
    assert cfg["kmeans_seed"] == 0


def test_check_init_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown init strategy"):
        _check_init_strategy("kmeansplusplus")


def test_ice_cfg_override_threads_through():
    mdl = PMCModel(HMC_IN_K2)
    cfg = _parse_ice_cfg(mdl, {"init": "kmeans", "kmeans_seed": 7})
    assert cfg["init"] == "kmeans"
    assert cfg["kmeans_seed"] == 7


# ---------------------------------------------------------------------------
# _kmeans_label_assignment — basic sanity
# ---------------------------------------------------------------------------

def test_kmeans_labels_range_and_shape():
    rng = np.random.default_rng(0)
    Y = np.concatenate([
        rng.normal(loc=-3, scale=0.5, size=300),
        rng.normal(loc=+3, scale=0.5, size=300),
    ])
    labels = _kmeans_label_assignment(Y, K=2, random_state=0)
    assert labels.shape == (600,)
    assert labels.dtype.kind == "i"
    assert set(np.unique(labels).tolist()) == {0, 1}


def test_kmeans_labels_reproducible():
    rng = np.random.default_rng(123)
    Y = rng.normal(size=400)
    a = _kmeans_label_assignment(Y, K=3, random_state=42)
    b = _kmeans_label_assignment(Y, K=3, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_kmeans_labels_separated_clusters_match_ground_truth():
    """Two very well-separated clusters → K-means recovers them up to a label flip."""
    rng = np.random.default_rng(0)
    X_true = np.concatenate([np.zeros(300, dtype=int), np.ones(300, dtype=int)])
    Y      = np.concatenate([
        rng.normal(loc=-10, scale=0.3, size=300),
        rng.normal(loc=+10, scale=0.3, size=300),
    ])
    labels = _kmeans_label_assignment(Y, K=2, random_state=0)
    # Accept either label permutation.
    err = min(
        np.mean(labels != X_true),
        np.mean(labels != (1 - X_true)),
    )
    assert err < 0.01


# ---------------------------------------------------------------------------
# _warmstart_from_kmeans — invariants
# ---------------------------------------------------------------------------

def _simulate_separated_hmc_in_k2(N: int = 800, seed: int = 0):
    """HMC-IN K=2 with well-separated Gaussian means ±3 → easy classification."""
    mdl = PMCModel(HMC_IN_K2)
    X, Y = simulate(mdl, N=N, seed=seed)
    return mdl, X, Y


def test_warmstart_returns_valid_pmc_model():
    mdl, _, Y = _simulate_separated_hmc_in_k2()
    warm = _warmstart_from_kmeans(
        mdl, Y,
        random_state=0,
        fit_margins=True,
        candidates=[],
        selection_criterion="mle",
        margin_selection_rule="mle",
    )
    # Variant / K / dimensionality are structural — must be preserved.
    assert warm.variant == mdl.variant
    assert warm.K       == mdl.K
    assert warm.d       == mdl.d


def test_warmstart_reproducible_same_seed():
    mdl, _, Y = _simulate_separated_hmc_in_k2()
    a = _warmstart_from_kmeans(
        mdl, Y,
        random_state=42, fit_margins=True,
        candidates=[], selection_criterion="mle", margin_selection_rule="mle",
    )
    b = _warmstart_from_kmeans(
        mdl, Y,
        random_state=42, fit_margins=True,
        candidates=[], selection_criterion="mle", margin_selection_rule="mle",
    )
    # Compare numerical params via raw dicts (TOML re-roundtrip).
    np.testing.assert_allclose(
        np.asarray(a.transition_A), np.asarray(b.transition_A), atol=1e-12,
    )


def test_warmstart_fit_margins_false_preserves_margin_params():
    """When fit_margins=False, warm-start must NOT touch the margin parameters."""
    mdl, _, Y = _simulate_separated_hmc_in_k2()
    original_margins = [dict(blk["params"]) for blk in mdl.raw.get("margins", [])]

    warm = _warmstart_from_kmeans(
        mdl, Y,
        random_state=0,
        fit_margins=False,
        candidates=[],
        selection_criterion="mle",
        margin_selection_rule="mle",
    )

    warm_margins = [dict(blk["params"]) for blk in warm.raw.get("margins", [])]
    assert warm_margins == original_margins


def test_warmstart_fit_margins_true_recovers_means():
    """With well-separated data and fit_margins=True, the warm-start refits μ_i
    close to the empirical cluster means (up to label permutation)."""
    rng = np.random.default_rng(0)
    Y = np.concatenate([
        rng.normal(loc=-3, scale=1.0, size=500),
        rng.normal(loc=+3, scale=1.0, size=500),
    ])
    mdl = PMCModel(HMC_IN_K2)
    warm = _warmstart_from_kmeans(
        mdl, Y,
        random_state=0,
        fit_margins=True,
        candidates=[],
        selection_criterion="mle",
        margin_selection_rule="mle",
    )
    locs = sorted(float(blk["params"]["loc"]) for blk in warm.raw["margins"])
    # Should be close to ±3 in some order.
    assert abs(locs[0] - (-3.0)) < 0.5
    assert abs(locs[1] - (+3.0)) < 0.5


# ---------------------------------------------------------------------------
# End-to-end ice() with init="kmeans"
# ---------------------------------------------------------------------------

def test_ice_init_kmeans_recovers_classification_on_hmc_in_k2():
    """End-to-end: ICE with K-means warm-start converges to near-Bayes
    error on the K=2 HMM (means ±1 → marginal Bayes error ≈ 16 %; the
    0.9/0.1 transitions reduce the sequence error to roughly 7-8 %)
    starting from an adversarial initial model where both margins are
    declared at loc=0."""
    mdl, X_ref, Y = _simulate_separated_hmc_in_k2(N=1500, seed=0)

    # Adversarial initial model: both margins at loc=0 — the user's
    # declared parameters are intentionally wrong so K-means must rescue
    # the fit.
    raw = mdl.raw
    for blk in raw["margins"]:
        blk["params"]["loc"]   = 0.0
        blk["params"]["scale"] = 1.0
    bad_init = PMCModel.from_dict(raw)

    fitted, trace = ice(
        bad_init, Y,
        ice_cfg={
            "init":         "kmeans",
            "kmeans_seed":  0,
            "fit_margins":  True,
            "max_iter":     10,
            "candidates":   [],
        },
    )
    assert len(trace.log_liks) >= 1
    X_hat, _, _ = classify(fitted, Y)
    er = error_rate(X_ref, X_hat)
    # Loose upper bound (well below the 16 % marginal Bayes error) ensuring
    # the warm-start actually recovered the two clusters from a degenerate
    # initial model.
    assert er < 0.10, f"expected <10 % error after K-means warm-start, got {er:.3f}"


def test_ice_init_kmeans_beats_bad_model_init():
    """K-means warm-start should reach a *higher* final log-likelihood than
    the same ICE run starting from an adversarial model — on a setting where
    the user's initial parameters are far from the data."""
    mdl, _, Y = _simulate_separated_hmc_in_k2(N=1200, seed=1)

    # Identical adversarial init for both runs.
    raw = mdl.raw
    for blk in raw["margins"]:
        blk["params"]["loc"]   = 0.0
        blk["params"]["scale"] = 1.0
    bad_init = PMCModel.from_dict(raw)

    cfg_common = {"fit_margins": True, "max_iter": 8, "candidates": []}

    _, trace_model = ice(bad_init, Y, ice_cfg={**cfg_common, "init": "model"})
    _, trace_kmean = ice(
        bad_init, Y,
        ice_cfg={**cfg_common, "init": "kmeans", "kmeans_seed": 0},
    )
    assert trace_kmean.log_liks[-1] > trace_model.log_liks[-1] + 1e-3, (
        f"K-means warm-start did not improve final LL: "
        f"model={trace_model.log_liks[-1]:.4f}  kmeans={trace_kmean.log_liks[-1]:.4f}"
    )


def test_ice_init_invalid_value_raises():
    mdl, _, Y = _simulate_separated_hmc_in_k2()
    with pytest.raises(ValueError, match="Unknown init strategy"):
        ice(mdl, Y, ice_cfg={"init": "not-a-thing", "max_iter": 1})


def test_ice_init_kmeans_reproducible_end_to_end():
    """Two ICE runs with the same K-means seed and same data → identical trace."""
    mdl, _, Y = _simulate_separated_hmc_in_k2(N=600, seed=0)
    cfg = {
        "init":         "kmeans",
        "kmeans_seed":  17,
        "fit_margins":  True,
        "max_iter":     5,
        "candidates":   [],
    }
    _, trace_a = ice(mdl, Y, ice_cfg=cfg)
    _, trace_b = ice(mdl, Y, ice_cfg=cfg)
    np.testing.assert_allclose(trace_a.log_liks, trace_b.log_liks, atol=1e-10)


# ---------------------------------------------------------------------------
# PMC variant smoke test — make sure K-means warm-start also runs for a
# variant that uses copulas (the M-step code path is wider there).
# ---------------------------------------------------------------------------

def test_ice_init_kmeans_runs_on_pmc_k2():
    """Smoke: PMC variant with copulas — K-means warm-start should not crash
    and should produce a finite final log-likelihood."""
    mdl = PMCModel(PMC_K2)
    _, Y = simulate(mdl, N=600, seed=0)
    _, trace = ice(
        mdl, Y,
        ice_cfg={
            "init":         "kmeans",
            "kmeans_seed":  0,
            "fit_margins":  True,
            "max_iter":     5,
            "candidates":   ["Gauss", "Clayton", "GH"],
        },
    )
    assert len(trace.log_liks) >= 1
    assert np.isfinite(trace.log_liks[-1])
