"""Tests for :func:`pmcprg.pmc.sem.sem` — Stochastic EM estimator.

Adds PR2: SEM as a sister estimator to ICE, sharing the M-step and the
optional K-means warm-start, but completing the latent X with a
Forward-Filter Backward-Sample draw at every iteration.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from pmcprg.pmc.inference import classify, error_rate
from pmcprg.pmc.model     import PMCModel
from pmcprg.pmc.sem       import (
    SemTrace,
    _parse_sem_cfg,
    sem,
    sem_image,
)
from pmcprg.pmc.simulate  import simulate


MODELS    = pathlib.Path("pmcprg/pmc/models")
HMC_IN_K2 = MODELS / "hmc_in_gauss_k2.toml"
PMC_K2    = MODELS / "pmc_gauss_k2.toml"


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_default_cfg_keys_present():
    mdl = PMCModel(HMC_IN_K2)
    cfg = _parse_sem_cfg(mdl, None)
    for key in (
        "max_iter", "candidates", "selection_criterion",
        "margin_selection_rule", "fit_margins",
        "init", "kmeans_seed", "sem_seed",
        "n_starts", "multistart_seed", "multistart_jitter",
    ):
        assert key in cfg, f"missing key {key}"


def test_cfg_override_threads_through():
    mdl = PMCModel(HMC_IN_K2)
    cfg = _parse_sem_cfg(mdl, {"sem_seed": 99, "max_iter": 7})
    assert cfg["sem_seed"] == 99
    assert cfg["max_iter"] == 7


def test_sem_section_overrides_ice_section():
    """A ``[sem]`` TOML block overrides shared ``[ice]`` keys (audit Q-5/A-5).

    Previously ``_parse_sem_cfg`` guarded the ``[sem]`` read behind
    ``hasattr(model, "sem_config")`` — a method that did not exist — so the
    ``[sem]`` section was silently ignored. ``PMCModel.sem_config()`` now makes
    it real.
    """
    mdl = PMCModel(HMC_IN_K2)
    raw = mdl.raw
    raw["ice"] = {"max_iter": 11, "fit_margins": True}
    raw["sem"] = {"max_iter": 22}            # [sem] wins over [ice] for shared key
    mdl2 = PMCModel.from_dict(raw)

    cfg = _parse_sem_cfg(mdl2, None)
    assert cfg["max_iter"] == 22             # from [sem]
    assert cfg["fit_margins"] is True        # inherited from [ice]

    # Caller-supplied dict still has the highest priority.
    cfg2 = _parse_sem_cfg(mdl2, {"max_iter": 33})
    assert cfg2["max_iter"] == 33


# ---------------------------------------------------------------------------
# Trace shape & types
# ---------------------------------------------------------------------------

def test_sem_trace_shapes():
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    fitted, trace = sem(mdl, Y, sem_cfg={"max_iter": 5, "sem_seed": 0})
    assert isinstance(trace, SemTrace)
    assert len(trace.log_liks) == 5
    assert trace.tau_history.shape       == (5, mdl.K, mdl.K)
    assert trace.p_history.shape         == (5, mdl.K, mdl.K)
    assert trace.sampled_X_history.shape == (5, len(Y))
    assert trace.sampled_X_history.dtype.kind == "i"


# ---------------------------------------------------------------------------
# Reproducibility under fixed sem_seed
# ---------------------------------------------------------------------------

def test_sem_reproducible_same_seed():
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    cfg = {"max_iter": 4, "sem_seed": 17, "fit_margins": True}
    _, trace_a = sem(mdl, Y, sem_cfg=cfg)
    _, trace_b = sem(mdl, Y, sem_cfg=cfg)
    np.testing.assert_allclose(trace_a.log_liks, trace_b.log_liks, atol=1e-10)
    np.testing.assert_array_equal(
        trace_a.sampled_X_history, trace_b.sampled_X_history,
    )


def test_sem_different_seeds_produce_different_traces():
    """Stochastic algorithm — different ``sem_seed`` values must lead to
    visibly different sampled paths (proves the seed actually controls
    the FFBS RNG)."""
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    cfg_a = {"max_iter": 3, "sem_seed": 0, "fit_margins": True}
    cfg_b = {"max_iter": 3, "sem_seed": 999, "fit_margins": True}
    _, trace_a = sem(mdl, Y, sem_cfg=cfg_a)
    _, trace_b = sem(mdl, Y, sem_cfg=cfg_b)
    # Sampled X paths must differ at *some* iteration.
    assert not np.array_equal(
        trace_a.sampled_X_history, trace_b.sampled_X_history,
    )


# ---------------------------------------------------------------------------
# End-to-end: SEM recovers a good fit from an adversarial init
# ---------------------------------------------------------------------------

def _adversarial_model() -> PMCModel:
    mdl = PMCModel(HMC_IN_K2)
    raw = mdl.raw
    for blk in raw["margins"]:
        blk["params"]["loc"]   = 0.0
        blk["params"]["scale"] = 1.0
    return PMCModel.from_dict(raw)


def test_sem_improves_log_lik_from_bad_init():
    mdl_truth = PMCModel(HMC_IN_K2)
    _, Y      = simulate(mdl_truth, N=1200, seed=0)
    bad_init  = _adversarial_model()

    _, trace = sem(
        bad_init, Y,
        sem_cfg={"max_iter": 10, "sem_seed": 0, "fit_margins": True},
    )
    assert trace.log_liks[-1] > trace.log_liks[0] + 10.0, (
        f"SEM did not improve LL: {trace.log_liks[0]:.2f} → "
        f"{trace.log_liks[-1]:.2f}"
    )


def test_sem_low_error_on_separated_hmc_in_k2():
    """End-to-end correctness: SEM with K-means warm-start reaches a
    realistic error rate (<10 % — below the 16 % marginal Bayes error)
    from an adversarial initial model."""
    pytest.importorskip("sklearn", reason="init='kmeans' needs scikit-learn (extra [ml])")
    mdl_truth     = PMCModel(HMC_IN_K2)
    X_ref, Y      = simulate(mdl_truth, N=1500, seed=0)
    bad_init      = _adversarial_model()

    fitted, _ = sem(
        bad_init, Y,
        sem_cfg={
            "init":         "kmeans",
            "kmeans_seed":  0,
            "max_iter":     10,
            "sem_seed":     0,
            "fit_margins":  True,
            "candidates":   [],
        },
    )
    X_hat, _, _ = classify(fitted, Y)
    er = error_rate(X_ref, X_hat)
    assert er < 0.10, f"expected <10 % error after SEM + K-means, got {er:.3f}"


# ---------------------------------------------------------------------------
# Multistart for SEM
# ---------------------------------------------------------------------------

def test_sem_multistart_runs_and_returns_best():
    mdl = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=600, seed=0)
    fitted, trace = sem(
        mdl, Y,
        sem_cfg={
            "max_iter":         3,
            "sem_seed":         0,
            "n_starts":         3,
            "multistart_seed":  1,
            "multistart_jitter": 0.10,
        },
    )
    # The best trace gets the non-best ones attached for the GUI's
    # multistart-comparison view.
    assert len(trace.multistart_runs) == 2
    assert all(isinstance(t, SemTrace) for t in trace.multistart_runs)
    # Best trace's final LL must dominate the losing runs.
    other_finals = [t.log_liks[-1] for t in trace.multistart_runs]
    assert trace.log_liks[-1] >= max(other_finals)


def test_sem_multistart_uses_independent_ffbs_streams():
    """Each multistart run must draw an *independent* FFBS stream (audit N-5).

    The s=0 (unperturbed) run uses seed_offset 0, so it reproduces a standalone
    single-start fit at the same sem_seed (backward compatibility); the
    perturbed runs use sem_seed + s, so their sampled paths differ from the
    unperturbed one rather than being byte-identical."""
    mdl  = PMCModel(HMC_IN_K2)
    _, Y = simulate(mdl, N=400, seed=0)
    cfg  = {"max_iter": 3, "sem_seed": 5}

    _, t_single = sem(mdl, Y, sem_cfg=cfg)
    _, t_multi  = sem(mdl, Y, sem_cfg={**cfg, "n_starts": 3, "multistart_seed": 1})

    all_runs = [t_multi, *t_multi.multistart_runs]
    # Exactly one run (the unperturbed s=0) reproduces the standalone stream.
    reproduces = [
        np.array_equal(t_single.sampled_X_history, r.sampled_X_history)
        for r in all_runs
    ]
    assert sum(reproduces) == 1, "unperturbed run must match standalone(seed)"
    # And the runs are not all sharing one stream (the bug): at least two
    # distinct sampled-path histories across the 3 runs.
    distinct = {r.sampled_X_history.tobytes() for r in all_runs}
    assert len(distinct) >= 2, "multistart runs share an identical FFBS stream"


# ---------------------------------------------------------------------------
# Smoke test: PMC variant with copulas
# ---------------------------------------------------------------------------

def test_sem_runs_on_pmc_variant_with_copulas():
    mdl = PMCModel(PMC_K2)
    _, Y = simulate(mdl, N=600, seed=0)
    _, trace = sem(
        mdl, Y,
        sem_cfg={
            "max_iter":   4,
            "sem_seed":   0,
            "fit_margins": True,
            "candidates": ["Gauss", "Clayton", "GH"],
        },
    )
    assert len(trace.log_liks) == 4
    assert np.isfinite(trace.log_liks[-1])
    # tau_history must contain finite values (copula present) at most pairs.
    assert np.isfinite(trace.tau_history).any()


# ---------------------------------------------------------------------------
# Smoke test: sem_image dispatches to sem() on a linearised image
# ---------------------------------------------------------------------------

def test_sem_image_shape_mismatch_raises():
    mdl = PMCModel(HMC_IN_K2)
    bad_img = np.zeros((4, 5, 2))   # d=2 but model is d=1
    with pytest.raises(ValueError, match="channels"):
        sem_image(mdl, bad_img, sem_cfg={"max_iter": 1})


def test_sem_image_runs_on_tiny_grayscale():
    mdl = PMCModel(HMC_IN_K2)
    rng = np.random.default_rng(0)
    img = rng.normal(size=(6, 8))   # 6×8 grayscale image, d=1
    _, trace = sem_image(mdl, img, sem_cfg={"max_iter": 2, "sem_seed": 0})
    assert len(trace.log_liks) == 2
