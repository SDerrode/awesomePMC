"""Multistart jitter clips its own out-of-range τ without a WARNING (F2).

``perturb_initial_model`` shifts every copula τ by ``jitter · span · N(0, 1)``
and clips it. The clip used to go through ``CopulaEnum.correct_tau``, which
logs a WARNING meant for user input — e.g. ``Gauss.correct_tau: τ=1.6615
outside valid range [-1.0000, 1.0000] — clipped to τ=1.0000`` for
``pmc_gauss_k2.toml``, seed 3, jitter 0.25 — about a value nobody gave, and
naming 1.0 while 0.9998 is stored. The clip is now logged at DEBUG; the result
is unchanged.

The reference values below were produced by the code *before* the change
(commit 0cb1283). τ and ``loc`` involve only IEEE +, ×, clip and the
platform-independent NumPy ``Generator``, so they are compared exactly; the
prior and ``scale`` go through ``np.exp``, whose last bit may differ between
platforms, hence ``rtol = 1e-12``.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum
from pmcprg.pmc._estim_common import build_multistart_inits, perturb_initial_model
from pmcprg.pmc.model import PMCModel

MODELS = "pmcprg/pmc/models"

# pmc_gauss_k2.toml, jitter 0.25: seed → (τ of the 4 copula blocks, prior p
# row-major, (loc, scale) of the 2 margins). Seeds 3 and 4 clip a τ above 1 and
# logged a WARNING before the change; seed 0 did not.
_REFERENCE = {
    0: ([0.24813238209650368, -0.6327107355230263, -0.3116372312686761, 0.6206629896736218],
        [0.4493675794344658, 0.05179942599609423, 0.05179942599609423, 0.4470335685733457],
        [(-1.1339173432902778, 1.0946106877262385), (1.3260000112825343, 1.2671499449466976)]),
    3: ([0.16739346186252912, 0.9998, 0.11289330661396088, 0.4236846028292023],
        [0.6134287213067451, 0.03351413267098475, 0.03351413267098475, 0.3195430133512854],
        [(-1.1131623230276115, 0.9475275264621629), (0.49500346771318726, 0.9436659005297728)]),
    4: ([-0.2040938920931945, 0.12088593843842566, 0.11769045936872738, 0.9998],
        [0.36883588705579723, 0.059643004056211354, 0.059643004056211354, 0.5118781048317801],
        [(-1.4103493236461617, 0.9987000296514684), (0.8441340647529016, 1.0378568655895692)]),
}


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("seed", sorted(_REFERENCE))
def test_perturbed_model_matches_the_pre_change_code_without_warning(seed, caplog):
    model = PMCModel(f"{MODELS}/pmc_gauss_k2.toml")
    with caplog.at_level(logging.DEBUG, logger="pmcprg"):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=0.25)
    assert _warnings(caplog) == []

    tau_ref, p_ref, margins_ref = _REFERENCE[seed]
    raw = out.raw
    assert [blk["tau"] for blk in raw["copulas"]] == tau_ref
    np.testing.assert_allclose(np.ravel(raw["prior"]["p"]), p_ref, rtol=1e-12, atol=0.0)
    assert [blk["params"]["loc"] for blk in raw["margins"]] == [m[0] for m in margins_ref]
    np.testing.assert_allclose([blk["params"]["scale"] for blk in raw["margins"]],
                               [m[1] for m in margins_ref], rtol=1e-12, atol=0.0)
    if max(tau_ref) == 0.9998:
        # The clip is still logged, at DEBUG.
        assert any("correct_tau" in r.getMessage() and r.levelno == logging.DEBUG
                   for r in caplog.records)


@pytest.mark.parametrize("fixture", ["pmc_gauss_k2.toml", "hmc_dn_gauss_k2.toml",
                                     "pmc_pair_gauss_k2.toml", "sp2016_gice_k2.toml"])
def test_jitter_logs_no_warning_over_seeds(fixture, caplog):
    model = PMCModel(f"{MODELS}/{fixture}")
    with caplog.at_level(logging.INFO, logger="pmcprg"):
        for seed in range(12):
            perturb_initial_model(model, np.random.default_rng(seed), jitter=0.25)
    assert _warnings(caplog) == []


def test_family_multistart_starts_log_no_warning(caplog):
    """Jitter + family redraws (``"random"``) and the diagonal sweep: no WARNING
    other than the sweep's own truncation notice."""
    model = PMCModel(f"{MODELS}/pmc_gauss_k2.toml")
    cands = ["Gauss", "GH", "Clayton", "Frank", "Joe", "Plackett", "FGM"]
    with caplog.at_level(logging.INFO, logger="pmcprg"):
        for mode in ("none", "random"):
            cfg = {"n_starts": 12, "multistart_seed": 3, "multistart_jitter": 0.25,
                   "multistart_families": mode, "candidates": cands}
            build_multistart_inits(model, cfg, label="ICE")
    assert _warnings(caplog) == []


def test_user_facing_correct_tau_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._base"):
        assert CopulaEnum.GAUSSIAN.correct_tau(1.6615) == 1.0
    assert any("outside valid range" in r.getMessage() for r in _warnings(caplog))
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="pmcprg.copulas._base"):
        assert CopulaEnum.GAUSSIAN.correct_tau(1.6615, warn=False) == 1.0
        assert CopulaEnum.GAUSSIAN.correct_tau(float("nan"), warn=False) == 0.0
    assert _warnings(caplog) == []
    assert len(caplog.records) == 2
