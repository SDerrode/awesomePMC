"""
test_multistart_parallel.py — process-parallel multistart must be a pure
wall-clock optimisation: identical results to the sequential path.

The starting points are pre-drawn with the same RNG consumption in both
modes, ICE is deterministic given its start, and SEM's FFBS stream is seeded
per start (``sem_seed + start_index``, audit N-5) — so
``multistart_workers > 1`` must reproduce the sequential run exactly
(log-likelihood traces and fitted parameters).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pmcprg.pmc import PMCModel, ice, sem, simulate
from pmcprg.pmc._estim_common import shared_estim_defaults

MODEL = Path(__file__).resolve().parents[1] / "pmc" / "models" / "pmc_gauss_k2.toml"


def test_multistart_workers_default_is_sequential():
    """The knob lives in the single-source defaults and defaults to 1."""
    assert shared_estim_defaults()["multistart_workers"] == 1


def _cfg(workers: int, **extra) -> dict:
    cfg = {
        "n_starts":           3,
        "multistart_seed":    5,
        "multistart_workers": workers,
        "max_iter":           2,
        "candidates":         ["Gauss", "Clayton"],
    }
    cfg.update(extra)
    return cfg


def _assert_same_fit(f_a, t_a, f_b, t_b):
    assert t_a.log_liks == t_b.log_liks
    losers_a = sorted(t.log_liks[-1] for t in t_a.multistart_runs)
    losers_b = sorted(t.log_liks[-1] for t in t_b.multistart_runs)
    assert losers_a == losers_b and len(losers_a) == 2
    np.testing.assert_array_equal(f_a.transition_A, f_b.transition_A)
    assert f_a.pdf(0, 1, -0.4) == f_b.pdf(0, 1, -0.4)


@pytest.mark.slow
def test_parallel_ice_matches_sequential():
    mdl = PMCModel(MODEL)
    _, Y = simulate(mdl, N=250, seed=11)
    f_seq, t_seq = ice(mdl, Y, ice_cfg=_cfg(1))
    f_par, t_par = ice(mdl, Y, ice_cfg=_cfg(2))
    _assert_same_fit(f_seq, t_seq, f_par, t_par)


@pytest.mark.slow
def test_parallel_sem_matches_sequential():
    mdl = PMCModel(MODEL)
    _, Y = simulate(mdl, N=250, seed=12)
    f_seq, t_seq = sem(mdl, Y, sem_cfg=_cfg(1, sem_seed=3))
    f_par, t_par = sem(mdl, Y, sem_cfg=_cfg(2, sem_seed=3))
    _assert_same_fit(f_seq, t_seq, f_par, t_par)
