"""Memory of the gap quadrature: transitions built when needed, within a budget.

A missing → missing transition that touches a local grid is a dense (K·G)²
block, one per step of a gap; a long gappy series kept thousands of them
(``gaps`` module docstring, "Memory": Intel Lab mote 48, 16 271 missing rows,
2.1 GB at G = 64 and 21 GB at G = 256). A chain now keeps at most
``_TRANSITION_BUDGET`` bytes of them and rebuilds the others when a pass
needs them again (backward, ξ, FFBS, the next chain of the filter-weighted
grids). A rebuilt transition is the same operations on the same inputs, so
every result must be the same *bit for bit* whatever the budget: every
comparison below is ``np.array_equal``, on the same machine and code.
"""
from __future__ import annotations

import gc
import logging
import math
import weakref

import numpy as np
import pytest

from pmcprg.pmc import (PMCModel, StateMarkovMissingness, StateMissingness, flag_outliers, gaps,
                        predictive_pit, simulate)


def _tau(rho):
    return 2.0 / math.pi * math.asin(rho)


def _state(means, sds, p, tau, variant="PMC"):
    K = len(means)
    return PMCModel.from_dict({
        "model": {"variant": variant, "K": K}, "prior": {"p" if variant == "PMC" else "A": p},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": means[i], "scale": sds[i]}}
                    for i in range(K)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(K) for j in range(K)]})


def _pair3():
    p = [[0.30, 0.03, 0.01], [0.02, 0.35, 0.03], [0.02, 0.02, 0.22]]
    loc = [[0.0, 0.4, 1.0], [0.6, 2.0, 2.5], [1.5, 3.0, 4.0]]
    sc = [[1.0, 1.2, 0.9], [1.1, 0.8, 1.0], [0.9, 1.0, 0.7]]
    cops = [["GH", "Frank", "Gauss"], ["Clayton", "GH", "Frank"], ["Gauss", "GH", "GH"]]
    taus = [[0.97, 0.6, 0.5], [0.6, 0.98, 0.7], [0.5, 0.8, 0.95]]
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 3}, "prior": {"p": p},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": loc[i][j], "scale": sc[i][j]}}
                    for i in range(3) for j in range(3)],
        "copulas": [{"i": i, "j": j, "name": cops[i][j], "tau": taus[i][j]}
                    for i in range(3) for j in range(3)]})


def _gapped(model, N, runs, seed):
    _, Y = simulate(model, N=N, seed=seed)
    Y = np.array(Y, dtype=float)
    for a, L in runs:
        Y[a:a + L] = np.nan
    return Y


def _cases():
    tau = _tau(0.999)
    m3 = _state([0.0, 1.0, 2.5], [1.0, 0.7, 1.3],
                [[0.30, 0.02, 0.01], [0.02, 0.30, 0.01], [0.01, 0.01, 0.32]], tau)
    m2 = _state([0.0, 1.5], [1.0, 0.8], [[0.45, 0.05], [0.05, 0.45]], 0.95)
    pair2 = PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml").raw
    for c in pair2["copulas"]:
        c["tau"] = 0.97
    pair2 = PMCModel.from_dict(pair2)
    hmc = _state([0.0, 0.0], [1.0, 1.0], [[0.9, 0.1], [0.2, 0.8]], tau, variant="HMC-DN")
    pair3 = _pair3()
    return {
        # a 17-row leading gap of known prior (transitions of the prior), long gap
        "state_K3_lead": (m3, _gapped(m3, 120, [(0, 17), (40, 1), (60, 3), (80, 12)], 2)),
        # non-ignorable "state": factors on the rebuilt transitions
        "state_K2_mnar": (m2.with_missingness(StateMissingness(rates=[0.05, 0.4])),
                          _gapped(m2, 120, [(0, 2), (30, 3), (70, 8), (110, 1)], 5)),
        # pair margins, "state-markov": the leading gap keeps the interior rows
        "pair_K2_markov": (pair2.with_missingness(StateMarkovMissingness(onset=[0.05, 0.3],
                                                                         persistence=[0.4, 0.85])),
                           _gapped(pair2, 120, [(0, 4), (35, 2), (60, 9), (100, 1)], 6)),
        "pair_K3": (pair3, _gapped(pair3, 110, [(0, 5), (25, 1), (50, 4), (80, 10)], 4)),
        "hmcdn": (hmc, _gapped(hmc, 100, [(0, 2), (20, 1), (50, 5), (70, 15)], 7)),
    }


G = 24


def _outputs(model, Y):
    out = {}
    p = gaps.gap_posterior(model, Y, gap_nodes=G, xi=True)
    for f in ("log_lik", "alpha_hat", "beta_hat", "gamma", "xi", "node_post", "nodes", "quad_error"):
        out[f"posterior.{f}"] = np.asarray(getattr(p, f))
    f = gaps.forecast(model, Y[:90], 3, gap_nodes=G)
    for k in ("mean", "sd", "quantile_values", "grid_nodes", "grid_mass", "state_probs"):
        out[f"forecast.{k}"] = np.asarray(getattr(f, k))
    im = gaps.impute(model, Y, gap_nodes=G, n_samples=2, rng=3)
    for k in ("mean", "sd", "quantile_values", "density", "grid_mass", "x_samples", "y_samples"):
        out[f"impute.{k}"] = np.asarray(getattr(im, k))
    pit = predictive_pit(model, Y, gap_nodes=G)
    out["pit.pvalue"] = np.asarray(pit.pvalue)
    fl = flag_outliers(model, Y, sequential=True, gap_nodes=G)
    out["flag.pvalue"] = np.asarray(fl.pvalue)
    X, Yd = gaps._grid_sample(model, Y, gaps.missing_mask(Y), np.random.default_rng(1), G, True)
    out["sample.X"], out["sample.Y"] = X, Yd
    return out


@pytest.fixture
def budget(monkeypatch):
    def set_(b):
        monkeypatch.setattr(gaps, "_TRANSITION_BUDGET", int(b))
        gaps._GRID_CACHE.clear()
    yield set_
    gaps._GRID_CACHE.clear()


#: the two largest cases (~3 s each) run in the full suite only
_SLOW = ("state_K3_lead", "pair_K3")


@pytest.mark.parametrize("name", [pytest.param(n, marks=pytest.mark.slow) if n in _SLOW else n
                                  for n in _cases()])
def test_results_do_not_depend_on_the_transition_budget(name, budget):
    model, Y = _cases()[name]
    miss = gaps.missing_mask(Y)
    ref = gaps.reference_grid(model, G)
    budget(1 << 40)
    chain, _, _, _ = gaps._run_chain(model, Y, miss, ref, backward=True)
    assert any(g.local for g in chain.grids.values()), "premise: local grids"
    assert chain._lazy, "premise: transitions built when needed"
    one_kb = sum(a.nbytes for a in (chain._store[min(chain._store)][0],))
    full = _outputs(model, Y)
    # every transition rebuilt when needed, then about half of them kept
    for b in (0, one_kb * len(chain._store) // 2):
        budget(b)
        got = _outputs(model, Y)
        for k, v in full.items():
            assert np.array_equal(got[k], v, equal_nan=True), (name, b, k)


@pytest.mark.parametrize("name", ["pair_K2_markov", "hmcdn"])
def test_results_do_not_depend_on_the_blocks_of_the_whole_sequence_temporaries(name, monkeypatch):
    """The (P, C, G) and (P·G, K, K) temporaries of the grid construction and
    the (P, 5G, K, K) ones of the Nyström densities of ``impute`` are
    computed by blocks of rows (``_CHUNK`` elements); blocks of a few rows
    give the same results as one block."""
    model, Y = _cases()[name]
    gaps._GRID_CACHE.clear()
    full = _outputs(model, Y)
    monkeypatch.setattr(gaps, "_CHUNK", 300)
    gaps._GRID_CACHE.clear()
    got = _outputs(model, Y)
    gaps._GRID_CACHE.clear()
    for k, v in full.items():
        assert np.array_equal(got[k], v, equal_nan=True), (name, k)


def test_the_grid_cache_keeps_the_newest_entry_within_its_bytes(monkeypatch):
    model, Y = _cases()["hmcdn"]
    miss = gaps.missing_mask(Y)
    ref = gaps.reference_grid(model, G)
    gaps._GRID_CACHE.clear()
    try:
        gaps._gap_grids(model, Y, miss, ref)
        gaps._gap_grids(model, Y[:-1], miss[:-1], ref)
        assert len(gaps._GRID_CACHE) == 2
        monkeypatch.setattr(gaps, "_GRID_CACHE_BYTES", 1)
        g = gaps._gap_grids(model, Y[:-2], miss[:-2], ref)
        assert len(gaps._GRID_CACHE) == 1                      # the newest, whatever its size
        hit = gaps._gap_grids(model, Y[:-2], miss[:-2], ref)
        assert all(hit[n] is g[n] for n in g)
    finally:
        gaps._GRID_CACHE.clear()


def test_the_kept_transitions_stay_within_the_budget(budget):
    model, Y = _cases()["state_K3_lead"]
    miss = gaps.missing_mask(Y)
    ref = gaps.reference_grid(model, G)
    budget(1 << 40)
    chain, _, _, _ = gaps._run_chain(model, Y, miss, ref, backward=True)
    total = chain._stored
    assert total > 0 and len(chain._store) == len(chain._lazy)
    budget(total // 3)
    chain, _, _, _ = gaps._run_chain(model, Y, miss, ref, backward=True)
    assert 0 < chain._stored <= total // 3
    assert 0 < len(chain._store) < len(chain._lazy)


def test_report_and_xi_rebuild_what_the_backward_pass_did_not_see(budget):
    """ξ and the quadrature report of messages from another backward pass
    (``_backward_chain`` without the forward messages) rebuild the
    transitions the chain did not keep: the same values."""
    model, Y = _cases()["pair_K3"]
    miss = gaps.missing_mask(Y)
    ref = gaps.reference_grid(model, G)
    budget(0)
    chain, alphas, _, betas = gaps._run_chain(model, Y, miss, ref, backward=True)
    fused = gaps._chain_posterior(chain, alphas, betas, want_xi=True)
    q_fused = gaps._quadrature_report(chain, alphas, betas)
    betas2 = gaps._backward_chain(chain)                     # no fused parts
    assert all(np.array_equal(a, b) for a, b in zip(betas, betas2))
    rebuilt = gaps._chain_posterior(chain, alphas, betas2, want_xi=True)
    assert np.array_equal(rebuilt["xi"], fused["xi"])
    assert gaps._quadrature_report(chain, alphas, betas2) == q_fused


def test_a_dropped_chain_is_freed_at_once():
    """No reference cycle: the chain of a failed linear pass, or the one a
    later chain took its transitions from, goes without the garbage collector."""
    model, Y = _cases()["state_K3_lead"]
    miss = gaps.missing_mask(Y)
    ref = gaps.reference_grid(model, G)
    gc.disable()
    try:
        chain = gaps._Chain(model, Y, miss, ref, log=False)
        assert gaps._forward_chain(chain) is not None
        r = weakref.ref(chain)
        del chain
        assert r() is None
    finally:
        gc.enable()


def test_a_kernel_overflow_in_a_transition_built_later_still_names_its_cause(monkeypatch, caplog):
    """A linear pass whose forward stops on an underflow first names an
    overflow of a transition it did not build yet, as when every transition
    was built before the pass; the results are those of the log-space pass."""
    model, Y = _cases()["state_K3_lead"]
    miss = gaps.missing_mask(Y)
    ref = gaps.reference_grid(model, G)
    grids = gaps._gap_grids(model, Y, miss, ref)
    log_chain = gaps._Chain(model, Y, miss, ref, log=True, grids=grids)
    a_log, ll_log = gaps._forward_chain(log_chain)
    last = max(n for n in range(len(Y) - 1) if miss[n] and miss[n + 1]
               and (grids[n].local or grids[n + 1].local))
    orig_ko, orig_fw = gaps._kernel_outer, gaps._forward_chain
    state = {}

    def ko(model_, fL, FL, fR, FR, *, log):
        out = orig_ko(model_, fL, FL, fR, FR, log=log)
        if not log and state.get("flag") and np.array_equal(fR, state["fB"]):
            return out[0], True                    # the last local transition overflows
        return out

    def fw(chain, **kw):
        if not chain.log:
            state["flag"] = True
            state["fB"] = chain._mg.nodes(chain._lazy[last][2])[0]
            return None                            # the linear forward underflows at once
        return orig_fw(chain, **kw)

    monkeypatch.setattr(gaps, "_kernel_outer", ko)
    monkeypatch.setattr(gaps, "_forward_chain", fw)
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gaps"):
        chain, (alphas, ll), _ = gaps._pass(model, Y, miss, ref, grids, None, backward=False)
    assert chain.log
    assert any("a kernel value overflows" in r.message for r in caplog.records)
    assert ll == ll_log and all(np.array_equal(a, b) for a, b in zip(alphas, a_log))
