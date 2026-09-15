"""
test_multistart_families.py — copula-family multistart (``multistart_families``).

ICE on the CSDA-2013 Exp. 3 design has several fixed points that differ by the
copula family of the diagonal pairs; the parameter-jitter multistart keeps the
families of the initial model and cannot leave a wrong-family basin. The
``multistart_families`` key changes the families of the starts:

* ``"none"``   — parameter jitter only, bit-identical to the historical driver;
* ``"random"`` — every pair redrawn among the candidates, on top of the jitter;
* ``"sweep"``  — every combination of candidate families on the diagonal pairs.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path

import numpy as np
import pytest

from pmcprg.copulas._base import CopulaEnum
from pmcprg.pmc import PMCModel, ice, sem, simulate
from pmcprg.pmc._estim_common import (
    MAX_SWEEP_COMBINATIONS,
    MULTISTART_FAMILY_MODES,
    RANDOM_TAU_BAND,
    SWEEP_TAU,
    build_multistart_inits,
    perturb_initial_model,
    sem_estim_defaults,
    shared_estim_defaults,
)
from pmcprg.pmc.ice import _parse_ice_cfg
from pmcprg.pmc.sem import _parse_sem_cfg

MODELS    = Path(__file__).resolve().parents[1] / "pmc" / "models"
PMC_STATE = MODELS / "pmc_gauss_k2.toml"
PMC_PAIR  = MODELS / "pmc_pair_gauss_k2.toml"
HMC_IN    = MODELS / "hmc_in_gauss_k2.toml"

CANDS3 = ["Gauss", "GH", "Clayton"]


def _cfg(**kw) -> dict:
    cfg = {**shared_estim_defaults(), "multistart_seed": 4, "candidates": list(CANDS3)}
    cfg.update(kw)
    return cfg


def _copulas(model) -> dict:
    return {(int(b["i"]), int(b["j"])): dict(b) for b in model.copula_blocks()}


def _without_copulas(model) -> dict:
    raw = model.raw
    raw.pop("copulas", None)
    return raw


@pytest.fixture(scope="module")
def pmc():
    return PMCModel(PMC_STATE)


@pytest.fixture(scope="module")
def data(pmc):
    _, Y = simulate(pmc, N=200, seed=21)
    return Y


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

def test_default_is_none_for_ice_and_sem(pmc):
    assert MULTISTART_FAMILY_MODES == ("none", "random", "sweep")
    assert shared_estim_defaults()["multistart_families"] == "none"
    assert sem_estim_defaults()["multistart_families"] == "none"
    assert _parse_ice_cfg(pmc, None)["multistart_families"] == "none"
    assert _parse_sem_cfg(pmc, None)["multistart_families"] == "none"


def test_toml_key_is_read(pmc):
    raw = pmc.raw
    raw["ice"] = {**raw.get("ice", {}), "multistart_families": "sweep"}
    raw["sem"] = {"multistart_families": "random"}
    mdl = PMCModel.from_dict(raw)
    assert _parse_ice_cfg(mdl, None)["multistart_families"] == "sweep"
    assert _parse_sem_cfg(mdl, None)["multistart_families"] == "random"


@pytest.mark.parametrize("bad", ["Sweep", "all", "", None, 1])
def test_unknown_mode_is_refused(pmc, data, bad):
    with pytest.raises(ValueError, match="multistart_families"):
        _parse_ice_cfg(pmc, {"multistart_families": bad})
    with pytest.raises(ValueError, match="multistart_families"):
        _parse_sem_cfg(pmc, {"multistart_families": bad})
    with pytest.raises(ValueError, match="multistart_families"):
        ice(pmc, data, ice_cfg={"multistart_families": bad, "max_iter": 1})


# ---------------------------------------------------------------------------
# "none" — the historical starts, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [PMC_STATE, PMC_PAIR, HMC_IN])
def test_none_reproduces_the_historical_starts(path):
    """Same labels and same raw dicts as the former driver's loop."""
    mdl = PMCModel(path)
    cfg = _cfg(n_starts=4, multistart_jitter=0.2)
    rng = np.random.default_rng(cfg["multistart_seed"])
    legacy = [(mdl, "unperturbed")] + [
        (perturb_initial_model(mdl, rng, jitter=0.2), f"perturbed-{s}") for s in range(1, 4)
    ]
    got = build_multistart_inits(mdl, cfg)
    assert [t for _, t in got] == [t for _, t in legacy]
    assert got[0][0] is mdl
    for (m_got, _), (m_old, _) in zip(got, legacy):
        assert m_got.raw == m_old.raw


@pytest.mark.parametrize("algo", ["ice", "sem"])
@pytest.mark.parametrize("n_starts", [1, 3])
def test_explicit_none_equals_default(pmc, data, algo, n_starts):
    base = {"max_iter": 2, "n_starts": n_starts, "multistart_seed": 2, "candidates": CANDS3}
    fn, key = (ice, "ice_cfg") if algo == "ice" else (sem, "sem_cfg")
    f_a, t_a = fn(pmc, data, **{key: base})
    f_b, t_b = fn(pmc, data, **{key: {**base, "multistart_families": "none"}})
    assert t_a.log_liks == t_b.log_liks and t_a.run_tag == t_b.run_tag
    assert [(t.run_tag, t.log_liks) for t in t_a.multistart_runs] == \
           [(t.run_tag, t.log_liks) for t in t_b.multistart_runs]
    assert f_a.raw == f_b.raw
    tags = {t.run_tag for t in (t_b, *t_b.multistart_runs)}
    assert tags == ({""} if n_starts == 1 else {"unperturbed", "perturbed-1", "perturbed-2"})


# ---------------------------------------------------------------------------
# "random"
# ---------------------------------------------------------------------------

def test_random_draws_only_candidate_families(pmc):
    cands = ["Clayton", "Frank", "FGM"]
    inits = build_multistart_inits(pmc, _cfg(n_starts=15, multistart_families="random",
                                             candidates=cands))
    assert len(inits) == 15
    assert inits[0] == (pmc, "unperturbed")
    assert [t for _, t in inits[1:]] == [f"family-random-{s}" for s in range(1, 15)]
    seen = set()
    for mdl, _ in inits[1:]:
        for blk in mdl.copula_blocks():
            entry = CopulaEnum.from_short_name(blk["name"])
            assert blk["name"] in cands
            seen.add(blk["name"])
            lo, hi = entry.value.TAU_MIN_MAX
            span = hi - lo
            assert lo + RANDOM_TAU_BAND * span - 1e-12 <= blk["tau"] <= hi - RANDOM_TAU_BAND * span + 1e-12
    assert seen == set(cands)          # 56 draws among 3 families


def test_random_is_reproducible_and_seed_dependent(pmc):
    def families(seed):
        inits = build_multistart_inits(pmc, _cfg(n_starts=6, multistart_families="random",
                                                 multistart_seed=seed))
        return [m.raw for m, _ in inits]
    assert families(11) == families(11)
    a, b = families(11), families(12)
    assert [r["copulas"] for r in a[1:]] != [r["copulas"] for r in b[1:]]


def test_random_keeps_the_jitter_of_none(pmc):
    """Family draws use their own stream: prior and margins match "none"."""
    none = build_multistart_inits(pmc, _cfg(n_starts=5))
    rand = build_multistart_inits(pmc, _cfg(n_starts=5, multistart_families="random"))
    for (m_n, _), (m_r, _) in zip(none, rand):
        assert _without_copulas(m_n) == _without_copulas(m_r)


def test_family_change_resets_extra_parameters(pmc):
    inits = build_multistart_inits(pmc, _cfg(n_starts=12, multistart_families="random",
                                             candidates=["Student", "Gauss"]))
    names = set()
    for mdl, _ in inits[1:]:
        for blk in mdl.copula_blocks():
            names.add(blk["name"])
            if blk["name"] == "Student":
                assert blk["df"] == 4.0
            else:
                assert "df" not in blk
    assert names == {"Student", "Gauss"}


# ---------------------------------------------------------------------------
# "sweep"
# ---------------------------------------------------------------------------

def test_sweep_order_and_content(pmc):
    inits = build_multistart_inits(pmc, _cfg(n_starts=10, multistart_families="sweep"))
    combos = list(itertools.product(CANDS3, repeat=2))
    assert [t for _, t in inits] == ["unperturbed"] + [f"family-sweep:{a}/{b}" for a, b in combos]
    assert inits[0][0] is pmc
    ref = _copulas(pmc)
    for (mdl, _), (a, b) in zip(inits[1:], combos):
        cop = _copulas(mdl)
        assert (cop[(0, 0)]["name"], cop[(1, 1)]["name"]) == (a, b)
        assert cop[(0, 0)]["tau"] == cop[(1, 1)]["tau"] == SWEEP_TAU
        assert cop[(0, 1)] == ref[(0, 1)] and cop[(1, 0)] == ref[(1, 0)]
        assert _without_copulas(mdl) == _without_copulas(pmc)     # no jitter


def test_sweep_clips_tau_to_the_family_range(pmc):
    inits = build_multistart_inits(pmc, _cfg(n_starts=5, multistart_families="sweep",
                                             candidates=["FGM", "Gauss"]))
    fgm = _copulas(inits[1][0])[(0, 0)]
    assert fgm["name"] == "FGM" and fgm["tau"] == pytest.approx(2.0 / 9.0)


def test_sweep_truncation_warns(pmc, caplog):
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc._estim_common"):
        inits = build_multistart_inits(pmc, _cfg(n_starts=4, multistart_families="sweep"),
                                       label="ICE")
    assert [t for _, t in inits] == ["unperturbed", "family-sweep:Gauss/Gauss",
                                     "family-sweep:Gauss/GH", "family-sweep:Gauss/Clayton"]
    msg = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
    assert "n_starts = 1 + 3^2 = 10" in msg and "drops 6 combination(s)" in msg
    assert "family-sweep:Clayton/Clayton" in msg


def test_sweep_with_a_single_start_warns_and_runs_the_model(pmc, data, caplog):
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc._estim_common"):
        _, trace = ice(pmc, data, ice_cfg={"max_iter": 1, "candidates": CANDS3,
                                           "multistart_families": "sweep"})
    assert trace.run_tag == "" and trace.multistart_runs == []
    assert any("drops 9 combination(s)" in r.getMessage() for r in caplog.records)


def test_sweep_fills_with_random_starts(pmc):
    inits = build_multistart_inits(pmc, _cfg(n_starts=13, multistart_families="sweep"))
    tags = [t for _, t in inits]
    assert tags[10:] == ["family-random-10", "family-random-11", "family-random-12"]
    assert all(t.startswith("family-sweep:") for t in tags[1:10])
    for mdl, _ in inits[10:]:
        assert {b["name"] for b in mdl.copula_blocks()} <= set(CANDS3)
    # Deterministic, sweep part included.
    again = build_multistart_inits(pmc, _cfg(n_starts=13, multistart_families="sweep"))
    assert [m.raw for m, _ in inits] == [m.raw for m, _ in again]


def test_sweep_refuses_too_many_combinations(pmc, data):
    every = [c.value.SHORT_NAME for c in CopulaEnum.available()]
    assert len(every) ** 2 > MAX_SWEEP_COMBINATIONS
    with pytest.raises(ValueError, match="random"):
        build_multistart_inits(pmc, _cfg(n_starts=2, multistart_families="sweep",
                                         candidates=every))
    with pytest.raises(ValueError, match="random"):
        ice(pmc, data, ice_cfg={"max_iter": 1, "n_starts": 3, "candidates": every,
                                "multistart_families": "sweep"})
    # "random" accepts the same candidate list.
    assert len(build_multistart_inits(pmc, _cfg(n_starts=3, multistart_families="random",
                                                candidates=every))) == 3


def test_sweep_candidates_are_deduplicated(pmc):
    inits = build_multistart_inits(pmc, _cfg(n_starts=20, multistart_families="sweep",
                                             candidates=["Gauss", "Clayton", "Gauss"]))
    assert sum(t.startswith("family-sweep:") for _, t in inits) == 4


# ---------------------------------------------------------------------------
# Fallback to "none"
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["random", "sweep"])
def test_variant_without_copulas_falls_back(mode, caplog):
    mdl = PMCModel(HMC_IN)
    with caplog.at_level(logging.INFO, logger="pmcprg.pmc._estim_common"):
        got = build_multistart_inits(mdl, _cfg(n_starts=3, multistart_families=mode))
    ref = build_multistart_inits(mdl, _cfg(n_starts=3))
    assert [(m.raw, t) for m, t in got] == [(m.raw, t) for m, t in ref]
    assert any("no copula" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("mode", ["random", "sweep"])
def test_empty_candidates_fall_back(pmc, mode, caplog):
    with caplog.at_level(logging.INFO, logger="pmcprg.pmc._estim_common"):
        got = build_multistart_inits(pmc, _cfg(n_starts=3, multistart_families=mode,
                                               candidates=[]))
    ref = build_multistart_inits(pmc, _cfg(n_starts=3, candidates=[]))
    assert [(m.raw, t) for m, t in got] == [(m.raw, t) for m, t in ref]
    assert any("no copula candidate" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Driver: winning label, pair margins, SEM, workers
# ---------------------------------------------------------------------------

def _all_runs(trace):
    return [trace, *trace.multistart_runs]


def test_ice_sweep_records_the_winning_label(pmc, data):
    fitted, trace = ice(pmc, data, ice_cfg={"max_iter": 3, "candidates": CANDS3,
                                            "n_starts": 10, "multistart_families": "sweep"})
    runs = _all_runs(trace)
    assert sorted(t.run_tag for t in runs) == sorted(
        ["unperturbed"] + [f"family-sweep:{a}/{b}" for a, b in itertools.product(CANDS3, repeat=2)])
    best = max(runs, key=lambda t: t.log_liks[-1])
    assert trace.run_tag == best.run_tag
    assert trace.log_liks[-1] == max(t.log_liks[-1] for t in runs)


def test_pair_margin_model_sweep_keeps_pair_structure():
    mdl = PMCModel(PMC_PAIR)
    _, Y = simulate(mdl, N=200, seed=3)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 2, "candidates": ["Clayton", "Frank"],
                                         "fit_margins": True, "n_starts": 5,
                                         "multistart_families": "sweep"})
    assert fitted.margin_structure == "pair" and len(fitted.margin_blocks()) == 4
    assert {t.run_tag for t in _all_runs(trace)} == {
        "unperturbed", "family-sweep:Clayton/Clayton", "family-sweep:Clayton/Frank",
        "family-sweep:Frank/Clayton", "family-sweep:Frank/Frank"}


def test_sem_random_runs_and_labels(pmc, data):
    _, trace = sem(pmc, data, sem_cfg={"max_iter": 2, "candidates": CANDS3, "n_starts": 3,
                                       "multistart_families": "random", "sem_seed": 1})
    assert {t.run_tag for t in _all_runs(trace)} == {
        "unperturbed", "family-random-1", "family-random-2"}


@pytest.mark.slow
@pytest.mark.parametrize("algo,mode", [("ice", "sweep"), ("sem", "random")])
def test_parallel_workers_match_sequential(pmc, data, algo, mode):
    cfg = {"max_iter": 2, "candidates": CANDS3, "n_starts": 4, "multistart_seed": 9,
           "multistart_families": mode, "sem_seed": 2}
    fn, key = (ice, "ice_cfg") if algo == "ice" else (sem, "sem_cfg")
    f_seq, t_seq = fn(pmc, data, **{key: {**cfg, "multistart_workers": 1}})
    f_par, t_par = fn(pmc, data, **{key: {**cfg, "multistart_workers": 2}})
    assert t_seq.run_tag == t_par.run_tag and t_seq.log_liks == t_par.log_liks
    assert sorted((t.run_tag, t.log_liks[-1]) for t in t_seq.multistart_runs) == \
           sorted((t.run_tag, t.log_liks[-1]) for t in t_par.multistart_runs)
    assert f_seq.raw == f_par.raw


# ---------------------------------------------------------------------------
# End to end — CSDA-2013 Exp. 3, setting 1 (a single start picks the wrong family)
# ---------------------------------------------------------------------------

# report/reproduce_csda2013.py::run_exp3, "exp1_orig_p": candidates Gauss, GH,
# Clayton; truth (0,0) Gauss τ=0.7, (0,1),(1,0) GH τ=0.4, (1,1) Clayton τ=0.7;
# Table 1 pair margins N(μ_ij, s_ij) with s read as a variance; N = 2500.
_EXP3_TRUTH = {(0, 0): ("Gauss", 0.7), (0, 1): ("GH", 0.4),
               (1, 0): ("GH", 0.4), (1, 1): ("Clayton", 0.7)}
_EXP3_MARGINS = {(0, 0): (0.0, 1.0), (0, 1): (0.3, 1.6), (1, 0): (1.1, 1.4), (1, 1): (1.5, 1.0)}
_EXP3_PRIOR = [[0.5, 0.05], [0.05, 0.4]]


def _exp3_model(layout: dict) -> PMCModel:
    return PMCModel.from_dict({
        "model": {"name": "exp3", "variant": "PMC", "K": 2, "N_default": 2500},
        "prior": {"p": _EXP3_PRIOR},
        "margins": [{"i": i, "j": j, "dist": "norm",
                     "params": {"loc": mu, "scale": float(np.sqrt(var))}}
                    for (i, j), (mu, var) in sorted(_EXP3_MARGINS.items())],
        "copulas": [{"i": i, "j": j, "name": name, "tau": tau}
                    for (i, j), (name, tau) in _EXP3_TRUTH.items()] if layout is None else
                   [{"i": i, "j": j, "name": name, "tau": tau}
                    for (i, j), (name, tau) in layout.items()],
    })


def test_exp3_model_matches_the_report_builder():
    """The layout above is the report's (guards against drift)."""
    rep = pytest.importorskip("report.reproduce_csda2013")
    truth = rep._build_pmc_model_per_pair(
        {(0, 0): ("c1", 0.7), (0, 1): ("c3", 0.4), (1, 0): ("c3", 0.4), (1, 1): ("c6", 0.7)},
        rep.MARGIN_SETS["pair"]["gauss"], _EXP3_PRIOR)
    mine = _exp3_model(None)
    assert mine.copula_blocks() == truth.copula_blocks()
    for a, b in zip(mine.margin_blocks(), truth.margin_blocks()):
        assert (a["i"], a["j"], a["dist"]) == (b["i"], b["j"], b["dist"])
        for k in ("loc", "scale"):
            assert a["params"][k] == pytest.approx(b["params"][k], rel=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exp3_sweep_recovers_the_diagonal_families(seed):
    truth = _exp3_model(None)
    start = _exp3_model({k: ("Gauss", 0.0) for k in _EXP3_TRUTH})   # first candidate, τ = 0
    _, Y = simulate(truth, N=2500, seed=seed)
    cfg = {"max_iter": 30, "candidates": CANDS3, "fit_margins": False,
           "selection_criterion": "mle"}

    single, t_single = ice(start, Y, ice_cfg=cfg)
    multi, t_multi = ice(start, Y, ice_cfg={**cfg, "n_starts": 10,
                                            "multistart_families": "sweep"})

    def diag(m):
        return (m.copula(0, 0).copula_enum.value.SHORT_NAME,
                m.copula(1, 1).copula_enum.value.SHORT_NAME)

    assert diag(single) != ("Gauss", "Clayton")          # the known failure
    assert diag(multi) == ("Gauss", "Clayton")
    assert t_multi.log_liks[-1] >= t_single.log_liks[-1]
    assert t_multi.run_tag.startswith("family-sweep:")
