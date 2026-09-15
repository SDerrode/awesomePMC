"""Multistart never builds a start the copula refuses: BB1's joint constraint (G1).

BB1 needs θ = 2/(δ(1 − τ)) − 2 > 0, i.e. δ < 1/(1 − τ). The multistart draws
move τ and δ each inside its own box — the jitter of ``perturb_initial_model``
and the family draws of ``multistart_families`` "random"/"sweep", which reset δ
to 1.5 (inadmissible for τ ≤ 1/3) or keep a jittered δ with a new τ — so a pair
could leave the admissible set and the start raised ``CopulaParameterError``:
51 of 200 seeds at jitter 0.1 and 143 at 0.25 on a BB1 model with τ = 0.5,
δ = 1.5, and e.g. every "random" start list of ``pmc_gauss_k2.toml`` with the
candidates Gauss and BB1.

``CopulaVirt.constructible_params`` now repairs such a pair — δ moved to
``1/(1 − τ)`` pulled in by ``max(TAU_PAD_REL · τ/(1 − τ), TAU_PAD_ABS)`` — and
returns every accepted pair untouched. No RNG draw is added, so a start that
built before is the same start.

The reference values below were produced by the code before the change (commit
0178bfa). τ involves only IEEE +, ×, clip and the NumPy ``Generator``: exact.
δ goes through ``np.exp``, whose last bit may differ between platforms: rtol
1e-12. A repaired δ is computed from τ alone: exact.
"""
from __future__ import annotations

import numpy as np
import pytest

from pmcprg.copulas import CopulaBB1, CopulaEnum, CopulaStudent
from pmcprg.copulas._base import TAU_PAD_ABS, TAU_PAD_REL
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS
from pmcprg.pmc._estim_common import build_multistart_inits, perturb_initial_model
from pmcprg.pmc.ice import EXTRA_PARAM_BOUNDS
from pmcprg.pmc.model import PMCModel

MODELS = "pmcprg/pmc/models"


def _model(blocks, fixture="pmc_gauss_k2.toml") -> PMCModel:
    raw = PMCModel(f"{MODELS}/{fixture}").raw
    for blk, new in zip(raw["copulas"], blocks):
        blk.pop("tau", None)
        blk.update(new)
    return PMCModel.from_dict(raw)


def _bb1(tau=0.5, delta=1.5) -> PMCModel:
    return _model([{"name": "BB1", "tau": tau, "delta": delta}] * 4)


def _refused(tau: float, delta: float) -> bool:
    try:
        CopulaBB1(tau_k=tau, delta=delta)
    except CopulaParameterError:
        return True
    return False


def _repaired_delta(tau: float) -> float:
    hi = 1.0 / (1.0 - tau)
    return max(1.0, hi - max(TAU_PAD_REL * (hi - 1.0), TAU_PAD_ABS))


# --------------------------------------------------------------------------
# The hook
# --------------------------------------------------------------------------

def test_accepted_pairs_are_returned_untouched():
    rng = np.random.default_rng(0)
    taus = np.concatenate([[EPS, 1e-9, 0.1, 1 / 3, 0.5, 0.9, 0.9999], rng.uniform(0, 1, 40)])
    n_accepted = 0
    for tau in taus:
        hi = 1.0 / (1.0 - tau)
        deltas = [1.0, 1.5, 3.0, 10.0, hi, np.nextafter(hi, 0.0), np.nextafter(hi, 2 * hi),
                  hi * (1 - 1e-15), float(rng.uniform(1.0, 10.0))]
        for delta in deltas:
            params = {"tau_k": float(tau), "delta": float(delta)}
            out = CopulaBB1.constructible_params(params)
            # The hook's test is the constructor's, to the last bit.
            assert (out is params) == (not _refused(float(tau), float(delta)))
            n_accepted += out is params
    assert n_accepted > 100
    for entry in CopulaEnum:
        if entry is not CopulaEnum.BB1:
            params = {"tau_k": 0.3, "df": 4.0}
            assert entry.klass.constructible_params(params) is params


@pytest.mark.parametrize("tau", [EPS, 1e-9, 1e-6, 0.05, 0.2, 1 / 3, 0.5, 0.8, 0.9, 0.9999])
@pytest.mark.parametrize("delta", [0.5, 1.5, 3.0, 10.0, None])
def test_refused_pairs_move_only_delta_just_inside(tau, delta):
    params = {"tau_k": tau} if delta is None else {"tau_k": tau, "delta": delta}
    if not _refused(tau, 1.5 if delta is None else delta) and (delta is None or delta >= 1.0):
        return
    out = CopulaBB1.constructible_params(params)
    assert out is not params and out["tau_k"] == tau
    cop = CopulaBB1(**out)
    assert cop.theta > 0.0 and 1.0 <= out["delta"] < 1.0 / (1.0 - tau)
    if delta is not None and delta < 1.0:
        assert out["delta"] == 1.0
    else:
        assert out["delta"] == _repaired_delta(tau)
        hi = 1.0 / (1.0 - tau)
        if out["delta"] > 1.0:
            # the refused end of [1, 1/(1 − τ)) pulled in as constructible_tau_range does
            pad = max(TAU_PAD_REL * (hi - 1.0), TAU_PAD_ABS)
            assert hi - out["delta"] == pytest.approx(pad, rel=1e-6)


# --------------------------------------------------------------------------
# Parameter jitter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("jitter,n_raised_before", [(0.1, 51), (0.25, 143)])
def test_bb1_jitter_reproducer_builds(jitter, n_raised_before):
    """Every seed builds; the seeds whose start was repaired are exactly as many
    as the seeds that raised before (a drawn δ never equals the repaired value)."""
    model = _bb1()
    repaired = 0
    for seed in range(200):
        out = perturb_initial_model(model, np.random.default_rng(seed), jitter=jitter)
        blocks = out.copula_blocks()
        for blk in blocks:
            assert not _refused(blk["tau"], blk["delta"])
        repaired += any(blk["delta"] == _repaired_delta(blk["tau"]) for blk in blocks)
    assert repaired == n_raised_before


# pmc all-BB1 (τ = 0.5, δ = 1.5), jitter 0.25: seed → [(τ, δ)] of the 4 blocks.
# Seeds 2 and 7 built before the change; seed 3 raised on blocks 0 and 3 — its
# values are the draws the code made before building.
_BUILT = {
    2: [(0.5703026674494123, 1.3060527218653077), (0.7443918627815089, 1.3879474087863275),
        (0.41779402398550936, 1.2305096329078584), (0.6137395178102139, 1.4632582029873105)],
    7: [(0.3769483703621676, 1.284470258498743), (0.6224605125462995, 1.639984614103972),
        (0.5263535622494746, 1.1886855263897507), (0.49268704438418165, 1.784772411821697)],
}
_RAISED = {
    3: [(0.28369673093126463, 3.442558645720224), (0.5564466533069804, 1.3734247108722921),
        (0.42967814546216243, 1.2692847103187146), (0.23621236219871972, 1.3603810764895685)],
}


@pytest.mark.parametrize("seed", sorted(_BUILT))
def test_a_start_that_built_is_unchanged(seed):
    out = perturb_initial_model(_bb1(), np.random.default_rng(seed), jitter=0.25)
    blocks = out.copula_blocks()
    assert [b["tau"] for b in blocks] == [t for t, _ in _BUILT[seed]]
    np.testing.assert_allclose([b["delta"] for b in blocks], [d for _, d in _BUILT[seed]],
                               rtol=1e-12, atol=0.0)


def test_a_start_that_raised_keeps_every_draw_but_the_refused_delta():
    out = perturb_initial_model(_bb1(), np.random.default_rng(3), jitter=0.25)
    blocks = out.copula_blocks()
    assert [b["tau"] for b in blocks] == [t for t, _ in _RAISED[3]]
    for blk, (tau, delta) in zip(blocks, _RAISED[3]):
        if _refused(tau, delta):
            assert blk["delta"] == _repaired_delta(tau) < delta
        else:
            assert blk["delta"] == pytest.approx(delta, rel=1e-12, abs=0.0)
    assert sum(_refused(t, d) for t, d in _RAISED[3]) == 2


def test_missing_delta_uses_the_default_and_is_written_when_repaired():
    model = _model([{"name": "BB1", "tau": 0.5}] * 4)
    n_written = 0
    for seed in range(40):
        for blk in perturb_initial_model(model, np.random.default_rng(seed),
                                         jitter=0.25).copula_blocks():
            if "delta" in blk:
                assert blk["delta"] == _repaired_delta(blk["tau"])
                n_written += 1
            else:
                assert not _refused(blk["tau"], 1.5)
    assert n_written > 0


@pytest.mark.parametrize("cls_name", sorted(EXTRA_PARAM_BOUNDS))
def test_every_multi_parameter_family_survives_strong_jitter(cls_name):
    entry = next(e for e in CopulaEnum if e.value.CLASS_NAME == cls_name)
    short = entry.value.SHORT_NAME
    lo, hi = entry.constructible_tau_range()
    for (key, (xlo, xhi, _)) in EXTRA_PARAM_BOUNDS[cls_name].items():
        for tau, x in ((hi, xlo), (hi, xhi), (0.5 * (lo + hi), xhi), (lo, xhi)):
            try:
                model = _model([{"name": short, "tau": tau, key: x}] * 4)
            except CopulaParameterError:
                continue            # not a valid user model to start from
            for seed in range(30):
                perturb_initial_model(model, np.random.default_rng(seed), jitter=0.5)


# --------------------------------------------------------------------------
# Family multistart
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["random", "sweep"])
@pytest.mark.parametrize("cands", [["Gauss", "BB1"], ["BB1", "Student", "Clayton", "Frank"]])
@pytest.mark.parametrize("start", ["gauss", "bb1_strong", "bb1_weak"])
def test_family_starts_with_bb1_candidates_build(mode, cands, start):
    model = {"gauss": lambda: PMCModel(f"{MODELS}/pmc_gauss_k2.toml"),
             "bb1_strong": lambda: _bb1(0.8, 3.0),
             "bb1_weak": lambda: _bb1(0.2, 1.1)}[start]()
    for seed in range(4):
        cfg = {"n_starts": 1 + len(cands) ** 2 + 4, "multistart_seed": seed,
               "multistart_jitter": 0.25, "multistart_families": mode, "candidates": cands}
        inits = build_multistart_inits(model, cfg, label="ICE")
        assert inits[0][0] is model and inits[0][1] == "unperturbed"
        for m, _ in inits:
            for blk in m.copula_blocks():
                if blk["name"] == "BB1":
                    assert not _refused(blk["tau"], blk.get("delta", 1.5))


def test_sweep_keeps_an_admissible_delta_and_repairs_a_refused_one():
    """Sweep τ = 0.5 on a diagonal BB1 pair whose family does not change keeps
    its δ = 3.0, which needs τ > 2/3 — it raised; now δ = 2 − 10⁻⁴."""
    model = _bb1(0.8, 3.0)
    cfg = {"n_starts": 3, "multistart_seed": 0, "multistart_jitter": 0.1,
           "multistart_families": "sweep", "candidates": ["BB1", "Clayton"]}
    inits = build_multistart_inits(model, cfg, label="ICE")
    tag = inits[1][1]
    diag = {(b["i"], b["j"]): b for b in inits[1][0].copula_blocks()}
    assert tag == "family-sweep:BB1/BB1"
    assert diag[(0, 0)]["tau"] == 0.5 and diag[(0, 0)]["delta"] == 2.0 - 1e-4
    assert diag[(0, 1)] == {"i": 0, "j": 1, "name": "BB1", "tau": 0.8, "delta": 3.0}
    # δ = 1.5 at τ = 0.5 was always admissible: a family change to BB1 keeps it.
    gauss = PMCModel(f"{MODELS}/pmc_gauss_k2.toml")
    cfg["candidates"] = ["Gauss", "BB1"]
    inits = build_multistart_inits(gauss, cfg, label="ICE")
    assert [b.get("delta") for b in inits[2][0].copula_blocks()] == [None, None, None, 1.5]


def test_student_family_draws_are_untouched():
    params = {"tau_k": 0.99, "df": 2.001}
    assert CopulaStudent.constructible_params(params) is params


@pytest.mark.parametrize("candidates", [["BB1", "Gauss"], ["Gauss", "BB1"], ["Student", "BB1"]])
def test_selection_placeholder_builds(candidates):
    # When no candidate has a finite score the M-step stores a placeholder block
    # (first candidate, τ pulled into its range). BB1 first used to store τ ≈ 0
    # without δ, which the model then refused to build (δ = 1.5 needs τ > 1/3).
    from pmcprg.pmc.ice import FIT_FAILED_KEY, _copula_candidate_fits, _copula_placeholder

    rng = np.random.default_rng(0)
    u, v = rng.uniform(size=50), rng.uniform(size=50)
    resolved, _ = _copula_candidate_fits(candidates, u, v, np.ones(50), "mle")
    blk = _copula_placeholder(candidates, resolved, "mle", {"name": candidates[0], "tau": 0.0})
    assert blk[FIT_FAILED_KEY] is True
    stored = {k: val for k, val in blk.items() if k != FIT_FAILED_KEY}
    if candidates[0] != "BB1":
        assert set(stored) == {"name", "tau"}          # unchanged for other families
    model = _model([stored])
    assert model.copula_blocks()[0]["name"] == candidates[0]
