"""Degenerate fitted states of ICE and SEM — ``degenerate_states`` (E2).

A state of negligible weight, a margin whose variance collapses (on an atom of
discretised data, typically) or a copula τ driven to ±1 make the likelihood
unbounded. The estimators check the model they return, store the findings in
``trace.degenerate`` and log one WARNING; nothing in the fit changes.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pmcprg.pmc import PMCModel, ice, sem, simulate
from pmcprg.pmc._estim_common import DegenerateFinding, degenerate_states

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "pmcprg" / "pmc" / "models"
PAIR = MODELS / "pmc_pair_gauss_k2.toml"
DIAG_LOGGER = "pmcprg.pmc._estim_common"


def _degenerate_state_model() -> PMCModel:
    """PMC state margins, K = 2: state 1 of weight 0.2 %, sd 1e-4, c_11 at τ = 0.9999."""
    raw = PMCModel(MODELS / "pmc_gauss_k2.toml").raw
    raw["prior"]["p"] = [[0.996, 0.002], [0.002, 0.0]]
    for blk in raw["margins"]:
        if int(blk["i"]) == 1:
            blk["params"]["scale"] = 1e-4
    for blk in raw["copulas"]:
        if (int(blk["i"]), int(blk["j"])) == (1, 1):
            blk["name"], blk["tau"] = "Clayton", 0.9999
    return PMCModel.from_dict(raw)


def test_constructed_degenerate_model_is_flagged_with_state_and_cause():
    mdl = _degenerate_state_model()
    _, Y = simulate(PMCModel(MODELS / "pmc_gauss_k2.toml"), N=500, seed=0)
    found = degenerate_states(mdl, Y)
    assert [(f.cause, f.i, f.j) for f in found] == [
        ("state_weight", 1, None), ("margin_sd", 1, None), ("copula_tau", 1, 1),
    ]
    weight, margin, tau = found
    assert weight.value == pytest.approx(0.002) and weight.threshold == 0.005
    assert margin.value == pytest.approx(1e-4)
    assert margin.threshold == pytest.approx(0.01 * np.std(Y))
    assert margin.family == "norm" and tau.family == "Clayton"
    assert str(weight).startswith("state 1: stationary weight")
    assert str(margin).startswith("margin f_1 (norm): sd = 0.0001 <")
    assert str(tau) == "copula c_11 (Clayton): τ = 0.9999 within 0.001 of +1"
    # Thresholds are keyword arguments.
    assert degenerate_states(mdl, Y, min_weight=0.001, sd_ratio=1e-5, tau_edge=1e-5) == []


def test_pair_margin_and_countermonotone_copula_are_flagged():
    raw = PMCModel(PAIR).raw
    for blk in raw["margins"]:
        if (int(blk["i"]), int(blk["j"])) == (0, 1):
            blk["params"]["scale"] = 2e-3
    for blk in raw["copulas"]:
        if (int(blk["i"]), int(blk["j"])) == (1, 0):
            blk["name"], blk["tau"] = "Gauss", -0.9995
    mdl = PMCModel.from_dict(raw)
    _, Y = simulate(PMCModel(PAIR), N=500, seed=1)
    Y[10:20] = np.nan                                  # ignored by the data sd
    found = degenerate_states(mdl, Y)
    assert [(f.cause, f.i, f.j) for f in found] == [("margin_sd", 0, 1), ("copula_tau", 1, 0)]
    assert found[0].where == "margin f_01" and found[0].data_sd == pytest.approx(np.nanstd(Y))
    assert str(found[1]).endswith("within 0.001 of -1")


def test_undefined_sd_uses_the_interquartile_range():
    raw = PMCModel(MODELS / "hmc_in_gauss_k2.toml").raw
    raw["margins"][0]["dist"], raw["margins"][0]["params"] = "cauchy", {"loc": 0.0, "scale": 1.0}
    Y = np.linspace(-3.0, 3.0, 101)
    assert degenerate_states(PMCModel.from_dict(raw), Y) == []
    raw["margins"][0]["params"]["scale"] = 1e-5
    found = degenerate_states(PMCModel.from_dict(raw), Y)
    assert [(f.cause, f.i) for f in found] == [("margin_sd", 0)]
    assert found[0].value == pytest.approx(1e-5 * 2.0 / 1.3489795003921634)


@pytest.mark.slow
@pytest.mark.parametrize("fixture", sorted(p.name for p in MODELS.glob("*.toml")))
def test_bundled_fixtures_are_not_flagged(fixture, caplog):
    """Every bundled model, as is and after a brief ICE and SEM fit from the truth."""
    truth = PMCModel(MODELS / fixture)
    _, Y = simulate(truth, N=600, seed=0)
    assert degenerate_states(truth, Y) == []
    with caplog.at_level(logging.WARNING, logger=DIAG_LOGGER):
        _, tr_ice = ice(truth, Y, ice_cfg={"max_iter": 3})
        _, tr_sem = sem(truth, Y, sem_cfg={"max_iter": 3})
    assert tr_ice.degenerate == [] and tr_sem.degenerate == []
    assert not [r for r in caplog.records if r.name == DIAG_LOGGER]


def _collapsing_start():
    """HMC-IN K = 2 whose state 1 starts 60 sd away from every observation."""
    truth = PMCModel(MODELS / "hmc_in_gauss_k2.toml")
    _, Y = simulate(truth, N=400, seed=0)
    raw = truth.raw
    for blk in raw["margins"]:
        if int(blk["i"]) == 1:
            blk["params"]["loc"] = 60.0
    return PMCModel.from_dict(raw), Y


@pytest.mark.parametrize("estimator", ["ice", "sem"])
def test_estimators_warn_once_and_store_the_findings(estimator, caplog):
    start, Y = _collapsing_start()
    run = ice if estimator == "ice" else sem
    with caplog.at_level(logging.WARNING, logger=DIAG_LOGGER):
        fitted, trace = run(start, Y, {"max_iter": 3})
    records = [r for r in caplog.records if r.name == DIAG_LOGGER]
    assert len(records) == 1 and records[0].levelno == logging.WARNING
    msg = records[0].getMessage()
    assert "\n" not in msg and msg.startswith(f"{estimator.upper()}: degenerate fitted model")
    assert "state 1: stationary weight" in msg
    assert trace.degenerate and all(isinstance(f, DegenerateFinding) for f in trace.degenerate)
    assert ("state_weight", 1) in [(f.cause, f.i) for f in trace.degenerate]
    assert trace.degenerate == degenerate_states(fitted, Y)


def test_the_check_changes_nothing_in_the_fit(monkeypatch):
    """Same fitted parameters and trace with the diagnostic replaced by a no-op."""
    import pmcprg.pmc._estim_common as ec

    start, Y = _collapsing_start()
    fit_a, tr_a = ice(start, Y, {"max_iter": 3})
    monkeypatch.setattr(ec, "report_degenerate_states", lambda *a, **k: [])
    fit_b, tr_b = ice(start, Y, {"max_iter": 3})
    assert fit_a.raw == fit_b.raw and tr_a.log_liks == tr_b.log_liks
    assert tr_a.degenerate and tr_b.degenerate is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, "-m", "pmcprg.pmc", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=120, env=env)


@pytest.mark.slow
def test_cli_prints_the_degenerate_warning_on_stderr(tmp_path: Path):
    degenerate, Y = _collapsing_start()
    degenerate.save(tmp_path / "degenerate.toml")
    with open(tmp_path / "hmc.csv", "w") as fh:
        fh.write("n,Y\n" + "".join(f"{n + 1},{float(y)!r}\n" for n, y in enumerate(Y)))
    res = _cli("estimate", "-m", str(tmp_path / "degenerate.toml"), "-d", str(tmp_path / "hmc.csv"),
               "--max-iter", "3", "--ref", "", "--out", str(tmp_path / "fit.toml"), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    lines = [ln for ln in res.stderr.splitlines() if "degenerate fitted model" in ln]
    assert len(lines) == 1 and "state 1: stationary weight" in lines[0], res.stderr
    assert "Degenerate    : state 1" in res.stdout
