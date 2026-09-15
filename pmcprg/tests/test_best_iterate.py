"""Best iterate of ICE and SEM — config key ``return_best_iterate`` (E1).

ICE is not monotone, and after ``patience`` regressions it used to stop below
its best iterate (55 of 216 ICE fits of the real-series study, by up to 50
nats). With the option the estimators return θ^q for the first q maximising
``trace.log_liks`` — ``log_liks[q]`` being computed with θ^q before the M-step
— and multistart ranks the starts by the log-likelihood of the model each one
returns. The default is unchanged (``test_estim_complete_data_identity.py``).

The regression case below (pair-margin PMC, seed 5, N = 200, hand-made start)
was found by a search over fixtures, seeds and starts: from θ^1 every ICE
M-step lowers the log-likelihood and ICE stops after three regressions, 4.6
nats below its iterate 1.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pmcprg.pmc import PMCModel, ice, sem, simulate
from pmcprg.pmc._estim_common import IceTrace, best_iter_of, run_multistart
from pmcprg.pmc.inference import forward
from pmcprg.pmc.sem import _parse_sem_cfg

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "pmcprg" / "pmc" / "models"
PAIR = MODELS / "pmc_pair_gauss_k2.toml"


def _log_lik(model, Y) -> float:
    """Log-likelihood of ``model`` (observed-data one when Y has NaN rows)."""
    return float(forward(model, Y)[1])


def _regression_case(*, missing: bool = False):
    """(start, Y) on which ICE stops after three regressions below its best iterate."""
    truth = PMCModel(PAIR)
    _, Y = simulate(truth, N=200, seed=5)
    if missing:
        rng = np.random.default_rng(5)
        Y = Y.copy()
        for s in rng.choice(200 - 10, size=200 // 40, replace=False):
            Y[s:s + 5] = np.nan
    raw = truth.raw
    for blk in raw["margins"]:
        blk["params"]["loc"] += -0.5 if int(blk["i"]) == 0 else 0.5
        blk["params"]["scale"] *= 0.7
    for blk in raw["copulas"]:
        blk["tau"] *= 0.8
    return PMCModel.from_dict(raw), Y


def _assert_is_snapshot(model, trace, q: int) -> None:
    """``model`` carries the parameters recorded at iterate q of ``trace``."""
    np.testing.assert_array_equal(model.prior_p, trace.p_history[q])
    assert model.margin_blocks() == trace.margin_history[q]
    tau = np.full((model.K, model.K), np.nan)
    for blk in model.copula_blocks():
        tau[int(blk["i"]), int(blk["j"])] = blk["tau"]
    np.testing.assert_array_equal(tau, trace.tau_history[q])


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "nan"])
def test_ice_returns_the_argmax_with_the_option_and_the_last_iterate_by_default(missing):
    start, Y = _regression_case(missing=missing)
    fit0, tr0 = ice(start, Y, ice_cfg={"max_iter": 12})
    fit1, tr1 = ice(start, Y, ice_cfg={"max_iter": 12, "return_best_iterate": True})
    lls = np.asarray(tr0.log_liks)
    T = len(lls)
    # The case: stopped early (patience), more than 1 nat below its best iterate.
    assert T < 12 and lls.max() - lls[-1] > 1.0, lls

    # Default: the last iterate, evaluated (early stop).
    assert tr0.best_iter == int(np.argmax(lls)) < T - 1
    assert tr0.returned_iter == T - 1
    assert _log_lik(fit0, Y) == pytest.approx(lls[-1], rel=1e-12, abs=1e-9)
    _assert_is_snapshot(fit0, tr0, T - 1)

    # Option: same run, same trace, the argmax returned.
    assert tr1.log_liks == tr0.log_liks
    assert tr1.best_iter == tr1.returned_iter == tr0.best_iter
    assert _log_lik(fit1, Y) == pytest.approx(lls.max(), rel=1e-12, abs=1e-9)
    _assert_is_snapshot(fit1, tr1, tr1.best_iter)
    assert _log_lik(fit1, Y) > _log_lik(fit0, Y) + 1.0


def test_option_evaluates_the_last_m_step_when_max_iter_is_reached():
    """log_liks[q] is θ^q: with max_iter = m the default returns θ^m, never evaluated.

    The option evaluates θ^m once more (one more log-likelihood, no M-step), so
    it is a candidate: with max_iter = 1 the fit is not the initial model.
    """
    start, Y = _regression_case()
    for m in (1, 2, 3):
        fit0, tr0 = ice(start, Y, ice_cfg={"max_iter": m})
        fit1, tr1 = ice(start, Y, ice_cfg={"max_iter": m, "return_best_iterate": True})
        assert len(tr0.log_liks) == m and tr0.returned_iter == m
        assert len(tr1.log_liks) == m + 1
        assert tr1.log_liks[:m] == tr0.log_liks
        # The extra entry is the log-likelihood of the default's returned model.
        assert tr1.log_liks[m] == pytest.approx(_log_lik(fit0, Y), rel=1e-12)
        assert len(tr1.p_history) == len(tr1.margin_history) == m + 1
        q = tr1.best_iter
        assert q == tr1.returned_iter == best_iter_of(tr1.log_liks) == 1
        assert _log_lik(fit1, Y) == pytest.approx(tr1.log_liks[q], rel=1e-12)
        _assert_is_snapshot(fit1, tr1, q)
    # θ^1 of every run above is the same parameter set.
    assert _log_lik(fit1, Y) == pytest.approx(_log_lik(ice(start, Y, {"max_iter": 1})[0], Y),
                                              rel=1e-12)


def _fake_run(lls_by_start, *, option):
    def single_run(model, tag, s):
        lls = list(lls_by_start[s])
        tr = IceTrace(log_liks=lls, run_tag=tag, best_iter=best_iter_of(lls))
        tr.returned_iter = tr.best_iter if option else len(lls) - 1
        return f"model-{s}", tr
    return single_run


@pytest.mark.parametrize("option,winner", [(False, 1), (True, 0)])
def test_multistart_ranks_starts_by_the_log_lik_of_the_returned_model(option, winner):
    """Start 0 ends lower (−9 vs −8) but its best iterate is higher (−5 vs −7)."""
    lls = {0: [-10.0, -5.0, -9.0], 1: [-10.0, -7.0, -8.0]}
    cfg = {"n_starts": 2, "multistart_seed": 0, "multistart_jitter": 0.1,
           "multistart_workers": 1, "multistart_families": "none",
           "return_best_iterate": option}
    fitted, trace = run_multistart(_fake_run(lls, option=option), PMCModel(PAIR), cfg,
                                   label="TEST")
    assert fitted == f"model-{winner}"
    assert [t.log_liks for t in trace.multistart_runs] == [lls[1 - winner]]


@pytest.mark.slow
def test_ice_multistart_with_the_option_keeps_the_best_returned_model():
    start, Y = _regression_case()
    cfg = {"max_iter": 12, "n_starts": 3, "multistart_seed": 2, "return_best_iterate": True}
    fit, tr = ice(start, Y, ice_cfg=cfg)
    returned = [t.log_liks[t.returned_iter] for t in [tr, *tr.multistart_runs]]
    assert len(returned) == 3
    for t in [tr, *tr.multistart_runs]:
        assert t.returned_iter == t.best_iter == best_iter_of(t.log_liks)
    assert returned[0] == max(returned)
    assert _log_lik(fit, Y) == pytest.approx(returned[0], rel=1e-12)


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "nan"])
def test_sem_best_iterate(missing):
    start, Y = _regression_case(missing=missing)
    m = 8
    cfg = {"max_iter": m, "sem_seed": 3}
    fit0, tr0 = sem(start, Y, sem_cfg=cfg)
    fit1, tr1 = sem(start, Y, sem_cfg={**cfg, "return_best_iterate": True})
    # Default: θ^m, returned without evaluation.
    assert len(tr0.log_liks) == m and tr0.returned_iter == m
    # Option: the same chain (the final evaluation draws nothing), θ^m evaluated.
    assert tr1.log_liks[:m] == tr0.log_liks
    np.testing.assert_array_equal(tr1.sampled_X_history, tr0.sampled_X_history)
    assert len(tr1.log_liks) == m + 1 and tr1.sampled_X_history.shape[0] == m
    assert tr1.log_liks[m] == pytest.approx(_log_lik(fit0, Y), rel=1e-12)
    q = tr1.best_iter
    assert q == tr1.returned_iter == int(np.argmax(tr1.log_liks))
    assert q < m, "the case needs an SEM chain whose best iterate is not the last"
    assert _log_lik(fit1, Y) == pytest.approx(tr1.log_liks[q], rel=1e-12)
    _assert_is_snapshot(fit1, tr1, q)


def test_return_best_iterate_config_key():
    from pmcprg.pmc._estim_common import ice_estim_defaults, sem_estim_defaults
    from pmcprg.pmc.ice import _parse_ice_cfg

    assert ice_estim_defaults()["return_best_iterate"] is False
    assert sem_estim_defaults()["return_best_iterate"] is False
    raw = PMCModel(PAIR).raw
    raw["ice"] = {"return_best_iterate": True}
    mdl = PMCModel.from_dict(raw)
    assert _parse_ice_cfg(mdl, None)["return_best_iterate"] is True
    assert _parse_sem_cfg(mdl, None)["return_best_iterate"] is True      # [ice] fallback
    raw["sem"] = {"return_best_iterate": False}
    assert _parse_sem_cfg(PMCModel.from_dict(raw), None)["return_best_iterate"] is False
    for bad in ("false", "yes", 2, 0.5):
        with pytest.raises(ValueError, match="return_best_iterate"):
            _parse_ice_cfg(mdl, {"return_best_iterate": bad})
        with pytest.raises(ValueError, match="return_best_iterate"):
            _parse_sem_cfg(mdl, {"return_best_iterate": bad})


def test_best_iter_ranks_nan_lowest_and_keeps_the_first_maximum():
    assert best_iter_of([]) == -1
    assert best_iter_of([float("nan"), -3.0, -1.0, -1.0, -2.0]) == 2
    assert best_iter_of([float("nan"), float("nan")]) == 0


# ---------------------------------------------------------------------------
# CLI and GUI
# ---------------------------------------------------------------------------

def _cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), env.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, "-m", "pmcprg.pmc", *args], cwd=cwd,
                          capture_output=True, text=True, timeout=120, env=env)


@pytest.mark.slow
def test_cli_best_iterate_flag(tmp_path: Path):
    start, _ = _regression_case()
    start.save(tmp_path / "start.toml")
    res = _cli("simulate", "-m", str(PAIR), "--N", "200", "--seed", "5",
               "--out", str(tmp_path / "sim.csv"), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    common = ("estimate", "-m", str(tmp_path / "start.toml"), "-d", str(tmp_path / "sim.csv"),
              "--max-iter", "12")
    res = _cli(*common, "--out", str(tmp_path / "last.toml"), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "Returned iter" not in res.stdout
    res = _cli(*common, "--best-iterate", "--out", str(tmp_path / "best.toml"), cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "Returned iter : 1  (best iterate" in res.stdout


@pytest.fixture(scope="session")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app                       # held for the session, as in test_gui_dialogs


def test_gui_tab_round_trips_return_best_iterate(qapp):
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    assert tab.get_cfg()["return_best_iterate"] is False
    tab.load({"return_best_iterate": True})
    assert tab.get_cfg()["return_best_iterate"] is True
    tab.load({"return_best_iterate": "yes"})          # not a boolean → the default
    assert tab.get_cfg()["return_best_iterate"] is False
