"""Frank never produces or stores a τ beyond the τ it reaches (F1).

``CopulaFrank`` caps θ at ±700, so its τ stops at ±``_FRANK_TAU_MAX`` ≈ 0.994299
while the registry advertises (−1 + ε, 1 − ε). A τ in between builds the copula
at θ = ±700, logs a clamping WARNING and stores ±``_FRANK_TAU_MAX`` in the
copula — but the bounded τ searches (``CopulaVirt.fit``, the ICE/SEM M-step,
the independence LR test) and the clips into
``CopulaEnum.constructible_tau_range`` (``fit(method='tau')``, multistart
jitter) returned such a τ, and the model's raw TOML block kept it: a fitted
model then warned at every reload (real-series study, 57 of 266 fits).

``CopulaEnum.constructible_tau_range`` is now cut to the reachable τ for Frank,
and ``CopulaEnum.reachable_tau`` maps a τ beyond it to the bound, where the
density is the same. Every other family is unaffected (its constructible range
and ``reachable_tau`` are unchanged — identity).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum, CopulaFrank, CopulaGaussian, independence_lr_test
from pmcprg.copulas._base import constructible_tau_range
from pmcprg.copulas.archimedean.frank import (
    _FRANK_THETA_MAX,
    _FRANK_TAU_MAX,
    find_theta_frank,
    kendall_tau_frank,
)
from pmcprg.pmc._estim_common import perturb_initial_model
from pmcprg.pmc.ice import (
    _copula_block,
    _fit_copula_params,
    _select_and_fit_copula,
    ice,
)
from pmcprg.numerics import EPS_MINUS_ONE, ONE_MINUS_EPS
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

FRANK = CopulaEnum.FRANK
MODELS = "pmcprg/pmc/models"


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _strong(sign: int, n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-observations more dependent than Frank can model (Gaussian τ = ±0.998)."""
    uv = CopulaGaussian(tau_k=sign * 0.998).sample(n, seed=seed)
    return uv[:, 0], uv[:, 1]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def test_frank_constructible_range_is_the_reachable_range():
    assert FRANK.constructible_tau_range() == (-_FRANK_TAU_MAX, _FRANK_TAU_MAX)
    assert FRANK.value.TAU_MIN_MAX == [EPS_MINUS_ONE, ONE_MINUS_EPS]     # registry unchanged
    assert abs(kendall_tau_frank(_FRANK_THETA_MAX)) == _FRANK_TAU_MAX
    assert FRANK.reachable_tau(0.995) == _FRANK_TAU_MAX
    assert FRANK.reachable_tau(-0.9999) == -_FRANK_TAU_MAX
    assert FRANK.reachable_tau(0.9) == 0.9


# Plackett declares its own bounds since G2 (test_plackett_reachable_tau.py);
# Galambos declares its own since FR-9 (test_galambos.py) — θ is capped at 1e6,
# so τ = 1.0 (a registered bound refused by the constructor, RB-6, but still
# checked here) reaches only ≈ 1 − 1e-6, not itself. Hüsler-Reiss declares its
# own for the same reason since FR-9 round 2 (test_husler_reiss.py) — λ capped
# at 1e6, reaching τ ≈ 1 − 1.13e-6.
@pytest.mark.parametrize("entry", [e for e in CopulaEnum if e.value.AVAILABLE and e is not FRANK
                                   and e is not CopulaEnum.PLACKETT
                                   and e is not CopulaEnum.GALAMBOS
                                   and e is not CopulaEnum.HUSLER_REISS],
                         ids=lambda e: e.value.SHORT_NAME)
def test_other_families_are_unchanged(entry):
    assert entry.klass.reachable_tau_abs is None
    assert entry.constructible_tau_range() == constructible_tau_range(*entry.value.TAU_MIN_MAX)
    for tau in (*entry.value.TAU_MIN_MAX, 0.995, -0.9999, 2.0):
        assert entry.reachable_tau(tau) is tau


def test_tau_at_the_bound_builds_silently_and_beyond_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.archimedean.frank"):
        cop = CopulaFrank(tau_k=-_FRANK_TAU_MAX)
        # A few ulps above — the bound as another platform's quadrature rounds it.
        assert abs(find_theta_frank(_FRANK_TAU_MAX + 1e-15)) == _FRANK_THETA_MAX
    assert _warnings(caplog) == []
    assert cop.theta == -_FRANK_THETA_MAX and cop.params["tau_k"] == -_FRANK_TAU_MAX

    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.archimedean.frank"):
        CopulaFrank(tau_k=0.995)                    # genuine user input
    assert len(_warnings(caplog)) == 1


# --------------------------------------------------------------------------
# Standalone fits and the LR test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_returns_the_tau_it_uses(method, sign, caplog):
    u, v = _strong(sign)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        r = CopulaFrank.fit(np.column_stack((u, v)), method=method)
    assert _warnings(caplog) == []
    assert r.tau_k == sign * _FRANK_TAU_MAX
    assert r.copula.params["tau_k"] == r.tau_k
    assert r.copula.theta == sign * _FRANK_THETA_MAX


def test_independence_lr_test_reports_a_reachable_tau(caplog):
    u, v = _strong(1)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        r = independence_lr_test(CopulaFrank, np.column_stack((u, v)))
    assert _warnings(caplog) == []
    assert r.tau_k == _FRANK_TAU_MAX


# --------------------------------------------------------------------------
# ICE / SEM M-step
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sign", [1, -1])
def test_m_step_fit_is_reachable_and_silent(sign, caplog):
    u, v = _strong(sign)
    w = np.random.default_rng(0).uniform(0.2, 1.0, u.size)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        params = _fit_copula_params(CopulaFrank, FRANK, u, v, w)
    assert _warnings(caplog) == []
    assert params == {"tau_k": sign * _FRANK_TAU_MAX}


@pytest.mark.parametrize("criterion", ["mle", "aic", "huard", "xvcic"])
def test_selected_block_is_reachable_and_silent(criterion, caplog):
    u, v = _strong(-1)
    w = np.ones(u.size)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        blk = _select_and_fit_copula(["Frank"], u, v, w, criterion=criterion)
    assert _warnings(caplog) == []
    assert blk == {"name": "Frank", "tau": -_FRANK_TAU_MAX}


def test_averaged_tau_is_stored_at_the_bound():
    """Multiple imputation averages the D fitted τ: a mean of values at the
    bound may exceed it by a few ulps (np.mean of 5 copies does)."""
    over = float(np.mean([_FRANK_TAU_MAX] * 5))
    assert over > _FRANK_TAU_MAX
    assert _copula_block("Frank", {"tau_k": over})["tau"] == _FRANK_TAU_MAX
    gauss = {"tau_k": 0.9997}
    assert _copula_block("Gauss", gauss)["tau"] is gauss["tau_k"]


@pytest.fixture(scope="module")
def fitted_strong_frank():
    """ICE with a Frank-only candidate set on data more dependent than Frank."""
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    truth_taus = [0.998, 0.1, 0.1, -0.998]
    start_taus = [0.9, 0.1, 0.1, -0.9]
    for blk, t in zip(raw["copulas"], truth_taus):
        blk["tau"] = t
    _, Y = simulate(PMCModel.from_dict(raw), N=600, seed=7)
    for blk, t in zip(raw["copulas"], start_taus):
        blk["name"], blk["tau"] = "Frank", t
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Keep(level=logging.WARNING)
    log = logging.getLogger("pmcprg")
    log.addHandler(handler)
    try:
        fitted, trace = ice(PMCModel.from_dict(raw), Y,
                            ice_cfg={"candidates": ["Frank"], "max_iter": 4})
    finally:
        log.removeHandler(handler)
    return fitted, trace, records


def test_ice_stores_the_tau_it_uses(fitted_strong_frank):
    fitted, trace, records = fitted_strong_frank
    assert [r.getMessage() for r in records if "Frank" in r.getMessage()] == []
    diag = {(b["i"], b["j"]): b["tau"] for b in fitted.copula_blocks()}
    assert diag[(0, 0)] == _FRANK_TAU_MAX and diag[(1, 1)] == -_FRANK_TAU_MAX
    for blk in fitted.copula_blocks():
        cop = fitted.copula(blk["i"], blk["j"])
        assert cop.params["tau_k"] == blk["tau"]
        assert kendall_tau_frank(cop.theta) == pytest.approx(blk["tau"], abs=1e-12)
    assert np.nanmax(np.abs(trace.tau_history)) <= _FRANK_TAU_MAX


def test_saved_fit_reloads_without_warning(fitted_strong_frank, tmp_path, caplog):
    fitted, _, _ = fitted_strong_frank
    path = fitted.save(tmp_path / "strong_frank.toml")
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        reloaded = PMCModel(path)
    assert _warnings(caplog) == []
    assert reloaded.copula_blocks() == fitted.copula_blocks()
    assert reloaded.copula(0, 0).theta == _FRANK_THETA_MAX


def test_jitter_keeps_frank_tau_reachable(tmp_path, caplog):
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    for blk, t in zip(raw["copulas"], [0.99, -0.99, 0.2, 0.9]):
        blk["name"], blk["tau"] = "Frank", t
    model = PMCModel.from_dict(raw)
    hits = 0
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        for seed in range(10):
            out = perturb_initial_model(model, np.random.default_rng(seed), jitter=0.25)
            taus = [blk["tau"] for blk in out.copula_blocks()]
            assert max(abs(t) for t in taus) <= _FRANK_TAU_MAX
            hits += sum(abs(t) == _FRANK_TAU_MAX for t in taus)
            PMCModel(out.save(tmp_path / f"jitter_{seed}.toml"))
    assert hits > 0
    assert _warnings(caplog) == []
