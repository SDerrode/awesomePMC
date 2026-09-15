"""Plackett never produces or stores a τ beyond the τ it reaches (G2).

``CopulaPlackett`` inverts τ(θ) on a table whose θ stops at 10^{±6}, so its τ
stops at the table ends τ(10⁻⁶) = −0.9935245713002141 and τ(10⁶) =
0.9935245713002134 — not exact opposites — while the registry advertises
(−1 + ε, 1 − ε). A τ in between builds the copula at the capped θ, logs a
clamping WARNING and stores the table end in the copula; but the τ searches
and clips returned the unreachable value and the model's raw block kept it, so
a fitted model warned at every reload. The mechanism of F1 (Frank) is
generalised: a family declares ``reachable_tau_bounds() -> (lo, hi)``; Frank's
default is ``±reachable_tau_abs``, Plackett's the table ends.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from pmcprg.copulas import CopulaEnum, CopulaGaussian, CopulaPlackett, independence_lr_test
from pmcprg.copulas.archimedean.frank import _FRANK_TAU_MAX
from pmcprg.copulas.explicit.plackett import (
    _PLACKETT_THETA_MAX,
    _plackett_tau_from_theta,
    _plackett_tau_table,
)
from pmcprg.numerics import EPS_MINUS_ONE, ONE_MINUS_EPS
from pmcprg.pmc._estim_common import build_multistart_inits, perturb_initial_model
from pmcprg.pmc.ice import (
    _copula_block,
    _fit_copula_params,
    _select_and_fit_copula,
    ice,
)
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

PLACKETT = CopulaEnum.PLACKETT
MODELS = "pmcprg/pmc/models"
LO, HI = (float(t) for t in (_plackett_tau_table()[0][0], _plackett_tau_table()[0][-1]))


def _warnings(caplog):
    return [r for r in caplog.records if r.levelno >= logging.WARNING]


def _strong(sign: int, n: int = 400, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Pseudo-observations more dependent than Plackett can model (Gaussian τ = ±0.998)."""
    uv = CopulaGaussian(tau_k=sign * 0.998).sample(n, seed=seed)
    return uv[:, 0], uv[:, 1]


def _bound(sign: int) -> float:
    return HI if sign > 0 else LO


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def test_plackett_constructible_range_is_the_table_range():
    # Ends of Plackett's numerically computed τ table: their last bits depend on
    # the SciPy version (−0.9935245713002137 with SciPy 1.10), hence rtol.
    np.testing.assert_allclose((LO, HI), (-0.9935245713002141, 0.9935245713002134),
                               rtol=1e-12, atol=0.0)
    assert CopulaPlackett.reachable_tau_bounds() == (LO, HI)
    assert CopulaPlackett.reachable_tau_abs is None
    assert PLACKETT.constructible_tau_range() == (LO, HI)
    assert PLACKETT.value.TAU_MIN_MAX == [EPS_MINUS_ONE, ONE_MINUS_EPS]   # registry unchanged
    assert PLACKETT.reachable_tau(0.995) == HI and PLACKETT.reachable_tau(-0.9999) == LO
    assert PLACKETT.reachable_tau(float(np.nextafter(HI, 1.0))) == HI
    for tau in (0.9, LO, HI, -0.99352, float("nan")):
        assert PLACKETT.reachable_tau(tau) is tau


def test_frank_bounds_are_the_symmetric_default():
    frank = CopulaEnum.FRANK
    assert frank.klass.reachable_tau_bounds() == (-_FRANK_TAU_MAX, _FRANK_TAU_MAX)
    assert frank.reachable_tau(-0.9999) == -_FRANK_TAU_MAX
    assert type(frank.reachable_tau(np.float64(0.9999))) is float


def test_tau_at_the_bounds_builds_silently_and_beyond_still_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.explicit.plackett"):
        cops = [CopulaPlackett(tau_k=LO), CopulaPlackett(tau_k=HI)]
        # A few ulps beyond — the ends as another platform's quadrature rounds them.
        near = [CopulaPlackett(tau_k=LO - 1e-15), CopulaPlackett(tau_k=HI + 1e-15)]
    assert _warnings(caplog) == []
    assert [c.params["tau_k"] for c in cops + near] == [LO, HI, LO, HI]
    assert cops[1].theta == near[1].theta == pytest.approx(_PLACKETT_THETA_MAX, rel=1e-14)
    assert cops[0].theta == near[0].theta == pytest.approx(1.0 / _PLACKETT_THETA_MAX, rel=1e-14)

    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas.explicit.plackett"):
        cop = CopulaPlackett(tau_k=0.995)                   # genuine user input
    assert len(_warnings(caplog)) == 1
    assert cop.params["tau_k"] == HI


# --------------------------------------------------------------------------
# Standalone fits and the LR test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_returns_the_tau_it_uses(method, sign, caplog):
    u, v = _strong(sign)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        r = CopulaPlackett.fit(np.column_stack((u, v)), method=method)
    assert _warnings(caplog) == []
    assert r.tau_k == _bound(sign)
    assert r.copula.params["tau_k"] == r.tau_k
    assert _plackett_tau_from_theta(r.copula.theta) == r.tau_k


@pytest.mark.parametrize("sign", [1, -1])
def test_independence_lr_test_reports_a_reachable_tau(sign, caplog):
    u, v = _strong(sign)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        r = independence_lr_test(CopulaPlackett, np.column_stack((u, v)))
    assert _warnings(caplog) == []
    assert r.tau_k == _bound(sign)


# --------------------------------------------------------------------------
# ICE / SEM M-step
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sign", [1, -1])
def test_m_step_fit_is_reachable_and_silent(sign, caplog):
    u, v = _strong(sign)
    w = np.random.default_rng(0).uniform(0.2, 1.0, u.size)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        params = _fit_copula_params(CopulaPlackett, PLACKETT, u, v, w)
    assert _warnings(caplog) == []
    assert params == {"tau_k": _bound(sign)}


@pytest.mark.parametrize("criterion", ["mle", "aic", "huard", "xvcic"])
@pytest.mark.parametrize("sign", [1, -1])
def test_selected_block_is_reachable_and_silent(criterion, sign, caplog):
    u, v = _strong(sign)
    w = np.ones(u.size)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        blk = _select_and_fit_copula(["Plackett"], u, v, w, criterion=criterion)
    assert _warnings(caplog) == []
    assert blk == {"name": "Plackett", "tau": _bound(sign)}


def test_averaged_tau_is_stored_at_the_bound():
    """Multiple imputation averages the D fitted τ: the mean of values at a
    bound may pass it by an ulp."""
    assert _copula_block("Plackett", {"tau_k": float(np.nextafter(HI, 1.0))})["tau"] == HI
    assert _copula_block("Plackett", {"tau_k": float(np.nextafter(LO, -1.0))})["tau"] == LO
    inside = {"tau_k": 0.99352}
    assert _copula_block("Plackett", inside)["tau"] is inside["tau_k"]


@pytest.fixture(scope="module")
def fitted_strong_plackett():
    """ICE with a Plackett-only candidate set on data more dependent than Plackett."""
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    truth_taus = [0.998, 0.1, 0.1, -0.998]
    start_taus = [0.9, 0.1, 0.1, -0.9]
    for blk, t in zip(raw["copulas"], truth_taus):
        blk["tau"] = t
    _, Y = simulate(PMCModel.from_dict(raw), N=600, seed=7)
    for blk, t in zip(raw["copulas"], start_taus):
        blk["name"], blk["tau"] = "Plackett", t
    records: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Keep(level=logging.WARNING)
    log = logging.getLogger("pmcprg")
    log.addHandler(handler)
    try:
        fitted, trace = ice(PMCModel.from_dict(raw), Y,
                            ice_cfg={"candidates": ["Plackett"], "max_iter": 4})
    finally:
        log.removeHandler(handler)
    return fitted, trace, records


def test_ice_stores_the_tau_it_uses(fitted_strong_plackett):
    fitted, trace, records = fitted_strong_plackett
    assert [r.getMessage() for r in records if "Plackett" in r.getMessage()] == []
    diag = {(b["i"], b["j"]): b["tau"] for b in fitted.copula_blocks()}
    assert diag[(0, 0)] == HI and diag[(1, 1)] == LO
    for blk in fitted.copula_blocks():
        cop = fitted.copula(blk["i"], blk["j"])
        assert cop.params["tau_k"] == blk["tau"]
        if blk["i"] == blk["j"]:
            assert _plackett_tau_from_theta(cop.theta) == blk["tau"]
    assert np.nanmax(trace.tau_history) <= HI and np.nanmin(trace.tau_history) >= LO


def test_saved_fit_reloads_without_warning(fitted_strong_plackett, tmp_path, caplog):
    fitted, _, _ = fitted_strong_plackett
    path = fitted.save(tmp_path / "strong_plackett.toml")
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        reloaded = PMCModel(path)
    assert _warnings(caplog) == []
    assert reloaded.copula_blocks() == fitted.copula_blocks()
    assert reloaded.copula(0, 0).theta == pytest.approx(_PLACKETT_THETA_MAX, rel=1e-14)


# --------------------------------------------------------------------------
# Multistart draws
# --------------------------------------------------------------------------

def test_jitter_keeps_plackett_tau_reachable(tmp_path, caplog):
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    for blk, t in zip(raw["copulas"], [0.99, -0.99, 0.2, 0.9]):
        blk["name"], blk["tau"] = "Plackett", t
    model = PMCModel.from_dict(raw)
    hits = 0
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        for seed in range(10):
            out = perturb_initial_model(model, np.random.default_rng(seed), jitter=0.25)
            taus = [blk["tau"] for blk in out.copula_blocks()]
            assert LO <= min(taus) and max(taus) <= HI
            hits += sum(t in (LO, HI) for t in taus)
            PMCModel(out.save(tmp_path / f"jitter_{seed}.toml"))
    assert hits > 0
    assert _warnings(caplog) == []


def test_family_starts_keep_plackett_tau_reachable(caplog):
    raw = PMCModel(f"{MODELS}/pmc_gauss_k2.toml").raw
    for blk, t in zip(raw["copulas"], [0.99, -0.99, 0.2, 0.9]):
        blk["name"], blk["tau"] = "Plackett", t
    model = PMCModel.from_dict(raw)
    with caplog.at_level(logging.WARNING, logger="pmcprg"):
        for mode in ("random", "sweep"):
            cfg = {"n_starts": 8, "multistart_seed": 1, "multistart_jitter": 0.5,
                   "multistart_families": mode, "candidates": ["Plackett", "Gauss"]}
            for m, _ in build_multistart_inits(model, cfg, label="ICE")[1:]:
                taus = [b["tau"] for b in m.copula_blocks() if b["name"] == "Plackett"]
                assert all(LO <= t <= HI for t in taus)
    assert [r for r in _warnings(caplog) if "Plackett" in r.getMessage()] == []
