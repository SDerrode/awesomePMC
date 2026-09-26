"""``forecast(check_nodes=...)``: convergence of the predictive laws in ``gap_nodes``.

``GapPosterior.quad_error`` estimates the error of log p(y_obs) only: a
forecast horizon is a trailing gap and contributes 0 to it. ``check_nodes``
computes the predictive laws a second time with more nodes and compares
them with the definitions of the forecasting study
(``report/forecasting/fc_common.py``: ``quad_diff``, ``node_law_quantiles``,
``QUAD_TOL``, ``QUAD_TOL_TAIL``). Checked here:

* the returned forecast is the same bit for bit with the check on or off,
  whatever the grid cache holds;
* a PMC at τ = 0.99 with G = 16 is flagged, far beyond both tolerances,
  with a WARNING that the studies' quad_error parsers cannot take for
  theirs; weak dependence at the default G passes without a WARNING;
* the numbers equal a recomputation from two forecasts, at G and 2G, with
  the study's formulas;
* the int form and the argument checks; the exact variants (d = 1 and
  d = 3) are not recomputed; a predictive sd that collapses gives inf.
"""
from __future__ import annotations

import dataclasses
import logging
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from pmcprg.pmc import NodeCheck, PMCModel
from pmcprg.pmc import forecast as pmc_forecast
from pmcprg.pmc import gaps

REPO = Path(__file__).resolve().parents[2]
LOGGER = "pmcprg.pmc.gaps"

#: The regexes the studies use to parse the quad_error WARNING, copied from
#: report/forecasting/fc_common.py (``_QUAD_RE``, and its ``kind`` test) and
#: report/erroneous_data/intel_lab/il_common.py (``_QUAD_RE``).
STUDY_QUAD_RES = (
    re.compile(r"relative error ([0-9.eE+-]+) > ([0-9.eE+-]+) .*\((\d+) missing rows, gap_nodes = (\d+)\)"),
    re.compile(r"Missing-data quadrature not converged: relative error "
               r"([0-9.eE+-]+) > ([0-9.eE+-]+)"),
)
STUDY_QUAD_PREFIX = "Missing-data quadrature not converged"


def _pair_model(tau):
    """Two regimes, pair margins, Gaussian copulas at Kendall's τ."""
    m, s = (0.0, 2.0), (1.0, 0.7)
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": m[j] + 0.3 * i, "scale": s[j]}}
                    for i in range(2) for j in range(2)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]})


def _state_model(tau):
    """Two regimes, state margins, Gaussian copulas at Kendall's τ."""
    m, s = (0.0, 2.0), (1.0, 0.7)
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": m[i], "scale": s[i]}} for i in range(2)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]})


def _ar1_model(rho):
    """A Gaussian AR(1) of coefficient ρ written as a two-state PMC (identical regimes)."""
    tau = 2.0 / math.pi * math.asin(rho)
    std = {"dist": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    return PMCModel.from_dict({
        "model": {"variant": "PMC", "K": 2}, "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": i, **std} for i in range(2)],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau} for i in range(2) for j in range(2)]})


#: A series far in the upper tail of the stationary law (state 1, mean 2.3,
#: sd 0.7): under τ = 0.99 its forecast needs more than 16 nodes.
Y_TAIL = np.array([3.54, 3.53, 3.52, 3.54, 3.55, 3.52, 3.52, 3.515])
#: A series with a leading, an interior and a two-row gap.
Y_GAPS = np.array([np.nan, 0.1, 0.2, np.nan, 0.3, 0.25, np.nan, np.nan, 0.6, 0.8, 1.0, 1.1])


@pytest.fixture
def cold_caches():
    """Empty the grid and kernel-moment caches of the gap quadrature (a
    function, called before each computation to compare; also on teardown)."""
    def clear():
        gaps._GRID_CACHE.clear()
        gaps._GRID_CACHE_NB.clear()
        gaps._MOM_CACHE.clear()
    clear()
    yield clear
    clear()


def _fields(fc):
    return {f.name: getattr(fc, f.name) for f in dataclasses.fields(fc) if f.name != "node_check"}


def _assert_same_forecast(a, b):
    fa, fb = _fields(a), _fields(b)
    for name, va in fa.items():
        vb = fb[name]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            assert np.array_equal(va, vb), name
        else:
            assert va == vb, name


def _study_node_law_quantiles(nodes, mass, levels):
    """``node_law_quantiles`` of report/forecasting/fc_common.py."""
    out = np.empty((nodes.shape[0], len(levels)))
    for k in range(nodes.shape[0]):
        o = np.argsort(nodes[k])
        x, m = nodes[k][o], np.clip(mass[k][o], 0.0, None)
        b = np.r_[x[0] - 0.5 * (x[1] - x[0]), 0.5 * (x[1:] + x[:-1]), x[-1] + 0.5 * (x[-1] - x[-2])]
        F = np.r_[0.0, np.cumsum(m / m.sum())]
        out[k] = np.interp(levels, F, b)
    return out


def _check_warnings(records):
    return [r.getMessage() for r in records
            if r.name == LOGGER and r.levelno == logging.WARNING
            and r.getMessage().startswith("Forecast not stable in gap_nodes")]


# ---------------------------------------------------------------------------
# The returned forecast does not depend on the check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["pair_tau099_G16", "pair_tau099_gaps_G24", "state_tau03_G64"])
def test_forecast_is_bit_identical_with_the_check_on_or_off(case, cold_caches):
    m, Y, G = {"pair_tau099_G16": (_pair_model(0.99), Y_TAIL, 16),
               "pair_tau099_gaps_G24": (_pair_model(0.99), Y_GAPS, 24),
               "state_tau03_G64": (_state_model(0.3), Y_GAPS, 64)}[case]
    off = gaps.forecast(m, Y, 6, gap_nodes=G)
    cold_caches()
    on = gaps.forecast(m, Y, 6, gap_nodes=G, check_nodes=True)
    assert off.node_check is None and isinstance(on.node_check, NodeCheck)
    _assert_same_forecast(off, on)
    # warm caches, the check's own grids (2G) among them: still the same
    again = gaps.forecast(m, Y, 6, gap_nodes=G)
    _assert_same_forecast(off, again)
    on_again = gaps.forecast(m, Y, 6, gap_nodes=G, check_nodes=2 * G)
    _assert_same_forecast(off, on_again)
    assert on_again.node_check == on.node_check


def test_the_check_is_exported_with_forecast():
    import pmcprg.pmc
    assert pmc_forecast is gaps.forecast
    assert "NodeCheck" in pmcprg.pmc.__all__ and "NodeCheck" in gaps.__all__
    fc = pmc_forecast(_state_model(0.3), Y_GAPS, 2, check_nodes=True)
    assert fc.node_check.gap_nodes == gaps.DEFAULT_GAP_NODES
    assert fc.node_check.gap_nodes_ref == 2 * gaps.DEFAULT_GAP_NODES


# ---------------------------------------------------------------------------
# Flagged, and not flagged
# ---------------------------------------------------------------------------

def test_strong_dependence_at_few_nodes_is_flagged(caplog):
    # Measured: mean_sd 0.20, tails 0.42 (G = 16 → 32, h = 10); for τ =
    # 0.985–0.99 and the series shifted by ±0.3, 0.17–0.27 and 0.35–0.57.
    # At G = 24 still 0.036 / 0.24.
    m = _pair_model(0.99)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        fc = gaps.forecast(m, Y_TAIL, 10, gap_nodes=16, check_nodes=True)
    c = fc.node_check
    assert (c.gap_nodes, c.gap_nodes_ref) == (16, 32)
    assert not c.converged
    assert c.mean_sd > 5 * gaps.FORECAST_CHECK_TOL
    assert c.tails > 4 * gaps.FORECAST_CHECK_TOL_TAIL
    msgs = _check_warnings(caplog.records)
    assert len(msgs) == 1
    msg = msgs[0]
    assert "gap_nodes = 16 to 32" in msg and "Increase gap_nodes." in msg
    assert "means" in msg and "sds" in msg and "quantiles" in msg
    # the studies must not take it for the quad_error WARNING
    assert not msg.startswith(STUDY_QUAD_PREFIX)
    assert not any(rx.search(msg) for rx in STUDY_QUAD_RES)


def test_weak_dependence_at_the_default_nodes_passes_quietly(caplog):
    # Measured (G = 64 → 128, h = 10): mean_sd 6.8e-8, tails 1.5e-3 (Y_TAIL
    # shifted to the body of the law: 7.0e-8, 2.2e-3). At G = 32, 1.2e-5 and
    # 8.4e-3; at G = 16 the node-law tails are 0.28 (first order in the
    # node spacing), for means and sds within 2.3e-3.
    m = _state_model(0.3)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        fc = gaps.forecast(m, Y_GAPS, 10, check_nodes=True)
    c = fc.node_check
    assert c.converged
    assert (c.gap_nodes, c.gap_nodes_ref) == (64, 128)
    assert c.mean_sd < 1e-5 and c.tails < 1e-2
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model, Y, G", [(_pair_model(0.99), Y_TAIL, 16),
                                         (_pair_model(0.99), Y_GAPS, 24),
                                         (_state_model(0.3), Y_GAPS, 32)],
                         ids=["tail_G16", "gaps_G24", "weak_G32"])
def test_numbers_equal_a_recomputation_from_two_forecasts(model, Y, G, cold_caches):
    h = 7
    c = gaps.forecast(model, Y, h, gap_nodes=G, check_nodes=True).node_check
    cold_caches()
    a = gaps.forecast(model, Y, h, gap_nodes=G, quantiles=(0.5,))
    cold_caches()
    b = gaps.forecast(model, Y, h, gap_nodes=2 * G, quantiles=(0.5,))
    # quad_diff of report/forecasting/fc_common.py, at one origin
    sd_y = float(np.nanstd(Y))
    assert min(np.min(a.sd), np.min(b.sd)) >= 1e-3 * sd_y          # no collapse here
    worst = max(float(np.max(np.abs(a.sd - b.sd) / b.sd)), float(np.max(np.abs(a.mean - b.mean) / b.sd)))
    qa = _study_node_law_quantiles(np.asarray(a.grid_nodes), np.asarray(a.grid_mass), (0.025, 0.975))
    qb = _study_node_law_quantiles(np.asarray(b.grid_nodes), np.asarray(b.grid_mass), (0.025, 0.975))
    tail = float(np.max(np.abs(qa - qb) / b.sd[:, None]))
    assert c.mean_sd == worst
    assert c.tails == tail
    assert c.converged == (worst <= 0.01 and tail <= 0.05)
    assert (gaps.FORECAST_CHECK_TOL, gaps.FORECAST_CHECK_TOL_TAIL) == (0.01, 0.05)


# ---------------------------------------------------------------------------
# Arguments, exact variants, collapse
# ---------------------------------------------------------------------------

def test_int_form_and_argument_checks(caplog):
    m = _pair_model(0.99)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        c_true = gaps.forecast(m, Y_TAIL, 4, gap_nodes=16, check_nodes=True).node_check
        c_int = gaps.forecast(m, Y_TAIL, 4, gap_nodes=16, check_nodes=32).node_check
        c_np = gaps.forecast(m, Y_TAIL, 4, gap_nodes=16, check_nodes=np.int64(32)).node_check
        c_48 = gaps.forecast(m, Y_TAIL, 4, gap_nodes=16, check_nodes=48).node_check
    assert c_true == c_int == c_np
    assert (c_48.gap_nodes, c_48.gap_nodes_ref) == (16, 48)
    assert c_48.mean_sd != c_true.mean_sd
    assert "gap_nodes = 16 to 48" in _check_warnings(caplog.records)[-1]
    for off in (False, None, np.False_):
        assert gaps.forecast(m, Y_TAIL, 2, gap_nodes=16, check_nodes=off).node_check is None
    for bad in (16, 8, 0, -32):
        with pytest.raises(ValueError, match="check_nodes"):
            gaps.forecast(m, Y_TAIL, 2, gap_nodes=16, check_nodes=bad)
    for bad in (32.0, "yes", [32]):
        with pytest.raises(TypeError, match="check_nodes"):
            gaps.forecast(m, Y_TAIL, 2, gap_nodes=16, check_nodes=bad)
    # checked before any computation
    with pytest.raises(ValueError, match="check_nodes"):
        gaps.forecast(m, Y_TAIL, 2, gap_nodes=None, check_nodes=64)


@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "pmc_in_gauss_k2.toml", "hmc_in_mvn_k2_d3.toml"])
def test_exact_variants_are_converged_without_recomputation(name, monkeypatch, caplog):
    m = PMCModel(REPO / "pmcprg" / "pmc" / "models" / name)
    assert not gaps.needs_grid(m)
    d = getattr(m, "d", 1)
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((12, d)) if d > 1 else rng.standard_normal(12)
    Y[3] = np.nan
    off = gaps.forecast(m, Y, 3, gap_nodes=32)

    def no_grid(*a, **k):
        raise AssertionError("the check recomputed an exact forecast")
    monkeypatch.setattr(gaps, "_run_chain", no_grid)
    monkeypatch.setattr(gaps, "_horizon_chain", no_grid)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        on = gaps.forecast(m, Y, 3, gap_nodes=32, check_nodes=True)
    assert on.method == "exact"
    _assert_same_forecast(off, on)
    assert on.node_check == NodeCheck(gap_nodes=32, gap_nodes_ref=64, mean_sd=0.0, tails=0.0,
                                      converged=True)
    assert not _check_warnings(caplog.records)


def test_a_collapsed_predictive_sd_is_flagged_not_divided(monkeypatch, caplog):
    # On the reference grid alone (the quadrature before local grids) an
    # AR(1) at ρ = 0.9999 forecast from one reading has sds of 4e-9 for
    # 0.014–0.024 (a law on one node). One reading has no sd: the scale is
    # the sd of the reference law, 1, and the floor 1e-3.
    m = _ar1_model(0.9999)
    Y = np.array([0.3])
    fine = gaps.forecast(m, Y, 3, check_nodes=True).node_check
    assert fine.converged and np.isfinite(fine.mean_sd)
    monkeypatch.setattr(gaps, "_LOCAL_GRIDS", False)
    with warnings.catch_warnings(), caplog.at_level(logging.WARNING, logger=LOGGER):
        warnings.simplefilter("error")                  # no division by 0, no nanstd of nothing
        fc = gaps.forecast(m, Y, 3, check_nodes=True)
    assert np.max(fc.sd) < 1e-3
    c = fc.node_check
    assert c.mean_sd == math.inf and c.tails == math.inf and not c.converged
    msgs = _check_warnings(caplog.records)
    assert len(msgs) == 1 and "collapses" in msgs[0] and "the reference law" in msgs[0]
    assert "Increase gap_nodes." in msgs[0]
    assert not any(rx.search(msgs[0]) for rx in STUDY_QUAD_RES)
