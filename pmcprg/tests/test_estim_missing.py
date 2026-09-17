"""Unsupervised estimation (ICE, SEM) with missing observations — mechanics.

What the missing-data code paths compute, checked against direct
computations: configuration keys, the observed-data log-likelihood trace, the
``"available"`` M-step (observed values / pairs only, exact posteriors), the
``"impute"`` combination rules, the k-means warm start on the observed rows,
SEM's joint completion, leading/trailing gaps, d > 1, multistart. The
statistical behaviour (parameter recovery against the missing rate, trace
behaviour, family selection) is in ``test_estim_missing_recovery.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from pmcprg.missing.patterns import mcar
from pmcprg.numerics import EPS, ONE_MINUS_EPS
from pmcprg.pmc import gaps
from pmcprg.pmc._estim_common import ice_estim_defaults, ice_missing_defaults, sem_estim_defaults
from pmcprg.pmc.ice import ice
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.sem import sem
from pmcprg.pmc.simulate import simulate

ICE = importlib.import_module("pmcprg.pmc.ice")
SEM = importlib.import_module("pmcprg.pmc.sem")
MODELS = Path(__file__).resolve().parents[1] / "pmc" / "models"


def _model(name: str) -> PMCModel:
    return PMCModel(MODELS / name)


def _data(name: str, N: int = 400, rate: float = 0.2, seed: int = 0, block: int = 5):
    mdl = _model(name)
    X, Y = simulate(mdl, N=N, seed=seed)
    if Y.ndim == 1:
        Ym, _ = mcar(Y, rate, block_size=block, seed=seed + 1)
    else:
        Ym = Y.copy()
        rows = np.random.default_rng(seed + 1).choice(np.arange(N // 10, N), int(rate * N),
                                                      replace=False)
        Ym[rows] = np.nan
    return mdl, X, Y, Ym


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_missing_keys_have_defaults_but_stay_out_of_the_gui_key_list():
    cfg = ICE._parse_ice_cfg(_model("hmc_in_gauss_k2.toml"), None)
    assert cfg["missing_strategy"] == "available"
    assert cfg["missing_draws"] == 5 and cfg["missing_seed"] == 0
    assert cfg["gap_nodes"] == gaps.DEFAULT_GAP_NODES
    assert SEM._parse_sem_cfg(_model("hmc_in_gauss_k2.toml"), None)["gap_nodes"] == 64
    # ice_estim_defaults() / sem_estim_defaults() is the *mandatory* GUI widget
    # contract (test_ice_tab_exposes_every_api_config_key); missing-data keys
    # are covered by _IceTab's own dedicated widgets instead (sourced from
    # ice_missing_defaults(), see test_ice_tab_round_trips_missing_keys).
    assert not set(ice_missing_defaults()) & (set(ice_estim_defaults()) | set(sem_estim_defaults()))


@pytest.mark.parametrize("bad", [{"missing_strategy": "drop"}, {"missing_draws": 0},
                                 {"missing_draws": 2.5}, {"gap_nodes": 1}])
def test_invalid_missing_keys_are_refused(bad):
    mdl = _model("hmc_in_gauss_k2.toml")
    with pytest.raises(ValueError):
        ICE._parse_ice_cfg(mdl, bad)
    if "gap_nodes" in bad:
        with pytest.raises(ValueError):
            SEM._parse_sem_cfg(mdl, bad)


def test_complete_data_never_enters_the_missing_data_code(monkeypatch):
    pytest.importorskip("sklearn")

    def boom(*a, **k):
        raise AssertionError("missing-data code reached on complete data")

    for name in ("_gap_e_step", "_m_step_impute", "_warmstart_from_kmeans_missing",
                 "_combine_margin_fits", "_combine_choice"):
        monkeypatch.setattr(ICE, name, boom)
    monkeypatch.setattr(SEM, "gap_e_step", boom)
    mdl, _, Y, _ = _data("pmc_gauss_k2.toml", N=200)
    for cfg in ({"max_iter": 2, "missing_strategy": "impute"},
                {"max_iter": 2, "init": "kmeans", "fit_margins": True, "n_starts": 2}):
        ice(mdl, Y, ice_cfg=cfg)
        sem(mdl, Y, sem_cfg=cfg)


# ---------------------------------------------------------------------------
# Log-likelihood trace = observed-data log-likelihood
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "pmc_gauss_k2.toml",
                                  "pmc_pair_gauss_k2.toml"])
@pytest.mark.parametrize("algo", ["ice", "ice_impute", "sem"])
def test_trace_starts_at_the_observed_data_log_likelihood(name, algo):
    mdl, _, _, Ym = _data(name)
    ref = gaps.gap_posterior(mdl, Ym).log_lik
    cfg = {"max_iter": 2, "fit_margins": True}
    if algo == "sem":
        _, tr = sem(mdl, Ym, sem_cfg=cfg)
    else:
        _, tr = ice(mdl, Ym, ice_cfg={**cfg, "missing_strategy": "impute" if algo == "ice_impute"
                                      else "available", "missing_draws": 2})
    assert np.isfinite(tr.log_liks).all()
    assert tr.log_liks[0] == pytest.approx(ref, rel=1e-12)


# ---------------------------------------------------------------------------
# "available": the M-step on observed values and pairs, exact posteriors
# ---------------------------------------------------------------------------

def test_available_state_margins_are_gamma_weighted_on_observed_values():
    mdl, _, _, Ym = _data("hmc_dn_gauss_k2.toml", N=500, rate=0.3)
    post = gaps.gap_posterior(mdl, Ym)
    obs = ~post.miss
    raw = mdl.raw
    ICE._m_step(raw, mdl, Ym, post.xi, post.gamma, fit_margins=True, candidates=["Gauss"],
                selection_criterion="mle", margin_selection_rule="mle", obs=obs)
    for blk in PMCModel.from_dict(raw).margin_blocks():
        w, y = post.gamma[obs, blk["i"]], Ym[obs]
        mu = np.sum(w * y) / np.sum(w)
        sd = np.sqrt(np.sum(w * (y - mu) ** 2) / np.sum(w))
        assert blk["params"]["loc"] == pytest.approx(mu, rel=1e-10)
        assert blk["params"]["scale"] == pytest.approx(sd, rel=1e-10)
    # prior from ξ over every n (gaps included)
    p = post.xi.sum(axis=0) / (len(Ym) - 1)
    p = 0.5 * (p + p.T)
    p /= p.sum()
    A = PMCModel.from_dict(raw).transition_A
    np.testing.assert_allclose(A, p / p.sum(axis=1, keepdims=True), rtol=1e-12)


def test_available_copulas_use_pairs_with_both_endpoints_observed():
    mdl, _, _, Ym = _data("pmc_gauss_k2.toml", N=500, rate=0.3)
    post = gaps.gap_posterior(mdl, Ym)
    obs = ~post.miss
    raw = mdl.raw
    ICE._m_step(raw, mdl, Ym, post.xi, post.gamma, fit_margins=False, candidates=["Gauss", "Clayton"],
                selection_criterion="mle", margin_selection_rule="mle", obs=obs)
    both = obs[:-1] & obs[1:]
    F = np.column_stack([mdl.margin(k).cdf_vec(np.where(obs, Ym, 0.0)) for k in range(2)])
    F = np.clip(F, EPS, ONE_MINUS_EPS)
    fitted = PMCModel.from_dict(raw)
    for blk in fitted.copula_blocks():
        i, j = blk["i"], blk["j"]
        ref = ICE._select_and_fit_copula(["Gauss", "Clayton"], F[:-1, i][both], F[1:, j][both],
                                         post.xi[both, i, j], criterion="mle")
        assert blk["name"] == ref["name"]
        assert blk["tau"] == pytest.approx(ref["tau"], abs=1e-9)


def test_available_pair_margins_keep_the_observed_endpoints_of_the_dual_view():
    mdl, _, _, Ym = _data("pmc_pair_gauss_k2.toml", N=500, rate=0.3)
    post = gaps.gap_posterior(mdl, Ym)
    obs = ~post.miss
    blocks = mdl.margin_blocks()
    ICE._m_step_pair_margins(blocks, Ym, post.xi, "mle", obs=obs)
    for blk in blocks:
        i, j = blk["i"], blk["j"]
        y = np.concatenate((Ym[:-1], Ym[1:]))
        w = 0.5 * np.concatenate((post.xi[:, i, j], post.xi[:, j, i]))
        keep = np.isfinite(y)
        mu = np.sum(w[keep] * y[keep]) / np.sum(w[keep])
        assert blk["params"]["loc"] == pytest.approx(mu, rel=1e-10)


# ---------------------------------------------------------------------------
# "impute": completions, combination rules, reproducibility
# ---------------------------------------------------------------------------

def test_combine_choice_pooled_and_majority(monkeypatch):
    table = np.array([[10.0, 9.0, -np.inf],
                      [10.0, 9.0, 30.0],
                      [1.0, 12.0, 30.0]])
    monkeypatch.setattr(ICE, "_IMPUTE_FAMILY_RULE", "pooled")
    assert ICE._combine_choice(table) == 1           # sums 21 < 30; column 2 unusable
    assert ICE._combine_choice(np.full((2, 2), -np.inf)) == -1
    monkeypatch.setattr(ICE, "_IMPUTE_FAMILY_RULE", "majority")
    assert ICE._combine_choice(table) == 0           # votes 2 : 1 over the usable columns
    tie = np.array([[1.0, 2.0], [3.0, 2.5]])         # one vote each; sums 4.0 < 4.5
    assert ICE._combine_choice(tie) == 1


def test_average_params_nested():
    out = ICE._average_params([{"mean": [0.0, 2.0], "cov": [[1.0, 0.0], [0.0, 1.0]], "fit_failed": True},
                               {"mean": [2.0, 4.0], "cov": [[3.0, 1.0], [1.0, 3.0]]}])
    assert out == {"mean": [1.0, 3.0], "cov": [[2.0, 0.5], [0.5, 2.0]]}


@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "pmc_pair_gauss_k2.toml"])
def test_gap_e_step_draws_fill_only_the_missing_rows(name):
    mdl, _, _, Ym = _data(name)
    miss = gaps.missing_mask(Ym)
    est = ICE._gap_e_step(mdl, Ym, miss, gap_nodes=None, posterior=True, n_draws=3,
                          rng=np.random.default_rng(0))
    assert est.Y_draws.shape == (3, len(Ym)) and est.X_draws.shape == (3, len(Ym))
    assert np.isfinite(est.Y_draws).all()
    np.testing.assert_array_equal(est.Y_draws[:, ~miss], np.broadcast_to(Ym[~miss], (3, (~miss).sum())))
    assert not np.array_equal(est.Y_draws[0, miss], est.Y_draws[1, miss])
    if gaps.needs_grid(mdl):                        # decoded on the quadrature nodes
        nodes = gaps.reference_grid(mdl).nodes
        assert np.isin(est.Y_draws[:, miss], nodes).all()


def test_impute_is_reproducible_and_seed_dependent():
    mdl, _, _, Ym = _data("pmc_gauss_k2.toml")
    cfg = {"max_iter": 3, "fit_margins": True, "missing_strategy": "impute", "missing_draws": 2}
    a, ta = ice(mdl, Ym, ice_cfg={**cfg, "missing_seed": 4})
    b, tb = ice(mdl, Ym, ice_cfg={**cfg, "missing_seed": 4})
    c, tc = ice(mdl, Ym, ice_cfg={**cfg, "missing_seed": 5})
    assert ta.log_liks == tb.log_liks and a.raw == b.raw
    assert ta.log_liks != tc.log_liks


def test_impute_gice_selects_a_candidate_family():
    mdl, _, _, Ym = _data("sp2016_gice_k2.toml", N=600, rate=0.2)
    fitted, tr = ice(mdl, Ym, ice_cfg={"max_iter": 2, "missing_strategy": "impute",
                                       "missing_draws": 2, "selection_criterion": "mle",
                                       "margin_selection_rule": "mle"})
    cands = {"norm", "gamma", "invgamma", "betaprime"}
    assert all(b["dist"] in cands for b in fitted.margin_blocks())
    assert all(b["name"] in {"Gauss", "GH", "Clayton", "Frank"} for b in fitted.copula_blocks())
    assert np.isfinite(tr.log_liks).all()


# ---------------------------------------------------------------------------
# k-means warm start on the observed rows
# ---------------------------------------------------------------------------

def test_kmeans_warm_start_clusters_observed_rows_and_skips_the_gaps():
    pytest.importorskip("sklearn")
    mdl, _, _, Ym = _data("hmc_in_gauss_k2.toml", N=600, rate=0.3)
    obs = np.isfinite(Ym)
    warm = ICE._warmstart_from_kmeans(mdl, Ym, random_state=3, fit_margins=True,
                                      candidates=["Gauss"], selection_criterion="mle",
                                      margin_selection_rule="mle")
    labels = ICE._align_kmeans_labels(
        ICE._kmeans_label_assignment(Ym[obs], 2, random_state=3), Ym[obs], mdl)
    for blk in warm.margin_blocks():
        vals = Ym[obs][labels == blk["i"]]
        assert blk["params"]["loc"] == pytest.approx(vals.mean(), rel=1e-10)
        assert blk["params"]["scale"] == pytest.approx(vals.std(), rel=1e-10)
    full = np.zeros(len(Ym), dtype=int)
    full[obs] = labels
    both = obs[:-1] & obs[1:]
    counts = np.zeros((2, 2))
    np.add.at(counts, (full[:-1][both], full[1:][both]), 1.0)
    p = 0.5 * (counts + counts.T)
    np.testing.assert_allclose(warm.transition_A, p / p.sum(axis=1, keepdims=True), rtol=1e-12)


# ---------------------------------------------------------------------------
# SEM: joint completion
# ---------------------------------------------------------------------------

def test_sem_fits_the_completed_series(monkeypatch):
    """The M-step receives Y with its gaps filled by the draw, not the NaN."""
    mdl, _, _, Ym = _data("pmc_gauss_k2.toml")
    seen = []
    real = SEM.m_step

    def spy(raw, current, Y, xi, gamma, **kw):
        seen.append(np.array(Y, copy=True))
        return real(raw, current, Y, xi, gamma, **kw)

    monkeypatch.setattr(SEM, "m_step", spy)
    _, tr = sem(mdl, Ym, sem_cfg={"max_iter": 2, "fit_margins": True, "sem_seed": 1})
    miss = np.isnan(Ym)
    assert len(seen) == 2
    for Yc in seen:
        assert np.isfinite(Yc).all()
        np.testing.assert_array_equal(Yc[~miss], Ym[~miss])
    assert not np.array_equal(seen[0][miss], seen[1][miss])
    assert tr.sampled_X_history.shape == (2, len(Ym))


# ---------------------------------------------------------------------------
# Edge cases: gaps at both ends, d > 1, multistart
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["hmc_in_gauss_k2.toml", "pmc_gauss_k2.toml",
                                  "pmc_pair_gauss_k2.toml"])
def test_leading_and_trailing_gaps(name):
    mdl, _, Y, _ = _data(name, N=300)
    Ym = Y.copy()
    Ym[:7] = np.nan
    Ym[-5:] = np.nan
    Ym[100:130] = np.nan
    for strategy in ("available", "impute"):
        fitted, tr = ice(mdl, Ym, ice_cfg={"max_iter": 3, "fit_margins": True, "missing_draws": 2,
                                           "missing_strategy": strategy})
        assert np.isfinite(tr.log_liks).all()
        assert all(np.isfinite(list(b["params"].values())).all() for b in fitted.margin_blocks())
    _, tr = sem(mdl, Ym, sem_cfg={"max_iter": 3, "fit_margins": True})
    assert np.isfinite(tr.log_liks).all()


def test_multivariate_rows_missing():
    pytest.importorskip("sklearn")
    mdl, _, _, Ym = _data("hmc_in_mvn_k2_d3.toml", N=500, rate=0.2)
    for cfg in ({"missing_strategy": "available"}, {"missing_strategy": "impute", "missing_draws": 2},
                {"init": "kmeans"}):
        fitted, tr = ice(mdl, Ym, ice_cfg={"max_iter": 3, **cfg})
        assert np.isfinite(tr.log_liks).all()
        for blk in fitted.margin_blocks():
            assert np.all(np.linalg.eigvalsh(np.asarray(blk["params"]["cov"])) > 0)
    fitted, tr = sem(mdl, Ym, sem_cfg={"max_iter": 3})
    assert np.isfinite(tr.log_liks).all()


@pytest.mark.parametrize("algo", ["ice", "sem"])
@pytest.mark.parametrize("missing_strategy", ["available", "impute"])
def test_gui_estimation_entry_point_accepts_missing_values(algo, missing_strategy, monkeypatch):
    """The GUI worker entry (``_do_estimate``) runs on NaN data and its result is taken in.

    ``missing_strategy`` is set on the actual ``_IceTab`` widgets (the combo
    and the imputation-draws spinbox — see ``_IceTab``'s missing-observations
    section), then read back through ``get_cfg()``, exactly as the real
    "Estimate" action does. SEM ignores ``missing_strategy`` (it always
    completes by one joint posterior draw), so both values must run cleanly
    for it too.
    """
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication, QMessageBox

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    app = QApplication.instance() or QApplication([])   # noqa: F841 — kept alive
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    path = MODELS / "pmc_pair_gauss_k2.toml"
    mdl, _, _, Ym = _data(path.name, N=300)
    w = PMCMainWindow(str(path))
    w._tab_ice._combo_missing_strategy.setCurrentText(missing_strategy)
    w._tab_ice._spn_missing_draws.setValue(2)
    cfg = w._tab_ice.get_cfg()
    assert cfg["missing_strategy"] == missing_strategy
    cfg.update(max_iter=2, algorithm=algo)
    result = PMCMainWindow._do_estimate(mdl, Ym, cfg)
    assert np.isfinite(result.trace.log_liks).all()
    w._on_est_done(result)
    assert w._model is result.fitted_model


@pytest.mark.slow
@pytest.mark.parametrize("algo", ["ice", "sem"])
def test_multistart_with_gaps_parallel_matches_sequential(algo):
    mdl, _, _, Ym = _data("pmc_gauss_k2.toml", N=300)
    cfg = {"max_iter": 3, "n_starts": 3, "multistart_families": "random", "fit_margins": True,
           "candidates": ["Gauss", "Clayton"], "missing_strategy": "impute", "missing_draws": 2}
    if algo == "sem":
        cfg = {k: v for k, v in cfg.items() if not k.startswith("missing")}
    fn = ice if algo == "ice" else sem
    f_seq, t_seq = fn(mdl, Ym, {**cfg, "multistart_workers": 1})
    f_par, t_par = fn(mdl, Ym, {**cfg, "multistart_workers": 2})
    assert t_seq.log_liks == t_par.log_liks and t_seq.run_tag == t_par.run_tag
    assert f_seq.raw == f_par.raw
    assert len(t_seq.multistart_runs) == 2
