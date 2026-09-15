"""General PMC — unit tests of the pair-margin M-step (ICE, SEM, K-means).

Fast, exact checks of the dual-view update of ``pmcprg.pmc.ice._m_step`` for
pair-indexed margins f_ij (A16 Eqs. 12–14), complementing the statistical
acceptance tests of ``test_general_pmc_estimation.py``:

* the dual-view sample of f_ij is {y_n : (x_n, x_{n+1}) = (i, j)} ∪
  {y_{n+1} : (x_n, x_{n+1}) = (j, i)} for hard labels, with weights ½ ξ;
* the Gaussian update is the weighted moments of that sample (soft ξ too);
* the copula c_ij is fitted on (F_ij(y_n), F_ji(y_{n+1})) with weight ξ_n(i, j);
* a pair absent from a hard labelling keeps its margin;
* GICE selects the family per pair on true labels;
* PMC-IN pair ICE does not re-tie f_ij = f_ik, SEM keeps K² distinct blocks;
* multivariate pair margins (PMC-IN, d > 1) are updated per pair.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

from pmcprg.numerics import EPS, ONE_MINUS_EPS
from pmcprg.pmc.ice import _m_step, _pair_margin_sample, ice
from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights, smooth
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.sem import sem
from pmcprg.pmc.simulate import simulate

# ``pmcprg.pmc`` re-exports the function ``ice``, which shadows the submodule name.
ice_mod = importlib.import_module("pmcprg.pmc.ice")

TABLE1 = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}


def _pair_raw(variant="PMC", p=((0.35, 0.15), (0.15, 0.35)), tau=0.5):
    raw = {
        "model": {"name": "pair", "variant": variant, "K": 2},
        "prior": {"p": [list(r) for r in p]},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                    for (i, j), (mu, sd) in sorted(TABLE1.items())],
    }
    if variant == "PMC":
        raw["copulas"] = [{"i": i, "j": j, "name": "Clayton", "tau": tau}
                          for i in range(2) for j in range(2)]
    return raw


def _one_hot(X, K=2):
    N = len(X)
    gamma = np.zeros((N, K))
    gamma[np.arange(N), X] = 1.0
    xi = np.zeros((N - 1, K, K))
    xi[np.arange(N - 1), X[:-1], X[1:]] = 1.0
    return xi, gamma


def _run_m_step(model, Y, xi, gamma, **kw):
    raw = model.raw
    _m_step(raw, model, Y, xi, gamma,
            fit_margins=kw.get("fit_margins", True),
            candidates=kw.get("candidates", ["Clayton"]),
            selection_criterion="mle",
            margin_selection_rule=kw.get("rule", "mle"))
    return raw


def _block(raw, i, j):
    return next(b for b in raw["margins"] if b["i"] == i and b["j"] == j)


def test_dual_view_sample_is_the_left_and_right_sub_samples():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=40)
    Y = rng.normal(size=40)
    xi, _ = _one_hot(X)
    for i in range(2):
        for j in range(2):
            y, w = _pair_margin_sample(Y, xi, i, j)
            assert y.shape == w.shape == (2 * 39,)
            left = Y[:-1][(X[:-1] == i) & (X[1:] == j)]
            right = Y[1:][(X[:-1] == j) & (X[1:] == i)]
            np.testing.assert_array_equal(np.sort(y[w > 0]), np.sort(np.concatenate((left, right))))
            assert set(np.unique(w)) <= {0.0, 0.5}
    # ½ ξ: every observation counts once over the K² blocks (N − 1 in total).
    total = sum(_pair_margin_sample(Y, xi, i, j)[1].sum() for i in range(2) for j in range(2))
    assert total == pytest.approx(39.0)


@pytest.mark.parametrize("variant", ["PMC", "PMC-IN"])
def test_gaussian_pair_update_is_the_dual_view_weighted_moments(variant):
    m = PMCModel.from_dict(_pair_raw(variant))
    X, Y = simulate(m, N=400, seed=5)
    # Hard labels: plain sub-sample moments.
    raw = _run_m_step(m, Y, *_one_hot(X))
    for i in range(2):
        for j in range(2):
            sub = np.concatenate((Y[:-1][(X[:-1] == i) & (X[1:] == j)],
                                  Y[1:][(X[:-1] == j) & (X[1:] == i)]))
            prm = _block(raw, i, j)["params"]
            assert prm["loc"] == pytest.approx(sub.mean(), rel=1e-12, abs=1e-12)
            assert prm["scale"] == pytest.approx(sub.std(), rel=1e-12)
    # Soft posteriors: weighted moments with ξ_n(i, j) on y_n and ξ_n(j, i) on y_{n+1}.
    W, f_pdf = precompute_weights(m, Y)
    a, _ = forward(m, Y, W=W, f_pdf=f_pdf)
    b = backward(m, Y, W=W)
    xi, gamma = joint_posteriors(a, W, b), smooth(a, b)
    raw = _run_m_step(m, Y, xi, gamma)
    for i in range(2):
        for j in range(2):
            y = np.concatenate((Y[:-1], Y[1:]))
            w = np.concatenate((xi[:, i, j], xi[:, j, i]))
            mu = np.sum(w * y) / np.sum(w)
            sd = np.sqrt(np.sum(w * (y - mu) ** 2) / np.sum(w))
            prm = _block(raw, i, j)["params"]
            assert prm["loc"] == pytest.approx(mu, rel=1e-10)
            assert prm["scale"] == pytest.approx(sd, rel=1e-10)
    assert len({round(_block(raw, *k)["params"]["loc"], 8) for k in TABLE1}) == 4


def test_pair_copula_pseudo_observations_use_f_ij_and_f_ji(monkeypatch):
    m = PMCModel.from_dict(_pair_raw("PMC"))
    X, Y = simulate(m, N=120, seed=2)
    xi, gamma = _one_hot(X)
    seen = {}
    real = ice_mod._select_and_fit_copula

    def spy(candidates, u, v, weights, criterion="mle"):
        k = len(seen)
        seen[k] = (np.array(u), np.array(v), np.array(weights))
        return real(candidates, u, v, weights, criterion=criterion)

    monkeypatch.setattr(ice_mod, "_select_and_fit_copula", spy)
    _run_m_step(m, Y, xi, gamma)
    pairs = [(b["i"], b["j"]) for b in m.copula_blocks() if xi[:, b["i"], b["j"]].sum() >= 1e-12]
    assert len(seen) == len(pairs) == 4
    for k, (i, j) in enumerate(pairs):
        u, v, w = seen[k]
        np.testing.assert_array_equal(u, np.clip(m.margin(i, j).cdf_vec(Y[:-1]), EPS, ONE_MINUS_EPS))
        np.testing.assert_array_equal(v, np.clip(m.margin(j, i).cdf_vec(Y[1:]), EPS, ONE_MINUS_EPS))
        np.testing.assert_array_equal(w, xi[:, i, j])


def test_pair_absent_from_hard_labels_keeps_its_margin():
    m = PMCModel.from_dict(_pair_raw("PMC-IN"))
    Y = np.random.default_rng(1).normal(size=50)
    X = np.zeros(50, dtype=int)                      # only the pair (0, 0) occurs
    raw = _run_m_step(m, Y, *_one_hot(X))
    for (i, j), (mu, sd) in TABLE1.items():
        prm = _block(raw, i, j)["params"]
        if (i, j) == (0, 0):
            # y_0 and y_49 enter one view each, interior points both.
            assert prm["loc"] == pytest.approx(np.concatenate((Y[:-1], Y[1:])).mean())
        else:
            assert prm == {"loc": mu, "scale": sd}
    PMCModel.from_dict(raw)                          # still a valid model


def test_gice_selects_the_family_per_pair_on_true_labels():
    params = {(0, 0): {"loc": 0.0, "scale": 1.0}, (1, 1): {"loc": 3.0, "scale": 1.0},
              (0, 1): {"a": 2.0, "loc": 0.0, "scale": 0.8}, (1, 0): {"a": 2.0, "loc": 0.5, "scale": 0.9}}
    raw = _pair_raw("PMC-IN")
    for b in raw["margins"]:
        b["params"] = dict(params[(b["i"], b["j"])])
        b["dist"] = "norm" if b["i"] == b["j"] else "gamma"
    X, Y = simulate(PMCModel.from_dict(raw), N=4000, seed=4)
    init = _pair_raw("PMC-IN")
    for b in init["margins"]:
        b["params"] = {"loc": float(Y.mean()), "scale": float(Y.std())}
        b["candidates"] = ["norm", "gamma"]
    out = _run_m_step(PMCModel.from_dict(init), Y, *_one_hot(X), rule="aic")
    chosen = {(b["i"], b["j"]): b["dist"] for b in out["margins"]}
    assert chosen == {(0, 0): "norm", (0, 1): "gamma", (1, 0): "gamma", (1, 1): "norm"}


def test_pmc_in_pair_ice_does_not_retie_the_margins():
    """Wave-1 defect: a γ_n(i)-weighted update gave f_00 = f_01 after 2 iterations."""
    m = PMCModel.from_dict(_pair_raw("PMC-IN"))
    _, Y = simulate(m, N=500, seed=8)
    fitted, trace = ice(m, Y, {"max_iter": 3, "fit_margins": True, "tol": 0.0})
    assert fitted.margin_structure == "pair" and len(fitted.margin_blocks()) == 4
    for i in range(2):
        assert fitted.margin(i, 0).params != fitted.margin(i, 1).params
    assert all(len(snap) == 4 for snap in trace.margin_history)


def test_sem_on_pair_model_keeps_k2_distinct_blocks():
    m = PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml")
    _, Y = simulate(m, N=300, seed=3)
    fitted, trace = sem(m, Y, {"max_iter": 3, "fit_margins": True, "candidates": ["Clayton"],
                                "sem_seed": 1})
    assert fitted.margin_structure == "pair"
    locs = [b["params"]["loc"] for b in fitted.margin_blocks()]
    assert len(locs) == 4 and len(set(locs)) == 4
    assert np.all(np.isfinite(trace.log_liks))


def test_multivariate_pair_margins_are_updated_per_pair():
    means = {(0, 0): [0.0, 0.0], (0, 1): [0.5, -0.5], (1, 0): [1.5, 1.0], (1, 1): [2.0, 2.0]}
    raw = {
        "model": {"name": "mvn pair", "variant": "PMC-IN", "K": 2, "d": 2},
        "prior": {"p": [[0.4, 0.1], [0.1, 0.4]]},
        "margins": [{"i": i, "j": j, "dist": "multivariate_normal",
                     "params": {"mean": means[(i, j)], "cov": [[1.0, 0.2], [0.2, 1.0]]}}
                    for i in range(2) for j in range(2)],
    }
    m = PMCModel.from_dict(raw)
    X, Y = simulate(m, N=300, seed=6)
    out = _run_m_step(m, Y, *_one_hot(X))
    for i in range(2):
        for j in range(2):
            sub = np.concatenate((Y[:-1][(X[:-1] == i) & (X[1:] == j)],
                                  Y[1:][(X[:-1] == j) & (X[1:] == i)]))
            np.testing.assert_allclose(_block(out, i, j)["params"]["mean"], sub.mean(axis=0), rtol=1e-10)
    assert PMCModel.from_dict(out).margin_structure == "pair"


@pytest.mark.slow
def test_parallel_multistart_on_pair_model_matches_sequential():
    m = PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml")
    _, Y = simulate(m, N=200, seed=11)
    cfg = {"n_starts": 2, "multistart_seed": 5, "max_iter": 2, "candidates": ["Clayton"],
           "fit_margins": True}
    f_seq, t_seq = ice(m, Y, {**cfg, "multistart_workers": 1})
    f_par, t_par = ice(m, Y, {**cfg, "multistart_workers": 2})
    assert t_seq.log_liks == t_par.log_liks
    assert f_seq.margin_blocks() == f_par.margin_blocks()
    assert f_par.margin_structure == "pair" and len(f_par.margin_blocks()) == 4
