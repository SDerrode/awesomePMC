"""K-means warm start: clusters are renumbered to match the declared state laws.

K-means numbers its clusters arbitrarily. Before this fix the warm start used
that numbering as the state labels, so with known margins
(``fit_margins = False``) about one start in two estimated the prior and the
copulas on labels permuted with respect to the margins. Measured on the CSDA
2013 Exp. 3 design (Gaussian margins, 10 seeds): the true Clayton copula of
state pair (1, 1) was recovered 5/10 times from the k-means start, 10/10 after
renumbering (report/out/csda_compare/ice_init_check.py).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

pytest.importorskip(
    "sklearn",
    reason="ICE init='kmeans' requires scikit-learn (pip install awesomepmc[ml]).",
)

import importlib

from pmcprg.pmc.ice import _align_kmeans_labels, _warmstart_from_kmeans
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

# ``pmcprg.pmc`` re-exports the function ``ice``, which shadows the submodule name.
ice_mod = importlib.import_module("pmcprg.pmc.ice")

MODELS = pathlib.Path("pmcprg/pmc/models")
HMC_IN_K2 = MODELS / "hmc_in_gauss_k2.toml"
HMC_IN_K3 = MODELS / "hmc_in_gauss_k3.toml"
PMC_K2 = MODELS / "pmc_gauss_k2.toml"
PMC_PAIR_K2 = MODELS / "pmc_pair_gauss_k2.toml"


@pytest.mark.parametrize("path", [HMC_IN_K2, PMC_K2, PMC_PAIR_K2])
def test_permuted_true_labels_are_renumbered_back_k2(path):
    mdl = PMCModel(path)
    X, Y = simulate(mdl, N=1500, seed=3)
    assert np.array_equal(_align_kmeans_labels(X, Y, mdl), X)
    assert np.array_equal(_align_kmeans_labels(1 - X, Y, mdl), X)


def test_cyclic_permutation_is_undone_k3():
    mdl = PMCModel(HMC_IN_K3)
    X, Y = simulate(mdl, N=1500, seed=4)
    for shift in (1, 2):
        assert np.array_equal(_align_kmeans_labels((X + shift) % 3, Y, mdl), X)


def test_identical_declared_margins_keep_the_numbering():
    raw = PMCModel(HMC_IN_K2).raw
    for blk in raw["margins"]:
        blk["params"] = {"loc": 0.0, "scale": 1.0}
    mdl = PMCModel.from_dict(raw)
    rng = np.random.default_rng(0)
    Y = rng.normal(size=400)
    labels = rng.integers(0, 2, size=400)
    assert np.array_equal(_align_kmeans_labels(labels, Y, mdl), labels)


def test_empty_cluster_is_tolerated():
    mdl = PMCModel(HMC_IN_K3)
    _, Y = simulate(mdl, N=300, seed=1)
    labels = np.where(Y < 0, 0, 2)          # cluster 1 empty
    out = _align_kmeans_labels(labels, Y, mdl)
    assert set(np.unique(out)) <= {0, 1, 2}
    assert len(np.unique(out)) == 2


@pytest.mark.parametrize("path", [PMC_K2, PMC_PAIR_K2])
def test_warm_start_does_not_depend_on_cluster_numbering(path, monkeypatch):
    mdl = PMCModel(path)
    _, Y = simulate(mdl, N=1200, seed=7)
    base = ice_mod._kmeans_label_assignment(Y, mdl.K, random_state=0)
    kw = dict(random_state=0, fit_margins=False, candidates=["Gauss", "Clayton"],
              selection_criterion="mle", margin_selection_rule="mle")

    monkeypatch.setattr(ice_mod, "_kmeans_label_assignment", lambda *a, **k: base)
    warm_a = _warmstart_from_kmeans(mdl, Y, **kw)
    monkeypatch.setattr(ice_mod, "_kmeans_label_assignment", lambda *a, **k: 1 - base)
    warm_b = _warmstart_from_kmeans(mdl, Y, **kw)

    assert np.array_equal(warm_a.prior_p, warm_b.prior_p)
    for i in range(mdl.K):
        for j in range(mdl.K):
            ca, cb = warm_a.copula(i, j), warm_b.copula(i, j)
            assert ca.copula_enum == cb.copula_enum
            assert ca.params["tau_k"] == cb.params["tau_k"]
