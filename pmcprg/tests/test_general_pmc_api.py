"""General PMC with pair-indexed margins — API details beyond the acceptance tests.

Complements ``test_general_pmc_core.py`` (the contract of the first
implementation wave) with: the refusal of ``margin(i)`` and its message, the
margin-structure resolution table, the once-per-process INFO notice, block
order and GICE ``candidates`` round trip, pair margins with multivariate
observations (supported for PMC-IN), FFBS on pair models against the exact
path posterior, the relation between ``PMCModel.weight`` and
``precompute_weights``, and the log-space weights. Formulas: Derrode &
Pieczynski (2013, DerrodePieczynski_CSDA2013) Eqs. 12–14 and the Proposition
of §2.1.
"""
from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import pmcprg.pmc.model as model_mod
from pmcprg.pmc.inference import (
    _log_transition_weights, backward, forward, joint_posteriors,
    precompute_weights, sample_posterior, smooth,
)
from pmcprg.pmc.model import PMCModel, Variant
from pmcprg.pmc.simulate import simulate

MODELS_DIR = Path(__file__).resolve().parents[1] / "pmc" / "models"
PAIR_FIXTURE = MODELS_DIR / "pmc_pair_gauss_k2.toml"

TABLE1 = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}
P_ORIG = [[0.50, 0.05], [0.05, 0.40]]
Y6 = np.array([0.2, -0.4, 1.9, 1.1, 2.4, 0.6])


def _pair_raw(variant="PMC", copula="Clayton", tau=0.7, margins=TABLE1, p=P_ORIG, **model_keys):
    raw = {
        "model": {"name": "pair", "variant": variant, "K": 2, "N_default": 100, **model_keys},
        "prior": {"p": [list(r) for r in p]},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                    for (i, j), (mu, sd) in sorted(margins.items())],
    }
    if variant == "PMC":
        raw["copulas"] = [{"i": i, "j": j, "name": copula, "tau": tau}
                          for i in range(2) for j in range(2)]
    return raw


def _state_raw(variant="PMC", copula="Clayton", tau=0.7, locs=(0.0, 1.1), scales=(1.0, 1.4)):
    raw = _pair_raw(variant=variant, copula=copula, tau=tau)
    raw["margins"] = [{"i": i, "dist": "norm", "params": {"loc": locs[i], "scale": scales[i]}}
                      for i in range(2)]
    return raw


# --------------------------------------------------------------------------
# margin(i) refusal, structure resolution, notice
# --------------------------------------------------------------------------

def test_margin_without_j_is_refused_with_an_explanation():
    m = PMCModel.from_dict(_pair_raw())
    with pytest.raises(ValueError) as exc:
        m.margin(1)
    msg = str(exc.value)
    assert "ambiguous" in msg and "margin_structure = 'pair'" in msg
    assert "margin(i, j)" in msg and "margin(j, i)" in msg
    with pytest.raises(ValueError, match="ambiguous"):
        m.pdf(0, None, 0.3)
    with pytest.raises(KeyError, match="pair"):
        m.margin(0, 2)
    # j is still ignored for state margins.
    s = PMCModel.from_dict(_state_raw())
    assert s.margin(1) is s.margin(1, 0) is s.margin(1, 1)


def test_margin_structure_resolution_table():
    assert PMCModel.from_dict(_state_raw()).margin_structure == "state"
    assert PMCModel.from_dict(_pair_raw()).margin_structure == "pair"
    assert PMCModel.from_dict(_pair_raw(margin_structure="pair")).margin_structure == "pair"
    assert PMCModel.from_dict(_pair_raw(margin_structure="state")).margin_structure == "state"
    raw = _state_raw()
    raw["model"]["margin_structure"] = "pair"
    with pytest.raises(ValueError, match="pair"):
        PMCModel.from_dict(raw)
    raw["model"]["margin_structure"] = "joint"
    with pytest.raises(ValueError, match="margin_structure"):
        PMCModel.from_dict(raw)


def test_hidden_markov_variant_refusal_cites_the_proposition():
    raw = _pair_raw(variant="PMC-IN")
    raw["model"]["variant"] = "HMC-IN2"
    raw["prior"] = {"A": [[0.9, 0.1], [0.1, 0.9]]}
    with pytest.raises(ValueError, match="Proposition"):
        PMCModel.from_dict(raw)
    raw["model"]["margin_structure"] = "state"                         # explicit collapse is fine
    assert PMCModel.from_dict(raw).margin_structure == "state"


def test_variant_margin_properties():
    assert {v for v in Variant if v.allows_pair_margins} == {Variant.PMC, Variant.PMC_IN}
    assert all(v.per_class_margin == (not v.allows_pair_margins) for v in Variant)


def test_pair_notice_is_logged_once_and_only_when_implicit(monkeypatch, caplog):
    monkeypatch.setattr(model_mod, "_pair_notice_logged", False)
    with caplog.at_level(logging.INFO, logger="pmcprg.pmc.model"):
        PMCModel.from_dict(_pair_raw(margin_structure="pair"))
        assert not [r for r in caplog.records if "pair-indexed" in r.message]
        PMCModel.from_dict(_pair_raw())
        PMCModel.from_dict(_pair_raw())
    notices = [r for r in caplog.records if "pair-indexed" in r.message]
    assert len(notices) == 1 and notices[0].levelno == logging.INFO
    assert 'margin_structure = "state"' in notices[0].message


def test_explicit_state_collapse_is_the_k_format_model_bit_for_bit():
    tied = {(i, j): ((0.0, 1.0) if i == 0 else (1.1, 1.4)) for i in range(2) for j in range(2)}
    collapsed = PMCModel.from_dict(_pair_raw(margins=tied, margin_structure="state"))
    k_format = PMCModel.from_dict(_state_raw())
    as_pair = PMCModel.from_dict(_pair_raw(margins=tied))
    _, Y = simulate(k_format, N=300, seed=2)
    W_c, f_c = precompute_weights(collapsed, Y)
    W_k, f_k = precompute_weights(k_format, Y)
    np.testing.assert_array_equal(W_c, W_k)
    assert forward(collapsed, Y)[1] == forward(k_format, Y)[1]
    # Tied pair margins describe the same law through the pair code path.
    W_p, _ = precompute_weights(as_pair, Y)
    np.testing.assert_allclose(W_p, W_k, rtol=1e-12)
    assert forward(as_pair, Y)[1] == pytest.approx(forward(k_format, Y)[1], abs=1e-9)


def test_margin_blocks_order_and_candidates_round_trip(tmp_path):
    raw = _pair_raw()
    raw["margins"] = list(reversed(raw["margins"]))
    raw["margins"][1]["candidates"] = ["norm", "gamma"]                 # block (1, 0)
    m = PMCModel.from_dict(raw)
    blocks = m.margin_blocks()
    assert [(b["i"], b["j"]) for b in blocks] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert blocks[2]["candidates"] == ["norm", "gamma"]
    assert all("candidates" not in b for k, b in enumerate(blocks) if k != 2)
    m2 = PMCModel(m.save(tmp_path / "pair.toml"))
    assert m2.margin_structure == "pair" and m2.margin_blocks() == blocks
    raw["margins"][1]["candidates"] = []
    with pytest.raises(ValueError, match="candidates"):
        PMCModel.from_dict(raw)


def test_pair_fixture_is_the_csda_table1_setting():
    m = PMCModel(PAIR_FIXTURE)
    assert m.variant is Variant.PMC and m.margin_structure == "pair"
    np.testing.assert_allclose(m.prior_p, P_ORIG)
    for (i, j), (mu, sd) in TABLE1.items():
        assert m.margin(i, j).params == {"loc": mu, "scale": sd}
        assert m.copula(i, j).__class__.__name__.lower().endswith("clayton")
    X1, Y1 = simulate(m, N=200, seed=4)
    X2, Y2 = simulate(m, N=200, seed=4)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(Y1, Y2)


# --------------------------------------------------------------------------
# Pair margins with multivariate observations — supported for PMC-IN
# --------------------------------------------------------------------------

def _mvn_pair_raw(variant="PMC-IN"):
    means = {(0, 0): [0.0, 0.0], (0, 1): [0.5, -0.5], (1, 0): [1.5, 1.0], (1, 1): [2.0, 2.0]}
    covs = {(0, 0): [[1.0, 0.3], [0.3, 1.0]], (0, 1): [[2.0, 0.0], [0.0, 1.5]],
            (1, 0): [[1.2, -0.4], [-0.4, 1.0]], (1, 1): [[1.0, 0.0], [0.0, 1.0]]}
    return {
        "model": {"name": "mvn pair", "variant": variant, "K": 2, "d": 2},
        "prior": {"p": [[0.4, 0.1], [0.1, 0.4]]},
        "margins": [{"i": i, "j": j, "dist": "multivariate_normal",
                     "params": {"mean": means[(i, j)], "cov": covs[(i, j)]}}
                    for i in range(2) for j in range(2)],
    }


def test_pair_margins_with_multivariate_observations():
    m = PMCModel.from_dict(_mvn_pair_raw())
    assert m.margin_structure == "pair" and m.d == 2
    X, Y = simulate(m, N=5, seed=3)
    assert X.shape == (5,) and Y.shape == (5, 2)
    P = m.prior_p

    def f(i, j, y):
        b = m.margin(i, j).params
        return stats.multivariate_normal.pdf(y, b["mean"], b["cov"])

    Z = 0.0
    for path in itertools.product(range(2), repeat=5):
        pr = sum(P[path[0], j] * f(path[0], j, Y[0]) for j in range(2))
        for n in range(4):
            i, j = path[n], path[n + 1]
            pr *= P[i, j] * f(i, j, Y[n]) / sum(P[i, k] * f(i, k, Y[n]) for k in range(2))
            pr *= f(j, i, Y[n + 1])
        Z += pr
    assert forward(m, Y)[1] == pytest.approx(math.log(Z), abs=1e-10)
    raw = _mvn_pair_raw()
    raw["model"]["variant"] = "PMC"
    raw["copulas"] = [{"i": i, "j": j, "name": "Gauss", "tau": 0.3} for i in range(2) for j in range(2)]
    with pytest.raises(ValueError, match="scalar margins"):
        PMCModel.from_dict(raw)


# --------------------------------------------------------------------------
# Inference on pair models
# --------------------------------------------------------------------------

def _exact_path_posterior(raw, Y):
    """P(x_{1:N} | y_{1:N}) for every path, from DerrodePieczynski_CSDA2013
    Eqs. 12–14."""
    m = PMCModel.from_dict(raw)
    P = m.prior_p
    uses_cop = m.variant.uses_copula
    post = {}
    for path in itertools.product(range(2), repeat=len(Y)):
        pr = sum(P[path[0], j] * m.pdf(path[0], j, Y[0]) for j in range(2))
        for n in range(len(Y) - 1):
            i, j = path[n], path[n + 1]
            pr *= P[i, j] * m.pdf(i, j, Y[n]) / sum(P[i, k] * m.pdf(i, k, Y[n]) for k in range(2))
            pr *= m.pdf(j, i, Y[n + 1])
            if uses_cop:
                pr *= float(m.copula(i, j).pdf([m.cdf(i, j, Y[n]), m.cdf(j, i, Y[n + 1])]))
        post[path] = pr
    Z = sum(post.values())
    return {k: v / Z for k, v in post.items()}


@pytest.mark.parametrize("variant", ["PMC", "PMC-IN"])
def test_sample_posterior_draws_paths_from_the_exact_posterior(variant):
    raw = _pair_raw(variant=variant, tau=0.4, p=[[0.35, 0.15], [0.15, 0.35]])
    m = PMCModel.from_dict(raw)
    exact = _exact_path_posterior(raw, Y6)
    assert max(exact.values()) < 0.3                                   # a spread-out posterior
    W, f = precompute_weights(m, Y6)
    a, _ = forward(m, Y6, W=W, f_pdf=f)
    rng = np.random.default_rng(12)
    n_draws = 20_000
    counts = dict.fromkeys(exact, 0)
    for _ in range(n_draws):
        counts[tuple(int(x) for x in sample_posterior(m, Y6, rng, W=W, f_pdf=f, alpha_hat=a))] += 1
    big = [k for k in exact if exact[k] * n_draws >= 5]
    small = [k for k in exact if k not in big]
    obs = [counts[k] for k in big]
    exp = [exact[k] * n_draws for k in big]
    if small:                                                          # pooled rare paths
        obs.append(sum(counts[k] for k in small))
        exp.append(sum(exact[k] for k in small) * n_draws)
    assert stats.chisquare(obs, exp).pvalue > 1e-3
    tv = 0.5 * sum(abs(counts[k] / n_draws - exact[k]) for k in exact)
    assert tv < 0.03, tv


def test_smoothing_on_pair_model_is_the_exact_marginal_posterior():
    raw = _pair_raw()
    m = PMCModel.from_dict(raw)
    exact = _exact_path_posterior(raw, Y6)
    gamma = np.zeros((6, 2))
    for path, pr in exact.items():
        gamma[np.arange(6), path] += pr
    W, f = precompute_weights(m, Y6)
    a, _ = forward(m, Y6, W=W, f_pdf=f)
    b = backward(m, Y6, W=W)
    np.testing.assert_allclose(smooth(a, b), gamma, atol=1e-12)
    xi = joint_posteriors(a, W, b)
    np.testing.assert_allclose(xi.sum(axis=2), gamma[:-1], atol=1e-12)


_WEIGHT_CASES = [pytest.param(p, id=p.name) for p in sorted(MODELS_DIR.glob("*.toml"))] + [
    pytest.param(_pair_raw(variant="PMC-IN"), id="pair-PMC-IN"),
    pytest.param(_pair_raw(copula="Gauss", tau=0.4), id="pair-PMC-Gauss"),
]


@pytest.mark.parametrize("source", _WEIGHT_CASES)
def test_weight_is_consistent_with_precompute_weights(source):
    m = PMCModel(source) if isinstance(source, Path) else PMCModel.from_dict(source)
    if m.d > 1:
        pytest.skip("weight() takes scalar observations")
    _, Y = simulate(m, N=25, seed=1)
    W, f_pdf = precompute_weights(m, Y)
    P = m.prior_p
    for n in range(len(Y) - 1):
        for i in range(m.K):
            D = sum(P[i, k] * m.pdf(i, k, Y[n]) for k in range(m.K))
            np.testing.assert_allclose(f_pdf[n, i], [m.pdf(i, k, Y[n]) for k in range(m.K)],
                                       rtol=1e-12)
            for j in range(m.K):
                w = m.weight(i, j, float(Y[n]), float(Y[n + 1]))
                if m.variant.has_markov_prior:
                    expected = w
                else:                  # D = 0: y_n outside every f_ik support → W = 0
                    expected = w / D if D > 0 else 0.0
                assert W[n, i, j] == pytest.approx(expected, rel=1e-9, abs=1e-300), (n, i, j)


@pytest.mark.parametrize("variant", ["PMC", "PMC-IN"])
def test_log_weights_are_the_log_of_the_linear_weights_for_pair_margins(variant):
    m = PMCModel.from_dict(_pair_raw(variant=variant))
    _, Y = simulate(m, N=200, seed=8)
    W, f_pdf = precompute_weights(m, Y)
    logW, log_a1 = _log_transition_weights(m, Y)
    np.testing.assert_allclose(np.exp(logW), W, rtol=1e-11)
    np.testing.assert_allclose(np.exp(log_a1), np.einsum("ij,ij->i", m.prior_p, f_pdf[0]),
                               rtol=1e-12)


def test_pmc_in_pair_simulator_draws_the_pair_margins():
    m = PMCModel.from_dict(_pair_raw(variant="PMC-IN", p=[[0.35, 0.15], [0.15, 0.35]]))
    X, Y = simulate(m, N=20_000, seed=21)
    for (i, j), (mu, sd) in TABLE1.items():
        left = Y[:-1][(X[:-1] == i) & (X[1:] == j)]
        assert stats.kstest(left, stats.norm(mu, sd).cdf).pvalue > 1e-3, (i, j)
