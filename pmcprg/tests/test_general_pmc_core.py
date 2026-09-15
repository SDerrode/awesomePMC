"""General pairwise Markov chain — pair-indexed margins f_ij (A16 Eqs. 12–14).

Acceptance tests of the first implementation wave (model, simulation,
inference). Since v0.5.0 the package collapsed the K² margins of a PMC to K
state margins, f_ij = f_i, attributing it to reversibility. Derrode &
Pieczynski (2013, §2.1, Proposition) state the opposite: for a stationary
reversible PMC, "X is a Markov chain" is *equivalent* to
p(y2 | x1, x2) = p(y2 | x2). With state margins the transition
p(x_{n+1} = j | x_n = i, y_n) ∝ p_ij f_ij(y_n) no longer depends on y_n: the
package's "PMC" was an SR hidden Markov chain with dependent noise (HMC-DN).

Contract under test
-------------------
* ``[[margins]]`` in K-format (key ``i``) → ``margin_structure == "state"``,
  unchanged behaviour (regression values pinned below);
* K²-format (keys ``i``, ``j``) → ``margin_structure == "pair"``: the K²
  densities are kept, ``margin(i, j)`` is f_ij, and ``margin(i)`` without
  ``j`` is refused (it is ambiguous);
* ``[model].margin_structure = "state"`` forces the former collapse;
* pair margins exist only for PMC and PMC-IN (for HMC-* variants X is
  Markov, so the Proposition forces state margins);
* the forward–backward pass is exact for pair margins (brute-force
  enumeration), the simulator follows Eqs. 13–14, and supervised restoration
  of the CSDA-2013 Table 1 setting reproduces an independent implementation.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from pmcprg.pmc.inference import (
    backward, classify, forward, joint_posteriors, precompute_weights, smooth,
)
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

FIXTURE = "pmcprg/pmc/models/pmc_gauss_k2.toml"

# CSDA-2013 Table 1, Gaussian margins (σ read as standard deviation), states
# 0/1 for the paper's 1/2.
TABLE1 = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}
P_ORIG = [[0.50, 0.05], [0.05, 0.40]]


def _pair_raw(copula="Clayton", tau=0.7, variant="PMC", margins=TABLE1, p=P_ORIG):
    raw = {
        "model": {"name": "general PMC", "variant": variant, "K": 2, "N_default": 500},
        "prior": {"p": [list(r) for r in p]},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                    for (i, j), (mu, sd) in sorted(margins.items())],
    }
    if variant == "PMC":
        raw["copulas"] = [{"i": i, "j": j, "name": copula, "tau": tau}
                          for i in range(2) for j in range(2)]
    return raw


# --------------------------------------------------------------------------
# Model contract
# --------------------------------------------------------------------------

def test_k_format_is_state_structured_and_unchanged():
    m = PMCModel(FIXTURE)
    assert m.margin_structure == "state"
    assert m.margin(0, 1) is m.margin(0) and m.margin(1, 0) is m.margin(1)


def test_k2_format_keeps_pair_margins():
    m = PMCModel.from_dict(_pair_raw())
    assert m.margin_structure == "pair"
    for (i, j), (mu, sd) in TABLE1.items():
        mg = m.margin(i, j)
        assert mg.pdf(mu) == pytest.approx(stats.norm.pdf(mu, mu, sd))
    assert m.margin(0, 1) is not m.margin(0, 0)
    with pytest.raises((ValueError, TypeError)):
        m.margin(0)


def test_explicit_state_structure_collapses_k2_margins():
    raw = _pair_raw()
    raw["model"]["margin_structure"] = "state"
    m = PMCModel.from_dict(raw)
    assert m.margin_structure == "state"
    assert m.margin(0, 1) is m.margin(0, 0)


@pytest.mark.parametrize("variant", ["HMC-IN", "HMC-IN2", "HMC-DN"])
def test_pair_margins_are_refused_for_hidden_markov_variants(variant):
    raw = _pair_raw(variant="PMC")
    raw["model"]["variant"] = variant
    if variant != "HMC-DN":
        raw.pop("copulas")
    raw["prior"] = {"A": [[0.9, 0.1], [0.1, 0.9]]} if variant.startswith("HMC") else raw["prior"]
    with pytest.raises(ValueError):
        PMCModel.from_dict(raw)


def test_pair_model_round_trips_through_toml(tmp_path):
    m = PMCModel.from_dict(_pair_raw())
    path = m.save(tmp_path / "pair.toml")
    m2 = PMCModel(path)
    assert m2.margin_structure == "pair"
    for (i, j), (mu, sd) in TABLE1.items():
        assert m2.margin(i, j).cdf(0.7) == pytest.approx(stats.norm.cdf(0.7, mu, sd))
    assert len(m2.margin_blocks()) == 4


# --------------------------------------------------------------------------
# Regression: state-structured models are bit-identical to v0.8
# --------------------------------------------------------------------------

def test_state_model_inference_is_unchanged():
    m = PMCModel(FIXTURE)
    X, Y = simulate(m, N=300, seed=5)
    assert X[:12].tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert float(Y.sum()) == pytest.approx(-22.668502632876308, abs=1e-9)
    W, f = precompute_weights(m, Y)
    a, ll = forward(m, Y, W=W, f_pdf=f)
    b = backward(m, Y, W=W)
    g = smooth(a, b)
    xi = joint_posteriors(a, W, b)
    assert ll == pytest.approx(-347.0432754805682, abs=1e-9)
    np.testing.assert_allclose(g[17], [0.6120412387055064, 0.3879587612944936], atol=1e-12)
    np.testing.assert_allclose(xi[40], [[0.9447269716161424, 0.00919098461461979],
                                        [0.002627208239631123, 0.043454835529606856]], atol=1e-12)


# --------------------------------------------------------------------------
# Inference is exact for pair margins
# --------------------------------------------------------------------------

def _brute_force(raw, Y):
    """log p(y), γ, ξ by enumerating every state path, from Eqs. 12–14."""
    P = np.array(raw["prior"]["p"])
    marg = {(b["i"], b["j"]): (b["params"]["loc"], b["params"]["scale"]) for b in raw["margins"]}
    m = PMCModel.from_dict(raw)
    uses_cop = raw["model"]["variant"] == "PMC"
    N, K = len(Y), 2

    def f(i, j, y):
        return stats.norm.pdf(y, *marg[(i, j)])

    def F(i, j, y):
        return stats.norm.cdf(y, *marg[(i, j)])

    joint = {}
    for path in itertools.product(range(K), repeat=N):
        i0 = path[0]
        p = sum(P[i0, j] * f(i0, j, Y[0]) for j in range(K))           # p(x1, y1)
        for n in range(N - 1):
            i, j = path[n], path[n + 1]
            trans = P[i, j] * f(i, j, Y[n]) / sum(P[i, k] * f(i, k, Y[n]) for k in range(K))
            dens = f(j, i, Y[n + 1])
            if uses_cop:
                u, v = F(i, j, Y[n]), F(j, i, Y[n + 1])
                dens *= float(m.copula(i, j).pdf([u, v]))
            p *= trans * dens
        joint[path] = p
    Z = sum(joint.values())
    gamma = np.zeros((N, K)); xi = np.zeros((N - 1, K, K))
    for path, p in joint.items():
        for n in range(N):
            gamma[n, path[n]] += p / Z
        for n in range(N - 1):
            xi[n, path[n], path[n + 1]] += p / Z
    return math.log(Z), gamma, xi


@pytest.mark.parametrize("variant,copula", [("PMC", "Clayton"), ("PMC", "Gauss"), ("PMC-IN", None)])
def test_forward_backward_is_exact_for_pair_margins(variant, copula):
    raw = _pair_raw(copula=copula or "Clayton", variant=variant)
    m = PMCModel.from_dict(raw)
    Y = np.array([0.2, -0.4, 1.9, 1.1, 2.4, 0.6])
    ll_ref, g_ref, xi_ref = _brute_force(raw, Y)
    W, f = precompute_weights(m, Y)
    a, ll = forward(m, Y, W=W, f_pdf=f)
    b = backward(m, Y, W=W)
    assert ll == pytest.approx(ll_ref, abs=1e-10)
    np.testing.assert_allclose(smooth(a, b), g_ref, atol=1e-12)
    np.testing.assert_allclose(joint_posteriors(a, W, b), xi_ref, atol=1e-12)


def test_log_space_pass_agrees_with_linear_pass_for_pair_margins():
    from pmcprg.pmc.inference import _backward_log_space, _forward_log_space
    m = PMCModel.from_dict(_pair_raw())
    _, Y = simulate(m, N=400, seed=3)
    W, f = precompute_weights(m, Y)
    a, ll = forward(m, Y, W=W, f_pdf=f)
    a_log, ll_log = _forward_log_space(m, Y)
    assert ll_log == pytest.approx(ll, abs=1e-8)
    np.testing.assert_allclose(a_log, a, atol=1e-10)
    np.testing.assert_allclose(_backward_log_space(m, Y), backward(m, Y, W=W), atol=1e-10)


# --------------------------------------------------------------------------
# The simulator follows Eqs. 13–14
# --------------------------------------------------------------------------

def test_simulated_pairs_have_the_pair_margins_and_prior():
    m = PMCModel.from_dict(_pair_raw(copula="Prod", tau=0.0, p=[[0.35, 0.15], [0.15, 0.35]]))
    X, Y = simulate(m, N=60_000, seed=7)
    counts = np.zeros((2, 2))
    np.add.at(counts, (X[:-1], X[1:]), 1)
    np.testing.assert_allclose(counts / counts.sum(), [[0.35, 0.15], [0.15, 0.35]], atol=0.01)
    for (i, j), (mu, sd) in TABLE1.items():
        left = Y[:-1][(X[:-1] == i) & (X[1:] == j)]                   # y_n given (x_n, x_{n+1}) = (i, j)
        right = Y[1:][(X[:-1] == j) & (X[1:] == i)]                   # y_{n+1} given (j, i): also f_ij
        assert stats.kstest(left, stats.norm(mu, sd).cdf).pvalue > 1e-3, (i, j, "left")
        assert stats.kstest(right, stats.norm(mu, sd).cdf).pvalue > 1e-3, (i, j, "right")


def test_hidden_process_is_not_markov_with_pair_margins():
    """P(x_{n+1} = 1 | x_n = 0, y_n) must depend on y_n (Eq. 13)."""
    m = PMCModel.from_dict(_pair_raw(copula="Prod", tau=0.0, p=[[0.35, 0.15], [0.15, 0.35]]))
    X, Y = simulate(m, N=80_000, seed=11)
    sel = X[:-1] == 0
    low = np.mean(X[1:][sel & (np.abs(Y[:-1] - 0.0) < 0.5)] == 1)
    high = np.mean(X[1:][sel & (np.abs(Y[:-1]) > 2.5)] == 1)
    # f_01 = N(0.3, 1.6) is wide, f_00 = N(0, 1) is narrow: far from 0 the switch is far more likely.
    assert high > low + 0.2, (low, high)


# --------------------------------------------------------------------------
# CSDA-2013 Table 1 setting — agrees with an independent implementation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("copula,tau,lo,hi", [
    ("Prod", 0.0, 8.5, 12.0),     # independent implementation: 10.18 % (sd 1.07), paper 11.06 %
    ("Clayton", 0.7, 7.5, 11.0),     # independent implementation: 9.13 % (sd 1.32), paper 6.31 %
])
def test_supervised_error_rate_matches_independent_implementation(copula, tau, lo, hi):
    m = PMCModel.from_dict(_pair_raw(copula=copula, tau=tau))
    errs = []
    for r in range(12):
        X, Y = simulate(m, N=2000, seed=1000 + r)
        errs.append(100 * np.mean(classify(m, Y)[0] != X))
    assert lo < np.mean(errs) < hi, errs
