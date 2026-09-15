"""General PMC — pseudo-observations and margin samples of the diagnostics.

For the pair margins of a general PMC (A16 Eqs. 12–14) the pair density is
f_ij(y_n) · f_ji(y_{n+1}) · c_ij(F_ij(y_n), F_ji(y_{n+1})). The diagnostics
therefore test

* the copula c_ij on (F_ij(y_n), F_ji(y_{n+1})) weighted by ξ_n(i, j);
* the margin f_ij in the dual view: F_ij(y_n) weighted by ξ_n(i, j) and
  F_ij(y_{n+1}) weighted by ξ_n(j, i);

and, for a state model, reduce to the per-state construction value for value.
Each construction is checked here against a direct computation on a short
series — scipy CDFs and ξ by enumerating every state path — and the PMM
(A16 §3.3) against the pair density written out by hand.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

from pmcprg.diagnostics import (
    copula_pseudo_obs, margin_cdfs, margin_keys, margin_of, margin_pit_dual,
    margin_sample,
)
from pmcprg.pmc.inference import (
    backward, forward, joint_posteriors, precompute_weights, smooth,
)
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.pmm import classify_pmm, simulate_pmm
from pmcprg.pmc.simulate import simulate

STATE_FIXTURE = "pmcprg/pmc/models/pmc_gauss_k2.toml"

# CSDA-2013 Table 1, Gaussian margins (σ = standard deviation), classes 0/1.
TABLE1 = {(0, 0): (0.0, 1.00), (0, 1): (0.3, 1.60), (1, 0): (1.1, 1.40), (1, 1): (1.5, 1.00)}
P_ORIG = [[0.50, 0.05], [0.05, 0.40]]
P_BAL = [[0.35, 0.15], [0.15, 0.35]]
CLIP = 1e-12
Y_SHORT = np.array([0.2, -0.4, 1.9, 1.1, 2.4, 0.6])


def _pair_raw(copula="Clayton", tau=0.7, p=P_ORIG):
    return {
        "model": {"name": "general PMC", "variant": "PMC", "K": 2, "N_default": 500},
        "prior": {"p": [list(r) for r in p]},
        "margins": [{"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
                    for (i, j), (mu, sd) in sorted(TABLE1.items())],
        "copulas": [{"i": i, "j": j, "name": copula, "tau": tau}
                    for i in range(2) for j in range(2)],
    }


def _F(i, j, y):
    """F_ij(y) from scipy, clipped like the diagnostics."""
    return float(np.clip(stats.norm.cdf(y, *TABLE1[(i, j)]), CLIP, 1.0 - CLIP))


def _xi_by_enumeration(model, Y):
    """ξ_n(i, j) = P(x_n = i, x_{n+1} = j | y) summed over every state path.

    p(x, y) = Σ_j p_{x1 j} f_{x1 j}(y_1) · Π_n p_ij f_ij(y_n) / Σ_k p_ik f_ik(y_n)
              · f_ji(y_{n+1}) · c_ij(F_ij(y_n), F_ji(y_{n+1}))   (A16 Eqs. 12–14)
    """
    P = model.prior_p
    K, N = model.K, len(Y)

    def f(i, j, y):
        return stats.norm.pdf(y, *TABLE1[(i, j)])

    joint = {}
    for path in itertools.product(range(K), repeat=N):
        i0 = path[0]
        p = sum(P[i0, j] * f(i0, j, Y[0]) for j in range(K))
        for n in range(N - 1):
            i, j = path[n], path[n + 1]
            p *= P[i, j] * f(i, j, Y[n]) / sum(P[i, k] * f(i, k, Y[n]) for k in range(K))
            p *= f(j, i, Y[n + 1]) * float(model.copula(i, j).pdf(
                [stats.norm.cdf(Y[n], *TABLE1[(i, j)]),
                 stats.norm.cdf(Y[n + 1], *TABLE1[(j, i)])]))
        joint[path] = p
    Z = sum(joint.values())
    xi = np.zeros((N - 1, K, K))
    for path, p in joint.items():
        for n in range(N - 1):
            xi[n, path[n], path[n + 1]] += p / Z
    return xi


def _xi(model, Y):
    W, f = precompute_weights(model, Y)
    a, _ = forward(model, Y, W=W, f_pdf=f)
    b = backward(model, Y, W=W)
    return joint_posteriors(a, W, b), smooth(a, b)


@pytest.fixture(scope="module")
def pair_model():
    return PMCModel.from_dict(_pair_raw())


# --------------------------------------------------------------------------
# F[n, i, j] = F_ij(y_n)
# --------------------------------------------------------------------------

def test_margin_cdfs_of_a_pair_model_are_the_pair_cdfs(pair_model):
    F = margin_cdfs(pair_model, Y_SHORT)
    assert F.shape == (len(Y_SHORT), 2, 2)
    for n, y in enumerate(Y_SHORT):
        for (i, j) in TABLE1:
            assert F[n, i, j] == pytest.approx(_F(i, j, y), rel=1e-12, abs=1e-300)
    # Off-diagonal margins differ, so the index inversion below is observable.
    assert not np.allclose(F[:, 0, 1], F[:, 1, 0])


def test_margin_cdfs_of_a_state_model_broadcast_the_state_cdfs_bit_for_bit():
    m = PMCModel(STATE_FIXTURE)
    _, Y = simulate(m, N=50, seed=4)
    F = margin_cdfs(m, Y)
    for i in range(m.K):
        ref = np.clip(m.margin(i).cdf_vec(Y), CLIP, 1.0 - CLIP)
        for j in range(m.K):
            assert F[:, i, j].tobytes() == np.ascontiguousarray(ref).tobytes()


@pytest.mark.parametrize("structure", ["pair", "state"])
def test_margin_cdfs_clip_pair_and_broadcast_view(pair_model, structure):
    """``clip=(lo, hi)`` clips asymmetrically; ``broadcast=True`` returns the
    state CDFs as a read-only view — same floats — and a pair model a new array."""
    m = pair_model if structure == "pair" else PMCModel(STATE_FIXTURE)
    Y = np.concatenate((Y_SHORT, [np.nan, -9.0, 9.0]))
    F = margin_cdfs(m, Y, clip=None)
    want = np.clip(F, 0.2, 0.7)
    got = margin_cdfs(m, Y, clip=(0.2, 0.7))
    assert got.tobytes() == want.tobytes() and got.flags.writeable
    assert margin_cdfs(m, Y, clip=0.2).tobytes() == np.clip(F, 0.2, 1.0 - 0.2).tobytes()
    view = margin_cdfs(m, Y, clip=(0.2, 0.7), broadcast=True)
    assert np.ascontiguousarray(view).tobytes() == want.tobytes()
    assert view.flags.writeable == (structure == "pair")


def test_margin_keys_follow_the_structure(pair_model):
    assert margin_keys(PMCModel(STATE_FIXTURE)) == [0, 1]
    assert margin_keys(pair_model) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert margin_of(pair_model, (0, 1)) is pair_model.margin(0, 1)
    with pytest.raises(ValueError):
        margin_of(pair_model, 0)           # margin(i) is refused for pair margins


# --------------------------------------------------------------------------
# Copula pseudo-observations: (F_ij(y_n), F_ji(y_{n+1})), weight ξ_n(i, j)
# --------------------------------------------------------------------------

def test_copula_pseudo_obs_against_a_direct_computation(pair_model):
    Y = Y_SHORT
    xi_ref = _xi_by_enumeration(pair_model, Y)
    xi, _ = _xi(pair_model, Y)
    np.testing.assert_allclose(xi, xi_ref, atol=1e-12)
    F = margin_cdfs(pair_model, Y)
    for (i, j) in TABLE1:
        uv, w = copula_pseudo_obs(F, xi, i, j)
        assert uv.shape == (len(Y) - 1, 2)
        for n in range(len(Y) - 1):
            assert uv[n, 0] == pytest.approx(_F(i, j, Y[n]), rel=1e-12)
            assert uv[n, 1] == pytest.approx(_F(j, i, Y[n + 1]), rel=1e-12)   # f_ji: Eq. 12
            assert w[n] == pytest.approx(xi_ref[n, i, j], abs=1e-12)


def test_copula_pseudo_obs_on_a_selection_of_transitions(pair_model):
    xi, _ = _xi(pair_model, Y_SHORT)
    F = margin_cdfs(pair_model, Y_SHORT)
    sel = np.array([0, 2, 4])
    uv, w = copula_pseudo_obs(F, xi, 1, 0, sel)
    full_uv, full_w = copula_pseudo_obs(F, xi, 1, 0)
    np.testing.assert_array_equal(uv, full_uv[sel])
    np.testing.assert_array_equal(w, full_w[sel])


def test_copula_pseudo_obs_of_a_state_model_are_the_state_pseudo_obs():
    """The construction the GoF scripts used before pair margins, bit for bit."""
    m = PMCModel(STATE_FIXTURE)
    _, Y = simulate(m, N=80, seed=9)
    xi, _ = _xi(m, Y)
    f = np.empty((len(Y), m.K))
    for k in range(m.K):
        f[:, k] = m.margin(k).cdf_vec(Y)
    f = np.clip(f, CLIP, 1.0 - CLIP)
    F = margin_cdfs(m, Y)
    N = len(Y)
    for i in range(m.K):
        for j in range(m.K):
            uv, w = copula_pseudo_obs(F, xi, i, j)
            ref = np.column_stack((f[: N - 1, i], f[1:, j]))
            assert uv.tobytes() == ref.tobytes()
            assert w.tobytes() == np.asarray(xi[:, i, j], dtype=float).tobytes()


# --------------------------------------------------------------------------
# Margin adequacy, dual view
# --------------------------------------------------------------------------

def test_margin_pit_dual_against_a_direct_computation(pair_model):
    Y = Y_SHORT
    N = len(Y)
    xi_ref = _xi_by_enumeration(pair_model, Y)
    xi, gamma = _xi(pair_model, Y)
    F = margin_cdfs(pair_model, Y)
    total = 0.0
    left_mass = np.zeros((N - 1, 2))
    right_mass = np.zeros((N - 1, 2))
    for (i, j) in TABLE1:
        u, w = margin_pit_dual(F, xi, i, j)
        assert u.shape == w.shape == (2 * (N - 1),)
        for n in range(N - 1):
            # left observation of the pair (i, j) ...
            assert u[n] == pytest.approx(_F(i, j, Y[n]), rel=1e-12)
            assert w[n] == pytest.approx(xi_ref[n, i, j], abs=1e-12)
            # ... and right observation of the pair (j, i), whose right margin is f_ij
            assert u[N - 1 + n] == pytest.approx(_F(i, j, Y[n + 1]), rel=1e-12)
            assert w[N - 1 + n] == pytest.approx(xi_ref[n, j, i], abs=1e-12)
        total += w.sum()
        left_mass[:, i] += w[: N - 1]
        right_mass[:, i] += w[N - 1:]
    assert total == pytest.approx(2 * (N - 1))
    np.testing.assert_allclose(left_mass, gamma[:-1], atol=1e-12)    # Σ_j ξ_n(i, j) = γ_n(i)
    np.testing.assert_allclose(right_mass, gamma[1:], atol=1e-12)    # Σ_j ξ_n(j, i) = γ_{n+1}(i)


def test_margin_pit_dual_is_uniform_under_the_true_model_and_the_inversion_matters():
    """The ξ-weighted PIT of the true f_ij is uniform; using f_ji on the right
    observation (no index inversion) is not, on the off-diagonal pairs."""
    m = PMCModel.from_dict(_pair_raw(tau=0.3, p=P_BAL))
    _, Y = simulate(m, N=20_000, seed=1)
    xi, _ = _xi(m, Y)
    F = margin_cdfs(m, Y)
    grid = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    def ecdf(u, w):
        return np.array([w[u <= t].sum() / w.sum() for t in grid])

    for (i, j) in TABLE1:
        u, w = margin_pit_dual(F, xi, i, j)
        assert np.abs(ecdf(u, w) - grid).max() < 0.05, (i, j)
        if i != j:
            u_wrong = np.concatenate((F[:-1, i, j], F[1:, j, i]))
            assert np.abs(ecdf(u_wrong, w) - grid).max() > 0.08, (i, j)


def test_margin_sample_against_a_loop():
    rng = np.random.default_rng(0)
    Y = rng.normal(size=40)
    lab = rng.integers(0, 2, size=40)
    np.testing.assert_array_equal(margin_sample(Y, lab, 1), Y[lab == 1])
    for i, j in TABLE1:
        left = [Y[n] for n in range(39) if (lab[n], lab[n + 1]) == (i, j)]
        right = [Y[n + 1] for n in range(39) if (lab[n], lab[n + 1]) == (j, i)]
        np.testing.assert_array_equal(margin_sample(Y, lab, (i, j)), np.array(left + right))
    # Every observation but the two ends is used twice over the four pairs.
    assert sum(margin_sample(Y, lab, k).size for k in TABLE1) == 2 * 39


# --------------------------------------------------------------------------
# PMM with pair margins (A16 §3.3 with the pair density of Eq. 12)
# --------------------------------------------------------------------------

def test_classify_pmm_pair_matches_the_pair_density_written_out(pair_model):
    Y = np.array([0.1, 1.7, -0.6, 0.4, 2.2, 1.3])
    X_hat, ll = classify_pmm(pair_model, Y)
    P = pair_model.prior_p
    ll_ref = 0.0
    X_ref = []
    for k in range(3):
        y1, y2 = Y[2 * k], Y[2 * k + 1]
        joint = np.zeros((2, 2))
        for (i, j) in TABLE1:
            u, v = _F(i, j, y1), _F(j, i, y2)
            joint[i, j] = (P[i, j] * stats.norm.pdf(y1, *TABLE1[(i, j)])
                           * stats.norm.pdf(y2, *TABLE1[(j, i)])
                           * float(pair_model.copula(i, j).pdf([u, v])))
        ll_ref += math.log(joint.sum())
        X_ref += [int(np.argmax(joint.sum(axis=1))), int(np.argmax(joint.sum(axis=0)))]
    assert ll == pytest.approx(ll_ref, rel=1e-9)
    assert X_hat.tolist() == X_ref


def test_simulate_pmm_pair_uses_f_ij_left_and_f_ji_right():
    m = PMCModel.from_dict(_pair_raw(copula="Prod", tau=0.0, p=P_BAL))
    X, Y = simulate_pmm(m, n_pairs=40_000, seed=3)
    x1, x2, y1, y2 = X[0::2], X[1::2], Y[0::2], Y[1::2]
    for (i, j), _ in TABLE1.items():
        sel = (x1 == i) & (x2 == j)
        mu_l, sd_l = TABLE1[(i, j)]
        mu_r, sd_r = TABLE1[(j, i)]
        se = max(sd_l, sd_r) / math.sqrt(sel.sum())
        assert abs(y1[sel].mean() - mu_l) < 4 * se, (i, j)
        assert abs(y2[sel].mean() - mu_r) < 4 * se, (i, j)
        assert abs(y1[sel].std() - sd_l) < 0.05 * sd_l, (i, j)
        assert abs(y2[sel].std() - sd_r) < 0.05 * sd_r, (i, j)


def test_simulate_pmm_pair_with_a_copula_restores_better_than_chance(pair_model):
    X, Y = simulate_pmm(pair_model, n_pairs=1000, seed=0)
    X_hat, ll = classify_pmm(pair_model, Y)
    assert np.isfinite(ll)
    assert np.mean(X_hat != X) < 0.30
