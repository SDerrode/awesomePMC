"""State-weighted empirical margins for the copula step of ICE/SEM (FR-7 a).

Two groups of tests.

**Mechanics** — the option's semantics, pinned against hand-computed numbers
and against the parametric path:

* ``_weighted_ecdf`` on a toy with explicit weights, a tie and the ``+ 1``;
* the state margin is the γ-weighted rescaled ECDF and the pair margin the
  ½ξ-weighted dual-view one (:func:`pmcprg.pmc.ice._empirical_margin_cdfs`),
  both hand-computed, both strictly inside (0, 1);
* pooling the pair weights over j returns γ (the derivation's consistency
  check between the two margin structures);
* one-hot weights (SEM's draw, the k-means labels) give exactly the classical
  ``#{n : x_n = i, y_n ≤ y} / (n_i + 1)`` of the sub-sample;
* the copula step — τ fit *and* family scores, i.e. the one
  ``_select_and_fit_copula`` call — receives those pseudo-observations, while
  the prior, the margin update and the E-step are bit-identical to the
  parametric run;
* the key is absent by default and the default path is unchanged *in this
  process* (no stored golden: the bit-exact cross-run comparison already
  lives in ``test_estim_complete_data_identity.py``);
* an invalid value raises; SEM shares the option; the missing-data paths
  (``"available"``, ``"impute"``) and the k-means warm start honour it;
* a non-default value is recorded in the fitted model's ``[ice]`` / ``[sem]``
  table, and the three standard-error modules of FR-4 refuse such a model.

**Monte Carlo** (slow) — what the option buys and what it costs, on a K = 2
PMC with N(±1, 1) state margins, Gaussian copulas (τ = 0.5 on the diagonal
pairs, 0 off-diagonal), N = 2000, ICE started from the truth with
``candidates = ["Gauss"]``. Contamination: 1 % of the y_n replaced by
``μ_{x_n} ± 7σ``, *away* from the other state (so the E-step cannot re-label
them), i.e. a true-margin CDF of 1.3e-12 — RB-5's extreme pseudo-observation
transplanted into observation space. Measured with the seeds below (R = 50,
bias and RMSE of τ̂ on the two diagonal pairs):

*Margins re-estimated inside ICE* (``fit_margins=True`` — the unsupervised
case, and the one the audit's "IFM is not robust" is about): parametric τ̂
bias **+0.258/+0.251**, RMSE 0.286/0.277, and the run still drifting at
``max_iter = 50``; empirical **+0.100/+0.051**, RMSE 0.176/0.203 — RMSE
ratio empirical/parametric **0.62/0.73**. Its own spread is large: the
empirical fits are not uniformly good, they are uniformly *less* biased.
With the outliers pushed *down* for both states instead of away, parametric
+0.345/+0.137 against empirical +0.128/+0.126 (ratio 0.57/0.92). The
mechanism is **not** the outliers' own extreme pseudo-observations but the
scale they add to the fitted margins: σ̂ grows by ~20 %, every clean y_n's
pseudo-observation is pulled toward ½, and a Gaussian copula fitted on
compressed normal scores reports a larger τ. Ranks do not see that scale.

*Margins held at the truth* (``fit_margins=False``): the parametric copula
step is **not** visibly hurt (bias +0.018/+0.018) and the empirical one is
*worse* (+0.091/+0.087). Two honest findings behind that: inside ICE the
E-step re-routes a contaminated pair into the blocks that tolerate it — the
off-diagonal pairs, whose τ̂ absorbs the damage instead (bias +0.15
parametric, +0.29 empirical there) — so RB-5's 10⁻¹² coordinates never reach
the diagonal blocks; and the empirical margins, which are built from *all* of
a state's observations (weight γ) while a diagonal block keeps only the pairs
the E-step left in it, put the outliers at the bottom ranks of F̂_i and
thereby compress the clean points' ranks, biasing τ̂ upward. The option is for
the case its literature is about — margins estimated from the same
contaminated data — not for oracle margins.

*Clean, correctly specified data* — efficiency cost, RMSE ratio empirical/
parametric: **0.99/1.00** with ``fit_margins=True`` (for the Gaussian copula
the rank estimator is semiparametrically efficient, and the parametric path
pays for its own estimated margins) and **1.64/1.46** with the margins fixed
at the truth, where the parametric path is an oracle (the i.i.d.
known-margin/rank variance ratio for a Gaussian copula at τ = 0.5 is
1 + ρ² = 1.5, i.e. 1.22 in RMSE; the rest is the posterior weighting).

*A second family* — the same design with **Clayton** copulas (τ = 0.5
diagonal, 0.05 off-diagonal, R = 40), where the pseudo-likelihood is *not*
efficient: clean-data RMSE ratio 1.05/1.07 with fitted margins and
3.38/2.12 with oracle ones; under contamination with fitted margins the bias
goes from +0.103/+0.143 (parametric) to +0.078/+0.065 (empirical), with the
RMSE ratio split 1.24/0.51 — the gain is in bias, not always in RMSE.

Seeds: ``zlib.crc32(repr(parts).encode())``, never ``hash()`` (salted per
process).
"""

from __future__ import annotations

import importlib
import zlib

import numpy as np
import pytest

from pmcprg.numerics import EPS, ONE_MINUS_EPS
from pmcprg.pmc._estim_common import (
    copula_margin_defaults,
    ice_estim_defaults,
    sem_estim_defaults,
)
from pmcprg.pmc.ice import (
    COPULA_MARGIN_MODES,
    DEFAULT_COPULA_MARGINS,
    _empirical_margin_cdfs,
    _m_step,
    _pair_margin_sample,
    _parse_ice_cfg,
    _weighted_ecdf,
    ice,
)
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.sem import sem
from pmcprg.pmc.simulate import simulate

ICE = importlib.import_module("pmcprg.pmc.ice")

MU = (-1.0, 1.0)
SD = (1.0, 1.0)
TAU = 0.5


def seed_of(*parts) -> int:
    """Deterministic seed — crc32 of the parts' repr (never ``hash()``)."""
    return zlib.crc32(repr(parts).encode())


def state_raw(tau_off: float = 0.0, family: str = "Gauss", N: int = 2000) -> dict:
    return {
        "model": {"name": "fr7a", "variant": "PMC", "K": 2, "N_default": N},
        "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": k, "dist": "norm", "params": {"loc": MU[k], "scale": SD[k]}}
                    for k in range(2)],
        "copulas": [{"i": i, "j": j, "name": family, "tau": TAU if i == j else tau_off}
                    for i in range(2) for j in range(2)],
    }


def pair_raw() -> dict:
    raw = state_raw()
    raw["model"]["margin_structure"] = "pair"
    raw["margins"] = [
        {"i": i, "j": j, "dist": "norm",
         "params": {"loc": MU[i] + 0.3 * j, "scale": SD[i]}}
        for i in range(2) for j in range(2)
    ]
    return raw


# ---------------------------------------------------------------------------
# The formula: hand-computed weighted empirical CDFs
# ---------------------------------------------------------------------------

def test_weighted_ecdf_hand_computed_with_a_tie_and_the_plus_one():
    y = np.array([0.3, -1.0, 0.3, 2.0, 0.5])
    w = np.array([0.5, 1.0, 0.25, 0.0, 0.75])       # Σw = 2.5 → denominator 3.5
    got = _weighted_ecdf(y, w, np.array([-2.0, -1.0, 0.0, 0.3, 0.4, 0.5, 2.0, 10.0]))
    want = np.array([0.0,                 # below every point
                     1.0 / 3.5,           # the only point ≤ −1
                     1.0 / 3.5,
                     (1.0 + 0.5 + 0.25) / 3.5,   # both tied 0.3 count (≤)
                     1.75 / 3.5,
                     2.5 / 3.5,
                     2.5 / 3.5,           # the 2.0 carries weight 0
                     2.5 / 3.5])
    assert np.allclose(got, want, rtol=0, atol=1e-15)
    assert got.max() < 1.0                          # the "+ 1" keeps F̂ < 1


def test_weighted_ecdf_of_an_empty_sample_is_zero():
    assert _weighted_ecdf(np.array([]), np.array([]), np.array([0.0, 1.0])).tolist() == [0.0, 0.0]


def test_state_empirical_margin_is_the_gamma_weighted_ecdf():
    model = PMCModel.from_dict(state_raw(N=5))
    Y = np.array([0.3, -1.0, 0.3, 2.0, 0.5])
    gamma = np.zeros((5, 2))
    gamma[:, 0] = [0.5, 1.0, 0.25, 0.0, 0.75]
    gamma[:, 1] = 1.0 - gamma[:, 0]
    xi = np.zeros((4, 2, 2))
    F = _empirical_margin_cdfs(model, Y, xi, gamma)
    assert F.shape == (5, 2)
    # State 0, by hand (Σγ = 2.5): the two tied 0.3 share the value.
    assert F[1, 0] == pytest.approx(1.0 / 3.5)
    assert F[0, 0] == pytest.approx(F[2, 0]) == pytest.approx(1.75 / 3.5)
    assert F[4, 0] == pytest.approx(2.5 / 3.5)
    assert F[3, 0] == pytest.approx(2.5 / 3.5)
    # State 1 (Σγ = 2.5 as well, weights 0.5, 0.0, 0.75, 1.0, 0.25).
    assert F[1, 1] == pytest.approx(EPS)            # weight 0 at the minimum → clipped
    assert F[3, 1] == pytest.approx(2.5 / 3.5)
    assert np.all((F > 0.0) & (F < 1.0))


def test_pair_empirical_margin_uses_the_dual_view_xi_weights():
    model = PMCModel.from_dict(pair_raw())
    assert model.margin_structure == "pair"
    Y = np.array([0.0, 1.0, 2.0, 3.0])
    xi = np.zeros((3, 2, 2))
    xi[0] = [[0.6, 0.1], [0.2, 0.1]]
    xi[1] = [[0.2, 0.5], [0.1, 0.2]]
    xi[2] = [[0.1, 0.3], [0.4, 0.2]]
    gamma = np.zeros((4, 2))
    F = _empirical_margin_cdfs(model, Y, xi, gamma)
    assert F.shape == (4, 2, 2)
    # f_01: {y_n, ½ξ_n(0,1)} ∪ {y_{n+1}, ½ξ_n(1,0)} — by hand, y sorted 0…3:
    #   y0 = 0.0 : ½·0.1                       = 0.05
    #   y1 = 1.0 : ½·0.5 (left) + ½·0.2 (right) = 0.35
    #   y2 = 2.0 : ½·0.3 (left) + ½·0.1 (right) = 0.20
    #   y3 = 3.0 :               ½·0.4 (right)  = 0.20
    total = 0.05 + 0.35 + 0.20 + 0.20
    assert total == pytest.approx(0.5 * (xi[:, 0, 1].sum() + xi[:, 1, 0].sum()))
    cum = np.cumsum([0.05, 0.35, 0.20, 0.20])
    assert np.allclose(F[:, 0, 1], cum / (total + 1.0))
    assert np.all((F > 0.0) & (F < 1.0))
    # The v-coordinate of block (0, 1) is F̂_10 — the right margin of a pair
    # (0, 1) is f_10, whose sample holds y_{n+1} with weight ½ξ_n(0, 1).
    y_s, w_s = _pair_margin_sample(Y, xi, 1, 0)
    assert np.allclose(F[:, 1, 0], _weighted_ecdf(y_s, w_s, Y))


def test_pair_weights_pooled_over_j_give_the_state_weights():
    """The derivation's consistency check: Σ_j (dual-view weight of y_n in
    f_ij) = ½Σ_j ξ_n(i, j) + ½Σ_j ξ_{n−1}(j, i) = γ_n(i) for interior n.

    It needs genuine posteriors — the identity rests on both marginals of ξ
    being γ, which a random ξ does not satisfy — so the weights come from a
    real forward–backward pass.
    """
    N = 40
    model = PMCModel.from_dict(state_raw(N=N))
    _, Y = simulate(model, N=N, seed=seed_of("pool", 7))
    gamma, xi = _posteriors(model, Y)
    index = {float(y): n for n, y in enumerate(Y)}
    for i in range(model.K):
        pooled = np.zeros(N)
        for j in range(model.K):
            y_s, w_s = _pair_margin_sample(Y, xi, i, j)
            for y, w in zip(y_s, w_s):
                pooled[index[float(y)]] += w
        assert np.allclose(pooled[1:N - 1], gamma[1:N - 1, i])
        # The two end points get half the state weight (they are seen once).
        assert np.allclose(pooled[[0, N - 1]], 0.5 * gamma[[0, N - 1], i])


def test_one_hot_weights_give_the_classical_rescaled_subsample_ecdf():
    """SEM's draw / the k-means labels: F̂_i = #{x_n = i, y_n ≤ y} / (n_i + 1)."""
    model = PMCModel.from_dict(state_raw(N=200))
    _, Y = simulate(model, N=200, seed=seed_of("onehot", 1))
    labels = np.asarray(simulate(model, N=200, seed=seed_of("onehot", 2))[0])
    gamma = np.zeros((200, 2))
    gamma[np.arange(200), labels] = 1.0
    xi = np.zeros((199, 2, 2))
    xi[np.arange(199), labels[:-1], labels[1:]] = 1.0
    F = _empirical_margin_cdfs(model, Y, xi, gamma)
    for k in range(2):
        sub = Y[labels == k]
        want = np.array([(sub <= y).sum() / (sub.size + 1.0) for y in Y])
        assert np.allclose(F[:, k], np.clip(want, EPS, ONE_MINUS_EPS))


# ---------------------------------------------------------------------------
# Wiring: what the copula step sees, and what stays untouched
# ---------------------------------------------------------------------------

def _posteriors(model, Y):
    from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights, smooth
    W, f_pdf = precompute_weights(model, Y)
    a, _ = forward(model, Y, W=W, f_pdf=f_pdf)
    b = backward(model, Y, W=W)
    return smooth(a, b), joint_posteriors(a, W, b)


@pytest.mark.parametrize("raw_fn", [state_raw, pair_raw])
def test_copula_step_receives_the_empirical_pseudo_observations(raw_fn, monkeypatch):
    model = PMCModel.from_dict(raw_fn())
    _, Y = simulate(model, N=300, seed=seed_of("wiring", raw_fn.__name__))
    gamma, xi = _posteriors(model, Y)
    seen: dict = {}
    real = ICE._select_and_fit_copula

    def spy(candidates, u, v, weights, criterion=ICE.DEFAULT_SELECTION_CRITERION):
        seen[(len(seen))] = (u.copy(), v.copy())
        return real(candidates, u, v, weights, criterion=criterion)

    monkeypatch.setattr(ICE, "_select_and_fit_copula", spy)
    raw = model.raw
    _m_step(raw, model, Y, xi, gamma, fit_margins=False, candidates=["Gauss"],
            selection_criterion="mle", margin_selection_rule="mle",
            copula_margins="empirical")
    F = _empirical_margin_cdfs(model, Y, xi, gamma)
    blocks = [(int(b["i"]), int(b["j"])) for b in raw["copulas"]]
    for k, (ii, jj) in enumerate(blocks):
        u_want, v_want = ICE._pair_pseudo_obs(model, F, ii, jj)
        u_got, v_got = seen[k]
        assert np.array_equal(u_got, u_want)
        assert np.array_equal(v_got, v_want)


@pytest.mark.parametrize("raw_fn", [state_raw, pair_raw])
def test_only_the_copula_blocks_change(raw_fn):
    """Prior and (GICE) margin updates are bit-identical; only copulas move."""
    model = PMCModel.from_dict(raw_fn())
    _, Y = simulate(model, N=400, seed=seed_of("only-copulas", raw_fn.__name__))
    gamma, xi = _posteriors(model, Y)
    out = {}
    for mode in COPULA_MARGIN_MODES:
        raw = model.raw
        _m_step(raw, model, Y, xi, gamma, fit_margins=True, candidates=["Gauss"],
                selection_criterion="mle", margin_selection_rule="mle",
                copula_margins=mode)
        out[mode] = raw
    assert out["parametric"]["prior"] == out["empirical"]["prior"]
    assert out["parametric"]["margins"] == out["empirical"]["margins"]
    assert out["parametric"]["copulas"] != out["empirical"]["copulas"]


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_default_is_parametric_and_out_of_the_gui_widget_contract():
    assert DEFAULT_COPULA_MARGINS == "parametric"
    assert copula_margin_defaults() == {"copula_margins": "parametric"}
    # Keys of these two dicts must have a GUI widget (test_gui_dialogs.py).
    assert "copula_margins" not in set(ice_estim_defaults()) | set(sem_estim_defaults())


def test_toml_ice_table_and_ice_cfg_both_reach_the_estimator():
    raw = state_raw()
    raw["ice"] = {"copula_margins": "empirical"}
    model = PMCModel.from_dict(raw)
    assert _parse_ice_cfg(model, None)["copula_margins"] == "empirical"
    assert _parse_ice_cfg(model, {"copula_margins": "parametric"})["copula_margins"] == "parametric"
    assert _parse_ice_cfg(PMCModel.from_dict(state_raw()), None)["copula_margins"] == "parametric"


@pytest.mark.parametrize("bad", ["ranks", "Empirical", "", True, None, 1])
def test_invalid_value_raises(bad):
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=120, seed=seed_of("invalid", 0))
    with pytest.raises(ValueError, match="copula_margins"):
        ice(model, Y, {"copula_margins": bad, "max_iter": 1})
    with pytest.raises(ValueError, match="copula_margins"):
        sem(model, Y, {"copula_margins": bad, "max_iter": 1})


def test_m_step_refuses_an_invalid_value():
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=120, seed=seed_of("invalid-mstep", 0))
    gamma, xi = _posteriors(model, Y)
    with pytest.raises(ValueError, match="copula_margins"):
        _m_step(model.raw, model, Y, xi, gamma, fit_margins=False, candidates=["Gauss"],
                selection_criterion="mle", margin_selection_rule="mle", copula_margins="ranks")


# ---------------------------------------------------------------------------
# The default path is unchanged
# ---------------------------------------------------------------------------

def _ice_run(model, Y, cfg):
    fitted, trace = ice(model, Y, cfg)
    return fitted.raw, list(trace.log_liks), trace.tau_history.copy()


@pytest.mark.parametrize("raw_fn", [state_raw, pair_raw])
def test_absent_key_and_explicit_parametric_agree_exactly(raw_fn):
    """No stored golden (platform-dependent): both runs happen here and now."""
    model = PMCModel.from_dict(raw_fn())
    _, Y = simulate(model, N=400, seed=seed_of("identity", raw_fn.__name__))
    base = {"max_iter": 6, "fit_margins": True, "candidates": ["Gauss", "Clayton"]}
    raw_a, ll_a, tau_a = _ice_run(model, Y, dict(base))
    raw_b, ll_b, tau_b = _ice_run(model, Y, {**base, "copula_margins": "parametric"})
    assert raw_a == raw_b
    assert ll_a == ll_b
    assert np.array_equal(tau_a, tau_b)
    assert "copula_margins" not in PMCModel.from_dict(raw_a).ice_config()


def test_sem_default_path_is_unchanged():
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=400, seed=seed_of("identity-sem", 0))
    base = {"max_iter": 5, "sem_seed": 3, "candidates": ["Gauss"], "fit_margins": True}
    f_a, t_a = sem(model, Y, dict(base))
    f_b, t_b = sem(model, Y, {**base, "copula_margins": "parametric"})
    assert f_a.raw == f_b.raw
    assert t_a.log_liks == t_b.log_liks
    assert "copula_margins" not in f_a.sem_config()


# ---------------------------------------------------------------------------
# SEM, missing data, warm start — the option reaches every M-step
# ---------------------------------------------------------------------------

def test_sem_uses_the_option_and_records_it():
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=600, seed=seed_of("sem-empirical", 0))
    cfg = {"max_iter": 4, "sem_seed": 1, "candidates": ["Gauss"], "fit_margins": True}
    f_par, _ = sem(model, Y, dict(cfg))
    f_emp, _ = sem(model, Y, {**cfg, "copula_margins": "empirical"})
    assert f_emp.sem_config()["copula_margins"] == "empirical"
    assert f_par.copula(0, 0).params["tau_k"] != f_emp.copula(0, 0).params["tau_k"]


@pytest.mark.parametrize("strategy", ["available", "impute"])
def test_missing_data_paths_honour_the_option(strategy):
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=400, seed=seed_of("missing", strategy))
    Ym = Y.copy()
    rng = np.random.default_rng(seed_of("missing-rows", strategy))
    Ym[rng.choice(400, 40, replace=False)] = np.nan
    cfg = {"max_iter": 3, "candidates": ["Gauss"], "missing_strategy": strategy,
           "missing_draws": 2, "missing_seed": 0}
    f_par, _ = ice(model, Ym, dict(cfg))
    f_emp, _ = ice(model, Ym, {**cfg, "copula_margins": "empirical"})
    assert f_emp.ice_config()["copula_margins"] == "empirical"
    assert np.isfinite(f_emp.copula(0, 0).params["tau_k"])
    assert f_par.copula(0, 0).params["tau_k"] != f_emp.copula(0, 0).params["tau_k"]


def test_kmeans_warm_start_honours_the_option():
    pytest.importorskip("sklearn")
    from pmcprg.pmc.ice import _warmstart_from_kmeans
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=500, seed=seed_of("kmeans", 0))
    kw = dict(random_state=0, fit_margins=True, candidates=["Gauss"],
              selection_criterion="mle", margin_selection_rule="mle")
    par = _warmstart_from_kmeans(model, Y, **kw)
    emp = _warmstart_from_kmeans(model, Y, **kw, copula_margins="empirical")
    assert par.margin_blocks() == emp.margin_blocks()      # margins untouched
    assert par.copula(0, 0).params["tau_k"] != emp.copula(0, 0).params["tau_k"]


# ---------------------------------------------------------------------------
# Recording, and the standard errors that no longer apply
# ---------------------------------------------------------------------------

def test_ice_records_the_option_in_the_fitted_model():
    model = PMCModel.from_dict(state_raw())
    _, Y = simulate(model, N=300, seed=seed_of("record", 0))
    fitted, _ = ice(model, Y, {"max_iter": 2, "candidates": ["Gauss"],
                               "copula_margins": "empirical"})
    assert fitted.ice_config()["copula_margins"] == "empirical"
    # …and an explicit override of a TOML "empirical" is recorded too.
    raw = state_raw()
    raw["ice"] = {"copula_margins": "empirical"}
    back, _ = ice(PMCModel.from_dict(raw), Y,
                  {"max_iter": 2, "candidates": ["Gauss"], "copula_margins": "parametric"})
    assert back.ice_config()["copula_margins"] == "parametric"


@pytest.mark.parametrize("table", ["ice", "sem"])
def test_standard_error_modules_refuse_empirical_margins(table):
    from pmcprg.pmc._godambe import ice_godambe_tau_se
    from pmcprg.pmc._lystig_hughes import ice_lh_information
    from pmcprg.pmc._oakes import ice_oakes_tau_se
    raw = state_raw()
    raw[table] = {"copula_margins": "empirical"}
    model = PMCModel.from_dict(raw)
    _, Y = simulate(model, N=200, seed=seed_of("se-refuse", table))
    for fn in (ice_oakes_tau_se, ice_godambe_tau_se, ice_lh_information):
        with pytest.raises(NotImplementedError, match="copula_margins"):
            fn(model, Y)
    # The parametric default is not refused (these pilots run on this fixture).
    plain = PMCModel.from_dict(state_raw())
    assert ice_oakes_tau_se(plain, Y)


# ---------------------------------------------------------------------------
# Monte Carlo: robustness and efficiency (slow)
# ---------------------------------------------------------------------------

def _contaminate(X, Y, rng, eps=0.01, shift=7.0):
    """1 % of the y_n pushed 7σ *away* from the other state's mean."""
    Y = Y.copy()
    idx = rng.choice(len(Y), size=int(round(eps * len(Y))), replace=False)
    sign = np.where(np.asarray(X)[idx] == 0, -1.0, 1.0)
    Y[idx] = np.asarray(MU)[np.asarray(X)[idx]] + sign * shift * np.asarray(SD)[np.asarray(X)[idx]]
    return Y


def _tau_study(*, fit_margins: bool, R: int, N: int = 2000, family: str = "Gauss"):
    truth = PMCModel.from_dict(state_raw(family=family, N=N))
    out: dict = {}
    for r in range(R):
        X, Y = simulate(truth, N=N, seed=seed_of("fr7a-study", family, N, r))
        Yc = _contaminate(X, Y, np.random.default_rng(seed_of("fr7a-study-cont", family, N, r)))
        for data_name, data in (("clean", Y), ("cont", Yc)):
            for mode in COPULA_MARGIN_MODES:
                fitted, _ = ice(truth, data, {
                    "max_iter": 50, "tol": 1e-6, "candidates": [family],
                    "fit_margins": fit_margins, "copula_margins": mode,
                })
                out.setdefault((data_name, mode), []).append(
                    [fitted.copula(k, k).params["tau_k"] for k in range(2)])
    err = {k: np.asarray(v) - TAU for k, v in out.items()}
    bias = {k: v.mean(axis=0) for k, v in err.items()}
    rmse = {k: np.sqrt((v ** 2).mean(axis=0)) for k, v in err.items()}
    return bias, rmse


#: Replications of the slow study. R = 30 here (≈ 90 s + ≈ 25 s); the
#: thresholds below are set from an R = 50 run (module docstring) with margin.
STUDY_R = 30


@pytest.fixture(scope="module")
def study_fitted_margins():
    return _tau_study(fit_margins=True, R=STUDY_R)


@pytest.fixture(scope="module")
def study_oracle_margins():
    return _tau_study(fit_margins=False, R=STUDY_R)


@pytest.mark.slow
def test_empirical_margins_cut_the_contamination_bias_when_margins_are_fitted(
        study_fitted_margins):
    """R = 50 reference: bias +0.258/+0.251 parametric, +0.100/+0.051
    empirical, RMSE ratio 0.62/0.73."""
    bias, rmse = study_fitted_margins
    par_b, emp_b = bias[("cont", "parametric")], bias[("cont", "empirical")]
    assert np.all(par_b > 0.15), par_b          # the IFM bias RB-5 is about
    assert np.all(emp_b < 0.18), emp_b
    assert np.all(emp_b < par_b - 0.07), (emp_b, par_b)
    ratio = rmse[("cont", "empirical")] / rmse[("cont", "parametric")]
    assert np.all(ratio < 0.95), ratio


@pytest.mark.slow
def test_efficiency_cost_on_clean_correctly_specified_data(
        study_fitted_margins, study_oracle_margins):
    """Clean data: ≈ no cost against *estimated* parametric margins (for the
    Gaussian copula the rank estimator is efficient), a real cost against
    margins fixed at the truth. R = 50 ratios: 0.99/1.00 and 1.64/1.46."""
    ratio_fit = (study_fitted_margins[1][("clean", "empirical")]
                 / study_fitted_margins[1][("clean", "parametric")])
    assert np.all(ratio_fit < 1.20), ratio_fit
    ratio_fix = (study_oracle_margins[1][("clean", "empirical")]
                 / study_oracle_margins[1][("clean", "parametric")])
    assert np.all(ratio_fix > 1.1), ratio_fix        # the oracle is hard to beat
    assert np.all(ratio_fix < 3.0), ratio_fix


@pytest.mark.slow
def test_with_oracle_margins_the_option_does_not_help(study_oracle_margins):
    """The unfavourable half of the study, pinned so it cannot quietly change:
    with the margins held at the truth the E-step re-routes the contaminated
    pairs out of the diagonal blocks, the parametric τ̂ stays nearly unbiased
    and the empirical one is worse (R = 50: +0.018/+0.018 vs +0.091/+0.087)."""
    bias = study_oracle_margins[0]
    par_b, emp_b = bias[("cont", "parametric")], bias[("cont", "empirical")]
    assert np.all(np.abs(par_b) < 0.06), par_b
    assert np.all(emp_b > np.abs(par_b)), (emp_b, par_b)
