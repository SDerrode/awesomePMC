"""Estimation of the missingness mechanism by ICE and SEM (P6, wave 2).

* **M-steps against the definitions** — :func:`estimate_state`,
  :func:`estimate_state_markov` and :func:`common_mechanism` maximise the
  expected complete-data log-likelihood of the mask written here from the
  definitions of ``"state"`` and ``"state-markov"`` (initial term with
  s = a / (1 − b + a) included) plus the guard's pseudo-observations,
  maximised by a generic optimiser;
* **config key** ``missingness`` — values, defaults, TOML, validation, the
  GUI-widget contract;
* **default bit-identity** — ``missingness = "model"`` (the default) gives,
  bit for bit, what ICE and SEM gave before the key existed (loaded from git);
* **modes** — ``"ignorable"``, the complete-Y fallback, the starting
  mechanism (common rate: the ignorable E-step), the first M-step of ICE
  (both strategies) and SEM against the formulas, the trace, the returned
  model;
* **statistics** — the common start is left on state-dependent data, the
  guard keeps a zero rate from being absorbing, both strategies with the
  k-means start and multistart.
"""
from __future__ import annotations

import importlib
import logging
import math
import re
import subprocess
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from pmcprg.missing.patterns import state_dependent, state_markov
from pmcprg.pmc import PMCModel, StateMarkovMissingness, StateMissingness, gaps, simulate
from pmcprg.pmc import missingness as MS
from pmcprg.pmc._estim_common import ice_estim_defaults, missingness_defaults, sem_estim_defaults
from pmcprg.pmc.ice import ice
from pmcprg.pmc.sem import sem

ICE = importlib.import_module("pmcprg.pmc.ice")
SEM = importlib.import_module("pmcprg.pmc.sem")
REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "pmcprg" / "pmc" / "models"


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) % 2**31


def _hmc_in(A=((0.95, 0.05), (0.05, 0.95)), mu=(-1.0, 1.0)):
    return PMCModel.from_dict({
        "model": {"variant": "HMC-IN", "K": 2}, "prior": {"A": [list(r) for r in A]},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": float(m), "scale": 1.0}}
                    for i, m in enumerate(mu)]})


def _state_data(m, N, rates, tag):
    X, Y = simulate(m, N=N, seed=_seed("x", tag, N))
    Yn, mk = state_dependent(Y, X, rates, seed=_seed("mask", tag, N))
    return X, Y, Yn, mk


# ---------------------------------------------------------------------------
# M-steps against the definitions
# ---------------------------------------------------------------------------

def _sig(u):
    return 1.0 / (1.0 + math.exp(-u))


def _pooled(k, n):
    return (k + 0.5) / (n + 1.0)


def _state_objective(g, m, i, c, p):
    """Σ_n γ_n(i) log P(m_n | π = p) + c [π̄ log p + (1 − π̄) log(1 − p)]."""
    pbar = _pooled(m.sum(), m.size)
    v = sum(g[n, i] * (math.log(p) if m[n] else math.log1p(-p)) for n in range(m.size))
    return v + c * (pbar * math.log(p) + (1 - pbar) * math.log1p(-p))


def _markov_objective(g, m, i, c, a, b):
    """Σ_{n≥1} γ_n(i) log p(m_n | m_{n−1}; a, b) + γ_0(i) log p(m_0 | s) + guard,
    s = a / (1 − b + a); the guard targets are the pooled transition rates."""
    N = m.size
    n01 = sum(1 for n in range(1, N) if not m[n - 1] and m[n])
    n0 = sum(1 for n in range(1, N) if not m[n - 1])
    n11 = sum(1 for n in range(1, N) if m[n - 1] and m[n])
    n1 = sum(1 for n in range(1, N) if m[n - 1])
    abar, bbar = _pooled(n01, n0), _pooled(n11, n1)
    s = a / (1 - b + a)
    v = g[0, i] * math.log(s if m[0] else 1 - s)
    for n in range(1, N):
        p1 = b if m[n - 1] else a
        v += g[n, i] * math.log(p1 if m[n] else 1 - p1)
    return v + c * (abar * math.log(a) + (1 - abar) * math.log1p(-a)
                    + bbar * math.log(b) + (1 - bbar) * math.log1p(-b))


def _ref_markov(g, m, i, c):
    """Nelder–Mead on the logits from three starts."""
    best = None
    for z0 in ([0.0, 0.0], [-3.0, 1.0], [-1.0, -1.0]):
        r = minimize(lambda z: -_markov_objective(g, m, i, c, _sig(z[0]), _sig(z[1])), z0,
                     method="Nelder-Mead",
                     options={"xatol": 1e-13, "fatol": 1e-15, "maxiter": 20000, "maxfev": 40000})
        best = r if best is None or r.fun < best.fun else best
    return _sig(best.x[0]), _sig(best.x[1]), -best.fun


def _random_case(trial):
    rng = np.random.default_rng(_seed("mstep", trial))
    N, K = 60, 3
    m = rng.random(N) < 0.3
    m[5:12] = True                        # a burst: persistence counts
    m[0] = bool(trial % 2)                # both initial masks
    return rng.dirichlet(np.ones(K), size=N), m


# Tolerances. The references are generic optimisers: their argmax is only
# determined to ~sqrt(eps) relative to the curvature scale; measured over
# these cases, max |implementation − reference| = 1.06e-8 ("state", bounded
# Brent) and 1.04e-8 ("state-markov", Nelder–Mead); common mechanism 1.4e-8.
# 5e-8 is 3.5× the largest. The objective values are compared one-sided: the
# implementation may not be worse than the reference by more than 1e-12
# relative (the rounding of a sum of 60 terms).
_ARG_TOL = 5e-8


@pytest.mark.parametrize("trial", range(4))
@pytest.mark.parametrize("c", [0.0, 1.0])
def test_state_m_step_maximises_the_expected_mask_log_likelihood(trial, c):
    g, m = _random_case(trial)
    est = MS.estimate_state(g, m, pseudo_count=c).rates
    for i in range(g.shape[1]):
        r = minimize_scalar(lambda p: -_state_objective(g, m, i, c, p), bounds=(1e-12, 1 - 1e-12),
                            method="bounded", options={"xatol": 1e-14})
        assert est[i] == pytest.approx(r.x, abs=_ARG_TOL)
        assert _state_objective(g, m, i, c, est[i]) >= -r.fun - 1e-12 * abs(r.fun)


@pytest.mark.parametrize("trial", range(4))
@pytest.mark.parametrize("c", [0.0, 1.0])
def test_state_markov_m_step_is_exact_with_the_initial_term(trial, c):
    g, m = _random_case(trial)
    est = MS.estimate_state_markov(g, m, pseudo_count=c)
    for i in range(g.shape[1]):
        a, b, F = _ref_markov(g, m, i, c)
        assert est.onset[i] == pytest.approx(a, abs=_ARG_TOL)
        assert est.persistence[i] == pytest.approx(b, abs=_ARG_TOL)
        assert _markov_objective(g, m, i, c, est.onset[i], est.persistence[i]) >= F - 1e-12 * abs(F)


def test_state_markov_m_step_is_the_closed_form_without_the_initial_term():
    """γ_0(i) = 0 (SEM's one-hot path at n = 0 elsewhere): the closed form is exact."""
    g, m = _random_case(0)
    g[0] = [1.0, 0.0, 0.0]
    est = MS.estimate_state_markov(g, m, pseudo_count=1.0)
    prev, cur = m[:-1], m[1:]
    abar = _pooled((~prev & cur).sum(), (~prev).sum())
    bbar = _pooled((prev & cur).sum(), prev.sum())
    for i in (1, 2):
        w = g[1:, i]
        a = (w[~prev & cur].sum() + abar) / (w[~prev].sum() + 1.0)
        b = (w[prev & cur].sum() + bbar) / (w[prev].sum() + 1.0)
        # measured 1.7e-16 relative (summation order): 1e-14
        assert est.onset[i] == pytest.approx(a, rel=1e-14)
        assert est.persistence[i] == pytest.approx(b, rel=1e-14)
    # state 0 carries the initial term and moves away from its closed form
    w = g[1:, 0]
    a0 = (w[~prev & cur].sum() + abar) / (w[~prev].sum() + 1.0)
    assert est.onset[0] != a0


@pytest.mark.parametrize("m0", [False, True])
def test_common_mechanism_is_the_mle_of_the_mask(m0):
    rng = np.random.default_rng(_seed("common", m0))
    m = rng.random(80) < 0.2
    m[10:15] = True
    m[0] = m0
    st = MS.common_mechanism("state", m, 3)
    assert st.rates == (m.mean(),) * 3
    cm = MS.common_mechanism("state-markov", m, 3)
    a, b, F = _ref_markov(np.ones((m.size, 1)), m, 0, 0.0)
    assert cm.onset == (cm.onset[0],) * 3 and cm.persistence == (cm.persistence[0],) * 3
    assert cm.onset[0] == pytest.approx(a, abs=_ARG_TOL)
    assert cm.persistence[0] == pytest.approx(b, abs=_ARG_TOL)
    # mask_log_likelihood is log p(m) of the chain, the reference's maximum:
    # measured 1.5e-15 relative (the maximum is flat to second order, so the
    # reference's 1e-8 argmax error does not show); 1e-13
    assert MS.mask_log_likelihood(cm, m) == pytest.approx(F, rel=1e-13)
    M, N = int(m.sum()), m.size
    # the M/N formula from the definition: measured 2.5e-16 relative
    assert MS.mask_log_likelihood(st, m) == pytest.approx(
        M * math.log(M / N) + (N - M) * math.log(1 - M / N), rel=1e-14)
    with pytest.raises(ValueError, match="state-independent"):
        MS.mask_log_likelihood(StateMissingness(rates=(0.1, 0.2)), m)


def test_common_markov_mechanism_at_a_zero_count():
    """No burst of two missing rows: b̂ = 0 at the MLE, kept at σ(−36)."""
    m = np.zeros(50, bool)
    m[[3, 20, 40]] = True
    cm = MS.common_mechanism("state-markov", m, 2)
    assert 0.0 < cm.persistence[0] < 1e-12
    # log p(m) at b = 0 exactly, from the definition: m_0 = 0 with probability
    # 1 − s = 1 / (1 + a); 46 steps from an observed row, 3 of them onsets;
    # the 3 steps from a missing row all end it (probability 1 − b = 1). The
    # bounded logit loses ~1e-15 of it.
    a = cm.onset[0]
    exact = -math.log1p(a) + 3 * math.log(a) + 43 * math.log1p(-a)
    assert MS.mask_log_likelihood(cm, m) == pytest.approx(exact, abs=1e-12)
    # and a is its maximiser: d/da [−log(1 + a) + 3 log a + 43 log(1 − a)] = 0
    assert -1 / (1 + a) + 3 / a - 43 / (1 - a) == pytest.approx(0.0, abs=1e-6)


def test_guard_keeps_every_rate_inside_and_fills_empty_states():
    m = np.zeros(40, bool)
    m[10:20] = True
    g = np.zeros((40, 3))
    g[:, 0] = ~m                             # state 0 never missing
    g[:, 1] = m                              # state 1 always missing
    st = MS.estimate_state(g, m)             # state 2: zero weight
    # one pseudo-observation at π̄ = 10.5 / 41 against 30 and 10 real ones
    assert 0.0 < st.rates[0] < 0.01 and 0.9 < st.rates[1] < 1.0
    assert st.rates[2] == pytest.approx(_pooled(10, 40), rel=1e-15)
    assert MS.estimate_state(g, m, pseudo_count=0.0).rates[:2] == (0.0, 1.0)
    mk = MS.estimate_state_markov(g, m)
    assert all(0.0 < v < 1.0 for v in mk.onset + mk.persistence)


def test_estimate_mechanism_dispatch_and_errors():
    g, m = _random_case(1)
    assert MS.estimate_mechanism("state", g, m) == MS.estimate_state(g, m)
    assert MS.estimate_mechanism("state-markov", g, m) == MS.estimate_state_markov(g, m)
    with pytest.raises(ValueError):
        MS.estimate_mechanism("ignorable", g, m)
    with pytest.raises(ValueError):
        MS.estimate_state(g[:-1], m)
    with pytest.raises(ValueError):
        MS.common_mechanism("bursty", m, 2)


def test_pseudo_count_default_is_read_at_call_time(monkeypatch):
    g, m = _random_case(2)
    monkeypatch.setattr(MS, "PSEUDO_COUNT", 0.0)
    assert MS.estimate_state(g, m) == MS.estimate_state(g, m, pseudo_count=0.0)


# ---------------------------------------------------------------------------
# Config key
# ---------------------------------------------------------------------------

def test_missingness_key_defaults_and_gui_contract():
    m = _hmc_in()
    assert ICE._parse_ice_cfg(m, None)["missingness"] == "model"
    assert SEM._parse_sem_cfg(m, None)["missingness"] == "model"
    assert missingness_defaults() == {"missingness": "model"}
    assert ICE.MISSINGNESS_MODES == ("model", "ignorable", "state", "state-markov")
    # no GUI widget yet: kept out of the widget-contract dicts
    assert "missingness" not in set(ice_estim_defaults()) | set(sem_estim_defaults())


@pytest.mark.parametrize("bad", ["State", "markov", "", None, 1])
def test_invalid_missingness_values_are_refused(bad):
    m = _hmc_in()
    with pytest.raises(ValueError, match="missingness"):
        ICE._parse_ice_cfg(m, {"missingness": bad})
    with pytest.raises(ValueError, match="missingness"):
        SEM._parse_sem_cfg(m, {"missingness": bad})


def test_missingness_key_from_the_toml_tables():
    raw = _hmc_in().raw
    raw["ice"] = {"missingness": "state"}
    raw["sem"] = {"missingness": "state-markov"}
    m = PMCModel.from_dict(raw)
    assert ICE._parse_ice_cfg(m, None)["missingness"] == "state"
    assert SEM._parse_sem_cfg(m, None)["missingness"] == "state-markov"
    assert ICE._parse_ice_cfg(m, {"missingness": "ignorable"})["missingness"] == "ignorable"
    _, _, Yn, _ = _state_data(m, 300, (0.05, 0.25), "toml")
    fitted, _ = ice(m, Yn, {"max_iter": 2})
    assert isinstance(fitted.missingness, StateMissingness)


# ---------------------------------------------------------------------------
# Default bit-identity: missingness = "model" is the estimators before the key
# ---------------------------------------------------------------------------

BASE_COMMIT = "2854d75"      # the last commit before the missingness key


@pytest.fixture(scope="module")
def pre_key():
    """``ice.py`` and ``sem.py`` at BASE_COMMIT, on today's other modules."""
    try:
        srcs = {f: subprocess.run(["git", "show", f"{BASE_COMMIT}:pmcprg/pmc/{f}.py"], cwd=REPO,
                                  capture_output=True, text=True, check=True).stdout
                for f in ("ice", "sem")}
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git history with commit {BASE_COMMIT} unavailable")
    mods = {}
    for f in ("ice", "sem"):
        name = f"_pre_key_{f}"
        mod = types.ModuleType(name)
        sys.modules[name] = mod                 # dataclasses resolve their module
        mods[f] = mod
    try:
        for f in ("ice", "sem"):
            exec(compile(srcs[f], f"_pre_key_{f}.py", "exec"), mods[f].__dict__)
        yield mods["ice"], mods["sem"]
    finally:
        for f in ("ice", "sem"):
            sys.modules.pop(f"_pre_key_{f}", None)


def _bits(fitted, trace):
    raw = fitted.raw
    floats = []

    def walk(o):
        if isinstance(o, dict):
            for k in sorted(o):
                walk(o[k])
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, (float, int)) and not isinstance(o, bool):
            floats.append(float(o).hex())
        else:
            floats.append(repr(o))

    walk({k: raw.get(k) for k in ("prior", "margins", "copulas", "missingness")})
    extra = (np.asarray(trace.sampled_X_history).tobytes()
             if hasattr(trace, "sampled_X_history") else b"")
    return floats, [float(v).hex() for v in trace.log_liks], extra


_IDENTITY_CASES = [
    # (fixture, mechanism, gaps, algo)
    ("hmc_in_gauss_k2.toml", None, True, "ice-available"),
    ("hmc_in_gauss_k2.toml", "state", True, "ice-impute"),
    ("hmc_in_gauss_k2.toml", "state-markov", False, "ice-available"),
    ("pmc_gauss_k2.toml", "state-markov", True, "ice-available"),
    ("pmc_pair_gauss_k2.toml", "state", True, "sem"),
    ("hmc_dn_gauss_k2.toml", None, True, "sem"),
    ("hmc_in_gauss_k2.toml", "state", False, "sem"),
]


@pytest.mark.parametrize("name,mech,with_gaps,algo", _IDENTITY_CASES)
def test_default_is_bit_identical_to_the_estimators_before_the_key(pre_key, name, mech,
                                                                  with_gaps, algo):
    old_ice, old_sem = pre_key
    m0 = PMCModel(MODELS / name)
    m = m0 if mech is None else m0.with_missingness(
        StateMissingness(rates=(0.05, 0.3)) if mech == "state"
        else StateMarkovMissingness(onset=(0.01, 0.05), persistence=(0.6, 0.9)))
    X, Y = simulate(m0, N=120, seed=_seed("bit", name))
    if with_gaps:
        Y = np.array(Y, dtype=float, copy=True)
        Y[[0, 1, 30, 31, 32, 70, 119]] = np.nan
    if algo == "sem":
        cfg = {"max_iter": 3, "sem_seed": 5, "fit_margins": True}
        new, old = sem(m, Y, cfg), old_sem.sem(m, Y, cfg)
        explicit = sem(m, Y, {**cfg, "missingness": "model"})
    else:
        cfg = {"max_iter": 3, "fit_margins": True, "missing_strategy": algo.split("-")[1],
               "missing_draws": 2}
        new, old = ice(m, Y, cfg), old_ice.ice(m, Y, cfg)
        explicit = ice(m, Y, {**cfg, "missingness": "model"})
    assert _bits(*new) == _bits(*old)
    assert _bits(*explicit) == _bits(*new)
    assert new[0].missingness == m.missingness
    # the trace records the (fixed) mechanism of every iterate
    tbl = None if m.missingness is None else m.missingness.to_table()
    assert new[1].missingness_history == [tbl] * len(new[1].log_liks)


# ---------------------------------------------------------------------------
# Modes: ignorable, complete Y, start, first M-step, trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("algo", ["ice", "sem"])
def test_ignorable_drops_the_mechanism(algo):
    m0 = _hmc_in()
    m = m0.with_missingness(StateMissingness(rates=(0.05, 0.3)))
    _, _, Yn, _ = _state_data(m0, 200, (0.05, 0.3), "ignorable")
    run = (lambda mm, **c: sem(mm, Yn, {"max_iter": 3, "sem_seed": 1, **c})) if algo == "sem" \
        else (lambda mm, **c: ice(mm, Yn, {"max_iter": 3, **c}))
    f1, t1 = run(m, missingness="ignorable")
    f0, t0 = run(m0)
    assert f1.missingness is None and "missingness" not in f1.raw
    assert _bits(f1, t1) == _bits(f0, t0)
    assert t1.missingness_history == [None] * len(t1.log_liks)


@pytest.mark.parametrize("algo", ["ice", "sem"])
@pytest.mark.parametrize("mode", ["state", "state-markov"])
def test_complete_y_falls_back_to_ignorable_with_a_warning(algo, mode, caplog):
    m = _hmc_in().with_missingness(StateMissingness(rates=(0.05, 0.3)))
    _, Y = simulate(m, N=150, seed=_seed("complete", algo, mode))
    run = (lambda **c: sem(m, Y, {"max_iter": 3, "sem_seed": 1, **c})) if algo == "sem" \
        else (lambda **c: ice(m, Y, {"max_iter": 3, **c}))
    with caplog.at_level(logging.WARNING):
        f1, t1 = run(missingness=mode)
    assert any("no missing rows" in r.getMessage() and "ignorable" in r.getMessage()
               for r in caplog.records)
    assert f1.missingness is None
    assert _bits(f1, t1) == _bits(*run(missingness="ignorable"))


def test_common_start_is_the_ignorable_e_step():
    """From a model without a mechanism the start is the common rate M/N: the
    first log-likelihood is the ignorable one plus log p(m), and the first
    M-step is the formula on the ignorable posteriors."""
    m0 = _hmc_in()
    _, _, Yn, mk = _state_data(m0, 400, (0.03, 0.3), "start")
    f, t = ice(m0, Yn, {"missingness": "state", "max_iter": 2, "fit_margins": True})
    M, N = int(mk.sum()), mk.size
    assert t.missingness_history[0] == {"mechanism": "state", "rates": [M / N, M / N]}
    post = gaps.gap_posterior(m0, Yn)
    shift = M * math.log(M / N) + (N - M) * math.log(1 - M / N)
    # the same forward pass but for the constant factor: measured 5.1e-16
    # relative; the first M-step, same γ: identical (0). Tolerances 1e-14.
    assert t.log_liks[0] == pytest.approx(post.log_lik + shift, rel=1e-14)
    first = MS.estimate_state(post.gamma, mk).rates
    np.testing.assert_allclose(t.missingness_history[1]["rates"], first, rtol=1e-14)
    # it left the common rate at once
    assert first[1] - first[0] > 0.1
    # state-markov: the common chain's MLE
    f, t = ice(m0, Yn, {"missingness": "state-markov", "max_iter": 2})
    assert t.missingness_history[0] == MS.common_mechanism("state-markov", mk, 2).to_table()


def test_start_from_the_model_mechanism():
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 200, (0.03, 0.3), "start-model")
    st = StateMissingness(rates=(0.02, 0.4))
    _, t = ice(m0.with_missingness(st), Yn, {"missingness": "state", "max_iter": 2})
    assert t.missingness_history[0] == st.to_table()
    # "state" → "state-markov": the nested a = b = π
    _, t = ice(m0.with_missingness(st), Yn, {"missingness": "state-markov", "max_iter": 2})
    assert t.missingness_history[0] == StateMarkovMissingness(onset=st.rates,
                                                              persistence=st.rates).to_table()
    # another kind: the common start
    mk = StateMarkovMissingness(onset=(0.01, 0.02), persistence=(0.5, 0.5))
    _, t = ice(m0.with_missingness(mk), Yn, {"missingness": "state", "max_iter": 2})
    assert t.missingness_history[0]["rates"][0] == t.missingness_history[0]["rates"][1]


@pytest.mark.parametrize("strategy", ["available", "impute"])
def test_ice_mechanism_m_step_uses_the_exact_posteriors(strategy):
    m0 = _hmc_in()
    start = m0.with_missingness(StateMarkovMissingness(onset=(0.02, 0.05), persistence=(0.5, 0.8)))
    X, Y = simulate(m0, N=300, seed=_seed("mstep-ice"))
    Yn, mk = state_markov(Y, X, (0.01, 0.05), (0.6, 0.9), seed=_seed("mstep-ice-mask"))
    _, t = ice(start, Yn, {"missingness": "state-markov", "max_iter": 2,
                           "missing_strategy": strategy, "missing_draws": 2})
    gamma = gaps.gap_posterior(start, Yn).gamma
    ref = MS.estimate_state_markov(gamma, mk).to_table()
    assert t.missingness_history[0] == start.missingness.to_table()
    # same γ (gap_posterior is ICE's E-step), same M-step: measured identical
    # for both strategies; 1e-14 leaves room for a reordered sum only
    np.testing.assert_allclose(t.missingness_history[1]["onset"], ref["onset"], rtol=1e-14)
    np.testing.assert_allclose(t.missingness_history[1]["persistence"], ref["persistence"],
                               rtol=1e-14)


@pytest.mark.parametrize("mode", ["state", "state-markov"])
def test_sem_mechanism_m_step_uses_the_drawn_path(mode):
    m0 = _hmc_in()
    _, _, Yn, mk = _state_data(m0, 300, (0.03, 0.3), "mstep-sem")
    f, t = sem(m0, Yn, {"missingness": mode, "max_iter": 3, "sem_seed": 2})
    assert len(t.missingness_history) == 3
    # the draw of iteration q gives the mechanism of iterate q + 1; SEM returns
    # θ^max_iter, that of the last draw
    after = t.missingness_history[1:] + [f.missingness.to_table()]
    for q in range(3):
        X = t.sampled_X_history[q]
        g = np.zeros((X.size, 2))
        g[np.arange(X.size), X] = 1.0
        assert after[q] == MS.estimate_mechanism(mode, g, mk).to_table()


def test_returned_model_carries_the_estimated_mechanism():
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 300, (0.03, 0.3), "returned")
    f, t = ice(m0, Yn, {"missingness": "state", "max_iter": 30, "tol": 1e-10,
                        "return_best_iterate": True})
    assert len(t.missingness_history) == len(t.log_liks)
    assert f.missingness.to_table() == t.missingness_history[t.returned_iter]
    assert f.raw["missingness"] == t.missingness_history[t.returned_iter]
    # the log-likelihood of the returned model is log p(y_obs, m) with it
    # (the same pass: measured identical; 1e-14)
    assert t.log_liks[t.returned_iter] == pytest.approx(gaps.gap_posterior(f, Yn).log_lik,
                                                        rel=1e-14)


# ---------------------------------------------------------------------------
# Statistics: recovery from the common start, the guard, strategies and starts
# ---------------------------------------------------------------------------

# report/missing_state/design_measurements.py (100 replications, N = 2000,
# truth π = (0.02, 0.3), ICE run to its fixed point): RMSE of π̂ = (0.0054,
# 0.0145), common start 0.16 left at the first M-step. The bounds below are
# 4 RMSE (a single fixed-seed data set).
_FIT = {"fit_margins": True, "tol": 1e-8, "max_iter": 500, "patience": 500}


def test_common_start_is_not_a_trap_on_state_dependent_data():
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 2000, (0.02, 0.3), "recovery")
    f, t = ice(m0, Yn, {**_FIT, "missingness": "state"})
    r = f.missingness.rates
    assert abs(r[0] - 0.02) < 4 * 0.0054 and abs(r[1] - 0.3) < 4 * 0.0145
    h = t.missingness_history
    assert h[0]["rates"][0] == h[0]["rates"][1]                 # common start
    assert h[1]["rates"][1] - h[1]["rates"][0] > 0.15           # left at once


def test_state_markov_recovery():
    # design_measurements.py "initial" (N = 2000, a = (0.005, 0.03), b = (0.7,
    # 0.9), 100 replications): sd of â = (0.0033, 0.0064), of b̂ = (0.143,
    # 0.028). Bounds: 4 sd.
    m0 = _hmc_in()
    X, Y = simulate(m0, N=2000, seed=_seed("x", "markov-recovery"))
    Yn, _ = state_markov(Y, X, (0.005, 0.03), (0.7, 0.9), seed=_seed("mask", "markov-recovery"))
    f, _ = ice(m0, Yn, {**_FIT, "missingness": "state-markov"})
    a, b = f.missingness.onset, f.missingness.persistence
    assert abs(a[0] - 0.005) < 4 * 0.0033 and abs(a[1] - 0.03) < 4 * 0.0064
    assert abs(b[0] - 0.7) < 4 * 0.143 and abs(b[1] - 0.9) < 4 * 0.028


def test_the_guard_keeps_a_zero_rate_from_being_absorbing(monkeypatch):
    # A start π_0 = 0 makes state 0 impossible at every missing row: without
    # the guard (c = 0) π̂_0 stays exactly 0 — measured in all 50
    # replications of design_measurements.py, 58 nat below the fit from the
    # common start at N = 2000. With c = 1 it recovers in 6–9 iterations to
    # the same log-likelihood (truth 0.05; smallest π̂_0 measured 0.031;
    # the two fixed points of ICE — not exactly EM — differ by ≤ 1.2e-3 nat
    # over 100 runs: tolerance 5e-3).
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 2000, (0.05, 0.3), "trap")
    start = m0.with_missingness(StateMissingness(rates=(0.0, 0.3)))
    f, t = ice(start, Yn, {**_FIT, "missingness": "state"})
    assert f.missingness.rates[0] > 0.03
    f_common, t_common = ice(m0, Yn, {**_FIT, "missingness": "state"})
    assert t.log_liks[-1] == pytest.approx(t_common.log_liks[-1], abs=5e-3)
    monkeypatch.setattr(MS, "PSEUDO_COUNT", 0.0)
    f0, t0 = ice(start, Yn, {**_FIT, "missingness": "state", "max_iter": 20})
    assert all(h["rates"][0] == 0.0 for h in t0.missingness_history)
    assert t0.log_liks[-1] < t.log_liks[-1] - 10.0


@pytest.mark.parametrize("strategy", ["available", "impute"])
def test_both_strategies_with_kmeans_and_multistart(strategy):
    pytest.importorskip("sklearn")
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 800, (0.03, 0.3), "strategies")
    f, t = ice(m0, Yn, {"missingness": "state", "missing_strategy": strategy, "init": "kmeans",
                        "n_starts": 2, "max_iter": 40, "fit_margins": True, "missing_draws": 2})
    r = f.missingness.rates
    # N = 800: RMSE of π̂ ≈ (0.011, 0.029) at N = 500 (design_measurements);
    # 4 × those, and every run of the multistart estimated its own mechanism.
    assert abs(r[0] - 0.03) < 0.045 and abs(r[1] - 0.3) < 0.12
    for tr in [t] + t.multistart_runs:
        assert tr.missingness_history[0]["rates"][0] == tr.missingness_history[0]["rates"][1]
        assert tr.missingness_history[-1]["rates"][1] > tr.missingness_history[-1]["rates"][0]
    fs, _ = sem(m0, Yn, {"missingness": "state-markov", "init": "kmeans", "n_starts": 2,
                         "max_iter": 10})
    assert isinstance(fs.missingness, StateMarkovMissingness)


@pytest.mark.slow
def test_parallel_multistart_estimates_the_same_mechanism():
    """The worker processes get the resolved ``missingness`` in their cfg."""
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 150, (0.03, 0.3), "parallel")
    cfg = {"missingness": "state", "n_starts": 2, "max_iter": 3}
    f1, _ = ice(m0, Yn, cfg)
    f2, _ = ice(m0, Yn, {**cfg, "multistart_workers": 2})
    assert f1.missingness == f2.missingness


def test_trace_and_raw_keep_the_table_format():
    m0 = _hmc_in()
    _, _, Yn, _ = _state_data(m0, 200, (0.03, 0.3), "format")
    f, t = ice(m0, Yn, {"missingness": "state-markov", "max_iter": 3})
    for h in t.missingness_history:
        assert set(h) == {"mechanism", "onset", "persistence"}
        assert MS.parse_missingness(h, 2) is not None
    # the fitted model round-trips through TOML with its mechanism
    assert PMCModel.from_dict(f.raw).missingness == f.missingness
    assert re.search(r"state-markov", repr(f.missingness.to_table()))
