"""Likelihood-ratio test of state-dependent missingness (P6, wave 2).

* the null log-likelihoods written from their definitions: ignorable fit
  plus the maximised log-probability of the mask (common rate M/N; common
  two-state Markov chain with stationary start, maximised here by a generic
  optimiser), the ``"state"`` fit for the nested test;
* the alternative's log-likelihood, the statistic, the degrees of freedom and
  the χ² p-value;
* the parametric bootstrap: p-value formula, reproducibility, separate seed
  streams for the paths and the masks, the null mechanism of the masks;
* validation, ``summary``;
* Monte Carlo (slow): size and power, reduced from
  ``report/missing_state/lr_study.py``.
"""
from __future__ import annotations

import importlib
import math
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

from pmcprg.missing import patterns
from pmcprg.missing.patterns import state_dependent, state_markov
from pmcprg.pmc import (MissingnessLRTest, PMCModel, StateMarkovMissingness, StateMissingness,
                        gaps, missingness_lr_test, simulate)
from pmcprg.pmc import missingness_lr as LR

SIM = importlib.import_module("pmcprg.pmc.simulate")      # the module (the package exports the function)
MODELS = Path(__file__).resolve().parents[1] / "pmc" / "models"


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode()) % 2**31


def _hmc_in(K=2):
    if K == 3:
        return PMCModel(MODELS / "hmc_in_gauss_k3.toml")
    return PMCModel.from_dict({
        "model": {"variant": "HMC-IN", "K": 2}, "prior": {"A": [[0.95, 0.05], [0.05, 0.95]]},
        "margins": [{"i": i, "dist": "norm", "params": {"loc": m, "scale": 1.0}}
                    for i, m in enumerate((-1.0, 1.0))]})


def _data(m, N, tag, rates=None, onset=None, persistence=None):
    X, Y = simulate(m, N=N, seed=_seed("x", tag, N))
    if rates is not None:
        return state_dependent(Y, X, rates, seed=_seed("mask", tag, N))[0]
    return state_markov(Y, X, onset, persistence, seed=_seed("mask", tag, N))[0]


def _mask_markov_loglik(m, a, b):
    """log p(m) of a two-state Markov chain, P(1 | 0) = a, P(1 | 1) = b, started
    from its stationary law s = a / (1 − b + a) — from the definition."""
    s = a / (1.0 - b + a)
    ll = math.log(s if m[0] else 1.0 - s)
    for prev, cur in zip(m[:-1], m[1:]):
        p1 = b if prev else a
        ll += math.log(p1 if cur else 1.0 - p1)
    return ll


def _max_mask_markov(m):
    sig = lambda u: 1.0 / (1.0 + math.exp(-u))          # noqa: E731
    best = None
    for z0 in ([-3.0, 0.0], [-1.0, 1.0]):
        r = minimize(lambda z: -_mask_markov_loglik(m, sig(z[0]), sig(z[1])), z0,
                     method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-14,
                                                    "maxiter": 20000, "maxfev": 40000})
        best = r if best is None or r.fun < best.fun else best
    return -best.fun


# Measured on these data sets: the forward pass of a model carrying a
# state-independent mechanism and "ignorable log-likelihood + log p(m)"
# differ by 1.7e-15 relative (the rounding of the constant factor in the
# scaled forward pass); the null of "state-markov" and the reference
# optimiser's maximum of log p(m) by 1.5e-15 (the maximum is flat to second
# order); the refitted log-likelihoods and a fresh gap_posterior are
# identical (same pass). 1e-13 relative is ≥ 60× those.
_REL = 1e-13
_FIT = {"fit_margins": True}


def test_state_against_common_rate():
    m = _hmc_in()
    Yn = _data(m, 600, "state", rates=(0.03, 0.2))
    r = missingness_lr_test(m, Yn, ice_cfg=_FIT)
    miss = gaps.missing_mask(Yn)
    M, N = int(miss.sum()), miss.size
    assert (r.n_obs, r.n_missing) == (N, M)
    ll_ign = gaps.gap_posterior(r.null_model.with_missingness(None), Yn).log_lik
    mask = M * math.log(M / N) + (N - M) * math.log(1 - M / N)
    assert r.log_lik_null == pytest.approx(ll_ign + mask, rel=_REL)
    assert r.log_lik_null == pytest.approx(gaps.gap_posterior(r.null_model, Yn).log_lik, rel=_REL)
    assert r.log_lik_alt == pytest.approx(gaps.gap_posterior(r.alt_model, Yn).log_lik, rel=_REL)
    assert r.null_params == {"mechanism": "state", "rates": [M / N, M / N]}
    assert r.alt_params == r.alt_model.missingness.to_table()
    assert r.statistic == 2 * (r.log_lik_alt - r.log_lik_null)
    assert r.df == 1 and r.p_value == chi2.sf(r.statistic, 1)
    assert r.statistic > 10.0 and r.alt_params["rates"][1] > r.alt_params["rates"][0]
    assert math.isnan(r.p_value_bootstrap) and r.n_bootstrap == 0
    assert r.bootstrap_statistics == ()


def test_state_markov_against_a_common_chain():
    m = _hmc_in()
    Yn = _data(m, 600, "markov", onset=(0.01, 0.04), persistence=(0.6, 0.9))
    r = missingness_lr_test(m, Yn, alternative="state-markov", ice_cfg=_FIT)
    miss = gaps.missing_mask(Yn)
    ll_ign = gaps.gap_posterior(r.null_model.with_missingness(None), Yn).log_lik
    # the common (a, b) maximises log p(m) (reference optimiser; see _REL)
    assert r.log_lik_null == pytest.approx(ll_ign + _max_mask_markov(miss), rel=_REL)
    assert r.null_params["mechanism"] == "state-markov"
    assert len(set(r.null_params["onset"])) == 1 and len(set(r.null_params["persistence"])) == 1
    assert r.log_lik_alt == pytest.approx(gaps.gap_posterior(r.alt_model, Yn).log_lik, rel=_REL)
    assert r.df == 2 and r.p_value == chi2.sf(r.statistic, 2)
    assert isinstance(r.alt_model.missingness, StateMarkovMissingness)


def test_state_markov_against_state():
    m = _hmc_in()
    Yn = _data(m, 600, "nested", onset=(0.01, 0.04), persistence=(0.6, 0.9))
    r = missingness_lr_test(m, Yn, alternative="state-markov", null="state", ice_cfg=_FIT)
    assert isinstance(r.null_model.missingness, StateMissingness)
    assert r.log_lik_null == pytest.approx(gaps.gap_posterior(r.null_model, Yn).log_lik, rel=_REL)
    assert r.df == 2
    # bursts of mean length 2.5 and 10: a Markov mask the "state" null misses
    assert r.statistic > 20.0 and r.p_value < 1e-4


def test_hmc_in_statistic_is_not_negative_under_the_null():
    # HMC-IN: ICE is EM up to the SR symmetrisation of the prior, so the
    # alternative started at the null fit does not lose likelihood: smallest
    # LR measured over the 1 200 null replications of lr_study.py ("state"
    # vs common, N = 500–2000) 5.7e-6. (The nested test can end slightly
    # below its start — min −0.096 at N = 500: the guard's pseudo-counts
    # target other pooled rates for "state" and "state-markov", so the two
    # fits maximise different penalised likelihoods; rerun without the guard
    # the five negative replications give LR = 0.007 to 0.21.)
    m = _hmc_in()
    for k in range(3):
        Yn = _data(m, 400, ("null", k), rates=(0.1, 0.1))
        r = missingness_lr_test(m, Yn, ice_cfg=_FIT)
        assert r.statistic > -1e-3


@pytest.mark.parametrize("alternative,null,df", [("state", "common", 2),
                                                 ("state-markov", "common", 4),
                                                 ("state-markov", "state", 3)])
def test_degrees_of_freedom_with_three_states(alternative, null, df):
    m = _hmc_in(3)
    Yn = _data(m, 200, "k3", rates=(0.05, 0.1, 0.2))
    r = missingness_lr_test(m, Yn, alternative=alternative, null=null, ice_cfg={"max_iter": 3})
    assert r.df == df and r.p_value == chi2.sf(r.statistic, df)


def test_fit_settings_resolution():
    m = _hmc_in()
    assert LR._fit_cfg(m, None) == {"tol": 1e-8, "max_iter": 500, "patience": 500}
    raw = m.raw
    raw["ice"] = {"tol": 1e-6}
    m2 = PMCModel.from_dict(raw)
    assert LR._fit_cfg(m2, None) == {"max_iter": 500, "patience": 500}
    assert LR._fit_cfg(m2, {"max_iter": 7})["max_iter"] == 7


def test_bootstrap(monkeypatch):
    m = _hmc_in()
    Yn = _data(m, 300, "boot", rates=(0.08, 0.12))
    seen = {"sim": [], "mask": []}
    real_sim, real_mask = SIM.simulate, patterns.state_dependent

    def sim(model, N=None, seed=None):
        seen["sim"].append(seed)
        return real_sim(model, N=N, seed=seed)

    def mask(Y, X, rates, *, seed=None):
        seen["mask"].append(seed)
        # the masks of the bootstrap follow the fitted null: a common rate
        assert len(set(rates)) == 1
        return real_mask(Y, X, rates, seed=seed)

    monkeypatch.setattr(SIM, "simulate", sim)
    monkeypatch.setattr(patterns, "state_dependent", mask)
    cfg = {"max_iter": 20, "tol": 1e-6}
    r = missingness_lr_test(m, Yn, ice_cfg=cfg, n_bootstrap=5, seed=3)
    assert r.n_bootstrap == 5 and r.n_bootstrap_valid == 5
    assert len(seen["sim"]) == len(seen["mask"]) == 5
    assert len(set(seen["sim"]) | set(seen["mask"])) == 10     # every seed distinct
    boot = np.array(r.bootstrap_statistics)
    assert r.p_value_bootstrap == (1 + np.sum(boot >= r.statistic)) / (boot.size + 1)
    again = missingness_lr_test(m, Yn, ice_cfg=cfg, n_bootstrap=5, seed=3)
    assert again.bootstrap_statistics == r.bootstrap_statistics
    other = missingness_lr_test(m, Yn, ice_cfg=cfg, n_bootstrap=5, seed=4)
    assert other.bootstrap_statistics != r.bootstrap_statistics
    # one replicate by hand: the public helper with the same seeds
    lr0 = LR.bootstrap_replicate(r.null_model, 300, seen["sim"][0], seen["mask"][0],
                                 ice_cfg=cfg)
    assert lr0 == r.bootstrap_statistics[0]


def test_bootstrap_replicate_with_an_empty_mask_is_dropped():
    m = _hmc_in().with_missingness(StateMissingness(rates=(0.0, 0.0)))
    assert LR.bootstrap_replicate(m, 50, 1, 2) is None


def test_validation():
    m = _hmc_in()
    _, Y = simulate(m, N=50, seed=1)
    with pytest.raises(ValueError, match="0 missing rows"):
        missingness_lr_test(m, Y)
    with pytest.raises(ValueError, match="50 missing rows of 50"):
        missingness_lr_test(m, np.full(50, np.nan))
    Yn = np.array(Y, dtype=float, copy=True)
    Yn[[3, 10]] = np.nan
    with pytest.raises(ValueError, match="no test"):
        missingness_lr_test(m, Yn, alternative="state", null="state")
    with pytest.raises(ValueError, match="no test"):
        missingness_lr_test(m, Yn, alternative="ignorable")
    for bad in (-1, 2.5):
        with pytest.raises(ValueError, match="n_bootstrap"):
            missingness_lr_test(m, Yn, n_bootstrap=bad)


def test_summary_and_repr():
    m = _hmc_in()
    Yn = _data(m, 300, "summary", rates=(0.05, 0.15))
    r = missingness_lr_test(m, Yn, ice_cfg={"max_iter": 10}, n_bootstrap=2, seed=1)
    assert isinstance(r, MissingnessLRTest)
    s = r.summary()
    assert "state-independent" in s and "'state'" in s
    assert f"{r.statistic:.4f}" in s and f"df = {r.df}" in s
    assert "bootstrap, 2/2" in s and f"missing = {r.n_missing}" in s
    assert "rates = [" in s
    assert repr(r).startswith("MissingnessLRTest(state vs common, LR=")
    with pytest.raises(AttributeError):
        r.statistic = 0.0                                     # frozen


# ---------------------------------------------------------------------------
# Monte Carlo (reduced from report/missing_state/lr_study.py)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_monte_carlo_size_and_power_hmc_in():
    """HMC-IN, N = 500, "state" against a common rate.

    Size: 200 masks with π = (0.1, 0.1). The full study (lr_study.py,
    400 replications per N) measured the χ² rejection rate at 5 %:
    0.058 / 0.045 / 0.060 (± 0.012) at N = 500 / 1000 / 2000 — HMC-IN is EM,
    Wilks applies. Here the bound is the binomial 99.9 % band of 0.05 at
    R = 200: [0.010, 0.105]. Power: 60 masks with π = (0.025, 0.175);
    measured 200 / 200 at N = 500 — asserted ≥ 0.85 (probability of fewer
    than 51 of 60 at a power of 0.99: 5e-10).
    """
    m = _hmc_in()
    rej = []
    for r in range(200):
        Yn = _data(m, 500, ("mc-size", r), rates=(0.1, 0.1))
        rej.append(missingness_lr_test(m, Yn, ice_cfg=_FIT).p_value < 0.05)
    size = float(np.mean(rej))
    assert 0.010 <= size <= 0.105, size
    power = np.mean([missingness_lr_test(m, _data(m, 500, ("mc-power", r), rates=(0.025, 0.175)),
                                         ice_cfg=_FIT).p_value < 0.05 for r in range(60)])
    assert power >= 0.85, power
