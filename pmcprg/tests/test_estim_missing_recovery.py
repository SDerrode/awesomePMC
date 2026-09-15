"""Unsupervised estimation with missing observations — statistical behaviour (slow).

Parameter recovery
------------------
Simulated data (N = 3000) from four fixtures — ``pmc_gauss_k2`` (state
margins, Gaussian copulas), ``pmc_pair_gauss_k2`` (pair margins, Clayton),
``hmc_dn_gauss_k2`` (HMC-DN) and ``sp2016_gice_k2`` (GICE: margin and copula
families selected) — with MCAR blocks of 10 values
(:func:`pmcprg.missing.patterns.mcar`) removed at 10, 20 and 40 %, fitted by
ICE ``"available"``, ICE ``"impute"`` and SEM with ``fit_margins = True`` from
the truth moved by a controlled random amount (:func:`_start`).

Tolerance rule. For every parameter θ (joint prior p_ij; mean and standard
deviation of each fitted margin law; Kendall's τ of each copula) the spread
of the complete-data estimator was measured: σ₀(θ) is the RMSE of the same
algorithm (ICE for both ICE strategies, SEM for SEM) on complete data over 30
replications of this design, 12 for the GICE fixture (``SIGMA0`` below, from
the study reported with this change). A fit at missing rate ρ passes when

    |θ̂ − θ| ≤ 4 · σ₀(θ) / √(1 − ρ) + 0.005

— four complete-data RMSEs, inflated by the loss of 1/(1 − ρ) of the
information of an MCAR sample, plus a floor for parameters whose σ₀ is tiny.
The RMSE includes the complete-data estimator's bias (e.g. the pair margins
f_01, f_10 of ``pmc_pair_gauss_k2``, which carry 5 % of the pairs), so a
missing-data fit is only required to be as good as a complete-data one up to
noise, not better. The seeds are fixed, so the test is deterministic.

Trace behaviour, family selection: see the individual tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from pmcprg.missing.patterns import mcar
from pmcprg.pmc.ice import ice
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.sem import sem
from pmcprg.pmc.simulate import simulate

MODELS = Path(__file__).resolve().parents[1] / "pmc" / "models"
N_OBS = 3000
RATES = (0.1, 0.2, 0.4)
STRATEGIES = ("available", "impute", "sem")

#: Complete-data RMSE per parameter, 30 replications of the design above
#: (ICE = the ICE fit on complete data, the same for both strategies).
SIGMA0: dict[str, dict[str, dict[str, float]]] = {
    "pmc_gauss_k2.toml": {
        "ice": {"p00": 0.0421, "p01": 0.0035, "p11": 0.0433, "mean0": 0.0849, "sd0": 0.0616,
                "mean1": 0.0923, "sd1": 0.0403, "tau00": 0.0245, "tau01": 0.0613, "tau11": 0.0157},
        "sem": {"p00": 0.0524, "p01": 0.0041, "p11": 0.0508, "mean0": 0.1, "sd0": 0.0714,
                "mean1": 0.1129, "sd1": 0.0519, "tau00": 0.0251, "tau01": 0.081, "tau11": 0.0207},
    },
    "pmc_pair_gauss_k2.toml": {
        "ice": {"p00": 0.073, "p01": 0.0087, "p11": 0.0692, "mean00": 0.1507, "sd00": 0.0866,
                "mean01": 0.3719, "sd01": 0.223, "mean10": 0.3734, "sd10": 0.2183,
                "mean11": 0.1857, "sd11": 0.1068, "tau00": 0.0346, "tau01": 0.0852, "tau11": 0.0301},
        "sem": {"p00": 0.0821, "p01": 0.0167, "p11": 0.0732, "mean00": 0.1813, "sd00": 0.1098,
                "mean01": 0.4603, "sd01": 0.2725, "mean10": 0.3721, "sd10": 0.2603,
                "mean11": 0.1964, "sd11": 0.1152, "tau00": 0.0385, "tau01": 0.1102, "tau11": 0.032},
    },
    "hmc_dn_gauss_k2.toml": {
        "ice": {"p00": 0.0458, "p01": 0.0036, "p11": 0.046, "mean0": 0.0858, "sd0": 0.0609,
                "mean1": 0.0954, "sd1": 0.0465, "tau00": 0.0244, "tau01": 0.061, "tau11": 0.0199},
        "sem": {"p00": 0.0486, "p01": 0.0046, "p11": 0.0492, "mean0": 0.1014, "sd0": 0.0709,
                "mean1": 0.1125, "sd1": 0.0547, "tau00": 0.0257, "tau01": 0.0694, "tau11": 0.0187},
    },
    # 12 replications (GICE runs are slow). Margin standard deviations are not
    # checked: the selected families are heavy-tailed (inverse gamma, beta
    # prime), the complete-data RMSE of sd1 is 0.32 and a fit can have none.
    "sp2016_gice_k2.toml": {
        "ice": {"p00": 0.0299, "p01": 0.0037, "p11": 0.0293, "mean0": 0.0339, "mean1": 0.044,
                "tau00": 0.0173, "tau01": 0.0446, "tau11": 0.0292},
        "sem": {"p00": 0.0304, "p01": 0.0037, "p11": 0.0285, "mean0": 0.0376, "mean1": 0.038,
                "tau00": 0.0209, "tau01": 0.0375, "tau11": 0.0273},
    },
}


def _recovery_cases():
    """Every (fixture, rate, strategy); the slow GICE fixture runs its 5–15× slower
    ``"impute"`` and SEM fits at 20 % only (``"available"`` at every rate)."""
    for fixture in ("pmc_gauss_k2.toml", "pmc_pair_gauss_k2.toml", "hmc_dn_gauss_k2.toml",
                    "sp2016_gice_k2.toml"):
        for rate in RATES:
            for strategy in STRATEGIES:
                if fixture == "sp2016_gice_k2.toml" and strategy != "available" and rate != 0.2:
                    continue
                yield fixture, rate, strategy


def _start(truth: PMCModel, rng: np.random.Generator) -> PMCModel:
    """Truth moved by a controlled random amount.

    Prior entries ×exp(0.1·N(0,1)) (symmetrised / renormalised); margin loc
    + 0.15·scale·N(0,1) (downwards only, by its absolute value, for a
    bounded-support family), other margin parameters ×exp(0.1·N(0,1)); τ +
    0.1·N(0,1) clipped to [−0.85, 0.85] ∩ the family's range. (The package's
    multistart jitter can put a τ at ±1, a start ICE does not recover from.)
    """
    from pmcprg.copulas._base import CopulaEnum
    raw = truth.raw
    pr = raw["prior"]
    key = "p" if "p" in pr else "A"
    M = np.asarray(pr[key], float) * np.exp(0.1 * rng.standard_normal((truth.K, truth.K)))
    if key == "p":
        M = 0.5 * (M + M.T)
        M /= M.sum()
    else:
        M /= M.sum(axis=1, keepdims=True)
    pr[key] = M.tolist()
    for blk in raw.get("margins", []):
        prm = blk["params"]
        sc = float(prm.get("scale", 1.0))
        for k in list(prm):
            if k == "loc":
                step = 0.15 * sc * float(rng.standard_normal())
                # a bounded support (gamma, beta-prime, …) must keep covering the data
                prm[k] = float(prm[k]) + (step if blk["dist"] == "norm" else -abs(step))
            else:
                prm[k] = float(prm[k]) * float(np.exp(0.1 * rng.standard_normal()))
    for blk in raw.get("copulas", []):
        lo, hi = CopulaEnum.from_short_name(blk["name"]).constructible_tau_range()
        blk["tau"] = float(np.clip(blk["tau"] + 0.1 * rng.standard_normal(),
                                   max(lo, -0.85), min(hi, 0.85)))
    return PMCModel.from_dict(raw)


def _summary(model: PMCModel) -> tuple[dict[str, float], dict[str, str]]:
    params, families = {}, {}
    p = (model.stationary_pi[:, None] * model.transition_A if model.variant.has_markov_prior
         else model.prior_p)
    for i in range(model.K):
        for j in range(i, model.K):
            params[f"p{i}{j}"] = float(p[i, j])
    for blk in model.margin_blocks():
        tag = f"{blk['i']}{blk.get('j', '')}"
        law = getattr(stats, blk["dist"])(**blk["params"])
        params[f"mean{tag}"], params[f"sd{tag}"] = float(law.mean()), float(law.std())
        families[f"f{tag}"] = blk["dist"]
    if model.variant.uses_copula:
        for blk in model.copula_blocks():
            if blk["i"] <= blk["j"]:
                params[f"tau{blk['i']}{blk['j']}"] = float(blk["tau"])
                families[f"c{blk['i']}{blk['j']}"] = blk["name"]
    return params, families


def _fit(fixture: str, rate: float, strategy: str, seed: int = 101):
    truth = PMCModel(MODELS / fixture)
    _, Y = simulate(truth, N=N_OBS, seed=50_000 + seed)
    Ym = mcar(Y, rate, block_size=10, seed=60_000 + seed)[0] if rate else Y
    start = _start(truth, np.random.default_rng(70_000 + seed))
    cfg = {"fit_margins": True}
    if fixture != "sp2016_gice_k2.toml":
        cfg["candidates"] = [truth.copula_blocks()[0]["name"]]
    if strategy == "sem":
        return truth, *sem(start, Ym, sem_cfg={**cfg, "sem_seed": seed})
    return truth, *ice(start, Ym, ice_cfg={**cfg, "missing_strategy": strategy,
                                            "missing_seed": seed})


@pytest.mark.slow
@pytest.mark.parametrize("fixture,rate,strategy", list(_recovery_cases()))
def test_parameter_recovery_with_missing_observations(fixture, rate, strategy):
    truth, fitted, trace = _fit(fixture, rate, strategy)
    sigma0 = SIGMA0[fixture]["sem" if strategy == "sem" else "ice"]
    ref, _ = _summary(truth)
    got, families = _summary(fitted)
    assert np.isfinite(trace.log_liks).all()
    if fixture == "sp2016_gice_k2.toml":
        assert set(families[k] for k in ("f0", "f1")) <= {"norm", "gamma", "invgamma", "betaprime"}
    bad = {}
    for k, theta in ref.items():
        if k not in sigma0:
            continue
        tol = 4.0 * sigma0[k] / np.sqrt(1.0 - rate) + 0.005
        if not abs(got[k] - theta) <= tol:
            bad[k] = (round(got[k], 4), theta, round(tol, 4))
    assert not bad, f"{fixture} rate={rate} {strategy}: (estimate, truth, tol) {bad}"


# ---------------------------------------------------------------------------
# Log-likelihood trace behaviour (observed-data log-likelihood)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("rate", [0.2, 0.4])
@pytest.mark.parametrize("fixture", ["pmc_gauss_k2.toml", "hmc_dn_gauss_k2.toml"])
def test_ice_available_trace_is_non_decreasing(fixture, rate):
    """ICE is not EM, so no monotonicity is guaranteed; measured, the
    observed-data log-likelihood of ``"available"`` never decreased on the
    state-margin fixtures (study: 0 decreasing runs of 30 per cell). Pair
    margins do decrease, on complete data too — see the report."""
    _, _, trace = _fit(fixture, rate, "available")
    steps = np.diff(trace.log_liks)
    assert steps.size >= 2
    assert steps.min() >= -1e-6 * abs(trace.log_liks[-1])


@pytest.mark.slow
def test_sem_trace_is_stationary_after_burn_in():
    """SEM's trace fluctuates around a level: after a burn-in of 20 iterations
    the means of iterations 20–39 and 40–59 differ by less than two standard
    deviations of the fluctuation, and the level lies below the ICE maximum
    by a few standard deviations at most."""
    truth = PMCModel(MODELS / "pmc_gauss_k2.toml")
    _, Y = simulate(truth, N=N_OBS, seed=123)
    Ym = mcar(Y, 0.2, block_size=10, seed=124)[0]
    start = _start(truth, np.random.default_rng(125))
    cfg = {"fit_margins": True, "candidates": ["Gauss"]}
    _, tr = sem(start, Ym, sem_cfg={**cfg, "max_iter": 60, "sem_seed": 3})
    ll = np.asarray(tr.log_liks)
    sd = ll[20:].std()
    assert abs(ll[20:40].mean() - ll[40:].mean()) <= 2.0 * sd
    _, tr_ice = ice(start, Ym, ice_cfg=cfg)
    gap = tr_ice.log_liks[-1] - ll[20:].mean()
    assert -2.0 * sd <= gap <= 10.0 * sd


# ---------------------------------------------------------------------------
# Copula family selection — CSDA-2013 Exp. 3, setting 1, with 20 % missing
# ---------------------------------------------------------------------------

# report/reproduce_csda2013.py::run_exp3 "exp1_orig_p" (as in
# test_multistart_families.py): candidates Gauss, GH, Clayton; truth (0,0)
# Gauss τ=0.7, (0,1)=(1,0) GH τ=0.4, (1,1) Clayton τ=0.7; Table 1 pair
# margins N(μ_ij, s_ij), s read as a variance; p_orig; N = 2500.
_EXP3_TRUTH = {(0, 0): ("Gauss", 0.7), (0, 1): ("GH", 0.4),
               (1, 0): ("GH", 0.4), (1, 1): ("Clayton", 0.7)}
_EXP3_MARGINS = {(0, 0): (0.0, 1.0), (0, 1): (0.3, 1.6), (1, 0): (1.1, 1.4), (1, 1): (1.5, 1.0)}
_EXP3_PRIOR = [[0.5, 0.05], [0.05, 0.4]]


def _exp3_model(layout: dict) -> PMCModel:
    return PMCModel.from_dict({
        "model": {"name": "exp3", "variant": "PMC", "K": 2, "N_default": 2500},
        "prior": {"p": _EXP3_PRIOR},
        "margins": [{"i": i, "j": j, "dist": "norm",
                     "params": {"loc": mu, "scale": float(np.sqrt(var))}}
                    for (i, j), (mu, var) in sorted(_EXP3_MARGINS.items())],
        "copulas": [{"i": i, "j": j, "name": name, "tau": tau}
                    for (i, j), (name, tau) in layout.items()],
    })


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_exp3_sweep_recovers_the_diagonal_families_with_20_percent_missing(seed):
    truth = _exp3_model(_EXP3_TRUTH)
    start = _exp3_model({k: ("Gauss", 0.0) for k in _EXP3_TRUTH})
    _, Y = simulate(truth, N=2500, seed=seed)
    Ym = mcar(Y, 0.2, block_size=10, seed=1000 + seed)[0]
    cfg = {"max_iter": 30, "candidates": ["Gauss", "GH", "Clayton"], "fit_margins": False,
           "selection_criterion": "mle", "n_starts": 10, "multistart_families": "sweep"}
    fitted, trace = ice(start, Ym, ice_cfg=cfg)
    assert (fitted.copula(0, 0).copula_enum.value.SHORT_NAME,
            fitted.copula(1, 1).copula_enum.value.SHORT_NAME) == ("Gauss", "Clayton")
    # the winner has the highest observed-data log-likelihood of the 10 starts
    assert trace.log_liks[-1] == max(t.log_liks[-1] for t in [trace, *trace.multistart_runs])
    assert fitted.copula(0, 0).params["tau_k"] == pytest.approx(0.7, abs=0.1)
    assert fitted.copula(1, 1).params["tau_k"] == pytest.approx(0.7, abs=0.1)
