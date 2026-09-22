"""Empty states: a silent wrong answer found by a real-data study.

**Empty states** (forecasting study, problem P2). A state whose row and
column of the PMC prior p are all zero — ICE and SEM produce one when a
state loses every observation (``degenerate_states`` reports π = 0) — must
simply carry zero probability: every result must be that of the model with
the state deleted. ``outliers`` read the row T_i· = p_i· / Σ_k p_ik = 0/0 of
the empty state as NaN and 0 · NaN made every PIT after row 1 NaN, so
nothing was flagged; the log-space weights of ``inference`` formed
log 0 − log 0 (NaN, RuntimeWarning) before masking it. References: the same
model with the empty state deleted (K − 1 states), for every variant, in
linear and log space; the report's own reproduction (a Gaussian AR(1)
written as a PMC whose first state is empty) against the closed form.

Seeds are fixed integers. Tolerances are the errors measured on macOS arm64,
quoted next to each constant with the margin taken.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy import stats

from pmcprg.pmc import PMCModel, classify, flag_outliers, predictive_pit
from pmcprg.pmc import inference as inf


def _tau(rho: float) -> float:
    return 2.0 / math.pi * math.asin(rho)


# ---------------------------------------------------------------------------
# Empty states — the K-state model and the same model without state 1
# ---------------------------------------------------------------------------

MU = (0.0, 3.0, 1.5)
SD = (1.0, 0.7, 1.3)
RHO = ((0.6, 0.3, -0.2), (0.3, 0.8, 0.1), (-0.2, 0.1, 0.5))
#: Dyadic priors, so that deleting state 1 leaves exactly the same numbers.
#: PMC: row and column 1 of p are zero. HMC: column 1 of A is zero, so π_1 = 0
#: and p = π A has a zero row and column; row 1 of A (the law out of a state
#: that is never entered) is the stationary law, so that the power iteration
#: of ``_stationary_distribution`` gives the same π for K and K − 1 states.
P_EMPTY = ((0.375, 0.0, 0.0625), (0.0, 0.0, 0.0), (0.0625, 0.0, 0.5))
A_EMPTY = ((0.875, 0.0, 0.125), (0.5, 0.0, 0.5), (0.125, 0.0, 0.875))
FULL, KEPT = (0, 1, 2), (0, 2)
CASES = [("PMC", "state"), ("PMC", "pair"), ("PMC-IN", "state"), ("PMC-IN", "pair"),
         ("HMC-IN", "state"), ("HMC-DN", "state")]

#: Absolute error on probabilities (α̂, γ, PIT, p-values) and on log
#: densities, K = 3 with an empty state against K = 2, over the six variants
#: in linear and log space, with and without missing rows: at most 8.9e-16
#: measured (a few ulps: sums of 3 terms with a zero one against sums of 2);
#: ×110.
TOL = 1e-13
#: Log-likelihoods (≈ −150, 80 rows): equal to the bit on macOS arm64; 1e-11
#: is ~350 ulps of the total, room for a libm that rounds each log C_n apart.
TOL_LL = 1e-11


def _empty_model(variant: str, structure: str, keep=FULL) -> PMCModel:
    K = len(keep)
    m = {"name": "empty-state", "K": K, "variant": variant}
    if structure == "pair":
        m["margin_structure"] = "pair"
        margins = [{"i": a, "j": b, "dist": "norm",
                    "params": {"loc": MU[i] + 0.25 * j, "scale": SD[i] * (1.0 + 0.125 * j)}}
                   for a, i in enumerate(keep) for b, j in enumerate(keep)]
    else:
        margins = [{"i": a, "dist": "norm", "params": {"loc": MU[i], "scale": SD[i]}}
                   for a, i in enumerate(keep)]
    raw = {"model": m, "margins": margins}
    idx = np.ix_(keep, keep)
    if variant.startswith("HMC"):
        raw["prior"] = {"A": np.asarray(A_EMPTY)[idx].tolist()}
    else:
        raw["prior"] = {"p": np.asarray(P_EMPTY)[idx].tolist()}
    if variant in ("PMC", "HMC-DN"):
        raw["copulas"] = [{"i": a, "j": b, "name": "Gauss", "tau": _tau(RHO[i][j])}
                          for a, i in enumerate(keep) for b, j in enumerate(keep)]
    return PMCModel.from_dict(raw)


def _series(missing: bool) -> np.ndarray:
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0.0, 1.0, 40), rng.normal(1.5, 1.3, 40)])
    y[30] = 6.0
    if missing:
        y[np.arange(len(y)) % 7 == 3] = np.nan
    return y


def _close(a, b, tol, what):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert np.array_equal(np.isnan(a), np.isnan(b)), f"{what}: NaN pattern differs"
    ok = ~np.isnan(a)
    err = float(np.max(np.abs(a[ok] - b[ok]), initial=0.0))
    assert err <= tol, f"{what}: {err:.3g} > {tol:.3g}"


@pytest.mark.parametrize("variant,structure", CASES)
def test_empty_state_inference_equals_reduced_model(variant, structure):
    """forward / backward / classify / FFBS, linear and log space: K = 3 with
    an empty state gives the K = 2 results, the empty state probability 0."""
    MK, MR = _empty_model(variant, structure), _empty_model(variant, structure, KEPT)
    y = _series(missing=False)
    kept = list(KEPT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        passes = {
            "linear": (inf.forward(MK, y), inf.backward(MK, y)),
            "log": (inf._forward_log_space(MK, y), inf._backward_log_space(MK, y)),
        }
        reduced = {
            "linear": (inf.forward(MR, y), inf.backward(MR, y)),
            "log": (inf._forward_log_space(MR, y), inf._backward_log_space(MR, y)),
        }
        cK, cR = classify(MK, y), classify(MR, y)
        XK = inf.sample_posterior(MK, y, np.random.default_rng(5))
        XR = inf.sample_posterior(MR, y, np.random.default_rng(5))
    for space in passes:
        (aK, llK), bK = passes[space]
        (aR, llR), bR = reduced[space]
        assert np.all(aK[:, 1] == 0.0), space
        _close(aK[:, kept], aR, TOL, f"{space} alpha")
        assert abs(llK - llR) <= TOL_LL, (space, llK - llR)
        gK, gR = inf.smooth(aK, bK), inf.smooth(aR, bR)
        assert np.all(gK[:, 1] == 0.0), space
        _close(gK[:, kept], gR, TOL, f"{space} gamma")
    assert np.array_equal(cK[0], np.asarray(kept)[cR[0]])
    _close(cK[1][:, kept], cR[1], TOL, "classify gamma")
    assert abs(cK[2] - cR[2]) <= TOL_LL
    assert np.array_equal(XK, np.asarray(kept)[XR])        # same draws, relabelled


@pytest.mark.parametrize("missing", [False, True], ids=["complete", "gaps"])
@pytest.mark.parametrize("variant,structure", CASES)
def test_empty_state_pit_equals_reduced_model(variant, structure, missing):
    """predictive_pit and flag_outliers (gated and not): K = 3 with an empty
    state gives the K = 2 results — no NaN, the same flags."""
    MK, MR = _empty_model(variant, structure), _empty_model(variant, structure, KEPT)
    y = _series(missing)
    kept = list(KEPT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        PK, PR = predictive_pit(MK, y), predictive_pit(MR, y)
    assert np.array_equal(np.isnan(PK.pit), np.isnan(y))
    for name in ("pit", "pvalue", "z", "log_pred"):
        _close(getattr(PK, name), getattr(PR, name), TOL, name)
    assert abs(PK.log_lik - PR.log_lik) <= TOL_LL
    assert np.all(PK.alpha_hat[:, 1] == 0.0)
    _close(PK.alpha_hat[:, kept], PR.alpha_hat, TOL, "alpha_hat")
    for sequential in (True, False):
        FK = flag_outliers(MK, y, sequential=sequential)
        FR = flag_outliers(MR, y, sequential=sequential)
        assert np.array_equal(FK.flagged, FR.flagged), sequential
        assert FK.flagged[30], sequential                     # the 6-sd spike
        _close(FK.pvalue, FR.pvalue, TOL, f"flag p-values (sequential={sequential})")


def test_empty_state_forecasting_reproduction():
    """The forecasting study's reproduction: an AR(1)(0.3) written as a PMC
    whose first state is empty, y ~ N(0, 1) (seed 0), an 8-sd spike at row
    100. Before the fix: 199 of 200 PITs NaN, nothing flagged."""
    rho = 0.3

    def ar1(p):
        return PMCModel.from_dict({
            "model": {"name": "AR(1)", "K": 2, "variant": "PMC"},
            "prior": {"p": p},
            "margins": [{"i": i, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}}
                        for i in range(2)],
            "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": _tau(rho)}
                        for i in range(2) for j in range(2)]})

    y = np.random.default_rng(0).normal(size=200)
    y[100] = 8.0
    empty = ar1([[0.0, 0.0], [0.0, 1.0]])
    P = predictive_pit(empty, y)
    assert not np.isnan(P.pit).any()
    for sequential in (True, False):
        assert flag_outliers(empty, y, sequential=sequential).index.tolist() == [100]
    # Two identical regimes, both occupied: the same AR(1). 1.1e-16 measured; ×100.
    _close(P.pit, predictive_pit(ar1([[0.45, 0.05], [0.05, 0.45]]), y).pit, 1e-14, "pit")
    # Closed form N(ρ y_{n−1}, 1 − ρ²): 2.2e-16 measured away from the spike.
    # At rows 100–101 the copula argument F(8) = 1 − 6.2e-16 is a float
    # within a few ulps of 1 (the PIT of row 101 is off by 3.6e-4 whatever
    # the prior), a limit of copula arguments, not of the empty state.
    ref = np.r_[stats.norm.cdf(y[0]),
                stats.norm.cdf((y[1:] - rho * y[:-1]) / math.sqrt(1.0 - rho * rho))]
    keep = np.ones(len(y), dtype=bool)
    keep[[100, 101]] = False
    _close(P.pit[keep], ref[keep], 1e-14, "pit vs AR(1)")
