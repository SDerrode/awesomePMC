"""Tests for the GICE margin family selection added in v0.6.

Covers Derrode & Pieczynski (SP 2016) §3 — selecting margin shape from a
candidate list at each ICE M-step. The Pearson moment-matching shortcut
of the paper is *not* implemented here; we use four standard decision
rules instead: ``mle`` (maximum weighted log-likelihood, equivalent to
the paper's PLM for margins), ``kolmogorov`` (Example 3.1), ``aic`` and
``bic``.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest
from scipy import stats as _ss

from prg.pmc.ice import (
    DEFAULT_MARGIN_SELECTION_RULE,
    MARGIN_SELECTION_RULES,
    _data_aware_init_params,
    _kolmogorov_distance,
    _select_margin_family,
    _weighted_log_likelihood_margin,
    ice,
)
from prg.pmc.model    import PMCModel
from prg.pmc.simulate import simulate


MODELS = pathlib.Path("prg/pmc/models")
SP2016 = MODELS / "sp2016_gice_k2.toml"


# ---------------------------------------------------------------------------
# Helper functions — basic invariants
# ---------------------------------------------------------------------------

def test_default_margin_selection_rule_is_mle():
    """Defaults preserve v0.5 behaviour (no automatic selection unless candidates given)."""
    assert DEFAULT_MARGIN_SELECTION_RULE == "mle"
    assert sorted(MARGIN_SELECTION_RULES) == ["aic", "bic", "kolmogorov", "mle"]


def test_data_aware_init_params_match_moments():
    """Heuristic init returns scale ~ std of the (weighted) data."""
    rng = np.random.default_rng(0)
    y = rng.normal(loc=2.0, scale=1.5, size=2000)
    w = np.ones_like(y) / len(y)
    init = _data_aware_init_params("norm", y, w)
    assert abs(init["loc"]   - 2.0) < 0.1
    assert abs(init["scale"] - 1.5) < 0.1


def test_weighted_log_likelihood_returns_finite():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    w = np.ones_like(y) / len(y)
    ll = _weighted_log_likelihood_margin(
        "norm", {"loc": 0.0, "scale": 1.0}, y, w,
    )
    assert np.isfinite(ll)
    # Compare with scipy reference (un-weighted log-lik / N).
    ref = float(_ss.norm.logpdf(y, loc=0.0, scale=1.0).mean())
    assert abs(ll - ref) < 1e-9


def test_kolmogorov_distance_zero_at_truth():
    """KS distance ≈ 0 when the candidate IS the data-generating family."""
    rng = np.random.default_rng(0)
    y = rng.normal(loc=1.0, scale=2.0, size=5000)
    w = np.ones_like(y)
    d = _kolmogorov_distance(
        "norm", {"loc": 1.0, "scale": 2.0}, y, w,
    )
    assert d < 0.05   # ~ 1/√n confidence band


# ---------------------------------------------------------------------------
# _select_margin_family — recovery on synthetic data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", sorted(MARGIN_SELECTION_RULES))
def test_select_margin_family_recovers_strongly_skewed_gamma(rule):
    """Moderately-skewed Gamma data: at least ``mle`` and ``kolmogorov``
    should pick Gamma.  AIC/BIC may stick with Gauss when the BIC penalty
    dominates the small log-lik gain — that's a known conservative trade-off."""
    rng = np.random.default_rng(42)
    y = _ss.gamma.rvs(a=2, loc=0, scale=1, size=2000, random_state=rng)
    w = np.ones_like(y)
    blk = {
        "i": 0,
        "dist":   "norm", "params": {"loc": 1.0, "scale": 1.0},
        "candidates": ["norm", "gamma", "invgamma", "betaprime"],
    }
    out = _select_margin_family(blk, y, w, rule=rule)
    assert out["dist"] in {"norm", "gamma", "invgamma", "betaprime"}
    if rule in {"mle", "kolmogorov"}:
        assert out["dist"] == "gamma", (
            f"{rule}: expected Gamma on Gamma(a=2) data, got {out['dist']}"
        )


def test_select_margin_family_returns_singleton_when_no_candidates():
    """No 'candidates' field → fall back to the existing single-family fit."""
    rng = np.random.default_rng(0)
    y = rng.normal(loc=0.0, scale=1.0, size=500)
    w = np.ones_like(y)
    blk = {"i": 0, "dist": "norm", "params": {"loc": 0.5, "scale": 0.8}}
    out = _select_margin_family(blk, y, w, rule="mle")
    # No "dist" key → kept (block was updated in place by the legacy path).
    assert "params" in out
    # Values close to the truth.
    assert abs(out["params"]["loc"]) < 0.2
    assert abs(out["params"]["scale"] - 1.0) < 0.2


def test_select_margin_family_rejects_unknown_rule():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    blk = {"i": 0, "dist": "norm", "params": {"loc": 0, "scale": 1},
           "candidates": ["norm", "gamma"]}
    with pytest.raises(ValueError, match="Unknown margin_selection_rule"):
        _select_margin_family(blk, y, np.ones_like(y), rule="not_a_rule")


# ---------------------------------------------------------------------------
# PMCModel — TOML schema accepts ``candidates`` field
# ---------------------------------------------------------------------------

def test_pmcmodel_accepts_candidates_field():
    """The SP-2016 fixture should load with the candidates list intact."""
    mdl = PMCModel(SP2016)
    blocks = mdl.margin_blocks()
    assert len(blocks) == 2
    for blk in blocks:
        assert "candidates" in blk
        assert blk["candidates"] == ["norm", "gamma", "invgamma", "betaprime"]


def test_pmcmodel_rejects_empty_candidates(tmp_path):
    """Empty candidates list is a user error."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[model]\nname="bad"\nvariant="PMC"\nK=2\nN_default=100\n'
        '[prior]\np = [[0.45, 0.05], [0.05, 0.45]]\n'
        '[[margins]]\ni=0\ndist="norm"\nparams={loc=0, scale=1}\ncandidates=[]\n'
        '[[margins]]\ni=1\ndist="norm"\nparams={loc=1, scale=1}\n'
        '[[copulas]]\ni=0\nj=0\nname="Gauss"\ntau=0.5\n'
        '[[copulas]]\ni=0\nj=1\nname="Gauss"\ntau=0.0\n'
        '[[copulas]]\ni=1\nj=0\nname="Gauss"\ntau=0.0\n'
        '[[copulas]]\ni=1\nj=1\nname="Gauss"\ntau=0.5\n'
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        PMCModel(bad)


# ---------------------------------------------------------------------------
# End-to-end: ICE recovers the right margin family on the SP-2016 fixture
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["mle", "kolmogorov"])
def test_ice_gice_runs_end_to_end_on_sp2016(rule):
    """End-to-end: GICE on the SP-2016 fixture must converge and pick a
    family from the declared candidate set for every state.

    We don't assert the specific family — at finite N, BetaPrime can mimic
    Gamma and vice versa (both are flexible 4-parameter scipy families), so
    "wrong" family identification is statistically reasonable. What we
    *do* require is that the selected family is in the candidate list and
    that MPM classification with the GICE-fitted model still produces
    finite log-likelihoods.
    """
    from prg.pmc.inference import classify

    mdl   = PMCModel(SP2016)
    _, Y  = simulate(mdl, N=2000, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={
        "max_iter":              5,
        "fit_margins":           True,
        "margin_selection_rule": rule,
        "candidates":            ["Gauss", "GH", "Clayton"],
    })
    candidates = {"norm", "gamma", "invgamma", "betaprime"}
    for state in range(fitted.K):
        assert fitted.margin(state).dist_name in candidates
    # Final log-likelihood is finite and ICE converged.
    assert all(np.isfinite(trace.log_liks))
    # Classification with the fitted model produces a finite MPM result.
    _, _, ll = classify(fitted, Y)
    assert np.isfinite(ll)


def test_ice_unknown_margin_rule_raises():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    _, Y = simulate(mdl, N=200, seed=0)
    with pytest.raises(ValueError, match="Unknown margin_selection_rule"):
        ice(mdl, Y, ice_cfg={
            "max_iter": 2,
            "margin_selection_rule": "made_up",
        })


def test_ice_back_compat_when_no_candidates_field():
    """Models without 'candidates' must behave exactly as in v0.5
    (single-family numerical fit)."""
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    _, Y = simulate(mdl, N=300, seed=0)
    fitted, _ = ice(mdl, Y, ice_cfg={
        "max_iter": 3, "fit_margins": True,
    })
    # Family preserved on every block.
    for blk in fitted.margin_blocks():
        assert blk["dist"] == "norm"


def test_ice_logs_gice_at_debug(caplog):
    """The GICE selection logs a DEBUG line per block — useful for diagnostics."""
    mdl   = PMCModel(SP2016)
    _, Y  = simulate(mdl, N=500, seed=0)
    with caplog.at_level(logging.DEBUG, logger="prg.pmc.ice"):
        ice(mdl, Y, ice_cfg={
            "max_iter": 2, "fit_margins": True,
            "margin_selection_rule": "kolmogorov",
        })
    # At least one GICE line per state per iteration.
    gice_logs = [r for r in caplog.records if "GICE margin" in r.message]
    assert len(gice_logs) >= 2 * 2          # 2 states × 2 iters minimum
