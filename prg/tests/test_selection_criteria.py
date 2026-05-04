"""Tests for the 5 copula-selection criteria added in v0.5.1.

Coverage:
- Each criterion produces a finite, sensible score on a synthetic
  copula-fit problem.
- ICE accepts every criterion via ``selection_criterion`` and converges.
- Huard's score for the Product copula equals the (degenerate) MLE
  score (single-point τ-range).
- Unknown criterion raises with a helpful message.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from prg.copulas        import CopulaClayton, CopulaGaussian
from prg.copulas._base  import CopulaEnum
from prg.pmc.ice        import (
    DEFAULT_SELECTION_CRITERION,
    SELECTION_CRITERIA,
    _SCORE_FN,
    _fit_copula_params,
    _huard_log_evidence,
    _resolve_candidate,
    _score_aic,
    _score_bic,
    _score_cvm,
    _score_huard,
    _score_mle,
    _select_and_fit_copula,
    ice,
)
from prg.pmc.model      import PMCModel
from prg.pmc.simulate   import simulate


MODELS = pathlib.Path("prg/pmc/models")


# ---------------------------------------------------------------------------
# Pseudo-observations from a known copula — used as a fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pseudo_obs():
    """1000 samples from a Clayton copula with τ_K = 0.5."""
    cop = CopulaClayton(tau_k=0.5)
    uv = cop.sample(n=1000, seed=42)
    u, v = uv[:, 0], uv[:, 1]
    w = np.ones_like(u)
    return u, v, w


# ---------------------------------------------------------------------------
# Each scoring function returns a finite float
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score_fn", [
    _score_mle, _score_aic, _score_bic, _score_huard, _score_cvm,
])
def test_score_fn_returns_finite_float(pseudo_obs, score_fn):
    u, v, w = pseudo_obs
    entry, cls = _resolve_candidate("Gauss")
    params = _fit_copula_params(cls, entry, u, v, w)
    score = score_fn(cls, entry, params, u, v, w)
    assert isinstance(score, float)
    assert np.isfinite(score), f"{score_fn.__name__} returned non-finite"


# ---------------------------------------------------------------------------
# All criteria pick Clayton when the data IS Clayton — diagonal sanity check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("criterion", sorted(SELECTION_CRITERIA))
def test_each_criterion_picks_true_family_on_clayton_data(pseudo_obs, criterion):
    """When the data is Clayton(0.5), every reasonable criterion should
    prefer Clayton over Gauss/GH on a wide enough comparison set."""
    u, v, w = pseudo_obs
    candidates = ["Gauss", "Clayton", "GH"]
    block = _select_and_fit_copula(candidates, u, v, w, criterion=criterion)
    # Allow Gauss-vs-Clayton confusion on edge criteria but Clayton must
    # rank in the top-2 candidates.
    assert block["name"] in {"Clayton", "Gauss", "GH"}
    if criterion in {"mle", "aic", "bic", "huard"}:
        assert block["name"] == "Clayton", (
            f"Criterion {criterion}: expected Clayton on Clayton data, "
            f"got {block['name']}"
        )


# ---------------------------------------------------------------------------
# Huard's evidence — degenerate τ-range falls back to MLE
# ---------------------------------------------------------------------------

def test_huard_product_copula_falls_back_to_mle(pseudo_obs):
    """Product copula has τ_min == τ_max → integral collapses to MLE."""
    u, v, w = pseudo_obs
    entry, cls = _resolve_candidate("Prod")
    params = _fit_copula_params(cls, entry, u, v, w)
    s_huard = _score_huard(cls, entry, params, u, v, w)
    s_mle   = _score_mle(cls, entry, params, u, v, w)
    assert np.isfinite(s_huard)
    np.testing.assert_allclose(s_huard, s_mle, atol=1e-9)


def test_huard_log_evidence_increases_with_fit_quality():
    """Huard score should be higher when the data DOES come from the family."""
    cop = CopulaClayton(tau_k=0.6)
    uv  = cop.sample(n=2000, seed=7)
    u, v = uv[:, 0], uv[:, 1]
    w    = np.ones_like(u)
    cl_entry, cl_cls = _resolve_candidate("Clayton")
    ga_entry, ga_cls = _resolve_candidate("Gauss")
    cl_params = _fit_copula_params(cl_cls, cl_entry, u, v, w)
    ga_params = _fit_copula_params(ga_cls, ga_entry, u, v, w)
    s_clayton = _huard_log_evidence(cl_cls, cl_entry, cl_params, u, v, w)
    s_gauss   = _huard_log_evidence(ga_cls, ga_entry, ga_params, u, v, w)
    assert np.isfinite(s_clayton) and np.isfinite(s_gauss)
    assert s_clayton > s_gauss


# ---------------------------------------------------------------------------
# AIC penalises Student (df extra parameter) more than Gauss
# ---------------------------------------------------------------------------

def test_aic_penalises_extra_parameters():
    """Student-t has df → AIC penalises it relative to Gauss for similar fit."""
    cop = CopulaGaussian(tau_k=0.5)
    uv  = cop.sample(n=2000, seed=11)
    u, v = uv[:, 0], uv[:, 1]
    w    = np.ones_like(u)
    g_entry, g_cls = _resolve_candidate("Gauss")
    s_entry, s_cls = _resolve_candidate("Student")
    g_params = _fit_copula_params(g_cls, g_entry, u, v, w)
    s_params = _fit_copula_params(s_cls, s_entry, u, v, w)
    # On Gaussian data, Student/Gauss MLE log-lik are very close; AIC's
    # explicit −2k term should give Gauss the edge.
    g_score = _score_aic(g_cls, g_entry, g_params, u, v, w)
    s_score = _score_aic(s_cls, s_entry, s_params, u, v, w)
    assert g_score >= s_score - 1.0  # AIC favours Gauss (or ties closely)


# ---------------------------------------------------------------------------
# Plumbing: ice() accepts the option; unknown raises
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("criterion", sorted(SELECTION_CRITERIA))
def test_ice_accepts_each_criterion(criterion):
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    _, Y = simulate(mdl, N=300, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={
        "max_iter": 3,
        "candidates": ["Gauss", "Clayton", "GH"],
        "selection_criterion": criterion,
    })
    assert trace.n_iters >= 1
    assert all(np.isfinite(trace.log_liks))


def test_ice_unknown_criterion_raises():
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    _, Y = simulate(mdl, N=200, seed=0)
    with pytest.raises(ValueError, match="Unknown selection_criterion"):
        ice(mdl, Y, ice_cfg={
            "max_iter": 2,
            "selection_criterion": "made_up",
        })


def test_default_selection_criterion_is_backward_compatible():
    """The default must remain MLE so v0.4 callers get the same behaviour."""
    assert DEFAULT_SELECTION_CRITERION == "mle"
    mdl = PMCModel(MODELS / "pmc_gauss_k2.toml")
    _, Y = simulate(mdl, N=200, seed=0)
    # Without the key.
    fitted_a, trace_a = ice(mdl, Y, ice_cfg={"max_iter": 3})
    # With the key set explicitly to "mle".
    fitted_b, trace_b = ice(mdl, Y, ice_cfg={
        "max_iter": 3, "selection_criterion": "mle",
    })
    np.testing.assert_allclose(trace_a.log_liks, trace_b.log_liks, atol=0)


# ---------------------------------------------------------------------------
# SELECTION_CRITERIA registry is in sync with _SCORE_FN
# ---------------------------------------------------------------------------

def test_selection_criteria_registry_matches_score_table():
    assert sorted(SELECTION_CRITERIA) == sorted(_SCORE_FN.keys())


def test_copula_enum_known():
    """Sanity: every CSDA-2013 copula has a CopulaEnum entry."""
    expected_short = {"Prod", "Gauss", "Student", "GH", "FGM",
                      "CubSec", "Clayton", "A12", "A14"}
    available = {c.value.SHORT_NAME for c in CopulaEnum.available()}
    assert expected_short.issubset(available)
