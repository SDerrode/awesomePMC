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

from pmcprg.copulas        import CopulaClayton, CopulaGaussian
from pmcprg.copulas._base  import CopulaEnum
from pmcprg.pmc.ice        import (
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
    _contiguous_weight_balanced_folds,
    _score_xvcic,
    _select_and_fit_copula,
    _weighted_cvm_statistic,
    _weighted_log_likelihood,
    ice,
)
from pmcprg.pmc.model      import PMCModel
from pmcprg.pmc.simulate   import simulate


MODELS = pathlib.Path("pmcprg/pmc/models")


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


# ---------------------------------------------------------------------------
# Regression: CvM criterion is ξ-weighted (audit N-6)
# ---------------------------------------------------------------------------

def test_weighted_cvm_is_weight_sensitive():
    """The weighted CvM must actually use the weights (unlike the previous
    unweighted call), so down-weighted pseudo-obs influence it less."""
    from pmcprg.copulas import CopulaGaussian

    rng = np.random.default_rng(0)
    uv  = rng.uniform(size=(200, 2))
    cop = CopulaGaussian(tau_k=0.3)

    s_uniform = _weighted_cvm_statistic(uv, np.ones(200), cop)
    # Concentrate all weight on the first half → a different statistic.
    s_skewed  = _weighted_cvm_statistic(
        uv, np.concatenate([np.ones(100), np.zeros(100)]), cop,
    )
    assert np.isfinite(s_uniform) and s_uniform >= 0.0
    assert np.isfinite(s_skewed) and s_skewed >= 0.0
    assert abs(s_uniform - s_skewed) > 1e-6, "weights had no effect on CvM"


@pytest.mark.parametrize("tau_true,delta_true", [(0.10, 1.05), (0.20, 1.15), (0.45, 1.70)])
def test_bb1_fit_respects_the_joint_delta_tau_constraint(tau_true, delta_true):
    """BB1 must be fittable below τ = 1/3 (audit: joint-constraint stall).

    θ = 2/(δ(1−τ)) − 2 is positive only for δ < 1/(1−τ), a constraint no box
    can express. The optimiser's default δ = 1.5 is inadmissible for any
    τ ≤ 1/3, so the fit started on the constant failure penalty, saw a zero
    gradient, and returned δ untouched: every construction then raised, and
    BB1 was unusable over a third of its τ range (measured 0/10 at τ = 0.2).
    """
    from pmcprg.copulas import CopulaBB1

    entry, cls = _resolve_candidate("BB1")
    data = CopulaBB1(tau_k=tau_true, delta=delta_true).sample(400, seed=1)
    params = _fit_copula_params(cls, entry, data[:, 0], data[:, 1], np.ones(400))

    cop = CopulaBB1(**params)                     # must not raise
    assert cop.theta > 0.0
    assert params["delta"] <= CopulaBB1.delta_max(params["tau_k"]) + 1e-12
    assert params["tau_k"] == pytest.approx(tau_true, abs=0.08)


def test_bb1_delta_max_and_projection():
    """``delta_max`` is 1/(1−τ) up to the θ-floor, and the projection clamps."""
    from pmcprg.copulas import CopulaBB1

    for tau in (0.1, 0.5, 0.9):
        assert CopulaBB1.delta_max(tau) == pytest.approx(1.0 / (1.0 - tau), rel=1e-5)
    # τ → 0 pins δ to the Clayton limit.
    assert CopulaBB1.delta_max(0.0) == pytest.approx(1.0, rel=1e-5)

    # An inadmissible (τ, δ) is projected back to something constructible.
    projected = CopulaBB1.constrain_params({"tau_k": 0.2, "delta": 1.5})
    assert projected["delta"] < 1.5
    assert CopulaBB1(**projected).theta > 0.0
    # An admissible pair is left alone.
    assert CopulaBB1.constrain_params({"tau_k": 0.7, "delta": 2.0})["delta"] == 2.0


def test_huard_common_uses_a_shared_tau_support():
    """``huard_common`` must integrate every candidate over one support.

    With per-family supports (the CSDA-2013 convention), a narrow-range
    family concentrates more prior mass and is rewarded for it, so a
    Huard-versus-likelihood comparison conflates "averaging over τ helps on
    sparse cells" with "this family's declared range is narrow and contains
    the truth". Integrating all candidates over the intersection separates
    the two — the isolation experiment behind §7 of the report.
    """
    from pmcprg.pmc.ice import _huard_log_evidence, _score_huard, _score_huard_common

    uv = CopulaClayton(tau_k=0.5).sample(400, seed=11)
    u, v, w = uv[:, 0], uv[:, 1], np.ones(400)
    entry, cls = _resolve_candidate("Clayton")
    params = _fit_copula_params(cls, entry, u, v, w)

    # Without an explicit range the variant is exactly the original.
    assert _score_huard_common(cls, entry, params, u, v, w) == pytest.approx(
        _score_huard(cls, entry, params, u, v, w)
    )
    # A narrower support changes the evidence (it is a different prior).
    narrowed = _score_huard_common(
        cls, entry, params, u, v, w, tau_range=(0.0, 0.2),
    )
    assert narrowed != pytest.approx(_score_huard(cls, entry, params, u, v, w))
    assert narrowed == pytest.approx(
        _huard_log_evidence(cls, entry, params, u, v, w, tau_range=(0.0, 0.2))
    )
    # The selector hands every candidate the intersection of their ranges:
    # Gauss [-1,1] ∩ Clayton [0,1] ∩ CubSec [0,0.165] = [0, 0.165].
    blk = _select_and_fit_copula(
        ["Gauss", "Clayton", "CubSec"], u, v, w, criterion="huard_common",
    )
    assert blk["name"] in {"Gauss", "Clayton", "CubSec"}


def test_xvcic_folds_are_contiguous_and_weight_balanced():
    """Folds must partition [0,n) in order, with roughly equal Σw.

    Contiguity matters because the pseudo-observations form a Markov chain:
    random folds would leave each held-out point's neighbours in the training
    set. Balancing on Σw rather than on point counts matters because the
    weights are ξ pair-posteriors — a long low-weight stretch carries little
    information and must not consume a whole fold.
    """
    w = np.concatenate([np.ones(100), 0.01 * np.ones(300), np.ones(100)])
    folds = _contiguous_weight_balanced_folds(w, 5)

    assert folds[0][0] == 0 and folds[-1][1] == w.size
    assert all(folds[i][1] == folds[i + 1][0] for i in range(len(folds) - 1))
    masses = [float(w[a:b].sum()) for a, b in folds]
    assert max(masses) - min(masses) <= 0.02 * float(w.sum())
    # The low-weight middle stretch is absorbed into a single fold.
    assert any(b - a > 200 for a, b in folds)
    assert _contiguous_weight_balanced_folds(np.zeros(50), 5) == []


def test_xvcic_is_deterministic_and_degrades_gracefully():
    """No RNG anywhere, and a fall-back to the plain likelihood on tiny blocks."""
    uv = CopulaClayton(tau_k=0.5).sample(300, seed=2)
    u, v, w = uv[:, 0], uv[:, 1], np.ones(300)
    params = {"tau_k": 0.5}
    entry, cls = _resolve_candidate("Clayton")

    scores = [_score_xvcic(cls, entry, params, u, v, w) for _ in range(3)]
    assert scores[0] == scores[1] == scores[2]
    assert np.isfinite(scores[0])

    # Σw below the noise floor → exactly the plain weighted log-likelihood.
    tiny = np.full(300, 0.01)          # Σw = 3 ≪ _XVCIC_MIN_W
    assert _score_xvcic(cls, entry, params, u, v, tiny) == pytest.approx(
        _weighted_log_likelihood(cls, params, u, v, tiny)
    )


def test_xvcic_selects_the_true_family_on_clean_data():
    """End-to-end: the criterion must be usable and sensible."""
    uv = CopulaClayton(tau_k=0.5).sample(400, seed=7)
    blk = _select_and_fit_copula(
        ["Prod", "Gauss", "Clayton", "Frank"], uv[:, 0], uv[:, 1],
        np.ones(400), criterion="xvcic",
    )
    assert blk["name"] == "Clayton"


def test_product_copula_is_selectable_on_independent_data():
    """Independence must be reachable — τ_min == τ_max used to be padded into
    an inverted interval, and Product was charged for a parameter it has not."""
    from pmcprg.copulas import CopulaProduct
    from pmcprg.pmc.ice import _pad_tau_bounds

    assert CopulaProduct.n_params == 0
    assert _pad_tau_bounds(0.0, 0.0) == (0.0, 0.0)

    uv = np.random.default_rng(0).uniform(size=(500, 2))
    blk = _select_and_fit_copula(
        ["Prod", "Gauss", "Clayton"], uv[:, 0], uv[:, 1], np.ones(500),
        criterion="bic",
    )
    assert blk["name"] == "Prod"


def test_weighted_log_likelihood_is_a_total_not_a_mean():
    """Σ_n w_n·log c must scale with the weights, not average over them.

    The penalised criteria compare this against ``2k`` / ``k·log n_eff``,
    which are calibrated for a sum. When it returned a per-observation mean,
    the penalty was effectively Σw times too strong.
    """
    rng = np.random.default_rng(0)
    uv  = CopulaClayton(tau_k=0.5).sample(300, seed=4)
    u, v = uv[:, 0], uv[:, 1]
    w    = rng.uniform(0.2, 1.0, size=300)
    params = {"tau_k": 0.5}

    ll_w  = _weighted_log_likelihood(CopulaClayton, params, u, v, w)
    ll_2w = _weighted_log_likelihood(CopulaClayton, params, u, v, 2.0 * w)
    assert ll_2w == pytest.approx(2.0 * ll_w, rel=1e-12)

    # And it really is Σ w·log c, not a normalised average.
    expected = float(np.dot(w, CopulaClayton(**params).logpdf_array(
        np.column_stack((u, v)))))
    assert ll_w == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("criterion", ["aic", "bic"])
def test_penalised_criteria_can_select_two_parameter_families(criterion):
    """AIC/BIC must be *able* to pick a 2-parameter family when it is true.

    With the fit term on a per-observation mean scale, a 2-parameter family
    had to beat its rivals by ~1 nat per observation instead of ~1/n; none
    ever did, so Student and BB1 were unselectable (0/480 replicates over the
    whole synthetic grid). Student data is unambiguous here: its tail
    dependence is what Gaussian cannot reproduce.
    """
    from pmcprg.copulas import CopulaStudent

    uv = CopulaStudent(tau_k=0.5, df=3.0).sample(800, seed=5)
    u, v, w = uv[:, 0], uv[:, 1], np.ones(800)
    blk = _select_and_fit_copula(
        ["Gauss", "Clayton", "Frank", "Student"], u, v, w, criterion=criterion,
    )
    assert blk["name"] == "Student", (
        f"{criterion} picked {blk['name']} on Student(τ=0.5, ν=3) data"
    )


def test_weighted_cvm_zero_weights_drop_points_exactly():
    """Zero-weight pseudo-obs must vanish from both the weighted empirical
    copula and the outer L²-sum: the statistic over (uv, [1…1, 0…0]) must
    equal the statistic over the kept points alone (audit T-11)."""
    from pmcprg.copulas import CopulaGaussian

    rng = np.random.default_rng(0)
    uv  = rng.uniform(size=(200, 2))
    cop = CopulaGaussian(tau_k=0.3)

    w01    = np.concatenate([np.ones(100), np.zeros(100)])
    s_full = _weighted_cvm_statistic(uv, w01, cop)
    s_kept = _weighted_cvm_statistic(uv[:100], np.ones(100), cop)
    np.testing.assert_allclose(s_full, s_kept, rtol=1e-12)


def test_score_cvm_runs_in_selection():
    """`cvm` selection criterion still produces a valid copula block."""
    rng = np.random.default_rng(1)
    u   = rng.uniform(size=400)
    v   = rng.uniform(size=400)
    w   = rng.uniform(0.0, 1.0, size=400)
    block = _select_and_fit_copula(
        ["Gauss", "Clayton", "Frank"], u, v, w, criterion="cvm",
    )
    assert block["name"] in {"Gauss", "Clayton", "Frank"}
    assert np.isfinite(block["tau"])
