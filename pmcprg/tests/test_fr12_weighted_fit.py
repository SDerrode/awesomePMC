"""The public weighted copula fit and its per-fit diagnostics — audit AUDIT_COPULES FR-12.

``CopulaVirt.fit(data, method, *, weights=None, pseudo_obs=False)`` and
``CopulaVirt.fit_best(..., *, weights=None, criterion='aic', pseudo_obs=False)``
run the weighted engine of ICE's M-step (:mod:`pmcprg.copulas._weighted`);
``FitResult`` reports the optimiser's status and, on demand, the gradient,
Hessian eigenvalues and boundary flag of the estimate.

Every tolerance below is a small multiple of an error measured on these data
(numpy 2 / scipy 1.18 and the minimum versions numpy 1.24 / scipy 1.10);
every sample has a fixed seed. The parity of the same engine against
pyvinecopulib and VineCopula is ``test_parity_weighted.py``.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from scipy.stats import kendalltau, rankdata

from pmcprg.copulas import (
    CopulaBB1,
    CopulaBB190,
    CopulaClayton,
    CopulaEnum,
    CopulaFrank,
    CopulaGaussian,
    CopulaGH,
    CopulaProduct,
    CopulaStudent,
    CopulaTawn1,
    CopulaVirt,
    FitBestResults,
    FitDiagnostics,
    validate_weights,
    weighted_kendall_tau,
    weighted_pseudo_obs,
)
from pmcprg.copulas._stderr import standard_errors
from pmcprg.diagnostics.model_selection import comparison_matrix
from pmcprg.pmc.ice import _fit_copula_params, _select_and_fit_copula, _weighted_kendall_tau

FAMILIES = [
    (CopulaGaussian, {"tau_k": 0.4}),
    (CopulaClayton, {"tau_k": 0.4}),
    (CopulaFrank, {"tau_k": -0.3}),
    (CopulaGH, {"tau_k": 0.5}),
    (CopulaStudent, {"tau_k": 0.4, "df": 4.0}),
    (CopulaBB1, {"tau_k": 0.5, "delta": 1.5}),
]
IDS = [cls.__name__ for cls, _ in FAMILIES]


def _sample(cls, kw, n=400, seed=11):
    return cls(**kw).sample(n, seed=seed)


def _ranks(x):
    n = x.shape[0]
    return np.column_stack([rankdata(x[:, 0]), rankdata(x[:, 1])]) / (n + 1)


def _weights(n, seed=3):
    return np.random.default_rng(seed).beta(0.5, 0.5, n)


def _same_fit(a, b):
    """Bit for bit: parameters, log-likelihood, pseudo-observations, status."""
    assert type(a.copula) is type(b.copula)
    assert dict(a.copula.params) == dict(b.copula.params)
    assert a.tau_k == b.tau_k and a.method == b.method
    assert a.log_likelihood == b.log_likelihood
    np.testing.assert_array_equal(a.uv, b.uv)
    assert a.converged == b.converged and a.n_obs == b.n_obs


# ---------------------------------------------------------------------------
# Weight validation — one place for ICE and the public API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad, match", [
    (np.ones((10, 2)), "1-D"),
    (np.ones(9), "same length"),
    (np.r_[np.ones(9), np.nan], "finite and non-negative"),
    (np.r_[np.ones(9), np.inf], "finite and non-negative"),
    (np.r_[np.ones(9), -1.0], "finite and non-negative"),
    (np.zeros(10), "positive sum"),
    (["a"] * 10, "array of numbers"),
])
def test_invalid_weights_raise_a_clear_value_error(bad, match):
    data = _sample(CopulaGaussian, {"tau_k": 0.3}, n=10)
    with pytest.raises(ValueError, match=match):
        validate_weights(bad, 10)
    for method in ("tau", "mle"):
        with pytest.raises(ValueError, match=match):
            CopulaGaussian.fit(data, method=method, weights=bad)
    with pytest.raises(ValueError, match=match):
        CopulaVirt.fit_best(data, families=[CopulaGaussian], weights=bad)
    with pytest.raises(ValueError, match=match):
        weighted_pseudo_obs(data, bad)


def test_validation_returns_the_same_array_and_allows_a_zero_sum_when_asked():
    w = np.linspace(0.0, 1.0, 7)
    assert validate_weights(w, 7) is w                    # no copy: ICE stays bit-identical
    z = np.zeros(4)
    assert validate_weights(z, 4, allow_zero_sum=True) is z
    with pytest.raises(ValueError, match="positive sum"):
        validate_weights(z, 4)


def test_too_few_positive_weights_and_unknown_method_raise():
    data = _sample(CopulaGaussian, {"tau_k": 0.3}, n=20)
    w = np.zeros(20)
    w[:3] = [0.5, 2.0, 1.0]
    with pytest.raises(ValueError, match="positive weight"):
        CopulaGaussian.fit(data, method="mle", weights=w)
    with pytest.raises(ValueError, match="method"):
        CopulaGaussian.fit(data, method="ml", weights=np.full(20, 0.5))


# ---------------------------------------------------------------------------
# Unit weights, {0, 1} weights, scale
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, kw", FAMILIES + [(CopulaBB190, {"tau_k": -0.5, "delta": 1.5}),
                                                (CopulaTawn1, {"tau_k": 0.4, "psi": 0.7})],
                         ids=IDS + ["CopulaBB190", "CopulaTawn1"])
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_unit_weights_are_no_weights_bit_for_bit(cls, kw, method):
    data = _sample(cls, kw, n=300)
    a = cls.fit(data, method=method)
    b = cls.fit(data, method=method, weights=np.ones(300))
    _same_fit(a, b)
    assert a.weights is None and b.weights is None


@pytest.mark.parametrize("cls, kw", FAMILIES, ids=IDS)
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_zero_one_weights_are_the_fit_on_the_subset_bit_for_bit(cls, kw, method):
    data = _sample(cls, kw, n=300)
    keep = np.random.default_rng(5).random(300) < 0.6
    a = cls.fit(data[keep], method=method)
    b = cls.fit(data, method=method, weights=keep.astype(float))
    _same_fit(a, b)
    # Zero weights drop their rows in the weighted engine too: same fit with or without them.
    w = _weights(300) * keep
    _same_fit(cls.fit(data[keep], method=method, weights=w[keep]),
              cls.fit(data, method=method, weights=w))


# Multiplying every weight by c (1/3 … 1e6, down to 2^-20), pseudo-observations
# given. Measured on these samples, largest parameter change: one parameter
# 1.7e-15 ('tau'), 1.5e-11 ('mle', Brent on the Σw-normalised objective); two
# parameters 1.8e-10 (the profile of itau), 9.7e-7 (the joint MLE, along the
# flat direction ν of the likelihood). ℓ(c·w)/c against ℓ(w): 6.1e-13
# relative (BB1, c = 2^-20). The one-parameter 'mle' is a Brent search whose
# objective differs by rounding under c: a flipped comparison changes its
# path, so the result moves by up to its xatol in principle — measured
# 1.5e-11 on macOS arm64, 3.95e-11 on the Linux runners (Clayton, c = 1e-3).
TOL_SCALE = {("tau", 1): 1e-14, ("mle", 1): 1e-9, ("tau", 2): 1e-8, ("mle", 2): 1e-5}
TOL_SCALE_LOGLIK_REL = 3e-12


@pytest.mark.parametrize("cls, kw", FAMILIES, ids=IDS)
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_scaling_the_weights_leaves_the_estimate_unchanged(cls, kw, method):
    uv = _ranks(_sample(cls, kw))
    w = _weights(400)
    base = cls.fit(uv, method=method, weights=w, pseudo_obs=True)
    k = len(base.copula.params)
    for c in (1 / 3, 7.0, 1e3, 1e6, 1e-3, 2.0 ** -20):
        r = cls.fit(uv, method=method, weights=c * w, pseudo_obs=True)
        dp = max(abs(r.copula.params[p] - base.copula.params[p]) for p in base.copula.params)
        assert dp <= TOL_SCALE[(method, min(k, 2))], (c, dp)
        # The reported log-likelihood is the total Σ wᵢ log cᵢ: it scales with c.
        assert abs(r.log_likelihood / c - base.log_likelihood) <= \
            TOL_SCALE_LOGLIK_REL * abs(base.log_likelihood), c
        assert r.n_eff == pytest.approx(c * base.n_eff, rel=1e-12)


def test_the_rank_transform_is_on_the_count_scale_of_the_weights():
    """FR-7(a)'s ``+ 1`` counts one observation: frequency weights.

    Measured on n = 400, Σw = 210: rescaling the weights by 10³ moves the
    pseudo-observations by 4.7e-3; to Σw = 0.21 by 0.82 — which the fit warns
    about (Σw < 10).
    """
    x = _sample(CopulaGaussian, {"tau_k": 0.4}, seed=12)
    w = _weights(400, seed=7)
    u1 = weighted_pseudo_obs(x, w)
    assert 1e-3 < np.abs(weighted_pseudo_obs(x, 1e3 * w) - u1).max() < 1e-2
    # Hand check of the convention: Σ_{x_k ≤ x_i} w_k / (Σw + 1).
    i = 17
    expect = w[x[:, 0] <= x[i, 0]].sum() / (w.sum() + 1.0)
    assert u1[i, 0] == pytest.approx(expect, rel=1e-15)


def test_weights_summing_below_one_warn(caplog):
    x = _sample(CopulaGaussian, {"tau_k": 0.4}, n=100, seed=12)
    w = _weights(100)
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._weighted"):
        CopulaGaussian.fit(x, method="mle", weights=w / w.sum())          # ranked with Σw = 1
    assert any("frequency weights" in r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="pmcprg.copulas._weighted"):
        CopulaGaussian.fit(_ranks(x), method="mle", weights=w / w.sum(), pseudo_obs=True)
        CopulaGaussian.fit(x, method="mle", weights=w)                    # Σw ≈ 50
    assert not any("frequency weights" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# pseudo_obs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, kw", FAMILIES, ids=IDS)
@pytest.mark.parametrize("method", ["tau", "mle"])
def test_pseudo_obs_true_and_false_agree(cls, kw, method):
    x = _sample(cls, kw)
    w = _weights(400)
    # Unweighted: ranks computed here, or by the fit.
    _same_fit(cls.fit(x, method=method), cls.fit(_ranks(x), method=method, pseudo_obs=True))
    # Rank pseudo-observations are a fixed point of the rank transform.
    _same_fit(cls.fit(_ranks(x), method=method), cls.fit(_ranks(x), method=method, pseudo_obs=True))
    # Weighted: the weighted ECDF computed here, or by the fit.
    a = cls.fit(x, method=method, weights=w)
    b = cls.fit(weighted_pseudo_obs(x, w), method=method, weights=w, pseudo_obs=True)
    _same_fit(a, b)
    np.testing.assert_array_equal(a.weights, b.weights)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, np.nan])
def test_pseudo_obs_outside_the_open_square_raise(bad):
    uv = _ranks(_sample(CopulaGaussian, {"tau_k": 0.3}, n=50))
    uv[3, 1] = bad
    with pytest.raises(ValueError, match="pseudo_obs"):
        CopulaGaussian.fit(uv, method="mle", pseudo_obs=True)
    with pytest.raises(ValueError, match="pseudo_obs"):
        CopulaGaussian.fit(uv, method="mle", weights=_weights(50), pseudo_obs=True)


# ---------------------------------------------------------------------------
# One engine: the public fit is ICE's M-step fit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("short", ["Gauss", "Clayton", "Frank", "GH", "Joe", "Student", "BB1",
                                   "BB190", "Tawn1", "Prod"])
def test_the_weighted_mle_is_ices_bit_for_bit(short):
    entry = CopulaEnum.from_short_name(short)
    cls = entry.klass
    kw = {"tau_k": -0.4 if entry.value.TAU_MIN_MAX[1] <= 0 else 0.4}
    if short == "Prod":
        kw = {"tau_k": 0.0}
    uv = _ranks(cls(**cls.constructible_params(kw)).sample(300, seed=9))
    w = _weights(300, seed=4)
    ice = _fit_copula_params(cls, entry, uv[:, 0], uv[:, 1], w)
    r = cls.fit(uv, method="mle", weights=w, pseudo_obs=True)
    assert dict(r.copula.params) == ice
    assert r.converged and not r.failed and r.n_iter is not None and r.n_eval >= 0


def test_weighted_itau_inverts_the_weighted_tau():
    uv = _ranks(_sample(CopulaGaussian, {"tau_k": 0.45}))
    w = _weights(400)
    r = CopulaGaussian.fit(uv, method="tau", weights=w, pseudo_obs=True)
    assert r.tau_k == _weighted_kendall_tau(uv[:, 0], uv[:, 1], w)
    assert r.tau_k == weighted_kendall_tau(uv[:, 0], uv[:, 1], w)
    assert r.n_iter == 0 and r.n_eval == 0 and r.converged
    # Clipped into the constructible range, like the unweighted fit (Clayton: τ̂ < 0 → ε).
    neg = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]])
    rc = CopulaClayton.fit(neg, method="tau", weights=w, pseudo_obs=True)
    assert rc.tau_k == CopulaEnum.CLAYTON.constructible_tau_range()[0]


def test_weighted_kendall_tau_without_weights_is_scipys_tau_b():
    rng = np.random.default_rng(1)
    u = rng.integers(0, 7, 500).astype(float)
    v = np.round(u + rng.normal(0.0, 2.0, 500))
    assert weighted_kendall_tau(u, v) == pytest.approx(kendalltau(u, v)[0], abs=1e-14)
    with pytest.raises(ValueError, match="same length"):
        weighted_kendall_tau(u, v[:-1])


# ---------------------------------------------------------------------------
# fit_best
# ---------------------------------------------------------------------------

FB_FAMILIES = [CopulaGaussian, CopulaClayton, CopulaGH, CopulaFrank, CopulaStudent, CopulaBB1]


@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_best_default_is_the_aic_ranking(method):
    data = _sample(CopulaStudent, {"tau_k": 0.45, "df": 5.0}, n=300)
    a = CopulaVirt.fit_best(data, families=FB_FAMILIES, method=method)
    b = CopulaVirt.fit_best(data, families=FB_FAMILIES, method=method, criterion="aic")
    assert isinstance(a, FitBestResults) and a.criterion == "aic"
    assert [r.aic for r in a] == sorted(r.aic for r in a)
    assert [(type(r.copula), r.log_likelihood) for r in a] == \
        [(type(r.copula), r.log_likelihood) for r in b]


def test_fit_best_criteria():
    data = _sample(CopulaStudent, {"tau_k": 0.45, "df": 5.0}, n=300)
    w = _weights(300)
    for crit, key in (("bic", lambda r: r.bic), ("loglik", lambda r: -r.log_likelihood)):
        res = CopulaVirt.fit_best(data, families=FB_FAMILIES, method="mle", weights=w,
                                  criterion=crit)
        assert res.criterion == crit
        assert [key(r) for r in res] == sorted(key(r) for r in res)
    r = res[0]
    assert r.bic == r.n_params * math.log(w.sum()) - 2.0 * r.log_likelihood   # log Σw
    with pytest.raises(ValueError, match="criterion"):
        CopulaVirt.fit_best(data, families=FB_FAMILIES, criterion="hqc")


@pytest.mark.parametrize("criterion, ice_criterion", [("aic", "aic"), ("bic", "bic"),
                                                      ("loglik", "mle")])
def test_weighted_fit_best_selects_as_ice(criterion, ice_criterion):
    shorts = ["Prod", "Gauss", "Clayton", "GH", "Frank", "Student", "BB1"]
    classes = [CopulaEnum.from_short_name(s).klass for s in shorts]
    for seed in range(3):
        uv = _ranks(CopulaBB1(tau_k=0.45, delta=1.4).sample(400, seed=seed))
        w = _weights(400, seed=seed)
        res = CopulaVirt.fit_best(uv, families=classes, method="mle", weights=w,
                                  criterion=criterion, pseudo_obs=True)
        blk = _select_and_fit_copula(shorts, uv[:, 0], uv[:, 1], w, criterion=ice_criterion)
        best = res[0].copula
        assert best.copula_enum.value.SHORT_NAME == blk["name"]
        assert best.params["tau_k"] == blk["tau"]
        assert all(best.params[k] == v for k, v in blk.items() if k not in ("name", "tau"))


def test_weighted_fit_best_comparison_uses_the_weights():
    uv = _ranks(_sample(CopulaGaussian, {"tau_k": 0.4}))
    w = _weights(400)
    fb = CopulaVirt.fit_best(uv, families=[CopulaGaussian, CopulaClayton, CopulaFrank],
                             method="mle", weights=w, pseudo_obs=True)
    mat = fb.compare(bandwidth=0)
    logc = {r.copula.copula_enum.value.SHORT_NAME: r.copula.logpdf_array(r.uv) for r in fb}
    ref = comparison_matrix(logc, w, n_params={k: 1 for k in logc}, bandwidth=0)
    np.testing.assert_array_equal(mat.statistic, ref.statistic)


# ---------------------------------------------------------------------------
# Status and what a weighted FitResult refuses
# ---------------------------------------------------------------------------

def test_a_failed_weighted_fit_is_flagged(monkeypatch, caplog):
    monkeypatch.setattr(CopulaClayton, "logpdf_array", lambda self, uv: np.full(len(uv), np.nan))
    uv = _ranks(_sample(CopulaGaussian, {"tau_k": 0.4}, n=100))
    with caplog.at_level(logging.WARNING):
        r = CopulaClayton.fit(uv, method="mle", weights=_weights(100), pseudo_obs=True)
    assert r.failed and not r.converged
    assert "not finite" in r.message
    assert any("CopulaClayton" in rec.getMessage() for rec in caplog.records)
    assert "not converged" in repr(r)


def test_weighted_result_refuses_the_unweighted_resampling_tools():
    x = _sample(CopulaGaussian, {"tau_k": 0.4}, n=200)
    w = _weights(200)
    r = CopulaGaussian.fit(x, method="mle", weights=w)
    for call in (lambda: r.gof_test(B=2), lambda: r.bootstrap_ci(B=2), lambda: r.cv_loglik(K=2),
                 lambda: r.plot_diagnostics("/nonexistent")):
        with pytest.raises(NotImplementedError, match="weighted"):
            call()
    se = r.standard_errors()                              # the weights are passed on
    ref = standard_errors(r.copula, r.uv, w, "mle")
    assert se.se == ref.se and se.n_eff == pytest.approx(w.sum())
    assert "Σw=" in repr(r)


# ---------------------------------------------------------------------------
# itau for multi-parameter families — the intended change of FR-12
# ---------------------------------------------------------------------------

def test_the_generic_tau_path_no_longer_leaves_the_extra_parameter_at_its_start():
    """``CopulaVirt.fit(method='tau')`` used to build a multi-parameter family at
    the registry's start value of its extra parameter (ν = 4 for Student). It
    now fits it by maximum likelihood at τ̂ (a profile likelihood)."""
    data = CopulaStudent(tau_k=0.7, df=3.0).sample(2000, seed=0)
    r = CopulaVirt.fit.__func__(CopulaStudent, data, method="tau")    # the generic path
    assert r.copula.df != 4.0 and r.method == "tau" and r.converged
    _same_fit(r, CopulaStudent.fit(data, method="tau"))


# Student(τ = 0.7, ν = 3), n = 2000, 10 seeds: itau ν̂ mean 3.22, sd 0.31,
# max |ν̂ − 3| 0.77 (the joint MLE: mean 3.23, sd 0.33). Seeds 0–4 below.
@pytest.mark.parametrize("seed", range(5))
def test_itau_recovers_student_nu(seed):
    data = CopulaStudent(tau_k=0.7, df=3.0).sample(2000, seed=seed)
    r = CopulaStudent.fit(data, method="tau")
    assert r.tau_k == kendalltau(data[:, 0], data[:, 1])[0]
    assert abs(r.copula.df - 3.0) < 1.2                 # ≈ 4 sd; 0.77 measured over 10 seeds
    assert not r.diagnostics.at_boundary


def test_weighted_itau_recovers_student_nu():
    """Generic weights, n = 3000, five seeds: ν̂ 3.40–4.89 for ν = 4 (measured)."""
    nus = []
    for seed in range(5):
        data = CopulaStudent(tau_k=0.5, df=4.0).sample(3000, seed=100 + seed)
        r = CopulaStudent.fit(data, method="tau", weights=_weights(3000, seed=seed))
        assert r.converged and r.method == "tau"
        nus.append(r.copula.df)
    assert 3.0 < np.mean(nus) < 5.2 and max(abs(np.array(nus) - 4.0)) < 1.5


# BB1(τ = 0.6, δ = 2), n = 2000, 10 seeds: itau δ̂ mean 1.97, sd 0.067, max err 0.19.
def test_itau_recovers_bb1_delta():
    ds = [CopulaBB1.fit(CopulaBB1(tau_k=0.6, delta=2.0).sample(2000, seed=s), method="tau")
          .copula.delta for s in range(3)]
    assert max(abs(np.array(ds) - 2.0)) < 0.35


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

# Interior MLE fits, n = 500 (the six families, unweighted and weighted):
# Newton decrement ≤ 1.7e-9 nat unweighted (the unweighted one-parameter search
# stops at Brent's default xatol 1e-5), ≤ 7.9e-12 nat weighted (xatol 1e-6 and
# the joint optimiser); every Hessian eigenvalue < 0.
@pytest.mark.parametrize("cls, kw", FAMILIES, ids=IDS)
@pytest.mark.parametrize("weighted", [False, True])
def test_an_interior_mle_has_a_small_gradient_and_a_negative_definite_hessian(cls, kw, weighted):
    data = _sample(cls, kw, n=500, seed=5)
    r = cls.fit(data, method="mle", weights=_weights(500, seed=8) if weighted else None)
    assert "diagnostics" not in r.__dict__                # lazy: nothing computed by the fit
    d = r.diagnostics
    assert isinstance(d, FitDiagnostics) and r.diagnostics is d     # cached
    assert d.names == tuple(r.copula.copula_enum.value.PARAMETERS_SET_NAME)
    assert d.names[0] == "tau_k" and len(d.names) == len(r.copula.params)
    assert d.negative_definite and np.all(d.eigenvalues < 0.0)
    assert d.newton_decrement < (1e-10 if weighted else 1e-8)
    assert not d.at_boundary and d.boundary == ()
    assert d.converged and d.n_iter == r.n_iter and d.n_eval == r.n_eval
    assert "negative definite" in d.summary()


def test_the_itau_gradient_is_reported_as_it_is():
    """The inversion of τ is no maximiser: its gradient in τ is not zero."""
    data = _sample(CopulaClayton, {"tau_k": 0.4}, n=500)
    d = CopulaClayton.fit(data, method="tau").diagnostics
    assert d.max_abs_gradient > 1.0 and d.negative_definite and d.n_iter == 0


@pytest.mark.parametrize("case", ["clayton_independence", "bb1_clayton_limit",
                                  "student_gaussian_limit", "tawn_edge"])
def test_a_boundary_estimate_raises_the_flag(case):
    rng = np.random.default_rng(2)
    if case == "clayton_independence":        # negative dependence: τ̂ at the end ε
        x = CopulaGaussian(tau_k=-0.3).sample(400, seed=1)
        r = CopulaClayton.fit(x, method="mle")
    elif case == "bb1_clayton_limit":           # Clayton data: δ̂ = 1
        x = CopulaClayton(tau_k=0.5).sample(1500, seed=1)
        r = CopulaBB1.fit(x, method="mle", weights=rng.random(1500) + 0.5)
        assert r.copula.delta == 1.0
    elif case == "student_gaussian_limit":      # Gaussian data: ν̂ at the end of its box
        x = CopulaGaussian(tau_k=0.4).sample(60, seed=3)
        r = CopulaStudent.fit(x, method="tau")
    else:                                       # Tawn: data far from the family
        x = CopulaGaussian(tau_k=-0.2).sample(300, seed=4)
        r = CopulaTawn1.fit(x, method="mle")
    d = r.diagnostics
    assert d.at_boundary and d.boundary, (case, r)
    assert "at_boundary" in repr(d)


def test_product_copula_diagnostics_are_empty():
    r = CopulaProduct.fit(_sample(CopulaGaussian, {"tau_k": 0.2}, n=50), method="mle")
    d = r.diagnostics
    assert d.names == () and d.gradient.size == 0 and not d.at_boundary
    assert d.newton_decrement == 0.0 and d.max_abs_gradient == 0.0
