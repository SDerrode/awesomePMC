"""Monte-Carlo size and power of the family-comparison tests (audit FR-6).

Fixed seeds throughout, so every rate below is a deterministic number; the
bounds leave room for the Monte-Carlo error of ``R`` replicates (±3.3 s.d. of
a binomial proportion unless stated otherwise) and the measured values are
quoted in the docstrings of ``pmcprg.diagnostics.model_selection``.

Designs
-------
* **Exact non-nested null**: Gaussian-copula data are radially symmetric, so
  Clayton and survival Clayton are equally close to them in Kullback–Leibler
  divergence (and ``m_n`` is symmetric about 0: the median null of Clarke's
  test holds too) — ``E[m] = 0`` with ``ω² > 0``.
* **Tiny perturbation at the true parameter**: ``m_n ≈ −δ·score_n``, so the
  Vuong statistic is the score statistic, N(0, 1) (a mean null only; the
  median of the score is not 0, so Clarke's test is not run on it).
* **Serial dependence**: a copula Markov chain whose transition copula
  switches between Clayton and survival Clayton with a persistent hidden
  regime — ``E[m] = 0`` by symmetry, strongly autocorrelated ``m_n``.
* **Power**: Clayton data (τ = 0.5), Clayton vs Gumbel.

The fast tests run in about a second; the others are marked ``slow``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

from pmcprg.copulas import CopulaClayton, CopulaGaussian, CopulaGH, CopulaStudent, SurvivalClayton
from pmcprg.diagnostics.model_selection import clarke_test, vuong_test

ALPHA = 0.05


def _rate(flags) -> float:
    return float(np.mean(np.asarray(flags, dtype=float)))


def _mle_tau(cls, uv, weights=None):
    """Maximum-likelihood τ of a one-parameter family on fixed pseudo-observations."""
    w = np.ones(len(uv)) if weights is None else weights

    def nll(t):
        try:
            ll = float(np.dot(w, cls(tau_k=float(t)).logpdf_array(uv)))
        except Exception:
            return 1e12
        return -ll if np.isfinite(ll) else 1e12

    return float(minimize_scalar(nll, bounds=(1e-4, 0.95), method="bounded").x)


def _symmetric_null_logc(n, seed, margins="true"):
    """Clayton vs survival Clayton, both fitted, on Gaussian-copula data."""
    s = CopulaGaussian(tau_k=0.5).sample(n=n, seed=seed)
    if margins == "ranks":
        s = np.column_stack([rankdata(s[:, 0]) / (n + 1), rankdata(s[:, 1]) / (n + 1)])
    ta, tb = _mle_tau(CopulaClayton, s), _mle_tau(SurvivalClayton, s)
    return (CopulaClayton(tau_k=ta).logpdf_array(s),
            SurvivalClayton(tau_k=tb).logpdf_array(s), s)


def _regime_chain(T, R, stay, tau, seed):
    """R chains of length T + 1 with a Clayton / survival-Clayton transition."""
    A, B = CopulaClayton(tau_k=tau), SurvivalClayton(tau_k=tau)
    rng = np.random.default_rng(seed)
    state = rng.integers(0, 2, size=R)
    u = np.empty((T + 1, R))
    u[0] = rng.random(R)
    for t in range(1, T + 1):
        w = rng.random(R)
        nxt = np.empty(R)
        m0 = state == 0
        nxt[m0] = A.inv_h_array(w[m0], u[t - 1][m0])
        nxt[~m0] = B.inv_h_array(w[~m0], u[t - 1][~m0])
        u[t] = nxt
        state = np.where(rng.random(R) > stay, 1 - state, state)
    return u, A, B


# ---------------------------------------------------------------------------
# Fast subset
# ---------------------------------------------------------------------------

def test_power_clayton_vs_gumbel_fast():
    for r in range(20):
        s = CopulaClayton(tau_k=0.5).sample(n=1000, seed=50000 + r)
        a = CopulaClayton.fit(s, method="mle")
        b = CopulaGH.fit(s, method="mle")
        la, lb = a.copula.logpdf_array(a.uv), b.copula.logpdf_array(b.uv)
        assert vuong_test(la, lb, names=("Clayton", "GH")).preferred == "Clayton"
        assert clarke_test(la, lb, names=("Clayton", "GH")).preferred == "Clayton"
        assert vuong_test(lb, la, names=("GH", "Clayton")).preferred == "Clayton"


def test_size_under_a_tiny_perturbation_at_the_true_parameter_fast():
    R = 400
    rej_auto, rej_iid = [], []
    for r in range(R):
        s = CopulaClayton(tau_k=0.5).sample(n=300, seed=30000 + r)
        la = CopulaClayton(tau_k=0.5).logpdf_array(s)
        lb = CopulaClayton(tau_k=0.5 * (1 + 1e-4)).logpdf_array(s)
        rej_auto.append(not vuong_test(la, lb).tie)
        rej_iid.append(not vuong_test(la, lb, bandwidth=0).tie)
    for rate in (_rate(rej_auto), _rate(rej_iid)):
        assert 0.014 <= rate <= 0.086, rate


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vuong_size_on_the_exact_null_with_true_margins():
    R = 1000
    rej = {"auto": [], "iid": []}
    for r in range(R):
        la, lb, _ = _symmetric_null_logc(500, 70000 + r)
        rej["auto"].append(not vuong_test(la, lb).tie)
        rej["iid"].append(not vuong_test(la, lb, bandwidth=0).tie)
    for key, flags in rej.items():
        assert 0.027 <= _rate(flags) <= 0.073, (key, _rate(flags))


@pytest.mark.slow
def test_vuong_size_on_rank_pseudo_observations_is_inflated_as_documented():
    """Chen & Fan (2006): estimating the margins changes the variance of the
    pseudo-likelihood ratio; the naive Vuong variance ignores it. Measured
    here, not corrected — the module docstring quotes this number."""
    R = 400
    flags, zs = [], []
    for r in range(R):
        la, lb, _ = _symmetric_null_logc(2000, 70000 + r, margins="ranks")
        res = vuong_test(la, lb, bandwidth=0)
        flags.append(not res.tie)
        zs.append(res.statistic)
    assert 0.06 <= _rate(flags) <= 0.20, _rate(flags)
    assert 1.05 <= float(np.std(zs)) <= 1.30, float(np.std(zs))


@pytest.mark.slow
def test_clarke_size_at_the_true_parameters_and_its_estimation_effect():
    R = 1000
    at_truth, fitted = [], []
    for r in range(R):
        la, lb, s = _symmetric_null_logc(500, 70000 + r)
        l0a = CopulaClayton(tau_k=0.4).logpdf_array(s)
        l0b = SurvivalClayton(tau_k=0.4).logpdf_array(s)
        at_truth.append(not clarke_test(l0a, l0b, bandwidth=0).tie)
        fitted.append(not clarke_test(la, lb, bandwidth=0).tie)
    assert 0.027 <= _rate(at_truth) <= 0.073, _rate(at_truth)
    # Documented limitation: the sign test ignores the estimation of θ̂.
    assert _rate(fitted) >= 0.40, _rate(fitted)


@pytest.mark.slow
def test_hac_restores_the_level_under_serial_dependence():
    T, R = 2000, 1000
    u, A, B = _regime_chain(T, R, stay=0.9, tau=0.5, seed=123)
    rej = {k: [] for k in ("V iid", "V auto", "C iid", "C auto")}
    for r in range(R):
        uv = np.column_stack([u[:-1, r], u[1:, r]])
        la, lb = A.logpdf_array(uv), B.logpdf_array(uv)
        rej["V iid"].append(not vuong_test(la, lb, bandwidth=0).tie)
        rej["V auto"].append(not vuong_test(la, lb).tie)
        rej["C iid"].append(not clarke_test(la, lb, bandwidth=0).tie)
        rej["C auto"].append(not clarke_test(la, lb).tie)
    rates = {k: _rate(v) for k, v in rej.items()}
    assert rates["V iid"] >= 0.15 and rates["C iid"] >= 0.10, rates
    # The Bartlett HAC with the Newey–West bandwidth is slightly liberal here
    # (measured 6.7 % and 5.5 %): bound at 10 %.
    assert 0.02 <= rates["V auto"] <= 0.10 and 0.02 <= rates["C auto"] <= 0.10, rates


@pytest.mark.slow
def test_sum_of_weights_is_conservative_and_kish_is_exact_for_independent_weights():
    """Weights independent of the data: Var(Σ w m) = σ² Σw², so Kish's n_eff is
    exact and the package's Σw (frequency) convention under-rejects."""
    R, n = 1000, 1000
    rej = {"sum": [], "kish": []}
    for r in range(R):
        s = CopulaGaussian(tau_k=0.5).sample(n=n, seed=80000 + r)
        w = np.random.default_rng(90000 + r).random(n)
        la = CopulaClayton(tau_k=0.4).logpdf_array(s)
        lb = SurvivalClayton(tau_k=0.4).logpdf_array(s)
        rej["sum"].append(not vuong_test(la, lb, w, bandwidth=0).tie)
        rej["kish"].append(not vuong_test(la, lb, w, bandwidth=0, n_eff="kish").tie)
    assert 0.027 <= _rate(rej["kish"]) <= 0.073, _rate(rej["kish"])
    assert _rate(rej["sum"]) < 0.035, _rate(rej["sum"])


@pytest.mark.slow
def test_gaussian_vs_student_at_the_df_bound():
    """Overlapping families: uncorrected, a tie; corrected, parsimony wins.

    ν̂ sits at its upper bound 100 in about 60 % of the replicates. Measured on
    200 replicates: 5.0 % rejections uncorrected, 63.5 % with ``"akaike"``
    (86 % with ``"schwarz"``); 100 are run here to keep the test short.
    """
    R = 100
    rej = {"none": [], "akaike": []}
    for r in range(R):
        s = CopulaGaussian(tau_k=0.5).sample(n=1000, seed=40000 + r)
        a = CopulaGaussian.fit(s, method="mle")
        b = CopulaStudent.fit(s, method="mle")
        la, lb = a.copula.logpdf_array(a.uv), b.copula.logpdf_array(b.uv)
        for corr in rej:
            res = vuong_test(la, lb, k_a=1, k_b=2, correction=corr)
            rej[corr].append(not res.tie)
            if corr == "akaike" and not res.tie:
                assert res.decision == "A"                  # the Gaussian
    assert _rate(rej["none"]) <= 0.10, _rate(rej["none"])
    assert _rate(rej["akaike"]) >= 0.30, _rate(rej["akaike"])


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_power_clayton_vs_gumbel():
    R = 200
    power = {"vuong": [], "clarke": [], "vuong n=100": []}
    for r in range(R):
        for n, key in ((1000, None), (100, "vuong n=100")):
            s = CopulaClayton(tau_k=0.5).sample(n=n, seed=50000 + r)
            a = CopulaClayton.fit(s, method="mle")
            b = CopulaGH.fit(s, method="mle")
            la, lb = a.copula.logpdf_array(a.uv), b.copula.logpdf_array(b.uv)
            if key is None:
                power["vuong"].append(vuong_test(la, lb).decision == "A")
                power["clarke"].append(clarke_test(la, lb).decision == "A")
            else:
                power[key].append(vuong_test(la, lb).decision == "A")
    assert _rate(power["vuong"]) == 1.0 and _rate(power["clarke"]) == 1.0
    assert 0.5 <= _rate(power["vuong n=100"]) < 1.0, _rate(power["vuong n=100"])
