"""Unit tests of ``pmcprg.diagnostics.model_selection`` (audit FR-6).

Is a copula family choice significant? Weighted Vuong (1989) and Clarke (2007)
tests with a Newey–West variance, the matrix over a candidate set, the
confidence set of families not significantly worse than the best, and the two
opt-in hooks (``FitBestResults.compare`` / ``.confidence_set`` and
``ice_pair_comparisons``). The Monte-Carlo size and power checks live in
``test_model_selection_mc.py``.
"""
from __future__ import annotations

import copy
import importlib
import math

import numpy as np
import pytest
from scipy import stats

from pmcprg.copulas import (
    CopulaClayton, CopulaFrank, CopulaGaussian, CopulaGH, CopulaStudent, CopulaVirt,
)
from pmcprg.diagnostics.model_selection import (
    ComparisonResult,
    clarke_test,
    comparison_matrix,
    confidence_set,
    fit_best_comparison,
    fit_best_confidence_set,
    hac_variance,
    ice_pair_comparisons,
    newey_west_bandwidth,
    vuong_test,
)
from pmcprg.pmc.inference import backward, forward, joint_posteriors, precompute_weights, smooth
from pmcprg.pmc.model import PMCModel
from pmcprg.pmc.simulate import simulate

ice_mod = importlib.import_module("pmcprg.pmc.ice")


def _logc_pair(n=400, seed=0, tau=0.5):
    """Log-densities of two fitted families on Clayton data (unit weights)."""
    s = CopulaClayton(tau_k=tau).sample(n=n, seed=seed)
    a = CopulaClayton.fit(s, method="mle")
    b = CopulaGH.fit(s, method="mle")
    return a.copula.logpdf_array(a.uv), b.copula.logpdf_array(b.uv)


def _close_pair(n=300, seed=1):
    """Two log-density arrays whose difference is small and noisy."""
    rng = np.random.default_rng(seed)
    la = rng.normal(0.3, 1.0, n)
    lb = la + rng.normal(0.02, 0.3, n)
    return la, lb


# ---------------------------------------------------------------------------
# HAC variance and bandwidth
# ---------------------------------------------------------------------------

def test_hac_variance_bandwidth_zero_is_the_sum_of_squares():
    z = np.random.default_rng(0).normal(size=137)
    assert hac_variance(z, 0) == pytest.approx(float(np.dot(z, z)), rel=1e-14)


@pytest.mark.parametrize("L", [1, 3, 10])
def test_hac_variance_matches_the_double_sum(L):
    z = np.random.default_rng(L).normal(size=60)
    idx = np.arange(z.size)
    lag = np.abs(idx[:, None] - idx[None, :])
    k = np.where(lag <= L, 1.0 - lag / (L + 1.0), 0.0)
    assert hac_variance(z, L) == pytest.approx(float(z @ k @ z), rel=1e-12)


def test_hac_variance_is_never_negative():
    # Alternating signs make every odd-lag autocovariance strongly negative.
    z = np.tile([1.0, -1.0], 50)
    for L in range(0, 30):
        assert hac_variance(z, L) >= 0.0


def test_newey_west_bandwidth_follows_the_autocorrelation():
    rng = np.random.default_rng(3)
    T = 2000
    iid = rng.normal(size=T)
    ar = np.empty(T)
    ar[0] = rng.normal()
    for t in range(1, T):
        ar[t] = 0.9 * ar[t - 1] + rng.normal()
    L_iid = newey_west_bandwidth(iid - iid.mean())
    L_ar = newey_west_bandwidth(ar - ar.mean())
    assert 0 <= L_iid < L_ar <= T - 1
    assert newey_west_bandwidth(np.zeros(50)) == 0
    assert newey_west_bandwidth(np.ones(2)) == 0


def test_auto_bandwidth_is_at_least_one():
    la, lb = _close_pair()
    assert vuong_test(la, lb).bandwidth >= 1
    assert vuong_test(la, lb, bandwidth=0).bandwidth == 0
    assert vuong_test(la, lb, bandwidth=7).bandwidth == 7


# ---------------------------------------------------------------------------
# Vuong
# ---------------------------------------------------------------------------

def test_vuong_unit_weights_iid_is_the_textbook_statistic():
    la, lb = _logc_pair()
    m = la - lb
    n = m.size
    omega = math.sqrt(np.mean(m ** 2) - np.mean(m) ** 2)
    z = m.sum() / (math.sqrt(n) * omega)
    r = vuong_test(la, lb, bandwidth=0)
    assert r.statistic == pytest.approx(z, rel=1e-10)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(abs(z)), rel=1e-8)
    assert r.method == "normal" and r.n_eff == n and r.sum_weights == n


@pytest.mark.parametrize("n_eff", ["sum", "kish"])
def test_vuong_hac_reduces_to_the_weighted_iid_variance_with_bandwidth_zero(n_eff):
    la, lb = _close_pair()
    w = np.random.default_rng(5).random(la.size)
    m = la - lb
    mean = np.dot(w, m) / w.sum()
    d = m - mean
    if n_eff == "sum":
        var = np.dot(w, d ** 2) / w.sum()
        ne = w.sum()
    else:
        var = np.dot(w ** 2, d ** 2) / np.dot(w, w)
        ne = w.sum() ** 2 / np.dot(w, w)
    r = vuong_test(la, lb, w, bandwidth=0, n_eff=n_eff)
    assert r.sd == pytest.approx(math.sqrt(var), rel=1e-12)
    assert r.n_eff == pytest.approx(ne, rel=1e-12)
    assert r.statistic == pytest.approx(math.sqrt(ne) * mean / math.sqrt(var), rel=1e-10)


@pytest.mark.parametrize("correction", ["none", "akaike", "schwarz"])
def test_integer_weights_equal_repeated_data(correction):
    la, lb = _close_pair(n=120, seed=7)
    reps = np.random.default_rng(8).integers(0, 4, size=la.size)
    la_rep, lb_rep = np.repeat(la, reps), np.repeat(lb, reps)
    kw = dict(k_a=2, k_b=1, correction=correction, bandwidth=0)
    for fn, extra in ((vuong_test, {}), (clarke_test, {"method": "exact"}),
                      (clarke_test, {"method": "normal"})):
        rw = fn(la, lb, reps.astype(float), **kw, **extra)
        rr = fn(la_rep, lb_rep, **kw, **extra)
        assert rw.statistic == pytest.approx(rr.statistic, rel=1e-10, abs=1e-12)
        assert rw.p_value == pytest.approx(rr.p_value, rel=1e-10, abs=1e-15)
        assert rw.decision == rr.decision
        assert rw.n_eff == pytest.approx(rr.n_eff, rel=1e-12)


@pytest.mark.parametrize("test_fn", [vuong_test, clarke_test])
@pytest.mark.parametrize("correction", ["none", "akaike", "schwarz"])
@pytest.mark.parametrize("bandwidth", [0, "auto"])
@pytest.mark.parametrize("weighted", [False, True])
def test_a_vs_b_is_minus_b_vs_a(test_fn, correction, bandwidth, weighted):
    la, lb = _close_pair(n=250, seed=11)
    w = np.random.default_rng(12).random(la.size) if weighted else None
    kw = dict(correction=correction, bandwidth=bandwidth)
    ab = test_fn(la, lb, w, k_a=2, k_b=1, names=("A", "B"), **kw)
    ba = test_fn(lb, la, w, k_a=1, k_b=2, names=("B", "A"), **kw)
    assert ab.statistic == pytest.approx(-ba.statistic, rel=1e-12, abs=1e-12)
    assert ab.p_value == pytest.approx(ba.p_value, rel=1e-10, abs=1e-15)
    assert ab.bandwidth == ba.bandwidth
    assert ab.preferred == ba.preferred
    sw = ab.swapped()
    assert sw.statistic == -ab.statistic and sw.names == ("B", "A")
    assert sw.preferred == ab.preferred


def test_kish_is_invariant_to_the_weight_scale_and_sum_is_a_frequency():
    la, lb = _close_pair(n=400, seed=13)
    w = np.random.default_rng(14).random(la.size)
    k1 = vuong_test(la, lb, w, n_eff="kish", bandwidth=3)
    k2 = vuong_test(la, lb, 10.0 * w, n_eff="kish", bandwidth=3)
    assert k1.statistic == pytest.approx(k2.statistic, rel=1e-10)
    s1 = vuong_test(la, lb, w, bandwidth=3)
    s4 = vuong_test(la, lb, 4.0 * w, bandwidth=3)
    assert s4.statistic == pytest.approx(2.0 * s1.statistic, rel=1e-10)
    # Σw ≤ Kish for weights in [0, 1].
    assert s1.n_eff < k1.n_eff


def test_penalties():
    la, lb = _close_pair(n=200, seed=15)
    w = np.full(la.size, 0.5)
    none = vuong_test(la, lb, w, k_a=2, k_b=1)
    aic = vuong_test(la, lb, w, k_a=2, k_b=1, correction="akaike")
    bic = vuong_test(la, lb, w, k_a=2, k_b=1, correction="schwarz")
    assert none.penalty == 0.0
    assert aic.penalty == 1.0
    assert bic.penalty == pytest.approx(0.5 * math.log(100.0))
    # Same variance, numerator shifted by −K/√n_eff/σ.
    assert aic.statistic == pytest.approx(
        none.statistic - 1.0 / (math.sqrt(100.0) * none.sd), rel=1e-10)
    # Clarke spreads K over the observations before taking signs.
    c = clarke_test(la, lb, w, k_a=2, k_b=1, correction="akaike", bandwidth=0, method="normal")
    mc = (la - lb) - 1.0 / 100.0
    assert c.count == pytest.approx(0.5 * np.sum(mc > 0))


def test_sign_of_the_akaike_statistic_agrees_with_the_aic_ranking():
    s = CopulaClayton(tau_k=0.4).sample(n=300, seed=21)
    fb = CopulaVirt.fit_best(s, families=[CopulaClayton, CopulaGaussian, CopulaFrank,
                                          CopulaGH, CopulaStudent], method="mle")
    mat = fb.compare(bandwidth=0)
    aic = {r.copula.copula_enum.value.SHORT_NAME: r.aic for r in fb}
    for (a, b), r in mat.results.items():
        assert np.sign(r.statistic) == np.sign(aic[b] - aic[a])


def test_decision_follows_the_p_value():
    la, lb = _logc_pair(n=200, seed=2)
    r = vuong_test(la, lb)
    assert r.decision == "A" and r.p_value <= 0.05 and r.preferred == "A" and not r.tie
    r = vuong_test(la, lb, alpha=1e-300)
    assert r.decision == "tie" and r.preferred is None and r.tie


def test_zero_weight_points_are_ignored_whatever_their_log_density():
    la, lb = _close_pair(n=150, seed=17)
    w = np.random.default_rng(18).random(la.size)
    w[[3, 40, 99]] = 0.0
    ref_v = vuong_test(la, lb, w)
    ref_c = clarke_test(la, lb, w)
    la2, lb2 = la.copy(), lb.copy()
    la2[3], lb2[40], la2[99], lb2[99] = -np.inf, np.nan, np.inf, -np.inf
    for ref, new in ((ref_v, vuong_test(la2, lb2, w)), (ref_c, clarke_test(la2, lb2, w))):
        assert new.statistic == ref.statistic and new.p_value == ref.p_value


@pytest.mark.parametrize("test_fn", [vuong_test, clarke_test])
def test_non_finite_log_densities_at_positive_weight(test_fn):
    la, lb = _close_pair(n=50, seed=19)
    a = la.copy()
    a[5] = -np.inf
    r = test_fn(a, lb)
    assert r.decision == "B" and r.statistic == -np.inf and r.p_value == 0.0
    assert r.method == "infinite"
    r = test_fn(la, a)
    assert r.decision == "A" and r.statistic == np.inf
    b = lb.copy()
    b[7] = -np.inf
    assert test_fn(a, b).decision == "undetermined"        # both likelihoods −∞
    c = la.copy()
    c[9] = np.nan
    r = test_fn(c, lb)
    assert r.decision == "undetermined" and math.isnan(r.p_value)


def test_identical_families_are_a_degenerate_tie():
    la, _ = _close_pair()
    v = vuong_test(la, la.copy())
    assert v.decision == "tie" and v.method == "degenerate" and v.p_value == 1.0
    c = clarke_test(la, la.copy())
    assert c.decision == "tie" and c.method == "degenerate" and c.n_trials == 0.0
    # Same densities, different parameter counts: still no evidence in the data.
    assert vuong_test(la, la.copy(), k_a=1, k_b=2, correction="akaike").decision == "tie"


def test_clarke_exact_is_the_binomial_test():
    la, lb = _close_pair(n=101, seed=23)
    r = clarke_test(la, lb, bandwidth=0)
    m = la - lb
    B, N = int(np.sum(m > 0)), int(np.sum(m != 0))
    assert r.method == "exact" and r.count == B and r.n_trials == N
    assert r.p_value == pytest.approx(stats.binomtest(B, N, 0.5).pvalue, rel=1e-12)
    assert r.statistic == pytest.approx((B - N / 2) / math.sqrt(N / 4))
    # The normal path at bandwidth 0 is the normal approximation of the same binomial.
    rn = clarke_test(la, lb, bandwidth=0, method="normal")
    assert rn.statistic == pytest.approx(r.statistic, rel=1e-12)
    assert rn.sd == pytest.approx(0.5)


def test_clarke_exact_needs_frequency_weights_and_no_hac():
    la, lb = _close_pair(n=40)
    with pytest.raises(ValueError, match="integer weights"):
        clarke_test(la, lb, np.full(40, 0.5), bandwidth=0, method="exact")
    with pytest.raises(ValueError, match="bandwidth=0"):
        clarke_test(la, lb, method="exact")
    assert clarke_test(la, lb).method == "normal"                 # auto bandwidth
    assert clarke_test(la, lb, np.full(40, 0.5), bandwidth=0).method == "normal"


@pytest.mark.parametrize("kwargs, match", [
    ({"correction": "aic"}, "correction"),
    ({"n_eff": "n"}, "n_eff"),
    ({"alpha": 0.0}, "alpha"),
    ({"bandwidth": -1}, "bandwidth"),
    ({"bandwidth": "nw"}, "bandwidth"),
    ({"bandwidth": 1.5}, "bandwidth"),
])
def test_invalid_options_raise(kwargs, match):
    la, lb = _close_pair(n=30)
    with pytest.raises(ValueError, match=match):
        vuong_test(la, lb, **kwargs)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError, match="same length"):
        vuong_test(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="non-negative"):
        vuong_test(np.zeros(3), np.ones(3), [1.0, -1.0, 1.0])
    with pytest.raises(ValueError, match="all weights are zero"):
        vuong_test(np.zeros(3), np.ones(3), np.zeros(3))


# ---------------------------------------------------------------------------
# Candidate sets
# ---------------------------------------------------------------------------

def _candidate_logc(n=600, seed=31, tau=0.5):
    s = CopulaClayton(tau_k=tau).sample(n=n, seed=seed)
    fits = {cls.__name__: cls.fit(s, method="mle")
            for cls in (CopulaClayton, CopulaGaussian, CopulaFrank, CopulaGH)}
    return ({k: r.copula.logpdf_array(r.uv) for k, r in fits.items()},
            {k: r.n_params for k, r in fits.items()})


@pytest.mark.parametrize("test", ["vuong", "clarke"])
def test_comparison_matrix_is_antisymmetric_and_consistent(test):
    logc, k = _candidate_logc()
    w = np.random.default_rng(32).random(600)
    mat = comparison_matrix(logc, w, n_params=k, test=test, correction="akaike")
    assert mat.names == list(logc)
    np.testing.assert_array_equal(mat.statistic, -mat.statistic.T)
    np.testing.assert_array_equal(mat.p_value, mat.p_value.T)
    assert np.all(np.diag(mat.statistic) == 0) and np.all(np.diag(mat.p_value) == 1)
    for a, na in enumerate(mat.names):
        for b, nb in enumerate(mat.names):
            if a == b:
                continue
            r = mat.result(na, nb)
            assert r.names == (na, nb)
            direct = (vuong_test if test == "vuong" else clarke_test)(
                logc[na], logc[nb], w, k_a=k[na], k_b=k[nb], correction="akaike")
            assert r.statistic == pytest.approx(direct.statistic, rel=1e-10, abs=1e-12)
            label = r.preferred or r.decision
            assert mat.decision[a][b] == label
    assert "significant" in str(mat)


def test_confidence_set_on_clearly_separated_families():
    logc, k = _candidate_logc(n=1000)
    cs = confidence_set(logc, n_params=k, correction="akaike")
    assert cs.best == "CopulaClayton"
    assert cs.members == ["CopulaClayton"]
    assert sorted(cs.excluded) == sorted(set(logc) - {"CopulaClayton"})
    assert "CopulaClayton" in cs and len(cs) == 1 and not cs.is_tie


def test_confidence_set_keeps_ties_and_holm_keeps_at_least_as_many():
    rng = np.random.default_rng(41)
    base = rng.normal(0.5, 1.0, 500)
    pm = np.tile([1.0, -1.0], 250)                  # mean 0, standard deviation 1
    # best − near has mean c and s.d. 0.05 exactly: Z = √500·c/0.05 = 2.19,
    # p = 0.028 — significant alone, not after multiplying by 3 comparisons.
    c = 2.19 * 0.05 / math.sqrt(500)
    logc = {
        "best": base,
        "twin": base - 1e-4 + 0.05 * pm,                        # Z ≈ 0.04: tied
        "near": base - c + 0.05 * pm,                           # borderline
        "bad": base - 0.5 + rng.normal(0.0, 0.3, 500),          # clearly worse
    }
    plain = confidence_set(logc, bandwidth=0)
    holm = confidence_set(logc, bandwidth=0, adjust="holm")
    bonf = confidence_set(logc, bandwidth=0, adjust="bonferroni")
    assert plain.best == "best" and "twin" in plain and "bad" not in plain
    # Bonferroni p-values ≥ Holm p-values ≥ raw ones: nested sets.
    assert set(plain.members) <= set(holm.members) <= set(bonf.members)
    assert "near" in bonf and "near" not in plain                 # the borderline case
    for cs in (plain, holm, bonf):
        assert cs.members[0] == "best"
        assert set(cs.members) | set(cs.excluded) == set(logc)
        assert not set(cs.members) & set(cs.excluded)
    # Holm / Bonferroni adjusted p-values
    raw = {n: r.p_value for n, r in plain.comparisons.items()}
    assert bonf.p_adjusted == {n: min(1.0, 3 * p) for n, p in raw.items()}
    order = sorted(raw, key=raw.get)
    running = 0.0
    for rank, n in enumerate(order):
        running = max(running, min(1.0, (3 - rank) * raw[n]))
        assert holm.p_adjusted[n] == pytest.approx(running)


def test_confidence_set_reference_and_scores():
    logc, k = _candidate_logc()
    cs = confidence_set(logc, n_params=k, best="CopulaFrank")
    assert cs.best == "CopulaFrank" and cs.members[0] == "CopulaFrank"
    # A family better than the reference is never excluded.
    assert "CopulaClayton" in cs.members
    scores = {n: float(i) for i, n in enumerate(logc)}          # last one is "best"
    cs = confidence_set(logc, n_params=k, scores=scores)
    assert cs.best == list(logc)[-1]
    with pytest.raises(ValueError, match="not among"):
        confidence_set(logc, best="nope")
    with pytest.raises(ValueError, match="adjust"):
        confidence_set(logc, adjust="bh")


# ---------------------------------------------------------------------------
# fit_best hook
# ---------------------------------------------------------------------------

def _snapshot_fit_best(fb):
    return ([(type(r.copula).__name__, r.tau_k, r.log_likelihood, r.aic, r.converged,
              dict(r.copula.params)) for r in fb],
            [(type(r.copula).__name__, r.log_likelihood) for r in fb.other_method],
            list(fb.failures), fb.method)


@pytest.mark.parametrize("method", ["tau", "mle"])
def test_fit_best_hooks_are_read_only(method):
    s = CopulaGaussian(tau_k=0.3).sample(n=250, seed=51)
    fams = [CopulaClayton, CopulaGaussian, CopulaFrank, CopulaGH, CopulaStudent]
    fb = CopulaVirt.fit_best(s, families=fams, method=method)
    before = _snapshot_fit_best(fb)
    mat = fb.compare()
    cs = fb.confidence_set(adjust="holm")
    clarke = fb.compare("clarke", bandwidth=0)
    assert _snapshot_fit_best(fb) == before
    assert _snapshot_fit_best(CopulaVirt.fit_best(s, families=fams, method=method)) == before
    names = [r.copula.copula_enum.value.SHORT_NAME for r in fb]
    assert mat.names == names and clarke.names == names
    assert cs.best == names[0]                        # the AIC winner
    assert mat.correction == "akaike"
    ref = fit_best_comparison(fb)
    np.testing.assert_array_equal(ref.statistic, mat.statistic)
    ref_cs = fit_best_confidence_set(fb, adjust="holm")
    assert ref_cs.members == cs.members
    if method == "tau":                               # Student is in other_method
        assert "Student" not in names


def test_fit_best_hook_refuses_fits_on_different_data():
    from pmcprg.copulas._fit import FitBestResults
    a = CopulaClayton.fit(CopulaClayton(tau_k=0.3).sample(n=100, seed=1), method="mle")
    b = CopulaGH.fit(CopulaClayton(tau_k=0.3).sample(n=100, seed=2), method="mle")
    with pytest.raises(ValueError, match="same pseudo-observations"):
        FitBestResults([a, b]).compare()


# ---------------------------------------------------------------------------
# ICE hook
# ---------------------------------------------------------------------------

def _state_raw(names=("Clayton", "GH", "Gauss", "Clayton"), taus=(0.6, 0.4, 0.3, 0.6)):
    return {
        "model": {"name": "fr6", "variant": "PMC", "K": 2, "N_default": 600},
        "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 1, "dist": "norm", "params": {"loc": 2.0, "scale": 1.0}}],
        "copulas": [{"i": i, "j": j, "name": names[2 * i + j], "tau": taus[2 * i + j]}
                    for i in range(2) for j in range(2)],
    }


def _pair_raw():
    raw = _state_raw()
    raw["margins"] = [
        {"i": i, "j": j, "dist": "norm", "params": {"loc": mu, "scale": sd}}
        for (i, j), (mu, sd) in {(0, 0): (0.0, 1.0), (0, 1): (0.3, 1.6),
                                 (1, 0): (1.1, 1.4), (1, 1): (2.0, 1.0)}.items()
    ]
    return raw


def _posteriors(model, Y):
    W, f = precompute_weights(model, Y)
    a, _ = forward(model, Y, W=W, f_pdf=f)
    b = backward(model, Y, W=W)
    return joint_posteriors(a, W, b), smooth(a, b)


CANDIDATES = ["Gauss", "Clayton", "GH", "Frank"]


@pytest.fixture(scope="module")
def fitted_state():
    truth = PMCModel.from_dict(_state_raw())
    _, Y = simulate(truth, N=600, seed=4)
    fitted, _ = ice_mod.ice(truth, Y, ice_cfg={"max_iter": 3, "candidates": CANDIDATES,
                                               "selection_criterion": "aic"})
    return fitted, Y


@pytest.mark.parametrize("criterion", ["mle", "aic", "bic", "cvm", "huard_common"])
@pytest.mark.parametrize("structure", ["state", "pair"])
def test_ice_pair_comparisons_select_what_the_m_step_selects(criterion, structure, fitted_state):
    if structure == "state":
        model, Y = fitted_state
    else:
        model = PMCModel.from_dict(_pair_raw())
        _, Y = simulate(model, N=500, seed=6)
    xi, gamma = _posteriors(model, Y)
    raw_before = copy.deepcopy(model.raw)
    res = ice_pair_comparisons(model, Y, xi, candidates=CANDIDATES, criterion=criterion)
    assert model.raw == raw_before                               # read-only

    # The M-step of ICE on the same (model, Y, ξ).
    raw = model.raw
    ice_mod._m_step(raw, model, Y, xi, gamma, fit_margins=False, candidates=CANDIDATES,
                    selection_criterion=criterion, margin_selection_rule="mle")
    blocks = {(int(b["i"]), int(b["j"])): b for b in raw["copulas"]}
    assert set(res) == set(blocks)
    for key, pc in res.items():
        assert pc.selected == blocks[key]["name"], key
        assert pc.params[pc.selected]["tau_k"] == blocks[key]["tau"]
        assert pc.criterion == criterion
        assert pc.runner_up in CANDIDATES and pc.runner_up != pc.selected
        assert pc.scores[pc.selected] == max(pc.scores.values())
        assert pc.sum_weights == pytest.approx(float(xi[:, key[0], key[1]].sum()))
        assert pc.confidence_set.best == pc.selected
        assert pc.vs_runner_up.names == (pc.selected, pc.runner_up)
        assert pc.tied_with_runner_up == (pc.vs_runner_up.decision != "A")
        expected = {"mle": "none", "aic": "akaike", "bic": "schwarz"}.get(criterion, "none")
        assert pc.vs_runner_up.correction == expected


def test_ice_pair_comparisons_runs_the_e_step_when_xi_is_omitted(fitted_state):
    model, Y = fitted_state
    xi, _ = _posteriors(model, Y)
    a = ice_pair_comparisons(model, Y, None, candidates=CANDIDATES, criterion="bic",
                             test="clarke", pairs=[(0, 1)])
    b = ice_pair_comparisons(model, Y, xi, candidates=CANDIDATES, criterion="bic",
                             test="clarke", pairs=[(0, 1)])
    assert list(a) == [(0, 1)]
    np.testing.assert_array_equal(a[(0, 1)].matrix.statistic, b[(0, 1)].matrix.statistic)
    assert a[(0, 1)].matrix.test == "clarke"
    # The stored log-densities and weights reproduce the stored tests.
    pc = a[(0, 1)]
    np.testing.assert_array_equal(pc.weights, xi[:, 0, 1])
    s, r = pc.selected, pc.runner_up
    again = clarke_test(pc.log_densities[s], pc.log_densities[r], pc.weights,
                        k_a=pc.n_params[s], k_b=pc.n_params[r], correction="schwarz")
    assert again.statistic == pc.vs_runner_up.statistic


def test_ice_pair_comparisons_defaults_come_from_the_model_config(fitted_state):
    model, Y = fitted_state
    cfg = ice_mod._parse_ice_cfg(model, None)
    res = ice_pair_comparisons(model, Y, pairs=[(0, 0)])
    pc = res[(0, 0)]
    assert set(pc.scores) == {c for c in cfg["candidates"]
                              if ice_mod.CopulaEnum.from_short_name(c) is not None}
    assert pc.criterion == cfg["selection_criterion"]


def test_ice_pair_comparisons_refuses_a_model_without_copula():
    model = PMCModel("pmcprg/pmc/models/hmc_in_gauss_k2.toml")
    _, Y = simulate(model, N=50, seed=0)
    with pytest.raises(ValueError, match="no copula"):
        ice_pair_comparisons(model, Y)


def test_ice_pair_comparisons_checks_the_shape_of_xi(fitted_state):
    model, Y = fitted_state
    with pytest.raises(ValueError, match="xi must have shape"):
        ice_pair_comparisons(model, Y, np.zeros((3, 2, 2)))


def test_result_types_are_exported():
    import pmcprg.diagnostics as d
    for name in ("vuong_test", "clarke_test", "comparison_matrix", "confidence_set",
                 "ice_pair_comparisons", "ComparisonResult", "ConfidenceSet"):
        assert hasattr(d, name)
    assert d.ComparisonResult is ComparisonResult
