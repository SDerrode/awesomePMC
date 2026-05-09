"""
test_pmc.py — Functional tests for the prg.pmc package.

Coverage
--------
* PMCModel loading & validation for the 5 variants
* weight() formula on a manually-checked example
* save/load round-trip on a temp file
* simulate(): empirical state frequencies match the stationary π
* classify(): low error rate on a well-separated HMC-IN problem
* error_rate(): handles K=2 label permutation
* ICE: convergence on a synthetic PMC sequence + τ recovery
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from prg.pmc.model     import PMCModel, Variant
from prg.pmc.simulate  import simulate
from prg.pmc.inference import (
    classify, forward, backward, smooth, error_rate, precompute_weights,
)
from prg.pmc.ice       import ice


MODELS_DIR = Path(__file__).resolve().parents[1] / "pmc" / "models"
ALL_MODELS = sorted(MODELS_DIR.glob("*.toml"))


# ===========================================================================
# 1. Model loading & validation
# ===========================================================================

@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_model_loads(toml_path: Path):
    """Every shipped TOML loads without error and exposes required attributes."""
    mdl = PMCModel(toml_path)
    assert isinstance(mdl.variant, Variant)
    assert mdl.K >= 2
    assert mdl.N_default >= 1
    assert mdl.name


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_model_prior_consistent(toml_path: Path):
    """π, A, p must be mutually consistent: p = π[:, None] * A and π = p.sum(axis=1)."""
    mdl = PMCModel(toml_path)

    pi = mdl.stationary_pi
    A  = mdl.transition_A
    p  = mdl.prior_p

    assert pi.shape == (mdl.K,)
    assert A.shape  == (mdl.K, mdl.K)
    assert p.shape  == (mdl.K, mdl.K)

    # π is a probability vector
    assert np.isclose(pi.sum(), 1.0, atol=1e-8)
    assert (pi >= 0).all()

    # A is row-stochastic
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-8)

    # p sums to 1
    assert np.isclose(p.sum(), 1.0, atol=1e-8)

    # Joint = marginal × conditional
    np.testing.assert_allclose(p, pi[:, None] * A, atol=1e-8)


@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_model_weight_positive(toml_path: Path):
    """``weight()`` returns positive finite values for in-support observations.

    Earlier versions of this test drew ``(y0, y1)`` from ``N(0, 1)`` for every
    model, but that breaks for fixtures whose marginal distributions live far
    from zero (e.g. ``betaprime(loc=2)`` in the SP-2016 GICE fixture, whose
    PDF/CDF return 0 / log(0) for any y < 2). Instead, we sample
    representative observations from the model itself via ``simulate``.
    """
    from prg.pmc.simulate import simulate
    mdl = PMCModel(toml_path)
    if mdl.d > 1:
        pytest.skip(
            "weight() takes scalar y_n / y_{n+1} arguments; multivariate "
            "models exercise the equivalent path through precompute_weights "
            "which is covered by test_multivariate.py."
        )
    _X, Y = simulate(mdl, N=50, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        i, j = rng.integers(0, mdl.K, size=2)
        y0, y1 = rng.choice(Y, size=2, replace=False)
        w = mdl.weight(int(i), int(j), float(y0), float(y1))
        assert w >= 0
        assert math.isfinite(w)


def test_invalid_variant_raises(tmp_path: Path):
    """Bogus variant name → ValueError."""
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[model]\nvariant = "NOPE"\nK = 2\n'
        '[prior]\nA = [[0.5,0.5],[0.5,0.5]]\n'
    )
    with pytest.raises(ValueError, match="Unknown variant"):
        PMCModel(bad)


def test_non_stochastic_A_raises(tmp_path: Path):
    """Rows of A that don't sum to 1 → ValueError."""
    bad = tmp_path / "bad_A.toml"
    bad.write_text(
        '[model]\nvariant = "HMC-IN"\nK = 2\n'
        '[prior]\nA = [[0.3,0.3],[0.5,0.5]]\n'
        '[[margins]]\ni=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
    )
    with pytest.raises(ValueError, match="rows must sum to 1"):
        PMCModel(bad)


def test_duplicate_margin_raises(tmp_path: Path):
    """Two margin blocks for the same i (HMC-IN) → ValueError."""
    bad = tmp_path / "dup_margin.toml"
    bad.write_text(
        '[model]\nvariant = "HMC-IN"\nK = 2\n'
        '[prior]\nA = [[0.5,0.5],[0.5,0.5]]\n'
        '[[margins]]\ni=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=0\ndist="norm"\nparams={loc=1.0, scale=1.0}\n'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        PMCModel(bad)


def test_duplicate_pair_margin_raises(tmp_path: Path):
    """Two margin blocks for the same (i,j) (PMC-IN) → ValueError."""
    bad = tmp_path / "dup_pair.toml"
    bad.write_text(
        '[model]\nvariant = "PMC-IN"\nK = 2\n'
        '[prior]\np = [[0.45,0.05],[0.05,0.45]]\n'
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=1\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        PMCModel(bad)


def test_missing_pair_margin_raises(tmp_path: Path):
    """K² blocks but a pair is missing (another duplicated) → ValueError."""
    bad = tmp_path / "missing_pair.toml"
    # K²=4 blocks, but (1,1) absent — (0,0) appears twice. Validator should
    # detect either the duplicate OR the missing pair.
    bad.write_text(
        '[model]\nvariant = "PMC-IN"\nK = 2\n'
        '[prior]\np = [[0.45,0.05],[0.05,0.45]]\n'
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=1\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
    )
    with pytest.raises(ValueError, match="Duplicate|missing"):
        PMCModel(bad)


def test_margin_missing_dist_raises(tmp_path: Path):
    """Margin block without 'dist' key → ValueError."""
    bad = tmp_path / "no_dist.toml"
    bad.write_text(
        '[model]\nvariant = "HMC-IN"\nK = 2\n'
        '[prior]\nA = [[0.5,0.5],[0.5,0.5]]\n'
        '[[margins]]\ni=0\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
    )
    with pytest.raises(ValueError, match="dist"):
        PMCModel(bad)


# ===========================================================================
# 2. weight() — manual reference value
# ===========================================================================

def test_weight_hmc_in_manual():
    """For HMC-IN, weight(i,j,y0,y1) = A[i,j] · f_j(y1)."""
    mdl = PMCModel(MODELS_DIR / "hmc_in_gauss_k2.toml")
    A   = mdl.transition_A
    # f_j is stored at margin (j, *); use (j, 0)
    expected = A[0, 1] * mdl.margin(1, 0).pdf(0.5)
    got      = mdl.weight(0, 1, -0.3, 0.5)
    assert math.isclose(got, expected, rel_tol=1e-12)


def test_weight_pmc_in_manual():
    """For PMC-IN, weight(i,j,y0,y1) = p[i,j] · f_{ij}(y0) · f_{ji}(y1)."""
    mdl = PMCModel(MODELS_DIR / "pmc_in_gauss_k2.toml")
    p   = mdl.prior_p
    expected = p[0, 1] * mdl.margin(0, 1).pdf(-0.2) * mdl.margin(1, 0).pdf(0.4)
    got      = mdl.weight(0, 1, -0.2, 0.4)
    assert math.isclose(got, expected, rel_tol=1e-12)


# ===========================================================================
# 3. save/load round-trip
# ===========================================================================

@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_save_load_roundtrip(toml_path: Path, tmp_path: Path):
    """save() then reload yields the same numerical content."""
    mdl1 = PMCModel(toml_path)
    out  = tmp_path / "saved.toml"
    mdl1.save(out)

    mdl2 = PMCModel(out)
    assert mdl1.variant == mdl2.variant
    assert mdl1.K       == mdl2.K
    np.testing.assert_allclose(mdl1.prior_p, mdl2.prior_p, atol=1e-10)
    np.testing.assert_allclose(mdl1.transition_A, mdl2.transition_A, atol=1e-10)


# ===========================================================================
# 4. simulate() — empirical π matches theoretical
# ===========================================================================

@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_simulate_empirical_pi(toml_path: Path):
    """For N large, the empirical state frequency matches π within 5 σ."""
    mdl = PMCModel(toml_path)
    N   = 10_000
    X, Y = simulate(mdl, N=N, seed=42)

    assert X.shape == (N,)
    expected_y_shape = (N,) if mdl.d == 1 else (N, mdl.d)
    assert Y.shape == expected_y_shape
    assert X.dtype.kind == "i"
    assert np.all((X >= 0) & (X < mdl.K))

    pi   = mdl.stationary_pi
    freq = np.bincount(X, minlength=mdl.K) / N
    # Multinomial sd ≈ sqrt(p(1-p)/N); 5 σ tolerance
    sd   = np.sqrt(pi * (1 - pi) / N)
    np.testing.assert_allclose(freq, pi, atol=5 * sd.max() + 0.01)


def test_simulate_seed_reproducible():
    """Same seed → identical (X, Y)."""
    mdl = PMCModel(MODELS_DIR / "pmc_gauss_k2.toml")
    X1, Y1 = simulate(mdl, N=200, seed=123)
    X2, Y2 = simulate(mdl, N=200, seed=123)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_allclose(Y1, Y2)


def test_simulate_tolerates_pi_drift_within_validator_atol():
    """``simulate`` renormalises ``π`` so float drift never crashes ``rng.choice``.

    Regression test for a GUI-reported crash:
    ``ValueError: Probabilities do not sum to 1``. The PMCModel validator
    accepts up to ``atol=1e-6`` but ``np.random.Generator.choice`` is
    strict to ~``sqrt(eps)≈1.5e-8``. A prior round-tripped through the
    prior tab's 6-decimal display can drift inside that gap.
    """
    mdl = PMCModel(MODELS_DIR / "pmc_gauss_k2.toml")
    # Inject a sub-validator drift directly into the cached π. This
    # mirrors what would happen if a fitted prior's row sums drifted
    # by ~1e-7 (well below the validator's 1e-6 ceiling but well above
    # rng.choice's tolerance).
    drifted = mdl.stationary_pi.copy()
    drifted[0] += 1e-7
    mdl._pi = drifted
    assert abs(drifted.sum() - 1.0) > 1e-8, "drift not large enough"

    # Same trick for the transition matrix rows — must also be safe.
    drifted_A = mdl.transition_A.copy()
    drifted_A[0, 0] += 1e-7
    mdl._A = drifted_A

    # No crash, even with the drifted internals.
    X, Y = simulate(mdl, N=50, seed=0)
    assert X.shape == (50,)
    assert Y.shape == (50,)
    assert np.all((X >= 0) & (X < mdl.K))


# ===========================================================================
# 5. forward/backward — log-likelihood is finite, gamma is row-stochastic
# ===========================================================================

@pytest.mark.parametrize("toml_path", ALL_MODELS, ids=lambda p: p.name)
def test_forward_backward(toml_path: Path):
    mdl = PMCModel(toml_path)
    _, Y = simulate(mdl, N=500, seed=7)

    W, f_pdf = precompute_weights(mdl, Y)
    assert W.shape == (len(Y) - 1, mdl.K, mdl.K)
    assert (W >= 0).all()

    alpha, ll  = forward(mdl, Y, W=W, f_pdf=f_pdf)
    beta       = backward(mdl, Y, W=W)
    gamma      = smooth(alpha, beta)

    assert math.isfinite(ll)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-8)
    assert (gamma >= 0).all() and (gamma <= 1).all()


# ===========================================================================
# 6. classify() — low error on well-separated HMC-IN
# ===========================================================================

def test_classify_hmc_in_k3():
    """K=3 well-separated Gaussian HMM: error rate < 5 % after Hungarian relabeling."""
    mdl = PMCModel(MODELS_DIR / "hmc_in_gauss_k3.toml")
    assert mdl.K == 3
    X_ref, Y = simulate(mdl, N=2000, seed=0)

    # Sanity: simulator hits all three classes
    assert set(np.unique(X_ref)) == {0, 1, 2}

    X_hat, gamma, ll = classify(mdl, Y)
    assert gamma.shape == (2000, 3)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-8)

    er = error_rate(X_ref, X_hat)
    assert er < 0.05, f"Expected < 5 %, got {er:.4f}"


def test_classify_separated_hmc_in(tmp_path: Path):
    """Two well-separated Gaussians (means ±3) → error rate < 5 %."""
    cfg = """
[model]
variant   = "HMC-IN"
K         = 2
N_default = 2000

[prior]
A = [[0.95, 0.05],
     [0.05, 0.95]]

[[margins]]
i = 0
dist = "norm"
params = {loc = -3.0, scale = 1.0}

[[margins]]
i = 1
dist = "norm"
params = {loc = 3.0, scale = 1.0}
"""
    p = tmp_path / "sep.toml"
    p.write_text(cfg)
    mdl = PMCModel(p)
    X_ref, Y = simulate(mdl, N=2000, seed=0)
    X_hat, _, _ = classify(mdl, Y)
    er = error_rate(X_ref, X_hat)
    assert er < 0.05, f"Expected < 5 %, got {er:.4f}"


def test_stationary_distribution_near_reducible(tmp_path: Path):
    """π converges to a valid distribution even when A is nearly reducible."""
    # K=3 chain with two almost-closed classes. Eigen-based extraction with
    # np.linalg.eig is sensitive to which of the near-1 eigenvectors it
    # returns; power iteration handles this gracefully.
    cfg = """
[model]
variant   = "HMC-IN"
K         = 3
N_default = 100

[prior]
A = [[0.999, 0.0005, 0.0005],
     [0.0005, 0.999, 0.0005],
     [0.0005, 0.0005, 0.999]]

[[margins]]
i = 0
dist = "norm"
params = {loc = -3.0, scale = 1.0}

[[margins]]
i = 1
dist = "norm"
params = {loc = 0.0, scale = 1.0}

[[margins]]
i = 2
dist = "norm"
params = {loc = 3.0, scale = 1.0}
"""
    p = tmp_path / "near_reducible.toml"
    p.write_text(cfg)
    mdl = PMCModel(p)

    pi = mdl.stationary_pi
    assert pi.shape == (3,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-10)
    assert (pi > 0).all()
    # By symmetry, all three states have equal stationary prob ≈ 1/3
    np.testing.assert_allclose(pi, np.full(3, 1 / 3), atol=1e-6)
    # And the resulting joint p is a valid probability matrix
    assert np.isclose(mdl.prior_p.sum(), 1.0, atol=1e-10)


def test_forward_raises_on_incompatible_observation(tmp_path: Path):
    """A Y[0] far outside every margin's support → IncompatibleObservationError."""
    from prg.exceptions import IncompatibleObservationError

    # Two narrow Gaussians around 0 and 5; pass Y[0]=1e6 → density underflows on both.
    cfg = """
[model]
variant   = "HMC-IN"
K         = 2
N_default = 10

[prior]
A = [[0.5, 0.5],
     [0.5, 0.5]]

[[margins]]
i = 0
dist = "norm"
params = {loc = 0.0, scale = 1.0}

[[margins]]
i = 1
dist = "norm"
params = {loc = 5.0, scale = 1.0}
"""
    p = tmp_path / "narrow.toml"
    p.write_text(cfg)
    mdl = PMCModel(p)
    Y = np.array([1e6, 0.0, 5.0])  # Y[0] is impossible under both margins
    with pytest.raises(IncompatibleObservationError):
        classify(mdl, Y)


def test_error_rate_label_flip():
    """error_rate handles the trivial K=2 label permutation."""
    X_true = np.array([0, 0, 1, 1, 0, 1])
    X_flip = 1 - X_true
    assert error_rate(X_true, X_flip) == 0.0
    assert error_rate(X_true, X_true) == 0.0


def test_error_rate_K3_permutation():
    """error_rate is invariant under arbitrary K≥3 label permutations."""
    rng = np.random.default_rng(0)
    X_true = rng.integers(0, 3, size=120)

    # Any bijection on {0,1,2} should yield error 0
    for perm in [np.array([2, 0, 1]),
                 np.array([1, 2, 0]),
                 np.array([0, 2, 1])]:
        assert error_rate(X_true, perm[X_true]) == 0.0

    # Same permutation but with 6 cells deliberately swapped to a wrong class
    perm = np.array([2, 0, 1])
    X_hat = perm[X_true].copy()
    wrong = [0, 5, 17, 42, 63, 99]
    for n in wrong:
        # set to a value different from perm[X_true[n]]
        bad = (perm[X_true[n]] + 1) % 3
        X_hat[n] = bad

    # The Hungarian recovers the inverse permutation; remaining errors = 6/120
    assert abs(error_rate(X_true, X_hat) - 6 / 120) < 1e-12


# ===========================================================================
# 7. ICE — log-likelihood (mostly) increases, recovers a sensible τ
# ===========================================================================

def test_ice_fit_margins_preserves_family():
    """ICE with ``fit_margins=True`` must preserve the declared scipy.stats family.

    Regression test for a previous bug in `_fit_gaussian_margin_weighted`
    that silently replaced every margin with a Gaussian, regardless of
    what the user had declared.
    """
    raw = {
        "model":   {"name": "test", "variant": "HMC-IN", "K": 2, "N_default": 500},
        "prior":   {"A": [[0.95, 0.05], [0.05, 0.95]]},
        "margins": [
            {"i": 0, "dist": "norm",    "params": {"loc": -3.0, "scale": 1.0}},
            {"i": 1, "dist": "lognorm", "params": {"s": 0.4, "loc": 5.0, "scale": 1.5}},
        ],
    }
    true_mdl = PMCModel.from_dict(raw)
    _, Y = simulate(true_mdl, N=2000, seed=0)

    # Init from perturbed parameters but the SAME families
    init_raw = {
        "model":   {"name": "init", "variant": "HMC-IN", "K": 2, "N_default": 500},
        "prior":   {"A": [[0.5, 0.5], [0.5, 0.5]]},
        "margins": [
            {"i": 0, "dist": "norm",    "params": {"loc": -1.0, "scale": 2.0}},
            {"i": 1, "dist": "lognorm", "params": {"s": 1.0, "loc": 4.0, "scale": 1.0}},
        ],
    }
    init_mdl = PMCModel.from_dict(init_raw)
    fitted, _trace = ice(init_mdl, Y, ice_cfg={"max_iter": 10, "fit_margins": True})

    fitted_families = [b["dist"] for b in fitted.raw["margins"]]
    assert fitted_families == ["norm", "lognorm"], (
        f"Margin families not preserved: got {fitted_families}, expected "
        f"['norm', 'lognorm']. The fitter is silently overwriting the family."
    )

    # Sanity: parameters should be in the right ballpark
    fitted_norm = next(b for b in fitted.raw["margins"] if b["i"] == 0)
    assert abs(fitted_norm["params"]["loc"]   - (-3.0)) < 0.3
    assert abs(fitted_norm["params"]["scale"] -  1.0)   < 0.3


def test_ice_convergence_pmc():
    """ICE on a perturbed init monotonically improves LL and yields ≥ baseline LL."""
    mdl   = PMCModel(MODELS_DIR / "pmc_gauss_k2.toml")
    X, Y  = simulate(mdl, N=1500, seed=11)

    # Perturbed initial model: shift τ by -0.2 on all copulas
    import copy
    raw = copy.deepcopy(mdl._raw)
    for blk in raw.get("copulas", []):
        blk["tau"] = max(-0.9, blk.get("tau", 0.0) - 0.2)
    init = PMCModel.from_dict(raw)

    fitted, trace = ice(
        init, Y,
        ice_cfg={"max_iter": 15, "candidates": ["Gauss", "Clayton", "GH"]},
    )
    lls = trace.log_liks

    # Final LL should be ≥ initial LL (non-monotonic on isolated steps is OK,
    # but the total improvement must be non-trivial).
    assert lls[-1] > lls[0] - 1e-6
    assert lls[-1] - lls[0] > 1.0   # at least 1 nat of improvement

    # Diagonal copula τ should recover near 0.6 (the true value)
    tau00 = fitted.copula(0, 0).params["tau_k"]
    tau11 = fitted.copula(1, 1).params["tau_k"]
    assert abs(tau00 - 0.6) < 0.15, f"τ00={tau00:.3f}"
    assert abs(tau11 - 0.6) < 0.15, f"τ11={tau11:.3f}"
