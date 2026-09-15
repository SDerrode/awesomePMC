"""
test_multivariate.py — vector observations (multivariate Gaussian margins).

Covers
------
* TOML schema: ``[model] d``, ``multivariate_normal`` margins.
* Validation: copula-using variants rejected when d > 1; mismatched d
  rejected; image dimensionality must match ``model.d``.
* Image transforms round-trip on (H, W, d) RGB-shaped arrays.
* ``simulate`` emits Y of shape (N, d) for vector models.
* ``classify`` and ``classify_image`` recover labels with low error on
  synthetic high-contrast multivariate Gaussian data.
* ICE multivariate M-step (closed-form) recovers means and covariance
  from a perturbed init.
* Backward compat: existing scalar models still load with ``d == 1``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pmcprg.pmc import (
    PMCModel,
    classify,
    classify_image,
    error_rate,
    ice_image,
    image_to_signal,
    signal_to_image,
    simulate,
)
from pmcprg.pmc.ice import _fit_multivariate_gaussian_weighted


# ---------------------------------------------------------------------------
# Schema / validation
# ---------------------------------------------------------------------------

def _mvn_dict(K=2, d=3, variant="HMC-IN"):
    """Build a minimal multivariate model dict for from_dict construction."""
    base = {
        "model": {"name": f"test {variant} d={d}", "variant": variant, "K": K, "d": d},
        "margins": [
            {
                "i": k,
                "dist": "multivariate_normal",
                "params": {
                    "mean": [float(m) for m in (np.arange(d) + k * 5.0)],
                    "cov":  np.eye(d).tolist(),
                },
            }
            for k in range(K)
        ],
    }
    if variant.startswith("HMC"):
        A = np.full((K, K), 0.05) + 0.85 * np.eye(K)
        A /= A.sum(axis=1, keepdims=True)
        base["prior"] = {"A": A.tolist()}
    else:
        # PMC-IN — joint prior summing to 1.
        p = np.full((K, K), 0.05 / (K * K - K)) + (0.95 / K) * np.eye(K)
        p /= p.sum()
        base["prior"] = {"p": p.tolist()}
    return base


def test_default_d_is_one_for_legacy_models():
    """All shipped scalar models load with d=1 (no [model] d field)."""
    for f in sorted((Path(__file__).parent.parent / "pmc" / "models").glob("*.toml")):
        m = PMCModel(f)
        # The new RGB model declares d=3 — exclude it from this check.
        if "mvn" in f.name:
            continue
        assert m.d == 1, f"{f.name} should default to d=1"


def test_loads_multivariate_model():
    m = PMCModel.from_dict(_mvn_dict(K=2, d=3, variant="HMC-IN"))
    assert m.d == 3
    assert m.K == 2
    mg = m.margin(0)
    assert mg.is_multivariate
    assert mg.d == 3
    # PDF on a vector
    p = mg.pdf(np.zeros(3))
    assert np.isfinite(p) and p > 0


@pytest.mark.parametrize("variant", ["HMC-DN", "PMC"])
def test_rejects_copula_variants_when_d_gt_1(variant):
    """Copulas need scalar margins — must be refused at d > 1."""
    raw = _mvn_dict(K=2, d=3, variant="HMC-IN")
    raw["model"]["variant"] = variant
    with pytest.raises(ValueError, match="copula"):
        PMCModel.from_dict(raw)


def test_rejects_mismatched_margin_dimension():
    """A multivariate margin with the wrong dim must be flagged."""
    raw = _mvn_dict(K=2, d=3)
    raw["margins"][0]["params"]["mean"] = [0.0, 0.0]  # length 2, but d=3
    raw["margins"][0]["params"]["cov"]  = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match=r"dimension 2.*\[model\]\.d = 3"):
        PMCModel.from_dict(raw)


def test_rejects_scalar_margin_when_d_gt_1():
    """A scalar dist mixed into a d>1 model must be rejected."""
    raw = _mvn_dict(K=2, d=3)
    raw["margins"][1] = {"i": 1, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}}
    with pytest.raises(ValueError, match=r"\b(d|dimension)\b"):
        PMCModel.from_dict(raw)


# ---------------------------------------------------------------------------
# Image transforms — vector images
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H, W, d", [
    (8, 8, 3), (5, 7, 3), (32, 48, 5), (16, 24, 1), (1, 7, 4),
])
def test_image_signal_roundtrip_vector(H, W, d):
    rng = np.random.default_rng(0)
    img = rng.normal(size=(H, W, d))
    sig = image_to_signal(img)
    rec = signal_to_image(sig, (H, W))
    assert sig.shape == (H * W, d)
    assert rec.shape == (H, W, d)
    assert np.array_equal(img, rec)


def test_image_signal_scalar_unchanged():
    """Scalar (H, W) → (N,) → (H, W) round-trip still works."""
    rng = np.random.default_rng(0)
    img = rng.normal(size=(10, 12))
    sig = image_to_signal(img)
    rec = signal_to_image(sig, (10, 12))
    assert sig.shape == (120,)
    assert rec.shape == (10, 12)
    assert np.array_equal(img, rec)


def test_image_to_signal_rejects_bad_ndim():
    with pytest.raises(ValueError, match="2D|3D"):
        image_to_signal(np.zeros((3, 4, 5, 6)))


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_emits_vector_y_for_d_gt_1():
    m = PMCModel.from_dict(_mvn_dict(K=2, d=3, variant="HMC-IN"))
    X, Y = simulate(m, N=200, seed=0)
    assert X.shape == (200,)
    assert Y.shape == (200, 3)


def test_simulate_emits_scalar_y_for_d_eq_1():
    m = PMCModel("pmcprg/pmc/models/hmc_in_gauss_k2.toml")
    X, Y = simulate(m, N=200, seed=0)
    assert X.shape == (200,) and Y.shape == (200,)


# ---------------------------------------------------------------------------
# classify on vector observations
# ---------------------------------------------------------------------------

def test_classify_recovers_high_contrast_multivariate():
    """K=2 well-separated 3D Gaussians → near-zero error."""
    m = PMCModel.from_dict(_mvn_dict(K=2, d=3, variant="HMC-IN"))
    rng = np.random.default_rng(0)
    N = 1000
    # Direct chain sample (avoid copula path)
    A = m.transition_A
    pi = m.stationary_pi
    X = np.empty(N, dtype=int); X[0] = rng.choice(2, p=pi)
    for n in range(1, N):
        X[n] = rng.choice(2, p=A[X[n-1]])
    Y = np.zeros((N, 3))
    for k in (0, 1):
        Y[X == k] = m.margin(k).rvs(int((X == k).sum()), rng)

    X_hat, gamma, ll = classify(m, Y)
    assert X_hat.shape == (N,)
    assert gamma.shape == (N, 2)
    assert np.isfinite(ll)
    # The class means here are (0, 1, 2) and (5, 6, 7) — extreme contrast.
    err = error_rate(X, X_hat)
    assert err < 0.02, f"expected near-perfect recovery, got err={err:.3f}"


def test_classify_image_rgb_synthetic():
    """End-to-end on an RGB image folded back via gilbert."""
    m = PMCModel("pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml")
    H, W = 32, 48; N = H * W
    rng = np.random.default_rng(42)
    A = m.transition_A; pi = m.stationary_pi
    X = np.empty(N, dtype=int); X[0] = rng.choice(2, p=pi)
    for n in range(1, N):
        X[n] = rng.choice(2, p=A[X[n-1]])
    Y_1d = np.zeros((N, 3))
    for k in (0, 1):
        Y_1d[X == k] = m.margin(k).rvs(int((X == k).sum()), rng)
    img = signal_to_image(Y_1d, (H, W))
    assert img.shape == (H, W, 3)

    X_hat_2d, gamma_2d, _ll = classify_image(m, img)
    assert X_hat_2d.shape == (H, W)
    assert gamma_2d.shape == (H, W, 2)
    err = error_rate(signal_to_image(X, (H, W)).ravel(), X_hat_2d.ravel())
    assert err < 0.10


def test_classify_image_rejects_dimensionality_mismatch():
    m = PMCModel("pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml")
    # Pass a 2D grayscale image to a d=3 model
    with pytest.raises(ValueError, match="Image channels"):
        classify_image(m, np.zeros((10, 12)))
    # Pass a 3D image with wrong number of channels
    m1d = PMCModel("pmcprg/pmc/models/hmc_in_gauss_k2.toml")
    with pytest.raises(ValueError, match="Image channels"):
        classify_image(m1d, np.zeros((10, 12, 3)))


# ---------------------------------------------------------------------------
# ICE M-step — multivariate
# ---------------------------------------------------------------------------

def test_fit_multivariate_gaussian_weighted_recovers_mean_cov():
    """Closed-form weighted MLE returns the analytical (μ, Σ)."""
    rng = np.random.default_rng(0)
    N = 5000
    mu_true  = np.array([1.0, -2.0, 0.5])
    cov_true = np.array([[2.0, 0.5, 0.3],
                         [0.5, 1.0, 0.2],
                         [0.3, 0.2, 1.5]])
    Y = rng.multivariate_normal(mu_true, cov_true, size=N)
    w = np.ones(N) / N
    out = _fit_multivariate_gaussian_weighted(Y, w)
    mu_est  = np.array(out["params"]["mean"])
    cov_est = np.array(out["params"]["cov"])
    np.testing.assert_allclose(mu_est,  mu_true,  atol=0.1)
    np.testing.assert_allclose(cov_est, cov_true, atol=0.2)


def test_ice_image_recovers_means_from_perturbed_init():
    """ICE on a synthetic RGB image shifts the means back towards truth."""
    m_true = PMCModel("pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml")

    # Build the synthetic image
    H, W = 32, 48; N = H * W
    rng = np.random.default_rng(42)
    A = m_true.transition_A; pi = m_true.stationary_pi
    X = np.empty(N, dtype=int); X[0] = rng.choice(2, p=pi)
    for n in range(1, N):
        X[n] = rng.choice(2, p=A[X[n-1]])
    Y_1d = np.zeros((N, 3))
    for k in (0, 1):
        Y_1d[X == k] = m_true.margin(k).rvs(int((X == k).sum()), rng)
    img = signal_to_image(Y_1d, (H, W))

    # Perturbed init
    raw = copy.deepcopy(m_true._raw)
    raw["margins"][0]["params"]["mean"] = [-0.3, -0.2, -0.1]
    raw["margins"][1]["params"]["mean"] = [ 0.3,  0.2,  0.1]
    init = PMCModel.from_dict(raw)

    fitted, trace = ice_image(init, img, ice_cfg={"max_iter": 12})
    assert len(trace.log_liks) >= 2
    # Log-lik should improve (allow tiny tolerance for already-converged starts)
    assert trace.log_liks[-1] > trace.log_liks[0] - 1e-6

    # Fitted means should be much closer to the truth than the init was
    for k in (0, 1):
        true_mu = np.array(m_true.margin(k).params["mean"])
        init_mu = np.array(init.margin(k).params["mean"])
        fit_mu  = np.array(fitted.margin(k).params["mean"])
        # Permitted: either the same labels (close to true), or swapped (close to other true mean)
        d_true_k    = np.linalg.norm(fit_mu - true_mu)
        d_true_swap = np.linalg.norm(fit_mu - np.array(m_true.margin(1 - k).params["mean"]))
        assert min(d_true_k, d_true_swap) < np.linalg.norm(init_mu - true_mu)
