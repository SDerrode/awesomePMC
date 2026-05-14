"""
test_peano.py — image ↔ signal transforms and end-to-end classification.

Covers
------
* Vendored ``gilbert2d`` correctness: every cell visited exactly once,
  consecutive cells differ by L1 = 1 (locality preserved).
* ``image_to_signal`` / ``signal_to_image`` round-trip on multiple shapes,
  including non-square and 1×N strips.
* End-to-end ``classify_image`` works on a synthetic 2D image and
  recovers a usable error rate.
* ``ice_image`` runs without raising and returns a fitted model.
* Pillow I/O round-trip through a real PNG file (skip if Pillow missing).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the package under test is importable when the suite runs from the
# project root without installation.
_HERE = Path(__file__).resolve().parents[2]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from prg.pmc._gilbert import gilbert2d
from prg.pmc.peano import (
    image_to_signal,
    peano_path,
    signal_to_image,
)


# ---------------------------------------------------------------------------
# Vendored gilbert2d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("W, H", [
    (1, 1), (1, 7), (7, 1),       # degenerate strips
    (8, 8), (16, 16),             # power-of-2 squares
    (5, 7), (7, 5),               # odd, non-square
    (100, 150), (150, 100),       # rectangular non-power-of-2
])
def test_gilbert_visits_every_cell_exactly_once(W, H):
    coords = list(gilbert2d(W, H))
    assert len(coords) == W * H, f"path length {len(coords)} ≠ W·H={W*H}"
    assert len(set(coords)) == W * H, "duplicate cells in path"
    # All coordinates inside the rectangle.
    for x, y in coords:
        assert 0 <= x < W and 0 <= y < H


@pytest.mark.parametrize("W, H", [
    (8, 8), (5, 7), (100, 150), (1, 7),
])
def test_gilbert_locality_preserved(W, H):
    """Consecutive cells must be 4-connected neighbours (L1 = 1)."""
    coords = list(gilbert2d(W, H))
    if len(coords) < 2:
        return
    diffs = np.abs(np.diff(np.array(coords), axis=0)).sum(axis=1)
    assert diffs.max() == 1, f"non-unit step in gilbert path: {diffs.max()}"


# ---------------------------------------------------------------------------
# image_to_signal / signal_to_image round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H, W", [
    (8, 8), (5, 7), (100, 150), (1, 7), (7, 1), (32, 48),
])
def test_image_signal_roundtrip(H, W):
    rng = np.random.default_rng(0)
    img = rng.normal(size=(H, W))
    sig = image_to_signal(img)
    rec = signal_to_image(sig, (H, W))
    assert sig.shape == (H * W,)
    assert np.array_equal(img, rec)


def test_signal_to_image_rejects_wrong_length():
    sig = np.arange(10)
    with pytest.raises(ValueError, match="signal length"):
        signal_to_image(sig, (3, 4))   # expects length 12


def test_image_to_signal_rejects_non_2d():
    """A 4D input is neither (H, W) nor (H, W, d) and must be rejected."""
    with pytest.raises(ValueError, match="2D|3D"):
        image_to_signal(np.zeros((3, 4, 5, 6)))


def test_peano_path_cached():
    """Two calls with the same shape should return the *same* array object
    (lru_cache hit) — confirms caching is wired."""
    p1 = peano_path((4, 6))
    p2 = peano_path((4, 6))
    assert p1 is p2


# ---------------------------------------------------------------------------
# End-to-end image classification
# ---------------------------------------------------------------------------

def test_classify_image_recovers_synthetic_segmentation():
    """High-contrast synthetic image: classify_image should recover labels
    with low error.

    Build a 2-class scene by simulating a 1D PMC chain with means ±3 and
    folding it onto a 2D grid via the gilbert path. classify with the
    same model recovers the labels nearly perfectly because the chain is
    Markov along the gilbert path *by construction*.
    """
    from prg.pmc import PMCModel, simulate, signal_to_image, classify_image, error_rate

    mdl0 = PMCModel("prg/pmc/models/hmc_in_gauss_k2.toml")
    raw = copy.deepcopy(mdl0._raw)
    raw["margins"][0]["params"]["loc"] = -3.0
    raw["margins"][1]["params"]["loc"] = +3.0
    mdl = PMCModel.from_dict(raw)

    H, W = 32, 48
    X1d, Y1d = simulate(mdl, N=H * W, seed=42)
    img      = signal_to_image(Y1d, (H, W))
    X_ref_2d = signal_to_image(X1d, (H, W))

    X_hat_2d, gamma_2d, log_lik = classify_image(mdl, img)

    assert X_hat_2d.shape == (H, W)
    assert gamma_2d.shape == (H, W, mdl.K)
    assert np.isfinite(log_lik)

    er = error_rate(X_ref_2d.ravel(), X_hat_2d.ravel())
    assert er < 0.05, f"unexpectedly high error rate {er:.3f}"


def test_ice_image_runs():
    """ICE on a 2D image should produce a model and a non-empty trace."""
    from prg.pmc import PMCModel, simulate, signal_to_image, ice_image

    mdl = PMCModel("prg/pmc/models/hmc_in_gauss_k2.toml")
    H, W = 16, 24
    _X1d, Y1d = simulate(mdl, N=H * W, seed=1)
    img = signal_to_image(Y1d, (H, W))

    fitted, trace = ice_image(
        mdl, img, ice_cfg={"max_iter": 3, "candidates": ["Gauss"]},
    )
    assert fitted.K == mdl.K
    assert len(trace.log_liks) >= 1


# ---------------------------------------------------------------------------
# Pillow I/O round-trip — skipped if Pillow not installed
# ---------------------------------------------------------------------------

def test_load_save_roundtrip_through_png(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    from prg.pmc.peano import load_grayscale, load_labels, save_segmentation

    # Save a known grayscale image, load it back, check shape + range.
    arr = (np.arange(256).reshape(16, 16)).astype("uint8")
    p = tmp_path / "gray.png"
    Image.fromarray(arr, mode="L").save(p)
    loaded = load_grayscale(p)
    assert loaded.shape == (16, 16)
    assert loaded.dtype == np.float64
    assert loaded.min() >= 0.0 and loaded.max() <= 1.0
    # Pixel values should match raw / 255 to floating-point tolerance.
    np.testing.assert_allclose(loaded, arr.astype(np.float64) / 255.0)

    # Save a segmentation map and read it back as labels (no normalisation).
    labels = (np.arange(16 * 16) % 4).reshape(16, 16).astype(np.int32)
    seg = tmp_path / "seg.png"
    save_segmentation(seg, labels, K=4)
    reloaded = load_labels(seg)
    assert reloaded.shape == (16, 16)
    assert reloaded.dtype == np.int32
    assert set(np.unique(reloaded).tolist()) == set(np.unique(labels).tolist())
    # Indices should match exactly (paletted PNG preserves indices).
    np.testing.assert_array_equal(reloaded, labels)


def test_classify_image_rejects_non_2d():
    """A scalar (d=1) model rejects a multi-channel image: dimension mismatch."""
    from prg.pmc import PMCModel, classify_image
    mdl = PMCModel("prg/pmc/models/hmc_in_gauss_k2.toml")
    with pytest.raises(ValueError, match="Image channels"):
        classify_image(mdl, np.zeros((3, 4, 5)))


def test_ice_image_rejects_non_2d():
    """A scalar (d=1) model rejects a multi-channel image: dimension mismatch."""
    from prg.pmc import PMCModel, ice_image
    mdl = PMCModel("prg/pmc/models/hmc_in_gauss_k2.toml")
    with pytest.raises(ValueError, match="Image channels"):
        ice_image(mdl, np.zeros((3, 4, 5)))
