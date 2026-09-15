"""
peano.py — 2D image ↔ 1D signal transforms via the Generalized Hilbert
("Gilbert") space-filling curve, plus grayscale image I/O.

Rationale
---------
The PMC/HMC machinery in :mod:`pmcprg.pmc.inference` and :mod:`pmcprg.pmc.ice`
operates on 1D observation sequences. To apply it to 2D images we need a
1D linearisation that **preserves spatial locality** — neighbours in 2D
should remain neighbours (or near-neighbours) in 1D, so the Markov
hypothesis on the 1D signal approximates spatial dependence in the image.

A naïve raster scan fails this requirement: the last column of a row and
the first column of the next row are 1D-adjacent but 2D-far. Space-filling
curves (Hilbert, Peano, …) solve this. We use the **Generalized Hilbert
curve** (Jakub Červený's "gilbert" algorithm, vendored in :mod:`._gilbert`)
because it works on **any rectangular shape** (no power-of-2 constraint).

The path is deterministic: `gilbert2d(W, H)` always returns the same
sequence of `(x, y)` for a given `(W, H)`. We therefore cache it.

The idea of running a chain model along a Hilbert–Peano scan of the image
is the one the PMC papers use: Skarbek (1992) for the scan itself, Giordana
& Pieczynski (1997) for hidden Markov chains on it, Derrode & Pieczynski
(2004) for pairwise Markov chains. The *code* is Červený's gilbert.

References
----------
* Skarbek, W. (1992). Generalized Hilbert scan in image printing. In
  R. Klette & W. Kropatsch (eds), *Theoretical Foundations of Computer
  Vision*, Akademie Verlag, Berlin.
* Giordana, N. & Pieczynski, W. (1997). Estimation of generalized
  multisensor hidden Markov chains and unsupervised image segmentation.
  *IEEE Trans. PAMI* 19(5), 465–475.
* Derrode, S. & Pieczynski, W. (2004). Signal and image segmentation using
  pairwise Markov chains. *IEEE Trans. Signal Processing* 52(9), 2477–2489.

Public API
----------
:func:`peano_path`        — list of (col, row) pairs for an H×W image.
:func:`image_to_signal`   — reshape 2D image to 1D signal along the path.
:func:`signal_to_image`   — reshape 1D signal back to 2D along inverse path.
:func:`load_grayscale`    — read as 2D float64 array in [0, 1].
:func:`load_color`        — read as 3D (H, W, 3) RGB float64 in [0, 1].
:func:`load_labels`       — read a class-label PNG as int32 (no rescaling).
:func:`save_segmentation` — save a 2D label map as PNG with a colour palette.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np

from pmcprg.pmc._gilbert import gilbert2d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path construction (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _path_xy(width: int, height: int) -> np.ndarray:
    """Return the gilbert path as an (N, 2) int array of (x, y) = (col, row).

    Cached because the iteration is pure-Python and slow on large grids
    (e.g. ~50 ms on a 512×512 image), and the path depends only on the
    shape — not on the image content.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got ({width}, {height}).")
    return np.array(list(gilbert2d(width, height)), dtype=np.int64)


def peano_path(shape: Tuple[int, int]) -> np.ndarray:
    """Public accessor: gilbert path for a 2D image of given shape.

    Parameters
    ----------
    shape : (H, W) tuple — image shape as numpy uses it (rows, cols).

    Returns
    -------
    path : np.ndarray, shape (H*W, 2), dtype int64
        ``path[k]`` = ``(col, row)`` of the k-th visited pixel.

    Notes
    -----
    Numpy uses ``(rows, cols) = (H, W)`` while gilbert uses ``(width, height)``.
    We swap once here so callers always pass numpy-style shapes.
    """
    H, W = shape
    return _path_xy(W, H)


# ---------------------------------------------------------------------------
# 2D image  →  1D signal
# ---------------------------------------------------------------------------

def image_to_signal(img: np.ndarray) -> np.ndarray:
    """Linearise a 2D image (mono- or multi-channel) along the gilbert path.

    Parameters
    ----------
    img : np.ndarray, shape (H, W) or (H, W, d)
        Mono-channel grayscale image (2D) or multi-channel image (3D, with
        the channel axis last — e.g. RGB has ``d=3``).

    Returns
    -------
    sig : np.ndarray
        Shape ``(H*W,)`` if ``img`` is 2D, else ``(H*W, d)``. Same dtype.
        ``sig[k] = img[row_k, col_k(, :)]`` where ``(col_k, row_k)`` is the
        k-th cell of the gilbert path.
    """
    if img.ndim not in (2, 3):
        raise ValueError(
            f"image_to_signal expects 2D (H, W) or 3D (H, W, d); "
            f"got shape {img.shape}."
        )
    H, W = img.shape[:2]
    path = peano_path((H, W))
    # Fancy indexing: img[rows, cols] (or img[rows, cols, :] for 3D, which
    # numpy collapses naturally when we omit the trailing slice).
    return img[path[:, 1], path[:, 0]]


# ---------------------------------------------------------------------------
# 1D signal  →  2D image
# ---------------------------------------------------------------------------

def signal_to_image(sig: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Re-fold a 1D (scalar or vector) signal back into a 2D image.

    Parameters
    ----------
    sig   : np.ndarray
        Shape ``(N,)`` for scalar pixels, or ``(N, d)`` for vector pixels
        (e.g. an RGB segmentation never produces this — only the original
        observations do).
    shape : (H, W) tuple — target image shape, with ``H * W = N``.

    Returns
    -------
    img : np.ndarray
        Shape ``(H, W)`` for scalar input or ``(H, W, d)`` for vector input.
        ``img[row_k, col_k(, :)] = sig[k(, :)]``. Same dtype as ``sig``.
    """
    H, W = shape
    if sig.ndim not in (1, 2):
        raise ValueError(
            f"signal_to_image expects a 1D or 2D array; got shape {sig.shape}."
        )
    if sig.shape[0] != H * W:
        raise ValueError(
            f"signal length {sig.shape[0]} does not match shape {shape} "
            f"(expected {H * W})."
        )
    path = peano_path((H, W))
    if sig.ndim == 1:
        img = np.empty((H, W), dtype=sig.dtype)
    else:
        img = np.empty((H, W, sig.shape[1]), dtype=sig.dtype)
    img[path[:, 1], path[:, 0]] = sig
    return img


# ---------------------------------------------------------------------------
# Grayscale image I/O (Pillow)
# ---------------------------------------------------------------------------

def load_grayscale(path: str | Path) -> np.ndarray:
    """Load an image file as a 2D grayscale array of float64 in [0, 1].

    Accepts any format Pillow can decode (PNG, JPG, BMP, TIFF, …). RGB
    inputs are converted with PIL's ``L`` mode (ITU-R 601-2 luma:
    ``L = R*299/1000 + G*587/1000 + B*114/1000``); already-grayscale or
    paletted inputs are converted via the same mode for uniformity. Pure
    RGB images whose three channels are identical (the case the user
    cares about) round-trip exactly.

    Parameters
    ----------
    path : str | pathlib.Path

    Returns
    -------
    img : np.ndarray, shape (H, W), dtype float64, values in [0, 1].
    """
    try:
        from PIL import Image
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "Pillow is required for image I/O. "
            "Install it with: pip install pillow  (or `pip install awesomepmc[image]`)."
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")

    with Image.open(p) as im:
        gray = im.convert("L")              # 8-bit grayscale 0..255
        arr  = np.asarray(gray, dtype=np.float64) / 255.0
    logger.debug("Loaded grayscale image %s (shape=%s)", p, arr.shape)
    return arr


def load_color(path: str | Path) -> np.ndarray:
    """Load an image as a 3D RGB array of float64 in [0, 1].

    Pillow handles palette / CMYK / 16-bit / etc. inputs by converting to
    ``RGB`` (3 channels). Pure-grayscale inputs are broadcast to RGB.

    Parameters
    ----------
    path : str | pathlib.Path

    Returns
    -------
    img : np.ndarray, shape (H, W, 3), dtype float64, values in [0, 1].
    """
    try:
        from PIL import Image
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "Pillow is required for image I/O. "
            "Install it with: pip install pillow  (or `pip install awesomepmc[image]`)."
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")

    with Image.open(p) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.float64) / 255.0
    logger.debug("Loaded color image %s (shape=%s)", p, arr.shape)
    return arr


def load_labels(path: str | Path) -> np.ndarray:
    """Load a class-label PNG as a 2D integer array (no normalisation).

    Reference label maps for segmentation are stored either as an 8-bit
    paletted PNG (mode ``"P"``) with raw indices ``0, 1, …, K-1``, or as
    an 8-bit grayscale PNG (mode ``"L"``) with the same raw indices. This
    helper preserves the integer values as-is, unlike :func:`load_grayscale`
    which normalises to [0, 1].

    Parameters
    ----------
    path : str | pathlib.Path

    Returns
    -------
    labels : np.ndarray, shape (H, W), dtype int32, values in 0..255.
    """
    try:
        from PIL import Image
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "Pillow is required for image I/O. "
            "Install it with: pip install pillow  (or `pip install awesomepmc[image]`)."
        ) from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"label image not found: {p}")

    with Image.open(p) as im:
        # For paletted PNGs, ``np.asarray(im)`` returns the palette indices
        # directly (which is what we want). For RGB or RGBA inputs we
        # collapse to a single channel via ``L``-mode, since label images
        # are intrinsically scalar — but we do NOT divide by 255.
        if im.mode in ("P", "L"):
            arr = np.asarray(im, dtype=np.int32)
        else:
            arr = np.asarray(im.convert("L"), dtype=np.int32)
    logger.debug("Loaded labels %s (shape=%s, unique=%s)", p, arr.shape, np.unique(arr).tolist())
    return arr


def save_segmentation(
    path: str | Path,
    labels: np.ndarray,
    K: int | None = None,
) -> None:
    """Save a 2D class-label map as a paletted PNG.

    Parameters
    ----------
    path : str | pathlib.Path
        Output PNG path.
    labels : np.ndarray, shape (H, W), integer dtype
        Class indices in ``{0, …, K-1}``.
    K : int, optional
        Number of classes (used to size the palette). Defaults to
        ``labels.max() + 1``.

    Notes
    -----
    The palette uses evenly-spaced hues from matplotlib's ``tab10`` for
    ``K ≤ 10``, falling back to ``viridis`` otherwise. The PNG has mode
    ``"P"`` so file size scales with ``H × W`` independently of ``K``.
    """
    try:
        from PIL import Image
    except ImportError as exc:                                # pragma: no cover
        raise ImportError(
            "Pillow is required for image I/O. "
            "Install it with: pip install pillow  (or `pip install awesomepmc[image]`)."
        ) from exc
    import matplotlib

    if labels.ndim != 2:
        raise ValueError(f"labels must be 2D, got shape {labels.shape}.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be integer dtype, got {labels.dtype}.")

    if K is None:
        K = int(labels.max()) + 1
    if K > 256:
        raise ValueError(f"PNG paletted mode supports K ≤ 256, got K={K}.")

    cmap_name = "tab10" if K <= 10 else "viridis"
    cmap      = matplotlib.colormaps[cmap_name]
    # 256-entry palette, first K entries = our class colours, rest black.
    palette = np.zeros((256, 3), dtype=np.uint8)
    for k in range(K):
        rgba = cmap(k / max(K - 1, 1))
        palette[k] = (np.array(rgba[:3]) * 255.0).astype(np.uint8)

    arr = labels.astype(np.uint8)
    im  = Image.fromarray(arr, mode="P")
    im.putpalette(palette.flatten().tolist())
    im.save(path)
    logger.debug("Saved segmentation %s (K=%d, shape=%s)", path, K, labels.shape)
