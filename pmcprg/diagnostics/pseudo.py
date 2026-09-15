"""
pseudo.py — pseudo-observations and margin samples for post-fit diagnostics.

Why this exists
---------------
Every diagnostic that looks at a fitted pairwise Markov chain starts by
pushing the observations through the fitted margin CDFs, and every one of
them used to write ``margin(k).cdf_vec(Y)`` for the K states. That is only
right for **state** margins (``f_ij = f_i``). A general PMC (A16 Eqs. 12–14)
attaches a density to the *pair* of states,

    f(y_n, y_{n+1} | x_n = i, x_{n+1} = j)
        = f_ij(y_n) · f_ji(y_{n+1}) · c_ij(F_ij(y_n), F_ji(y_{n+1})),

so the left observation of pair ``(i, j)`` follows ``f_ij`` and the right one
follows ``f_ji`` — the index inversion of Eq. 12. ``margin(i)`` refuses to
answer for such a model, and a diagnostic that collapsed to ``f_i`` would test
a model other than the one that was fitted.

The helpers below build the objects the diagnostics consume from an
``(N, K, K)`` array ``F[n, i, j] = F_ij(y_n)``. For a state model that array is
the K state CDFs broadcast over ``j``, so every construction below reduces
**value for value** to the one the package used before: the pseudo-observations
are the same floats, not merely the same up to rounding.

What is built
-------------
* :func:`is_pair` — whether a model carries pair margins.
* :func:`margin_cdfs` — ``F[n, i, j] = F_ij(y_n)``, clipped away from 0 and 1.
* :func:`copula_pseudo_obs` — the sample a copula ``c_ij`` is tested (or
  fitted) on: ``(F_ij(y_n), F_ji(y_{n+1}))`` with weight ``ξ_n(i, j)``.
* :func:`margin_pit_dual` — the sample a pair margin ``f_ij`` is assessed on,
  in the dual view: every ``y_n`` enters as the left observation of the pair
  ``(i, j)`` (PIT ``F_ij(y_n)``, weight ``ξ_n(i, j)``) and every ``y_{n+1}`` as
  the right observation of the pair ``(j, i)``, whose right margin is
  ``f_ij`` (PIT ``F_ij(y_{n+1})``, weight ``ξ_n(j, i)``). This is the update
  ICE/SEM use for ``f_ij``, so the diagnostic and the estimator look at the
  same weighted sample.
* :func:`margin_keys` / :func:`margin_sample` — the per-margin sample of one
  label path (a posterior draw, the MPM path, the true labels): ``{y_n :
  x_n = k}`` for a state margin, and for a pair margin the hard-label version
  of the dual view, ``{y_n : (x_n, x_{n+1}) = (i, j)} ∪ {y_{n+1} : (x_n,
  x_{n+1}) = (j, i)}``.
* :func:`state_law` — the law of ``y_n`` given ``x_n = i`` alone, which a
  per-state sample ``{y_n : x_n = i}`` follows: ``f_i`` for state margins, and
  for pair margins A16 Eq. 12's left margin summed over ``x_{n+1}``,

      g_i(y) = Σ_j p(x_{n+1} = j | x_n = i) f_ij(y) = Σ_j (p_ij / p_i) f_ij(y),
      p_i = Σ_j p_ij

  (:class:`StateMixture`, weights :func:`state_weights`, family label
  :func:`state_law_name`). On a state model it returns the model's own
  density object, so nothing computed on a state model changes.

The GUI uses these helpers through :mod:`pmcprg.pmc.gui._margins`, which
re-exports them and only adds its ``lo``/``hi`` form of :func:`margin_cdfs`.

Like the rest of :mod:`pmcprg.diagnostics`, nothing here imports the model class:
:func:`margin_cdfs` and :func:`margin_keys` only need ``K``,
``margin_structure`` and ``margin(i, j)``; :func:`state_law` also reads
``transition_A``, and :func:`state_law_name` ``margin_blocks()``.

References
----------
* Derrode, S. & Pieczynski, W. (2013). Unsupervised data classification using
  pairwise Markov chains with automatic copulas selection. *Comput. Statist.
  Data Anal.* 63, 81–98 (A16) — Eq. 12 (pair density and index inversion),
  Eqs. 13–14 (transition), §2.1 Proposition (state margins ⇔ X Markov).
* Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for
  copulas: A review and a power study. *Insurance Math. Econom.* 44(2),
  199–213 — pseudo-observations as the input of copula GoF statistics.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "StateMixture",
    "copula_pseudo_obs",
    "is_pair",
    "margin_cdfs",
    "margin_keys",
    "margin_of",
    "margin_pit_dual",
    "margin_sample",
    "mixture_label",
    "state_law",
    "state_law_name",
    "state_weights",
]

#: Default clipping of the CDF values — the one every diagnostic of the
#: package used when it built the K state CDF vectors.
CDF_CLIP = 1e-12


def is_pair(model) -> bool:
    """``True`` when ``model`` carries pair margins f_ij."""
    return getattr(model, "margin_structure", "state") == "pair"


def margin_keys(model) -> list:
    """Keys of the distinct margins of ``model``, in a stable order.

    ``[0, …, K−1]`` for state margins, ``[(0, 0), (0, 1), …, (K−1, K−1)]`` for
    pair margins. ``model.margin(*key)`` (or ``model.margin(key)`` for an
    integer key) returns the corresponding density.
    """
    K = int(model.K)
    if is_pair(model):
        return [(i, j) for i in range(K) for j in range(K)]
    return list(range(K))


def margin_of(model, key):
    """The margin density behind a key of :func:`margin_keys`."""
    if isinstance(key, tuple):
        return model.margin(*key)
    return model.margin(key)


def _clip_bounds(clip):
    """``(lo, hi)`` from the ``clip`` argument of :func:`margin_cdfs`."""
    if clip is None:
        return None
    if np.ndim(clip) == 0:
        return clip, 1.0 - clip
    lo, hi = clip
    return lo, hi


def margin_cdfs(model, Y, *, clip=CDF_CLIP, broadcast: bool = False) -> np.ndarray:
    """``F[n, i, j] = F_ij(y_n)`` for every observation and every pair.

    Parameters
    ----------
    model     : fitted model exposing ``K``, ``margin_structure`` and
                ``margin(i, j)`` (a :class:`~pmcprg.pmc.model.PMCModel`).
    Y         : observations, shape ``(N,)``.
    clip      : the bounds the values are clipped to — a float ``c`` for
                ``[c, 1 − c]``, a pair ``(lo, hi)`` for ``[lo, hi]``, ``None``
                for no clipping. The default is the bound the package's
                diagnostics used.
    broadcast : state margins only. ``False`` (default) returns a writable
                ``(N, K, K)`` array; ``True`` a read-only broadcast view of the
                ``(N, K)`` state CDFs over ``j`` — the same floats without the
                K copies (the GUI's form). Pair margins always return a new
                writable array.

    Returns
    -------
    np.ndarray, shape ``(N, K, K)``, float64.

    For state margins only K CDFs are evaluated and ``F[:, i, j]`` is the
    vector ``F_i(Y)`` for every ``j``; for pair margins K² are evaluated.
    Clipping is element-wise, so clipping the K state vectors before
    broadcasting them gives the floats of clipping the broadcast array.
    """
    Y = np.asarray(Y)
    K = int(model.K)
    N = len(Y)
    bounds = _clip_bounds(clip)
    if is_pair(model):
        F = np.empty((N, K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                F[:, i, j] = model.margin(i, j).cdf_vec(Y)
        if bounds is not None:
            np.clip(F, *bounds, out=F)
        return F
    S = np.empty((N, K), dtype=float)
    for k in range(K):
        S[:, k] = model.margin(k).cdf_vec(Y)
    if bounds is not None:
        np.clip(S, *bounds, out=S)
    F = np.broadcast_to(S[:, :, None], (N, K, K))
    return F if broadcast else F.copy()


def copula_pseudo_obs(F, xi, i: int, j: int, sel=None):
    """Weighted pseudo-observations of the copula ``c_ij`` (A16 Eq. 12).

    Parameters
    ----------
    F   : ``(N, K, K)`` array from :func:`margin_cdfs`.
    xi  : ``(N−1, K, K)`` pair posteriors ``ξ_n(i, j) = P(x_n=i, x_{n+1}=j | y)``
          (or one-hot draws).
    i, j: the pair.
    sel : optional indices ``n`` of the transitions to keep (a thinning
          stride, a window); all ``N−1`` transitions by default.

    Returns
    -------
    uv : ``(M, 2)`` — ``(F_ij(y_n), F_ji(y_{n+1}))``.
    w  : ``(M,)``   — ``ξ_n(i, j)``.
    """
    F = np.asarray(F)
    if sel is None:
        N = F.shape[0]
        uv = np.column_stack((F[: N - 1, i, j], F[1:, j, i]))
        w = np.asarray(xi[:, i, j], dtype=float)
    else:
        sel = np.asarray(sel)
        uv = np.column_stack((F[sel, i, j], F[sel + 1, j, i]))
        w = np.asarray(xi[sel, i, j], dtype=float)
    return uv, w


def margin_pit_dual(F, xi, i: int, j: int):
    """Weighted probability integral transform of the pair margin ``f_ij``.

    Dual view (A16 Eq. 12): ``f_ij`` is the left margin of the pair ``(i, j)``
    and the right margin of the pair ``(j, i)``, so it accounts for

    * ``u = F_ij(y_n)``     with weight ``ξ_n(i, j)``, ``n = 0 … N−2``;
    * ``u = F_ij(y_{n+1})`` with weight ``ξ_n(j, i)``, ``n = 0 … N−2``.

    Under the model, ``P(x_n=i, x_{n+1}=j, y_n ∈ dy) = p_ij f_ij(y) dy`` and
    ``P(x_n=j, x_{n+1}=i, y_{n+1} ∈ dy) = p_ji f_ij(y) dy`` (A16 Eqs. 12–13,
    symmetric ``p``), so the ξ-weighted empirical CDF of ``u`` is uniform in
    expectation when ``f_ij`` is right. Returned as one
    sample of length ``2(N−1)``, left view first. Summed over all pairs the
    weights total ``2(N−1)``: every observation is counted once as a left
    and once as a right observation, except the two ends.

    For a state model (``F[:, i, j] = F_i``) the dual view of ``(i, j)`` is a
    subset of the state-``i`` sample; summing its weights over ``j`` gives
    ``γ_n(i)`` on the left view and ``γ_{n+1}(i)`` on the right one.

    Returns
    -------
    u : ``(2(N−1),)``
    w : ``(2(N−1),)``
    """
    F = np.asarray(F)
    N = F.shape[0]
    u = np.concatenate((F[: N - 1, i, j], F[1:, i, j]))
    w = np.concatenate((np.asarray(xi[:, i, j], dtype=float),
                        np.asarray(xi[:, j, i], dtype=float)))
    return u, w


def margin_sample(Y, labels, key):
    """Observations a margin accounts for under one label path.

    * ``key = k`` (state margin ``f_k``): ``Y[labels == k]``.
    * ``key = (i, j)`` (pair margin ``f_ij``): the hard-label dual view,
      ``{y_n : (x_n, x_{n+1}) = (i, j)}`` followed by
      ``{y_{n+1} : (x_n, x_{n+1}) = (j, i)}``. With one-hot ``ξ`` this is
      exactly the sample :func:`margin_pit_dual` weights.

    ``labels`` is any path — the true labels, a posterior draw, the MPM path.
    """
    Y = np.asarray(Y)
    lab = np.asarray(labels)
    if not isinstance(key, tuple):
        return Y[lab == key]
    i, j = key
    left = (lab[:-1] == i) & (lab[1:] == j)
    right = (lab[:-1] == j) & (lab[1:] == i)
    return np.concatenate((Y[:-1][left], Y[1:][right]))


# ---------------------------------------------------------------------------
# The law of y_n given x_n = i alone
# ---------------------------------------------------------------------------

class StateMixture:
    """g_i = Σ_j w_j f_ij — the law of y_n given x_n = i on a pair model.

    Exposes the subset of the margin interface the diagnostics use
    (``pdf``/``cdf`` and their vectorised forms). A mixture of CDFs is the CDF
    of the mixture, so ``cdf_vec`` is exact, multivariate margins included.
    """

    def __init__(self, components, weights):
        self.components = list(components)
        self.weights    = np.asarray(weights, dtype=float)
        fams = [getattr(c, "dist_name", "?") for c in self.components]
        self.dist_name = mixture_label(fams)
        self.d = getattr(self.components[0], "d", 1) if self.components else 1
        self.is_multivariate = bool(
            getattr(self.components[0], "is_multivariate", False)
        ) if self.components else False

    def _mix(self, method: str, y):
        out = None
        for w, comp in zip(self.weights, self.components):
            if w == 0.0:
                continue
            term = w * np.asarray(getattr(comp, method)(y), dtype=float)
            out = term if out is None else out + term
        if out is None:                                  # all-zero weights
            out = 0.0 * np.asarray(getattr(self.components[0], method)(y),
                                   dtype=float)
        return out

    def pdf_vec(self, y):
        return self._mix("pdf_vec", y)

    def cdf_vec(self, y):
        return self._mix("cdf_vec", y)

    def pdf(self, y) -> float:
        return float(self._mix("pdf", y))

    def cdf(self, y) -> float:
        return float(self._mix("cdf", y))

    def __repr__(self) -> str:
        return f"StateMixture({self.dist_name}, weights={self.weights.tolist()})"


def mixture_label(families) -> str:
    """Short name of a mixture: ``"norm mix"`` or ``"mix(norm+t)"``."""
    uniq = list(dict.fromkeys(str(f) for f in families))
    if len(uniq) == 1:
        return f"{uniq[0]} mix"
    return "mix(" + "+".join(uniq) + ")"


def state_weights(model, i: int) -> np.ndarray:
    """``p(x_{n+1} = j | x_n = i)`` for j = 0…K−1 (uniform if p_i = 0)."""
    K   = int(model.K)
    row = np.asarray(model.transition_A[i], dtype=float)
    s   = float(row.sum())
    if not np.isfinite(s) or s <= 0.0:
        return np.full(K, 1.0 / K)
    return row / s


def state_law(model, i: int):
    """The law of y_n given x_n = i: f_i (state model) or g_i (pair model)."""
    if not is_pair(model):
        return model.margin(i)
    K = int(model.K)
    return StateMixture([model.margin(i, j) for j in range(K)],
                        state_weights(model, i))


def state_law_name(model, i: int) -> str:
    """Family label of :func:`state_law` — the ``dist`` of f_i, or a mixture name."""
    blocks = model.margin_blocks()
    if not is_pair(model):
        for blk in blocks:
            if int(blk.get("i", -1)) == i:
                return str(blk.get("dist", "?"))
        return "?"
    fams = [blk.get("dist", "?") for blk in
            sorted((b for b in blocks if int(b.get("i", -1)) == i),
                   key=lambda b: int(b.get("j", -1)))]
    return mixture_label(fams) if fams else "?"
