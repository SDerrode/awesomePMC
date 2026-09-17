"""
dependent_multiplier.py — serially dependent multipliers (audit FR-5).

Why this exists
---------------
Two consecutive pairs of a Markov chain share an observation: the pair
``(y_n, y_{n+1})`` and the pair ``(y_{n+1}, y_{n+2})`` have ``y_{n+1}`` in
common, so the pseudo-observations a state pair's copula is tested on are
**not i.i.d.** (Darsow, Nguyen & Olsen 1992, doi:10.1215/ijm/1255987328;
Chen & Fan 2006, doi:10.1016/j.jeconom.2005.03.004). Every bootstrap in this
package that resamples *within* such a sample — the i.i.d. multiplier
bootstraps of :mod:`pmcprg.diagnostics.radial_symmetry`,
:mod:`pmcprg.diagnostics.exchangeability` and
:mod:`pmcprg.diagnostics.rosenblatt` — reproduces only the *short-run*
variance of the empirical process, and is therefore calibrated against a null
that is too narrow. Fermanian, Radulović & Wegkamp (2004,
doi:10.3150/bj/1099579158) state the i.i.d. requirement; the remedy FR-5 asks
for is the **dependent multiplier bootstrap** (Bücher & Ruppert 2013,
doi:10.1016/j.jmva.2012.12.002; Bücher & Kojadinovic 2016,
doi:10.3150/14-BEJ682).

This module provides only the multiplier *sequence*. The three tests above
plug it into their existing linearisations, unchanged otherwise, through
``bootstrap="dependent-multiplier"``.

The construction
----------------
An i.i.d. multiplier bootstrap draws ``ξ_1, …, ξ_n`` i.i.d., mean 0, variance
1, and replaces the empirical process by ``n^{-1/2} Σ_i ξ_i g_i``. Its
conditional variance is ``n^{-1} Σ_i g_i²`` — the i.i.d. variance of the
series ``g``. Under serial dependence the quantity to reproduce is the
**long-run** variance ``n^{-1} Σ_{i,j} Cov(g_i, g_j)`` instead.

A *dependent* multiplier sequence is obtained by smoothing i.i.d. variates
with a kernel of bandwidth ``ℓ`` (Bücher & Kojadinovic's construction; the
mechanism is Bühlmann's 1993 dependent wild bootstrap):

    ξ_i = Σ_{k=0}^{m-1} w_k Z_{i+k},        Z_1, …, Z_{n+m-1} i.i.d. (0, 1).

The weights are the kernel sampled at ``k/ℓ`` and normalised by
``Σ_k w_k² = 1``, which is exactly what keeps ``Var(ξ_i) = 1``; because
``m − 1`` extra ``Z``'s are drawn, the sequence is *exactly* stationary (no
edge correction), with

    Corr(ξ_i, ξ_{i+h}) = Σ_k w_k w_{k+h}.

**Deriving the normalisation** is the whole of it: ``Var(ξ_i) = Σ_k w_k²``
since the ``Z`` are uncorrelated with unit variance, so dividing the raw
kernel values by ``(Σ_k φ(k/ℓ)²)^{1/2}`` — not by ``Σ_k φ(k/ℓ)``, the
normalisation a *smoother* would use — is the one that gives unit variance.
Both statements are checked numerically in
``pmcprg/tests/test_dependent_multiplier.py``, on generated sequences rather
than on paper.

Two kernels, and why ``"bartlett"`` is the default
--------------------------------------------------
``kernel="bartlett"`` uses the **rectangular** window ``w_k = ℓ^{-1/2}``,
``k = 0 … ℓ-1`` — i.e. a plain moving average of ``ℓ`` consecutive i.i.d.
variates, rescaled. Its autocorrelation is the self-convolution of that
window, which is *exactly* the Bartlett kernel:

    ρ(h) = (ℓ − |h|)/ℓ = 1 − |h|/ℓ     for |h| < ℓ,   0 otherwise,

so the sequence is exactly ``ℓ``-dependent. (This is why the name refers to
the *autocorrelation*, not to the window: a triangular window would give a
smoother, longer-ranged autocorrelation, not a triangular one.)

That choice is not cosmetic — it makes this module's ``ℓ`` and the HAC
bandwidth ``L`` of :mod:`pmcprg.diagnostics.model_selection` **the same
quantity**. For any fixed series ``a_1, …, a_n``,

    Var(Σ_i a_i ξ_i) = Σ_{|h| < ℓ} (1 − |h|/ℓ) Σ_i a_i a_{i+h}
                     = hac_variance(a, L)      with  L = ℓ − 1,

because ``model_selection.hac_variance`` uses Bartlett weights
``k(h) = 1 − |h|/(L + 1)``. The dependent multiplier bootstrap with this
kernel therefore reproduces, *exactly and by construction*, the Newey–West
(1987) long-run variance the package already uses for the Vuong/Clarke
statistics — the equality is asserted in the test module, not just claimed
here. It also means ``ℓ = 1`` is the i.i.d. bootstrap (``L = 0``), bit for
bit: ``m = 1``, ``w_0 = 1`` and ``ξ = Z``, the very array the i.i.d. path
draws.

``kernel="parzen"`` uses the Parzen kernel on ``k/ℓ``, ``|k| ≤ ℓ-1``
(``m = 2ℓ-1`` terms), normalised the same way. Its autocorrelation is
smoother but reaches ``2ℓ - 2``, so ``ℓ`` is *not* the dependence length
there; the automatic rule below rescales it accordingly. Offered because
FR-5's remedy is stated for a kernel-smoothed multiplier in general, and a
smoother kernel is the usual robustness check on the block width.

Choosing ``ℓ``
--------------
:func:`auto_block_length` reuses
:func:`pmcprg.diagnostics.model_selection.newey_west_bandwidth` — the Newey &
West (1994) plug-in already in this package — rather than inventing a second
rule, which the identity above makes legitimate: selecting ``L`` for the
Bartlett HAC variance of a series *is* selecting ``ℓ = L + 1`` for this
construction.

One extension is needed, and it is the only place this departs from the HAC
precedent. Newey–West selects ``L`` for **one scalar series**; here the
bootstrap must be valid for a whole empirical process ``{g_i(u, v)}`` indexed
by ``(u, v)``, and a single ``ℓ`` has to serve all of it. The rule taken is
the **median** over a coarse 3×3 quantile grid of the centred indicator
series ``1{u_i ≤ a, v_i ≤ b} − C_n(a, b)`` — the actual summands of the
empirical copula process at those points. Two reasons for indicators rather
than the ranks themselves: they are what the process is built from, and they
see the serial dependence a rank autocorrelation misses (a copula Markov
chain with tail dependence can have near-zero lag-1 rank correlation and
strongly dependent joint exceedances — Beare 2010, doi:10.3982/ECTA8152, on
such chains being ρ-mixing but not more).

Median and not maximum, which was the first thing tried: the Newey–West
plug-in has enough sampling noise of its own that the maximum over nine
series is essentially all noise. Measured over 60 replicates of a Gaussian
copula at τ = 0.5, i.i.d. pairs vs. the corresponding Markov chain's
consecutive pairs (``n = 200``): the maximum averages ``L = 9.1`` on i.i.d.
data and ``8.6`` on the chain — no discrimination at all — while the median
averages ``4.2`` and ``7.8``, and at ``n = 800``, ``6.1`` and ``14.7``.

**Honest about the plug-in's bias.** Those numbers also show that the rule
does *not* return ``ℓ = 1`` on i.i.d. data: it returns ``ℓ ≈ 5`` at
``n = 200``. That is the Newey–West plug-in's own well-known upward bias when
the true autocovariances vanish — ``γ̂ ∝ (ŝ₁/ŝ₀)^{2/3}`` is built from a
squared quantity, so estimation noise cannot cancel — and it is inherited, not
introduced here; :mod:`pmcprg.diagnostics.model_selection` uses the same rule
with the same property. The direction is the safe one: with Bartlett weights
the extra lag terms have mean ≈ 0 under independence, so an ``ℓ`` that is too
large makes the bootstrap null noisier and mildly *wider* — conservative
size, some lost power — whereas an ``ℓ`` that is too small leaves exactly the
anticonservative test FR-5 is about. A caller who knows the sample is i.i.d.
should pass ``bootstrap="multiplier"`` (or ``block_length=1``) rather than
ask the plug-in to discover it.

``ℓ`` is capped at ``n // 4``: beyond that the long-run variance estimate is
dominated by a handful of near-independent blocks.

What is verified, and what is assumed
--------------------------------------
Verified numerically (``pmcprg/tests/test_dependent_multiplier.py``), in the
posture every FR-9/FR-10 round of this audit has taken:

* the generated ``ξ`` have mean 0 and variance 1, and their sample
  autocorrelation matches ``Corr(ξ_i, ξ_{i+h}) = Σ_k w_k w_{k+h}``;
* for ``kernel="bartlett"`` that autocorrelation is the Bartlett kernel, and
  ``Var(Σ a_i ξ_i)`` equals ``hac_variance(a, ℓ-1)`` to Monte-Carlo error;
* ``ℓ = 1`` reproduces the existing i.i.d. multiplier bootstrap **exactly**
  (identical arrays, identical p-values);
* the size and power consequences on genuinely serially dependent data
  simulated from a fitted PMC — see ``test_fr5_dependent_multiplier.py``.

Assumed, and *not* claimed to be a transcription: that this is Bücher &
Kojadinovic's (2016) multiplier sequence in its published form. Neither paper
could be consulted from this worktree (offline), so what is implemented is
the construction derived above from first principles — i.i.d. variates
smoothed by a kernel, normalised for unit variance — which is the
construction those papers are cited for, with the normalisation and the
autocorrelation worked out here and checked by simulation. The *tests* it
feeds also inherit their own documented simplifications (no
parameter-estimation correction in :mod:`pmcprg.diagnostics.rosenblatt`, a
hand-derived linearisation in the other two); making the multipliers
serially dependent fixes the i.i.d.-data assumption, not those.

References
----------
* Bücher, A. & Kojadinovic, I. (2016). A dependent multiplier bootstrap for
  the sequential empirical copula process under strong mixing. *Bernoulli*
  22(2), 927–968. doi:10.3150/14-BEJ682
* Bücher, A. & Ruppert, M. (2013). Consistent testing for a constant copula
  under strong mixing based on the tapered block multiplier technique.
  *J. Multivariate Anal.* 116, 208–229. doi:10.1016/j.jmva.2012.12.002
* Bühlmann, P. (1993). *The Blockwise Bootstrap in Time Series and Empirical
  Processes*. PhD thesis, ETH Zürich — the moving-average smoothing of
  i.i.d. variates this construction is an instance of.
* Fermanian, J.-D., Radulović, D. & Wegkamp, M. (2004). Weak convergence of
  empirical copula processes. *Bernoulli* 10(5), 847–860.
  doi:10.3150/bj/1099579158 — the i.i.d. assumption being relaxed.
* Newey, W. K. & West, K. D. (1994). Automatic lag selection in covariance
  matrix estimation. *Rev. Econom. Stud.* 61(4), 631–653 — the bandwidth
  rule reused for ``ℓ``.
* Beare, B. K. (2010). Copulas and temporal dependence. *Econometrica* 78(1),
  395–410. doi:10.3982/ECTA8152 — why the mixing coefficients of a
  tail-dependent copula chain do not decay as fast as one would like.
"""
from __future__ import annotations

import logging
import math

import numpy as np

from pmcprg.diagnostics.model_selection import newey_west_bandwidth

logger = logging.getLogger(__name__)

__all__ = [
    "MULTIPLIER_KERNELS",
    "MULTIPLIER_LAWS",
    "auto_block_length",
    "draw_multipliers",
    "multiplier_autocorrelation",
    "multiplier_weights",
    "resolve_block_length",
]

#: Multiplier laws — i.i.d., mean 0, variance 1, as the multiplier CLT
#: requires of the *underlying* variates ``Z``. Shared with the three tests.
MULTIPLIER_LAWS: tuple[str, ...] = ("normal", "rademacher")

#: Smoothing kernels for the dependent multiplier sequence. ``"bartlett"``
#: (the rectangular window whose autocorrelation is the Bartlett kernel) is
#: the default — see the module docstring on why it ties ``ℓ`` to the HAC
#: bandwidth exactly.
MULTIPLIER_KERNELS: tuple[str, ...] = ("bartlett", "parzen")

#: ``ℓ`` is capped at ``n // BLOCK_CAP_FRACTION`` by :func:`auto_block_length`.
BLOCK_CAP_FRACTION = 4


def _parzen(x: np.ndarray) -> np.ndarray:
    """Parzen kernel on ``[-1, 1]``, 0 outside."""
    a = np.abs(np.asarray(x, dtype=float))
    out = np.where(a <= 0.5, 1.0 - 6.0 * a ** 2 + 6.0 * a ** 3, 2.0 * (1.0 - a) ** 3)
    return np.where(a <= 1.0, out, 0.0)


def multiplier_weights(block_length: int, kernel: str = "bartlett") -> np.ndarray:
    """Weights ``w`` of the moving average, normalised so ``Σ w_k² = 1``.

    ``block_length = 1`` returns ``array([1.0])`` for every kernel, which is
    what makes the dependent bootstrap reduce to the i.i.d. one exactly.

    ``"bartlett"``: ``ℓ`` equal weights ``ℓ^{-1/2}`` (a rectangular window,
    whose self-convolution *is* the Bartlett kernel — module docstring).
    ``"parzen"``: ``2ℓ-1`` weights ``∝ Parzen(k/ℓ)``, ``|k| ≤ ℓ-1``.
    """
    if isinstance(block_length, str) or isinstance(block_length, (bool, np.bool_)) \
            or not float(block_length).is_integer():
        raise ValueError(f"block_length must be a positive int, got {block_length!r}.")
    ell = int(block_length)
    if ell < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length!r}.")
    if kernel not in MULTIPLIER_KERNELS:
        raise ValueError(f"kernel must be one of {MULTIPLIER_KERNELS}, got {kernel!r}.")
    if ell == 1:
        return np.ones(1, dtype=float)
    if kernel == "bartlett":
        return np.full(ell, 1.0 / math.sqrt(ell), dtype=float)
    raw = _parzen(np.arange(-(ell - 1), ell, dtype=float) / ell)
    return raw / math.sqrt(float(np.dot(raw, raw)))


def multiplier_autocorrelation(block_length: int, kernel: str = "bartlett") -> np.ndarray:
    """``[ρ(0), ρ(1), …]`` of the sequence, ``ρ(h) = Σ_k w_k w_{k+h}``.

    Returned for ``h = 0 … m-1`` (``m = len(w)``); ``ρ(h) = 0`` beyond. For
    ``kernel="bartlett"`` this is exactly ``1 − h/ℓ``.
    """
    w = multiplier_weights(block_length, kernel)
    m = w.size
    return np.array([float(np.dot(w[: m - h], w[h:])) for h in range(m)])


def _draw_iid(n: int, B: int, rng: np.random.Generator, law: str) -> np.ndarray:
    """``(n, B)`` i.i.d. mean-0, variance-1 variates.

    Byte-for-byte the draw the three tests' own i.i.d. multiplier path made
    before FR-5; keeping the call identical is what keeps every existing
    default unchanged.
    """
    if law == "normal":
        return rng.standard_normal(size=(n, B))
    if law == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=(n, B))
    raise ValueError(f"multiplier must be one of {MULTIPLIER_LAWS}, got {law!r}.")


def draw_multipliers(
    n: int,
    B: int,
    rng: np.random.Generator,
    law: str = "normal",
    *,
    block_length: int = 1,
    kernel: str = "bartlett",
) -> np.ndarray:
    """``(n, B)`` multipliers, mean 0, variance 1, ``ℓ``-dependent in ``n``.

    Parameters
    ----------
    n, B         : sequence length and number of bootstrap replicates; the
                   ``B`` columns are independent replicates, the ``n`` rows
                   carry the serial dependence.
    rng          : generator. ``n + m - 1`` variates per column are drawn
                   (``m = 1`` when ``block_length = 1``), so the sequence is
                   exactly stationary rather than edge-corrected.
    law          : ``'normal'`` (default) or ``'rademacher'`` — the law of
                   the *underlying* i.i.d. ``Z``. After smoothing with
                   ``ℓ > 1`` the ``ξ`` are no longer ``±1`` in the
                   Rademacher case; they remain mean 0, variance 1, which is
                   all the multiplier CLT asks of them.
    block_length : ``ℓ ≥ 1``. ``1`` returns the i.i.d. draw itself.
    kernel       : ``'bartlett'`` (default) or ``'parzen'``.
    """
    if int(n) < 0 or int(B) < 0:
        raise ValueError(f"n and B must be >= 0, got n={n!r}, B={B!r}.")
    w = multiplier_weights(block_length, kernel)
    m = w.size
    z = _draw_iid(int(n) + m - 1, int(B), rng, law)
    if m == 1:
        # w[0] == 1.0 exactly: return ``z`` untouched so that ``block_length=1``
        # is bit-identical to the pre-FR-5 i.i.d. path, not merely equal to it.
        return z
    xi = np.zeros((int(n), int(B)), dtype=float)
    for k in range(m):
        xi += w[k] * z[k : k + int(n), :]
    return xi


def auto_block_length(
    u: np.ndarray,
    v: np.ndarray,
    *,
    kernel: str = "bartlett",
    grid: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_block: int | None = None,
) -> int:
    """Newey & West (1994) plug-in ``ℓ`` for a bivariate sequence in ``(0,1)²``.

    ``ℓ = 1 + median_{(a,b) ∈ grid²} L_NW(1{u_i ≤ a, v_i ≤ b} − C_n(a, b))``
    for ``kernel='bartlett'``, where ``L_NW`` is
    :func:`pmcprg.diagnostics.model_selection.newey_west_bandwidth` — the
    package's existing HAC bandwidth rule, reused rather than duplicated
    because ``ℓ = L + 1`` is an exact identity for this kernel (module
    docstring). For ``kernel='parzen'`` the same target dependence length
    ``L + 1`` is converted to that kernel's ``ℓ``, whose autocorrelation
    reaches ``2ℓ - 2``: ``ℓ = ⌈(L + 2)/2⌉``.

    ``u, v`` must be in **time order** — the whole point is that row ``i`` is
    observation ``i`` of the chain. Returns at least 1, at most
    ``max_block`` (default ``n // 4``).
    """
    if kernel not in MULTIPLIER_KERNELS:
        raise ValueError(f"kernel must be one of {MULTIPLIER_KERNELS}, got {kernel!r}.")
    u = np.asarray(u, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    n = u.size
    if n != v.size:
        raise ValueError(f"u and v must be same-sized; got {u.size} vs {v.size}.")
    cap = int(max_block) if max_block is not None else max(1, n // BLOCK_CAP_FRACTION)
    if n < 3 or cap <= 1:
        return 1
    per_point = []
    for a in grid:
        for b in grid:
            ind = ((u <= a) & (v <= b)).astype(float)
            z = ind - ind.mean()
            per_point.append(int(newey_west_bandwidth(z)) if np.any(z) else 0)
    # Median, not maximum — see the module docstring for the measurement that
    # settled it: over nine series the plug-in's own noise swamps the maximum.
    L = int(np.median(per_point)) if per_point else 0
    if kernel == "bartlett":
        ell = L + 1
    else:
        ell = int(math.ceil((L + 2) / 2.0))
    ell = int(max(1, min(ell, cap)))
    logger.debug("auto_block_length: n=%d kernel=%s L_NW=%d -> ell=%d", n, kernel, L, ell)
    return ell


def resolve_block_length(
    block_length: int | str,
    u: np.ndarray,
    v: np.ndarray,
    kernel: str = "bartlett",
) -> int:
    """``'auto'`` -> :func:`auto_block_length`; an int is validated and returned."""
    if isinstance(block_length, str):
        if block_length != "auto":
            raise ValueError(
                f"block_length must be a positive int or 'auto', got {block_length!r}."
            )
        return auto_block_length(u, v, kernel=kernel)
    if isinstance(block_length, (bool, np.bool_)) or not float(block_length).is_integer():
        raise ValueError(f"block_length must be a positive int or 'auto', got {block_length!r}.")
    ell = int(block_length)
    if ell < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length!r}.")
    return ell
