"""
pmcprg.copulas._nonparametric — the empirical beta copula (AUDIT_COPULES
FR-9, last remaining item: "une copule non paramétrique de comparaison").

What this is, and why it is not a :class:`~pmcprg.copulas._base.CopulaVirt`
----------------------------------------------------------------------------
Every family in :class:`~pmcprg.copulas._base.CopulaEnum` (Gaussian, the
Archimedean family, BB1/BB6/BB7/BB8, the extreme-value family, …) is
**parametric**: one τ (plus at most a couple of extras), fittable by MLE,
selectable against its siblings by ICE. The empirical beta copula is not
that — it has no parameter to fit and nothing to select against. Given a
bivariate sample, it *is* the nonparametric copula that sample suggests, and
its purpose is to serve as an assumption-free **benchmark**: does a fitted
parametric family's pdf/cdf track what the data alone say? Accordingly it is
**not** registered in :class:`CopulaEnum`, is not wired into ICE's
family-selection machinery, and has no ``fit()`` in the sense the other
families have one — it is built directly from data, the way
:mod:`pmcprg.copulas._robust` (audit FR-7 c) sits beside the registry without
touching it.

The estimator (re-derived, not transcribed)
--------------------------------------------
Segers, Sibuya & Tsukahara (2017, *J. Multivariate Anal.* 155, 35–51,
doi:10.1016/j.jmva.2016.11.010 — verified against Crossref by bibliographic
query, not assumed from the brief) start from the empirical copula of a
sample of size n,

.. math::  C_n(u, v) = \\frac1n \\sum_{i=1}^n
           \\mathbf 1\\{R_{1i} \\le n u\\}\\,\\mathbf 1\\{R_{2i} \\le n v\\},

a step function, and replace each indicator by the CDF of the point i would
occupy under the *order statistics* of a continuous U(0,1) sample: the k-th
order statistic of n i.i.d. U(0,1) draws is Beta(k, n+1−k), so
``1{R ≤ nu}`` is smoothed into ``P(order stat R ≤ u) = I_u(R, n+1−R)``, the
regularised incomplete beta function. Averaging the product over the n data
points gives the **empirical beta copula**

.. math::  C_n^\\beta(u, v) = \\frac1n \\sum_{i=1}^n
           I_u(R_{1i}, n+1-R_{1i})\\; I_v(R_{2i}, n+1-R_{2i}) ,

which is what this module computes. Differentiating once in each argument
(the two factors are independent of each other, so the product rule needs no
cross term) gives the density and the two h-functions in closed form —
derived here and checked against a numerical derivative and an independent
brute-force loop (:mod:`pmcprg.tests.test_empirical_beta_copula`):

.. math::

    c_n^\\beta(u, v)   &= \\frac1n \\sum_i \\beta(u; R_{1i}, n+1-R_{1i})\\,
                            \\beta(v; R_{2i}, n+1-R_{2i}) ,\\\\
    h_1(v \\mid u)      &= \\partial_u C_n^\\beta
                         = \\frac1n \\sum_i \\beta(u; R_{1i}, n+1-R_{1i})\\,
                            I_v(R_{2i}, n+1-R_{2i}) ,\\\\
    h_2(u \\mid v)      &= \\partial_v C_n^\\beta
                         = \\frac1n \\sum_i I_u(R_{1i}, n+1-R_{1i})\\,
                            \\beta(v; R_{2i}, n+1-R_{2i}) ,

with ``β(·; a, b)`` the Beta(a, b) density and ``I_x(a, b)`` its regularised
incomplete beta CDF (``scipy.special.betainc`` / ``scipy.stats.beta``, the
same primitives :mod:`pmcprg.copulas.elliptical.student` and
:mod:`pmcprg.copulas.extreme_value.t_ev` already use via ``betaln`` — no new
special-function dependency is introduced).

R's ``copula`` package computes the same object under ``C.n(..., smoothing =
"beta")`` / the ``empCopula``/``betacopula`` family; its documentation
(consulted, not accessed as source code here) describes the identical
average-of-Beta-CDF construction, which is the second, independent check the
brief asks for beyond the paper itself.

Ranks and ties
---------------
The paper assumes continuous margins, hence a.s. distinct ranks in
``{1, …, n}``. This package's own rank-based code breaks ties by *averaging*
— :meth:`CopulaVirt.fit(method='tau')` builds pseudo-observations with
``scipy.stats.rankdata(data) / (n + 1)`` (default ``method="average"``,
:mod:`pmcprg.copulas._base`), and the weighted-ECDF convention of
:func:`pmcprg.pmc.ice._weighted_ecdf` gives a tied block a single shared
value at its upper end. This module follows the **average-rank** convention
of ``_base.py`` (rather than the "shared upper end" of ``_weighted_ecdf``,
which is a *weighted* construction this module has no counterpart of): ties
in a coordinate get the mean of the ranks they span, e.g. two tied smallest
values both get rank 1.5 instead of one 1 and one 2. Because the Beta CDF
and density are continuous in their shape parameters, this simply gives the
tied points the *same* Beta(1.5, n−0.5) marginal shape instead of picking an
arbitrary order between them — a smooth relaxation, not a special case, and
it costs nothing: ranking a set of *raw data* or of *pseudo-observations*
(any monotone rescaling of the data) gives identical results, since rank is
invariant under a strictly increasing transform of each coordinate
separately. :func:`EmpiricalBetaCopula.from_data` therefore accepts either.

Numerical stability — log-space for the density
--------------------------------------------------
``cdf`` and the h-functions average n terms each in [0, 1]: summing n such
bounded terms and dividing by n cannot overflow, and the only loss is the
ordinary O(√n) rounding error of a mean, so they are computed directly in
linear space.

The **density** is different. A single term ``β(u; R_{1i}, n+1−R_{1i})`` is
a Beta density evaluated away from its own mode when point i's rank is far
from ``(u, v)``, and for a moderately large n that term underflows to an
exact ``0.0`` in float64 long before the true value is negligible: e.g. for
n = 500 and u = 0.01, the term for a point ranked at the top (R₁ = 500) has
``β(0.01; 500, 1) = 500·0.01^{499}``, i.e. ``log`` ≈ −2295 — far below the
smallest representable double (``exp(x)`` underflows to 0 for x ≲ −745). If
*every* term at a given (u, v) is this far out — plausible deep in a corner
with a sample concentrated elsewhere — a naive ``mean(exp(logpdf_i))``
returns exactly 0.0 and ``log`` of that is ``-inf``, even though the true
log-density (around −2295 − log n, itself un-representable in linear scale
but perfectly meaningful in log scale) is a well-defined, informative
number for a diagnostic that only ever needs the log (e.g. a
cross-entropy/KL comparison against a fitted parametric family — exactly
this module's purpose). :func:`EmpiricalBetaCopula.logpdf_array` therefore
computes ``scipy.special.logsumexp`` of the n per-term log-densities minus
``log n`` — the maximum-subtraction trick avoids the intermediate underflow
entirely and is exact down to the point where the *true* log-density itself
exceeds float64's exponent range, which no algorithm can fix. Following the
house convention documented in :class:`~pmcprg.copulas._base.CopulaVirt`
("a subclass with a native ``logpdf_array`` gets ``pdf_array =
exp(logpdf_array)``"), :meth:`pdf_array` is implemented as
``exp(logpdf_array)`` rather than averaging linear-space terms directly, so
the two can never disagree and the same protection carries over to the
non-log density (it still floors to 0.0 in the same regime — a
representable-precision limit, not a bug).

What "exact" means here — checked, and better than expected
----------------------------------------------------------------
``C_n^β`` is an **estimator** built from n points, not an exact copula in
general, but its margins are a genuine exception, not merely "close for
large n". Setting v = 1 (``I_1(a, b) = 1`` identically),

.. math::  C_n^\\beta(u, 1) = \\frac1n \\sum_{i=1}^n I_u(R_{1i}, n+1-R_{1i}).

**Without ties**, ``{R_{1i}}`` is a permutation of ``{1, …, n}``, so this is
``\\frac1n\\sum_{k=1}^n I_u(k, n+1-k)``, a sum that no longer depends on the
data at all — and it equals ``u`` **exactly** (not just in expectation): the
CDF of the k-th order statistic of n i.i.d. U(0,1) draws is ``I_u(k,
n+1-k)``, and the sum, over k, of "is the k-th order statistic ≤ u" is
*identically* the count of the n points that fall in [0, u], whose sum over
all n has expectation ``nu`` — but because every term ``I_u(k, n+1-k)`` is a
fixed function of u alone, this expectation identity is in fact a
**deterministic identity of the regularised incomplete beta function**,
``\\sum_{k=1}^n I_u(k, n+1-k) \\equiv nu``, true for every n and u ∈ [0, 1]
regardless of any sample. Checked numerically to float64 rounding (≤
4·10⁻¹⁵) for n ∈ {3, 5, 20, 100} on a grid of u
(:func:`pmcprg.tests.test_empirical_beta_copula.test_margin_is_exactly_uniform_without_ties`)
— this was **derived and then verified**, not assumed, and is the actual
reason the "empirical beta copula" deserves the name "copula": unlike the
plain empirical copula (a step function, margins only asymptotically
uniform), its beta-smoothed version has *exactly* uniform margins whenever
there are no ties.

**With ties** (average, non-integer ranks) the permutation argument breaks
— ``{R_{1i}}`` is no longer ``{1, …, n}`` — and a small, quantifiable
discrepancy appears: e.g. one tied pair out of n = 6 gives a sup
discrepancy of 6.7·10⁻³, and the same single tied pair diluted into larger
samples (n = 20, 60, 200, 600) gives 6.0·10⁻⁴, 6.7·10⁻⁵, 6.0·10⁻⁶, 6.7·10⁻⁷
— shrinking roughly like ``n⁻²`` as the one non-generic pair is diluted
among more order statistics (measured, not derived analytically, in
``test_margin_uniformity_with_ties_shrinks_with_n``). This is reported as
what it is: a small, ties-only, empirically-shrinking discrepancy — not
hidden behind a blanket "asymptotically uniform" claim that would also be
true but far less informative than the exact identity above.

Sampling
--------
Segers, Sibuya & Tsukahara's estimator is itself a finite mixture over the n
data points, each with weight 1/n and independent Beta margins conditional
on the chosen point — so it can be sampled *exactly* (not by Rosenblatt
inversion, unlike every family in :mod:`pmcprg.copulas._base`): draw an
index I ~ Uniform{1, …, n}, then ``U ~ Beta(R_{1I}, n+1-R_{1I})`` and
``V ~ Beta(R_{2I}, n+1-R_{2I})`` independently. :meth:`sample` implements
exactly this two-stage draw; its agreement with :meth:`cdf_array` is checked
by simulation in the tests, with a Monte-Carlo error bound stated from the
sample size rather than an arbitrary tolerance.

Scope deliberately left out
------------------------------
* **Weights.** Unlike :mod:`pmcprg.pmc.ice`'s ``_weighted_ecdf``/rank-MLE
  machinery, this first version takes an unweighted sample only — a
  weighted empirical beta copula (replacing 1/n by ``w_i / Σw`` and the rank
  by a weighted rank) is a natural extension but is not needed by any
  current caller (this module is not wired into ICE) and is left for when
  one exists, rather than guessed at.
* **h-function inverses** (``inv_h``/``inv_h_array``, i.e. sampling by
  Rosenblatt inversion). Not needed: the exact two-stage mixture sampler
  above is both simpler and exact, so no root-finding on ``h`` is added.
* **Multivariate (d > 2) version.** The paper and this module are bivariate,
  matching the rest of :mod:`pmcprg.copulas`.

References
----------
* Segers, J., Sibuya, M. & Tsukahara, H. (2017). The empirical beta copula.
  *J. Multivariate Anal.* 155, 35–51. doi:10.1016/j.jmva.2016.11.010
* Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). A semiparametric estimation
  procedure of dependence parameters in multivariate families of
  distributions. *Biometrika* 82(3), 543–552. — the rank/(n+1)
  pseudo-observation convention this module's rank step matches.
"""

from __future__ import annotations

import numpy as np
from scipy.special import betainc, logsumexp
from scipy.stats import beta as _beta_dist
from scipy.stats import rankdata


class EmpiricalBetaCopula:
    """The empirical beta copula of Segers, Sibuya & Tsukahara (2017).

    A standalone nonparametric comparison tool (AUDIT_COPULES FR-9) — not a
    :class:`~pmcprg.copulas._base.CopulaVirt` family: it has no parameter,
    is not in :class:`~pmcprg.copulas._base.CopulaEnum`, and is not used by
    ICE's family selection. See the module docstring for the full
    derivation, the tie convention, and the numerical-stability discussion.

    Parameters
    ----------
    data : (n, 2) array_like
        Either raw bivariate observations or pseudo-observations already in
        [0, 1] — ranking is invariant to any strictly increasing per-column
        rescaling, so both give identical results. n ≥ 2 required (n ≥ 1
        gives ranks a1 = a2 = 1, b1 = b2 = 1, i.e. the independence copula,
        which is degenerate as an *estimate* but not undefined; n ≥ 2 is
        required here so the estimator's own uniform-margin discrepancy —
        see the module docstring — is at least meaningful to quote).

    Attributes
    ----------
    n : int
        Sample size.
    a1, b1, a2, b2 : (n,) ndarray
        Beta shape parameters ``(R_i, n + 1 - R_i)`` for each coordinate;
        ``a1 + b1 == a2 + b2 == n + 1`` exactly.
    """

    def __init__(self, data) -> None:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError(f"data must be shape (n, 2); got {data.shape}.")
        n = data.shape[0]
        if n < 2:
            raise ValueError(f"at least 2 observations are required; got {n}.")
        self.n = n
        self.a1 = rankdata(data[:, 0], method="average")
        self.a2 = rankdata(data[:, 1], method="average")
        self.b1 = (n + 1) - self.a1
        self.b2 = (n + 1) - self.a2

    # Alias documenting that raw data and pseudo-observations are
    # interchangeable inputs (see module docstring, "Ranks and ties").
    from_data = classmethod(lambda cls, data: cls(data))
    from_pseudo_obs = classmethod(lambda cls, uv: cls(uv))

    # ------------------------------------------------------------------
    # cdf
    # ------------------------------------------------------------------
    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """``C_n^β(u, v)`` at M point pairs — shape (M, 2) in, (M,) out."""
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"cdf_array expects shape (M, 2); got {uv.shape}.")
        u = uv[:, 0][:, None]           # (M, 1)
        v = uv[:, 1][:, None]
        Fu = betainc(self.a1[None, :], self.b1[None, :], u)   # (M, n)
        Fv = betainc(self.a2[None, :], self.b2[None, :], v)
        return (Fu * Fv).mean(axis=1)

    def cdf(self, u: float, v: float) -> float:
        """Scalar convenience wrapper around :meth:`cdf_array`."""
        return float(self.cdf_array(np.array([[u, v]]))[0])

    # ------------------------------------------------------------------
    # density — log-space (see module docstring)
    # ------------------------------------------------------------------
    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """``log c_n^β(u, v)`` at M point pairs, via ``logsumexp`` over the
        n per-point terms (not floored; ``-inf`` where the true value
        underflows float64's range — see module docstring)."""
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"logpdf_array expects shape (M, 2); got {uv.shape}.")
        u = uv[:, 0][:, None]
        v = uv[:, 1][:, None]
        log_terms = (_beta_dist.logpdf(u, self.a1[None, :], self.b1[None, :])
                     + _beta_dist.logpdf(v, self.a2[None, :], self.b2[None, :]))
        return logsumexp(log_terms, axis=1) - np.log(self.n)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """``c_n^β(u, v) = exp(logpdf_array(uv))`` — see module docstring
        for why this is derived from the log form rather than the other
        way round (house convention, :class:`CopulaVirt`)."""
        return np.exp(self.logpdf_array(uv))

    def pdf(self, u: float, v: float) -> float:
        return float(self.pdf_array(np.array([[u, v]]))[0])

    def logpdf(self, u: float, v: float) -> float:
        return float(self.logpdf_array(np.array([[u, v]]))[0])

    # ------------------------------------------------------------------
    # h-functions (conditional cdfs) — closed form, see module docstring
    # ------------------------------------------------------------------
    def h1_array(self, uv: np.ndarray) -> np.ndarray:
        """``h_1(v | u) = ∂C_n^β/∂u`` — the conditional cdf of V given U=u."""
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"h1_array expects shape (M, 2); got {uv.shape}.")
        u = uv[:, 0][:, None]
        v = uv[:, 1][:, None]
        fu = _beta_dist.pdf(u, self.a1[None, :], self.b1[None, :])
        Fv = betainc(self.a2[None, :], self.b2[None, :], v)
        return (fu * Fv).mean(axis=1)

    def h2_array(self, uv: np.ndarray) -> np.ndarray:
        """``h_2(u | v) = ∂C_n^β/∂v`` — the conditional cdf of U given V=v."""
        uv = np.asarray(uv, dtype=float)
        if uv.ndim != 2 or uv.shape[1] != 2:
            raise ValueError(f"h2_array expects shape (M, 2); got {uv.shape}.")
        u = uv[:, 0][:, None]
        v = uv[:, 1][:, None]
        Fu = betainc(self.a1[None, :], self.b1[None, :], u)
        fv = _beta_dist.pdf(v, self.a2[None, :], self.b2[None, :])
        return (Fu * fv).mean(axis=1)

    def h1(self, u: float, v: float) -> float:
        return float(self.h1_array(np.array([[u, v]]))[0])

    def h2(self, u: float, v: float) -> float:
        return float(self.h2_array(np.array([[u, v]]))[0])

    # ------------------------------------------------------------------
    # sampling — exact two-stage mixture (see module docstring)
    # ------------------------------------------------------------------
    def sample(self, n: int = 500, seed: int | None = None) -> np.ndarray:
        """Draw n samples from the mixture ``C_n^β`` exactly represents.

        For each draw: pick a data index I ~ Uniform{0, …, n_data − 1}, then
        draw U ~ Beta(a1[I], b1[I]) and V ~ Beta(a2[I], b2[I]) independently
        (no Rosenblatt inversion needed — see module docstring).
        """
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.n, size=n)
        u = rng.beta(self.a1[idx], self.b1[idx])
        v = rng.beta(self.a2[idx], self.b2[idx])
        return np.column_stack([u, v])
