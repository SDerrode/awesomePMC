"""
smooth.py — Neyman smooth-test components for a fitted margin.

What they are
-------------
Push a sample through the probability integral transform of the distribution
it is supposed to follow, ``u = F(y)``. Under H₀ the ``u`` are uniform, so any
departure shows as a departure from uniformity. Project that onto orthonormal
Legendre polynomials::

    V_j = n^{-1/2} Σ h_j(u_i)

Each ``V_j`` is asymptotically ``N(0, 1)`` under a fully specified null;
with an estimated margin the null law changes (Kallenberg & Ledwina 1997),
which is why the package calibrates them by
:func:`~pmcprg.diagnostics.bootstrap.parametric_bootstrap`. Each answers a
different question — ``V₁`` location, ``V₂`` dispersion, ``V₃`` asymmetry,
``V₄`` tail weight — the mean/variance/skewness/kurtosis reading of Rayner &
Best (1989).

Why these and not an EDF statistic
----------------------------------
Measured against five alternatives, 200 datasets each, all calibrated by the
same bootstrap so the comparison is purely about power
(``report/margin_statistic_panel.py``): rejection of a wrong margin goes from
0.098 to 0.338 against Student-t tails and from 0.152 to 0.480 against
skew-normal asymmetry, where the multivariate KS test the package shipped
with sat near the nominal level.

The reason is *not* that a component has a tighter null. On i.i.d. uniforms
``V²`` has an interquartile range 2.68 times its median against 0.95 for
Anderson-Darling — nearly three times wider. Relative dispersion compares
statistics inside the EDF family and does not transfer across families.
What governs power is the non-centrality, and a component puts the whole
deviation of one kind into a single degree of freedom where Cramér-von Mises
and Anderson-Darling dilute it across an integral over the entire support.

The multiplicity is real and must be paid
-----------------------------------------
Four components is four tests. Taking the smallest p-value and reporting it
raises the level from 0.05 to **0.175** — measured, not feared. A Bonferroni
factor of the number of components restores it (0.052 measured) and still
beats every alternative on the table, including Neyman's own omnibus
``Σ V_j²``, because the components are near-orthogonal under H₀ and the
correction is therefore close to exact rather than conservative.

:func:`neyman_test` applies that correction. Its ``component`` field names
where the deviation lives, which a single p-value cannot: a margin can be
wrong in the tails or wrong in its skewness, and the fix differs. The
caveat in :class:`NeymanResult` — heavy tails load ``V₂`` *and* ``V₄``, so a
component name is a summary, not a diagnosis — is exactly Henze's (1997)
point: the components of a smooth test are not strictly diagnostic.

References
----------
* Neyman, J. (1937). "Smooth test" for goodness of fit. *Skandinavisk
  Aktuarietidskrift* 20, 149–199.
* Rayner, J. C. W. & Best, D. J. (1989). *Smooth Tests of Goodness of Fit*.
  Oxford University Press — the components read as mean, variance, skewness
  and kurtosis departures.
* Ledwina, T. (1994). Data-driven version of Neyman's smooth test of fit.
  *JASA* 89(427), 1000–1005.
* Kallenberg, W. C. M. & Ledwina, T. (1997). Data-driven smooth tests when
  the hypothesis is composite. *JASA* 92(439), 1094–1104 — the null law of
  the components with estimated parameters.
* Henze, N. (1997). Do components of smooth tests of fit have diagnostic
  properties? *Metrika* 45, 121–130.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["COMPONENT_NAMES", "NeymanResult", "neyman_components",
           "neyman_test"]

#: What each component is sensitive to, in the order they are returned.
COMPONENT_NAMES = ("location", "dispersion", "asymmetry", "tail weight")

_EPS = 1e-12


@dataclass(frozen=True)
class NeymanResult:
    """Components of a smooth test, and the multiplicity-corrected verdict.

    Fields
    ------
    components : ``np.ndarray`` — the signed ``V_j``, asymptotically N(0, 1)
                 under a fully specified null; with an estimated margin the
                 null law changes (Kallenberg & Ledwina 1997), which is why
                 the package calibrates them by ``parametric_bootstrap``.
    squares    : ``np.ndarray`` — ``V_j²``, the per-component statistics.
    n          : int — sample size the components were computed on.
    component  : str — name of the largest component: *where* the margin
                 departs, which a single p-value cannot say.
    index      : int — its 1-based position.

    Read the **signed** components, not only the largest. A single name is a
    summary and summaries mislead: heavy tails show up as ``V₂`` negative
    *and* ``V₄`` positive, because extra mass in the extremes comes with a
    hollowed-out shoulder. Taking the argmax alone would label that
    "dispersion" and lose half the diagnosis.
    """

    components: np.ndarray
    squares:    np.ndarray
    n:          int
    component:  str = field(default="")
    index:      int = 0


def neyman_components(sample, cdf=None, *, order: int = 4) -> NeymanResult:
    """Smooth-test components of ``sample`` against ``cdf``.

    Parameters
    ----------
    sample : array-like, shape ``(n,)``
        The observations. One-dimensional: the construction rests on the
        probability integral transform, which has no direct multivariate
        analogue here.
    cdf : callable, optional
        Vectorised CDF of the reference distribution. Pass ``None`` when
        ``sample`` already holds the transformed values.
    order : int
        Number of components, at most 4 — beyond ``V₄`` the polynomials
        answer questions no margin diagnostic asks, and each one costs
        multiplicity.

    Returns
    -------
    NeymanResult
    """
    if not 1 <= order <= 4:
        raise ValueError(f"order must be in 1..4, got {order}")
    x = np.asarray(sample, dtype=float).ravel()
    if x.size < 8:
        raise ValueError(f"at least 8 observations required, got {x.size}")

    u = np.asarray(cdf(x), dtype=float).ravel() if cdf is not None else x
    u = np.clip(u, _EPS, 1.0 - _EPS)
    n = u.size

    # Orthonormal Legendre polynomials on [0, 1]: E[h_j(U)] = 0 and
    # Var[h_j(U)] = 1 for U uniform, which is what makes V_j ~ N(0, 1).
    z = 2.0 * u - 1.0
    basis = [
        math.sqrt(3.0) * z,
        math.sqrt(5.0) * 0.5 * (3.0 * z ** 2 - 1.0),
        math.sqrt(7.0) * 0.5 * (5.0 * z ** 3 - 3.0 * z),
        3.0 * 0.125 * (35.0 * z ** 4 - 30.0 * z ** 2 + 3.0),
    ][:order]

    v = np.array([float(np.sum(b) / math.sqrt(n)) for b in basis])
    sq = v ** 2
    k = int(np.argmax(sq))
    return NeymanResult(components=v, squares=sq, n=n,
                        component=COMPONENT_NAMES[k], index=k + 1)


def neyman_test(p_values, *, alpha: float = 0.05) -> tuple[float, int]:
    """Combine per-component p-values into one verdict, paying multiplicity.

    ``p_values`` are the bootstrap p-values of ``V_j²``, in component order.
    Returns ``(corrected_p, index)`` where ``index`` is the 1-based position
    of the component that drove it.

    Bonferroni, not the omnibus ``Σ V_j²``. Both hold their level; measured
    over 200 datasets per alternative, Bonferroni rejects a Student-t margin
    at 0.338 against the omnibus's 0.180 and a skew-normal one at 0.480
    against 0.290. Concentrating on one component and paying a factor of four
    beats spreading over four degrees of freedom, because a real margin is
    usually wrong in *one* way.

    Reporting the uncorrected minimum would take the level from 0.05 to 0.175
    — measured on the same datasets. The factor is not optional.
    """
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0 or not np.any(np.isfinite(p)):
        return float("nan"), 0
    finite = np.where(np.isfinite(p), p, np.inf)
    k = int(np.argmin(finite))
    return float(min(1.0, p.size * finite[k])), k + 1
