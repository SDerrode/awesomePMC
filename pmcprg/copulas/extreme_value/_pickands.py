"""
Shared numerics for bivariate extreme-value copulas built from a Pickands
dependence function A(t), t in [0, 1] (A(0) = A(1) = 1, max(t, 1-t) <= A(t) <= 1).

Factored out of the Galambos module (FR-9 pilot) once the Hüsler–Reiss family
(FR-9, round 2) needed the *same* two pieces of machinery: the adaptive
quadrature of the Genest & MacKay (1986) identity

    tau = int_0^1  t(1-t) A''(t) / A(t)  dt

and the tau -> parameter Brent inversion built on top of it. Both families'
single dependence parameter increases with dependence (Galambos's theta;
Hüsler–Reiss's lambda — see that module's docstring for why, contrary to a
plausible-looking but wrong first guess, lambda runs the *same* way as
theta rather than the opposite way), so one pair of conventions serves both:

* The quadrature breakpoints at 0.5 +/- eps_scale/param localise the ridge
  that A''(t) develops near t = 1/2 as the parameter grows towards the
  comonotone limit (the Pickands function of the comonotone copula M is the
  non-smooth max(t, 1-t), so A'' concentrates near t = 1/2 as either family
  approaches it). Without the breakpoints, ``scipy.integrate.quad`` silently
  returns a wrong (too small, sometimes negative) integral for a large
  parameter — caught during the Galambos pilot by comparison against a
  10,000-node Gauss-Legendre grid, not by ``quad``'s own error estimate,
  which stays small even when wrong. See each family's module docstring
  for the numbers that pin its own parameter range.
* The parameter -> tau map is only known to be monotone numerically (no
  proof attempted for either family); tau(param) can be any function
  handed to :func:`invert_tau`, so the Brent bracket is a plain interval
  search, not a closed-form inversion.

Kept deliberately small: this is a two-family case, and both families'
closed-form A(t), A'(t), A''(t) differ enough (Galambos: a power-mean
kernel of t^{-theta}, (1-t)^{-theta}; Hüsler–Reiss: the standard normal CDF
and PDF) that forcing them into a shared ``_A_terms`` would only obscure
each family's own derivation (see ``galambos.py`` and ``husler_reiss.py``
for those). Each family also keeps its own pdf, cdf and h-function, built
from its own A(t), A'(t), A''(t) via the general EV-copula identities
(Joe 2014, §6.1.2) but not shared here, since only the derivatives feed
this module.

References
----------
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280-283 (Kendall's tau of a copula from its Pickands dependence function).
* Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman & Hall/CRC,
  §6.1 (the Pickands representation and the general EV-copula pdf/cdf/h
  identities in terms of A, A', A'').
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from pmcprg.exceptions import CopulaParameterError

ATerms = Callable[[np.ndarray, float], tuple]


def tau_from_A_terms(A_terms: ATerms, param: float, *,
                     eps_scale: float = 5.0, quad_limit: int = 200) -> float:
    """Kendall's tau of the copula whose Pickands function has ``A_terms``.

    Parameters
    ----------
    A_terms : callable(t, param) -> (A, A', A'')
        The family's own closed-form Pickands function and its first two
        derivatives at ``t`` (array or scalar), e.g.
        :func:`pmcprg.copulas.extreme_value.galambos._galambos_A_terms` or
        :func:`pmcprg.copulas.extreme_value.husler_reiss._husler_reiss_A_terms`.
        Only ``A`` and ``A''`` are used by the Genest & MacKay integrand;
        ``A'`` is accepted (and returned by every family here) because it is
        needed by the pdf/h-function formulas that call the same function.
    param : float
        The family's single dependence parameter (theta, lambda, ...).
        ``param <= 0`` returns 0.0 (independence) without quadrature — the
        boundary every family here excludes from its own admissible range,
        handled the same way the two families' own code did before this was
        factored out.
    eps_scale : float
        Quadrature breakpoints at ``0.5 +/- min(eps_scale / param, 0.49)``
        (module docstring); 5.0 is the value the Galambos pilot verified
        (against a 10,000-node Gauss-Legendre grid) up to theta = 1e6, and
        is re-verified for Hüsler–Reiss up to its own parameter cap in that
        module's tests.
    """
    param = float(param)
    if param <= 0.0:
        return 0.0

    def integrand(t):
        A, _, App = A_terms(t, param)
        return t * (1.0 - t) * App / A

    eps = min(eps_scale / param, 0.49)
    points = sorted({p for p in (0.5 - eps, 0.5, 0.5 + eps) if 0.0 < p < 1.0})
    val, _ = quad(integrand, 0.0, 1.0, points=points, limit=quad_limit)
    return float(np.clip(val, 0.0, 1.0))


def invert_tau(tau_from_param: Callable[[float], float], tau_target: float,
               lo: float, hi: float, *, family_name: str,
               xtol: float = 1e-12, maxiter: int = 200) -> float:
    """Brent inversion of a family's tau(param) on ``[lo, hi]``.

    ``tau_target <= 0`` raises :class:`CopulaParameterError` — no
    extreme-value copula here has negative or null dependence (independence
    is a limit at the low end of the parameter range, never attained
    exactly). A target at or beyond ``tau_from_param(hi)`` is clamped to
    ``hi`` (the RB-10/Frank/Plackett/Galambos convention: the copula is
    built at the capped parameter and reports the tau it actually realises,
    not the one requested).
    """
    tau_target = float(tau_target)
    if tau_target <= 0.0:
        raise CopulaParameterError(
            f'{family_name}: tau_k={tau_target!r} must be > 0 (no negative or '
            f'zero dependence in an extreme-value copula; independence is a '
            f'limit of the parameter range, never attained exactly).')
    hi_tau = tau_from_param(hi)
    if tau_target >= hi_tau:
        return float(hi)
    return float(brentq(lambda p: tau_from_param(p) - tau_target, lo, hi,
                        xtol=xtol, rtol=4.0 * np.finfo(float).eps, maxiter=maxiter))
