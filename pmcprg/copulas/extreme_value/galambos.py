"""
Galambos copula — bivariate extreme-value family, upper-tail dependence only.

Pickands dependence function: A(t) = 1 − (t^{−θ} + (1−t)^{−θ})^{−1/θ}, θ > 0,
t ∈ [0, 1]  (A(0) = A(1) = 1, so the family contains independence at the
degenerate limit θ → 0).

CDF:  C(u,v) = u·v·exp(S),      S = (w^{−θ} + z^{−θ})^{−1/θ},
      w = −ln u, z = −ln v      (Galambos 1975, eq. for the bivariate model;
      Joe 2014, §6.3 — the "negative logistic" bivariate EV copula).
PDF:  c(u,v) = e^S · [(1 − S·r/w)(1 − S·(1−r)/z) + S(1+θ)·r(1−r)/(wz)],
      r = w^{−θ} / (w^{−θ} + z^{−θ}).
h:    h(v|u) = ∂C/∂u = v·e^S·(1 − S·r/w).

Both pdf and h were re-derived here from C (``sympy``, then cross-checked
numerically against the closed form at several (u, v, θ) — see
``scripts`` history / the module tests) rather than transcribed, because the
literature's h/c formulas for Galambos are usually quoted without a
derivation and are easy to mistranscribe (a sign, a missing factor of θ+1).

θ → 0⁺: independence.  θ → ∞: comonotone copula M.
λ_U = 2(1 − A(1/2)) = 2^{−1/θ},  λ_L = 0 — verified from the general EV
identity λ_U = 2(1 − A(1/2)) (Nelsen 2006, Thm 5.7.3 for EV copulas via the
Pickands function; A(1/2) = 1 − 2^{−1−1/θ} here).

Kendall's τ(θ) — **no closed form**. The value τ = 1/(θ+2) that is
sometimes quoted (e.g. in secondary summaries) is wrong: it is *decreasing*
in θ, from 1/2 at θ = 0 to 0 as θ → ∞ — the opposite direction from
λ_U = 2^{−1/θ}, which must increase with θ since larger θ means stronger
extremal dependence. This module computes τ(θ) instead from the Genest &
MacKay (1986) identity

    τ = ∫₀¹ [t(1−t)/A(t)] · A''(t) dt

applied to the Pickands function above (closed-form A, A', A'' — see
``_galambos_A_terms``). Verified two independent ways before use:

1. The identity itself, applied to Gumbel's Pickands function
   A(t) = (t^θ + (1−t)^θ)^{1/θ}, reproduces Gumbel's closed form
   τ = 1 − 1/θ to 1e-12 (30-digit ``mpmath``, θ = 1.5, 2, 5) — so the
   integral formula itself, not just this module's arithmetic, is right.
2. Applied to the Galambos A above, at θ = 1, 2, 5, 20 it gives
   τ = 0.41840, 0.63116, 0.82485, 0.95171 (30-digit ``mpmath``); a
   200,000-pair Monte-Carlo Kendall's τ from this module's own ``sample()``
   at θ = 1, 2, 5 agrees to within its Monte-Carlo standard error
   (≈ 0.0022 at n = 200,000): 0.4186, 0.6297, 0.8241 respectively (seed 0).

τ(θ) is strictly increasing on (0, ∞) (checked numerically; no proof
attempted), so θ(τ) is obtained by Brent's method on ``tau_from_theta``
directly (no cached table: unlike Plackett's 2-D quadrature, one evaluation
here costs ~1-2 ms, so a Brent search of ~40 evaluations per copula
construction, ~50-100 ms, is cheap enough for this family's scope — a
production ICE loop rebuilding this copula thousands of times would want
Plackett's cached-table approach instead; noted as a follow-up).

θ is capped at ``_THETA_MAX`` = 1e6 (``adaptive quadrature loses accuracy
resolving `A''`'s increasingly sharp ridge at t = 1/2 beyond it without
extra breakpoints — see ``_galambos_tau_from_theta``); the table this caps
reaches τ ≈ 1 − 1e-6, declared via ``reachable_tau_bounds`` (Frank/Plackett
pattern) so the package never stores a τ beyond what a finite θ realises.

References
----------
* Galambos, J. (1975). Order statistics of samples from multivariate
  distributions. *JASA* 70(351), 674–680,
  doi:10.1080/01621459.1975.10482485.
* Joe, H. (2014). *Dependence Modeling with Copulas*, Chapman & Hall/CRC,
  §6.3 (bivariate extreme-value copulas; the Galambos/"negative logistic"
  family, its CDF and tail dependence).
* Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed., Springer,
  §5.7 and Theorem 5.7.3 (Pickands representation; upper-tail dependence
  of an EV copula as 2(1 − A(1/2))).
* Genest, C. & MacKay, J. (1986). The joy of copulas: bivariate
  distributions with uniform marginals. *The American Statistician* 40(4),
  280–283 (Kendall's τ of a copula from its dependence function A).

Numerics
--------
``w = −ln u`` can be as small as ≈ 2.2·10⁻¹⁶ (u = 1 − EPS); computing
``w^{−θ}`` directly then overflows for θ ≳ 26. Every path below instead
works from ``logp = −θ·ln w``, ``logq = −θ·ln z`` — plain products, never
exponentiated until the final, always-bounded quantities are formed::

    logT = logaddexp(logp, logq),        S = exp(−logT/θ)
    r    = expit(logp − logq)  [= p/(p+q), sigmoid — avoids forming p, q, T]
    log C = ln u + ln v + S              (≤ 0 always: S ≤ min(w, z), proof
                                           below, so C = exp(log C) never
                                           overflows)
    h(v|u) = v·exp(S)·(1 − S·r/w)         (0 ≤ S·r/w ≤ 1: S ≤ w, so the
                                           bracket never overflows either)

*Why S ≤ min(w, z).* T = p + q ≥ max(p, q), and t ↦ t^{−1/θ} is decreasing,
so S = T^{−1/θ} ≤ max(p, q)^{−1/θ} = min(w, z) (the larger of p, q belongs
to the smaller of w, z, since x ↦ x^{−θ} is itself decreasing). The same
bound applied to the Pickands function's own kernel (``w = −ln t``,
``z = −ln(1−t)``, used only inside :func:`_galambos_tau_from_theta`) keeps
its ``g = B^{−1/θ}`` term (the Pickands-level analogue of S) bounded too.

``inv_h`` has no closed form (unlike Clayton; and Gumbel's monotone-Newton
trick does not carry over — h is not separable in one transformed
variable here), so this module relies on :meth:`CopulaVirt.inv_h`'s
Brent-bracketed default over the closed-form ``conditional_cdf`` above —
adequate for a one-parameter pilot family; a shared Pickands/EV scaffold
for the other extreme-value families (Hüsler–Reiss, Tawn, t-EV) would be
the natural place to add a faster vectorised inverse.
"""
if __name__ == '__main__':
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from pmcprg.copulas._base import CopulaVirt
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Log-space kernel on ka = log a, kb = log b (module docstring)
# ---------------------------------------------------------------------------

def _galambos_terms(ka, kb, theta):
    """``(w, z, S, a1, a2, logr, log1mr)`` from ``ka = log u``, ``kb = log v``.

    ``a1 = 1 − S·r/w`` and ``a2 = 1 − S·(1−r)/z`` (r = p/(p+q), the module
    docstring's bracket terms) are the ones that cancel catastrophically
    when u and v pull θ hard in opposite directions (w ≪ z or z ≪ w with θ
    large: r rounds to exactly 1.0 or 0.0 in double precision, and
    ``1 − S·r/w`` then loses every digit — caught during development at
    (u, v, θ) = (0.3, 1e-12, 19.3), where the naive formula returned exactly
    0.0 against a true density ≈ 3.3e-26). Both are computed instead from

        d = logq − logp,   softplus(x) = log(1 + eˣ) = logaddexp(0, x)
        a1 = −expm1(−(1+θ)/θ · softplus(d)),   a2 = −expm1(−(1+θ)/θ · softplus(−d))

    a closed form obtained by writing ``logS − log w + log r`` (whose
    exponential is ``S·r/w``) as ``−(1+θ)/θ·softplus(d)`` directly, using
    ``log w = −logp/θ`` and ``log r = −softplus(d)`` — never forming the two
    near-equal O(1)…O(θ) terms whose difference is the tiny result, so
    ``expm1`` sees its argument at full relative precision however small
    ``a1`` or ``a2`` is.
    """
    w = -ka
    z = -kb
    logp = -theta * np.log(w)
    logq = -theta * np.log(z)
    logT = np.logaddexp(logp, logq)
    S = np.exp(-logT / theta)
    d = logq - logp
    softplus_d = np.logaddexp(0.0, d)
    softplus_nd = np.logaddexp(0.0, -d)
    coef = -(1.0 + theta) / theta
    a1 = -np.expm1(coef * softplus_d)
    a2 = -np.expm1(coef * softplus_nd)
    logr = -softplus_d
    log1mr = -softplus_nd
    return w, z, S, a1, a2, logr, log1mr


def _galambos_logpdf(ka, kb, theta):
    """log c(u, v) = S + log(bracket) (module docstring)."""
    w, z, S, a1, a2, logr, log1mr = _galambos_terms(ka, kb, theta)
    third = S * (1.0 + theta) * np.exp(logr + log1mr) / (w * z)
    bracket = a1 * a2 + third
    with np.errstate(invalid='ignore', divide='ignore'):
        return S + np.log(bracket)


def _galambos_cdf(ka, kb, theta):
    """``(C, 1 − C)``. ``log C = ka + kb + S`` is always ≤ 0 (module docstring)."""
    w = -ka
    z = -kb
    logp = -theta * np.log(w)
    logq = -theta * np.log(z)
    logT = np.logaddexp(logp, logq)
    S = np.exp(-logT / theta)
    logC = ka + kb + S
    return np.exp(logC), -np.expm1(logC)


def _galambos_h(kb, ka, theta):
    """``(h, 1 − h)`` of h(b | a) = ∂C(a, b)/∂a."""
    _, _, S, a1, _, _, _ = _galambos_terms(ka, kb, theta)
    logh = kb + S + np.log(a1)
    with np.errstate(invalid='ignore', divide='ignore'):
        h = np.exp(logh)
    return h, -np.expm1(logh)


# ---------------------------------------------------------------------------
# Pickands function A(t) = 1 − (t^{-θ} + (1-t)^{-θ})^{-1/θ} and its
# derivatives, in the same overflow-free log-space kernel as above
# (t plays the role of ``u``, 1 − t of ``v``; module docstring).
# ---------------------------------------------------------------------------

def _galambos_A_terms(t, theta):
    """``(A, A', A'')`` at ``t`` (array or scalar), closed form.

    Verified against ``mpmath`` numerical differentiation of ``A`` at 30
    digits, several (t, θ), during development (see module docstring).
    """
    t = np.asarray(t, dtype=float)
    tm = 1.0 - t
    w = -np.log(t)
    z = -np.log(tm)
    x = theta * w
    y = theta * z
    m = np.maximum(x, y)
    ex = np.exp(x - m)
    ey = np.exp(y - m)
    denom = ex + ey
    g = np.exp(-(m + np.log(denom)) / theta)          # Pickands-level S
    d1 = (ey / tm - ex / t) / denom                    # (B'/B) / θ
    d2 = theta * (theta + 1.0) * (ey / tm ** 2 + ex / t ** 2) / denom  # B''/B
    Ap = g * d1
    App = (g / theta) * d2 - g * (1.0 + theta) * d1 ** 2
    A = 1.0 - g
    return A, Ap, App


def _galambos_tau_integrand(t, theta):
    A, _, App = _galambos_A_terms(t, theta)
    return t * (1.0 - t) * App / A


def _galambos_tau_from_theta(theta: float) -> float:
    """τ(θ) by adaptive quadrature of the Genest & MacKay (1986) identity.

    Breakpoints at ``0.5 ± 5/θ`` localise the ridge that ``A''`` develops
    near t = 1/2 as θ grows (the comonotone limit's Pickands function is
    the non-smooth ``max(t, 1-t)``); without them ``quad`` silently returns
    a wrong (too small, even negative) integral for θ beyond a few
    thousand — this was caught by comparing against a 10,000-node
    Gauss–Legendre grid during development, not by ``quad``'s own error
    estimate, which stays small even when wrong.
    """
    theta = float(theta)
    if theta <= 0.0:
        return 0.0
    eps = min(5.0 / theta, 0.49)
    points = sorted({0.5 - eps, 0.5, 0.5 + eps} & {p for p in (0.5 - eps, 0.5, 0.5 + eps) if 0.0 < p < 1.0})
    val, _ = quad(_galambos_tau_integrand, 0.0, 1.0, args=(theta,),
                  points=points, limit=200)
    return float(np.clip(val, 0.0, 1.0))


_THETA_LO = 1.0e-4    # tau_from_theta(_THETA_LO) underflows to 0.0 (« EPS)
_THETA_HI = 1.0e6     # quadrature verified accurate here (module docstring)


def _galambos_theta_from_tau(tau_target: float) -> float:
    """Invert τ(θ) by Brent's method on ``[_THETA_LO, _THETA_HI]``.

    τ(θ) is strictly increasing (checked numerically, not proved) from
    ≈ 0 to τ(_THETA_HI) < 1. A target beyond that reachable maximum is
    clamped to ``_THETA_HI`` (module docstring; ``reachable_tau_bounds``
    keeps the package from ever requesting more).
    """
    tau_target = float(tau_target)
    if tau_target <= 0.0:
        raise CopulaParameterError(
            f'Galambos: tau_k={tau_target!r} must be > 0 (no negative or '
            f'zero dependence in an extreme-value copula; independence is '
            f'the θ → 0 limit, never attained).')
    hi_tau = _galambos_tau_from_theta(_THETA_HI)
    if tau_target >= hi_tau:
        return _THETA_HI
    return float(brentq(lambda th: _galambos_tau_from_theta(th) - tau_target,
                        _THETA_LO, _THETA_HI, xtol=1e-12, rtol=4.0 * np.finfo(float).eps,
                        maxiter=200))


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaGalambos(CopulaVirt):
    """Galambos copula — bivariate extreme-value family, θ > 0.

    Parameters
    ----------
    tau_k : float
        Kendall's τ. Must be > 0 (extreme-value copulas have no negative or
        null dependence — the module docstring); τ = 0 is the θ → 0 limit
        and is not attained. Reachable up to ``reachable_tau_bounds()[1]``
        (≈ 1 − 1e-6): a larger request is clamped to θ = 1e6 and
        ``params['tau_k']`` becomes the τ this θ realises (RB-10).
    """

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float]:
        """``(EPS, τ(1e6))`` — the τ this family's finite θ range realises.

        The lower end is not restrictive (θ ≈ 0.02 already reaches τ = EPS,
        far inside ``[_THETA_LO, _THETA_HI]``); only the upper end narrows
        the registered ``[EPS, 1.0]``.
        """
        return float(EPS), _galambos_tau_from_theta(_THETA_HI)

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        tau = self.params['tau_k']
        self.theta = _galambos_theta_from_tau(tau)
        if self.theta >= _THETA_HI:
            # θ was clamped: store the τ it realises, not the one requested
            # (audit RB-10 convention, as Frank/Plackett).
            self.params['tau_k'] = _galambos_tau_from_theta(self.theta)

    # -- public API ------------------------------------------------------

    def pdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(_galambos_logpdf(np.log(u), np.log(v), self.theta)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF in log space (module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return _galambos_logpdf(np.log(u), np.log(v), self.theta)

    def cdf(self, uv):
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _galambos_cdf(np.log(u), np.log(v), self.theta)
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (NaN where it cannot be evaluated)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _galambos_cdf(np.log(u), np.log(v), self.theta)
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C(u,v)/∂u, in log space (module docstring)."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = _galambos_h(np.log(v), np.log(u), self.theta)
        return float(np.clip(h, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """Galambos copula: λ_L = 0, λ_U = 2^{−1/θ} (module docstring)."""
        return 0.0, float(2.0 ** (-1.0 / self.theta))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaGalambos(tau_k=0.5)
    lam_u = 2.0 ** (-1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}')
    print(f'tau back : {_galambos_tau_from_theta(cop.theta):.6f}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = 0,  λ_U = {lam_u:.4f}  [= 2^(-1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
