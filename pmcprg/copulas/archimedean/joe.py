"""
Joe copula — upper-tail dependence only.

Generator: φ(t) = -log(1-(1-t)^θ), θ ≥ 1.
CDF: C(u,v) = 1 − [(1−u)^θ + (1−v)^θ − (1−u)^θ(1−v)^θ]^{1/θ}
τ_K range: [0, 1)  (positive dependence only; θ=1 → independence)
λ_L = 0, λ_U = 2 − 2^{1/θ}

Reference: Joe, H. (1997). *Multivariate Models and Dependence Concepts*,
Chapman & Hall, ch. 5 (one-parameter families); Nelsen, R. B. (2006). *An Introduction
to Copulas*, 2nd ed., Springer, Table 4.1, family (4.2.6); Genest, C. &
MacKay, J. (1986). The joy of copulas. *The American Statistician* 40(4),
280–283 (τ = 1 + 4∫φ/φ′).

Numerics
--------
*Kernel.* Every path works on the complement logarithms ``kā = log(1 − a)``,
``kb̄ = log(1 − b)`` of its two arguments — never on ``1 − a`` formed in
floating point — so that the survival copula (``survival.py``) can pass
``log u`` itself, which is exact, where the base copula is evaluated at 1 − u
(audit FR-1). With ``a′ = θ kā ≤ 0``, ``b′ = θ kb̄ ≤ 0``, ``m = max(a′, b′)``,
``n = min(a′, b′)``::

    log W = m + λ,   λ = log1p(e^{n−m} · (−expm1(m))) ∈ [0, log 2]

where ``W = ā^θ + b̄^θ − ā^θ b̄^θ``: the two terms are of the same sign, so
neither cancellation (near independence, ā^θ → 1) nor underflow (θ ≈ 39 puts
W at 10⁻⁴⁰ on a sixth of the square, audit K-1) can occur. Then::

    log c   = (θ − 1)(kā + kb̄) + (1/θ − 2) log W + log(θ − 1 + W)
    C       = −expm1(log W / θ),        1 − C = exp(log W / θ)
    log h   = (1/θ − 1)·((m − a′) + λ) + log1mexp(−b′)          [h(b | a)]

the last line having the two leading terms of the textbook
``(1/θ − 1) log W + (θ − 1) kā`` cancelled exactly, so that 1 − h is
accurate when h → 1. ``log1mexp(x) = log(1 − e^{−x})`` is evaluated with
Mächler's switch at log 2 (Mächler, M. (2012). *Accurately computing
log(1 − exp(−|a|))*, vignette of the R package Rmpfr, no DOI).

*Inverse h-function* (no closed form). With ``x = ā^θ``, ``r = 1/x − 1``,
``κ = 1 − 1/θ`` and ``y = b̄^θ``, h = (1 − y)(1 + ry)^{−κ}, so ``h = w`` is
``F(ℓ) = −log(1 − e^ℓ) + κ·softplus(log r + ℓ) = −log w`` in ``ℓ = log y``
(y itself underflows on the diagonal once θ is in the thousands). F is
increasing and convex in ℓ: Newton's method started right of the root
decreases monotonically to it, from a start built from explicit bounds on
the root (``_joe_inv_h``). This is the Newton iteration in a transformed
variable of vinecopulib's ``joe.ipp`` (roadmap technique 16), with a
bracketing start instead of a reflected-Clayton guess and ``log r``,
``log y`` never leaving log space.

*Kendall's τ* (audit RB-2). Integrating Genest & MacKay's ``1 + 4∫φ/φ′``
gives τ = 1 + 2[ψ(2) − ψ(1 + 2/θ)]/(2 − θ) — the form R ``copula`` and
vinecopulib implement — which is 0/0 at θ = 2 and cancels to 0 at θ = 1.
The former 2000-term series stopped at 4.99·10⁻⁷ at θ = 1, so no θ existed
for τ < 5·10⁻⁷. Writing s = 2/θ and ψ(1+s) = −γ + Σ_k s/(k(k+s)) gives,
exactly,

    τ = 1 − s·S(s),   S(s) = Σ_{k≥1} 1/((k+1)(k+s))
      = (2 − s)·[1/2 − s·T(s)],   T(s) = Σ_{k≥1} 1/((k+1)(k+2)(k+s)),

evaluated as:

* θ ≤ 4/3 (s ≥ 3/2): the second form, T expanded in powers of (s − 2) with
  coefficients t_j = Σ_{i≥1} ζ(j+2+i, 3) (Hurwitz ζ, positive terms), so
  τ carries the factor 2 − s = 2(θ − 1)/θ explicitly and τ(1) = 0 exactly;
* 4/3 < θ ≤ 8/3: S expanded in powers of (s − 1), coefficients ζ(j+2, 2),
  which removes the singularity at θ = 2;
* θ > 8/3: S(s) = [ψ(1+s) − ψ(2)]/(s − 1), no singularity left, and
  1 − τ = s·S(s) keeps its relative precision as τ → 1.

θ(τ) is found by Brent's method on a bracket derived from 1/2 ≤ S ≤ 1:
θ ∈ [1/(1−τ), 2/(1−τ)], in the variable θ − 1 for τ ≤ 1/2 (relative
precision near independence) and in s on 1 − τ for τ > 1/2.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import digamma, zeta

from pmcprg.copulas._base import CopulaVirt
from pmcprg.exceptions    import CopulaParameterError
from pmcprg.numerics      import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

_LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Kendall's tau ↔ theta
# ---------------------------------------------------------------------------

def _series_coefficients():
    """Taylor coefficients of T(2 + ε) (in −ε) and S(1 + d) (in −d)."""
    t = np.array([sum(float(zeta(j + 2 + i, 3)) for i in range(1, 60)) for j in range(30)])
    z = np.array([float(zeta(j + 2, 2)) for j in range(40)])
    return t, z


_JOE_T, _JOE_Z = _series_coefficients()


def _joe_s_times_S(s: float) -> float:
    """``s·S(s)`` = 1 − τ at s = 2/θ, for 0 < s < 3/2 (module docstring)."""
    if s >= 0.75:
        return s * float(np.polynomial.polynomial.polyval(1.0 - s, _JOE_Z))
    return s * float((digamma(1.0 + s) - digamma(2.0)) / (s - 1.0))


def _joe_tau_from_theta(theta: float) -> float:
    """Kendall's τ of the Joe copula at θ ≥ 1, exact from θ = 1 to θ → ∞.

    See the module docstring for the three expansions; τ(1) = 0 exactly and
    the removable singularity of the digamma form at θ = 2 is absent.
    """
    theta = float(theta)
    if not theta >= 1.0:
        return math.nan
    if math.isinf(theta):
        return 1.0
    s = 2.0 / theta
    if s >= 1.5:
        tt = float(np.polynomial.polynomial.polyval(2.0 - s, _JOE_T))
        return (2.0 * (theta - 1.0) / theta) * (0.5 - s * tt)
    return 1.0 - _joe_s_times_S(s)


def _joe_theta_from_tau(tau_target: float) -> float:
    """Invert τ(θ) for the Joe copula; every τ ∈ [0, 1) has its θ.

    The bracket θ ∈ [1/(1−τ), 2/(1−τ)] follows from 1/2 ≤ S(s) ≤ 1 on
    s ∈ (0, 2]. τ = 0 is independence (θ = 1); τ = 1 (comonotonicity) has
    no finite θ and is refused, as are τ < 0 and non-finite τ.
    """
    tau = float(tau_target)
    if not (0.0 <= tau < 1.0):
        raise CopulaParameterError(
            f"Joe copula: no parameter θ for τ = {tau!r} (τ must lie in [0, 1)).")
    if tau == 0.0:
        return 1.0
    if tau <= 0.5:
        lo, hi = tau / (1.0 - tau), (1.0 + tau) / (1.0 - tau)
        delta = brentq(lambda d: _joe_tau_from_theta(1.0 + d) - tau,
                       lo, hi, xtol=1e-300, rtol=4.0 * EPS, maxiter=200)
        return 1.0 + delta
    one_minus = 1.0 - tau                     # exact for τ ≥ 1/2
    lo, hi = one_minus, 2.0 * one_minus       # bracket on s = 2/θ
    g_lo = _joe_s_times_S(lo) - one_minus
    if g_lo >= 0.0:                           # 1 − τ below ~1e-15: S(s) = 1 to double precision
        return 2.0 / lo
    s = brentq(lambda s_: _joe_s_times_S(s_) - one_minus,
               lo, hi, xtol=1e-300, rtol=4.0 * EPS, maxiter=200)
    return 2.0 / s


# ---------------------------------------------------------------------------
# Log-space kernel on complement logarithms (see module docstring)
# ---------------------------------------------------------------------------

def _log1mexp(x):
    """``log(1 − e^{−x})`` for x ≥ 0, Mächler's (2012) switch at log 2."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x <= _LOG2, np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))


def _joe_logw(ka, kb, th):
    """``(a′, b′, m, λ)`` with ``a′ = θ kā``, ``b′ = θ kb̄`` and ``log W = m + λ``."""
    a = th * ka
    b = th * kb
    m = np.maximum(a, b)
    lam = np.log1p(np.exp(np.minimum(a, b) - m) * -np.expm1(m))
    return a, b, m, lam


def _joe_logpdf(ka, kb, th):
    """log c from ``kā = log(1 − a)``, ``kb̄ = log(1 − b)``."""
    _, _, m, lam = _joe_logw(ka, kb, th)
    logw = m + lam
    # log(θ − 1 + W): at θ = 1 the constant vanishes, logaddexp(−inf, x) = x.
    with np.errstate(divide='ignore'):
        log_t = np.logaddexp(np.log(max(th - 1.0, 0.0)), logw)
    return (th - 1.0) * (ka + kb) + (1.0 / th - 2.0) * logw + log_t


def _joe_cdf(ka, kb, th):
    """``(C, 1 − C)`` with ``1 − C = W^{1/θ}``."""
    _, _, m, lam = _joe_logw(ka, kb, th)
    e = (m + lam) / th
    return -np.expm1(e), np.exp(e)


def _joe_h(kb, ka, th):
    """``(h, 1 − h)`` of h(b | a) = W^{1/θ−1} ā^{θ−1} (1 − b̄^θ)."""
    a, b, m, lam = _joe_logw(ka, kb, th)
    logh = (1.0 / th - 1.0) * ((m - a) + lam) + _log1mexp(-b)
    return np.exp(logh), -np.expm1(logh)


def _joe_inv_h(lw, ka, th, max_iter=100):
    """``(b, 1 − b)`` solving h(b | a) = w, from ``lw = log w`` and ``kā``.

    Newton on ``F(ℓ) = −log(1 − e^ℓ) + κ·softplus(log r + ℓ) = q``, q = −log w,
    in ``ℓ = log y = θ log b̄`` — y itself underflows on the diagonal once θ
    is in the thousands. F is increasing and convex in ℓ, so Newton started
    right of the root decreases monotonically to it, and one Newton step
    taken from left of the root lands right of it. Bounds, each nearly exact
    in one regime:

    * right (F ≥ q): ``log1mexp(q)`` (first term alone), ``q/κ − log r``
      (softplus(z) ≥ z), ``log q − log(1 + κ log 2 · r)`` where
      ``r e^ℓ ≤ 1`` (softplus(z) ≥ e^z log 2 for z ≤ 0);
    * left (F ≤ q): ``ℓ = log(1 − e^{−t})`` with ``t = q/(1 + κr)`` or
      ``t = q − κ log1p(r)`` (log1p(ry) ≤ min(ry, log1p r), y ≤ t).

    The start is the smallest right bound or the Newton step from the
    largest left bound, whichever is smaller. The iteration stops on a
    relative step of 4ε or a residual of 16ε·q — the rounding floor of F,
    below which the step no longer converges but oscillates.
    """
    q = -np.asarray(lw, dtype=float)
    ka = np.asarray(ka, dtype=float)
    kappa = 1.0 - 1.0 / th
    z = -th * ka                                         # log(1/x) ≥ 0
    lr = z + np.log(-np.expm1(-z))                       # log r = log expm1(z)

    def newton_step(ell):
        zz = lr + ell
        sp = np.logaddexp(0.0, zz)
        f = -_log1mexp(-ell) + kappa * sp - q
        fp = 1.0 / np.expm1(-ell) + kappa * np.exp(zz - sp)
        return f, f / fp

    with np.errstate(divide='ignore', invalid='ignore'):
        log_kappa = np.log(kappa)
        ell_a = _log1mexp(q)
        ell_b = q / kappa - lr if kappa > 0.0 else np.full_like(q, np.inf)
        ell_c = np.log(q) - np.logaddexp(0.0, log_kappa + math.log(_LOG2) + lr)
        ell_c = np.where(lr + ell_c <= 0.0, ell_c, np.inf)
        ell_hi = np.minimum(np.minimum(ell_a, ell_b), ell_c)
        t2 = q - kappa * np.logaddexp(0.0, lr)
        log_t = np.maximum(np.log(q) - np.logaddexp(0.0, log_kappa + lr),
                           np.where(t2 > 0.0, np.log(t2), -np.inf))
        ell_lo = np.where(log_t < -30.0, log_t - 0.5 * np.exp(log_t),
                          np.log(-np.expm1(-np.exp(log_t))))
        _, step_lo = newton_step(ell_lo)
        ell_lo_next = ell_lo - step_lo
        ell = np.where(np.isfinite(ell_lo_next), np.minimum(ell_hi, ell_lo_next), ell_hi)
    done = np.zeros(np.shape(ell), dtype=bool)
    for _ in range(max_iter):
        f, step = newton_step(ell)
        step = np.where(np.isfinite(step), step, 0.0)
        new = ell - step
        # converged entries are frozen so that each result is independent of the batch
        done_now = ((np.abs(step) <= 4.0 * EPS * np.abs(new)) | (np.abs(f) <= 16.0 * EPS * q)
                    | ~np.isfinite(f))
        ell = np.where(done, ell, new)
        done |= done_now
        if np.all(done):
            break
    kbb = ell / th                                       # log b̄
    return -np.expm1(kbb), np.exp(kbb)


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaJoe(CopulaVirt):
    """Joe copula — upper-tail dependence only.

    θ = _joe_theta_from_tau(τ_K),  θ ≥ 1.
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = _joe_theta_from_tau(self.params['tau_k'])

    # -- kernel interface (consumed by survival.py) -------------------------
    # The kernel coordinate of a point x is log(1 − x): ``_kcoord(x)`` computes
    # it for x itself, ``_kcoord_reflected(x)`` for 1 − x — which is log x.

    @staticmethod
    def _kcoord(x):
        return np.log1p(-x)

    @staticmethod
    def _kcoord_reflected(x):
        return np.log(x)

    def _k_logpdf(self, ka, kb):
        return _joe_logpdf(ka, kb, self.theta)

    def _k_cdf(self, ka, kb):
        return _joe_cdf(ka, kb, self.theta)

    def _k_h(self, kb, ka):
        return _joe_h(kb, ka, self.theta)

    def _k_inv_h(self, lw, ka):
        return _joe_inv_h(lw, ka, self.theta)

    # -- public API ----------------------------------------------------------

    def pdf(self, uv):
        """c(u,v) = ū^{θ-1} · v̄^{θ-1} · W^{1/θ-2} · (θ−1+W)

        where  ū=1−u, v̄=1−v, W = ū^θ + v̄^θ − ū^θ·v̄^θ.

        This is ∂²C/∂u∂v: differentiating C = 1 − W^{1/θ} twice gives
        ū^{θ-1}v̄^{θ-1}W^{1/θ-2}[(θ−1)(1−ū^θ)(1−v̄^θ) + θW], which reduces to
        the expression above because (1−ū^θ)(1−v̄^θ) = 1 − W. There is **no**
        leading θ factor: one used to be present here, making the density
        integrate to θ rather than 1.
        """
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return float(np.exp(_joe_logpdf(np.log1p(-u), np.log1p(-v), self.theta)))

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF, computed entirely in log space (see module docstring).

        With ū = 1 − u, v̄ = 1 − v, a = ū^θ, b = v̄^θ, W = a + b − a·b:
            log c(u, v) = (θ − 1)(log ū + log v̄)
                        + (1/θ − 2) log W
                        + log(θ − 1 + W)
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            return _joe_logpdf(np.log1p(-u), np.log1p(-v), self.theta)

    def cdf(self, uv):
        """C(u,v) = 1 − W^{1/θ}."""
        u = np.float64(minmaxEPS(uv[0]))
        v = np.float64(minmaxEPS(uv[1]))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _joe_cdf(np.log1p(-u), np.log1p(-v), self.theta)
        return float(np.clip(c, 0.0, 1.0))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (NaN where it cannot be evaluated)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            c, _ = _joe_cdf(np.log1p(-u), np.log1p(-v), self.theta)
        return np.clip(c, 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = W^{1/θ-1} · ū^{θ-1} · (1−v̄^θ)."""
        u = np.float64(minmaxEPS(u))
        v = np.float64(minmaxEPS(v))
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            h, _ = _joe_h(np.log1p(-v), np.log1p(-u), self.theta)
        return float(np.clip(h, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """v such that h(v|u) = w, by the monotone Newton iteration of the module docstring."""
        return float(self.inv_h_array(np.array([w], dtype=float), np.array([u], dtype=float))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised inverse h-function (monotone Newton, module docstring)."""
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        if w.shape != u.shape:
            raise ValueError(f"inv_h_array: w {w.shape} and u {u.shape} must match.")
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            v, _ = _joe_inv_h(np.log(w), np.log1p(-u), self.theta)
        return np.clip(v, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Joe copula: λ_L = 0, λ_U = 2 − 2^{1/θ}."""
        return 0.0, float(2.0 - 2.0 ** (1.0 / self.theta))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaJoe(tau_k=0.5)
    lam_u = 2.0 - 2.0 ** (1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tau(θ=1) = {_joe_tau_from_theta(1.0):.6f}  [should be 0]')
    print(f'tail dep : λ_L = 0,  λ_U = {lam_u:.4f}  [= 2−2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
