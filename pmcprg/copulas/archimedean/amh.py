"""
Ali-Mikhail-Haq (AMH) copula.

Generator: φ(t) = log((1 − θ(1−t)) / t),   θ ∈ [−1, 1).

CDF:  C(u,v) = u·v / W,            W = 1 − θ(1−u)(1−v)
PDF:  c(u,v) = [1 − θ(2−u−v−uv) + θ²(1−u)(1−v)] / W³
h:    h(v|u) = v·(1 − θ(1−v)) / W²

τ_K = 1 − 2[θ + (1−θ)²·log(1−θ)] / (3θ²)
    (limit τ_K → 0 as θ → 0, i.e. independence)

τ range:  [τ(−1), 1/3)  ≈ [−0.1817, 0.3333),  τ(−1) = (5 − 8 ln 2)/3
           (very narrow — AMH is mainly for mild dependence)

λ_L = λ_U = 0  (no tail dependence).

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, Table 4.1 family (4.2.3) and ch. 5 (τ); the τ(θ) expression is
τ = 1 + 4∫₀¹ φ(t)/φ'(t) dt applied to the AMH generator — Genest, C. &
MacKay, J. (1986). The joy of copulas: bivariate distributions with
uniform marginals. *The American Statistician* 40(4), 280–283.

Numerics
--------
*Kendall's τ.* The closed form cancels twice as θ → 0 — θ + (1−θ)² log(1−θ)
is O(θ²) from O(θ) terms, and 1 − 2(…)/(3θ²) is O(θ) — so its relative
error reaches 10⁻¹¹ for |θ| ≤ 0.05, 4·10⁻⁸ at θ = 10⁻⁴ and 100 % below 10⁻⁸. The
former "Taylor" branch used τ ≈ 2θ/3, which is the wrong slope
(τ'(0) = 2/9), and the inversion floored |τ| < 10⁻¹⁰ to θ = 0; together they
built τ = −10⁻⁸ as θ = −1.2·10⁻⁶, a copula realising τ = −2.7·10⁻⁷ (audit
RB-10, FR-1). For |θ| ≤ ½ the series

    τ = (4θ/3) Σ_{k≥0} θ^k / ((k+1)(k+2)(k+3)) = (2θ/9)(1 + θ/4 + θ²/10 + θ³/20 + θ⁴/35 + …)

is used, carried to θ⁵⁰ (truncation < 10⁻¹⁹); its first five terms are
``tauAMH`` of the R package ``copula``, which switches at |θ| ≤ 0.01.
Against 80-digit ``decimal`` evaluations of the closed form, the series is
good to 3.3·10⁻¹⁶ for |θ| ≤ ½ and the closed form to 7.6·10⁻¹⁵ beyond.

*Density.* Written with ū = 1 − u, v̄ = 1 − v, s = u + v − uv = 1 − ūv̄, the
numerator and W are sums of non-negative terms on each side of θ = 0, so
nothing cancels even at θ → 1 in the corner u, v → 0, where the textbook
numerator 1 − 2θ + θ² = (1 − θ)² is formed from O(1) terms:

    θ ≥ 0:  N = (1−θ)² + θ[(1−θ)(u+v) + (1+θ)uv],    W = (1−θ) + θ s,
    θ < 0:  N = (1+θ)[(1+θ) − θ s] − 2θ(ū+v̄),        W = 1 − θ ūv̄.

N > 0 on the open square for every θ ∈ [−1, 1); at θ = −1 it vanishes at the
corner (1, 1) only, so the density is never floored: ``pdf_array`` is
``exp(logpdf_array)`` and no path returns an ``EPS`` sentinel (audit RB-9).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from scipy.optimize import brentq

from pmcprg.copulas._base import CopulaVirt
from pmcprg.exceptions import CopulaParameterError
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# τ_K ↔ θ relationship
# ---------------------------------------------------------------------------

# Series coefficients 1/((k+1)(k+2)(k+3)), k = 0..50, for Horner evaluation.
_AMH_TAU_SERIES = tuple(1.0 / ((k + 1) * (k + 2) * (k + 3)) for k in range(51))
_AMH_TAU_SERIES_MAX_THETA = 0.5

# Admissible θ bracket: θ = −1 is included, θ = 1 is not (the generator
# degenerates there). The largest double below 1 reaches every τ < 1/3 to
# within one ulp, so the registered upper bound 1/3 − EPS has an antecedent.
_AMH_THETA_LO = -1.0
_AMH_THETA_HI = float(np.nextafter(1.0, 0.0))


def _amh_tau_from_theta(theta: float) -> float:
    """τ_K(θ) = 1 − 2[θ + (1−θ)²·log(1−θ)] / (3θ²), θ ∈ [−1, 1).

    For |θ| ≤ ½ the power series of the module docstring (exact to
    rounding); the closed form with ``log1p`` elsewhere.
    """
    theta = float(theta)
    if abs(theta) <= _AMH_TAU_SERIES_MAX_THETA:
        s = 0.0
        for coef in reversed(_AMH_TAU_SERIES):
            s = s * theta + coef
        return float(4.0 * theta / 3.0 * s)
    one_minus_t = 1.0 - theta
    if one_minus_t <= 0.0:
        return float('nan')              # θ ≥ 1: outside the family
    log_term = one_minus_t ** 2 * np.log1p(-theta)
    return float(1.0 - 2.0 * (theta + log_term) / (3.0 * theta ** 2))


def _amh_theta_from_tau(tau_target: float) -> float:
    """Invert :func:`_amh_tau_from_theta` by Brent's method on [−1, 1).

    τ is increasing in θ, from τ(−1) = (5 − 8 ln 2)/3 ≈ −0.1817 to 1/3 as
    θ → 1. The root is solved to relative precision (``xtol`` negligible,
    ``rtol`` = 4 ulp), so a τ of 10⁻¹² gets θ = 4.5·10⁻¹² rather than an
    absolute 10⁻¹² tolerance. A τ within 10⁻¹⁵ of the τ at an end of the
    bracket returns that end; a τ further outside the family raises
    :class:`CopulaParameterError` instead of being silently clamped.
    """
    tau_target = float(tau_target)
    if tau_target == 0.0:
        return 0.0                       # exact: τ(0) = 0
    f_lo = _amh_tau_from_theta(_AMH_THETA_LO) - tau_target
    f_hi = _amh_tau_from_theta(_AMH_THETA_HI) - tau_target
    if f_lo >= 0.0:
        if f_lo <= 1e-15:
            return _AMH_THETA_LO
        raise CopulaParameterError(
            f'AMH: tau_k = {tau_target!r} is below τ(θ = −1) = '
            f'{tau_target + f_lo!r}.')
    if f_hi <= 0.0:
        if f_hi >= -1e-15:
            return _AMH_THETA_HI
        raise CopulaParameterError(
            f'AMH: tau_k = {tau_target!r} is not below τ(θ → 1) = 1/3.')
    return float(brentq(lambda t: _amh_tau_from_theta(t) - tau_target,
                        _AMH_THETA_LO, _AMH_THETA_HI,
                        xtol=1e-300, rtol=4.0 * np.finfo(float).eps, maxiter=200))


# ---------------------------------------------------------------------------
# Cancellation-free kernel (see module docstring)
# ---------------------------------------------------------------------------

def _amh_numer_w(u, v, th):
    """``(N, W)`` with c = N / W³, for arrays or numpy scalars in (0, 1)."""
    uv_ = u * v
    upv = u + v
    if th >= 0.0:
        numer = (1.0 - th) ** 2 + th * ((1.0 - th) * upv + (1.0 + th) * uv_)
        w = (1.0 - th) + th * (upv - uv_)
    else:
        ub = 1.0 - u                     # exact for u ≥ ½ (Sterbenz)
        vb = 1.0 - v
        numer = (1.0 + th) * ((1.0 + th) - th * (upv - uv_)) - 2.0 * th * (ub + vb)
        w = 1.0 - th * ub * vb
    return numer, w


def _amh_w(u, v, th):
    """W = 1 − θ(1−u)(1−v), as a sum of non-negative terms."""
    if th >= 0.0:
        return (1.0 - th) + th * (u + v - u * v)
    return 1.0 - th * (1.0 - u) * (1.0 - v)


def _amh_logpdf(u, v, th):
    """log c = log N − 3 log W."""
    numer, w = _amh_numer_w(u, v, th)
    return np.log(numer) - 3.0 * np.log(w)


def _amh_h(v, u, th):
    """h(v|u) = v·(1 − θ(1−v)) / W², with 1 − θ(1−v) = (1−θ) + θv for θ ≥ 0."""
    lead = (1.0 - th) + th * v if th >= 0.0 else 1.0 - th * (1.0 - v)
    return v * lead / _amh_w(u, v, th) ** 2


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaAMH(CopulaVirt):
    """Ali-Mikhail-Haq copula.

    Parameters
    ----------
    tau_k : float
        Kendall's τ.  Must lie in [τ_min, 1/3 − ε] ≈ [−0.1817, 0.3333).
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = _amh_theta_from_tau(self.params['tau_k'])

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv) -> float:
        """C(u,v) = u·v / W,  W = 1 − θ(1−u)(1−v)."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        return float(np.clip(u * v / _amh_w(u, v, self.theta), 0.0, 1.0))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF, log c = log N − 3 log W (no floor; see module docstring)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        with np.errstate(invalid='ignore', divide='ignore'):
            return _amh_logpdf(u, v, self.theta)

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (AMH has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return np.clip(u * v / _amh_w(u, v, self.theta), 0.0, 1.0)

    def pdf(self, uv) -> float:
        """c(u,v) = [1 − θ(2−u−v−uv) + θ²(1−u)(1−v)] / W³."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        numer, w = _amh_numer_w(u, v, self.theta)
        return float(numer / w ** 3)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        return np.exp(self.logpdf_array(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v·(1 − θ(1−v)) / W²."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        return float(np.clip(_amh_h(v, u, self.theta), 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """AMH has no tail dependence: λ_L = λ_U = 0."""
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaAMH(tau_k=0.2)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}')
    print(f'tau back : {_amh_tau_from_theta(cop.theta):.6f}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print('tail dep : λ_L = 0,  λ_U = 0')

    # Verify independence at tau=0
    cop0 = CopulaAMH(tau_k=0.0)
    print(f'\ntau=0: theta={cop0.theta:.2e}')
    print(f'  cdf(0.3, 0.7) = {cop0.cdf([0.3, 0.7]):.6f}  [should be 0.21]')

    # Verify negative dependence
    tau_test = _amh_tau_from_theta(-1.0)
    print(f'\ntau(-1) = {tau_test:.4f}  [≈ -0.1817]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
