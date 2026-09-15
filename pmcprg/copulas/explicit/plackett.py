"""
Plackett copula — defined by the constant cross-product ratio θ > 0.

CDF:  C(u,v) = [S − √Δ] / (2(θ−1))  for θ ≠ 1,   else u·v
      S = 1 + (θ−1)(u+v),   Δ = S² − 4θ(θ−1)·u·v

PDF:  c(u,v) = θ·[1 + (θ−1)(u+v−2uv)] / Δ^{3/2}

h:    h(v|u) = [1 − (S − 2θv)/√Δ] / 2   (= v at θ=1)

ρ_S = (θ+1)/(θ−1) − 2θ·log(θ)/(θ−1)²     (Spearman's ρ, Nelsen 2006, ch. 5;
     limit ρ_S → 0 as θ → 1, i.e. independence)
Kendall's τ of the Plackett copula has **no closed form**: τ(θ) is obtained
here by Gauss–Legendre quadrature of Hoeffding's identity
τ = 1 − 4∫∫ h(v|u)·h(u|v) du dv, and θ(τ) by monotone interpolation of a
cached (τ, log θ) table (``_plackett_tau_from_theta``,
``_plackett_theta_from_tau``). The ρ_S line above used to be labelled τ_K
(audit K-10).

τ range: (−1, 1)  (positive and negative dependence, full range).
θ range: (0, ∞),  θ = 1 → independence. θ is capped at 10^{±6} numerically,
where the table ends at τ = −0.9935245713002141 and 0.9935245713002134 (not
exact opposites: two separate quadratures); the tabulated τ(θ) is accurate
to about 1e-3 at τ = 0.99 and 3e-3 at τ = 0.993. A requested τ beyond the
table is clamped and the copula then stores the τ it realises,
``_plackett_tau_from_theta(θ)`` (audit RB-10) — the table end.
``CopulaPlackett.reachable_tau_bounds`` declares those ends, so the package
never produces or stores a Plackett τ beyond them (fits, ICE/SEM M-step,
multistart draws: :meth:`CopulaEnum.reachable_tau`, G2).

λ_L = λ_U = 0  (no tail dependence).

Numerics
--------
All three textbook expressions cancel near independence (θ → 1: ``S − √Δ``
and ``θ − 1`` both vanish, so ``cdf_array`` lost 8 digits at τ = −10⁻⁸) and
``h`` cancels in its lower tail. With η = θ − 1, a = u(1−v) + v(1−u) and
d = (u − v)², every quantity below is a sum of terms of one sign:

    Δ       = 1 + t₁,   t₁ = η(2a + η·d)                      (η ≥ −½; 2a ≥ d)
            = S² + 4θ(−η)uv                                    (η < −½)
    S       = (1 − u − v) + θ(u + v),   1 − u − v taken as (1 − max(u,v)) − min(u,v)
    log c   = log θ + log1p(η·a) − (3/2)·log1p(t₁)            (η ≥ −½)
            = log θ + log(θa + (1−u)(1−v) + uv) − (3/2)·log Δ  (η < −½)
    C       = 2θuv / (S + √Δ)          if S ≥ 0   (rationalised numerator)
            = (S − √Δ) / (2η)          if S < 0   (then η < 0)
    h(v|u)  = 2θv(1−v) / (√Δ(√Δ + T))  if T ≥ 0,   T = S − 2θv = (1 − u − v) + θ(u − v)
            = (√Δ − T) / (2√Δ)         if T < 0

The density is the algebraic form of Joe (1997, p. 141) used by R
``copula`` (technique 9 of the numerics notes); C and h are rationalised the
same way, using S² − Δ = 4θηuv and Δ − T² = 4θv(1 − v). They are exact at
θ = 1 (C = uv, h = v, c = 1) without a special case, and verified against
the high-precision references of ``pmcprg/tests/test_copula_limits.py``.

Reference: Plackett, R. L. (1965). A class of bivariate distributions.
*Journal of the American Statistical Association* 60(310), 516–522,
doi:10.1080/01621459.1965.10480807; Nelsen, R. B. (2006). *An Introduction
to Copulas*, 2nd ed., Springer, ch. 3 (Plackett distributions) and ch. 5
(ρ_S; τ has no closed form); Joe, H. (1997). *Multivariate Models and
Dependence Concepts*, Chapman & Hall/CRC, doi:10.1201/b13150 (density).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math
from functools import lru_cache

import numpy as np

from pmcprg.copulas._base import CopulaVirt
from pmcprg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cancellation-free kernels (module docstring). u, v are numpy arrays in (0, 1).
# ---------------------------------------------------------------------------

def _one_minus_sum(u, v):
    """1 − u − v as (1 − max) − min: exact subtraction when max ≥ ½ (Sterbenz)."""
    return (1.0 - np.maximum(u, v)) - np.minimum(u, v)


def _plackett_parts(theta: float, u, v):
    """(η, a, S, Δ, t₁ or None) — t₁ is returned when Δ = 1 + t₁ is the form used."""
    eta = theta - 1.0
    a = u * (1.0 - v) + v * (1.0 - u)
    S = _one_minus_sum(u, v) + theta * (u + v)
    if eta >= -0.5:
        t1 = eta * (2.0 * a + eta * (u - v) ** 2)
        return eta, a, S, 1.0 + t1, t1
    return eta, a, S, S * S + 4.0 * theta * (-eta) * u * v, None


def _plackett_logpdf(theta: float, u, v):
    eta, a, _, delta, t1 = _plackett_parts(theta, u, v)
    if t1 is not None:
        return math.log1p(eta) + np.log1p(eta * a) - 1.5 * np.log1p(t1)
    return (math.log(theta) + np.log(theta * a + (1.0 - u) * (1.0 - v) + u * v)
            - 1.5 * np.log(delta))


def _plackett_cdf(theta: float, u, v):
    eta, _, S, delta, _ = _plackett_parts(theta, u, v)
    root = np.sqrt(delta)
    pos = S >= 0.0
    out = np.empty(np.broadcast(u, v).shape)
    out[pos] = 2.0 * theta * (u * v)[pos] / (S[pos] + root[pos])
    if not np.all(pos):                            # S < 0 only when η < 0
        neg = ~pos
        out[neg] = (S[neg] - root[neg]) / (2.0 * eta)
    return out


def _plackett_h(v, u, theta):
    """h(v|u) = ∂C/∂u (module docstring), well defined at θ = 1 (h = v)."""
    _, _, _, delta, _ = _plackett_parts(theta, u, v)
    root = np.sqrt(delta)
    T = _one_minus_sum(u, v) + theta * (u - v)
    return np.where(T >= 0.0,
                    2.0 * theta * v * (1.0 - v) / (root * (root + np.abs(T))),
                    (root - T) / (2.0 * root))


# ---------------------------------------------------------------------------
# τ_K ↔ θ relationship
# ---------------------------------------------------------------------------

# Unlike its Spearman ρ, the Plackett copula has **no closed-form Kendall τ**.
# This module used to return the ρ_S expression from a function named
# ``_plackett_tau_from_theta``, so every Plackett was parameterised by
# Spearman's ρ while the rest of the package speaks Kendall τ:
# ``CopulaPlackett(tau_k=0.5)`` actually realised τ ≈ 0.35 (and ρ_S = 0.50).
# τ is now computed from Hoeffding's identity
#     τ = 1 − 4·∫∫ h(v|u)·h(u|v) du dv,
# whose integrand is bounded in [0,1] — far better conditioned than the
# equivalent C·c form, whose Δ^{-3/2} spike defeats a fixed quadrature grid.

_GL_NODES = 200                 # Gauss–Legendre nodes per dimension.
_PLACKETT_THETA_MAX = 1.0e6     # Table ends at |τ| ≈ 0.9935; the quadrature
                                # τ(θ) is only good to ~1e-3 at τ = 0.99 and
                                # ~3e-3 at 0.993 (audit K-10); a fixed
                                # 200-node grid cannot resolve the ridge of
                                # the density beyond.
_TABLE_POINTS = 200             # grid points per side of θ = 1.

# _plackett_theta_from_tau warns only when τ is beyond a table end by more than
# this. A τ stored at the end by the package (CopulaEnum.constructible_tau_range,
# CopulaEnum.reachable_tau) is exactly that end where it was written, but the
# quadrature may round a few ulps differently on another platform; such a τ
# must reload silently (θ = 10^{±6} either way). Same rule as Frank's
# _FRANK_TAU_WARN_TOL.
_PLACKETT_TAU_WARN_TOL = 1e-12


@lru_cache(maxsize=1)
def _gl_square():
    """Gauss–Legendre tensor grid on (0,1)², built once."""
    x, w = np.polynomial.legendre.leggauss(_GL_NODES)
    u  = 0.5 * (x + 1.0)
    wu = 0.5 * w
    U, V = np.meshgrid(u, u, indexing='ij')
    return U, V, np.outer(wu, wu)


def _plackett_rho_s_from_theta(theta: float) -> float:
    """Spearman's ρ_S(θ) = (θ+1)/(θ−1) − 2θ·log θ/(θ−1)² — closed form.

    Kept because it is the classical Plackett association measure (and the
    expression this module once mislabelled as Kendall's τ).
    """
    if theta <= 0.0:
        return -1.0
    eps = theta - 1.0
    if abs(eps) < 1e-7:
        return float(eps / 3.0 - eps ** 2 / 6.0 + eps ** 3 / 10.0)
    return float((theta + 1.0) / eps - 2.0 * theta * np.log(theta) / eps ** 2)


def _plackett_tau_from_theta(theta: float) -> float:
    """Kendall's τ(θ) by quadrature of Hoeffding's identity."""
    if theta <= 0.0:
        return -1.0
    if abs(theta - 1.0) < 1e-12:
        return 0.0
    U, V, W = _gl_square()
    tau = 1.0 - 4.0 * float(np.sum(W * _plackett_h(V, U, theta)
                                     * _plackett_h(U, V, theta)))
    return float(np.clip(tau, -1.0, 1.0))


@lru_cache(maxsize=1)
def _plackett_tau_table():
    """Strictly increasing (τ, log θ) table used to invert τ(θ).

    Built once (~0.3 s). Inverting by interpolation instead of running Brent
    on the quadrature keeps construction at a few µs, which matters because
    ICE rebuilds copulas inside its M-step loop.
    """
    log_max = np.log(_PLACKETT_THETA_MAX)
    log_th  = np.concatenate([
        np.linspace(-log_max, 0.0, _TABLE_POINTS, endpoint=False),
        np.zeros(1),
        np.linspace(0.0, log_max, _TABLE_POINTS)[1:],
    ])
    taus = np.array([_plackett_tau_from_theta(float(np.exp(t))) for t in log_th])
    return taus, log_th


def _plackett_theta_from_tau(tau_target: float) -> float:
    """Invert τ(θ) by monotone interpolation of the cached table.

    A τ outside the table ``[τ(10⁻⁶), τ(10⁶)]`` is **clamped** to the table end
    with a warning rather than raising — matching the graceful saturation of
    the other families (see
    :func:`pmcprg.copulas.archimedean.frank.find_theta_frank`). No warning
    within ``_PLACKETT_TAU_WARN_TOL`` of the end, where the package stores
    fitted values.
    """
    if abs(tau_target) < 1e-10:
        return 1.0  # independence

    taus, log_th = _plackett_tau_table()
    if not (taus[0] <= tau_target <= taus[-1]):
        if not (taus[0] - _PLACKETT_TAU_WARN_TOL <= tau_target
                <= taus[-1] + _PLACKETT_TAU_WARN_TOL):
            logger.warning(
                'Plackett: requested tau=%.6f is outside the numerically reachable '
                'range [%.6f, %.6f] (theta in [1e-6, 1e6]); clamping.',
                tau_target, taus[0], taus[-1],
            )
        tau_target = float(np.clip(tau_target, taus[0], taus[-1]))

    return float(np.exp(np.interp(tau_target, taus, log_th)))


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaPlackett(CopulaVirt):
    """Plackett copula.

    Parameters
    ----------
    tau_k : float
        Kendall's τ.  Full range (−1, 1); beyond the table ends (|τ| ≈ 0.9935)
        θ is clamped and ``params['tau_k']`` becomes the realised τ.
    """

    @classmethod
    def reachable_tau_bounds(cls) -> tuple[float, float]:
        """The table ends ``(τ(10⁻⁶), τ(10⁶))`` — the τ a clamped copula stores.

        Not opposite (−0.9935245713002141 / 0.9935245713002134) and computed on
        first use of the table, hence a method rather than ``reachable_tau_abs``.
        """
        taus, _ = _plackett_tau_table()
        return float(taus[0]), float(taus[-1])

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        tau = self.params['tau_k']
        self.theta = _plackett_theta_from_tau(tau)
        taus, _ = _plackett_tau_table()
        if not (taus[0] <= tau <= taus[-1]):
            # θ was clamped: store the τ this θ realises, not the one requested
            # (audit RB-10), so traces and saved models do not overstate it.
            self.params['tau_k'] = _plackett_tau_from_theta(self.theta)

    @staticmethod
    def _clip(uv):
        uv = np.asarray(uv, dtype=float)
        return (np.clip(uv[:, 0], EPS, ONE_MINUS_EPS),
                np.clip(uv[:, 1], EPS, ONE_MINUS_EPS))

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv) -> float:
        """C(u,v), rationalised (module docstring)."""
        u = np.array([minmaxEPS(uv[0])])
        v = np.array([minmaxEPS(uv[1])])
        return float(np.clip(_plackett_cdf(self.theta, u, v), 0.0, 1.0)[0])

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised CDF (module docstring); rounding is clipped to [0, 1]."""
        u, v = self._clip(uv)
        return np.clip(_plackett_cdf(self.theta, u, v), 0.0, 1.0)

    def pdf(self, uv) -> float:
        """``exp`` of the native log-density — no floor."""
        u = np.array([minmaxEPS(uv[0])])
        v = np.array([minmaxEPS(uv[1])])
        return float(np.exp(_plackett_logpdf(self.theta, u, v))[0])

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        with np.errstate(over='ignore', under='ignore'):
            return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF: log θ + log[1 + η(u+v−2uv)] − (3/2) log Δ, without cancellation."""
        u, v = self._clip(uv)
        return _plackett_logpdf(self.theta, u, v)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = ∂C/∂u, rationalised (module docstring). Equals v at θ = 1."""
        uu = np.array([minmaxEPS(u)])
        vv = np.array([minmaxEPS(v)])
        return float(np.minimum(_plackett_h(vv, uu, self.theta), 1.0)[0])

    def tail_dependence(self) -> tuple[float, float]:
        """Plackett copula: λ_L = λ_U = 0 (no tail dependence)."""
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaPlackett(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}')
    print(f'tau back : {_plackett_tau_from_theta(cop.theta):.6f}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print('tail dep : λ_L = 0,  λ_U = 0')

    # Independence at tau=0
    cop0 = CopulaPlackett(tau_k=0.0)
    print(f'\ntau=0: theta={cop0.theta:.6f}')
    print(f'  cdf(0.3, 0.7) = {cop0.cdf([0.3, 0.7]):.6f}  [should be 0.21]')
    print(f'  pdf(0.3, 0.7) = {cop0.pdf([0.3, 0.7]):.6f}  [should be 1.0]')

    # Negative tau
    cop_neg = CopulaPlackett(tau_k=-0.5)
    print(f'\ntau=-0.5: theta={cop_neg.theta:.6f}')
    print(f'  pdf(0.3, 0.7) = {cop_neg.pdf([0.3, 0.7]):.6f}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
