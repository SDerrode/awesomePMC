"""
Frank copula — symmetric, captures both positive and negative dependence.

Generator: φ(t) = -log((e^{-θt}-1)/(e^{-θ}-1))
CDF: C(u,v) = -(1/θ) log(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1))
τ_K range: (−1, 1)  [full range via Debye function inversion]

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, Table 4.1 family (4.2.5) and ch. 5 (τ via the Debye function
D₁); Genest, C. (1987). Frank's family of bivariate distributions.
*Biometrika* 74(3), 549–555, doi:10.1093/biomet/74.3.549; Hofert, M.,
Mächler, M. & McNeil, A. J. (2012). Likelihood inference for Archimedean
copulas in high dimensions under known margins. *Journal of Multivariate
Analysis* 110, 133–150, doi:10.1016/j.jmva.2012.02.019 (log-scale
evaluation of the Archimedean densities, ``log1mexp``).

Numerics
--------
With ``A = expm1(−θu)``, ``B = expm1(−θv)``, ``E = expm1(−θ)`` and
``x = A·B/E`` the textbook expressions are, exactly,

    C(u,v)      = −log1p(x) / θ
    log c(u,v)  = g(θ) − θ(u+v) − 2·log1p(x),     g(θ) = log(θ / (1 − e^{−θ}))
    h(v|u)      = e^{−θu} · (B/E) / (1 + x)
    inv_h(w|u)  = −log1p(y) / θ,   y = E / (1 + e^{−θu}(1−w)/w)

(Genest 1987; the inverse is the closed form used by every vine library).
Every factor is computed to full *relative* precision, so no difference of
nearly equal quantities is formed — neither at the independence limit θ → 0,
where the textbook ``e^{−θ} − 1`` and the log-space form of audit K-3 both
cancel (``cdf_array`` lost 5 digits at τ = 10⁻⁴), nor in the tails.

The only cancellation left is ``1 + x`` (resp. ``1 + y``) when it is close to
0, which happens for θ > 0 in strong dependence; there, ``log(1+x)`` is taken
from the log-space kernel of audit K-3, in which every sum has positive
terms:

    log(1+x) = log den − log(1 − e^{−θ}),
    den = e^{−θ}·[expm1(θ(1−u)) + e^{θ(1−v)}·(−expm1(−θu))].

The switch is at ``x = −½``, where both forms are accurate. For θ < 0 all of
``A, B, E, x, y`` are positive, so nothing cancels; they are carried as
logarithms (``log expm1(a) = a + log1mexp(a)``) to avoid overflow and
``log1p(x)`` is a ``logaddexp(0, log x)``. ``log1mexp(a) = log(1 − e^{−a})``
switches between ``log(−expm1(−a))`` and ``log1p(−e^{−a})`` at a = log 2
(Hofert, Mächler & McNeil 2012). ``g(θ)`` uses its Taylor series
θ/2 − θ²/24 + θ⁴/2880 for |θ| ≤ 10⁻².

For |θ| ≤ 10⁻¹² (``_FRANK_THETA_SERIES``) the first-order expansion is used,
Frank = FGM(α = θ/2) + O(θ²):  log c = α(1−2u)(1−2v),
C = uv[1 + α(1−u)(1−v)],  h = v[1 + α(1−2u)(1−v)],
inv_h = w[1 − α(1−2u)(1−w)]; its relative error on C, h and c is O(θ²)
≤ 10⁻²⁴ and on log c ≈ θ/10 ≤ 10⁻¹³ (away from the zero lines u or v = ½),
and it keeps θ = 0 exact.

τ(θ) = 1 − (4/θ²)∫₀^θ [1 − t/(e^t − 1)] dt is taken by quadrature for
|θ| > 0.1 and from its Debye series τ = θ/9 − θ³/900 + θ⁵/52920 − θ⁷/2721600
below (relative truncation error < 10⁻¹⁵), where the quadrature form cancels;
the series is also inverted (Newton) for small τ, so a τ of 10⁻¹² gets
θ = 9·10⁻¹² instead of being rounded to independence.

Kernels are verified against the high-precision references of
``pmcprg/tests/test_copula_limits.py`` (Python ``decimal``) at u, v ∈
{10⁻¹², 10⁻⁶, 0.3, 0.5, 1 − 10⁻⁶, 1 − 10⁻¹²} for τ ∈ {−0.7, 0.3, 0.9} and the
independence limit τ ∈ {−10⁻⁸, 10⁻¹², 10⁻⁴}.
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize  import root_scalar

from pmcprg.copulas._base import CopulaVirt
from pmcprg.numerics   import EPS, MIN_POSITIVE, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)

_LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Kendall's tau ↔ theta relationship for the Frank copula
# ---------------------------------------------------------------------------

def _frank_tau_integrand(t: float) -> float:
    """f(t) = 1 − t/(e^t − 1).  Limit f(0) = 0 via series."""
    if abs(t) < 1e-7:
        # 1 − t/(e^t−1) = t/2 − t²/12 + t⁴/720 − ...
        return t * (0.5 + t * (-1.0 / 12.0 + t * t / 720.0))
    if t > 0.0:                                   # t e^{−t}/(1 − e^{−t}): no overflow
        return 1.0 - t * math.exp(-t) / -math.expm1(-t)
    return 1.0 - t / math.expm1(t)


# Below this |θ| τ(θ) is taken from its Debye series (module docstring).
_FRANK_TAU_SERIES_THETA = 0.1


def _frank_tau_series(theta: float) -> float:
    t2 = theta * theta
    return theta * (1.0 / 9.0 + t2 * (-1.0 / 900.0 + t2 * (1.0 / 52920.0 - t2 / 2721600.0)))


def kendall_tau_frank(theta: float) -> float:
    """τ_K as a function of θ for the Frank copula.

    τ = 1 − (4/θ²) ∫_0^θ [1 − t/(e^t − 1)] dt

    The integrand vanishes at t = 0, which avoids the cancellation of the
    equivalent 1 − D₁(θ) form; for |θ| ≤ 0.1 the difference 1 − (4/θ²)∫ still
    cancels (τ ≈ θ/9), so the Debye series is used there.
    """
    if theta == 0.0:
        return 0.0
    if abs(theta) <= _FRANK_TAU_SERIES_THETA:
        return _frank_tau_series(theta)
    integral, _ = quad(_frank_tau_integrand, 0.0, theta, epsabs=0.0, epsrel=1e-13)
    return 1.0 - (4.0 / (theta * theta)) * integral


# Largest |θ| at which kendall_tau_frank is still numerically well-behaved.
# Beyond θ ≈ 745 the term exp(−θ) underflows to 0.0 in float64 and the τ
# integral loses meaning; |τ| is already ≈ 0.994 at θ = 700. The log-space
# evaluation below is verified up to this bound (module docstring), so the
# cap costs no practically useful dependence.
_FRANK_THETA_MAX = 700.0

# Largest |τ| reachable at θ = ±θ_max. kendall_tau_frank is odd in θ
# (τ(−θ) = −τ(θ)), so a single quad at import time covers both signs — this
# used to be recomputed at every CopulaFrank construction (audit N-10).
_FRANK_TAU_MAX = abs(kendall_tau_frank(_FRANK_THETA_MAX))

# find_theta_frank warns only when |τ| exceeds _FRANK_TAU_MAX by more than this.
# A τ stored at the bound by the package (CopulaEnum.constructible_tau_range,
# CopulaEnum.reachable_tau) is exactly _FRANK_TAU_MAX where it was written, but
# the quadrature above may round a few ulps differently on another platform;
# such a τ must reload silently (θ = θ_max either way).
_FRANK_TAU_WARN_TOL = 1e-12

# At or below this |θ| the first-order (FGM) expansion is used (module docstring).
_FRANK_THETA_SERIES = 1e-12


def find_theta_frank(tau_target: float) -> float:
    """Invert :func:`kendall_tau_frank`.

    τ = 0 → θ = 0 (independence). For |τ| ≤ τ(0.1) the Debye series is
    inverted by Newton's method from θ₀ = 9τ; above, Brent's method brackets
    ``[0.1, θ_max]`` (|τ| up to ≈ 0.994).

    If ``|tau_target|`` exceeds the largest value reachable with a stable θ,
    it is **clamped** to that maximum (with a warning; none at the maximum
    itself, where the package stores fitted values) instead of raising —
    matching the graceful τ-saturation of the other archimedean families and
    the ``[EPS_MINUS_ONE, ONE_MINUS_EPS]`` τ-range the registry advertises for
    Frank. Previously the bracket was hard-capped at θ = 500 (|τ| ≈ 0.992), so
    any ``tau_k`` in the gap (0.992, 1) — reachable from a TOML file or an ICE
    fit in strong dependence — raised ``ValueError: f(a) and f(b) must have
    different signs``.
    """
    if tau_target == 0.0:
        return 0.0  # independence

    sign    = 1.0 if tau_target > 0.0 else -1.0
    tau_abs = abs(tau_target)

    if tau_abs >= _FRANK_TAU_MAX:
        if tau_abs > _FRANK_TAU_MAX + _FRANK_TAU_WARN_TOL:
            logger.warning(
                'Frank: requested |tau|=%.6f exceeds the largest numerically '
                'reachable value %.6f (theta=±%.0f); clamping.',
                tau_abs, _FRANK_TAU_MAX, _FRANK_THETA_MAX,
            )
        return sign * _FRANK_THETA_MAX

    if tau_abs <= _frank_tau_series(_FRANK_TAU_SERIES_THETA):
        theta = 9.0 * tau_abs
        for _ in range(20):
            t2 = theta * theta
            dtau = (1.0 / 9.0 + t2 * (-3.0 / 900.0 + t2 * (5.0 / 52920.0 - 7.0 * t2 / 2721600.0)))
            step = (_frank_tau_series(theta) - tau_abs) / dtau
            theta -= step
            if abs(step) <= 4e-16 * theta:
                break
        return sign * theta

    sol = root_scalar(
        lambda theta: kendall_tau_frank(theta) - tau_abs,
        bracket=(_FRANK_TAU_SERIES_THETA, _FRANK_THETA_MAX), method='brentq',
        xtol=1e-14,
    )
    if not sol.converged:
        raise ValueError(f'Frank theta inversion did not converge for tau={tau_target}.')
    return sign * sol.root


# ---------------------------------------------------------------------------
# Kernels (module docstring). All take numpy arrays u, v, w in (0, 1).
# ---------------------------------------------------------------------------

def _log1mexp(a):
    """log(1 − e^{−a}) for a > 0 (switch at log 2; Hofert, Mächler & McNeil 2012)."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide='ignore'):          # the branch np.where discards
        return np.where(a <= _LOG2, np.log(-np.expm1(-a)), np.log1p(-np.exp(-a)))


def _log_expm1(a):
    """log(e^a − 1) for a > 0, without overflow."""
    return a + _log1mexp(a)


def _frank_g(th: float) -> float:
    """g(θ) = log(θ / (1 − e^{−θ})), θ ≠ 0, accurate as θ → 0."""
    if abs(th) <= 1e-2:
        t2 = th * th
        return th / 2.0 - t2 / 24.0 + t2 * t2 / 2880.0
    if th > 0.0:
        return math.log(th) - float(_log1mexp(th))
    return math.log(-th) - float(_log_expm1(-th))


def _frank_log1px_pos(th, u, v):
    """θ > 0: (log(1 + x), log(B/E)) with x = A·B/E ∈ (−1, 0]."""
    A = np.expm1(-th * u)
    BE = np.expm1(-th * v) / math.expm1(-th)
    x = A * BE
    with np.errstate(divide='ignore'):          # x = −1 is replaced below
        L = np.log1p(x)
    far = x < -0.5
    if np.any(far):
        uf, vf = u[far], v[far]
        a = np.log(np.expm1(th * (1.0 - uf)))
        b = th * (1.0 - vf) + np.log(-np.expm1(-th * uf))
        L[far] = -th + np.logaddexp(a, b) - float(_log1mexp(th))
    return L, np.log(BE)


def _frank_log1px_neg(a, u, v):
    """θ = −a < 0: (log(1 + x), log(B/E)) with x = A·B/E > 0, in logs."""
    lBE = _log_expm1(a * v) - float(_log_expm1(a))
    lx = _log_expm1(a * u) + lBE
    return np.logaddexp(0.0, lx), lBE


def _frank_logpdf(th, u, v):
    if abs(th) <= _FRANK_THETA_SERIES:
        return 0.5 * th * (1.0 - 2.0 * u) * (1.0 - 2.0 * v)
    if th > 0.0:
        L, _ = _frank_log1px_pos(th, u, v)
    else:
        L, _ = _frank_log1px_neg(-th, u, v)
    return _frank_g(th) - th * (u + v) - 2.0 * L


def _frank_cdf(th, u, v):
    if abs(th) <= _FRANK_THETA_SERIES:
        return u * v * (1.0 + 0.5 * th * (1.0 - u) * (1.0 - v))
    if th > 0.0:
        L, _ = _frank_log1px_pos(th, u, v)
        return -L / th
    L, _ = _frank_log1px_neg(-th, u, v)
    return L / -th


def _frank_h(th, v, u):
    """h(v|u) = ∂C/∂u."""
    if abs(th) <= _FRANK_THETA_SERIES:
        return v * (1.0 + 0.5 * th * (1.0 - 2.0 * u) * (1.0 - v))
    if th > 0.0:
        L, lBE = _frank_log1px_pos(th, u, v)
    else:
        L, lBE = _frank_log1px_neg(-th, u, v)
    return np.exp(-th * u + lBE - L)


def _frank_inv_h(th, w, u):
    """Solve h(v|u) = w for v (module docstring)."""
    if abs(th) <= _FRANK_THETA_SERIES:
        return w * (1.0 - 0.5 * th * (1.0 - 2.0 * u) * (1.0 - w))
    log_r = np.log1p(-w) - np.log(w)                  # r = (1 − w)/w
    if th > 0.0:
        # y = E/D, D = 1 + r e^{−θu} ≥ 1, y ∈ (−1, 0)
        y = math.expm1(-th) / (1.0 + np.exp(log_r - th * u))
        with np.errstate(divide='ignore'):      # y = −1 is replaced below
            L = np.log1p(y)
        far = y < -0.5
        if np.any(far):
            ell = log_r[far] - th * u[far]
            # 1 + y = (r e^{−θu} + e^{−θ}) / D — two positive terms
            L[far] = np.logaddexp(ell, -th) - np.logaddexp(0.0, ell)
        return -L / th
    a = -th
    ly = float(_log_expm1(a)) - np.logaddexp(0.0, log_r + a * u)   # y = E/D > 0
    return np.logaddexp(0.0, ly) / a


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

def _pair(uv):
    return (np.array([minmaxEPS(uv[0])]), np.array([minmaxEPS(uv[1])]))


class CopulaFrank(CopulaVirt):
    """Frank copula — symmetric, positive and negative dependence.

    θ ↔ τ_K via the Debye function:  τ = 1 − 4/θ · (1 − D_1(θ)).
    """

    # θ is capped at ±_FRANK_THETA_MAX: the τ the family represents stops at
    # ±_FRANK_TAU_MAX, below the registered range (−1 + ε, 1 − ε). Cuts
    # CopulaEnum.constructible_tau_range, where fitted / drawn τ are clipped.
    reachable_tau_abs = _FRANK_TAU_MAX

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        tau = self.params['tau_k']
        self.theta = find_theta_frank(tau)
        if abs(tau) > _FRANK_TAU_MAX:
            # find_theta_frank clamped θ; clip the stored τ to the value the
            # clamped θ actually models, so traces / saved TOMLs / the GUI do
            # not overstate the dependence (audit N-9, RB-10).
            self.params['tau_k'] = float(np.copysign(_FRANK_TAU_MAX, tau))

    def pdf(self, uv):
        """``exp`` of the native log-density — no floor."""
        u, v = _pair(uv)
        with np.errstate(over='ignore', under='ignore'):
            return float(np.exp(_frank_logpdf(self.theta, u, v))[0])

    def cdf(self, uv):
        u, v = _pair(uv)
        return float(np.clip(_frank_cdf(self.theta, u, v), 0.0, 1.0)[0])

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised PDF, ``exp`` of :meth:`logpdf_array` (no floor)."""
        with np.errstate(over='ignore', under='ignore'):
            return np.exp(self.logpdf_array(uv))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF (module docstring), exact as θ → 0 and in log space for large |θ|."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return _frank_logpdf(self.theta, u, v)

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised CDF; C = −log1p(x)/θ (module docstring). Rounding is clipped to [0, 1]."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return np.clip(_frank_cdf(self.theta, u, v), 0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = e^{−θu}(B/E)/(1 + x)  (module docstring); NaN if not computable."""
        uu = np.array([minmaxEPS(u)])
        vv = np.array([minmaxEPS(v)])
        return float(np.minimum(_frank_h(self.theta, vv, uu), 1.0)[0])

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u) (module docstring); NaN if not computable."""
        return float(self.inv_h_array(np.array([w], dtype=float), np.array([u], dtype=float))[0])

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised closed-form inverse h-function; NaN propagates (no fallback).

        w is clipped to [MIN_POSITIVE, 1 − EPS] only — an h-value below EPS
        still inverts to its own v — and v to [EPS, 1 − EPS] like every input.
        """
        w = np.clip(np.asarray(w, dtype=float), MIN_POSITIVE, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        shape = w.shape
        v = _frank_inv_h(self.theta, w.ravel(), u.ravel()).reshape(shape)
        return np.clip(v, EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Frank copula: λ_L = λ_U = 0 (no tail dependence for any finite θ)."""
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaFrank(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [from Debye inversion]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print('tail dep : λ_L = 0,  λ_U = 0  [Frank has no tail dependence]')

    # Test negative tau
    cop_neg = CopulaFrank(tau_k=-0.5)
    print(f'\ntau_k=-0.5 → theta={cop_neg.theta:.6f}')
    print(f'pdf(0.3, 0.7) = {cop_neg.pdf([0.3, 0.7]):.6f}')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
