"""
Frank copula — symmetric, captures both positive and negative dependence.

Generator: φ(t) = -log((e^{-θt}-1)/(e^{-θ}-1))
CDF: C(u,v) = -(1/θ) log(1 + (e^{-θu}-1)(e^{-θv}-1)/(e^{-θ}-1))
τ_K range: (−1, 1)  [full range via Debye function inversion]
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from scipy.integrate import quad
from scipy.optimize  import root_scalar
from statsmodels.distributions.copula.api import FrankCopula

from prg.copulas._base import CopulaVirt
from prg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kendall's tau ↔ theta relationship for the Frank copula
# ---------------------------------------------------------------------------

def _frank_tau_integrand(t: float) -> float:
    """f(t) = 1 − t/(e^t − 1).  Limit f(0) = 0 via series."""
    if abs(t) < 1e-7:
        # 1 − t/(e^t−1) = t/2 − t²/12 + t⁴/720 − ...
        return t * (0.5 + t * (-1.0 / 12.0 + t * t / 720.0))
    return 1.0 - t / (np.exp(t) - 1.0)


def kendall_tau_frank(theta: float) -> float:
    """τ_K as a function of θ for the Frank copula.

    τ = 1 − (4/θ²) ∫_0^θ [1 − t/(e^t − 1)] dt

    Numerically stable for all θ ≠ 0 because the integrand vanishes at t = 0,
    avoiding the catastrophic cancellation of the equivalent 1 − D₁(θ) form.
    """
    if theta == 0.0:
        return 0.0
    integral, _ = quad(_frank_tau_integrand, 0.0, theta)
    return 1.0 - (4.0 / (theta * theta)) * integral


def find_theta_frank(tau_target: float) -> float:
    """Invert kendall_tau_frank via Brent's method.

    Works for |τ| up to ≈ 0.999.  τ = 0 → θ = 0 (independence).
    """
    if abs(tau_target) < 1e-10:
        return 0.0  # independence — FrankCopula accepts theta=0

    if tau_target > 0.0:
        bracket = (1e-9, 500.0)   # τ(500) ≈ 0.992
    else:
        bracket = (-500.0, -1e-9)

    sol = root_scalar(
        lambda theta: kendall_tau_frank(theta) - tau_target,
        bracket=bracket, method='brentq'
    )
    if not sol.converged:
        raise ValueError(f'Frank theta inversion did not converge for tau={tau_target}.')
    return sol.root


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaFrank(CopulaVirt):
    """Frank copula — symmetric, positive and negative dependence.

    θ ↔ τ_K via the Debye function:  τ = 1 − 4/θ · (1 − D_1(θ)).
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = find_theta_frank(self.params['tau_k'])
        self._model = FrankCopula(theta=self.theta)

    def pdf(self, uv):
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = float(self._model.pdf([u, v]))
        if not (np.isfinite(result) and result > 0):
            logger.debug(
                'Frank.pdf: statsmodels retourne %s (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                result, self.theta, u, v,
            )
            return float(EPS)
        return result

    def cdf(self, uv):
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = float(self._model.cdf([u, v]))
        if not np.isfinite(result):
            logger.debug(
                'Frank.cdf: statsmodels retourne %s (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                result, self.theta, u, v,
            )
            return float(EPS)
        return float(np.clip(result, 0.0, 1.0))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = exp(−θu)·(exp(−θv)−1) / [(exp(−θ)−1) + (exp(−θu)−1)·(exp(−θv)−1)].

        For θ ≈ 0 the copula is near-independent: h(v|u) → v.
        """
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        try:
            th = self.theta
            if abs(th) < 1e-10:
                return float(v)  # near-independence
            eu = np.exp(-th * u)
            ev = np.exp(-th * v)
            ed = np.exp(-th) - 1.0
            numer = eu * (ev - 1.0)
            denom = ed + (eu - 1.0) * (ev - 1.0)
            if denom == 0.0 or not np.isfinite(denom):
                raise ZeroDivisionError('denom=0 or non-finite')
            result = numer / denom
            if not np.isfinite(result):
                raise ValueError(f'non-finite result: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug(
                'Frank.h: %s (θ=%.3f, u=%.3e, v=%.3e) → fallback %.1f',
                exc, self.theta, u, v, fallback,
            )
            return fallback

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF — bypasses ``log(max(pdf, EPS))``.

        With g = e^{−θ} − 1, g_u = e^{−θu} − 1, g_v = e^{−θv} − 1:
            c(u, v) = θ · g · e^{−θ(u+v)} / (g + g_u·g_v)²
        and so
            log c = log|θ| + log|g| − θ(u+v) − 2 log|g + g_u g_v|.

        For θ ≈ 0 the copula is near-independence; we fall back to log(1)=0.
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th = self.theta
        if abs(th) < 1e-10:
            return np.zeros_like(u)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            g  = np.exp(-th)        - 1.0
            gu = np.exp(-th * u)    - 1.0
            gv = np.exp(-th * v)    - 1.0
            denom = g + gu * gv
            return (np.log(abs(th))
                    + np.log(np.maximum(np.abs(g),     EPS))
                    + (-th) * (u + v)
                    - 2.0 * np.log(np.maximum(np.abs(denom), EPS)))

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u): solve h(v|u) = w analytically.

        Derivation (from h(v|u) = exp(−θu)·(exp(−θv) − 1) / D, where
        D = (exp(−θ) − 1) + (exp(−θu) − 1)·(exp(−θv) − 1)):

            exp(−θv) = 1 + w·(exp(−θ) − 1) / [exp(−θu) − w·(exp(−θu) − 1)]
            v        = −1/θ · log(1 + w(e^{-θ}-1) / (e^{-θu} - w(e^{-θu}-1)))

        For θ ≈ 0 the copula is near-independent: v = w (Brent fallback would
        suffice but the closed form is numerically stable here too).
        """
        u = minmaxEPS(u)
        w = minmaxEPS(w)
        th = self.theta
        if abs(th) < 1e-10:
            return float(w)   # near-independence
        try:
            eu     = np.exp(-th * u)
            ed     = np.exp(-th) - 1.0
            denom  = eu - w * (eu - 1.0)
            if denom == 0.0 or not np.isfinite(denom):
                raise ZeroDivisionError(f"denom={denom}")
            arg    = 1.0 + w * ed / denom
            if arg <= 0.0 or not np.isfinite(arg):
                raise ValueError(f"non-positive arg={arg}")
            v = -np.log(arg) / th
            return minmaxEPS(float(v))
        except (ZeroDivisionError, ValueError, OverflowError):
            # Fall back to numerical inversion on edge cases
            return super().inv_h(w, u)

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised closed-form inverse h-function."""
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        th = self.theta
        if abs(th) < 1e-10:
            return w.copy()        # near-independence
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            eu    = np.exp(-th * u)
            ed    = np.exp(-th) - 1.0
            denom = eu - w * (eu - 1.0)
            arg   = 1.0 + w * ed / np.where(denom != 0.0, denom, EPS)
            v     = -np.log(np.maximum(arg, EPS)) / th
        return np.clip(np.where(np.isfinite(v), v, w), EPS, ONE_MINUS_EPS)

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
