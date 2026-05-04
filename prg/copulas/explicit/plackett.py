"""
Plackett copula — defined by the constant cross-product ratio θ > 0.

CDF:  C(u,v) = [S − √Δ] / (2(θ−1))  for θ ≠ 1,   else u·v
      S = 1 + (θ−1)(u+v),   Δ = S² − 4θ(θ−1)·u·v

PDF:  c(u,v) = θ·[1 + (θ−1)(u+v−2uv)] / Δ^{3/2}

h:    h(v|u) = [1 − (S − 2θv)/√Δ] / 2   (= v at θ=1)

τ_K = (θ+1)/(θ−1) − 2θ·log(θ)/(θ−1)²
     (limit τ_K → 0 as θ → 1, i.e. independence)

τ range: (−1, 1)  (positive and negative dependence, full range).
θ range: (0, ∞),  θ = 1 → independence.

λ_L = λ_U = 0  (no tail dependence).
"""
if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from scipy.optimize import brentq

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# τ_K ↔ θ relationship
# ---------------------------------------------------------------------------

def _plackett_tau_from_theta(theta: float) -> float:
    """τ_K(θ) = (θ+1)/(θ−1) − 2θ·log(θ)/(θ−1)².

    Numerically stable:
    • θ = 1  → τ = 0 via Taylor (avoids 0/0).
    • θ → 0+ → τ → −1.
    • θ → ∞  → τ → +1.
    """
    if theta <= 0.0:
        return -1.0
    eps = theta - 1.0
    if abs(eps) < 1e-7:
        # Taylor around θ=1: τ ≈ ε/3 − ε²/6 + ε³/10 − ...
        return float(eps / 3.0 - eps ** 2 / 6.0 + eps ** 3 / 10.0)
    return float((theta + 1.0) / eps - 2.0 * theta * np.log(theta) / eps ** 2)


def _plackett_theta_from_tau(tau_target: float) -> float:
    """Invert _plackett_tau_from_theta via Brent's method.

    τ is monotone increasing in θ:
    • τ < 0 → θ ∈ (0, 1),  bracket (ε, 1−ε)
    • τ = 0 → θ = 1
    • τ > 0 → θ ∈ (1, ∞),  bracket (1+ε, M)
    """
    if abs(tau_target) < 1e-10:
        return 1.0  # independence

    if tau_target > 0.0:
        lo, hi = 1.0 + 1e-10, 1.0e7
    else:
        lo, hi = 1e-10, 1.0 - 1e-10

    f_lo = _plackett_tau_from_theta(lo) - tau_target
    f_hi = _plackett_tau_from_theta(hi) - tau_target
    if f_lo * f_hi > 0:
        # tau_target nearly at boundary → clamp
        return lo if f_lo > 0 else hi

    return float(brentq(lambda t: _plackett_tau_from_theta(t) - tau_target,
                        lo, hi, xtol=1e-12, rtol=1e-12))


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _plackett_delta(theta: float, u: float, v: float) -> tuple[float, float]:
    """Return (S, Δ) for the Plackett CDF.

    S = 1 + (θ−1)(u+v),  Δ = S² − 4θ(θ−1)·uv.

    Returns S and Δ; caller must handle Δ < 0 (should not occur for valid inputs).
    """
    S   = 1.0 + (theta - 1.0) * (u + v)
    Dlt = S * S - 4.0 * theta * (theta - 1.0) * u * v
    return S, Dlt


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaPlackett(CopulaVirt):
    """Plackett copula.

    Parameters
    ----------
    tau_k : float
        Kendall's τ.  Full range (−1, 1).
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = _plackett_theta_from_tau(self.params['tau_k'])

    # ------------------------------------------------------------------
    # CDF / PDF / h-function
    # ------------------------------------------------------------------

    def cdf(self, uv) -> float:
        """C(u,v) = [S − √Δ] / (2(θ−1))  or u·v at θ=1."""
        u  = minmaxEPS(uv[0])
        v  = minmaxEPS(uv[1])
        th = self.theta
        try:
            eps = th - 1.0
            if abs(eps) < 1e-10:
                return float(u * v)  # independence limit

            S, Dlt = _plackett_delta(th, u, v)
            if Dlt < 0.0:
                Dlt = 0.0  # numerical guard against tiny negatives
            result = (S - np.sqrt(Dlt)) / (2.0 * eps)
            if not np.isfinite(result):
                raise ValueError(f'non-finite: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            logger.debug(
                'Plackett.cdf: %s (θ=%.4f, u=%.3e, v=%.3e) → fallback EPS',
                exc, th, u, v,
            )
            return float(EPS)

    def pdf(self, uv) -> float:
        """c(u,v) = θ·[1 + (θ−1)(u+v−2uv)] / Δ^{3/2}."""
        u  = minmaxEPS(uv[0])
        v  = minmaxEPS(uv[1])
        th = self.theta
        try:
            S, Dlt = _plackett_delta(th, u, v)
            if Dlt <= 0.0:
                raise ValueError(f'Δ={Dlt:.3e} ≤ 0')
            numer  = th * (1.0 + (th - 1.0) * (u + v - 2.0 * u * v))
            result = numer / Dlt ** 1.5
            if not (np.isfinite(result) and result > 0.0):
                raise ValueError(f'non-positive or non-finite: {result}')
            return float(result)
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            logger.debug(
                'Plackett.pdf: %s (θ=%.4f, u=%.3e, v=%.3e) → fallback EPS',
                exc, th, u, v,
            )
            return float(EPS)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = [√Δ − (S − 2θv)] / (2√Δ).

        Derived as ∂C/∂u.  Equals v at θ=1 (independence).
        """
        u  = minmaxEPS(u)
        v  = minmaxEPS(v)
        th = self.theta
        try:
            eps = th - 1.0
            if abs(eps) < 1e-10:
                return float(v)  # independence

            S, Dlt = _plackett_delta(th, u, v)
            if Dlt <= 0.0:
                Dlt = 0.0
            sqrt_D = np.sqrt(Dlt)
            if sqrt_D < 1e-15:
                raise ZeroDivisionError(f'√Δ≈0')
            result = (sqrt_D - (S - 2.0 * th * v)) / (2.0 * sqrt_D)
            if not np.isfinite(result):
                raise ValueError(f'non-finite: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug(
                'Plackett.h: %s (θ=%.4f, u=%.3e, v=%.3e) → fallback %.1f',
                exc, th, u, v, fallback,
            )
            return fallback

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
    print(f'tail dep : λ_L = 0,  λ_U = 0')

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
