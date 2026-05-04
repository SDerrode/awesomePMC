"""
Joe copula — upper-tail dependence only.

Generator: φ(t) = -log(1-(1-t)^θ), θ ≥ 1.
CDF: C(u,v) = 1 − [(1−u)^θ + (1−v)^θ − (1−u)^θ(1−v)^θ]^{1/θ}
τ_K range: [0, 1)  (positive dependence only; θ=1 → independence)
λ_L = 0, λ_U = 2 − 2^{1/θ}
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

# Pre-computed index array for the series sum (2000 terms is more than enough;
# the series is O(1/k³) so it converges extremely fast)
_JOE_K = np.arange(1, 2001, dtype=float)


# ---------------------------------------------------------------------------
# Kendall's tau ↔ theta
# ---------------------------------------------------------------------------

def _joe_tau_from_theta(theta: float) -> float:
    """τ_K(θ) for the Joe copula via convergent series.

    τ = 1 − α²·Σ_{k=1}^{2000} 1 / [k·(k+α−1)·(k+α)],   α = 2/θ.

    Verified: τ(θ=1) = 0  (independence),  τ→1 as θ→∞.
    """
    alpha = 2.0 / theta
    terms = 1.0 / (_JOE_K * (_JOE_K + alpha - 1.0) * (_JOE_K + alpha))
    return float(1.0 - alpha ** 2 * np.sum(terms))


def _joe_theta_from_tau(tau_target: float) -> float:
    """Invert τ(θ) for the Joe copula via Brent's method.

    τ=0 → θ=1 (independence).  τ≥1−1e-10 → θ capped at 1e9.
    """
    if tau_target <= 0.0:
        return 1.0
    if tau_target >= 1.0 - 1e-10:
        return 1e9  # near-perfect upper dependence
    return float(brentq(
        lambda th: _joe_tau_from_theta(th) - tau_target,
        1.0 + 1e-12, 1e7, xtol=1e-10,
    ))


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

    def pdf(self, uv):
        """c(u,v) = θ · ū^{θ-1} · v̄^{θ-1} · W^{1/θ-2} · (θ−1+W)

        where  ū=1−u, v̄=1−v, W = ū^θ + v̄^θ − ū^θ·v̄^θ.
        """
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        try:
            ub  = 1.0 - u           # ū
            vb  = 1.0 - v           # v̄
            th  = self.theta
            a   = ub ** th          # ū^θ
            b   = vb ** th          # v̄^θ
            W   = a + b - a * b
            if W <= 0.0:
                raise ValueError(f'W={W} ≤ 0')
            result = (
                th
                * (ub ** (th - 1.0))
                * (vb ** (th - 1.0))
                * (W ** (1.0 / th - 2.0))
                * (th - 1.0 + W)
            )
            if not (np.isfinite(result) and result > 0.0):
                raise ValueError(f'non-finite or non-positive: {result}')
            return float(result)
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            logger.debug(
                'Joe.pdf: %s (θ=%.3f, u=%.3e, v=%.3e) → fallback EPS',
                exc, self.theta, u, v,
            )
            return float(EPS)

    def cdf(self, uv):
        """C(u,v) = 1 − W^{1/θ}."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        try:
            ub = 1.0 - u
            vb = 1.0 - v
            th = self.theta
            a  = ub ** th
            b  = vb ** th
            W  = a + b - a * b
            if W <= 0.0:
                return 1.0   # W→0 ⟹ C→1
            result = 1.0 - W ** (1.0 / th)
            if not np.isfinite(result):
                raise ValueError(f'non-finite: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            logger.debug(
                'Joe.cdf: %s (θ=%.3f, u=%.3e, v=%.3e) → fallback EPS',
                exc, self.theta, u, v,
            )
            return float(EPS)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = W^{1/θ-1} · ū^{θ-1} · (1−v̄^θ)."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        try:
            ub = 1.0 - u
            vb = 1.0 - v
            th = self.theta
            a  = ub ** th
            b  = vb ** th
            W  = a + b - a * b
            if W <= 0.0:
                raise ValueError(f'W={W} ≤ 0')
            result = (W ** (1.0 / th - 1.0)) * (ub ** (th - 1.0)) * (1.0 - b)
            if not np.isfinite(result):
                raise ValueError(f'non-finite: {result}')
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as exc:
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug(
                'Joe.h: %s (θ=%.3f, u=%.3e, v=%.3e) → fallback %.1f',
                exc, self.theta, u, v, fallback,
            )
            return fallback

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
