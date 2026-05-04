"""
Ali-Mikhail-Haq (AMH) copula.

Generator: φ(t) = log((1 − θ(1−t)) / t),   θ ∈ [−1, 1).

CDF:  C(u,v) = u·v / W,            W = 1 − θ(1−u)(1−v)
PDF:  c(u,v) = [1 − θ(2−u−v−uv) + θ²(1−u)(1−v)] / W³
h:    h(v|u) = v·(1 − θ(1−v)) / W²

τ_K = 1 − 2[θ + (1−θ)²·log(1−θ)] / (3θ²)
    (limit τ_K → 0 as θ → 0, i.e. independence)

τ range:  [τ(−1), 1/3)  ≈ [−0.1817, 0.3333)
           (very narrow — AMH is mainly for mild dependence)

λ_L = λ_U = 0  (no tail dependence).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from scipy.optimize import brentq

from prg.copulas._base import CopulaVirt
from prg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# τ_K ↔ θ relationship
# ---------------------------------------------------------------------------

def _amh_tau_from_theta(theta: float) -> float:
    """τ_K(θ) = 1 − 2[θ + (1−θ)²·log(1−θ)] / (3θ²).

    Numerically stable:
    • θ = 0  → τ = 0 (independence).
    • |θ| ≪ 1 → τ ≈ 2θ/3 via Taylor expansion (avoids 0/0).
    """
    if abs(theta) < 1e-8:
        # Taylor: τ ≈ (2/3)θ − (1/3)θ² + ...
        return float((2.0 / 3.0) * theta)

    one_minus_t = 1.0 - theta
    if one_minus_t <= 0.0:
        # θ ≥ 1 is outside domain; return limit
        return 1.0 / 3.0
    log_term = one_minus_t ** 2 * np.log(one_minus_t)
    return float(1.0 - 2.0 * (theta + log_term) / (3.0 * theta ** 2))


def _amh_theta_from_tau(tau_target: float) -> float:
    """Invert _amh_tau_from_theta via Brent.

    τ is monotone increasing in θ on [−1, 1).
    τ(−1) ≈ −0.1817,  τ(1−ε) ≈ 1/3 − ε.
    """
    if abs(tau_target) < 1e-10:
        return 0.0  # independence

    # Tight bracket: θ ∈ [−1+ε, 1−ε]
    lo, hi = -1.0 + 1e-10, 1.0 - 1e-10
    f_lo = _amh_tau_from_theta(lo) - tau_target
    f_hi = _amh_tau_from_theta(hi) - tau_target
    if f_lo * f_hi > 0:
        # tau_target out of effective range → return nearest endpoint
        return lo if f_lo > 0 else hi

    return float(brentq(lambda t: _amh_tau_from_theta(t) - tau_target,
                        lo, hi, xtol=1e-12, rtol=1e-12))


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
        W = 1.0 - self.theta * (1.0 - u) * (1.0 - v)
        if W <= 0.0:
            logger.debug('AMH.cdf: W=%.3e ≤ 0 (θ=%.3f) → fallback EPS', W, self.theta)
            return float(EPS)
        return float(np.clip(u * v / W, 0.0, 1.0))

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF — bypasses ``log(max(pdf, EPS))``.

            log c = log[1 − θ(2−u−v−uv) + θ²(1−u)(1−v)] − 3 log W,
                W = 1 − θ(1−u)(1−v).
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th = self.theta
        W      = np.maximum(1.0 - th * (1.0 - u) * (1.0 - v), EPS)
        numer  = 1.0 - th * (2.0 - u - v - u * v) + th * th * (1.0 - u) * (1.0 - v)
        return np.log(np.maximum(numer, EPS)) - 3.0 * np.log(W)

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (AMH has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        W  = 1.0 - self.theta * (1.0 - u) * (1.0 - v)
        with np.errstate(invalid='ignore', divide='ignore'):
            result = u * v / np.maximum(W, EPS)
        return np.clip(np.where(np.isfinite(result), result, EPS), 0.0, 1.0)

    def pdf(self, uv) -> float:
        """c(u,v) = [1 − θ(2−u−v−uv) + θ²(1−u)(1−v)] / W³."""
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        th = self.theta
        W = 1.0 - th * (1.0 - u) * (1.0 - v)
        if W <= 0.0:
            logger.debug('AMH.pdf: W=%.3e ≤ 0 (θ=%.3f) → fallback EPS', W, th)
            return float(EPS)
        numer = 1.0 - th * (2.0 - u - v - u * v) + th ** 2 * (1.0 - u) * (1.0 - v)
        result = numer / W ** 3
        if not (np.isfinite(result) and result > 0.0):
            logger.debug(
                'AMH.pdf: result=%s (θ=%.3f, u=%.3e, v=%.3e) → fallback EPS',
                result, th, u, v,
            )
            return float(EPS)
        return float(result)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (AMH has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th = self.theta
        W = 1.0 - th * (1.0 - u) * (1.0 - v)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            numer  = 1.0 - th * (2.0 - u - v - u * v) + th ** 2 * (1.0 - u) * (1.0 - v)
            result = numer / np.power(np.maximum(W, EPS), 3)
        return np.where(np.isfinite(result) & (result > 0.0), result, EPS)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v·(1 − θ(1−v)) / W²."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        th = self.theta
        W = 1.0 - th * (1.0 - u) * (1.0 - v)
        if W == 0.0:
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug('AMH.h: W=0 → fallback %.1f', fallback)
            return fallback
        result = v * (1.0 - th * (1.0 - v)) / W ** 2
        if not np.isfinite(result):
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug(
                'AMH.h: result=%s (θ=%.3f, u=%.3e, v=%.3e) → fallback %.1f',
                result, th, u, v, fallback,
            )
            return fallback
        return float(np.clip(result, 0.0, 1.0))

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
