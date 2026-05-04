"""
Archimedean copula A12 (Huard, Evin & Favre 2006, Table 1, family 12).

Generator: φ(t) = (1/t − 1)^θ
CDF:       C(u,v) = 1 / (1 + (U1 + U2)^{1/θ})
           where  U_i = (1/u_i − 1)^θ
θ = 2 / (3(1 − τ_K)),  τ_K ∈ [1/3, 1).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np

from prg.copulas._base import CopulaVirt
from prg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


class CopulaA12(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.params['tau_k']))

    def pdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        inv_th = 1.0 / self.theta
        # Log-space arithmetic prevents intermediate over/underflow for large θ.
        # Root cause: U_i = (1/u_i−1)^θ underflows near u≈1 for large θ, making
        # S≈0, and then S^{1/θ−2} (negative exponent) overflows.
        log_a1  = np.log(1.0 / u1 - 1.0)
        log_a2  = np.log(1.0 / u2 - 1.0)
        log_U1  = self.theta * log_a1
        log_U2  = self.theta * log_a2
        log_max = max(log_U1, log_U2)
        log_S   = log_max + np.log(np.exp(log_U1 - log_max) + np.exp(log_U2 - log_max))
        S1th    = np.exp(log_S * inv_th)
        factor  = (self.theta - 1.0) + (self.theta + 1.0) * S1th
        if factor <= 0.0:
            logger.debug('A12.pdf: factor ≤ 0 (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                         self.theta, u1, u2)
            return float(EPS)
        log_result = (
            log_U1 - np.log(u1) - np.log(1.0 - u1)
            + log_U2 - np.log(u2) - np.log(1.0 - u2)
            + np.log(factor)
            + (inv_th - 2.0) * log_S
            - 3.0 * np.log1p(S1th)
        )
        result = np.exp(log_result)
        if not np.isfinite(result):
            logger.debug('A12.pdf: résultat non-fini (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                         self.theta, u1, u2)
            return float(EPS)
        return float(result)

    def cdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        U1 = np.pow(1.0 / u1 - 1.0, self.theta)
        U2 = np.pow(1.0 / u2 - 1.0, self.theta)
        return float(1.0 / (1.0 + np.pow(U1 + U2, 1.0 / self.theta)))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (A12 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u1 = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        u2 = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th = self.theta
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            U1 = (1.0 / u1 - 1.0) ** th
            U2 = (1.0 / u2 - 1.0) ** th
            result = 1.0 / (1.0 + (U1 + U2) ** (1.0 / th))
        return np.clip(np.where(np.isfinite(result), result, EPS), 0.0, 1.0)

    def _logpdf_array_core(self, uv: np.ndarray) -> np.ndarray:
        """Log-space helper used by both ``pdf_array`` and ``logpdf_array``."""
        uv = np.asarray(uv, dtype=float)
        u1 = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        u2 = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th     = self.theta
        inv_th = 1.0 / th
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            log_a1  = np.log(1.0 / u1 - 1.0)
            log_a2  = np.log(1.0 / u2 - 1.0)
            log_U1  = th * log_a1
            log_U2  = th * log_a2
            log_max = np.maximum(log_U1, log_U2)
            log_S   = log_max + np.log(np.exp(log_U1 - log_max) + np.exp(log_U2 - log_max))
            S1th    = np.exp(log_S * inv_th)
            factor  = (th - 1.0) + (th + 1.0) * S1th
            return (
                log_U1 - np.log(u1) - np.log(1.0 - u1)
                + log_U2 - np.log(u2) - np.log(1.0 - u2)
                + np.log(np.maximum(factor, EPS))
                + (inv_th - 2.0) * log_S
                - 3.0 * np.log1p(S1th)
            )

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (A12 has no statsmodels backend)."""
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = np.exp(self._logpdf_array_core(uv))
        return np.where(np.isfinite(result) & (result > 0.0), result, EPS)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF — bypasses ``log(max(pdf, EPS))`` floor."""
        return self._logpdf_array_core(uv)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−2} · S^{1/θ−1} · (1/u−1)^{θ−1} / u².

        For τ → 1 (θ → ∞) the term ``(1/v − 1)^θ`` overflows for any v ≪ u
        (typical in the brentq bracket of :meth:`inv_h`). In that limit the
        copula concentrates on the diagonal v = u, so h(v|u) → 𝟙{v ≥ u}.
        We catch the overflow and return that limiting indicator — the same
        pattern used by :class:`CopulaA14`.
        """
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        try:
            U1 = (1.0/u - 1.0) ** self.theta
            U2 = (1.0/v - 1.0) ** self.theta
            S   = U1 + U2
            S1t = S ** (1.0 / self.theta)
            result = ((1.0 + S1t)**(-2.0) * S**(1.0/self.theta - 1.0) *
                      (1.0/u - 1.0)**(self.theta - 1.0) / u**2)
            if not np.isfinite(result):
                fallback = 1.0 if v >= u else 0.0
                logger.debug(
                    "A12.h: result non-fini (θ=%.3f, u=%.3e, v=%.3e) → fallback %.1f",
                    self.theta, u, v, fallback,
                )
                return fallback
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            fallback = 1.0 if v >= u else 0.0
            logger.debug(
                "A12.h: exception numérique (θ=%.3f, u=%.3e, v=%.3e): %s → fallback %.1f",
                self.theta, u, v, e, fallback,
            )
            return fallback

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaA12(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2/(3(1−τ))]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    lL, lU = cop.tail_dependence()
    print(f'tail dep : λ_L = {lL:.4f}  [= 2^(−1/θ)],  λ_U = {lU:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
