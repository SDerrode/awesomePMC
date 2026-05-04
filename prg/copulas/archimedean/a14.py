"""
Archimedean copula A14 (Huard, Evin & Favre 2006, Table 1, family 14).

CDF:  C(u,v) = (1 + (U1 + U2)^{1/θ})^{-θ}
      where  U_i = (u_i^{-1/θ} − 1)^θ
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


class CopulaA14(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.params['tau_k']))

    def pdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        try:
            U1  = pow(pow(u0, -1.0 / self.theta) - 1.0, self.theta)
            U2  = pow(pow(u1, -1.0 / self.theta) - 1.0, self.theta)
            S   = U1 + U2
            S1t = pow(S, 1.0 / self.theta)
            result = (
                U1 * pow(S, 1.0 / self.theta - 2.0) *
                U2 * pow(1.0 + S1t, -2.0 - self.theta) /
                (self.theta * u0 * u1 *
                 (pow(u0, 1.0 / self.theta) - 1.0) *
                 (pow(u1, 1.0 / self.theta) - 1.0)) *
                (self.theta - 1.0 + 2.0 * self.theta * S1t)
            )
            if not (np.isfinite(result) and result > 0):
                logger.debug('A14.pdf: résultat non-fini/négatif (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                             self.theta, u0, u1)
                return float(EPS)
            return float(result)
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            # Cas limites : u ou v → 1 avec θ grand → densité ≈ 0 en frontière
            logger.debug('A14.pdf: exception numérique (θ=%.3f, u=(%.3e,%.3e)): %s → fallback EPS',
                         self.theta, u0, u1, e)
            return float(EPS)

    def cdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        U1 = pow(pow(u0, -1.0 / self.theta) - 1.0, self.theta)
        U2 = pow(pow(u1, -1.0 / self.theta) - 1.0, self.theta)
        return float(pow(1.0 + pow(U1 + U2, 1.0 / self.theta), -self.theta))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (A14 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th     = self.theta
        inv_th = 1.0 / th
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            U1 = (u ** (-inv_th) - 1.0) ** th
            U2 = (v ** (-inv_th) - 1.0) ** th
            result = (1.0 + (U1 + U2) ** inv_th) ** (-th)
        return np.clip(np.where(np.isfinite(result), result, EPS), 0.0, 1.0)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (A14 has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th     = self.theta
        inv_th = 1.0 / th
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            U1  = (u ** (-inv_th) - 1.0) ** th
            U2  = (v ** (-inv_th) - 1.0) ** th
            S   = U1 + U2
            S1t = S ** inv_th
            denom = (
                th * u * v
                * (u ** inv_th - 1.0)
                * (v ** inv_th - 1.0)
            )
            result = (
                U1 * S ** (inv_th - 2.0)
                * U2 * (1.0 + S1t) ** (-2.0 - th)
                / np.where(denom != 0.0, denom, EPS)
                * (th - 1.0 + 2.0 * th * S1t)
            )
        return np.where(np.isfinite(result) & (result > 0.0), result, EPS)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF for A14 — log-space form of the closed-form PDF.

        With U_i = (u_i^{−1/θ} − 1)^θ > 0, S = U_1 + U_2 > 0, and noting
        ``u^{1/θ} − 1 < 0`` for u ∈ (0, 1), we use ``log|u^{1/θ} − 1| =
        log(1 − u^{1/θ})``:

            log c = log U_1 + log U_2 + (1/θ − 2) log S + (−2 − θ) log(1 + S^{1/θ})
                  − log(θ uv) − log(1 − u^{1/θ}) − log(1 − v^{1/θ})
                  + log(θ − 1 + 2θ S^{1/θ}).
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th     = self.theta
        inv_th = 1.0 / th
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            U1  = np.maximum((u ** (-inv_th) - 1.0) ** th, EPS)
            U2  = np.maximum((v ** (-inv_th) - 1.0) ** th, EPS)
            S   = U1 + U2
            S1t = S ** inv_th
            # u^{1/θ} - 1 < 0 for u ∈ (0,1); use log|u^{1/θ} - 1| = log(1 - u^{1/θ}).
            return (
                np.log(U1) + np.log(U2)
                + (inv_th - 2.0) * np.log(S)
                + (-2.0 - th)    * np.log1p(S1t)
                - np.log(th * u * v)
                - np.log(np.maximum(1.0 - u ** inv_th, EPS))
                - np.log(np.maximum(1.0 - v ** inv_th, EPS))
                + np.log(np.maximum(th - 1.0 + 2.0 * th * S1t, EPS))
            )

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−θ−1} · S^{1/θ−1} · (u^{−1/θ}−1)^{θ−1} · u^{−1/θ−1}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        try:
            inv_th = 1.0 / self.theta
            U1  = (u**(-inv_th) - 1.0) ** self.theta
            U2  = (v**(-inv_th) - 1.0) ** self.theta
            S   = U1 + U2
            S1t = S ** inv_th
            result = ((1.0 + S1t)**(-self.theta - 1.0) * S**(inv_th - 1.0) *
                      (u**(-inv_th) - 1.0)**(self.theta - 1.0) * u**(-inv_th - 1.0))
            if not np.isfinite(result):
                fallback = 1.0 if v > 0.5 else 0.0
                logger.debug('A14.h: résultat non-fini (θ=%.3f, u=%.3e, v=%.3e) → fallback %.1f',
                             self.theta, u, v, fallback)
                return fallback
            return float(np.clip(result, 0.0, 1.0))
        except (ZeroDivisionError, ValueError, OverflowError) as e:
            # S → 0 quand u et v → 1 : h(v|u) → 1
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug('A14.h: exception numérique (θ=%.3f, u=%.3e, v=%.3e): %s → fallback %.1f',
                         self.theta, u, v, e, fallback)
            return fallback

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaA14(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2/(3(1−τ))]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    lL, lU = cop.tail_dependence()
    print(f'tail dep : λ_L = {lL:.4f}  [≈ 1/2 for any θ],  λ_U = {lU:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
