if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from statsmodels.distributions.copula.api import ClaytonCopula

from prg.copulas._base import CopulaVirt
from prg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS

logger = logging.getLogger(__name__)


class CopulaClayton(CopulaVirt):
    """
    Clayton copula — lower-tail dependence.

    θ = 2 τ_K / (1 − τ_K).
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        tau = self.params['tau_k']
        self.theta = 2.0 * tau / (1.0 - tau)
        self._model = ClaytonCopula(theta=self.theta)

    def pdf(self, uv):
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = float(self._model.pdf([u, v]))
        if not (np.isfinite(result) and result > 0):
            logger.debug('Clayton.pdf: statsmodels retourne %s (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                         result, self.theta, u, v)
            return float(EPS)
        return result

    def cdf(self, uv):
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = float(self._model.cdf([u, v]))
        if not np.isfinite(result):
            logger.debug('Clayton.cdf: statsmodels retourne %s (θ=%.3f, u=(%.3e,%.3e)) → fallback EPS',
                         result, self.theta, u, v)
            return float(EPS)
        return float(np.clip(result, 0.0, 1.0))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = C(u,v)^{1+θ} · u^{−θ−1}.
        Pour u → 0+, Clayton a une dépendance de queue inférieure : h(v|u) → 1.
        """
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        c_uv = self.cdf([u, v])
        result = (c_uv ** (1.0 + self.theta)) * (u ** (-self.theta - 1.0))
        if not np.isfinite(result):
            fallback = 1.0 if v > 0.5 else 0.0
            logger.debug('Clayton.h: résultat non-fini (θ=%.3f, u=%.3e, v=%.3e) → fallback %.1f',
                         self.theta, u, v, fallback)
            return fallback
        return float(np.clip(result, 0.0, 1.0))

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u): solve h(v|u) = w analytically.

        From h(v|u) = (1 + u^θ·(v^{−θ} − 1))^{−1−1/θ} = w:
            v = ((w^{−θ/(1+θ)} − 1)·u^{−θ} + 1)^{−1/θ}
        """
        u  = minmaxEPS(u)
        w  = minmaxEPS(w)
        th = self.theta
        try:
            v = ((w ** (-th / (1.0 + th)) - 1.0) * u ** (-th) + 1.0) ** (-1.0 / th)
            return minmaxEPS(float(v))
        except (ZeroDivisionError, ValueError, OverflowError):
            # Fall back to numerical inversion on edge cases
            return super().inv_h(w, u)

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised closed-form inverse h-function."""
        w  = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u  = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        th = self.theta
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            v = ((w ** (-th / (1.0 + th)) - 1.0) * u ** (-th) + 1.0) ** (-1.0 / th)
        return np.clip(np.where(np.isfinite(v), v, 0.5), EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Clayton copula: λ_L = 2^{−1/θ}, λ_U = 0."""
        return 2.0 ** (-1.0 / self.theta), 0.0

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF — bypasses the ``log(max(pdf, EPS))`` floor.

            log c(u, v) = log(1 + θ) + (−1 − θ)(log u + log v)
                        + (−1/θ − 2) log(u^{−θ} + v^{−θ} − 1)
        """
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        th = self.theta
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            # u^{-θ} = exp(-θ log u); the "−1" inside the log is small
            # compared to the dominant term when u is small, so we use
            # ``np.log`` directly on a positive quantity.
            inner = np.maximum(u ** (-th) + v ** (-th) - 1.0, EPS)
            return (np.log1p(th)
                    + (-1.0 - th) * (np.log(u) + np.log(v))
                    + (-1.0 / th - 2.0) * np.log(inner))

    def majorant(self, u_left: float) -> float:
        """Closed-form maximum of c(u, v) over v.

        Derivation
        ----------
        With ``c(u,v) = (1+θ) (uv)^{-1-θ} (u^{-θ}+v^{-θ}-1)^{-1/θ-2}``,
        setting ∂/∂v log c = 0 gives the critical value

            v*^{-θ} = (1+θ)/θ · (u^{-θ} − 1).

        If u^{-θ} ≤ 1 (i.e. u → 1) then v*^{-θ} ≤ 0 and the interior
        critical point does not exist; the max then occurs at the
        boundary v → 1, where c blows up. We clip to v = 1 − EPS.

        The critical value v* may also fall outside ``(EPS, 1-EPS)`` for
        extreme u; in that case we fall back to evaluating c at both
        boundaries and taking the larger one.
        """
        u = minmaxEPS(u_left)
        th = self.theta
        # Critical point exists only if u^{-θ} > 1, i.e. u < 1.
        u_pow = u ** (-th)
        z = (1.0 + th) / th * (u_pow - 1.0)
        if z > 0.0:
            v_star = z ** (-1.0 / th)
            v_star = minmaxEPS(v_star)
            f_star = self.pdf([u, v_star])
        else:
            f_star = 0.0
        # Boundary candidates (v → 0+ or v → 1−)
        f_lo = self.pdf([u, EPS])
        f_hi = self.pdf([u, ONE_MINUS_EPS])
        return float(max(f_star, f_lo, f_hi))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaClayton(tau_k=0.5)
    lam_l = 2.0 ** (-1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2τ/(1−τ)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lam_l:.4f}  [= 2^(−1/θ)],  λ_U = 0')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
