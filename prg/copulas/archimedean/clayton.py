if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import logging
import numpy as np
from statsmodels.distributions.copula.api import ClaytonCopula

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import EPS, minmaxEPS

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

    def tail_dependence(self) -> tuple[float, float]:
        """Clayton copula: λ_L = 2^{−1/θ}, λ_U = 0."""
        return 2.0 ** (-1.0 / self.theta), 0.0

    # Numerical majorant (inherited from CopulaVirt)


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
