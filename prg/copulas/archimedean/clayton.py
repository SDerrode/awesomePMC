if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
from statsmodels.distributions.copula.api import ClaytonCopula

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


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
        return float(self._model.pdf(uv))

    def cdf(self, uv):
        return float(self._model.cdf(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = C(u,v)^{1+θ} · u^{−θ−1}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        c_uv = self.cdf([u, v])
        result = (c_uv ** (1.0 + self.theta)) * (u ** (-self.theta - 1.0))
        return float(np.clip(result, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """Clayton copula: λ_L = 2^{−1/θ}, λ_U = 0."""
        return 2.0 ** (-1.0 / self.theta), 0.0

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from prg.tools.tools import set_dir

    cop = CopulaClayton(tau_k=0.5)
    lam_l = 2.0 ** (-1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2τ/(1−τ)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = {lam_l:.4f}  [= 2^(−1/θ)],  λ_U = 0')

    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
