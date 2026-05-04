if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import math
import numpy as np
from scipy.stats import norm as _norm
from statsmodels.distributions.copula.api import GaussianCopula

from prg.copulas._base   import CopulaVirt
from prg.tools.tools     import EPS, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS


class CopulaGaussian(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = math.sin(self.params['tau_k'] * math.pi / 2.0)
        self.theta = max(EPS_MINUS_ONE, min(ONE_MINUS_EPS, self.theta))
        corr = np.array([[1., self.theta], [self.theta, 1.]])
        self._model = GaussianCopula(corr=corr)

    def pdf(self, uv):
        u = minmaxEPS(uv[0])
        v = minmaxEPS(uv[1])
        return float(self._model.pdf([u, v]))

    def cdf(self, uv):
        return float(self._model.cdf(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = Φ((Φ⁻¹(v) − θ·Φ⁻¹(u)) / √(1−θ²))."""
        x_u = _norm.ppf(minmaxEPS(u))
        x_v = _norm.ppf(minmaxEPS(v))
        return float(_norm.cdf((x_v - self.theta * x_u) / math.sqrt(1.0 - self.theta**2)))

    def tail_dependence(self) -> tuple[float, float]:
        """Gaussian copula has zero tail dependence for any |θ| < 1."""
        return 0.0, 0.0

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaGaussian(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= sin(π·τ/2)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = λ_U = 0  (for all |θ| < 1)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
