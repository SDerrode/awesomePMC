"""
Cubic-section copula.

τ_K ∈ [0, 33/200].
Analytical majorant (verified, piecewise):
  u_left ∈ [0, 0.5] : M = 1 + 2θ(1 − 3·u²)
  u_left ∈ (0.5, 1] : M = 1 + 2θ(−3·u² + 6·u − 2)
"""
if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import math
import numpy as np

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


class CopulaCubSec(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        dis        = (100.0 - 24.0 * self.params['tau_k']) / 225.0
        self.theta = (-2.0 / 3.0 + math.sqrt(dis)) / (-12.0 / 225.0)

    def pdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        result = 1.0 + 2.0 * self.theta * (
              (1.0 - u1) * (1.0 - u2) * (-8.0*u2*u1 + 2.0*u1 + 2.0*u2 + 1.0)
            + u1          * (1.0 - u2) * (+4.0*u2*u1 - u1      - 2.0*u2 - 1.0)
            + (1.0 - u1) * u2          * (+4.0*u2*u1 - 2.0*u1  - u2      - 1.0)
            + u1          * u2          * (-2.0*u2*u1 + u1      + u2      + 1.0)
        )
        return result

    def cdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        return u1 * u2 * (1.0 + 2.0*self.theta * (1.0 - u1) * (1.0 - u2) * (1.0 + u1 + u2 - 2.0*u1*u2))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v · [1 + 2θ(1−v)(1+v−3u²−6uv+6u²v)]."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        result = v * (1.0 + 2.0*self.theta*(1.0-v)*(1.0+v - 3.0*u*u - 6.0*u*v + 6.0*u*u*v))
        return float(np.clip(result, 0.0, 1.0))

    def majorant(self, u_left: float) -> float:
        """Analytical majorant (verified)."""
        u = u_left
        if u <= 0.5:
            return 1.0 + 2.0 * self.theta * (-3.0*u*u + 1.0)
        else:
            return 1.0 + 2.0 * self.theta * (-3.0*u*u + 6.0*u - 2.0)

    def tail_dependence(self) -> tuple[float, float]:
        """Cubic-section copula has zero tail dependence."""
        return 0.0, 0.0


if __name__ == '__main__':
    from prg.tools.tools import set_dir

    cop = CopulaCubSec(tau_k=0.15)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}')
    print(f'pdf(0.3, 0.7)  = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7)  = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)   = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'majorant(0.2)  = {cop.majorant(0.2):.6f}  [piecewise analytical]')
    print(f'majorant(0.8)  = {cop.majorant(0.8):.6f}')
    print(f'tail dep : λ_L = λ_U = 0  (τ_max = 33/200 = 0.165)')

    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
