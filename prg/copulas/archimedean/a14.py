"""
Archimedean copula A14 (Huard, Evin & Favre 2006, Table 1, family 14).

CDF:  C(u,v) = (1 + (U1 + U2)^{1/θ})^{-θ}
      where  U_i = (u_i^{-1/θ} − 1)^θ
θ = 2 / (3(1 − τ_K)),  τ_K ∈ [1/3, 1).
"""
if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


class CopulaA14(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.params['tau_k']))

    def pdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        U1 = pow(pow(u0, -1.0 / self.theta) - 1.0, self.theta)
        U2 = pow(pow(u1, -1.0 / self.theta) - 1.0, self.theta)
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
        return float(result)

    def cdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        U1 = pow(pow(u0, -1.0 / self.theta) - 1.0, self.theta)
        U2 = pow(pow(u1, -1.0 / self.theta) - 1.0, self.theta)
        return float(pow(1.0 + pow(U1 + U2, 1.0 / self.theta), -self.theta))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−θ−1} · S^{1/θ−1} · (u^{−1/θ}−1)^{θ−1} · u^{−1/θ−1}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        inv_th = 1.0 / self.theta
        U1 = (u**(-inv_th) - 1.0) ** self.theta
        U2 = (v**(-inv_th) - 1.0) ** self.theta
        S   = U1 + U2
        S1t = S ** inv_th
        result = ((1.0 + S1t)**(-self.theta - 1.0) * S**(inv_th - 1.0) *
                  (u**(-inv_th) - 1.0)**(self.theta - 1.0) * u**(-inv_th - 1.0))
        return float(np.clip(result, 0.0, 1.0))

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from prg.tools.tools import set_dir

    cop = CopulaA14(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2/(3(1−τ))]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    lL, lU = cop.tail_dependence()
    print(f'tail dep : λ_L = {lL:.4f}  [≈ 1/2 for any θ],  λ_U = {lU:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
