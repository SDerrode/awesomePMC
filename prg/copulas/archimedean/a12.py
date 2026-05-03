"""
Archimedean copula A12 (Huard, Evin & Favre 2006, Table 1, family 12).

Generator: φ(t) = (1/t − 1)^θ
CDF:       C(u,v) = 1 / (1 + (U1 + U2)^{1/θ})
           where  U_i = (1/u_i − 1)^θ
θ = 2 / (3(1 − τ_K)),  τ_K ∈ [1/3, 1).
"""
if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import EPS, ONE_MINUS_EPS, minmaxEPS


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
            return float(EPS)
        log_result = (
            log_U1 - np.log(u1) - np.log(1.0 - u1)
            + log_U2 - np.log(u2) - np.log(1.0 - u2)
            + np.log(factor)
            + (inv_th - 2.0) * log_S
            - 3.0 * np.log1p(S1th)
        )
        result = np.exp(log_result)
        return float(result) if np.isfinite(result) else float(EPS)

    def cdf(self, uv):
        u1 = minmaxEPS(uv[0])
        u2 = minmaxEPS(uv[1])
        U1 = np.pow(1.0 / u1 - 1.0, self.theta)
        U2 = np.pow(1.0 / u2 - 1.0, self.theta)
        return float(1.0 / (1.0 + np.pow(U1 + U2, 1.0 / self.theta)))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = (1+S^{1/θ})^{−2} · S^{1/θ−1} · (1/u−1)^{θ−1} / u²."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        U1 = (1.0/u - 1.0) ** self.theta
        U2 = (1.0/v - 1.0) ** self.theta
        S   = U1 + U2
        S1t = S ** (1.0 / self.theta)
        result = ((1.0 + S1t)**(-2.0) * S**(1.0/self.theta - 1.0) *
                  (1.0/u - 1.0)**(self.theta - 1.0) / u**2)
        return float(np.clip(result, 0.0, 1.0))

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from prg.tools.tools import set_dir

    cop = CopulaA12(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 2/(3(1−τ))]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    lL, lU = cop.tail_dependence()
    print(f'tail dep : λ_L = {lL:.4f}  [= 2^(−1/θ)],  λ_U = {lU:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
