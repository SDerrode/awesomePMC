if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
from statsmodels.distributions.copula.api import GumbelCopula

from prg.copulas._base import CopulaVirt
from prg.numerics   import minmaxEPS


class CopulaGH(CopulaVirt):
    """
    Gumbel-Hougaard copula — positive upper-tail dependence.

    Generator: φ(t) = (−ln t)^θ,  θ = 1/(1 − τ_K).
    θ = 1 → independence;  θ → ∞ → comonotonicity.
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 1.0 / (1.0 - self.params['tau_k'])
        self._model = GumbelCopula(theta=self.theta)

    def pdf(self, uv):
        return float(self._model.pdf(uv))

    def cdf(self, uv):
        return float(self._model.cdf(uv))

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = C(u,v)/u · (−ln u)^{θ−1} · A^{(1−θ)/θ}
        where A = ((−ln u)^θ + (−ln v)^θ)^{1/θ}."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        ln_u = -np.log(u)
        ln_v = -np.log(v)
        A = (ln_u**self.theta + ln_v**self.theta) ** (1.0 / self.theta)
        c_uv = np.exp(-A)
        result = c_uv * (A ** (1.0 - self.theta)) * (ln_u ** (self.theta - 1.0)) / u
        return float(np.clip(result, 0.0, 1.0))

    def tail_dependence(self) -> tuple[float, float]:
        """GH copula: λ_L = 0, λ_U = 2 − 2^{1/θ}."""
        return 0.0, 2.0 - 2.0 ** (1.0 / self.theta)

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaGH(tau_k=0.5)
    lam_u = 2.0 - 2.0 ** (1.0 / cop.theta)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 1/(1−τ)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = 0,  λ_U = {lam_u:.4f}  [= 2 − 2^(1/θ)]')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
