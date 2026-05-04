if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import math
import numpy as np
from scipy.stats import t as _t
from statsmodels.distributions.copula.api import StudentTCopula

from prg.copulas._base   import CopulaVirt
from prg.tools.tools     import EPS_MINUS_ONE, ONE_MINUS_EPS, minmaxEPS


class CopulaStudent(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.df    = 4  # degrees of freedom (fixed for now)
        self.theta = math.sin(self.params['tau_k'] * math.pi / 2.0)
        self.theta = max(EPS_MINUS_ONE, min(ONE_MINUS_EPS, self.theta))
        corr = np.array([[1., self.theta], [self.theta, 1.]])
        self._model = StudentTCopula(corr=corr, df=self.df)

    def pdf(self, uv):
        return float(self._model.pdf(uv))

    def cdf(self, uv):
        raise NotImplementedError('CopulaStudent: CDF has no closed form (bivariate Student-t integral).')

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = t_{ν+1}((t_ν⁻¹(v) − θ·t_ν⁻¹(u)) · √((ν+1)/((ν+(t_ν⁻¹(u))²)(1−θ²))))."""
        x_u = _t.ppf(minmaxEPS(u), df=self.df)
        x_v = _t.ppf(minmaxEPS(v), df=self.df)
        scale = math.sqrt((self.df + 1.0) / ((self.df + x_u**2) * (1.0 - self.theta**2)))
        return float(_t.cdf((x_v - self.theta * x_u) * scale, df=self.df + 1))

    def tail_dependence(self) -> tuple[float, float]:
        """Symmetric tail dependence: λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))."""
        arg = -math.sqrt((self.df + 1.0) * (1.0 - self.theta) / (1.0 + self.theta))
        lam = 2.0 * float(_t.cdf(arg, df=self.df + 1))
        return lam, lam

    # Numerical majorant (inherited from CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaStudent(tau_k=0.5)
    lam = 2.0 * _t.cdf(
        -math.sqrt((cop.df + 1) * (1 - cop.theta) / (1 + cop.theta)),
        df=cop.df + 1
    )
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= sin(π·τ/2)]')
    print(f'df       : {cop.df}  (fixed)')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'tail dep : λ_L = λ_U = {lam:.4f}  [symmetric, heavier than Gaussian]')
    print(f'(CDF has no closed form — plot_cdf silently skipped)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)   # skipped gracefully
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
