if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import math
import numpy as np
from scipy.stats import norm as _norm
from statsmodels.distributions.copula.api import GaussianCopula

from prg.copulas._base   import CopulaVirt
from prg.numerics     import EPS, ONE_MINUS_EPS, EPS_MINUS_ONE, minmaxEPS


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

    def inv_h(self, w: float, u: float) -> float:
        """Closed-form inverse of h(v|u): solve h(v|u) = w in one step.

        Derivation:  Φ⁻¹(w) = (Φ⁻¹(v) − θ·Φ⁻¹(u)) / √(1−θ²)
                  ⟹  v = Φ(Φ⁻¹(w)·√(1−θ²) + θ·Φ⁻¹(u))
        """
        x_u = _norm.ppf(minmaxEPS(u))
        x_w = _norm.ppf(minmaxEPS(w))
        x_v = x_w * math.sqrt(1.0 - self.theta ** 2) + self.theta * x_u
        return minmaxEPS(_norm.cdf(x_v))

    def inv_h_array(self, w: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Vectorised closed-form inverse h-function (M points at once)."""
        w = np.clip(np.asarray(w, dtype=float), EPS, ONE_MINUS_EPS)
        u = np.clip(np.asarray(u, dtype=float), EPS, ONE_MINUS_EPS)
        x_u = _norm.ppf(u)
        x_w = _norm.ppf(w)
        x_v = x_w * math.sqrt(1.0 - self.theta ** 2) + self.theta * x_u
        return np.clip(_norm.cdf(x_v), EPS, ONE_MINUS_EPS)

    def tail_dependence(self) -> tuple[float, float]:
        """Gaussian copula has zero tail dependence for any |θ| < 1."""
        return 0.0, 0.0

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF — bypasses the ``log(max(pdf, EPS))`` floor.

        For x = Φ⁻¹(u), y = Φ⁻¹(v), ρ = θ:
            log c(u, v) = -½ log(1 − ρ²)
                        + (2ρxy − ρ²(x² + y²)) / (2(1 − ρ²))

        Accurate down to log c ≈ −700 (where ``np.exp`` underflows but
        ``log c`` itself remains representable). The default
        ``log(max(pdf, EPS))`` saturates around −36.
        """
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        x = _norm.ppf(u)
        y = _norm.ppf(v)
        rho     = self.theta
        one_m_r2 = 1.0 - rho * rho
        return -0.5 * math.log(one_m_r2) + \
               (2.0 * rho * x * y - rho * rho * (x * x + y * y)) / (2.0 * one_m_r2)

    def majorant(self, u_left: float) -> float:
        """Closed-form maximum of c(u, v) over v.

        Derivation
        ----------
        Let x = Φ⁻¹(u), y = Φ⁻¹(v), ρ = θ. Then
            log c(u,v) = -½ log(1-ρ²) + (2ρxy - ρ²(x²+y²)) / (2(1-ρ²))
        ∂/∂y log c = -(ρ²y - ρx) / (1-ρ²) = 0  ⟹  y* = x/ρ.
        Plugging y* back into the exponent yields x²/2, so

            max_v c(u, v) = (1 - ρ²)^{-1/2} · exp((Φ⁻¹(u))² / 2).

        For ρ = 0 the copula PDF is identically 1, so the max is 1.
        """
        if abs(self.theta) < 1e-12:
            return 1.0
        x = _norm.ppf(minmaxEPS(u_left))
        return float(math.exp(0.5 * x * x) / math.sqrt(1.0 - self.theta ** 2))


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaGaussian(tau_k=0.5)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= sin(π·τ/2)]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print('tail dep : λ_L = λ_U = 0  (for all |θ| < 1)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
