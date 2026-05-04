if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
from prg.copulas._base import CopulaVirt
from prg.numerics   import EPS, ONE_MINUS_EPS, minmaxEPS


class CopulaFGM(CopulaVirt):
    """
    Farlie-Gumbel-Morgenstern copula.

    C(u,v) = u·v·[1 + θ(1−u)(1−v)],  θ = 9/2 · τ_K.
    c(u,v) = 1 + θ(1−2u)(1−2v).
    τ_K ∈ [−2/9, 2/9].

    Analytical majorant:  M(u_L) = 1 + |θ| · |1 − 2·u_L|
    """

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 4.5 * self.params['tau_k']

    def pdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        return 1.0 + self.theta * (1.0 - 2.0 * u0) * (1.0 - 2.0 * u1)

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form PDF (FGM has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        result = 1.0 + self.theta * (1.0 - 2.0 * u) * (1.0 - 2.0 * v)
        return np.maximum(result, EPS)

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Native log-PDF: log(1 + θ(1−2u)(1−2v))."""
        uv = np.asarray(uv, dtype=float)
        u = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return np.log(np.maximum(
            1.0 + self.theta * (1.0 - 2.0 * u) * (1.0 - 2.0 * v),
            EPS,
        ))

    def cdf(self, uv):
        u0 = minmaxEPS(uv[0])
        u1 = minmaxEPS(uv[1])
        return u0 * u1 * (1.0 + self.theta * (1.0 - u0) * (1.0 - u1))

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """Vectorised closed-form CDF (FGM has no statsmodels backend)."""
        uv = np.asarray(uv, dtype=float)
        u  = np.clip(uv[:, 0], EPS, ONE_MINUS_EPS)
        v  = np.clip(uv[:, 1], EPS, ONE_MINUS_EPS)
        return np.clip(u * v * (1.0 + self.theta * (1.0 - u) * (1.0 - v)),
                       0.0, 1.0)

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v · [1 + θ(1−v)(1−2u)]."""
        u = minmaxEPS(u)
        v = minmaxEPS(v)
        result = v * (1.0 + self.theta * (1.0 - v) * (1.0 - 2.0*u))
        return float(np.clip(result, 0.0, 1.0))

    def majorant(self, u_left: float) -> float:
        """Analytical majorant (verified)."""
        return 1.0 + abs(self.theta * (1.0 - 2.0 * u_left))

    def tail_dependence(self) -> tuple[float, float]:
        """FGM has zero tail dependence at both extremes."""
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaFGM(tau_k=0.15)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'theta    : {cop.theta:.6f}  [= 9/2 · τ]')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}')
    print(f'majorant(0.5) = {cop.majorant(0.5):.6f}  [= 1 + |θ||1−2u|]')
    print('tail dep : λ_L = λ_U = 0  (τ_max = 2/9 ≈ 0.222)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    cop.plot_multi_tau(plot_dir)
