"""
Product (independence) copula Π(u,v) = uv:  c ≡ 1,  h(v|u) = v,
τ_K = ρ_S = 0,  λ_L = λ_U = 0.  No free parameter (``n_params = 0``).

Reference: Nelsen, R. B. (2006). *An Introduction to Copulas*, 2nd ed.,
Springer, ch. 2 (the product copula Π).

Numerics
--------
The array paths are closed forms: ``pdf_array`` ≡ 1, ``logpdf_array`` ≡ 0
and ``cdf_array`` = uv. They used to fall through to the base class — a
Python loop over the scalar ``pdf`` (≈ 25 ms per 10⁵ points, the slowest
density in the package) followed by ``log(max(pdf, EPS))`` (audit RB-9).
"""
if __name__ == '__main__':
    import sys
    import pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np

from pmcprg.copulas._base import CopulaVirt
from pmcprg.numerics   import minmaxEPS


def _check_uv(uv) -> np.ndarray:
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"expected shape (M, 2); got {uv.shape}.")
    return uv


class CopulaProduct(CopulaVirt):
    """Independence (product) copula: C(u,v) = u·v,  c(u,v) = 1."""

    # No free parameter: τ is identically 0, nothing is estimated from the
    # data. The base default of 1 made the penalised criteria charge the
    # independence hypothesis for a parameter it does not have, which cost it
    # 2 nats of AIC and kept it from ever winning — even on exactly
    # independent data.
    n_params: int = 0

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 0.0

    def pdf(self, uv):
        return 1.0

    def pdf_array(self, uv: np.ndarray) -> np.ndarray:
        """c ≡ 1 on the unit square."""
        return np.ones(_check_uv(uv).shape[0])

    def logpdf_array(self, uv: np.ndarray) -> np.ndarray:
        """log c ≡ 0 on the unit square."""
        return np.zeros(_check_uv(uv).shape[0])

    def cdf(self, uv):
        return float(uv[0]) * float(uv[1])

    def cdf_array(self, uv: np.ndarray) -> np.ndarray:
        """C(u, v) = uv, with u and v clipped to [0, 1]."""
        uv = np.clip(_check_uv(uv), 0.0, 1.0)
        return uv[:, 0] * uv[:, 1]

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v  (independent margins)."""
        return float(minmaxEPS(v))

    def majorant(self, u_left: float) -> float:
        return 1.0

    def tail_dependence(self) -> tuple[float, float]:
        return 0.0, 0.0


if __name__ == '__main__':
    from pathlib import Path

    cop = CopulaProduct(tau_k=0.)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}  [= 1 everywhere]')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}  [= u·v]')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}  [= v, independence]')
    print('tail dep : λ_L = λ_U = 0')
    print('(plot_multi_tau skipped: single point τ=0)')

    plot_dir = Path('./data/Plots/Copulas')
    plot_dir.mkdir(parents=True, exist_ok=True)
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    # plot_multi_tau: no range to explore (τ fixed at 0)
