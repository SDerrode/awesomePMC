if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


class CopulaProduct(CopulaVirt):
    """Independence (product) copula: C(u,v) = u·v,  c(u,v) = 1."""

    def __init__(self, **kwargs):
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = 0.0

    def pdf(self, uv):
        return 1.0

    def cdf(self, uv):
        return float(uv[0]) * float(uv[1])

    def conditional_cdf(self, v: float, u: float) -> float:
        """h(v|u) = v  (independent margins)."""
        return float(minmaxEPS(v))

    def majorant(self, u_left: float) -> float:
        return 1.0

    def tail_dependence(self) -> tuple[float, float]:
        return 0.0, 0.0


if __name__ == '__main__':
    from prg.tools.tools import set_dir

    cop = CopulaProduct(tau_k=0.)
    print(f'Copula   : {cop.copula_enum.value.LONG_NAME}')
    print(f'tau_k    : {cop.params["tau_k"]:.4f}  range={cop.tau_range}')
    print(f'pdf(0.3, 0.7) = {cop.pdf([0.3, 0.7]):.6f}  [= 1 everywhere]')
    print(f'cdf(0.3, 0.7) = {cop.cdf([0.3, 0.7]):.6f}  [= u·v]')
    print(f'h(0.7 | 0.3)  = {cop.conditional_cdf(0.7, 0.3):.6f}  [= v, independence]')
    print(f'tail dep : λ_L = λ_U = 0')
    print(f'(plot_multi_tau skipped: single point τ=0)')

    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plot_pdf(plot_dir)
    cop.plot_cdf(plot_dir)
    cop.plot_h_function(plot_dir)
    cop.plot_samples(plot_dir)
    cop.plot_overview(plot_dir)
    # plot_multi_tau: no range to explore (τ fixed at 0)
