"""
Cubic-section copula.

τ_K ∈ [0, 33/200].
Analytical majorant (verified, piecewise):
  uleft ∈ [0, 0.5] : M = 1 + 2θ(1 − 3·u²)
  uleft ∈ (0.5, 1] : M = 1 + 2θ(−3·u² + 6·u − 2)
"""
import math

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


class CopulaCubSec(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        dis        = (100.0 - 24.0 * self.CopParamDict['tauK']) / 225.0
        self.theta = (-2.0 / 3.0 + math.sqrt(dis)) / (-12.0 / 225.0)

    def PdfCopule(self, VectU):
        u1 = minmaxEPS(VectU[0])
        u2 = minmaxEPS(VectU[1])
        result = 1.0 + 2.0 * self.theta * (
              (1.0 - u1) * (1.0 - u2) * (-8.0*u2*u1 + 2.0*u1 + 2.0*u2 + 1.0)
            + u1          * (1.0 - u2) * (+4.0*u2*u1 - u1      - 2.0*u2 - 1.0)
            + (1.0 - u1) * u2          * (+4.0*u2*u1 - 2.0*u1  - u2      - 1.0)
            + u1          * u2          * (-2.0*u2*u1 + u1      + u2      + 1.0)
        )
        return result

    def CdfCopule(self, VectU):
        u1 = minmaxEPS(VectU[0])
        u2 = minmaxEPS(VectU[1])
        return u1 * u2 * (1.0 + 2.0*self.theta * (1.0 - u1) * (1.0 - u2) * (1.0 + u1 + u2 - 2.0*u1*u2))

    def MajorantCopula(self, uleft: float) -> float:
        """Analytical majorant (verified)."""
        u = uleft
        if u <= 0.5:
            return 1.0 + 2.0 * self.theta * (-3.0*u*u + 1.0)
        else:
            return 1.0 + 2.0 * self.theta * (-3.0*u*u + 6.0*u - 2.0)


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaCubSec(tauK=0.15)
    print(cop, f'  theta={cop.theta:.4f}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Cdf([0.5, 0.7]) = {cop.CdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.2)   = {cop.MajorantCopula(0.2):.6f}')
    print(f'  Majorant(0.8)   = {cop.MajorantCopula(0.8):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
