"""
Archimedean copula A14 (Huard, Evin & Favre 2006, Table 1, family 14).

CDF:  C(u,v) = (1 + (U1 + U2)^{1/θ})^{-θ}
      where  U_i = (u_i^{-1/θ} − 1)^θ
θ = 2 / (3(1 − τ_K)),  τ_K ∈ [1/3, 1).
"""
import numpy as np

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


class CopulaA14(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.CopParamDict['tauK']))

    def PdfCopule(self, VectU):
        u0 = minmaxEPS(VectU[0])
        u1 = minmaxEPS(VectU[1])
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

    def CdfCopule(self, VectU):
        u0 = minmaxEPS(VectU[0])
        u1 = minmaxEPS(VectU[1])
        U1 = pow(pow(u0, -1.0 / self.theta) - 1.0, self.theta)
        U2 = pow(pow(u1, -1.0 / self.theta) - 1.0, self.theta)
        return float(pow(1.0 + pow(U1 + U2, 1.0 / self.theta), -self.theta))

    # Majorant numérique (hérité de CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaA14(tauK=0.5)
    print(cop, f'  theta={cop.theta:.4f}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Cdf([0.5, 0.7]) = {cop.CdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.2)   = {cop.MajorantCopula(0.2):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
