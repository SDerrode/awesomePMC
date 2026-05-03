"""
Archimedean copula A12 (Huard, Evin & Favre 2006, Table 1, family 12).

Generator: φ(t) = (1/t − 1)^θ
CDF:       C(u,v) = 1 / (1 + (U1 + U2)^{1/θ})
           where  U_i = (1/u_i − 1)^θ
θ = 2 / (3(1 − τ_K)),  τ_K ∈ [1/3, 1).
"""
import numpy as np

from prg.copulas._base import CopulaVirt
from prg.tools.tools   import EPS, UNMOINSEPS, minmaxEPS


class CopulaA12(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.theta = 2.0 / (3.0 * (1.0 - self.CopParamDict['tauK']))

    def PdfCopule(self, VectU):
        u1 = minmaxEPS(VectU[0])
        u2 = minmaxEPS(VectU[1])
        U1 = np.pow(1.0 / u1 - 1.0, self.theta)
        U2 = np.pow(1.0 / u2 - 1.0, self.theta)
        S  = U1 + U2
        S1th = np.pow(S, 1.0 / self.theta)
        base = np.pow(S, 1.0 / self.theta - 2.0)
        if base == 0.0:
            return EPS
        result = (
            U1 / (u1 * (u1 - 1.0)) *
            U2 / (u2 * (u2 - 1.0)) *
            (self.theta - 1.0 + (self.theta + 1.0) * S1th) *
            base / np.pow(1.0 + S1th, 3.0)
        )
        return float(result) if np.isfinite(result) else EPS

    def CdfCopule(self, VectU):
        u1 = minmaxEPS(VectU[0])
        u2 = minmaxEPS(VectU[1])
        U1 = np.pow(1.0 / u1 - 1.0, self.theta)
        U2 = np.pow(1.0 / u2 - 1.0, self.theta)
        return float(1.0 / (1.0 + np.pow(U1 + U2, 1.0 / self.theta)))

    # Majorant numérique (hérité de CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaA12(tauK=0.5)
    print(cop, f'  theta={cop.theta:.4f}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Cdf([0.5, 0.7]) = {cop.CdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.2)   = {cop.MajorantCopula(0.2):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
