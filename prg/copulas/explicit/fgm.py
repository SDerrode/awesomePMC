from prg.copulas._base import CopulaVirt
from prg.tools.tools   import minmaxEPS


class CopulaFGM(CopulaVirt):
    """
    Farlie-Gumbel-Morgenstern copula.

    C(u,v) = u·v·[1 + θ(1−2u)(1−2v)],  θ = 9/2 · τ_K.
    τ_K ∈ [−2/9, 2/9].

    Analytical majorant:  M(u_L) = 1 + |θ| · |1 − 2·u_L|
    """

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.theta = 4.5 * self.CopParamDict['tauK']

    def PdfCopule(self, VectU):
        u0 = minmaxEPS(VectU[0])
        u1 = minmaxEPS(VectU[1])
        return 1.0 + self.theta * (1.0 - 2.0 * u0) * (1.0 - 2.0 * u1)

    def CdfCopule(self, VectU):
        u0 = minmaxEPS(VectU[0])
        u1 = minmaxEPS(VectU[1])
        return u0 * u1 * (1.0 + self.theta * (1.0 - 2.0 * u0) * (1.0 - 2.0 * u1))

    def MajorantCopula(self, uleft: float) -> float:
        """Analytical majorant (verified)."""
        return 1.0 + abs(self.theta * (1.0 - 2.0 * uleft))


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaFGM(tauK=0.15)
    print(cop, f'  theta={cop.theta:.4f}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Cdf([0.5, 0.7]) = {cop.CdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.5)   = {cop.MajorantCopula(0.5):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
