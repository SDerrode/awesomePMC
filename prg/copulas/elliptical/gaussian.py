import math
import numpy as np
from statsmodels.distributions.copula.api import GaussianCopula

from prg.copulas._base   import CopulaVirt
from prg.tools.tools     import EPS, UNMOINSEPS, EPSMOINSUN, minmaxEPS


class CopulaGaussian(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.theta = math.sin(self.CopParamDict['tauK'] * math.pi / 2.0)
        self.theta = max(EPSMOINSUN, min(UNMOINSEPS, self.theta))
        matCorr = np.array([[1., self.theta], [self.theta, 1.]])
        self.copulastatmodels = GaussianCopula(corr=matCorr)

    def PdfCopule(self, VectU):
        u = minmaxEPS(VectU[0])
        v = minmaxEPS(VectU[1])
        return float(self.copulastatmodels.pdf([u, v]))

    def CdfCopule(self, VectU):
        return float(self.copulastatmodels.cdf(VectU))

    # Majorant numérique (hérité de CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaGaussian(tauK=0.3)
    print(cop, f'  theta={cop.theta:.4f}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Cdf([0.5, 0.7]) = {cop.CdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.2)   = {cop.MajorantCopula(0.2):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
