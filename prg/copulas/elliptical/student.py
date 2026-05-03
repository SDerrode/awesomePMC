import math
import numpy as np
from statsmodels.distributions.copula.api import StudentTCopula

from prg.copulas._base   import CopulaVirt
from prg.tools.tools     import EPS, UNMOINSEPS, EPSMOINSUN


class CopulaStudent(CopulaVirt):

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.df    = 4  # degrees of freedom (fixed for now — see point 4b)
        self.theta = math.sin(self.CopParamDict['tauK'] * math.pi / 2.0)
        self.theta = max(EPSMOINSUN, min(UNMOINSEPS, self.theta))
        matCorr = np.array([[1., self.theta], [self.theta, 1.]])
        self.copulastatmodels = StudentTCopula(corr=matCorr, df=self.df)

    def PdfCopule(self, VectU):
        return float(self.copulastatmodels.pdf(VectU))

    def CdfCopule(self, VectU):
        raise NotImplementedError('CopulaStudent: CDF has no closed form (bivariate Student-t integral).')

    # Majorant numérique (hérité de CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaStudent(tauK=0.5)
    print(cop, f'  theta={cop.theta:.4f}  df={cop.df}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.2)   = {cop.MajorantCopula(0.2):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    # CDF not available in closed form for StudentTCopula (statsmodels)
