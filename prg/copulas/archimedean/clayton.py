from statsmodels.distributions.copula.api import ClaytonCopula

from prg.copulas._base import CopulaVirt


class CopulaClayton(CopulaVirt):
    """
    Clayton copula — lower-tail dependence.

    θ = 2 τ_K / (1 − τ_K).
    """

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        tau = self.CopParamDict['tauK']
        self.theta = 2.0 * tau / (1.0 - tau)
        self.copulastatmodels = ClaytonCopula(theta=self.theta)

    def PdfCopule(self, VectU):
        return float(self.copulastatmodels.pdf(VectU))

    def CdfCopule(self, VectU):
        return float(self.copulastatmodels.cdf(VectU))

    # Majorant numérique (hérité de CopulaVirt)


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaClayton(tauK=0.5)
    print(cop, f'  theta={cop.theta:.4f}')
    print(f'  Pdf([0.5, 0.7]) = {cop.PdfCopule([0.5, 0.7]):.6f}')
    print(f'  Cdf([0.5, 0.7]) = {cop.CdfCopule([0.5, 0.7]):.6f}')
    print(f'  Majorant(0.2)   = {cop.MajorantCopula(0.2):.6f}')
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
