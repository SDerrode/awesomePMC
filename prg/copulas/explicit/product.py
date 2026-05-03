from prg.copulas._base import CopulaVirt


class CopulaProduct(CopulaVirt):
    """Independence (product) copula: C(u,v) = u·v,  c(u,v) = 1."""

    def __init__(self, **kwargs):
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.theta = 0.0

    def PdfCopule(self, VectU):
        return 1.0

    def CdfCopule(self, VectU):
        return float(VectU[0]) * float(VectU[1])

    def MajorantCopula(self, uleft: float) -> float:
        return 1.0


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from prg.tools.tools import set_dir

    cop = CopulaProduct(tauK=0.)
    print(cop)
    plot_dir = set_dir('./data/Plots', 'Copulas')
    cop.plotPdfCopule(plot_dir)
    cop.plotCdfCopule(plot_dir)
