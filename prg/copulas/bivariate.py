import os
import secrets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from prg.tools.tools            import minmaxEPS, EPS
from prg.settings.plot_settings import facecolor, dpi, BIGGER_SIZE


def _margin_to_dict(margin: tuple) -> dict:
    """Convert a scipy margin tuple (dist, *params) to a parameter dictionary."""
    dist   = margin[0]
    params = list(margin[1:])
    names: list[str] = []
    if dist.shapes is not None:
        names += [s.strip() for s in dist.shapes.split(',')]
    names += ['loc', 'scale']
    return {
        'Distribution'     : dist,
        'Distribution name': dist.name,
        'Parameters'       : params,
        'Parameters name'  : names,
    }


class Loi2DCopule:
    """
    Bivariate distribution built from a copula and two marginals (Sklar's theorem):

        f(x, y) = f1(x) · f2(y) · c(F1(x), F2(y))

    Parameters
    ----------
    copula       : CopulaVirt instance
    left_margin  : tuple  (scipy_dist, *params)
    right_margin : tuple  (scipy_dist, *params)
    """

    def __init__(self, copula, left_margin: tuple, right_margin: tuple):
        self.CopulaLaw          = copula
        self.LeftMarginLawDict  = _margin_to_dict(left_margin)
        self.RightMarginLawDict = _margin_to_dict(right_margin)

        self.generateNewSeed()

        # Plotting grid on the [5%, 95%] quantile range of each margin
        self.ticks_nbr = 15
        N = 200
        left_range  = self.LeftMarginLawDict['Distribution'].ppf(
            [0.05, 0.95], *self.LeftMarginLawDict['Parameters'])
        right_range = self.RightMarginLawDict['Distribution'].ppf(
            [0.05, 0.95], *self.RightMarginLawDict['Parameters'])
        self.x = np.linspace(left_range[0],  left_range[1],  N)
        self.y = np.linspace(right_range[0], right_range[1], N)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.z = np.zeros(self.X.shape)

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------
    def getSeed(self) -> int:
        return self.__Seed

    def setSeed(self, sKey: int) -> None:
        self.__Seed = sKey
        self.__rng  = np.random.default_rng(self.__Seed)

    def generateNewSeed(self) -> int:
        self.setSeed(secrets.randbits(128))
        return self.__Seed

    def getDimObservations(self) -> int:
        return 2

    # ------------------------------------------------------------------
    # Probability computations
    # ------------------------------------------------------------------
    def Pdf(self, VectCoord) -> float:
        """Joint PDF at (x, y) in the original scale."""
        u1 = self.LeftMarginLawDict['Distribution'].cdf(
            VectCoord[0], *self.LeftMarginLawDict['Parameters'])
        u2 = self.RightMarginLawDict['Distribution'].cdf(
            VectCoord[1], *self.RightMarginLawDict['Parameters'])
        f1 = self.LeftMarginLawDict['Distribution'].pdf(
            VectCoord[0], *self.LeftMarginLawDict['Parameters'])
        f2 = self.RightMarginLawDict['Distribution'].pdf(
            VectCoord[1], *self.RightMarginLawDict['Parameters'])
        c  = self.CopulaLaw.PdfCopule(np.array([u1, u2]))
        result = f1 * f2 * c
        if not np.isfinite(result):
            return EPS
        return result if result > 0.0 else EPS

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def randomsampling_leftmargin(self) -> float:
        return self.LeftMarginLawDict['Distribution'].rvs(
            *self.LeftMarginLawDict['Parameters'], random_state=self.__rng)

    def randomsampling_rightmargin_condto_leftmargin(self, yleft: float) -> float:
        """
        Conditional sample of the right margin given yleft, via acceptance-rejection.

        The majorant M = max_{v} c(F1(yleft), v) is provided by CopulaLaw.MajorantCopula.
        """
        Yleft    = minmaxEPS(
            self.LeftMarginLawDict['Distribution'].cdf(
                yleft, *self.LeftMarginLawDict['Parameters']))
        Majorant = self.CopulaLaw.MajorantCopula(Yleft)

        vectU     = np.array([Yleft, -1.0])
        cpt       = 0
        nbitermax = 80_000
        paramright = self.RightMarginLawDict['Parameters']

        while cpt < nbitermax:
            cpt   += 1
            yright = self.RightMarginLawDict['Distribution'].rvs(
                *paramright, random_state=self.__rng)
            Yright = minmaxEPS(
                self.RightMarginLawDict['Distribution'].cdf(yright, *paramright))
            vectU[1] = Yright
            seuil    = self.CopulaLaw.PdfCopule(vectU) / Majorant
            if self.__rng.uniform() <= seuil:
                return yright

        raise RuntimeError(
            f'Loi2DCopule: acceptance-rejection did not converge after {nbitermax} iterations. '
            f'yleft={yleft}, Majorant={Majorant}, last seuil={seuil:.4e}. '
            f'Check the copula majorant or increase nbitermax.'
        )

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def __repr__(self):
        return str(self)

    def __str__(self):
        lp = ', '.join(f'{v:.2f}' for v in self.LeftMarginLawDict['Parameters'])
        rp = ', '.join(f'{v:.2f}' for v in self.RightMarginLawDict['Parameters'])
        return (
            f'[\n  COPULA:       {self.CopulaLaw}'
            f"\n  LEFT MARGIN:  {self.LeftMarginLawDict['Distribution name']} ({lp})"
            f"\n  RIGHT MARGIN: {self.RightMarginLawDict['Distribution name']} ({rp})"
            '\n]'
        )

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def _compute_pdf_grid(self):
        for i, a in enumerate(self.x):
            for j, b in enumerate(self.y):
                self.z[j, i] = self.Pdf([a, b])

    def plotPdfLois2D(self, plot_dir: str, string: str = ''):
        self._compute_pdf_grid()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor=facecolor)
        min_ = np.nanpercentile(self.z, 0.)
        max_ = np.nanpercentile(self.z, 100.)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        cs = ax.contourf(self.X, self.Y, self.z, vticks, antialiased=True, vmin=0, vmax=max_)
        plt.colorbar(cs, ticks=vticks)
        ax.set_aspect('equal')
        ax.set_xlabel('Left')
        ax.set_ylabel('Right')
        ax.set_facecolor('red')
        plt.suptitle("2D representation of law's pdf", y=0.85, fontsize=BIGGER_SIZE)
        plt.savefig(os.path.join(plot_dir, f'{string}2D_PdfLaw.png'),
                    bbox_inches='tight', dpi=dpi, facecolor=facecolor)
        plt.close()

    def plotPdfLois2DwithMargins(self, plot_dir: str, title: str, string: str = ''):
        self._compute_pdf_grid()

        fig = plt.figure(figsize=(6, 6), facecolor=facecolor)
        gs  = gridspec.GridSpec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                                left=0.1, right=0.9, bottom=0.1, top=0.9,
                                wspace=0.05, hspace=0.05)
        ax             = fig.add_subplot(gs[1, 0])
        ax_marginleft  = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_marginright = fig.add_subplot(gs[1, 1], sharey=ax)

        min_ = np.nanpercentile(self.z, 0.)
        max_ = np.nanpercentile(self.z, 100.)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        ax.contourf(self.X, self.Y, self.z, vticks, antialiased=True, vmin=min_, vmax=max_)
        ax.set_xlabel('Left')
        ax.set_ylabel('Right')
        ax.set_facecolor('red')

        ax_marginleft.tick_params(axis='x', labelbottom=False)
        ax_marginright.tick_params(axis='y', labelleft=False)

        lparams = self.LeftMarginLawDict['Parameters']
        rparams = self.RightMarginLawDict['Parameters']
        ax_marginleft.plot(self.x,
                           self.LeftMarginLawDict['Distribution'].pdf(self.x, *lparams),
                           'b-', lw=2, alpha=0.6)
        ax_marginright.plot(self.RightMarginLawDict['Distribution'].pdf(self.y, *rparams),
                            self.y, 'b-', lw=2, alpha=0.6)

        lname = ', '.join(f'{v:.2f}' for v in lparams)
        rname = ', '.join(f'{v:.2f}' for v in rparams)
        ax_marginleft.text(0.5, 0.9,
                           f"Left: {self.LeftMarginLawDict['Distribution name']} ({lname})",
                           transform=ax_marginleft.transAxes, va='center', ha='center')
        ax_marginright.text(0.9, 0.5,
                            f"Right: {self.RightMarginLawDict['Distribution name']} ({rname})",
                            transform=ax_marginright.transAxes, va='center', ha='center',
                            rotation=270)

        tau = self.CopulaLaw.CopParamDict['tauK']
        cname = self.CopulaLaw.copulaEnum.value.LONG_NAME
        fig.suptitle(f'{title} — Copula: {cname} (τ={tau})', y=0.95, fontsize=BIGGER_SIZE)
        plt.savefig(os.path.join(plot_dir, f'{string}2D_PdfLawWithMarginPdfs.png'),
                    bbox_inches='tight', dpi=dpi, facecolor=facecolor)
        plt.close()


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import scipy as sc
    from prg.tools.tools import set_dir
    from prg.copulas import (
        Loi2DCopule,
        CopulaGaussian, CopulaProduct, CopulaFGM,
        CopulaA12, CopulaA14, CopulaGH, CopulaClayton,
        CopulaCubSec, CopulaStudent,
    )

    left_margin  = (sc.stats.triang, 0.158, 0., 3.)
    right_margin = (sc.stats.norm,   5.,    0.5)

    copulas = [
        CopulaGaussian(tauK=0.5),
        CopulaProduct(tauK=0.),
        CopulaFGM(tauK=0.2),
        CopulaA12(tauK=0.7),
        CopulaA14(tauK=0.7),
        CopulaGH(tauK=0.2),
        CopulaClayton(tauK=0.2),
        CopulaCubSec(tauK=0.1),
        CopulaStudent(tauK=0.1),
    ]
    for cop in copulas:
        loi = Loi2DCopule(cop, left_margin, right_margin)
        print(loi)
        plot_dir = set_dir('./data/Plots', '2DLaws')
        loi.plotPdfLois2D(plot_dir, f'{cop.copulaEnum.SHORT_NAME}_')
        loi.plotPdfLois2DwithMargins(plot_dir, r'$p(y_1, y_2)$', f'{cop.copulaEnum.SHORT_NAME}_')
