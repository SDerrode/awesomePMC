import os
import matplotlib.pyplot as plt
import numpy as np

from enum import Enum, unique
from dataclasses import dataclass, field

from prg.tools.tools            import EPS, UNMOINSEPS, EPSMOINSUN, minmaxEPS
from prg.settings.plot_settings import facecolor, dpi, BIGGER_SIZE


@dataclass
class CopulaDataMixin:
    ID:                  int
    SHORT_NAME:          str
    LONG_NAME:           str
    CLASS_NAME:          str
    AVAILABLE:           bool
    PARAMETERS_SET_NAME: list[str]   = field(default_factory=list)
    TAU_MIN_MAX:         list[float] = field(default_factory=list)


@unique
class CopulaEnum(CopulaDataMixin, Enum):
    PRODUCT = 1,  'Prod',    'Product',                   'CopulaProduct',  True,  ['tauK'], [0.,          0.         ]
    GAUSSIAN = 2, 'Gauss',   'Gaussian',                  'CopulaGaussian', True,  ['tauK'], [-1.,         1.         ]
    STUDENT = 3,  'Student', 'Student',                   'CopulaStudent',  True,  ['tauK'], [-1.,         1.         ]
    GH = 4,       'GH',      'Gumbel-Hougaard',           'CopulaGH',       True,  ['tauK'], [0. + EPS,    1.         ]
    FGM = 5,      'FGM',     'Farlie-Gumbel-Morgenstern', 'CopulaFGM',      True,  ['tauK'], [-2. / 9.,    2. / 9.    ]
    CUBSEC = 6,   'CubSec',  'Cubique Section',           'CopulaCubSec',   True,  ['tauK'], [0.,          33. / 200. ]
    CLAYTON = 7,  'Clayton', 'Clayton',                   'CopulaClayton',  True,  ['tauK'], [0. + EPS,    1.         ]
    A12 = 8,      'A12',     'Archimedean12',             'CopulaA12',      True,  ['tauK'], [1. / 3.,     1.         ]
    A14 = 9,      'A14',     'Archimedean14',             'CopulaA14',      True,  ['tauK'], [1. / 3.,     1.         ]
    FRANK = 10,   'Frank',   'Frank',                     'CopulaFrank',    False, ['tauK'], [EPSMOINSUN,  UNMOINSEPS ]

    def describe(self):
        return self.name, self.value

    def correctTau(self, tau: float) -> float:
        """Clip tau to the valid range; replace non-finite values with the midpoint."""
        if not np.isfinite(tau):
            return (self.value.TAU_MIN_MAX[0] + self.value.TAU_MIN_MAX[1]) / 2.0
        return float(np.clip(tau, self.value.TAU_MIN_MAX[0], self.value.TAU_MIN_MAX[1]))

    @classmethod
    def favorite_copula(cls):
        return cls.GAUSSIAN

    @classmethod
    def printAvailableCopulas(cls):
        print('Available copulas:')
        for c in CopulaEnum:
            if c.value.AVAILABLE:
                print('  ', c.describe())

    @classmethod
    def getListAvailableCopulas(cls):
        return [c for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def getListAvailableCopulaClass(cls):
        return [c.CLASS_NAME for c in CopulaEnum if c.value.AVAILABLE]

    @classmethod
    def get_ID_From_ShortName(cls, short: str):
        for c in CopulaEnum:
            if c.value.SHORT_NAME == short:
                return c
        return None


class CopulaVirt:

    def __init__(self, className: str, copParamDict: dict):
        self.className = className

        # Locate the enum entry for this class
        self.copulaEnum = None
        for c in CopulaEnum.getListAvailableCopulas():
            if c.CLASS_NAME == self.className:
                self.copulaEnum = c
                self.tauMin     = c.value.TAU_MIN_MAX[0]
                self.tauMax     = c.value.TAU_MIN_MAX[1]
                break
        if self.copulaEnum is None:
            raise ValueError(f'CopulaVirt: copula {self.className!r} is not available.')

        # Validate tau admissibility
        if not (self.tauMin <= copParamDict['tauK'] <= self.tauMax):
            raise ValueError(
                f'CopulaVirt: tauK={copParamDict["tauK"]} out of [{self.tauMin}, {self.tauMax}] '
                f'for {self.className}.'
            )

        # Validate parameter names
        for k in copParamDict:
            if k not in self.copulaEnum.PARAMETERS_SET_NAME:
                raise ValueError(
                    f'CopulaVirt: unexpected parameter {k!r} for {self.className}. '
                    f'Expected: {self.copulaEnum.PARAMETERS_SET_NAME}'
                )

        self.CopParamDict = copParamDict
        if 'tauK' not in self.CopParamDict:
            self.updateTauK(0.0)

        self.updateInternalParam()

        # Grid for the numerical majorant (150 points on (0,1))
        self.N         = 150
        self.ticks_nbr = 15
        self.X  = np.linspace(EPS, UNMOINSEPS, self.N)
        self.Y  = np.zeros(self.N)

        # Grid for plotting (150×150)
        self.X1 = np.linspace(EPS, UNMOINSEPS, self.N)
        self.Y1 = np.linspace(EPS, UNMOINSEPS, self.N)
        self.X2, self.Y2 = np.meshgrid(self.X1, self.Y1)
        self.Z2          = np.zeros(self.X2.shape)

    # ------------------------------------------------------------------
    # Parameter update (must be overridden by every subclass)
    # ------------------------------------------------------------------
    def updateInternalParam(self):
        raise NotImplementedError(f'{self.__class__.__name__} must implement updateInternalParam().')

    # ------------------------------------------------------------------
    # PDF / CDF (must be overridden by every subclass)
    # ------------------------------------------------------------------
    def PdfCopule(self, VectU):
        raise NotImplementedError(f'{self.__class__.__name__} must implement PdfCopule().')

    def CdfCopule(self, VectU):
        raise NotImplementedError(f'{self.__class__.__name__} must implement CdfCopule().')

    # ------------------------------------------------------------------
    # Majorant  max_{v} c(uleft, v)   — overridden analytically when possible
    # ------------------------------------------------------------------
    def MajorantCopula(self, uleft: float) -> float:
        """Numerical majorant via grid search on 150 points. Subclasses may override."""
        u = minmaxEPS(uleft)
        for i, v in enumerate(self.X):
            self.Y[i] = self.PdfCopule([u, v])
        return float(np.max(self.Y))

    # ------------------------------------------------------------------
    # tau update
    # ------------------------------------------------------------------
    def updateTauK(self, newTauK: float):
        if self.tauMin <= newTauK <= self.tauMax:
            self.CopParamDict['tauK'] = newTauK
            self.updateInternalParam()
        else:
            self.CopParamDict['tauK'] = (self.tauMin + self.tauMax) / 2.0

    def getTauMinMax(self) -> tuple[float, float]:
        return self.tauMin, self.tauMax

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Copula name = {self.copulaEnum}; Tau Kendall= {self.CopParamDict['tauK']:.2f}"

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------
    def plotPdfCopule(self, plot_dir: str, string: str = ''):
        for i, a in enumerate(self.X1):
            for j, b in enumerate(self.Y1):
                self.Z2[j, i] = self.PdfCopule([a, b])

        fig, ax = plt.subplots(figsize=(6, 6), facecolor=facecolor)
        min_ = np.nanpercentile(self.Z2, 0)
        max_ = np.nanpercentile(self.Z2, 97)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        cs = ax.contourf(self.X2, self.Y2, self.Z2, vticks, antialiased=True, vmin=0, vmax=max_)
        plt.colorbar(cs, ticks=vticks)
        ax.set_aspect('equal')
        ax.set_xlabel('Left')
        ax.set_ylabel('Right')
        ax.set_facecolor('red')
        fig.suptitle(
            f"{self.copulaEnum.LONG_NAME} copula pdf — Tau: {self.CopParamDict['tauK']}",
            y=0.85, fontsize=BIGGER_SIZE
        )
        fname = os.path.join(plot_dir, f"{string}2D_PdfCopula{self.copulaEnum.LONG_NAME}-Tau={self.CopParamDict['tauK']}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=dpi, facecolor=facecolor)
        plt.close()

    def plotCdfCopule(self, plot_dir: str, string: str = ''):
        try:
            for i, a in enumerate(self.X1):
                for j, b in enumerate(self.Y1):
                    self.Z2[j, i] = self.CdfCopule([a, b])
        except NotImplementedError as e:
            print(f'plotCdfCopule skipped: {e}')
            return

        fig, ax = plt.subplots(figsize=(6, 6), facecolor=facecolor)
        min_ = np.nanpercentile(self.Z2, 0)
        max_ = np.nanpercentile(self.Z2, 100)
        vticks = np.linspace(min_, max_, num=self.ticks_nbr)
        cs = ax.contourf(self.X2, self.Y2, self.Z2, vticks, antialiased=True, vmin=0, vmax=max_)
        plt.colorbar(cs, ticks=vticks)
        ax.set_aspect('equal')
        ax.set_xlabel('Left')
        ax.set_ylabel('Right')
        ax.set_facecolor('red')
        fig.suptitle(
            f"{self.copulaEnum.LONG_NAME} copula cdf — Tau: {self.CopParamDict['tauK']}",
            y=0.85, fontsize=BIGGER_SIZE
        )
        fname = os.path.join(plot_dir, f"{string}2D_CdfCopula{self.copulaEnum.LONG_NAME}-Tau={self.CopParamDict['tauK']}.png")
        plt.savefig(fname, bbox_inches='tight', dpi=dpi, facecolor=facecolor)
        plt.close()


if __name__ == '__main__':
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    CopulaEnum.printAvailableCopulas()
    print('favorite:', CopulaEnum.favorite_copula())
