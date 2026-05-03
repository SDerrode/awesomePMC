"""
Frank copula — symmetric, captures both positive and negative dependence.

Status: disabled pending investigation of numerical instability in theta
estimation via the Debye function for large |tau|. See point 4a of the
quality roadmap.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize  import root_scalar
from statsmodels.distributions.copula.api import FrankCopula

from prg.copulas._base import CopulaVirt, CopulaEnum


# ---------------------------------------------------------------------------
# Kendall's tau ↔ theta relationship for the Frank copula
# ---------------------------------------------------------------------------

def _debye_1(theta: float) -> float:
    """Debye function of order 1: D_1(θ) = (1/θ) ∫_0^θ t/(e^t − 1) dt."""
    if theta == 0.0:
        return 1.0
    integral, _ = quad(lambda t: t / (np.exp(t) - 1.0), 0.0, theta)
    return integral / theta


def kendall_tau_frank(theta: float) -> float:
    """τ_K as a function of θ for the Frank copula."""
    if theta == 0.0:
        return 0.0
    return 1.0 - (4.0 / theta) * (1.0 - _debye_1(theta))


def find_theta_frank(tau_target: float, bracket: tuple = (1e-5, 50.0)) -> float:
    """Invert kendall_tau_frank via Brent's method."""
    sol = root_scalar(
        lambda theta: kendall_tau_frank(theta) - tau_target,
        bracket=bracket, method='brentq'
    )
    if not sol.converged:
        raise ValueError(f'Frank theta inversion did not converge for tau={tau_target}.')
    return sol.root


# ---------------------------------------------------------------------------
# Copula class
# ---------------------------------------------------------------------------

class CopulaFrank(CopulaVirt):

    def __init__(self, **kwargs):
        # Guard: Frank is marked AVAILABLE=False in CopulaEnum.
        # This check is redundant (CopulaVirt.__init__ enforces it) but explicit.
        if not CopulaEnum.FRANK.AVAILABLE:
            raise NotImplementedError(
                'CopulaFrank is currently disabled (numerical instability — see frank.py). '
                'Set CopulaEnum.FRANK.AVAILABLE = True after fixing theta inversion for |tau| > 0.7.'
            )
        super().__init__(className=self.__class__.__name__, copParamDict=kwargs)

    def updateInternalParam(self):
        self.theta = find_theta_frank(self.CopParamDict['tauK'])
        self.copulastatmodels = FrankCopula(theta=self.theta)

    def PdfCopule(self, VectU):
        return float(self.copulastatmodels.pdf(VectU))

    def CdfCopule(self, VectU):
        return float(self.copulastatmodels.cdf(VectU))

    # Majorant numérique (hérité de CopulaVirt)
