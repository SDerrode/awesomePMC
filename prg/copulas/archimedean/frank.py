"""
Frank copula — symmetric, captures both positive and negative dependence.

Status: disabled pending investigation of numerical instability in theta
estimation via the Debye function for large |tau|. See point 4a of the
quality roadmap.
"""
if __name__ == '__main__':
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.integrate import quad
from scipy.optimize  import root_scalar
from statsmodels.distributions.copula.api import FrankCopula

from prg.copulas._base import CopulaVirt, CopulaEnum
from prg.exceptions    import CopulaNotAvailableError


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
        if not CopulaEnum.FRANK.AVAILABLE:
            raise CopulaNotAvailableError(
                'CopulaFrank is currently disabled (numerical instability). '
                'Set CopulaEnum.FRANK.AVAILABLE = True after fixing theta inversion for |tau| > 0.7.'
            )
        super().__init__(class_name=self.__class__.__name__, params=kwargs)

    def _update_params(self):
        self.theta = find_theta_frank(self.params['tau_k'])
        self._model = FrankCopula(theta=self.theta)

    def pdf(self, uv):
        return float(self._model.pdf(uv))

    def cdf(self, uv):
        return float(self._model.cdf(uv))

    # Numerical majorant (inherited from CopulaVirt)
