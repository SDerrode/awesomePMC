import os
import sys
import numpy as np


EPS        = sys.float_info.epsilon   # ≈ 2.22e-16
UNMOINSEPS = 1.0 - EPS
EPSMOINSUN = EPS - 1.0


def minmaxEPS(u: float) -> float:
    """Clamp u to [EPS, 1-EPS] to keep values in the open unit interval."""
    return max(EPS, min(UNMOINSEPS, float(u)))


def set_dir(base: str, subdir: str) -> str:
    """Create base/subdir if absent and return its path."""
    path = os.path.join(base, subdir)
    os.makedirs(path, exist_ok=True)
    return path


def list_parameters(dist) -> list[str]:
    """Return the parameter names of a scipy continuous distribution."""
    params: list[str] = []
    if dist.shapes is not None:
        params += [s.strip() for s in dist.shapes.split(',')]
    params += ['loc', 'scale']
    return params


def check_CovMatrix(M: np.ndarray) -> bool:
    """Return True if M is positive semi-definite (Cholesky check)."""
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False
