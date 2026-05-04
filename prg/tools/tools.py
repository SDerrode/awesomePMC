import sys
import numpy as np


EPS           = sys.float_info.epsilon   # ≈ 2.22e-16
ONE_MINUS_EPS = 1.0 - EPS
EPS_MINUS_ONE = EPS - 1.0


def minmaxEPS(u: float) -> float:
    """Clamp u to [EPS, 1-EPS] to keep values in the open unit interval."""
    return max(EPS, min(ONE_MINUS_EPS, float(u)))


def check_cov_matrix(M: np.ndarray) -> bool:
    """Return True if M is positive semi-definite (Cholesky check)."""
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False
