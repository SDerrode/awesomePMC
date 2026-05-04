"""
prg.numerics — numerical constants and small utilities used throughout the package.

Constants
---------
EPS           : float — machine epsilon (≈ 2.22e-16).
ONE_MINUS_EPS : float — 1 − EPS, the largest float strictly less than 1.
EPS_MINUS_ONE : float — EPS − 1, the smallest float strictly greater than -1.
MIN_POSITIVE  : float — smallest positive normal float (≈ 2.225e-308). Use as a
                non-zero floor in normalisations where ``log()`` is then taken
                (``log(MIN_POSITIVE) ≈ -708`` is finite).

Helpers
-------
minmaxEPS(u)  : clamp ``u`` to the open unit interval ``(EPS, 1-EPS)``.
"""

import sys


EPS           = sys.float_info.epsilon   # ≈ 2.22e-16
ONE_MINUS_EPS = 1.0 - EPS
EPS_MINUS_ONE = EPS - 1.0
MIN_POSITIVE  = sys.float_info.min       # ≈ 2.225e-308


def minmaxEPS(u: float) -> float:
    """Clamp u to [EPS, 1-EPS] to keep values in the open unit interval."""
    return max(EPS, min(ONE_MINUS_EPS, float(u)))
