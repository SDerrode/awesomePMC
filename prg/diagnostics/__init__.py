"""
prg.diagnostics — goodness-of-fit and inference diagnostics.

This sub-package collects standalone diagnostic tools that complement the
estimation routines in :mod:`prg.pmc`. They are intentionally independent
of any specific model class: each utility takes raw NumPy arrays (and
optionally a CDF callable) and returns a typed result object.

Public API
----------
mks_1samp, mks_2samp, mks_test : multivariate Kolmogorov-Smirnov tests
    (Naaman 2021 finite-sample extension).
MKSResult                       : typed result of an MKS test.
"""

from prg.diagnostics.mks import (
    MKSResult,
    mks_1samp,
    mks_2samp,
    mks_test,
)

__all__ = [
    "MKSResult",
    "mks_1samp",
    "mks_2samp",
    "mks_test",
]
