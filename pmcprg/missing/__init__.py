"""pmcprg.missing — data layer for missing observations.

Two modules, both free of any inference code:

* :mod:`pmcprg.missing.patterns` — artificial missingness patterns that follow
  the geometry of ImputeGAP's ``GenGap`` contamination module (Khayati et al.,
  arXiv:2503.15250; scenarios of *Mind the Gap*, PVLDB 13(5), 2020,
  doi:10.14778/3377369.3377383): ``mcar``, ``aligned``, ``scattered``,
  ``blackout``, ``disjoint``, ``overlap``, ``gaussian``, ``distribution``,
  plus :func:`~pmcprg.missing.patterns.mask_from_nan`, and two state-dependent
  (non-ignorable) masks drawn given a hidden state path, ``state_dependent``
  and ``state_markov`` (the ``[missingness]`` mechanisms of
  :mod:`pmcprg.pmc.missingness`). Each pattern returns ``(Y_masked, mask)``
  with ``mask`` True where a value was removed.

* :mod:`pmcprg.missing.metrics` — scores restricted to the masked positions:
  point metrics as in ImputeGAP (RMSE, MAE, 10-bin mutual information,
  Pearson), probabilistic ones (sample and Gaussian CRPS, interval coverage)
  and the classification error split between missing and observed positions.

Conventions: a univariate series is ``(N,)``; a multichannel one is
``(N, d)`` with one series per *column* (ImputeGAP's own orientation after
its ``logic_by_series`` transpose). Missing values are float NaN.
ImputeGAP is not a dependency; masks follow its geometry but are not
bit-identical to it (different random generator, see each docstring).
"""

from pmcprg.missing.metrics import (
    ErrorRates,
    crps_from_samples,
    crps_gaussian,
    error_rate_split,
    interval_coverage,
    mae,
    mutual_information,
    pearson,
    rmse,
)
from pmcprg.missing.patterns import (
    aligned,
    blackout,
    disjoint,
    distribution,
    gaussian,
    mask_from_nan,
    mcar,
    overlap,
    scattered,
    state_dependent,
    state_markov,
)

__all__ = [
    # patterns
    "mask_from_nan",
    "mcar",
    "aligned",
    "scattered",
    "blackout",
    "disjoint",
    "overlap",
    "gaussian",
    "distribution",
    "state_dependent",
    "state_markov",
    # metrics
    "rmse",
    "mae",
    "mutual_information",
    "pearson",
    "crps_from_samples",
    "crps_gaussian",
    "interval_coverage",
    "error_rate_split",
    "ErrorRates",
]
