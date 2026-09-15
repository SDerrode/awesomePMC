"""
test_majorant.py — verify that ``cop.majorant(u)`` is a valid upper bound
on ``c(u, ·)`` for every copula family.

A ``majorant`` is correct iff  ``majorant(u) ≥ max_v c(u, v)`` for every
``u ∈ (0, 1)``. The acceptance-rejection sampler in
:class:`pmcprg.copulas.bivariate.BivariateLaw.sample_conditional` relies
on this property; an underestimating majorant biases the sampled
distribution silently.

We compare ``cop.majorant(u)`` against a high-precision reference
``true_max = max_v c(u, v)`` obtained by 1-D scalar optimisation, on a
random grid of u-values.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from pmcprg.copulas._base import CopulaEnum


# Canonical τ value per family (inside the valid range, away from boundary).
_FAMILY_TAU = {
    "Prod":   0.0,
    "FGM":    0.18,    # within [-2/9, 2/9]
    "CubSec": 0.10,    # within [0, 33/200]
    "AMH":    0.20,    # within [≈-0.18, 1/3)
}


def _tau_for(short: str, tau_min: float, tau_max: float) -> float:
    if short in _FAMILY_TAU:
        return _FAMILY_TAU[short]
    return 0.5 if tau_max > 0.5 else 0.5 * (tau_min + tau_max)


_RNG = np.random.default_rng(0)
_U_GRID = _RNG.uniform(0.05, 0.95, 50)


@pytest.mark.parametrize(
    "entry",
    [c for c in CopulaEnum if c.value.AVAILABLE],
    ids=lambda c: c.value.SHORT_NAME,
)
def test_majorant_is_upper_bound(entry):
    """``cop.majorant(u) ≥ max_v c(u, v)`` for every copula and every u."""
    short = entry.value.SHORT_NAME
    tau   = _tau_for(short, *entry.value.TAU_MIN_MAX)
    kwargs = {"tau_k": tau}
    if "delta" in entry.value.PARAMETERS_SET_NAME:
        kwargs["delta"] = 1.5
    if "df" in entry.value.PARAMETERS_SET_NAME:
        kwargs["df"] = 4.0

    cop = entry.klass(**kwargs)

    # Find the worst-case deficit across the random u-grid.
    max_deficit = 0.0
    for u in _U_GRID:
        bound = cop.majorant(float(u))
        # High-precision reference via 1-D bounded optimisation.
        res = minimize_scalar(
            lambda v: -cop.pdf([float(u), float(v)]),
            bounds=(1e-5, 1 - 1e-5), method="bounded",
            options={"xatol": 1e-9},
        )
        true_max = cop.pdf([float(u), float(res.x)])
        if true_max > bound:
            max_deficit = max(max_deficit, (true_max / bound) - 1.0)

    # Tolerance: 1e-4. Closed-form majorants give exact bounds; the
    # default-with-safety-margin overrides give ≥ 2 % overshoot.
    assert max_deficit < 1e-4, (
        f"{short} τ={tau:.3f}: majorant under-estimates max c(u,v) by "
        f"up to {max_deficit*100:.4f} %. The AR sampler will be biased."
    )
