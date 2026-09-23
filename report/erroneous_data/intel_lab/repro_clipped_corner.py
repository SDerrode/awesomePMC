"""
repro_clipped_corner.py — minimal reproduction of the library problem found by
the Intel Lab study (erroneous data): the non-gated predictive PIT of a
reading that follows another reading far beyond the margins.

``predictive_pit`` clips the copula arguments to [EPS, 1 − EPS]. When
y_{n−1} and y_n both lie beyond F⁻¹(1 − EPS) (about 8.1 sd for a Gaussian
margin), u and v are both clipped to 1 − EPS and the PIT becomes
h(1 − EPS | 1 − EPS): ≈ 0.5 for a strongly dependent Gaussian or
Gumbel–Hougaard copula, although the log predictive density of the same
row is about −1 200 (the likelihood rejects the value, the PIT accepts it).
A sensor stuck at 50 sd reads as "ordinary persistence": p ≈ 0.95. With
``sequential=True`` the filter never conditions on the flagged reading and
every row is flagged; Clayton and Frank give p ~ 1e-13 at the same corner.

Usage (from the repository root)
--------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/repro_clipped_corner.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmcprg.pmc import PMCModel, flag_outliers, predictive_pit  # noqa: E402


def model(family: str, tau: float) -> PMCModel:
    """PMC, K = 2, state margins N(0, 1) and N(1, 1), one copula family, τ on every pair."""
    return PMCModel.from_dict({
        "model": {"name": "repro", "variant": "PMC", "K": 2, "N_default": 7},
        "prior": {"p": [[0.45, 0.05], [0.05, 0.45]]},
        "margins": [{"i": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}}],
        "copulas": [{"i": i, "j": j, "name": family, "tau": tau}
                    for i in range(2) for j in range(2)],
    })


def main():
    logging.getLogger("pmcprg").setLevel(logging.ERROR)
    Y = np.array([0.0, 0.1, 0.2, 50.0, 50.0, 50.0, 100.0])
    np.set_printoptions(precision=3, linewidth=120)
    print("Y =", Y)
    for family in ("Gauss", "GH", "Clayton", "Frank"):
        m = model(family, 0.99)
        P = predictive_pit(m, Y)
        F = flag_outliers(m, Y, alpha=1e-3, sequential=True)
        print(f"{family:8s} non-gated p-value : {P.pvalue}")
        print(f"{'':8s} log predictive    : {np.round(P.log_pred, 0)}")
        print(f"{'':8s} gated p-value     : {F.pvalue}  flagged {F.flagged.astype(int)}")


if __name__ == "__main__":
    main()
