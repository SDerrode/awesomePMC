"""
repro_gap_nodes.py — minimal reproduction of the second library problem found
by the Intel Lab study (erroneous data): with a strongly dependent copula,
the gap quadrature of :mod:`pmcprg.pmc.gaps` does not converge at the
default ``gap_nodes = 64``.

A missing row is integrated out on a fixed grid of G nodes, the quantiles of
the reference mixture of the margins. When the one-step conditional law is
much narrower than the node spacing, the quadrature cannot resolve it:

* the log-likelihood of a series with single-row gaps moves by hundreds to
  thousands of nats between G = 64, 128, 256 and 512 (it does not move at
  τ = 0.6);
* the PIT of the row right after a gap is over-dispersed (normal scores
  with sd ≫ 1), so ``flag_outliers`` flags it too often. With
  ``sequential=True`` every flagged row becomes a gap, which cascades.

Series: a PMC with K = 2, N(0, 1) and N(1, 1) state margins and Gaussian
copulas with τ on every pair. N = 3 000 rows are simulated and every 10th
row is removed.

Usage (from the repository root)
--------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/repro_gap_nodes.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pmcprg.pmc import PMCModel, predictive_pit, simulate  # noqa: E402


def model(tau: float) -> PMCModel:
    return PMCModel.from_dict({
        "model": {"name": "repro", "variant": "PMC", "K": 2, "N_default": 3000},
        "prior": {"p": [[0.49, 0.01], [0.01, 0.49]]},
        "margins": [{"i": 0, "dist": "norm", "params": {"loc": 0.0, "scale": 1.0}},
                    {"i": 1, "dist": "norm", "params": {"loc": 1.0, "scale": 1.0}}],
        "copulas": [{"i": i, "j": j, "name": "Gauss", "tau": tau}
                    for i in range(2) for j in range(2)],
    })


def main():
    logging.getLogger("pmcprg").setLevel(logging.ERROR)
    for tau in (0.6, 0.99):
        m = model(tau)
        _, Y = simulate(m, 3000, seed=1)
        Y = np.asarray(Y, float)
        Y[9::10] = np.nan
        after_gap = np.r_[False, ~np.isfinite(Y[:-1])] & np.isfinite(Y)
        after_obs = np.r_[False, np.isfinite(Y[:-1])] & np.isfinite(Y)
        print(f"tau = {tau}: {int(after_gap.sum())} rows after a gap")
        for G in (64, 128, 256, 512):
            P = predictive_pit(m, Y, gap_nodes=G)
            print(f"  G = {G:3d}: log-lik {P.log_lik:10.1f}   sd of z after a gap "
                  f"{np.std(P.z[after_gap]):.2f} (after an observed row "
                  f"{np.std(P.z[after_obs]):.2f})   p < 1e-3 after a gap: "
                  f"{int((P.pvalue[after_gap] < 1e-3).sum())}")


if __name__ == "__main__":
    main()
