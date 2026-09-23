"""
repro_leading_gap.py — minimal reproduction of the third library problem found
by the Intel Lab study (erroneous data): a leading gap under strong serial
dependence is misintegrated, silently.

When the first L rows of a series are missing, the rows after the gap of a
stationary PMC have the law of the series started at row L + 1, so the exact
log p(y_obs) is the gap-free forward pass on Y[L:]. With τ ≥ 0.99 on the
diagonal copulas the local grids of :mod:`pmcprg.pmc.gaps` do not reproduce
it:

* both the batch pass (``gap_posterior``, hence ``classify`` and the ICE /
  SEM E-steps) and the PIT filter (``predictive_pit``, ``flag_outliers``)
  are off by nats at the default ``gap_nodes = 64``, and by tenths of a nat
  to nats at 256;
* the batch pass also loses the prior inside the gap: P(x_n | nothing
  observed) should be the stationary π at every missing row (the PIT filter
  keeps it to 1e-15), and it drifts by up to 0.3; its log-likelihood then
  differs from the filter's, although both are documented as equal;
* ``quad_error`` stays far below its WARNING threshold (1e-3), so nothing is
  logged: the diagnostic leaves the rows of a leading gap out.

Series: the PMC of repro_gap_nodes.py (K = 2, N(0, 1) and N(1, 1) state
margins, symmetric p, Gaussian copulas with τ on every pair), N = 400 rows
simulated with seed 1, the first 17 rows removed (the leading gap of Intel
Lab motes 48 and 47).

Usage (from the repository root)
--------------------------------
    .venv/bin/python report/erroneous_data/intel_lab/repro_leading_gap.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pmcprg.pmc import classify, gaps, predictive_pit, simulate  # noqa: E402
from repro_gap_nodes import model  # noqa: E402

L = 17


def main():
    logging.getLogger("pmcprg").setLevel(logging.ERROR)
    for tau in (0.997, 0.999):
        m = model(tau)
        pi = np.asarray(m.prior_p).sum(axis=1)
        _, Y = simulate(m, 400, seed=1)
        Y = np.asarray(Y, float)
        Yn = Y.copy()
        Yn[:L] = np.nan
        exact = float(classify(m, Y[L:])[2])
        limit = gaps.quad_warn_limit(~np.isfinite(Yn))
        print(f"tau = {tau}: rows 0-{L - 1} missing, exact log p(y_obs) = {exact:.3f}, "
              f"WARNING above quad_error = {limit:.0e}")
        for G in (64, 256, 1024):
            post = gaps.gap_posterior(m, Yn, gap_nodes=G, xi=False)
            P = predictive_pit(m, Yn, gap_nodes=G)
            print(f"  G = {G:4d}: gap_posterior {post.log_lik - exact:+8.3f} nats, "
                  f"max |alpha_hat - pi| {np.abs(post.alpha_hat[:L] - pi).max():.3f}, "
                  f"quad_error {post.quad_error:.0e}   predictive_pit {P.log_lik - exact:+8.3f} nats, "
                  f"max |alpha_hat - pi| {np.abs(P.alpha_hat[:L] - pi).max():.0e}")


if __name__ == "__main__":
    main()
