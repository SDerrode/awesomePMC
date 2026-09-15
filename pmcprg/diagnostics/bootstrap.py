"""
bootstrap.py — model-level parametric bootstrap for post-fit diagnostics.

Why this exists
---------------
Every diagnostic in this package that produces a p-value hit the same wall,
and the same fix was written five times before it was written once here.

A statistic applied to a *fitted* model does not have its textbook null
distribution. Three assumptions break at once, and in opposite directions:

* the reference distribution is estimated from the same data (the Durbin
  effect), which makes the statistic stochastically smaller;
* the observations are a Markov chain, not an i.i.d. sample, which inflates
  it;
* in a latent-state model the per-state sample has to be *assigned*, and the
  assignment depends on the data, which biases it again.

Which effect wins cannot be argued, only measured — and it was. Applying the
tabulated critical value of the multivariate KS test to a fitted PMC rejects
a **correct** model 10% of the time at a nominal 5% (1000 datasets): serial
dependence dominates and the test is anticonservative by a factor of two.

Resampling at the level of the *model* absorbs all three at once, because
each replicate goes through the identical pipeline — simulate a whole series
from the fitted model, re-run whatever the diagnostic does, compare. What
comes back is calibrated by construction rather than by argument. Measured:
0.043 and 0.048 for the copula goodness-of-fit test over 1200 measurements,
and 0.040 to 0.052 across eleven different margin statistics — all against a
nominal 0.05.

The caveat, deliberate and load-bearing
---------------------------------------
The model is **not refitted** on each replicate. Refitting would mean running
ICE B times, which is not affordable behind a GUI button. The null being
tested is therefore *"the fitted model, taken as given, generated Y"*, not
the composite null of Genest-Rémillard. Every caller inherits that reading
and should say so where it reports a p-value.

References
----------
* Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and Their
  Application*. Cambridge University Press, ch. 4 (the Monte-Carlo p-value).
* Phipson, B. & Smyth, G. K. (2010). Permutation p-values should never be
  zero. *Stat. Appl. Genet. Mol. Biol.* 9(1), Art. 39.
* Genest, C. & Rémillard, B. (2008). Validity of the parametric bootstrap for
  goodness-of-fit testing in semiparametric models. *Ann. Inst. H. Poincaré
  Probab. Statist.* 44(6), 1096–1127 — the composite-null bootstrap this
  harness deliberately does **not** run (no refit); without re-estimation
  the replicate statistics lack the Durbin shrinkage, so the test is
  conservative, not anticonservative — consistent with the levels above.
* Durbin, J. (1973). Weak convergence of the sample distribution function
  when parameters are estimated. *Ann. Statist.* 1(2), 279–290.
* Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit tests for
  copulas: A review and a power study. *Insurance Math. Econom.* 44(2),
  199–213.

Usage
-----
The caller supplies a ``statistic`` callable. It receives a model, a series
and a random generator, and returns a mapping from any hashable key to a
float. The keys are opaque here — a state index, a pair ``(i, j)``, a
statistic name — and come back as the keys of the result. Returning several
statistics from one call is the point: the expensive part is almost always
the forward-backward pass they share, not the statistics themselves.

    from pmcprg.diagnostics import parametric_bootstrap

    def stat(model, Y, rng):
        return {"ks": ..., "ad": ...}

    res = parametric_bootstrap(model, Y, stat, B=200)
    res["ad"].p_value
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["BootstrapResult", "parametric_bootstrap"]


@dataclass(frozen=True)
class BootstrapResult:
    """One statistic's observed value and its bootstrap p-value.

    Fields
    ------
    observed  : float — the statistic on the real data.
    p_value   : float — ``(1 + #{T_b ≥ T_obs}) / (n_valid + 1)``, the exactly
                valid Monte-Carlo p-value (Davison & Hinkley 1997, ch. 4; Phipson & Smyth 2010): it can never be 0, and the
                smallest reportable value is ``1/(B + 1)``. NaN when no
                replicate landed.
    n_valid   : int   — replicates that produced a finite value. Report it:
                a p-value resting on 12 of 200 draws is not the same claim as
                one resting on 200, and the difference is invisible in the
                number itself.
    B         : int   — replicates requested.
    n_series  : int   — length of the series actually used, which may be
                shorter than the input (see ``max_len``).
    """

    observed: float
    p_value: float
    n_valid: int
    B: int
    n_series: int


def parametric_bootstrap(
    model,
    Y,
    statistic: Callable[[Any, np.ndarray, np.random.Generator],
                        Mapping[Hashable, float]],
    *,
    B: int = 200,
    seed: int = 0,
    max_len: int | None = None,
    progress_cb: Callable[[int, int, float, str], None] | None = None,
    simulate_fn: Callable[..., tuple] | None = None,
) -> dict[Hashable, BootstrapResult]:
    """Calibrate ``statistic`` against series simulated from ``model``.

    Parameters
    ----------
    model : PMCModel
        The **fitted** model. It is both what is being tested and what
        generates the null — see the module docstring on what that means for
        the hypothesis actually being tested.
    Y : array-like
        The observations, shape ``(N,)`` or ``(N, d)``.
    statistic : callable
        ``statistic(model, Y, rng) -> {key: float}``. Called once on the real
        data and once per replicate. It may consume randomness — a posterior
        draw, for instance — and gets a generator for that; observed and
        replicates draw from the same stream, so a statistic that uses one
        draw is compared against nulls that use one draw too.
    B : int
        Replicates. Cost is linear in ``N·B``.
    max_len : int, optional
        Cap on the series length. Above it a **contiguous** prefix is used,
        and replicates are simulated at that same length so calibration is
        untouched — what is lost is sample size, not validity. Contiguous
        because thinning would destroy the serial dependence the bootstrap
        exists to absorb.
    progress_cb : callable, optional
        ``progress_cb(b, B, 0.0, "bootstrap")`` before each replicate.
    simulate_fn : callable, optional
        Defaults to :func:`pmcprg.pmc.simulate.simulate`. Injectable for tests
        and for model classes that simulate differently.

    Returns
    -------
    dict mapping each key returned by ``statistic`` to a
    :class:`BootstrapResult`. Keys seen only in replicates and never in the
    observed pass are dropped: there is nothing to compare them to.
    """
    if simulate_fn is None:
        from pmcprg.pmc.simulate import simulate as simulate_fn  # noqa: N813

    Y = np.asarray(Y)
    if max_len is not None and len(Y) > max_len:
        Y = Y[:max_len]                      # contiguous — see the docstring
    N = len(Y)

    rng = np.random.default_rng(seed)
    observed = dict(statistic(model, Y, rng))
    null: dict[Hashable, list[float]] = {k: [] for k in observed}

    for b in range(B):
        if progress_cb is not None:
            try:
                progress_cb(b, B, 0.0, "bootstrap")
            except Exception:                              # pragma: no cover
                pass
        try:
            _, Y_b = simulate_fn(model, N=N,
                                 seed=int(rng.integers(0, 2**31 - 1)))
            for key, value in statistic(model, Y_b, rng).items():
                if key in null and np.isfinite(value):
                    null[key].append(float(value))
        except Exception as exc:                           # pragma: no cover
            logger.debug("bootstrap replicate %d failed: %s", b, exc)

    out: dict[Hashable, BootstrapResult] = {}
    for key, value in observed.items():
        draws = np.asarray(null[key], dtype=float)
        # (1 + #{T_b ≥ T_obs}) / (B + 1), not the plain proportion: the observed
        # statistic counts as one draw from the null it is compared to, which
        # is what makes the p-value exactly uniform under H0 and never 0
        # (Davison & Hinkley 1997, ch. 4). Genest, Rémillard & Beaudoin
        # (2009, App. A) write the plain proportion; the two differ by at most
        # 1/(B + 1) — 0.005 at B = 200 — in the conservative direction. The
        # level measurements quoted in the module docstring were made with the
        # plain proportion. (Audit K-7.)
        p = (float((1 + np.sum(draws >= value)) / (draws.size + 1))
             if draws.size and np.isfinite(value) else float("nan"))
        out[key] = BootstrapResult(
            observed=float(value) if np.isfinite(value) else float("nan"),
            p_value=p, n_valid=int(draws.size), B=int(B), n_series=int(N),
        )
    return out
