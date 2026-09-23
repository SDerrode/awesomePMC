# Forecasting: pmcprg against pmmforecast, and outlier gating (forecasting)

> **Status (2026-09-23).** Parts 1, 3 and 4 were rerun in full on pmcprg
> commit **39f249f** (P1: local gap-quadrature grids; P2: empty state; P3:
> forecast/impute quantiles), with pmmforecast 58ac3b6. The run took 92.6 min
> of wall time with 0 task errors. `repro_pmcprg_problems.py` confirms that
> P1–P3 are gone. Part 2 is unchanged. Two caveats remain:
>
> - **Pending the leading-gap fix** (pmcprg's open problem, see
>   `report/erroneous_data/intel_lab/repro_leading_gap.py`): only the four
>   copula rows of `results/mote20_select.csv`. Mote 20 starts with one
>   missing row and those fits have τ up to 0.997. No forecast of this study
>   is conditioned on a series that starts with a gap (`run_info.json`,
>   `leading_gaps`).
> - **Quadrature converged to 3 %, not 1 %:** five fits of contaminated
>   Aotizhongxin by the copula PMC (marked † below).

This study compares pmcprg's forecasts with **pmmforecast**. pmmforecast is the
author's companion package for stationary 1-D Gaussian pairwise Markov models:
a continuous hidden state X and five parameters (a, b, c, d, e). Its reference
is Escudier, Abdelkefi, Fernandes & Pieczynski, *Forecasting with Pairwise
Gaussian Markov Models*, CMCSI 2023, arXiv 2402.07532. The study also measures
what outlier gating (`flag_outliers`, `robust_estimate`) does to forecasts.

## Conclusions (part 4)

**When does the discrete-regime PMC beat the continuous-state Gaussian PMM, and
conversely?** The paired CRPS differences below are per-origin means over h,
with 95 % bootstrap CIs.

- **The PMC wins when the data have regimes or isolated departures**
  (relative CIs are the bootstrap bounds divided by the PMM's CRPS).
  - *Series simulated from regime models* (clean fixtures): PMC −1.1 %
    [−1.6, −0.5] and HMC-IN −0.7 % [−1.0, −0.4] against the PMM. The gain is
    largest at h = 1 (−2.6 % and −2.7 %) and shrinks with h (−0.8 % and
    −0.1 % at h = 7–10).
  - *Aotizhongxin with spikes* (seed 0): −6.8 % [−13, −0.1] at 1 % and
    −13.7 % [−21, −5.8] at 5 %; −43 % and −58 % at h = 1.
  - The reason: ICE turns the third regime into a **spike regime**. Its
    probability is the spike rate (0.011, 0.049), its self-transition is 0
    and its pair margins sit at the spike level (ln PM2.5 of 9.6 and 10.8 at
    5 %; this is the model of P3). The fit therefore forecasts
    a return to normal after a spike, and its CRPS barely moves: 0.557 clean,
    0.556 at 1 %, 0.561 at 5 %. The PMM goes 0.561 → 0.595 → 0.649.
- **Clean Aotizhongxin: a tie.** The PMC pair K = 3 is −1.2 % [−6, +3.5]
  against the PMM at every horizon band. Its 95 % intervals are narrower
  (width 3.08 against 3.28) and cover less (0.88 against 0.91). In this
  study the PMM behaves as an AR(1): paired difference +0.0001.
- **The PMM wins on smooth, trending series.** tsNH4: the PMM is 7.0 %
  [+0.1, +14] better than the PMC state K = 2, 7.3 % at h = 7–24. The fitted
  PMM is a genuine ARMA(2, 1) (tr A = 1.91, det A = 0.91, spectral radius
  0.979): it carries momentum that a PMC with an AR(1)-like regime lacks. The
  AR(1) is 5.6 % behind the PMM too.
- **Mote 20 (30 s, near-deterministic).**
  - pmcprg cannot forecast it with a copula PMC: none converges in the
    quadrature at 256 nodes (see "Checks"). HMC-IN, which ignores the
    y_n → y_{n+1} dependence, has 7.7× the PMM's CRPS.
  - Persistence beats the PMM by 16.8 % [4, 26]. The PMM's predictive variance
    is 4–6× its empirical MSE at h ≤ 6.
- **HMC-IN is always last among the model-based forecasts on real series:**
  +7 % (Aotizhongxin), +43 % (tsNH4) and +680 % (mote 20) against the PMM.
  Real series need the dependence of y_{n+1} on y_n that the copulas or the
  PMM carry.

**What does gating buy?**

- **With the fit's own model: little.** On every clean real series, the
  gated forecasts of the PMC and HMC-IN are within 0.3 % of the ungated ones,
  for 0–15 test flags per 1 000.
- **On contaminated data, gating helps only when the fit has not absorbed
  the spikes.**
  - PMC fixture at 1 %: raw + gate −0.3 %, robust + gate −1.0 %.
  - At 5 % the raw ICE fit makes the spikes a regime. Gating then flags
    nothing (test detection 0.000), and `robust_estimate` stops at the same
    fit (mask detection 0.001).
- **The Hampel pre-screen is what works at 5 %.** Its robust + gate matches
  the true model gated:
  - PMC fixture: 0.746 against 0.743;
  - HMC-IN fixture: 0.770 against 0.770, where the raw fit gets 0.818;
  - HMC-IN on Aotizhongxin: −6.3 % [−12, −0.1].
- **The discrete PMC needs no gating on contaminated Aotizhongxin:** its spike
  regime already does the job (+0.0 % raw + gate against raw).
- **Gating the true model** is the ceiling: 5 % spikes cost 8 % of CRPS
  ungated and 0.4 % gated (PMC fixture, 0.800 → 0.743).

**Would innovation gating in pmmforecast's Kalman filter be worthwhile?**
Measured on the Gaussian AR(1) with the closed-form innovation gate
(`ar1_flags`, the gate a Kalman filter would apply). pmmforecast's fits on
the real series are AR(1)-like, and on Aotizhongxin its forecasts equal the
AR(1)'s to +0.0001 CRPS.

- **A gate with the fit's own parameters: no.** It changes the CRPS by 0 to
  −0.8 % on contaminated data (fixtures and Aotizhongxin, 1 and 5 %). The fit
  on contaminated data inflates the innovation variance, so spikes pass: the
  Aotizhongxin AR(1) goes from σ² = 0.12, φ = 0.95 to σ² = 2.8, φ = 0.31 at 5 %.
- **A gate plus flag-and-mask refitting: large gains on contaminated data.**
  - Fixtures at 5 %: −9.9 % and −12.6 %, back to the clean fit (0.791
    against 0.790 on the HMC-IN fixture).
  - Aotizhongxin at 5 %: −9.3 %, 0.713 → 0.650, level with the PMM's 0.649.
- **It is costly or catastrophic on clean real series.** The Gaussian gate
  loses lock after level shifts:
  - Aotizhongxin: +4.9 % (raw gate, 24 test flags per 1 000) and +13 %
    (robust, 119 per 1 000);
  - tsNH4: +1.6 % and +4 %;
  - mote 20: 266 and 904 flags per 1 000, CRPS ×17 and ×53.
- **Answer from these measurements:** not as a default.
  - Innovation gating pays only with a refit loop, and then only where the
    contamination is heavy. With refitting, the Aotizhongxin AR(1) loses
    13 % on the clean series and gains 9 % at 5 % spikes.
  - On the 30 s mote it fails outright: the Gaussian gate flags the level
    shifts and loses lock.
  - The discrete PMC's alternative, a regime for the departures, gave the
    best contaminated forecasts here without any gating.

## Library problems, pmcprg

### P1–P3: fixed at 39f249f (reproductions: `repro_pmcprg_problems.py`)

The stopped run found three silent problems: P1, the gap quadrature
collapsing under strong copula dependence; P2, an empty state giving NaN
PITs; P3, forecast quantiles disagreeing with the returned law. On 39f249f:

```bash
PYTHONPATH=. .venv/bin/python report/forecasting/repro_pmcprg_problems.py
```

```
P1 — forecast / predictive_pit after a missing row, AR(1)(ρ) as a PMC (G = 64 nodes):
   ρ = 0.99: 2-step sd 0.1985 (exact 0.1985); PIT after a gap 0.5321 (exact 0.5321)
   ρ = 0.998: 2-step sd 0.0893 (exact 0.0893); PIT after a gap 0.5499 (exact 0.5499)
   ρ = 0.999: 2-step sd 0.0632 (exact 0.0632); PIT after a gap 0.5666 (exact 0.5666)
   ρ = 0.9999: 2-step sd 0.0200 (exact 0.0200); PIT after a gap 0.6925 (exact 0.6925)
P2 — predictive_pit / flag_outliers with an all-zero row of p (8 sd spike at row 100):
   p = [[0.0, 0.0], [0.0, 1.0]]: NaN PITs 0/200, log_lik -321.865, flagged 1
   p = [[1e-12, 1e-12], [1e-12, 0.999999999997]]: NaN PITs 0/200, log_lik -321.865, flagged 1
P3 — forecast quantiles against the forecast's own node law (ICE fit of Aotizhongxin with 5 % spikes):
   G =  32: h = 2 mean 4.264, sd 1.437; returned 2.5/50/97.5 % [ 3.039  4.007 10.464]; median of the node law 3.993
   G =  64: h = 2 mean 4.257, sd 1.424; returned 2.5/50/97.5 % [ 3.037  4.004 10.468]; median of the node law 4.055
   G = 128: h = 2 mean 4.256, sd 1.424; returned 2.5/50/97.5 % [ 3.04   4.003 10.468]; median of the node law 4.023
   G = 256: h = 2 mean 4.256, sd 1.424; returned 2.5/50/97.5 % [ 3.04   4.003 10.468]; median of the node law 4.007
```

- **P1 is fixed.** The 2-step sd and the PIT after a gap equal the closed form
  at every ρ. Before: 0.0000 for 0.0200 at ρ = 0.9999, and a PIT of 0.6473
  for 0.6925.
- **P2 is fixed.** No PIT is NaN, the 8 sd spike is flagged, and the
  log-likelihood is that of the model without the empty state. Before: 199
  NaN PITs and nothing flagged.
- **P3 is fixed.** The returned quantiles are stable from G = 32 to 256.
  Before: 13.40 / 15.77 / 16.69 at G = 64, for a law of mean 4.26. The last
  column is the median of the reference nodes weighted by `density`, a coarse
  discretisation that moves by 0.05 around 4.00.

### Leading gap under strong dependence (open; found by report/erroneous_data)

A series that starts with missing rows is misintegrated when τ ≥ 0.997, and
no WARNING is logged. In this study:

- **The only series that starts with a gap is mote 20** (epoch 1 absent,
  1 row).
- **No fit or forecast conditioned on it is used.** Every copula PMC of the
  mote fails the quadrature check (below). HMC-IN has no grid.
- **Affected: the four copula rows of `mote20_select.csv`** (τ max 0.994–0.997,
  leading gap 1), pending the fix. Their log-likelihoods come from `classify`
  on the training part, and their ICE fits had the same leading gap.
- **Measured on the fitted PMC state K = 2**, first 40 and 400 rows, the
  series against the same series without its first row:
  - log-likelihood: 0.025 nats off at G = 64 and 0.0000 at G = 256, the same
    through `gap_posterior` and `predictive_pit`;
  - 6-step forecasts: 1e-14 sd.
- **No other fitted, conditioning or gated series of the campaign starts with
  a missing row.** `fits.csv`: `lead_gap_fit`, `lead_gap_cond`,
  `lead_gap_gated`.

### Observations for the library author (not defects)

- **The `quad_error` WARNING is conservative on forecast chains.**
  - The §4 Aotizhongxin PMC (pair K = 3, τ ≤ 0.86) logs it on all 91
    forecasts at G = 64: quad_error median 0.029, max 0.75, against a limit
    of 0.004.
  - At the three worst origins, its forecasts agree with G = 512 to
    6.4e-3 predictive sd in mean and 5.7e-3 in sd.
  - In the campaign it fired 3 710 times over 131 task steps. It is useful
    only as a screen, and the study's own check decides G.
- **Models with a spike regime converge slowly in G.** These are the ICE fits
  of contaminated Aotizhongxin: 1–5 % of mass at ln PM2.5 ≈ 10, the rest
  near 4.
  - Their means and sds change by 1.7–2.8 % of the predictive sd between
    128 and 256 nodes.
  - The returned 97.5 % quantile departs from its own node law by up to
    0.04 sd from one value and 0.064 sd from a 50-row window at G = 512,
    with the quantile-fallback WARNING. The pre-fix code gave −0.033 sd on
    the same window.
  - The study scores the node law, so it does not depend on the returned
    quantiles.
- **Mote 20 (τ up to 0.997, 4 373 missing rows) still does not converge at
  G = 256.**
  - Between 128 and 256 nodes the forecast means and sds change by 6.7–17 %
    of the predictive sd; quad_error is 5–34× its limit, so pmcprg warns.
  - One forecast of the full series costs 18 s at G = 256 (one quantile
    level), so the mote campaign at G = 512 would take hours.
  - For the PMC state K = 2 fit, the log-likelihood of the training part
    (21 600 rows, 3 409 missing) is 41 043.29 at G = 64, 41 032.48 at 128,
    41 032.51 at 256 and 41 032.97 at 512.

## pmmforecast issues (for its author; its repository was not modified)

pmmforecast is exported from commit **`58ac3b624d9bc006b0fdf96a03c9db5f566c8f9f`**
(v0.22.1): `git archive HEAD` into the session scratchpad, then a fresh venv
(Python 3.12, numpy 2.5.3, scipy 1.18.1). The snippets below run in that venv.

**M1. Y-only identifiability is wider than documented.** The docstring of
`estimate_pmm_mle_y_only`, Tutorial 09 and the property test describe the
ambiguity as the sign flip (b, d, e) → −(b, d, e). In fact the law of Y
depends on the tuple only through (c, tr A, det A), with

- tr A = (a + c − b(d + e)) / (1 − b²)
- det A = (ac − de) / (1 − b²)

(derivation in part 2). This leaves a 2-dimensional set of tuples with the
same Y-only likelihood; the swap d ↔ e is one of them. Checked on one path in
the table of part 2: tuples with b = 0.2, 0.5 and 0.7 give the same NLL, with
a difference of exactly 0.0 in floating point. The estimated tuple is
therefore not interpretable; its forecasts of Y are.

**M2. Default starts miss the maximum on persistent series.** The default
starts of `estimate_pmm_mle_y_only` are (0.5, 0.3, 0.1, 0.4, 0.2), then
uniform draws in [−0.7, 0.7]^5. On persistent series they converge to a
spurious region. Full campaign (log-likelihood on the data scale; default
starts, then the added start (0.5, 0.3, r, 0.3 r, 0.3 r), an AR(1) inside
the PMM family, r being the lag-1 autocorrelation of the standardised
training part, clipped at 0.999):

| series | r | default | AR(1) start | exact AR(1) MLE | default c → chosen c |
|---|---|---|---|---|---|
| tsNH4 | 0.994 | 3 009.4 | 3 051.1 | 3 017.9 | 0.077 → 0.994 |
| mote 20, suspect removed | 0.999 | 31 962.4 | 39 308.4 | 28 252.4 | 0.481 → 0.9999 |
| mote 20, as recorded | 0.960 | −27 216.5 | −25 585.3 | −27 070.2 | 0.938 → 0.959 |
| Aotizhongxin | 0.945 | −2 253.0 | −2 252.1 | −2 258.4 | 0.946 → 0.946 |

The default fit also lies below the AR(1) sub-model on tsNH4 and mote 20 as
recorded. Its forecasts are not always worse: on mote 20 the default fit
scores CRPS 0.091 against 0.098 for the higher-likelihood fit (difference
−8 % [−17, +3]).

```python
import numpy as np
from prg.models import ParamPMM, PMMSimulator1D
from prg.inference.estimation import estimate_pmm_mle_y_only, neg_log_likelihood_y_only, standardise
sim = PMMSimulator1D(ParamPMM(0.0, 0.0, 0.99, 0.0, 0.0), n_samples=2000); sim.set_seed(7)
y = np.array([tuple(v) for v in sim.simulate_generator()])[:, 1]
print(neg_log_likelihood_y_only(np.array([0, 0, 0.99, 0, 0]), standardise(y)))   # -1243.6
print(estimate_pmm_mle_y_only(y, n_restarts=4, seed=0)[1].fun)                      # -1230.2, c ≈ 0.50
```

This study's workaround is `pmm_side.fit_pmm`, which uses pmmforecast's own
API (`x0`, a user parameter). It adds one start at (0.5, 0.3, r, 0.3 r, 0.3 r)
and keeps the higher likelihood. The default-start fit is kept and forecast
too, as `pmm_default`.

**M3. A NaN gives a bare `AssertionError`.** One NaN in y makes
`standardise` return all-NaN values. Every restart's NLL is then +inf, and
`estimate_pmm_mle_y_only` fails on `assert best_result is not None` after
thousands of RuntimeWarnings from scipy. `neg_log_likelihood_y_only` returns
+inf silently. pmmforecast has no missing-value handling:
`filter_observations` raises `NumericError` at the first NaN.

**M4. `TheoreticalMSE`: which output is Y's, and memory.**

- `compute_th_mse_for_pmm()[0]` is the MSE of **X**. The Y forecast MSE is the
  (Y, Y) entry of the covariance table `[1][:, 1, 1]`; the docstring does not
  say so.
- The constructor materialises an n × n Toeplitz table: 350 MB at n = 6 570,
  3.7 GB at n = 21 600. This study evaluates it at n0 = 500, where the filter
  variance has long converged.

**M5. Cost of the Y-only MLE on long series.** The two
`estimate_pmm_mle_y_only` calls on mote 20 (21 599 training rows) took 29 min
(suspect readings removed) and 18 min (as recorded). The four-worker
pmmforecast side took 126.5 min in total (`results/pmm_fits.csv`, `seconds`).

## Part 1 — forecasts on real series

Scores are averaged over the origins: 91, 93 and 120. Rows with a missing
truth are skipped, which leaves 90 and 106 at h = 1 for Aotizhongxin and
mote 20. Every table is in `results/tables.md`.

**Beijing Aotizhongxin (ln PM2.5, 1 h)**

| h | PMC pair K=3 | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|---|
| RMSE 1 | 0.396 | 0.591 | 0.396 | 0.398 | 0.397 | 0.410 |
| RMSE 6 | 1.040 | 1.087 | 1.011 | 1.012 | 1.011 | 1.109 |
| RMSE 24 | 1.214 | 1.207 | 1.187 | 1.187 | 1.187 | 1.489 |
| RMSE 1–24 | 1.009 | 1.050 | 0.996 | 0.996 | 0.996 | 1.158 |
| CRPS 1 | **0.199** | 0.334 | 0.206 | 0.207 | 0.206 | 0.210 |
| CRPS 2 | 0.320 | 0.403 | 0.318 | 0.319 | 0.318 | 0.339 |
| CRPS 3 | 0.386 | 0.449 | 0.389 | 0.390 | 0.389 | 0.414 |
| CRPS 6 | 0.572 | 0.624 | 0.577 | 0.577 | 0.576 | 0.614 |
| CRPS 12 | 0.594 | 0.632 | 0.608 | 0.608 | 0.609 | 0.650 |
| CRPS 24 | 0.709 | 0.707 | 0.689 | 0.689 | 0.689 | 0.859 |
| CRPS 1–24 | **0.557** | 0.600 | 0.561 | 0.562 | 0.561 | 0.632 |
| coverage 50 / 80 / 95 % | 0.35 / 0.67 / 0.88 | 0.35 / 0.68 / 0.89 | 0.38 / 0.74 / 0.91 | 0.38 / 0.74 / 0.91 | 0.38 / 0.75 / 0.91 | 0.42 / 0.69 / 0.89 |
| 95 % width | 3.08 | 3.43 | 3.28 | 3.28 | 3.30 | 4.17 |

§4 of `report/real_series` is reproduced to the third decimal for all four
models it had (RMSE at h = 1 / 6 / 24):

- PMC 0.396 / 1.040 / 1.214;
- HMC-IN 0.591 / 1.087 / 1.207;
- AR(1) 0.397 / 1.011 / 1.187;
- persistence 0.410 / 1.109 / 1.489.

**tsNH4 (ln NH4, 10 min)**

| h | PMC state K=2 | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|---|
| RMSE 1 | 0.089 | 0.304 | 0.087 | 0.087 | 0.088 | 0.088 |
| RMSE 6 | 0.186 | 0.341 | 0.158 | 0.182 | 0.183 | 0.188 |
| RMSE 24 | 0.586 | 0.608 | 0.547 | 0.546 | 0.570 | 0.612 |
| RMSE 1–24 | 0.388 | 0.471 | **0.352** | 0.368 | 0.379 | 0.400 |
| CRPS 1 | 0.050 | 0.185 | 0.049 | 0.050 | 0.050 | 0.050 |
| CRPS 2 | 0.065 | 0.181 | 0.063 | 0.064 | 0.064 | 0.065 |
| CRPS 3 | 0.073 | 0.190 | 0.071 | 0.074 | 0.074 | 0.074 |
| CRPS 6 | 0.104 | 0.207 | 0.097 | 0.104 | 0.105 | 0.106 |
| CRPS 12 | 0.200 | 0.265 | 0.186 | 0.194 | 0.199 | 0.204 |
| CRPS 24 | 0.327 | 0.327 | 0.305 | 0.301 | 0.316 | 0.336 |
| CRPS 1–24 | 0.194 | 0.260 | **0.181** | 0.185 | 0.191 | 0.198 |
| coverage 50 / 80 / 95 % | 0.51 / 0.74 / 0.89 | 0.52 / 0.84 / 0.99 | 0.67 / 0.88 / 0.97 | 0.61 / 0.81 / 0.91 | 0.60 / 0.81 / 0.91 | 0.54 / 0.83 / 0.99 |
| 95 % width | 1.10 | 2.17 | 1.54 | 1.21 | 1.26 | 1.83 |

**Intel Lab mote 20 (°C, 30 s).** pmcprg has no copula PMC here: none of the
four passed the quadrature check at 256 nodes (`results/mote20_select.csv`).
HMC-IN K = 3 is the BIC choice among the converged fits.

| h | HMC-IN K=3 | PMM (pmmforecast) | PMM, default starts | AR(1) | persistence |
|---|---|---|---|---|---|
| RMSE 1 | 1.317 | 0.019 | 0.025 | 0.021 | 0.021 |
| RMSE 6 | 1.337 | 0.060 | 0.080 | 0.080 | 0.081 |
| RMSE 24 | 1.417 | 0.401 | 0.427 | 0.426 | 0.427 |
| RMSE 1–24 | 1.359 | **0.272** | 0.289 | 0.289 | 0.289 |
| CRPS 1 | 0.738 | 0.013 | 0.017 | 0.015 | **0.011** |
| CRPS 6 | 0.742 | 0.043 | 0.043 | 0.045 | **0.036** |
| CRPS 24 | 0.795 | 0.174 | 0.153 | 0.155 | **0.142** |
| CRPS 1–24 | 0.759 | 0.098 | 0.091 | 0.092 | **0.083** |
| coverage 50 / 80 / 95 % | 0.44 / 0.78 / 0.93 | 0.90 / 0.97 / 0.98 | 0.78 / 0.90 / 0.94 | 0.81 / 0.91 / 0.95 | 0.57 / 0.84 / 0.95 |

**Paired CRPS differences against the PMM** (per-origin mean over h, 95 %
bootstrap CI, 2 000 resamples; `results/part1_paired.csv`):

| series | model | mean difference | CI | better in |
|---|---|---|---|---|
| Aotizhongxin | PMC pair K=3 | −0.0067 (−1.2 %) | [−0.033, +0.018] | 52 % |
| Aotizhongxin | HMC-IN K=3 | +0.040 (+7.0 %) | [+0.014, +0.067] | 33 % |
| Aotizhongxin | AR(1) | +0.0001 | [−0.0009, +0.0011] | 47 % |
| tsNH4 | PMC state K=2 | +0.013 (+7.0 %) | [−0.000, +0.026] | 59 % |
| tsNH4 | AR(1) | +0.010 (+5.6 %) | [+0.000, +0.021] | 57 % |
| tsNH4 | HMC-IN K=3 | +0.079 (+43 %) | [+0.058, +0.101] | 18 % |
| mote 20 | persistence | −0.016 (−17 %) | [−0.026, −0.005] | 84 % |
| mote 20 | AR(1) | −0.0067 (−7 %) | [−0.015, +0.004] | 84 % |
| mote 20 | HMC-IN K=3 | +0.66 (+680 %) | [+0.56, +0.76] | 1 % |

By horizon band, PMC against PMM on Aotizhongxin: −3.6 % at h = 1, −1.2 % at
h = 2–6, −1.1 % at h = 7–24; no CI excludes 0. On tsNH4: +1.0 %, +4.2 % and
+7.3 %.

**PMM: empirical MSE against `TheoreticalMSE`** (Y entry, paper Eqs. 18–24,
n0 = 500; ratio = empirical / theoretical):

| series | h = 1 | 2 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|
| Aotizhongxin | 1.35 | 1.64 | 1.64 | 1.93 | 1.43 | 1.38 |
| tsNH4 | 0.77 | 0.59 | 0.45 | 0.35 | 0.67 | 0.85 |
| mote 20 | 0.23 | 0.20 | 0.18 | 0.18 | 1.32 | 0.70 |

The theoretical MSE equals the mean predictive variance of the forecasts,
so the ratio measures calibration:

- Aotizhongxin: the PMM is over-confident (its 95 % intervals cover 0.91).
- tsNH4 and mote 20: it is under-confident at short horizons (coverage 0.97
  and 0.98).

The test parts are not stationary with respect to the training parts. Part 2
shows that, under the model, the formula matches Monte Carlo.

## Part 2 — Gaussian cross-check (complete, unchanged)

**When do the two packages describe the same law of Y?**

*PMM side.* Write Z_n = (X_n, Y_n), with Q₁ = Cov(Z_n) = [[1, b], [b, 1]],
Q₂ = Cov(Z_{n+1}, Z_n) = [[a, e], [d, c]] and A = Q₂Q₁⁻¹. The autocovariance
of Y is γ(1) = c and γ(k) = [A^{k−1}Q₂]_{YY}.

- Cayley–Hamilton (A² = tr A · A − det A · I) and A⁻¹Q₂ = Q₁ give
  γ(k) = tr A · γ(k−1) − det A · γ(k−2) for k ≥ 2, with γ(0) = 1.
- So Y is a Gaussian ARMA(2, 1) whose law depends on (c, tr A, det A) only
  (M1).
- It is an AR(1) iff γ(2) = c², that is **(d − bc)(e − bc) = 0**:
  - if d = bc, the Y-row of A is (0, c), so Y_{n+1} = cY_n + noise
    independent of the past;
  - if e = bc, the same holds in reversed time, and a stationary Gaussian
    process has the same law reversed.
- The HMM (c = ab², d = e = ab) is never an AR(1) unless ab = 0.

*pmcprg side.* Take a PMC with Gaussian margins and Gaussian copulas.

- The law of (y_n, y_{n+1}) is the finite mixture Σ p_ij of bivariate
  Gaussians. By identifiability of finite Gaussian mixtures, it is Gaussian
  only if all components with p_ij > 0 are the same bivariate Gaussian.
- Then the transition of Y does not depend on the regimes: Y is a Markov
  Gaussian process, an AR(1).

*The coincidence.* The two describe the same law of Y exactly when:

- the PMM has (d − bc)(e − bc) = 0;
- the PMC has identical regimes (K ≥ 2 in pmcprg): margins N(μ, σ²), Gaussian
  copulas of ρ = c (Kendall τ = (2/π) arcsin c) on every pair, any prior;
- y = μ + σY.

No other coincidence exists. A generic PMM (ARMA(2, 1)) or an HMM has no exact
pmcprg counterpart.

**Checks.** Paths are simulated by pmmforecast (`pmm_side.py crosscheck`).
The pmcprg model is two identical regimes at y = 2 + 0.5 Y.

Log-likelihoods (N = 2 000):

| PMM (a, b, c, d, e) | d − bc | e − bc | Y AR(1) | pmmforecast Y-only | pmcprg | AR(1) closed form | pmcprg − pmmforecast |
|---|---|---|---|---|---|---|---|
| 0.8 0.5 0.7 0.35 0.6 | 0 | 0.25 | yes | −755.0942 | −755.0942 | −755.0942 | +3.0e-12 |
| 0.8 0.5 0.7 0.6 0.35 | 0.25 | 0 | yes | −773.5301 | −773.5301 | −773.5301 | +1.6e-12 |
| 0.8 0.5 0.7 0.5 0.65 | 0.15 | 0.30 | no | −728.3943 | −757.1987 | −757.1987 | −28.8 |
| HMM 0.9 0.8 | 0.26 | 0.26 | no | −963.8185 | −1057.5029 | −1057.5029 | −93.7 |

In the last two rows, even the best AR(1) (MLE) stays 27.4 and 75.8 nats below
the PMM.

Forecasts at origins 50, 500, 1 000 and 2 000, h = 1…20, largest absolute
differences:

- pmcprg (quadrature grid) against the AR(1) closed form: 3e-8 in the mean,
  1.7e-7 in the sd, 1e-6 in the quantiles.
- pmmforecast (filter, A^k, covariance recursion) against the closed form:
  2e-16 when Y is an AR(1).
- For the generic PMM and the HMM, pmmforecast's own forecasts differ from the
  AR(1)(c) by up to 0.05 and 0.35 in the mean, as they should.

`TheoreticalMSE` (Y entry, paper Eqs. 18–24) equals pmmforecast's predictive
variance exactly. Monte-Carlo MSE, 40 000-step path, 975 origins, model
scale:

| PMM | h | TheoreticalMSE | 1 − c^{2h} | AR(1)(c) MSE under the PMM | empirical, pmmforecast | empirical, pmcprg AR(1)(c) |
|---|---|---|---|---|---|---|
| d = bc | 1 | 0.5100 | 0.5100 | 0.5100 | 0.500 ± 0.023 | 0.500 ± 0.023 |
| d = bc | 3 | 0.8824 | 0.8824 | 0.8824 | 0.918 ± 0.042 | 0.918 ± 0.042 |
| e = bc | 1 | 0.5100 | 0.5100 | 0.5100 | 0.491 ± 0.021 | 0.491 ± 0.021 |
| generic | 1 | 0.4985 | 0.5100 | 0.5100 | 0.476 ± 0.022 | 0.493 ± 0.022 |
| generic | 3 | 0.7725 | 0.8824 | 0.8028 | 0.732 ± 0.033 | 0.768 ± 0.034 |
| HMM | 1 | 0.5975 | 0.6682 | 0.6682 | 0.577 ± 0.026 | 0.634 ± 0.029 |
| HMM | 3 | 0.7359 | 0.9635 | 0.8582 | 0.722 ± 0.035 | 0.820 ± 0.038 |

The empirical MSEs agree with the formulas within about 2 standard errors at
every h = 1…10 (`results/crosscheck_mse.csv`). For the AR(1)-type PMMs, the
two packages' forecasts coincide origin by origin. The AR(1)(c) approximation
of a generic PMM costs 2–5 % of MSE; for the HMM it costs 12–17 %.

Identifiability (M1): pmmforecast's Y-only NLL on the "generic" path.

| tuple (a, b, c, d, e) | c | tr A | det A | NLL |
|---|---|---|---|---|
| 0.8 0.5 0.7 0.5 0.65 | 0.7 | 1.2333 | 0.3133 | 2114.688681 |
| −(b, d, e) | 0.7 | 1.2333 | 0.3133 | 2114.688681 |
| d ↔ e | 0.7 | 1.2333 | 0.3133 | 2114.688681 |
| 0.6458 0.2 0.7 0.2934 0.5156 | 0.7 | 1.2333 | 0.3133 | 2114.688681 |
| 0.8825 0.7 0.7 0.7579 0.6042 | 0.7 | 1.2333 | 0.3133 | 2114.688681 |
| Y-only MLE, 4 restarts | 0.680 | 1.3177 | 0.3697 | 2114.326530 |

## Part 3 — robust forecasting

CRPS averaged over horizons and origins (and replicates), then at h = 1.
Scores are always against the clean values. "raw": fit and condition on the
series as observed; "+ gate": condition on it with the fit's own sequentially
flagged rows removed; "robust": `robust_estimate`; "Hampel robust":
`robust_estimate` from the Hampel pre-screen.

**Simulated fixtures** (`report/erroneous_data`, τ = 0.6 for the PMC; 2 000 +
1 000 rows, spikes of 6 sd, 10 replicates, h = 1…10):

| method | HMC-IN fixture 0 % | 1 % | 5 % | h=1 5 % | PMC fixture 0 % | 1 % | 5 % | h=1 5 % |
|---|---|---|---|---|---|---|---|---|
| true model | 0.768 | 0.767 | 0.777 | 0.698 | 0.740 | 0.755 | 0.800 | 0.702 |
| true model + gate | 0.768 | 0.768 | 0.770 | 0.675 | 0.740 | 0.741 | 0.743 | 0.492 |
| raw | 0.768 | 0.778 | 0.818 | 0.805 | 0.742 | 0.751 | 0.769 | 0.530 |
| raw + gate | 0.768 | 0.778 | 0.818 | 0.805 | 0.742 | 0.749 | 0.769 | 0.530 |
| robust | 0.768 | 0.768 | 0.817 | 0.805 | 0.742 | 0.757 | 0.769 | 0.530 |
| robust + gate | 0.768 | 0.768 | 0.817 | 0.805 | 0.743 | 0.744 | 0.769 | 0.530 |
| Hampel robust + gate | 0.768 | 0.768 | **0.770** | 0.675 | 0.743 | 0.744 | **0.746** | 0.496 |
| clean fit, clean series | 0.768 | 0.768 | 0.768 | 0.670 | 0.742 | 0.742 | 0.742 | 0.482 |
| PMM (pmmforecast) | 0.774 | 0.782 | 0.869 | 0.827 | 0.750 | 0.771 | 0.870 | 0.781 |
| AR(1) | 0.790 | 0.797 | 0.878 | 0.856 | 0.746 | 0.779 | 0.883 | 0.816 |
| AR(1) + gate | 0.790 | 0.797 | 0.877 | 0.851 | 0.758 | 0.773 | 0.881 | 0.797 |
| AR(1) robust + gate | 0.790 | 0.790 | 0.791 | 0.714 | 0.770 | 0.770 | 0.772 | 0.557 |
| AR(1), clean fit | 0.790 | 0.790 | 0.790 | 0.708 | 0.746 | 0.746 | 0.746 | 0.494 |
| persistence | 0.983 | 1.010 | 1.310 | 1.183 | 0.924 | 1.003 | 1.219 | 0.838 |

Flags (`fits.csv`, means over replicates):

- **5 %: the raw ICE fit models the spikes as a regime.** Its gate detects
  0.4 % (HMC-IN) and 0 % (PMC) of the test spikes. `robust_estimate` stops
  there (mask detection 0.7 % and 0.1 %).
- **From the Hampel pre-screen, the loop ends with every spike masked** (mask
  detection 1.00, 1.0–1.5 false flags per 1 000).
- **1 %: gating works with the fit's own model.** Test detection is 0.99
  (HMC-IN) and 0.77 (PMC); `robust_estimate` reaches detection 1.00.

**Beijing Aotizhongxin with injected spikes, seed 0.** Every method ran on
the same series. The copula PMC ran on seed 0 only. The 3-seed table of the
other methods is in `results/tables.md`; its means are within 0.01 of these.

| method | clean | h=1 clean | 1 % | h=1 1 % | 5 % | h=1 5 % |
|---|---|---|---|---|---|---|
| PMC pair K=3 | 0.557 | 0.199 | **0.556**† | **0.197**† | **0.561**† | **0.207**† |
| PMC pair K=3, raw + gate | 0.557 | 0.199 | 0.556† | 0.197† | 0.561† | 0.207† |
| PMC pair K=3, robust | 0.564 | 0.200 | 0.558† | 0.198† | 0.565† | 0.205† |
| PMC pair K=3, robust + gate | 0.564 | 0.200 | 0.558† | 0.198† | 0.566† | 0.205† |
| PMC pair K=3, Hampel robust + gate | – | – | 0.564 | 0.198 | 0.566† | 0.205† |
| HMC-IN K=3 | 0.600 | 0.334 | 0.626 | 0.419 | 0.642 | 0.436 |
| HMC-IN K=3, raw + gate | 0.600 | 0.334 | 0.626 | 0.419 | 0.642 | 0.436 |
| HMC-IN K=3, robust + gate | 0.601 | 0.334 | 0.626 | 0.420 | 0.641 | 0.436 |
| HMC-IN K=3, Hampel robust + gate | – | – | 0.602 | 0.335 | 0.602 | 0.338 |
| PMM (pmmforecast) | 0.561 | 0.206 | 0.595 | 0.348 | 0.649 | 0.496 |
| AR(1) | 0.561 | 0.206 | 0.649 | 0.309 | 0.713 | 0.588 |
| AR(1) + gate | 0.589 | 0.272 | 0.649 | 0.309 | 0.713 | 0.574 |
| AR(1) robust + gate | 0.636 | 0.313 | 0.636 | 0.313 | 0.650 | 0.329 |
| persistence | 0.632 | 0.210 | 0.631 | 0.210 | 0.817 | 0.415 |

† These fits converge in the quadrature to 1.7–2.8 % of the predictive sd in
mean and sd, between G = 128 and 256, instead of 1 %. Their node-law tails
change by 2–9 %. Such a change is small beside the differences quoted: a
2.8 % sd shift of the mean changes a Gaussian CRPS by less than 0.1 %.

- **The raw pair PMC gives its spike state** (see "Conclusions") a
  probability of 0.011 and 0.049, a self-transition of 0 and pair margins
  at ln PM2.5 ≈ 10.
- **Its gate flags almost nothing:** 2–3 test rows, detecting 3–12 % of the
  test spikes.
- **`robust_estimate`, even from the Hampel mask (97 % of the spikes), returns
  to that fit at 5 %** (mask of 21 rows, 0.9 % of the spikes). At 1 % the
  Hampel start keeps 87 masked rows (detection 1.00): a clean model, flagged
  degenerate by the real_series rule.

**Intel Lab mote 20 as recorded** (three −38.4 °C readings, the sensor's zero
count, in the training part; the test part is the same):

| method | CRPS, suspect removed | CRPS, as recorded | RMSE h=1, removed | RMSE h=1, as recorded |
|---|---|---|---|---|
| HMC-IN K=3 | 0.759 | 0.750 | 1.317 | 1.303 |
| HMC-IN K=3, robust + gate | 0.778 | 0.778 | 1.335 | 1.335 |
| PMM (pmmforecast) | 0.098 | **0.252** | 0.019 | 0.066 |
| AR(1) | 0.092 | **0.969** | 0.021 | 0.221 |
| AR(1) + gate | 1.617 | 0.969 | 3.384 | 0.221 |
| AR(1) robust + gate | 4.940 | 4.940 | 5.873 | 5.873 |
| persistence | 0.083 | 0.083 | 0.021 | 0.021 |

- **Three readings out of 21 600 degrade the Gaussian fits.** The PMM loses a
  factor 2.6, the AR(1) a factor 10: φ drops from 0.9999 to 0.95, σ² rises
  from 0.0023 to 1.03.
- **The Gaussian gate cannot repair this.** On the raw fit it flags 0 test
  rows. On the clean fit it flags 266 per 1 000 and loses lock.
- **HMC-IN and persistence are unaffected.**

## Design of parts 1 and 3

**Series.** The data are read from the data folder; nothing is copied.

| series | what is fitted | truth | split | origins | h |
|---|---|---|---|---|---|
| Beijing Aotizhongxin | dequantised ln PM2.5 (37 NaN), as real_series | ln PM2.5 as reported | first 75 % | 91, every 24 h | 1…24 |
| tsNH4 | ln tsNH4Complete (the imputeTS NaN were inserted by its authors; the complete record is used) | same | first 75 % | 93, every 12 steps (2 h) | 1…24 |
| Intel Lab mote 20 | temperature, epochs 1…28 800 (4 373 absent, epoch 1 among them), dequantised by U(±0.0049) (readings are multiples of 0.0098 °C, 29 % ties; copula fits were degenerate without it); suspect readings removed (part 1) or kept (part 3) | readings as recorded | first 75 % | 120, every 30 min | 1…24 |

**Models.**

- *pmcprg*: the real_series §4 models at their BIC K. Aotizhongxin: PMC pair
  K = 3 and HMC-IN K = 3. tsNH4: PMC state K = 2 and HMC-IN K = 3. Mote 20:
  BIC on the training part, among the fits that are neither degenerate nor
  unconverged in the quadrature. Fits use ICE "available" from the k-means
  start (`rs_common`).
- *pmmforecast*: the Gaussian PMM by Y-only Kalman MLE on the same training
  part.
  - Interior NaN are filled linearly, since pmmforecast cannot take them.
  - A trailing gap before an origin is skipped exactly: the forecast starts
    from the last observed row.
  - Starts as in M2.
- *AR(1)*: Gaussian, exact likelihood with gaps (`rs_common.ar1_fit`).
- *Persistence*: the last observed value. Its predictive law is that value
  plus the empirical quantiles of the training h-step changes.

**Scores** by horizon: RMSE of the predictive mean, CRPS, and the coverage of
the central 50 / 80 / 95 % intervals.

- *Gaussian laws* (AR(1), PMM): closed-form CRPS, Gaussian quantiles.
- *HMC-IN*: exact Gaussian-mixture CRPS; coverage from its quantiles.
- *Copula PMCs (grid models)*: everything comes from the forecast's own
  discrete law, the quadrature nodes and masses of each horizon
  (`Forecast.grid_nodes` / `grid_mass`). Each mass is spread uniformly over
  its cell.
  - From that piecewise-linear CDF: the CRPS in closed form, the PIT, the
    coverage (PIT inside the central interval) and the 95 % width
    (`fc_common.node_law_scores`).
  - The returned quantiles (7 levels) are only checked.
  - Why: with 207 levels, pmcprg's quantile code costs 10–30× the forecast;
    and the node law is the law the mean and sd come from.
  - Accuracy: the node-law CRPS is within 2e-4–3e-3 (mean; max 7.6e-3) of
    the closed form on AR(1)s written as PMCs. It is within 7.4e-4 (mean;
    max 2.3e-2; 2 832 rows) of the 200-level quantile CRPS, computed at the
    middle origin of every grid model of the campaign.
- *Persistence*: CRPS from 200 empirical quantiles. For the Gaussian and
  HMC-IN laws, the 200-quantile CRPS agrees with the exact one to 3.7e-5 and
  8.3e-5 (mean relative).
- *Paired comparisons*: bootstrap CI (2 000 resamples) of the per-origin
  difference of CRPS (mean over h).
- *PMM theoretical MSE*: `TheoreticalMSE` (Y entry) against the empirical MSE.

**Contamination (part 3).**

- *Simulated series*: the `report/erroneous_data` fixtures (`hmc_in_gauss_k2`,
  `pmc_gauss_k2`). Isolated spikes of 6 sd at 0, 1 and 5 %; 10 replicates;
  crc32 seeds, as in that study. Fits start at the true model.
- *Aotizhongxin*: the same spikes injected into the observed rows, at 1 and
  5 %, 3 seeds. The copula PMC (pair K = 3) runs on seed 0 only
  (`REAL_PMC_SEEDS`). Its contaminated fits need 256 nodes, where one
  forecast of the full series costs 9 s: 77 and 79 min per task.
- *Mote 20*: as recorded.
- *Scoring*: always against the clean values.

**Methods per fit.**

- *raw*: fit the series as observed, condition on it.
- *+ gate*: condition on the series with the rows flagged by the fit's own
  sequential gating (`flag_outliers`, α = 1e-3) set to NaN.
- *robust*: `robust_estimate`, whose refits restart from the k-means start of
  the unmasked rows.
- *Hampel robust*: `robust_estimate` with the NaN-aware Hampel pre-screen of
  erroneous_data as `initial_mask`.
- *References*: the clean fit and, for the fixtures, the true model.
- *AR(1)*: the same variants with a closed-form Gaussian gate
  (`fc_common.ar1_flags`), the innovation gating of a Kalman filter. It
  equals `flag_outliers` on the AR(1) as a pmcprg model (asserted in
  `--quick`).

The gated filter is causal, so one `flag_outliers` call serves every origin.

**Seeds and settings.**

- Fixture simulations and spikes: crc32 of their labels.
- Mote dequantisation: seed (11, 20).
- ICE: k-means seed 0, missing seed 1000.
- pmmforecast: 4 restarts, seed 0.
- Bootstrap: `default_rng(0)`.
- Every value is in `run_info.json` (`settings`).

## Checks (kept from P1–P3), all run on this campaign

**Quadrature convergence per model** (`choose_nodes`, `fits.csv`).

- The chosen G is the first of 64, 128 and 256 whose predictive means and
  sds change by less than 1 %, and whose node-law 2.5 / 97.5 % quantiles by
  less than 5 %, of the predictive sd when G is doubled. The check uses the
  first, middle and last origins.
- That G is used for forecasting and gating.
- pmcprg's own diagnostic, quad_error / its WARNING limit, is recorded too:
  for the same forecast chains, and for every conditioning series (observed
  and gated). It is not required.

| study | model | fits | G = 64 / 128 / 256 | unconverged | mean/sd check at 64 | at chosen G | tails at G | quad_error / limit at G |
|---|---|---|---|---|---|---|---|---|
| Aotizhongxin | PMC pair K=3 | 8 | 1 / 2 / 5 | 5† | 4.7e-2 | 2.8e-2 | 9.0e-2 | up to 230 |
| tsNH4 | PMC state K=2 | 2 | 0 / 2 / 0 | 0 | 2.2e-2 | 1.4e-4 | 4.5e-2 | 2.3 |
| PMC fixture | PMC state K=2 | 150 | 150 / 0 / 0 | 0 | 4.4e-4 | 4.4e-4 | 4.5e-2 | 5.6 |
| mote 20 (selection) | 4 copula PMCs | 4 | 0 / 0 / 4 | 4 (not used) | 0.10–0.24 | 0.067–0.17 | 0.13–0.70 | 5–34 |

Aotizhongxin fits:

- *Converged:* the clean raw fit (§4, G = 64, 3e-4), the clean robust fit
  (G = 128, 4.2e-3) and the 1 % Hampel fit (G = 128, 5.2e-3).
- *† (1.7–2.8 %):* the 1 % raw and robust fits and all three 5 % fits.

**pmcprg WARNINGs** (`results/pmcprg_warnings.csv`; the tasks used to
silence them):

- *quad_error:* 3 710 records in 131 task steps. They come from ICE fits,
  `classify`, the forecasts and the checks. The largest values are on the
  mote-20 copula fits (up to 160), which are not used. On the forecasts used,
  it is a conservative screen (see "Observations for the library author").
- *Quantile fallback:* 489 records, mostly Aotizhongxin forecasts.
- *Other:* 289 records: ICE log-likelihood regressions, the degenerate-model
  WARNING (tsNH4's state of π = 0.0013, as in real_series), and log-space
  recomputations.

**Quantiles** (`q_check`, `q_check_tail` per row; `run_info.json`).

- *Median check:* 66 of 276 726 grid rows exceed 0.25, all tsNH4 at h ≤ 2
  and G = 128 (max 0.376).
  - The node spacing there is 0.57 predictive sd, and one node holds 27–29 %
    of the mass. The discrete node median therefore jumps by half a spacing.
  - Checked on two such rows: the returned median sits at node-law CDF
    0.499–0.501 and does not move between G = 128 and 1 024 (2.6317 at every
    G), while the node median moves by 0.3 sd.
  - Before the P3 fix, 1 332 rows were flagged, up to 9 796 per case.
- *Tails:* `q_check_tail` has p99 0.017; 463 rows (0.17 %) exceed 0.05,
  max 0.35.

**Leading gaps** (`run_info.json`, `leading_gaps`): no fitted, conditioning
or gated series of a grid model starts with a missing row. Only the four
mote-20 selection fits do (see "Leading gap").

## Reproduce

pmmforecast lives in a separate venv. Export pmmforecast HEAD with `git
archive`, then `pip install` it into a new venv. awesomePMC's `.venv` stays
the pmcprg interpreter. The two exchange CSV/JSON files. Without
`--pmm-python`, the PMM part is skipped with a message.

```bash
# part 2 (≈ 6 min; its pmmforecast side ≈ 45 s)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/cross_check.py --pmm-python <venv>/bin/python
# pmcprg problems P1–P3 (fixed at 39f249f)
PYTHONPATH=. .venv/bin/python report/forecasting/repro_pmcprg_problems.py
# parts 1 and 3 (smoke run ≈ 35 min, then the full campaign ≈ 1.5 h)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/run_forecasting.py --quick --pmm-python <venv>/bin/python
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/run_forecasting.py --jobs 6 --pmm-workers 4 --pmm-python <venv>/bin/python
# tables (results/tables.md, tables.tex) and figures (figures/), from the CSVs only
.venv/bin/python report/forecasting/summarise.py
```

Cost of the campaign, on an Apple arm64 laptop (Python 3.14.7, numpy 2.5.3)
shared with another session's jobs:

- 92.6 min of wall time on 6 pmcprg and 4 pmmforecast processes;
- pmcprg task time 367 min in total:
  - mote-20 selection 42 min;
  - real-series tasks 207 min, of which the two contaminated Aotizhongxin
    copula tasks took 77 and 79 min;
  - fixtures 117 min;
- pmmforecast 126.5 min.

Tasks are cached in `report/out/forecasting/full/tasks/`. Seeds are fixed in
the task specs.

```
report/forecasting/
├── README.md
├── cross_check.py            part 2, pmcprg side
├── fc_common.py              data, models, scores, checks, tasks (pmcprg side)
├── pmm_side.py               every use of pmmforecast (its own venv)
├── repro_pmcprg_problems.py  P1–P3
├── run_forecasting.py        parts 1 and 3 driver
├── summarise.py              tables and figures from results/
├── figures/                  part 1 CRPS by horizon, PMM theoretical MSE, part 3
└── results/                  every CSV of parts 1–3, run_info.json, tables.md, tables.tex
```
