# Forecasting: pmcprg against pmmforecast, and outlier gating (forecasting)

> **Status (2026-09-23).** Of the library problems reported below, the empty
> state (P2) is fixed on main. The PIT at the clipped copula corner is left as
> is and documented in `pmcprg/pmc/outliers.py`: without gating, the model
> itself finds a plateau after a jump plausible, so use `sequential=True`. The
> gap quadrature under strong dependence (P1) and the forecast/impute
> quantiles (P3) are still open, so the copula-model (PMC) numbers here are
> provisional. The HMC-IN numbers, the baselines and the Gaussian cross-check
> are not affected.

This study compares pmcprg's forecasts with **pmmforecast**. pmmforecast is the
author's companion package for stationary 1-D Gaussian pairwise Markov models:
a continuous hidden state X and five parameters (a, b, c, d, e). Its reference
is Escudier, Abdelkefi, Fernandes & Pieczynski, *Forecasting with Pairwise
Gaussian Markov Models*, CMCSI 2023, arXiv 2402.07532. The study also measures
what outlier gating (`flag_outliers`, `robust_estimate`) does to forecasts.

## Status (2026-09-22): part 2 done, parts 1, 3 and 4 stopped

The campaign of parts 1 and 3 was stopped after 20 minutes. The study found
three pmcprg problems that make its forecasts silently wrong in configurations
the study needs (see **Library problems, pmcprg**):

- **P1.** Missing and future rows are integrated on a fixed quadrature grid,
  and that grid collapses under strong copula dependence.
- **P2.** With an emptied state, the PIT is NaN and nothing is flagged.
- **P3.** Forecast quantiles can disagree with the law whose mean and standard
  deviation `forecast` returns.

The brief says to stop on a library bug and report it; no library code was
changed. What is delivered:

- **Part 2 (Gaussian cross-check): complete.** It is not affected by P1–P3.
  Tables are below; the script is `cross_check.py`, the numbers are in
  `results/crosscheck_*.csv`.
- **Minimal reproductions** of P1–P3 (`repro_pmcprg_problems.py`) and of the
  pmmforecast issues (snippets below).
- **The pipeline of parts 1 and 3**, implemented and smoke-tested:
  `run_forecasting.py`, `fc_common.py`, `pmm_side.py` and `summarise.py`.
  It detects P1 with a per-model quadrature convergence check and P3 with a
  per-row quantile check. Rerun it once P1–P3 are fixed.
- **Observations from the interrupted run**, labelled as such below. Their
  numbers come from `report/out/forecasting/`, which is not versioned.

## Library problems, pmcprg (reproductions: `repro_pmcprg_problems.py`)

```bash
PYTHONPATH=. .venv/bin/python report/forecasting/repro_pmcprg_problems.py
```

**P1. Gap quadrature collapses under strong copula dependence (silent).**
`forecast`, `predictive_pit`, `flag_outliers` and `impute` integrate every
missing or future row on a fixed grid of `gap_nodes` nodes (default 64). The
nodes come from the reference (stationary) law. With strong dependence, the
conditional law of y_{n+1} given y_n is narrower than the node spacing, and the
result is silently wrong. The test model is a Gaussian AR(1)(ρ) written as a
PMC with two identical regimes. On complete conditioning data it is exact to
1e-14. After one unobserved row:

| ρ (Gaussian copula τ) | 2-step forecast sd, pmcprg | exact | PIT after a gap, pmcprg | exact |
|---|---|---|---|---|
| 0.99 (0.910) | 0.1985 | 0.1985 | 0.5321 | 0.5321 |
| 0.998 (0.960) | 0.0848 | 0.0893 | 0.5477 | 0.5499 |
| 0.999 (0.972) | 0.0378 | 0.0632 | 0.5546 | 0.5666 |
| 0.9999 (0.991) | **0.0000** | 0.0200 | 0.6473 | 0.6925 |

Consequences:

- At ρ = 0.9999, with 15 % missing rows, `flag_outliers` flags 843 of 3 000
  rows. The exact Gaussian rule flags 2.
- More nodes help only partly. G = 256 fixes ρ = 0.999; ρ = 0.9999 needs
  G = 1024.
- This hits every h ≥ 2 forecast: the future rows are a trailing gap.
- It hits the gated conditioning data too: a flagged row becomes a missing row.

Real data reach this regime. On Intel Lab mote 20 (temperature, 30 s), every
copula PMC fitted has diagonal τ between 0.98 and 0.998. They beat HMC-IN by
more than 110 000 in BIC. Their forecasts still change by 7–115 % between 128
and 256 nodes (`results/mote20_select.csv`), so pmcprg cannot forecast this
series with them.

**P2. Empty state → NaN PIT, nothing flagged (silent).** A PMC prior p with an
all-zero row is a state that ICE emptied. The study's k-means starts on the
mote extracts produced such fits, with a margin reset to N(0, 1e-4). On such a
model:

- `predictive_pit` returns NaN for every row after the first.
- `flag_outliers` flags nothing, not even an 8 sd spike.
- The log-likelihood is right.
- Replacing the zeros by 1e-12 gives finite PITs and flags the spike.

The NaN comes from `np.log(p) − np.log(p.sum(axis=1))` for that row: a
RuntimeWarning at `pmcprg/pmc/inference.py:505`.
`robust_estimate` started from such a fit stops at once with an empty mask.

**P3. `forecast` quantiles disagree with the returned law (silent).** Take a
3-state pair-margin PMC. It was fitted by ICE on Aotizhongxin with 5 % spikes,
and one regime sits at the spike level. From y = 4.0, the two-step law has
mean 4.257 and sd 1.424. Its median, computed from `forecast`'s own
`density × ω`, is 4.055. The returned quantiles do not match:

| gap_nodes | returned 2.5 / 50 / 97.5 % quantiles | median of the node law |
|---|---|---|
| 32 | 3.136 / 4.089 / **15.822** | 3.993 |
| 64 (default) | **13.396 / 15.767 / 16.685** | 4.055 |
| 128 | 3.105 / 4.054 / **16.475** | 4.023 |
| 256 | 3.040 / 4.003 / 10.470 | 4.007 |

- The moments and the node density are consistent with each other. The
  quantiles come from the Nyström / Legendre path (`_nystrom_masses`,
  `_grid_summary`), and that path fails on this bimodal law.
- `impute` shows the same symptom: the second of two missing rows between two
  4.0 values gets a 97.5 % quantile of 16.3.
- In the interrupted run, this corrupted the quantile CRPS (values of 11–15
  where the CRPS of the mean and sd is about 0.6) and the coverage:
  - up to 91 % of the PMC rows of an Aotizhongxin contaminated case
    (9 796 of 10 785);
  - 382 of 8 628 robust-fit rows of the clean case.

Part 2 is not affected. Its grid forecasts (ρ = 0.7, unimodal) agree with the
closed form to 1e-6 in their quantiles.

## pmmforecast issues (for its author; its repository was not modified)

pmmforecast is exported from commit **`58ac3b624d9bc006b0fdf96a03c9db5f566c8f9f`**
(v0.22.1): `git archive HEAD` into the session scratchpad, then a fresh venv.
The snippets below run in that venv.

**M1. Y-only identifiability is wider than documented.** The docstring of
`estimate_pmm_mle_y_only`, Tutorial 09 and the property test describe the
ambiguity as the sign flip (b, d, e) → −(b, d, e). In fact the law of Y
depends on the tuple only through (c, tr A, det A), with

- tr A = (a + c − b(d + e)) / (1 − b²)
- det A = (ac − de) / (1 − b²)

(derivation in part 2). This leaves a 2-dimensional set of tuples with the
same Y-only likelihood; the swap d ↔ e is one of them. Checked on one path in
the table below: tuples with b = 0.2, 0.5 and 0.7 give the same NLL, with a
difference of exactly 0.0 in floating point.
The estimated tuple is therefore not interpretable; its forecasts of Y are.

**M2. Default starts miss the maximum on persistent series.** The default
starts of `estimate_pmm_mle_y_only` are (0.5, 0.3, 0.1, 0.4, 0.2), then
uniform draws in [−0.7, 0.7]^5. On series with a lag-1 autocorrelation of
0.99 or more they converge to a spurious region: c ≈ 0.47–0.50,
det A ≈ −0.99.

- In that region the likelihood is tens of nats below the AR(1) sub-model,
  (0, 0, r, 0, 0), which is a valid PMM.
- tsNH4, first 2 250 rows (r = 0.9969), standardised NLL:
  - default, 4 restarts: −2352.9;
  - AR(1) tuple: −2396.8;
  - from x0 = (0, 0, r, 0, 0): −2402.6.
- Simulated PMM with Y an AR(1) of 0.99, N = 2 000: the default estimate's
  NLL is −1230.2, the true tuple's −1243.6.

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
API (`x0`, a user parameter). It adds one start at (0.5, 0.3, r, 0.3 r, 0.3 r),
an AR(1)(r) inside the PMM family, and keeps the higher likelihood. The
default-start fit is kept and forecast too, as `pmm_default`.

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

## Part 2 — Gaussian cross-check (complete)

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

All tables: `results/tables.md` (Markdown) and `results/tables.tex`.

## Parts 1 and 3: design (implemented, campaign stopped)

**Series.** The data are read from the data folder; nothing is copied.

| series | what is fitted | truth | split | origins | h |
|---|---|---|---|---|---|
| Beijing Aotizhongxin | dequantised ln PM2.5 (37 NaN), as real_series | ln PM2.5 as reported | first 75 % | 91, every 24 h | 1…24 |
| tsNH4 | ln tsNH4Complete (the imputeTS NaN were inserted by its authors; the complete record is used) | same | first 75 % | 93, every 12 steps (2 h) | 1…24 |
| Intel Lab mote 20 | temperature, epochs 1…28 800, dequantised by U(±0.0049) (readings are multiples of 0.0098 °C, 29 % ties; copula fits were degenerate without it); suspect readings removed (part 1) or kept (part 3: three −38.4 °C readings, the sensor's zero count, in the training part) | readings as recorded | first 75 % | 120, every 30 min | 1…24 |

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

**Scores** by horizon: RMSE, CRPS and the coverage of the central 50 / 80 /
95 % intervals.

- *CRPS of Gaussian laws* (AR(1), PMM): closed form.
- *CRPS of HMC-IN*: the exact Gaussian-mixture CRPS.
- *CRPS of the grid models*: from 200 predictive quantiles (quantile
  decomposition). On the smoke run it agreed with the exact CRPS to 4e-5 on
  average (max 1.5e-3) for the Gaussian and mixture laws. For comparison with
  §4, the Gaussian CRPS of the predictive mean and sd is kept alongside.
- *Paired comparisons*: bootstrap CI (2 000 resamples) of the per-origin
  difference against the PMM.
- *PMM theoretical MSE*: `TheoreticalMSE` (Y entry) against the empirical MSE.

**Contamination (part 3).**

- *Simulated series*: the `report/erroneous_data` fixtures (`hmc_in_gauss_k2`,
  `pmc_gauss_k2`), 2 000 + 1 000 rows. Isolated spikes of 6 sd at 0, 1 and
  5 %; 10 replicates; crc32 seeds, as in that study. Fits start at the true
  model.
- *Aotizhongxin*: the same spikes injected into the observed rows, at 1 and
  5 %, 3 seeds.
- *Mote 20*: as recorded.
- *Scoring*: always against the clean values.

**Methods per fit.**

- *raw*: fit the series as observed, condition on it.
- *+gate*: condition on the series with the rows flagged by the fit's own
  sequential gating (`flag_outliers`, α = 1e-3) set to NaN.
- *robust*: `robust_estimate`, whose refits restart from the k-means start of
  the unmasked rows.
- *Hampel robust*: `robust_estimate` with the NaN-aware Hampel pre-screen of
  erroneous_data as `initial_mask`.
- *References*: the clean fit and, for the fixtures, the true model.
- *AR(1)*: the same variants with a closed-form Gaussian gate (see below).

The gated filter is causal, so one `flag_outliers` call serves every origin.
The closed-form Gaussian gate is `fc_common.ar1_flags`. It equals
`flag_outliers` on the AR(1) as a pmcprg model where P1 does not bite; this is
asserted in `--quick`.

**Detection of P1 and P3 in the pipeline.**

- *P1*: every grid model gets `choose_nodes`. It takes the first G in 64, 128,
  256 whose forecasts change by less than 1 % of the predictive sd when G is
  doubled, and uses that G for forecasting and gating. G, the final check and
  the check at 64 are recorded in `fits.csv`.
- *P3*: every grid forecast row carries `q_check`, the distance of the
  returned median to the median of the node law, in predictive sd.
  `run_info.json` counts the rows above 0.25.
  - The check is a coarse screen. A bimodal law whose CDF is flat at 0.5 can
    trip it without any bug: one row of the true PMC fixture did in the smoke
    run.
  - Its output after the final code: 1 332 rows flagged, mostly Aotizhongxin
    PMC pair rows, plus 14 tsNH4 PMC-state rows per variant.

## Observations from the interrupted run (not versioned, indicative)

- **§4 is reproduced exactly.** PMC pair K = 3 on Aotizhongxin: RMSE 0.396 /
  1.040 / 1.214 at h = 1 / 6 / 24. HMC-IN K = 3: 0.591 / 1.087 / 1.207.
  Both use G = 64, which passed the quadrature check (1e-4).
- **pmmforecast on Aotizhongxin behaves as an AR(1).**
  - RMSE 0.396 / 1.011 / 1.187; §4's AR(1) had 0.397 / 1.011 / 1.187.
  - Exact CRPS 0.206 / 0.577 / 0.689; the AR(1)'s CRPS at h = 1 is 0.206.
  - The fit: c = 0.946, tr A = 0.42, det A = −0.50.
  - Its predictive variance is too small. The empirical MSE is 1.35, 1.94 and
    1.38 times the theoretical MSE at h = 1, 6 and 24. The 95 % intervals
    cover 0.84–0.91.
- **tsNH4.** pmmforecast's default starts end 42 nats below the
  AR(1)-embedding start: standardised NLL −2553.9 against −2595.5 (M2).
- **Mote 20.** The copula PMCs cannot be forecast (P1, table above), so only
  HMC-IN remained for pmcprg. HMC-IN ignores the dependence of y_{n+1} on
  y_n, and on this near-deterministic 30 s series it is far behind
  persistence: RMSE 1.3 against about 0.06 in the smoke run.
- **Smoke run, 3 000-row extracts: the Gaussian AR(1) gate is poorly
  calibrated on real series.**
  - At α = 1e-3 it flagged 16 (Beijing) and 68 (tsNH4) clean test rows per
    1 000, and more than half of the mote-20 training extract: the gate loses
    lock after level jumps.
  - Gating raised the AR(1)'s CRPS on tsNH4 from 0.248 to 0.353.
  - This is the first measurement relevant to "should pmmforecast's Kalman
    filter gate innovations?". The full part 3 is needed to answer it.

## Reproduce

pmmforecast lives in a separate venv. Export pmmforecast HEAD with `git
archive`, then `pip install` it into a new venv. awesomePMC's `.venv` stays
the pmcprg interpreter. The two exchange CSV/JSON files. Without
`--pmm-python`, the PMM part is skipped with a message.

```bash
# part 2 (≈ 6 min; its pmmforecast side ≈ 45 s)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/cross_check.py --pmm-python <venv>/bin/python
# pmcprg problems P1–P3
PYTHONPATH=. .venv/bin/python report/forecasting/repro_pmcprg_problems.py
# parts 1 and 3 — after P1–P3 are fixed (smoke run < 3 min, then the full campaign)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/run_forecasting.py --quick --pmm-python <venv>/bin/python
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/forecasting/run_forecasting.py --jobs 6 --pmm-workers 4 --pmm-python <venv>/bin/python
# tables (results/tables.md, tables.tex) and figures (figures/), from the CSVs only
.venv/bin/python report/forecasting/summarise.py
```

Estimated cost of the full campaign, measured on the interrupted run:

- one Aotizhongxin PMC-pair task (fits, robust fits, 91 origins × 5 variants,
  quadrature checks) takes 10–20 min;
- the whole campaign should take about 1 h of wall time on 6 + 4 processes.

Tasks are cached in `report/out/forecasting/full/tasks/`. Seeds are fixed in
the task specs.

```
report/forecasting/
├── README.md
├── cross_check.py            part 2, pmcprg side
├── fc_common.py              data, models, scores, tasks (pmcprg side)
├── pmm_side.py               every use of pmmforecast (its own venv)
├── repro_pmcprg_problems.py  P1–P3
├── run_forecasting.py        parts 1 and 3 driver
├── summarise.py              tables and figures from results/
└── results/                  crosscheck_*.csv, mote20_select.csv, tables.md, tables.tex
```
