# Real series with gaps: unsupervised estimation, imputation, classification (A4)

Everything measured so far on missing observations used simulated data and known
models (`report/missing_benchmark/`). This study asks the practical question on
three real series with gaps: **does unsupervised estimation with gaps work, is the
fitted model useful for imputation and classification, and where does it fail?**

All numbers below are printed by `summarise.py` from the CSVs in `results/`
(`results/tables.md` holds every table in full). 266 unsupervised fits, 180 masked
cross-validation tasks, 91 forecast origins; no fit crashed and no estimate was
non-finite.

## Summary

* **Estimation with gaps runs on real data, and the missing-data machinery is not
  the weak point.** With 19.4 % missing (tsNH4), 3.8 % (Huairou) or 11–12 %
  (PAMAP2) the three strategies run without failure, and on PAMAP2 a model fitted
  on the gapped series classifies the complete series within 0.4 points (median)
  of the model fitted on the complete series. What decides the result is what
  also decides it without gaps: the likelihood surface of the copula models has
  many basins (median spread over 3 starts 95.5 nats; median lowest agreement of a
  start with the best fit of its cell 0.736), and on discretised data it is
  unbounded.
* **ICE "available" is the strategy to use.** From the same k-means start, "impute"
  ends within 1 nat of it in 38 % of the copula-model cells (median −0.6 nats) at
  4.9× the time, SEM in none (median −3.4 nats) at 1.9× the time. Neither is
  better on average: from the same start each wins some cells by tens of nats
  (up to +71) and loses others by as much (down to −81). For HMC-IN the three
  strategies end within 1 nat of each other from the same start.
* **Copulas are the big win in likelihood** (thousands of nats over HMC-IN on every
  series) **but not in imputation quality**. At the real gaps of tsNH4 the PMC
  posterior is well calibrated (90 % intervals cover 92 %) but its CRPS ties the
  Gaussian AR(1) smoother (0.114–0.132 vs 0.122) and its RMSE does not beat linear
  interpolation (best PMC 0.251, BIC-chosen PMC 0.315, linear 0.242). On Beijing
  PM2.5 the refitted PMC has a lower CRPS than the AR(1) smoother in 30 of 30
  masked (rate, block, seed) cells at Huairou (cell means 6–14 % lower) and 23 of
  30 at Aotizhongxin (5–10 % lower for 1- and 6-hour blocks, a tie for 48-hour
  blocks); its forecasts tie AR(1).
* **PAMAP2: gap handling is fine, unsupervised segmentation is not.** With the
  supervised oracle, exact marginalisation (or the posterior-mean plug-in)
  classifies the missing windows better than linear or LOCF fill-in (K = 2,
  mean of 3 subjects: 5.8–6.5 % vs 8.0–13.6 %). Unsupervised, the likelihood-best
  states are not the activity groups: K = 3 errors 27.7–43.7 % against
  6.7–16.9 % for the HMC-IN and pair-margin oracles, and the K = 2 copula fits of
  subject 105 split the data by dependence rather than by level (44.2–45.7 %).
* **The dropouts are state-dependent** (subject 102: 0.00 % of rest windows
  missing, 2.07 % of locomotion, 10.82 % of vigorous), but ignoring it costs
  nothing measurable here: the real gaps are short, their neighbours carry the
  state, and the ignorable oracle gets them right (≤ 3.3 % error; ≤ 4.3 % with
  the "any NaN" rule). A naive
  missingness-aware model that treats every missing window as independent
  evidence for the state makes it worse (102, K = 3: 8.8 % vs 0.0 %; "any NaN"
  rule: 34.1 % vs 2.1 %).
* **Failure cases** (details below): unbounded likelihood on discretised data;
  degenerate states (a margin sd → 0 or a state with π < 0.5 %) in 13 of 64 tsNH4
  fits, one of which has 90 % intervals covering 27 % of the truth; likelihood-best
  segmentations that do not match the labels; ICE returning a model up to 50 nats
  below its own best iterate.

## Data and preparation

The data are read from `--data` (default
`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/data/timeseries`) and are
never copied into the repository. Each dataset folder documents its own
preprocessing (`README.md`, section *Prétraitement*); the choices below are made on
top of those outputs, in `rs_common.py`.

| series | file | N | what is fitted | truth for scoring |
|---|---|---|---|---|
| tsNH4 | `imputeTS/output/tsNH4_study_log.csv` | 4 552 (10 min) | `Y` = ln NH4, 883 NaN (19.4 %) in 155 real gaps | `Y_true` (tsNH4Complete) at the real gaps |
| Huairou | `beijing_air_quality/output/huairou_pm25_study.csv` | 8 760 (1 h, first year) | dequantised ln PM2.5 (below), 334 NaN, one gap of 208 h | none at the real gaps: masked CV |
| Aotizhongxin | `beijing_air_quality/output/aotizhongxin_pm25_study.csv` | 8 760 | dequantised ln PM2.5, 37 NaN | masked CV, forecasts |
| PAMAP2 102 / 108 / 105 | `pamap2/output/subject10x_100Hz.csv` | 8 940 / 8 160 / 7 495 windows (2 Hz) | log-SD of the hand accelerometer norm (below) | activity labels |

**Beijing: dequantisation (choice to validate).** PM2.5 is reported in whole
µg/m³ with a floor at 3 µg/m³ (402 hours in Huairou's first year, dataset README),
so many consecutive hours are exactly equal. On `ln PM2.5` as provided, a K = 3 PMC
fitted by ICE puts one state on the atom ln 3 with a standard deviation of 1e-4
and a Clayton copula at τ = 1: the likelihood is unbounded and the fit "wins" by
thousands of nats (fits `huairou_rawlog`, section 2). The fitted series is
therefore `ln(PM2.5 + U(−0.5, 0.5))` (fixed seed per station); every imputation
and forecast is scored against `ln PM2.5` as reported.

**PAMAP2: feature, windows and missingness rule (choices to validate).**

* Windows of 50 consecutive 100 Hz rows (0.5 s → a 2 Hz series), aligned on the
  first row of the file, last incomplete window dropped.
* `Y` = log of the standard deviation of ‖a‖ (column `Y` of the `_100Hz` files:
  norm of the ±16 g hand accelerometer, gravity included) over the valid samples
  of the window — the intensity of the hand motion. The window mean of ‖a‖, the
  other obvious choice, separates the groups less well: supervised HMC-IN error
  on the complete series, K = 2, 6.7 / 4.7 / 10.6 % with the log-SD against
  14.2 / 16.2 / 11.4 % with the mean (subjects 102 / 105 / 108;
  `pamap2_feature_check.csv`).
* Label of a window = majority activityID; activity 0 (transient) windows stay in
  the series but are excluded from every error rate.
* **Groups.** K = 2: rest = {lying, sitting, standing, ironing}, active = all other
  protocol activities. K = 3: rest as above; locomotion = {walking, cycling,
  ascending stairs, descending stairs, vacuum cleaning}; vigorous = {running, Nordic
  walking, rope jumping}. Nordic walking is "vigorous" because the hand sensor sees
  the pole swing. The grouping was chosen with the feature in view: with ascending
  stairs vigorous and Nordic walking locomotion instead, the supervised HMC-IN
  error is 20.8 / 20.6 / 17.6 % against 13.0 / 9.4 / 11.7 % (log-SD).
* **Real gaps.** A window is missing when ≥ 5 of its 50 samples are NaN (≥ 10 %).
  A rule on a *mostly* missing window leaves almost no missing window (the longest
  hand dropout is 77 samples, dataset README); the "any NaN" rule is kept as a
  sensitivity check. A window with fewer than 2 valid samples (rare) gets its SD
  from linearly interpolated samples in the complete series.
* **Gapped series** = real gaps + MCAR blocks of 20 windows (10 s) removing 10 % of
  the series (`pmcprg.missing.patterns.mcar`, seed = subject number).

## Methods

**Models.** `hmc_in` (HMC-IN, Gaussian state margins); `pmc_state` (PMC with state
margins = stationary reversible HMC-DN, Gaussian margins); `pmc_pair` (general PMC,
K² Gaussian pair margins); `pmc_state_gice` (state margins whose family is selected
at every M-step among norm, gamma, invgamma, betaprime by BIC — tsNH4 and Huairou
only). Copula families are selected at every M-step among Gauss, Clayton,
Gumbel–Hougaard and Frank by maximum likelihood. K = 2 and K = 3.

**Starts.** Start 0 is the library's k-means warm start (observed rows only, seed 0).
Starts 1 and 2 are the library multistart draws from it (`build_multistart_inits`,
jitter 0.25, `multistart_families = "random"` for the copula models), identical for
all strategies of a cell, so the strategies are compared from the same starting
points. Each start is a separate single-start run.

**Strategies.** ICE `missing_strategy = "available"` and `"impute"` (5 completed
series per iteration, library default), `max_iter = 50`, `tol = 1e-4`; SEM with
30 iterations (library default; the returned model is the last iterate, as a user
gets it). The log-likelihood of every fitted model is recomputed exactly as
log p(y_obs) with `classify`, whatever the strategy. Starts per cell: tsNH4 3 for
every strategy (GICE: impute and SEM 1); Huairou and PAMAP2 subject 102: 3 for
"available", 3 for impute and SEM at K = 2 and for HMC-IN, 1 otherwise; PAMAP2
108 and 105: "available" only (3 starts); complete PAMAP2 series: ICE, 3 starts;
Aotizhongxin: "available", 1 start.

**Model and K selection.** BIC = −2 log p(y_obs) + k ln n_obs, k = free prior
parameters (K(K−1) for A, K(K+1)/2 − 1 for a symmetric p) + margin parameters +
one τ per copula pair; the family choices are not counted.

**Imputation.** `impute` of the fitted model: posterior mean (point), 5/95 %
posterior quantiles (90 % interval) and 200 joint FFBS draws (CRPS,
`crps_from_samples`; drawn on the quadrature nodes for the grid variants).
Baselines: linear interpolation and LOCF (point only: their CRPS is their MAE),
and an exact Gaussian AR(1) smoother written here (`rs_common.ar1_fit` /
`ar1_smooth`: maximum likelihood of the AR(1) on the observed values with gaps,
Gaussian bridge between the nearest observed neighbours = the Kalman/RTS smoother
of that model; Gaussian CRPS and ±1.645 sd interval).

* tsNH4: scored at the 883 real gaps against `Y_true`, split by gap length (1–2,
  3–20, > 20).
* Beijing: masked cross-validation on Huairou and Aotizhongxin. `patterns.mcar`
  hides 5 % or 20 % of the series in blocks of 1, 6 or 48 hours (5 seeds each) among
  the observed values; HMC-IN and the best Gaussian-margin PMC (each at its BIC
  K, from the fits on the real series) are refitted on each masked series (ICE
  "available", k-means start) and scored on the hidden positions only. The same
  refit gives the stability of the segmentation: label agreement with the fit on
  the unmasked series, on observed and on hidden positions.

**Classification (PAMAP2).** For every fit, MPM error rates on non-transient windows,
states matched to groups by the Hungarian permutation of all scored windows,
split by `error_rate_split` into missing / observed windows and further into
real-gap / MCAR windows:

* complete-series fit on the complete series;
* gapped-series fit on the gapped series: exact marginalisation (`classify` with
  NaN) vs fill-in then classify (posterior mean = `plugin`, linear, LOCF);
* gapped-series fit applied to the complete series (estimation damage only);
* supervised oracle: prior, margins and copulas from one library M-step on the
  labels (transient windows skipped, prior smoothed by 1e-4), same inference —
  separates the estimation error (unsupervised − oracle) from the model error
  (oracle).

**State-dependent missingness.** Missing-window rates per group; and on the series
with the real gaps only, a supervised HMC-IN classified with the library's
ignorable rule (emission 1 at a missing window) vs a missingness-aware HMC-IN whose
emission is f_k(y)(1 − r_k) when observed and r_k when missing, r_k = P(missing |
group k) from the labels (a numpy forward–backward, checked against the library in
the ignorable case).

**Forecasts** (Aotizhongxin). Models fitted on the first 75 % of the year; origins
every 24 h in the last 25 %, each conditioned on all earlier observations; `forecast`
h = 1…24 vs persistence (last observed value) and the AR(1) forecast. The CRPS of
the model forecasts is the Gaussian CRPS of their predictive mean and sd.

## Results

### 1. Model and K selection (BIC, ICE "available", best of the starts)

| series | lowest BIC | next | HMC-IN (best K) |
|---|---|---|---|
| tsNH4 | PMC state margins, K = 2 (BIC −7415.4) | PMC state K = 3, ΔBIC +52.3; GICE K = 3 +59.4; PMC pair K = 2 +70.1 | K = 3, +9681.2 |
| Huairou | GICE state margins (gamma, gamma, norm), K = 3 (877.3) | PMC state K = 3 +118.9; PMC pair K = 3 +221.4 | K = 3, +13169.5 |
| Aotizhongxin | PMC pair margins, K = 3 (−925.3) | PMC pair K = 2 +64.8; PMC state K = 3 +257.9 | K = 3, +14252.9 |
| PAMAP2 102 / 105 / 108, complete | PMC pair margins, K = 3 (all three) | PMC state K = 3: +722.3 / +878.0 / +1066.3 | K = 3 |
| PAMAP2 102 / 105 / 108, gapped | PMC pair margins, K = 3 (all three) | PMC state K = 3: +556.1 / +585.0 / +982.8 | K = 3 |

* The copulas are worth thousands of nats on every series (tsNH4: log p(y_obs)
  3748.7 for the PMC against −1083.7 for HMC-IN K = 3): within-regime temporal
  dependence, not the regime switches, carries most of the likelihood.
* K = 3 wins everywhere except tsNH4, including PAMAP2, where the K = 3
  unsupervised segmentations are 27.7–43.7 % wrong (section 5): BIC selects a
  density, not a segmentation.
* Huairou's K = 3 PMC with Gaussian margins spends a state (π = 0.036) on the
  3 µg/m³ detection floor: mean 1.0895 ≈ ln 3, sd 0.097 — the width of the
  dequantisation jitter on the log scale. A real feature of the data, but not a
  pollution regime.

### 2. Unbounded and degenerate likelihoods

| series | K | model | log p(y_obs) | state means | state sd |
|---|---|---|---|---|---|
| Huairou, ln PM2.5 as provided | 3 | PMC state | **4399.1** | 1.0986, 3.0934, 4.3085 | **0.0001**, 1.0054, 0.7941 (copula c₀₀: Clayton τ = 1.000) |
| Huairou, dequantised | 3 | PMC state | −407.7 | 1.0895, 3.0699, 4.3347 | 0.0972, 1.0702, 0.7785 |
| Huairou, ln PM2.5 as provided | 2 | PMC state | −668.2 | 2.8929, 4.2897 | 1.1517, 0.8452 |
| Huairou, dequantised | 2 | PMC state | −697.7 | 2.8671, 4.2925 | 1.1497, 0.8384 |

Degenerate fits (a margin sd below 1 % of the series sd, or a state with
stationary probability below 0.5 %) also occur on the continuous tsNH4 series:
13 of its 64 fits, all K = 3 copula models except one SEM start at K = 2, mostly
SEM and "impute" (10 of 13). They are not harmless: the best-likelihood start of
the K = 3 pair-margin PMC ("available") is degenerate, and its 90 % imputation
intervals cover 27 % of the truth with a mean width of 0.11, where the K = 2 pair
model covers 92 % with a width of 0.73. The runner excludes degenerate fits from
the phase-2 model choice.

### 2b. ICE "available" vs ICE "impute" vs SEM

Pooled over tsNH4, Huairou and PAMAP2 subject 102 (gapped), 22 (K, model) cells:

| strategy | models | cells | median ΔLL (best start) | range | median ΔLL, same k-means start | within 1 nat | median LL spread, 3 starts | median lowest agreement | time vs "available" |
|---|---|---|---|---|---|---|---|---|---|
| available | copula models | 16 | 0 | – | 0 | – | 95.5 | 0.736 | 1× |
| impute | copula models | 16 | −10.7 | −308.4 … +33.1 | −0.6 | 38 % | 70.0 | 0.758 | 4.9× |
| SEM | copula models | 16 | −12.9 | −276.3 … +52.5 | −3.4 | 0 % | 47.6 | 0.723 | 1.9× |
| available | HMC-IN | 6 | 0 | – | 0 | – | 0.1 | 0.990 | 1× |
| impute | HMC-IN | 6 | −0.0 | −0.3 … 0.0 | −0.2 | 100 % | 0.4 | 0.990 | 7.7× |
| SEM | HMC-IN | 6 | +0.7 | −0.2 … +9.7 | −0.3 | 100 % | 1.0 | 0.994 | 7.0× |

(ΔLL: log p(y_obs) minus that of "available" in the same cell; "impute" and SEM
have a single start in 8 of the 16 copula cells, hence the −308 and −276 of
PAMAP2 102 K = 3 pair margins, −37.2 and −5.2 from the common start.)

* **Does the result depend on the strategy?** For HMC-IN, no. For the copula
  models the three strategies often end in different basins — lowest label
  agreement with the best fit of the cell 0.72–0.76 — but so do three starts of
  the same strategy (spread 48–96 nats). The strategy is one more source of
  multimodality, not a systematic bias.
* Wins of "impute" or SEM exist but are not predictable: Huairou GICE K = 3,
  "impute" +10.0 and SEM −5.6; Huairou GICE K = 2, SEM +27.3 from the same start;
  tsNH4 pair margins K = 2, SEM +52.5 (a non-degenerate start), "impute" −81.2.
* GICE margin families agree across strategies on Huairou K = 2 (gamma | norm,
  same copula families); on tsNH4 K = 2 only "impute" selects a gamma margin. The
  library's advice to prefer "impute" when margin families are selected
  (`DEFAULT_MISSING_STRATEGY` docstring, simulated data) is not borne out here:
  from the same start GICE "impute" is +10.0 (Huairou K = 3), −0.6 (Huairou K = 2),
  −45.0 and −68.5 nats (tsNH4 K = 2 and 3) relative to "available".
* ICE traces are not monotone on these data: 48 of the 112 "available" fits
  decrease at least once; 55 of 216 ICE fits return a model more than 1 nat below
  the best iterate of their own trace (median 16.6 nats, largest 50.2) because the
  early stop keeps the last iterate.

Figure `figures/strategies_ll_vs_time.png`: log-likelihood below the best fit of
the cell against estimation time, every start.

### 3. tsNH4: imputation at the 883 real gap positions (log scale)

Positions per gap length: 1–2: 120, 3–20: 300, > 20: 463. Best-likelihood start
of each cell; CRPS of linear and LOCF = their MAE.

| method | RMSE all | 1–2 | 3–20 | > 20 | CRPS all | 1–2 | > 20 | 90 % coverage (width) |
|---|---|---|---|---|---|---|---|---|
| linear interpolation | **0.242** | 0.102 | 0.087 | **0.322** | 0.161 | 0.057 | 0.251 | – |
| AR(1) smoother | 0.266 | 0.102 | 0.087 | 0.357 | 0.122 | 0.044 | 0.189 | 0.94 (0.75) |
| LOCF | 0.394 | 0.099 | 0.193 | 0.519 | 0.234 | 0.069 | 0.345 | – |
| PMC state K = 2, available (BIC choice) | 0.315 | 0.102 | 0.087 | 0.426 | 0.132 | 0.045 | 0.207 | 0.92 (0.90) |
| PMC pair K = 2, available | 0.257 | 0.095 | 0.088 | 0.345 | 0.122 | 0.043 | 0.189 | 0.92 (0.73) |
| PMC pair K = 2, SEM | 0.251 | 0.098 | 0.088 | 0.335 | **0.114** | 0.044 | **0.174** | 0.93 (0.72) |
| PMC pair K = 2, impute | 0.286 | **0.083** | 0.087 | 0.386 | 0.131 | **0.042** | 0.203 | 0.92 (0.79) |
| HMC-IN K = 3, available | 0.452 | 0.356 | 0.389 | 0.509 | 0.242 | 0.193 | 0.278 | 0.96 (1.62) |
| PMC pair K = 3, available (degenerate) | 0.298 | 0.097 | 0.087 | 0.402 | 0.133 | 0.044 | 0.209 | 0.27 (0.11) |

* Gaps of 1–2 steps: the PMC posterior mean is the best point imputation (pair
  margins 0.083–0.098 against 0.102 for linear and AR(1)); gaps of 3–20 steps:
  every method except LOCF and HMC-IN ties at 0.087–0.088.
* Gaps longer than 20 steps (9 gaps, up to 26 h): linear interpolation has the
  lowest RMSE. The model posteriors pull towards the regime means inside the gap
  (figure `figures/tsnh4_longest_gap.png`), which the truth does not do.
* The PMC intervals are calibrated (0.92–0.93) and the pair-margin PMC matches or
  beats AR(1) on CRPS; the BIC choice (state margins) is 8 % worse than AR(1) on
  CRPS. HMC-IN, which imputes a mixture of state means, is useless for imputation.

Figure `figures/tsnh4_imputation_by_gap.png`.

### 4. Beijing PM2.5: masked cross-validation and forecasts

Refitted HMC-IN and the best Gaussian-margin PMC (BIC: Huairou PMC state K = 3,
Aotizhongxin PMC pair K = 3; HMC-IN K = 3 at both). Mean over 5 seeds, 20 % hidden
(the 5 % tables are in `results/tables.md`):

| station | block (h) | CRPS PMC | HMC-IN | AR(1) | linear (MAE) | LOCF (MAE) | RMSE PMC / AR(1) / linear | coverage PMC / AR(1) | agreement obs. / hidden |
|---|---|---|---|---|---|---|---|---|---|
| Huairou | 1 | **0.160** | 0.288 | 0.180 | 0.228 | 0.300 | 0.375 / 0.382 / 0.412 | 0.91 / 0.92 | 0.941 / 0.891 |
| Huairou | 6 | **0.263** | 0.356 | 0.290 | 0.377 | 0.554 | 0.552 / 0.564 / 0.587 | 0.91 / 0.93 | 0.901 / 0.791 |
| Huairou | 48 | **0.495** | 0.528 | 0.525 | 0.767 | 0.981 | 0.934 / 0.945 / 1.044 | 0.91 / 0.94 | 0.987 / 0.709 |
| Aotizhongxin | 1 | **0.118** | 0.250 | 0.131 | 0.159 | 0.232 | 0.258 / 0.260 / 0.260 | 0.91 / 0.92 | 0.642 / 0.617 |
| Aotizhongxin | 6 | **0.239** | 0.331 | 0.252 | 0.331 | 0.486 | 0.496 / 0.499 / 0.498 | 0.87 / 0.90 | 0.753 / 0.643 |
| Aotizhongxin | 48 | 0.531 | 0.538 | **0.527** | 0.783 | 1.013 | 0.963 / 0.958 / 1.044 | 0.86 / 0.89 | 0.736 / 0.498 |

* Paired over the 30 (rate, block, seed) cells of a station, the PMC has a lower
  CRPS than the AR(1) smoother in 30 of 30 at Huairou (median −0.025) and 23 of 30
  at Aotizhongxin (median −0.012); HMC-IN in 4 of 60 (median +0.077).
* The gain is in the predictive distribution, not in the point: the PMC's RMSEs
  are within 3 % of AR(1)'s in every cell. Both beat linear interpolation clearly
  for 48 h blocks.
* The pair-margin PMC's intervals under-cover (0.86–0.88 for blocks ≥ 6 h).
* **Segmentation stability.** On the observed positions the refit on a masked
  series agrees with the unmasked fit at 0.971–0.993 for HMC-IN and 0.901–0.987
  for Huairou's state-margin PMC, but only 0.642–0.881 for Aotizhongxin's
  pair-margin PMC (over both rates): the pair model's segmentation is not
  reproducible from one masked copy of the data to the next. On the hidden
  positions agreement drops with the block length (48 h: 0.498–0.723).
* **Forecasts** (91 daily origins in the last quarter): RMSE at h = 1 / 6 / 24 h,
  PMC pair K = 3 0.396 / 1.040 / 1.214, AR(1) 0.397 / 1.011 / 1.187, persistence
  0.410 / 1.109 / 1.489, HMC-IN 0.591 / 1.087 / 1.207; Gaussian CRPS at h = 1,
  0.200 (PMC) vs 0.206 (AR(1)). The PMC's 90 % forecast intervals cover 0.700–0.844
  against 0.809–0.867 for AR(1).

Figure `figures/beijing_cv_crps.png`.

### 5. PAMAP2: classification

**Missingness is state-dependent.** Share of 2 Hz windows missing (≥ 10 % rule /
any NaN), per group:

| subject | rest | locomotion | vigorous | transient | whole series (≥ 10 % / any / added MCAR) |
|---|---|---|---|---|---|
| 102 | 0.00 / 0.75 | 2.07 / 13.29 | 10.82 / 46.65 | 0.27 / 3.46 | 1.89 / 10.34 / 9.84 |
| 105 | 0.24 / 1.37 | 4.02 / 13.83 | 0.43 / 8.19 | 0.15 / 3.22 | 1.33 / 6.54 / 9.87 |
| 108 | 0.19 / 1.85 | 0.83 / 13.36 | 0.83 / 22.81 | 2.43 / 11.81 | 1.24 / 11.09 / 9.80 |

(Subject 102: running 17.93 %, Nordic walking 12.10 %, walking 7.07 %; subject 105:
walking 13.57 %.) Figure `figures/pamap2_missing_by_group.png`.

**Model error vs estimation error** (complete series, error %):

| subject | K | oracle HMC-IN | oracle PMC state | oracle PMC pair | unsup. HMC-IN | unsup. PMC state | unsup. PMC pair |
|---|---|---|---|---|---|---|---|
| 102 | 2 | 6.7 | 3.9 | 3.6 | 9.6 | 6.3 | 12.7 |
| 105 | 2 | 4.7 | 2.6 | 2.6 | 21.5 | 44.2 | 45.7 |
| 108 | 2 | 10.6 | 14.2 | 13.9 | 18.8 | 19.2 | 17.0 |
| 102 | 3 | 13.0 | 15.7 | 6.7 | 38.6 | 39.5 | 40.7 |
| 105 | 3 | 9.4 | 16.9 | 14.0 | 43.6 | 38.5 | 33.1 |
| 108 | 3 | 11.7 | 35.8 | 12.9 | 43.7 | 36.6 | 27.7 |

The models can represent the groups (oracle 2.6–14.2 % at K = 2, 6.7–16.9 % at
K = 3 except the state-margin PMC of 108) but unsupervised estimation does not
find them: at K = 3 no fit is below 27.7 %. The strip of subject 102 shows how:
its 5-minute vigorous bout is split between the states matched to "locomotion"
and "rest" instead of forming its own state (figure `figures/pamap2_strip.png`).

**Gap handling** (gapped series, error % on the missing windows, mean over subjects):

| K | fit | model | marginalise | plug-in mean | linear | LOCF | marginalise, real gaps | marginalise, MCAR |
|---|---|---|---|---|---|---|---|---|
| 2 | oracle | HMC-IN | 6.5 | 6.4 | 11.7 | 9.0 | 1.1 | 7.1 |
| 2 | oracle | PMC state | 5.8 | 4.3 | 13.6 | 8.0 | 1.1 | 6.3 |
| 2 | oracle | PMC pair | 6.1 | 4.1 | 11.3 | 8.0 | 1.1 | 6.7 |
| 3 | oracle | HMC-IN | 11.2 | 11.3 | 14.8 | 16.6 | 1.1 | 12.3 |
| 3 | oracle | PMC pair | 7.2 | 8.0 | 16.7 | 15.5 | 1.0 | 8.0 |
| 2 | unsup. | PMC state | 18.3 | 13.9 | 14.5 | 9.9 | 2.5 | 20.6 |
| 3 | unsup. | PMC pair | 32.0 | 41.7 | 46.4 | 48.3 | 40.1 | 32.5 |

* With a good model, marginalising (or plugging in the posterior mean) beats
  linear and LOCF fill-in by 2–10 points on the missing windows. With a poorly
  estimated model the ordering is erratic (K = 2 PMC state: LOCF best).
* The real-gap windows are easy (1.0–1.1 % for the oracles above): isolated
  windows inside long activity bouts. The 10 s MCAR blocks are where the errors are.
* **Complete vs gapped estimation.** The gapped-series fit classifies the complete
  series within a median 0.4 points of the complete-series fit; in 5 of 18
  (subject, K, model) cells the difference exceeds 5 points, in both directions
  (105 K = 2 pair margins: 45.7 → 10.8 %; 108 K = 3 pair margins: 27.7 → 46.7 %)
  — a change of basin, not a loss of information.

Figure `figures/pamap2_errors.png`.

**Non-ignorable missingness** (supervised HMC-IN, real gaps only; error % on the
real-gap windows):

| subject | K | rule | missing windows | ignorable | aware | truth share among missing | aware posterior share |
|---|---|---|---|---|---|---|---|
| 102 | 3 | ≥ 10 % | 159 | 0.0 | 8.8 | 0.000 / 0.289 / 0.711 | 0.000 / 0.209 / 0.791 |
| 102 | 3 | any | 797 | 2.1 | 34.1 | 0.019 / 0.370 / 0.611 | 0.014 / 0.063 / 0.924 |
| 105 | 3 | any | 424 | 3.1 | 2.8 | 0.068 / 0.705 / 0.226 | 0.052 / 0.722 / 0.226 |
| 108 | 3 | any | 560 | 4.3 | 3.8 | 0.070 / 0.489 / 0.441 | 0.042 / 0.501 / 0.457 |
| 102 | 2 | any | 797 | 0.9 | 0.8 | 0.019 / 0.981 | 0.014 / 0.986 |

The missingness is informative — a missing window of subject 102 (≥ 10 % rule) is
vigorous in 71 % of cases, against 1044 of its 5268 scored windows overall — but
the ignorable posterior already puts it there (mean posterior share 0.711 with the
≥ 10 % rule, 0.608 with any NaN; `mnar_pamap2.csv`), from the neighbouring windows.
Adding P(missing | state) as an independent emission over-counts runs of missing
windows and pushes them to "vigorous" (0.924 against a true 0.611); elsewhere it
changes the error by at most 0.5 points. A missingness model for these data
would need to be dependent in time (bursts), which the library does not offer;
for gaps this short, ignoring the mechanism is the better choice. The hand-written
ignorable recursion matches the library's posteriors to 2.7e-12.

## Failure cases

1. **Unbounded likelihood on discretised data** (Beijing): a variance collapses on
   an atom and a copula goes to τ = 1; the likelihood gains thousands of nats and
   would win every model-selection criterion. Nothing in the fit warns. Workaround
   here: dequantisation.
2. **Degenerate states on continuous data** (tsNH4 K = 3, 13 of 64 fits): tiny or
   near-constant states, with overconfident imputation (27 % coverage).
3. **Likelihood-best ≠ label-best segmentation** (PAMAP2): at K = 3, 27.7–43.7 %
   unsupervised error against 6.7–16.9 % for the supervised HMC-IN and pair-margin
   models; the K = 2 copula fits of subject 105 split by dependence, not intensity
   (44.2–45.7 %).
4. **Multimodality of the copula models**: 3 starts differ by a median 95.5 nats
   and a label agreement of 0.736; the pair-margin PMC's segmentation is not
   reproducible under masking (agreement 0.642–0.881).
5. **Long gaps**: beyond ~20 steps (tsNH4) no model beats linear interpolation on
   RMSE; the posterior mean reverts to regime means.
6. **State-dependent missingness, modelled naively**, is worse than ignoring it.

## Library observations (no library code changed)

* **Frank τ range.** `CopulaEnum.from_short_name("Frank").constructible_tau_range()`
  is (−1 + ε, 1 − ε) but `CopulaFrank` only reaches |τ| = 0.994299 (θ = 700): a
  τ in between logs a WARNING and is clamped. ICE's bounded τ search and the
  multistart jitter both produce such values (57 fits with the warning), and a
  fitted model can store a Frank τ it does not actually use (e.g. `pamap2_108_complete`
  K = 3 PMC state, c₂₀ = Frank(−0.996)), which warns again when reloaded. Reproducer:
  `CopulaFrank(tau_k=0.995)` → `WARNING … clamping`, `theta = 700`.
* **Multistart jitter warnings.** `perturb_initial_model` shifts τ by
  `jitter × span × N(0, 1)` and clips it with `correct_tau`, which logs a WARNING
  about a value the user never gave, and the message names τ = 1.0000 while the
  stored value is then pulled inside the constructible range:
  `perturb_initial_model(PMCModel("pmcprg/pmc/models/pmc_gauss_k2.toml"),
  np.random.default_rng(3), jitter=0.25)` logs `Gauss.correct_tau: τ=1.6615 outside
  valid range [-1.0000, 1.0000] — clipped to τ=1.0000.` and returns τ₀₁ = 0.9998.
  With jitter 0.25 the clipped starts sit at |τ| ≈ 1 (up to 40 fits per family
  carry such a warning).
* **Early stop returns the last iterate.** After `patience` consecutive
  regressions ICE stops and returns the model of the last iteration, not the best
  one: 55 of 216 ICE fits end more than 1 nat below their own best iterate (up to
  50.2 nats). E.g. `tsnh4__K2__pmc_state__available__s0`: trace … 3720.2, 3719.6,
  3714.8, 3713.3, returned 3713.3.
* **No guard on degenerate likelihoods**: no warning when a margin sd reaches
  1e-4 or a copula τ = 1 (failure case 1).

No crash, no `IncompatibleObservationError`, no non-finite estimate in 266 fits,
180 cross-validation tasks and 18 oracles.

## Runtime

Sum of task times (single-process equivalent, `tasks.csv`): 74.9 min — fits
48.9 min, masked CV 24.8 min, forecasts 0.8 min. The campaign ran on 3 processes;
its last invocation took 1015 s (`run_info_campaign.json`), after an interrupted
first invocation whose finished tasks were reused from the cache. Median
estimation time per start (s): tsNH4 3.3 / 19.9 / 6.9 ("available" / "impute" /
SEM), Huairou 7.6 / 24.2 / 10.4 (longest: GICE K = 3 "impute", 163.8 s), PAMAP2 102
gapped 2.6 / 9.7 / 10.4. A masked-CV refit of the K = 3 PMC takes 13–40 s.

## Reproduce

From the repository root:

```bash
# smoke run (< 1 min): short extracts → report/out/real_series/quick/
PYTHONPATH=. .venv/bin/python report/real_series/run_real_series.py --quick
# full campaign (3 processes) → results/ (small CSVs) and report/out/real_series/full/
PYTHONPATH=. .venv/bin/python report/real_series/run_real_series.py --jobs 3
# tables (printed, and results/tables.md) and figures/, from the CSVs only
# (the two time-series figures read report/out/real_series/full/figure_data/)
.venv/bin/python report/real_series/summarise.py
# smoke-run tables and figures
.venv/bin/python report/real_series/summarise.py --results report/out/real_series/quick/results \
    --figures report/out/real_series/quick/figures --extracts report/out/real_series/quick/figure_data
```

Tasks are cached in `report/out/real_series/<full|quick>/tasks/` (a rerun skips
finished tasks; `--force` recomputes). Seeds are fixed in the task specs: `--jobs 1`
and `--jobs 3` give identical CSVs apart from the timing columns (checked on
`--quick`). Threads of the numerical libraries are pinned to 1 per process.

```
report/real_series/
├── README.md               this file
├── run_real_series.py      campaign runner (phases, task cache, aggregation)
├── rs_common.py            data preparation, models, baselines, tasks
├── summarise.py            tables and figures from the CSVs
├── results/                fits.csv (one row per fit), imputation_tsnh4.csv,
│                           cv_beijing.csv, forecast_beijing.csv,
│                           classification_pamap2.csv, oracle_pamap2.csv,
│                           mnar_pamap2.csv, missingness_pamap2.csv,
│                           pamap2_feature_check.csv, tasks.csv,
│                           run_info.json (last invocation),
│                           run_info_campaign.json (the invocation that ran the
│                           campaign), tables.md — metrics and parameters only
└── figures/                PNG figures
report/out/real_series/     (git-ignored) <full|quick>/: tasks/ (cache), models/
                            (fitted models, JSON), labels/, imputations/,
                            figure_data/ (the extracts behind tsnh4_longest_gap.png
                            and pamap2_strip.png: they contain observations, so
                            they stay out of the repository)
```
