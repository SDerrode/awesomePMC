# Erroneous data on real sensors: the Intel Berkeley Lab temperatures

> **Status (2026-09-25, pmcprg d14e91d).** Every number below comes from
> the rerun on the current gap quadrature. Steps 1, 2 and 3 ran on pmcprg
> d14e91d (commits c151f21, d637a06 and 2bbe5b8). The 256-node check of step
> 2 ran on pmcprg 5f23fc9 (commit 6ef6a31), which only bounds the memory of
> the quadrature and gives bit for bit the results of d14e91d (CHANGELOG
> `[Unreleased]`, "memory of the gap quadrature").
>
> * **Library fixes since the first run** (CHANGELOG `[Unreleased]`): local
>   grids for missing values under strong dependence (P1), the forecast and
>   impute quantiles (P3), the empty state (P2), leading gaps, a calibrated
>   `quad_error` diagnostic and the cost of the quadrature (P5–P7), and local
>   grids placed from the filter (cause 4). They change every PMC number.
>   The HMC-IN results, the baselines and the labels are reproduced bit for
>   bit.
> * **Clean days: converged in practice.** At the default G = 64 the PMC
>   log-likelihood of days 1–10 is within 0.04 nats of G = 256, and the
>   flags are the same, although the quadrature WARNING fires.
> * **Failure windows: not converged, at G = 64 or 256.** Wherever a PMC
>   pass conditions on failing readings, its results move with G: the
>   non-sequential flags of motes 48 and 47, and through gating the BH lead
>   on mote 47 (section 2, "The check at 256 nodes"). The sequential recall
>   and the false alarms in normal operation do not move.
> * **Flag-and-mask finds no fixed point in 9 of the 12 PMC loops**
>   (section 3). The failure still becomes a state in every raw fit.
> * **`quad_error` saturates on long runs of missing rows** (library
>   problem 4, open): about 1 per missing row whatever the error, which puts
>   it in the thousands on the detection and robust windows.
> * **Clipped corner: decided, not changed.** The PIT at the clipped copula
>   corner (library problem 1) is left as is. `pmcprg/pmc/outliers.py`
>   documents why: without gating, a run of erroneous readings is judged
>   given its own first value, so `sequential=True` is the detection mode.

The simulation study of `report/erroneous_data` measured `pmcprg.pmc.outliers`
on isolated spikes. This study runs the same tools on a real failure mode.
In the Intel Berkeley Lab deployment, a mote whose battery dies reports a
temperature that climbs smoothly from about 22 °C to 122 °C over about a day,
then stays stuck at 122.15 °C. The dataset's `suspect` flag (temperature
outside [−10, 60] °C or humidity outside [0, 100] %) catches only the part
above 60 °C. The climb that precedes it stays inside the thresholds.

Three questions:

1. Is a model fitted on clean data calibrated?
2. Does that model, held fixed, flag the failing readings, how early, and at
   what false-alarm cost, compared with simple baselines?
3. Does flag-and-mask estimation (`robust_estimate`) survive a window that
   mixes clean and failing readings?

Every number below is printed by `summarise.py` from the CSVs in `results/`.
`results/tables.md` holds every table; `results/tables.tex` holds the key
ones. The 256-node checks are scripted too (`fit_clean.py`, `detect.py
--check`). The breakdowns of `quad_error` by run of missing rows are the
exception: one-off diagnostics, quoted in the text.

> **Quadrature caveats.** Every PMC computation uses the default
> `gap_nodes = 64`; the HMC-IN uses the exact K-state message.
>
> * **Clean days (step 1).** Converged in practice: the log-likelihood
>   moves by 0.006–0.036 nats from G = 64 to 256, and the flags are the
>   same.
> * **Detection (step 2).** Converged where the gated filter has integrated
>   the failing readings out; not converged where a pass conditions on
>   them, far outside every clean margin. The
>   non-sequential PMC results of motes 48 and 47 and the sequential BH
>   lead of mote 47 depend on G.
> * **Robust estimation (step 3).** Every PMC E-step warns, and the fits
>   were not checked at 256 nodes. The failure states are unlikely to
>   depend on G; the masks and cycles may.
> * **The WARNING itself.** On long runs of missing rows `quad_error`
>   reports about 1 per missing row, whatever the error (library problem
>   4). Its thousands on these windows are not nats.

## Summary

* **Calibration on clean data: no model passes the tests, but the copula
  model is close in the margin and the HMC-IN is far off.** On the held-out
  days 8–10 (n ≈ 7 600 rows per mote), every KS and Ljung–Box test rejects.
  - **PMC (K = 3).** The PIT is nearly uniform (KS D = 0.03–0.12) and too
    wide (sd of the normal scores 0.75–0.97). The lag-1 autocorrelation of
    the scores is 0.01–0.09, but 0.2–0.3 from lag 2 on: temperature has
    momentum that a first-order chain cannot carry. The rows after a gap
    are now calibrated like the others.
  - **HMC-IN.** Its scores have a lag-1 autocorrelation of 1.00. At 30 s, a
    model whose observations are independent given the state is only a
    level test.
* **Detection with the clean model held fixed works.** Sequential flagging
  catches every reading above 60 °C (recall 1.00 on all three motes, at 64
  nodes, and at 256 on the check window).
  - **PMC, sequential, BH at α = 1e-3.** The alarm that reaches the failure
    starts 6.1, 1.5 and 16.2 h before the threshold hit (motes 48, 22, 47);
    its first flag of the last 24 h comes 11.1, 17.3 and 16.2 h before it.
    It raises 0, 0 and 2 flags in 14–15 days of normal operation (one
    episode).
  - **The mote-47 lead depends on the quadrature.** At G = 256 the alarm on
    the climb breaks up, and the episode that reaches the failure starts
    3.2 h before it, against 15.2 h at G = 64 on the same window; its first
    flag does not move. The other sequential leads move by at most 2.3 h
    (mote 48 at α = 1e-3: 8.8 → 11.1 h), and the false alarms not at all.
  - **HMC-IN.** It warns earlier on motes 48 and 22 (13.4 and 6.5 h; 15.2 h
    on mote 47) but raises about 900–1 300 flags in the normal operation of
    motes 48 and 47, on warm afternoons.
  - **Fixed thresholds.** A plain 35 °C threshold gives 5.3, 6.8 and 13.1 h
    of warning with no false alarm on these three motes.
  - **Rolling filters.** The Hampel filter and the rolling robust z-score do
    not detect this failure: their recall is 0.00–0.13. Both adapt to a smooth
    drift and to a stuck value.
* **The flags before the threshold are mostly true early drift.** On motes
  48 and 47, 57 % and 89 % of the out-of-range climb is flagged by the
  sequential PMC at α = 1e-3; the rest is its lower end. Mote 22 is the
  exception (31 %): its clean-window PMC has a wide margin (sd 5.17 °C)
  that accepts a climb to about 50 °C. The false alarms, examined on the
  plots, fall into two classes:
  - warm afternoon peaks above the fit-window maximum (mote 48, 26–30 °C,
    599 of its 605 flags; mote 47, 30.7–32.2 °C);
  - morning events at 07:01 that recur on several days: a jump from about
    22 to 28–30 °C on mote 22, a 0.3 °C dip on mote 47 (probably a
    building schedule).

  Both are real temperatures, not sensor errors.
* **Innovation gating has a real-data cost: lock-out.** A gated legitimate
  reading makes its successors look wrong, because the gated filter keeps
  predicting the old level while the true level moves on. On mote 48's warm
  afternoons of 03-13 and 03-20, 4 and 2 non-sequential flags become runs
  of 331 and 268 sequential ones (605 against 12 normal-operation flags),
  at G = 64 and at G = 256 alike. On the held-out clean days the lock-out
  is small (10, 3 and 30 sequential flags against 3, 2 and 2). The
  Benjamini–Hochberg correction or α = 1e-4 removes almost all of it.
* **Robust re-estimation breaks down, as the simulation predicts.** In every
  raw fit (6 of 6), the failing readings form a state of their own:
  mean 59–120 °C, sd 3–37 °C, stay ≥ 0.9998 for the HMC-IN; a broad state
  of 39–52 °C, sd 26–35 °C, stay 0.97–1.00 for the PMC, which on motes 22
  and 47 spends a second state on the stuck 122.15 °C value.
  - **Default flag-and-mask.** Under that state almost nothing is flagged
    (0–22 rows), and every result keeps a failure state.
  - **No fixed point in 9 of the 12 PMC loops** (3 of 12 for the HMC-IN,
    all on mote 22). On mote 22 three loops end in one deterministic cycle:
    the raw fit, then the raw fit with its 13 flagged climb rows masked,
    every fit at the ICE cap of 50 iterations. On mote 47 the default and
    Hampel loops cycle through 3–15 masked rows. With the threshold
    pre-mask the masks drift (267–350 rows). The three PMC loops that
    converge keep a failure state.
  - **Hampel `initial_mask`.** It does not help, because it catches spikes,
    not drifts (6 of 6 break down).
  - **Other starts.** Starting from the clean-window model does not help
    either (6 of 6 keep a failure state).
  - **Threshold `initial_mask`.** A sensor-level pre-mask (the 60 °C
    threshold) gets 4 of 6 fits to mask every suspect row: both models on
    motes 48 and 47. The HMC-IN on mote 48 also masks the whole climb, the
    PMC on mote 48 66 % of it; on mote 47 the PMC keeps the climb as a
    33 °C state. On mote 22 (19 % of the window failing) both models
    release the pre-mask round after round and end in the raw fit's cycle.
* **Even the threshold oracle is contaminated.** Masking only the suspect
  rows leaves the climb in: the top state keeps an sd of 6–8 °C, against
  1.7–2.7 °C once the climb is masked too (HMC-IN).
* **A contamination component is needed for estimation, not for detection.**
  The failure is persistent: the fitted stay probability of the broad
  failure states is ≥ 0.97, and ≥ 0.9998 for the HMC-IN. It is also broad:
  sd 26–37 °C in 5 of the 6 raw fits. The indicator must therefore be
  Markov, not Bernoulli, and its law must be fixed, not a regular state
  that ICE can reshape. Conclusion below.
* **Four library problems found** (below, each with a reproduction or a
  measurement).
  1. **Clipped corner** (left as is, documented). The non-gated PIT of a
     reading that follows another reading beyond F⁻¹(1 − EPS) is about 0.5
     for the Gaussian and Gumbel–Hougaard copulas, although its log
     predictive density is about −1 200. With the current local grids it
     also reaches most rows after a gap inside the failure: the
     non-sequential recall of the PMC is 0.26 and 0.02 on motes 48 and 47.
     The sequential (default) flags are not affected.
  2. **Gap quadrature** (fixed at 39f249f, refined up to d14e91d). With
     strongly dependent copulas (τ ≈ 0.99–0.997 at 30 s), the quadrature
     had not converged at the default `gap_nodes = 64`, nor at 512. On the
     clean days of this study G = 64 is now within 0.04 nats of G = 256.
  3. **Leading gap** (fixed, P5). When a series started with missing rows,
     the first reading after them was misintegrated under strong
     dependence, with no WARNING (−4.3 nats at τ = 0.997 and −22 nats at
     τ = 0.999 on a 400-row simulated series, G = 64).
  4. **`quad_error` on long runs of missing rows** (open). It reports about
     1 per missing row, whatever the error: 4 533 for a trailing gap of
     4 656 rows, which contributes exactly 0 to the log-likelihood.

## Data and labels

The data are read from
`/Users/MacBook_Derrode/Documents/ProjetsRecherche/Markov/data/timeseries/intel_lab`
(`--data`) and are never copied into the repository. The files are
`output/mote<id>_epochs.csv`, preprocessed as in the dataset README: a 30 s
epoch grid of 102 706 epochs, with `Y_raw` the raw temperature, `suspect`
and `Y`. The study uses motes 48, 22 and 47, the selection of that README.

**Choices** (constants in `il_common.py`):

* **Dequantisation.** The readings are quantised at 0.0098 °C, and a third
  of the 30 s increments are exactly 0. The copula models resolve steps of
  that size, so their one-step predictive law would be discrete. Every
  present reading gets `+ U(−0.0049, 0.0049)` (seed `[11, mote]`): a
  randomised PIT, as for the Beijing PM2.5 series of `report/real_series`.
  The suspect readings (including the 122.15 °C plateau) are dequantised
  too.
* **Windows.**
  - Fit on days 1–7 (epochs 1–20 160) and hold out days 8–10 (20 161–28 800).
    Neither has a suspect reading.
  - Detection runs on epochs 28 801–102 706. That is 14–15 days of normal
    operation (with the network-wide outages of the dataset), then the
    failure and the plateau.
  - Robust estimation uses, per mote, the 4 days before the first reading
    above 60 °C and the day after it.
* **Labels** (`truth_labels`). The suspect flag is only a partial truth.
  Every observed row of a scored window gets one label:
  - `suspect` is the dataset flag. It covers the steep rise from 60 to
    122 °C and the plateau.
  - `climb` covers the rows from the **climb onset** to the first suspect
    reading. The onset is the last epoch before the first suspect reading at
    which the trailing 1 h median of the raw temperature was still at or
    below the maximum of the fit window (27.9, 34.2 and 29.0 °C for motes
    48, 22 and 47). From there the readings exceed a week of normal
    operation and climb to the threshold without coming back. These rows are
    very probably wrong: there is no nightly cooling, and the climb is
    continuous with the impossible readings.
  - `ambiguous` covers the rest of the 24 h before the first suspect
    reading, where possible early drift is mixed with a normal morning
    warm-up. On mote 22 it holds a 35–40 °C episode at 06:00 on 03-23,
    which this mote also shows on clean days: 35 °C on 03-08.
  - `normal` covers everything else; flags there are false alarms.

| mote | first reading > 60 °C | climb onset | climb (h) |
|---|---|---|---|
| 48 | epoch 74 940 (2004-03-25 01:28) | 73 479 (03-24 13:17) | 12.2 |
| 22 | epoch 71 746 (2004-03-23 22:51) | 71 044 (03-23 17:00) | 5.9 |
| 47 | epoch 74 454 (2004-03-24 21:25) | 72 679 (03-24 06:37) | 14.8 |

## Methods

**Step 1: `fit_clean.py`.**

* **Fits.** HMC-IN, and PMC with state margins, the copula variant of
  `report/real_series` (a stationary reversible HMC-DN).
  - Gaussian margins. Copula family selected at every M-step among Gauss,
    Clayton, Gumbel–Hougaard and Frank.
  - K ∈ {2, 3}; ICE `missing_strategy = "available"`, 50 iterations,
    tol 1e-4, best iterate returned. The gaps are NaN rows.
  - 3 starts per cell: the k-means warm start plus 2 library multistart
    draws (jitter 0.25; the copula families are redrawn for the PMC).
  - The best start is kept by its exact log p(y_obs), and K is chosen by
    BIC.
* **Calibration.** `predictive_pit` runs over days 1–10: the filter starts
  at epoch 1, so every held-out row is predicted from its whole past.
  `pit_checks` runs with 10 lags on the held-out rows, and on the fit rows
  for reference. The table also gives the flag counts at α = 1e-2, 1e-3 and
  1e-4.
* **Regimes.** MPM states of days 1–10 by hour of day.

**Step 2: `detect.py`.** The selected models are held fixed, and
`flag_outliers` runs on the later window:

* sequential and non-sequential;
* per-row α = 1e-2, 1e-3 and 1e-4;
* Benjamini–Hochberg at FDR α = 1e-2 and 1e-3.

The baselines run on the same rows:

* **Fixed thresholds.** 60 °C (the temperature half of the suspect rule,
  the reference), and 35, 40 and 50 °C upper thresholds. These round values
  were chosen after looking at the raw series, before scoring.
* **Clean envelope.** The fit-window range ± 2 °C.
* **Hampel identifier.** A centred window of 11 observed rows, as in the
  simulation study, with t = 3, 4 and 5 robust sd and the MAD floored at one
  quantum.
* **Rolling robust z-score.** The median and MAD of the observed rows of
  the trailing 24 h (the current row excluded, at least 30 rows), with
  t = 3, 4 and 5.

The scores:

* recall on the suspect rows, and on the rising suspect rows below 120 °C;
* precision against `suspect`, and against suspect ∪ climb;
* flags per label;
* false alarms in normal operation, counted in flags and in alarm episodes
  (flags less than 1 h apart form one episode);
* three lead times, measured before the first suspect reading:
  - the first flag of the whole window;
  - the first flag of the last 24 h;
  - the start of the alarm episode that reaches the failure (the main one).

**Step 3: `robust.py`.** Each mote and model (selected K) gets the fits
below. Every fit is ICE ("available", best iterate), started from the
k-means warm start of the rows left observed unless stated.

| fit | what |
|---|---|
| `raw` | the window as it is |
| `oracle` | the `suspect` rows set to NaN |
| `oracle_ext` | the `suspect` and `climb` rows set to NaN (the best truth available) |
| `fm` | `robust_estimate`, α = 1e-3, sequential flags, at most 10 refits |
| `fm_hampel` | the same, with `initial_mask` = Hampel (t = 4) |
| `fm_thr60` | the same, with `initial_mask` = the 60 °C threshold (a sensor-level flag, re-tested by the model) |
| `raw_cs`, `fm_cs` | `raw` and `fm` started from the clean-window model |

The scores are the parameters, the presence of a failure state (a state
whose mean exceeds the fit-window maximum by more than 2 °C), the mask
against the labels, and the MPM classification of the `normal` rows. That
classification is compared with the `oracle_ext` fit and with the
clean-window model (Hungarian alignment).

## Run

From the repository root (the scripts put the repository first on
`sys.path` and check that `pmcprg` is imported from it):

```bash
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/fit_clean.py --jobs 4
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/detect.py --jobs 4
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/robust.py --jobs 4   # --resume after an interruption
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/detect.py --check --jobs 4
.venv/bin/python report/erroneous_data/intel_lab/summarise.py
.venv/bin/python report/erroneous_data/intel_lab/repro_clipped_corner.py   # library problem 1
.venv/bin/python report/erroneous_data/intel_lab/repro_gap_nodes.py        # library problem 2 (≈ 20 s)
.venv/bin/python report/erroneous_data/intel_lab/repro_leading_gap.py      # library problem 3 (≈ 20 s)
```

`robust.py` saves each task to `results/cache/robust_tasks/` as it
finishes; `--resume` reuses the saved tasks, so an interrupted run loses
only its running tasks. `--max-rounds` (default 10) bounds the refits of
every flag-and-mask loop.

The scripts keep the WARNINGs of pmcprg instead of hiding them. Every task
counts the quadrature WARNINGs of its passes and records `quad_error`
against its threshold (0.05):

* `fits.csv`: per ICE start;
* `pit_checks.csv`: days 1–10;
* `detect_scores.csv`: the non-gated pass over the detection window;
* `robust_fits.csv`: the returned fit on its masked window.

The PMC computations use the library default `gap_nodes = 64`. Two checks
at 256 nodes are scripted:

* `fit_clean.py` reruns the selected PMC's PIT and sequential flags on days
  1–10 (the `check_*` columns of `pit_checks.csv`);
* `detect.py --check` reruns the PMC flags at 64 and 256 nodes on the rows
  up to 6 h after the first reading above 60 °C (`results/detect_check.csv`).

The non-sequential settings of `detect.py` share one `predictive_pit` pass
per model, thresholded as `flag_outliers(sequential=False)` does; the flags
are identical.

The wall times on an Apple arm64 laptop (32 GB; Python 3.14.7, numpy
2.5.3, scipy 1.18.1), from the `*_info.json` files and the run logs:

* `fit_clean.py`: 6.2 min with 4 processes (217 s of fits, 152 s of PIT
  checks);
* `detect.py`: 26 min with 4 processes (97 min of task time);
* `robust.py`: 3 h 05 min with 4 processes (13.2 h of task time). It was
  resumed with `--resume` after a first 1 h 14 min with 2 processes, which
  had finished 2 of the 48 tasks. The 12 PMC flag-and-mask tasks take
  11–105 min each; the 24 HMC-IN tasks take 0.6–17 s.
* `detect.py --check`: 56 min with 4 processes. Its G = 256 passes take
  2.6–38 min each, against 7–199 s at G = 64.

**Memory.** Run `--check` on pmcprg 5f23fc9 or later. Before 5f23fc9 the
G = 256 passes on these windows reached 21–23 GB per process (CHANGELOG,
"memory of the gap quadrature"). With 5f23fc9 a G = 256 sequential pass on
the mote-48 check window takes about 2.7 GB of footprint and 11 min on its
own. With 4 processes the check peaked at about 14.5 GB of footprint in all
(3.9 GB for the largest process), and `robust.py` at about 7.6 GB (a memory
guard sampling every 60 s).

The per-row flags and labels go to `results/cache/`, which is not
versioned. `detect.py --figures-only` redraws the figures from that cache.
Nothing is subsampled: the models run at the native 30 s resolution on
every observed epoch.

## Results

### 1. The clean window

| mote | model | BIC K=2 | BIC K=3 | K | means (°C) | sd (°C) | diagonal copulas (τ) | spread over starts (nats) |
|---|---|---|---|---|---|---|---|---|
| 48 | HMC-IN | 59224 | 43874 | 3 | 17.34 / 20.02 / 22.53 | 0.89 / 0.47 / 1.36 | – | 3 |
| 48 | PMC | −90106 | −92827 | 3 | 19.13 / 21.89 / 22.05 | 2.03 / 1.76 / 2.01 | Gauss(0.996) / Frank(0.946) / Gauss(0.989) | 2329 |
| 22 | HMC-IN | 82622 | 67679 | 3 | 18.20 / 23.49 / 31.79 | 1.36 / 2.15 / 1.22 | – | 1 |
| 22 | PMC | −79769 | −81512 | 3 | 22.45 / 22.75 / 23.36 | 1.79 / 5.17 / 3.40 | Gauss(0.975) / Clayton(0.997) / Clayton(0.920) | 939 |
| 47 | HMC-IN | 64679 | 47486 | 3 | 17.26 / 20.25 / 23.73 | 0.94 / 0.59 / 1.86 | – | 0 |
| 47 | PMC | −74363 | −79747 | 3 | 18.71 / 21.61 / 25.49 | 1.95 / 1.89 / 2.40 | Gauss(0.996) / GH(0.983) / GH(0.969) | 2823 |

* **K.** BIC chooses K = 3 for every mote and model. With about 17 600
  strongly dependent rows, BIC would keep adding states. K = 3 is the upper
  end of the range the study allows, and it is interpretable (next point).
* **Starts.** The PMC likelihood has many basins: the 3 starts end
  939–2 823 nats apart, as in `report/real_series`.
* **The PMC gains 63 700–74 600 nats of likelihood over the HMC-IN.** Its
  diagonal copulas have τ = 0.92–0.997, and τ = 0.996–0.997 for the state
  that holds the night: consecutive 30 s readings are almost deterministic
  given the previous one.
* **The HMC-IN fits are those of the first run**, bit for bit: they use the
  exact K-state message, which no library change touched.

**Regimes.** The day/night split is real for the HMC-IN on all motes and for
the PMC on mote 47. It is weaker for the PMC on motes 48 and 22, whose
night state also holds most of the day.

| mote | model | night share by state (00–06 h) | day share by state (10–16 h) | largest gap |
|---|---|---|---|---|
| 48 | HMC-IN | 0.64 / 0.23 / 0.13 | 0.02 / 0.43 / 0.54 | 0.62 |
| 48 | PMC | 1.00 / 0.00 / 0.00 | 0.61 / 0.03 / 0.36 | 0.38 |
| 22 | HMC-IN | 0.86 / 0.14 / 0.00 | 0.00 / 0.71 / 0.29 | 0.86 |
| 22 | PMC | 0.00 / 1.00 / 0.00 | 0.26 / 0.74 / 0.00 | 0.26 |
| 47 | HMC-IN | 0.66 / 0.21 / 0.12 | 0.01 / 0.49 / 0.50 | 0.65 |
| 47 | PMC | 1.00 / 0.00 / 0.00 | 0.19 / 0.70 / 0.11 | 0.81 |

The two models split the day differently:

* **HMC-IN: level regimes.** The cool state holds the night, and the warm
  states hold the day.
* **PMC: dynamics regimes.** On every mote one calm state, the most
  strongly dependent (τ = 0.996–0.997, stay probability 0.9988–0.9990),
  holds every night row and the smooth evening cooling. By day it still
  holds 61 %, 74 % and 19 % of the rows (motes 48, 22, 47); the other
  states take the fast morning rises, the afternoon peaks and the jumps.
  The margin means of the PMC states are within 2.9, 0.9 and 6.8 °C of each
  other (`figures/clean_regimes.png`).

**Calibration on the held-out days 8–10** (`figures/clean_pit.png`):

| mote | model | n | KS D (p) | LB(10) z (p) | LB(10) z² (p) | z mean | z sd | acf1(z) | flags α=1e-2 / 1e-3 / 1e-4 (expected) | sequential flags α=1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 48 | HMC-IN K=3 | 7551 | 0.247 (< 1e-300) | 74631 (< 1e-300) | 75353 (< 1e-300) | 0.76 | 1.61 | 1.00 | 911 / 678 / 393 (76 / 8 / 0.8) | 694 |
| 48 | PMC K=3 | 7551 | 0.045 (9.9e-14) | 4071 (< 1e-300) | 1006 (1.2e-209) | 0.02 | 0.90 | 0.05 | 80 / 3 / 1 (76 / 8 / 0.8) | 10 |
| 22 | HMC-IN K=3 | 7736 | 0.235 (< 1e-300) | 74996 (< 1e-300) | 74146 (< 1e-300) | 0.55 | 1.04 | 1.00 | 34 / 0 / 0 (77 / 8 / 0.8) | 0 |
| 22 | PMC K=3 | 7736 | 0.118 (1e-94) | 2151 (< 1e-300) | 3103 (< 1e-300) | −0.08 | 0.75 | 0.01 | 23 / 2 / 0 (77 / 8 / 0.8) | 3 |
| 47 | HMC-IN K=3 | 7605 | 0.148 (5.9e-146) | 74346 (< 1e-300) | 75703 (< 1e-300) | 0.53 | 1.36 | 1.00 | 730 / 384 / 266 (76 / 8 / 0.8) | 394 |
| 47 | PMC K=3 | 7605 | 0.032 (3.7e-07) | 4888 (< 1e-300) | 2011 (< 1e-300) | 0.04 | 0.97 | 0.09 | 97 / 2 / 0 (76 / 8 / 0.8) | 30 |

* **No model is calibrated in the sense of the tests.** With n ≈ 7 600 and
  30 s data, KS and Ljung–Box reject everything.
* **The magnitudes separate the models.**
  - **HMC-IN.** Its normal scores are a smooth curve (acf1 = 1.00). It is
    biased on the held-out days (z mean 0.53–0.76), which were warmer than
    the fit days. At α = 1e-3 it flags 384–678 held-out rows instead of 8 on
    motes 48 and 47. It is a level test whose tail is the fit week's range.
  - **PMC.** Nearly uniform (KS D 0.03–0.12), too wide: z sd 0.75–0.97, a
    hump in the PIT histogram, strongest on mote 22 (z sd 0.75, KS D
    0.12). Residual dependence: acf1 = 0.01–0.09, but about 0.2–0.3 from
    lag 2 on, still 0.1–0.2 at lag 60 (the figure): the momentum of warming
    and cooling phases. Its non-sequential flag counts are near or below
    nominal: 2–3 at α = 1e-3 against 8 expected, 23–97 at α = 1e-2 against
    76–77.
* **Sequential flags on clean data: little lock-out.** The sequential PMC
  flags 10, 3 and 30 held-out rows at α = 1e-3, against 3, 2 and 2 without
  gating, and the same counts at 256 nodes. The first run's 239 and 513
  (motes 48 and 47) came from models fitted and filtered through the
  unconverged quadrature. The lock-out that remains is measured on the
  detection window (section 2).
* **The PMC PIT after a gap is no longer quadrature-limited.** For the 933,
  809 and 897 held-out rows that follow a gap, the sd of the normal scores
  is 0.88, 0.74 and 1.01, against 0.91, 0.76 and 0.96 after an observed
  row. In the first run it was 1.53 on mote 48 at G = 64.

**Gap quadrature on the clean days** (`fit_clean.py`, the selected PMC on
days 1–10 at 64 and 256 nodes; `results/tables.md`):

| mote | `quad_error` G = 64 (limit 0.05) | `quad_error` G = 256 | log-lik days 1–10, G = 64 | log-lik, G = 256 | held-out flags α = 1e-3, non-sequential / sequential (both G) | same sequential rows |
|---|---|---|---|---|---|---|
| 48 | 0.19 | 0.0067 | 67 114.06 | 67 114.07 | 3 / 10 | yes |
| 22 | 1.72 | 0.034 | 59 688.92 | 59 688.88 | 2 / 3 | yes |
| 47 | 0.23 | 0.0004 | 58 729.45 | 58 729.43 | 2 / 30 | yes |

* **Converged in practice at G = 64.** The log-likelihood of days 1–10
  moves by 0.006–0.036 nats from G = 64 to 256. The sd of the scores after
  a gap (to 3 decimals), the flag counts and the sequentially flagged rows
  are the same.
* **The WARNING fires anyway.** `quad_error` is 0.19–1.72 at G = 64, 10–50
  times the change it estimates, and falls below its threshold at
  G = 256. The ICE E-steps of the PMC fits warn too (29–42 passes of the 3
  starts per cell; `quad_error` 0.08–2.5 for the kept fits).
* **The two passes agree.** The PIT filter and the batch pass give the same
  log-likelihood (67 114.06 against 67 114.06 on mote 48), as the
  leading-gap fix (P5) makes them: the clean windows start with a 17-, 2-
  and 17-row gap.

### 2. Detection on the later window

The clean-window models are held fixed. Values are for motes 48 / 22 / 47.
Normal operation covers 29 448, 26 937 and 28 658 observed rows (14–15
days per mote). Every PMC number here uses the default G = 64; the check
at 256 nodes follows the discussion.

| method | recall (suspect) | precision (suspect) | precision (suspect + climb) | share of climb rows flagged | normal-operation flags (episodes) |
|---|---|---|---|---|---|
| PMC seq α=1e-3 | 1.00 / 1.00 / 1.00 | 0.77 / 0.94 / 0.84 | 0.80 / 0.98 / 0.97 | 0.57 / 0.31 / 0.89 | 605 (8) / 106 (9) / 49 (7) |
| PMC seq α=1e-4 | 1.00 / 1.00 / 1.00 | 0.96 / 0.97 / 0.87 | 1.00 / 1.00 / 0.99 | 0.54 / 0.24 / 0.79 | 0 (0) / 0 (0) / 3 (1) |
| PMC seq BH α=1e-3 | 1.00 / 1.00 / 1.00 | 0.96 / 0.97 / 0.87 | 1.00 / 1.00 / 0.99 | 0.54 / 0.24 / 0.78 | 0 (0) / 0 (0) / 2 (1) |
| PMC nonseq α=1e-3 | 0.26 / 1.00 / 0.02 | 0.93 / 0.96 / 0.27 | 0.98 / 1.00 / 0.87 | 0.20 / 0.30 / 0.28 | 12 (8) / 9 (9) / 8 (7) |
| PMC nonseq BH α=1e-3 | 0.26 / 1.00 / 0.02 | 0.96 / 0.97 / 0.35 | 1.00 / 1.00 / 0.97 | 0.15 / 0.24 / 0.21 | 0 (0) / 0 (0) / 0 (0) |
| HMC-IN seq α=1e-3 | 1.00 / 1.00 / 1.00 | 0.61 / 0.87 / 0.63 | 0.66 / 0.97 / 0.74 | 1.00 / 0.84 / 1.00 | 1290 (3) / 0 (0) / 924 (3) |
| HMC-IN nonseq α=1e-3 | 1.00 / 1.00 / 1.00 | 0.62 / 0.87 / 0.63 | 0.66 / 0.97 / 0.74 | 1.00 / 0.84 / 1.00 | 1271 (3) / 0 (0) / 905 (3) |
| HMC-IN seq BH α=1e-3 | 1.00 / 1.00 / 1.00 | 0.67 / 0.89 / 0.69 | 0.72 / 0.98 / 0.81 | 1.00 / 0.80 / 0.95 | 967 (3) / 0 (0) / 636 (3) |
| threshold 35 °C | 1.00 / 1.00 / 1.00 | 0.97 / 0.84 / 0.92 | 1.00 / 0.95 / 1.00 | 0.44 / 0.96 / 0.51 | 0 (0) / 0 (0) / 0 (0) |
| threshold 40 °C | 1.00 / 1.00 / 1.00 | 0.97 / 0.93 / 0.97 | 1.00 / 1.00 / 1.00 | 0.35 / 0.58 / 0.20 | 0 (0) / 0 (0) / 0 (0) |
| threshold 50 °C | 1.00 / 1.00 / 1.00 | 0.98 / 0.97 / 0.99 | 1.00 / 1.00 / 1.00 | 0.18 / 0.22 / 0.06 | 0 (0) / 0 (0) / 0 (0) |
| clean envelope ±2 °C | 1.00 / 1.00 / 1.00 | 0.92 / 0.88 / 0.69 | 0.98 / 0.98 / 0.81 | 0.88 / 0.81 / 0.95 | 54 (1) / 0 (0) / 617 (3) |
| Hampel t=4 | 0.00 / 0.00 / 0.00 | 0.05 / 0.05 / 0.02 | 0.07 / 0.20 / 0.19 | 0.01 / 0.03 / 0.03 | 19 (15) / 41 (19) / 34 (25) |
| robust z 24 h t=4 | 0.05 / 0.13 / 0.09 | 0.13 / 0.27 / 0.13 | 0.33 / 0.35 / 0.38 | 1.00 / 0.29 / 1.00 | 587 (5) / 1412 (4) / 863 (4) |

**Lead times** (hours before the first reading above 60 °C; negative =
after it):

| method | episode reaching the failure | first flag in the last 24 h | first flag of the whole window |
|---|---|---|---|
| PMC seq α=1e-3 | 8.8 / 2.1 / 16.2 | 11.1 / 17.8 / 16.2 | 277.6 / 351.8 / 270.5 |
| PMC seq α=1e-4 | 6.1 / 1.5 / 16.2 | 11.1 / 17.3 / 16.2 | 11.1 / 17.3 / 38.4 |
| PMC seq BH α=1e-3 | 6.1 / 1.5 / 16.2 | 11.1 / 17.3 / 16.2 | 11.1 / 17.3 / 38.4 |
| PMC nonseq α=1e-3 | 8.8 / 2.1 / 3.5 | 11.1 / 17.8 / 16.2 | 277.6 / 351.8 / 270.5 |
| HMC-IN seq α=1e-3 | 13.4 / 6.5 / 15.2 | 13.4 / 16.3 / 15.2 | 276.0 / 16.3 / 271.1 |
| threshold 35 °C | 5.3 / 6.8 / 13.1 | 5.3 / 16.4 / 13.1 | 5.3 / 16.4 / 13.1 |
| threshold 40 °C | 3.5 / 3.8 / 8.1 | 3.5 / 15.8 / 10.6 | 3.5 / 15.8 / 10.6 |
| threshold 50 °C | 1.5 / 1.3 / 1.6 | 1.5 / 1.3 / 1.6 | 1.5 / 1.3 / 1.6 |
| clean envelope ±2 °C | 10.6 / 6.0 / 15.2 | 10.6 / 16.2 / 15.2 | 273.2 / 16.2 / 270.4 |
| Hampel t=4 | −0.0 / −1.2 / −4.0 | 23.4 / 23.8 / 23.9 | 370.8 / 325.8 / 372.5 |
| robust z 24 h t=4 | 14.2 / 2.1 / 16.2 | 14.2 / 17.2 / 16.2 | 378.0 / 351.8 / 374.0 |

The HMC-IN and baseline rows are those of the first run, which the rerun
reproduced. The figures are `figures/detect_overview.png` (the whole
window), `figures/detect_zoom.png` (36 h before to 6 h after the failure),
`figures/detect_pvalues.png` and `figures/detect_false_alarms.png`.

* **The model flags come early.** The sequential PMC and the HMC-IN flag
  every suspect reading. On motes 48 and 47 the alarm that reaches the
  failure starts 6–16 h before the threshold hit: 6.1–8.8 and 16.2 h for
  the PMC, 13.4 and 15.2 h for the HMC-IN. That is 1.7–3.8 times the
  warning of a 40 °C threshold. Both behave like an adaptive level
  threshold: the climb's 30 s increments are ordinary, and what gives it
  away is the level relative to the clean margins.
  - **PMC, mote 48.** Its alarm starts where the level leaves the bulk of
    the clean margins: at 31.7 °C (α = 1e-3) and 33.4 °C (BH), where the
    top state's mean plus 4 sd is 30.1 °C. Above F⁻¹(1 − EPS) (38.4 °C) the
    margins put no mass at all. The climb rows it leaves unflagged lie at
    28.0–34.7 °C.
  - **PMC, mote 47.** Its alarm starts at 28.7 °C, at 05:15 on 03-24, in
    the ambiguous zone just before the climb onset. That is below the top
    state's mean plus 4 sd (35.1 °C): there the innovation test fires, not
    the level.
  - **Mote 22 is where the PMC is late (1.5–2.1 h).** Its clean-window PMC
    has a state with sd 5.17 °C, whose F⁻¹(1 − EPS) is 64.7 °C. The climb
    from 34.8 to about 50 °C stays inside that margin (430–474 climb rows
    unflagged), and the alarm that reaches the failure starts at 48–50 °C.
* **The HMC-IN pays for its lead in false alarms.** Its lead is the longest
  on mote 48 (13.4 h). Its tail threshold (about 27–30 °C on motes 48 and
  47) sits inside the range of the warm afternoons of the next two weeks:
  3 episodes per mote, 900–1 300 flags. The clean envelope ± 2 °C is the
  same test without a model, with nearly the same numbers.
* **Examined false alarms of the PMC** (sequential, α = 1e-3;
  `results/alarm_episodes.csv`, every episode listed in
  `results/tables.md`; `figures/detect_false_alarms.png`):
  - **Mote 48.** 8 episodes, 605 flags. Two of them hold 599 flags: 03-13
    from 12:56 (4.6 h, 331 flags) and 03-20 from 13:58 (2.5 h, 268 flags),
    smooth afternoon peaks of 26.2–29.9 and 27.1–29.3 °C, up to 2 °C above
    the fit week's maximum of 27.9 °C. Without gating the same peaks give 4
    and 2 flags: this is lock-out, and it holds at G = 256 (below). The 6
    other episodes are single flags.
  - **Mote 22.** 9 episodes, 106 flags. The two largest start at 07:01 on
    03-09 and 03-10 (47 and 38 flags): the reading jumps from about 22 °C
    to 28 °C within minutes and to 30 °C within half an hour. A third
    starts at 07:01 on 03-22 (4 flags). The others hold 1–4 flags.
  - **Mote 47.** 7 episodes, 49 flags. Four start between 07:01 and 07:12
    (03-17, 03-18, 03-19 and 03-23; 2–19 flags). The two largest (18 and
    19 flags) sit on a 0.3 °C dip in the night cooling at 07:01, the same
    minute on both days. Two fall on the 03-13 afternoon at 30.7–32.2 °C
    (1 and 2 flags), one on 03-19 at 21:18 (1 flag).
  - **Summary.** No PMC false alarm in normal operation looks like a sensor
    glitch. They are afternoon peaks warmer than the fit week, and morning
    events at 07:01 that recur on several days, probably a building
    schedule. With BH at α = 1e-3 or with α = 1e-4, 2–3 flags remain, in
    one episode on mote 47.
* **Flags before the threshold are mostly true early drift.**
  - Between the climb onset and the first suspect reading, the sequential
    PMC (α = 1e-3) flags 57 % (mote 48) and 89 % (mote 47) of the rows.
    The rows it leaves are the lower part of the climb: 28.0–34.7 °C on
    mote 48, and on mote 47 the first 1.4 h of the climb (30.0–33.4 °C).
    The HMC-IN flags 100 % on both motes.
  - In the ambiguous zone, the sequential PMC flags 0, 9 and 29 rows.
    - **Mote 47.** The 29 flags run from 05:15 on 03-24 to the climb onset
      at 06:37 (28.0–31.3 °C), continuous with the climb: probably early
      drift.
    - **Mote 22.** The 9 flags fall between 05:04 and 13:55 on 03-23
      (29.1–35.8 °C), around a 35–40 °C bump at 06:00, then 33–35 °C. The
      bump looks like the legitimate morning jumps of this mote (35 °C on
      03-08, and the 07:01 jumps above) and cannot be settled from this
      mote alone.
* **The simple baselines split in two.**
  - **Fixed thresholds.** No false alarm at 35–50 °C on these motes. The
    lead is 1.3–1.6 h at 50 °C and 5–13 h at 35 °C, but 35 °C is only
    1 °C above mote 22's fit-window maximum (34.2 °C). A rule that works
    on these three motes need not transfer.
  - **Rolling filters.** They are the wrong tool for this failure. The
    Hampel filter looks for spikes: recall 0.00, and 15–25 episodes of
    single-row false alarms on 30 s noise. The trailing 24 h robust z-score
    sees the climb (lead 14–16 h on motes 48 and 47). Then its window fills
    with failing readings and it stops flagging, which gives a recall of
    0.05–0.13 on the suspect rows. It also flags every warm afternoon
    (4–5 episodes, 587–1 412 flags).
* **Non-sequential PMC: recall 0.26 and 0.02 on motes 48 and 47.**
  - Every suspect row that directly follows another observed suspect row
    is missed (1 146 and 1 064 rows, median p 0.94 and 0.96). This is the
    clipped-corner problem below.
  - Of the 1 207 suspect rows that follow a missing epoch on each mote,
    621 and 51 are flagged (median p 8.6e-16 and 0.82). The first run
    flagged all of them. These flags depend on the quadrature: at G = 256
    even fewer are flagged (below).
  - On mote 22, whose top-state copula is a Clayton, every suspect row gets
    p ~ 1e-13 and is flagged: recall 1.00.
  - The sequential run is not affected: recall 1.00 on every mote.
* **Sequential vs non-sequential.**
  - **During a persistent failure.** Gating is what keeps a stuck sensor
    flagged: recall 1.00 against 0.02–0.26.
  - **During legitimate transients.** Gating turns a single flag into a
    run: 605 against 12 normal-operation flags on mote 48, 106 against 9
    on mote 22, 49 against 8 on mote 47. The simulation study measured a
    gating cost of 0.3 extra flags per 1 000 rows on its fixtures. On this
    near-unit-root series it reaches 20 per 1 000 normal rows (mote 48).
  - **For the HMC-IN it barely matters.** No y-dependence: 3 846 against
    3 826 flags on mote 48.
  - **Corrections.** BH (or a smaller per-row α) is the practical fix for
    the lock-out on this data. Per-row α = 1e-2 makes it worse: 2 351,
    1 173 and 943 normal-operation flags.

#### The check at 256 nodes

`detect.py --check` reruns the PMC at G = 64 and at G = 256 on the rows
from epoch 28 801 to 6 h after the first reading above 60 °C (epochs
75 660, 72 466 and 75 174 for motes 48, 22 and 47). That covers normal
operation, the ambiguous zone, the climb and the start of the failure. The
settings are sequential α = 1e-3, sequential BH α = 1e-3 and every
non-sequential one, each run at both node counts on that window.

* **The G = 64 flags equal the main run's.** Cut to the window, they are
  the same rows for every setting without BH on every mote
  (`same_as_main_run`: the filter is causal). A BH threshold depends on the
  window, so the BH rows below differ slightly from the main table.

Values for motes 48 / 22 / 47, on the check window
(`results/detect_check.csv`):

| PMC setting | G | rows differing from G = 64 | recall (suspect) | climb rows flagged | normal-operation flags (episodes) | lead, episode (h) | lead, first flag of the last 24 h (h) |
|---|---|---|---|---|---|---|---|
| seq α=1e-3 | 64 | – | 1.00 / 1.00 / 1.00 | 104 / 193 / 358 | 605 (8) / 106 (9) / 49 (7) | 8.8 / 2.1 / 16.2 | 11.1 / 17.8 / 16.2 |
| seq α=1e-3 | 256 | 26 / 1 / 0 | 1.00 / 1.00 / 1.00 | 114 / 193 / 358 | 621 (8) / 105 (9) / 49 (7) | 11.1 / 2.1 / 16.2 | 11.1 / 17.8 / 16.2 |
| seq BH α=1e-3 | 64 | – | 1.00 / 1.00 / 1.00 | 86 / 127 / 273 | 0 / 0 / 0 | 6.1 / 1.2 / 15.2 | 10.6 / 1.2 / 15.2 |
| seq BH α=1e-3 | 256 | 1 / 0 / 81 | 1.00 / 1.00 / 1.00 | 87 / 127 / 211 | 0 / 0 / 0 | 6.1 / 1.2 / 3.2 | 10.6 / 1.2 / 15.2 |
| nonseq α=1e-3 | 64 | – | 0.30 / 1.00 / 0.36 | 37 / 188 / 114 | 12 (8) / 9 (9) / 8 (7) | 8.8 / 2.1 / 3.5 | 11.1 / 17.8 / 16.2 |
| nonseq α=1e-3 | 256 | 35 / 0 / 97 | 0.07 / 1.00 / 0.00 | 31 / 188 / 67 | 12 (8) / 9 (9) / 8 (7) | 1.0 / 2.1 / none | 11.1 / 17.8 / 16.2 |
| nonseq BH α=1e-3 | 64 | – | 0.30 / 1.00 / 0.36 | 24 / 117 / 87 | 0 / 0 / 0 | 6.1 / 1.2 / 3.5 | 8.8 / 1.2 / 15.2 |
| nonseq BH α=1e-3 | 256 | 34 / 0 / 94 | 0.07 / 1.00 / 0.00 | 17 / 117 / 42 | 0 / 0 / 0 | 1.0 / 1.2 / none | 8.8 / 1.2 / 15.2 |

The other non-sequential settings (α = 1e-2, 1e-4, BH 1e-2) differ in
34–36 rows on mote 48, 94–102 on mote 47 and 0–3 on mote 22. The
quadrature diagnostic of the non-sequential pass over the window's gaps
is in the thousands at both G, against a threshold of 0.05: `quad_error`
6.2e3 / 3.8e3 / 6.3e3 at G = 64 and 3.2e3 / 1.8e3 / 3.4e3 at G = 256.

* **What does not change.**
  - The sequential recall: 1.00 at both G on every mote and setting.
  - The normal-operation false alarms: none under BH at either G, and at
    α = 1e-3 the same episodes (8, 9 and 7) with 605 → 621, 106 → 105 and
    49 → 49 flags. The mote-48 lock-out is genuine.
  - The first flag of the last 24 h, for every setting and mote.
  - Mote 22, every setting: 0–3 rows differ.
* **What changes.**
  - **Mote 47, sequential BH:** 81 rows, mostly in the climb (273 → 211 of
    401 climb rows flagged). The alarm then breaks up, and the episode that
    reaches the failure starts 3.2 h before it instead of 15.2 h. Its first
    flag of the last 24 h is still 15.2 h ahead.
  - **Mote 48, sequential α = 1e-3:** 26 rows: 10 more climb rows and 16
    more normal-operation flags. The episode that reaches the failure starts
    11.1 h ahead instead of 8.8 h.
  - **Non-sequential, motes 48 and 47:** 34–102 rows, on the failure. The
    share of the window's suspect rows flagged falls from 0.30 to 0.07 and
    from 0.36 to 0.00. No non-sequential alarm reaches the failure more than
    1.0 h ahead at G = 256 on mote 48, and none reaches it on mote 47.
* **Why: the passes that condition on failing readings.** Above 38.4, 64.7
  and 45.0 °C (motes 48, 22, 47) a reading lies beyond F⁻¹(1 − EPS) of
  every clean state, where the copula arguments are clipped (library
  problem 1).
  - The non-sequential pass conditions on them. Its log-likelihood over
    the window moves by 1 312 and 3 956 nats between G = 64 and 256 on
    motes 48 and 47, and by 2.4 nats on mote 22.
  - The gated passes integrate the flagged readings out. Where they flag
    the same rows (or all but one), their log-likelihoods at G = 64 and 256
    agree within 0.1–2.4 nats, although they cross the same network
    outages.
* **Why `quad_error` is in the thousands.** A one-off breakdown by run of
  missing rows of the mote-48 non-sequential pass at G = 64
  (`gaps._quadrature_report(..., per_run=True)`, 6 247 in all):
  - 5 015 on two network outages of normal operation, 2 944 and 2 341
    missing rows between ordinary readings (from epochs 32 657 and
    47 871): about 1 per missing row, the per-block cap. The gated passes
    cross the same outages and agree within 0.1–2.4 nats, so there the
    report is not an estimate of nats (library problem 4).
  - 980 on the 123 runs next to a reading above 40 °C, again about 1 per
    missing row. There the log-likelihood does move by thousands of nats.
  - 252 on the other 3 723 runs.
* **Caveat.** At the default G = 64, the PMC detection results on the
  failure depend on the quadrature wherever a pass conditions on failing
  readings: the non-sequential recall, climb coverage and lead on motes 48
  and 47, and through gating the BH lead on mote 47 and the α = 1e-3 lead
  on mote 48. The sequential recall, the false alarms in normal operation
  and the first-flag leads do not. G = 256 has not converged either, so
  neither node count gives the reference numbers on the failure. The model
  is also far outside its validity there: the plateau and most of the rise
  above 60 °C lie more than 8 sd above every clean margin.

### 3. Robust re-estimation on a window that mixes clean and failing readings

| mote | window (epochs) | observed | normal | ambiguous 24 h | climb | suspect | suspect + climb |
|---|---|---|---|---|---|---|---|
| 48 | 63 420–77 820 | 6 126 | 4 879 | 850 | 182 | 211 | 6.4 % |
| 22 | 60 226–74 626 | 8 946 | 5 451 | 1 786 | 623 | 1 079 | 19.0 % |
| 47 | 62 934–77 334 | 6 364 | 4 771 | 925 | 401 | 267 | 10.5 % |

The HMC-IN fits below are exact, and those of the first run. The PMC fits
and their flags use the default `gap_nodes = 64`; the quadrature on these
windows is discussed after the tables.

**HMC-IN** (K = 3). "Masked" shows the rows masked, then in brackets the
share of the suspect rows and of the climb rows among them. The failure
state is a state whose mean exceeds the fit-window maximum by more than
2 °C.

| mote | fit | fits | masked (suspect / climb) | means (°C) | sd (°C) | failure state: mean, sd, stay | agreement with oracle_ext |
|---|---|---|---|---|---|---|---|
| 48 | raw | 1 | 0 | 19.61 / 22.74 / 68.99 | 1.09 / 1.22 / 36.98 | 69, 37, 1.000 | 0.88 |
| 48 | oracle | 1 | 211 (1.00 / 0.00) | 19.59 / 22.09 / 27.85 | 1.08 / 0.56 / 7.73 | – | 0.99 |
| 48 | oracle_ext | 1 | 393 (1.00 / 1.00) | 19.60 / 22.03 / 24.68 | 1.08 / 0.50 / 2.71 | – | 1.00 |
| 48 | fm | 1, converged | 0 | = raw | | 69, 37, 1.000 | 0.88 |
| 48 | fm_hampel | 2, converged | 0 | = raw | | 69, 37, 1.000 | 0.88 |
| 48 | fm_thr60 | 8, converged | 416 (1.00 / 1.00) | 18.81 / 20.66 / 22.77 | 0.76 / 0.20 / 1.25 | – | 0.68 |
| 48 | fm_cs | 1, converged | 0 | 18.15 / 21.50 / 68.42 | 0.33 / 1.58 / 37.07 | 68, 37, 1.000 | 0.50 |
| 22 | raw | 1 | 0 | 20.96 / 38.02 / 119.65 | 2.13 / 14.78 / 3.14 | 120, 3, 1.000 | 0.71 |
| 22 | oracle | 1 | 1079 (1.00 / 0.00) | 20.84 / 31.16 / 47.15 | 2.04 / 3.71 / 6.41 | 47, 6, 0.999 | 0.73 |
| 22 | oracle_ext | 1 | 1702 (1.00 / 1.00) | 19.51 / 24.92 / 34.18 | 1.08 / 2.35 / 2.47 | – | 1.00 |
| 22 | fm | 11, cycling | 0 | = raw | | 120, 3, 1.000 | 0.71 |
| 22 | fm_hampel | 11, cycling | 0 | = raw | | 120, 3, 1.000 | 0.71 |
| 22 | fm_thr60 | 11, cycling | 0 | = raw | | 120, 3, 1.000 | 0.71 |
| 22 | fm_cs | 1, converged | 0 | 20.42 / 30.45 / 94.36 | 1.73 / 4.58 / 29.91 | 94, 30, 0.998 | 0.80 |
| 47 | raw | 1 | 0 | 18.87 / 22.87 / 59.22 | 0.84 / 1.70 / 36.78 | 59, 37, 1.000 | 0.77 |
| 47 | oracle | 1 | 267 (1.00 / 0.00) | 18.89 / 22.26 / 30.15 | 0.85 / 1.00 / 6.14 | – | 0.86 |
| 47 | oracle_ext | 1 | 668 (1.00 / 1.00) | 18.91 / 21.78 / 25.03 | 0.86 / 0.59 / 1.74 | – | 1.00 |
| 47 | fm | 1, converged | 0 | = raw | | 59, 37, 1.000 | 0.77 |
| 47 | fm_hampel | 2, converged | 0 | = raw | | 59, 37, 1.000 | 0.77 |
| 47 | fm_thr60 | 4, converged | 299 (1.00 / 0.08) | 18.90 / 22.26 / 29.45 | 0.85 / 0.99 / 4.78 | – | 0.87 |
| 47 | fm_cs | 1, converged | 0 | 18.07 / 21.98 / 54.71 | 0.27 / 1.88 / 35.88 | 55, 36, 1.000 | 0.58 |

**PMC** (K = 3; "empty" = a state emptied by ICE, π < 0.005, which the
library reports as degenerate; "< 0.01": the stuck 122.15 °C value, whose
spread is the dequantisation jitter). The raw PMC fits of motes 22 and 47
have two failure states; both are shown.

| mote | fit | fits | masked (suspect / climb) | means (°C) | sd (°C) | failure state: mean, sd, stay | agreement with oracle_ext |
|---|---|---|---|---|---|---|---|
| 48 | raw | 1 | 0 | 20.56 / 21.60 / 41.04 | 2.09 / 1.07 / 33.09 | 41, 33, 0.998 | 0.81 |
| 48 | oracle | 1 | 211 (1.00 / 0.00) | 19.89 / 22.09 / 22.53 | 1.38 / 5.84 / 1.47 | – | 0.80 |
| 48 | oracle_ext | 1 | 393 (1.00 / 1.00) | empty / 20.81 / 21.65 | – / 2.37 / 1.33 | – | 1.00 |
| 48 | fm | 7, converged | 22 (0.01 / 0.01) | 19.55 / 21.33 / 43.32 | 0.83 / 1.99 / 33.98 | 43, 34, 0.9995 | 0.79 |
| 48 | fm_hampel | 11, not converged | 26 (0.01 / 0.01) | 19.11 / 21.69 / 38.42 | 0.92 / 1.75 / 31.48 | 38, 31, 0.999 | 0.56 |
| 48 | fm_thr60 | 11, not converged | 350 (1.00 / 0.66) | 19.93 / 20.14 / 21.66 | 2.01 / 1.31 / 2.01 | – | 0.76 |
| 48 | fm_cs | 11, not converged | 2 (0.01 / 0.00) | 20.77 / 20.86 / 43.08 | 2.01 / 0.59 / 33.91 | 43, 34, 0.999 | 0.95 |
| 22 | raw | 1 | 0 | 24.33 / 51.83 / 122.15 | 5.74 / 34.67 / < 0.01 | 52, 35, 0.978; 122, < 0.01, 0.999 | 0.97 |
| 22 | oracle | 1 | 1079 (1.00 / 0.00) | 24.44 / 25.19 / 48.08 | 5.81 / 5.21 / 6.04 | 48, 6, 0.999 | 1.00 |
| 22 | oracle_ext | 1 | 1702 (1.00 / 1.00) | empty / 24.09 / 24.85 | – / 5.53 / 5.31 | – | 1.00 |
| 22 | fm | 11, cycling | 0 | = raw | | = raw | 0.97 |
| 22 | fm_hampel | 11, cycling | 13 (0.00 / 0.02) | 24.27 / 51.76 / 122.15 | 5.64 / 34.60 / < 0.01 | 52, 35, 0.981; 122, < 0.01, 0.998 | 0.97 |
| 22 | fm_thr60 | 11, cycling | 13 (0.00 / 0.02) | = fm_hampel | | = fm_hampel | 0.97 |
| 22 | fm_cs | 5, converged | 36 (0.02 / 0.01) | 24.41 / 24.99 / 89.85 | 5.76 / 5.09 / 32.20 | 90, 32, 0.9998 | 1.00 |
| 47 | raw | 1 | 0 | 21.51 / 38.56 / 122.15 | 2.64 / 26.38 / < 0.01 | 39, 26, 0.970; 122, < 0.01, 0.174 | 0.55 |
| 47 | oracle | 1 | 267 (1.00 / 0.00) | 21.43 / 21.56 / 33.08 | 2.68 / 1.04 / 7.31 | 33, 7, 0.989 | 0.74 |
| 47 | oracle_ext | 1 | 668 (1.00 / 1.00) | 21.18 / 21.25 / 27.19 | 2.18 / 0.23 / 1.17 | – | 1.00 |
| 47 | fm | 11, not converged | 11 (0.04 / 0.00) | 21.59 / 57.28 / empty | 2.60 / 37.24 / – | 57, 37, 0.996 | 0.56 |
| 47 | fm_hampel | 11, not converged | 4 (0.00 / 0.00) | 21.49 / 37.70 / 122.15 | 2.62 / 25.19 / < 0.01 | 38, 25, 0.966; 122, < 0.01, 0.191 | 0.55 |
| 47 | fm_thr60 | 11, not converged | 286 (1.00 / 0.02) | 21.43 / 21.53 / 32.76 | 2.65 / 1.01 / 6.87 | 33, 7, 0.993 | 0.71 |
| 47 | fm_cs | 3, converged | 31 (0.00 / 0.02) | 20.09 / 23.24 / 47.57 | 1.71 / 1.82 / 34.47 | 48, 34, 0.9995 | 0.46 |

The `raw_cs` rows (the raw fit from the clean model) and the quadrature
columns are in `results/tables.md`. The mask count of every refit and the
ICE iterations of every fit are in `results/robust_fits.csv`
(`masks_per_fit`, `iters_per_fit`). The states of the raw, `oracle_ext`,
`fm` and `fm_thr60` fits are drawn in `figures/robust_hmc_in.png` and
`figures/robust_pmc_state.png`.

* **The failing readings form their own state, as the simulation predicts.**
  Every raw fit has one:
  - HMC-IN: mean 59–120 °C, sd 3–37 °C, stay 0.9998–1.0000, an absorbing
    state;
  - PMC: a broad state of mean 39–52 °C, sd 26–35 °C, stay 0.970–0.998.
    On motes 22 and 47 the PMC spends a second state on the stuck
    122.15 °C plateau (π 0.085 and 0.014), so two of its three states
    describe the failure.

  Starting ICE from the clean-window model changes nothing (`raw_cs`,
  `fm_cs`: a failure state in 6 of 6, PMC means 43–90 °C). The breakdown is
  a property of the likelihood, not of the k-means start.
* **Under that state almost nothing is flagged**, and the default loop
  ends at or near the raw fit: 0 masked rows for the HMC-IN, 22, 0 and 11
  for the PMC, every fit with a failure state.
* **Flag-and-mask finds no fixed point in 9 of the 12 PMC loops** (3 of
  the 12 HMC-IN loops, all on mote 22). `robust_estimate` stops at
  `max_rounds` = 10 refits, and its result is the 11th fit, not a fixed
  point.
  - **Mote 22, `fm`, `fm_hampel` and `fm_thr60`: one deterministic cycle.**
    The raw fit flags 13 climb rows; with them masked the fit flags none,
    so the next fit is the raw fit again: 0 | 13 | 0 | 13 … `fm_hampel`
    joins the cycle at its 6th fit, and `fm_thr60` at its 10th, after
    releasing the pre-mask round by round (1 079, 951, 989, 1 027, 835,
    137, 154, 80, 46, 0). Every fit of the cycle runs to the ICE cap of 50
    iterations, and so does the raw fit: ICE itself does not converge on
    this window. The HMC-IN cycles the same way on this mote (0 | 33).
  - **Mote 47, `fm` and `fm_hampel`: a longer cycle.** After the first fit
    the mask moves between 3 and 15 rows (`fm`: 0, 4, 7, 11, 15, 6, 6, 12,
    4, 7, 11) while every ICE fit converges in 6–17 iterations. The counts repeat with
    period 7, and `fm_hampel` runs through the same counts from its 4th
    fit: probably one cycle.
  - **Motes 48 and 47, `fm_thr60`: a drifting mask.** It grows from the
    threshold's 211 and 267 rows and then wanders: 267–350 rows on mote 48
    (211, 267, 284, 294, 332, 292, 293, 283, 286, 324, 350), 281–333 on
    mote 47. One fit of each runs to the ICE cap.
  - **Mote 48, `fm_hampel` and `fm_cs`:** 12–31 and 0–46 rows; `fm_hampel`
    ends alternating between 26 and 17.
  - **Converged:** the PMC `fm` on mote 48 (7 fits, 22 rows masked) and
    `fm_cs` on motes 22 and 47 (5 and 3 fits, 36 and 31 rows). All three
    keep a failure state (43, 90 and 48 °C, sd 32–34 °C): they converge
    to the breakdown.

  The HMC-IN, which uses no quadrature, cycles too: non-convergence comes
  from flag-and-mask on a window where the failure is a state. The
  particular masks of the PMC may depend on the quadrature (below).
* **Hampel `initial_mask`.** It masks 31–130 rows, spikes in the 30 s noise
  and not the drift or the plateau, so the loop falls back into the failure
  basin (6 of 6). The first run's exception, the PMC on mote 48, is gone:
  it now ends with 26 rows masked and a 38 °C failure state. The
  simulation's remedy for frequent spikes does not transfer to a smooth
  persistent failure.
* **Threshold `initial_mask`.** The sensor-level flag is re-tested by the
  model. 4 of the 6 fits end with every suspect row masked:
  - **HMC-IN on mote 47.** 299 rows masked, converged, means 18.90 /
    22.26 / 29.45 against the oracle's 18.89 / 22.26 / 30.15.
  - **HMC-IN on mote 48.** 416 rows masked, every suspect and climb row,
    converged. The model extends the mask into the drift.
  - **PMC on mote 48.** 350 rows masked, every suspect row and 66 % of the
    climb, no failure state; not converged.
  - **PMC on mote 47.** Every suspect row masked, but the climb left in
    forms a 33 °C state (sd 6.9 °C), as in the `oracle` fit; not converged.

  On mote 22 (19 % failing) both models release the pre-mask round by
  round and end in the raw fit's cycle. The HMC-IN goes from 1 079 to 978,
  887, 796, 755, 567 and 0 masked rows: the climb, left in, forms a state
  at about 47 °C (the `oracle` fit's) that makes the lower suspect
  readings plausible, which widens the state, and so on.
* **The threshold oracle is not clean either.** Masking only the suspect
  rows leaves the climb in the fit. The HMC-IN's top state keeps an sd of
  7.7, 6.4 and 6.1 °C, against 2.7, 2.5 and 1.7 °C when the climb is masked
  too. The PMC's `oracle` fits keep a 5.8 °C-sd state on mote 48 and a
  48 °C and a 33 °C state on motes 22 and 47. The partial ground truth
  biases the estimate as much as a mild contamination would.
* **Classification of the clean rows.**
  - **HMC-IN.** The level regimes are identifiable. Against `oracle_ext`,
    the raw fits agree on 71–88 % of the normal rows and the threshold
    oracle on 73–99 %. The raw fit spends a state on the failure and merges
    two day/night levels into the remaining two.
  - **PMC.** The agreement cannot rank the fits. The `oracle_ext` fits of
    motes 48 and 22 empty a state, and on mote 22 one state has π = 0.83,
    so almost any fit agrees with it (0.94–1.00). On mote 47 the
    two oracles agree on 74 % of the normal rows. The parameters (the
    failure states, the top-state sd) rank the fits.

**Quadrature of the PMC fits.** Every ICE E-step of a PMC fit on these
windows logs the quadrature WARNING (11–562 WARNINGs per task), and
`quad_error` of the returned fits on their masked windows is 9.5–3 970
(`results/tables.md`). The values in the thousands go with long runs of
missing rows: the failure block that a mask removes, and the day after the
failure, where a failing mote reports on 7–38 % of the epochs. There the
report is about 1 per missing row whatever the error (library problem 4),
but the gaps among failing readings are also where the detection check
found the quadrature unconverged. The PMC fits were not rerun at 256
nodes: a flag-and-mask task already takes 11–105 min at G = 64. The
failure states of the raw fits are unlikely to depend on G (the failing
rows are 6–19 % of the window, tens of degrees above the clean margins);
the masks, cycles and non-convergence may.

**The 3 WARNINGs logged after the fits.** Once the tasks are done,
`robust.py` classifies the normal rows with the clean-window model on each
window with the suspect and climb rows set to NaN, for the
`agree_clean_model` column of `results/robust_fits.csv`. For the PMC that
pass runs outside the tasks, so its WARNINGs reach the run log:
`quad_error` 4.2e3, 2.3e3 and 4.5e3 on 8 668, 7 157 and 8 705 missing rows
(motes 48, 22 and 47; the counts identify the windows). A one-off
recomputation at G = 64, broken down by run, gives 4 170 and 4 548 for
motes 48 and 47:

* **Mote 47.** 4 533 on one trailing gap of 4 656 rows, the masked climb
  and failure to the end of the window. A trailing gap contributes exactly
  0 to the log-likelihood, and for state margins the state law across a
  gap is exact (`gaps` module docstring): nothing observed depends on it.
  The other 993 runs report 15.
* **Mote 48.** 4 140 on two runs: 1 464 rows from the climb onset to a
  59.55 °C reading, and a 2 875-row trailing gap after a 59.70 °C reading.
  These are failing readings just below 60 °C that the suspect flag misses
  (4 on mote 48, 7 on mote 22), left observed by the `oracle_ext` mask. The
  other 917 runs report 29.
* **Consequence.** The normal rows end 9–12 h before these gaps, with the
  850–925 readings of the ambiguous zone in between, so the
  `agree_clean_model` column of the PMC is hardly affected on motes 48 and
  47. Mote 22 was not recomputed; its window keeps 7 such readings.

### 4. Conclusion

**What works.**

* **Detection with a model fitted on clean data, held fixed, and gated
  (sequential).** It flags every failing reading, at 64 and 256 nodes. On
  motes 48 and 47 its alarm starts 6.1–8.8 and 16.2 h before the 60 °C
  rule, depending on the setting. Under BH the mote-47 start depends on
  the quadrature (3.2 h at G = 256); its first flag does not.
* **Its false alarms are real events outside the training week**, not
  sensor noise. With BH at α = 1e-3, they are 0, 0 and 2 flags (one
  episode) in 14–15 days, at G = 64 and 256.
* **The copula model beats the HMC-IN on false alarms by two orders of
  magnitude or more.** With BH at α = 1e-3 it raises 0 and 2 flags on
  motes 48 and 47, against 967 and 636 for the HMC-IN with BH. Its alarm
  starts 6.5 and 4.1 h later than the HMC-IN's on motes 48 and 22, and
  1.0 h earlier on mote 47.

**What fails, or needs care.**

* **Calibration.** The tests reject on 30 s data. The PIT is usable as a
  score, not as an exact p-value.
* **Lock-out of the gated filter** after a legitimate transient. It is
  genuine on mote 48's warm afternoons (605 against 12 flags at G = 64,
  621 at G = 256). It needs a correction (BH or α = 1e-4); a
  re-acquisition rule would be the principled fix.
* **The gap quadrature on the failure.** Where a pass conditions on
  readings far outside the clean margins, the PMC results move between
  G = 64 and 256 (the non-sequential flags of motes 48 and 47, the BH lead
  of mote 47), and neither is converged. `quad_error` cannot say by how
  much on these windows (library problem 4).
* **The PMC's lead depends on the width of its clean margins.** It is
  1.5–2.1 h on mote 22.
* **The non-gated copula PIT at the clipped corner** (library problem 1):
  non-sequential recall 0.26 and 0.02 on motes 48 and 47.
* **Rolling filters** (Hampel, trailing robust z) do not see a smooth
  drift or a stuck sensor.
* **Flag-and-mask estimation on a failing window.** It breaks down in
  every case without a sensor-level pre-mask (Hampel pre-screen
  included), and 9 of its 12 PMC loops find no fixed point. It only
  recovers with the 60 °C pre-mask, and only where at most about 10 % of
  the window is failing (motes 48 and 47). Even then the PMC does not
  converge, and on mote 47 it keeps the climb as a state.

**Is the contamination model needed?**

* **For detection, no.** A clean model plus gating already flags the
  failure; the missing piece is calibration, not a contamination law.
* **For estimation on data that contain a failure, yes.** Every fit gives
  the failure a regular state, and flag-and-mask cannot leave that basin.
  The data say which contamination model:
  - **The indicator must be Markov, not Bernoulli.** The failure lasts
    until the mote stops reporting (2 271–4 816 suspect readings per mote).
    The fitted stay probabilities of the broad failure states are ≥ 0.97,
    and 0.9998–1.0 for the HMC-IN (about 1 exit in 5 000 rows); in reality
    the state is absorbing. A
    Bernoulli indicator at the rate that explains 10–19 % of a window
    would put isolated contaminated rows everywhere and would not
    explain a block.
  - **The broad law g must be fixed** (e.g. uniform over the sensor
    range), not re-estimated. A re-estimated g becomes the 39–120 °C
    states seen here, with an sd that ICE fits to the plateau: 3–37 °C, and
    under 0.01 °C for the PMC states that hold the stuck value alone.
  - **The climb is the hard part.** The rows are continuous, smooth, and
    below 60 °C. At 30 s a clean copula state predicts each step within
    about 0.05 °C, a far higher density than any broad g. A contamination
    indicator would therefore take the climb only where its level leaves
    the clean margins: the information the HMC-IN, the envelope and the
    PMC's ceiling already use.

  A drift component (a contaminated state whose law follows the previous
  reading, i.e. a copula state with a fixed broad margin) or the known
  physics (a voltage covariate: the supply voltage is at its lowest,
  2.2–2.3 V, during the climb) would be needed to separate the early climb
  from a warm day.

## Library problems found

### 1. The non-gated PIT at the clipped corner

`predictive_pit` (and so `flag_outliers(sequential=False)`) clips the
copula arguments to [EPS, 1 − EPS], EPS = 2.2e-16, as `precompute_weights`
does. Take y_{n−1} and y_n both beyond F⁻¹(1 − EPS), about 8.1 sd above
a Gaussian margin, as with a stuck sensor. Then u and v are both clipped to
1 − EPS, and the PIT becomes h(1 − EPS | 1 − EPS) instead of about 1:

* ≈ 0.5 for a strongly dependent Gaussian or Gumbel–Hougaard copula,
  p ≈ 0.95;
* p ~ 1e-13 for Clayton and Frank.

The log predictive density of the same row is about −1 200, so the
likelihood rejects the reading that the PIT accepts. The module's claim
that "the PIT is the CDF of the law the likelihood uses" fails in that
corner: the clipped conditional CDF saturates below 1.

Reproduction (`repro_clipped_corner.py`, a PMC with K = 2, N(0, 1) and
N(1, 1) margins, τ = 0.99 on every pair):

```text
Y = [  0.    0.1   0.2  50.   50.   50.  100. ]
Gauss    non-gated p-value : [0.659 0.124 0.122 0.    0.949 0.949 0.949]
         log predictive    : [-1.000e+00 -1.700e+01 -1.700e+01 -1.285e+05 -1.164e+03 -1.164e+03 -4.864e+03]
         gated p-value     : [0.659 0.124 0.122 0.    0.    0.    0.   ]  flagged [0 0 0 1 1 1 1]
GH       non-gated p-value : [0.659 0.125 0.016 0.    0.993 0.993 0.993]
Clayton  non-gated p-value : [6.587e-01 1.245e-01 2.000e-01 0.000e+00 8.837e-14 8.837e-14 8.837e-14]
Frank    non-gated p-value : [6.587e-01 1.245e-01 1.589e-03 2.842e-14 1.137e-13 1.137e-13 1.137e-13]
```

* **Scope.** Only the non-gated filter conditions on a clipped y_{n−1}.
  The gated filter integrates a flagged row out. In the first run, the
  grid path after a missing row used quadrature nodes inside the margins.
  The local grids of the current quadrature are placed where the filter
  and the next reading put the missing value, so inside the failure they
  probably reach beyond F⁻¹(1 − EPS) as well.
* **Here.** It caps the non-sequential PMC's recall on motes 48 and 47 at
  0.26 and 0.02 (0.51 and 0.53 in the first run).
  - Every suspect row that follows an observed suspect row is missed:
    1 146 and 1 064 rows, median p 0.94 and 0.96.
  - Of the 1 207 suspect rows that follow a missing epoch on each mote,
    586 and 1 156 are missed now (none in the first run); the median p of
    those 1 207 rows is 8.6e-16 on mote 48 and 0.82 on mote 47. At G = 256
    fewer still are flagged (section 2, the check at 256 nodes), which
    fits the local grids reaching the clipped corner.
  - No sequential number depends on it.
* **Fix (not attempted, library code untouched).** Evaluate the
  conditional CDF with v unclipped, since h(1 | u) = 1 for every u, or
  take the upper tail from the margin's survival function when
  F(y_n) > 1 − EPS.

### 2. The gap quadrature does not converge for near-deterministic transitions

**Status: fixed** at 39f249f, refined since (P5–P7 and cause 4, pmcprg
d14e91d). The numbers of this subsection up to the fix are those of the
first run.

`pmcprg.pmc.gaps` integrates a missing y_{n−1} on a fixed grid of G nodes,
the quantiles of the margins' reference mixture (`reference_grid`). The
30 s temperatures have diagonal τ ≈ 0.97–0.997, so the one-step
conditional law has an sd of about 0.04 °C. At the default G = 64, the node
spacing is 0.3 °C between 20 and 30 °C on mote 48 (0.35–0.7 °C on motes 47
and 22). The quadrature cannot resolve a law that narrow:

* **Log-likelihood.** For the fitted PMC on mote 48's clean days 1–10
  (`predictive_pit(...).log_lik`), it is 51 798, 57 574, 63 096 and
  65 364 nats at G = 64, 128, 256 and 512. It has not converged at 512.
* **PIT after a gap.** The normal scores of the 933 held-out rows that
  follow a missing epoch have an sd of 1.53, 1.29, 1.04 and 0.80, with
  22, 7, 4 and 0 p-values below 1e-3. After an observed row the sd is
  0.70 at G = 64 and 0.78 at G = 256.
* **Gated flags.** Under gating every flagged row becomes a gap, and the
  error cascades. On mote 47's held-out days, the sequential count falls
  from 513 (G = 64) to 52 (G = 256).

Reproduction on simulated data (`repro_gap_nodes.py`): a PMC with K = 2,
N(0, 1) and N(1, 1) margins, Gaussian copulas; 3 000 rows with every 10th
removed.

```text
tau = 0.6: 299 rows after a gap
  G =  64: log-lik    -2541.2   sd of z after a gap 0.99 (after an observed row 0.99)   p < 1e-3 after a gap: 1
  G = 512: log-lik    -2541.2   sd of z after a gap 0.99 (after an observed row 0.99)   p < 1e-3 after a gap: 1
tau = 0.99: 299 rows after a gap
  G =  64: log-lik     6341.8   sd of z after a gap 1.87 (after an observed row 0.99)   p < 1e-3 after a gap: 17
  G = 128: log-lik     6887.0   sd of z after a gap 1.18 (after an observed row 0.99)   p < 1e-3 after a gap: 0
  G = 256: log-lik     7020.8   sd of z after a gap 0.96 (after an observed row 0.99)   p < 1e-3 after a gap: 0
  G = 512: log-lik     7023.5   sd of z after a gap 0.96 (after an observed row 0.99)   p < 1e-3 after a gap: 0
```

At τ = 0.6 (the simulation fixtures) G is irrelevant. At τ = 0.99, G = 64
loses 682 nats over 299 single-row gaps. It also flags 17 rows after a gap
at α = 1e-3, against 0.3 expected.

* **Scope.** The problem is not specific to outliers. It concerns every
  grid-variant computation with missing rows: `gap_posterior`, `classify`,
  `impute`, `forecast`, ICE and SEM with `missing_strategy = "available"`,
  `predictive_pit` and `flag_outliers`. It grows with the dependence and
  with the ratio of the margin's spread to the conditional spread.
  - **Affected here.** Every PMC fit (the E-step at G = 64 with 12 % single
    gaps), the PMC log-likelihoods and BIC of step 1, the PMC PIT after a
    gap, and every gated PMC flag.
  - **Not affected.** The HMC-IN, which uses the exact K-state message.
* **Fix (not attempted, library code untouched).** The fix belongs to the
  grid. Options:
  - nodes adapted to the conditional law (centred on the last observed
    value, with the conditional scale of the diagonal copulas);
  - an exact one-dimensional integral for single-row gaps;
  - at least, a warning when the conditional spread at the nodes is below
    the node spacing, and a documented range of validity of the default G.

**Fixed at 39f249f** (local grids per missing position, and the
`GapPosterior.quad_error` WARNING; see the CHANGELOG). The same script now
prints:

```text
tau = 0.6: 299 rows after a gap
  G =  64: log-lik    -2541.2   sd of z after a gap 0.99 (after an observed row 0.99)   p < 1e-3 after a gap: 1
  G = 128: log-lik    -2541.2   sd of z after a gap 0.99 (after an observed row 0.99)   p < 1e-3 after a gap: 1
  G = 256: log-lik    -2541.2   sd of z after a gap 0.99 (after an observed row 0.99)   p < 1e-3 after a gap: 1
  G = 512: log-lik    -2541.2   sd of z after a gap 0.99 (after an observed row 0.99)   p < 1e-3 after a gap: 1
tau = 0.99: 299 rows after a gap
  G =  64: log-lik     7023.5   sd of z after a gap 0.96 (after an observed row 0.99)   p < 1e-3 after a gap: 0
  G = 128: log-lik     7023.5   sd of z after a gap 0.96 (after an observed row 0.99)   p < 1e-3 after a gap: 0
  G = 256: log-lik     7023.3   sd of z after a gap 0.96 (after an observed row 0.99)   p < 1e-3 after a gap: 0
  G = 512: log-lik     7023.5   sd of z after a gap 0.96 (after an observed row 0.99)   p < 1e-3 after a gap: 0
```

A one-off check with more digits, at G = 32, 64, 128, 256, 512 and 1024:

* **Log-likelihood at τ = 0.99.**
  - At G ≥ 64 it is within 0.008 nats of 7023.515 (the value at
    G = 512), except at G = 256: −0.26 nats.
  - A WARNING is logged at G = 32–256 (`quad_error` 2.1, 0.29, 0.10 and
    0.073, against a threshold of 0.017), and none at G ≥ 512.
  - The residual is therefore flagged, but it is not monotone in G.
* **Flags do not depend on G.**
  - `flag_outliers` at α = 1e-3 flags the same 4 rows (sequential) and 3
    rows (non-sequential) at every G.
  - Before the fix, 17 rows after a gap had p < 1e-3 at G = 64.
* **τ = 0.6.** Nothing moves: 1e-6 nats between G = 64 and 1024.

The outputs above are those of 39f249f; `repro_gap_nodes.py` was not rerun
on d14e91d. On this study's data, d14e91d gives at G = 64 a log-likelihood
of mote 48's clean days 1–10 within 0.006 nats of G = 256 (the refitted
model; section 1), and every PMC fit, PIT and flag of the study was rerun
on it.

### 3. A leading gap under strong dependence is misintegrated, with no WARNING

**Status: fixed** (P5, CHANGELOG `[Unreleased]`). Measured there: the
reproduction's leading gaps (L = 2, 5 and 17) within 2.5e-4 nats of the
gap-free pass at G = 64 and 4.8e-8 at G = 256, the state filter equal to π
to 4e-15, and the batch pass and the PIT filter within 2.3e-13 nats of
each other. In the rerun of step 1 the two passes give the same
log-likelihood on days 1–10 (section 1). The outputs below are those of
39f249f; `repro_leading_gap.py` was not rerun on d14e91d.

Found while rerunning step 1 on 39f249f.

**The exact reference.** Take a stationary PMC (symmetric p) whose first L
rows are missing. The rows after the gap have the law of the series started
at row L + 1. So the exact log p(y_obs) is the gap-free forward pass on
Y[L:], and P(x_n | nothing observed) = π at every missing row n < L.

Reproduction (`repro_leading_gap.py`, the PMC of `repro_gap_nodes.py`,
N = 400, the first 17 rows removed):

```text
tau = 0.997: rows 0-16 missing, exact log p(y_obs) = 1470.175, WARNING above quad_error = 1e-03
  G =   64: gap_posterior   -4.324 nats, max |alpha_hat - pi| 0.171, quad_error 3e-05   predictive_pit   -4.324 nats, max |alpha_hat - pi| 3e-16
  G =  256: gap_posterior   -0.130 nats, max |alpha_hat - pi| 0.000, quad_error 1e-08   predictive_pit   -0.130 nats, max |alpha_hat - pi| 2e-16
  G = 1024: gap_posterior   -0.000 nats, max |alpha_hat - pi| 0.000, quad_error 4e-14   predictive_pit   -0.000 nats, max |alpha_hat - pi| 6e-16
tau = 0.999: rows 0-16 missing, exact log p(y_obs) = 1889.843, WARNING above quad_error = 1e-03
  G =   64: gap_posterior  -22.096 nats, max |alpha_hat - pi| 0.290, quad_error 3e-05   predictive_pit  -23.019 nats, max |alpha_hat - pi| 2e-15
  G =  256: gap_posterior   -1.654 nats, max |alpha_hat - pi| 0.063, quad_error 1e-08   predictive_pit   -0.921 nats, max |alpha_hat - pi| 2e-16
  G = 1024: gap_posterior   +0.085 nats, max |alpha_hat - pi| 0.000, quad_error 4e-14   predictive_pit   +0.085 nats, max |alpha_hat - pi| 8e-16
```

* **Both passes are off.** The batch pass (`gap_posterior`, hence
  `classify` and the ICE / SEM E-steps) and the PIT filter
  (`predictive_pit`, `flag_outliers`) are both wrong.
  - The error is 4–23 nats at the default G, and still 0.1–1.7 nats at
    256.
  - It grows with τ and with the length of the gap. A scan of the same
    series at G = 64 gives:
    - at τ = 0.99: −0.04, −0.03 and −0.10 nats for L = 2, 5 and 17;
    - at τ = 0.995 and L = 17: −0.62 nats;
    - at τ = 0.9: nothing (below 1e-3 nats).
* **The diagnostic is silent.** `quad_error` is 3e-5 at G = 64, far below
  its threshold. `_quadrature_report` leaves out the rows inside a leading
  gap, on the ground that "the renormalisation conserves that mass wherever
  it relocates it".
  - The renormalisation does conserve the state masses in log space: the
    PIT filter keeps π to 1e-15.
  - Probably, it does not conserve the law of y on the nodes. The prior is
    broad, while the local grids of the gap are narrow, pinned by the first
    reading after it. The raw block masses inside the gap are far from
    their targets (up to 92 against at most 1 on mote 48 at G = 64), and
    the renormalisation moves that mass onto nodes where the stationary
    law does not put it.
* **The two passes disagree.** The batch pass also drifts from π inside
  the gap, by up to 0.29.
  - **Mote 48.** In the 17-row gap of the clean window (old model, G = 64)
    it gives P(x_8) = (0.46, 0.14, 0.39) against π = (0.52, 0.09, 0.39).
  - **Cause.** The same chain run in log space keeps π. In linear space,
    18 blocks into rows 8, 15 and 16 have a kernel that underflows to
    zero, so they cannot be renormalised and lose their mass. The drift
    starts at row 8.
  - **Consequence.** The batch and filter log-likelihoods differ, although
    `outliers.py` and the CHANGELOG state that they are equal:
    - in the reproduction, errors of −22.10 and −23.02 nats at τ = 0.999
      and G = 64, and −1.65 and −0.92 nats at G = 256;
    - on mote 48's days 1–10 at G = 64, 0.36 nats. The grids of the two
      passes are identical there (checked on the first 3 000 rows), and
      the passes agree to 4e-12 at G = 256.
  - **Interior runs too.** The two passes also differ after interior runs
    of 2–3 missing rows: state probabilities up to 1e-2 apart on mote 48.
    There the diagnostic does warn, at G = 64 on this data.
* **Exposure of this study: small.**
  - Only the clean windows start with a gap: 17, 2 and 17 rows on motes
    48, 22 and 47. The detection and robust-estimation windows start with
    a reading.
  - The refitted models of step 1 on 39f249f (stationary p) were checked
    by dropping the leading gap, which leaves log p(y_obs) unchanged for a
    stationary model. At G = 64 it changes:
    - the days 1–7 log-likelihood of the batch pass by 0.06, 1.5 and
      0.005 nats (motes 48, 22 and 47);
    - that of the PIT filter, on the first 200 readings, by 0.06, 0.015
      and 0.0005 nats.

    At G = 256 it changes nothing (≤ 1e-4 nats).
  - The PIT of the first reading moves by 4e-4 at most.
  - Inside the gap, the batch state probabilities are off by up to 0.27
    (mote 47).
* **Fix (not attempted, library code untouched).**
  - Inside a leading gap, keep the reference (stationary) grid, which
    carries the prior, up to the positions where the backward piece is
    narrower than the prior.
  - Or give the first reading its exact stationary law when the model is
    stationary.
  - Count the leading-gap rows in `_quadrature_report` (at least the exit
    into the first reading).
  - Make the batch pass renormalise zero-mass blocks in log space, as the
    filter does.

### 4. `quad_error` saturates on long runs of missing rows

**Status: open.** Found in the rerun, from the WARNINGs of the robust
tables step and of the detection check.

`quad_error` sums, over the missing positions of each run, a weighted mean
of relative errors capped at 1 per block (`gaps._quadrature_report`). On a
long run where the checks of every position fail, the report grows with
the length of the run, by about 1 per missing row, and no longer estimates
nats.

* **A trailing gap.** On mote 47's robust window, with the suspect and
  climb rows set to NaN, the clean PMC reports 4 548 at G = 64, of which
  4 533 on one trailing gap of 4 656 rows (0.97 per row). A trailing gap
  contributes exactly 0 to the log-likelihood, and for state margins the
  state law across a gap is exact (`gaps` module docstring): no result that
  involves an observed row depends on it.
* **Network outages.** On mote 48's detection-check window, 5 015 of the
  6 247 of the non-sequential pass come from two outages of 2 944 and 2 341
  rows between ordinary readings (0.95 and 0.94 per row). The gated passes
  cross the same outages, and where they flag the same rows their
  log-likelihoods at G = 64 and 256 agree within 0.1–2.4 nats.
* **Shorter runs.** On the two robust windows, runs of 11–100 rows report
  0.006–0.011 per missing row, runs of 2–10 rows under 5e-4, single rows
  under 5e-5.
* **Why it matters.** The WARNING then fires on any window with long
  outages, at every G (it only halves from G = 64 to 256 on the check
  windows), and it cannot tell the outages from the failure, where the
  log-likelihood of the non-sequential pass does move by thousands of
  nats. On the clean days, with runs of 1–17 rows, it overstates the error
  10–50 times but falls below its threshold at G = 256.
* **How it was measured.** One-off passes at G = 64 (`gaps._posterior`,
  then `gaps._quadrature_report(..., per_run=True)`), not scripted here:
  the clean PMC on the robust windows of motes 48 and 47 with the suspect
  and climb rows set to NaN (25 and 41 s; 1.3 and 2.4 GB of footprint),
  and on mote 48's detection-check window (61 s, 2.1 GB).
* **Fix (not attempted, library code untouched).** Leave trailing gaps out
  of the report (their parts are exact by construction). On long interior
  runs, weight the positions by what reaches an observed row (the exit
  integral), rather than summing capped errors over every position, or
  report the largest per-run part next to the sum.

## Limits

* **Three motes and one failure each.** The failures are the same battery
  failure mode, three times. There is no replication in the statistical
  sense; the lead times are three numbers per method.
* **Labels.** `suspect` is a partial truth. The `climb` label is a rule
  (the trailing 1 h median above the fit-week maximum) and the 24 h
  ambiguous zone is a convention. Other rules would move the precision
  against suspect ∪ climb and the lead to the climb, not the recall on
  `suspect`. The false alarms were examined visually, not against an
  independent reference. The lab-wide median of the other motes is itself
  contaminated at the end, because most motes fail within days of each
  other.
* **Models.** Gaussian margins; K chosen by BIC within {2, 3} on strongly
  dependent data; single best-of-3 fits in a multimodal likelihood; one
  model per mote held fixed for three weeks. The late-March afternoons
  are warmer than the fit week's, which drives most false alarms. A model
  refitted on a rolling window would have fewer.
* **Thresholds.** The baseline thresholds (35 / 40 / 50 °C, envelope
  ± 2 °C, Hampel and z at t = 3 / 4 / 5) were chosen with the raw series in
  view, but before scoring. The 35 °C rule's clean record is specific to
  these motes.
* **Temperature only.** Humidity and voltage, which carry the failure's
  physics, are not used.
* **Gap quadrature.** Every PMC computation used the default
  `gap_nodes = 64`. On the clean days it is converged in practice. On the
  failure windows it is not, at 64 or 256 nodes, wherever a pass
  conditions on failing readings; the detection check measures what moves
  there. The robust PMC fits were not checked at 256 nodes, and
  `quad_error` does not measure the error on long runs of missing rows
  (library problem 4).
* **Robust estimation.** One window per mote, K fixed at the clean
  selection (3). With K = 4 a failure state would not have to take a
  day/night level, which was not tried. 9 of the 12 PMC `robust_estimate`
  loops and 3 of the 12 HMC-IN loops stopped at `max_rounds` without a
  fixed point, so their masks are the 11th fit's. Several PMC ICE fits
  stopped at the cap of 50 iterations: the raw fit of mote 22 and every fit
  of its cycle among them.

## Files

| file | content |
|---|---|
| `il_common.py` | constants, data loader, dequantisation, labels, model helpers, baselines, scores |
| `fit_clean.py` | step 1 → `results/fits.csv`, `results/models/*.toml` (the 12 best clean fits), `results/selected_models.json`, `results/pit_checks.csv`, `results/regimes_by_hour.csv`, `figures/clean_pit.png`, `figures/clean_regimes.png` |
| `detect.py` | step 2 → `results/detect_scores.csv` (one row per mote × method × setting), `results/alarm_episodes.csv`, `results/detect_info.json`, `figures/detect_*.png`; `--check` → `results/detect_check.csv`, `results/detect_check_info.json` |
| `robust.py` | step 3 → `results/robust_fits.csv`, `results/robust_info.json`, `figures/robust_*.png`; the tasks in `results/cache/robust_tasks/` (for `--resume`, not versioned) |
| `summarise.py` | `results/tables.md`, `results/tables.tex` |
| `repro_clipped_corner.py` | library problem 1 (clipped corner) |
| `repro_gap_nodes.py` | library problem 2 (gap quadrature, fixed at 39f249f) |
| `repro_leading_gap.py` | library problem 3 (leading gap, fixed in P5) |
