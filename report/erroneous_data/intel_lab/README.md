# Erroneous data on real sensors: the Intel Berkeley Lab temperatures

> **Status (2026-09-26, pmcprg 5351de0 / 35e897d).** Every number below
> comes from the last rerun of the whole study, on the fix of library
> problem 4 (long runs of missing rows, and a `quad_error` in nats). Step 1
> ran on pmcprg 5351de0; step 2, the 256-node check and step 3 on 35e897d,
> which adds only a fix for an exactly singular system in the quadrature
> diagnostic (commits f92da76 and 700dcd4). Where a number moved since the
> previous rerun (pmcprg d14e91d / 5f23fc9, 2026-09-25), the text says so.
>
> * **Library fixes since the first run** (CHANGELOG `[Unreleased]`): local
>   grids for missing values under strong dependence (P1), the forecast and
>   impute quantiles (P3), the empty state (P2), leading gaps, a calibrated
>   `quad_error` diagnostic and the cost of the quadrature (P5–P7), local
>   grids placed from the filter (cause 4), bounded memory (5f23fc9), and
>   long runs of missing rows with `quad_error` in nats (5351de0). The
>   HMC-IN results, the baselines and the labels are reproduced bit for
>   bit.
> * **What the long-run fix changed here.** Almost nothing on the clean
>   days, and no lead time or sequential recall. `quad_error` no longer
>   counts long gaps: 4 533 → under 0.01 for a 4 656-row trailing gap,
>   5 015 → under 2.8 for two network outages (library problem 4, fixed). Where
>   it stays high, on the failure, the pass is not converged. At G = 64 the
>   non-sequential passes on the failure moved closer to G = 256 and the
>   gated ones further (74 / 17 / 7 rows differ at α = 1e-3 instead of
>   26 / 1 / 0). Three flag-and-mask loops end differently.
> * **Clean days: converged in practice.** At the default G = 64 the PMC
>   log-likelihood of days 1–10 is within 0.042 nats of G = 256, and the
>   flags are the same, although the quadrature WARNING fires.
> * **Failure windows: not converged, at G = 64 or 256.** Wherever a PMC
>   pass conditions on failing readings, its results move with G: the
>   non-sequential flags of motes 48 and 47, and through gating the BH lead
>   on mote 47 (section 2, "The check at 256 nodes"). The sequential recall
>   and the false-alarm episodes in normal operation do not move.
> * **Flag-and-mask finds no fixed point in 10 of the 12 PMC loops**
>   (section 3). The failure still becomes a state in every raw fit.
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
>   moves by 0.006–0.042 nats from G = 64 to 256, and the flags are the
>   same.
> * **Detection (step 2).** Not converged where a pass conditions on the
>   failing readings, far outside every clean margin. The non-sequential
>   PMC results of motes 48 and 47 and the sequential BH lead of mote 47
>   depend on G. The gated passes at G = 64 are also some rows and a few
>   nats off G = 256 in normal operation (mote 48: 65 more false flags at
>   G = 256, in the same episodes).
> * **Robust estimation (step 3).** Every PMC E-step warns, and the fits
>   were not checked at 256 nodes. The failure states are unlikely to
>   depend on G; the masks and cycles do (the long-run fix changed three
>   loops' endings).
> * **The WARNING itself.** `quad_error` now estimates nats and counts 0
>   for a trailing gap. On the clean days it overstates the change to
>   G = 256 by 10–17 times; on the failure it is capped at 4.6 nats per run
>   and understates the change (142 against 3 669 nats on mote 47).

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
    These leads did not move with the long-run fix.
  - **The mote-47 lead depends on the quadrature.** At G = 256 the alarm on
    the climb breaks up, and the episode that reaches the failure starts
    3.2 h before it, against 15.2 h at G = 64 on the same window; its first
    flag does not move. The other sequential leads move by at most 2.3 h
    (mote 48 at α = 1e-3: 8.8 → 11.1 h), and the false-alarm episodes not
    at all.
  - **HMC-IN.** It warns earlier on motes 48 and 22 (13.4 and 6.5 h; 15.2 h
    on mote 47) but raises about 900–1 300 flags in the normal operation of
    motes 48 and 47, on warm afternoons.
  - **Fixed thresholds.** A plain 35 °C threshold gives 5.3, 6.8 and 13.1 h
    of warning with no false alarm on these three motes.
  - **Rolling filters.** The Hampel filter and the rolling robust z-score do
    not detect this failure: their recall is 0.00–0.13. Both adapt to a smooth
    drift and to a stuck value.
* **The flags before the threshold are mostly true early drift.** On motes
  48 and 47, 58 % and 88 % of the out-of-range climb is flagged by the
  sequential PMC at α = 1e-3; the rest is its lower end. Mote 22 is the
  exception (31 %): its clean-window PMC has a wide margin (sd 5.17 °C)
  that accepts a climb to about 50 °C. The false alarms, examined on the
  plots, fall into two classes:
  - warm afternoon peaks above the fit-window maximum (mote 48, 26–30 °C,
    553 of its 559 flags; mote 47, 30.7–32.2 °C);
  - morning events at 07:01 that recur on several days: a jump from about
    22 to 28–30 °C on mote 22, a 0.3 °C dip on mote 47 (probably a
    building schedule).

  Both are real temperatures, not sensor errors.
* **Innovation gating has a real-data cost: lock-out.** A gated legitimate
  reading makes its successors look wrong, because the gated filter keeps
  predicting the old level while the true level moves on. On mote 48's warm
  afternoons of 03-13 and 03-20, 4 and 2 non-sequential flags become runs
  of 320 and 233 sequential ones (559 against 12 normal-operation flags;
  624 at G = 256). On the held-out clean days the lock-out
  is small (10, 3 and 30 sequential flags against 3, 2 and 2). The
  Benjamini–Hochberg correction or α = 1e-4 removes almost all of it.
* **Robust re-estimation breaks down, as the simulation predicts.** In every
  raw fit (6 of 6), the failing readings form a state of their own:
  mean 59–120 °C, sd 3–37 °C, stay ≥ 0.9998 for the HMC-IN; a broad state
  of 38–52 °C, sd 25–35 °C, stay 0.97–1.00 for the PMC, which on motes 22
  and 47 spends a second state on the stuck 122.15 °C value.
  - **Default flag-and-mask.** Under that state almost nothing is flagged
    (0–16 rows), and every result keeps a failure state.
  - **No fixed point in 10 of the 12 PMC loops** (9 in the previous rerun;
    3 of 12 for the HMC-IN, all on mote 22). On mote 22 three loops end in
    one deterministic cycle: the raw fit, then the raw fit with its 13
    flagged rows masked, every fit at the ICE cap of 50 iterations.
    On motes 48 and 47 the masks drift: 12–25 rows for the default loop of
    mote 48 (a fixed point of 22 rows in the previous rerun), 5–15 for
    that of mote 47, and 265–333 with the threshold pre-mask until mote
    48's loop drops it (below). The two PMC loops that
    converge (from the clean model, motes 22 and 47) keep a failure
    state.
  - **Hampel `initial_mask`.** It does not help, because it catches spikes,
    not drifts (6 of 6 break down).
  - **Other starts.** Starting from the clean-window model does not help
    either (6 of 6 keep a failure state).
  - **Threshold `initial_mask`.** A sensor-level pre-mask (the 60 °C
    threshold) gets 3 of 6 fits to mask every suspect row: the HMC-IN on
    motes 48 and 47, and the PMC on mote 47, which keeps the climb in a
    broad 27 °C state (sd 6.5 °C). The HMC-IN on mote 48 also masks the
    whole climb. The PMC on mote 48 now drops the pre-mask at its 9th fit
    and falls into the failure basin (in the previous rerun it
    masked every suspect row and 66 % of the climb). On mote 22 (19 % of
    the window failing) both models release the pre-mask round after round
    and end in the raw fit's cycle.
* **Even the threshold oracle is contaminated.** Masking only the suspect
  rows leaves the climb in: the top state keeps an sd of 6–8 °C, against
  1.7–2.7 °C once the climb is masked too (HMC-IN).
* **A contamination component is needed for estimation, not for detection.**
  The failure is persistent: the fitted stay probability of the broad
  failure states is ≥ 0.97, and ≥ 0.9998 for the HMC-IN. It is also broad:
  sd 25–37 °C in 5 of the 6 raw fits. The indicator must therefore be
  Markov, not Bernoulli, and its law must be fixed, not a regular state
  that ICE can reshape. Conclusion below.
* **Four library problems found** (below, each with a reproduction or a
  measurement).
  1. **Clipped corner** (left as is, documented). The non-gated PIT of a
     reading that follows another reading beyond F⁻¹(1 − EPS) is about 0.5
     for the Gaussian and Gumbel–Hougaard copulas, although its log
     predictive density is about −1 200. With the current local grids it
     also reaches most rows after a gap inside the failure: the
     non-sequential recall of the PMC is 0.15 and 0.02 on motes 48 and 47.
     The sequential (default) flags are not affected.
  2. **Gap quadrature** (fixed at 39f249f, refined up to 5351de0). With
     strongly dependent copulas (τ ≈ 0.99–0.997 at 30 s), the quadrature
     had not converged at the default `gap_nodes = 64`, nor at 512. On the
     clean days of this study G = 64 is now within 0.042 nats of G = 256.
  3. **Leading gap** (fixed, P5). When a series started with missing rows,
     the first reading after them was misintegrated under strong
     dependence, with no WARNING (−4.3 nats at τ = 0.997 and −22 nats at
     τ = 0.999 on a 400-row simulated series, G = 64).
  4. **`quad_error` on long runs of missing rows** (fixed at 5351de0). It
     reported about 1 per missing row, whatever the error: 4 533 for a
     trailing gap of 4 656 rows, which contributes exactly 0 to the
     log-likelihood. Now under 0.01 for that gap, and long runs converge in
     G.

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
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/robust.py --jobs 3 --resume &   # --resume: reuse saved tasks
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/detect.py --check --jobs 2 &     # side by side
wait
.venv/bin/python report/erroneous_data/intel_lab/summarise.py
.venv/bin/python report/erroneous_data/intel_lab/repro_clipped_corner.py   # library problem 1
.venv/bin/python report/erroneous_data/intel_lab/repro_gap_nodes.py        # library problem 2 (≈ 5 s)
.venv/bin/python report/erroneous_data/intel_lab/repro_leading_gap.py      # library problem 3 (≈ 10 s)
```

`robust.py` saves each task to `results/cache/robust_tasks/` as it
finishes; `--resume` reuses the saved tasks, so an interrupted run loses
only its running tasks. `--max-rounds` (default 10) bounds the refits of
every flag-and-mask loop.

The scripts keep the WARNINGs of pmcprg instead of hiding them. Every task
counts the quadrature WARNINGs of its passes and records `quad_error`
against its threshold (0.05). Since pmcprg 5351de0 `quad_error` estimates
the error of the log-likelihood in nats, summed over the runs of missing
rows; before, it summed capped relative errors (library problem 4). It is
recorded in:

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

**A second process pool for `robust.py`.** A task saved to the cache is
loaded, not rerun, when `--resume` reaches it. So a helper that runs the
same task specifications in reverse order into the same cache, while the
main run goes forward, meets it in the middle; each of its workers can at
most duplicate the task the main run is on. The helper is not versioned
(`report/out/reruns/robust_helper.py`). Once both have stopped, a last
`robust.py --resume` loads every task and writes the tables.

The wall times of the last rerun on an Apple arm64 laptop (32 GB; Python
3.14.7, numpy 2.5.3, scipy 1.18.1), from the run logs and the
`*_info.json` files:

* `fit_clean.py`: 6.4 min with 4 processes (256 s of fits, 126 s of PIT
  checks), on pmcprg 5351de0. 35e897d only changes what happens on an
  exactly singular system, which stops 5351de0; step 1 did not stop, so
  its results are those of 35e897d too.
* `detect.py`: 27 min with 4 processes (99 min of task time), on
  35e897d. The first attempt on 5351de0 stopped on a singular 5×5 system
  of the quadrature diagnostic (`LinAlgError`), which 35e897d fixes.
* `detect.py --check`: 94 min with 2 processes, alongside `robust.py`. Its
  G = 256 passes take 2.6–29 min each, against 17–245 s at G = 64.
* `robust.py`: 5 h 53 min of wall time, 25.8 h of task time (13.2 h in the
  previous rerun). The main run used 3 processes from 13:16 to 19:07. A
  reverse-order helper on 4 processes ran from 14:50 to 18:55 into the
  same cache; one task (mote 22, PMC, `fm_hampel`, 2.3 h) ran in both. The
  last `--resume` took 103 s. Its `results/robust_info.json` records only
  that last run's pool: `wall_seconds` is 1.4 s, and `jobs` 3.
  `fit_seconds_sum` (92 892 s) is the sum over the 48 saved tasks. The 12
  PMC flag-and-mask tasks take 16–202 min each; the 24 HMC-IN tasks
  0.6–17 s.

**Memory.** Run `--check` on pmcprg 5f23fc9 or later. Before 5f23fc9 the
G = 256 passes on these windows reached 21–23 GB per process (CHANGELOG,
"memory of the gap quadrature"). With 5f23fc9 a G = 256 sequential pass on
the mote-48 check window takes about 2.7 GB of footprint and 11 min on its
own. In the last rerun the processes of the study peaked at 16.8 GB of
footprint in all (4.1 GB for the largest), with the check and `robust.py`
side by side; `detect.py` alone at 11.6 GB (a memory guard sampling every
60 s).

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
| 22 | PMC | −79768 | −81512 | 3 | 22.45 / 22.75 / 23.36 | 1.79 / 5.17 / 3.40 | Gauss(0.975) / Clayton(0.997) / Clayton(0.920) | 940 |
| 47 | HMC-IN | 64679 | 47486 | 3 | 17.26 / 20.25 / 23.73 | 0.94 / 0.59 / 1.86 | – | 0 |
| 47 | PMC | −74363 | −79747 | 3 | 18.71 / 21.61 / 25.49 | 1.95 / 1.89 / 2.40 | Gauss(0.996) / GH(0.983) / GH(0.969) | 2838 |

* **K.** BIC chooses K = 3 for every mote and model. With about 17 600
  strongly dependent rows, BIC would keep adding states. K = 3 is the upper
  end of the range the study allows, and it is interpretable (next point).
* **Starts.** The PMC likelihood has many basins: the 3 starts end
  940–2 838 nats apart, as in `report/real_series`.
* **The PMC gains 63 700–74 600 nats of likelihood over the HMC-IN.** Its
  diagonal copulas have τ = 0.92–0.997, and τ = 0.996–0.997 for the state
  that holds the night: consecutive 30 s readings are almost deterministic
  given the previous one.
* **The HMC-IN fits are those of the first run**, bit for bit: they use the
  exact K-state message, which no library change touched.
* **The PMC fits barely moved with the long-run fix** (5351de0, against the
  previous rerun on d14e91d). The same starts are kept; their parameters
  change by at most 7.7e-4 in relative terms (mote 22) and their
  log-likelihoods by at most 0.19 nats. Only a start that no cell keeps
  moves by more (−15 nats, mote 47, K = 3). The tables above and below are
  the previous rerun's to their last digit, except where stated.

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
| 22 | PMC K=3 | 7736 | 0.118 (1e-94) | 2152 (< 1e-300) | 3103 (< 1e-300) | −0.08 | 0.75 | 0.01 | 23 / 2 / 0 (77 / 8 / 0.8) | 3 |
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
| 48 | 0.057 | 0.0048 | 67 114.06 | 67 114.07 | 3 / 10 | yes |
| 22 | 0.64 | 0.0068 | 59 689.09 | 59 689.05 | 2 / 3 | yes |
| 47 | 0.42 | 0.0003 | 58 729.45 | 58 729.42 | 2 / 30 | yes |

* **Converged in practice at G = 64.** The log-likelihood of days 1–10
  moves by 0.006–0.042 nats from G = 64 to 256. The sd of the scores after
  a gap (to 3 decimals), the flag counts and the sequentially flagged rows
  are the same.
* **The WARNING fires anyway.** `quad_error`, now an estimate in nats, is
  0.057–0.64 at G = 64 (0.19–1.72 before the long-run fix), 10–17 times the
  change to G = 256, and falls below its threshold at G = 256. The ICE
  E-steps of the PMC fits warn too (28–41 passes of the 3 starts per cell;
  `quad_error` 0.04–0.92 for the kept fits).
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
| PMC seq α=1e-3 | 1.00 / 1.00 / 1.00 | 0.78 / 0.94 / 0.84 | 0.81 / 0.97 / 0.97 | 0.58 / 0.31 / 0.88 | 559 (8) / 122 (9) / 49 (7) |
| PMC seq α=1e-4 | 1.00 / 1.00 / 1.00 | 0.96 / 0.97 / 0.87 | 1.00 / 1.00 / 0.99 | 0.55 / 0.24 / 0.80 | 0 (0) / 0 (0) / 3 (1) |
| PMC seq BH α=1e-3 | 1.00 / 1.00 / 1.00 | 0.96 / 0.97 / 0.87 | 1.00 / 1.00 / 0.99 | 0.54 / 0.24 / 0.80 | 0 (0) / 0 (0) / 2 (1) |
| PMC nonseq α=1e-3 | 0.15 / 1.00 / 0.02 | 0.87 / 0.96 / 0.26 | 0.97 / 1.00 / 0.86 | 0.21 / 0.30 / 0.28 | 12 (8) / 9 (9) / 8 (7) |
| PMC nonseq BH α=1e-3 | 0.15 / 1.00 / 0.02 | 0.93 / 0.97 / 0.34 | 1.00 / 1.00 / 0.98 | 0.15 / 0.24 / 0.21 | 0 (0) / 0 (0) / 0 (0) |
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

The HMC-IN and baseline rows are those of the first run, which the reruns
reproduced. Against the previous rerun (d14e91d), the long-run fix leaves
every lead time and every sequential recall unchanged; it moves the
sequential PMC's false alarms at α = 1e-3 (605 → 559 on mote 48, 106 →
122 on mote 22) and the non-sequential recall of mote 48 (0.26 → 0.15).
The figures are `figures/detect_overview.png` (the whole window),
`figures/detect_zoom.png` (36 h before to 6 h after the failure),
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
  - **Mote 48.** 8 episodes, 559 flags. Two of them hold 553 flags: 03-13
    from 12:56 (4.5 h, 320 flags) and 03-20 from 13:58 (2.2 h, 233 flags),
    smooth afternoon peaks of 26.2–29.9 and 27.1–29.3 °C, up to 2 °C above
    the fit week's maximum of 27.9 °C. Without gating the same peaks give 4
    and 2 flags: this is lock-out, and it holds at G = 256 (below). The 6
    other episodes are single flags.
  - **Mote 22.** 9 episodes, 122 flags. The two largest start at 07:01 on
    03-09 and 03-10 (61 and 40 flags): the reading jumps from about 22 °C
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
    PMC (α = 1e-3) flags 58 % (mote 48) and 88 % (mote 47) of the rows.
    The rows it leaves are the lower part of the climb: 28.0–34.7 °C on
    mote 48, 30.0–38.9 °C on mote 47 (8.5–14.8 h before the threshold
    hit). The HMC-IN flags 100 % on both motes.
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
* **Non-sequential PMC: recall 0.15 and 0.02 on motes 48 and 47.**
  - Every suspect row that directly follows another observed suspect row
    is missed (1 146 and 1 064 rows, median p 0.94 and 0.96). This is the
    clipped-corner problem below.
  - Of the 1 207 suspect rows that follow a missing epoch on each mote,
    350 and 48 are flagged (median p 0.87 and 0.83). The first run flagged
    all of them, the previous rerun 621 and 51. These flags depend on the
    quadrature: at G = 256 even fewer are flagged (below).
  - On mote 22, whose top-state copula is a Clayton, every suspect row gets
    p ~ 1e-13 and is flagged: recall 1.00.
  - The sequential run is not affected: recall 1.00 on every mote.
* **Sequential vs non-sequential.**
  - **During a persistent failure.** Gating is what keeps a stuck sensor
    flagged: recall 1.00 against 0.02–0.15.
  - **During legitimate transients.** Gating turns a single flag into a
    run: 559 against 12 normal-operation flags on mote 48, 122 against 9
    on mote 22, 49 against 8 on mote 47. The simulation study measured a
    gating cost of 0.3 extra flags per 1 000 rows on its fixtures. On this
    near-unit-root series it reaches 19 per 1 000 normal rows (mote 48).
  - **For the HMC-IN it barely matters.** No y-dependence: 3 846 against
    3 826 flags on mote 48.
  - **Corrections.** BH (or a smaller per-row α) is the practical fix for
    the lock-out on this data. Per-row α = 1e-2 makes it worse: 2 336,
    1 172 and 957 normal-operation flags.

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
| seq α=1e-3 | 64 | – | 1.00 / 1.00 / 1.00 | 105 / 193 / 351 | 559 (8) / 122 (9) / 49 (7) | 8.8 / 2.1 / 16.2 | 11.1 / 17.8 / 16.2 |
| seq α=1e-3 | 256 | 74 / 17 / 7 | 1.00 / 1.00 / 1.00 | 114 / 193 / 358 | 624 (8) / 105 (9) / 49 (7) | 11.1 / 2.1 / 16.2 | 11.1 / 17.8 / 16.2 |
| seq BH α=1e-3 | 64 | – | 1.00 / 1.00 / 1.00 | 86 / 127 / 279 | 0 / 0 / 0 | 6.1 / 1.2 / 15.2 | 10.6 / 1.2 / 15.2 |
| seq BH α=1e-3 | 256 | 1 / 0 / 73 | 1.00 / 1.00 / 1.00 | 87 / 127 / 211 | 0 / 0 / 0 | 6.1 / 1.2 / 3.2 | 10.6 / 1.2 / 15.2 |
| nonseq α=1e-3 | 64 | – | 0.21 / 1.00 / 0.34 | 39 / 188 / 111 | 12 (8) / 9 (9) / 8 (7) | 8.8 / 2.1 / 3.5 | 11.1 / 17.8 / 16.2 |
| nonseq α=1e-3 | 256 | 25 / 0 / 91 | 0.07 / 1.00 / 0.00 | 31 / 188 / 67 | 12 (8) / 9 (9) / 8 (7) | 1.0 / 2.1 / none | 11.1 / 17.8 / 16.2 |
| nonseq BH α=1e-3 | 64 | – | 0.21 / 1.00 / 0.34 | 25 / 117 / 83 | 0 / 0 / 0 | 6.1 / 1.2 / 3.5 | 8.8 / 1.2 / 15.2 |
| nonseq BH α=1e-3 | 256 | 25 / 0 / 88 | 0.07 / 1.00 / 0.00 | 17 / 117 / 41 | 0 / 0 / 0 | 1.0 / 1.2 / none | 8.8 / 1.2 / 15.2 |

The other non-sequential settings (α = 1e-2, 1e-4, BH 1e-2) differ in
25–27 rows on mote 48, 89–92 on mote 47 and 0–3 on mote 22.

**Against the previous rerun (d14e91d).** The G = 256 results hardly
moved: 847 / 703 / 564 flags at α = 1e-3 against 844 / 703 / 564, the same
BH counts, non-sequential counts within 1 flag, and log-likelihoods within
0.7 nats where the flags agree (37 nats for the non-sequential pass of
mote 47). The G = 64 results moved:

* **Non-sequential: closer to G = 256.** 25–27 rows differ on mote 48
  (34–36 before) and 88–92 on mote 47 (94–102). The log-likelihood of the
  pass moves by 944 and 3 669 nats between G = 64 and 256 (1 312 and 3 956
  before), and by 4.6 nats on mote 22 (2.4).
* **Sequential: further from G = 256.** At α = 1e-3, 74, 17 and 7 rows
  differ (26, 1 and 0 before); under BH 1, 0 and 73 (1, 0 and 81). Where
  the flags agree, the log-likelihoods of G = 64 and 256 are 4.5 nats apart
  on mote 22 under BH (2.4 before) and 5.6 nats on mote 48 (1 row differs;
  0.9 before). Probably the new rule for kernel test pieces: the CHANGELOG
  warns that it can move G = 64 results on runs too short for the moment
  tilt (under 32 rows), and the gated filter turns every flagged reading
  into such a run.

`quad_error` of the non-sequential pass over the window's gaps, now an
estimate in nats: 400 / 22 / 142 at G = 64 and 16 / 0.092 / 71 at G = 256,
against a threshold of 0.05 (6.2e3 / 3.8e3 / 6.3e3 and 3.2e3 / 1.8e3 /
3.4e3 before). Over the whole detection window at G = 64 it is 2 957 /
33 / 170 (28 528 / 15 072 / 31 593 before; `detect_scores.csv`).

* **What does not change.**
  - The sequential recall: 1.00 at both G on every mote and setting.
  - The false-alarm episodes in normal operation: none under BH at either
    G, and at α = 1e-3 the same 8, 9 and 7 episodes, with 559 → 624,
    122 → 105 and 49 → 49 flags. The mote-48 lock-out is genuine.
  - The first flag of the last 24 h, for every setting and mote.
  - Mote 22, non-sequential and BH: 0–3 rows differ.
* **What changes.**
  - **Mote 47, sequential BH:** 73 rows, mostly in the climb (279 → 211 of
    401 climb rows flagged). The alarm then breaks up, and the episode that
    reaches the failure starts 3.2 h before it instead of 15.2 h. Its first
    flag of the last 24 h is still 15.2 h ahead.
  - **Mote 48, sequential α = 1e-3:** 74 rows: 9 more climb rows, 65 more
    normal-operation flags in the same 8 episodes. The episode that reaches
    the failure starts 11.1 h ahead instead of 8.8 h.
  - **Non-sequential, motes 48 and 47:** 25–92 rows, on the failure. The
    share of the window's suspect rows flagged falls from 0.21 to 0.07 and
    from 0.34 to 0.00. No non-sequential alarm reaches the failure more than
    1.0 h ahead at G = 256 on mote 48, and none reaches it on mote 47.
* **Why: the passes that condition on failing readings.** Above 38.4, 64.7
  and 45.0 °C (motes 48, 22, 47) a reading lies beyond F⁻¹(1 − EPS) of
  every clean state, where the copula arguments are clipped (library
  problem 1). The non-sequential pass conditions on them, and its
  log-likelihood moves by hundreds to thousands of nats between G = 64 and
  256. A one-off breakdown by run of missing rows of the mote-48
  non-sequential pass at G = 64 (`gaps._quadrature_report(..., per_run=True)`,
  400 in all):
  - 368 on the 123 runs next to a reading above 40 °C (980 before). Many of
    them sit at the report's floor of 4.6 nats per run (|log 0.01|): the
    pass is not converged there;
  - 2.8 on the four runs of more than 100 rows (5 042 before), the two
    network outages of 2 944 and 2 341 rows among them (5 015 before,
    library problem 4);
  - 29 on the other 3 721 runs (226 before).
* **Caveat.** At the default G = 64, the PMC detection results on the
  failure depend on the quadrature wherever a pass conditions on failing
  readings: the non-sequential recall, climb coverage and lead on motes 48
  and 47, and through gating the BH lead on mote 47 and the α = 1e-3 lead
  on mote 48. The sequential recall, the false-alarm episodes in normal
  operation and the first-flag leads do not. G = 256 has not converged
  either (`quad_error` 16 and 71 nats on motes 48 and 47), so neither node
  count gives the reference numbers on the failure. The model is also far
  outside its validity there: the plateau and most of the rise above
  60 °C lie more than 8 sd above every clean margin.

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
| 48 | raw | 1 | 0 | 20.61 / 21.53 / 40.62 | 2.07 / 1.11 / 32.80 | 41, 33, 0.998 | 0.80 |
| 48 | oracle | 1 | 211 (1.00 / 0.00) | 19.45 / 20.12 / 24.14 | 1.67 / 0.97 / 5.48 | – | 0.77 |
| 48 | oracle_ext | 1 | 393 (1.00 / 1.00) | empty / 20.24 / 21.31 | – / 2.08 / 2.21 | – | 1.00 |
| 48 | fm | 11, not converged | 16 (0.01 / 0.02) | 19.39 / 21.34 / 42.25 | 0.87 / 1.93 / 33.50 | 42, 34, 0.999 | 0.81 |
| 48 | fm_hampel | 11, not converged | 19 (0.01 / 0.02) | 20.74 / 21.98 / 36.48 | 1.99 / 0.76 / 30.24 | 36, 30, 0.999 | 0.97 |
| 48 | fm_thr60 | 11, not converged | 15 (0.01 / 0.02) | 19.38 / 21.34 / 42.16 | 0.87 / 1.93 / 33.46 | 42, 33, 0.999 | 0.81 |
| 48 | fm_cs | 11, not converged | 10 (0.00 / 0.00) | 20.74 / 20.77 / 42.01 | 0.67 / 2.00 / 33.39 | 42, 33, 0.999 | 0.96 |
| 22 | raw | 1 | 0 | 24.33 / 51.83 / 122.15 | 5.74 / 34.67 / < 0.01 | 52, 35, 0.978; 122, < 0.01, 0.999 | 0.97 |
| 22 | oracle | 1 | 1079 (1.00 / 0.00) | 24.47 / 24.80 / 46.73 | 5.84 / 4.95 / 7.10 | 47, 7, 0.995 | 1.00 |
| 22 | oracle_ext | 1 | 1702 (1.00 / 1.00) | empty / 24.09 / 24.85 | – / 5.53 / 5.31 | – | 1.00 |
| 22 | fm | 11, cycling | 0 | = raw | | = raw | 0.97 |
| 22 | fm_hampel | 11, cycling | 0 | = raw | | = raw | 0.97 |
| 22 | fm_thr60 | 11, cycling | 0 | = raw | | = raw | 0.97 |
| 22 | fm_cs | 5, converged | 36 (0.02 / 0.01) | 24.41 / 24.99 / 89.85 | 5.76 / 5.09 / 32.20 | 90, 32, 0.9998 | 1.00 |
| 47 | raw | 1 | 0 | 21.49 / 37.76 / 122.15 | 2.62 / 25.33 / < 0.01 | 38, 25, 0.967; 122, < 0.01, 0.451 | 0.60 |
| 47 | oracle | 1 | 267 (1.00 / 0.00) | 21.42 / 21.56 / 33.08 | 2.67 / 1.04 / 7.31 | 33, 7, 0.989 | 0.48 |
| 47 | oracle_ext | 1 | 668 (1.00 / 1.00) | 20.74 / 23.64 / empty | 1.73 / 3.17 / – | – | 1.00 |
| 47 | fm | 11, not converged | 5 (0.02 / 0.00) | 21.52 / 37.77 / 122.15 | 2.64 / 24.87 / < 0.01 | 38, 25, 0.968; 122, < 0.01, 0.469 | 0.60 |
| 47 | fm_hampel | 11, not converged | 8 (0.02 / 0.00) | 21.48 / 37.76 / 122.15 | 2.61 / 24.88 / < 0.01 | 38, 25, 0.967; 122, < 0.01, 0.000 | 0.61 |
| 47 | fm_thr60 | 11, not converged | 305 (1.00 / 0.04) | 19.84 / 23.15 / 27.35 | 1.61 / 1.97 / 6.45 | – | 0.64 |
| 47 | fm_cs | 3, converged | 31 (0.00 / 0.02) | 20.09 / 23.24 / 47.50 | 1.71 / 1.82 / 34.44 | 48, 34, 0.9995 | 0.74 |

The `raw_cs` rows (the raw fit from the clean model) and the quadrature
columns are in `results/tables.md`. The mask count of every refit and the
ICE iterations of every fit are in `results/robust_fits.csv`
(`masks_per_fit`, `iters_per_fit`). The states of the raw, `oracle_ext`,
`fm` and `fm_thr60` fits are drawn in `figures/robust_hmc_in.png` and
`figures/robust_pmc_state.png`.

**Against the previous rerun (d14e91d).** The HMC-IN fits and loops are the
same. The PMC fits of mote 22 are the same to the table's digits but for
its `oracle` fit, and the raw fits of motes 48 and 47 move by less than
1 °C in their failure state. Three loops end differently: mote 48's `fm`
no longer converges, its `fm_thr60` no longer recovers, and mote 47's
`fm_thr60` no longer has a failure state. The masks of most other loops
changed too.

* **The failing readings form their own state, as the simulation predicts.**
  Every raw fit has one:
  - HMC-IN: mean 59–120 °C, sd 3–37 °C, stay 0.9998–1.0000, an absorbing
    state;
  - PMC: a broad state of mean 38–52 °C, sd 25–35 °C, stay 0.967–0.998.
    On motes 22 and 47 the PMC spends a second state on the stuck
    122.15 °C plateau (π 0.085 and 0.023), so two of its three states
    describe the failure.

  Starting ICE from the clean-window model changes nothing (`raw_cs`,
  `fm_cs`: a failure state in 6 of 6, PMC means 42–90 °C). The breakdown is
  a property of the likelihood, not of the k-means start.
* **Under that state almost nothing is flagged**, and the default loop
  ends at or near the raw fit: 0 masked rows for the HMC-IN, 16, 0 and 5
  for the PMC, every fit with a failure state.
* **Flag-and-mask finds no fixed point in 10 of the 12 PMC loops** (9 in
  the previous rerun; 3 of the 12 HMC-IN loops, all on mote 22).
  `robust_estimate` stops at `max_rounds` = 10 refits, and its result is
  the 11th fit, not a fixed point.
  - **Mote 22, `fm`, `fm_hampel` and `fm_thr60`: one deterministic cycle,
    as before.** The raw fit flags 13 rows (climb rows in the previous
    rerun); with them masked the fit flags none, so the next fit is the raw
    fit again: 0 | 13 | 0 | 13 …
    `fm_hampel` joins the cycle at its 5th fit, and `fm_thr60` at its 11th,
    after releasing the pre-mask round by round (1 079, 947, 989, 1 023,
    839, 123, 144, 70, 15, 13, 0). Every fit of the cycle runs to the ICE
    cap of 50 iterations, and so does the raw fit: ICE itself does not
    converge on this window. All three now end on the raw fit. The HMC-IN
    cycles the same way on this mote (0 | 33).
  - **Mote 48, `fm`: no longer converges.** The mask moves between 12 and
    25 rows and ends with two different 16-row masks (0, 12, 23, 25, 21,
    19, 23, 20, 23, 16, 16). The previous rerun reached a fixed point of 22
    rows after 7 fits. Either way it keeps a 42–43 °C failure state.
  - **Mote 47, `fm` and `fm_hampel`: a drifting mask.** After the first fit
    it moves between 5 and 15 rows (`fm`: 0, 7, 11, 15, 7, 5, 8, 8, 11, 7,
    5) while every ICE fit converges in 6–13 iterations; `fm_hampel` runs
    through `fm`'s counts from its 6th fit.
  - **Motes 48 and 47, `fm_thr60`.** On mote 47 the mask wanders between
    281 and 333 rows after the threshold's 267, as before. On mote 48 it
    grows from 211 to 265–298 rows, then collapses to 6, 13 and 15 in the
    last three rounds, and the loop falls into the failure basin of `fm`
    (the same 42 °C state, sd 33 °C). The previous rerun ended at 350 rows
    with no failure state. One fit of mote 48's loop and two of mote 47's
    run to the ICE cap.
  - **Mote 48, `fm_hampel` and `fm_cs`:** 9–31 and 0–56 rows.
  - **Converged:** `fm_cs` on motes 22 and 47 (5 and 3 fits, 36 and 31
    rows), as before. Both keep a failure state (90 and 48 °C, sd 32–34
    °C): they converge to the breakdown.

  The HMC-IN, which uses no quadrature, cycles too: non-convergence comes
  from flag-and-mask on a window where the failure is a state. The
  particular masks of the PMC depend on the quadrature: the long-run fix
  changed the ending of 3 of the 12 loops.
* **Hampel `initial_mask`.** It masks 31–130 rows, spikes in the 30 s noise
  and not the drift or the plateau, so the loop falls back into the failure
  basin (6 of 6). The simulation's remedy for frequent spikes does not
  transfer to a smooth persistent failure.
* **Threshold `initial_mask`.** The sensor-level flag is re-tested by the
  model. 3 of the 6 fits end with every suspect row masked (4 in the
  previous rerun):
  - **HMC-IN on mote 47.** 299 rows masked, converged, means 18.90 /
    22.26 / 29.45 against the oracle's 18.89 / 22.26 / 30.15.
  - **HMC-IN on mote 48.** 416 rows masked, every suspect and climb row,
    converged. The model extends the mask into the drift.
  - **PMC on mote 47.** 305 rows masked, every suspect row and 4 % of the
    climb; not converged. The climb left in widens the top state (27 °C, sd
    6.5 °C, against 3.2 °C at most in the `oracle_ext` fit), whose mean
    stays below the failure-state threshold (31.0 °C). The previous rerun
    kept it as a 33 °C failure state.

  The PMC on mote 48 releases the pre-mask and ends in the failure basin
  (above). On mote 22 (19 % failing) both models release it round by
  round and end in the raw fit's cycle. The HMC-IN goes from 1 079 to 978,
  887, 796, 755, 567 and 0 masked rows: the climb, left in, forms a state
  at about 47 °C (the `oracle` fit's) that makes the lower suspect
  readings plausible, which widens the state, and so on.
* **The threshold oracle is not clean either.** Masking only the suspect
  rows leaves the climb in the fit. The HMC-IN's top state keeps an sd of
  7.7, 6.4 and 6.1 °C, against 2.7, 2.5 and 1.7 °C when the climb is masked
  too. The PMC's `oracle` fits keep a 24 °C state of sd 5.5 °C on mote 48,
  and a 47 °C and a 33 °C state on motes 22 and 47. The partial ground
  truth biases the estimate as much as a mild contamination would.
* **Classification of the clean rows.**
  - **HMC-IN.** The level regimes are identifiable. Against `oracle_ext`,
    the raw fits agree on 71–88 % of the normal rows and the threshold
    oracle on 73–99 %. The raw fit spends a state on the failure and merges
    two day/night levels into the remaining two.
  - **PMC.** The agreement cannot rank the fits. The `oracle_ext` fits of
    all three motes empty a state, and on mote 22 one state has π = 0.83,
    so almost any fit agrees with it (0.94–1.00). On motes 48 and 47 the
    two oracles agree on 77 % and 48 % of the normal rows. The parameters
    (the failure states, the top-state sd) rank the fits.
* **Cost.** Every PMC task took 0.9–3.3 times (median 1.8) its time in the
  previous rerun, 1.3–2.7 times (median 1.9) per ICE iteration: 26 h of
  task time against 13 h. The moment tilt of the long runs costs time
  (CHANGELOG), and this run shared the machine with up to 6 other
  processes (Run).

**Quadrature of the PMC fits.** Every ICE E-step of a PMC fit on these
windows still logs the quadrature WARNING (8–562 WARNINGs per task). Now
that `quad_error` estimates nats, the returned fits report 2.2–67 nats on
their masked windows, and 211 for the raw fit of mote 47 and its `fm` loop
(9.5–3 970 before, in the old units). The PMC fits were not rerun at 256
nodes: a flag-and-mask task takes 16–202 min at G = 64. The failure states
of the raw fits are unlikely to depend on G (the failing rows are 6–19 %
of the window, tens of degrees above the clean margins, and the long-run
fix moved them by less than 1 °C); the masks, cycles and non-convergence
do (above).

**The 3 WARNINGs logged after the fits.** Once the tasks are done,
`robust.py` classifies the normal rows with the clean-window model on each
window with the suspect and climb rows set to NaN, for the
`agree_clean_model` column of `results/robust_fits.csv` (every PMC row of
the mote). For the PMC that pass runs outside the tasks, so its WARNINGs
reach the run log: `quad_error` 7.3, 9.6 and 1.1 nats on 8 668, 7 157 and
8 705 missing rows (motes 48, 22 and 47; 4.2e3, 2.3e3 and 4.5e3 before, in
the old units). One-off recomputations of that pass:

* **The trailing gaps now count 0**: 1.2e-11 and 3.9e-12 for the 2 875
  and 2 872 rows of motes 48 and 22, under 0.01 for the 4 656 rows of
  mote 47 (4 533 before).
* **Where the rest sits.** On mote 48, 5.8 of 7.3 on three network gaps of
  34–58 rows among the normal rows, 61–63 h before the first suspect
  reading, and 0.45 on the 1 464-row gap that ends at a 59.55 °C reading
  (1 363 before). On mote
  22, 4.6 (the per-run floor) on a 14-row gap 29 h before it, 2.5 on the
  704-row masked block that ends at a 59.87 °C reading, 1.7 on a 9-row gap.
  On mote 47, 1.1 spread over 993 short runs.
* **How much it matters.** Rerun at G = 128, the same pass changes the
  MPM state of 0 of the 4 879 normal rows of mote 48 and 3 of the 5 451 of
  mote 22, with log-likelihoods 1.0 and 0.7 nats from G = 64 (the reports
  say 7.3 and 9.6). The `agree_clean_model` column is therefore right to
  about 1e-3; no conclusion uses it.

### 4. Conclusion

**What works.**

* **Detection with a model fitted on clean data, held fixed, and gated
  (sequential).** It flags every failing reading, at 64 and 256 nodes. On
  motes 48 and 47 its alarm starts 6.1–8.8 and 16.2 h before the 60 °C
  rule, depending on the setting, before and after the long-run fix.
  Under BH the mote-47 start depends on the quadrature (3.2 h at
  G = 256); its first flag does not.
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
  genuine on mote 48's warm afternoons (559 against 12 flags at G = 64,
  624 at G = 256). It needs a correction (BH or α = 1e-4); a
  re-acquisition rule would be the principled fix.
* **The gap quadrature on the failure.** Where a pass conditions on
  readings far outside the clean margins, the PMC results move between
  G = 64 and 256 (the non-sequential flags of motes 48 and 47, the BH lead
  of mote 47), and neither is converged. `quad_error` now estimates nats
  and flags these windows, but it is capped at 4.6 nats per run and
  understates the change there (142 against 3 669 nats on mote 47).
* **The PMC's lead depends on the width of its clean margins.** It is
  1.5–2.1 h on mote 22.
* **The non-gated copula PIT at the clipped corner** (library problem 1):
  non-sequential recall 0.15 and 0.02 on motes 48 and 47.
* **Rolling filters** (Hampel, trailing robust z) do not see a smooth
  drift or a stuck sensor.
* **Flag-and-mask estimation on a failing window.** It breaks down in
  every case without a sensor-level pre-mask (Hampel pre-screen
  included), and 10 of its 12 PMC loops find no fixed point. It only
  recovers with the 60 °C pre-mask, only where at most about 10 % of the
  window is failing (motes 48 and 47), and fully only for the HMC-IN. The
  PMC does not converge there: on mote 48 it drops the pre-mask and breaks
  down (it recovered in the previous rerun), and on mote 47 it keeps the
  climb in a broad state.

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
    range), not re-estimated. A re-estimated g becomes the 38–120 °C
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
  0.15 and 0.02 (0.51 and 0.53 in the first run, 0.26 and 0.02 in the
  previous rerun).
  - Every suspect row that follows an observed suspect row is missed:
    1 146 and 1 064 rows, median p 0.94 and 0.96.
  - Of the 1 207 suspect rows that follow a missing epoch on each mote,
    857 and 1 159 are missed now (none in the first run, 586 and 1 156 in
    the previous rerun); the median p of those 1 207 rows is 0.87 and
    0.83. At G = 256 fewer still are flagged (section 2, the check at 256
    nodes), which fits the local grids reaching the clipped corner.
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

The outputs above are those of 39f249f. On the current code (the
worktree at 3df8f55, with 5351de0's long-run fix) the script prints the
same lines, except G = 256 at τ = 0.99: 7023.5 instead of 7023.3, so the
non-monotone residual is gone. On this study's data, G = 64 gives a
log-likelihood of mote 48's clean days 1–10 within 0.006 nats of G = 256
(the refitted model; section 1), and every PMC fit, PIT and flag of the
study was rerun on the current quadrature.

### 3. A leading gap under strong dependence is misintegrated, with no WARNING

**Status: fixed** (P5, CHANGELOG `[Unreleased]`). Measured there: the
reproduction's leading gaps (L = 2, 5 and 17) within 2.5e-4 nats of the
gap-free pass at G = 64 and 4.8e-8 at G = 256, the state filter equal to π
to 4e-15, and the batch pass and the PIT filter within 2.3e-13 nats of
each other. In the rerun of step 1 the two passes give the same
log-likelihood on days 1–10 (section 1). The outputs below are those of
39f249f. On the current code (3df8f55) `repro_leading_gap.py` prints an
error of 0.000 nats and max |α̂ − π| of 0.000 (batch pass) and 7e-15 or
less (PIT filter) at every G and both τ, with `quad_error` 1e-2 at G = 64
and 3e-9 at G = 256, under the threshold.

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

**Status: fixed at pmcprg 5351de0** (CHANGELOG `[Unreleased]`, "long runs
of missing rows converge in G"). Runs of 32 missing rows or more take
moment-matched transitions and converge in G, and `quad_error` became a
per-run estimate of the log-likelihood error in nats, in which a trailing
gap counts 0. Found in the previous rerun (d14e91d), from the WARNINGs of
the robust tables step and of the detection check.

**The problem.** `quad_error` summed, over the missing positions of each
run, a weighted mean of relative errors capped at 1 per block. On a long
run where the checks of every position failed, the report grew with the
length of the run, by about 1 per missing row, and no longer estimated
nats.

* **A trailing gap.** On mote 47's robust window, with the suspect and
  climb rows set to NaN, the clean PMC reported 4 548 at G = 64, of which
  4 533 on one trailing gap of 4 656 rows (0.97 per row). A trailing gap
  contributes exactly 0 to the log-likelihood, and for state margins the
  state law across a gap is exact (`gaps` module docstring): no result that
  involves an observed row depends on it.
* **Network outages.** On mote 48's detection-check window, 5 015 of the
  6 247 of the non-sequential pass came from two outages of 2 944 and
  2 341 rows between ordinary readings (0.95 and 0.94 per row). The gated
  passes crossed the same outages, and where they flagged the same rows
  their log-likelihoods at G = 64 and 256 agreed within 0.1–2.4 nats.
* **Why it mattered.** The WARNING fired on any window with long outages,
  at every G, and could not tell the outages from the failure, where the
  log-likelihood of the non-sequential pass does move by thousands of
  nats.

**After the fix.** The same one-off passes at G = 64 (`gaps._posterior`,
then `gaps._quadrature_report(..., per_run=True)`; not scripted here),
before (d14e91d) and after (35e897d):

| pass at G = 64 | before | after |
|---|---|---|
| mote 47, robust window, clean PMC, suspect and climb rows as NaN | 4 548 | 1.15 |
| — its 4 656-row trailing gap | 4 533 | < 0.01 |
| mote 48, the same | 4 170 | 7.25 |
| — its 2 875-row trailing gap | 2 777 | 1.2e-11 |
| — the 1 464-row gap that ends at a 59.55 °C reading | 1 363 | 0.45 |
| — the other 917 runs | 29 | 6.8 |
| mote 48, check window, non-sequential pass | 6 247 | 400 |
| — the four runs of more than 100 rows (the two outages among them) | 5 042 | 2.8 |
| — the 123 runs next to a reading above 40 °C | 980 | 368 |
| — the other 3 721 runs | 226 | 29 |
| clean days 1–10, the selected PMC, motes 48 / 22 / 47 (`pit_checks.csv`) | 0.19 / 1.72 / 0.23 | 0.057 / 0.64 / 0.42 |

* **Trailing gaps and outages no longer count.** What remains sits on
  short gaps among the readings, and on the failure.
* **Against the change it estimates.** On the clean days the report is
  10–17 times the change of the log-likelihood from G = 64 to 256. On the
  robust windows of motes 48 and 22, 7.3 and 9.6 against a change of 1.0
  and 0.7 nats from G = 64 to 128. On the failure, where many runs sit at
  the per-run cap of 4.6 nats (|log 0.01|), it understates the change on
  motes 48 and 47 (400 and 142 against 944 and 3 669 nats to G = 256) and
  overstates it on mote 22 (22 against 4.6).
* **The price at G = 64.** The gated detection passes moved away from
  their G = 256 results (section 2); the CHANGELOG notes that the new rule
  for kernel test pieces can move G = 64 results on short runs.
* **Cost.** The robust windows' passes took 23–40 s at G = 64 (25–41 s
  before), the check window's 97 s (61 s before), with 1.1–2.2 GB of
  footprint.

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
  there. At G = 64 the gated passes are also some rows off G = 256 in
  normal operation. The robust PMC fits were not checked at 256 nodes, and
  their masks moved with the long-run fix. `quad_error` is an estimate in
  nats since 5351de0, but capped per run on the failure.
* **Robust estimation.** One window per mote, K fixed at the clean
  selection (3). With K = 4 a failure state would not have to take a
  day/night level, which was not tried. 10 of the 12 PMC `robust_estimate`
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
