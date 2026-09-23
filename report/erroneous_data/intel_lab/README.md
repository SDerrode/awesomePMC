# Erroneous data on real sensors: the Intel Berkeley Lab temperatures

> **Status (2026-09-23, pmcprg 39f249f).** The copula-model (PMC) numbers
> below are still provisional: they come from the run with the unconverged
> gap quadrature. The HMC-IN numbers, the baselines and the labels are final.
>
> * **Fixed on main.** The gap quadrature under strong dependence (library
>   problem 2 below) is fixed at 39f249f, with local grids and the
>   `GapPosterior.quad_error` WARNING. `repro_gap_nodes.py` now gives the
>   same log-likelihood and the same flags at every G (section "Library
>   problems found"). The empty state (NaN PIT) is fixed too.
> * **Clipped corner: decided, not changed.** The PIT at the clipped copula
>   corner (library problem 1) is left as is. `pmcprg/pmc/outliers.py`
>   documents why: without gating, a run of erroneous readings is judged
>   given its own first value, so `sequential=True` is the detection mode.
> * **The rerun stopped at a new library problem.** The rerun of the PMC
>   parts on 39f249f began with step 1 (`fit_clean.py`), whose results are
>   not committed. It stopped there, as the brief requires for a library
>   bug: a leading gap under strong dependence is misintegrated, and no
>   WARNING is logged (library problem 3, `repro_leading_gap.py`).
>   - In this study only the clean windows start with a gap (17, 2 and 17
>     rows on motes 48, 22 and 47).
>   - The detection and robust-estimation windows start with a reading.
> * **HMC-IN.** It does not depend on anything that changed: it uses the
>   exact K-state message, and none of its fits has an empty state. The
>   step-1 rerun reproduced its fits and PIT checks bit for bit.

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
ones. The sensitivity numbers at G = 256 quadrature nodes are an exception:
they come from one-off runs, quoted in the text.

> **Why the PMC numbers are provisional.** They were computed before the
> fix of library problem 2: a gap quadrature that had not converged at the
> default `gap_nodes = 64`. That affects every PMC number that involves a
> missing row:
>
> * the PMC fits (the ICE E-step, with 12 % single-epoch gaps);
> * the PMC log-likelihoods and BIC;
> * the PIT of the rows that follow a gap;
> * under gating, every row that follows a flag.
>
> The HMC-IN results (exact K-state filter), the baselines and the labels
> are not affected. The main PMC detection numbers were checked at G = 256
> in that run (section 2). Recall and lead times were unchanged there; the
> false-alarm counts were not.

## Summary

* **Calibration on clean data: no model passes the tests, but the copula
  model is close in the margin and the HMC-IN is far off.** On the held-out
  days 8–10 (n ≈ 7 600 rows per mote), every KS and Ljung–Box test rejects.
  - **PMC (K = 3).** The PIT is nearly uniform (KS D = 0.03–0.09) and slightly
    too wide (sd of the normal scores 0.81–0.91). The lag-1 autocorrelation of
    the scores is 0.06–0.10: temperature has momentum that a first-order
    chain cannot carry. The 12 % of rows that follow a gap have an
    over-dispersed PIT at the default quadrature (library problem 2).
  - **HMC-IN.** Its scores have a lag-1 autocorrelation of 1.00. At 30 s, a
    model whose observations are independent given the state is only a
    level test.
* **Detection with the clean model held fixed works.** Sequential flagging
  catches every reading above 60 °C (recall 1.00 on all three motes).
  - **PMC, sequential, BH at α = 1e-3.** Its first flag comes 9.7, 0.9 and
    13.8 h before the threshold hit (motes 48, 22, 47). Its 0, 0 and 21
    flags in 14–15 days of normal operation make 0, 0 and 2 alarm episodes.
  - **PMC, sequential, α = 1e-3, checked at G = 256.** Recall and lead
    times are unchanged (11.1, 1.2 and 19.1 h). The normal-operation flags
    fall from 425 / 0 / 484 to 0 / 0 / 164.
  - **HMC-IN.** It warns earlier on all three motes (13.4, 6.5 and 15.2 h)
    but raises about 900–1 300 flags in the normal operation of motes 48
    and 47, on warm afternoons.
  - **Fixed thresholds.** A plain 35 °C threshold gives 5.3, 6.8 and 13.1 h
    of warning with no false alarm on these three motes.
  - **Rolling filters.** The Hampel filter and the rolling robust z-score do
    not detect this failure: their recall is 0.00–0.13. Both adapt to a smooth
    drift and to a stuck value.
* **The flags before the threshold are mostly true early drift.** On motes
  48 and 47, 93–100 % of the out-of-range climb is flagged by the sequential
  PMC. Mote 22 is the exception (22 %): its clean-window PMC has a
  wide margin (sd 5.65 °C) that accepts a climb to about 55 °C. The false
  alarms, examined on the plots, fall into two classes:
  - warm afternoon peaks above the fit-window maximum (mote 48, 29–30 °C,
    which vanish at G = 256; mote 47, 31.7 °C);
  - fast rises of 2–7 °C within minutes, repeated in the late afternoon on
    mote 47 (probably sun on the sensor).

  Both are real temperatures, not sensor errors.
* **Innovation gating has a real-data cost: lock-out.** A gated legitimate
  reading makes its successors look wrong, because the gated filter keeps
  predicting the old level while the true level moves on. On the held-out
  afternoon peak of mote 48 (03-07, 29.1–30.4 °C), one flag becomes a run
  of hundreds:
  - 239 sequential flags against 22 non-sequential ones at G = 64;
  - 265 against 4 at G = 256, where the problem is genuine.

  Elsewhere, most of the lock-out measured at the default G = 64 is the
  quadrature problem. On mote 47's held-out days, 513 sequential flags
  fall to 52 at G = 256 (non-sequential: 21 and 20). The
  Benjamini–Hochberg correction or α = 1e-4 removes almost all of them
  at G = 64.
* **Robust re-estimation breaks down, as the simulation predicts.** In every
  raw fit (6 of 6), the failing readings form a state of their own, with a
  mean of 33–120 °C, an sd of 3–45 °C and a stay probability of 0.80–1.00.
  - **Default flag-and-mask.** Under that state almost nothing is flagged,
    and the loop ends at the raw fit. It stops at once, or, on mote 22,
    after cycling between two small masks. The one exception is the PMC on
    mote 48, which masks 89 % of the suspect rows, none of the climb, and
    still changes after 10 refits.
  - **Hampel `initial_mask`.** It does not help, because it catches spikes,
    not drifts. The only exception is the PMC on mote 48 (258 masked: every
    suspect row and 22 % of the climb).
  - **Other starts.** Starting from the clean-window model does not help
    either (6 of 6 break down).
  - **Threshold `initial_mask`.** A sensor-level pre-mask (the 60 °C
    threshold) gets 3 of 6 fits to mask every suspect row: HMC-IN on motes
    48 and 47, and the PMC on mote 47. Only the HMC-IN on mote 48 also masks
    the climb. On mote 22 (19 % of the window failing), the HMC-IN releases
    the pre-mask round after round and ends at the raw fit.
* **Even the threshold oracle is contaminated.** Masking only the suspect
  rows leaves the climb in: the top state keeps an sd of 6–8 °C, against
  1.7–2.7 °C once the climb is masked too (HMC-IN).
* **A contamination component is needed for estimation, not for detection.**
  The failure is persistent: its fitted stay probability is ≥ 0.80, and
  ≥ 0.9998 for the HMC-IN. It is also broad: sd 28–45 °C in 5 of the 6 raw
  fits. The indicator must therefore be Markov, not Bernoulli, and its law
  must be fixed, not a regular state that ICE can reshape. Conclusion
  below.
* **Three library problems found** (below, each with a reproduction).
  1. **Clipped corner** (left as is, documented). The non-gated PIT of a
     reading that follows another reading beyond F⁻¹(1 − EPS) is about 0.5
     for the Gaussian and Gumbel–Hougaard copulas, although its log
     predictive density is about −1 200. It explains the 0.51–0.53
     non-sequential recall of the PMC on motes 48 and 47. The sequential
     (default) flags are not affected.
  2. **Gap quadrature** (fixed at 39f249f). With strongly dependent copulas
     (τ ≈ 0.99–0.997 at 30 s), the quadrature had not converged at the
     default `gap_nodes = 64`, nor at 512.
     - The PMC log-likelihood of mote 48's clean days 1–10 was 51 798,
       57 574, 63 096 and 65 364 nats at G = 64, 128, 256 and 512.
     - The normal scores of the rows after a gap had an sd of 1.53, 1.29,
       1.04 and 0.80.
     - Every PMC fit, PIT and gated flag of the study used G = 64.
  3. **Leading gap** (open). When a series starts with missing rows, the
     first reading after them is misintegrated under strong dependence,
     with no WARNING. On a 400-row simulated series with a 17-row leading
     gap, the error is −4.3 nats at τ = 0.997 and −22 nats at τ = 0.999
     (G = 64).

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
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/robust.py --jobs 3
OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/intel_lab/detect.py --check --jobs 4
.venv/bin/python report/erroneous_data/intel_lab/summarise.py
.venv/bin/python report/erroneous_data/intel_lab/repro_clipped_corner.py   # library problem 1
.venv/bin/python report/erroneous_data/intel_lab/repro_gap_nodes.py        # library problem 2 (≈ 20 s)
.venv/bin/python report/erroneous_data/intel_lab/repro_leading_gap.py      # library problem 3 (≈ 20 s)
```

The scripts now keep the WARNINGs of pmcprg instead of hiding them. The
CSVs committed here predate that change and lack these columns. Every
task counts the quadrature WARNINGs of its passes and records `quad_error`
against its threshold:

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

The wall times on an Apple arm64 laptop (Python 3.14.7, numpy 2.5.3, scipy
1.18.1) add up to 32 min:

* `fit_clean.py`: 2.1 min with 4 processes;
* `detect.py`: 8.3 min with 4 processes (33 min of CPU);
* `robust.py`: 22.1 min with 3 processes (63 min of CPU).

The one-off quadrature checks (G = 128, 256 and 512 on the clean days, and
the G = 256 detection runs) took about 25 more minutes. They are not
scripted beyond `repro_gap_nodes.py`.

The per-row flags and labels go to `results/cache/`, which is not
versioned. `detect.py --figures-only` redraws the figures from that cache.
Nothing is subsampled: the models run at the native 30 s resolution on
every observed epoch.

## Results

### 1. The clean window

| mote | model | BIC K=2 | BIC K=3 | K | means (°C) | sd (°C) | diagonal copulas (τ) | spread over starts (nats) |
|---|---|---|---|---|---|---|---|---|
| 48 | HMC-IN | 59224 | 43874 | 3 | 17.34 / 20.02 / 22.53 | 0.89 / 0.47 / 1.36 | – | 3 |
| 48 | PMC | −71434 | −72223 | 3 | 18.74 / 21.10 / 21.50 | 1.96 / 2.08 / 2.36 | Gauss(0.996) / Gauss(0.992) / GH(0.965) | 273 |
| 22 | HMC-IN | 82622 | 67679 | 3 | 18.20 / 23.49 / 31.79 | 1.36 / 2.15 / 1.22 | – | 1 |
| 22 | PMC | −59229 | −59530 | 3 | 19.81 / 22.55 / 23.60 | 2.53 / 3.28 / 5.65 | Frank(0.994) / Clayton(0.996) / Clayton(0.997) | 382 |
| 47 | HMC-IN | 64679 | 47486 | 3 | 17.26 / 20.25 / 23.73 | 0.94 / 0.59 / 1.86 | – | 0 |
| 47 | PMC | −58417 | −61965 | 3 | 18.88 / 18.93 / 21.99 | 2.13 / 2.23 / 2.68 | Gauss(0.997) / Gauss(0.997) / Gauss(0.986) | 2328 |

* **K.** BIC chooses K = 3 for every mote and model. With about 17 600
  strongly dependent rows, BIC would keep adding states. K = 3 is the upper
  end of the range the study allows, and it is interpretable (next point).
* **Starts.** The PMC likelihood has many basins: the 3 starts end
  273–2 328 nats apart, as in `report/real_series`.
* **The PMC gains 55 000–64 000 nats of likelihood over the HMC-IN.** Its
  diagonal copulas have τ = 0.97–0.997: consecutive 30 s readings are
  almost deterministic given the previous one.

**Regimes.** The day/night split is real for the HMC-IN on all motes and for
the PMC on motes 48 and 47. It is weak for the PMC on mote 22.

| mote | model | night share by state (00–06 h) | day share by state (10–16 h) | largest gap |
|---|---|---|---|---|
| 48 | HMC-IN | 0.64 / 0.23 / 0.13 | 0.02 / 0.43 / 0.54 | 0.62 |
| 48 | PMC | 0.87 / 0.12 / 0.01 | 0.23 / 0.72 / 0.06 | 0.64 |
| 22 | HMC-IN | 0.86 / 0.14 / 0.00 | 0.00 / 0.71 / 0.29 | 0.86 |
| 22 | PMC | 0.21 / 0.12 / 0.67 | 0.04 / 0.18 / 0.78 | 0.17 |
| 47 | HMC-IN | 0.66 / 0.21 / 0.12 | 0.01 / 0.49 / 0.50 | 0.65 |
| 47 | PMC | 0.81 / 0.16 / 0.03 | 0.25 / 0.05 / 0.70 | 0.66 |

The two models split the day differently:

* **HMC-IN: level regimes.** The cool state holds the night, and the warm
  states hold the day.
* **PMC: dynamics regimes.** On motes 48 and 47, a calm, strongly dependent
  state holds the nocturnal cooling (81–87 % of the night rows), and the
  other states hold the more agitated daytime. The margin means of the PMC
  states are within 3 °C of each other (`figures/clean_regimes.png`).

**Calibration on the held-out days 8–10** (`figures/clean_pit.png`):

| mote | model | n | KS D (p) | LB(10) z (p) | LB(10) z² (p) | z mean | z sd | acf1(z) | flags α=1e-2 / 1e-3 / 1e-4 (expected) | sequential flags α=1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 48 | HMC-IN K=3 | 7551 | 0.247 (< 1e-300) | 74631 (< 1e-300) | 75353 (< 1e-300) | 0.76 | 1.61 | 1.00 | 911 / 678 / 393 (76 / 8 / 0.8) | 694 |
| 48 | PMC K=3 | 7551 | 0.076 (1.5e-38) | 1810 (< 1e-300) | 181 (1.6e-33) | 0.05 | 0.85 | 0.10 | 32 / 22 / 6 (76 / 8 / 0.8) | 239 |
| 22 | HMC-IN K=3 | 7736 | 0.235 (< 1e-300) | 74996 (< 1e-300) | 74146 (< 1e-300) | 0.55 | 1.04 | 1.00 | 34 / 0 / 0 (77 / 8 / 0.8) | 0 |
| 22 | PMC K=3 | 7736 | 0.089 (1.4e-53) | 1970 (< 1e-300) | 1345 (9.4e-283) | −0.06 | 0.81 | 0.07 | 1 / 0 / 0 (77 / 8 / 0.8) | 0 |
| 47 | HMC-IN K=3 | 7605 | 0.148 (5.9e-146) | 74346 (< 1e-300) | 75703 (< 1e-300) | 0.53 | 1.36 | 1.00 | 730 / 384 / 266 (76 / 8 / 0.8) | 394 |
| 47 | PMC K=3 | 7605 | 0.032 (4.9e-07) | 1830 (< 1e-300) | 3329 (< 1e-300) | 0.01 | 0.91 | 0.06 | 45 / 21 / 2 (76 / 8 / 0.8) | 513 |

* **No model is calibrated in the sense of the tests.** With n ≈ 7 600 and
  30 s data, KS and Ljung–Box reject everything.
* **The magnitudes separate the models.**
  - **HMC-IN.** Its normal scores are a smooth curve (acf1 = 1.00). It is
    biased on the held-out days (z mean 0.53–0.76), which were warmer than
    the fit days. At α = 1e-3 it flags 384–678 held-out rows instead of 8 on
    motes 48 and 47. It is a level test whose tail is the fit week's range.
  - **PMC.** Nearly uniform (KS D ≤ 0.09), slightly too wide (z sd
    0.81–0.91, a hump in the PIT histogram). Residual dependence: acf1 =
    0.06–0.10, about 0.2 at lag 2, decaying slowly to about 0.1 at lag 60,
    the momentum of warming and cooling phases. Its non-sequential flag
    counts are within a factor of 3 of nominal: 0–22 at α = 1e-3 against 8
    expected.
* **Sequential flags on clean data: lock-out, partly a quadrature
  artifact.** At the default G = 64, the sequential PMC flags 239 (mote 48)
  and 513 (mote 47) held-out rows at α = 1e-3, against 22 and 21 without
  gating.
  - **Mote 47, 03-07 afternoon.** One row flagged during a steep
    legitimate warming starts a 4.3 h run of 455 flags. Gated, the filter
    predicts the next rows through the gap quadrature, which is too coarse
    at G = 64: at G = 256 the mote's held-out total falls to 52 (20
    without gating).
  - **Mote 48.** The lock-out is genuine: 265 sequential against 4
    non-sequential flags at G = 256. They form one 2.6 h episode from 15:05
    on 03-07, over an afternoon peak of 29.1–30.4 °C, above the fit week's
    maximum of 27.9 °C.

  The simulation study measured a gating cost of 0.3 extra flags per 1 000
  rows on its fixtures. On this near-unit-root series it reaches 35 per
  1 000 held-out rows (mote 48, G = 256).
* **The PMC PIT after a gap is quadrature-limited.** For the 933 held-out
  rows of mote 48 that follow a gap, the sd of the normal scores is 1.53,
  1.29, 1.04 and 0.80 at G = 64, 128, 256 and 512, against 0.70–0.78 after
  an observed row. Part of the PMC's KS and Ljung–Box rejection above comes
  from these rows (12 % of the held-out rows).

### 2. Detection on the later window

The clean-window models are held fixed. Values are for motes 48 / 22 / 47.
Normal operation covers 29 448, 26 937 and 28 658 observed rows (14–15
days per mote).

| method | recall (suspect) | precision (suspect) | precision (suspect + climb) | share of climb rows flagged | normal-operation flags (episodes) |
|---|---|---|---|---|---|
| PMC seq α=1e-3 | 1.00 / 1.00 / 1.00 | 0.80 / 0.97 / 0.69 | 0.85 / 0.99 / 0.81 | 0.93 / 0.22 / 1.00 | 425 (3) / 0 (0) / 484 (20) |
| PMC seq α=1e-4 | 1.00 / 1.00 / 1.00 | 0.94 / 0.98 / 0.82 | 1.00 / 1.00 / 0.97 | 0.87 / 0.14 / 1.00 | 0 (0) / 0 (0) / 22 (2) |
| PMC seq BH α=1e-3 | 1.00 / 1.00 / 1.00 | 0.94 / 0.98 / 0.88 | 1.00 / 1.00 / 0.97 | 0.87 / 0.15 / 0.59 | 0 (0) / 0 (0) / 21 (2) |
| PMC nonseq α=1e-3 | 0.51 / 1.00 / 0.53 | 0.91 / 0.97 / 0.84 | 0.99 / 0.99 / 0.94 | 0.60 / 0.21 / 0.36 | 14 (4) / 0 (0) / 59 (21) |
| PMC nonseq BH α=1e-3 | 0.51 / 1.00 / 0.53 | 0.93 / 0.98 / 0.96 | 1.00 / 1.00 / 0.99 | 0.50 / 0.15 / 0.10 | 0 (0) / 0 (0) / 3 (2) |
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
| PMC seq α=1e-3 | 11.1 / 1.2 / 19.1 | 11.1 / 16.6 / 23.0 | 273.8 / 16.6 / 367.8 |
| PMC seq α=1e-4 | 9.9 / 0.8 / 17.4 | 9.9 / 15.3 / 17.4 | 9.9 / 15.3 / 102.7 |
| PMC seq BH α=1e-3 | 9.7 / 0.9 / 13.8 | 9.7 / 15.3 / 17.4 | 9.7 / 15.3 / 102.7 |
| PMC nonseq α=1e-3 | 11.1 / 1.2 / 19.1 | 11.1 / 16.6 / 23.0 | 273.8 / 16.6 / 367.8 |
| HMC-IN seq α=1e-3 | 13.4 / 6.5 / 15.2 | 13.4 / 16.3 / 15.2 | 276.0 / 16.3 / 271.1 |
| threshold 35 °C | 5.3 / 6.8 / 13.1 | 5.3 / 16.4 / 13.1 | 5.3 / 16.4 / 13.1 |
| threshold 40 °C | 3.5 / 3.8 / 8.1 | 3.5 / 15.8 / 10.6 | 3.5 / 15.8 / 10.6 |
| threshold 50 °C | 1.5 / 1.3 / 1.6 | 1.5 / 1.3 / 1.6 | 1.5 / 1.3 / 1.6 |
| clean envelope ±2 °C | 10.6 / 6.0 / 15.2 | 10.6 / 16.2 / 15.2 | 273.2 / 16.2 / 270.4 |
| Hampel t=4 | −0.0 / −1.2 / −4.0 | 23.4 / 23.8 / 23.9 | 370.8 / 325.8 / 372.5 |
| robust z 24 h t=4 | 14.2 / 2.1 / 16.2 | 14.2 / 17.2 / 16.2 | 378.0 / 351.8 / 374.0 |

**Sensitivity of the PMC to the gap quadrature** (α = 1e-3, one-off runs
with `gap_nodes = 256` instead of the default 64; the same fitted models):

| PMC, α = 1e-3 | G | flags | recall | share of climb rows flagged | normal-operation flags (episodes) | lead, episode (h) |
|---|---|---|---|---|---|---|
| sequential | 64 | 2952 / 4981 / 3315 | 1.00 / 1.00 / 1.00 | 0.93 / 0.22 / 1.00 | 425 (3) / 0 (0) / 484 (20) | 11.1 / 1.2 / 19.1 |
| sequential | 256 | 2527 / 4984 / 2870 | 1.00 / 1.00 / 1.00 | 0.93 / 0.22 / 0.94 | 0 (0) / 0 (0) / 164 (19) | 11.1 / 1.2 / 19.1 |
| non-sequential | 64 | 1330 / 4977 / 1430 | 0.51 / 1.00 / 0.53 | 0.60 / 0.21 / 0.36 | 14 (4) / 0 (0) / 59 (21) | 11.1 / 1.2 / 19.1 |
| non-sequential | 256 | 1281 / 4980 / 1312 | 0.51 / 1.00 / 0.53 | 0.41 / 0.21 / 0.09 | 0 (0) / 0 (0) / 49 (19) | 2.7 / 1.2 / 1.3 |

Recall, the sequential lead times and the clipped-corner recall of the
non-sequential run are unchanged at G = 256. The false-alarm counts are
not. The mote-48 afternoon episodes vanish, and mote 47 keeps 164
sequential flags in 19 episodes. The non-sequential climb coverage and lead
fall on motes 48 and 47: at G = 64 part of those flags came from the coarse
quadrature after a missing epoch. G = 256 has not converged either (the
log-likelihood still moves by 2 300 nats from 256 to 512 on mote 48's
clean days). The PMC false-alarm counts are therefore uncertain: lower at
G = 256 (0 and 164 against 425 and 484 on motes 48 and 47), but not
bounded.

The figures are `figures/detect_overview.png` (the whole window),
`figures/detect_zoom.png` (36 h before to 6 h after the failure),
`figures/detect_pvalues.png` and `figures/detect_false_alarms.png`.

* **The model flags come early.** The sequential PMC and the HMC-IN flag
  every suspect reading and most of the climb, starting 11–19 h before the
  threshold hit on motes 48 and 47. That is 2–4 times the warning of a 40 °C
  threshold. Both behave like an adaptive level threshold: the climb's
  30 s increments are ordinary, and what gives it away is the level
  relative to the clean margins.
  - **PMC.** Its flags start when the level leaves the bulk of the clean
    margins: about 30 °C on motes 48 and 47, where the top state's mean
    plus 4 sd is 30.9 and 32.7 °C. Above F⁻¹(1 − EPS) (40.6 and 43.8 °C)
    the margins put no mass at all. Its innovation test adds flags wherever
    the climb steepens.
  - **Mote 22 is where the PMC is late (0.9–1.2 h).** Its clean-window PMC
    has a state with sd 5.65 °C, so the climb to about 55 °C stays inside
    the margins, and the Clayton copulas follow it.
* **The HMC-IN pays for its lead in false alarms.** Its lead is the longest
  on mote 48 (13.4 h). Its tail threshold (about 27–30 °C on motes 48 and
  47) sits inside the range of the warm afternoons of the next two weeks:
  3 episodes per mote, 900–1 300 flags. The clean envelope ± 2 °C is the
  same test without a model, with nearly the same numbers.
* **Examined false alarms of the PMC** (`results/alarm_episodes.csv`, every
  episode listed in `results/tables.md`):
  - **Mote 48.** 3 episodes, on 03-13, 03-14 and 03-20 between 14:47 and
    18:00. The temperature is 28–30 °C, the smooth peak of a normal diurnal
    cycle, up to 2 °C above the fit-week maximum of 27.9 °C (see the
    figure). These are real temperatures. The sequential run turns 3–6
    flags into 171–241 (lock-out). All of them vanish at G = 256: they are
    quadrature artifacts on real peaks.
  - **Mote 47.** 20 episodes, 17 of them starting between 13:00 and
    18:30. They start with a 30 s step of 0.2–0.6 °C. Several continue with
    a rise of 2–7 °C within minutes, then return to the daily curve: fast
    afternoon transients, probably sun on the sensor. They are real, not
    faults. The 03-20 episode (310 sequential flags) is a 31.7 °C afternoon
    peak.
  - **Summary.** No PMC false alarm in normal operation is a sensor glitch.
    All are real temperatures that the fit week had not shown: warmer
    peaks, faster transients. With BH at α = 1e-3 or with α = 1e-4, only
    2 episodes remain (mote 47).
* **Flags before the threshold are mostly true early drift.**
  - Between the climb onset and the first suspect reading, the sequential
    PMC flags 93 % (mote 48) and 100 % (mote 47) of the rows. The HMC-IN
    flags 100 % on both motes.
  - In the ambiguous zone, the sequential PMC flags 0, 24 and 159 rows.
    - **Mote 47.** The 159 flags (60 at G = 256) run from 22:23 on 03-23 to
      the climb onset at 06:37, while the reading rises from 22 to 31 °C
      overnight. The mote's clean nights cool down, so this is probably
      early drift.
    - **Mote 22.** The 24 flags fall between 06:13 and 16:53 on 03-23: a
      35–40 °C bump, then 33–35 °C all day, then the climb. The bump looks
      like the legitimate morning jumps of this mote (35 °C on 03-08) and
      cannot be settled from this mote alone.
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
* **Non-sequential PMC: recall 0.51 and 0.53 on motes 48 and 47.** Every
  missed row directly follows another observed suspect row (1 147 and
  1 064 rows). Every suspect row that follows a missing epoch is flagged
  (1 207 of 1 207 on each mote). This is the clipped-corner problem below.
  On mote 22, whose top-state copula is a Clayton, the same rows get
  p ~ 1e-13 and are flagged: recall 1.00. The sequential run is not
  affected: the flagged predecessor is integrated out, so the filter never
  conditions on a clipped value.
* **Sequential vs non-sequential.**
  - **During a persistent failure.** Gating is what keeps a stuck sensor
    flagged: recall 1.00 against 0.51–0.53.
  - **During legitimate transients.** Gating turns a single flag into a
    run of tens to hundreds of flags. Part of this is the coarse quadrature
    that predicts the rows after a gated one; the genuine part is measured
    on the held-out peak of mote 48 (section 1).
  - **For the HMC-IN it barely matters.** No y-dependence: 3 846 against
    3 826 flags on mote 48.
  - **Corrections.** BH (or a smaller per-row α) is the practical fix for
    the lock-out on this data. Per-row α = 1e-2 makes it worse: 2 831
    normal-operation flags on mote 47.

### 3. Robust re-estimation on a window that mixes clean and failing readings

| mote | window (epochs) | observed | normal | ambiguous 24 h | climb | suspect | suspect + climb |
|---|---|---|---|---|---|---|---|
| 48 | 63 420–77 820 | 6 126 | 4 879 | 850 | 182 | 211 | 6.4 % |
| 22 | 60 226–74 626 | 8 946 | 5 451 | 1 786 | 623 | 1 079 | 19.0 % |
| 47 | 62 934–77 334 | 6 364 | 4 771 | 925 | 401 | 267 | 10.5 % |

The HMC-IN fits below are exact. The PMC fits and their flags use the
default `gap_nodes = 64`, which has not converged on this data (library
problem 2). The PMC's qualitative outcome, a failure state in every raw
fit, is unlikely to depend on the quadrature. The failing rows are 6–19 %
of the window and lie tens of degrees above the clean margins, while the
quadrature only changes how the gaps are integrated. The PMC's masks,
cycles and non-convergence may depend on it; this was not rerun.

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
library reports as degenerate):

| mote | fit | fits | masked (suspect / climb) | means (°C) | sd (°C) | failure state: mean, sd, stay | agreement with oracle_ext |
|---|---|---|---|---|---|---|---|
| 48 | raw | 1 | 0 | 19.97 / 21.08 / 33.26 | 2.33 / 2.33 / 28.46 | 33, 28, 0.918 | 0.53 |
| 48 | oracle | 1 | 211 (1.00 / 0.00) | 20.78 / 21.04 / 23.55 | 2.09 / 1.51 / 6.96 | – | 0.56 |
| 48 | oracle_ext | 1 | 393 (1.00 / 1.00) | empty / 20.20 / 21.49 | – / 1.57 / 2.43 | – | 1.00 |
| 48 | fm | 11, still growing | 187 (0.89 / 0.00) | 20.51 / 21.06 / 23.56 | 2.13 / 2.17 / 8.48 | – | 0.50 |
| 48 | fm_hampel | 11, not converged | 258 (1.00 / 0.22) | 20.44 / 20.50 / 22.85 | 1.50 / 1.89 / 3.93 | – | 0.55 |
| 48 | fm_thr60 | 11, not converged | 128 (0.60 / 0.01) | 20.24 / 22.23 / 28.55 | 1.58 / 2.91 / 18.06 | – | 0.76 |
| 48 | fm_cs | 11, not converged | 3 (0.01 / 0.00) | 19.65 / 21.39 / 35.67 | 2.13 / 2.00 / 29.94 | 36, 30, 0.853 | 0.67 |
| 22 | raw | 1 | 0 | 22.68 / 34.02 / 62.71 | 4.04 / 13.71 / 45.40 | 63, 45, 0.799 | 0.63 |
| 22 | oracle | 1 | 1079 (1.00 / 0.00) | 24.93 / 27.29 / 28.89 | 5.96 / 1.38 / 13.68 | – | 0.65 |
| 22 | oracle_ext | 1 | 1702 (1.00 / 1.00) | 22.03 / 23.73 / 25.09 | 2.66 / 5.98 / 5.78 | – | 1.00 |
| 22 | fm | 11, cycling | 0 | = raw | | 63, 45, 0.799 | 0.63 |
| 22 | fm_hampel | 11, not converged | 10 (0.01 / 0.00) | empty / 23.87 / 48.41 | – / 4.87 / 36.71 | 48, 37, 0.940 | 0.74 |
| 22 | fm_thr60 | 11, not converged | 809 (0.70 / 0.02) | 21.79 / 22.82 / 43.44 | 3.64 / 3.58 / 15.99 | 43, 16, 0.998 | 0.61 |
| 22 | fm_cs | 11, not converged | 28 (0.02 / 0.00) | 21.62 / 25.23 / 58.03 | 2.66 / 5.41 / 39.37 | 58, 39, 0.912 | 0.56 |
| 47 | raw | 1 | 0 | empty / 21.68 / 42.79 | – / 2.63 / 34.31 | 43, 34, 0.957 | 0.49 |
| 47 | oracle | 1 | 267 (1.00 / 0.00) | 21.64 / 21.86 / 28.04 | 2.44 / 0.19 / 9.03 | – | 0.53 |
| 47 | oracle_ext | 1 | 668 (1.00 / 1.00) | 18.03 / 21.22 / 22.88 | 0.25 / 1.65 / 2.56 | – | 1.00 |
| 47 | fm | 1, converged | 0 | = raw | | 43, 34, 0.957 | 0.49 |
| 47 | fm_hampel | 2, converged | 0 | = raw | | 43, 34, 0.957 | 0.49 |
| 47 | fm_thr60 | 11, not converged | 333 (1.00 / 0.13) | 21.31 / 21.85 / 23.25 | 1.73 / 2.25 / 5.64 | – | 0.45 |
| 47 | fm_cs | 11, not converged | 0 | 21.44 / 21.46 / 33.17 | 2.18 / 2.35 / 27.03 | 33, 27, 0.993 | 0.59 |

The `raw_cs` rows (the raw fit from the clean model) equal the first fit
of `fm_cs` and are in `results/tables.md`. The mask of every refit is in
`results/robust_fits.csv` (`masks_per_fit`). The states of the raw,
`oracle_ext`, `fm` and `fm_thr60` fits are drawn in
`figures/robust_hmc_in.png` and `figures/robust_pmc_state.png`.

* **The failing readings form their own state, as the simulation predicts.**
  Every raw fit has one:
  - HMC-IN: mean 59–120 °C, sd 3–37 °C, stay 0.9998–1.0000, an absorbing
    state;
  - PMC: mean 33–63 °C, sd 28–45 °C, stay 0.80–0.96.

  Starting ICE from the clean-window model changes nothing (`raw_cs`,
  `fm_cs`: a failure state in 6 of 6). The breakdown is a property of the
  likelihood, not of the k-means start.
* **Under that state almost nothing is flagged, and the loop ends at the
  raw fit**, usually at once. The other outcomes:
  - **Cycles.** On mote 22 (HMC-IN and PMC) the loop alternates between two
    masks (0 and 33 rows for the HMC-IN, 0 and 10 for the PMC) until
    `max_rounds`. The cycle is deterministic, so `converged = False`, and
    the result is the raw fit.
  - **Partial recovery.** Only the PMC on mote 48 escapes. Its raw failure
    state has the lowest mean (33 °C, sd 28 °C), and the first fit flags
    42 rows.
    The mask then changes unevenly (42, 98, 119, 105, 69, 113, 147, 137,
    176, 187 rows) and reaches 89 % of the suspect rows after 10 refits,
    still changing. It never masks the climb.
* **Hampel `initial_mask`.** It masks 31–130 rows, spikes in the 30 s noise
  and not the drift or the plateau, so the loop falls back into the failure
  basin (5 of 6). The exception is again the PMC on mote 48: after 10
  refits it masks every suspect row and 22 % of the climb. The simulation's
  remedy for frequent spikes does not transfer to a smooth persistent
  failure.
* **Threshold `initial_mask`.** The sensor-level flag is re-tested by the
  model. It recovers the `oracle` level where the contamination is modest:
  - **HMC-IN on mote 47.** 299 rows masked, means 18.90 / 22.26 / 29.45
    against the oracle's 18.89 / 22.26 / 30.15.
  - **HMC-IN on mote 48.** 416 rows masked, every suspect and climb row.
    The model extends the mask into the drift.
  - **PMC on mote 47.** Every suspect row masked, not converged.

  On mote 22 (19 % failing) the pre-mask is released round by round. The
  HMC-IN goes from 1 079 to 978, 887, 796, 755, 567 and 0 masked rows,
  then cycles at the raw fit: the climb, left in, forms a state at about
  47 °C (the `oracle` fit's) that makes the lower suspect readings
  plausible, which widens the state, and so on.
* **The threshold oracle is not clean either.** Masking only the suspect
  rows leaves the climb in the fit. The HMC-IN's top state keeps an sd of
  7.7, 6.4 and 6.1 °C, against 2.7, 2.5 and 1.7 °C when the climb is masked
  too. On mote 22 it keeps a 47 °C state. The partial ground truth biases
  the estimate as much as a mild contamination would.
* **Classification of the clean rows.**
  - **HMC-IN.** The level regimes are identifiable. Against `oracle_ext`,
    the raw fits agree on 71–88 % of the normal rows and the threshold
    oracle on 73–99 %. The raw fit spends a state on the failure and merges
    two day/night levels into the remaining two.
  - **PMC.** The dynamics regimes are not reproducible across fits. Even
    the two oracles agree on only 53–65 % of the normal rows, so the
    classification cannot rank the PMC fits here. Its parameters (the
    failure state, the top-state sd) can.

### 4. Conclusion

**What works.**

* **Detection with a model fitted on clean data, held fixed, and gated
  (sequential).** It flags every failing reading, and on 2 of 3 motes it
  starts 9.7–19.1 h before the 60 °C rule, depending on the setting.
* **Its false alarms are real events outside the training week**, not
  sensor noise. With BH at α = 1e-3, they are 0, 0 and 2 episodes in
  14–15 days (G = 64).
* **The copula model beats the HMC-IN on false alarms by one to two orders
  of magnitude.** With BH at α = 1e-3 it raises 0 and 21 flags on motes 48
  and 47, against 636–1 290 for the HMC-IN. At α = 1e-3 with G = 256 it
  raises 0 and 164. It gives 1–6 h less warning.

**What fails, or needs care.**

* **Calibration.** The tests reject on 30 s data. The PIT is usable as a
  score, not as an exact p-value.
* **Lock-out of the gated filter** after a legitimate transient. It is
  genuine on mote 48's held-out peak (265 against 4 flags at G = 256, one
  2.6 h episode) and inflated elsewhere by the gap quadrature. It needs a
  correction (BH or α = 1e-4); a re-acquisition rule would be the
  principled fix.
* **The gap quadrature** (library problem 2). At the default G = 64 it
  distorts the PMC's likelihood, fits and gated flags on this data, and
  even G = 512 has not converged.
* **The PMC's lead depends on the width of its clean margins.** It is 1 h
  on mote 22.
* **The non-gated copula PIT at the clipped corner** (library problem 1).
* **Rolling filters** (Hampel, trailing robust z) do not see a smooth
  drift or a stuck sensor.
* **Flag-and-mask estimation on a failing window.** It breaks down in
  almost every case (Hampel pre-screen included). It only recovers with a
  sensor-level pre-mask, and only where at most about 10 % of the window
  is failing.

**Is the contamination model needed?**

* **For detection, no.** A clean model plus gating already flags the
  failure; the missing piece is calibration, not a contamination law.
* **For estimation on data that contain a failure, yes.** Every fit gives
  the failure a regular state, and flag-and-mask cannot leave that basin.
  The data say which contamination model:
  - **The indicator must be Markov, not Bernoulli.** The failure lasts
    until the mote stops reporting (2 271–4 816 suspect readings per mote).
    The fitted stay probabilities are ≥ 0.80, and 0.9998–1.0 for the
    HMC-IN (about 1 exit in 5 000 rows); in reality the state is absorbing. A
    Bernoulli indicator at the rate that explains 10–19 % of a window
    would put isolated contaminated rows everywhere and would not
    explain a block.
  - **The broad law g must be fixed** (e.g. uniform over the sensor
    range), not re-estimated. A re-estimated g becomes the 33–120 °C state
    seen here, with an sd of 3–45 °C that ICE fits to the plateau.
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
  The gated filter integrates a flagged row out, and the grid path after a
  missing row uses quadrature nodes inside the margins.
* **Here.** It caps the non-sequential PMC's recall on motes 48 and 47
  at 0.51–0.53, with every missed row following an observed suspect
  row. No other number in this study depends on it.
* **Fix (not attempted, library code untouched).** Evaluate the
  conditional CDF with v unclipped, since h(1 | u) = 1 for every u, or
  take the upper tail from the margin's survival function when
  F(y_n) > 1 − EPS.

### 2. The gap quadrature does not converge for near-deterministic transitions

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

### 3. A leading gap under strong dependence is misintegrated, with no WARNING

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
  `gap_nodes = 64`, which has not converged on this data (library problem
  2). The PMC fits, BIC, PIT after a gap, gated flags and robust fits are
  affected. Only the key detection settings were checked at G = 256, which
  has not converged either.
* **Robust estimation.** One window per mote, K fixed at the clean
  selection (3). With K = 4 a failure state would not have to take a
  day/night level, which was not tried. Several `robust_estimate` runs
  stopped at `max_rounds` without converging, so their masks are the 11th
  fit's, not fixed points.

## Files

| file | content |
|---|---|
| `il_common.py` | constants, data loader, dequantisation, labels, model helpers, baselines, scores |
| `fit_clean.py` | step 1 → `results/fits.csv`, `results/models/*.toml` (the 12 best clean fits), `results/selected_models.json`, `results/pit_checks.csv`, `results/regimes_by_hour.csv`, `figures/clean_pit.png`, `figures/clean_regimes.png` |
| `detect.py` | step 2 → `results/detect_scores.csv` (one row per mote × method × setting), `results/alarm_episodes.csv`, `figures/detect_*.png`; `--check` → `results/detect_check.csv` |
| `robust.py` | step 3 → `results/robust_fits.csv`, `figures/robust_*.png` |
| `summarise.py` | `results/tables.md`, `results/tables.tex` |
| `repro_clipped_corner.py` | library problem 1 (clipped corner) |
| `repro_gap_nodes.py` | library problem 2 (gap quadrature, fixed at 39f249f) |
| `repro_leading_gap.py` | library problem 3 (leading gap) |
