# State-dependent missingness on real data: PAMAP2 (P6)

The data preparation of PAMAP2 found that hand-IMU dropouts depend on the
activity. This study measures that dependence on the three subjects of the
real-series study (102, 108, 105) and tests it with `missingness_lr_test`. It
then asks whether a model of it (`[missingness]`: `"state"` or
`"state-markov"`, estimated by ICE) improves classification and imputation.

Every number below is printed by `summarise.py` from the CSV files in
`results/`. `results/tables.md` holds every table in full, and `tables/*.tex`
holds the main tables in LaTeX (booktabs).

## Summary

* **The dependence is real, strong and in the same direction for all three
  subjects: rest vs active.** At 100 Hz, 0.24–4.3 % of the hand samples are
  missing during walking, running, Nordic walking and rope jumping, against
  0–0.15 % for lying, sitting, standing, ironing, cycling, stairs and
  vacuum cleaning. On the 2 Hz windows (≥ 10 % rule), 0–0.24 % of the rest
  windows are missing, against 0.8–4.9 % of the active ones. Tested with the
  labels, "state" beats a common rate with LR = 11–1118. **Which activity
  loses the most samples depends on the subject.** For 102 it is running and
  Nordic walking (vigorous 10.8 % vs locomotion 2.1 %), for 105 walking
  (locomotion 4.0 % vs vigorous 0.4 %), and for 108 the transient windows
  (2.4 %). The K = 3 ordering is therefore not stable, and within a group
  the rate varies by activity: under the "any" rule, 39–44 % of walking
  windows are missing, against 0–4 % for cycling, stairs and vacuum cleaning.
* **The gaps are bursty, and `"state-markov"` fits them far better than
  `"state"`.** At 100 Hz the persistence b is 0.1–0.7 while the rates are
  below 3 % (b/π = 11–755). On the windows, masks redrawn under `"state"`
  give bursts that are too short in every cell, and too many in 4 of 6 (e.g.
  102, "any" rule: 653 bursts [618, 683] against 487 observed).
  `"state-markov"` reproduces the number and mean length of the bursts. It
  does not reproduce their clustering in time: the index of dispersion over
  10 s blocks is 3.3–8.0 observed against 1.4–5.0 redrawn (p = 0.01
  everywhere), and at 100 Hz the onset dispersion also exceeds the Markov
  value for all three subjects.
  Within an activity, the dropout rate varies slowly over time.
* **The LR test detects the mechanism on the real series** (unsupervised
  HMC-IN; the profile statistic of library commit 0758e3a; details in
  section 2). With the ≥ 10 % rule, "state" vs common gives LR = 30–170 for
  102 and 105 (p_boot = 0.01, the smallest possible with B = 99), and
  "state-markov" vs "state" gives LR = 61–145 for all three subjects.
  Subject 108 is borderline for the dependence on the state: at K = 2,
  "state" LR = 5.6 (p χ² 0.018, bootstrap 0.03) and "state-markov" LR = 4.2
  (0.12 / 0.15). With the "any" rule every test gives LR = 70–827. Every
  series has 69–558 bursts, and HMC-IN is not a grid variant, so the known
  over-rejection with few bursts (grid variant, ~9 bursts) does not apply.
  The fitted states are intensity levels, not activities: at K = 2 they
  match rest/active with 10–22 % error, but at K = 3 the error is 23–44 %.
  The fitted rates follow intensity (e.g. 102, K = 3: π = 0.05 / 0.03 /
  3.5 %).
* **Classification: using the mask helps when the model is right, and hurts
  on the real series.**
  * *Controlled experiment* (masks drawn on the true groups at the real rates):
    - oracle HMC-IN: the error at the missing windows drops by 1.2–3.4 points
      in all 16 cells (paired s.e. 0.15–0.65), and the log-loss by 0.12–0.27;
    - PMC pair oracle: the drop is 2.2–5.2 points, and `"state-markov"` beats
      `"state"` on Markov masks (−3.85 vs −3.13 and −5.15 vs −4.02).
  * *Real gaps, K = 2*: the oracle already gets the missing windows right
    (1.1–1.8 % error), and the mask changes that by at most 0.7 points (PMC
    pair, "any" rule: 1.5 → 0.8 %).
  * *Real gaps, K = 3*: the mask hurts.
    - For subject 102 ("any" rule), the oracle HMC-IN error at the missing
      windows goes from 2.1 % to 34.1 % with `"state"` and to 33.6 % with
      `"state-markov"`. The PMC pair oracle of 108 goes from 2.1 % to 44 %.
    - The cause: whole walking bouts flip to "vigorous" (89–100 % error on
      walking's missing windows, 48–97 % on its observed windows). Walking's
      dropout is that of the vigorous group, while the other locomotion
      activities lose almost nothing.
    - A group-level mechanism adds a small likelihood ratio at every window
      of a bout, and over hundreds of windows the sum overwhelms the
      observations. `"state-markov"` removes the over-counting inside a
      burst, not this accumulation across a bout.
* **Unsupervised: estimate the mechanism from the ignorable fit.** Started
  from the k-means start, ICE with `missingness = "state"` or
  `"state-markov"` ends more than 1 nat *below* "ignorable fit + common mask"
  in up to 17 % of the HMC-IN fits (8–17 % at K = 2; down to −100 nat), in
  another basin.
  - One such fit (105, K = 3, "any" rule): the error at the missing windows
    goes from 5.4 % to 71.7 %.
  - Warm-started from the ignorable fit, as `missingness_lr_test` does for
    its alternatives, the fit always gains (at least +0.4 nat; median 25–160
    nat) and keeps the segmentation.
  - The same holds for the library's nested test: its `"state"` null is
    fitted from the start (section 2).
* **Posteriors at the missing windows are overconfident in every setting.**
  The observed MPM error is 5–25 times the predicted one (1 − max posterior)
  for the oracles, and 12–70 times for the unsupervised fits. The mechanism
  lowers both, but does not fix the ratio.
* **Imputation: using the mask is slightly worse in 53 of 56 comparisons.**
  - The RMSE of the posterior mean rises by up to 0.10 (median 0.017), the
    CRPS by up to 0.04, and the 90 % coverage falls by up to 0.02. The 3
    exceptions are unsupervised K = 2 fits, better by at most 0.005.
  - For example, oracle HMC-IN, K = 3, Markov masks at the "any" rates: RMSE
    0.658 ignorable, 0.728 `"state-markov"`, 0.745 `"state"`. PMC pair:
    0.496, 0.549, 0.573.
  - On Markov masks, `"state-markov"` is always the less harmful of the two;
    on "state" masks the two are within 0.005.
  - The cause: at a missing window, the state posterior of the ignorable
    model follows the neighbouring windows. It tracks the local level of the
    log-SD, which HMC-IN does not otherwise model. On 102, imputing the true
    group mean gives RMSE 0.72 on the locomotion windows, against 0.51 for
    the ignorable posterior. The mask pulls the posterior towards the label
    and away from that level.

**Verdict.**
* State-dependent missingness holds on PAMAP2, and it is bursty:
  `"state-markov"` is the mechanism that fits, and `"state"` is rejected
  (nested LR = 61–735; 735 and 520 measured after the fix of the nested
  test's null, see "Library" below).
* For classification it pays only when the dropout is homogeneous within
  each model state. That is true in the controlled experiment (1–5 points
  fewer errors at the missing windows), and it holds at K = 2 on the real
  data (no loss). It is false for the K = 3 groups of the real series, where
  walking behaves like a vigorous activity, and the loss there is large.
* For imputation it costs a little.
* Use the mask with states whose dropout rates are homogeneous, estimate it
  from the ignorable fit, and check it (segmentation, calibration) against
  the ignorable model.

## Data and design

The windows, feature, groups, gap rules and models are those of the
real-series study (`report/real_series/`, imported from `rs_common.py`), so
the numbers are comparable:

* **Series.** Windows of 50 samples of the 100 Hz hand accelerometer
  (2 Hz), Y = log of the SD of ‖a‖. Subjects 102, 108 and 105 have 8 940,
  8 160 and 7 495 windows. The data stay outside the repository.
* **Groups.** K = 2: rest / active. K = 3: rest / locomotion / vigorous
  (running, Nordic walking and rope jumping are vigorous). Activity 0
  (transient) is kept in the series and scored nowhere.
* **Gap rules.** A window is missing when ≥ 5 of its 50 samples are NaN
  (`ge10`, the main rule of the real-series study: 100, 101 and 169 windows,
  1.2–1.9 %) or when any sample is NaN (`any`, the sensitivity rule: 490–924
  windows, 6.5–11.1 %). The real-series study also added MCAR blocks. They
  are not used here: they would dilute the mechanism being tested.
* **Models.**
  - Oracles: one supervised M-step on the labels of the complete series,
    HMC-IN or PMC pair margins (`rs_common.supervised_model`). The
    transitions are observed only between labelled windows, so A is the
    identity plus the 10⁻⁴ floor, a very sticky prior.
  - Unsupervised: ICE "available", `max_iter = 50`, `tol = 1e-4`, the
    k-means start and the two library multistart draws. The best of the 3
    starts is kept by its own likelihood.
  - Unsupervised PMC pair K = 3 (the BIC choice of the real-series study):
    k-means start only.
* **Mechanisms.**
  - In the oracles: `estimate_state` / `estimate_state_markov` (the library
    M-step, with its guard) on the one-hot labels, transient rows with zero
    weight.
  - Unsupervised: the ICE config key `missingness`, estimated either from the
    start itself or from the ignorable fit of the same start (**warm**).
* **Seeds.** `zlib.crc32` of named tuples. Masks, FFBS draws and the
  bootstrap streams have distinct seeds.

## 1. The real gaps

### 1a. 100 Hz, per activity (% of hand samples missing / number of bursts / persistence b)

| activity | 102 | 108 | 105 |
|---|---|---|---|
| lying / sitting / standing / ironing | 0.004 / 0.009 / 0.016 / 0.038 % | 0.008 / 0.100 / 0.095 / 0.152 % | 0.051 / 0.141 / 0.032 / 0.127 % |
| cycling / asc. / desc. stairs / vacuum | 0.004 / 0.006 / 0.039 / 0.010 % | 0.039 / 0 / 0 / 0.103 % | 0.045 / 0.042 / 0.031 / 0.029 % |
| walking | 2.43 % (558 bursts, b = 0.29) | 1.54 % (385, 0.21) | **3.51 %** (679, 0.40) |
| running | **4.28 %** (260, 0.34) | 1.17 % (168, 0.13) | 0.24 % (50, 0.17) |
| Nordic walking | 3.51 % (685, 0.34) | 0.40 % (98, 0.14) | 0.29 % (59, 0.21) |
| rope jumping | 1.42 % (139, 0.26) | 0.90 % (61, 0.23) | 0.45 % (21, 0.40) |
| transient | 0.15 % | **0.79 %** (567, 0.51) | 0.12 % |

Spearman correlation of the 12 activity rates between subjects: 0.55–0.76.
The split is stable: in each subject, every dynamic activity loses more
than every static one, by factors from 1.7 (105: running 0.24 % vs sitting
0.14 %) to over 1 000. The ranking among dynamic activities is not stable.
Figure `figures/gaps_by_activity.png`.

**Bursts at 100 Hz.**
- The persistence b = P(missing | previous missing) is 0.11–0.71 per group,
  against missing rates ≤ 3.1 %: b/π = 11–755. The mask is far from
  independent.
- The bursts are short: mean length 1.1–3.4 samples, and 62–88 % are single
  samples.
- The per-activity `"state-markov"` does not describe them fully. Redrawn
  masks (40 per subject) give fewer single-sample bursts (0.63–0.67 against
  0.70–0.75 observed) and shorter longest bursts (8–15 against 20–77).
- They also cluster less in time. The dispersion of burst onsets is
  2.7–4.2 per 1 s block (1.3–2.4 redrawn) and 14–29 per 10 s block
  (4.6–15.4 redrawn), outside the redrawn range for all three subjects.

### 1b. 2 Hz windows, per K = 3 group (π = missing rate; a, b = onset, persistence)

| subject | rule | rest π | locomotion π (a, b) | vigorous π (a, b) | transient π | whole series: bursts, mean length |
|---|---|---|---|---|---|---|
| 102 | ge10 | 0 | 2.07 % (1.84 %, 0.13) | **10.8 %** (9.2 %, 0.24) | 0.27 % | 135, 1.25 |
| 105 | ge10 | 0.24 % | **4.02 %** (2.80 %, 0.33) | 0.43 % (0.43 %, 0) | 0.15 % | 69, 1.45 |
| 108 | ge10 | 0.19 % | 0.83 % (0.79 %, 0.06) | 0.83 % (0.84 %, 0) | **2.43 %** | 84, 1.20 |
| 102 | any | 0.75 % | 13.3 % (7.2 %, 0.53) | **46.7 %** (42.9 %, 0.51) | 3.5 % | 487, 1.90 |
| 105 | any | 1.37 % | **13.8 %** (7.0 %, 0.57) | 8.2 % (7.1 %, 0.21) | 3.2 % | 290, 1.69 |
| 108 | any | 1.85 % | 13.4 % (9.7 %, 0.37) | **22.8 %** (18.9 %, 0.36) | 11.8 % | 558, 1.62 |

### 1c. Do the mechanisms reproduce the window masks?

Masks were redrawn 200 times with `pmcprg.missing.state_dependent` or
`state_markov` on the true class path, with the plain per-class rates
(classes = K = 3 groups + transient, or the 13 activities). The table gives
the observed statistic and the median [2.5 %, 97.5 %] of the redrawn masks,
for classes = K = 3 groups:

| subject, rule | bursts: obs. / "state" / "state-markov" | share of missing windows in runs ≥ 2 | longest run | dispersion, 10 s blocks |
|---|---|---|---|---|
| 102, ge10 | 135 / 158 [137, 178] / 136 [117, 156] | 0.33 / 0.15 / 0.36 | 6 / 3 / 4 | 4.8 / 2.1 / 2.4 [1.9, 2.8] |
| 105, ge10 | 69 / 96 [78, 114] / 69 [52, 84] | 0.47 / 0.07 / 0.51 | 8 / 2 / 5 | 7.1 / 1.4 / 2.1 [1.7, 2.8] |
| 108, ge10 | 84 / 97 [76, 117] / 85 [65, 103] | 0.27 / 0.04 / 0.30 | 6 / 2 / 3 | 3.3 / 1.1 / 1.4 [1.2, 1.8] |
| 102, any | 487 / 653 [618, 683] / 488 [453, 521] | 0.64 / 0.47 / 0.71 | 23 / 8 / 10 | 8.0 / 4.4 / 5.0 [4.5, 5.5] |
| 105, any | 290 / 438 [402, 474] / 289 [258, 316] | 0.56 / 0.20 / 0.61 | 12 / 4 / 10 | 7.1 / 1.7 / 2.7 [2.2, 3.3] |
| 108, any | 558 / 769 [731, 812] / 558 [521, 595] | 0.55 / 0.27 / 0.62 | 18 / 4 / 8 | 5.5 / 1.6 / 2.2 [2.0, 2.6] |

- `"state"` is rejected by the run-length statistics in every cell (mean
  length, share in runs, longest run: p ≤ 0.02). It also gives too many
  bursts in 4 of 6 cells.
- `"state-markov"` matches the number of bursts and their mean length.
  Their share of the missing windows is matched too, except with the "any"
  rule, where it is slightly high.
- `"state-markov"` misses the longest runs with the "any" rule, and in every
  cell it misses the clustering of the gaps in time: the 10 s dispersion is
  outside the redrawn range (p = 0.01).
- Per-activity classes do not close the gap (dispersion redrawn 1.5–6.5
  against 3.3–8.0).

## 2. Likelihood-ratio tests

### 2a. Supervised view: the mechanisms fitted on the true groups

Plain MLE; groups ordered rest / (locomotion /) vigorous. The label-based LR
(df) of the three tests is also given. The Bernoulli ("state") likelihood
treats the windows as independent.

| subject | rule | K | π (%) | a (%) | b | state vs common | markov vs common | markov vs state |
|---|---|---|---|---|---|---|---|---|
| 102 | ge10 | 3 | 0.00 / 2.07 / 10.82 | 0.00 / 1.84 / 9.24 | 0.20 / 0.13 / 0.24 | 263 (2) | 212 (4) | 31 (3) |
| 105 | ge10 | 3 | 0.24 / 4.02 / 0.43 | 0.19 / 2.80 / 0.43 | 0.20 / 0.33 / 0.00 | 109 (2) | 72 (4) | 97 (3) |
| 108 | ge10 | 3 | 0.19 / 0.83 / 0.83 | 0.14 / 0.79 / 0.84 | 0.25 / 0.06 / 0.00 | 11 (2) | 15 (4) | 11 (3) |
| 102 | any | 3 | 0.75 / 13.29 / 46.65 | 0.75 / 7.22 / 42.91 | 0.00 / 0.53 / 0.51 | 1118 (2) | 748 (4) | 340 (3) |
| 105 | any | 3 | 1.37 / 13.83 / 8.19 | 1.29 / 6.98 / 7.06 | 0.07 / 0.57 / 0.21 | 271 (2) | 167 (4) | 405 (3) |
| 108 | any | 3 | 1.85 / 13.36 / 22.81 | 1.79 / 9.68 / 18.90 | 0.05 / 0.37 / 0.36 | 398 (2) | 282 (4) | 152 (3) |

K = 2 rows: `results/tables.md`. Rest vs active: π = 0.00–0.24 % vs 0.83–4.87 %
(ge10) and 0.75–1.85 % vs 11.9–24.0 % (any).

### 2b. `missingness_lr_test` on the real series (unsupervised HMC-IN)

HMC-IN from the k-means start of the real-series study, with the test's ICE
defaults (`tol = 1e-8`, `patience = 500`).

* **Which statistic.** The numbers below use the **profile statistic** of
  library commit `0758e3a`: profile log-likelihoods of the mechanism at both
  fits' θ, LR ≥ 0 (`results/lr_tests.csv`, rerun after that commit).
* **The first run** used the earlier statistic, the difference of the two
  ICE fixed points (`results/lr_tests_fits_statistic.csv`). The rerun keeps
  that statistic too, as `LR_fits` (the library's `statistic_fits`). On
  these HMC-IN fits the two differ by at most 2.3
  (102, K = 2, "state-markov" vs common: 113.2 against 110.9), and every
  conclusion is the same.
* **p-values.**
  - p χ²: from the profile statistic.
  - p boot for subject 108 (ge10): a new bootstrap of the profile statistic,
    B = 99.
  - p boot for the other cells: the first run's bootstrap of the fits'
    statistic, B = 99 (ge10) or 19 (any), so the smallest value is 0.01 or
    0.05. Every one of those LR is far beyond the largest bootstrap
    replicate.
* **Bursts** (runs of missing windows): 135 / 69 / 84 (102 / 105 / 108,
  ge10) and 487 / 290 / 558 (any).

Rule `ge10`:

| K | subject | M (bursts) | "state" vs common: LR (p χ², p boot) | "state-markov" vs common | "state-markov" vs "state" | states vs groups (error) |
|---|---|---|---|---|---|---|
| 2 | 102 | 169 (135) | 134.8 (4e-31, 0.01) | 113.2 (3e-25, 0.01) | 89.9 (3e-20, 0.01) | 0.10 |
| 2 | 105 | 100 (69) | 30.0 (4e-8, 0.01) | 19.4 (6e-5, 0.01) | 145.2 (3e-32, 0.01) | 0.22 |
| 2 | 108 | 101 (84) | **5.6 (0.018, 0.03)** | **4.2 (0.12, 0.15)** | 61.2 (5e-14, 0.01) | 0.19 |
| 3 | 102 | 169 (135) | 170.5 (9e-38, 0.01) | 140.5 (2e-29, 0.01) | 81.5 (1e-17, 0.01) | 0.38 |
| 3 | 105 | 100 (69) | 114.1 (2e-25, 0.01) | 79.3 (2e-16, 0.01) | 121.0 (5e-26, 0.01) | 0.44 |
| 3 | 108 | 101 (84) | 9.7 (0.008, 0.01) | 9.6 (0.048, 0.04) | 62.4 (2e-13, 0.01) | 0.44 |

Rule `any`: LR = 129–827 ("state"), 70–438 ("state-markov" vs common) and
287–665 (nested; 520 and 735 for the two cells affected by the null-start
issue once it is fixed, see "Library" below). Every p χ² is below 1e-15, and every p_boot is 0.05 (the
first run, B = 19).

* **Estimated parameters (H1), states sorted by margin mean.**
  - 102, K = 2, ge10: π = 0.03 / 3.02 %; a = 0.06 / 2.47 %, b = 0.37 / 0.20.
  - 102, K = 3, any: π = 0.19 / 1.72 / 18.30 %; a = 0.14 / 1.89 / 11.07 %,
    b = 0.25 / 0.10 / 0.50.
  - The dropout sits in the top intensity state, as the labels say.
  - For 108 (ge10), whose missing windows are 70 % transient, the states
    differ little (π = 0.79 / 1.44 %).
* **The fitted states are not the activities.**
  - At K = 2 they match rest/active with 10–22 % error.
  - At K = 3 the error is 38–44 % (ge10) and 23–44 % (any). The three
    states are intensity levels, and the top one mixes locomotion and
    vigorous (102: 62 % locomotion, 31 % vigorous, 6 % rest).
  - The test therefore says "the mask depends on the hand-motion level",
    not "on the activity".
  - The supervised view (2a) says the same about the activities.
* **Bootstrap against χ²** (first run, fits' statistic, 36 cells).
  - The mean of the bootstrap LR is close to the df for the common-null
    tests: 0.6–1.3 for df 1, 1.7–2.7 for df 2 and 3.7–4.8 for df 4.
  - The largest replicates reach 21.6 at df 4. That suggests a heavier tail
    than χ²(4), but B = 99 cannot resolve it.
  - The nested test's bootstrap is below χ²(K) (mean 0.9–2.3 for df 2–3),
    so its χ² p-value is conservative.
  - The profile-statistic bootstrap of subject 108 gives the same picture.
    Its 95 % quantiles are 3.60 (df 1; χ² 3.84), 5.91 and 5.52 (df 2;
    χ² 5.99) and 8.44 (df 4; χ² 9.49).
  - Only subject 108 (ge10) is near the 5 % threshold, and there χ² and the
    bootstrap agree.
* **No grid variant was tested.** On the copula models (PMC state or pair
  margins) one ICE iteration takes 0.5–1 s at N ≈ 9 000. The test's 500
  iterations per fit cost 5–10 min per fit and 20 min per test, which is
  outside the budget. The known anticonservative case is also far from these
  series. On a grid variant (HMC-DN, N = 500, ~9 bursts;
  `report/missing_state/README.md`) the Markov test rejects about 12.5 % at
  5 %, and it holds its level from about 20 bursts on. These series have
  N ≈ 8 000–9 000 and 69–558 bursts.

## 3. Does using the mask help classification?

Scores: MPM error on the non-transient windows, overall and at the missing
windows. Calibration at the missing windows: mean confidence (max posterior),
Brier score, log-loss of the true group, and ECE. Unsupervised fits are
relabelled by the Hungarian permutation.

### 3a. Real gaps (mean over subjects; error at the missing windows for 102 / 108 / 105)

| rule | K | fit | variant | err all % | err missing % | per subject | log-loss miss. |
|---|---|---|---|---|---|---|---|
| ge10 | 2 | oracle HMC-IN | ignorable / state / state-markov | 7.4 / 7.3 / 7.3 | 1.1 / 1.1 / 1.1 | 0.0 / 3.3 / 0.0 (all) | 0.069 / 0.085 / 0.088 |
| any | 2 | oracle HMC-IN | ignorable / state / state-markov | 7.6 / 7.3 / 7.3 | 1.8 / 1.5 / 1.5 | 0.9 / 3.2 / 1.2 → 0.8 / 2.7 / 1.2 | 0.124 / 0.128 / 0.129 |
| any | 2 | oracle PMC pair | ignorable / state / state-markov | 6.7 / 4.6 / 5.8 | 1.5 / 0.8 / 1.1 | | 0.103 / 0.032 / 0.054 |
| ge10 | 3 | oracle HMC-IN | ignorable / state / state-markov | 11.4 / 10.8 / 10.6 | 1.8 / **4.7** / 1.8 | 102: 0.0 → 8.8 / 0.0 | 0.095 / 0.200 / 0.129 |
| ge10 | 3 | oracle PMC pair | ignorable / state / state-markov | 11.2 / 12.0 / 12.1 | 1.0 / **6.7 / 6.9** | 102: 0.0 → 17.0 / 17.6 | 0.254 / 0.601 / 0.518 |
| any | 3 | oracle HMC-IN | ignorable / state / state-markov | 11.5 / 13.1 / 13.2 | 3.2 / **13.6 / 13.5** | 102: 2.1 → 34.1 / 33.6 | 0.238 / 1.404 / 1.120 |
| any | 3 | oracle PMC pair | ignorable / state / state-markov | 11.2 / 15.3 / 15.4 | 5.3 / **30.8 / 30.9** | 102: 1.3 → 35.5 / 36.1; 108: 2.1 → 44.5 / 43.9 | 0.921 / 3.645 / 3.319 |
| ge10 | 2 | unsup. HMC-IN | ignorable / state (warm) / state-markov (warm) | 16.7 / 16.7 / 16.7 | 2.9 / 4.0 / 4.0 | 108: 6.7 → 10.0 | 0.143 / 0.177 / 0.172 |
| any | 2 | unsup. HMC-IN | ignorable / state (warm) / state-markov (warm) | 18.5 / 18.5 / 18.6 | 4.4 / 4.5 / 4.5 | | 0.285 / 0.378 / 0.358 |
| ge10 | 3 | unsup. HMC-IN | ignorable / state (warm) / state-markov (warm) | 42.1 / 42.1 / 42.1 | 38.0 / 39.4 / 38.3 | | 2.52 / 3.55 / 3.13 |
| any | 3 | unsup. HMC-IN | ignorable / state / state-markov (direct) | 35.0 / 41.4 / 42.3 | 39.7 / **62.0 / 46.4** | 105: 5.4 → 71.7 / 25.2 | 2.25 / 5.27 / 4.79 |
| any | 3 | unsup. HMC-IN | state (warm) / state-markov (warm) | 35.2 / 35.1 | 39.9 / 39.8 | | 3.23 / 2.98 |
| any | 3 | unsup. PMC pair | ignorable / state (warm) / state-markov (warm) | 41.7 / 37.4 / 42.8 | 44.0 / 45.0 / 51.5 | | 2.75 / 2.53 / 3.57 |

**Why the oracles lose at K = 3.** Error of the oracle at the windows of
each activity, missing / observed ("any" rule; every activity, rule and
model: `results/tables.md`, column `err_by_activity` of
`classification.csv`):

| subject, model | variant | walking (missing: 44 % / 39 %) | running | Nordic walking |
|---|---|---|---|---|
| 102, HMC-IN | ignorable | 0 % / 3 % | 0 / 10 | 1 / 9 |
| 102, HMC-IN | state / state-markov | **90 % / 48 %**, **89 % / 52 %** | 0 / 10 | 1–2 / 11–15 |
| 108, PMC pair | ignorable | 0 % / 3 % | 1 / 13 | 0 / 4 |
| 108, PMC pair | state / state-markov | **100 % / 97 %**, **98 % / 96 %** | 1 / 13 | 0 / 4 |

The other locomotion activities (cycling, stairs, vacuum cleaning) are
unchanged or better with the mask.

Walking is "locomotion", but its dropout (39–44 % of its windows) is that
of the vigorous group. Cycling, stairs and vacuum cleaning (0–4 %) pull the
locomotion rate down to 13 %. Every window of a walking bout, missing or
observed, then carries a small likelihood ratio towards "vigorous"
(`"state-markov"`: an onset 43 % vs 7 %, an observed window 57 % vs 93 %).
Over a bout the ratios add up, and the whole bout flips. `"state-markov"`
counts a burst as one onset plus its continuations, but it cannot stop
evidence accumulating across a bout in which the group-level rate is wrong.
Subject 105 shows the counter-case. Its walking loses as much (43 % of its
windows), but its vigorous activities lose less (8.2 %), so walking sets
the locomotion rate (13.8 %) and nothing flips.
The same effect makes the direct unsupervised fits capture a
"missing-prone" state (105, K = 3, any: posterior share of the missing
windows 0.02 / 0.10 / 0.88 against a truth of 0.07 / 0.71 / 0.23).

### 3b. Controlled masks: the real series with masks drawn on the true groups

For each subject, K, intensity (the plain rates of the `ge10` or `any`
masks, per group + transient) and mechanism, 5 masks were drawn with
`state_dependent` or `state_markov`: 120 series. The oracle mechanism is
estimated from the labels and the drawn mask. The unsupervised HMC-IN uses
the k-means start. Δ is the paired difference to the ignorable fit (mean ±
s.e. over 15 series).

| rule | mask | K | fit | err missing %: ignorable | Δ "state" (pts) | Δ "state-markov" (pts) | Δ log-loss, "state-markov" |
|---|---|---|---|---|---|---|---|
| ge10 | state | 2 | oracle HMC-IN | 4.9 | −1.65 ± 0.34 | −1.65 ± 0.34 | −0.116 ± 0.027 |
| ge10 | state-markov | 2 | oracle HMC-IN | 7.8 | −2.27 ± 0.47 | −1.73 ± 0.55 | −0.207 ± 0.039 |
| ge10 | state | 3 | oracle HMC-IN | 7.8 | −2.33 ± 0.65 | −2.33 ± 0.64 | −0.177 ± 0.037 |
| ge10 | state-markov | 3 | oracle HMC-IN | 7.3 | −1.75 ± 0.48 | −1.21 ± 0.32 | −0.189 ± 0.051 |
| any | state | 2 | oracle HMC-IN | 5.5 | −1.48 ± 0.17 | −1.46 ± 0.15 | −0.154 ± 0.015 |
| any | state-markov | 2 | oracle HMC-IN | 5.9 | −2.53 ± 0.35 | −2.40 ± 0.37 | −0.163 ± 0.013 |
| any | state | 3 | oracle HMC-IN | 8.2 | −2.18 ± 0.49 | −2.15 ± 0.46 | −0.221 ± 0.041 |
| any | state-markov | 3 | oracle HMC-IN | 9.0 | −2.74 ± 0.36 | −3.42 ± 0.46 | −0.274 ± 0.037 |
| ge10 | state-markov | 3 | oracle PMC pair | 11.4 | −3.13 ± 0.84 | −3.85 ± 0.76 | −0.164 ± 0.042 |
| any | state | 3 | oracle PMC pair | 12.9 | −4.87 ± 1.01 | −4.88 ± 1.03 | −0.273 ± 0.051 |
| any | state-markov | 3 | oracle PMC pair | 13.7 | −4.02 ± 0.77 | −5.15 ± 0.69 | −0.289 ± 0.026 |
| any | state-markov | 2 | unsup. HMC-IN | 5.7 | −1.55 ± 0.24 | −0.84 ± 0.14 | −0.030 ± 0.007 |
| any | state-markov | 3 | unsup. HMC-IN (warm) | 49.9 | −2.71 ± 0.79 | −1.83 ± 0.58 | **+0.368 ± 0.112** |

- **Oracles.** Every cell gains, 1.2–5.2 points at the missing windows.
  - The overall error falls by 0.1–1.7 points with HMC-IN, and by 0.6–4.4
    with PMC pair ("any": 12.1 → 7.7 %).
  - On Markov masks the right mechanism is better with the PMC pair oracle,
    and with HMC-IN at the "any" rates. With HMC-IN at the ge10 rates,
    `"state"` is better: the runs are short (mean 1.2–1.9 windows), so
    `"state"` over-counts little.
- **Unsupervised K = 2.** The gain is 0.2–1.6 points.
- **Unsupervised K = 3** (states ≠ groups; 47–51 % error at the missing
  windows). The error falls by 0.3–3.0 points, but the log-loss rises by
  0.37–0.87. The model becomes more confident about states that are not the
  groups.
- Full table (every mechanism, both masks and all K): `results/tables.md`,
  section 3b. Figure `figures/errors_missing.png` shows the real gaps and
  the controlled masks side by side. Its error bars are the s.e. over
  subjects or series.

### 3c. Estimating the mechanism from the start or from the ignorable fit

For each series and start, the gain is log p(y_obs, m) of the MNAR fit minus
[log p(y_obs) of the ignorable fit + log p(m) of the common mechanism]. That
bracket is where the warm fit starts, and a lower bound of its fixed point
when ICE is EM.

| setting | K | variant | fits | median gain (nat) | gain < −1 nat | min | Δ err missing (pts) |
|---|---|---|---|---|---|---|---|
| real | 2 | state / state-markov (direct) | 18 | 65.6 / 45.1 | 17 % / 17 % | −74 / −78 | +0.6 / +1.0 |
| real | 2 | state / state-markov (warm) | 18 | 66.4 / 45.7 | 0 / 0 | +3.3 / +2.7 | +0.6 / +0.6 |
| real | 3 | state / state-markov (direct) | 18 | 85.2 / 70.0 | 0 % / 11 % | +4.8 / −100 | **+10.6** / +3.8 |
| real | 3 | state / state-markov (warm) | 18 | 85.7 / 69.2 | 0 / 0 | +5.3 / +5.2 | +0.8 / +0.2 |
| controlled | 2 | state / state-markov (direct) | 60 | 31.3 / 22.8 | 13 % / 8 % | −80 / −77 | −0.8 / −0.5 |
| controlled | 2 | state / state-markov (warm) | 60 | 38.7 / 25.3 | 0 / 0 | +0.4 / +0.6 | −0.8 / −0.5 |
| controlled | 3 | state / state-markov (direct) | 60 | 66.0 / 50.2 | 0 % / 2 % | −0.2 / −63 | −1.8 / −1.3 |
| controlled | 3 | state / state-markov (warm) | 60 | 66.7 / 51.6 | 0 / 0 | +0.4 / +1.2 | −2.2 / −1.8 |

- A direct fit starts from the common mask and the k-means θ. At the first
  M-steps, poorly placed states pick up the mask, and ICE can settle in
  another basin, sometimes far worse (−100 nat). The warm fit never loses.
- A direct fit can also land in a *better-aligned* basin (105, K = 2, ge10:
  overall error 21.6 → 13.9 %), so the direct protocol mostly adds
  start-to-start variance.

### Calibration at the missing windows

The posteriors are nearly all above 0.95, so a binned reliability diagram is
a single bin. The table compares the mean predicted error (1 − max
posterior) with the observed MPM error, for HMC-IN (both K):

| setting | rule | fit | ignorable: predicted / observed % | "state-markov": predicted / observed % |
|---|---|---|---|---|
| real | ge10 | oracle | 0.06 / 1.45 | 0.17 / 1.45 |
| real | any | oracle | 0.42 / 2.46 | 0.50 / 7.54 |
| real | any | unsupervised (best start; mechanism direct) | 1.06 / 22.1 | 0.40 / 25.5 |
| controlled | any | oracle | 1.35 / 7.15 | 0.68 / 4.79 |
| controlled | any | unsupervised | 2.25 / 28.1 | 1.64 / 27.3 |

The observed error is 5–25 times the predicted one for the oracles, and 12–70
times for the unsupervised fits. The oracle's sticky A is one cause: its
posterior at a missing window is that of its neighbours. The mechanism
sharpens the posteriors without making them calibrated. Figure
`figures/calibration_missing.png` plots one point per series.

## 4. Imputation of the simulated gaps

`gaps.impute` scores the controlled masks (truth known): the posterior mean
for RMSE, 200 FFBS draws for the CRPS, and the 5–95 % posterior interval.
Means over 15 series, K = 3, Markov masks:

| rule | fit | ignorable: RMSE / CRPS / cov90 | "state" | "state-markov" |
|---|---|---|---|---|
| ge10 | oracle HMC-IN | 0.633 / 0.364 / 0.981 | 0.667 / 0.376 / 0.974 | 0.650 / 0.372 / 0.976 |
| ge10 | oracle PMC pair | 0.477 / 0.256 / 0.864 | 0.516 / 0.269 / 0.852 | 0.487 / 0.260 / 0.855 |
| any | oracle HMC-IN | 0.658 / 0.372 / 0.979 | 0.745 / 0.402 / 0.962 | 0.728 / 0.397 / 0.962 |
| any | oracle PMC pair | 0.496 / 0.262 / 0.879 | 0.573 / 0.287 / 0.862 | 0.549 / 0.281 / 0.865 |
| any | unsup. HMC-IN | 0.719 / 0.405 / 0.882 | 0.769 / 0.427 / 0.866 | 0.734 / 0.411 / 0.877 |

- Using the mask makes imputation worse in 53 of the 56 (cell, variant)
  comparisons: RMSE up to +0.10 (median +0.017), CRPS up to +0.04, coverage
  down to −0.02. The 3 exceptions (unsupervised K = 2) are better by at most
  0.005. On Markov masks `"state-markov"` is always less harmful than
  `"state"`.
- The oracle HMC-IN intervals are too wide (coverage 0.95–0.99): they are
  mixtures of broad state margins. The unsupervised HMC-IN intervals are
  near 0.90 (0.87–0.90), and the PMC pair ones too narrow (0.85–0.88).
- **Why.** At a missing window the imputation is a mixture over the states,
  weighted by their posterior. With the ignorable model that posterior
  follows the neighbouring windows, so it tracks the local level of the
  log-SD within an activity (slow vs brisk walking). HMC-IN does not model
  that level otherwise.
  - On 102 (K = 3, "any" Markov mask, oracle HMC-IN), imputing the *true*
    group mean gives RMSE 0.72 on the locomotion windows. The ignorable
    posterior mean gives 0.51, although 20 % of those windows are
    misclassified.
  - The mask pulls the posterior towards the label and away from the local
    level. `classify` and `impute` give identical posteriors (difference
    0.0), and the mean equals γ·μ exactly, so this is the model, not the
    code.
- By group and the other cells: `results/tables.md`, section 4.

## Limits

* **Sample.** Three subjects, one feature, one window length. The groups
  were chosen for classification, not for missingness. The within-group
  heterogeneity that drives the K = 3 loss is a property of this grouping.
* **The window mask is a design choice.** Both rules threshold 100 Hz
  dropouts. Under "any", a window with one NaN sample out of 50 is
  "missing" although its feature is almost fully observed.
* **Oracles.** Their A is the identity plus a 10⁻⁴ floor (transitions only
  through transient windows). That makes them sticky and overconfident, and
  it shapes both the size of the gain in 3b and the flipping of whole bouts
  in 3a.
* **Unsupervised K = 3 states are intensity levels, not the groups.** The
  unsupervised scores measure the mechanism's effect on a misaligned model.
  The unsupervised PMC pair uses one start, warm variants only.
* **Tests.** HMC-IN only: the copula models cost ~20 min per test.
  - The bootstrap B (99, or 19 for "any") only resolves p down to 0.01 or
    0.05, which is enough here: every LR is ≫ the bootstrap maximum except
    for subject 108.
  - The tests use the k-means start only. Their alternatives are fitted
    from different nulls and can end at different fixed points (below).
* **Controlled masks** are homogeneous within each group by construction,
  and redrawn with the plain rates. They isolate the model's gain; they do
  not reproduce the real clustering (1c).
* **Compute.**
  - The campaign: 64.8 CPU-minutes (297 tasks, 4 processes, 34 min wall).
  - The rerun of the tests with the profile statistic: 16.8 CPU-minutes
    (36 tasks, 7 min wall).
  - About 20 CPU-minutes more for timing, smoke runs and an interrupted
    first launch.
  - In all, about 100 CPU-minutes, over the ~1.5 h budget because of the
    rerun the statistic change required.
  - The machine was heavily loaded (load average 30–70 on 10 cores), so task
    wall times run 2× the CPU times. `results/tasks.csv` has both.
  - No series was shortened.

## Library observations (no library code changed)

* **The nested test's null is fitted from the start, not from the ignorable
  fit.** `missingness_lr_test(null="state")` runs `ice(model,
  missingness="state")` directly from `model`. In 2 of the 12 real-series
  cells (102 K = 2 and 105 K = 3, "any" rule), that null ended 33–41 nat
  below the `"state"` fit that the `"state"` vs common test reaches from
  the ignorable fit. The alternative, started from that null, ended 76–150
  nat below the `"state-markov"` fit reached from the common null. The
  reported nested LR is 665 and 287, against 735 and 520 from the best fits.
  The profile statistic of `0758e3a` does not change this: it profiles the
  mechanism at the two fits' θ, and both θ lie in the inferior basins
  (sup_H0 = −12004.05, sup_H1 = −11671.71 for 102). The conclusions do not
  change here, but the statistic compares two local optima.
  **Fixed on main after this study:** the `"state"` null is now fitted from
  the ignorable fit. Rerun on the two cells: nested LR 735.0 (102, K = 2)
  and 520.1 (105, K = 3), the best-fit values; the null of 102 now ends at
  −11963.46, the `"state"` fit of the `"state"` vs common test. The tables
  above keep the numbers of the study run. Reproduction
  (≈ 1 min, checked after `0758e3a`):

  ```python
  import sys; sys.path.insert(0, "report/real_series")
  import numpy as np, rs_common as rc
  from pmcprg.pmc.missingness_lr import missingness_lr_test
  d = rc.prep_pamap2(rc.DEFAULT_DATA, 102, False)
  Y = d["Y_complete"].copy(); Y[d["real_any"]] = np.nan
  init, _ = rc.starting_model("hmc_in", 2, Y, 0, 1)
  cfg = {"fit_margins": True, "missing_strategy": "available"}
  a = missingness_lr_test(init, Y, alternative="state", null="common", ice_cfg=cfg)
  b = missingness_lr_test(init, Y, alternative="state-markov", null="state", ice_cfg=cfg)
  print(a.log_lik_alt, b.log_lik_null)   # -11963.46  -12004.05: the same model, 40.6 nat apart
  print(b.statistic)                       # 664.7; the best fits give 735.0
  ```

  A possible remedy: fit the `"state"` null from the ignorable fit, as the
  alternatives already are.
* **The same holds for ICE with an estimated mechanism** (section 3c). From
  a fresh start, up to 17 % of the HMC-IN fits end more than 1 nat below
  "ignorable fit + common mask". The docstring of `missingness` in
  `pmcprg/pmc/missingness.py` ("Start") could recommend the two-stage
  protocol.
* No crash, no non-finite estimate, no `IncompatibleObservationError` in
  the 357 tasks run. `classify` and `impute` agree exactly on the posteriors at the
  missing rows with a mechanism.

## Reproduce

From the repository root (the data folder is read, never written):

```bash
# every result file but lr_tests_fits_statistic.csv (library ≥ 0758e3a: the profile
# statistic; bootstrap B = 99 for subject 108, rule ge10, as committed)
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python report/missing_state/pamap2/run_pamap2.py --jobs 4 \
    --boot 99 --boot-any 0 --boot-subjects 108
.venv/bin/python report/missing_state/pamap2/summarise.py        # tables.md, tables/*.tex, figures/
# smoke run (subject 105, B = 2, 1 seed, 1 start; → results_quick/, not versioned)
OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python report/missing_state/pamap2/run_pamap2.py --quick
```

Finished tasks are cached in `report/out/missing_state_pamap2/` (git-ignored),
so a rerun only aggregates them; `--force` recomputes.

`results/lr_tests_fits_statistic.csv` comes from the first campaign
(`--boot 99 --boot-any 19`, all subjects), run on the library before
`0758e3a`, whose statistic was the difference of the two fits. With the
current library, `--boot 99 --boot-any 19` (no `--boot-subjects`) runs the
bootstrap of the profile statistic in every cell instead: about 60
CPU-minutes.

```
report/missing_state/pamap2/
├── README.md        this file
├── pm_common.py     data (via report/real_series/rs_common.py), mask statistics,
│                    mechanisms from labels, scores
├── run_pamap2.py    the four parts (describe, test, real, controlled), task cache
├── summarise.py     tables and figures from the CSVs
├── results/         describe_100hz.csv, describe_windows.csv, gof.csv,
│                    supervised_mechanism.csv, supervised_lr.csv, lr_tests.csv
│                    (profile statistic), lr_tests_fits_statistic.csv (first run),
│                    classification.csv, imputation.csv, tasks.csv, run_info.json,
│                    run_info_test.json, tables.md — statistics only, no observations
├── tables/          LaTeX tables
└── figures/         gaps_by_activity.png, errors_missing.png, calibration_missing.png
```
