# State-dependent missingness: estimation and likelihood-ratio test (P6)

Three scripts, run from the repository root with the repo venv.

| script | what | time (loaded Apple arm64 laptop) |
|---|---|---|
| `design_measurements.py` | the design choices of the mechanism M-step of ICE/SEM: initial term of `"state-markov"`, boundary guard, starts | 5 min (6 processes) |
| `lr_study.py` | size, power (asymptotic χ² and parametric bootstrap) of `missingness_lr_test`, recovery of π̂, â, b̂ | 59 min (6 processes; `--direct`, `--profile`, `--direct-study`: see below) |
| `lr_diagnosis.py` | why the Markov test was too wide on HMC-DN at N = 500: decomposition of the statistic, direct maximisation of the likelihood | 5 + 15 min (5 processes) |

Every seed is a `zlib.crc32` of a named tuple; a path and its mask never share
a seed. Results (one row per replication) are in `results/`.

```bash
.venv/bin/python report/missing_state/design_measurements.py --jobs 6
.venv/bin/python report/missing_state/lr_study.py --jobs 4 --quick    # smoke: 4 per cell → results/lr_study_quick.csv (not versioned)
.venv/bin/python report/missing_state/lr_study.py --jobs 6            # full study (the fits' statistic, see below)
.venv/bin/python report/missing_state/lr_study.py --jobs 6 --direct   # B = 99 bootstrap check
.venv/bin/python report/missing_state/lr_study.py --jobs 6 --profile  # paired rerun, profile statistic
.venv/bin/python report/missing_state/lr_study.py --jobs 6 --direct-study   # B = 99 on HMC-DN nulls
.venv/bin/python report/missing_state/lr_study.py --summarise         # tables from the CSV files
.venv/bin/python report/missing_state/lr_diagnosis.py --decompose --jobs 6
.venv/bin/python report/missing_state/lr_diagnosis.py --mle --jobs 5 --r 111,78,197,11,198,89,7,143,$(seq -s, 0 29)
```

**The statistic.** The first study (`results/lr_study.csv`, `lr_direct.csv`,
the "Size and power" table) was measured when the statistic was the
difference of the two ICE fits, 2 (LL1(θ̂1, φ̂1) − LL0(θ̂0)) — now the
result field `statistic_fits`. The test now uses the profile statistic
(`pmcprg.pmc.missingness_lr`, "The statistic"; "Diagnosis" below for why);
`--profile` reran HMC-IN at N = 500 and the HMC-DN Markov cells with it, same
seeds (`results/lr_study_profile.csv`, the "paired" table).

## Design measurements

HMC-IN, K = 2, A = [[0.95, 0.05], [0.05, 0.95]], margins N(∓1, 1); ICE with
`fit_margins = True`, `tol = 1e-8`, `max_iter = 500`, `patience = 500` (to
ICE's fixed point), `missing_strategy = "available"`.

### 1. The initial term of `"state-markov"` (`results/design_initial.csv`)

The n = 0 term γ_0(i) log p(m_0 | s_i), s_i = a_i / (1 − b_i + a_i), couples
a_i and b_i. ICE with the exact per-state maximisation (the package) against
ICE with the closed form that drops it; masks a = (0.005, 0.03),
b = (0.7, 0.9); 100 data sets per N.

| N | parameter | sd (exact) | mean \|Δ\| | max \|Δ\| | mean Δ / sd |
|---|---|---|---|---|---|
| 500 | a_0 | 6.86e-3 | 2.79e-4 | 4.54e-3 | +0.015 |
| 500 | a_1 | 1.32e-2 | 9.10e-4 | 4.60e-3 | −0.011 |
| 500 | b_0 | 0.141 | 6.51e-3 | 5.80e-2 | +0.018 |
| 500 | b_1 | 0.096 | 6.11e-3 | 5.04e-2 | −0.020 |
| 2000 | a_0 | 3.28e-3 | 4.65e-5 | 1.25e-3 | +0.001 |
| 2000 | a_1 | 6.39e-3 | 2.08e-4 | 1.11e-3 | −0.001 |
| 2000 | b_0 | 0.143 | 2.37e-3 | 1.25e-2 | +0.001 |
| 2000 | b_1 | 2.84e-2 | 1.22e-3 | 1.06e-2 | +0.001 |

Log-likelihood reached, exact − closed form: median 0.0022 nat (N = 500) and
0.00065 (N = 2000), mean 0.074 and 0.0048; above 0.1 nat in 11 of 100 runs at
N = 500 (up to 2.48 nat), none at N = 2000; below −1e-9 in 19 and 22 runs, by
at most 0.032 nat (ICE is not exactly EM: the SR symmetrisation of the prior).
The large gains are series that start inside a long burst of a rarely missing
state, where log s_i is the only term tying a small a_i to a b_i near 1 (the
2.48-nat run: â_0 = 9.7e-4, b̂_0 = 0.9998 exact; 9.2e-5, 0.986 without).
**Choice: the exact maximisation** — no systematic change of the estimates,
but a real gain on those series, for 8 % of an ICE iteration (N = 2000).

### 2. The boundary guard (`results/design_guard.csv`)

Each M-step adds c pseudo-observations at the pooled rate of the mask (a Beta
prior worth c observations; the posterior mode). Bias and RMSE of the
estimates, 100 data sets per cell:

| truth | N | parameter | c = 0 | c = 0.1 | c = 1 | c = 10 |
|---|---|---|---|---|---|---|
| π = (0.02, 0.3) | 500 | π_0 = 0.02 | +0.0002 / 0.0114 | +0.0003 / 0.0113 | +0.0012 / 0.0112 | +0.0082 / 0.0136 |
| | | π_1 = 0.3 | −0.0017 / 0.0292 | −0.0018 / 0.0292 | −0.0025 / 0.0291 | −0.0089 / 0.0296 |
| | 2000 | π_0 | −0.0003 / 0.0054 | −0.0003 / 0.0054 | −0.0001 / 0.0054 | +0.0018 / 0.0056 |
| | | π_1 | −0.0004 / 0.0145 | −0.0004 / 0.0145 | −0.0006 / 0.0145 | −0.0021 / 0.0145 |
| π = (0, 0.2) | 500 | π_0 = 0 | +0.0010 / 0.0026 | +0.0011 / 0.0027 | +0.0019 / 0.0031 | +0.0074 / 0.0087 |
| | | π_1 = 0.2 | −0.0003 / 0.0260 | −0.0003 / 0.0260 | −0.0009 / 0.0259 | −0.0054 / 0.0260 |
| | 2000 | π_0 | +0.0005 / 0.0009 | +0.0005 / 0.0009 | +0.0007 / 0.0011 | +0.0023 / 0.0025 |
| | | π_1 | +0.0004 / 0.0121 | +0.0004 / 0.0121 | +0.0002 / 0.0121 | −0.0010 / 0.0120 |
| a = (0.002, 0.03), b = (0.5, 0.9) | 500 | a_0 | +0.0013 / 0.0049 | +0.0013 / 0.0049 | +0.0014 / 0.0049 | +0.0022 / 0.0051 |
| (state 0: 1–2 bursts) | | a_1 | −0.0015 / 0.0116 | −0.0015 / 0.0116 | −0.0016 / 0.0116 | −0.0025 / 0.0111 |
| | | b_0 = 0.5 | +0.075 / 0.371 | +0.110 / 0.336 | +0.212 / 0.285 | +0.302 / 0.317 |
| | | b_1 = 0.9 | −0.048 / 0.110 | −0.048 / 0.109 | −0.048 / 0.103 | −0.054 / 0.097 |
| | 2000 | a_0 | +0.0005 / 0.0022 | +0.0005 / 0.0022 | +0.0006 / 0.0023 | +0.0012 / 0.0025 |
| | | a_1 | +0.0000 / 0.0064 | +0.0000 / 0.0064 | −0.0001 / 0.0064 | −0.0008 / 0.0062 |
| | | b_0 | −0.043 / 0.288 | −0.017 / 0.256 | +0.090 / 0.188 | +0.252 / 0.262 |
| | | b_1 | −0.010 / 0.034 | −0.010 / 0.034 | −0.012 / 0.034 | −0.022 / 0.038 |

(bias / RMSE.) Smallest estimates: without the guard onset rates of 2.3e-16
(the logit bound) and b̂_0 of 1.2e-7; c = 0.1 still leaves 9.5e-7; c = 1
keeps every rate above 9.6e-6 (the onset of a state with true rate 0.002).

**The trap** (`results/design_trap.csv`, 50 data sets per N): a start
π_0 = 0 (truth π = (0.05, 0.3)) makes state 0 impossible at every missing row.
Without the guard π̂_0 stays exactly 0 in 50 of 50 runs, the fit ending 12.6
nat (N = 500, max 24.4) and 58.4 nat (N = 2000, max 92.6) below the fit from
the common start. With c = 1 π̂_0 is within 10 % of its final value after 6
iterations (median; max 9), and the fit reaches the same log-likelihood as
from the common start (difference ≤ 1.2e-3 nat); smallest π̂_0 0.019 / 0.031.

**Choice: c = 1**, the smallest value measured that keeps every rate away from
the bounds; its bias on well-populated states is ≤ 0.001 (within 0.1 RMSE),
and on a state with one or two bursts it trades bias for a lower RMSE (the
persistence of such a state is shrunk towards the pooled one). Not a config
key.

### 3. The common-rate start

From a model without a mechanism, ICE starts from the state-independent MLE
of the mask (π_i = M/N), under which the E-step is the ignorable one. Truth
π = (0.02, 0.3), 50 data sets:

| states | N | start | after the first M-step | final π̂ (mean ± sd) |
|---|---|---|---|---|
| Markov (A = 0.95 diag.) | 500 | 0.157 | (0.054, 0.273) | (0.021 ± 0.012, 0.297 ± 0.029) |
| | 2000 | 0.160 | (0.055, 0.274) | (0.020 ± 0.005, 0.300 ± 0.018) |
| i.i.d. (A rows equal) | 500 | 0.158 | (0.146, 0.173) | (0.176 ± 0.078, 0.139 ± 0.075) |
| | 2000 | 0.161 | (0.149, 0.174) | (0.139 ± 0.056, 0.192 ± 0.080) |

The common rate is not a trap: the first M-step leaves it and ICE converges
to the truth. With i.i.d. states π is not identified (only the observed
mixture weights ∝ p_i (1 − π_i) and Σ p_i π_i are) and the estimates wander —
the persistence of the states is what identifies a state-dependent mechanism.

## Monte-Carlo study of the test

`missingness_lr_test` (`pmcprg.pmc.missingness_lr`) on two models, K = 2,
ICE with `fit_margins = True` started from the true θ without mechanism and
run to its fixed point (the test's defaults):

* `hmc_in` — HMC-IN, A = [[0.95, 0.05], [0.05, 0.95]], margins N(∓1, 1): the
  exact K-state shortcut, where ICE is EM (up to the SR symmetrisation of the
  prior);
* `hmc_dn` — `pmcprg/pmc/models/hmc_dn_gauss_k2.toml` (A = 0.9 on the
  diagonal, N(∓1, 1), Gaussian copulas τ = 0.6 / 0): the quadrature grid
  (64 nodes), ICE with `candidates = ["Gauss"]` — not EM (copula step on
  the pairs with both ends observed).

Masks: `"state"` with π = (0.1, 0.1) under H0 and π = 0.1 ∓ δ/2, δ = 0.05,
0.10, 0.15; `"state-markov"` with a common (a, b) = (0.02, 0.8) under H0 (9 %
missing in bursts of mean length 5), and a = (0.013, 0.027), b = 0.8
("onset") or a = (0.01, 0.03), b = (0.7, 0.9) ("both"). The nested test
("state-markov" vs "state") runs on HMC-IN with π = (0.05, 0.15) under H0 and
the "both" Markov masks. 400 replications per null cell and 200 per
alternative on HMC-IN, 200 and 100 on HMC-DN; recovery at N = 5000 with 100.

**Bootstrap by the warp-speed method** (Giacomini, Politis & White 2013):
each replication draws one bootstrap replicate LR*_r from its own fitted null
model (`bootstrap_replicate`); the bootstrap test rejects when LR_r exceeds
the 95 % quantile of {LR*_r} pooled over the cell ("crit"). Two fits per
replication instead of 2B. `--direct` checks it on one cell with a genuine
B = 99 bootstrap per replication (below).

Cost per replication (test + one bootstrap replicate, 4 ICE fits; loaded
machine): HMC-IN 0.37 / 0.52 / 0.88 s at N = 500 / 1000 / 2000, 1.1 s at
5000 (test only); HMC-DN 9.4 / 14.4 s at N = 500 / 1000. The full study:
8 800 replications, 21 000 CPU-seconds, 59 min on 6 processes.

### Size and power at the 5 % level (`results/lr_study.csv`; the fits' statistic)

Rejection rate ± binomial standard error; "crit" is the warp-speed
bootstrap critical value (χ²: 3.84 for df = 1, 5.99 for df = 2). The
statistic is the difference of the fits (above); the paired table after this
one gives the profile statistic on the rerun cells.

| model | test | masks | N | R | mean M | mean LR (df) | reject χ² | reject bootstrap (crit) |
|---|---|---|---|---|---|---|---|---|
| HMC-IN | state vs common | **H0** π = (0.1, 0.1) | 500 | 400 | 50 | 1.07 (1) | 0.058 ± 0.012 | 0.050 ± 0.011 (3.97) |
| | | | 1000 | 400 | 100 | 0.99 (1) | 0.045 ± 0.010 | 0.045 ± 0.010 (3.89) |
| | | | 2000 | 400 | 201 | 1.05 (1) | 0.060 ± 0.012 | 0.065 ± 0.012 (3.52) |
| | | π = (0.075, 0.125) | 500 | 200 | 50 | 3.70 (1) | 0.365 ± 0.034 | 0.380 ± 0.034 (3.75) |
| | | | 1000 | 200 | 100 | 6.11 (1) | 0.620 ± 0.034 | 0.640 ± 0.034 (3.49) |
| | | | 2000 | 200 | 198 | 11.98 (1) | 0.940 ± 0.017 | 0.935 ± 0.017 (3.99) |
| | | π = (0.05, 0.15) | 500 | 200 | 50 | 11.46 (1) | 0.885 ± 0.023 | 0.910 ± 0.020 (3.14) |
| | | | 1000 | 200 | 99 | 24.06 (1) | 0.995 ± 0.005 | 0.995 ± 0.005 (4.02) |
| | | | 2000 | 200 | 201 | 46.31 (1) | 1.000 | 1.000 (2.91) |
| | | π = (0.025, 0.175) | 500–2000 | 200 each | 49–200 | 27.7–112.1 (1) | 1.000 | 1.000 |
| HMC-IN | state-markov vs common | **H0** a = 0.02, b = 0.8 | 500 | 400 | 45 | 2.07 (2) | 0.045 ± 0.010 | 0.043 ± 0.010 (6.56) |
| | | | 1000 | 400 | 89 | 1.94 (2) | 0.028 ± 0.008 | 0.028 ± 0.008 (6.11) |
| | | | 2000 | 400 | 180 | 2.02 (2) | 0.050 ± 0.011 | 0.058 ± 0.012 (5.89) |
| | | onset | 500 | 200 | 46 | 2.82 (2) | 0.135 ± 0.024 | 0.085 ± 0.020 (6.76) |
| | | | 1000 | 200 | 91 | 3.75 (2) | 0.210 ± 0.029 | 0.125 ± 0.023 (7.53) |
| | | | 2000 | 200 | 177 | 5.50 (2) | 0.365 ± 0.034 | 0.320 ± 0.033 (6.66) |
| | | both | 500 | 200 | 58 | 4.36 (2) | 0.255 ± 0.031 | 0.255 ± 0.031 (5.96) |
| | | | 1000 | 200 | 118 | 6.53 (2) | 0.485 ± 0.035 | 0.530 ± 0.035 (5.52) |
| | | | 2000 | 200 | 234 | 10.72 (2) | 0.805 ± 0.028 | 0.820 ± 0.027 (5.83) |
| HMC-IN | state-markov vs state | **H0** π = (0.05, 0.15) | 500 | 400 | 50 | 1.67 (2) | 0.020 ± 0.007 | 0.022 ± 0.007 (5.82) |
| | | | 1000 | 400 | 101 | 2.06 (2) | 0.048 ± 0.011 | 0.072 ± 0.013 (5.35) |
| | | | 2000 | 400 | 200 | 2.05 (2) | 0.045 ± 0.010 | 0.048 ± 0.011 (5.90) |
| | | both | 500 | 200 | 60 | 161.2 (2) | 0.990 ± 0.007 | 0.995 ± 0.005 (4.66) |
| | | | 1000, 2000 | 200 each | 119, 232 | 352, 698 (2) | 1.000 | 1.000 |
| HMC-DN | state vs common | **H0** π = (0.1, 0.1) | 500 | 200 | 50 | 1.04 (1) | 0.060 ± 0.017 | 0.040 ± 0.014 (4.49) |
| | | | 1000 | 200 | 100 | 1.02 (1) | 0.060 ± 0.017 | 0.060 ± 0.017 (3.85) |
| | | π = (0.05, 0.15) | 500 | 100 | 50 | 8.97 (1) | 0.750 ± 0.043 | 0.710 ± 0.045 (4.61) |
| | | | 1000 | 100 | 102 | 16.48 (1) | 0.990 ± 0.010 | 0.980 ± 0.014 (4.19) |
| | | π = (0.025, 0.175) | 500 | 100 | 49 | 18.84 (1) | 0.990 ± 0.010 | 0.960 ± 0.020 (4.78) |
| | | | 1000 | 100 | 99 | 35.24 (1) | 1.000 | 1.000 (3.97) |
| HMC-DN | state-markov vs common | **H0** a = 0.02, b = 0.8 | 500 | 200 | 45 | 2.73 (2) | **0.135 ± 0.024** | **0.175 ± 0.027** (5.06) |
| | | | 1000 | 200 | 88 | 1.96 (2) | 0.030 ± 0.012 | 0.015 ± 0.009 (6.58) |
| | | both | 500 | 100 | 54 | 3.60 (2) | 0.210 ± 0.041 | 0.170 ± 0.038 (6.51) |
| | | | 1000 | 100 | 110 | 4.38 (2) | 0.240 ± 0.043 | 0.210 ± 0.041 (7.51) |

Reading:

* **HMC-IN** (ICE is EM): both references hold the level — χ² 0.028–0.060,
  bootstrap 0.022–0.072 over the nine null cells — and agree with each
  other. The smallest LR over the 1 200 "state"-vs-common nulls is 5.7e-6
  (never negative). The nested test has 5 slightly negative LR at N = 500
  (min −0.096): the guard's pseudo-counts pull "state" and "state-markov"
  towards different pooled rates (without the guard those replications
  give 0.007 to 0.21; the profile statistic, unguarded, has none).
* **Power.** A difference of 5 points in the missing rate (7.5 % vs 12.5 %)
  needs N ≈ 2000 on HMC-IN; 10 points is detected at N = 500 (0.885). A
  Markov mask whose onset alone depends on the state (1.3 % vs 2.7 %) is
  hard to see — 0.37 at N = 2000: a burst is one piece of evidence, not
  five, and ~9 to 35 bursts are all there is; with the persistence too, 0.81
  at N = 2000. "state" against "state-markov" on bursty masks: power 0.99 at
  N = 500.
* **HMC-DN** (quadrature grid, ICE not EM): "state" holds the level (0.040–
  0.060) with a power close to HMC-IN's. The Markov test is
  **anticonservative at N = 500** under both references (0.135 and 0.175):
  its null distribution has a heavier tail than χ²(2) (95 % quantile of LR
  8.05 against 5.99) and than its bootstrap (5.06). 9 of the 200 null LR
  are negative (min −1.42) and the largest is 16.6. Taken apart in
  "Diagnosis" below: the negative values are the two ICE fixed points
  optimising θ unequally (fixed by the profile statistic), the excess of
  rejections is the likelihood ratio's own at that N. At N = 1000 the test
  is conservative (0.030 / 0.015) and its power low (0.24).

### The profile statistic, paired (`results/lr_study_profile.csv`)

`lr_study.py --profile`: HMC-IN at N = 500 (every scenario, the three
tests) and the HMC-DN Markov cells at N = 500 and 1000, the seeds of the
table above — the fits are bit-identical to it (3 000 of 3 000 replications:
`LR_fits` equals the first study's `LR`), so the two statistics are compared
on the same series, masks and warp-speed replicates' seeds. 36 min on 6
processes on a heavily loaded machine.

| model | test | masks | N | R | fits: mean LR (# < 0) | reject χ² | reject bootstrap (crit) | profile: mean LR (min) | reject χ² | reject bootstrap (crit) |
|---|---|---|---|---|---|---|---|---|---|---|
| HMC-IN | state vs common | **H0** π = (0.1, 0.1) | 500 | 400 | 1.07 (0) | 0.058 ± 0.012 | 0.050 ± 0.011 (3.97) | 1.07 (4e-6) | 0.058 ± 0.012 | 0.050 ± 0.011 (3.97) |
| | | π = (0.075, 0.125) | 500 | 200 | 3.70 (0) | 0.365 ± 0.034 | 0.380 ± 0.034 (3.75) | 3.70 | 0.365 ± 0.034 | 0.380 ± 0.034 (3.75) |
| | | π = (0.05, 0.15) | 500 | 200 | 11.46 (0) | 0.885 ± 0.023 | 0.910 ± 0.020 (3.14) | 11.46 | 0.885 ± 0.023 | 0.910 ± 0.020 (3.15) |
| | | π = (0.025, 0.175) | 500 | 200 | 27.71 (0) | 1.000 | 1.000 (3.35) | 27.71 | 1.000 | 1.000 (3.35) |
| HMC-IN | state-markov vs common | **H0** a = 0.02, b = 0.8 | 500 | 400 | 2.07 (0) | 0.045 ± 0.010 | 0.043 ± 0.010 (6.56) | 2.16 (0.008) | 0.052 ± 0.011 | 0.035 ± 0.009 (6.93) |
| | | onset | 500 | 200 | 2.82 (0) | 0.135 ± 0.024 | 0.085 ± 0.020 (6.76) | 2.91 | 0.140 ± 0.025 | 0.080 ± 0.019 (7.06) |
| | | both | 500 | 200 | 4.36 (0) | 0.255 ± 0.031 | 0.255 ± 0.031 (5.96) | 4.56 | 0.290 ± 0.032 | 0.270 ± 0.031 (6.26) |
| HMC-IN | state-markov vs state | **H0** π = (0.05, 0.15) | 500 | 400 | 1.67 (5) | 0.020 ± 0.007 | 0.022 ± 0.007 (5.82) | 1.79 (0.007) | 0.020 ± 0.007 | 0.020 ± 0.007 (5.94) |
| | | both | 500 | 200 | 161.2 (0) | 0.990 ± 0.007 | 0.995 ± 0.005 (4.66) | 161.2 | 0.995 ± 0.005 | 0.995 ± 0.005 (4.74) |
| HMC-DN | state-markov vs common | **H0** a = 0.02, b = 0.8 | 500 | 200 | 2.73 (9) | **0.135 ± 0.024** | **0.175 ± 0.027** (5.06) | 2.83 (0.0005) | **0.125 ± 0.023** | **0.150 ± 0.025** (5.31) |
| | | | 1000 | 200 | 1.96 (1) | 0.030 ± 0.012 | 0.015 ± 0.009 (6.58) | 2.04 (0.024) | 0.045 ± 0.015 | 0.010 ± 0.007 (7.01) |
| | | both | 500 | 100 | 3.60 (0) | 0.210 ± 0.041 | 0.170 ± 0.038 (6.51) | 3.81 | 0.220 ± 0.041 | 0.270 ± 0.044 (5.51) |
| | | | 1000 | 100 | 4.38 (0) | 0.240 ± 0.043 | 0.210 ± 0.041 (7.51) | 4.37 | 0.200 ± 0.040 | 0.200 ± 0.040 (5.92) |

Reading:

* **HMC-IN is not hurt.** The "state" test is unchanged (|LR − fits| ≤ 0.017
  under H0). On the Markov tests the profile statistic is larger by a median
  0.008 (common null) and 0.098 (nested), up to 2.1 and 0.43: that is the
  boundary guard, which the fits carry (posterior modes) and the profiles do
  not — 2 (ℓ1(θ̂1) − LL1) accounts for the largest difference (2.14; a state
  with one or two bursts, whose b̂ the guard shrinks), while the θ-part has
  an sd of 0.06. Sizes 0.052 / 0.035 and 0.020 / 0.020; the power of the
  Markov test against onset and persistence rises from 0.255 to 0.290. No
  negative value: the nested test's five (min −0.096) are gone.
* **HMC-DN.** No negative value (the fits' statistic: 9 at N = 500, down to
  −1.42); |LR − fits| up to 5.4 at N = 500 and 2.6 at N = 1000. At N = 500
  the size is still 0.125 (χ²) and 0.150 (warp-speed bootstrap): the tail is
  the likelihood ratio's own, see "Diagnosis". At N = 1000 χ² holds the level
  (0.045 ± 0.015; the fits gave 0.030), the bootstrap is conservative
  (0.010), and the power against onset and persistence differences is 0.20.
* Not rerun: HMC-IN at N = 1000, 2000 and the HMC-DN "state" test (the fits'
  statistic in the first table). On HMC-IN at N = 500 the two statistics'
  χ² decisions differ on 0, 3 and 0 of the 400 null series of the three
  tests (on HMC-DN: 8 of 200 at N = 500, 3 of 200 at N = 1000).
* Cost of the profiles (two EM runs over the mechanism, θ fixed; CPU time,
  4 series each): 17 % of the test on HMC-DN, N = 500 (2.35 s); 38–45 % on
  HMC-IN (0.19–0.23 s), where ICE converges in few iterations.

### Recovery of the mechanism (the alternative's fit; bias / RMSE)

| model | masks | parameter | N = 500 | N = 1000 | N = 2000 | N = 5000 |
|---|---|---|---|---|---|---|
| HMC-IN | π = (0.1, 0.1) | π_0 | −0.0012 / 0.0216 | −0.0003 / 0.0139 | +0.0002 / 0.0102 | |
| | | π_1 | +0.0009 / 0.0210 | +0.0009 / 0.0146 | +0.0007 / 0.0103 | |
| | π = (0.075, 0.125) | π_0 | +0.0016 / 0.0217 | +0.0003 / 0.0133 | −0.0008 / 0.0085 | |
| | | π_1 | +0.0004 / 0.0219 | −0.0014 / 0.0150 | −0.0007 / 0.0101 | |
| | π = (0.05, 0.15) | π_0 | +0.0012 / 0.0146 | −0.0007 / 0.0108 | +0.0000 / 0.0069 | +0.0010 / 0.0046 |
| | | π_1 | −0.0008 / 0.0254 | −0.0008 / 0.0167 | +0.0002 / 0.0126 | −0.0000 / 0.0078 |
| | π = (0.025, 0.175) | π_0 | +0.0001 / 0.0116 | +0.0000 / 0.0082 | −0.0000 / 0.0055 | |
| | | π_1 | −0.0015 / 0.0276 | −0.0014 / 0.0172 | +0.0007 / 0.0128 | |
| | a = 0.02, b = 0.8 (common) | a_0 | +0.0009 / 0.0107 | −0.0004 / 0.0071 | +0.0000 / 0.0052 | |
| | | a_1 | +0.0005 / 0.0108 | −0.0006 / 0.0073 | −0.0001 / 0.0050 | |
| | | b_0 | −0.061 / 0.153 | −0.016 / 0.083 | −0.012 / 0.065 | |
| | | b_1 | −0.043 / 0.134 | −0.028 / 0.099 | −0.014 / 0.061 | |
| | onset: a = (0.013, 0.027), b = 0.8 | a_0 | +0.0008 / 0.0082 | +0.0003 / 0.0064 | −0.0004 / 0.0041 | |
| | | a_1 | +0.0010 / 0.0124 | +0.0007 / 0.0080 | −0.0002 / 0.0061 | |
| | | b_0 | −0.044 / 0.151 | −0.040 / 0.122 | −0.032 / 0.091 | |
| | | b_1 | −0.047 / 0.132 | −0.013 / 0.069 | −0.001 / 0.045 | |
| | both: a = (0.01, 0.03), b = (0.7, 0.9) | a_0 | +0.0011 / 0.0084 | +0.0006 / 0.0057 | −0.0001 / 0.0035 | +0.0003 / 0.0025 |
| | | a_1 | +0.0004 / 0.0148 | +0.0007 / 0.0091 | −0.0002 / 0.0062 | −0.0002 / 0.0039 |
| | | b_0 | −0.012 / 0.177 | −0.011 / 0.152 | −0.008 / 0.111 | −0.008 / 0.098 |
| | | b_1 | −0.028 / 0.090 | −0.016 / 0.050 | −0.006 / 0.031 | −0.003 / 0.015 |
| HMC-DN | π = (0.1, 0.1) | π_0 | −0.0027 / 0.0274 | +0.0018 / 0.0190 | | |
| | | π_1 | −0.0016 / 0.0251 | −0.0022 / 0.0181 | | |
| | π = (0.05, 0.15) | π_0 | −0.0006 / 0.0225 | −0.0012 / 0.0147 | | |
| | | π_1 | +0.0025 / 0.0352 | +0.0022 / 0.0191 | | |
| | π = (0.025, 0.175) | π_0 | −0.0009 / 0.0154 | +0.0012 / 0.0113 | | |
| | | π_1 | −0.0040 / 0.0292 | −0.0007 / 0.0219 | | |
| | a = 0.02, b = 0.8 (common) | a_0, a_1 | +0.0003 / 0.0143, +0.0008 / 0.0162 | +0.0000 / 0.0093, −0.0002 / 0.0092 | | |
| | | b_0, b_1 | −0.055 / 0.165, −0.031 / 0.148 | −0.038 / 0.119, −0.031 / 0.124 | | |
| | both | a_0, a_1 | +0.0024 / 0.0171, +0.0009 / 0.0158 | +0.0018 / 0.0080, −0.0000 / 0.0104 | | |
| | | b_0, b_1 | −0.015 / 0.189, −0.043 / 0.115 | +0.011 / 0.129, −0.023 / 0.062 | | |

The rates π̂ and onsets â are unbiased to within a tenth of their RMSE,
which falls as 1/√N (π_1 of δ = 0.10: 0.0254, 0.0167, 0.0126, 0.0078 at
N = 500 to 5000). The persistence b̂ is biased down, by 0.01–0.06 at
N ≤ 1000 and ≤ 0.032 at N = 2000 (cause not isolated), and its RMSE falls
slowly: it is estimated from the continuations of the bursts of one state
only — for the state with 1 % onsets, b̂_0 still has an RMSE of 0.098 at
N = 5000. HMC-DN (whose states are less persistent, A = 0.9): RMSE 1.1–1.5×
HMC-IN's at the same N.

### Warp-speed against a genuine bootstrap (`results/lr_direct.csv`)

HMC-IN, `"state"` vs common, π = (0.1, 0.1), N = 500: 200 replications,
each with its own B = 99 bootstrap (`missingness_lr_test(...,
n_bootstrap=99)`; every replicate valid), 510 s on 6 processes.

| method | R | reject χ² | reject bootstrap |
|---|---|---|---|
| genuine bootstrap, B = 99 | 200 | 0.070 ± 0.018 | 0.050 ± 0.015 |
| warp-speed (table above, other seeds) | 400 | 0.058 ± 0.012 | 0.050 ± 0.011 |

The two estimates of the bootstrap test's size agree (0.050 and 0.050).

## Diagnosis: the Markov test on HMC-DN at N = 500 (`lr_diagnosis.py`)

The cell "HMC-DN, `"state-markov"` vs common, H0, N = 500" of the first
table, taken apart on its 200 null replications (same seeds; the fits are
bit-identical to the study's). At the time the statistic was the difference
of the two ICE fixed points, LL1(θ̂1, φ̂1) − LL0(θ̂0). ℓ_h(θ) is the
log-likelihood maximised over hypothesis h's mechanism at θ fixed (here
L-BFGS-B on the logits with the exact gradient, no guard), so ℓ0(θ) =
LL_ignorable(θ) + log p(m | common MLE).

`results/lr_decompose_hmc_dn_markov-null_500.csv` (5 min on 5 processes):

| statistic (× 2) | mean | sd | min | 95 % quantile | max | # < 0 | reject at χ²(2) 5.99 |
|---|---|---|---|---|---|---|---|
| fits: LL1(θ̂1, φ̂1) − LL0(θ̂0) | 2.73 | 2.68 | −1.42 | 8.05 | 16.63 | 9 | 0.135 |
| its θ-part: LL0(θ̂1) − LL0(θ̂0) | −0.09 | 0.90 | −5.87 | 0.76 | 4.73 | 135 | — |
| its mechanism part: LL1(θ̂1, φ̂1) − LL0(θ̂1) | 2.82 | 2.52 | 0.001 | 8.17 | 14.30 | 0 | 0.120 |
| mechanism at θ̂0: ℓ1(θ̂0) − ℓ0(θ̂0) | 2.52 | 2.27 | 0.001 | 7.20 | 11.13 | 0 | 0.105 |
| mechanism at θ̂1: ℓ1(θ̂1) − ℓ0(θ̂1) | 2.98 | 2.63 | 0.001 | 8.58 | 14.31 | 0 | 0.135 |
| one guarded mechanism M-step at θ̂0 | 1.38 | 1.47 | 0.000 | 4.11 | 9.94 | 0 | 0.015 |
| **profile statistic** (below) | 2.83 | 2.53 | 0.0005 | 8.24 | 14.31 | 0 | 0.125 |
| χ²(2) | 2 | 2 | 0 | 5.99 | | | 0.05 |

`results/lr_mle_hmc_dn_markov-null_500.csv`: a direct maximisation of
log p(y_obs, m) over all the parameters (A, two Gaussian margins, four
Gaussian copulas; + a_i, b_i: 10 and 14 parameters), L-BFGS-B with
forward-difference gradients, best of three starts per hypothesis (null: θ̂0,
θ̂1, the true θ; alternative: the alternative fit, ICE's best iterate, the
null maximum with the common mask). 36 replications — 0 to 29, and 78, 197,
198 (largest fits' statistics with 11) and 89, 111, 143 (most negative, with
7); 15 min on 5 processes. Error of each statistic against that LR:

| statistic | mean \|err\| | rms | max \|err\| | mean err | # \|err\| > 1 |
|---|---|---|---|---|---|
| fits (the study's) | 1.02 | 1.71 | 6.15 | −0.31 | 13 |
| fits + the null at θ̂1 | 0.82 | 1.46 | 4.73 | −0.42 | 10 |
| mechanism at θ̂0 only | 1.11 | 1.88 | 6.30 | −0.69 | 13 |
| **profile statistic** | 0.61 | 1.12 | 3.88 | −0.08 | 9 |

(a) **The two fits differ in how well they optimise θ.** ICE on HMC-DN is
not EM (copula step on the pairs with both ends observed, margin step without
the copula): its fixed points are below the maxima — the null fit a median
0.27 nat below the direct maximum, the alternative 0.51, both up to 4.6. The
alternative's run, started at the null fit, can climb and then drift to a
fixed point at another θ: replication 111 gains 2.29 nat in 17 iterations,
then ends 0.71 nat *below* its start (fits' statistic −1.42, exact LR 4.73;
the state-0 mean moved from −0.92 to +0.07). The alternative ends ≥ 0.1 nat
below its own best iterate in 15 of 200 runs (max 3.0), the ignorable null
in 48 (max 2.3); the alternative's margins move by more than 0.3 (sum of
|Δμ_i|) in 17 runs. The θ-part (sd 0.90) explains every negative value
(the mechanism part is ≥ 0.001) and adds spread both ways.

(b) **Not maximising is not what makes the tail.** The exact LR of the
first 30 replications has a mean of 3.40 — against 2 for χ²(2) — and 5 of 30
exceed 5.99 (the fits: mean 3.23, 5 of 30). The four largest fits'
statistics (11.1–16.6) are confirmed by the exact LR (10.6–14.0). The
mechanism's LR at θ̂0 alone, where no θ-difference enters, still rejects
0.105.

(c) **Not few bursts, not the guard.** The rejection rate of the fits'
statistic grows with the number of bursts: 0.034 with 3–7 bursts (59
series), 0.190 with 8–9 (58), 0.137 with 10–11 (51), 0.219 with 12–15 (32).
The guard lowers the statistic: ℓ1(θ̂1) − LL1 (the alternative fit is a
posterior mode) is 0.16 on average (× 2), up to 1.5. In 12 of the 36 exact
alternatives one state has no onset (â_i < 1e-4): a boundary maximum, which
shrinks an LR rather than inflating it.

(d) **The warp-speed bootstrap.** In the same cell its replicates LR*,
simulated from each fitted null θ̂0, have a mean of 1.99 and exceed 5.99 in
0.030 of the replications, against 2.73 and 0.135 for LR: the bootstrap world
does not reproduce the tail of the real one, and its critical value (5.06)
is below χ²'s. A **genuine** bootstrap (B = 99 per replication, the profile
statistic, `results/lr_direct_dn.csv`, `lr_study.py --direct-study`; 13–23
min per replication on the loaded machine) on the four largest null values
and on replications 0 and 1:

| r | LR (fits) | LR (profile) | p χ² | p bootstrap | bootstrap 95 % quantile | bootstrap mean |
|---|---|---|---|---|---|---|
| 78 | 16.63 | 14.31 | 0.0008 | 0.010 | 6.25 | 2.24 |
| 198 | 11.09 | 11.43 | 0.0033 | 0.010 | 5.74 | 1.98 |
| 197 | 11.85 | 11.24 | 0.0036 | 0.010 | 6.94 | 2.73 |
| 11 | 11.72 | 10.60 | 0.0050 | 0.010 | 5.16 | 2.09 |
| 0 | 6.20 | 6.27 | 0.0436 | 0.050 | 5.92 | 2.01 |
| 1 | 1.22 | 1.27 | 0.5309 | 0.680 | 6.73 | 2.45 |

Every replicate valid. Each series' own bootstrap distribution is close to
χ²(2) (95 % quantiles 5.2–6.9), and none of the 99 replicates reaches any of
the four largest values: the genuine bootstrap rejects them at p = 0.01,
like χ² — it does not fix the size either. Series simulated from the fitted
null do not have the real series' tail.

**Conclusion.** The negative values and part of the spread were the fits'
unequal optimisation of θ — the profile statistic removes them. The excess of
rejections is the likelihood ratio's own at N = 500 on this model (~9
bursts): neither a better maximiser nor the bootstrap, warp-speed or
genuine, removes it. At N = 1000 (~18 bursts) χ² holds the level with the
profile statistic (0.045 ± 0.015, paired table). **Use the Markov test on a
grid variant from ~20 bursts (here N ≈ 1000) on; below, a rejection is not
evidence at the nominal level.** The cause of the tail was not isolated
further: it is not the number of bursts alone (the rate grows with it at
N = 500), not ICE, not the guard.

### Reference

* Giacomini, R., Politis, D. N. & White, H. (2013). A warp-speed method for
  conducting Monte Carlo experiments involving bootstrap estimators.
  *Econometric Theory* 29(3), 567–589.
