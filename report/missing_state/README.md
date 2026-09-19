# State-dependent missingness: estimation and likelihood-ratio test (P6)

Two scripts, both run from the repository root with the repo venv.

| script | what | time (6 processes, loaded Apple arm64 laptop) |
|---|---|---|
| `design_measurements.py` | the design choices of the mechanism M-step of ICE/SEM: initial term of `"state-markov"`, boundary guard, starts | 5 min |
| `lr_study.py` | size, power (asymptotic χ² and parametric bootstrap) of `missingness_lr_test`, recovery of π̂, â, b̂ | 59 min (`--direct`: see below) |

Every seed is a `zlib.crc32` of a named tuple; a path and its mask never share
a seed. Results (one row per replication) are in `results/`.

```bash
.venv/bin/python report/missing_state/design_measurements.py --jobs 6
.venv/bin/python report/missing_state/lr_study.py --jobs 4 --quick    # smoke: 4 per cell → results/lr_study_quick.csv (not versioned)
.venv/bin/python report/missing_state/lr_study.py --jobs 6            # full study
.venv/bin/python report/missing_state/lr_study.py --jobs 6 --direct   # B = 99 bootstrap check
.venv/bin/python report/missing_state/lr_study.py --summarise         # tables from the CSV files
```

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

### Size and power at the 5 % level (`results/lr_study.csv`)

Rejection rate ± binomial standard error; "crit" is the warp-speed
bootstrap critical value (χ²: 3.84 for df = 1, 5.99 for df = 2).

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
  give 0.007 to 0.21).
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
  are negative (min −1.42) and the largest is 16.6: in some replications
  the two ICE fixed points differ by more than the mechanism's contribution,
  and the bootstrap world (simulated from the fitted θ̂0, ~9 bursts per
  series) does not reproduce that tail. The cause was not isolated. At
  N = 1000 the test is conservative (0.030 / 0.015) and its power low
  (0.24). Use the Markov test on grid variants with care below ~20 bursts.

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

### Reference

* Giacomini, R., Politis, D. N. & White, H. (2013). A warp-speed method for
  conducting Monte Carlo experiments involving bootstrap estimators.
  *Econometric Theory* 29(3), 567–589.
