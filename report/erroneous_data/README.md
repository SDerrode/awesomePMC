# Erroneous data (pilot): isolated spikes, PIT flags, flag-and-mask ICE

Simulation study of `pmcprg.pmc.outliers` (erroneous-data pilot). Wrong
observations (isolated spikes, a sensor glitch) are flagged by the model's
own one-step-ahead predictive test, and the flagged rows are neutralised by
treating them as missing values:

- `predictive_pit`: PIT_n = P(Y_n ≤ y_n | past) from the forward filter and
  the transition's conditional CDFs, with exact gaps (see the module
  docstring for the formulas per variant);
- `flag_outliers`: flag p_n = 2 min(PIT_n, 1 − PIT_n) < α. With
  `sequential=True` a flagged row is integrated out of the filter of the
  rows after it (innovation gating);
- `robust_estimate`: fit, flag, set the flags to NaN, refit from the
  initial model, until the mask is a fixed point. With ICE's
  `missing_strategy = "available"` this is a trimmed ICE: a masked row
  weighs nothing in the margins, and the pairs that touch it weigh nothing
  in the copulas.

## Design

**Models.** `hmc_in_gauss_k2` (HMC-IN, N(−1, 1) / N(1, 1), A_ii = 0.9) and
`pmc_gauss_k2` (PMC with state margins N(−1, 1) / N(1, 1), p_ii = 0.45,
Gaussian copulas τ = 0.6 on the diagonal pairs and 0 off it). With state
margins its transitions are those of the HMC-DN fixture.

**Contamination.** A clean series of N = 2000 rows is simulated. Then
round(rate·N) isolated positions get y_n ← y_n + k·sd, where sd is the
standard deviation of the clean series (≈ 1.41). Two spikes are never
adjacent, and neither the first nor the last row is contaminated. The
grid is k ∈ {4, 6, 8} × rate ∈ {1 %, 5 %}, plus a clean cell (rate 0), with
30 replicates per cell (420 series).

**Seeds** are `zlib.crc32` of labels. The clean series depends on (model,
replicate) only, so every cell of a replicate contaminates the same series.
The spike positions depend on (model, k, rate, replicate).

**Methods.** Every fit is ICE started at the true model, with
`fit_margins = True`, Gaussian copulas only for PMC, and the package
defaults otherwise (`tol = 1e-4`, `missing_strategy = "available"`).

| method | what |
|---|---|
| `true_seq`, `true_nonseq` | `flag_outliers` with the true model, α = 1e-3, with and without gating; `true_seq` also classifies the series with its flags masked |
| `true_raw` | classification by the true model of the raw series |
| `clean` | ICE on the clean series (the spikes never happened) |
| `raw` | ICE on the contaminated series |
| `oracle` | ICE with the true spike rows set to NaN: the best a mask can do |
| `fm` | `robust_estimate` (default: refits from the initial model, α = 1e-3) |
| `fm_hampel` | `robust_estimate` with `initial_mask` = a Hampel pre-screen: \|y_n − median\| > 4 · 1.4826 · MAD over a centred window of 11 rows |
| `fm_warm` | the same loop, each refit started from the previous fit (the rejected design) |
| `dpd` | PMC only: τ of the diagonal copulas by minimum density power divergence (α_DPD = 0.25, `pmcprg.copulas._robust.dpd_fit`) on the raw fit's pseudo-pairs (F_i(y_n), F_i(y_{n+1})), weights ξ_n(i, i) of the raw fit; the other parameters are those of `raw` |

**Scores.**

- Detection rate on the spike rows; false-flag rate on the clean rows; and
  the false flags on the row right after a spike (the swamping that gating
  prevents).
- Bias and RMSE of μ_i, σ_i, A_ii = p_ii / Σ_j p_ij and τ_ii.
- Classification error at the clean rows. Each fit classifies the series it
  was fitted on, with its masked rows integrated out; labels are aligned by
  the Hungarian assignment.

## Run

From the repository root:

```bash
# Smoke run: 2 replicates, N = 1000 → results/quick/ (not versioned), 17 s with 4 processes
PYTHONPATH=. .venv/bin/python report/erroneous_data/run_study.py --quick --jobs 4
# Full study (the tables below)
PYTHONPATH=. OMP_NUM_THREADS=1 .venv/bin/python report/erroneous_data/run_study.py --reps 30 --jobs 4
# Tables only, from results/runs.csv
PYTHONPATH=. .venv/bin/python report/erroneous_data/run_study.py --summarise
```

Outputs: `results/runs.csv` (one row per series × method), `results/run_info.json`
and `results/tables.md` (the tables below, generated).

## Findings (30 replicates per cell, N = 2000, α = 1e-3)

The full run took 5.9 min with 4 processes on an Apple arm64 laptop
(Python 3.14, numpy 2.5, scipy 1.18).

1. **Detection with the true model.**
   - Spikes of 6 and 8 sd are all found (detection ≥ 0.999).
   - Spikes of 4 sd are found 90 % of the time on HMC-IN and 92–94 % on
     PMC: a +5.7 jump can land within the range of the upper regime.
   - The false-flag rate stays near α: 0.86–1.12 per 1000 clean rows on
     HMC-IN, 0.97–2.26 on PMC (next point).
2. **Innovation gating** helps under heavy contamination and costs a little
   on clean data.
   - PMC at 5 % of 6 sd: false flags drop from 2.26 to 1.26 per 1000 with
     gating, and false flags on the row right after a spike from 11 to 1
     (total over 30 series). At 8 sd: 7 to 5; at 4 sd: 12 and 13, no gain.
   - HMC-IN at 5 %: neighbour false flags drop from 8–11 to 4–5.
   - Cost on clean PMC series: 1.27 false flags per 1000 with gating
     against 0.97 without. A legitimate extreme value, once gated, makes its
     dependent successor surprising too.
   - The effect is modest on these fixtures because 10 % of the transitions
     go to independent pairs (τ = 0 off the diagonal), which keeps the
     neighbour's predictive law wide. With strong dependence on every
     transition (the AR(1) regime with ρ = 0.9 of `test_outliers.py`), the
     neighbour of a spike gets p = 5e-65 without gating and is not flagged
     with it.
3. **Spikes spoil the classification of the clean rows, even with the true
   model.** With the true model, the clean-row error is 14–25 % on the raw
   PMC series against 11.3 % without spikes, and 11.5–11.7 % once the
   sequential flags are masked. On HMC-IN: 6.3–7.8 % against 6.0 %, and
   6.1–6.6 % once masked.
4. **Raw ICE breaks down early.**
   - PMC: at 1 % of spikes already, the raw fit turns a regime into a
     spike state: A_11 < 0.7 in 88 of 90 runs, clean-row error 44–46 %, τ
     biased by +0.10 and +0.22 to +0.29.
   - HMC-IN: at 1 % the raw fit inflates σ_1 (+0.25 to +0.84). At 5 % of 6
     and 8 sd it fits a state at the spikes' level (μ_1 = 8.5 and 11.3) in
     60 of 60 runs, with a clean-row error of 47.6 %.
5. **Flag-and-mask reaches the oracle mask whenever the first fit still
   flags some spikes.** This covers HMC-IN at 1 % and at 5 % of 4 sd, and
   PMC at 1 %.
   - Bias within 0.02 of the oracle, except the scales, shrunk by about 0.01
     by the trimming.
   - RMSE within 15 % of the oracle's.
   - Clean-row error 11.7 % on PMC (oracle 11.5–11.6 %) and 6.2–6.8 % on
     HMC-IN (oracle 6.1–6.3 %).
   - It takes 2 to 6 fits on PMC at 1 %, 2 to 5 times the cost of a single
     ICE (0.15–2.0 s per series).
   - On HMC-IN at 5 % of 4 sd, σ_1 stays biased by +0.05 (oracle +0.002):
     13 % of those spikes are plausible under the upper regime and are never
     flagged.
6. **It breaks down when the spikes form a regime**: PMC at 5 % for every
   k, and HMC-IN at 5 % of 6 and 8 sd.
   - The first fit's spike state explains the spikes, nothing is flagged,
     and the loop stops after 1–3 fits with the raw estimate.
   - A Hampel pre-screen as `initial_mask` restores the oracle level at
     5 % of 6 and 8 sd, in all 120 runs of both models. Every mean bias is
     within 0.012 of the oracle's. PMC's clean-row error is 11.9 %, against
     11.7 % for the oracle.
   - The exception is PMC at 5 % of 4 sd: the pre-screen catches only 68 %
     of the spikes, and 27 of 30 runs stay in the spike basin.
   - The pre-screen masks 7–9 clean rows per 1000, which the model then
     releases.
7. **Warm restarts fail.** On PMC at 1 %, refits started from the previous
   fit end with a clean-row error above 20 % in 72 of 90 runs (regimes
   merged in 70). Refits from the initial model: 0 of 90. This is why
   `robust_estimate` restarts every refit from the initial model.
8. **Copula-level robustness is no substitute.** The DPD re-estimate of the
   diagonal τ on the raw fit's pseudo-pairs moves τ by less than 0.01 (τ_11
   bias +0.216 against +0.219 for raw, at 1 % of 4 sd). The contaminated
   states and margins, not the copula fit, carry the damage, so the rows
   themselves have to be handled.
9. **Cost on clean data.**
   - Rows masked: 1.0 (HMC-IN) and 1.3 (PMC) per 1000.
   - σ bias: −0.008 and −0.020, against −0.002 and −0.009 for ICE.
   - Clean-row error: +0.02 and +0.16 points.

**Next.** Point 6 is a breakdown of any flag-and-refit scheme started from
a contaminated fit. The principled remedy is a contamination component in
the model itself: a per-row indicator c_n with P(c_n = 1) = ε, where a
contaminated y_n is drawn from a broad law g and its clean value is
integrated out, as for a gap. The spikes then have a better explanation
than a regular state. This is proposal (6) of the research note.

## Tables


N = 2000, 30 replicates per cell, α = 0.001; 420 series, 5.9 min with 4 processes (Darwin arm64).

### Detection with the true model (flag_outliers, α = 1e-3)

Detection rate on the spikes / false-flag rate on the clean rows (per 1000) / false flags right after a spike (total over the replicates).

| model | k | rate | sequential | non-sequential |
|---|---|---|---|---|
| HMC-IN | – | 0% | – / 0.98 / – | – / 0.98 / – |
| HMC-IN | 4 | 1% | 0.903 / 0.98 / 0 | 0.903 / 0.99 / 1 |
| HMC-IN | 6 | 1% | 1.000 / 0.98 / 0 | 1.000 / 0.98 / 0 |
| HMC-IN | 8 | 1% | 1.000 / 0.99 / 0 | 1.000 / 1.03 / 2 |
| HMC-IN | 4 | 5% | 0.900 / 1.00 / 4 | 0.897 / 1.12 / 9 |
| HMC-IN | 6 | 5% | 0.999 / 0.95 / 5 | 0.999 / 0.96 / 8 |
| HMC-IN | 8 | 5% | 1.000 / 0.86 / 4 | 1.000 / 1.02 / 11 |
| PMC | – | 0% | – / 1.27 / – | – / 0.97 / – |
| PMC | 4 | 1% | 0.937 / 1.28 / 1 | 0.937 / 1.11 / 1 |
| PMC | 6 | 1% | 1.000 / 1.26 / 0 | 1.000 / 1.16 / 2 |
| PMC | 8 | 1% | 1.000 / 1.26 / 0 | 1.000 / 1.16 / 2 |
| PMC | 4 | 5% | 0.920 / 1.44 / 13 | 0.918 / 1.79 / 12 |
| PMC | 6 | 5% | 1.000 / 1.26 / 1 | 1.000 / 2.26 / 11 |
| PMC | 8 | 5% | 1.000 / 1.28 / 5 | 1.000 / 1.82 / 7 |

### HMC-IN: bias of the ICE estimates (mean over replicates of estimate − truth)

| k | rate | method | mu0 | mu1 | sd0 | sd1 | A00 | A11 | fits | converged |
|---|---|---|---|---|---|---|---|---|---|---|
| – | 0% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| – | 0% | fm | +0.009 | -0.007 | -0.008 | -0.004 | -0.001 | -0.000 | 1.8 | 30/30 |
| 4 | 1% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 4 | 1% | raw | -0.003 | +0.005 | -0.003 | +0.248 | -0.005 | +0.006 |  |  |
| 4 | 1% | fm | +0.006 | -0.005 | -0.009 | +0.004 | -0.002 | -0.001 | 3.2 | 30/30 |
| 4 | 1% | fm_hampel | +0.006 | -0.005 | -0.009 | +0.004 | -0.002 | -0.001 | 2.8 | 30/30 |
| 4 | 1% | oracle | +0.007 | -0.003 | -0.002 | +0.003 | -0.000 | -0.000 |  |  |
| 6 | 1% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 6 | 1% | raw | +0.028 | -0.018 | +0.004 | +0.551 | +0.001 | +0.017 |  |  |
| 6 | 1% | fm | +0.008 | -0.007 | -0.008 | -0.005 | -0.001 | -0.000 | 2.8 | 30/30 |
| 6 | 1% | fm_hampel | +0.008 | -0.007 | -0.009 | -0.005 | -0.001 | -0.000 | 2.2 | 30/30 |
| 6 | 1% | oracle | +0.006 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 8 | 1% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 8 | 1% | raw | +0.115 | +0.362 | +0.028 | +0.836 | +0.010 | -0.012 |  |  |
| 8 | 1% | fm | +0.009 | -0.007 | -0.008 | -0.005 | -0.000 | -0.000 | 2.7 | 30/30 |
| 8 | 1% | fm_hampel | +0.009 | -0.007 | -0.008 | -0.005 | -0.000 | -0.000 | 2.2 | 30/30 |
| 8 | 1% | oracle | +0.008 | -0.004 | -0.002 | +0.001 | -0.000 | -0.000 |  |  |
| 4 | 5% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 4 | 5% | raw | +0.052 | +0.201 | -0.006 | +0.865 | -0.039 | -0.004 |  |  |
| 4 | 5% | fm | -0.002 | +0.002 | -0.010 | +0.050 | -0.011 | -0.005 | 8.5 | 29/30 |
| 4 | 5% | fm_hampel | -0.002 | +0.002 | -0.010 | +0.050 | -0.011 | -0.005 | 6.5 | 30/30 |
| 4 | 5% | oracle | +0.007 | -0.006 | -0.001 | +0.002 | -0.000 | -0.000 |  |  |
| 6 | 5% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 6 | 5% | raw | +1.003 | +7.456 | +0.411 | +0.382 | +0.048 | -0.900 |  |  |
| 6 | 5% | fm | +1.003 | +7.436 | +0.411 | +0.358 | +0.048 | -0.900 | 1.6 | 30/30 |
| 6 | 5% | fm_hampel | +0.008 | -0.008 | -0.008 | -0.002 | -0.001 | -0.000 | 2.9 | 30/30 |
| 6 | 5% | oracle | +0.006 | -0.005 | -0.003 | +0.004 | -0.001 | -0.000 |  |  |
| 8 | 5% | clean | +0.007 | -0.004 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |
| 8 | 5% | raw | +1.000 | +10.293 | +0.409 | +0.399 | +0.047 | -0.900 |  |  |
| 8 | 5% | fm | +1.001 | +10.275 | +0.409 | +0.378 | +0.048 | -0.900 | 1.5 | 30/30 |
| 8 | 5% | fm_hampel | +0.007 | -0.009 | -0.008 | -0.004 | -0.001 | -0.000 | 2.6 | 30/30 |
| 8 | 5% | oracle | +0.005 | -0.006 | -0.002 | +0.002 | -0.000 | -0.000 |  |  |

### PMC: bias of the ICE estimates (mean over replicates of estimate − truth)

| k | rate | method | mu0 | mu1 | sd0 | sd1 | A00 | A11 | tau00 | tau11 | fits | converged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| – | 0% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| – | 0% | fm | +0.031 | +0.000 | -0.020 | -0.008 | -0.002 | -0.003 | -0.011 | -0.004 | 2.0 | 30/30 |
| 4 | 1% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| 4 | 1% | raw | +0.924 | -0.420 | +0.350 | +0.899 | -0.010 | -0.344 | +0.111 | +0.219 |  |  |
| 4 | 1% | fm | +0.034 | +0.004 | -0.019 | -0.004 | -0.004 | -0.005 | -0.011 | -0.004 | 3.9 | 30/30 |
| 4 | 1% | fm_hampel | +0.034 | +0.005 | -0.019 | -0.005 | -0.004 | -0.005 | -0.011 | -0.004 | 3.0 | 30/30 |
| 4 | 1% | oracle | +0.032 | -0.000 | -0.009 | +0.002 | -0.002 | -0.003 | -0.007 | +0.000 |  |  |
| 4 | 1% | dpd | +0.924 | -0.420 | +0.350 | +0.899 | -0.010 | -0.344 | +0.108 | +0.216 |  |  |
| 6 | 1% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| 6 | 1% | raw | +0.969 | -0.298 | +0.345 | +1.548 | +0.006 | -0.366 | +0.104 | +0.268 |  |  |
| 6 | 1% | fm | +0.032 | -0.003 | -0.018 | -0.005 | -0.002 | -0.003 | -0.010 | -0.003 | 3.4 | 30/30 |
| 6 | 1% | fm_hampel | +0.033 | -0.004 | -0.017 | -0.004 | -0.002 | -0.003 | -0.009 | -0.002 | 2.6 | 30/30 |
| 6 | 1% | oracle | +0.031 | -0.000 | -0.010 | +0.001 | -0.002 | -0.004 | -0.007 | -0.002 |  |  |
| 6 | 1% | dpd | +0.969 | -0.298 | +0.345 | +1.548 | +0.006 | -0.366 | +0.103 | +0.266 |  |  |
| 8 | 1% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| 8 | 1% | raw | +0.976 | -0.100 | +0.336 | +2.192 | +0.016 | -0.349 | +0.097 | +0.290 |  |  |
| 8 | 1% | fm | +0.034 | -0.004 | -0.018 | -0.004 | -0.002 | -0.003 | -0.008 | -0.001 | 3.2 | 30/30 |
| 8 | 1% | fm_hampel | +0.034 | -0.004 | -0.018 | -0.004 | -0.002 | -0.003 | -0.008 | -0.001 | 2.6 | 30/30 |
| 8 | 1% | oracle | +0.034 | +0.000 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.001 |  |  |
| 8 | 1% | dpd | +0.976 | -0.100 | +0.336 | +2.192 | +0.016 | -0.349 | +0.098 | +0.288 |  |  |
| 4 | 5% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| 4 | 5% | raw | +0.972 | +1.495 | +0.374 | +1.928 | -0.034 | -0.827 | +0.109 | +0.169 |  |  |
| 4 | 5% | fm | +0.971 | +1.494 | +0.374 | +1.919 | -0.034 | -0.826 | +0.109 | +0.169 | 1.3 | 30/30 |
| 4 | 5% | fm_hampel | +0.869 | +1.333 | +0.337 | +1.724 | -0.034 | -0.747 | +0.098 | +0.275 | 4.4 | 30/30 |
| 4 | 5% | oracle | +0.034 | +0.001 | -0.011 | +0.001 | -0.002 | -0.003 | -0.007 | -0.000 |  |  |
| 4 | 5% | dpd | +0.972 | +1.495 | +0.374 | +1.928 | -0.034 | -0.827 | +0.110 | +0.169 |  |  |
| 6 | 5% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| 6 | 5% | raw | +0.987 | +2.984 | +0.373 | +3.283 | -0.018 | -0.839 | +0.097 | +0.193 |  |  |
| 6 | 5% | fm | +0.987 | +2.975 | +0.373 | +3.282 | -0.018 | -0.839 | +0.097 | +0.193 | 1.1 | 30/30 |
| 6 | 5% | fm_hampel | +0.033 | -0.000 | -0.017 | -0.009 | -0.002 | -0.004 | -0.009 | -0.003 | 3.4 | 30/30 |
| 6 | 5% | oracle | +0.032 | +0.000 | -0.009 | -0.001 | -0.003 | -0.004 | -0.006 | -0.001 |  |  |
| 6 | 5% | dpd | +0.987 | +2.984 | +0.373 | +3.283 | -0.018 | -0.839 | +0.101 | +0.193 |  |  |
| 8 | 5% | clean | +0.032 | +0.001 | -0.009 | +0.001 | -0.002 | -0.004 | -0.006 | -0.000 |  |  |
| 8 | 5% | raw | +0.991 | +4.680 | +0.374 | +4.654 | -0.008 | -0.849 | +0.089 | +0.098 |  |  |
| 8 | 5% | fm | +0.991 | +4.666 | +0.373 | +4.649 | -0.008 | -0.848 | +0.089 | +0.098 | 1.2 | 30/30 |
| 8 | 5% | fm_hampel | +0.033 | +0.000 | -0.021 | -0.008 | -0.002 | -0.003 | -0.011 | -0.005 | 3.0 | 30/30 |
| 8 | 5% | oracle | +0.032 | -0.002 | -0.010 | +0.002 | -0.002 | -0.003 | -0.007 | +0.000 |  |  |
| 8 | 5% | dpd | +0.991 | +4.680 | +0.374 | +4.654 | -0.008 | -0.849 | +0.096 | +0.099 |  |  |

### HMC-IN: RMSE of the ICE estimates

| k | rate | method | mu0 | mu1 | sd0 | sd1 | A00 | A11 |
|---|---|---|---|---|---|---|---|---|
| 4 | 1% | clean | 0.030 | 0.033 | 0.023 | 0.026 | 0.010 | 0.008 |
| 4 | 1% | raw | 0.034 | 0.043 | 0.024 | 0.252 | 0.012 | 0.011 |
| 4 | 1% | fm | 0.028 | 0.034 | 0.025 | 0.025 | 0.011 | 0.008 |
| 4 | 1% | fm_hampel | 0.028 | 0.034 | 0.026 | 0.025 | 0.011 | 0.008 |
| 4 | 1% | oracle | 0.029 | 0.034 | 0.023 | 0.025 | 0.010 | 0.008 |
| 6 | 1% | clean | 0.030 | 0.033 | 0.023 | 0.026 | 0.010 | 0.008 |
| 6 | 1% | raw | 0.047 | 0.045 | 0.025 | 0.553 | 0.011 | 0.019 |
| 6 | 1% | fm | 0.029 | 0.033 | 0.024 | 0.025 | 0.010 | 0.008 |
| 6 | 1% | fm_hampel | 0.029 | 0.033 | 0.025 | 0.025 | 0.010 | 0.008 |
| 6 | 1% | oracle | 0.030 | 0.033 | 0.023 | 0.025 | 0.010 | 0.008 |
| 8 | 1% | clean | 0.030 | 0.033 | 0.023 | 0.026 | 0.010 | 0.008 |
| 8 | 1% | raw | 0.212 | 1.938 | 0.073 | 0.844 | 0.021 | 0.166 |
| 8 | 1% | fm | 0.030 | 0.034 | 0.025 | 0.026 | 0.010 | 0.008 |
| 8 | 1% | fm_hampel | 0.030 | 0.034 | 0.025 | 0.026 | 0.010 | 0.008 |
| 8 | 1% | oracle | 0.030 | 0.034 | 0.023 | 0.026 | 0.010 | 0.008 |
| 4 | 5% | clean | 0.030 | 0.033 | 0.023 | 0.026 | 0.010 | 0.008 |
| 4 | 5% | raw | 0.071 | 0.208 | 0.028 | 0.866 | 0.041 | 0.013 |
| 4 | 5% | fm | 0.031 | 0.034 | 0.028 | 0.062 | 0.016 | 0.011 |
| 4 | 5% | fm_hampel | 0.031 | 0.034 | 0.028 | 0.062 | 0.016 | 0.011 |
| 4 | 5% | oracle | 0.032 | 0.035 | 0.025 | 0.023 | 0.010 | 0.008 |
| 6 | 5% | clean | 0.030 | 0.033 | 0.023 | 0.026 | 0.010 | 0.008 |
| 6 | 5% | raw | 1.004 | 7.458 | 0.412 | 0.389 | 0.048 | 0.900 |
| 6 | 5% | fm | 1.005 | 7.437 | 0.411 | 0.368 | 0.048 | 0.900 |
| 6 | 5% | fm_hampel | 0.028 | 0.037 | 0.024 | 0.025 | 0.010 | 0.008 |
| 6 | 5% | oracle | 0.028 | 0.037 | 0.022 | 0.026 | 0.010 | 0.008 |
| 8 | 5% | clean | 0.030 | 0.033 | 0.023 | 0.026 | 0.010 | 0.008 |
| 8 | 5% | raw | 1.002 | 10.296 | 0.410 | 0.407 | 0.047 | 0.900 |
| 8 | 5% | fm | 1.002 | 10.277 | 0.409 | 0.386 | 0.048 | 0.900 |
| 8 | 5% | fm_hampel | 0.030 | 0.036 | 0.025 | 0.024 | 0.010 | 0.008 |
| 8 | 5% | oracle | 0.031 | 0.036 | 0.023 | 0.025 | 0.010 | 0.008 |

### PMC: RMSE of the ICE estimates

| k | rate | method | mu0 | mu1 | sd0 | sd1 | A00 | A11 | tau00 | tau11 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1% | clean | 0.079 | 0.075 | 0.050 | 0.056 | 0.013 | 0.013 | 0.021 | 0.025 |
| 4 | 1% | raw | 0.937 | 0.472 | 0.357 | 0.916 | 0.017 | 0.355 | 0.112 | 0.220 |
| 4 | 1% | fm | 0.083 | 0.077 | 0.056 | 0.061 | 0.014 | 0.014 | 0.025 | 0.028 |
| 4 | 1% | fm_hampel | 0.083 | 0.079 | 0.056 | 0.062 | 0.014 | 0.014 | 0.025 | 0.028 |
| 4 | 1% | oracle | 0.080 | 0.079 | 0.050 | 0.054 | 0.013 | 0.013 | 0.022 | 0.024 |
| 4 | 1% | dpd | 0.937 | 0.472 | 0.357 | 0.916 | 0.017 | 0.355 | 0.110 | 0.218 |
| 6 | 1% | clean | 0.079 | 0.075 | 0.050 | 0.056 | 0.013 | 0.013 | 0.021 | 0.025 |
| 6 | 1% | raw | 0.975 | 0.345 | 0.350 | 1.561 | 0.012 | 0.372 | 0.105 | 0.269 |
| 6 | 1% | fm | 0.081 | 0.076 | 0.055 | 0.058 | 0.013 | 0.013 | 0.025 | 0.025 |
| 6 | 1% | fm_hampel | 0.080 | 0.076 | 0.054 | 0.056 | 0.013 | 0.013 | 0.024 | 0.024 |
| 6 | 1% | oracle | 0.079 | 0.076 | 0.051 | 0.056 | 0.013 | 0.013 | 0.022 | 0.025 |
| 6 | 1% | dpd | 0.975 | 0.345 | 0.350 | 1.561 | 0.012 | 0.372 | 0.104 | 0.266 |
| 8 | 1% | clean | 0.079 | 0.075 | 0.050 | 0.056 | 0.013 | 0.013 | 0.021 | 0.025 |
| 8 | 1% | raw | 0.982 | 0.252 | 0.342 | 2.218 | 0.019 | 0.358 | 0.099 | 0.291 |
| 8 | 1% | fm | 0.082 | 0.073 | 0.054 | 0.058 | 0.013 | 0.013 | 0.024 | 0.025 |
| 8 | 1% | fm_hampel | 0.082 | 0.073 | 0.054 | 0.058 | 0.013 | 0.013 | 0.024 | 0.025 |
| 8 | 1% | oracle | 0.081 | 0.076 | 0.049 | 0.054 | 0.012 | 0.013 | 0.021 | 0.024 |
| 8 | 1% | dpd | 0.982 | 0.252 | 0.342 | 2.218 | 0.019 | 0.358 | 0.099 | 0.289 |
| 4 | 5% | clean | 0.079 | 0.075 | 0.050 | 0.056 | 0.013 | 0.013 | 0.021 | 0.025 |
| 4 | 5% | raw | 0.976 | 1.510 | 0.379 | 1.931 | 0.035 | 0.827 | 0.110 | 0.400 |
| 4 | 5% | fm | 0.975 | 1.509 | 0.379 | 1.922 | 0.035 | 0.827 | 0.110 | 0.400 |
| 4 | 5% | fm_hampel | 0.915 | 1.419 | 0.362 | 1.821 | 0.036 | 0.787 | 0.105 | 0.292 |
| 4 | 5% | oracle | 0.077 | 0.078 | 0.048 | 0.055 | 0.012 | 0.012 | 0.020 | 0.025 |
| 4 | 5% | dpd | 0.976 | 1.510 | 0.379 | 1.931 | 0.035 | 0.827 | 0.110 | 0.402 |
| 6 | 5% | clean | 0.079 | 0.075 | 0.050 | 0.056 | 0.013 | 0.013 | 0.021 | 0.025 |
| 6 | 5% | raw | 0.990 | 3.006 | 0.377 | 3.289 | 0.019 | 0.840 | 0.098 | 0.464 |
| 6 | 5% | fm | 0.990 | 2.997 | 0.377 | 3.288 | 0.019 | 0.839 | 0.098 | 0.465 |
| 6 | 5% | fm_hampel | 0.079 | 0.075 | 0.057 | 0.060 | 0.014 | 0.013 | 0.025 | 0.025 |
| 6 | 5% | oracle | 0.078 | 0.075 | 0.052 | 0.054 | 0.013 | 0.013 | 0.022 | 0.023 |
| 6 | 5% | dpd | 0.990 | 3.006 | 0.377 | 3.289 | 0.019 | 0.840 | 0.102 | 0.467 |
| 8 | 5% | clean | 0.079 | 0.075 | 0.050 | 0.056 | 0.013 | 0.013 | 0.021 | 0.025 |
| 8 | 5% | raw | 0.995 | 4.694 | 0.378 | 4.659 | 0.010 | 0.850 | 0.090 | 0.562 |
| 8 | 5% | fm | 0.994 | 4.681 | 0.377 | 4.654 | 0.010 | 0.849 | 0.090 | 0.562 |
| 8 | 5% | fm_hampel | 0.083 | 0.078 | 0.055 | 0.065 | 0.013 | 0.013 | 0.026 | 0.029 |
| 8 | 5% | oracle | 0.080 | 0.078 | 0.048 | 0.056 | 0.013 | 0.012 | 0.022 | 0.023 |
| 8 | 5% | dpd | 0.995 | 4.694 | 0.378 | 4.659 | 0.010 | 0.850 | 0.097 | 0.565 |

### Classification error at the clean rows (%), and the masks of the fitted procedures

Error: mean over replicates. Mask: detection rate / false-flag rate per 1000 clean rows.

| model | k | rate | true_raw | true_seq | clean | raw | fm | fm_hampel | fm_warm | oracle | mask fm | mask fm_hampel | mask fm_warm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HMC-IN | – | 0% | 6.03 | 6.06 | 6.10 | 6.10 | 6.12 | 6.12 | 6.13 | 6.10 | – / 1.02 | – / 1.07 | – / 1.02 |
| HMC-IN | 4 | 1% | 6.38 | 6.17 | 6.09 | 6.88 | 6.27 | 6.26 | 6.36 | 6.13 | 0.895 / 0.98 | 0.897 / 0.99 | 0.893 / 0.99 |
| HMC-IN | 6 | 1% | 6.31 | 6.07 | 6.09 | 7.75 | 6.15 | 6.15 | 6.21 | 6.12 | 1.000 / 1.04 | 1.000 / 1.08 | 1.000 / 1.04 |
| HMC-IN | 8 | 1% | 6.34 | 6.07 | 6.09 | 9.92 | 6.16 | 6.16 | 7.52 | 6.13 | 1.000 / 1.01 | 1.000 / 1.03 | 0.968 / 1.01 |
| HMC-IN | 4 | 5% | 7.44 | 6.55 | 6.12 | 9.70 | 6.77 | 6.77 | 6.98 | 6.29 | 0.868 / 0.82 | 0.869 / 0.82 | 0.860 / 0.86 |
| HMC-IN | 6 | 5% | 7.69 | 6.25 | 6.10 | 47.59 | 47.59 | 6.33 | 47.59 | 6.28 | 0.006 / 0.09 | 0.999 / 0.93 | 0.006 / 0.09 |
| HMC-IN | 8 | 5% | 7.81 | 6.25 | 6.11 | 47.66 | 47.66 | 6.37 | 47.66 | 6.32 | 0.005 / 0.05 | 1.000 / 0.91 | 0.005 / 0.05 |
| PMC | – | 0% | 11.32 | 11.44 | 11.48 | 11.48 | 11.64 | 11.64 | 11.67 | 11.48 | – / 1.32 | – / 1.35 | – / 1.30 |
| PMC | 4 | 1% | 14.15 | 11.48 | 11.49 | 43.62 | 11.74 | 11.75 | 37.90 | 11.52 | 0.925 / 1.35 | 0.928 / 1.38 | 0.863 / 0.66 |
| PMC | 6 | 1% | 14.64 | 11.54 | 11.50 | 45.57 | 11.74 | 11.74 | 36.56 | 11.63 | 1.000 / 1.31 | 1.000 / 1.33 | 0.997 / 0.76 |
| PMC | 8 | 1% | 14.33 | 11.52 | 11.50 | 45.89 | 11.73 | 11.73 | 37.95 | 11.58 | 1.000 / 1.36 | 1.000 / 1.36 | 1.000 / 0.81 |
| PMC | 4 | 5% | 24.08 | 11.69 | 11.49 | 46.00 | 45.98 | 42.57 | 45.98 | 11.79 | 0.003 / 0.00 | 0.093 / 0.07 | 0.003 / 0.00 |
| PMC | 6 | 5% | 24.76 | 11.67 | 11.54 | 46.48 | 46.48 | 11.89 | 46.47 | 11.71 | 0.001 / 0.00 | 1.000 / 1.37 | 0.001 / 0.00 |
| PMC | 8 | 5% | 25.01 | 11.67 | 11.54 | 46.65 | 46.65 | 11.94 | 46.65 | 11.74 | 0.002 / 0.00 | 1.000 / 1.39 | 0.002 / 0.00 |

### Hampel pre-screen and cost

Pre-screen detection / false rate per 1000; mean wall time per series (s).

| model | k | rate | pre-screen | raw s | fm s | fm_hampel s | fm_warm s |
|---|---|---|---|---|---|---|---|
| HMC-IN | – | 0% | – / 8.77 | 0.04 | 0.15 | 0.18 | 0.14 |
| HMC-IN | 4 | 1% | 0.527 / 8.27 | 0.06 | 0.28 | 0.24 | 0.30 |
| HMC-IN | 6 | 1% | 0.908 / 8.35 | 0.06 | 0.26 | 0.18 | 0.28 |
| HMC-IN | 8 | 1% | 0.977 / 8.25 | 0.07 | 0.26 | 0.19 | 0.29 |
| HMC-IN | 4 | 5% | 0.514 / 7.60 | 0.06 | 0.81 | 0.60 | 0.73 |
| HMC-IN | 6 | 5% | 0.833 / 6.86 | 0.26 | 0.50 | 0.27 | 0.36 |
| HMC-IN | 8 | 5% | 0.955 / 6.84 | 0.16 | 0.31 | 0.23 | 0.24 |
| PMC | – | 0% | – / 9.23 | 0.13 | 0.55 | 0.69 | 0.42 |
| PMC | 4 | 1% | 0.725 / 8.89 | 0.86 | 2.01 | 1.24 | 2.30 |
| PMC | 6 | 1% | 0.890 / 8.75 | 0.70 | 1.75 | 1.04 | 2.24 |
| PMC | 8 | 1% | 0.950 / 8.75 | 0.43 | 1.36 | 0.97 | 1.94 |
| PMC | 4 | 5% | 0.684 / 7.32 | 0.33 | 0.59 | 3.92 | 0.48 |
| PMC | 6 | 5% | 0.864 / 7.23 | 0.27 | 0.42 | 3.29 | 0.39 |
| PMC | 8 | 5% | 0.939 / 7.63 | 0.23 | 0.42 | 2.71 | 0.37 |
