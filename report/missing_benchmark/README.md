# Missing-data benchmark: exact marginalisation vs naive fill-ins

Classification and imputation under missing observations with **known** models
(supervised). Each replicate simulates (X, Y) from the model, removes values of
Y with a `pmcprg.missing.patterns` generator, and runs six methods, all with the
true model:

| method | classification | imputation |
|---|---|---|
| `complete` | `classify` on the full Y (lower bound on the error) | none |
| `marginalise` | `classify(model, Y_masked)`: missing values integrated out | `impute`: mean, sd, quantiles 2.5/5/50/95/97.5 %, 200 FFBS draws |
| `plugin` | `impute` posterior mean written into Y, then `classify` | the same mean, Gaussian law N(mean, sd²) with the exact posterior sd |
| `linear` | linear interpolation (nearest value at the edges), then `classify` | the interpolated values |
| `locf` | last observation carried forward (first value backward), then `classify` | the carried values |
| `mean` | mean of the observed values, then `classify` | that mean |

**Models.** `hmc_in_gauss_k2` (exact shortcut), `hmc_dn_gauss_k2`,
`pmc_gauss_k2` (state margins, Gaussian copulas), `pmc_pair_gauss_k2` (pair
margins, Clayton τ = 0.7) and a pair-margin Gamma model with Gumbel copulas
τ = 0.7 (margins `reproduce_csda2013.MARGIN_SETS["pair"]["gamma"]`, prior of
CSDA 2013 Table 1). The last four go through the quadrature grid (G = 64).
**Patterns.** `mcar` in blocks of 1, 10 and 50, `scattered` (one block at a
random start), `aligned` (one block right after the protected first 10 %).
**Grid.** Rates 5, 10, 20, 40, 60 %; N = 2000; 20 replicates: 2500 sequences.

**Scores** (`pmcprg.missing.metrics`, on the missing positions unless stated):

* Error on missing / observed / all positions (`error_rate_split`); mean
  posterior confidence max_k γ_n(k), Brier score and log-loss −log γ_n(x_n).
* `loglik`, what `classify` returns: log p(y_obs) for `marginalise`, log p(y)
  for `complete`, the likelihood of the *filled* series for the fill-ins (not
  comparable).
* RMSE, MAE; CRPS, whose kind is recorded in `crps_kind`: `samples`
  (`crps_from_samples` on the FFBS draws) for `marginalise`, `gaussian`
  (`crps_gaussian`, posterior sd) for `plugin`, `gaussian-sd-obs` for the point
  methods (N(point, s²), s = sd of the observed values; the CRPS of the point
  itself equals the MAE, also recorded).
* Coverage and mean width of 90 % and 95 % intervals: posterior quantiles for
  `marginalise`, point ± 1.645 / 1.96 sd for `plugin` (posterior sd) and the
  point methods (s).
* Wall time: filling or imputation, classification, total.

Seeds are keyed by names: the simulated sequence depends on (model, rep) only,
so every pattern and rate of a replicate masks the same sequence; the results do
not depend on `--jobs` (checked: `--jobs 1` and `--jobs 4` give identical rows
apart from timings).

## Run

From the repository root:

```bash
# Smoke run: 3 reps, rates 0.1 and 0.4, N = 1000, 150 sequences
# → results/missing_benchmark/quick/ (not versioned); 13 s with 4 processes
.venv/bin/python report/missing_benchmark/run_benchmark.py --quick --jobs 4

# Full benchmark: 6 min with 4 processes (Apple arm64 laptop)
.venv/bin/python report/missing_benchmark/run_benchmark.py --jobs 4
.venv/bin/python report/missing_benchmark/summarise.py      # tables + figures, no simulation
```

Options: `--models`, `--patterns`, `--rates`, `--reps`, `--n-obs`, `--draws`,
`--seed`, `--jobs`, `--out`; `summarise.py --results DIR --tables DIR --figures DIR`.

Outputs:

```
report/results/missing_benchmark/   runs.csv (one row per model/pattern/rate/rep/method),
                                    summary.csv (mean and sd over reps), run_info.json
report/tables/missing_benchmark/    headline, err_missing_<model>, imputation_<model>,
                                    coverage90, coverage95, overconfidence, timing (.tex)
report/figures/missing_benchmark/   error_missing_<pattern>, crps_<pattern>,
                                    coverage90_<pattern> (.png, one panel per model)
```

## Findings (full run, 20 replicates, N = 2000)

**The library passes the optimality checks.** The MPM rule on the exact
posterior minimises the expected error, and the exact posterior minimises the
expected log-loss and CRPS; the posterior mean minimises the expected squared
error. On paired replicates, over the 500 comparisons (25 cells × 5 rates × 4
alternatives), another method has a lower error on the missing positions at
z < −2 in 2 cells (by at most 1.5 points), a lower log-loss in 0, a lower RMSE
in 0 of 375. The exact intervals are calibrated: pooled 90 % coverage per model
89.8–90.2 % (single cells 86–92 %), 95 % coverage 94.9–95.2 %.

**Error on the missing positions (%), 20 % | 40 % missing:**

| model | pattern | complete | marginalise | plugin | linear |
|---|---|---|---|---|---|
| HMC-IN Gauss | MCAR b=1 | 6.3 \| 6.4 | 14.4 \| 17.8 | 14.5 \| 17.7 | 14.9 \| 18.2 |
| | MCAR b=10 | 6.4 \| 6.3 | 32.4 \| 36.6 | 32.4 \| 36.6 | 33.4 \| 38.1 |
| | aligned | 6.7 \| 6.4 | 50.4 \| 49.9 | 49.7 \| 49.7 | 51.9 \| 50.5 |
| HMC-DN Gauss | MCAR b=1 | 12.6 \| 12.4 | 20.1 \| 21.2 | 21.3 \| 23.3 | 21.5 \| 23.3 |
| | MCAR b=10 | 11.7 \| 12.6 | 33.1 \| 37.8 | 32.7 \| 38.9 | 33.4 \| 38.6 |
| | aligned | 13.7 \| 12.9 | 52.4 \| 50.1 | 50.9 \| 48.8 | 51.0 \| 49.7 |
| PMC state Gauss | MCAR b=1 | 12.0 \| 12.1 | 19.3 \| 21.7 | 20.9 \| 23.8 | 20.9 \| 23.9 |
| | MCAR b=10 | 11.3 \| 12.0 | 34.6 \| 37.5 | 35.2 \| 38.1 | 36.3 \| 37.8 |
| | aligned | 11.2 \| 12.0 | 50.6 \| 50.3 | 48.3 \| 49.5 | 49.6 \| 50.7 |
| PMC pair Gauss–Clayton | MCAR b=1 | 9.4 \| 9.5 | 16.6 \| 19.8 | 21.4 \| 28.2 | 21.5 \| 28.3 |
| | MCAR b=10 | 8.8 \| 10.0 | 28.8 \| 33.6 | 43.9 \| 50.0 | 44.0 \| 49.8 |
| | aligned | 9.1 \| 9.7 | 43.2 \| 45.7 | 55.6 \| 54.2 | 55.9 \| 54.1 |
| PMC pair Gamma–Gumbel | MCAR b=1 | 17.7 \| 17.2 | 21.8 \| 22.9 | 22.3 \| 25.1 | 22.3 \| 25.2 |
| | MCAR b=10 | 17.8 \| 17.0 | 32.3 \| 31.4 | 37.1 \| 42.2 | 37.3 \| 40.9 |
| | aligned | 17.3 \| 16.9 | 41.9 \| 44.5 | 58.5 \| 55.3 | 51.3 \| 49.3 |

All patterns, methods and standard deviations: `tables/missing_benchmark/`.

* **The gain from marginalising depends on how the transition uses y_n.**
  Averaged over patterns and rates, the extra error of `plugin` / `linear`
  over `marginalise` is −0.0 / +0.7 points for HMC-IN, +0.3 / +0.4 for HMC-DN,
  +0.5 / +1.0 for the state-margin PMC, and +11.5 / +11.4 (Clayton) and
  +8.9 / +6.0 (Gamma–Gumbel) for the pair-margin PMCs (up to +23 points).
  With a Gaussian copula and state margins a filled-in value changes little; with
  pair margins the value of y_n drives p(x_{n+1} | x_n, y_n) directly.
* **Filling a long gap is worse than guessing** for the pair models. In
  `aligned`/`scattered` blocks `marginalise` has 37–46 % error, tending to the
  stationary minority share (45 %) as the block grows. Gauss–Clayton: every
  fill-in has 53–61 %; Gamma–Gumbel: `plugin` and `mean` 53–61 %, `linear` and
  `locf` 43–58 %. A constant or smooth filled run reads as one persistent
  regime and gets a single label, here mostly the minority class (presumably the
  tail dependence of Clayton and Gumbel rewarding such runs).
* **Fill-ins also degrade the observed positions.** Gauss–Clayton, MCAR b=1 at
  40 %: error on observed positions 12.9 % (`marginalise`) vs 18.6 % (`plugin`),
  18.9 % (`linear`), 27.3 % (`mean`); complete data 9.7 %. The observed mean is
  a poor fill for every copula model (HMC-DN MCAR b=1 at 40 %: 54.7 % on the
  missing positions, 18.7 % on the observed ones); it is harmless only for
  HMC-IN, where a value between the two symmetric class means carries almost no
  information.
* **Plug-in classification is overconfident.** Mean confidence vs accuracy on
  the missing positions at 40 %: HMC-IN MCAR b=10, `plugin` 83.6 % vs 63.4 %
  (`marginalise` 63.9 % vs 63.4 %); Gauss–Clayton aligned, `plugin` 99.8 % vs
  45.8 %, log-loss 3.96 against 0.69 (`marginalise` 55.3 % vs 54.3 %).
* **Long blocks are at chance for state-margin models.** Inside a gap the
  posterior decays to the stationary law within a few tens of steps, so every
  method gives about 50 % error in `scattered`/`aligned` blocks for the
  symmetric HMC-IN, HMC-DN and PMC models; those cells separate the methods
  only for the pair models.
* **Imputation.** The posterior mean has the lowest RMSE in every cell. Linear
  interpolation is close for isolated gaps (HMC-DN MCAR b=1 at 20 %: 0.766 vs
  0.760) and far behind in blocks (aligned: 1.785 vs 1.356). The CRPS of the
  exact posterior and of its Gaussian approximation differ by at most 1.2 %
  (Gauss–Clayton MCAR b=1 at 20 %: 0.262 vs 0.263); the Gaussian ±1.645 sd band
  slightly over-covers (pooled 90.5–91.1 %, cells up to 93 %) and ±1.96 sd gives
  95.3–96.0 %. Point imputations with a band of width set by the observed sd
  over-cover isolated gaps (linear up to 100 % for the pair models) and
  under-cover long ones (LOCF down to 68 %).
* **Cost** (median per sequence, N = 2000): `classify` with gaps 8.6 ms (exact
  shortcut) and 25–34 ms (grid), against 9–11 ms without gaps; `impute` with
  five quantiles and 200 draws 60 ms (exact) and 221–294 ms (grid); `plugin`
  22–45 ms.

## Limitations

* Known parameters only: no estimation with gaps. K = 2, scalar observations,
  N = 2000, 20 replicates; the `aligned` mask is the same in every replicate.
* For the grid variants `impute` draws the missing values on the quadrature
  nodes, which biases the sample CRPS upwards by about 0.5–1 % for short gaps
  (checked against a 200-level quantile CRPS with 2000 draws; e.g.
  Gauss–Clayton MCAR b=1 at 20 %: 0.2539 vs 0.2518). The `marginalise` vs
  `plugin` CRPS differences are within that bias: `plugin` is lower at z < −2
  in 3 of 500 cells. The benchmark does not rank these two methods on CRPS.
* The CRPS and coverage of the point methods depend on the chosen spread (sd of
  the observed values), a convention. Their log-likelihoods are those of the
  filled series.
* The error split aligns the labels by the Hungarian permutation on all
  positions; with the true model the permutation is the identity.
* Timings come from four single-threaded workers running side by side.
* The Gamma–Gumbel pair model reads the second parameter of Table 1 as a
  variance (the default of `MARGIN_SETS["pair"]`), whereas
  `pmc_pair_gauss_k2.toml` reads it as a standard deviation: the two pair models
  do not share the same margin spreads.
