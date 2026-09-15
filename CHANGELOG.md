# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

## [1.0.0] - 2026-09-15

First public release, under the new name **awesomePMC** (PyPI `awesomepmc`,
import package `pmcprg`). Identifiers such as "audit K-1", "FR-4" or "RB-6" in
this changelog and in code comments refer to internal review reports that are
not part of the public repository.

### Highlights

- **General pairwise Markov chains**: pair-indexed margins f_ij (A16 Eqs. 12–14)
  alongside state margins, for the five HMC/PMC variants, with 17 copula
  families computed in log space and checked against high-precision references.
- **Unsupervised estimation**: ICE, SEM and GICE with automatic copula- and
  margin-family selection (MLE, AIC, BIC, Huard, Cramér–von Mises, `xvcic`),
  k-means warm start, multistart over parameters or over copula families.
- **Missing observations**: exact classification, imputation (`impute`) and
  forecasting (`forecast`) with NaN; ICE and SEM estimation with gaps;
  missingness patterns and metrics (`pmcprg.missing`).
- **Is a family choice significant?** Analytic standard errors, weighted
  Vuong and Clarke tests with HAC variance, confidence sets of families.
- **Reproducible research**: reproduction of Derrode & Pieczynski (CSDA 2013),
  a missing-data benchmark and a study on real series with gaps (`report/`).
- **Packaging**: Python 3.11–3.14, `pmc` command line, PyQt6 GUI (`[gui]`),
  CI on the lowest declared dependencies, PyPI Trusted Publishing.

### Breaking changes since 0.8.0

- Import package `prg` → `pmcprg`; distribution `copulasformm` → `awesomepmc`;
  log directory `~/.copulasformm` → `~/.awesomepmc`.
- K²-format TOML files (`[[margins]]` keyed by `(i, j)`) now load as pair
  margins on PMC and PMC-IN; set `[model].margin_structure = "state"` for the
  former collapse. K-format files are unchanged.
- statsmodels is no longer a dependency.

(Detailed entries below are unchanged from the development log.)

### Fixed — BB1 multistart crashes; Plackett τ beyond its table; standard errors on a τ bound

- **Multistart no longer crashes on BB1.** Parameter jitter and the family draws
  (`multistart_families` "random"/"sweep") moved τ and δ independently and could
  violate δ < 1/(1 − τ): `CopulaParameterError` in 51/200 seeds at jitter 0.1 and
  143/200 at 0.25 on a BB1 model (τ = 0.5, δ = 1.5), and in every "random" start
  list with a BB1 candidate below τ = 1/3. New hook
  `CopulaVirt.constructible_params`: only a refused pair moves (δ just inside its
  admissible interval, τ kept, no extra RNG draw). All 1 125 previously crashing
  perturbations and 1 148 start lists now build; every start that built before is
  bit-identical (6 915 perturbations, 3 321 starts). The copula-selection
  placeholder (no candidate with a finite score, BB1 first) builds too.
- **Plackett** clamped at |τ| ≈ 0.99352 with a warning while the raw τ was
  kept, like Frank. Families can now declare asymmetric reachable bounds
  (`CopulaVirt.reachable_tau_bounds()`, default ±`reachable_tau_abs`, Frank
  unchanged); Plackett uses its τ-table ends. Same guarantees as Frank: never
  produced or stored beyond the bound, silent reload, a user value beyond it warns
  once; the GUI spin box range rounds inwards. Only Plackett values beyond the
  table end change (514 in the reference comparison); scores, likelihoods and
  trajectories identical; 3 163 spurious clamping warnings gone.
- **`standard_errors` at a τ̂ on the Frank or Plackett bound** used a finite
  difference crossing the bound (clamping warning, one-sided derivative, Frank
  dθ/dτ halved). Such an estimate is now flagged `at_boundary` (NaN 'mle' SEs and
  intervals, as for other boundaries); only the 48 reference results exactly on
  the bound change.

### Added — `return_best_iterate` and a warning on degenerate fits (ICE, SEM)

- **`return_best_iterate`** (config key of `[ice]` and `[sem]`, default `false`,
  results unchanged): return the iterate θ^q with the highest `trace.log_liks[q]`
  instead of the last one. In the real-series study 55 of 216 ICE fits ended
  more than 1 nat below their best iterate (up to 50 nats). `log_liks[q]` is
  computed with θ^q before the M-step; a run reaching `max_iter` evaluates its
  last M-step output once more, so its trace has `max_iter + 1` entries.
  Multistart ranks starts by the log-likelihood of the model each returns. For
  SEM it is a heuristic (best point of a noisy chain). New trace fields
  `best_iter` and `returned_iter`; CLI `pmc estimate --best-iterate` (also
  `estimate-image`); GUI checkbox **Return best iterate**.
- **Degenerate fits**: `degenerate_states(model, Y)` (`pmcprg.pmc._estim_common`)
  flags a state with stationary weight below 0.5 %, a margin (state or pair)
  with a standard deviation below 1 % of that of the observed data (IQR when the
  sd is undefined), and a copula τ within 1e-3 of ±1. ICE and SEM log one
  WARNING on the returned model and store the findings in `trace.degenerate`;
  `pmc estimate` prints a `Degenerate` summary line. On the 266 saved
  real-series fits the rule flags exactly the 15 degenerate ones and none of the
  251 others. No numerical result changes.

### Fixed — Frank τ beyond its reachable range; misleading multistart warnings (found by the real-series study)

- **Frank** only reaches |τ| = 0.994299 (θ capped at 700), but its registered
  range is ±(1 − ε): fitted models could store a τ they did not use and warned
  "clamping" on every reload (57 of 266 real-series fits). A family can now
  declare `reachable_tau_abs`; `CopulaEnum.constructible_tau_range()` is cut to it
  and `CopulaEnum.reachable_tau(τ)` caps values beyond it. Fits (`tau`, `mle`),
  the ICE M-step, Huard evidence, multiple-imputation averaging, the independence
  LR test and the GUI spin box never produce or store a Frank τ beyond the bound;
  a saved model reloads silently; a user-supplied τ beyond it still warns once.
  `TAU_MIN_MAX` is unchanged. Other families are bit-identical; for Frank only
  values that exceeded the bound change (203 in the reference comparison; scores,
  likelihoods and complete-data ICE/SEM trajectories identical). With ICE
  `missing_strategy="impute"`, the average of the capped per-series τ̂ moves
  accordingly.
- **Multistart jitter** logged `correct_tau` warnings ("τ=1.6615 outside valid
  range … clipped") for values the user never gave; they now go to DEBUG
  (`correct_tau(..., warn=False)`). Perturbed models are bit-identical.

### Added — ICE and SEM estimation with missing observations (wave 2 of the missing-data plan)

- **`ice` and `sem` accept `NaN` rows** and estimate from the observed data;
  `trace.log_liks` is the observed-data log-likelihood log p(y_obs). Complete
  data give bit-identical results to the previous code (90 golden cases: every
  fixture × 2 seeds × ICE, SEM, family multistart random/sweep, k-means, GICE;
  `test_estim_complete_data_identity.py`).
- **SEM**: data augmentation — each iteration draws the states and the missing
  values jointly (FFBS on the gap grid), fills them in and runs the unchanged
  M-step.
- **ICE, config key `missing_strategy`**:
  - `"available"` (default): exact posteriors given the observed data; the prior
    uses every position, margins the observed values (pair margins: their
    observed endpoints), copulas the pairs with both ends observed.
    Deterministic.
  - `"impute"`: `missing_draws` (default 5) completed series per iteration
    (seed `missing_seed`, offset by the start index in a multistart), complete-data
    ICE step on each, parameters averaged on their natural scale (τ, loc, scale,
    mean vector, covariance), prior from the exact posterior, each family chosen
    by the criterion summed over the completions.
  - `gap_nodes` (default 64) sets the quadrature grid of both. The new keys are
    validated and merged by the config parsers; the GUI has no widget for them
    yet and runs with the defaults.
- **k-means warm start** clusters the observed rows only; GICE and multistart
  over families work unchanged.
- **Measured choice of the default** (N = 3000, MCAR blocks of 10, start near
  the truth, 30 replications): RMSE of prior / margin means / τ on the state
  Gaussian PMC at 40 % missing `.027/.099/.040` (available), `.044/.140/.045`
  (impute), `.038/.131/.054` (SEM), against `.030/.089/.034` on complete data.
  `"available"` matches or beats `"impute"` almost everywhere, is 3–4× faster
  (8–19× with GICE) and its log-likelihood never decreased on the state-margin
  models. `"impute"` is better run to convergence (prior and margin-mean RMSE
  up to 25 % lower) and keeps GICE margin-family selection at its complete-data
  level (57 % correct vs 39 %; complete data 61 %). Exp. 3 of CSDA 2013 with
  a sweep multistart recovers the diagonal families in every run at 20 % and
  40 % missing. Drawing missing values on the grid nodes adds no detectable
  bias to τ; spreading them within the cell does (−0.002 to −0.003), so node
  draws are kept.
- **`pmc estimate`** accepts missing values and prints one warning line with
  their count (it exited with status 2 before).
- `gaps.refuse_missing` is removed (unreleased helper, no longer used).
- 74 new tests (mechanics of each missing-data M-step against direct
  calculations, trace = observed-data log-likelihood, gaps at both ends, d = 3,
  parallel = sequential multistart, GUI estimation entry point; parameter
  recovery at 10/20/40 % with a tolerance scaled from the complete-data RMSE;
  ICE trace non-decreasing on state-margin models, SEM stationary after burn-in).
- Limits: only block MCAR was studied; a multivariate row with a missing
  component is treated as missing as a whole; GICE can raise
  `IncompatibleObservationError` when a fitted location moves above an observed
  value (1 of 78 runs in the family study).

### Added — real series with gaps (`report/real_series/`)

- Unsupervised estimation, imputation, classification and forecasting on tsNH4
  (real gaps with ground truth), Beijing PM2.5 (Huairou, Aotizhongxin; masked
  cross-validation) and PAMAP2 subjects 102/108/105 (activity labels, real
  state-dependent sensor dropouts); 266 fits, no crash. Data are read from a
  folder outside the repository (`--data`); `--quick` smoke run about 15 s.
- Findings: ICE `"available"` is the strategy to use (the others differ by local
  optima, not systematically); copulas gain thousands of nats over HMC-IN but
  imputation only ties an AR(1) Kalman smoother in CRPS on tsNH4 (well calibrated,
  90 % intervals cover 92 %) and beats it at Huairou in 30/30 masked cells; on
  PAMAP2 exact marginalisation beats fill-ins on missing windows (5.8–6.5 % vs
  8.0–13.6 % error with the supervised model) but likelihood-best unsupervised
  states are not the activity groups (27.7–43.7 % at K = 3).
- Failure cases documented: unbounded likelihood on discretised data (a state
  collapsing on the ln PM2.5 floor), degenerate states in 13/64 tsNH4 fits, ICE
  returning its last rather than its best iterate (up to 50 nats below).

### Added — missing-data benchmark (`report/missing_benchmark/`)

- Classification and imputation with gaps under known models: exact
  marginalisation against `plugin` (posterior-mean fill-in), `linear`, `LOCF`
  and `mean` fill-ins; 5 models × 5 patterns × 5 rates × 20 replicates,
  N = 2000 (2500 sequences, 6 min with 4 processes); error on missing/observed
  positions, Brier, log-loss, RMSE, MAE, CRPS, 90/95 % coverage, wall time.
- Result: over 500 paired comparisons another method has a lower error on the
  missing positions at z < −2 in 2 cells (by at most 1.5 points), a lower
  log-loss or RMSE in none; exact intervals are calibrated (pooled 90 %
  coverage 89.8–90.2 %). Filling in costs +0.3 to +1 point with state margins
  and +6 to +11.5 points on average (up to +23) with pair margins.

### Changed — packaging, CI and release for 1.0

- `pyproject.toml`: description and keywords, `Development Status :: 4 - Beta`,
  Python 3.11–3.14, GitHub URLs, SPDX `license = "MIT"` (PEP 639), anchored
  sdist includes; **statsmodels is no longer a dependency** (nothing imports it).
- CI: Python 3.14 in the matrix, a `min-versions` job on the lowest declared
  dependencies (Python 3.11), the smoke job builds and installs the wheel.
- `publish.yml`: TestPyPI (manual) and PyPI (GitHub Release) through Trusted
  Publishing, with a tag/version guard; `RELEASING.md`; `CITATION.cff` keywords
  and public URLs.
- README reworked for PyPI (badges, "Why awesomePMC?", absolute links).
- `pmc --help` usage lines say `pmc` for the console script (they always said
  `python -m pmcprg.pmc`, which is kept when invoked that way).
- `scripts/publish_github.py` exports `main` without the private files to a
  local `public` branch for the public GitHub repository (never pushes).
- `scripts/update_readme_structure.sh` removed (README no longer has the
  markers it edited).

### Changed — one implementation of the margin CDFs used by diagnostics and the GUI

- `pmcprg.diagnostics.pseudo` now holds `margin_cdfs` (options `clip` — float,
  `(lo, hi)` or `None` — and `broadcast`), `is_pair` and the state-law helpers
  (`StateMixture`, `mixture_label`, `state_weights`, `state_law`,
  `state_law_name`); `pmcprg.pmc.gui._margins` re-exports them and
  `gui.views.compute_marginal_cdfs` calls them (K CDF evaluations instead of K²
  on state models). 751 recorded outputs (diagnostics, GoF, Vuong comparisons,
  GUI plot data, report probes) are bit-identical before and after.

### Tests — cached high-precision copula references (palier-1 deferred item)

- `test_copula_limits.py` reads its Decimal references (68 family/θ/δ entries ×
  36 (u, v) points: C, log c, h) from `pmcprg/tests/data/copula_limits_references.json`
  instead of recomputing them at up to 1920 digits: 267 s → about 5 s, assertions
  and the 407 original outcomes unchanged, cached arrays bit-identical to the
  computed ones (438 arrays). Values are stored as the shortest float64 `repr`.
- Drift checks: the file must cover exactly the cases built by the tests
  (missing cases fall back to Decimal with a warning); a seeded sample of 19 rows
  is recomputed exactly (`slow`, about 4 s); `PMC_REGEN_COPULA_REFS=1` enables the
  full regeneration test (about 140 s). `scripts/regen_copula_references.py
  [--check] [-j N]` rebuilds or checks the file in parallel.

### Fixed — warnings and tests

- `error_rate` warned whenever the label *values* of the reference and of the
  estimate differed (e.g. a {0, 255} mask image against {0, 1}), although it is
  invariant under relabelling; it now warns only when the numbers of classes
  differ.
- `_stationary_distribution` reported ‖Δ‖∞ = 0 when power iteration did not
  converge (periodic or very slowly mixing chains); it now solves π(A − I) = 0,
  Σπ = 1 directly in that case and warns only if that fails. Converged cases are
  unchanged.
- Tests pass at the declared lower bounds (`scipy.integrate.trapezoid`
  instead of `np.trapezoid`, scikit-learn tests skipped without it, Student
  inverse-h tolerance on scipy < 1.17) and on scipy 1.18 (`kstest` with a frozen
  CDF); a GICE test no longer depends on platform-specific rounding.

### Changed — project renamed `awesomePMC`, import package `pmcprg`

- **The project is now `awesomePMC`** (PyPI distribution `awesomepmc`) and the
  import package `prg` is now **`pmcprg`**: `from pmcprg.pmc import ice`,
  `python -m pmcprg.pmc`, `pip install 'awesomepmc[ml]'`. The command-line
  entry point is still `pmc`. `prg` clashed with other packages installing a
  top-level `prg` (e.g. awesomePKF) in the same environment.
- The per-user log directory moves from `~/.copulasformm` to `~/.awesomepmc`;
  the GUI settings are stored under the organisation `awesomePMC` (previous
  settings are not migrated).
- Repository URLs: `gitlab.ec-lyon.fr/sderrode/awesomePMC` and the GitHub mirror
  `SDerrode/awesomePMC` (the old URLs redirect).
- File paths `prg/…` quoted in earlier CHANGELOG entries and in the `AUDIT*.md`
  reports refer to the pre-rename layout (`pmcprg/…` now).

### Added — missing observations: exact inference, imputation, forecasting (P1, P2 of the missing-data plan)

- **`classify`, `forward`, `backward`, `sample_posterior` accept `NaN`** and
  integrate missing observations out. Exact shortcut (density 1) for HMC-IN,
  HMC-IN2 and PMC-IN with state margins; otherwise an augmented forward–backward
  on (state, value) over a Gauss–Legendre grid in the quantiles of the stationary
  law of y (`gap_nodes`, default 64, endpoint transform s²/(s²+(1−s)²),
  per-destination renormalisation, Nyström interpolation for quantiles).
  New `prg.pmc.gaps`: `gap_posterior` (γ, ξ, node posteriors), `impute` (mean,
  sd, quantiles, density, FFBS draws of states and values) and `forecast`
  (state probabilities and predictive law of y_{N+k}). Complete data go through
  the previous code, bit-identical.
- **Accuracy against exact references** (worst case, G = 32/64/128):
  Gaussian AR(1) ρ = 0.5 log-likelihood 4e-6/6e-8/6e-9, imputation mean/sd
  6e-6/5e-7/2e-8; distinct Gaussian regimes (brute force over paths)
  7e-4/6e-8/1e-9; Clayton τ = 0.7 pair model, isolated gap against `quad`
  2e-2/1.4e-4/1.6e-9 (lower-tail worst case); forecasts against 20 000
  simulated trajectories. 146 new tests. Cost with gaps: `classify` 1.6× on
  N = 10 000 with 10 % missing in gaps of 10.
- **`prg.missing`**: missingness patterns with ImputeGAP `GenGap` geometry
  (Khayati et al., PVLDB 13(5), 2020) — `mcar`, `aligned`, `scattered`,
  `blackout`, `disjoint`, `overlap`, `gaussian`, `distribution` — and metrics
  on masked positions — `rmse`, `mae`, `mutual_information`, `pearson`,
  `crps_from_samples`, `crps_gaussian`, `interval_coverage`, `error_rate_split`.
  124 new tests.
- **Loaders**: the GUI and CLI CSV readers accept empty, `NaN` and `NA` cells as
  missing (warning with count and share); the GUI classification view shades
  the gaps.
- Estimation with missing data: see the next section (the first version of
  this wave refused NaN in `ice`/`sem`).
- Limits: a multivariate row with a missing component is treated as missing as
  a whole; the grid is univariate (pair-margin PMC-IN with d > 1 raises);
  imputation draws for grid variants lie on the nodes.

### Added — multistart over copula families (ICE, SEM)

- **New config key `multistart_families`** (`[ice]`/`[sem]` TOML, cfg dict, GUI
  **Start families** combobox; default `"none"`). ICE on the CSDA-2013 Exp. 3
  design has fixed points that differ by the copula family of the diagonal
  pairs, and the parameter-jitter multistart keeps the families of the initial
  model, so it could not leave a wrong-family basin.
  - `"none"`: parameter jitter only — outputs bit-identical to the previous
    driver (24 ICE/SEM runs on the state PMC, pair PMC, HMC-DN and HMC-IN
    fixtures, `n_starts` 1 and 3, `multistart_workers` 1 and 2: log-likelihood,
    τ/family/prior/margin histories, fitted model and losing traces equal).
  - `"random"`: every extra start also redraws the family of every pair (i, j)
    uniformly among `candidates`, τ uniform on the central 60 % of the family's
    registered range, clipped to its constructible range; family draws on their
    own RNG stream of `multistart_seed` (prior and margins jittered as under
    `"none"`).
  - `"sweep"`: after the model itself, every combination of candidate families
    on the diagonal pairs (i, i) at τ = 0.5 (clipped), in `itertools.product`
    order, off-diagonal pairs unchanged, no jitter. Needs `n_starts = 1 + |C|^K`;
    fewer drops the last combinations (WARNING), more adds `"random"` starts;
    more than 256 combinations is refused (use `"random"`).
  - Variants without copulas or an empty candidate list fall back to `"none"`
    (INFO). The winning start is `trace.run_tag` (`family-sweep:Gauss/Clayton`,
    `family-random-4`); `multistart_workers > 1` gives the same result.
  - Measured on Exp. 3 setting 1 (Gauss/GH/Clayton candidates, pair Gaussian
    margins, p = [[0.5, 0.05], [0.05, 0.4]], N = 2500, known margins, MLE
    criterion, start on Gauss τ = 0, seeds 3000–3009): a single start ends on
    (GH, Clayton) on the diagonal 10/10 times; `"sweep"` (`n_starts = 10`) on
    the true (Gauss, Clayton) 10/10, final log-likelihood +30 nat (median;
    11–41), unsupervised error 13.9 % against 17.5 % single-start and 14.0 %
    supervised; `"random"` (`n_starts = 10`) 10/10 as well, +24 nat, 14.7 %.
- `n_starts` in the GUI now goes up to 257 (a full sweep).
- `prg.pmc._estim_common`: `build_multistart_inits`, `check_multistart_families`,
  `MULTISTART_FAMILY_MODES`, `SWEEP_TAU`, `RANDOM_TAU_BAND`,
  `MAX_SWEEP_COMBINATIONS`. New `prg/tests/test_multistart_families.py`.

### Fixed — ICE k-means warm start

- **Clusters are renumbered to match the declared state laws.** K-means numbers
  its clusters arbitrarily and the warm start used that numbering as state
  labels: with known margins (`fit_margins = False`) about one start in two
  estimated the prior and the copulas on labels permuted with respect to the
  margins. On the CSDA-2013 Exp. 3 design (Gaussian margins, 10 seeds) the true
  Clayton copula of pair (1, 1) was recovered 5/10 times from the k-means start,
  10/10 after the fix. Assignment by maximum Σ log g_k(y_n) (Hungarian
  algorithm), g_k = f_k or the mixture Σ_j (p_kj/p_k) f_kj for pair margins.

### Changed — CSDA-2013 reproduction report rewritten

- `report/csda2013_reproduction.tex` now reproduces the paper with the general
  PMC (pair margins), reads Table 1's Gaussian second parameter as a variance
  (the only reading consistent with its Gamma parameters), prints Tables 2–5 in
  the paper's orientation (rows = tested copula, columns = true copula) and
  runs Experiment 3 with a multistart ICE. Tables 2(a), 2(b), 3(b), 4(a) are
  matched to within 0.3–0.5 point on average (were 7–12 points off with one
  margin per state); the multistart recovers the diagonal copulas of Tables 6–7
  in 97–100 % of runs with an unsupervised error equal to the supervised one.
  Open gaps are documented (Table 3(a), PMM Tables 4(b) and 5(b)).
- `reproduce_csda2013.py`: `--exp3-starts multi` (independence, k-means and a
  sweep of the diagonal copula families; best final log-likelihood) and the
  paper's orientation in the LaTeX grid tables (CSV files unchanged).
- New `report/csda2013_tables.py` rebuilds every report table from the
  committed CSVs, with the paper's Tables 2–5 in `report/data/`; the report's
  `Makefile` builds the PDF without re-running simulations by default.

### Added — palier 2 of AUDIT_COPULES, first half (FR-4, FR-6)

- **Standard errors of copula parameters, outside ICE (FR-4).**
  `FitResult.standard_errors()` and `copula.standard_errors(uv, weights,
  method='mle'|'tau')` return a `StandardErrors` object (estimate, se, cov over
  τ and native parameters, `ci()`, `at_boundary`). `'mle'`: pseudo-likelihood
  sandwich of Genest, Ghoudi & Rivest (1995) with the estimated-rank terms and
  frequency weights; `'tau'`: 16·Var{2Cₙ − û − v̂}/n (Genest & Favre 2007) mapped
  to θ. Parameters at a boundary (independence, δ = 1, ν at a bound) are
  flagged and get no Wald interval; `independence_lr_test` uses ½χ²₀ + ½χ²₁
  where independence is on the boundary (Self & Liang 1987). Monte-Carlo
  SE/SD ratios 0.93–1.10 and 95 % coverage 93.5–96 % on Gaussian, Clayton,
  Gumbel, Frank, Plackett, BB1 and Student. Opt-in: 71 existing outputs
  byte-identical. Not yet: SEs inside ICE (Oakes, Godambe), serial dependence
  (FR-5).
- **Is a family choice significant? (FR-6).** `prg.diagnostics.model_selection`:
  weighted Vuong and Clarke tests with Newey–West HAC variance (consecutive
  pairs share an observation), decisions A / B / tie, comparison matrix,
  confidence set of families not significantly worse than the best;
  `FitBestResults.compare()` / `.confidence_set()` and `ice_pair_comparisons()`
  after an ICE run (same pseudo-observations, estimator and scores as the
  M-step). Measured: Vuong size 5.1 % with known margins but 9–11 % on rank
  pseudo-observations (the margin-estimation term of Chen & Fan 2006 is not
  included); **Clarke's sign test rejects 66–72 % at 5 % once parameters are
  fitted — do not use it to separate close families**; without HAC the size is
  21 % on a hidden-regime chain, 6.7 % with it. On Exp. 3 (exp2, p_orig) the
  sparse CubSec blocks are statistical ties in 75–100 % of runs and the true
  family is always in the confidence set (`report/family_tie_study.py`).
  Selection decisions unchanged (bit-identical against 05c2878).

### Changed

- **ICE weighted Kendall τ in O(N log N) time and O(N) memory** (weighted
  Knight merge sort) instead of N×N arrays that took 95–98 % of an ICE run:
  one ICE run ×17 faster at N = 2500, ×33 at N = 6000. Agreement with the old
  implementation ≤ 1e-12 (fuzzed: 1.9e-14). Exp. 3 replay (96 runs): 0 of 384
  family decisions change, max |Δτ̂| 4.9e-7 (Student start value), recorded
  `exp3_*__runs.csv` reproduced at their 4 decimals.
- **`reproduce_csda2013.py --table1-sigma sd|variance`.** Table 1's Gaussian
  margins N(μ, s) read with s a variance by default in `--margins pair` — the
  only reading consistent with the printed Gamma parameters; `--margins state`
  keeps the standard deviation and every recorded output.

### Changed — general PMC with pair-indexed margins f_ij (A16 Eqs. 12–14)

The package's "PMC" had one margin per state (f_ij = f_i), attributed to
reversibility. By the Proposition of A16 §2.1 that is exactly the case where X
is Markov: it was a stationary reversible HMC-DN, not the general PMC of the
paper. On the CSDA-2013 settings this was the main cause of the gap with the
published error rates (smoke run of `report/reproduce_csda2013.py --margins
pair`, 10 replicates: Tables 2 and 4 within about a point of the paper;
Table 3(a) and 5(b) gaps remain).

- **Breaking for K²-format files.** `[[margins]]` blocks keyed by `(i, j)` are
  now kept as pair margins on PMC and PMC-IN, and estimated separately.
  Files that relied on the collapse must set `[model].margin_structure =
  "state"`. HMC-* variants refuse pair margins. K-format files are unchanged.
- **Model, simulation, inference.** `PMCModel.margin_structure` ("state" |
  "pair"), `margin(i, j)` (`margin(i)` raises on pair models),
  `margin_blocks()` with K² blocks, simulation by Eqs. 13–14, forward-backward
  and log-space passes with Σ_k p_ik f_ik(y_n). State models are
  bit-identical (simulate, W, forward, backward, ξ, `sample_posterior`,
  `weight()`: 240 arrays on 8 fixtures × 2 seeds).
- **Estimation.** ICE/SEM fit f_ij by weighted MLE on both views of the pair
  density — y_n with weight ½ξ_n(i,j), y_{n+1} with weight ½ξ_n(j,i) — an
  ICE-style estimator, not an exact EM step; copula pseudo-observations
  (F_ij(y_n), F_ji(y_{n+1})) with weight ξ_n(i,j); pairs with no weight keep
  their margin. GICE per-pair selection, k-means init and multistart keep K²
  blocks. State models bit-identical (92 ICE/SEM/init/multistart/GICE runs).
- **GUI and CLI.** Margins tab with a K×K grid and a state/pair selector;
  diagnostics on pair models (copula GoF on F_ij/F_ji, margin adequacy
  against the state mixture g_i = Σ_j (p_ij/p_i) f_ij); pair-aware export;
  one-line CLI error instead of a traceback when estimation fails. Unticking
  every GICE candidate now removes `candidates` from the block.
- **Diagnostics and research scripts.** `prg/diagnostics/pseudo.py`
  (pair pseudo-observations, dual-view margin PIT), pair-aware PMM, the six
  research scripts off `margin(k)`, `reproduce_csda2013.py --margins
  pair|state` (default `state`, recorded results unchanged; pair outputs in
  `results/margins_pair/`, `tables/margins_pair/`).
- **Tests.** `test_general_pmc_core.py`, `_api.py`, `_estimation.py`,
  `_mstep.py`, `_diagnostics.py`, `test_gui_pair_margins.py`. Two contract
  tolerances were revised with measurements: ICE recovery 0.2 → 0.25 (ICE
  started at the truth converges 0.202 away on that sample) and the GICE start
  moved off the symmetric fixed point (identical blocks never separate).

### Fixed (palier 1 of AUDIT_COPULES — numerical robustness of copulas, RB-1..RB-10)

Acceptance tests first (`prg/tests/test_copula_limits.py`, 407 tests against
high-precision decimal references from the published CDFs, down to u = 1e-12,
1 − 1e-12 and τ = 1e-12): 118 failed on the previous code, all pass now.

- **Clayton near independence (RB-1)** cancelled catastrophically: on
  independent data `fit('tau')` gave LL = +2.86 instead of 0 and survival
  Clayton won `fit_best`. Log-space kernel with no cancellation as θ → 0 and
  no u^{−θ}; native scalar paths (statsmodels backend dropped).
- **Joe τ(θ = 1) was 5e-7 (RB-2)**, so `CopulaJoe(tau_k=1e-7)` raised and
  `fit_best` dropped Joe silently. Exact τ(θ) (Hurwitz-ζ series at θ = 1,
  series at θ = 2, digamma beyond); θ(τ) invertible on the whole domain.
- **Two-parameter ICE fits stopped at the start value (RB-3, RB-8)**: Student
  ν = 4.0 for a true 2.5, BB1 δ = 1.5 up to 47 nat below the optimum. One
  joint-MLE engine for ICE and `fit('mle')` on the unnormalised likelihood,
  Student in (atanh τ, 1/ν), BB1 in (log θ, log δ) without δ projection;
  failures logged at WARNING and flagged in the returned parameters.
- **Student log-density was floored at −36 nat (RB-4)**: native log-density
  (−79.6 nat at ν = 30, τ = 0.95), tail-accurate h, closed-form inverse h;
  ν bounds from one constant, lower bound 2.001.
- **Survival copulas, A12, BB1, Frank, Plackett, AMH, FGM, cubic section**:
  cancellation-free kernels (complements carried exactly, `expm1`/`log1p`
  forms, series near independence); BB1 no longer NaN at τ ≥ 0.999; AMH τ
  series (the old small-θ slope was 2/3 instead of 2/9).
- **Gaussian CDF** was scipy's randomized QMC (0 instead of 7.4e-20 at
  u = v = 1e-14, different results per call): deterministic
  Drezner–Wesolowsky/Genz integral, relative error ≤ 2e-10 for |ρ| ≤ 0.99,
  11× faster.
- **Singular τ (RB-6)**: |τ| = 1 raises `CopulaParameterError` before any
  family code runs (no more ZeroDivisionError / LinAlgError); `update_tau_k`
  refreshes θ (K-9); `fit('tau')` clips into the constructible range.
- **No silent fallbacks (RB-7, RB-9)**: zero-weight points never poison an
  objective, NaN penalised like exceptions; no EPS / 0.5 / w / 0-1 sentinels
  in any family; `pdf_array` = `exp(logpdf_array)` unfloored. Inference
  recomputes a pass in log space when every weight of a step underflows (the
  old floor over-stated the log-likelihood, −2990 vs −146204, and zeroed the
  posteriors of the whole sequence); an observation impossible under every
  state even in log space raises `IncompatibleObservationError`.
- **`fit_best` (RB-10)** ranks only fits made with the requested method and
  reports failures; a clamped Plackett stores its realised τ.
- **Speed**: GH/Joe `inv_h_array` 100× faster (vectorised Newton), Gaussian
  `cdf_array` 11× faster, Student log-density 1.7× faster.
- **Impact on recorded results, measured with the same seeds**: CSDA Tables
  2–5 bit-identical (168/168); Experiment 3 unchanged for the Gauss/GH/Clayton
  configurations; 3 of 384 decisions change in the configurations where
  Student is a candidate (|Δτ̂| ≤ 0.01).
- **Behaviour changes**: Joe θ shifts by ~6e-7; seeded Student samples shift
  at ~1e-8; `CopulaJoe`/`CopulaAMH` refuse out-of-range τ instead of
  returning an endpoint; `fit_best` returns a list subclass with
  `.other_method` and `.failures`.

### Added (citation metadata and references — audit B-6, B-8, K-16)

- **`CITATION.cff`** (CFF 1.2.0): version and release date from the git tag,
  MIT, both repositories, Derrode & Pieczynski (2013) as `preferred-citation`
  and (2016) under `references`, with a message asking for both to be cited.
  GitHub's *Cite this repository* button and Zenodo read it.
- **`REFERENCES.md`**: every reference the code, the reports and the note
  rely on — 60-odd entries, each with a DOI resolved against Crossref on
  2026-09-13 (or confirmed to have none) and the module that uses it.
- **`Reference:` paragraphs in the docstrings** of the eleven copula modules
  that had none (`gaussian.py` gains a module docstring), of the Neyman
  smooth test, the copula fit (`method='tau'` = Genest & Rivest 1993,
  `'mle'` = Genest, Ghoudi & Rivest 1995), ICE (Pieczynski 1992; Delignon et
  al. 1997; Giordana & Pieczynski 1997), the forward–backward pass (Devijver
  1985; Rabiner 1989), FFBS (Carter & Kohn 1994; Frühwirth-Schnatter 1994;
  Chib 1996), SEM (Celeux & Diebolt 1985, in full), the Hilbert–Peano scan
  (Skarbek 1992; Giordana & Pieczynski 1997; Derrode & Pieczynski 2004),
  conditional-inverse sampling (Rosenblatt 1952; Nelsen 2006 ch. 2 (random variate generation)) and the
  statistics of the report scripts (Anderson–Darling, Székely–Rizzo,
  Gretton et al., Wilson).

### Fixed (documentation — audit B-3, B-4, B-5, K-10, K-11, K-13, K-14)

- `AUDIT_RECHERCHE.md`: four wrong author lists (M7, M12/M13, E3, C22), one
  off-topic supporting example removed (M17), the UCI dataset DOI no longer
  presented as the DOI of the ESANN 2013 paper (also in the README), 20
  missing DOIs of published versions added and volumes/pages completed; no
  [NON VÉRIFIÉ] marker remains.
- Docstring formulas: the Plackett module labelled Spearman's ρ_S as
  Kendall's τ (τ has no closed form and is computed by quadrature, as the
  code already did); BB1's "δ → ∞ → Gumbel" limit (it is θ → 0, and
  λ_L → 1 as δ → ∞); the cubic-section copula now states its formula,
  τ = 2θ/3 − 2θ²/75 and its source; "Caillault & Guégan (2009)" is 2005;
  `xvcic` is described as an exact purged K-fold cross-validated
  log-likelihood, not the analytic xv-CIC of Grønneberg & Hjort (2014).
- The GUI's Kendall-process statistic is documented: the Lebesgue-weighted
  form of Wang & Wells (2000) rather than the dK_θ form of Genest, Quessy &
  Rémillard (2006), W_i including j = i, and the Monte-Carlo error of the
  K_θ reference (≤ 0.005 per node, common to observed and replicates).
- `inference.py` no longer calls its backward scaling "Devijver's": α̂ is
  normalised per Devijver (1985), β̂ is rescaled by its own sum (Rabiner
  1989 §V.A) and γ/ξ are renormalised downstream — same result.
- The provenance of every selection criterion is stated: only `huard` is
  from A16 (Eq. 20); `mle` is A23's PLM rule (Eq. 11), `kolmogorov` A23
  Eq. (10); AIC/BIC/CvM/xvcic/huard_common/huard_global and the margin
  rules mle/aic/bic are package additions.

### Fixed (audit K-6, K-7 — Huard evidence quadrature; bootstrap p-values)

- **The Huard evidence was integrated on a fixed 50-point τ grid.** The
  likelihood peak narrows like 1/√n_eff while the grid does not, and
  families with a (−1, 1) support get a grid twice as coarse as the
  positive-only ones: at n_eff = 1500 the Gauss evidence was off by
  −0.28…+0.21 nat depending on where τ̂ fell between two nodes — a
  family-dependent artefact of the size of the prior-width effects the
  `huard_*` variants are meant to isolate (no selection flipped in 40
  datasets, but the note compares criteria at 0.69 nat). The grid is now
  refined around the maximum until it resolves the peak and integrated by
  the trapezoid rule, normalised by the prior width: ≤ 1e-4 nat from
  `scipy.integrate.quad` at n_eff = 1500 and ≤ 0.003 at 20 000. Every Huard
  score moves by +0.02 nat uniformly (the old log-mean over 50 nodes carried
  log(49/50)); rankings are unaffected by that constant. Replayed on 6 ICE
  runs of Experiment 3 with the `huard` criterion (and 40 datasets in the
  audit), no selection changed; the 400 `huard` runs of Experiment 3 and the
  Huard-variant runs of `prior_width_isolation.py`, on which
  `note/penalty_scale.tex` rests, are the only recorded results worth
  re-running to confirm the hit counts.
- **Bootstrap p-values could be exactly 0.** `parametric_bootstrap` and
  `FitResult.gof_test` reported the plain proportion #{T_b ≥ T_obs}/B; both
  now use `(1 + #{T_b ≥ T_obs})/(B + 1)` (Davison & Hinkley 1997, ch. 4;
  Phipson & Smyth 2010), never 0, smallest 1/(B + 1). The difference is at
  most 0.005 at B = 200, in the conservative direction; the level
  measurements on record were made with the plain proportion.

### Changed (documentation — audit K-4, K-5, B-1, B-2)

- `ice()` is documented as the ξ-weighted variant of ICE: A16 §4.2
  (Eqs. 22–24) and A23 §3 (b)–(c) estimate copulas and margins on **one**
  posterior draw (L = 1) and reserve the conditional expectation for the
  prior; `sem()` is the hard-draw estimator. Said in `ice.py`, the
  feature ↔ paper map, the README's ICE-vs-SEM box and §Exp. 3 of the
  reproduction report, which no longer claims a faithful reproduction of the
  estimator.
- `n_eff` is Σw everywhere it is named: two GUI docstrings claimed Kish's
  (Σw)²/Σw² and the pair-scatter panel computed it under the same label.
- `note/penalty_scale.bib`: `ko2019` had the DOI of a different paper;
  `shemyakin2016` did not exist under the cited title/co-author and is
  removed (the citing sentence is now the note's own argument, with a
  TODO(author) naming the two real candidate references).

### Fixed (copula log-densities were wrong in strong dependence — audit K-1, K-2, K-3)

- **Joe, A14, Gumbel–Hougaard: the vectorised paths that ICE, the Huard
  evidence and the GoF reference consume floored an intermediate at `EPS`
  that is legitimately far below it.** Joe's `W = ū^θ + v̄^θ − ū^θ v̄^θ` is
  10⁻⁴⁰ on a sixth of the unit square at τ = 0.95 — where Joe puts its mass
  — so `cdf_array` was flattened at 0.606 and `logpdf_array` was off by 100
  to 290 nat there; A14's `U_i = (u^{−1/θ} − 1)^θ` is 7·10⁻²⁹ at the *centre*
  of the square, so its `logpdf_array` was off by 0.77 nat everywhere and by
  ~100 nat in the corners; GH inherited the generic `log(max(pdf, EPS))` and
  was capped at −36 nat where the true value runs to −130. All three are
  now evaluated in log space (`log W`, `log U_i`, `log A` via `logaddexp`),
  from one kernel shared by the scalar and vectorised paths.
- **Frank collapsed beyond |τ| ≈ 0.85** — the textbook `e^{−θ} − 1` and
  `(e^{−θu} − 1)(e^{−θv} − 1)` cancel catastrophically for θ ≳ 25: the
  statsmodels-backed scalar `pdf` returned 2·10⁻¹⁶ at the centre of the
  square for τ = 0.9 (true value 9.57), `cdf(0.98, 0.99)` returned 0 (true
  0.975), and `inv_h`, which `simulate()` relies on, was off by 10⁻² at
  τ = 0.9 and degenerate at 0.95 — while the registry advertised |τ| ≤ 0.994.
  Rewritten natively in log space for θ > 0 and by the 90° rotation for
  θ < 0; verified against the former backend to 10⁻¹³ where it was still
  accurate, and at θ = 38, 200, 700 against finite differences of C, the
  inverse round trip and ∫c = 1. The advertised range is now held.
- **A14 `tail_dependence()` returned λ_L = 0.65 / 0.77 / 0.94 at
  τ = 0.9 / 0.95 / 0.99** (numeric limit at u = 10⁻⁴, converging like
  u^{1/θ}) where Nelsen gives exactly ½; closed form now.
- Gaussian scalar `pdf` underflowed to exactly 0.0 where its own native
  `logpdf_array` was finite; both paths now share the log-density.
- New `prg/tests/test_copula_numerics.py`: every family at τ ∈ {0.8, 0.9,
  0.95} (and τ < 0 where admissible) — vectorised paths equal scalar paths,
  density equals the mixed finite difference of the CDF, h equals ∂C/∂u,
  `inv_h ∘ h = id`, plus the exact numbers the audit reported (424 cases).
  None of the existing tests looked above τ = 0.8.
- Impact on recorded results, **measured rather than assumed** (old code at
  e7317cf vs new, same seeds): at the (family, τ) pairs the experiments use,
  the primitives agree to ≤ 1e-11 except GH and A14 at τ = 0.7 (extreme
  corners only) and Joe at τ ≥ 0.5. Replaying the exposed experiments gave
  bit-identical error rates for all 84 replicate × copula rows of CSDA
  Tables 2–5 at τ = 0.70; the same families, τ and error rates for 14 ICE
  runs of Experiment 3 (mle, aic, huard); and no family change in 120
  refits of `gof_weighting_diagnosis.py` (Joe selected 21 times, always where
  the old code was exact). The GoF power panels and the O7 margin studies
  use Gauss/Clayton/GH at τ ≤ 0.6 and are unaffected. No recorded result
  needs re-running for K-1..K-3; see K-6 for the Huard criterion.

### Fixed (2-D density plots hid their highest-density regions)

- **Filled-contour plots left over-range cells unpainted.** Every 2-D density
  plot clips its levels to a percentile of the field (97th for copula
  densities, 1st/99th for joint laws) — sensible, since copula densities
  diverge at the corners, with a measured max/p97 ratio of ~10¹⁵ for Clayton,
  GH and Student. But `contourf` given explicit levels paints *nothing* above
  the last one, so the clipped region came out **white**: precisely the
  regions of highest density were rendered as if there were no data. On the
  Student copula the two white blobs were its tail-dependence lobes — the
  family's signature feature, and the reason one plots it.
- Fixed at all six sites (`plot_pdf`, `plot_samples`, `plot_overview`,
  `plot_multi_tau`, `BivariateLaw.plot_pdf`, `plot_pdf_with_margins`) with
  `extend="max"` (or `"both"` where the floor is clipped too), which paints
  over-range cells in the end colour and marks the colour bar with an arrow,
  so the saturation stays visible as saturation.
- Colour bars now name their quantity (`density c(u, v)`, `joint density
  f(x, y)`), and the joint-law bar labels every third level instead of all 15
  — it was carrying fifteen 4-decimal numbers.

### Added (`huard_global`: the clean prior manipulation, and its exact algebra)

- **New `selection_criterion="huard_global"`** — the Huard evidence under a
  prior of common *mass* rather than common support: each family keeps its own
  integration domain, only the normalisation is shared. This is an exact
  additive shift, so it isolates the prior term and nothing else:
  `score_global(r) = score_Eq20(r) − log(Λ_g / Λ_r)`, verified to machine
  precision. Eq. (20) therefore hands every candidate a free, data-independent
  bonus of **−log Λ_r**, tabulable before any data is seen: +2.49 nat to a
  family declaring `[0, 0.165]` over one declaring `[−1, 1]`, +1.80 over
  `[0, 1]`, 0 to the widest. On a block with ~100 effective observations those
  constants are the same order as the log-likelihood differences between
  neighbouring copulas — so they can decide the selection.
- **This is the referee-proof version of the isolation experiment.** The
  earlier `huard_common` integrates over the *intersection* of the declared
  ranges, which in the second candidate set is `[0, 0.165]` — it confines every
  family to a near-independence regime, conflating "common support" with
  "support too narrow to discriminate". Results with the clean manipulation are
  stronger: on the two configurations whose true off-diagonal family is the
  narrow one, Huard's recovery falls from 41 %/38 % to **1.5 %/5.0 %** — i.e.
  *below* the maximised likelihood it was beating (6.5 %) — with paired
  discordance 79:0 and 66:0 (p = 3.3e-24, 2.7e-20). Where the true family is
  wide, the effect is small (35.0 → 30.0 %, 17:7, p = 0.064).
- **Control**: on the diagonal blocks, whose true families declare wide ranges,
  all three variants agree within one point (57.0/56.5/57.5, 97.0/97.5/96.5,
  50.0/50.0/50.0, 90.0/90.5/89.5) — the prior term decides the sparse,
  narrow-family blocks and nothing else, exactly as the algebra predicts.

### Added (methodological note draft)

- **`note/`** — draft of a short methodological note built on these results:
  the `−log Λ_r` identity, both isolation manipulations with the diagonal
  control, the 100-run paired benchmark, and the negative result on the
  cross-validated criterion. The penalty-scale defect is demoted to a
  cautionary appendix: verification against reference implementations
  (VineCopula, vinecopulib, scikit-learn, MATLAB `aicbic`) shows the correct
  convention is well established there, so it is a documented implementation
  pitfall rather than a finding. The appendix also records that there is **no
  consensus on the weighted sample size** — vinecopulib uses Kish's
  `(Σw)²/Σw²` where this package uses `Σw`, and for responsibilities
  `Σw ≤ Kish ≤ n`, making our choice the least penalising of the three.

### Added (`huard_common`: isolating the prior-width effect)

- **New `selection_criterion="huard_common"`** — the Huard evidence
  integrated over a τ-support *shared* by all candidates (the intersection
  of their declared ranges) instead of each family's own. Its purpose is
  diagnostic. With per-family supports, which is what CSDA-2013 Eq. (20)
  prescribes, a narrow-range family concentrates more prior mass and is
  rewarded for it, so comparing Huard against a likelihood criterion
  conflates *"averaging over τ is robust on sparse cells"* with *"this
  family's declared range happens to be narrow and to contain the truth"*.
- **The isolation experiment settles which it is**
  (`report/prior_width_isolation.py`, 1200 ICE runs, results committed).
  Same data, truth, candidates and seeds; only the prior support changes.
  Given a common support, Huard's off-diagonal advantage over MLE retains
  **0–5 %** of its size in three of the four configurations and 38 % (a
  mere +3.0 points) in the fourth; the paired comparison of the two
  variants is one-sided at 67:0, 12:0 and 60:0 discordant cells. So Huard's
  superiority in this benchmark is almost entirely an effect of the
  candidates' declared τ-ranges, not of the τ-averaging — legitimate
  Bayesian behaviour, but a property of the candidate set rather than of
  the criterion. §7 of the report now states this, and the recommendation
  is qualified accordingly.

### Added (research opportunity O1 — cross-validated selection criterion)

- **New `selection_criterion="xvcic"`** — a cross-validated out-of-fold
  weighted log-likelihood, the answer to the objection that AIC's `2k` and
  BIC's `k·log n` are *fixed* penalties derived for a genuine likelihood with
  known margins, whereas here the margins are plug-in estimates and the
  weights are posterior probabilities. Cross-validation measures out-of-sample
  fit instead of assuming a penalty, so the effective penalty adapts to the
  family (Grønneberg & Hjort 2014 show the analogous breakdown under
  rank-based pseudo-likelihood, where for Gumbel and Joe no finite correction
  exists at all). The folds are **contiguous** — the pseudo-observations form
  a Markov chain, so random folds would leave each held-out point's
  neighbours in the training set — **Σξ-balanced** rather than count-balanced,
  and **purged** by one point on each side because consecutive pairs share an
  observation. Fully deterministic; degrades to the plain weighted
  log-likelihood when a block carries too little mass to split.
- **Honest assessment**: on a synthetic grid with a known true family
  (n = 400, 7 candidates), `xvcic` recovers 81 % against 90 %/92 % for the
  *corrected* AIC/BIC, 84 % for Huard and 70 % for MLE — at 5.6× the cost of
  MLE per block. It beats the un-penalised likelihood and is theoretically
  better founded, but it does **not** outperform AIC/BIC here: once the
  penalty is on the right scale, the fixed-penalty approximation holds up
  well in this regime. It also validates the copula stage only
  *conditionally on the margins* (the pseudo-observations come from margins
  fitted on all the data), so it approximates the two-stage criterion of
  Ko & Hjort rather than a rank-based CIC, and does not carry the
  Grønneberg–Hjort guarantee. Documented as such in the docstring.

### Fixed (the GUI erased TOML keys it has no widget for)

- **`_rebuild_model_from_widgets` now merges into `[ice]` instead of
  replacing it.** It did `raw["ice"] = self._tab_ice.get_cfg()`, and the tab
  only emits the twelve keys it owns a widget for — so loading a model that
  declared `selection_criterion` or `margin_selection_rule` and saving it
  back **silently deleted them**. Estimation itself was never affected (the
  merge order in `_parse_ice_cfg` keeps a TOML value the GUI cfg omits, which
  was verified before changing anything); the loss was to the file on disk.
  Guarded by a round-trip test. The GUI still offers no way to *choose* a
  criterion — that remains a feature gap, not a defect.

### Fixed (A14 realised the wrong Kendall τ)

- **`CopulaA14` inverted the wrong τ↔θ relation.** It used
  `θ = 2/(3(1−τ))`, whereas Nelsen's family 14 has `τ = 1 − 2/(1+2θ)` and
  therefore `θ = 1/(1−τ) − 1/2`. The two expressions coincide at exactly one
  point — τ = 1/3, the family's lower bound, where both give θ = 1 — so any
  fixture pinned near the boundary looked correct while the copula silently
  realised **τ = 0.539 when asked for 0.6**. Now accurate to 1e-4 across the
  whole range.
- **A registry-wide sampling test now closes this class of defect** (`slow`):
  for every registered family, the Kendall τ *realised* by `sample()` must
  match the τ requested. This is the one property the rest of the suite
  structurally cannot check — every other test compares a family against
  itself (pdf against the derivative of its own cdf, `pdf_array` against
  `pdf`, a survival copula against its base), and a self-consistent family
  with a wrong τ→θ map passes them all. It is how Plackett's Spearman-ρ
  inversion was found, and it would have caught A14. All 17 families now
  pass; the two defects found this way were the only ones.

### Fixed (BB1 was unfittable over a third of its τ range)

- **BB1's joint δ–τ constraint is now enforced by projection.** The family
  admits a parameter pair only when `θ = 2/(δ(1−τ)) − 2 > 0`, i.e.
  `δ < 1/(1−τ)` — a *joint* constraint that a box-bounded optimiser cannot
  express. The optimiser's default `δ = 1.5` is inadmissible for any
  `τ ≤ 1/3`, so a fit on moderately-dependent data started on the constant
  failure penalty, saw a zero gradient, and returned δ at its untouched
  initial value; every subsequent construction then raised. Measured before
  the fix: **0/10** valid fits at τ = 0.2, 1/10 at τ = 0.3, 10/10 only from
  τ ≈ 0.45 up.
- New `CopulaVirt.constrain_params()` hook (identity by default) projects a
  parameter dict onto the family's admissible set; `CopulaBB1` overrides it
  with `CopulaBB1.delta_max(τ)`. `_fit_copula_params` applies it to the
  starting point, inside the objective and to the result, so the optimiser
  never evaluates an inadmissible point. After the fix: 10/10 valid fits at
  every τ tested from 0.10 to 0.70, with accurate estimates
  (δ̂ = 1.19/1.35/1.75/2.07 for δ = 1.15/1.30/1.70/2.00), and BB1 does not
  usurp other families (Clayton data still selects Clayton 8/8).

### Fixed (independence copula could never be selected)

- **`CopulaProduct` is now reachable.** Two independent defects kept it out:
  `_pad_tau_bounds` padded its degenerate range `τ_min == τ_max == 0` into
  the *inverted* interval `(+1e-8, −1e-8)`, which every bounded optimiser
  rejects; and the family inherited the default `n_params = 1`, so the
  penalised criteria charged the independence hypothesis 2 nats of AIC for a
  parameter it does not have. On exactly independent data no criterion ever
  selected it. It is now picked by BIC in 5 runs out of 5 and by AIC in 3
  (MLE still prefers a dependent family, as an un-penalised likelihood must),
  while dependent data still select the true family.

### Fixed (selection criteria — research opportunity O1, first instalment)

- **The penalised criteria compared a mean against a sum.**
  `_weighted_log_likelihood` normalised its weights and so returned a
  per-observation *mean* log-density, while `_score_aic` (`2k`) and
  `_score_bic` (`k·log n`) apply penalties calibrated for a *total*
  log-likelihood. The margin a k-parameter family had to clear was therefore
  ~1 nat **per observation** instead of ~1/n — a threshold no real family
  clears — so AIC and BIC degenerated into "fewest parameters wins" and were
  in fact the *same* selector. On a synthetic grid with a known true family,
  Student and BB1 were picked in **0 of 480** replicates in which they were
  the truth. `_score_bic` compounded this by using the sequence length
  `len(u)` as n instead of the block's effective sample size Σξ, which is
  what the weighted likelihood actually accumulates over.
- **The same defect ran through the Bayesian evidence and the GICE margin
  path.** `_huard_log_evidence` marginalised a *mean* log-density over its
  τ-grid, which flattens L(τ) and destroys the Occam factor that is the whole
  point of an evidence — the criterion recovered the true family in 3.6 % of
  cases. `_select_margin_family` scored margins the same way (mean versus
  `2·n_params` / `n_params·log n`), and its n was a *count of non-zero
  weights* rather than Σw. `cvm` and `kolmogorov` were unaffected (no penalty
  term), as is the standalone `prg.copulas` fitting layer, which already used
  total log-likelihoods.
- **Effect** — family recovery on a synthetic grid (n = 500, uniform weights,
  true family among 6, candidates among 7), before → after: `mle`
  76.5 → 77 %, `aic` 66.1 → **93 %**, `bic` 66.1 → **95 %**, `cvm`
  36.4 → 37 %, `huard` 3.6 → **80 %**. The unchanged figures for the two
  unpenalised criteria confirm the fix is confined to the penalty scale. On
  the package's own GICE fixture, `margin_selection_rule="aic"` now matches
  `mle`/`kolmogorov` on final log-likelihood instead of trailing far behind.
  Guarded by an invariant test (doubling the weights must double the
  likelihood) and a behavioural one (AIC/BIC must be able to select a
  2-parameter family). The margin test that excused AIC/BIC as "a known
  conservative trade-off" now requires every rule to recover the true family.

### Fixed (copula correctness — found while preparing the O1 work)

- **The Joe density carried a spurious leading θ factor.** `∫∫c` integrated
  to θ instead of 1 (θ ≈ 2.86 at τ = 0.5), inflating every Joe log-density by
  a constant log θ (≈ 1.05 nat at τ = 0.5). `pdf`, `pdf_array` and
  `logpdf_array` were all affected, and `SurvivalJoe` inherits them by
  delegation. **Joe belongs to the default ICE candidate list**
  (`Gauss, GH, Clayton, Frank, Joe`), so family selection in the default
  configuration was systematically biased towards Joe — measured at 100 % of
  replicates on synthetic data, whatever the true family, for every
  likelihood-based criterion. The CDF, the h-function and the sampler were
  correct throughout, so simulated data and τ↔θ conversion were never
  affected. After the fix: ∫∫c = 1.000000 from τ = 0.1 to 0.95, `pdf` equals
  ∂²C/∂u∂v to 1e-7, and the likelihood recovers the true τ.
- **Plackett was parameterised by Spearman's ρ, not Kendall's τ.**
  `_plackett_tau_from_theta` returned `(θ+1)/(θ−1) − 2θ·log θ/(θ−1)²`, the
  closed form of ρ_S — Kendall's τ has none for this family. So
  `CopulaPlackett(tau_k=0.5)` realised τ ≈ 0.35 (with ρ_S = 0.50) while every
  other family honours τ, breaking the uniform τ contract that ICE fits
  against. τ now comes from Hoeffding's identity by Gauss–Legendre quadrature,
  inverted through a cached monotone table (round-trip error ≤ 2.3e-5, ~40 µs
  per construction); |τ| beyond the reachable 0.9935 clamps with a warning, as
  Frank already does. The closed-form ρ_S stays available, correctly named, as
  `_plackett_rho_s_from_theta`.

### Tests

- **The scientific verification suite now covers all 17 registered copula
  families** instead of a hard-coded 9. Normalisation (∫∫c = 1), PDF–CDF
  consistency and Kendall-τ agreement had never run on Joe, SJoe, SClayton,
  SGH, BB1, AMH, Frank or Plackett — which is exactly how both defects above
  survived. The parametrisation is now derived from `CopulaEnum`, so a newly
  registered family is covered automatically; extending it immediately caught
  the Plackett defect. Two targeted regression tests name each bug.

### Performance

- **Student-t 2-D MLE vectorised — ×48** (2.46 s → 0.05 s at n = 300): the
  `(ρ, ν)` objective evaluated the copula density through a per-row Python
  loop (`cop.pdf(row)` × n × every optimiser step); it now goes through the
  vectorised `pdf_array` fast path (identical values to 5e-15, same EPS
  floor semantics). This also speeds up every ICE fit that considers the
  Student candidate.
- **Optional process-parallel multistart** (new shared cfg key
  `multistart_workers`, default 1 = sequential): `run_multistart` can fan
  the ICE/SEM starts out over worker processes. Starting points are
  pre-drawn with the same RNG consumption as the sequential path and SEM
  keeps its per-start FFBS seed offset, so results are **identical** either
  way (regression-tested for both estimators) — only wall-clock and log
  order change. Measured ×2.4 with 4 workers on 4 starts (spawn overhead
  amortises further on longer runs). Per-iteration `progress_cb` reporting
  is unavailable while parallel (workers log per-run completion instead);
  set `multistart_workers` in `[ice]`/`[sem]` TOML or the cfg dict.

### Changed (GUI import surface — audit A-1)

- **The GUI now imports exclusively from the public `prg.pmc` package**:
  `EXTRA_PARAM_BOUNDS`, `GICE_KNOWN_FAMILIES` and `SP2016_DEFAULT_CANDIDATES`
  are re-exported from `prg.pmc` (new "estimation constants" group in
  `__all__`), and the five GUI import sites (`dialogs`, `main_window`,
  `views`) that reached into `prg.pmc.ice` / `prg.pmc.sem` directly now go
  through `prg.pmc`. Renaming or moving estimator internals can no longer
  break the GUI as long as the package façade holds.

### Tests (Student copula — audit T-4)

- **Dedicated test module for `CopulaStudent`** (`test_student.py`, 12
  tests), the least-covered copula (49 % → 94 %): ν > 2 domain validation,
  ρ = sin(πτ/2) link, no-closed-form CDF contract, h-function CDF
  properties + near-independence limit, tail-dependence formula checked
  against an independent evaluation (with ν-monotonicity and the Gaussian
  ν → ∞ limit), 2-D MLE input validation, the `method='tau'`
  under-determination warning, and a `slow` (τ, ν)-recovery test on
  simulated data.

### Packaging / CI (last follow-up audit nits)

- **The wheel no longer ships `prg/tests/`** (audit P-13): the exclusion
  existed only on the sdist target, so `pip install .` put the 27 test files
  into site-packages. Verified: freshly built wheel has 0 test files, the
  package and its model TOMLs intact.
- **GitLab and GitHub CI now block identically**: smoke-install moved into
  GitLab's `test` stage, so packaging and code are independent, parallel
  signals on both CIs (previously GitLab serialised lint→smoke→test while
  GitHub ran smoke ∥ test).
- Both CI file headers now document the smoke job; the `slow` marker
  description no longer implies slow tests are opt-in (they run by default).

### CI

- **Explicit ruff rule selection** (`select = ["E4", "E7", "E9", "F"]` in
  pyproject): both CIs install the *latest* ruff, and ruff 0.16 changed its
  default rule set (import-sorting `I`, `RUF` rules) — turning the tree red
  overnight with 246 spurious errors while local (0.15) stayed green.
  Lint semantics no longer depend on the ruff release.

### Changed (weighted CvM de-duplication — 2026-08-05 follow-up audit)

- **The ξ-weighted CvM statistic now lives in `prg.copulas._fit`**
  (audit Q-20): `_empirical_copula` / `_cvm_statistic` accept an optional
  `weights=` (normalised; `None` keeps the classic uniform/unweighted forms
  bit-for-bit), and `ice._weighted_cvm_statistic` is a thin delegator.
  The weighted formula introduced by N-6 previously re-implemented the
  O(N²) indicator matrices in `prg.pmc`, so the two copies could drift.
  Guarded by the existing exactness + weight-sensitivity tests.

### Changed (single-source estimator defaults — 2026-08-05 follow-up audit)

- **Estimator defaults resolve from one place** (audit Q-9 — the last open
  "source of truth" drift). New `ice_estim_defaults()` / `sem_estim_defaults()`
  (re-exported from `prg.pmc`) merge the shared defaults with each
  estimator's own keys; `_parse_ice_cfg`, `_parse_sem_cfg` **and the GUI**
  (spinbox initial values + `load()` fallbacks in `_IceTab`) now all read
  them. Previously `max_iter=50`, `tol=1e-4`, `patience=3`, `max_iter=30`
  (SEM), `multistart_jitter=0.10`, … were re-hardcoded at ~15 sites — the
  GUI twice — so changing a default did not propagate.
- Dead literal fallbacks (`cfg.get(key, literal)`) removed on all
  parse-guaranteed paths (`run_multistart`, both single-run functions, both
  K-means warm-start blocks, the GUI worker) — the parse step already
  guarantees every key, so the fallbacks could only mask drift. The GUI
  multistart-seed spinbox keeps its deliberate, tested 42 (vs API 0) —
  now commented as such.

### Fixed (estimator polish — 2026-08-05 follow-up audit)

- **Frank: a clamped θ now also clips the reported τ** (audit N-9). Beyond
  the largest numerically-reachable |τ| ≈ 0.9943, `find_theta_frank` clamps
  θ to ±700 but `params['tau_k']` kept the *requested* τ — so traces, saved
  TOMLs and the GUI could overstate the modelled dependence by up to 0.0057
  with only a log WARNING as a signal. `params['tau_k']` is now clipped to
  the τ the clamped θ actually models (regression test added). The reachable
  maximum is also a module constant now (`_FRANK_TAU_MAX`) instead of a
  fresh `quad` integration at every `CopulaFrank` construction (audit N-10).
- **`joint_posteriors` degenerate steps fall back to uniform 1/K²**
  (audit N-11), mirroring `smooth`'s 1/K fallback (N-8): a fully-underflowed
  step previously kept an all-zero ξ slice while γ went uniform, silently
  breaking the γ_n(i) = Σ_j ξ_n(i, j) invariant the M-step relies on.
  Covered by a new degenerate-step invariant test.

### Changed (GUI polish — 2026-08-05 follow-up audit)

- SEM-vs-ICE result detection uses `isinstance(result, SemResult)` instead
  of the stringly `type(result).__name__ == "SemResult"` (audit Q-4 residue).
- Dead `_ICE_VIEWS` set removed from `main_window.py` — its only consumer
  was the if/elif ladder replaced by the Q-11 dispatch table (audit Q-18).
- Stale "dispatch table" comment block removed from `views.py` — the table
  lives in `main_window.py` (audit Q-19).
- `views.compute_marginal_cdfs` is public: `main_window` was reaching into
  another module's underscore-private helper (audit A-11).

### Docs (README refresh — 2026-08-05 follow-up audit)

- **"Folder structure" block regenerated and re-automated** (audit P-4): the
  `PROJECT_STRUCTURE_START/END` markers `scripts/update_readme_structure.sh`
  expects were missing, so the script had been a silent no-op while the block
  drifted (phantom `prg/settings/`, `prg/tools/`, gitignored `docs/`; ~15
  real modules missing; 5 of 27 test files listed). The markers are back, the
  block is now generated output, and a script run is a verified no-op. The
  script's exclusions were tuned (`*.tex`/`*.log` out, the two example
  notebooks in).
- **"Bivariate joint laws" example rewritten on the real API** (audit P-12):
  it used non-existent `left_dist=`/`right_dist=` kwargs, a `seed=` parameter
  `sample()` does not take, an undefined `data` variable, string margin
  families that `fit()` rejects, and `fit.tau_k` instead of
  `fit.copula_fit.tau_k`. The block was executed end-to-end as part of this
  refresh (fitted τ̂ ≈ 0.595 for true τ = 0.6).
- **ICE example** (audit P-14): `fitted, log_lik_history = ice(...)` renamed
  to `trace` — the second element has been an `IceTrace` since v0.5.0 — with
  `trace.log_liks` shown, and the example now uses the public deep-copying
  `mdl.raw` property instead of `copy.deepcopy(mdl._raw)`.
- **Sundry accuracy fixes**: GUI diagram "Margins — K×K grid" → K states
  (v0.5.0 SR-PMC contract; same fix in the `gui/tabs.py` module docstring —
  audit P-15); "Five example models" → the actual eight, table completed
  (audit P-16); hardcoded `v0.8.0` dropped from the README title (audit
  P-17); logging "two sinks" → three; `prg/pmc/__init__.py` no longer points
  at the gitignored `docs/` PDFs (audit P-7 residue).

### Tests (environment robustness — 2026-08-05 follow-up audit)

- **Notebook tests now pin the Jupyter kernel to the running interpreter**
  (audit T-8). `kernel_name="python3"` resolved through the standard Jupyter
  search path, where a user-level kernelspec (e.g.
  `~/Library/Jupyter/kernels/python3`) shadows the venv's kernel and can point
  at a *different* interpreter — on the audit machine both notebook tests
  failed with `ModuleNotFoundError: statsmodels` raised from another project's
  venv. A shared `venv_kernel_manager` fixture (conftest) restricts the
  kernelspec search so `jupyter_client` falls back to the native kernel built
  for `sys.executable`.
- **Notebook tests now actually run in CI** (audit T-9): `nbformat`,
  `nbclient` and `ipykernel` joined the `dev` extra. Both tests
  `importorskip` these, so CI (`pip install -e ".[dev,gui]"`) silently
  SKIPPED them — the README's "re-run as a regression test" guarantee only
  held locally.
- **Sharper assertions** (audit T-10, T-11): the bootstrap-CI test now
  brackets the point estimate strictly (the previous ±0.1 slack made the
  assertion effectively unfalsifiable), and a new exactness test pins the
  weighted-CvM formula — zero-weight pseudo-observations must drop out of
  both the weighted empirical copula and the outer L²-sum.

### Changed (GUI refactor — post-v0.8.0 audit follow-up)

- **GUI view renderers extracted to `prg/pmc/gui/views.py`** (audit Q-10). The
  18 `_plot_*` renderers, the two shared `_draw_*` ribbon helpers, and the
  formatting/colour helpers (~1000 lines) moved out of the 2761-line
  `main_window.py` god-object into plain functions of a
  `matplotlib.figure.Figure` plus their data. They are now **unit-testable
  headless** — no `QMainWindow` required (`views.plot_ice_dashboard(fig, trace,
  Y)` renders on a bare `Figure`). `main_window.py` keeps thin two-line
  delegators + layout + state + dispatch and drops from **2766 to 1748 lines**.
  Behaviour is unchanged — a new `test_all_views_render_without_error` safety
  net renders every available view through the real dispatch and the full GUI
  suite passes.
- **View availability folded into the dispatch table** (audit Q-11).
  `_VIEW_DISPATCH` now maps each view to an `(available, render)` pair, so a
  view's gating predicate and renderer are registered in **one** place instead
  of the 35-line `_is_view_available` if/elif ladder that ran parallel to the
  table. `_is_view_available` is now a one-line lookup. Verified behaviour-
  identical to the old ladder across fresh / trace-only / fully-estimated /
  HMC-no-copula states.

### Added

- **`joint_posteriors(alpha_hat, W, beta_hat)` is now public** in
  `prg.pmc.inference` (and re-exported from `prg.pmc`), the pairwise ξ
  companion of the marginal `smooth` (γ). It was the ICE-private
  `_joint_posteriors`; promoting it gives the test suite a public entry point
  for the forward-backward pairwise posteriors (audit T-5) and rounds out the
  documented inference API.

### Changed (GUI / test coupling — post-v0.8.0 audit follow-up)

- **GUI estimate config filtered per estimator** (audit Q-15). `_do_estimate`
  now drops the ICE-only `tol` / `patience` keys before forwarding to SEM (it
  already dropped the SEM-only `sem_seed` before ICE), so neither estimator
  receives the other's knobs from the shared ICE tab. Harmless before (the
  keys were ignored) — now the contract is clean.
- **`test_forward_backward_invariants.py` imports the public
  `joint_posteriors`** instead of `prg.pmc.ice._joint_posteriors` (audit T-5).
  The remaining private-symbol imports in the test suite (selection-criteria /
  GICE / multivariate-MLE / fitting helpers, config parsers) are deliberate
  direct unit tests of internal numerics and are kept as such.

### CI (post-v0.8.0 audit follow-up)

- **New `smoke-install` CI job** on both GitHub Actions and GitLab CI (audit
  P-6). The existing test jobs install `-e ".[dev,gui]"`, which always puts the
  source tree on `sys.path` and so cannot catch a runtime dependency missing
  from `[project].dependencies`, a broken entry point, or packaging gaps. The
  new job does a clean `pip install .` (no extras), then — from a directory
  *outside* the repo — imports `prg` / `prg.copulas` / `prg.pmc` /
  `prg.diagnostics` and runs `pmc --help`. Verified locally: the install pulls
  exactly numpy / scipy / matplotlib / statsmodels / tomli_w and the smoke
  passes.

### Tests (coverage — post-v0.8.0 audit follow-up)

- **New `prg/tests/test_fit.py`** (17 tests) covering the previously
  under-tested fitting layer (audit T-4): `FitResult` information criteria
  (`aic` / `bic` / `aicc` incl. the small-sample `nan` branch / `hqc`), the
  parametric-bootstrap `gof_test` (structure, reproducibility, no-reject on the
  correct family), the non-parametric `bootstrap_ci`, `cv_loglik`, the pure
  helpers (`_empirical_copula`, `_cvm_statistic`, `_empirical_tail_dep`), and
  `BivariateLaw.fit` / `fit_best` (recovery, scoring, input validation,
  AIC-sorted candidates). Small bootstrap counts keep it off the `slow` path.

### Changed (numerical hygiene — post-v0.8.0 audit follow-up)

- **`smooth` degenerate-row fallback is now uniform `1/K`** (audit N-8). A γ row
  that sums to ≈ 0 (only in pathological / underflowed sequences) previously
  became a zero row, biasing `mpm`'s argmax — and any γ-weighted downstream step
  — toward state 0. It now falls back to a uniform posterior (matching
  `sample_posterior`), so every row is a valid distribution summing to 1.
- **Numerical-regularisation constants centralised in `ice.py`** (audit Q-14).
  The τ-bound padding formula (copied three times with subtly different floors),
  the variance floor, the covariance jitter, the Huard τ-grid size, and the
  K-means `n_init` are now named module constants (`_TAU_PAD_REL/_ABS`,
  `_VAR_FLOOR`, `_COV_JITTER`, `_HUARD_N_GRID`, `_KMEANS_N_INIT`) plus a
  `_pad_tau_bounds()` helper, so the values can't drift. No behaviour change on
  any non-degenerate input.

### Examples

- **`examples/quickstart.ipynb` refreshed** to match the current package:
  intro now describes **three** layers (adds `prg.diagnostics`); a new SEM
  section (§5b) runs the stochastic estimator alongside ICE on the same data
  and overlays their log-likelihood traces; a new goodness-of-fit section (§7)
  demos `mks_2samp` (simulated-from-fitted vs observed); the stale "294 tests"
  footer is gone and the CLI cheat-sheet now lists `--algorithm {ice,sem}` and
  the `classify-image` / `estimate-image` subcommands. Verified end-to-end via
  `test_quickstart_notebook.py`.

### Changed (finishing touches — post-v0.8.0 audit follow-up)

- **Optimiser-failure sentinels centralised** (audit Q-8). The scattered,
  inconsistent magic penalties in `ice.py` (`1e12`, `1e15`, `-1e12`, `-1e8`,
  `-1e6`) are now three named module constants — `_OPT_FAIL_PENALTY` (1e12),
  `_LOGLIK_FAIL` (-1e12), `_LOGPDF_FLOOR` (-1e8) — so the magnitudes stay
  consistent (the previous `1e15` / `-1e6` outliers are folded into the
  shared values).
- **Huard evidence normalised over the full τ-grid** (audit N-7):
  `_huard_log_evidence` divided the integrated likelihood by the *surviving*
  grid-point count, which silently re-normalised a numerically-fragile family
  onto a sub-interval and inflated its evidence. It now divides by the fixed
  `n_grid` (identical to the old result when every grid point is finite).
- **`prg/copulas/__init__.py` translated to English** (audit A-10), matching
  the rest of the codebase.

### Changed (API & type hygiene — post-v0.8.0 audit follow-up)

- **`mks_test` accepts `other` / `cdf` positionally** (audit A-3). They were
  keyword-only, so the documented `mks_test(a, b)` two-sample call raised
  `TypeError`; `alpha` / `asymptotic` stay keyword-only. New test.
- **Estimation result/trace types re-exported from `prg.pmc`** (audit A-4):
  `IceTrace`, `IceResult`, `SemTrace`, `SemResult` are now importable from the
  package root (the `ice()`/`sem()` return types were previously reachable only
  via the submodules). New test.
- **`prg.pmc.inference` gained an `__all__`** (audit A-7) matching its
  documented public API.
- **`MULTIVARIATE_DISTS` is now public in `prg.pmc.model`** (audit A-8), so
  `ice.py` no longer imports the underscore-private `_MULTIVARIATE_DISTS`
  across the module boundary (kept as a back-compat alias).
- **`EXTRA_PARAM_BOUNDS` lookup keyed by the registry's `CLASS_NAME`** (audit
  A-9) instead of `cls.__name__`, matching the source the dict is built from.
- **`Callable` type annotations** (audit Q-12): `_SCORE_FN` (ice) and
  `_VIEW_DISPATCH` (GUI) used the builtin `callable` as a type; now
  `collections.abc.Callable`, valid for type-checkers.

### Changed (docs & single-source — post-v0.8.0 audit follow-up)

- **Extra-parameter copula bounds single-sourced** (audit A-2). The
  `(lower, upper, init)` bounds for BB1 `delta` / Student `df` were hand-copied
  in `prg/copulas/_base.py` (param-keyed, inside `fit`) and `prg/pmc/ice.py`
  (class-keyed) — a silent-drift hazard. The canonical param-keyed dict now
  lives once in `prg.copulas._base.EXTRA_PARAM_BOUNDS_BY_PARAM`; `ice.py`
  derives its class-keyed `EXTRA_PARAM_BOUNDS` from it plus the registry
  (verified identical to the previous literal).
- **README CLI table completed** (audit P-5): adds the `classify-image` and
  `estimate-image` subcommands and notes `--algorithm {ice,sem}` on the
  estimate commands.
- **`docs/` PDF references corrected** (audit P-7): the `docs/` directory is
  git-ignored and absent from clones, so "PDFs shipped in `docs/`" (README +
  `prg/pmc/README.md`) is replaced by a pointer to the papers' DOIs.
- **Top-level docstring mentions `prg.diagnostics`** (audit A-6): the package
  overview now lists three layers (copulas / pmc / diagnostics) instead of two.

### Changed (ICE/SEM de-duplication — post-v0.8.0 audit follow-up)

- **Multistart driver shared** (audit Q-1). The ~40-line "perturb → run →
  keep best → attach losing traces" loop, previously copy-pasted in `ice()`
  and `sem()`, is now a single `prg.pmc._estim_common.run_multistart()` that
  both call with a thin per-estimator closure (SEM threads its per-start FFBS
  `seed_offset` through it; ICE ignores the start index). Behaviour unchanged.
- **Image wrappers de-duplicated** (audit Q-3). `ice_image` / `sem_image` had
  byte-identical 2D/3D validation; both now call
  `_estim_common.linearise_image(model, img, what=…)`.
- **Trace/Result dataclasses share a base** (audit Q-4). `SemTrace` now
  subclasses `IceTrace` (adding only `sampled_X_history`) and `SemResult`
  subclasses `IceResult` (kept as a distinct class so the GUI can still tell
  the two apart by type). Eliminates ~45 lines of mirrored field declarations.
- Net effect: `ice.py` −47 and `sem.py` −75 lines, consolidated into
  `_estim_common`. (The single-run loop body, audit Q-2, is intentionally
  left per-estimator: its E-step and convergence logic differ enough that a
  shared template would be a worse abstraction than the small remaining
  duplication.)

### Fixed (numerical — post-v0.8.0 audit follow-up)

- **`_kolmogorov_distance` now handles tied observations** (audit N-4). The
  weighted empirical CDF used `argsort`+`cumsum`, giving *tied* values distinct
  partial sums (`[1,1,1,2]` → 0.25/0.50/0.75 instead of 0.75/0.75/0.75) and so
  fabricating spurious KS deviations on quantised/discrete data (image gray
  levels, sensor counts). It is now right-continuous (each tied run shares the
  run's final cumulative weight), so the GICE `margin_selection_rule="kolmogorov"`
  is correct on discrete margins. New test in `test_gice_margins.py`.
- **`cvm` copula-selection criterion is now ξ-weighted** (audit N-6). `_score_cvm`
  passed raw pseudo-observations to the *unweighted* CvM helper, so observations
  that barely belong to pair `(i, j)` drove family selection — inconsistent with
  the ξ-weighted M-step used by every other criterion (mle/aic/bic/huard). A new
  `_weighted_cvm_statistic` uses the weighted empirical copula `C_n^w` and a
  weighted L²-sum (the shared standalone-fit `_cvm_statistic` is left unchanged).
  New tests in `test_selection_criteria.py`.
- **SEM multistart now draws independent FFBS streams** (audit N-5). Every start
  reused the same `sem_seed`, so all runs drew byte-identical stochastic
  completions (only the initial model differed). Each run now uses
  `sem_seed + start_index`; the unperturbed run keeps offset 0, so single-start
  fits stay bit-for-bit reproducible. New test in `test_sem.py`.

### Tests (robustness — post-v0.8.0 audit follow-up)

- **CLI tests no longer depend on install state** (audit T-1). `test_cli.py`'s
  subprocess helper now prepends the repo root to `PYTHONPATH`, so
  `python -m prg.pmc` resolves from this checkout even when the editable
  install is stale/broken (the previous `ModuleNotFoundError: No module named
  'prg'` from a temp cwd is gone). The 7 CLI tests now pass without a working
  `pip install -e`.
- **Suite runnable from any working directory** (audit T-2). Added
  `prg/tests/conftest.py` that (a) prepends the repo root to `sys.path` so
  `import prg` works without an editable install, and (b) runs the session
  from the repo root so the ~37 cwd-relative `prg/pmc/models/...` paths across
  9 test files resolve. Verified: `pytest prg/tests/test_sem.py` from `/tmp`
  now passes (previously `FileNotFoundError`).
- **MKS Type-I calibration tests less flake-prone** (audit T-3): tolerance
  widened from α+0.03 (≤ 6/80 — ~10 % binomial false-failure margin at the
  nominal rate) to α+0.05 (≤ 8/80), with a comment recording that the fixed
  seed makes the result deterministic and the Naaman bound keeps the empirical
  rate ≈ 0.

### Fixed (post-v0.8.0 audit — verified bugs)

- **Frank copula no longer crashes for strong dependence** (audit N-2).
  `find_theta_frank` hard-capped its Brent bracket at θ = 500 (|τ| ≈ 0.992),
  so any `tau_k` in the gap (0.992, 1) — reachable from a TOML `[[copulas]]`
  block or an ICE fit in strong dependence — raised
  `ValueError: f(a) and f(b) must have different signs`. The bracket now spans
  the full numerically-stable range (θ = ±700, |τ| ≈ 0.994) and **clamps with a
  warning** above the largest reachable |τ| instead of raising — matching the
  graceful τ-saturation of the other archimedean families and the
  `[EPS_MINUS_ONE, ONE_MINUS_EPS]` τ-range the registry advertises for Frank.
- **A12 copula scalar CDF no longer requires NumPy ≥ 2.0** (audit N-3).
  `CopulaA12.cdf` used `np.pow` (an alias added only in NumPy 2.0) while the
  project floor is `numpy>=1.24`, so the scalar CDF raised `AttributeError` on
  NumPy 1.x. Replaced with `np.power` (matching the vectorised `cdf_array` and
  the sibling A14 family).
- 8 new regression tests in `prg/tests/test_copulas.py`: Frank strong-dependence
  construction across the former crash gap (both signs), clamp-with-warning above
  the reachable τ, θ↔τ round-trip accuracy in the safe range, and A12 scalar-vs-
  vectorised CDF agreement.

### Packaging / docs (post-v0.8.0 audit follow-up)

- **`requirements.txt` now matches `pyproject.toml`** (audit P-1). It listed
  `pandas` and `rich` (imported nowhere in the repo) and omitted `statsmodels`
  and `tomli_w` (both hard runtime deps), so `pip install -r requirements.txt`
  produced an env where `import prg.copulas` failed. Now: numpy, scipy,
  matplotlib, statsmodels, tomli_w.
- **README dependency table corrected** (audit P-2): dropped the phantom
  pandas/rich rows, added statsmodels/tomli_w; the GUI install now uses the
  `[gui]` extra (`pip install 'copulasformm[gui]'`) instead of a bare
  `pip install PyQt6` (audit P-10).
- **Single source of truth for the version** (audit P-3): the package version
  now lives only in `prg/__init__.py::__version__`, read at build time by
  Hatchling (`[tool.hatch.version]` + `dynamic = ["version"]`). The hardcoded
  `version` in `pyproject.toml` and the literal assertion in
  `prg/tests/test_smoke.py` (now a format check) are gone. Verified: a wheel
  build resolves the version correctly.
- **`dev` extra de-duplicated** (audit P-8): it now references
  `copulasformm[image,ml]` instead of re-pinning Pillow / scikit-learn, so each
  optional dep is pinned in exactly one place.
- **Ruff config tightened** (audit P-9): `E402` moved from a blanket
  project-wide ignore to scoped `per-file-ignores` (so import-ordering mistakes
  are still caught in the top-level modules). `E702` stays ignored but is now
  documented as deliberate (used in ~37 compact paired-assignment lines —
  contrary to the audit's assumption that it was unused).
- **Trove classifier** bumped `Development Status :: 2 - Pre-Alpha` →
  `3 - Alpha` (audit P-11).

### Changed (dead-code / cruft — post-v0.8.0 audit follow-up)

- **`PMCModel.sem_config()` implemented** (audit Q-5/A-5). `_parse_sem_cfg`
  guarded its `[sem]`-section read behind `hasattr(model, "sem_config")` — a
  method that did not exist — so a `[sem]` TOML block was silently ignored and
  SEM only ever read `[ice]`. SEM now merges `[ice]` then `[sem]` (the latter
  overriding shared keys), matching the documented resolution order. New test
  in `test_sem.py`.
- **Removed a redundant exception clause** (audit Q-7): `_score_cvm` caught
  `except (NotImplementedError, Exception)` — identical to `except Exception`
  since `NotImplementedError ⊂ Exception`. Simplified, with the
  Student-no-CDF rationale moved into the comment.
- **Removed a stale `# noqa: ARG001`** (audit Q-6) on `_huard_log_evidence`'s
  `params` argument, which claimed the argument was unused — it is used to fix
  multi-parameter extras (Student `df`, BB1 `δ`).

### Changed (hygiene — post-v0.8.0 audit follow-up)

- **New internal module `prg/pmc/_estim_common.py`** — single source of
  truth for the M-step / warm-start infrastructure shared between ICE
  and SEM (snapshots, `m_step`, `warmstart_from_kmeans`,
  `perturb_initial_model`, `kmeans_label_assignment`,
  `check_init_strategy`, shared selection-criteria constants, and a new
  `shared_estim_defaults()` helper). `sem.py` no longer reaches into
  `prg.pmc.ice._xxx` underscore-private symbols; the dependency
  direction is now explicit and documented.
- **Cfg consolidation.** `_parse_ice_cfg` and `_parse_sem_cfg` both
  build their defaults on top of `shared_estim_defaults()`, eliminating
  the duplicated shared-key list and preventing silent default-drift
  between the two estimators.
- **`mks_1samp` / `mks_2samp` vectorised.** Replaced the inner double
  for-loop over corner-grid points with a batch empirical-CDF helper
  (`_mecdf_batch`) that uses NumPy broadcasting. `prg/tests/test_mks.py`
  is ~25 % faster as a side effect.
- **Explicit `__all__`** added to `prg/pmc/ice.py`, `prg/pmc/sem.py`,
  `prg/pmc/_estim_common.py`, and `prg/diagnostics/mks.py` — pins the
  public surface and stops `from … import *` from leaking internals.
- **Tests updated** to import the shared estimator helpers from
  `_estim_common` rather than reaching into `prg.pmc.ice`'s
  underscore-private namespace (`test_ice_kmeans_init.py`). Tests that
  target genuine ICE-internals (`test_gice_margins.py`,
  `test_multivariate.py`) keep their private imports — those helpers
  are deliberately not part of the shared surface.
- **Dead code removed**: the deprecated `_fit_gaussian_margin_weighted`
  shim, the `_EXTRA_PARAM_BOUNDS` back-compat alias, and the unused
  `_mecdf` (superseded by `_mecdf_batch`).

No behaviour change. 685 tests pass, ruff clean.

---

## [0.8.0] - 2026-05-14

This release closes the four-PR migration from the `markovchain_todelete`
companion project by adding a standalone **multivariate goodness-of-fit
test** (PR3) and a **real-data benchmark notebook** (PR4) on top of the
v0.7.0 SEM / K-means changes.

### Added

- **`prg.diagnostics` sub-package** with the **multivariate
  Kolmogorov-Smirnov test** (`mks_1samp`, `mks_2samp`, `mks_test`
  dispatcher, `MKSResult` typed result). Standalone goodness-of-fit
  utility — independent of any specific model class. Default
  critical value uses Naaman's finite-sample union bound (conservative,
  safe); `asymptotic=True` switches to the tighter large-`N`
  approximation.
  - 16 new tests in `prg/tests/test_mks.py`: input validation, H0
    calibration over many trials (Type-I error ≤ α + 0.03), H1 power on
    contrast settings, univariate sanity vs `scipy.stats.ks_1samp` /
    `ks_2samp`, 3-D smoke test, dispatcher routing, asymptotic-vs-
    finite-sample critical-value ordering.
- **Real-data benchmark notebook** at
  `examples/uci_har_smartphone.ipynb` exercising ICE + SEM + MKS on the
  *UCI Human Activity Recognition Using Smartphones* dataset (3-D
  accelerometer, 50 Hz, walking vs laying). Dataset downloaded on first
  run and cached in `data/uci_har/` (gitignored); the notebook falls
  back transparently to a synthetic 3-D signal simulated from the
  `hmc_in_mvn_k2_d3.toml` fixture if the download is unavailable. The
  `data/uci_har/README.md` cache pointer (with source / license /
  citation) **is** committed.
  - 1 new slow regression test (`prg/tests/test_uci_har_notebook.py`)
    that patches the notebook to force the offline fallback and executes
    every cell with a fresh Jupyter kernel.

### Origin
Ported (rewritten in copulasformm style — typed result, NumPy docstrings,
validation, logger) from the `markovchain_todelete` companion project
(`prg/mks_test/`). Source-of-truth algorithm:
[o-laurent/multivariate-ks-test](https://github.com/o-laurent/multivariate-ks-test).
The UCI HAR benchmark notebook generalises the smartphone-data pipeline
of `markovchain_todelete/prg/pipeline_SmartphoneData.py`.

---

## [0.7.0] - 2026-05-14

This release adds **SEM (Stochastic EM)** as a sister estimator to ICE
and introduces a **K-means warm-start** option shared by both
algorithms — both ideas cherry-picked and rewritten from the
`markovchain_todelete` companion project (files `prg/PMC_Estim.py` and
`prg/tools/probaPMC.py::simulRealisationAP`).

### Added

- **SEM — Stochastic EM estimator** in the new `prg/pmc/sem.py` module.
  Public API: `sem(model, Y, sem_cfg=None) -> (PMCModel, SemTrace)` and
  the image wrapper `sem_image(model, img, sem_cfg=None)`. SEM differs
  from ICE by drawing a single posterior realisation
  `X̃ ~ P(X | Y)` at every iteration (Forward-Filter Backward-Sample)
  and running the same supervised-style M-step on these hard labels.
  Shares the M-step (`_m_step`) and the K-means warm-start
  (`_warmstart_from_kmeans`) with ICE, so the two estimators stay
  exactly aligned on the parameter-update logic.
  - `SemTrace` mirrors `IceTrace` (same fields plus
    `sampled_X_history` capturing the per-iteration FFBS draws).
  - `SemResult` mirrors `IceResult` for GUI/diagnostics layers.
  - New config keys: `max_iter` (default 30), `sem_seed` (RNG seed for
    the FFBS draws). Other keys (`fit_margins`, `candidates`,
    `selection_criterion`, `margin_selection_rule`, `init`,
    `kmeans_seed`, `n_starts`, `multistart_seed`, `multistart_jitter`)
    are reused from ICE's config namespace.
  - `[sem]` TOML section recognised (falls back to `[ice]` for shared
    keys).
- **`sample_posterior(model, Y, rng)`** in `prg/pmc/inference.py` —
  Forward-Filter Backward-Sample (FFBS) draw of `X | Y`. Standalone
  utility, also used by SEM internally. Pre-computed `W`/`alpha_hat`
  can be passed in to avoid recomputation.
- **CLI `--algorithm {ice,sem}` flag** on both `estimate` and
  `estimate-image` (default `ice` — backward-compatible). Companion
  flag `--sem-seed` controls SEM's stochastic completion seed. Example:
  `python -m prg.pmc estimate --model M.toml --data Y.csv
  --algorithm sem --max-iter 30 --sem-seed 0 --fit-margins`.
- **GUI `_IceTab`** (ICE config panel) renamed conceptually to the
  "estimator" panel: new top "Algorithm" section with an `Estimator`
  combobox (`ice` / `sem`) and a `SEM seed` spinbox (auto-disabled
  when `ice` is selected). The worker (`_do_estimate`) dispatches to
  `ice()` or `sem()` based on the combobox value; the result-handling
  code is identical because `SemResult` and `IceResult` expose the
  same diagnostic fields.
- **K-means warm-start for ICE & SEM.** New TOML `[ice]` keys `init`
  (one of `"model"` — default, backward-compatible — or `"kmeans"`)
  and `kmeans_seed` (RNG seed forwarded to `sklearn.cluster.KMeans`).
  When `init = "kmeans"`, the estimator clusters `Y` with k-means++ and
  derives a warm-start model from the hard labels via a single
  supervised-style M-step (prior + copula τ always re-estimated;
  margins re-estimated only if `fit_margins` is true). The variant,
  K, margin distribution families and copula candidates are preserved.
  Multistart, when enabled, perturbs the warm-started model.
- **Optional `[ml]` extras** in `pyproject.toml` adding `scikit-learn` —
  required only by the K-means warm-start. Install with
  `pip install 'copulasformm[ml]'`.

### Changed

- **Refactor (no behaviour change)**: the ICE M-step (prior / margins /
  copulas update from posterior weights) is now a stand-alone
  `_m_step()` helper, called by both the ICE iteration loop, the SEM
  iteration loop, and the K-means warm-start path. This eliminates
  the duplication that was about to appear between ICE and SEM.

### Tests

- 16 new tests for the K-means warm-start (`test_ice_kmeans_init.py`).
- 8 new tests for the FFBS posterior sampler (`test_posterior_sampling.py`),
  including a Monte-Carlo check that the empirical marginal of many FFBS
  draws converges to the smoothed posterior γ_n(j).
- 11 new tests for SEM (`test_sem.py`): config plumbing, trace shapes,
  reproducibility under fixed `sem_seed`, stochastic variance across
  seeds, LL improvement from an adversarial init, end-to-end recovery
  with K-means warm-start, multistart, PMC variant smoke test, image
  wrapper smoke test.
- 4 new GUI tests (`_IceTab`): `init` / `kmeans_seed` widgets and the
  new `algorithm` / `sem_seed` widgets.

### Origin

The SEM and K-means warm-start ideas are cherry-picked from the
`markovchain_todelete` companion project (`prg/PMC_Estim.py:65` for
K-means; `prg/tools/probaPMC.py::simulRealisationAP` for FFBS-based
stochastic completion).

---

## [0.6.0] - 2026-05-05

This release adds **GICE** — automatic margin family selection from
Derrode-Pieczynski (Signal Process. 2016) — on top of the existing
ICE machinery, polishes the PyQt6 GUI substantially, and parallelises
the CSDA-2013 reproduction script. No public-API breaking changes.

### Added

- **GICE — automatic margin family selection (paper A23 / SP 2016, §3).**
  Margin blocks now accept an optional ``candidates`` list (a set of
  ``scipy.stats`` family names). When non-empty, the ICE M-step fits
  each candidate and picks the winner via a configurable rule:
  ``mle`` (default), ``kolmogorov`` (SP 2016, Example 3.1),
  ``aic`` or ``bic``. Eight families ship with data-aware
  initialisation heuristics: ``norm, gamma, invgamma, betaprime,
  lognorm, expon, weibull_min, beta``. Other ``scipy.stats`` names
  still work; the M-step falls back to ``scipy.stats.<dist>.fit`` for
  the init point. Public constants ``ice.GICE_KNOWN_FAMILIES`` and
  ``ice.SP2016_DEFAULT_CANDIDATES`` exposed.
- **New shipped fixture** ``models/sp2016_gice_k2.toml`` reproducing
  the SP-2016 §5.1 setup (Gamma + BetaPrime margins).
- **GUI: new ICE view "Parameters: true vs fitted"** (view *M*) — at
  a glance comparison of seed and ICE-fitted parameters: per-state
  margins (true vs fitted family + params), per-pair copulas
  (family + ``τ`` + ``Δτ``, with colour flags for family changes
  and ``|Δτ| > 0.10``), and three K×K joint-prior heatmaps
  (true / fitted / ``fitted − true``).
- **GUI: inline Export button** next to the View combobox, mirroring
  ``File → Export plot…``.
- **GUI: figure-level legend** for the dashboard's family-ribbon
  panel — the ribbon colours now have a decoder, no longer hidden
  inside a stripped-out per-axes legend.
- **GUI: auto-load** the bundled fixture
  ``prg/pmc/models/pmc_gauss_k2.toml`` when the GUI is launched
  without a path argument, so the user lands on a workable state
  instead of a fully-disabled UI.
- **GUI: GICE candidate sets edited as checkboxes**, both at the
  ICE-config level (``_IceTab``) and per-margin (``_MarginDialog``).
  Replaces the previous comma-separated text boxes; quick-pick
  buttons (All / None / SP-2016 §3 / Defaults) cover the common
  selections.
- **GUI: 🎲 random-seed button** for the multistart seed widget,
  drawing from ``secrets.randbelow(2³¹)``. Default seed changed
  ``0 → 42``.
- **GUI: LaTeX-mathtext labels** across every plot (axes, titles,
  legends use ``$Y_n$``, ``$P(X_n = k \mid Y)$``,
  ``$\Vert p^{(t)} - p^{(t-1)} \Vert_F$``, ``$\hat{\tau}$``, etc.)
  rendered with the bundled Computer-Modern font set. Compact font
  sizes: titles 10 pt, axes 9 pt, ticks 8 pt; constrained_layout on
  by default so 3×2 dashboards no longer overlap.
- **GUI: canvas figsize** bumped from (7, 5) to (9.5, 6.5) with
  ``minimumSize=(640, 460)`` — multi-panel figures get room to
  breathe.
- **GUI: Save action** now correctly avoids overwriting the bundled
  fixture when the GUI was auto-loaded; the first ``Ctrl+S`` falls
  through to ``Save As…`` so the user picks a destination.
- **Report: parallel mode (default).** ``reproduce_csda2013.py``
  now dispatches reps and ICE runs across a
  ``ProcessPoolExecutor`` with ``--jobs N`` (default
  ``cpu_count() // 2``). On a 10-core laptop ``--full`` drops from
  ~3 h to ~30-50 min. Results are bit-identical to the sequential
  path for the same seeds (verified end-to-end with a CSV diff).
- **Report: progress bar + ETA** for every experiment row / ICE
  config. Stdlib-only (no tqdm dependency); TTY rewrites in place,
  non-TTY emits one line per ~10 % rate-limited to ≤ 1 / 3 s.
- **Documentation: dedicated ``prg/pmc/README.md``** that opens
  with both source-paper citations (A16 / A23), a feature ↔ paper
  table, and BibTeX entries.
- **Documentation: GitHub Actions workflow** ``.github/workflows/ci.yml``
  mirroring ``.gitlab-ci.yml`` (lint + matrix tests on Python
  3.11 / 3.12 / 3.13, headless Qt via ``QT_QPA_PLATFORM=offscreen``).

### Changed

- ICE selection criteria expanded from MLE-only to
  ``{mle, aic, bic, huard, cvm}`` for copulas (A16 Eq. 20 implemented
  as ``huard``) and ``{mle, kolmogorov, aic, bic}`` for margins
  (A23 §3).
- Default ``N_default`` lowered from 5000 → 1000 in every shipped
  fixture except ``sp2016_gice_k2.toml`` (which keeps the paper's
  3000), so the auto-loaded default lands on a fast turnaround.
- Top-level README's "References" section rewritten with full
  citations for **both** A16 (CSDA 2013) and **A23** (SP 2016),
  including BibTeX. Fixed pre-existing author typo
  "Piecini" → "Pieczynski"; added ``docs/SP_2016.pdf`` to the
  documented directory tree.
- ``[project.urls]`` in ``pyproject.toml`` gains a
  ``Mirror = https://github.com/SDerrode/copulasformm`` entry.

### Fixed

- ``simulate`` no longer crashes with
  ``ValueError: Probabilities do not sum to 1`` when the model's
  cached ``π`` / ``A`` rows have drifted within the validator's
  ``atol=1e-6`` tolerance but past ``rng.choice``'s tighter
  ``√eps ≈ 1.5e-8`` bound. Defensive renormalisation in
  ``simulate.py``.
- Margin & copula ``QTableWidget`` headers now display state indices
  in italic math (``$i=0$`` / ``$j=0$``) instead of plain ASCII.
- Multi-panel figure layouts (ICE dashboard, K×K small multiples)
  no longer stack their suptitle onto the first row's tick labels.
- Param-comparison view: section titles can no longer overlap the
  table headers (each section now lives in its own ``subgridspec``
  with a dedicated header row).
- ``_draw_tau_trajectories`` no longer emits a ``No artists with
  labels found`` UserWarning on HMC variants whose τ-history is all
  NaN.

### Source-paper attribution

Two papers by S. Derrode and W. Pieczynski:

- **A16** — *Unsupervised data classification using pairwise Markov
  chains with automatic copulas selection*, Comput. Stat. Data Anal.
  63 (2013), 81-98.
  [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
- **A23** — *Unsupervised classification using hidden Markov chain
  with unknown noise copulas and margins*, Signal Process. 128
  (2016), 8-17.
  [doi:10.1016/j.sigpro.2016.03.008](https://doi.org/10.1016/j.sigpro.2016.03.008)

See [`prg/pmc/README.md`](prg/pmc/README.md) for the feature ↔
paper map.

---

## [0.5.0] - 2026-05-04

This is a substantial release covering three audit cycles, a new ICE
diagnostics layer (12 GUI views + animated playback), and a
**breaking-change** clean-up of the SR-PMC marginal-density contract.

Highlights:

- **Breaking** — ``ice()`` now returns ``(PMCModel, IceTrace)`` instead
  of ``(PMCModel, list[float])``. Use ``trace.log_liks`` for the plain
  history.
- **Breaking** — TOML margin schema migrated to K-format (one block per
  state). Legacy K² format is still accepted (with INFO/WARNING).
- **Breaking** — ``PMCModel.margin(i, j)`` now ignores the second
  argument under the SR-PMC contract; ``j`` is optional.

### Breaking — SR-PMC marginals contract enforced

The package now enforces the SR-PMC reversibility identity from CSDA 2013:
the marginal density of an observation depends **only on its current
state**, never on the other state of the bivariate pair. There are
therefore exactly **K marginal densities** in every variant — not K² —
and they are indexed by ``i`` only.

* **TOML schema** — the canonical [[margins]] format now lists K blocks
  (``i``, ``dist``, ``params``); the legacy K² (``i``, ``j``)-keyed
  format is still accepted but:
    * INFO-logged as legacy when entries respect tying;
    * WARN-logged listing each conflicting (i, j) when entries violate
      tying — the (i, 0) anchor is kept as the canonical density.
* **`PMCModel.margin(i, j=None)`** — second argument is now optional and
  ignored under SR-PMC. ``margin(i, 0) is margin(i, 1) is margin(i)``.
* **`PMCModel._state_margins: dict[int, _MarginDist]`** is the new
  internal source of truth (K entries instead of K²); the legacy
  ``_margins[(i, j)]`` mapping is still present for back-compat but all
  entries with the same ``i`` point to the same instance.
* **`PMCModel.margin_blocks()`** always returns the canonical K-format
  (regardless of how the source TOML was written).
* **`PMCModel.weight()` formulas** rewritten in per-state notation
  (``f_i(y_n) · f_j(y_{n+1}) · c_{ij}(F_i(y_n), F_j(y_{n+1}))``).
* **ICE M-step** now fits exactly K marginal densities (one per state)
  weighted by the full ``γ_n(i) = P(X_n=i | Y)``, instead of K² densities
  with partial dual-view weights. Saves K-fold redundant MLE per
  iteration and eliminates the "K replicas drift apart from each other"
  failure mode at finite N.
* **`precompute_weights`** evaluates K margin PDFs/CDFs (vector form)
  and broadcasts to the legacy (N, K, K) tensor shape. Same numerical
  result, K× less work.
* **`simulate._sample_copula_conditional`** uses
  ``model.margin(i).cdf`` and ``model.margin(j).ppf`` (state-only
  indexing) — Rosenblatt inversion is unchanged.
* **TOML model fixtures rewritten** in K-format:
  ``pmc_gauss_k2.toml``, ``pmc_in_gauss_k2.toml``,
  ``hmc_in2_gauss_k2.toml``, ``hmc_dn_gauss_k2.toml``.
  ``hmc_in_gauss_k2.toml`` and ``hmc_in_gauss_k3.toml`` already used the
  K-format.
* **GUI Margins tab** — collapsed from K×K grid to **K cells (one per
  state)**, with explanatory caption. The Copulas tab remains K×K
  (copulas ARE pair-indexed).
* **`Variant.per_class_margin`** is now ``True`` for every variant
  (was: only HMC-IN). Property kept for API stability; it now describes
  a contract, not a per-variant difference.

### Tests added

- ``test_pmcmodel_collapses_legacy_k2_margins`` — tied legacy TOML loads
  with INFO log.
- ``test_pmcmodel_warns_on_legacy_k2_untied_margins`` — untied legacy
  TOML logs a WARNING listing conflicts.
- ``test_pmcmodel_k_format_canonical`` — round-trip and identity of
  ``margin(i, *)`` lookups.
- ``test_ice_preserves_state_margin_tying`` — after ICE,
  ``margin(i, j) is margin(i)`` for every (i, j).

### Refactor (codage audit — third pass)

- **`EXTRA_PARAM_BOUNDS` deduplicated** — single source of truth in
  ``prg.pmc.ice``; the GUI dialog imports it instead of holding a copy.
- **`_BlockGridTab` base class** in ``prg.pmc.gui.tabs`` — Margins and
  Copulas tabs shared 90 % of their code; now each subclass only sets four
  class attributes and overrides ``_format_block`` (~80 LOC saved).
- **`IceResult` dataclass** packages
  ``(initial_model, fitted_model, Y, trace)`` — replaces the worker's
  ad-hoc 4-tuple and shrinks ``PMCMainWindow`` instance state.
- **`_compute_marginal_cdfs(model, Y)`** helper — replaces three
  duplicated ``f_cdf`` K×K loops in ``_do_gof_test``,
  ``_plot_ice_pseudos``, ``_plot_ice_pp``.
- **`_begin_figure()` / `_finalize_plot()` lifecycle helpers** — every
  ``_plot_*`` method now relies on the single canvas-clear and the single
  ``_has_plot``/Export-plot toggle in ``_render_view``. ~30 lines of
  boilerplate eliminated.
- **`_VIEW_DISPATCH` table** replaces the 17-branch ``if/elif`` chain in
  ``_render_view``; adding a view is a one-line entry + a new
  ``_plot_*`` method.
- **`_render_error()` UI** — render failures inside a view now paint a
  friendly message on the canvas and append the exception to the log
  panel (was: silent debug log).
- **Imports hoisted** — ``EPS``, ``ONE_MINUS_EPS``, ``classify``,
  ``error_rate``, ``_empirical_copula``, ``IceTrace`` moved to module
  scope; redundant ``Qt as _Qt`` alias and ``np as _np`` alias removed.
- **Type hints on ICE plot methods** — ``_plot_ice_*`` now declare
  ``trace: IceTrace`` (and ``model: PMCModel`` / ``Y: np.ndarray`` where
  applicable); the diagnostics layer is statically checkable.
- **`IceTrace` type hints tightened** — ``tau_history`` and
  ``p_history`` are now plain ``np.ndarray`` (never ``None``); empty
  traces use ``np.empty((0, 0, 0))``. Docstring gains an ASCII timeline
  diagram explaining snapshot semantics.
- **HMC-IN trace test** verifies ``tau_history`` is all-NaN /
  ``family_history`` all-empty / ``p_history`` sums to 1 for the
  no-copula variant.
- **Top-of-file docstring of ``main_window``** rewritten to describe the
  View-selector architecture and the dispatch flow.

### Added (ICE diagnostics — visual layer)

- **`IceTrace` dataclass** captured per iteration by the ICE driver:
  ``log_liks``, ``tau_history`` ``(T, K, K)``, ``family_history``
  ``(T × K × K SHORT_NAMEs)``, ``p_history`` ``(T, K, K)``,
  ``margin_history`` (per-block params), ``multistart_runs`` (losing
  traces when ``n_starts > 1``), ``run_tag``, ``candidates``.
- **`ice()` now returns ``(PMCModel, IceTrace)``** — backward-incompatible
  signature change. Use ``trace.log_liks`` for the plain LL history.
- **GUI View selector** (combobox above the canvas) — single dispatch
  point for every visualisation; unavailable views are kept in the list
  but disabled with an explanatory tooltip.
- **12 ICE diagnostic views** accessible from the View selector:
  - **A.** τ trajectories per pair (★ markers at family changes)
  - **B.** Family-selection ribbon (one row per pair × T columns,
        candidate-coloured bands)
  - **C.** Pseudo-observations + fitted log-PDF contours, K×K
        small-multiples
  - **D.** Multistart comparison (winner highlighted, losing runs grey)
  - **E.** Joint-prior evolution + ‖Δp‖_F log-y
  - **F.** Margin-parameter evolution (per-block small-multiples)
  - **G.** ICE dashboard (3×2: LL, τ, family, ‖Δp‖, AIC/BIC, summary)
  - **H.** PP plot of fitted copulas (K×K)
  - **I.** Tail dependence λ_L / λ_U heatmap per pair
  - **J.** γ posterior comparison BEFORE vs AFTER ICE
  - **K.** Animated playback — QSlider + Play/Pause button advances
        through iterations, refreshing a 4-panel snapshot view
        (LL / τ / family / p heatmap)
- **CLI / notebook updated** to consume ``IceTrace`` (``trace.log_liks``).

### Added (GUI — second pass)

- **Auto-mirror of SR-PMC `p` cells** — editing `p[i,j]` now mirrors to
  `p[j,i]` live (with an explanatory caption above the table).
- **Modified-indicator (`*`)** in the window title flags unsaved edits;
  reset on Save / Save As / Open / ICE finish.
- **Recent-files submenu** in *File* — last 5 TOML paths persisted via
  `QSettings`; "Clear list" entry to wipe the history.
- **GoF p-value heatmap** — after running GoF, the canvas now shows a
  K×K colour-coded heatmap (red→green, with a 0.05-contour and ‘×’
  markers for errored pairs) in addition to the textual log entry.
- **Per-pair GoF progress** — the determinate progress bar advances one
  step per copula pair (pair tag shown in the status bar).
- **Confirmation before Simulate overwrite** — if `(X, Y)` are already
  loaded, the user is asked before they get overwritten.
- **Auto-scroll log panel** — newly-appended lines are always visible.
- **Pointing-hand cursor** on margin/copula table cells (visual hint
  that double-click opens an editor).
- **Package version in *About*** — reads `prg.__version__`.
- **Export-plot disabled when canvas is empty** — no more accidental
  blank PNG dumps.
- **Symmetrize feedback** — clicking *Symmetrize p* logs
  ``"Symmetrized prior: max asymmetry was 3.4e-04"`` in the log panel
  via the new `_PriorTab.symmetrized(asym, was_renorm)` signal.

### Fixed (GUI — second pass)

- **`closeEvent` waits for the worker** — closing the window during a
  long ICE / GoF run prompts the user (Wait / Terminate / Cancel)
  instead of leaking the `QThread`.
- **`_PriorTab` cell validation** — invalid cells (typos like ``"0.4x"``)
  now raise a typed `PriorTabError` caught by the host, surfaced as a
  friendly QMessageBox; previously they were silently treated as 0.
- **`_Worker.set_progress_cb`** — public setter replaces the previous
  `_kwargs` mutation hack from the host window.
- **Imports cleanup** — `QHBoxLayout/QWidget` now imported at the top of
  `dialogs.py`; `matplotlib.pyplot` imported once at the top of
  `main_window.py` (after `matplotlib.use("QtAgg")`).

### Tests

- 14 new tests in `test_gui_dialogs.py` covering: auto-mirror, prior
  cell validation, symmetrize signal, ICE-tab `changed` signal,
  `_do_gof_test` schema and progress-callback, `_perturb_initial_model`
  non-trivial perturbation, `_read_data_csv` happy path / empty / missing
  Y / non-numeric Y / non-integer X.

### Added (GUI)

- **Multistart controls in the ICE tab** — exposes the four new options
  (`n_starts`, `multistart_seed`, `multistart_jitter`, `patience`).
  Seed/jitter spinboxes auto-disable when `n_starts == 1`.
- **Family-aware copula dialog** — the τ spinbox is reranged dynamically
  to match the selected family's `TAU_MIN_MAX`, and extra-parameter
  widgets (BB1 `δ`, Student `df`) appear only when the chosen family
  declares them. Eliminates the silent τ-clip footgun.
- **Robust margin parameters parser** — accepts both spaces and commas
  as separators (`"loc=0, scale=1"` works as well as `"loc=0 scale=1"`),
  flags unparseable tokens via a friendly `QMessageBox` instead of
  silently dropping them.
- **Symmetrize-p button** in the Prior tab — one-click enforcement of
  `p[i,j] = p[j,i]` for SR-PMC joint priors (visible only for those
  variants).
- **GoF test action** — a new button runs the Cramér-von Mises GoF
  (parametric bootstrap, `B=200`) on every pair-copula and reports
  per-pair p-values in a dialog and the log panel.
- **File ▸ Save data (CSV)** — saves the current `(X, Y)` sequence,
  symmetric with *Load data*.
- **File ▸ Export plot…** — writes the matplotlib canvas to PNG/PDF/SVG.
- **Edit ▸ Reset session (Ctrl+R)** — clears loaded data, canvas, and log.
- **Determinate ICE progress bar** — `ice()` now accepts
  `progress_cb=callable`. The GUI worker connects this to a Qt signal so
  the bar advances iteration-by-iteration, with per-run tagging when
  multistart is on (`ICE [perturbed-1] iter 3/8 …`).

### Fixed (GUI)

- `_load_model` no longer commits `_model_path` until `PMCModel()`
  succeeds — failed loads leave the GUI consistent.
- `_on_load_data` now validates that every `Y` cell parses as a float
  and reports the offending row number; non-integer `X` cells are
  ignored with a warning rather than crashing.
- Removed dead code (`_IceTab._cop_visible` was set and never read;
  unused `mdl` parameter on `_MarginTab._refresh_table`).

### Added

- **ICE multistart** — `ice()` now accepts `n_starts` (default `1`),
  `multistart_seed` and `multistart_jitter` config keys. With
  `n_starts > 1` the driver runs ICE several times from random
  perturbations of the initial model and returns the run with the highest
  final log-likelihood. The first start is always the unperturbed model,
  so the legacy single-start behaviour is preserved by default.
  Hardens fits against multimodal log-likelihoods (Archimedean copula
  selection, HMC variants with similar margins).
- **Data-aware init in `_fit_copula_params`** — weighted Kendall's τ on
  the pseudo-observations is now computed once and used (a) as the
  initial value for the multi-parameter L-BFGS-B (BB1, Student) and
  (b) as the fallback when the 1-D `minimize_scalar` path errors.
  Replaces the previous mid-range / Pearson-based fallbacks.
- **Boundary tests for Archimedean copulas** — `prg/tests/test_copulas.py`
  now exercises every Archimedean family (GH, Clayton, Joe, A12, A14,
  AMH, Frank, BB1, plus the three Survival rotations) at τ_K = τ_min +
  0.95·span and τ_min + 0.05·span. Verifies (i) finite PDF / CDF /
  conditional CDF, (ii) `correct_tau` clips out-of-range τ with a
  WARNING, (iii) `correct_tau` handles NaN, (iv) `logpdf_array` does
  not silently saturate near `-36`, (v) `inv_h_array` returns values
  strictly inside (0, 1).

### Fixed

- **`CopulaA12.conditional_cdf`** now traps `OverflowError` /
  `ZeroDivisionError` and returns the diagonal limit `𝟙{v ≥ u}` (same
  pattern as A14). Previously raised for τ ≳ 0.85 when probed at
  v = EPS, which made `inv_h(brentq)` fail near the boundary. Surfaced
  by the new boundary tests.

### Changed

- **`CopulaVirt.sample` uses Rosenblatt inversion via `inv_h`** instead
  of inlined Brent search. Combined with the new vectorised
  `inv_h_array` (Gaussian, Clayton, Frank), this gives:
    - Gaussian sample(2000): **1.67 s → 0.001 s** (~1500×)
    - Clayton  sample(2000): **0.21 s → 0.001 s** (~200×)
    - Frank    sample(2000): **~50× faster**
- **`BivariateLaw.sample_conditional` uses Rosenblatt** instead of
  acceptance-rejection. Removes the `MAX_AR_ITER` failure mode for
  high-correlation cases and gives O(1) per sample for closed-form-`inv_h`
  copulas. The `MAX_AR_ITER` constant and `SamplingConvergenceError`
  exception are kept exported for backward compatibility.
- **`CopulaVirt.fit(method='mle')` for multi-parameter families** —
  the base-class implementation now does joint L-BFGS-B over (τ, *extras)
  for any family declared with extra parameters in `CopulaEnum`. Student
  and BB1 keep their existing per-class overrides; the base path is the
  fallback for any future 2-parameter family.
- **`CopulaEnum.correct_tau` warns on clipping** — when the input τ is
  outside the family's valid range or non-finite, a `WARNING` is logged
  with the original and clipped values.
- **Bootstrap progress logging** — `gof_test`, `bootstrap_ci`,
  `BivariateFitResult.gof_test` and `bootstrap_ci` now emit a
  `logger.info(...)` at every 10 % of progress for B ≥ 10.

### Added

- **`CopulaVirt.inv_h_array(w, u)`** — vectorised inverse h-function.
  Default fallback is a scalar Python loop; closed-form vectorised
  overrides on Gaussian, Clayton, Frank.
- **Native `logpdf_array` overrides** on six additional copulas:
  AMH, A12, A14, FGM, CubSec, Plackett. Combined with the existing five
  (Gaussian, Clayton, Frank, Joe, BB1) and the three Survival wrappers
  that delegate to their base, **14/17 copulas now have a native
  log-PDF** with extended dynamic range.

- **Native `logpdf_array` overrides** on the five most-used copulas:
  Gaussian, Clayton, Frank, Joe, BB1. These compute log c(u, v) directly
  in the log domain, bypassing the ``log(max(pdf, EPS))`` floor of the
  base implementation. Dynamic range extends below −36 nats (the
  saturation point of the floored path), useful for tails and
  goodness-of-fit on small-density regions.
- **`prg/tests/test_pdf_array.py::test_logpdf_array_native_extends_dynamic_range`**
  — verifies that for the 5 native overrides, (a) the regular range
  matches the floored path to 1e-9 and (b) at least one extreme-tail
  point goes strictly below the floored saturation.
- **SR-PMC dual-view margin update in ICE** — for pair-indexed margins
  (`HMC-IN2`, `HMC-DN`, `PMC-IN`, `PMC`), each `Y[k]` now contributes to
  the weighted MLE of `f_{ij}` from BOTH its "first observation" view
  (weighted by `ξ[k, i, j]`) and its "second observation" view of the
  reversed pair (weighted by `ξ[k-1, j, i]`). Effectively doubles the
  sample size used for each margin, giving ~√2× tighter parameter
  estimates without violating the SR-PMC factorisation.
- **SR-PMC stationary-reversibility diagnostic** — `PMCModel` warns at
  construction if `[prior].p` is not symmetric (max |p[i,j] − p[j,i]| ≥
  1e-6). The package commits to the SR-PMC factorisation; non-symmetric
  `p` matrices give a non-reversible chain and the inference may
  produce statistically dubious results.
- **`error_rate` symmetric warning** — also warns when `X̂` uses labels
  *outside* the true set (typical of a mis-specified K), in addition to
  the existing warning when `X̂` collapses to fewer classes.

### Fixed

- **🔴 ICE `fit_margins=True` no longer overwrites the declared margin family.**
  The previous `_fit_gaussian_margin_weighted` always returned
  `dist="norm"`, silently replacing any non-Gaussian margin (`expon`,
  `lognorm`, `gamma`, …) with a Gaussian. The new `_fit_margin_weighted`:

  - keeps the closed-form Gaussian fast path,
  - dispatches to a generic `scipy.optimize.minimize` (L-BFGS-B) on the
    weighted negative log-likelihood for any other `scipy.stats` family,
  - preserves the `dist` field — only `params` are updated,
  - logs a warning and keeps initial parameters on optimiser failure.

  Verified on `(norm, lognorm)` mixed-margin model: families preserved,
  parameters recovered to within ~10 % on N=2000.

- **🟡 `precompute_weights` (and therefore `forward`, `classify`, `ice`)
  now reject `N < 2`** with a clear error message instead of crashing
  on a cryptic NumPy ``zero-size array reduction`` deep inside the
  call stack.

- **🟡 `CopulaVirt.majorant` is now a guaranteed upper bound** on
  `max_v c(u, v)`. The default has been replaced by a hybrid:
  150-point grid → 1-D scalar refinement → ×1.02 safety multiplier.
  In addition, `CopulaGaussian.majorant` and `CopulaClayton.majorant`
  now use closed-form expressions and return exact bounds.

  Empirical deficit (`true_max / bound − 1`) across 11 families went
  from up to **0.17 %** (Clayton τ=0.5) down to **0**; the AR sampler
  in `BivariateLaw.sample_conditional` is now unconditionally unbiased.

### Added

- **`prg/tests/test_majorant.py`** — 17 tests (one per copula family)
  verifying `cop.majorant(u) ≥ max_v c(u, v)` against a high-precision
  reference computed via `scipy.optimize.minimize_scalar`.
- **`prg/tests/test_pmc.py::test_ice_fit_margins_preserves_family`** —
  regression test for the silent-overwrite fix.

- **`prg/tests/test_forward_backward_invariants.py`** — 36 tests
  (6 invariants × 6 demo models) verifying the probability-conservation
  laws of the forward-backward algorithm:
  - `Σ_j α̂_n(j) = 1` (filtered posterior),
  - `Σ_j β̂_n(j) = 1` (Devijver-normalised backward),
  - `Σ_j γ_n(j) = 1` (smoothed posterior),
  - `Σ_{i,j} ξ_n(i, j) = 1` (joint pair posterior),
  - `Σ_j ξ_n(i, j) = γ_n(i)` (left-marginal consistency),
  - `Σ_i ξ_n(i, j) = γ_{n+1}(j)` (right-marginal consistency).

  Observed deviations are at machine epsilon (~2e-16) on all six demo
  models. These invariants would have caught the PMC kernel bug fixed
  earlier in this release.

- **`CopulaVirt.cdf_array(uv)`** — vectorised CDF on an `(M, 2)` array,
  same fast-path strategy as `pdf_array`. Closed-form overrides for FGM,
  AMH, A12, A14, BB1, CubSec, Plackett, Joe, and the three Survival
  wrappers. Used by the GoF bootstrap.
- **`prg/tests/test_pdf_array.py::test_cdf_array_matches_scalar`** —
  parametrised over all 17 copulas; Student is correctly skipped (no
  closed-form CDF).
- **`logger.warning`** in `prg.pmc.inference.error_rate` when X̂ uses
  fewer classes than X_true (typical symptom of a degenerate ICE solution).
- **Sanity assertion** in `CopulaVirt.inv_h` — verifies that
  `conditional_cdf` is monotone in v before launching Brent. Catches
  buggy custom copulas with a clear message instead of an opaque
  `bracket bracket` error from scipy.

### Changed

- **`prg.copulas._fit._eval_log_likelihood`** uses the vectorised
  `logpdf_array` (≈ **300×** faster on N=5000 — was 0.30 s, now 0.001 s).
- **`prg.copulas._fit._cvm_statistic`** uses `cdf_array` (~5× faster on
  the CDF-evaluation step; the `(n×n)` empirical-copula matrix remains
  the dominant cost for large N).
- **`_class_colors(n_classes)`** indexes the `tab10` palette modulo 10
  rather than dividing the discrete index. K ≤ 10 callers get the
  canonical colours; K > 10 callers cycle (rather than getting
  interpolated intermediates).
- **`BivariateFitResult.cv_loglik`** docstring documents the choice of
  −100 nats as the support-violation floor (vs `log(MIN_POSITIVE) ≈ -708`).

### Fixed

- **🔴 Critical: `prg.pmc.inference.precompute_weights` for PMC and PMC-IN
  variants double-counted the marginal density `f_{ij}(y_n)`** at every
  recursion step. The transition kernel was the *joint* pair density rather
  than the *conditional* kernel. As a consequence:
  - the reported log-likelihood was off by an additive Y-dependent term
    (≈ -2 nats per step on the demo models);
  - smoothed posteriors `γ_n(j)` and joint posteriors `ξ_n(i, j)` were
    biased — by up to 0.6 in extreme cases;
  - MPM error rate was degraded by 1–3 percentage points;
  - ICE optimised a wrong objective and reported wrong AIC/BIC.

  The fix divides the joint kernel by `D[n, i] = Σ_{j'} p[i, j'] · f_{ij'}(y_n)`
  (the marginal `p(X_n=i, Y_n=y_n)`), recovering the correct conditional
  transition kernel. After the fix:
  - all 5 variants match brute-force enumeration to **machine precision**
    (1e-14 on K=2, N=5);
  - PMC-IN now coincides with HMC-IN2 when `p[i,j] = π_i · A[i,j]` (as
    expected mathematically);
  - PMC classification error on `pmc_gauss_k2` drops from **10.7 % to 7.7 %**.

### Added

- **`prg/tests/test_forward_correctness.py`** — 19 tests guarding the
  Devijver kernel against future regressions:
  - 18 parametrised cases (5 demo models + K=3 × 3 seeds) compare
    `forward()` to brute-force enumeration of all `K^(N+1)` state
    sequences (N=5);
  - one test verifies that PMC-IN and HMC-IN2 give identical log-lik
    when `p[i,j] = π_i · A[i,j]` and margins coincide.

- **`examples/quickstart.ipynb`** — end-to-end demo notebook (25 cells, 14
  code) covering: copula PDF / fitting / sampling, vectorised `pdf_array`
  benchmark, Sklar bivariate law, PMC simulation → classification → ICE,
  K=3 example.
- **`prg/tests/test_quickstart_notebook.py`** — `slow`-marked regression
  test that re-runs every cell with a fresh kernel; fails on any cell
  exception.

- **`prg/pmc/models/hmc_in_gauss_k3.toml`** — first K=3 example model (three
  well-separated Gaussian regimes, sticky transitions). All `test_pmc.py`
  parametrised tests pick it up automatically; a dedicated
  `test_classify_hmc_in_k3` verifies error rate < 5 %.
- **`prg/tests/test_logging_setup.py`** — 5 tests covering `configure()`
  (no-file, explicit-path, idempotence, unwriteable-path) and
  `add_widget_handler()` (PyQt6 round-trip).  Coverage of
  `logging_setup.py`: 27 % → **95 %**.
- **`CSVLoadError`** in `prg.exceptions` (replaces the private
  `_DataLoadError` from `cli.py`). Now reusable from library code; keeps
  the same hierarchy (`PMCError, IOError`).

### Changed

- **`MAX_AR_ITER`** in `prg.copulas.bivariate` — renamed from `_NBITERMAX`,
  now public and documented (Sphinx-compatible `#:` comments). Tweak via
  `prg.copulas.bivariate.MAX_AR_ITER = …` if a custom copula has a poor
  majorant.
- **`CopulaEnum` access aligned** — every iteration now uses
  `c.value.X` (CLASS_NAME, MODULE, AVAILABLE, …) instead of mixing
  `c.X` and `c.value.X`. The dataclass mixin still accepts both, but the
  codebase is internally consistent.
- **`_class_colors(K)` → `_class_colors(n_classes)`** — PEP-8 compliant
  parameter name. The `import matplotlib.pyplot as plt` was hoisted out
  of the function body (lazy import was unnecessary in a Qt module).

### Added

- **`Variant` enum properties** — `variant.uses_copula`, `variant.has_markov_prior`,
  `variant.per_class_margin`. The legacy `USES_COPULA` / `MARKOV_PRIOR` /
  `PER_CLASS_MARGIN` sets are kept as backward-compatible aliases.
- **`prg/tests/test_cli.py`** — 7 subprocess-based smoke tests for the
  ``pmc`` console script (``--help``, simulate, classify, estimate;
  reproducibility, missing-data error path).
- **`pytest.mark.slow`** — marker registered in pyproject; CLI estimate
  (the longest single test at ~2 s) is tagged. Run `pytest -m "not slow"`
  for fast iteration.
- **CI updated** (`.gitlab-ci.yml`) — Python 3.11 / 3.12 / 3.13 matrix,
  ruff lint stage, coverage gauge, Cobertura report artefact, headless
  Qt deps installed for `test_gui_dialogs`.

### Changed

- **`prg/__init__.py`** docstring — replaced the obsolete "Public API
  (to be defined)" placeholder with three concrete quickstart examples
  (copula, PMC, CLI).

### Added

- **`CopulaVirt.inv_h(w, u)`** — Rosenblatt inverse step (used by `simulate`).
  Default uses Brent's method; `CopulaGaussian` and `CopulaClayton` override
  with closed-form expressions (~50× faster).
- **`[project.scripts] pmc`** — installs a `pmc` console script
  (`pmc simulate ...`, `pmc classify ...`, `pmc estimate ...`, `pmc gui`).
- **`[project.optional-dependencies] gui`** — `pip install copulasformm[gui]`
  pulls PyQt6.
- **`[tool.coverage]` config** — `pytest --cov` runs out of the box.
- **`prg/copulas/_fit.py`** — extracted from `_base.py`: `FitResult`,
  `GoFResult`, `_eval_log_likelihood`, `_empirical_copula`, `_cvm_statistic`,
  `_empirical_tail_dep`. `_base.py` re-exports for backward compatibility.
- **`prg/pmc/gui/dialogs.py`** — extracted `_MarginDialog` and `_CopulaDialog`
  from `main_window.py`. Now unit-testable in isolation.
- **`prg/pmc/gui/tabs.py`** — extracted the four parameter tabs (`_PriorTab`,
  `_MarginTab`, `_CopulaTab`, `_IceTab`) from `main_window.py`.
- **`prg/pmc/gui/worker.py`** — extracted `_Worker` (background `QThread`).
- **`prg/copulas/_bivariate_fit.py`** — extracted `BivariateFitResult`,
  `BivariateBootstrapCI`, `_empirical_joint_cdf` and `_joint_cvm_statistic`
  from `bivariate.py`. The runtime cycle is broken by lazy imports of
  `BivariateLaw` inside the bootstrap / cv methods.
- **`CopulaFrank.inv_h`** — closed-form override (Aas et al. 2009 derivation):
    v = -1/θ · log(1 + w(e^{-θ}-1) / (e^{-θu} - w(e^{-θu}-1)))
- **`prg/tests/test_gui_dialogs.py`** — 7 unit tests for `_MarginDialog` and
  `_CopulaDialog` (round-trip, defaults, parameter parsing, full copula list).
  Skipped gracefully when PyQt6 is not installed.
- **`prg/tests/test_pdf_array.py`** — 36 new tests verifying that every
  copula's `pdf_array` matches `[pdf([u,v]) for u,v in uv]` and that
  `inv_h` is a true inverse of `conditional_cdf` for Gaussian and Clayton.

### Changed

- **`pyproject.toml`** — multiple correctness fixes:
  - `requires-python = ">=3.11"` (was `>=3.10`; we use `tomllib` from stdlib).
  - Removed `pandas`, `rich` from dependencies (never imported).
  - Added `statsmodels>=0.14` (used by 5 copulas, was undeclared → install
    on a clean env was broken).
  - Classifier list updated to reflect 3.11 / 3.12 / 3.13.
- **`prg.pmc.simulate`** — uses the new `cop.inv_h(w, u)` instead of an
  in-module `brentq` call. The two `simulate(N=10_000)` tests went from
  8.1 s → 1.2 s each; total test suite **22.78 s → 7.25 s**.
- **`prg.pmc.ice._joint_posteriors`** — vectorised via numpy broadcasting;
  removed the `for n in range(N-1)` loop.
- **`BIGGER_SIZE` → `FONT_SIZE`** across the package (the legacy alias was
  misleading; the rename clarifies intent).
- **Logger placement** — moved `logger = logging.getLogger(__name__)` after
  the imports in `prg.pmc.gui.main_window` (was the only file with it before
  imports).
- **`scripts/update_readme_structure.sh`** — moved from the repo root to
  `scripts/` and the in-script usage doc updated.

### Removed

- **`prg.pmc.ice.ice_from_files`** — dead helper (61 LOC), never called from
  anywhere. The CLI uses `cmd_estimate` directly.
- **`prg/tests/__init__.py`** and **`prg/tests/conftest.py`** (empty).
- **`prg.pmc.simulate._inv_h`** — superseded by `CopulaVirt.inv_h`.

### Added (earlier in this cycle, kept here for completeness)

- **`CopulaVirt.pdf_array(uv)` / `logpdf_array(uv)`** — vectorised PDF / log-PDF
  evaluation on an `(M, 2)` array of pseudo-observations. Default fast path
  uses `self._model.pdf` (statsmodels) when available; falls back to a Python
  loop otherwise.
- **`CopulaJoe.pdf_array`** — vectorised closed-form override (Joe has no
  statsmodels backend).
- **`PMCModel` public read API** — `raw`, `path`, `ice_config()`,
  `margin_blocks()`, `copula_blocks()`. All return deep copies — safe to
  mutate. External callers (ICE, GUI) no longer reach into `_raw`.
- **`prg.configure_logging`** re-exported at the top-level package.
- **`prg/tests/test_pmc.py`** — 39 new tests covering model loading,
  prior consistency, weight() formula, save/load round-trip, simulate()
  empirical π, forward/backward invariants, classify() error rate,
  K=3 label permutations, and ICE convergence + τ recovery.

### Changed

- **`prg.pmc.inference.precompute_weights`** — vectorised:
  - HMC-IN/IN2/PMC-IN: full numpy broadcast (no Python loop on N).
  - HMC-DN/PMC: the `(N-1)` Python loop on copula PDFs is replaced by a
    single `cop.pdf_array(uv)` call per `(i, j)` pair.
  - **Speedup**: classify on PMC/HMC-DN, N=5000 → ≈55× (1.2 s → 0.02 s).
- **`prg.pmc.ice._neg_wll` and `_weighted_log_likelihood`** — use
  `cop.logpdf_array(uv)` instead of a Python list comprehension over scalars.
  - **Speedup**: ICE 5 iterations on PMC, N=2000 → ≈130× (27 s → 0.2 s).
- **`prg.pmc.inference.error_rate`** — handles arbitrary K via the
  Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on the
  confusion matrix; returns the minimum error over all label permutations.
- **`prg.pmc.model.PMCModel.save`** — uses `tomli_w` instead of a custom
  TOML serialiser (~50 LOC removed).
- **`PMCModel.from_dict(raw)`** now deep-copies `raw`, eliminating the
  aliasing pitfall where mutating the source dict silently affected the model.

- **`PMCModel._parse`** — exhaustive `[[margins]]` and `[[copulas]]`
  validation: detects duplicate `(i, j)` keys, missing pairs, and missing
  required fields (`dist`, `i`, `j`). 4 new tests cover these paths.
- **`prg.pmc.ice.ice()`** — distinguishes log-likelihood **regression** from
  convergence: emits a `WARNING` on each decreasing step and stops early
  after `patience` (default 3) consecutive regressions.
- **`prg.copulas._base.CopulaEnum.klass`** — cached lazy property that
  returns the implementation class. Replaces repeated
  `importlib.import_module` calls in `_build_copula` and ICE candidate
  resolution.
- **GUI** — added a **Seed** field to the model header. Empty = random;
  any integer = reproducible simulation. Parsing is tolerant: a non-integer
  value logs a warning and falls back to ``None``.
- **`prg.pmc.gui._MarginDialog` / `_CopulaDialog`** — now inherit directly
  from `QDialog` instead of wrapping a contained `QDialog` in a `QWidget`.
- **CLI** — extracted `_read_observations(path, ref_col)` helper; removed
  duplicated CSV-reading code from `cmd_classify` and `cmd_estimate`.

- **Vectorised `pdf_array` overrides** for every copula without a
  `statsmodels` backend: `BB1`, `AMH`, `A12`, `A14`, `FGM`, `Plackett`,
  `CubSec`, and the three `Survival*` wrappers. Bench against the scalar
  loop on N=2000 random pseudo-observations: speedups from **14× to 100×**;
  numerical match within 1 ulp on every family.
- **`prg/tools/tools.py`** — added `MIN_POSITIVE = sys.float_info.min`
  (smallest strictly-positive normal float, used as a safe non-zero floor
  in normalisation). Previously `1e-300` was hardcoded in five places.
- **`prg.pmc.inference.forward`** — when α₁ has zero density under every
  state (e.g. Y[0] outside every margin's support), the function now
  raises `IncompatibleObservationError` instead of silently flooring
  `C₁` to `1e-300` and producing a meaningless log-likelihood. Test
  added.
- **`prg.pmc.model._stationary_distribution`** — replaces the
  `np.linalg.eig`-based extraction of π with **power iteration**. The
  previous approach could pick the wrong eigenvector for nearly-reducible
  chains (multiple eigenvalues close to 1). Test added on a near-reducible
  3-state chain.
- **`prg.pmc.ice._fit_copula_params`** — multi-parameter copula fitting
  via L-BFGS-B. BB1 (`tau_k`, `delta`) and Student (`tau_k`, `df`) now
  jointly maximise the weighted log-likelihood instead of using the
  hardcoded defaults `delta = 1.5` / `df = 4.0`.

### Removed

- **`prg/logging_config.py`** — superseded by `prg/pmc/logging_setup.py`
  (unused since 0.4.0).

### Dependencies

- Added **`tomli_w>=1.0`** (TOML serialisation).

---

## [0.4.0] - 2026-05-04

### Added

- **`prg/pmc/`** — full Pairwise Markov Chain (PMC) package with 5 model variants:
  - **`HMC-IN`** — Hidden Markov Chain, one marginal per class (classical HMM).
  - **`HMC-IN2`** — HMC with pair-indexed marginals f_{ij}.
  - **`HMC-DN`** — HMC with copula-based temporal dependence.
  - **`PMC-IN`** — Pairwise Markov Chain, independent observations.
  - **`PMC`** — Full PMC with copula-based transitions.

- **`PMCModel`** (`prg/pmc/model.py`) — loads/validates/saves models from TOML.
  - Supports all 5 variants; validates K, prior (A or p), K² margins, K² copulas.
  - `prior_p`, `transition_A`, `stationary_pi` properties.
  - `pdf(i,j,y)`, `cdf(i,j,y)`, `ppf(i,j,q)` — margin access.
  - `copula(i,j)` — returns `CopulaVirt` instance.
  - `weight(i,j,y_n,y_{n+1})` — unified transition weight for all 5 variants.
  - Custom TOML writer (no `tomli_w` dependency).

- **`simulate`** (`prg/pmc/simulate.py`) — sequence generator for all 5 variants.
  - Latent chain sampled from row-stochastic A; observations via Rosenblatt h-inversion for copula variants.
  - `simulate(model, N=None, seed=None) → (X, Y)`.

- **`classify`** / **`forward`** / **`backward`** / **`smooth`** / **`mpm`** (`prg/pmc/inference.py`) — Baum-Welch forward-backward with Devijver normalization.
  - Initialization: α_1(j) = Σ_i p[i,j]·f_{ji}(y_1) — works for all 5 variants.
  - `classify(model, Y) → (X_hat, gamma, log_lik)`.
  - `error_rate(X_true, X_hat)` — handles K=2 label permutation.

- **`ice`** (`prg/pmc/ice.py`) — Iterative Conditional Estimation (ICE) for unsupervised parameter fitting.
  - E-step: forward-backward → marginal γ and joint ξ posteriors.
  - M-step: prior update, copula selection by weighted log-likelihood, τ by weighted MLE.
  - Optional margin re-estimation (Gaussian: weighted mean/std).
  - Convergence by relative log-likelihood change.

- **CLI** (`prg/pmc/cli.py`, `prg/pmc/__main__.py`):
  - `python -m prg.pmc simulate --model M.toml --N 5000 --seed 42 --out seq.csv`
  - `python -m prg.pmc classify --model M.toml --data seq.csv --out cls.csv`
  - `python -m prg.pmc estimate --model INIT.toml --data seq.csv --out fitted.toml`
  - `python -m prg.pmc gui [M.toml]`

- **PyQt6 GUI** (`prg/pmc/gui/`):
  - Single window with three action buttons: **Simulate**, **Classify**, **Estimate**.
  - Tabbed model editor: Prior (K×K editable matrix), Margins (K×K cells), Copulas (K×K cells), ICE Config.
  - Double-click cell dialogs for per-pair margin and copula editing.
  - Matplotlib result panel: scatter + histogram + pair-scatter for simulation; posterior + error map for classification; log-lik convergence for ICE.
  - Background QThread worker keeps GUI responsive during long computations.
  - File menu: Open / Save / Save As TOML; Load data CSV.

- **5 example TOML models** (`prg/pmc/models/`):
  - `hmc_in_gauss_k2.toml`, `hmc_in2_gauss_k2.toml`, `hmc_dn_gauss_k2.toml`
  - `pmc_in_gauss_k2.toml`, `pmc_gauss_k2.toml`

---

## [0.3.5] - 2026-05-04

### Changed

- **`CopulaStudent`** (`prg/copulas/elliptical/student.py`) — `df` (degrees of freedom ν) promoted from a hardcoded constant (4) to a free parameter.
  - `n_params = 2`; `PARAMETERS_SET_NAME = ["tau_k", "df"]` in `CopulaEnum`.
  - `CopulaStudent(tau_k=τ)` still works — `df` defaults to 4.0 if omitted (backward-compatible).
  - `df > 2` enforced in `_update_params` (raises `CopulaParameterError` otherwise).
  - `df` persisted back to `self.params['df']` so `plot_multi_tau`, bootstrap CI, and GoF bootstraps preserve it across round-trips.
  - **2-parameter MLE** — `CopulaStudent.fit` overrides the base class and jointly optimises (ρ, ν) via L-BFGS-B with bounds `ρ ∈ (−1, 1)`, `ν ∈ (2, ∞)`. `method='tau'` is accepted for API compatibility and warns before falling back.
  - Numerical guard added to `pdf` (errstate + isfinite/positive check → fallback `EPS`).
  - Tail dependence λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) now uses the fitted ν; correctly → 0 as ν → ∞ (recovers Gaussian).

---

## [0.3.4] - 2026-05-04

### Added

- **`CopulaAMH`** (`prg/copulas/archimedean/amh.py`) — Ali-Mikhail-Haq copula. Generator φ(t) = log((1−θ(1−t))/t), θ ∈ [−1, 1). Closed-form CDF C=uv/W, PDF c=[1−θ(2−u−v−uv)+θ²(1−u)(1−v)]/W³, analytical h-function h(v|u)=v(1−θ(1−v))/W². τ_K = 1−2[θ+(1−θ)²log(1−θ)]/(3θ²), τ range ≈ [−0.182, 1/3). λ_L = λ_U = 0. Numerically stable τ→0 limit via Taylor series (avoids 0/0 at θ=0).
- **`CopulaPlackett`** (`prg/copulas/explicit/plackett.py`) — Plackett copula (constant cross-product ratio θ>0). Explicit CDF C=[S−√Δ]/(2(θ−1)) with S=1+(θ−1)(u+v), Δ=S²−4θ(θ−1)uv; θ=1→independence (uv). PDF c=θ[1+(θ−1)(u+v−2uv)]/Δ^{3/2}. Analytical h-function h(v|u)=[√Δ−(S−2θv)]/(2√Δ). τ_K=(θ+1)/(θ−1)−2θlogθ/(θ−1)², full range (−1,1). λ_L = λ_U = 0. Numerically stable τ→0 limit via Taylor at θ=1.
- **`CopulaEnum.AMH`** (ID 16) and **`CopulaEnum.PLACKETT`** (ID 17); auto-imported and picked up by `fit_best`.
- **`_AMH_TAU_MIN`** / **`_AMH_TAU_MAX`** constants in `_base.py` for the AMH τ range (computed at import time from (5−8ln2)/3 and 1/3−ε).

---

## [0.3.3] - 2026-05-04

### Added

- **`CopulaBB1`** (`prg/copulas/archimedean/bb1.py`) — BB1 (Joe-Clayton) copula with **both** lower- and upper-tail dependence. Generator φ(t) = (t^{−θ}−1)^δ, θ > 0, δ ≥ 1. Closed-form τ_K = 1 − 2/(δ(θ+2)), λ_L = 2^{−1/(θδ)}, λ_U = 2 − 2^{1/δ}. Special case δ = 1 reduces to Clayton (verified analytically).
- **2-parameter fitting** — `CopulaBB1.fit` always uses 2-D L-BFGS-B MLE over (θ, δ); `method='tau'` is accepted for API compatibility (falls back to MLE with a warning). `n_params = 2` so AIC/BIC correctly penalise the extra parameter.
- **`CopulaEnum.BB1`** (ID 15) with `PARAMETERS_SET_NAME = ["tau_k", "delta"]`; picked up automatically by `fit_best` and `__init__.py` auto-import.

### Fixed

- **`CopulaVirt.plot_multi_tau`** — replaced `self.__class__(tau_k=float(tau))` with `self.__class__(**{**self.params, 'tau_k': float(tau)})` so extra parameters (e.g., BB1's `delta`) are preserved when sweeping τ.

---

## [0.3.2] - 2026-05-04

### Added

- **`CopulaFrank`** — enabled (`AVAILABLE=True`); τ↔θ inversion now numerically stable for the full range τ ∈ (−1, 1) including high |τ| (tested up to ±0.95). Analytical `conditional_cdf` and `tail_dependence` (λ_L = λ_U = 0). Robustness guards (`minmaxEPS`, `np.errstate`, `isfinite`) on `pdf`/`cdf`.
- **`CopulaJoe`** (`prg/copulas/archimedean/joe.py`) — new Archimedean family with upper-tail dependence only. Generator φ(t) = −log(1−(1−t)^θ), θ ≥ 1. τ↔θ via a fast convergent series (2000 terms, O(1/k³)). Analytical `pdf`, `cdf`, `conditional_cdf`, `tail_dependence` (λ_L = 0, λ_U = 2 − 2^{1/θ}). Full numerical guard suite.
- **`SurvivalCopula`** base class + **`SurvivalClayton`**, **`SurvivalGH`**, **`SurvivalJoe`** (`prg/copulas/archimedean/survival.py`) — 180° rotation wrappers: ĉ(u,v) = c(1−u,1−v), ĥ(v|u) = 1 − h(1−v|1−u), (λ_L, λ_U) swapped. τ_K preserved under rotation.
- **`CopulaEnum`** — 4 new entries: `JOE` (ID 11), `SURVIVAL_CLAYTON` (12), `SURVIVAL_GH` (13), `SURVIVAL_JOE` (14). Auto-import and `fit_best` pick them up automatically via the existing enum-driven mechanism.

### Fixed

- **`kendall_tau_frank`** — replaced the `1 − D₁(θ)` formula with a direct integration of `1 − t/(e^t − 1)`, which vanishes at t = 0 and eliminates catastrophic float cancellation for small |θ| (previously τ(10⁻⁹) returned 1.0 instead of ≈ 0).
- **`find_theta_frank`** — bracket widened from the hardcoded `(1e-5, 50)` (failed for |τ| ≥ 0.93) to dynamic `(±1e-9, ±500)` covering |τ| up to ≈ 0.992; τ = 0 handled as a special case returning θ = 0 directly.

---

## [0.3.1] - 2026-05-04

### Added

- **`CopulaEnum.MODULE`** — champ `MODULE: str` dans `CopulaDataMixin` ; chaque entrée de `CopulaEnum` porte le chemin pointillé de son module (ex. `"prg.copulas.elliptical.gaussian"`).
- **Auto-import piloté par l'enum** — `prg/copulas/__init__.py` boucle sur `CopulaEnum` pour importer et exposer toutes les familles ; `__all__` est généré depuis l'enum. Ajouter une nouvelle copule ne nécessite plus de toucher `__init__.py`.
- **`fit_best` enum-driven** — `CopulaVirt.fit_best` et `BivariateLaw.fit_best` résolvent leurs familles par défaut via `CopulaEnum` + `importlib` (plus d'import en dur).
- **Logging sur les fallbacks numériques** — `a12.py`, `a14.py` et `clayton.py` émettent un `logger.debug(...)` à chaque retour de secours (EPS, 0 ou 1) pour traçabilité.

### Fixed

- **`CopulaA14.pdf`** — `ZeroDivisionError` silencieux quand `u` ou `v` = `ONE_MINUS_EPS` avec θ > 2 (τ > 0,67) : `pow(u, 1/θ) − 1` s'annule en float64. Protégé par `try/except` → retourne `EPS`.
- **`CopulaA14.conditional_cdf`** — même cause (`S^{1/θ−1}` avec S → 0). Protégé par `try/except` → retourne 1 si v > 0,5, sinon 0.
- **`CopulaClayton.pdf / cdf`** — statsmodels retournait `nan` pour `u` = `EPS` et θ grand. Entrées clampées par `minmaxEPS`, calcul sous `np.errstate`, résultat vérifié par `isfinite`.
- **`CopulaClayton.conditional_cdf`** — `inf × 0 = nan` pour `u` → 0⁺. Garde `isfinite` ajoutée → retourne 1 (dépendance de queue inférieure).
- **`set_dir` supprimée** de `prg/tools/tools.py` ; remplacée par `pathlib.Path(...).mkdir(parents=True, exist_ok=True)` dans tous les blocs `__main__`.
- **`list_parameters` supprimée** de `prg/tools/tools.py` ; la clé `"param_names"` du dict interne de `BivariateLaw` n'était jamais lue.

---

## [0.3.0] - 2026-05-03

### Added

- **Parameter estimation** — `CopulaVirt.fit(data, method='tau'|'mle')` returns a `FitResult`. Pseudo-observations via ranks; method-of-moments (Kendall's τ) or maximum-likelihood (Brent on τ_K).
- **Model selection** — `CopulaVirt.fit_best(data, families=...)` ranks candidate families by AIC. `BivariateLaw.fit_best(data, left_family, right_family)` does the same with fixed margins.
- **Information criteria** — `aic`, `bic`, `aicc`, `hqc` properties on `FitResult` and `BivariateFitResult`.
- **Cross-validation** — `cv_loglik(K=5)` on both result classes; honest model-selection signal that penalises overfitting.
- **Goodness-of-fit** — `FitResult.gof_test(B=100)` and `BivariateFitResult.gof_test(B=100)`: Cramér-von Mises with parametric bootstrap. Joint variant catches misspecified margins too.
- **Bootstrap CIs** — `FitResult.bootstrap_ci(B=500)` for τ_K. `BivariateFitResult.bootstrap_ci(B=500)` for τ_K + every margin parameter.
- **IFM bivariate fit** — `BivariateLaw.fit(data, copula_class, left_family, right_family)` two-step inference for margins.
- **Tail dependence** — `CopulaVirt.tail_dependence() → (λ_L, λ_U)`. Default numerical limit on the diagonal CDF; analytical overrides on Product, Gaussian, Student, GH, Clayton, FGM, CubSec.
- **Visual diagnostics** — `FitResult.plot_diagnostics(plot_dir)` (6-panel: pseudo-obs+PDF, PP plot, λ_L curve, empirical copula, residuals, λ_U curve). `BivariateFitResult.plot_diagnostics(plot_dir)` (6-panel: data+PDF, two QQ plots, joint PP plot, λ_L/λ_U curves).
- **Result classes** — `FitResult`, `GoFResult`, `BivariateFitResult`, `BivariateBootstrapCI` exported from `prg.copulas`.

### Fixed

- `CopulaA12.pdf` no longer overflows for high τ. The original implementation evaluated `(1/u−1)^θ` directly, which underflows to 0 near u≈1 for large θ; `S^{1/θ−2}` then overflowed. Rewritten in log-space.
- Tail-dependence comments in `a12.py` and `a14.py` `__main__` blocks were incorrect (claimed λ_L = λ_U = 0). They now print the actual analytical values from `tail_dependence()`.
- All copula `__main__` scripts: `sys.path` injection moved above the first `from prg ...` import so `python prg/copulas/.../foo.py` works directly.
- `plot_samples` and `plot_overview` default sample count raised from 500 to 2000 for clearer scatter plots.

---

## [0.2.0] - 2026-05-03

### Added

- `BivariateLaw` and `ConditionalLaw` classes (Sklar joint, conditional pdf/cdf/sample).
- Sample method on every copula (Rosenblatt h-inversion via Brent).
- Plot suite: `plot_pdf`, `plot_cdf`, `plot_h_function`, `plot_samples`, `plot_overview`, `plot_multi_tau`.
- Test suite for conditional copulas, scientific identities, and bivariate construction.

---

## [0.1.0] - 2026-05-03

### Added

- Coherent exception system (`CopulaParameterError`, `CopulaNotAvailableError`, `SamplingConvergenceError`).
- Logging infrastructure throughout the package.
- Initial copula families: Product, Gaussian, Student, GH, Clayton, Frank, A12, A14, FGM, CubSec.

---

## [0.0.0] - 2026-05-03

### Added

- Initial project scaffold: `pyproject.toml`, `README.md`, `CHANGELOG.md`, `.gitignore`, `prg/__init__.py`

[Unreleased]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.4.0...HEAD
[0.4.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.5...v0.4.0
[0.3.5]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.4...v0.3.5
[0.3.4]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.3...v0.3.4
[0.3.3]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.2...v0.3.3
[0.3.2]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.1...v0.3.2
[0.3.1]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.0...v0.3.1
[0.3.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.0.0...v0.1.0
[0.0.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/tags/v0.0.0
