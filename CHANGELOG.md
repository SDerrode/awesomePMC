# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Fixed — an empty state gave NaN in the one-step PIT and hid every outlier

- When a row of the prior p is all zero (a state that ICE or SEM emptied,
  which both can do: see the degenerate-state WARNING), its transition row is
  0/0. `predictive_pit` then returned NaN for every row after the first, and
  `flag_outliers` flagged nothing, not even an 8-sd spike. The forecasting
  study found it (199 NaN PITs of 200).
- The empty row is now −∞ in log space (`inference._log_transition_weights`)
  and in the PIT filter (`outliers`). An empty state carries zero
  probability: results equal the model with that state deleted to 9e-16, for
  PMC and PMC-IN (state and pair margins) and HMC, with and without missing
  rows.
- Models without an empty state are bit-identical.
- `outliers` documents that without gating a run of erroneous readings is
  judged given its own first value (a plateau is then plausible), and why
  `sequential=True` is the mode for detection.

### Added — erroneous data: predictive PIT, outlier flags, flag-and-mask estimation (pilot)

- **Why.** Missing values are integrated out exactly, but a wrong value,
  such as a spike or a sensor glitch, still enters the likelihood as if it
  were true. On simulated data, 1 % of +4 sd spikes are enough for PMC's
  ICE to turn a regime into a "spike state". Its clean-row classification
  error then rises from 11.5 % to 44–46 %.
- **What.** A new module, `pmcprg.pmc.outliers`, exported from
  `pmcprg.pmc`:
  - `predictive_pit` gives, for every observed row, PIT_n = P(Y_n ≤ y_n |
    past), a two-sided p-value, a normal score and the log predictive
    density. It is derived from the forward filter and the transition's
    conditional CDFs. That is the margin CDF, or the copula h-function
    h_ij(F_ji(y) | F_ij(y_{n−1})), weighted as in `precompute_weights`:
    state or pair margins, and the division by D for the PMC variants.
    Missing rows before n are integrated out through the grid message of
    `gaps.py`. Scalar series only.
  - `flag_outliers` flags a row when its p-value falls below α, per row,
    with Bonferroni, or with Benjamini–Hochberg. With `sequential=True`, a
    flagged row is integrated out of the filter of the rows after it
    (innovation gating), so a spike does not also corrupt the prediction of
    its neighbour.
  - `pit_checks` runs a KS test on the PITs and Ljung–Box tests on the
    normal scores and on their squares.
  - `robust_estimate` fits, flags, sets the flagged rows to NaN and refits
    from the initial model, until the mask is a fixed point. With
    `missing_strategy = "available"` this is a trimmed ICE (or SEM). An
    optional `initial_mask` takes a pre-screen.
  - The docstrings give the formulas per variant, the multiplicity issue
    and the design of the loop. The API page `apidoc/api/pmc.md` lists the
    new functions.
- **Measured, exactness** (`test_outliers.py`, 32 tests, each tolerance
  quoted with its measured error):
  - brute force over the state paths: HMC-IN 2e-16, Gaussian regimes 6e-16,
    pair-margin PMC 4.5e-14;
  - across gaps, the quadrature error of the grid: 1.6e-8 at G = 64,
    1.1e-10 at G = 128;
  - the Gaussian AR(1) closed form: 1.3e-10;
  - the quadrature of the predictive density (Clayton, Clayton90, Frank):
    1e-15;
  - the filter equals `gap_posterior`'s to 6e-16, and its log-likelihood to
    6e-14;
  - the PIT of `forecast`'s quantiles is their level to 7e-15 (HMC-IN) and
    1.3e-9 (PMC).
- **Measured, calibration.** Under the true model the PITs are uniform and
  independent. Over 20 series of 2000 rows per fixture, the pooled KS
  p-values are 0.61, 0.99 and 0.65. The minimum over the 36 per-series KS
  and Ljung–Box p-values in the test is 0.03.
- **Measured, simulation** (`report/erroneous_data`: HMC-IN and PMC,
  N = 2000, spikes of 4, 6 and 8 sd at 1 % and 5 %, 30 replicates per cell,
  6 min):
  - with the true model, spikes of 6 and 8 sd are all flagged, and 90–94 %
    of those of 4 sd, at about α false flags per clean row;
  - masking the flags brings the true model's clean-row error on PMC from
    14–25 % back to 11.5–11.7 % (11.3 % without spikes);
  - at 1 %, flag-and-mask reaches the oracle mask: biases within 0.02,
    RMSE within 15 %, clean-row error 11.7 % on PMC. It takes 2 to 6 fits,
    2 to 5 times the cost of one ICE;
  - it breaks down when the spikes form a state: PMC at 5 %, and HMC-IN at
    5 % of 6–8 sd. A Hampel `initial_mask` restores the oracle level at 6
    and 8 sd in all 120 runs, but not for PMC at 5 % of 4 sd (3 of 30
    runs);
  - refits started from the previous fit merge the PMC regimes in 70 of 90
    runs at 1 %, which is why every refit restarts from the initial model;
  - a DPD re-estimate of τ on the raw fit's pairs changes τ by less than
    0.01. The damage is in the states and the margins, not in the copula
    fit;
  - the cost of gating on clean PMC series is 1.27 false flags per 1000
    against 0.97 without. Under 5 % of 6 sd spikes it halves them (1.26
    against 2.26).
- No existing function changes. The ICE/SEM goldens and the gap and
  missingness suites pass unchanged.

---

## [1.4.0] - 2026-09-21

### Highlights

- **Weighted copula fitting in the public API (FR-12).**
  `CopulaVirt.fit(data, method, weights=…, pseudo_obs=…)` and `fit_best(…,
  weights=…, criterion=…)` use the same weighted engine as ICE's M-step,
  now moved to `pmcprg.copulas._weighted`. Every fit reports its status and,
  on demand, diagnostics: gradient, Hessian eigenvalues and a boundary flag.
  The new API is documented on a local MkDocs site (`apidoc/`) and in
  `CONTRIBUTING.md`.
- **The copula layer is cross-validated against R and pyvinecopulib (FR-11).**
  Every family is compared, through versioned tables and an mpmath oracle,
  with pyvinecopulib, VineCopula, copula, copBasic and fCopulae: densities,
  CDFs, h-functions and their inverses, τ, tail coefficients, and weighted
  estimation and selection. pmcprg agrees to 1e-13 or better wherever a
  reference is accurate. Every disagreement is traced, almost always to the
  reference.
- **Results that change, each measured and small:**
  - the generic h-inverse and BB1's rotation inverse converge to the last
    digits; draws of those families move beyond the 9th and 13th digit;
  - the two-parameter MLE no longer depends on the scale of the weights;
  - BB1's four rotations are penalised for their two parameters in AIC/BIC;
  - `fit(method='tau')` of the multi-parameter families is now itau, where
    it used to fall back to the joint MLE;
  - the weighted τ is a τ-b on tied data, as in the references;
  - BB1 is labelled "Clayton-Gumbel".
  PMC/HMC results on data without ties are otherwise unchanged.

### Fixed — BB1's four rotations counted one free parameter instead of two (FR-12)

- `BB190`, `BB1270`, `SURVIVAL_BB190` and `SURVIVAL_BB1270` fit (τ, δ) but
  inherited `CopulaVirt`'s `n_params = 1`. Their AIC was therefore 2 too low,
  and their BIC log n (or log Σw) too low. That penalty is what `fit_best`
  and ICE's family selection use, so these four families were favoured
  wrongly whenever they were candidates.
- A rotation now takes `n_params` from its base family.
- `test_copula_n_params.py` checks that `n_params` equals the number of
  registered parameters for every family; it fails on the old code.
- ICE's family choice may change when a BB1 rotation is a candidate. It is
  not one of the default candidates.

### Added — CONTRIBUTING, an API documentation site, and the state-labelling rule (FR-12)

- **`CONTRIBUTING.md`** covers:
  - the development install and how to run the tests;
  - the numerical-rigour norms the suite follows: measured tolerances,
    mpmath references, parity tables regenerated only by `scripts/parity/`,
    deterministic `zlib.crc32` seeds, and bit-exact comparisons on macOS
    arm64 only;
  - how CI runs, how to report an issue, and how pull requests go through
    the GitHub mirror.
- **API documentation site**: MkDocs + Material + mkdocstrings, built
  locally only, with 0 warnings under `mkdocs build --strict`.
  - The sources are in `apidoc/` (`mkdocs.yml`, new `docs` extra).
  - Pages cover `pmcprg.copulas`, `pmcprg.pmc`, `pmcprg.missing`,
    `pmcprg.diagnostics`, installation and the state-labelling rule.
  - A new `docs` CI job builds it in strict mode. Nothing is deployed.
- **State labelling, documented** (`apidoc/state-labelling.md`, README).
  - A state's index comes from the initial model, and ICE/SEM keep it.
  - `init = "kmeans"` renumbers its arbitrary clusters to the declared states
    with the Hungarian assignment of `_align_kmeans_labels`.
- The docstrings of `clarke_test` and `comparison_matrix` had prose inside
  `Parameters`, which a strict docstring parser misreads. It moved to
  `Notes`.
### Added — weighted copula fitting and per-fit diagnostics in the public API (FR-12, part A)

- **One weighted engine.** The weighted Kendall τ, the weighted MLE and the
  weighted family selection of ICE's M-step move from `pmcprg/pmc/ice.py` to
  `pmcprg.copulas._weighted`, unchanged. ICE imports them under their former
  names. ICE's golden and bit-identity tests pass unchanged.
- **`CopulaVirt.fit(data, method, *, weights=None, pseudo_obs=False)`.**
  - `pseudo_obs=True` takes data in (0, 1)² as they are. Otherwise the data
    are rank-transformed; with weights, by the weighted empirical CDF of
    ICE's empirical copula margins (FR-7 a): Σ_{x_k ≤ x} w_k / (Σw + 1).
  - Weights are frequency weights. They are validated in one place
    (`pmcprg.copulas.validate_weights`, also used by ICE): 1-D, one per
    observation, finite, ≥ 0, with a positive sum.
  - Rows of weight 0 are dropped. If every other weight is 1, the result is
    the unweighted fit, bit for bit: unit weights change nothing, and
    {0, 1} weights give the fit on the subset.
  - Otherwise the fit runs ICE's engine: `'mle'` is ICE's weighted MLE, bit
    for bit, and `'tau'` inverts the weighted τ. The reported
    log-likelihood is Σ wᵢ log cᵢ, and the BIC charges k·log Σw.
    VineCopula charges log n instead, and pyvinecopulib rescales the
    weights to mean 1.
  - Parameters do not depend on the weights' scale when pseudo-observations
    are passed. The rank transform does: a WARNING fires when Σw < 10.
- **`CopulaVirt.fit_best(..., *, weights=None, criterion='aic', pseudo_obs=False)`.**
  `criterion` is `'aic'` (the default, unchanged), `'bic'` or `'loglik'`.
  With weights and `method='mle'`, it makes ICE's weighted selection.
- **Weighted τ-b on ties.** The weighted Kendall τ is now the weighted τ-b
  of pyvinecopulib and VineCopula. Without ties it keeps the former τ-a
  code, bit for bit. On the FR-11 tie grid it matches both references to
  1.3e-14 (it used to differ from them by 5e-2). ICE sees ties only with
  tied observations, such as integer-valued data. Its joint two-parameter
  MLE then starts from τ-b. In a fingerprint of ICE/SEM runs on rounded
  data, 5 of the 6 runs with copulas moved: τ by at most 1.3e-7, the copula
  parameters by at most 2.4e-6 relative, the log-likelihood trace by at most
  4.3e-5 nat. No family changed.
- **`FitResult` reports its status.** It gains `message`, `n_iter`,
  `n_eval` and `weights`, plus the `failed` and `n_eff` properties. The
  `diagnostics` property returns a `FitDiagnostics` in the manner of GJRM's
  `conv.check()`:
  - the gradient and Hessian of the weighted log-likelihood on the
    parameter scale;
  - the Hessian's eigenvalues, with `negative_definite`;
  - the Newton decrement;
  - a boundary flag, with FR-4's rules where they exist;
  - the optimiser's counts.

  It is computed on first access: 3 likelihood evaluations for one
  parameter, 9 for two, 0.3 to 19 ms at n = 1000. ICE pays nothing.
- New public names: `validate_weights`, `weighted_kendall_tau`,
  `weighted_pseudo_obs`, `FitDiagnostics`, `FitBestResults`.

### Changed — `fit(method='tau')` of the multi-parameter families is itau (FR-12)

- For the three-parameter Tawn, the profile over (ψ_u, ψ_v) at τ̂ is a
  Nelder–Mead search, started from an explicit simplex inside the box.
  scipy's default simplex is clipped onto a bound when the start sits on one,
  and before scipy 1.11 that collapses it. The minimum-versions CI job
  (scipy 1.10) caught this before release: the search stayed at the start,
  Gumbel's corner (1, 1), with log-likelihood 176.1 against 214.6. It now
  reaches the same optimum, to the digits shown, under scipy 1.10 and 1.18.
- **Before.**
  - Student, BB1, BB1's rotations and survival, BB6, BB7, BB8, Tawn 1/2/3
    and t-EV logged a warning and ran the joint MLE. The result was
    reported as `method='mle'`, so `fit_best(method='tau')` put these
    families in `other_method`, unranked.
  - The generic code path left the extra parameter at its registered
    start value (ν = 4 for Student).
- **Now.** τ̂ is Kendall's τ, as for the one-parameter families. The other
  parameters are fitted by maximum likelihood at that τ, a profile
  likelihood, as VineCopula's itau fits Student's ν. With weights, the
  weighted τ and the weighted profile are used. Student's weighted itau ν
  matches pyvinecopulib's to 1e-6 relative.
- **Measured** (n = 500, seed 1; before → after):

  | family | τ | extra parameter | log-likelihood |
  |---|---|---|---|
  | Student (τ 0.5, ν 8) | 0.4941 → 0.4851 | ν 7.700 → 7.204 | 171.270 → 171.173 |
  | Student (τ 0.7, ν 3) | 0.6924 → 0.6921 | ν 3.245 → 3.239 | 400.074 → 400.074 |
  | BB1 (τ 0.5, δ 1.5) | 0.4921 → 0.4833 | δ 1.408 → 1.387 | 181.731 → 181.618 |
  | Tawn 1 (τ 0.4, ψ 0.6) | 0.3774 → 0.3848 | ψ 0.533 → 0.546 | 134.905 → 134.831 |

- **What did not change.** Every `'mle'` fit and every one-parameter `'tau'`
  fit is bit-identical: an 86-fit fingerprint covered every family.
- **What follows.**
  - `fit_best(method='tau')` now ranks the multi-parameter families.
  - `standard_errors()` of an itau fit of these families raises: the
    variance of that estimator is not implemented.
  - The unweighted Student and BB1 MLEs now report L-BFGS-B's own success
    in `converged`.
- ICE never takes this path and is unchanged.

### Fixed — the two-parameter MLE depended on the scale of the weights (FR-11)

- The joint fit of the two-parameter families (Student, BB1, BB6–BB8, …)
  stopped L-BFGS-B on an absolute `gtol = 1e-6`. That bound applies to the
  gradient of the *total* weighted log-likelihood, which scales with the
  weights, so a small Σw stopped the search early. Found by the FR-11
  weighted-parity tests: at Σw = 1.4e-4 the fit ended 2.6e-3 nat below its
  maximum, with τ off by 5.6e-4.
- `gtol` is now scaled by the mean positive weight. Multiplying every weight
  by a constant no longer changes the fit: Δℓ ≤ 9.5e-12 and Δτ̂ ≤ 1.4e-7 for
  scales from 2⁻²⁰ to 10⁶.
- Unit weights and `weights=None` are bit-identical. ICE's fits change only
  for pairs of states with small posterior mass.

### Added — FR-11 parity, wave 3: weighted estimation and family selection

- **What.** ICE's weighted copula fit, taken exactly as the M-step calls it
  (`_weighted_kendall_tau`, `_fit_copula_params`, `_select_and_fit_copula`),
  is compared with pyvinecopulib 1.0.0 and VineCopula 2.6.1 fitting with
  weights. Two oracles of our own complete the comparison: the exact
  weighted τ in integer arithmetic, and a polished optimum of pmcprg's
  weighted log-likelihood.
  - Data: `scripts/parity/gen_weighted_data.py`.
  - Generators: `gen_pyvinecopulib_weighted.py` and `gen_r_weighted.R`;
    data and tables together weigh 520 kB.
  - Tests: `test_parity_weighted.py`, which reads the tables only.
- **Definitions, measured.**
  - Weighted τ: pmcprg's is a product-weight τ-a, the references' a weighted
    τ-b. They are identical without ties, and ICE has none.
  - Reported log-likelihood: pmcprg and VineCopula's MLE give Σ wᵢ log cᵢ;
    VineCopula's itau ignores the weights; pyvinecopulib rescales them to
    mean 1.
  - BIC: pmcprg charges log Σw, VineCopula log n.
- **Measured.**

  | | pmcprg | pyvinecopulib | VineCopula |
  |---|---|---|---|
  | weighted τ against the exact value | 1.7e-16 | 5.6e-17 | 1.8e-13 |
  | MLE, nat below the maximum | 1.3e-11 | 2.6e-10 | 3.0e-8 |

  With {0, 1} weights, pmcprg gives the fit on the kept subset bit for bit.
  One-parameter fits do not change when all weights are scaled.
- **Selection.** 960 decisions among {independence, Gauss, true family},
  |τ| ≈ 0.4, n = 500. The true family is recovered every time except for
  BB1, which is taken for Gauss:
  - unweighted BIC: 19/20;
  - weighted AIC: 19/20;
  - weighted BIC: 15/20.

  `BiCopSelect` agrees in 959 of 960 cases; the other differs through BIC's
  sample size.
### Added — FR-11 parity, wave 2: BB1, BB6, BB7 and BB8

- **What.** Interior parity against pyvinecopulib 1.0.0 and VineCopula 2.6.1
  for BB1 in all four rotations (and its survival's rotations), BB6, BB7 and
  BB8. An mpmath oracle recomputes everything from the textbook CDFs and
  generators alone. Its τ is the Genest–MacKay integral, within 2.2e-20 of
  the literature's closed forms.
  - Generators: `scripts/parity/gen_pyvinecopulib_bb.py` and `gen_r_bb.R`
    (tables 175 kB).
  - Tests: `test_parity_bb.py`. Without mpmath its 61 table comparisons
    still run.
- **Parametrisations, measured.** Both packages take Joe's (1997) [θ, δ],
  and VineCopula's 90°/270° families negate both parameters.
- **Measured.** pmcprg agrees with mpmath to:

  | Density (relative) | CDF | h | τ | λ |
  |---|---|---|---|---|
  | 1.3e-14 | 2.0e-16 | 9.5e-15 | 1.7e-16 | 1.1e-16 |

  The same holds under the minimum versions, except τ (6.1e-16 there).
  No pmcprg defect was found.
- **Package discrepancies**, registered and checked against mpmath:
  - numerical h-inverses in both libraries;
  - formulas that lose precision near (1, 1): the BB6 density is 3.5e-7
    relative off;
  - VineCopula's integrated τ, ≤ 1.1e-7.

### Fixed — BB1 was labelled "Joe-Clayton"

- BB1 nests Clayton (δ = 1) and Gumbel (θ → 0), as its own module says, and
  VineCopula calls it "Clayton-Gumbel". "Joe-Clayton" is BB7.
- The display name of `BB1` and `SURVIVAL_BB1` (plot titles, messages),
  the module docstrings and the README now say "Clayton-Gumbel".
- Short names, IDs and saved models are unchanged.

### Changed — the generic h-inverse now converges to the last digits (FR-11)

- `CopulaVirt.inv_h`, the Brent search used by the families without a
  closed-form or Newton inverse, stopped at `xtol = 1e-8`. The FR-11 parity
  tests measured what that left:
  - on h⁻¹ itself, up to 2.5e-9 against mpmath;
  - on the round trip h(h⁻¹(w)) − w, up to 1.2e-6 (A12) and 2.9e-7
    (Hüsler–Reiss).
  The families concerned are Galambos, Hüsler–Reiss, Plackett, AMH, FGM, A12
  and A14; their rotations have their own inverse.
- It now stops at `xtol = 1e-15` (`maxiter` 100).
  - Round trip after the change: ≤ 1.2e-13 on 2 000 points per family.
  - h⁻¹ against mpmath: ≤ 2.7e-15, except where h is flat — A14 2.2e-14
    and A12 1.2e-14 at v = 0.99, the limit set by the rounding of h itself.
  - Cost: 5–10 % more time per call.
- **Results change:** sampling, simulation and every quantity that inverts h
  for these seven families move beyond the ninth digit. The other families
  are bit-identical.
- **The BB1 rotations and survival** (`BB190`, `BB1270`, `SURVIVAL_BB1` and
  its rotations) have their own inverse on kernel coordinates. It stopped at
  `xtol = 1e-13` on log v, which left 1.5e-14. It now also stops at 1e-15:
  7.1e-15 against mpmath, at the same cost. Their draws move beyond the
  thirteenth digit.

### Added — FR-11 parity, wave 1: the families only R has

- **What.** Galambos, Hüsler–Reiss, t-EV, the three Tawn models, Plackett,
  AMH, FGM, and A12 and A14 with their 90°/270° rotations. The references
  are pyvinecopulib 1.0.0, VineCopula 2.6.1, copula 1.1-7, copBasic 2.2.16
  and fCopulae 4052.86.
  - The tables are generated offline (`scripts/parity/gen_r_extra.R`,
    `gen_pyvinecopulib_extra.py`; 164 kB with provenance).
  - An mpmath oracle recomputes every quantity from the textbook CDF alone,
    so that no family rests on a single reference.
  - `test_parity_extra.py` needs neither R nor pyvinecopulib, and without
    mpmath its 96 table comparisons still run.
- **Measured.** pmcprg agrees with mpmath to 4e-14 or better everywhere,
  with two exceptions:
  - the generic Brent `inv_h`, 2.5e-9 on h⁻¹ for Galambos, Hüsler–Reiss,
    Plackett, AMH, FGM, A12 and A14;
  - the `quad`-based τ of Galambos and Hüsler–Reiss, 1.8e-12.
  The tests pass unchanged from the minimum versions (numpy 1.24,
  scipy 1.10) to the current ones.
- **Correction (documentation only).** pmcprg's `TAWN1` (ψ_u = 1) is
  VineCopula's family **204** and `TAWN2` is **104**, the reverse of what
  `tawn.py` said. Four references agree on this.
- **Defects found in the references**, recorded in `scripts/parity/README.md`:
  - fCopulae's default AMH density is wrong (e^y − 1 in place of e^y − θ);
  - copula's spline τ for t-EV ignores ν and the sign of ρ;
  - several densities lose all their digits in the corners.

---

## [1.3.1] - 2026-09-21

A patch release: one bug fix found by the new parity tests, plus the tests
themselves. `BivariateLaw` users working with a rotated (90°/270°) or Tawn
copula and conditioning on the right margin should upgrade. The PMC/HMC
estimators and classifiers were not affected.

### Fixed — `BivariateLaw` conditioned on its right margin by swapping the copula's arguments (FR-11)

- `conditional_cdf`, `conditional_pdf`, `sample_conditional` and the
  left-given-right sampler with `which="right"` evaluated the copula with its
  arguments swapped. That is right only for an exchangeable copula. For the
  90°/270° rotations (14 registry entries) and the three Tawn models, they
  returned the conditional law of the other side: P(U ≤ u | V = v) was off by
  up to 0.46 (Joe), 0.42 (Clayton) and 0.19 (Gumbel). Clayton 90°, τ = −0.5,
  at (0.3, 0.8): 0.601 instead of 0.535.
- Found by the parity tests below. Checked independently against the closed
  form ∂C90/∂v = 1 − ∂C0/∂b(1 − u, v): agreement 1e-16 after the fix, error
  0.07–0.14 before, while `which="left"` was already right.
- The fix adds `CopulaVirt.transposed()`, the copula of (V, U), and an
  `exchangeable` flag:
  - a 90° rotation returns the 270° rotation of the same base, and conversely;
  - Tawn types 1 and 2 return each other;
  - Tawn 3 returns itself with ψ_u and ψ_v swapped;
  - an exchangeable copula returns itself, so its results are bit-identical.
- The PMC/HMC inference and simulation condition on the previous observation
  (the left argument) and were not affected. Every golden test is unchanged.

### Added — interior parity with pyvinecopulib, VineCopula and R `copula` (FR-11, pilot)

- **What.** `scripts/parity/` generates reference tables offline, from
  pyvinecopulib 1.0.0, VineCopula 2.6.1 and copula 1.1-7:
  - quantities: pdf, cdf, h1, h2, both h-inverses, τ(θ), λ_L and λ_U;
  - families: Gaussian, Student and Frank, plus Clayton, Gumbel and Joe in
    all four rotations;
  - 60 parameter sets × 25 interior points.
  The tables live in `pmcprg/tests/data/parity/` with their provenance.
  `test_parity_interior.py` needs neither R nor pyvinecopulib.
- **Conventions**, measured and written in `scripts/parity/README.md`:
  - parameters: τ for pmcprg, and ρ, θ, ν for the references;
  - VineCopula codes 23/24/26 and 33/34/36 take −θ;
  - rotations: 90° reflects u, 270° reflects v;
  - h1 = ∂C/∂u, h2 = ∂C/∂v, and the argument each inverse is taken in.
- **Measured.** pmcprg agrees with every reference to 1e-13 or better
  wherever the reference is accurate. Each remaining disagreement is traced
  with mpmath:
  - pyvinecopulib's Frank cdf/h near (1, 1);
  - the numerical h-inverses of pyvinecopulib and VineCopula, up to 1.2e-10;
  - VineCopula's interpolated Frank τ, 5.5e-4;
  - copula's Frank h at θ = −30 and its symbolic rotated-Joe density, 1.5e-3;
  - pmcprg's Gaussian CDF, 7e-11 relative, within the documented accuracy of
    its quadrature.
- **The Student copula's accuracy follows scipy's `t` distribution.** The
  first run on the minimum-versions CI job (scipy 1.10) showed this. Against
  the three references, over the whole Student grid:

  | scipy | density (relative) | h and h-inverses |
  |---|---|---|
  | 1.10 to 1.12 | 2.8e-8 | 1.0e-9 |
  | 1.13 to 1.16 | 3.1e-11 | 4.1e-12 |
  | from 1.17 | 3.9e-14 | 4.4e-15 |

  This matches scipy's `t.ppf`, which is 2.3e-9 relative from mpmath up to
  1.12 and 1.6e-11 in 1.13–1.16. The parity tests use tiered tolerances,
  measured under each of these versions. No change is needed for practical
  use; a newer scipy gives the last digits.

---

## [1.3.0] - 2026-09-21

### Highlights

- **State-dependent missingness (P6).** Until 1.2.0 a missing observation was
  integrated out under an *ignorable* mechanism: the gaps said nothing about
  the hidden states. A model can now declare, in a `[missingness]` table,
  that the probability of a gap depends on the hidden state:
  - `"state"`: one missing rate per state;
  - `"state-markov"`: an onset and a persistence probability per state, for
    gaps that come in bursts.
  Every inference function then uses the mask as evidence on the states:
  classification, posteriors, imputation, forecasting and FFBS draws. The
  default is unchanged and bit-identical to 1.2.0.
- **Estimation and a test.** ICE and SEM estimate the mechanism (config key
  `missingness`), and `missingness_lr_test` tests whether the gaps depend on
  the state, with a χ² and a parametric-bootstrap p-value. Both are
  reachable from the CLI (`pmc estimate --missingness`,
  `pmc missingness-lr-test`) and from the GUI.
- **Measured.**
  - On simulated data, using the mask cuts the classification error at the
    gaps from 31 % to 3 %.
  - For gaps in bursts, only `"state-markov"` stays calibrated.
  - The test holds its level on HMC-IN. On grid variants, `"state-markov"`
    needs about 20 bursts of gaps.
  - On PAMAP2 the dependence is real and bursty, but it helps classification
    only where each state's dropout rate is homogeneous. The study and its
    advice are in `report/missing_state/pamap2/`.

### Added — state-dependent missingness on real data: PAMAP2 (P6, study)

- **Why.** The work on state-dependent missingness was motivated by PAMAP2's
  activity-dependent hand-IMU dropouts. This study measures them, tests them
  with `missingness_lr_test`, and asks whether modelling them helps
  classification and imputation.
- **What.** `report/missing_state/pamap2/` (`run_pamap2.py`, `summarise.py`,
  `pm_common.py`). It reuses the three subjects, windows, groups, gap rules,
  oracles and ICE settings of the real-series study, and covers:
  - the real gaps, described at 100 Hz and 2 Hz;
  - LR tests based on the labels and on ICE (HMC-IN, profile statistic);
  - classification on the real gaps, and on masks drawn by
    `state_dependent` / `state_markov` on the true groups at the real rates;
  - imputation of the simulated gaps.
  No raw data is committed.
- **Measured.**
  - **The gaps.** The dependence holds in all three subjects: rest against
    active, label LR 11–1118. Its K = 3 ordering is subject-specific. The
    gaps are bursty: `"state"` is rejected by the run lengths, and
    `"state-markov"` fits them but not their clustering over 10 s.
  - **ICE-based tests.** LR 30–170 (≥ 10 % rule) and 70–827 ("any") for
    subjects 102 and 105, nested LR 61–735. Subject 108 is borderline
    (p 0.018, bootstrap 0.03).
  - **Classification.** The mask cuts the oracle error at the missing windows
    by 1.2–5.2 points when the dropout is homogeneous within each state
    (simulated masks). It changes nothing at K = 2 on the real gaps and hurts
    at K = 3 (102: 2.1 → 34 %): walking has the dropout of the vigorous group,
    and whole bouts flip.
  - **Starts.** Estimated from a fresh start, up to 17 % of unsupervised fits
    end in a worse basin than the ignorable fit (down to −100 nat).
    Estimated from the ignorable fit, none do.
  - **Imputation** is slightly worse with the mask: 53 of 56 comparisons,
    RMSE up to +0.10.
  - **Calibration.** Posteriors at the missing windows are 5–70×
    overconfident, with or without the mask.
- **Advice.** Use the mask with states whose dropout rates are homogeneous,
  estimate it from the ignorable fit, and check segmentation and calibration
  against the ignorable model.

### Fixed — the nested missingness test fits its `"state"` null from the ignorable fit (P6)

- `missingness_lr_test(alternative="state-markov", null="state")` used to fit
  the `"state"` null directly from the model it was given. On PAMAP2 it
  settled in worse basins in 2 of 12 cells (33–41 nat lower), and the nested
  LR came out at 665 and 287. The null is now fitted from the ignorable fit,
  as the alternatives already were from their null. The two cells now give
  **735.0 and 520.1**, the best-fit values; the null of subject 102 ends at
  −11 963.46, the `"state"` fit of the `"state"` vs common test. This costs
  one more ICE fit.

### Added — state-dependent missingness in the CLI, the GUI and the README (P6)

- **CLI.** `pmc estimate --missingness {model,ignorable,state,state-markov}`
  sets the ICE/SEM `missingness` key and prints the fitted mechanism, which
  is also saved in the output model's `[missingness]` table. A new
  subcommand, `pmc missingness-lr-test --model --data [--alternative]
  [--null] [--bootstrap] [--seed]`, runs `missingness_lr_test` and prints its
  `summary()`.
- **GUI.** The ICE/SEM tab gets a "Missingness mechanism" combobox (the four
  modes, with tooltips); the estimation log shows the fitted mechanism; the
  Analysis menu gets "Missingness LR test…" (the default test, `"state"`
  against a common rate, run in the worker thread, `summary()` in the log).
  A model's own `[missingness]` table survives GUI edits and saves.
- **README.** A "State-dependent missingness" subsection: when to use it, the
  TOML table, estimation with ICE and the test, each with a runnable snippet.
- `pmcprg.pmc.missingness.describe_mechanism` gives the one-line summary the
  CLI and the GUI print.

### Added — estimating the missingness mechanism, and testing it (P6, wave 2)

- **Why.** The pilot below made the mask evidence on the states, but its
  rates had to be known: ICE and SEM held them fixed. On real data they are
  what one wants to learn — and before relying on them, whether the mask
  depends on the states at all.
- **What — estimation.** A config key `missingness` for ICE and SEM
  (`ice_cfg` / `sem_cfg`, TOML `[ice]` / `[sem]`; `MISSINGNESS_MODES`):
  `"model"` (default: the model's mechanism, held fixed — bit-identical to
  before), `"ignorable"` (dropped), `"state"` / `"state-markov"` (estimated).
  The M-step adds `pmcprg.pmc.missingness.estimate_mechanism` on the exact
  posteriors given (y_obs, m) (both `missing_strategy` values) or on SEM's
  drawn path: π_i = Σ γ_n(i) m_n / Σ γ_n(i); onset and persistence split on
  m_{n−1}, with the n = 0 term γ_0(i) log p(m_0 | s_i) maximised exactly per
  state. Start: the model's mechanism of that kind, else the
  state-independent MLE of the mask, whose first E-step is the ignorable
  one. `trace.missingness_history` records the mechanism of every iterate;
  the returned model carries the estimate. A complete Y (m = 0: MLE rates 0,
  i.e. ignorable) logs a WARNING and fits the ignorable model.
- **What — test.** `pmcprg.pmc.missingness_lr_test(model, Y, alternative=,
  null=, ice_cfg=, n_bootstrap=, seed=)` → a frozen `MissingnessLRTest`
  (`statistic`, `df`, `p_value`, `p_value_bootstrap`, the fitted models and
  mechanisms, `summary()`): `"state"` vs a common rate (LL0 = the ignorable
  fit + M log(M/N) + (N − M) log(1 − M/N) exactly, df = K − 1),
  `"state-markov"` vs a common Markov mask (df = 2(K − 1)), and
  `"state-markov"` vs `"state"` (df = K). Asymptotic χ² and a parametric
  bootstrap (paths and masks from the fitted null, separate seed streams,
  both models refitted); both fits run to ICE's fixed point.
- **What — the statistic.** LR = 2 (sup_H1 − sup_H0), each supremum
  approximated by the best its hypothesis reaches at the θ of either fit:
  ℓ_h(θ), the log-likelihood maximised over hypothesis h's mechanism with θ
  fixed (EM over the mechanism alone, exact M-step without the boundary
  guard), sup_H0 ≈ max(ℓ0(θ̂0), ℓ0(θ̂1)), sup_H1 ≈ max(ℓ1(θ̂0), ℓ1(θ̂1),
  sup_H0). LR ≥ 0 by construction, without clipping. Where ICE is not EM
  (the grid variants) the difference of the two fixed points mixed how well
  each fit optimised θ with the mechanism's effect: on HMC-DN, N = 500,
  Markov null, 9 of 200 values were negative (min −1.42, whose LR by a
  direct maximisation over all 14 parameters is 4.73), and the error against
  that direct LR (36 replications) had a mean 1.02 and a max 6.2 — 0.61 and
  3.9 with the profile statistic. New result fields `sup_log_lik_null`,
  `sup_log_lik_alt`, `profile_log_liks`, and `statistic_fits` (the
  difference of the fits, a diagnostic); `log_lik_null`, `log_lik_alt`
  remain those of the fits. Cost: 17 % of the test's CPU on HMC-DN
  (N = 500), 38–45 % on HMC-IN (0.2 s per test).
- **Measured — design** (`report/missing_state/design_measurements.py`,
  HMC-IN, 100 runs per cell). *Initial term*: dropping it moves the
  estimates by 4–7 % of an sd at N = 500 with no systematic shift, but loses
  up to 2.5 nat (11 runs of 100 above 0.1 nat) on series that start inside a
  long burst — kept exact, for 8 % of an ICE iteration. *Boundary guard*: a
  rate at 0 is absorbing — from π_0 = 0 (truth 0.05) ICE stays at 0 in 50/50
  runs, 58 nat below the fit from the common start (N = 2000); one
  pseudo-observation at the pooled rate of the mask recovers it in 6–9
  iterations and changes bias and RMSE by ≤ 0.001 on well-populated states
  (not a config key). *Start*: the common rate is left at the first M-step
  ((0.16, 0.16) → (0.055, 0.27), truth (0.02, 0.30)); with i.i.d. states π is
  not identified.
- **Measured — test** (`report/missing_state/lr_study.py`, K = 2, 5 %
  level; bootstrap by the warp-speed method). First study, 8 800
  replications, with the difference of the fits as the statistic: size on
  HMC-IN (ICE is EM) 0.045–0.060 (χ²) and 0.045–0.065 (bootstrap) for
  `"state"`, 0.028–0.050 / 0.028–0.058 for `"state-markov"`, 0.020–0.048 /
  0.022–0.072 nested, N = 500–2000; on HMC-DN (grid, ICE not EM) `"state"`
  0.060 / 0.040–0.060, `"state-markov"` 0.135 / 0.175 at N = 500 (~9 bursts
  per series) and 0.030 / 0.015 at N = 1000. Rerun with the profile
  statistic, same seeds (HMC-IN at N = 500, the HMC-DN Markov cells; 3 000
  replications, the fits bit-identical): HMC-IN 0.058 / 0.050 (`"state"`),
  0.052 / 0.035 (`"state-markov"`), 0.020 / 0.020 (nested), no negative
  value (the fits' statistic: 5 in the nested test); HMC-DN Markov 0.125 /
  0.150 at N = 500, 0.045 / 0.010 at N = 1000, no negative value (the fits':
  9 at N = 500). Power of `"state"` on HMC-IN: 0.37 / 0.62 / 0.94 at N = 500
  / 1000 / 2000 for rates 7.5 % vs 12.5 %, 0.885 / 0.995 / 1 for 5 % vs
  15 % (the fits' statistic; unchanged at N = 500 with the profile one); of
  `"state-markov"` against onset and persistence differences 0.26 / 0.49 /
  0.81 (0.29 at N = 500 with the profile statistic). Recovery: π̂ and â
  unbiased to a tenth of their RMSE (π̂_1 = 0.15: 0.025, 0.017, 0.013, 0.008
  at N = 500, 1000, 2000, 5000); b̂ biased down by 0.01–0.06 at N ≤ 1000,
  RMSE 0.10 at N = 5000 for a state with 1 % onsets.
- **Measured — limit** (`report/missing_state/lr_diagnosis.py`, README
  "Diagnosis"). The Markov test's excess of rejections on HMC-DN at N = 500
  is the likelihood ratio's own: a direct maximisation of the likelihood over
  all parameters gives an LR of mean 3.40 (χ²(2): 2) on 30 null series, 5
  above the 5 % point; a genuine bootstrap (B = 99) rejects the four largest
  null values at p = 0.01 (its own distributions χ²-like, 95 % quantiles
  5.2–6.9). Use the Markov test on grid variants from ~20 bursts (N ≈ 1000
  here) on.
- **Unchanged by default.** `missingness = "model"` is bit-identical to the
  previous `ice.py` / `sem.py` (loaded from git: ICE with both strategies,
  SEM, with and without a mechanism, gaps and complete Y); the 90 golden
  bit-identity cases of `test_estim_complete_data_identity.py` pass
  unchanged.

### Added — state-dependent missingness: the mask as evidence on the hidden states (P6, pilot)

- **Why.** Missing rows were integrated out under an *ignorable* mechanism
  (MCAR/MAR): the mask said nothing about the states. On real data it does —
  on PAMAP2 the hand-IMU dropout is 1–4 % of the rows during walking or
  running against < 0.02 % lying or sitting, so a gap is itself evidence on
  the activity. A model can now say so.
- **What.** An optional `[missingness]` table in the model TOML,
  `PMCModel.missingness` (read-only; `None` = ignorable, the default) and
  `PMCModel.with_missingness(...)` (a copy with another mechanism, or none).
  Two mechanisms (`pmcprg.pmc.missingness`), the mask m independent of Y given
  the states, p(m | x) = Π_n e_n(x_n) — a selection model for data missing not
  at random (Little & Rubin 2019):
  - `mechanism = "state"`, `rates = [π_i]`: P(row n missing | x_n = i) = π_i,
    independently;
  - `mechanism = "state-markov"`, `onset = [a_i]`, `persistence = [b_i]`: a
    Markov mask, P(m_n = 1 | m_{n-1} = 0, x_n = i) = a_i,
    P(m_n = 1 | m_{n-1} = 1, x_n = i) = b_i, m_0 from the stationary law
    a_i / (1 − b_i + a_i). Gaps come in bursts: `"state"` counts a burst of L
    rows as L independent pieces of evidence, `"state-markov"` as one onset
    and L − 1 continuations.
  Inference then uses p(y_obs, m) = Σ_x ∫ p(x, y) Π_n e_n(x_n) dy_miss in every
  entry point — `classify`, `forward`, `backward` (and their log-space
  fallbacks), `sample_posterior`, `gap_posterior`, `impute` (Nyström
  quantiles and FFBS draws included) and `forecast`. The factor multiplies the
  message at each position after the transition (after the Tauchen–Hussey
  block renormalisation on the quadrature grid); for the exact shortcut it
  lives in the tensors of `precompute_weights` (columns of W, rows of
  f_pdf), so user calls `forward(model, Y, W=W, f_pdf=f_pdf)` stay exact. A
  complete Y still has a mask (none missing) and gets its factors; `forecast`
  gives none to the appended rows, whose mask is unknown. Rates 0 and 1 are
  allowed (a state becomes impossible at missing or observed rows).
- **Estimation.** Not in this pilot: ICE and SEM carry the mechanism of the
  initial model unchanged (π held fixed) and compute every E-step given
  (y_obs, m); ICE's `"impute"` strategy keeps the observed mask in the
  E-step of each completed series. (This is now the default,
  `missingness = "model"`, of the entry above.)
- **Simulation.** `pmcprg.missing.state_dependent(Y, X, rates)` and
  `state_markov(Y, X, onset, persistence)` draw such masks from a state path.
- **Measured.** Against path enumeration (N = 6, every path weighted by its
  factors written from the definitions): exact-shortcut variants to 7e-15;
  Gaussian-copula HMC-DN / PMC against the exact Gaussian path mixture at
  G = 64 to 1.2e-7 (log-likelihood, γ, ξ), 9e-7 (imputation sd), 3e-7
  (quantiles) — the ignorable model's own quadrature error on the same data
  is 7e-8 / 1.1e-7 / 1.4e-6; Clayton PMC (state and pair margins) and PMC-IN
  with pair margins against adaptive quadrature to 5.7e-7 at G = 64 and
  6.2e-10 at G = 128. A state-independent mechanism leaves every posterior
  unchanged to 9e-15 and shifts the log-likelihood by the log-probability of
  the mask to 1.3e-13. Monte Carlo (HMC-IN, 40 sequences): with π = (0.01,
  0.3) the MPM error at the missing rows falls from 0.311 (ignorable) to 0.034,
  and the posterior probabilities there are calibrated (overall z = −0.66;
  the ignorable ones are not, z = +56). With bursty masks (a = (0.002, 0.02),
  b = (0.8, 0.95)) the error at the missing rows is 0.193 ignorable, 0.090
  with `"state"` at the matched marginal rates and 0.062 with
  `"state-markov"`; `"state"` is over-confident (top bin predicts 0.998, 0.912
  observed; z = −13.9), `"state-markov"` calibrated (z = +0.69, every bin
  within 0.04).
- **Unchanged by default.** With `missingness = None` every result is
  bit-identical to 1.2.0 — checked on every fixture against the 1.2.0 code
  itself (`classify`, `forward`, `backward`, `sample_posterior`,
  `gap_posterior`, `impute`, `forecast`, the augmented chain), besides the
  existing golden and bit-identity tests. Old pickles load as ignorable.

---

## [1.2.0] - 2026-09-19

### Changed — CI: the fast suite on every push, the full suite on tags, weekly and on demand

- **The problem, measured.** Every push ran the whole suite (7 100 tests) in
  one process on four Python versions: 94 min (3.14), 97 (3.13), 145 (3.11)
  and 150 (3.12) per job on the last full run, plus 18 min for the
  minimum-versions job. A burst of pushes cancelled itself into silence: of the
  53 CI runs between 13 and 18 September, **11 were green, 22 red and 20
  cancelled** — several reds were noticed hours late, and the near-bit-exact
  tolerance failures had to be found by waiting for a whole matrix.
- **Now** (`.github/workflows/ci.yml`; `.gitlab-ci.yml` is removed — the GitLab
  server has no runner, so GitLab is a code mirror only and GitHub is the CI):
  - *push / pull request* — lint, smoke, minimum-versions and the **fast**
    suite (`-m "not slow"`, `pytest -n auto`) on Python 3.11 and 3.14;
  - *release tag `v*`, weekly schedule (Mondays 03:23 UTC), manual run* — the
    **full** suite on Python 3.11 to 3.14, coverage on 3.14 only. A green run
    on the release tag is the release gate (`RELEASING.md` step 11).
  A small `plan` job decides the matrix from the event; a run that is the
  release gate, the weekly one or a manual one is no longer cancelled by a
  push (they used to share one concurrency group per ref).
- **Measured** (4 xdist workers, one BLAS thread each, on the development
  machine): the fast suite — 6 627 tests — passes in **2 min 43 s**; the 458
  `slow` tests pass in **19 min 27 s**, 10 of them minutes each (the
  Rosenblatt, Lystig–Hughes and dependent-multiplier Monte-Carlo studies).
  No test failed for lack of isolation under xdist. On GitHub's runners
  (4 vCPU, slower per core) a push now takes about **10 minutes end to end**
  (jobs run in parallel: fast suite 9–10 min, minimum-versions 10 min, lint and
  smoke under 1 min), against about 2 h 30 before.
- **First full run on GitHub found an order-dependent test failure** (Python 3.13,
  4 tests of `test_estim_missing.py`): importing the GUI window before a
  `QApplication` exists makes matplotlib refuse the `QtAgg` backend on a headless
  Linux. It passed whenever another test had built an application earlier in the
  same xdist worker, hence green on every push. The suite now creates one
  `QApplication` for the session (`pmcprg/tests/conftest.py`); reproduced by
  simulating a headless matplotlib, fixed, and checked on the 211 GUI tests. The
  four failures were the only ones on 3.11–3.14 (7 051 passed); the full suite
  takes 47–70 min per Python version on GitHub's runners.
- `pytest-xdist` joins the `dev` extra; the `slow` marker's description now says
  when CI runs those tests. `README.md` and `RELEASING.md` give the parallel
  commands. No test was removed or skipped: the same tests run, less often.

### Changed — importing the package no longer changes matplotlib's settings

- **`import pmcprg.copulas` used to rewrite ten `matplotlib.rcParams`** (dpi,
  figure/axes/savefig facecolor, every font size) through
  `plot_style.apply_style()` run at import — and therefore so did
  `import pmcprg.pmc` and `import pmcprg.diagnostics`, which import it. Every
  figure a user drew afterwards, with no connection to this package, came out
  at 150 dpi and 12 pt. Measured on the previous code: 10 `rcParams` changed
  by `import pmcprg.copulas`; on this code: **none**, for `pmcprg`,
  `pmcprg.copulas`, `pmcprg.diagnostics`, `pmcprg.pmc` and `pmcprg.missing`.
- The package look is now **opt-in**, in three levels
  (`pmcprg.plot_style`): the package's own plot methods (`plot_pdf`,
  `plot_cdf`, `plot_samples`, `plot_h_function`, `plot_overview`,
  `plot_multi_tau`, `plot_diagnostics`, `BivariateLaw.plot_pdf`,
  `plot_pdf_with_margins`) apply it inside a `style_context()` and leave
  `rcParams` exactly as they found it; `style_context()` scopes it to your own
  `with` block; `apply_style()` restyles a whole session on purpose. The GUI
  is an application and now calls `apply_style()` then
  `apply_gui_compact_style()` explicitly.
- **The plots themselves are unchanged**: all nine plot methods were rendered
  with the previous code and with this one and compared pixel by pixel —
  maximum difference 0.0. Only code that *relied* on `import pmcprg.copulas`
  restyling its own figures is affected (the example notebooks now draw with
  matplotlib's defaults); call `apply_style()` to get the old look back.
- **A latent failure fixed on the way:** the GUI enables
  `figure.constrained_layout.use` globally, and in a process that had imported
  it `plot_diagnostics` (which uses `tight_layout`) raised `RuntimeError:
  Colorbar layout of new layout engine not compatible with old engine`.
  Reproduced on the previous code. `style_context()` pins constrained layout
  off, so the package's figures no longer depend on the caller's settings.
  New: `pmcprg/tests/test_plot_style_isolation.py` (20 cases, including
  fresh-interpreter import checks for five packages and the GUI restyle).

### Changed — pickled copulas are 1 000× lighter, and survive registry edits

- **Weight.** Every `CopulaVirt` built three 150 × 150 float64 plotting
  meshes (180 kB each) in its constructor, so a pickled copula was **0.5 MB**
  before holding a single parameter, 1.1–1.6 MB for a rotated or survival one
  (they hold a base copula), a `BivariateLaw` 1.2 MB and a two-state PMC model
  **2.2 MB** — copied into every worker of a multistart and into every
  `copy.deepcopy`. The meshes are deterministic in `N`: they are left out of
  the pickled state and rebuilt on load, bit for bit. Now 268–633 bytes per
  copula (all 41 families), 15 kB per `BivariateLaw`, 16 kB per packaged PMC
  model; classification after a round trip is identical, and 20 deep copies of
  a model take 13 ms.
- **Fragility.** `Enum` pickles by *value*, i.e. the whole registry entry
  (τ-range, parameter names, MODULE path…), so editing any field of an entry —
  this registry has seen a τ-range fixed, parameters added and a module moved —
  made every pickle of that family unloadable. `CopulaEnum` now pickles by
  **name**. A pickle written by an earlier version still loads (tested with a
  by-value payload, and with a copula state that still carries the meshes),
  as long as its entry has not changed since.
- New: `pmcprg/tests/test_pickle.py` (134 cases): every one of the 41
  families round-trips with `logpdf` and meshes bit-identical, deep-copies and
  stays under 4 kB; a fresh interpreter loads the pickle with identical
  numbers; fit results, `BivariateLaw`, `EmpiricalBetaCopula`, test results, a
  PMC model and an ICE result round-trip.

### Fixed — two audit-flagged bugs, both reproduced before the fix

- **The GUI's copula-block dialog could hand back a pair the constructor
  refuses.** `_CopulaDialog.get_block` returned τ and each extra parameter
  straight from their spinboxes, each independently bounded — but some
  families (BB1: `δ < 1/(1 − τ)`, BB6/BB7/BB8, Tawn) constrain the two
  *jointly*, a constraint no single spinbox range can express. Reproduced:
  `CopulaBB1(tau_k=0.5, delta=10.0)` raises `CopulaParameterError` (needs
  `τ > 1 − 1/δ = 0.9`), yet both values are individually in-range and the
  dialog handed them back unchanged. Now routed through the family's own
  `constructible_params` — the same hook `pmcprg.pmc.ice._copula_placeholder`
  already uses for the identical reason: τ is the value the user chose and
  stays, only a refused extra moves, to the nearest value the constructor
  accepts at that τ (`0.5 → 1.9999` in the reproduction above). A pre-existing
  test (`test_copula_dialog_bb1_exposes_delta`) turned out to itself encode
  an invalid pair (τ = 0.5, δ = 2.5, refused) without anyone noticing —
  `get_block` never validated anything, so the round trip "worked" — fixed
  to a genuinely admissible pair.
- **`submodel_lr_test` could not test Tawn 3 against its own Tawn 1/2
  sub-models.** Every previously-registered nesting (BB1 → Clayton, BB6 →
  Joe, …) has a **one**-parameter sub-model, fitted by a 1-D profile over τ
  alone. Tawn 1/2 are themselves **two**-parameter families (their own
  `psi`), so profiling only τ would silently leave `psi` at the
  constructor's default instead of its own MLE, understating the sub-model's
  likelihood and biasing the statistic upward — which is why the pair was
  never registered rather than shipped wrong. Now dispatched on
  `len(PARAMETERS_SET_NAME)`: a one-parameter sub-model keeps the existing
  profile, a two-parameter one is fitted by the same joint MLE
  (`_fit_two_parameter_mle`) the full model uses. `("CopulaTawn3",
  "CopulaTawn1")` and `("CopulaTawn3", "CopulaTawn2")` added to
  `_SUBMODEL_NESTING` (`psi_u = 1` / `psi_v = 1`, each the upper end of
  Tawn 3's registered range — verified against the family's own module
  docstring, not assumed). Size and power measured (Monte Carlo, n = 400):
  rejection at nominal 5 %/10 % under H0 within the existing tests' noise
  band; power against a genuine Tawn 3 alternative (ψ_u = 0.5, ψ_v = 0.9)
  clears 50 % by a wide margin.

### Added — posterior-weighted empirical margins for the copula step of ICE/SEM (audit FR-7 a)

- New ICE/SEM config key **`copula_margins`** (`"parametric"`, the default and
  the unchanged historical behaviour, or `"empirical"`), settable from
  `ice_cfg` / `sem_cfg` and from the TOML `[ice]` / `[sem]` tables; an invalid
  value raises. With `"empirical"` the copula step's pseudo-observations —
  the τ fit **and** the family-selection scores, in the same
  `_select_and_fit_copula` call — are built from the posterior-weighted
  empirical counterpart of *the margin the M-step estimates, with the weights
  it uses*: `F̂_i(y) = Σ_n γ_n(i)·1{y_n ≤ y} / (Σ_n γ_n(i) + 1)` for state
  margins, and for pair margins f_ij (general PMC, DerrodePieczynski_CSDA2013 Eq. 12) the ½ξ-weighted
  **dual-view** sample — `y_n` with weight ½ξ_n(i,j) (left observation of a
  pair (i, j)) and `y_{n+1}` with weight ½ξ_n(j,i) (right observation of a
  pair (j, i)), the same sample the parametric pair update is fitted on, so
  the copula c_ij receives `u_n = F̂_ij(y_n)`, `v_n = F̂_ji(y_{n+1})`. Ties are
  handled with `≤` (tied values share one value), the `+1` keeps every value
  strictly inside (0, 1), and pooling the pair weights over j returns γ.
  The E-step keeps the parametric margins — a step function has no density
  and `c(F̂, F̂)` beside a parametric `f` is not a proper transition kernel —
  so the copula step alone becomes the (weighted) second stage of Chen &
  Fan's (2006, doi:10.1016/j.jeconom.2005.03.004) two-step semiparametric
  estimator. SEM shares it through the same M-step (one-hot weights: exactly
  `#{n : x̃_n = i, y_n ≤ y}/(n_i + 1)`), as do the k-means warm start and both
  missing-data strategies (`"available"`, `"impute"`).
- **Measured** (K = 2 PMC, N(±1, 1) state margins, Gaussian copulas τ = 0.5
  diagonal / 0 off-diagonal, N = 2000, R = 50, 1 % of the y_n moved 7σ away
  from the other state — a true-margin CDF of 1.3e-12, RB-5's extreme
  pseudo-observation in observation space):
  - with **`fit_margins=True`** (the unsupervised case the audit's "IFM is not
    robust" is about) the parametric τ̂ is biased **+0.258/+0.251** on the two
    diagonal pairs and still drifting at `max_iter = 50`, the empirical one
    **+0.100/+0.051** (RMSE ratio 0.62/0.73); with the outliers pushed down
    for both states, +0.345/+0.137 against +0.128/+0.126. The mechanism is not
    the outliers' own pseudo-observations but the ~20 % of scale they add to
    the fitted margins, which compresses every clean point's
    pseudo-observation; ranks do not see that scale;
  - **efficiency cost on clean, correctly specified data**: RMSE ratio
    empirical/parametric **0.99/1.00** with fitted margins (Gaussian copula,
    where the rank estimator is semiparametrically efficient) and **1.64/1.46**
    when the parametric margins are held at the truth. For a Clayton copula,
    1.05/1.07 and 3.38/2.12.
  - **Not covered, and measured as such**: with margins *held at the truth*
    the option does **not** help — inside ICE the E-step re-routes a
    contaminated pair into the blocks that tolerate it (the off-diagonal
    pairs' τ̂ absorbs the damage, +0.15), so RB-5's coordinates never reach the
    diagonal blocks, and the empirical margins — built from all of a state's
    observations while a block keeps only its own pairs — put the outliers at
    the bottom ranks and bias τ̂ up (+0.091/+0.087 against +0.018/+0.018).
- With `"empirical"` the ICE fixed point is no longer a stationary point of
  the (parametric) observed-data likelihood that `trace.log_liks` reports, and
  the standard errors of `pmcprg.pmc._oakes`, `_godambe` and `_lystig_hughes`
  do not apply: they would need a Chen–Fan-type correction for the estimated
  margins, which is **not** implemented. ICE and SEM therefore record a
  non-default value in the fitted model's `[ice]` / `[sem]` table, and the
  three entry points (`ice_oakes_tau_se`, `ice_godambe_tau_se`,
  `ice_lh_information`) raise `NotImplementedError` on such a model.
- The default path is unchanged: with the key absent the M-step is the
  historical one (`pmcprg/tests/test_estim_complete_data_identity.py`'s
  bit-exact golden still passes), and a same-process comparison of "key
  absent" against `copula_margins="parametric"` is pinned in the new
  `pmcprg/tests/test_fr7a_empirical_copula_margins.py`. No GUI widget yet:
  the key is deliberately kept out of `ice_estim_defaults()` /
  `sem_estim_defaults()` (the widget contract) and travels through the TOML
  tables, which the GUI round trip preserves.
### Added — BB7, the Joe-Clayton copula (FR-9)

- **`BB7` (ID 40, `pmcprg/copulas/archimedean/bb7.py`)**, the next BB family
  after BB6 and the registry's fortieth. Generator
  `φ(t) = [1 − (1 − t)^θ]^{−δ} − 1`, `θ ≥ 1`, `δ > 0` — **not** transcribed:
  the candidate was first shown to be a copula at all (uniform margins to
  10⁻³⁰, no negative rectangle volume on an 11 × 11 grid at sixteen (θ, δ),
  at 120 digits in `mpmath`) before anything was built on it. BB7 nests
  **Clayton** at θ = 1 (checked to 4.8·10⁻¹²²) and **Joe** as δ → 0⁺ (error
  exactly linear in δ), so it is a two-way nest, unlike BB6's Joe/Gumbel; its
  tail coefficients are `λ_U = 2 − 2^{1/θ}` and `λ_L = 2^{−1/δ}`, one
  parameter each.
- **Kendall's τ has a closed form**, contrary to the usual claim, and the
  audit's guess of "a Beta form with a removable singularity" is confirmed:
  integrating Genest & MacKay's `1 + 4∫φ/φ′` gives
  `τ = 1 − 2[1 − (2/θ)(δ+1)·B(2/θ, δ+1)]/(δ(2 − θ))`. The **removable
  singularity is θ = 2** — the same θ at which Joe's own τ is 0/0 — where the
  bracket vanishes with `2 − θ`; the limit is `1 − [ψ(δ+2) − ψ(2)]/δ`. A
  second, weaker 0/0 at δ → 0 gives Joe's τ. Both are removed by writing the
  bracket as `1 − e^L` with `L = lnΓ(2+δ) + lnΓ(2+κ) − lnΓ(2+δ+κ)`,
  `κ = 2/θ − 1` — symmetric in (δ, κ), vanishing when either does — and
  expanding `L/(δκ)` in positive-term Hurwitz-ζ series. No quadrature is used.
- **Measured, not estimated.** τ against a 120-digit `mpmath` evaluation of
  the Beta form: |Δτ| ≤ 1.2·10⁻¹⁵ absolute over θ ∈ [1, 10¹²] × δ ∈ [10⁻¹², 10];
  the Beta form itself against an independent Genest–MacKay quadrature of
  BB7's own generator: 9·10⁻⁶¹ to 4·10⁻⁵⁸ relative, the quadrature's limit.
  C, h and ln c against an adaptive-precision (120 → 3840 digits) `mpmath`
  ground truth on `[10⁻¹², 1 − 10⁻¹²]²` at eight (τ, θ): **C ≤ 1.0·10⁻¹⁴**,
  **ln c ≤ 2.4·10⁻¹⁴**, **h ≤ 2.3·10⁻¹³**. The reference ladder itself had to
  be corrected twice (it "converged" at 240 digits on `C = 1` exactly, and on
  a difference quotient of exactly 0) — recorded in the test file, because a
  reference is a measurement too.
- **τ is not monotone in θ**, which is why BB7 keeps **θ** and not δ as its
  free parameter — the one place where it could not simply follow BB1 and
  BB6. At fixed δ, τ starts at Clayton's δ/(δ + 2), *dips*, and only then
  rises to 1; the dip appears at δ* ≈ 3.44121781421 and reaches 0.059 below
  the Clayton value at δ = 20. Keeping δ would have made (τ, δ) two-to-one,
  hidden a slab of genuine BB7 copulas that no (τ, δ) pair can name, and
  handed the joint MLE exactly the flat plateau audit RB-8 is about. τ **is**
  strictly increasing in δ at fixed θ (checked at 50 digits over
  θ ∈ [1, 10⁵] × δ ∈ [10⁻⁸, 10⁴]), so `theta7` is registered and δ recovered
  by Brent. The joint constraint is therefore `τ > τ_Joe(θ)`, i.e.
  `θ < θ_Joe(τ)`, with θ = 1 (Clayton, δ = 2τ/(1 − τ) in closed form) at the
  lower end and the Joe limit at the upper one; **θ = 1 is admissible at
  every registered τ**, so a block without `theta7` always builds.
- **Honest limitation.** The *relative* accuracy of τ degrades in the
  doubly-degenerate corner θ → 1 **and** δ → 0, where τ → 0 and the closed
  form cancels: 7·10⁻⁵ relative at τ = 2·10⁻¹² (still 1.4·10⁻¹⁶ absolute).
  This is weaker than `joe.py`'s τ, which is exact from θ = 1 (audit RB-2);
  no factored form was found and the corner is only reachable when the joint
  constraint already forces δ down to the order of τ.
- Wiring: `CopulaEnum.BB7`, `EXTRA_PARAM_BOUNDS_BY_PARAM["theta7"]`
  (`(1.0, 10.0, 1.0)`), a `theta7` branch of `_fit._two_parameter_spec`
  optimising in `(ln(θ − 1), ln δ)` — a box that *is* the admissible set, with
  δ as the nuisance coordinate — and two `_stderr._SUBMODEL_NESTING` entries
  (BB7 ⊃ Clayton at θ = 1, BB7 ⊃ Joe at δ → 0), both with the one-sided
  `½χ²₀ + ½χ²₁` null. The parameter is named `theta7`, not `delta` or
  `delta6`, because `_two_parameter_spec` dispatches on the parameter *name*
  and each of the three carries a different τ map.
- Tests: `pmcprg/tests/test_bb7.py` (237 cases), and four BB7 cases added to
  the `decimal` reference grid of `test_copula_limits.py`
  (`pmcprg/tests/data/copula_limits_references.json` regenerated), one of
  them the Clayton sub-model and one the Joe end.

### Changed

- `pmcprg/tests/test_copulas.py::test_available_count` and every documented
  copula-family count (README, `pmcprg/__init__.py`) move from 39 to 40
  with BB7, and to 41 with BB8 below.

### Known gaps (BB7)

- No 90°/270° rotation: FR-8 closed on a fixed list of six families that does
  not include the BB6/BB7/BB8 pairs. BB7 reaches τ > 0 only.
- BB7 is not in `_stderr._spec_of`'s Oakes / Lystig–Hughes analytic
  standard-error machinery, which FR-4 defers for every two- and
  three-parameter family beyond BB1 and Student.
- `mle_tau_discrepancy_test` and `dpd_fit` (FR-7) are one-parameter
  estimators and refuse **every** two-parameter family, BB7 included, with a
  documented `NotImplementedError` — Kendall's τ does not identify a second
  parameter. The test suite pins that this is the reasoned refusal (it names
  BB7's own `theta7`), not a crash; lifting it is a separate FR-7 job.
- BB8 and the non-parametric comparison copula remain open under FR-9.

### Added — BB8 (FR-9)

- **`CopulaBB8` (`BB8`, registry ID 41 — 40 was BB7's, integrated first), the
  last of the BB families the audit names** — `pmcprg/copulas/archimedean/bb8.py`, generator
  `φ(t) = −ln([1 − (1 − δt)^θ]/[1 − (1 − δ)^θ])`, θ ≥ 1, 0 < δ ≤ 1. The
  generator was re-derived and checked rather than transcribed: `φ(1) = 0`
  exactly (the numerator at t = 1 *is* the normalising constant) and
  `φ(0) = +∞`, so it is **strict for every admissible (θ, δ)**, δ < 1
  included — the boundary the audit suspected of trouble is not one. The
  closed form `C = (1 − (1 − A)^{1/θ})/δ`,
  `A = (1 − (1 − δu)^θ)(1 − (1 − δv)^θ)/(1 − (1 − δ)^θ)`, agrees with
  `φ^{-1}(φ(u) + φ(v))` to 9.7·10⁻⁵² at 60 digits; margins uniform to
  6.5·10⁻¹⁷ and the density positive (min 0.0498 over the points tried).
- **Sub-models, verified numerically — one popular claim confirmed, one
  refuted.** δ = 1 gives **Joe** *exactly* and with no limit to take
  (`(1 − δ)^θ = 0` makes the denominator 1): `max |C_BB8 − C_Joe| ≤
  1.4·10⁻⁵⁹`, and the two build the same θ bit for bit. θ = 1 gives
  **independence, not Frank** — `φ(t) = −ln t` whatever δ is,
  `|C − uv| ≤ 1.7·10⁻¹⁶` — so the "Joe–Frank" nickname does *not* describe
  any parameter value. δ → 0 at fixed θ is independence too, linearly in δ
  (`max |C/uv − 1|` on the corner grid = 1.5·10⁻³, 1.5·10⁻⁶, 1.5·10⁻⁹ at
  θ = 4, δ = 10⁻³, 10⁻⁶, 10⁻⁹). **Frank appears
  only as a joint limit** θ → ∞, δ → 0 at fixed θδ = κ, where
  `(1 − δt)^θ → e^{−κt}`: `max |C_BB8 − C_Frank|` = 1.2·10⁻², 9.9·10⁻⁴,
  9.7·10⁻⁶, 9.7·10⁻⁸ at θ = 10, 10², 10⁴, 10⁶ (κ = 5), i.e. O(1/θ), and only
  the positive-dependence half of Frank is reached. `_stderr` therefore
  registers **one** nesting, BB8 ⊃ Joe (δ = 1, the upper end of (0, 1], so a
  one-sided null), and not the two BB6 has.
- **Kendall's τ needed a quadrature — the audit's note stands, with a
  caveat.** `sympy` returns the Genest–MacKay integral unevaluated
  (indefinite, definite, with and without `meijerg`); the piece it does
  evaluate comes back as a Lerch transcendent. Summing the series by hand
  gives an exact **₃F₂ closed form**,
  `τ = 1 − (η²/(θ²δ²))·₃F₂(2, 2, 2 − 2/θ; 3, 3; η)` with `η = 1 − (1 − δ)^θ`,
  verified against a 50-digit quadrature of φ/φ′ itself to 6·10⁻⁴⁴–9·10⁻²⁶.
  It is **not usable**: the series converges like `Σ k^{−1−2/θ}`, `mpmath`'s
  own `hyper` raises `NoConvergence` from θ ≈ 10³, and `scipy` has no ₃F₂. So
  τ is computed by a 24-node composite Gauss-Legendre rule on the package's
  existing `graded_mesh`, after the substitution that flattens the integrand
  (`m = −2 ln x`, which makes `G(m) → −e^{−m}` whatever θ is):
  **36 panels, 864 vectorised integrand evaluations, 0.077 ms per τ**. Against
  a 60-digit `mpmath` reference (two independent breakpoint sets agreeing to
  10⁻⁵²) over θ ∈ [1 + 10⁻⁷, 10⁷] × δ ∈ [10⁻³, 1 − 10⁻¹²]: **max absolute
  error 6.0·10⁻¹⁶**; the max *relative* error, 3.9·10⁻⁵, occurs only where
  τ itself is 1.1·10⁻¹¹. δ = 1 does not go through the rule at all — it is
  Joe's own closed form, so the Joe member's τ is Joe's to the last bit.
  Cost of one `fit()` on n = 2000: **≈ 520 τ evaluations ≈ 4.5·10⁵ integrand
  evaluations, 0.04 s** — the τ memo (Tawn's, reused) keeps the constructor
  from re-inverting τ by Brent at every likelihood evaluation, which would
  have multiplied that by ≈ 40.
- **A cancellation the reference itself fell into, recorded as a warning.**
  `ln(A₁/η)` formed as `ln A₁ − ln η` loses 44 of 50 digits at θ = 10³,
  δ = 0.2, and a naive 50-digit `mpmath` reference disagreed with the truth
  in the *third* digit while looking converged — two wrong reference values
  before the cancellation-free form (`log1p` of a ratio, with a `y ≥ ½`
  branch) was put in both the kernel and the reference.
- **Tail dependence: λ_L = 0 always, λ_U = 0 for every δ < 1, and
  2 − 2^{1/θ} at δ = 1** — a genuine discontinuity of the family, not an
  approximation, derived from the finite non-zero slope of `1 − φ^{-1}(s)` at
  s = 0 when η < 1 and checked on the diagonal: `2 − (1 − C(u,u))/(1 − u)` at
  `1 − u = 10^{-k}`, k = 3…8, lands on λ_U and (for δ < 1) falls a decade per
  decade; k stops at 8 because below that the double's resolution of
  `ln(1 − A)` is coarser than the quantity measured.
- **No joint (τ, δ) constraint** — the first two-parameter family here
  without one. τ(θ, δ) rises from 0 at θ = 1 to 1 as θ → ∞ at *every* δ, so
  the admissible set is the full rectangle and `constructible_params` only
  has to clip δ into (0, 1]. Measured: τ(10⁷, δ) = 0.99960, 0.99996, 0.999992,
  0.999998, 0.9999994, 0.9999998 at δ = 10⁻³, 0.01, 0.05, 0.2, 0.5, 1. The
  approach to τ = 1 has Joe's own rate, `1 − τ = O(1/θ)`, but a constant that
  blows up as δ falls: the measured `(1 − τ)·θ` at θ = 10⁷ is **`4/δ − 2`** to
  1.3·10⁻⁷–1.6·10⁻⁵ relative (2.000000, 2.444444, 6.000000, 18.00000,
  77.9998, 397.993 at δ = 1, 0.9, 0.5, 0.2, 0.05, 0.01; δ = 1 is Joe's
  `1 − 2/θ`). Hence the θ cap 10⁷ and the RB-10 convention (build at the cap,
  store the τ realised), and hence BB8 is the **first Archimedean family here
  to declare `reachable_tau_bounds`** — the mechanism Frank, Plackett and the
  extreme-value families use; it is excluded from
  `test_frank_reachable_tau.test_other_families_are_unchanged` and pinned by
  its own test instead, which is what that list is for.
- Registered as `delta8` — neither BB1's `delta` nor BB6's `delta6` — so
  `_two_parameter_spec` cannot route BB8 through another family's τ map; its
  own branch optimises in `(ln(θ − 1), δ)`, a box that **is** the admissible
  set with nothing to project. `EXTRA_PARAM_BOUNDS_BY_PARAM["delta8"] =
  (0.01, 1.0, 1.0)`, the init being the Joe member.
- **Ground truth against adaptive-precision `mpmath`** (480 → 7680 digits, C
  from the generator alone, c and h as its finite differences) on the 6×6
  corner grid `[10⁻¹², 1 − 10⁻¹²]²` at fourteen (τ, δ), 504 points:
  **C ≤ 6.2·10⁻¹⁵, h ≤ 4.5·10⁻¹³, ln c ≤ 1.1·10⁻¹⁴, all relative.** The one
  exclusion is `h` where the truth is below 10⁻²⁹⁰ and no double holds it
  (1.0·10⁻¹¹⁹⁶ at τ = 0.99, δ = 1, (1 − 10⁻⁶, 10⁻¹²); `ln c` there is right
  to 6·10⁻¹⁶).
- **A reference that stops on two agreeing precisions can stop on two wrong
  ones.** At (10⁻¹², 1 − 10⁻¹²), τ = 0.99, δ = 0.5, the mixed finite
  difference of C needs ≈ 230 digits of cancellation; a 240-digit reference
  returned ln c = −407.8454 against the truth −407.836452727273792 — the
  double-precision kernel was right to 2·10⁻¹⁶ and the *reference* wrong in
  the fourth digit. The measurement above starts at 480 digits and requires
  **three** successive precisions to agree to 10⁻²⁵; the 14-row reference
  table of `test_bb8.py` was re-verified under that rule (0 rows changed).
- **Two cancellations found by the corner grid, not suspected in advance.**
  `ln(1 − A)` as `ln D − ln η` is exact towards (1, 1) and catastrophic
  towards (0, 0) — at (10⁻¹², 10⁻¹²) the true value is −10⁻²⁴ while both logs
  are rounded numbers of order 1, so C came out **exactly 0** instead of
  10⁻²⁴; `ln A` plus `log1p(−e^{ln A})` is now used below A = ½. And the
  complement `1 − C = ((1 − A)^{1/θ} − E)/δ` cancels completely at (1, 1),
  where `1 − A = E^θ` exactly; it is now `(E/δ)·expm1(ln(1 − A)/θ − ln E)`,
  whose argument is ≥ 0 everywhere and exactly 0 at the corner.
- Four BB8 cases added to the `decimal` reference
  file of `test_copula_limits.py` (τ = 0.3, 0.5, 0.7, 0.95 with δ = 0.5, 0.2,
  1.0, 0.05 — the third being the Joe sub-model exactly). Regenerating the
  file rewrote every pre-existing byte unchanged.

### Known limitations (BB8)

- δ is weakly identified at moderate n: on n = 2000 from (τ, δ) = (0.5, 0.2)
  the joint MLE returns δ̂ = 0.37 with τ̂ = 0.496 — the likelihood ridge in
  (θ, δ) is long and flat, as it is for BB1 and BB6, and this is worse here
  because θ and δ both control the same slow approach to τ = 1.
- Two-parameter standard errors are **not** implemented for BB8: `_spec_of`
  refuses it exactly as it refuses BB1, BB6 and Tawn (the deferred FR-4
  `_stderr` job). `mle_tau_discrepancy_test` and `dpd_fit` likewise raise
  their existing, documented `NotImplementedError` for any family with more
  than one parameter — BB8 changes nothing there. The generic
  `integral_c_power` (`∫∫ c^{1+α}`) *does* work on BB8 and has **no** closed
  form, as for every non-Gaussian family; measured against its own refined
  design, the default grid is exact to all 12 printed digits for δ < 1 (a
  bounded density, no tail dependence) and only degrades at δ = 1, the Joe
  member: 7.4·10⁻⁶ relative at α = 0.1, 5.8·10⁻⁴ at α = 0.5.
- No 90°/270° rotations (out of scope: FR-8 closed on a fixed list of six
  families).

### Added — the empirical beta copula (FR-9)

- `pmcprg.copulas.EmpiricalBetaCopula` (`pmcprg/copulas/_nonparametric.py`),
  the last item of audit FR-9: a nonparametric comparison tool, Segers,
  Sibuya & Tsukahara (2017, doi:10.1016/j.jmva.2016.11.010 — verified against
  Crossref). It is **not** a 40th `CopulaEnum` family: no τ, no `fit()`, not
  wired into ICE's family selection — given a bivariate sample it *is* the
  nonparametric copula that sample suggests, for checking a fitted
  parametric family's `cdf`/`pdf` against what the data alone say.
- API: `EmpiricalBetaCopula(data)` (raw data or pseudo-observations —
  ranking makes them equivalent) / `.from_data` / `.from_pseudo_obs`;
  `cdf`/`cdf_array`, `pdf`/`pdf_array`, `logpdf`/`logpdf_array` (native, via
  `scipy.special.logsumexp`, with `pdf_array = exp(logpdf_array)` per the
  `CopulaVirt` convention), the two h-functions `h1`/`h1_array`,
  `h2`/`h2_array` (closed form, ``∂C/∂u`` and ``∂C/∂v``), and `sample(n,
  seed=...)` — an **exact** two-stage mixture draw (pick a data index
  uniformly, then two independent Beta draws), not Rosenblatt inversion.
- Ties use the same average-rank convention as `CopulaVirt.fit`'s
  `rankdata(...)/(n+1)` pseudo-observations.
- **Derived and verified, not assumed**: without ties the margins are
  **exactly** uniform — `C_n^β(u, 1) ≡ u` is a deterministic identity of the
  regularised incomplete beta function (`Σ_{k=1}^n I_u(k, n+1-k) ≡ nu`),
  checked to float64 rounding (≤ 4e-15) — a stronger and more useful fact
  than the usual "asymptotically uniform" claim. With ties a small
  discrepancy appears and shrinks (empirically ~n⁻²) as the tied points are
  diluted into a larger sample. The density is evaluated in log space
  (`logsumexp`) because a naive `log(mean(exp(per-point logpdf)))`
  underflows to `-inf` at an extreme corner even though the true log-density
  is finite (concretely demonstrated at n = 500, (u, v) = (1e-30, 1e-30):
  naive gives `-inf`, `logpdf_array` gives −1804.8).
- Tests: `pmcprg/tests/test_empirical_beta_copula.py` — a hand-verified
  n = 3 case, an independent brute-force loop cross-check, h-functions vs.
  numerical derivative, tie handling, the exact/approximate margin-
  uniformity results above, sampler-vs-cdf agreement with a Monte-Carlo
  error bound, the log-space underflow scenario, and n = 2 / n = 300 edge
  cases.
- Out of scope for this version (documented, not silently skipped):
  frequency/posterior weights (no current caller needs them — this module is
  not wired into ICE) and an inverse h-function (not needed: the exact
  two-stage sampler makes Rosenblatt inversion unnecessary).

---

## [1.1.0] - 2026-09-18

### Highlights

- **The registry grows from 17 to 39 copula families.** Six extreme-value
  families with a closed-form or quadrature-based τ (Galambos, Hüsler–Reiss,
  t-EV, Tawn types 1/2/3), BB6, the survival BB1, and **90°/270° rotations
  for all six one-signed families** (Clayton, GH, Joe, BB1, A12, A14):
  negative dependence is representable everywhere the audit asked for it,
  through one generic mechanism (`pmcprg.copulas.archimedean.rotated`)
  rather than per-family code.
- **Four new goodness-of-fit / screening tests** (`pmcprg.diagnostics`):
  radial symmetry (with a 20.7× faster multiplier-bootstrap option), a
  Rosenblatt-transform test, an exchangeability test, and a **dependent
  multiplier bootstrap** for serially dependent pseudo-observations. The
  last one matters for this library in particular: on Markov data the
  i.i.d. bootstrap rejects a true null in 19.8 % of replications at a
  nominal 5 %, the dependent one in 3.2 %.
- **Standard errors of a copula parameter *inside* ICE**, the audit's three
  ranked options, all implemented: Oakes' (1999) observed information,
  Godambe's (2005) IFM correction for re-estimated margins, and Lystig &
  Hughes' (2002) exact joint information over the prior and every pair's
  copula — the last now covering two-parameter families and re-estimated
  margins. On overlapping states their coverage at a nominal 95 % is
  94.8 % (Lystig–Hughes), 73.5 % (Godambe), 67.7 % (Oakes) and 58.1 %
  (naive sandwich); the exact information with the margins *held fixed*
  reaches only 68.7 %, so no single channel of uncertainty accounts for
  the gap on its own.
- **Robustness diagnostics for a fitted copula**: an MLE-vs-τ discrepancy
  test (Hausman-style) and a density-power-divergence estimator,
  `pmcprg.copulas._robust`.
- Two research-library norms reinforced by this batch: every new formula
  was re-derived and checked against an independent high-precision
  reference rather than transcribed (three formulas given in early task
  briefs for this release turned out to be wrong, and were caught this
  way); every new Monte-Carlo test seeds itself deterministically
  (`zlib.crc32`, never Python's salted `hash()`).

(Detailed entries below are unchanged from the development log.)

### Added — a "Scientific lineage" section (README) and foundational references

- The README's model description cited Derrode & Pieczynski (2013, 2016) but
  not the work they build on. New **Scientific lineage** section: Pieczynski
  (2003) introduced the pairwise Markov chain itself; Brunel & Pieczynski
  (2003, 2005) first combined it with copulas; Pieczynski (1992) and
  Delignon, Marzouki & Pieczynski (1997) introduced ICE; Derrode & Pieczynski
  (2004) applied pairwise Markov chains to segmentation. All five DOIs
  resolved against Crossref.
- **`CITATION.cff`**: the two 2003/2005 foundational papers added to
  `references` (informational — citing them is not asked for, only
  DerrodePieczynski_CSDA2013/DerrodePieczynski_SP2016 are).

### Fixed — five tests that held only on the platform that wrote them

- The GitHub CI (Linux x86-64) failed 19 test cases that pass on
  macOS/arm64, all of one kind: numbers compared to the last bit, or to a
  tolerance tighter than libm differences allow. None was a library bug. Each
  drift was measured (CI logs, and a Linux aarch64 container) before any
  tolerance was chosen.
- **`test_pilot_path_is_bit_identical`** (`test_fr4_ice_lystig_hughes.py`)
  and **`test_existing_families_fit_to_the_same_bits`** (`test_tawn3.py`)
  stay bit-exact on macOS/arm64, where their goldens were written. Elsewhere
  Lystig–Hughes compares within measured bounds (log-likelihood 10⁻¹², SEs
  10⁻³ against a drift of 3·10⁻⁶ from the information matrix's
  conditioning). Tawn-3 compares *maxima*, not maximisers: the golden
  parameters' log-likelihood on the fit's own pseudo-observations, to 10⁻³.
  Student's `df` moves by 2 % on its nearly flat likelihood (seed 23) while
  the maximum moves by 1.6·10⁻⁴, in the Linux fit's favour.
- **`test_fr4_ice_oakes_bb1_student.py`**: the scalar and matrix Oakes paths
  reach τ(ψ ± h) through `expit` and `tanh` respectively, and a one-ulp
  disagreement is multiplied by 1/h² = 10⁸ in the second difference —
  10⁻¹⁵ apart on macOS, 2.6·10⁻⁸ on x86-64 Linux. Tolerance 10⁻⁹ → 10⁻⁶.
- **`test_fr4_submodel_lr.py`**: two LR statistics of ≈ 2·10⁻¹⁰ (BB1's δ̂ at
  its boundary on Clayton data) were compared with pytest's default absolute
  floor of 10⁻¹²; now `abs=1e-8`.
- **`test_t_ev.py`**: a λ_U round trip through SciPy's Student-t quantile and
  cdf reached 1.1·10⁻⁹ with the lowest supported dependencies, against
  `rel=1e-9`; now `1e-8`.

### Fixed — documentation back in step with the 39-family registry

- **`README.md`** advertised "17 copula families" in three places and its
  table stopped at Plackett, 22 families behind `CopulaEnum`. The list is now
  grouped by construction (elliptical, Archimedean, survival, extreme-value,
  explicit, 90°/270° rotations), and A12/A14 are labelled *both* tails, not
  *upper* — `λ_L = 2^(−1/θ)` and `1/2` respectively are not zero.
- **`pmcprg/tests/test_docs_copula_registry.py`** (new) fails when a count in
  `README.md` or `pmcprg/__init__.py` diverges from
  `len(CopulaEnum.available())`, or when a registered `SHORT_NAME` is absent
  from the README, naming what to update. Two assertions, ~0.7 s.
- **`_kendall_reference`** (`pmcprg.pmc.gui.main_window`) claimed a closed-form
  `K_θ` "for the Archimedean ones" only. Since FR-9 it also holds for the six
  extreme-value families — `K_θ(t) = t − (1 − τ)·t·log t` whatever the
  Pickands function (Ghoudi, Khoudraji & Rivest 1998), verified against 10⁵
  draws (deviation ≤ 0.0023 vs. a Monte-Carlo se of 0.0016, where Clayton,
  Frank, BB6, survival Joe and Gaussian miss by 0.02–0.07).
- **`MAX_AR_ITER`** (`pmcprg.copulas.bivariate`) was documented as the cap
  "comfortable for all 17 copulas". Its number was not stale, its premise was:
  `sample_conditional` has drawn by Rosenblatt inversion since 0.5.0, nothing
  reads the constant and `SamplingConvergenceError` is never raised. Documented
  as retained for backward compatibility instead of renumbered.

### Added — Lystig–Hughes extended to two-parameter copulas and to re-estimated margins (FR-4)

- **`ice_lh_information(model, Y, fit_margins=…)`**
  (`pmcprg.pmc._lystig_hughes`) now covers the two axes its own report named
  as next. The scaled derivative recursion is unchanged: θ grows, and every
  new coordinate enters through the per-step weight derivatives `W_r`,
  `W_rs` (and, for a margin coordinate, the initial step `a_1 = π ∘ f(y_1)`).
- **Two-parameter families (BB1, Student).** A pair contributes two θ
  coordinates, `psi_ij_0`/`psi_ij_1`, taken from
  `pmcprg.copulas._stderr._spec_of` — the same working coordinate
  (`(log θ, log(δ − 1))` for BB1, `(atanh τ, log(ν − 2))` for Student) and
  the same analytic Jacobian to the reported (τ, δ, θ) / (τ, ν, ρ) that
  `pmcprg.pmc._oakes`'s own two-parameter path uses. The per-observation
  `∂ log c/∂ψ` and the 2×2 `∂² log c/∂ψ∂ψᵀ` (4-point mixed central
  difference off the diagonal) come from the new
  `pmcprg.pmc._oakes._logc_derivatives`, split out of `_hessian_and_phi`
  operation for operation so Oakes' numbers are unchanged. Cross terms
  between two *different* pairs' ψ are exactly zero. **Tawn and t-EV are
  out of scope and refused**: `_spec_of` has no `_Spec` for them, i.e. no
  registered working coordinate and no analytic Jacobian — registering
  those is a `_stderr` job.
- **Re-estimated margins (`fit_margins=True`).** The `2K` coordinates
  `(μ_k, log σ_k)` of the Gaussian state margins — the case
  `pmcprg.pmc._godambe` piloted, chosen so the two methods compare on one
  fixture — join θ between the prior and the copula blocks. A margin
  coordinate enters `log W[n, i, j] = log A_ij + log f_j(y_{n+1}) +
  log c_ij(F_i(y_n), F_j(y_{n+1}))` through **two** paths: the emission
  `∂ log f_k`, analytic (`∂/∂μ = z/σ`, `∂/∂log σ = z² − 1`, and the three
  second derivatives), and the copula's arguments. The second is
  differenced *as a composition* — perturb (μ, σ) by ±10⁻⁴, recompute the
  clipped CDF column, re-evaluate `logpdf_array` — rather than chaining
  `∂ log c/∂u · ∂F/∂η`, which needs no step in `u` near 0 or 1 and no
  chain-rule bookkeeping. Mixed margin × margin and margin × ψ terms use
  the 4-point stencil (divisor `4·h_eta·h_psi` for the latter); mixed
  prior × margin and prior × ψ terms of `log W` are exactly zero.
- **New result fields**: `pair_names`, `estimate`, `se`, `se_partial` (per
  pair, per reported quantity — so BB1's δ and Student's ν get SEs too),
  `margin_names`, `margin_se` (natural units, σ not log σ), `fit_margins`;
  `ci(pair, level, name="tau_k")` gains the `name` argument. A pair whose
  `_spec_of` flags any boundary (τ̂ at a range end, δ̂ = 1, ν̂ at the fitting
  box) is held fixed exactly as a boundary τ̂ already was — its copula stays
  in `W`, it simply leaves θ.
- **The pre-existing path is bit-identical.** The one-parameter ψ stencil
  and the scalar delta method are kept verbatim;
  `test_pilot_path_is_bit_identical` freezes `log_lik`, `grad`,
  `newton_step` and both SE dicts against the constants commit `df4e662`
  produced, and compares them exactly.
- **Exactness** (max relative error against finite differences of
  `forward`'s log-likelihood, N = 600, at parameters away from any fit):
  gradient 2.0·10⁻⁷ (BB1), 1.6·10⁻⁷ (Student), 2.1·10⁻⁸ (Gauss +
  `fit_margins`), 8.7·10⁻⁸ (BB1 + `fit_margins`); Hessian 9.5·10⁻⁷ /
  2.2·10⁻⁵ / 5.1·10⁻⁷ / 2.0·10⁻⁶ against a difference of the exact
  gradient and 1.8·10⁻⁶ / 6.7·10⁻⁵ / 4.0·10⁻⁵ / 2.2·10⁻⁵ against a 4-point
  second difference of ℓ.
- **The exact joint information closes Godambe's residual gap.** Run on
  `pmcprg.pmc._godambe`'s own two fixtures, seeds and `fit_margins=True`
  ICE configuration (R = 300, N = 600, τ = 0.6, pair (0, 0); the Godambe,
  Oakes and naive columns reproduce that module's published numbers to the
  third decimal, so the replicates are identical). Standard,
  heavily-overlapping-states fixture (n = 291): 95 % coverage **0.948** (LH)
  vs 0.735 (Godambe), 0.677 (Oakes), 0.581 (naive) — and 0.687 for LH with
  the margins *held fixed*, so neither channel alone suffices; 90 %
  coverage 0.904 vs 0.674 / 0.605 / 0.519; SE ratio 1.128 vs 0.579 / 0.494 /
  0.403. Well-separated states (n = 300): 95 % coverage 0.937 vs 0.913 /
  0.627 / 0.637. Godambe's diagnosis was right — the missing channel was
  the latent-state one, and it dominates where the states overlap.
- **The 13 % that remains** is the ICE/MLE gap, not a defect of the matrix:
  `I(θ̂)⁻¹` is the *maximum-likelihood* estimator's variance, and ICE's
  margin M-step is the γ-weighted Gaussian MLE, blind to the copula's
  dependence on (μ, σ), so ICE with `fit_margins=True` is a two-stage
  estimator rather than an EM. Measured: the median one-Newton-step
  distance at an ICE fit is 0.51 SE in the margin coordinates against 0.21
  SE in ψ₀₀ (0.40 vs 0.12 on the well-separated fixture), and the median
  score 4.6 against 1.7. The resulting SE error is *conservative*.
- **Monte-Carlo coverage, two-parameter families** (R = 150, N = 600,
  margins fixed, `pmcprg.pmc._oakes`'s own BB1/Student fixtures and seeds;
  n = 135 / 132 after the same boundary exclusions). BB1 — τ: 95 %/90 %
  coverage 0.933/0.919 (LH joint), 0.911/0.904 (LH partial ≡ Oakes),
  0.859/0.815 (naive), SE ratio 1.092 / 0.994 / 0.774; δ: 0.956/0.926,
  0.956/0.896, 0.822/0.748, ratio 1.063 / 0.957 / 0.669. Student — τ:
  0.955/0.917, 0.939/0.886, 0.826/0.735, ratio 1.157 / 0.964 / 0.732; ν:
  0.924/0.902, 0.917/0.902, 0.894/0.864 (the SE ratio is not reported for
  ν, for the reason Oakes' own study gives). LH's *partial* SE reproduces
  Oakes' matrix SE to three decimals for every reported quantity — the
  cross-check that the ψ vector, the stencil and the Jacobian are shared.
- **Cost** (N = 600, K = 2, ms per call; LH always returns the *whole*
  joint matrix): Gauss margins fixed (P = 6) 17 vs Oakes 47 (four pairs);
  Gauss `fit_margins` (P = 10) 31 vs Godambe 51 (four pairs); BB1 margins
  fixed (P = 10) 20 vs Oakes multi 52; BB1 `fit_margins` (P = 14) 32;
  Student margins fixed (P = 10) 31 vs Oakes multi 44 (one pair).
- **Still out of scope**, stated explicitly in the module docstring: **gaps**
  (`pmcprg.pmc.gaps`) on both axes — the recursion would have to run on the
  augmented (state, quadrature-node) chain, and a margin coordinate would
  move the grid itself; **pair margins** (general PMC — the normaliser
  `Σ_k p_ik f_ik(y_n)` no longer cancels, so `∂ log W/∂η` gains a
  y-dependent term); other margin families; Tawn/t-EV.
- **Checks** (`pmcprg.tests.test_fr4_ice_lystig_hughes`, 52 fast tests,
  6 slow): the seven-model exactness grid above, the bit-identity freeze,
  the Oakes cross-checks (scalar and 2×2), the Gaussian margins' analytic
  `∂ log f` and the perturbed-CDF helper against `ice._margin_cdfs`, the
  two boundary conventions, and the scope guards (Tawn, pair margins,
  missing rows with and without `fit_margins`, non-Gaussian margins).

### Added — dependent multiplier bootstrap for serially dependent pseudo-observations (FR-5)

- **`pmcprg.diagnostics.dependent_multiplier`** (new module, re-exported from
  `pmcprg.diagnostics`) supplies the serially dependent multiplier sequence
  FR-5 asks for (Bücher & Ruppert 2013, doi:10.1016/j.jmva.2012.12.002;
  Bücher & Kojadinovic 2016, doi:10.3150/14-BEJ682). Two consecutive pairs of
  a Markov chain share an observation, so a state pair's pseudo-observations
  are not i.i.d. (Darsow, Nguyen & Olsen 1992; Chen & Fan 2006) and the
  i.i.d. multiplier bootstrap of FR-10's three Cramér–von Mises tests is
  calibrated against too narrow a null (Fermanian, Radulović & Wegkamp 2004).
- **Construction and normalisation, derived here and checked numerically**,
  not transcribed: `ξ_i = Σ_{k=0}^{m-1} w_k Z_{i+k}` with `Z` i.i.d. mean 0,
  variance 1 and weights `w_k ∝ φ(k/ℓ)` normalised by `Σ_k w_k² = 1` — the
  normalisation that gives `Var(ξ_i) = 1`, as opposed to the `Σ_k w_k` a
  smoother would use. `n + m − 1` variates are drawn per replicate, so the
  sequence is exactly stationary and `Corr(ξ_i, ξ_{i+h}) = Σ_k w_k w_{k+h}`
  with no edge correction. Measured over 40 000 replicates of length 120:
  mean within 0.01 of 0, variance within 0.01 of 1, and the empirical
  autocorrelation within 0.01 of `Σ_k w_k w_{k+h}` at every lag, for both
  kernels and both laws (`pmcprg/tests/test_dependent_multiplier.py`).
- **`kernel="bartlett"`** (the default) is the rectangular window
  `w_k = ℓ^{-1/2}`, `k = 0 … ℓ-1`, whose self-convolution is *exactly* the
  Bartlett kernel: `ρ(h) = 1 − |h|/ℓ`, and the sequence is exactly
  `ℓ`-dependent. That choice makes `ℓ` and the HAC bandwidth `L` of
  `pmcprg.diagnostics.model_selection` the same quantity — for any series
  `a`, `Var(Σ_i a_i ξ_i) = hac_variance(a, ℓ − 1)`, asserted to within 3 % of
  the exact value over 200 000 replicates rather than merely claimed.
  `kernel="parzen"` is the smoother alternative (`2ℓ-1` weights, reach
  `2ℓ-2`).
- **Block-width selection reuses the existing Newey & West (1994) plug-in**
  (`newey_west_bandwidth`) rather than introducing a second rule, which the
  identity above legitimises: `ℓ = 1 + L_NW`. The one extension is that
  `L_NW` selects for a single scalar series whereas `ℓ` must serve a whole
  empirical process, so `auto_block_length` takes the **median** of `L_NW`
  over a 3×3 quantile grid of centred indicator series
  `1{u_i ≤ a, v_i ≤ b} − C_n(a, b)`. Median, not maximum: measured over 60
  replicates at `n = 200`, the maximum over nine series averages `L = 9.1` on
  i.i.d. data and 8.6 on a chain — no discrimination, all plug-in noise —
  while the median averages 4.2 and 7.8 (6.1 and 14.7 at `n = 800`). The
  plug-in's known upward bias when the true autocovariances vanish is
  documented rather than papered over: on i.i.d. data the rule returns
  `ℓ ≈ 5`, not 1, and the resulting error is in the conservative direction.
- **Exposed as `bootstrap="dependent-multiplier"`** on `radial_symmetry_test`,
  `exchangeability_test` and `rosenblatt_gof_test`, with new keyword-only
  `block_length` (positive int or `"auto"`, the default) and `block_kernel`
  arguments and a new `block_length` field on all three result dataclasses.
  **Every existing default is bit-identical**: `bootstrap="parametric"`
  (still the default) ignores the new keywords entirely, and
  `block_length=1` returns the *same array* the pre-FR-5 i.i.d. path drew, so
  it reproduces `bootstrap="multiplier"`'s statistic and p-value exactly —
  asserted, not approximated.
- **Validated on genuinely serially dependent data**, not i.i.d. samples: each
  replicate simulates a two-state PMC with `pmcprg.pmc.simulate` (persistence
  0.98, Gaussian transition copulas at `τ = 0.7`) and extracts the `(0, 0)`
  state pair's observations by their true hidden labels. Size of
  `radial_symmetry_test` at a nominal 5 %, R = 500, B = 200, `n ≈ 392`
  (±3.3 binomial s.d. ≈ ±3.2 points): **i.i.d. multipliers reject a true null
  19.8 % of the time**, against 3.2 % for the same code on a matched i.i.d.
  sample — FR-5's claim, quantified. The dependent version repairs it:
  12.8 % (`ℓ = 2`), 6.6 % (5), 3.4 % (10), 3.0 % (20), 3.2 % (`"auto"`,
  mean `ℓ = 14.1`), 4.0 % (auto/Parzen), while the i.i.d. control stays at
  4.0–4.8 % at every `ℓ` — a strict generalisation, not a different test.
- **Sensitivity to the block width, both directions.** Raw power against a
  Clayton alternative at `n ≈ 127` falls from 49.6 % (i.i.d.) to 15.2 %
  (`ℓ = 20`), but that comparison is invalid: the `ℓ = 1` end is the test
  whose level is 14.5 % at this `n`. Judged against each method's own
  empirical 5 % null quantile, size-adjusted power is flat — 33.0 %
  (i.i.d.), 33.6 % (`ℓ = 2`), 33.6 % (5), 30.4 % (10), 24.0 % (20), 26.6 %
  (`"auto"`) — so the i.i.d. bootstrap's apparent advantage was the level
  distortion, and over-shooting `ℓ` costs real but modest power. On the
  i.i.d. control, power is 86 % at every `ℓ`.
- `exchangeability_test` on the same null is already far below nominal
  (0.4 % i.i.d., 0.0 % dependent): the over-rejection does not arise for that
  statistic on this design, so the dependent version only adds
  conservativeness there. Reported because it was measured.
- **Left for later, explicitly**: `CopulaVirt.bootstrap_ci` and
  `CopulaVirt.gof_test` (`pmcprg/copulas/_bivariate_fit.py`,
  `pmcprg/copulas/_fit.py`), the other i.i.d. bootstrap FR-5 names, resample
  pairs instead of reweighting an empirical process, so the multiplier
  sequence does not drop into them — a block bootstrap would be the
  analogue. Their coverage under serial dependence is not measured.
- 75 new tests (`pmcprg/tests/test_dependent_multiplier.py`,
  `pmcprg/tests/test_fr5_dependent_multiplier.py`), 4 of them `slow`.

### Added — BB6 copula, the outer power of Joe (FR-9)

- **`CopulaBB6`** (`pmcprg.copulas.archimedean.bb6`, `CopulaEnum.BB6`,
  short name `"BB6"`) — the first of the two-parameter BB families the audit
  still lists, added as the pilot for BB7/BB8 exactly as Galambos piloted the
  extreme-value families. Generator
  `phi(t) = (-ln(1 - (1 - t)^theta))^delta`, `theta >= 1`, `delta >= 1`: the
  **outer power** (Gumbel transform) of Joe's generator, giving
  `C(u,v) = 1 - [1 - exp(-((-ln(1-ubar^theta))^delta +
  (-ln(1-vbar^theta))^delta)^(1/delta))]^(1/theta)` with `ubar = 1 - u`.
  Every formula was re-derived from the generator and checked against an
  independent `mpmath` ground truth (C from `phi^-1(phi(u) + phi(v))`, the
  density and h-function as its finite differences at a step `1e-20` of the
  distance to the edge, the working precision raised from 120 to 3840 digits
  until two successive ones agree to 12 digits) — the package's own closed
  forms appear nowhere in it.
- **Kendall's tau has a closed form**, contrary to the usual claim for the BB
  families. An outer power leaves the Genest–MacKay integral scaled:
  `(phi^delta)/(phi^delta)' = phi/(delta phi')`, so
  **`tau(theta, delta) = 1 - (1 - tau_Joe(theta))/delta`** exactly — the same
  identity that gives BB1 (the outer power of Clayton) its
  `1 - 2/(delta(theta+2))`. Checked against a 60-digit `mpmath` quadrature of
  `1 + 4 int phi/phi'` of BB6's *own* generator (mesh graded towards `t = 1`,
  where the integrand concentrates once `theta` is large) at `theta` in
  {1, 1+1e-7, 1.2, 2, 3.7, 8, 50, 200} x `delta` in {1, 1.4, 2.6, 5}. Compared
  in `mpmath` at 60 digits the identity holds to a relative 0–9e-19, the
  quadrature's own limit; the package's float evaluation matches the recorded
  table to 1e-13 over the whole range. (On a plain [0, 1/4, 1/2, 3/4, 1] split
  the quadrature was itself 8e-5 off at `theta = 50` and the closed form was
  right — recorded in `test_bb6.py` as a reminder that a reference is a
  measurement too.) **No Archimedean
  tau quadrature was added to the package**: the closed form reuses
  `joe._joe_tau_from_theta` / `_joe_theta_from_tau`, whose three expansions
  are exact from `theta = 1` (`tau = 0`) to `theta -> infinity` (RB-2), so BB6
  inherits Joe's accuracy near independence. Against a 40 000-pair Monte-Carlo
  Kendall's tau over 5 seeds at (0.3, 1.2), (0.7, 2.5) and (0.9, 3.0), every
  |t| on the mean is below 1.5.
- **Sub-models, both exact and both boundaries**: `delta = 1` is **Joe** at the
  same `theta` (the constructor uses `tau_Joe = tau` directly there, so the
  member builds the very `theta` `CopulaJoe` builds, bit for bit) and
  `theta = 1` is **Gumbel–Hougaard** at parameter `delta`. Measured over the
  8x8 edge grid `[1e-12, 1-1e-12]^2`: `max |ln c_BB6 - ln c_Joe| = 2.3e-13`
  and `max |ln c_BB6 - ln c_GH| = 5.7e-14`, `max |C_BB6 - C| = 1.1e-16`;
  `theta = delta = 1` gives `ln c = 0` identically. Both sit at the *lower*
  end of the admissible `theta >= 1`, `delta >= 1`, so
  **`submodel_lr_test` accepts BB6 -> Joe and BB6 -> Gumbel** with the
  one-sided null `0.5*chi2(0) + 0.5*chi2(1)` (Self & Liang 1987) and no
  change to that mechanism — two new entries in `_SUBMODEL_NESTING`, which
  now holds six pairs. Unlike Student's `df -> infinity`, neither boundary
  needs a fitting-box end to stand in for it.
- **Tail dependence** `lambda_U = 2 - 2^(1/(theta*delta))` and
  `lambda_L = 0`, both re-derived from `phi^-1` and checked numerically: the
  diagonal ratio `(1 - 2u + C(u,u))/(1-u)` settles on the closed form to 12
  digits, and `C(u,u)/u` falls monotonically to `<= 1.2e-18` at `u = 1e-250`
  (probed on the kernel, which is not clipped at `EPS`; the decay is only a
  small power of `u` — still 0.226 at `u = 1e-9` for `theta = 1`,
  `delta = 10`). `lambda_U` depends on `(theta, delta)` only through their
  product, so it alone cannot separate them.
- **Parametrisation**: `delta` is the extra parameter and `theta` is
  recovered from `(tau, delta)`, as for BB1. `tau_Joe in [0, 1)` forces
  `delta in [1, 1/(1-tau)]` — a *joint* constraint like BB1's `(tau, delta)`
  and Tawn's `(tau, psi)` — but **closed at both ends**: `delta = 1/(1-tau)`
  is `theta = 1`, a Gumbel copula, not BB1's degenerate `theta = 0`. So
  `CopulaBB6.delta_max` returns the largest *double* with
  `delta*(1-tau) <= 1` (walked with `nextafter` in both directions, since
  `1/(1-tau)` can round either side of it — `tau = 0.9` gives a product of
  `1 + 2e-16`, `tau = 1/3` leaves `delta = 1.5` admissible one ulp above),
  and `constructible_params` repairs a refused draw **onto** that end without
  BB1's `TAU_PAD_REL` pull-in. A block without `delta6` is the **Joe member
  `delta = 1`**, admissible at every registered `tau` (Tawn's `psi = 1`
  precedent), not BB1's 1.5, which is inadmissible below `tau = 1/3`; every
  generic per-family sweep in the test suite therefore builds BB6 unchanged.
- **The extra parameter is registered as `delta6`, not `delta`.**
  `_fit._two_parameter_spec` dispatches on the parameter *name*, and the
  `"delta"` branch carries BB1's own `tau = 1 - 2/(delta(theta+2))` map: a BB6
  registered under `delta` would have been fitted with the wrong
  parametrisation, silently, since every `(theta, delta)` pair is numerically
  plausible. This is the reason t-EV's `nu` is not Student's `df`.
  `test_bb6.py` proves BB6 takes its own branch and that BB1's is untouched.
- **Fitting**: `fit` is always the joint two-parameter MLE; `method='tau'`
  logs a warning and falls back to it (BB1's and Tawn's precedent), so the
  two methods return identical estimates. The `delta6` branch optimises in
  `(ln(theta - 1), ln delta)`, whose box **is** the admissible set, with
  `tau` mapped back by the closed form; two stages, the second linear in
  `theta - 1` to reach the Gumbel boundary. Recovery over 25 replicates at
  `n = 3000`: RMSE on `tau` 0.0020–0.0126 (bias `<= 0.0033`), RMSE on `delta`
  0.031–0.375. `delta` is the weakly identified one and its RMSE grows with
  `tau`: 0.031 at (0.2, 1.0), 0.084 at (0.5, 1.5), 0.125 at (0.7, 2.5), 0.235
  at (0.9, 3.0) and 0.375 at (0.9, 10.0). The bias is one-sided by
  construction wherever the true `delta` is on a boundary — `+0.017` and
  `+0.039` at the two `delta = 1` (Joe) points, `-0.25` at
  `(0.9, delta_max = 10)` (Gumbel, and the top of the box) — and `delta` is
  simply weakly identified at high `tau`, `+0.14` at the interior
  `(0.9, 3.0)`.
- **Numerics**: everything is evaluated from `ka = ln(1-u)`, `kb = ln(1-v)`
  and no power is ever formed. Two cancellations were found *by* the ground
  truth and fixed: `ln p` needs the asymptotic branch `ln p = -z` beyond
  `z = -theta*ka = 700`, where `(1-u)^theta` underflows and
  `ln(1 - e^-z)` returns exactly 0 — without it `logpdf_array` is NaN on the
  whole `u = 1 - 1e-12` edge once `theta >= 2000`; and `ln G = ln(1 - e^-S)`
  needs `ln S - S/2` below `ln S = -20`, since `S` underflows in linear scale
  (`ln S = -1068` at `(1-1e-12, 1-1e-12)` for `tau = 0.95`, `delta = 1`,
  where the density is an ordinary `e^29.9`). `p + q - S` and `p - S`, which
  cancel to zero at `delta = 1`, are formed as
  `-exp(ln S + ln1mexp(ln S - ln p))` and bracketed on the smaller of `p, q`
  (Tawn's rule); `ln B` is a two-fold `logaddexp` of three non-negative
  terms, so `ln(theta-1) = -inf` and `ln(delta-1) = -inf` are absorbed rather
  than cancelled. `ln1mexp` is imported from `joe.py` rather than copied.
  Max errors against the `mpmath` ground truth on the 6x6 grid
  `[1e-12, 1-1e-12]^2` at fourteen `(tau, delta)` from `(1e-6, 1)` to
  `(0.99, 1)`: **`ln c <= 1.4e-14`, `C <= 2.9e-14` relative, `h <= 4.6e-13`
  relative**.
- **A `delta` one ulp below `delta_max` used to raise `ValueError`.**
  `tau_Joe = 1 - delta*(1 - tau)` lands on exactly `eps/2 = 1.11e-16` there,
  and `joe._joe_theta_from_tau`'s Brent bracket `[tau/(1-tau),
  (1+tau)/(1-tau)]` has no sign change at that value — `1 + tau/(1-tau)`
  rounds *up* to `1 + eps`, whose `tau` already exceeds the target. It hit 37
  of the 376 `(tau, delta)` pairs the hook test draws and 5 of 9 jittered
  multistart configurations. Joe is left untouched: BB6 snaps any `tau_Joe`
  below `tau_Joe(1 + eps) = 1.288e-16` — below which *no* `theta > 1` is
  representable as a double — to the Gumbel member `theta = 1`, moving `tau`
  by at most `1.288e-16/delta`, under one ulp of `tau`.
- `inv_h` has no closed form: `ln h(v|u) = ln w` is solved by 68 steps of
  vectorised bisection on `logit v` (final bracket `< 1e-19`), never forming
  `1 - v` in linear scale. Round trip `|h(inv_h(w,u)|u) - w| <= 2.5e-13` over
  2000 points at each of eight `(tau, delta)`; the sampler reproduces
  `CopulaJoe`'s and `CopulaGH`'s draws exactly at the two sub-models, with
  uniform margins (KS `p > 0.01`) and no `tau` bias over 20 seeds.
- **Tests and references**: new `pmcprg/tests/test_bb6.py` (216 tests), and
  four new entries in the `decimal` reference grid of
  `pmcprg/tests/test_copula_limits.py` at `tau` = 0.4, 0.5, 0.7, 0.9 with
  `delta` = 1.6, 2.0, 1.0, 3.0 — `(0.5, 2.0)` being the Gumbel sub-model
  (`delta*(1-tau) = 1` exactly in binary, so `theta = 1` to the last bit) and
  `(0.7, 1.0)` the Joe one, so the reference file itself pins both limits.
  Regenerated incrementally: **+144 rows / +152 lines, 0 lines removed** —
  every pre-existing entry byte-identical (161 -> 165 entries, 5796 -> 5940
  rows). The hardcoded family count of `test_copulas.py` goes 37 -> 38.

### Added — robust options outside ICE: MLE-vs-τ diagnostic and density power divergence (FR-7 b, c)

- **`mle_tau_discrepancy_test`** (`pmcprg.copulas._stderr`, re-exported from
  `pmcprg.copulas`, result `MleTauDiscrepancyTest`) is the systematic
  `|θ̂_MLE − θ̂_τ|` diagnostic FR-7 (b) asks for: a Hausman-style
  specification test of "both estimators are consistent for the same τ".
  Kendall's τ̂ has a bounded influence function and the log-density score
  does not (Croux & Dehon 2010, doi:10.1007/s10260-010-0142-z), so under
  contamination or misspecification the pseudo-MLE moves and τ̂ does not.
- The **variance of the difference is not** the Hausman
  `Var(slow) − Var(fast)` shortcut, which needs an efficient estimator under
  H0 — the rank-based pseudo-MLE is not semiparametrically efficient in
  general (Genest & Werker 2002) and the shortcut returns a *negative*
  variance on real samples (pinned: `CopulaFrank(tau_k=0.4)`, n = 600,
  seed 0, −1.36e-06 against the joint construction's +4.10e-06). Instead
  both estimators are written as sums of their influence functions on the
  same sample — `B⁻¹(φ + Ŵ₁ + Ŵ₂)` carried to τ for the pseudo-MLE, and
  `4(ẑ − z̄)` with `ẑ = 2C_n − û − v̂` for Kendall's τ̂ — and `Var(τ̂_MLE −
  τ̂_τ)` is the weighted empirical variance of their difference, O(n log n)
  and deterministic. The two marginal variances it reproduces are *exactly*
  what `standard_errors` reports for `method='mle'` and `method='tau'`, so
  the diagnostic cannot describe a different estimator than the module's own
  standard errors; the measured correlation between the two estimators is
  0.79 (Clayton) to 0.996 (Frank) at τ = 0.4, n = 2000, which is why the
  cross term may not be dropped.
- **Level** (R = 500, six families × τ ∈ {0.2, 0.4, 0.6} × n ∈ {500, 2000}):
  at n = 2000, rejection at nominal 5 % is 3.4-9.0 % and `sd(Z)` 0.94-1.08.
  At n = 500 the statistic carries an O(n^{-1/2}) mean offset (E[Z] ≈ 0.45
  for the Gaussian, 0.15-0.25 elsewhere, halving by n = 2000) that makes it
  mildly liberal for the Gaussian (7.2-8.0 %) and conservative for Frank
  (2.2-3.4 %, `sd(Z)` 0.87-0.93). Reported as measured, not tuned.
- **Power** (n = 1000, R = 500, τ = 0.4, rejection at 5 % for ε = 0/1/2/5/
  10 % of discordant corner pairs): Gaussian 6.2/98.2/100/100/100 %, Frank
  5.6/83.0/100/100/**0.2** %, Clayton 6.0/5.4/5.4/18.2/92.8 %. Two honest
  weak spots: Clayton's pseudo-likelihood is dominated by the lower tail and
  barely notices upper-corner pairs, and Frank's power is **not monotone** —
  at ε = 10 % the contaminated sample really is close to a Frank copula at a
  smaller τ, both estimators agree on it, and the contrast (rightly) stops
  firing. Contamination by draws from an opposite Gaussian copula
  (τ = −0.8) leaves the Gaussian family at nominal throughout (5.4-10.0 %),
  since a mixture of two Gaussian copulas is still nearly one.
- **`pmcprg/copulas/_robust.py`** (new module, re-exported: `dpd_fit`,
  `dpd_objective`, `integral_c_power`, `select_alpha`, `DPDFit`,
  `DPDAlphaSelection`, `DPD_ALPHAS`) implements FR-7 (c), the weighted
  density-power-divergence estimator of Basu, Harris, Hjort & Jones (1998,
  doi:10.1093/biomet/85.3.549), derived here for the copula case:
  `H_n(θ; α) = ∫∫ c_θ^{1+α} − (1 + 1/α) Σ w̄_i c_θ(û_i, v̂_i)^α`, with
  `H_n(θ; 0) = −Σ w̄_i log c_θ` the pseudo-MLE objective. It is **purely
  additive**: `CopulaVirt.fit`, `pmcprg.copulas._fit` and the ICE M-step are
  untouched (pinned by a test that greps their sources).
- The `∫∫ c^{1+α}` integral is the crux and is documented as such. A
  **closed form is derived** for the Gaussian copula —
  `I(ρ, α) = (1 − ρ²)^{−α/2}(1 − α²ρ²)^{−1/2}`, from
  `A = (1 + α)Σ⁻¹ − αI` and `|A| = (1 − α²ρ²)/(1 − ρ²)` — and used both as
  the fast path and as the exact reference the general quadrature is
  measured against. Every other family uses a **composite Gauss-Legendre
  rule in `s = logit u`** with geometrically widening panels (edges 0, ¼, ½,
  1, 2, 4, 8, 16, 32, 36.04 — the representable limit — 12 nodes per panel,
  216 nodes per axis), summed in log space with a maximum subtraction.
  Measured relative error over α = 0.1-0.75: ≤ 4.6e-14 (Gaussian τ = 0.2),
  ≤ 7.9e-9 (τ = 0.5), ≤ 4.6e-4 (τ = 0.9), machine precision for
  Frank/Plackett/AMH/FGM, ≤ 7.3e-4 (Clayton τ = 0.4), ≤ 4.8e-2 (Clayton
  τ = 0.7) and ≤ 2.0e-1 (Clayton τ = 0.85, where the density concentrates in
  a band of width ≈ 1/θ = 0.09 that 216 nodes do not resolve). Rather than
  hide that ceiling, every `DPDFit` carries `integral_rel_error`, the
  discrepancy with a refined design at the fitted parameter. Two further
  exact references are pinned: `I = 1` for the independence copula (which
  also checks that the weights integrate to 1, to 4e-16) and FGM's
  `1 + θ²/9` at α = 1, matched to 1e-12.
- **α selection** (`select_alpha`) minimises the empirical-MSE criterion
  `(τ̂_α − τ̂_P)² + V̂_α/n_eff` of Warwick & Jones (2005,
  doi:10.1080/00949650412331299120) with an explicit pilot (`pilot_alpha`,
  default 0.5). Ghosh & Basu (2015, doi:10.1080/02664763.2015.1016901) is
  unreachable offline, so their pilot-free refinement is **not** claimed to
  be reproduced — the same posture the FR-9 and FR-10 rounds took. `V̂_α =
  K_α/J_α²` is the Basu et al. sandwich on the same grid; it is the
  **known-margin** variance, without the rank-margin correction
  `standard_errors(ranks=True)` applies, so it ranks α values but is not
  reported as a standard error (the field is named `se_hint`).
- A structural limit found and measured while validating, not assumed:
  `K_α` integrates `c^{1+2α}`, so it exists over **half** the α range the
  objective does — for a family with tail dependence (`c ~ C/r` on a corner
  diagonal) it already diverges at α = 0.5. `_dpd_variance` detects this by
  recomputing `V̂_α` with `|logit u|` capped at 24 instead of 36.04 and
  returns NaN when the two differ by more than 1 %: measured at τ = 0.4,
  Clayton/Gumbel/Joe/A12 give a variance to α = 0.25 (change ≤ 1.2e-4) and
  NaN from α = 0.5 (0.39-0.50), Frank and Plackett agree to 4e-10 over the
  whole grid, the Gaussian copula is finite to α = 0.5 and NaN at 0.75. The
  **estimate** is returned in every case; only its variance is withheld, so
  for a tail-dependent family α̂ is in practice chosen from {0, 0.05, 0.1,
  0.25}.
- **Validation of the estimator** (n = 1000). *Clean data* (R = 200,
  τ = 0.4, Gaussian/Clayton/Frank): RMS `|τ̂_α − τ̂_MLE|` and the SD ratio
  `sd(τ̂_α)/sd(τ̂_MLE)` are 5.9e-4 to 8.9e-4 and 1.000-1.003 at α = 0.05,
  1.2e-3 to 1.8e-3 and 1.003-1.006 at α = 0.1, 2.8e-3 to 5.2e-3 and
  1.018-1.051 at α = 0.25, 5.3e-3 to 1.5e-2 and 1.066-1.365 at α = 0.5 —
  against an estimator SD of 0.0147-0.0165, so α ≤ 0.1 costs under 1 % of
  efficiency and moves τ̂ by under a tenth of its own standard error, and
  α = 0 reproduces `fit(method='mle')` to the optimiser's tolerance.
  *Contaminated data* (R = 200, τ = 0.5, bias of τ̂ at ε = 0/1/5/10 % of
  discordant corner pairs): Gaussian MLE +0.002/−0.075/−0.288/−0.457 against
  DPD α = 0.5 −0.003/+0.001/−0.002/−0.047; Clayton MLE
  −0.001/−0.033/−0.160/−0.349 against −0.001/+0.001/+0.003/−0.022; Frank MLE
  +0.001/−0.020/−0.120/−0.270 against +0.001/−0.003/−0.033/−0.112. α = 0.25
  holds to ε = 5 % and breaks at 10 %; α = 0.5 still holds at 10 %.
  Kendall's τ̂ is **not** immune either (−0.27 at ε = 10 %): contamination
  changes the population τ, so downweighting, not ranking, is what protects
  the estimate. *α selection* (R = 100): mean α̂ 0.12-0.15 on clean data and
  0.25-0.50 at ε = 5 %, with the selected τ̂ biased by at most 0.034 where
  the MLE is biased by 0.12-0.29.
- **FR-7 (a) is not in this round.** State-weighted empirical margins
  `F̂_i(y) = Σ_n γ_n(i)·1{y_n ≤ y} / (Σ_n γ_n(i) + 1)` turn ICE into a
  rank-based pseudo-likelihood (Kim, Silvapulle & Silvapulle 2007; Chen &
  Fan 2006) and require changing the M-step in `pmcprg/pmc/ice.py`, which a
  concurrent round owns; parametric margins stay the efficient option when
  validated (Genest & Werker 2002). Both deliverables here are outside ICE
  and leave `ice.py` untouched, so (a) remains open for a later round.
- Tests: `pmcprg/tests/test_fr7_robust_options.py` (42 fast, 13 slow), run
  twice under different `PYTHONHASHSEED`; seeds are integer literals or
  `zlib.crc32`.

### Added — the full three-parameter Tawn copula, and an n-parameter joint fit (FR-9)

- **`CopulaTawn3`** (`pmcprg.copulas.extreme_value.tawn`, `CopulaEnum.TAWN3`,
  short name `Tawn3`) — Tawn's (1988) asymmetric logistic extreme-value model
  with **both** weights free: `tau_k` plus the extras `psi_u`, `psi_v`, each
  in `(0, 1]`. It is the 38th registered family and the **first with three
  parameters**; `n_params = 3`, so every information criterion penalises it
  accordingly. Nesting: `Gumbel ⊂ {Tawn 1, Tawn 2} ⊂ Tawn 3`, with
  `psi_u = 1` giving type 1 and `psi_v = 1` giving type 2.
- The evaluation kernel needed **no change at all**: `_tawn_parts`,
  `_tawn_logpdf`, `_tawn_log_cdf`, `_tawn_logh`, `_tawn_A_terms` and
  `_tawn_tau_quad` already took both weights (round 3 fixed one weight in the
  parameter layer only). Round 5 split the round-3 body of `_CopulaTawn` into
  a shared `_TawnBase`, so the restrictions are reproduced **bit for bit**,
  not to a tolerance: at `psi_u = 1.0` the three-parameter member runs the
  same floating-point operations as `CopulaTawn1`, and `ln c`, `ln C`, `ln h`,
  `theta` and `tail_dependence()` compare equal exactly on the corner grid
  `(1e-12 … 1 − 1e-12)²`.
- **The general reachable-τ cap, re-derived rather than transcribed.** As
  `θ → ∞` the Pickands function tends to Marshall–Olkin's, whose only
  curvature is a slope jump of `ψ_u + ψ_v` at `t* = ψ_v/(ψ_u + ψ_v)`; with
  `A(t*) = 1 − ψ_uψ_v/(ψ_u + ψ_v)` the Genest–MacKay identity gives
  `τ_∞ = ψ_uψ_v/(ψ_u + ψ_v − ψ_uψ_v) = 1/(1/ψ_u + 1/ψ_v − 1)`. The
  round-3 statement `τ < ψ` is its `ψ_other = 1` case — **confirmed**, not
  contradicted. Checked against the quadrature on eleven weight pairs: the
  relative gap `(τ_∞ − τ(θ))/τ_∞` is `1e-7 … 1.1e-6` at `θ = 1e6` and shrinks
  by a factor 10 from `θ = 1e5`, and τ is increasing in θ on 300-point
  log-grids from `1 + 1e-6` to `1 + 1e6` (so τ_∞ is a supremum, not just a
  limit point). Examples: `(0.6, 0.8) → 0.521739130435` vs
  `τ(1e6) = 0.521738858188`; `(0.3, 0.3) → 0.176470588235` vs `0.176470557088`.
- `reachable_tau_cap` uses **both** algebraic forms, each where it is the
  exact one: the reciprocal in general (the product `ψ_uψ_v` underflows to 0
  below ≈ 1e-154, which the constructor accepts), and the restriction faces
  returned directly — `1/(1 + 1/0.9 − 1)` is `0.8999999999999999`, one ulp
  below 0.9, and a τ in that gap would be refused by `CopulaTawn3` while
  `CopulaTawn1` accepted it at the same parameters.
- **`CopulaTawn3.constructible_params` / `repaired_psis`** repair a refused
  triple so that no multistart draw, family draw or ICE placeholder ever
  builds one. Both weights travel together towards the Gumbel corner along
  `ψ_i(s) = 1 − s(1 − ψ_i)`, which keeps the direction of the asymmetry the
  draw expressed; the cap equation becomes the quadratic
  `(1 + τ)·P·s² − S·s + (1 − τ) = 0` (`S = Σ(1 − ψ_i)`, `P = Π(1 − ψ_i)`),
  whose smaller root is taken in the cancellation-free form
  `s* = 2(1 − τ)/(S + √(S² − 4P(1 − τ²)))`, and the repair goes to `s*/2` —
  the middle, since the boundary itself is the singular Marshall–Olkin limit
  `θ → ∞`. On the face `P = 0` this collapses to `(1 + τ)/2`, i.e. round 3's
  own `repaired_psi`, and is *evaluated* as that expression so the two rules
  agree bit for bit. An accepted triple is returned untouched (same object),
  as `CopulaVirt.constructible_params`'s contract requires.
- **The joint MLE is now n-parameter.** `_fit_two_parameter_mle`
  (`pmcprg.copulas._fit`) never knew the dimension — it only calls the stage
  maps and hands the box to L-BFGS-B — so the driver was reused unchanged;
  `_two_parameter_spec` became a dispatcher that hands a family with two or
  more extras to the new **`_multi_extra_spec`**. Tawn 3 fits in
  `(ln(θ − 1), ψ_u, ψ_v)`, a box that *is* the admissible set (every
  `θ > 1`, `ψ ∈ (0, 1]` is a Tawn copula), run twice as types 1/2 and t-EV
  are; a generic `(τ, x₁, …, xₙ)` fallback covers any future family.
- **No existing family's numbers moved.** The dispatch is an early return
  taken only when a family declares more than one extra, so for BB1,
  SurvivalBB1, BB190, Student, Tawn 1/2 and t-EV the executed code is
  unchanged — verified empirically, not merely argued: the fitted parameters
  and log-likelihoods of 15 cases through 4 entry points (`fit(method='mle')`,
  the unweighted and weighted driver, and the ICE M-step
  `_fit_copula_params`) are **byte-identical** to the same run against commit
  `df4e662`, iteration and evaluation counts included. The 15 `fit` results
  are pinned as exact `float.hex` literals in
  `test_tawn3.py::test_existing_families_fit_to_the_same_bits`.
- **ICE, multistart and the GUI needed no code change**, which this round
  verified rather than assumed: `EXTRA_PARAM_BOUNDS` is derived from
  `EXTRA_PARAM_BOUNDS_BY_PARAM` + the registry and is iterated everywhere
  (M-step, jitter, `_set_copula_family`, `_copula_placeholder`), and
  `_CopulaDialog` pre-builds one spin box per distinct extra name across all
  families — so `psi_u`/`psi_v` get correct widgets, ranges and defaults
  automatically. New tests pin all of it, including the counterfactual that
  with `constructible_params` neutralised the same multistart draws do raise.
- **Validation.** Independent `mpmath` ground truth (written from
  `C = exp(−ℓ)`, itself cross-checked against `mpmath.diff` of `C` to
  ≤ 3.4e-12) over 12 `(τ, ψ_u, ψ_v)` triples × the 36-point corner grid, both
  `ψ = 1` edges and the box's lower bound 0.01 included: max relative error
  **2.1e-14** on `C`, **1.5e-13** on `h`, **2.2e-14** on `c`, and
  **2.2e-14 nat** on `ln c`. τ by quadrature vs Monte-Carlo Kendall's τ at
  n = 200 000: `|Δ| ≤ 0.0021` (2·se ≈ 0.003). Recovery at n = 3000 over 40
  replicates a point, 320/320 fits converged: RMSE on `(τ̂, ψ̂_u, ψ̂_v)` is
  `0.008/0.007/0.010` at τ = 0.7 and `0.009/0.019/0.013` at τ = 0.4, but
  `0.016/0.077/0.096` at τ = 0.2 and `0.011/0.246/0.343` at τ = 0.05 — **the
  weights are not identified below τ ≈ 0.1**, where the copula is close to Π
  whatever the weights; τ̂ stays accurate throughout (RMSE ≤ 0.016). This is
  documented in the module docstring, not hidden.
- Exchangeability (`pmcprg.diagnostics.exchangeability`, FR-10) behaves as the
  model predicts — `ℓ` is symmetric in `(w, z)` exactly when `ψ_u = ψ_v`. At
  n = 500, B = 200 multiplier replicates, 40 samples per point, α = 0.05:
  rejection 5 % at `(0.9, 0.9)`, 15 % at `(0.7, 0.7)` and 10 % at the Gumbel
  corner (at or somewhat above nominal — the CvM statistic's own
  finite-sample behaviour at this n), against **100 %** at `(1.0, 0.6)`,
  `(0.95, 0.55)` and `(0.9, 0.4)` for τ ≥ 0.3. At τ = 0.15 the same
  `(0.9, 0.4)` asymmetry is seen only 15 % of the time — the same flatness
  near independence that limits estimation. This is the first family here
  whose asymmetry is a *free parameter* rather than a rotation artefact.
- Standard errors are deliberately **not** implemented:
  `pmcprg.copulas._stderr` raises `NotImplementedError` for `n_params = 3`,
  since a Wald interval on a weight that the likelihood barely constrains
  would be misleading.
- `CopulaTawn3` is deliberately **kept out of**
  `pmcprg/tests/data/copula_limits_references.json`: that file's key is
  `(family, θ, δ)` with a single slot for a second parameter, and widening it
  would rewrite all ~5 900 pre-existing rows. The `mpmath` ground truth above
  covers the same corner grid with the same independence-from-the-code
  discipline, and every stored Tawn 1/2 row is also a Tawn 3 row at a `ψ = 1`
  edge by the bit-identity above.

### Added — ICE/SEM missing-observations widgets in the GUI

- `_IceTab` (`pmcprg.pmc.gui.tabs`) gains a "Missing observations" section
  with three widgets for the ICE config keys that `ice()`/`sem()` already
  accepted but that had no way to be set from the interface: a
  `missing_strategy` combo (`"available"` / `"impute"`, driven by
  `pmcprg.pmc.ice.MISSING_STRATEGIES`), a `missing_draws` spinbox (default
  from `DEFAULT_MISSING_DRAWS` = 5), and a `missing_seed` spinbox (default
  0, range `0`–`2_147_483_647` like the other seed spinboxes). Previously
  these three keys could only be set by hand-editing the underlying
  TOML/dict, even though the GUI already loads and plots gapped data.
- `missing_draws` and `missing_seed` are disabled (greyed out) whenever
  `missing_strategy != "impute"`, the same convention as the K-means-seed /
  `init` and multistart / `n_starts` pairs already in the tab. Tooltips
  cite `ice.py`'s own measured tradeoff: `"available"` is deterministic and
  3-4x (grid variants) to 8-19x (GICE) faster than `"impute"` with 5 draws,
  while `"impute"` recovers the true GICE margin families in 57 % of fits
  on `sp2016_gice_k2` against 39 % for `"available"` (61 % on complete
  data).
- All three keys round-trip through `_IceTab.load()` / `get_cfg()` exactly
  like the tab's other keys, so they reach `ice()`/`sem()` via the same
  `PMCMainWindow._do_estimate` path already exercised by
  `test_gui_estimation_entry_point_accepts_missing_values`, now
  parametrized over both strategies.

### Added — likelihood-ratio tests of two-parameter families' sub-models (FR-4)

- **`submodel_lr_test`** (`pmcprg.copulas._stderr`, re-exported from
  `pmcprg.copulas`) tests whether a two-parameter family's extra parameter
  is needed, i.e. whether the data are consistent with the one-parameter
  sub-model nested inside it — the analogue, for a two-parameter family, of
  `independence_lr_test`'s test of independence in a one-parameter family.
  It fits the full family by joint MLE
  (`pmcprg.copulas._fit._fit_two_parameter_mle`, the same routine
  `CopulaVirt.fit(method='mle')` uses on a two-parameter family) and the
  sub-model by the 1-D profile `independence_lr_test` already uses,
  ``LR = 2 max(ℓ_full − ℓ_sub, 0)``.
- Three nestings are implemented, each verified numerically against the
  family's own module docstring rather than assumed: **BB1 → Clayton**
  (`delta = 1`) and **BB1 → Gumbel–Hougaard** (`theta → 0`, `delta =
  1/(1 − tau)`; `pmcprg.copulas.archimedean.bb1` — this module's own K-10
  note on a previously crossed statement of BB1's tail-dependence limits is
  a reminder not to transcribe such a claim), **Student → Gauss** (`df` at
  the fitting box's upper end, `EXTRA_PARAM_BOUNDS_BY_PARAM['df'] =
  (2.001, 100.0)`, standing in for the unreachable `df → ∞`), and **Tawn
  (type 1 or 2) → Gumbel–Hougaard** (`psi = 1`, the upper end of the
  registered `psi ∈ (0, 1]`). All three sit at a *boundary* of the full
  family's admissible extra-parameter range (not interior), so
  `LR → ½χ²₀ + ½χ²₁` (Self & Liang 1987) in every case — the mixture
  `SubmodelLRTest.null_distribution` always reports, unlike
  `independence_lr_test`'s null, which depends on the one-parameter family.
- **Monte-Carlo validation** (`pmcprg/tests/test_fr4_submodel_lr.py`, n =
  400, 400 replicates): empirical rejection under each sub-model's own null
  at nominal 5%/10% — BB1 vs Clayton 4.5%/9.0%, BB1 vs Gumbel 4.0%/9.25%,
  Student vs Gauss 5.75%/13.75%, Tawn1 vs Gumbel 5.75%/10.25% — all within
  Monte-Carlo reach of nominal; power away from the sub-model (BB1 delta=3,
  BB1 theta at delta=1.2, Student df=5, Tawn1 psi=0.5) at least 96% at the
  5% level (100% for three of the four cases).
- Student → Gauss is handled with the same finite-box convention as
  everywhere else in this module (`standard_errors`'s own boundary flag
  already treats `df`'s upper bound as standing in for `df → ∞`): no
  separate treatment or caveat was needed for this nesting.

### Added — Chen & Fan correction, ω² pre-test and a genuine-ICE level study, closing FR-6

- **`vuong_test(..., ranks=True, uv=, dm_du=, dm_dv=)`** (`pmcprg.diagnostics.model_selection`):
  a Chen & Fan (2006)-style correction to the variance of the Vuong statistic
  on rank-based pseudo-observations, adapted from the ``W₁, W₂`` margin
  correction `pmcprg.copulas._stderr.standard_errors(ranks=True)` already
  applies to its sandwich estimator — same ``O(n log n)`` suffix-sum
  construction (`_upper_weighted_sums`), applied to the log-density
  *difference* `m_n = log c_A − log c_B` instead of one family's score. The
  new helper `margin_correction_derivatives(copula_a, copula_b, uv)` supplies
  the required `∂m/∂u`, `∂m/∂v` by central finite differences in logit space,
  mirroring `_stderr._derivatives`. Not a term re-derived word for word from
  Chen & Fan's paper (offline; documented in `vuong_test`'s docstring) —
  validated by simulation instead: on Gaussian-copula data with rank
  pseudo-observations (Clayton vs survival Clayton, N=2000, 300 replicates)
  the level stays within the module's already-documented 9-11% band without
  regressing, and the corrected Z's standard deviation moves back toward 1.
  Default unchanged (`ranks=False`).
- **`omega2_test` / `vuong_test(..., pretest_omega2=True)`**: Vuong's (1989)
  own recommendation to test `H0: ω² = 0` before trusting the normal
  Z-test — when the two fitted densities are observationally near-equivalent,
  the normal limit is invalid. The exact construction (a weighted sum of χ²
  variables from both models' score covariance and information matrices) is
  **not implemented** — it needs score vectors and Hessians this module's
  generic log-density inputs do not carry. What is implemented is a
  documented, honestly-limited substitute: a one-sided HAC test of
  `H0: E[d_n²] = 0` on the same centred series the main test's `σ̂²` comes
  from. Measured (`pmcprg/tests/test_model_selection_mc.py`): it reliably
  does not reject only when ω² is *exactly* zero (a family against itself,
  `m_n ≡ 0`); on Vuong's own overlapping-models example (Gaussian vs
  Student, ν̂ often at its upper bound) ω² is small but strictly positive,
  and the pre-test rejects `H0: ω² = 0` there about 95% of the time
  (N=1000, 100 replicates) — it does not, in practice, screen out that
  near-degenerate case the way Vuong's exact construction would. Attached to
  the result as `ComparisonResult.omega2`; when it fails to reject, the
  decision is short-circuited to `"tie"` (`method="omega2_pretest"`).
  Default unchanged (`pretest_omega2=False`).
- **Level under a genuine ICE E-step** (`test_ice_weight_level_from_a_genuine_e_step`,
  `pmcprg/tests/test_model_selection_mc.py`): the module's own level
  measurements (2.3% Kish, 5.0% Σw) used weights independent of the data;
  this test runs `ice_pair_comparisons` on real simulated data from a 2-state
  PMC with a Gaussian (radially symmetric) transition copula — Clayton vs
  survival Clayton are equally close to it, the module's usual exact null —
  and measures the level on the resulting genuine, data-dependent ξ from a
  real forward-backward E-step (a single E-step/M-step pass at the true
  parameters, not run to ICE convergence, to stay within budget). Measured
  over 200 replicates: Vuong ≈ 2.5% (conservative, consistent with the Σw
  convention), Clarke ≈ 64.5% (confirms, on real ICE weights, the
  estimated-parameters bias already documented for Clarke's test).
- `ComparisonResult` gained two informational fields, both `None`/`False` by
  default and backward compatible: `ranks` (whether the Chen & Fan
  correction was applied) and `omega2` (the `Omega2Test` result, when
  `pretest_omega2=True`).

### Added — multiplier bootstrap for Rosenblatt and exchangeability, closing FR-10

- **`bootstrap="multiplier"` added to `rosenblatt_gof_test` and
  `exchangeability_test`** (`pmcprg.diagnostics.rosenblatt`,
  `pmcprg.diagnostics.exchangeability`), alongside the unchanged
  `bootstrap="parametric"` default, mirroring `radial_symmetry_test`'s own
  `bootstrap`/`multiplier` parameters exactly. This closes FR-10's
  multiplier-bootstrap request (Kojadinovic & Yan 2011,
  doi:10.1007/s11222-009-9142-y) for all three Cramér–von Mises screening
  tests in this package (radial symmetry, Rosenblatt, exchangeability).
- **Exchangeability**: `T_n = n·mean_i[C_n(û_i,v̂_i) − C_n(v̂_i,û_i)]²` needed
  the *same* multiplier linearisation as radial symmetry's own `T_n`
  (`α_n(u,v) − α_n(1−u,1−v)` under H0, with plug-in `Ċ_1, Ċ_2` derivative
  corrections), with the reflected reference `(1−u, 1−v)` replaced by the
  transposed one `(v, u)` — derived and sign-checked rather than assumed to
  transfer mechanically (the query points are swaps of the observed pairs,
  not reflections, so no boundary or orientation surprises). Speed-up at
  N=200, B=200: parametric 0.113 s vs multiplier 0.0024 s (**48×**). Size
  (N=200, 100 reps, B=150, α=0.05): Gaussian τ=0.3/0.6 → 2.0%/2.0%; Frank
  τ=0.3/0.6 → 6.0%/4.0%; Clayton τ=0.3/0.6 → 4.0%/4.0% — comparable to the
  parametric bootstrap's own numbers for the same grid (4.0%/1.0%,
  5.0%/11.0%, 8.0%/3.0%), all within Monte-Carlo reach of nominal. Power
  (90°-rotated families, same reps/B): `CopulaClayton90` τ=−0.2/−0.4/−0.6 →
  16%/65%/73% at N=100, 67%/98%/100% at N=300 (parametric: 24%/61%/88% and
  61%/99%/100%); `CopulaGH90` → 5%/14%/23% (N=100), 25%/57%/56% (N=300)
  (parametric: 10%/20%/15% and 25%/56%/61%); `CopulaJoe90` → 24%/71%/81%
  (N=100), 71%/97%/100% (N=300) (parametric: 31%/71%/86% and 81%/100%/100%)
  — comparable magnitude throughout, as expected from a shared derivation.
- **Rosenblatt**: a genuinely different linearisation was needed, not a
  copy of the copula-process one, because `S_n` compares an ordinary
  empirical CDF of the Rosenblatt-transformed sample `{(Û_i,V̂_i)}` to a
  *fixed* reference `Π(u,v) = u·v`, not to a data-dependent functional of
  itself — the classical one-sample multiplier CLT for an empirical process
  against a known reference, no `Ċ_1, Ċ_2` plug-in needed (see the module
  docstring for the argument this is a different case in kind from radial
  symmetry's and exchangeability's own). **Documented simplification**: the
  construction treats the fitted `θ̂` as fixed, omitting the
  parameter-estimation correction a fully rigorous residual-empirical-process
  treatment would need (no citation available offline to transcribe it, and
  no in-house derivative-of-`h`-by-`θ` machinery to build it from scratch).
  Speed-up at N=200, B=200: parametric 0.550 s vs multiplier 0.0037 s
  (**149×**). This omission has a measurable, honestly-reported cost: the
  simulation shows the multiplier bootstrap's replicate variance runs larger
  than the true sampling variance (the classical direction, Durbin 1973, for
  omitting a residual-process correction), making it markedly
  **conservative** — size (N=200, 200 reps, B=150, α=0.05) at essentially
  0.0% for Gauss/Clayton/Frank τ∈{0.3,0.6} (parametric: 5.3%/5.3%, 4.7%/5.3%,
  6.7%/7.3%, all near nominal), and power correspondingly reduced: true
  Clayton / fit Gauss, τ=0.2/0.4/0.6 → 0%/0%/0% at N=100, 0%/0%/26% at N=300
  (parametric: 19%/52%/85% and 45%/98%/100%); true GH / fit Gauss, τ=0.2/0.4/
  0.6 → 0%/0%/0% at N=100 (parametric: 6%/7%/11%) — a substantial, quantified
  loss of power in exchange for the speed-up. `bootstrap="parametric"`
  remains the recommended default for this test when power matters;
  `bootstrap="multiplier"` is offered for cost-constrained rapid screening
  where its conservative bias (never over-rejecting) is an acceptable
  trade-off.
- **Regression tests**: for both modules, `bootstrap="parametric"` (the
  default, unchanged) reproduces bit-identical results to before this
  option existed (`test_multiplier_bootstrap_default_unchanged` in both
  `pmcprg/tests/test_rosenblatt.py` and `pmcprg/tests/test_exchangeability.py`),
  plus unit tests for the new option, a wall-clock comparison, and
  reduced-budget size/power studies with `@pytest.mark.slow` full-grid
  mirrors — structured exactly like `test_radial_symmetry.py`'s own
  multiplier-bootstrap section.
- `ruff check pmcprg` clean; full fast suite (`pytest -q -m "not slow"`,
  5022 tests) passes.

### Added — survival BB1 and its own 90°/270° rotations, closing FR-8 completely

- **Survival BB1 and its own 90°/270° rotations**, closing FR-8 completely:
  the audit's own list of families needing 90°/270° rotations named "BB1 de
  survie" (survival BB1) separately from plain BB1, and it was the one
  family still missing after the previous round's six one-signed families.
  `SurvivalBB1` (`pmcprg.copulas.archimedean.survival`) is BB1's 180°
  rotation — τ unchanged, tail roles swapped (λ_L ↔ λ_U) — and
  `SurvivalBB190`/`SurvivalBB1270` (`pmcprg.copulas.archimedean.rotated`)
  are its 90°/270° children, registered as three new `CopulaEnum` entries
  (bringing the registry to 37). `delta` passes through `SurvivalCopula`
  unchanged, same as `tau_k` already did; `SurvivalBB1` is the first base
  family `SurvivalCopula` wraps with a joint `constrain_params`/
  `constructible_params` constraint (BB1's own `δ < 1/(1 − τ)`), which
  needed a new generic delegation added to `SurvivalCopula` itself (audit
  G1, mirroring the fix `RotatedCopula` needed for plain BB1's own
  rotations). The 180° and 90°/270° identities hold to 10⁻¹⁰–10⁻¹⁶; a
  numerically verified (not assumed) finding: `SurvivalBB190` and
  `SurvivalBB1270` are the same copula as `CopulaBB1270`/`CopulaBB190`
  (plain BB1's own rotations, swapped) at the same (τ, δ) — the dihedral
  composition 180° ∘ 90° = 270° for an exchangeable base — registered as
  their own families regardless, per the audit's own list. Reference file
  extended by pure addition (9 new entries, 0 changed bytes in the 161
  pre-existing ones).

### Added — 90°/270° rotations of A12 and A14, closing FR-8 for all six families (FR-8, last round)

- **A12 and A14 did not expose the kernel interface** either; both were
  refactored first, as BB1 was, with **every public value unchanged bit for
  bit**: `pdf`, `cdf`, `cdf_array`, `pdf_array`, `logpdf_array`,
  `conditional_cdf` and `sample` compared byte for byte with the pre-refactor
  modules on 3 725 points (grid down to 10⁻³⁰⁰ and 1 − 2.2·10⁻¹⁶) at 9 τ per
  family, 0 mismatches. The 132 existing tests selected by `-k "A12 or A14"`
  across 11 files (`test_copula_limits.py`, `test_palier1_*`,
  `test_scientific.py`, `test_pdf_array.py`, …) give the same results
  before and after. **Kernel coordinates:** A14's is `log u`, like GH's and
  BB1's (its generator `(t^{−1/θ} − 1)^θ` is a function of `log t` alone).
  A12's is the **pair** `(log u, log(1 − u))`: its generator `(1/t − 1)^θ`
  and every formula of the module use both logarithms separately, the
  reflection is the swapped pair (exact), and `rotated.py` never looks
  inside a kernel coordinate. A scalar `log((1 − u)/u)` would have worked
  too, but only by recomputing `log u` differently from the existing code.
- **A shared inverse h-function.** In their own transformed coordinate
  `a` (A12: `(1 − u)/u`; A14: `u^{−1/θ} − 1`), both families have
  `h(v|u) = (a/t)^{θ−1} ((1 + a)/(1 + t))^p`, p = 2 for A12 and θ + 1 for
  A14. `_k_inv_h` solves `h = w` by a monotone Newton iteration in
  `D = log(t/a)` (convex increasing, closed-form upper bounds, as in GH's).
  It is vectorised and accurate to one ulp of v (h at the returned v
  brackets w between neighbouring floats, up to τ = 0.97), unlike BB1's
  Brent loop. The `1 − h` member of `_k_h` uses the matching
  cancellation-free form `−(θ−1)D − p·log1p(expm1(D)·r)`. The public
  `inv_h`/`inv_h_array` of the bases are still `CopulaVirt`'s (Brent,
  ≈2.5·10⁻⁹ in v); `_k_inv_h` only serves the rotations. As a side effect,
  sampling a rotated A12/A14 is fast (40 000 points in ≈20 ms).
- **`CopulaA1290`/`CopulaA12270`/`CopulaA1490`/`CopulaA14270`**, the same
  two-line subclasses as every earlier round, registered as
  `CopulaEnum.A1290`/`A12270`/`A1490`/`A14270` (IDs 31–34) with
  `TAU_MIN_MAX = [−1, −1/3]`, the mirror of `[1/3, 1]`. As one-parameter
  families they need no `fit` override: `CopulaVirt.fit` works on the
  padded registered range directly.
- **The first rotated range that does not touch τ = 0**, checked in every
  range consumer. `constructible_tau_range` pads only the singular end
  (→ `[−0.99993, −1/3]`); `correct_tau` clips 0.5 or −0.1 to −1/3, not to 0.
  A `"sweep"` multistart start (τ = 0.5) lands at −1/3 (θ = 1). `"random"`
  draws fall in `[−13/15, −7/15]`. The selection placeholder pulls τ = 0
  onto −1/3. On a τ = −0.6 sample (n = 800) the ICE M-step returns −0.592
  and the Huard evidence is finite. On independent data `fit`/ICE stop at
  −1/3. All of this worked unchanged. **One check did not generalise:**
  `pmcprg.copulas.independence_lr_test` recognised independence only at a
  *lower* range end (`0 ≤ τ_min ≤ 10⁻¹²`). It therefore refused every
  earlier rotation (`CopulaClayton90`, …, range `[−1, −ε]`) with "does not
  contain the independence copula", as if it were A12. It now also accepts
  an upper end in `[−10⁻¹², 0]` (null ½χ²₀ + ½χ²₁, boundary estimate taken
  at the padded upper end). On `(u, v)` it gives the same statistic and
  p-value as Clayton's test on `(1 − u, v)`, with a mirrored τ̂; positive
  families take the unchanged code path. A12/A14 and their rotations are
  still, correctly, refused.
- **The generic test sweep assumed it too.** `test_scientific.py` built
  every family at `clip(0.5, range)`. For Clayton90 … BB1270 that is
  τ = −ε, so none of the earlier rotations was ever swept away from
  independence. For the new families it is −1/3, a non-exchangeable θ = 1
  copula, and `test_kendall_tau_numerical` failed on it (−0.280 for
  −0.333): its Hoeffding helper used `conditional_cdf(u, v)` as ∂C/∂v,
  which assumes exchangeability. Negative ranges are now swept at the
  mirror value −0.5. ∂C/∂v of a rotation comes from its base through the
  reflection (`1 − h(1−u|v)` at 90°, `h(u|1−v)` at 270°). The quadrature is
  now Gauss–Legendre on the whole of (0, 1): the former trapezoid on
  [0.01, 0.99] dropped boundary strips worth +0.07 on every rotation at
  −0.5, against +0.001 to +0.04 on the positive families. Every registered
  family is now within 10⁻⁴, and the test's tolerance is 10⁻³ instead of
  0.05 (its docstring already said 0.02). All 34 families, the eight
  earlier rotations now at τ = −0.5, pass the whole file, slow tests
  included (normalisation, PDF–CDF consistency, Fréchet bounds, realised
  τ).
- **Correctness.** On a 25×25 grid at τ ∈ {−1/3, −0.4, −0.6, −0.9}, the
  VineCopula identities `C90(u,v) = v − C(1−u,v)` / `C270(u,v) = u − C(u,1−v)`
  hold to 2.2·10⁻¹⁶. log c matches the base at the reflected point to
  2.8·10⁻¹⁴, `h90(v|u) = h(v|1−u)` / `h270(v|u) = 1 − h(1−v|u)` hold to
  7·10⁻¹⁵, and `inv_h` round-trips to 9·10⁻¹⁵. τ_rot = −τ_base to < 10⁻⁴,
  checked by Gauss–Legendre quadrature of `1 − 4∫∫ ∂C/∂u ∂C/∂v` (measured
  < 5·10⁻⁶). `test_copula_limits.py` covers the four families at
  τ ∈ {−0.4, −0.7, −0.9} (the base's decimal CDF reflected, no
  independence cases since −1/3 is not independence). The reference file
  was extended by **pure addition**: 12 new entries (+456 lines, 0 removed),
  the 140 existing ones byte-identical, all new rows converged, 2.8 s
  incremental computation.
- **`fit` recovery** (n = 3 000, `method='tau'` / `'mle'`): at
  τ = −0.35/−0.5/−0.7/−0.9, A1290 gives −0.355/−0.353, −0.497/−0.498,
  −0.704/−0.701, −0.898/−0.898, and A14270 −0.351/−0.350, −0.482/−0.488,
  −0.696/−0.691, −0.898/−0.899; all 32 fits are within 0.03.
  `fit_best` picks the correct rotation at τ = −0.4 (n = 1 500, both
  directions, both families).
- **Tail signature, measured.** Both bases have tail dependence in both
  diagonal corners, so both rotations populate both anti-diagonal corners,
  as BB1's do. For A12, `λ_L − λ_U = 2^{−1/θ} + 2^{1/θ} − 2 ≥ 0` at every θ,
  so A1290's lower-right corner never loses to its upper-left one. At
  corner side 0.05 and 40 000 points, the two corners weigh 0.0271 vs
  0.0056 at τ = −0.35 and 0.0379 vs 0.0323 at −0.7, reaching parity at
  −0.9 (ratio 1.02). For A14, `λ_L = 1/2` is crossed by `λ_U = 2 − 2^{1/θ}`
  at θ = 1/log₂1.5 (τ ≈ 0.547), and **A14's dominant rotated corner flips**
  there. A1490 has 0.0267 vs 0.0061 (lower-right first) at τ = −0.35 but
  0.0318 vs 0.0358 (upper-left first) at −0.7; at 200 000 points and
  τ = −0.8, the upper-left/lower-right ratio is ≈1.17. The 270° rotations
  mirror each case, and the diagonal corners stay empty (< 0.002).
- **Exchangeability** (FR-10 sanity check): at τ = −0.36 and −0.45
  (n = 1 000, B = 100, 5 replicates each), 38 of the 40 replicates across
  the four families reject at 5 % (all 20 at −0.36). At τ = −0.8, where the
  two corners balance, the power drops (1–5 of 5 per family). The test file
  asserts rejection on 3 replicates per family at τ = −0.4.
- **Tests.** New `pmcprg/tests/test_rotated_a12_a14.py` (277 tests, ≈15 s,
  identical under two `PYTHONHASHSEED` values; seeds are literals or
  `zlib.crc32`). `test_copula_limits.py` extended with the four families.
  `test_copulas.py::test_available_count` updated 30 → 34.
  `test_pdf_array.py` gives the four families an explicit τ = −0.5 (its
  generic branch would pick −0.4, only 0.07 from the interior bound).
  `test_frank_reachable_tau.py` needed no change (no reachable-τ cap).
  `test_scientific.py`: see above.
  **FR-8 is now complete for all six one-signed families** (Clayton, GH,
  Joe, BB1, A12, A14).

### Added — Monte-Carlo calibration of the outside-ICE standard errors for the six remaining families (FR-4)

- `pmcprg/tests/test_fr4_standard_errors_mc.py` covers the six families the
  audit's FR-4 "Reste" note still listed: Joe, AMH, FGM, CubSec, A12, A14.
  Each is simulated at a τ chosen well inside its own registered
  `TAU_MIN_MAX`, fitted by both `method='mle'` and `method='tau'` (Joe, AMH,
  FGM, A12, A14; n = R = 400) or both at n = 1000, R = 300 (CubSec, whose
  range [0, 33/200] is the package's narrowest — n = 400 produced occasional
  boundary hits there), and `standard_errors()` checked the same way as the
  six families already covered (Gauss, Clayton, Gumbel, Frank, Plackett,
  Student): the SE ratio and the 95 % Wald coverage, against the same
  statistically justified bands.
- All six calibrate within their bands: Joe 0.962-0.971/0.938-0.943; AMH
  0.957-0.982/0.948-0.950; FGM 0.942-0.957/0.935-0.945; CubSec
  1.034-1.056/0.950-0.957; A12 0.982-0.983/0.943-0.950; A14
  0.976-0.986/0.938-0.938. No bug found in `pmcprg/copulas/_stderr.py`.
- Joe (independence at τ → 0⁺, like Clayton) additionally checked for the
  independence LR test's level under H₀: 5.1 % rejection at the 5 % level,
  54.2 % share of zero statistics, both matching Clayton's ½χ²₀ + ½χ²₁
  bands. A12 and A14 do not contain the independence copula in their
  registered range ([1/3, 1)) and are confirmed refused by
  `independence_lr_test` (`ValueError`) by the pre-existing
  `test_fr4_standard_errors.py::test_lr_test_refuses_families_without_a_one_parameter_independence`.
- Closes the audit's "Reste" note for the outside-ICE Monte-Carlo
  calibration of FR-4: all ten one/two-parameter families with a
  `standard_errors()` path now have a Monte-Carlo coverage study.

### Added — Lystig & Hughes' exact observed information of an ICE fit, the third FR-4 option (FR-4)

- **`ice_lh_information(model, Y)`** (new module `pmcprg.pmc._lystig_hughes`)
  returns an `LHInformation`: the full observed information matrix
  `−∂²ℓ/∂θ∂θᵀ` of the observed-data log-likelihood over the prior and every
  copula pair, its inverse, the score, a one-Newton-step `I⁻¹g`, and per-pair
  τ standard errors, both joint (`se_tau`) and partial (`se_tau_partial`,
  the diagonal alone). Lystig & Hughes (2002) differentiate the forward
  recursion; here the package's **scaled** (Devijver) recursion is
  differentiated directly — `α̂_r = (a_r − C_r α̂)/C`,
  `α̂_rs = (a_rs − C_s α̂_r − C_r α̂_s − C_rs α̂)/C`,
  `∂²log C = C_rs/C − C_r C_s/C²` — so no step underflows and no log space is
  needed. The recursion is exact; only the per-observation `∂ log c/∂ψ` and
  `∂² log c/∂ψ²` are central differences (step 10⁻⁴, as `_stderr`/`_oakes`).
  `lh_loglik_derivatives` and `model_from_theta` are exposed for checks.
- **Parametrisation.** The ICE M-step estimates a *symmetric* joint prior
  (SR-PMC), so the free prior coordinates are the K(K+1)/2 − 1 softmax
  logits `η_ij = log(q_ij / q_00)` of its distinct entries (2 at K = 2, not
  K² − 1 = 3), with `A_ij = p_ij / π_i`. Copula pairs use Oakes' ψ
  (2·atanh τ for Gauss, logit τ for Clayton). A pair whose τ̂ lies at a
  boundary of its range is held fixed and gets a NaN SE (Self & Liang 1987).
- **Scope (pilot):** HMC-DN and state-margin PMC, margins fixed
  (`fit_margins=False`), Gauss and Clayton copulas, complete data. Validated
  at K = 2.
- **Checks** (`pmcprg.tests.test_fr4_ice_lystig_hughes`, 22 fast tests):
  - The value equals `forward`'s log-likelihood (relative error 10⁻¹³).
  - The gradient matches a central difference of `forward` to a maximum
    relative error of 6·10⁻⁷.
  - The Hessian matches a difference of the exact gradient to 2·10⁻⁷, and a
    4-point second difference of ℓ to 2·10⁻⁵. This holds on HMC-DN Gauss,
    HMC-DN Clayton and PMC Gauss.
  - The partial SE equals Oakes' SE to 10⁻⁹ relative, because Oakes' identity
    is exactly `−∂²ℓ/∂ψ²`.
  - Cost at K = 2, N = 600: 18 ms for the full 6×6 matrix, against 47 ms for
    Oakes' four one-pair computations.
  - ICE is not the exact MLE: its prior update ignores the initial-state term,
    and it stops at a tolerance. At an ICE fit the score is O(1), not zero,
    and in the checked examples a Newton step is at most ≈ 0.2 SE.
- **Monte-Carlo coverage** (slow, R = 300, N = 600, pair (0, 0), margins
  known, the Oakes pilot's seeds; each triple is joint / Oakes / naive):

  | Family, τ     | SE ratio              | 95 % coverage         | 90 % coverage         |
  |---------------|-----------------------|-----------------------|-----------------------|
  | Gauss, 0.6    | 1.00 / 0.95 / 0.77    | 0.943 / 0.936 / 0.863 | 0.890 / 0.866 / 0.789 |
  | Clayton, 0.5  | 1.03 / 1.01 / 0.91    | 0.947 / 0.943 / 0.927 | 0.910 / 0.903 / 0.863 |

  The joint SE is 2–5 % above Oakes' partial SE. This fixture is the
  overlapping-states one of the Godambe study, run with margins fixed. The
  joint information closes Oakes' remaining small shortfall. The
  under-coverage in the Godambe study appears only when margins are
  re-estimated. It needs the margin coordinates in θ, together with their
  pseudo-observation chain rule and the IFM correction for the ICE margin
  step, which is not an MLE step. That extension is documented in the
  module docstring, along with two-parameter copulas, pair margins and gaps.

### Added — t-EV copula, the extreme-value limit of the Student-t copula (FR-9, last round)

- **`CopulaTEV`** (`pmcprg/copulas/extreme_value/t_ev.py`, `CopulaEnum.TEV`,
  ID 30, short name `tEV`, `TAU_MIN_MAX = [EPS, 1.0]`, parameters `tau_k`
  and `nu`): Demarta & McNeil's (2005) t-EV copula,
  `ℓ(w, z) = w·T_{ν+1}(a) + z·T_{ν+1}(b)`, `a = k((w/z)^{1/ν} − ρ)`,
  `b = k((z/w)^{1/ν} − ρ)`, `k = √((ν+1)/(1−ρ²))`, `w = −ln u`, `z = −ln v`.
  This completes FR-9 for the families the package's τ + one-extra
  parameter machinery can hold (Galambos, Hüsler–Reiss, Tawn types 1/2,
  t-EV); the three-parameter asymmetric logistic Tawn model remains out of
  scope.
- **Formula re-derived, not transcribed — and the brief's is right.**
  ℓ_w is the limit of the Student-t h-function at (1 − s w, 1 − s z), s → 0
  (the t quantiles grow like s^{−1/ν}, so their ratio tends to (w/z)^{1/ν}),
  which gives ℓ_w = T_{ν+1}(a), and Euler's relation gives ℓ. In the
  package's convention t = w/(w + z) the brief's argument assignment
  ((t/(1−t))^{1/ν} in the t-term) is the only valid one: with the other, A(1)
  = T_{ν+1}(−kρ) ≠ 1. The family is exchangeable (A(t) = A(1 − t) by
  construction). Checked in `mpmath`: the Student h-limit at s = 10⁻⁴⁰
  matches T_{ν+1}(a) to 12 digits; bounds, convexity and symmetry of A hold
  at 400 random 40-digit points; λ_U = 2T_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))), the
  Student-t copula's own coefficient, equals 2(1 − A(½)) to 7·10⁻¹⁶. The
  tests also check the limit against the package's own `CopulaStudent`
  (the gap closes like s^{2/ν}).
- **Densities in log space.** ℓ_wz = −t_{ν+1}(a)·k·q/(νz) = −t_{ν+1}(b)·k/(qνw)
  (q = (w/z)^{1/ν}; equal because t(b)/t(a) = q^{ν+2}), h = C·ℓ_w/u and
  c = C(ℓ_wℓ_z − ℓ_wz)/(uv), with every term non-negative:
  ln c = wT(−a) + zT(−b) + ln(T(a)T(b) + m). a is formed from √η,
  η = (1 − ρ)/(1 + ρ), and expm1 (ρ is never formed); m takes the form whose
  t-argument is bounded; ln T in the far lower tail uses the incomplete-Beta
  series (DLMF 8.17.8), within 7·10⁻¹⁶ of 50-digit `mpmath` for
  n ∈ [1.05, 1001]. Against an independent `mpmath` ground truth (C from ℓ;
  h and c by numerical differentiation of C in (ln w, ln z) at 40–640 digits)
  on 1744 points — u, v ∈ {10⁻¹², 10⁻⁶, 0.01, 0.3, 0.5, 0.9, 1 − 10⁻⁶,
  1 − 10⁻¹²}, 29 (τ, ν) with ν ∈ {0.05, 0.5, 1, 2, 4, 50, 100, 1000} and τ
  from 10⁻¹⁰ to 0.999 — the largest relative errors are 3.5·10⁻¹³ (pdf),
  1.1·10⁻¹⁴ (cdf) and 7.6·10⁻¹⁴ (h). The 112 points whose differentiation
  would need more digits (ln c down to −6308) agree with the closed forms
  in 60-digit `mpmath` to 1.4·10⁻¹² (pdf) and 1.1·10⁻¹⁴ (cdf).
- **Parametrisation: τ and ν, no joint constraint.** ρ is recovered from
  (τ, ν), internally as s = ln η. Every τ ∈ (0, 1) is reached at every ν:
  ρ → 1 is comonotone, ρ → −1 is independence, so **negative ρ is
  admissible** (weak positive dependence, e.g. τ = 0.0087 at ρ = −0.5,
  ν = 4) and τ = 0 is never attained; τ decreases in s on every grid tried.
  `constructible_params` therefore stays the identity (jittered starts,
  family draws and the selection placeholder all build). The numerical cap
  is ν-free, τ ≤ 1 − 1.25·10⁻⁶ (x = √((ν+1)η) ≈ 10⁻⁶, Hüsler–Reiss's
  λ ≤ 10⁶); a larger τ is built at the cap and stores it (RB-10). ν is
  admissible on [0.05, 1000], the range the kernel was validated on.
  Proved in passing, for any symmetric EV copula: λ_U/2 ≤ τ ≤ 2λ_U
  (observed here: τ ≤ λ_U); it brackets the τ → s Brent search.
- **`nu`, not `df`.** `_two_parameter_spec` (joint MLE, ICE M-step),
  `EXTRA_PARAM_BOUNDS_BY_PARAM` and the GUI's extra-parameter widgets are all
  keyed by the parameter *name*: under `df`, t-EV would have taken
  Student's (atanh τ, 1/ν) branch with Student's bounds (ν > 2.001) and
  shared Student's GUI widget. It would still have fitted (the constructor
  does the τ map), through one Brent inversion per likelihood evaluation.
  `EXTRA_PARAM_BOUNDS_BY_PARAM["nu"] = (0.5, 100.0, 4.0)`; Student's `df`
  entry is unchanged. A test checks that the t-EV spec is not Student's,
  that `CopulaTEV(df=…)` is refused, and that the ICE M-step returns
  `{tau_k, nu}`.
- **Kendall's τ in logit space.** τ = 2∫_{−∞}^0 t_{ν+1}(a)·k·e^{L/ν}·σ(L)/(νA) dL
  (L = logit t, Genest–MacKay with A'' = −ℓ_wz/(t(1 − t)), symmetry), by a
  16-point composite Gauss–Legendre rule on [−1024, 0] graded around L = 0
  and around the zero ν ln ρ of a: within 7·10⁻¹⁵ of a 30-digit `mpmath`
  quadrature (A', A'' numerical derivatives of A), 36 points
  ν ∈ [0.5, 100], τ ∈ [10⁻²¹, 1 − 5·10⁻⁵], at 0.2 ms per τ. The shared
  t-space rule `tau_from_A_terms_gl` was off by up to 1.8·10⁻⁹ there, and
  `quad` by 100 % at ν = 100, τ → 1. Monte-Carlo Kendall's τ (16 × 50 000
  pairs, four points) agrees within 1.3 standard errors.
- **Hüsler–Reiss limit.** With (ν + 1)(1 − ρ)/(1 + ρ) = 1/λ² held fixed,
  ν → ∞ gives Hüsler–Reiss(λ). At τ_HR ∈ {0.1, 0.5, 0.9}, max |ΔC| on the
  edge grid is 6·10⁻⁴, 6·10⁻⁵ and 6·10⁻⁶ at ν = 10², 10³ and 10⁴, and |Δτ| is
  ≤ 2.4·10⁻⁴ at ν = 10³ — rate 1/ν. ln c converges at the same rate where it
  is not a far tail (0.036 nat at ν = 10³), but only pointwise in the
  corners (polynomial against Gaussian tails).
- **Joint MLE in its own coordinates.** `_two_parameter_spec` gets a `nu`
  branch in (logit λ_U, 1/ν): λ_U has a closed-form inverse in s (`stdtrit`),
  so no Brent search is needed per evaluation. λ_U ∈ [10⁻⁶, τ_hi] keeps
  τ ∈ [5·10⁻⁷, τ_hi], and a memo hands the constructor the s back. The box is
  run twice, as Tawn's is. `fit(method='tau')` warns and falls back to MLE.
  Recovery at n = 3000 (20 replicates per point, 160/160 converged,
  0.15 s per fit): τ̂ RMSE 0.002–0.012 everywhere. ν̂ RMSE is 0.10 at
  (τ, ν) = (0.3, 1), 0.08 at (0.7, 1), 0.22 at (0.5, 2), 0.35 at (0.9, 3),
  0.79 at (0.7, 5) and 1.2 at (0.2, 4). **Large ν is weakly identified**:
  at (0.5, 10), ν̂ ranges over [6.0, 74.6] (median 9.8); at (0.5, 50), over
  [16, 100], with 10 of 20 fits at the box end 100. Five contrived starts
  (τ_start from 10⁻⁶ to 0.95) reach the same optimum within 10⁻¹² nat on
  five data sets. Standard errors raise `NotImplementedError`, as for Tawn.
- **Tests**: `pmcprg/tests/test_t_ev.py` (129 cases, 11 s; integer seeds
  only; identical outcomes under `PYTHONHASHSEED` 0 and 12345).
  `test_copula_limits.py` gains t-EV: 6 (ρ, ν) entries, with ρ as the
  file's θ and ν in its `delta` slot, and a decimal CDF whose Student-t CDF
  comes from `mpmath`'s incomplete Beta. The cases have |ρ| ∈ [0.27,
  1 − 8·10⁻⁴], so the double ρ pins η. The references were regenerated
  **incrementally**: 228 lines added, none removed, and every pre-existing
  entry is unchanged. Family count 29 → 30 (`test_copulas.py`); t-EV joins
  the families with their own `reachable_tau_bounds`
  (`test_frank_reachable_tau.py`).

### Added — Tawn extreme-value copulas, types 1 and 2 (FR-9, round 3)

- **`CopulaTawn1`/`CopulaTawn2`** (`pmcprg/copulas/extreme_value/tawn.py`,
  `CopulaEnum.TAWN1`/`TAWN2`, IDs 28–29, `TAU_MIN_MAX = [EPS, 1.0]`,
  parameters `tau_k` and `psi`): the two two-parameter restrictions of
  Tawn's (1988) asymmetric logistic model
  `ℓ(w, z) = (1 − ψ_u)w + (1 − ψ_v)z + ((ψ_u w)^θ + (ψ_v z)^θ)^{1/θ}`,
  `w = −ln u`, `z = −ln v` — type 1 fixes ψ_u = 1 (ψ = ψ_v free), type 2
  fixes ψ_v = 1, as VineCopula's families 104/204 do (assignment from
  VineCopula's C code, which parametrises A by the *v*-share of w + z;
  recalled, not re-read — no network). The two types are exact transposes,
  `C₂(u, v) = C₁(v, u)`, so a reader of the other convention only swaps the
  names. The full three-parameter model and t-EV stay out of scope.
- **Formulas re-derived, not transcribed.** The brief's Pickands function is
  only meaningful once `t` is tied to a margin (the literature ties it both
  ways); the module names each weight by the margin it multiplies. Bounds
  `max(t, 1 − t) ≤ A ≤ 1`, `A(0) = A(1) = 1` and convexity follow by hand
  from the ℓ^θ-norm sitting between the ℓ^∞- and ℓ¹-norms, and were checked
  at 1000 random 40-digit `mpmath` points (no violation). ℓ_w, ℓ_z, ℓ_wz,
  h = C·ℓ_w/u and c = C(ℓ_wℓ_z − ℓ_wz)/(uv) are sums of non-negative terms,
  evaluated in log space (softplus/`expm1`, no power formed). Against an
  independent `mpmath` ground truth (C from ℓ; h and c by 60–960-digit
  central differences in (w, z)) on 3072 points — u, v ∈ {10⁻¹², 10⁻⁶, 0.01,
  0.3, 0.5, 0.9, 1 − 10⁻⁶, 1 − 10⁻¹²}, ψ ∈ {0.02, 0.1, 0.5, 0.9, 0.999, 1},
  τ/ψ ∈ {10⁻⁶, 0.05, 0.5, 0.95}, both types — the largest relative errors
  are 4.3·10⁻¹⁴ (pdf), 3.8·10⁻¹⁴ (cdf) and 1.5·10⁻¹³ (h). (A first
  version lost 5·10⁻¹⁴ nat where y ≫ x by adding y to −N; ``x + y − N`` is
  now formed on the side of the larger of x, y.)
- **The reachable τ is capped at ψ.** As θ → ∞ the model tends to the
  Marshall–Olkin copula, whose τ is ψ_uψ_v/(ψ_u + ψ_v − ψ_uψ_v) — ψ for both
  types (derived by hand from the Stieltjes form of Genest & MacKay's
  identity; `mpmath` at θ = 10⁵: τ = 0.3999984 for ψ = 0.4). (τ, ψ) is
  therefore jointly constrained, like BB1's (τ, δ): the constructor refuses
  τ ≥ ψ; θ ∈ (1, 10⁶], a τ in [τ(10⁶, ψ), ψ) builds at the cap and stores the
  τ it realises (RB-10). `constructible_params` moves only ψ of a refused
  pair, to the middle `(1 + τ)/2` of the admissible interval (τ, 1] — not
  "just inside" as BB1 does, because Tawn's refused end is the *singular*
  Marshall–Olkin limit — or to 1 (Gumbel) within ≈ 10⁻⁶ of τ = 1.
  Jittered starts (jitter 0.1/0.25/0.5, 60 seeds), family draws with Tawn
  candidates and the selection placeholder all build.
- **ψ = 1 is Gumbel–Hougaard, to machine precision.** It uses Gumbel's
  closed-form τ ↔ θ map, so θ is `CopulaGH`'s bit for bit; cdf values are
  identical, h agrees to 2·10⁻¹³, ln c to rounding — Tawn's value is within
  8·10⁻¹⁵ nat of `mpmath` where `CopulaGH`'s own is up to 8.6·10⁻¹³ nat off
  (τ = 0.99, diagonal: its `(1 − 2θ)·ln A` term amplifies rounding by 199).
- **Kendall's τ by a graded Gauss–Legendre rule** (`tau_from_A_terms_gl`,
  new in `_pickands.py`; Galambos/Hüsler–Reiss keep `quad`, unchanged): a
  vectorised composite 16-point rule on a mesh graded towards the ridge
  t* = ψ_v/(ψ_u + ψ_v) at scale t*(1 − t*)/θ and towards both ends. `quad`
  with the Galambos breakpoints was 4·10⁻⁹ off at θ ≥ 10³ (Gumbel included)
  and cost 3–7 ms per τ; the graded rule costs 0.2 ms and is within
  3·10⁻¹³ (θ ≤ 50) and 10⁻¹⁰ (θ = 10⁶) of a 60-digit `mpmath` quadrature
  whose A'' is a finite difference of A, θ ∈ [1 + 10⁻⁶, 10⁶], ψ ∈ [0.01, 1].
  θ(τ) is a Brent search on ln(θ − 1) (1–2.5 ms per construction). A
  Monte-Carlo Kendall's τ on 4 × 200 000 pairs agrees within 7·10⁻⁴ at
  (θ, ψ) = (1.5, 0.3), (3, 0.6), (50, 0.5), both types.
- **Joint MLE in its own coordinates.** `_two_parameter_spec` gets an
  explicit `psi` branch (the generic (τ, extra) box would have projected
  onto τ < ψ and created flat directions, RB-8): the box is
  (ln(θ − 1), ψ) ∈ [ln 10⁻⁶, ln(1/(1 − τ_hi) − 1)] × [0.01, 1], every point
  of which is a Tawn copula; a memo hands the constructor back the θ that
  produced each τ. `EXTRA_PARAM_BOUNDS_BY_PARAM["psi"] = (0.01, 1.0, 1.0)`
  (init = the Gumbel member, admissible at every τ). `fit(method='tau')`
  cannot identify ψ and falls back to MLE with a warning (BB1's precedent).
  Recovery at n = 3000, 20 replicates per point, both types: τ̂ RMSE ≤ 0.010
  everywhere; ψ̂ RMSE 0.009–0.020 at (τ, ψ) = (0.2, 0.3), (0.4, 0.6),
  (0.7, 0.8), 0.012 at ψ = 1 (τ = 0.5), but **ψ is weakly identified at
  small τ or near symmetry**: RMSE 0.07–0.08 at (0.1, 0.3) (ψ̂ from 0.20
  to 0.50) and 0.05–0.06 at (0.3, 0.9). All 240 fits converged. The box is
  run twice (a restart with fresh curvature memory): the likelihood is a
  narrow curved ridge in (ln(θ − 1), ψ), and from contrived starts a single
  L-BFGS-B pass stopped short on 3 of 50 (5 data sets × 5 start τ, both
  types), once by 131 nat; with the restart all 50 reach the same optimum.
- **Asymmetry is visible.** C₁(u, v) ≤ C₁(v, u) for u < v (strictly away
  from θ = 1, ψ = 1; θ → ∞ limit (1 − ψ)z > 0 by hand). The FR-10
  exchangeability test (n = 500, B = 100, 40 replicates) rejects 40/40 at
  (τ, ψ) = (0.3, 0.35), 17/40 at (0.1, 0.15), 13/40 at (0.2, 0.5), 3/40 at
  (0.4, 0.9), and 1/40 at ψ = 1 (Gumbel, exchangeable).
- **`h⁻¹`** by vectorised bisection on logit v (68 halvings), so `sample`
  draws 20 000 pairs in 0.16 s.
- **Tests**: `pmcprg/tests/test_tawn.py` (≈ 290 cases, 6 s; integer seeds
  only). `test_copula_limits.py` gains both types (12 (θ, ψ) entries,
  ψ ∈ {0.001, 0.05, 0.5, 0.35, 0.9, 0.97}, ψ carried in the file's `delta`
  slot), regenerated **incrementally** — 456 lines added, none removed;
  every pre-existing entry is byte-for-byte unchanged. Family count 27 → 29
  (`test_copulas.py`); Tawn joins Galambos/Hüsler–Reiss among the families
  with their own `reachable_tau_bounds` (`test_frank_reachable_tau.py`).
  Standard errors (`standard_errors`, Oakes/Godambe) are not implemented
  for Tawn and say so (`NotImplementedError`, as for any unsupported
  two-parameter family).

### Added — 90°/270° rotations of BB1, the kernel-interface prerequisite included (FR-8, third round)

- **BB1 did not expose the kernel interface** (`_kcoord`, `_kcoord_reflected`,
  `_k_logpdf`, `_k_cdf`, `_k_h`, `_k_inv_h`) the `rotated.py` mechanism needs
  — unlike Clayton/GH/Joe, which already had it via their own survival-copula
  wrappers. `pmcprg/copulas/archimedean/bb1.py` was refactored first: its
  `cdf`/`pdf`/`logpdf_array`/`conditional_cdf` are now built from the kernel
  interface instead of calling the log-space helpers directly, with **no
  change to BB1's external behaviour** — the existing BB1 tests
  (`test_palier1_b1_families.py`, `test_copula_limits.py`'s BB1 cases) pass
  bit-identically before and after. BB1's own natural kernel coordinate
  turned out to coincide with Gumbel's (`ka = log u`, not Joe's
  `log(1 − u)`), since its generator `(t^{-θ} − 1)^δ` is built from `t`
  directly. `_k_inv_h` has no closed form for BB1 (unlike GH/Joe's monotone
  Newton iteration) and solves `h(v|u) = w` by Brent's method on the kernel
  coordinate `kb = log v` itself — still a genuine kernel-interface
  implementation, not a fallback to direct composition.
- **`CopulaBB190`/`CopulaBB1270`** (`_base_class = CopulaBB1`) then follow the
  same two-line `RotatedCopula90`/`RotatedCopula270` pattern as every prior
  round, registered as `CopulaEnum.BB190`/`BB1270` (IDs 26–27),
  `TAU_MIN_MAX = [-1.0, -EPS]` mirroring BB1's own `[EPS, 1.0]`; `delta`
  passed through unchanged, unaffected by the rotation.
- **The `delta` pass-through, actually exercised for the first time.** Both
  prior rounds' reports noted (by inspection only) that `RotatedCopula`
  already passes a second parameter through to the base class unchanged.
  This round is the first to build a rotated family that actually has one,
  and it surfaced two real bugs the inspection could not have found:
  1. **`constrain_params`/`constructible_params`** (audit G1) are BB1's own
     joint δ-vs-τ repair hooks, not the identity — `RotatedCopula` needed a
     new generic delegation (τ negated, delegate to `_base_class`, τ negated
     back), preserving `CopulaVirt.constructible_params`'s identity contract
     (`params` returned as the same object when the base's own hook would
     not touch it either) so multistart's "was this pair repaired?" checks
     stay correct. Harmless for Clayton/GH/Joe's rotations, whose base
     hooks are the identity (round-trips to the same dict, verified).
  2. **`pmcprg.copulas._fit._fit_two_parameter_mle`'s BB1-specific `delta`
     branch hardcoded a positive τ range** (its θ-bound is built from the
     padded τ *ceiling*, `2/(1−τ_max) − 2`, meant to stay finite as
     τ_max → 1): on BB190/BB1270's negative range this degenerated and the
     joint optimiser returned a τ̂ outside the family's own registered
     range — found by actually running the fit, not by inspection. Fixed by
     solving the shadow positive-τ problem (`sign = −1` when the registered
     range is ≤ 0) and negating τ back in `params_of`; `CopulaBB1` itself is
     untouched (`sign = 1` there). `CopulaBB190`/`CopulaBB1270` additionally
     override `fit` to reuse BB1's own already-validated 2-D MLE via the
     reflection identity (`(U,V) ~ BB190` iff `(1−U,V) ~ BB1`) rather than
     rely on this routine directly — ICE's M-step
     (`pmcprg.pmc.ice._fit_copula_params`) still calls the routine directly,
     so the sign-fix was needed regardless.
- **Correctness.** The VineCopula identity `C90(u,v) = v − C(1−u,v)` (and the
  270° analogue) holds to machine precision across (τ, δ) ∈ {(−0.3, 1.2),
  (−0.6, 2.0), (−0.8, 3.0), (−0.9, 5.0)} on a 25×25 grid: max |ΔC| ≈ 2×10⁻¹⁶.
  Densities match the base's own `pdf_array` at the reflected point (max
  |Δc| ≈ 1.6×10⁻¹²); `h90(v|u) = h_base(v|1−u)` and
  `h270(v|u) = 1 − h_base(1−v|u)` hold to ≤5×10⁻¹⁶ on 100 random points per
  (τ, δ); `inv_h(h(v|u), u)` round-trips to ≤1×10⁻¹⁴. Cross-checked against
  `test_copula_limits.py`'s decimal reference file, extended by reflecting
  BB1's own already-validated CDF through the same machinery — pure
  addition, 6 new entries, zero changed bytes in any of the 116
  pre-existing ones (including Clayton90/270, GH90/270, Joe90/270).
  `fit` recovers both τ and δ: e.g. τ = −0.85, δ = 4.0, n = 3000 →
  τ̂ = −0.847, δ̂ = 3.92 (both `method='tau'` and `method='mle'` — BB1's own
  `fit` always uses MLE regardless, so both give the same answer here too).
- **Tail asymmetry, measured rather than assumed to be Clayton's or
  GH/Joe's.** BB1 is a Joe-Clayton hybrid with generally λ_L, λ_U both > 0
  (unlike Clayton's pure-lower or GH/Joe's pure-upper asymmetry), so a plain
  BB1 sample has residual mass in **both** diagonal corners (verified:
  λ_L=0.707, λ_U=0.413 at τ=0.6, δ=1.5, and both (0,0)/(1,1) corner masses
  are nonzero, the larger one at (0,0) matching λ_L > λ_U). The rotations
  therefore populate **both** anti-diagonal corners too, not a single
  dominant one: at τ=−0.6, δ=1.5 (40 000 simulated points), `CopulaBB190`'s
  lower-right corner (0.0362) exceeds its upper-left (0.0226), while
  `CopulaBB1270`'s upper-left (0.0348) exceeds its lower-right (0.0242) —
  the two rotations disagree on which corner dominates, unlike Clayton/GH/
  Joe's rotations, which always agree with each other's single opposite
  corner. `fit_best` still picks the correctly-rotated family cleanly when
  δ is close to 1 (near-Clayton, effectively one-sided: AIC −5086.89 vs.
  −4594.19 for the wrong rotation at τ=−0.8, δ=1.1), but the two rotations
  become harder to separate at higher δ, where both diagonal tails are
  comparable — a genuine property of this family, not a test flake.
- **Scoping note for A12/A14 (not touched this round).** BB1 confirms that a
  second free parameter is not, by itself, an obstacle to the rotation
  mechanism — the kernel-interface refactor and the two-line subclass
  pattern both went through unmodified. What actually needed new code was
  everything *downstream* of construction that has its own opinion about
  τ's sign: the joint-constraint hooks and the two-parameter MLE fitter.
  A12/A14 are one-parameter families, so if their kernel interface turns
  out to fit as cleanly as GH/Joe's did, no analogous fitter fix should be
  needed for them — but they should still be checked for any
  family-specific fitting path (like BB1's) that assumes a positive τ.
- **Tests.** `pmcprg/tests/test_rotated_bb1.py` (registration, the kernel
  interface, the VineCopula identity, h/h⁻¹ round-trips, τ/δ pass-through,
  `constructible_params`/`constrain_params` delegation — including a check
  that the one-parameter rotations are unaffected by it — measured corner
  asymmetry, `fit` recovery of τ *and* δ, `fit_best`); `test_copula_limits.py`
  extended with BB190/BB1270; `test_scientific.py`'s generic family sweep
  now builds each family via `constructible_params` instead of a bare
  `cls(tau_k=tau)` (BB190/BB1270's clipped τ landed exactly where the
  default δ = 1.5 is inadmissible, the same joint constraint BB1 itself
  needed this hook for); `test_multistart_joint_constraints.py`'s "identity
  for every family without the joint constraint" check now derives the
  excluded set from the registry (`"delta" in PARAMETERS_SET_NAME`) instead
  of naming only `CopulaEnum.BB1`. `test_copulas.py::test_available_count`
  updated 25 → 27. `ruff check pmcprg` clean; the full fast suite passes
  (3861 passed, 33 skipped, 298 deselected `slow`); every existing family
  (including Clayton/GH/Joe's rotations and plain BB1) is untouched.

### Added — exchangeability screening test, closing FR-10 (fourth and last item)

- **`pmcprg/diagnostics/exchangeability.py`.** Tests H0: the copula of a
  pair is exchangeable, `C(u,v) = C(v,u)`, with a Cramer-von Mises statistic
  comparing the empirical copula `C_n` to its transpose,
  `T_n = n * sum_i [C_n(u_i,v_i) - C_n(v_i,u_i)]^2`, calibrated by the same
  parametric (Gaussian-surrogate) bootstrap as `radial_symmetry_test` —
  chosen deliberately over a multiplier-bootstrap follow-up as the simpler,
  faster-to-validate option for a first version (FR-10's own text leaves
  that choice open for this item, unlike radial symmetry's explicit
  multiplier-bootstrap ask). No internet access to verify Genest, Neslehova
  & Quessy (2012, doi:10.1007/s10463-011-0337-6)'s exact statistic verbatim
  — this is the natural, defensible CvM construction their citation calls
  for ("compare `C_n` to its transpose"), built on the identical skeleton
  `radial_symmetry.py` already uses for the same job, and validated by
  simulation rather than by claimed fidelity to the paper's exact formula
  (same posture as both prior FR-10 modules).
- **Verified, not assumed.** Checked numerically before relying on either
  claim: every bivariate Gauss copula is exchangeable unconditionally (its
  CDF is `Phi_rho(Phi^-1(u), Phi^-1(v))`, symmetric in `u,v` for every
  `rho` — max |C(u,v)-C(v,u)| at machine precision across 5 tau values and
  3 points); the 180 degree survival rotation preserves exchangeability of
  an exchangeable base (the transform commutes with the swap); the 90/270
  degree rotations of FR-8 (`CopulaClayton90/270`, `CopulaGH90/270`,
  `CopulaJoe90/270`) break it concretely, e.g. `CopulaClayton90(tau=-0.5)`:
  `C(0.2,0.7) = 0.080221` vs `C(0.7,0.2) = 0.031237` (Delta ~= 0.049) — used
  as the power study's test bed instead of a bespoke asymmetric copula.
- **Size study** (alpha=0.05, N=200, 100 replicates, B=150): Gaussian
  tau=0.3/0.6 rejects 4.0%/1.0%; Frank tau=0.3/0.6 rejects 5.0%/11.0%;
  Clayton tau=0.3/0.6 rejects 8.0%/3.0% — all within Monte-Carlo noise of
  the 5% nominal level (binomial sd ~= 2.2% at 100 replicates).
- **Power study** (alpha=0.05, 100 replicates, B=150), rejection rate
  growing with |tau| and N as expected: `CopulaClayton90` at tau=-0.2/-0.4/
  -0.6, N=100 reaches 24%/61%/88% and N=300 reaches 61%/99%/100%;
  `CopulaJoe90` reaches 31%/71%/86% (N=100) and 81%/100%/100% (N=300);
  `CopulaGH90`, a visibly weaker alternative for this test, reaches
  10%/20%/15% (N=100, noisy near the tau=-0.4/-0.6 boundary at only 100
  replicates) and 25%/56%/61% (N=300).
- **Tests.** `pmcprg/tests/test_exchangeability.py` — result shape,
  independence never rejects, degenerate/tiny samples, mismatched shapes
  and rejected weights raise, a concrete numerical check that the 90-degree
  rotated families are and Gauss is not asymmetric at several points, fast
  qualitative size/power studies plus the full-scale ones above
  (`@pytest.mark.slow`, ~110s). Monte-Carlo seeds derived with the same
  `zlib.crc32`-based `_stable_seed()` helper the FR-10 seed fix
  (`5e56fda`) introduced for `test_radial_symmetry.py`/`test_rosenblatt.py`
  — never Python's salted `hash()`. Fast suite run 3x in a row (including
  once with `PYTHONHASHSEED=random`): identical pass, no flakes.

### Added — 90°/270° rotations of GH and Joe, applying the FR-8 pilot's mechanism (FR-8, second round)

- **Closing the pilot's own prediction.** The FR-8 pilot (`rotated.py`, one
  round above) built the generic `RotatedCopula`/`RotatedCopula90`/
  `RotatedCopula270` mechanism for Clayton and predicted that GH and Joe —
  already exposing the *kernel interface* `_kcoord`, `_kcoord_reflected`,
  `_k_logpdf`, `_k_cdf`, `_k_h`, `_k_inv_h` (used by `SurvivalGH`/
  `SurvivalJoe` in `survival.py`) — would each be a two-line subclass, same
  as Clayton's. This round adds `CopulaGH90`, `CopulaGH270`, `CopulaJoe90`,
  `CopulaJoe270` and confirms that prediction: no change to `rotated.py`'s
  wrapper machinery was needed, only four new `_base_class = CopulaGH` /
  `CopulaJoe` subclasses. BB1, A12 and A14 remain out of scope — they do not
  yet expose the kernel interface, the same prerequisite the pilot and
  `survival.py` both stopped at.
- **Registered τ ranges verified, not assumed.** GH and Joe are both
  registered with `TAU_MIN_MAX = [EPS, 1.0]` (`CopulaEnum.GH`/`JOE`),
  identical to Clayton's own range — so `GH90`/`GH270`/`JOE90`/`JOE270`
  mirror it the same way Clayton90/270 did: `TAU_MIN_MAX = [-1.0, -EPS]`
  (`CopulaEnum` IDs 22–25). Confirmed by reading the registry rather than
  assumed from the family name.
- **Correctness.** The VineCopula identity `C90(u,v) = v − C(1−u,v)` (and
  the 270° analogue `C270(u,v) = u − C(u,1−v)`) holds to machine precision
  for both families on a 25×25 grid at τ ∈ {−10⁻⁴, −0.3, −0.6, −0.95}: max
  |ΔC| = 2.2×10⁻¹⁶ (GH), 1.7×10⁻¹⁶ (Joe). Densities match the base family's
  own already-validated `pdf_array` at the reflected point directly (max
  |Δc| = 2.8×10⁻¹⁰ for GH, 1.1×10⁻⁹ for Joe — cheaper and independent of
  `test_copula_limits.py`'s decimal machinery). Against that decimal
  reference file — extended by reflecting GH's and Joe's own
  already-validated CDF formulas through the same `decimal` code the pilot
  used for Clayton, rather than rebuilding fresh ground truth — the
  package's closed forms reach, over the existing 6×6 grid × 6 τ (moderate,
  strong and independence-limit, τ down to −10⁻¹²), for both 90°/270°:
  GH max |Δ log c| = 1.1×10⁻¹³, max |ΔC| = 1.1×10⁻¹⁶, max |Δh| = 1.4×10⁻¹⁴;
  Joe max |Δ log c| = 2.3×10⁻¹³, max |ΔC| = 1.1×10⁻¹⁶, max |Δh| = 6.5×10⁻¹⁵.
  `inv_h(h(v|u), u)` round-trips to ≤2.3×10⁻¹⁶ for both. `fit(method='tau')`
  and `fit(method='mle')` both recover τ̂ within 0.06 of τ ∈ {−0.2, −0.5,
  −0.8} from n = 3000 simulated points, for all four new families (worst
  case observed: |τ̂ − τ| = 8.3×10⁻³).
  **The corner swap is the mirror image of the Clayton pilot's, not a
  repeat of it**: GH and Joe are upper-tail families (mass at (1,1)), not
  lower-tail like Clayton (mass at (0,0)), so reflecting through (1−U,V)
  pushes that mass to the *upper-left* corner for the 90° rotation and the
  *lower-right* corner for 270° — the opposite assignment from
  `CopulaClayton90`/`CopulaClayton270`. Caught by actually running the
  simulation rather than copying the pilot's assertion direction: a first
  draft asserted Clayton's corner assignment for GH and failed immediately.
  Simulating 40 000 points at τ = −0.6 confirms it for both new families —
  Joe's own upper-tail asymmetry is sharper than GH's at the same τ (plain
  Joe: corner mass ≈0.083 at (1,1) vs. ≈0.030 at (0,0); plain GH: ≈0.072 vs.
  ≈0.047), so Joe reproduces the pilot's own ≈2.7× corner-mass ratio at the
  pilot's own threshold (0.1), while GH needed a tighter corner threshold
  (0.03) for the same ≥2× separation.
- **Registration.** `CopulaEnum.GH90`/`GH270`/`JOE90`/`JOE270` (IDs 22–25);
  `reachable_tau_bounds()` is `None` for all four (nothing narrows the
  registered range). No hardcoded family list needed updating beyond the
  count test: `pmcprg.copulas.__init__` derives its public API from
  `CopulaEnum` directly, and `test_pdf_array.py`'s per-family τ picker
  already handled a negative-τ wide-span family generically (the pilot's own
  fix, `tau = 0.4 if tau_max > 0.0 else -0.4`, needed no further change).
  `test_copulas.py::test_available_count` updated 21 → 25.
- **Tests.** `pmcprg/tests/test_rotated_gh.py`, `pmcprg/tests/test_rotated_joe.py`
  (registration, boundary rejection, the VineCopula identity, h/h⁻¹
  round-trips, θ(τ) against the base family, sampling's realised τ and
  corner asymmetry, `fit` recovery, `fit_best`); `test_copula_limits.py`
  and its reference file extended with GH90/GH270/Joe90/Joe270 at the same
  τ grid as GH/Joe themselves, negated (116 entries, 4176 rows total; the
  regenerated file's diff against the pre-existing one is pure addition —
  zero changed bytes in any prior entry, including Clayton90/270).
  `ruff check pmcprg` clean; the full fast suite passes (3691 passed, 33
  skipped, 267 deselected `slow`); every existing family (including plain
  GH, Joe and Clayton90/270) is untouched.

### Added — Godambe's (IFM) information for a copula τ fitted inside ICE alongside re-estimated margins (FR-4)

- **`ice_godambe_tau_se(model, Y, pairs=None)`** (new module `pmcprg.pmc._godambe`),
  the second of FR-4's "Reste" three ranked options for a copula parameter's
  standard error inside ICE (after Oakes' identity, `pmcprg.pmc._oakes`;
  Lystig & Hughes 2002 remains out of scope): Joe's (2005) Inference
  Functions for Margins / Godambe sandwich, `Avar(η̂, θ̂) = D⁻¹ M D⁻ᵀ` for the
  joint estimating-equation vector `g(η, θ)` stacking the margin
  weighted-MLE scores (η, weighted by the ICE E-step's `γ_n(k)`) and the
  copula score (θ = τ, weighted by `ξ_n(i, j)`). Unlike Oakes' identity,
  which corrects for `ξ̂`'s own latent-state uncertainty but assumes margins
  are fixed, this targets the complementary gap `pmcprg.copulas._stderr`'s
  own docstring names explicitly: margins re-estimated inside ICE (`dist ==
  "norm"` state margins, `fit_margins=True`) whose estimation error is
  correlated with the copula score through the pseudo-observations they
  build — the missing "IFM Godambe matrix" that module says it does not
  provide. Pilot scope: K = 2, `margin_structure == "state"`, Gaussian
  margins, **Gauss** copula only, one pair at a time; `D`'s margin block is
  closed-form (elementary Gaussian weighted-MLE curvature), its
  margin/copula cross-block (the correction itself) and copula block are
  central finite differences in the same working coordinates
  (`pmcprg.pmc._oakes`'s ψ for τ, `(μ, log σ)` for margins) the sibling
  modules already use.
- Monte-Carlo coverage (`pmcprg.tests.test_fr4_ice_godambe`, R = 300, N =
  600, τ = 0.6, pair (0, 0), `fit_margins=True`): on the package's standard
  K = 2 fixture (states overlap a great deal), SE ratio (RMS reported /
  empirical SD) 0.58 (Godambe) vs 0.49 (Oakes) vs 0.40 (naive sandwich); 95 %
  coverage 0.74 vs 0.68 vs 0.58; 90 % coverage 0.67 vs 0.60 vs 0.52. On a
  well-separated-states variant (near-hard classification, isolating the
  margin-reestimation channel from Oakes' latent-state channel), Godambe's
  coverage is close to nominal: ratio 0.86, 95 % coverage 0.91, 90 %
  coverage 0.83 — vs ratio ≈ 0.48 and coverage ≈ 0.53-0.64 for both Oakes
  and the naive sandwich in that regime. Godambe beats both alternatives in
  every case measured, and reaches near-nominal coverage once the E-step's
  own latent-state ambiguity (Oakes' own channel, not extended here to the
  margins) is not the dominant source of uncertainty — an identified,
  documented next step, not a defect of the pilot.

### Added — Rosenblatt-transform goodness-of-fit test for a copula family (FR-10)

- **`rosenblatt_gof_test(x, y, family_cls, ...)`** (new module
  `pmcprg.diagnostics.rosenblatt`), the third FR-10 item: a GoF test that
  "reuses h and h⁻¹" (Rosenblatt 1952; Genest, Rémillard & Beaudoin 2009,
  doi:10.1016/j.insmatheco.2007.10.005). For pseudo-observations `(û, v̂)`
  the Rosenblatt transform `(Û, V̂) = (û, h(v̂|û))` — `h(v|u) =
  copula.conditional_cdf(v, u)`, already implemented for every family — is
  i.i.d. Uniform([0,1]²) and mutually independent under H0, so testing a
  candidate family reduces to testing the transformed sample against the
  independent-uniform reference `Π(u,v) = u·v`. The statistic is a
  Cramér–von Mises functional evaluated at the sample's own points, `Sₙ = n ·
  mean_i[Ĝₙ(Ûᵢ,V̂ᵢ) − Ûᵢ·V̂ᵢ]²` (`Ĝₙ` the empirical CDF of the transformed
  sample), the same skeleton as `radial_symmetry_statistic`'s own CvM
  statistic with the reflected-empirical-copula reference replaced by the
  fixed `Π`. Calibrated by a **parametric bootstrap that refits the
  candidate family on every replicate** (fit `θ̂`, simulate `n` points from
  `C_θ̂`, refit `θ̂_b`, Rosenblatt-transform with `C_θ̂_b`, recompute `Sₙ⁽ᵇ⁾`) —
  this is the calibration GRB (2009) themselves use for their own CvM
  statistics, not a simplification borrowed from elsewhere in this codebase.
  `mks_1samp` (already in the package, and able to test against any
  reference CDF including `Π`) was considered and **not** used: it is a
  Kolmogorov–Smirnov (max-deviation) statistic, with materially lower power
  than a CvM (average-deviation) statistic against a wrong copula family's
  diffuse departure from `Π`, and it is not what GRB (2009) actually propose
  for the Rosenblatt variant the audit cites — see the module docstring's
  "The test statistic" section for the full reasoning, and its "Honesty
  about fidelity" section for what is verified by citation vs. by the
  simulation study below (the exact GRB 2009 formula for this variant is not
  reproduced verbatim — no internet access from this worktree — the CvM
  construction here is the natural one their general framework calls for,
  validated by simulation as the radial-symmetry pilot's own statistic was).
  New typed result `RosenblattGoFResult` (`statistic, p_value, reject, alpha,
  n, family, tau_hat, B, n_valid`), unweighted only (same reasoning as
  `radial_symmetry_test` — no published weighted form of this empirical-CDF
  functional).
  **Size study** (N_reps=150, B=100, N=200, true family fitted to itself —
  by design this test's size should not depend on whether the true family is
  radially symmetric, unlike `radial_symmetry_test`): Gauss τ=0.3/0.6 →
  0.053/0.053 at α=0.05, 0.120/0.100 at α=0.10; Clayton τ=0.3/0.6 →
  0.047/0.053 at α=0.05, 0.087/0.093 at α=0.10; Frank τ=0.3/0.6 →
  0.067/0.073 at α=0.05, 0.160/0.107 at α=0.10 — all within Monte-Carlo
  reach of nominal (150 reps: binomial sd ≈ 0.018 at 5%, ≈ 0.024 at 10%),
  including for Clayton, the one asymmetric family tested, confirming size
  is not tied to radial symmetry here.
  **Power study** (N_reps=150, B=100, α=0.05, wrong family fitted): true
  Clayton / fit Gauss, τ∈{0.2,0.4,0.6} → 0.193/0.520/0.853 at N=100,
  0.447/0.980/1.000 at N=300 — power rising sharply with both τ and N; true
  Gumbel-Hougaard / fit Gauss, τ∈{0.2,0.4,0.6} → 0.060/0.073/0.113 at N=100,
  0.107/0.173/0.160 at N=300 — much weaker (Gauss and GH share the same
  correlation-driven bulk dependence at moderate τ, so this is the harder
  misspecification, as expected); true Gauss / fit Frank, τ∈{0.4,0.6} →
  0.107/0.207 at N=200 — weak but nonzero, again two nearly-radially-similar
  families. See `pmcprg/tests/test_rosenblatt.py` for fast, reduced-budget
  versions of both studies (`test_size_study_true_family_near_nominal`,
  `test_power_study_wrong_family_rejects_more_often`) plus a
  `@pytest.mark.slow` full-grid mirror of these numbers.

### Added — 90°/270° rotations of Clayton, as a reusable mechanism (FR-8 pilot)

- **The gap.** Clayton, GH, Joe, BB1, A12 and A14 are one-signed: their θ maps
  only reach τ ≥ 0 (or τ ≥ 1/3 for A12/A14), so none of them can represent
  negative dependence — the same limitation VineCopula (its rotation codes
  23–40), vinecopulib, GJRM, VC2copula and gofCopula's `flip` all correct via
  90°/180°/270° rotations. The 180° case (the survival copula, negating
  neither margin) was already in this codebase (`archimedean/survival.py`);
  90° and 270° (negating one margin) were not. This round adds them for
  **Clayton only**, as a pilot for the generic mechanism the other five
  families still need.
- **Generic mechanism, not six one-off classes.** New module
  `pmcprg/copulas/archimedean/rotated.py`: `RotatedCopula` (shared parameter
  plumbing — τ negated, every other parameter, e.g. a future BB1 `delta`,
  passed through unchanged) plus two one-method-per-formula mixins,
  `RotatedCopula90`/`RotatedCopula270`, composing any base family's existing
  *kernel interface* (`_kcoord`, `_kcoord_reflected`, `_k_logpdf`, `_k_cdf`,
  `_k_h`, `_k_inv_h` — the same interface `survival.py`'s 180° rotation
  already requires) with the (1−u, v) / (u, 1−v) reflection, so the
  complement is never formed in floating point (same numerical motivation as
  `survival.py`'s module docstring). `CopulaClayton90`/`CopulaClayton270` are
  each a two-line subclass setting `_base_class = CopulaClayton`. GH and Joe
  already expose the kernel interface (used by `SurvivalGH`/`SurvivalJoe`)
  and would need the same two lines each; BB1, A12 and A14 do not yet expose
  it (same prerequisite the 180° rotation already stopped at) and are the
  actual remaining cost of a fast follow, not a limitation of this mechanism.
- **The two rotations were derived, not assumed, and verified to differ.**
  C90(u,v) = v − C(1−u,v) and C270(u,v) = u − C(u,1−v) both give
  τ_rot = −τ_base, but only 270°'s h-function carries a genuine "1 −" term
  (h270(v|u) = 1 − h(1−v|u)); 90°'s does not (h90(v|u) = h(v|1−u) — the two
  sign flips from differentiating through the reflected argument cancel).
  Confirmed numerically: simulating 40 000 points from each at τ = −0.6,
  Clayton90 concentrates near the lower-right corner (u→1, v→0; corner mass
  0.082 vs. 0.030 for the mirror corner) and Clayton270 near the upper-left
  (u→0, v→1; 0.080 vs. 0.030) — not the same joint law despite the same τ.
- **Correctness.** The VineCopula identity C90(u,v) = v − C(1−u,v) (and the
  270° analogue) holds to machine precision (max |ΔC| = 1.1×10⁻¹⁶ on a
  25×25 grid at τ ∈ {−10⁻⁴, −0.3, −0.6, −0.95}); the density likewise matches
  c_base(1−u,v) / c_base(u,1−v) directly (max |Δc| ≈ 2×10⁻¹⁵). Against
  `test_copula_limits.py`'s high-precision decimal references — extended by
  reflecting Clayton's own already-validated CDF formula, `v − C(1−u,v)`,
  through the same `decimal` machinery rather than rebuilding fresh ground
  truth — the package's closed forms reach max |Δ log c| = 2.3×10⁻¹³,
  max |ΔC| = 1.2×10⁻¹⁶, max |Δh| = 1.1×10⁻¹⁴ over the existing
  6×6 grid × 6 τ (including the two independence-limit tails, τ down to
  −10⁻¹²). `inv_h(h(v|u), u)` round-trips to ≤3×10⁻¹⁰. `fit(method='tau')`
  and `fit(method='mle')` both recover τ̂ within 0.06 of τ ∈ {−0.2, −0.5,
  −0.8} from n = 3000 simulated points.
- **Registration.** `CopulaEnum.CLAYTON90` / `CLAYTON270` (IDs 20–21),
  `TAU_MIN_MAX = [-1.0, -EPS]` — the mirror image of Clayton's own
  `[EPS, 1.0]`; `reachable_tau_bounds()` is `None` for both (nothing narrows
  the registered range, unlike Frank/Plackett/Galambos). No hardcoded family
  list needed updating: `pmcprg.copulas.__init__` derives its public API from
  `CopulaEnum` directly, and `tail_dependence()` returns `(0, 0)` for both —
  a 90°/270° rotation's negative dependence sits on the anti-diagonal, which
  the base class's diagonal-probe default (built for the 180°/positive case)
  does not see, matching VineCopula's own convention for its rotated
  one-parameter families.
- Tests: `pmcprg/tests/test_rotated_clayton.py` (registration, boundary
  rejection, the VineCopula identity, h/h⁻¹ round-trips, θ(τ) against the
  base family, sampling's realised τ and corner asymmetry, `fit` recovery,
  `fit_best`); `pmcprg/tests/test_copula_limits.py` and its reference file
  extended with `Clayton90`/`Clayton270` at the same τ grid as `Clayton`
  itself, negated. `ruff check pmcprg` clean; the full fast suite passes;
  Clayton (unrotated) and every other family's cached references are
  bit-for-bit unchanged.

### Added — Oakes' observed information inside ICE, generalised to BB1 and Student (FR-4)

- **Two-parameter Oakes' information.** `ice_oakes_tau_se` now also accepts
  BB1 and Student copula blocks (`pmcprg.pmc._oakes.SUPPORTED_FAMILIES` is
  now `("Gauss", "Clayton", "BB1", "Student")`), generalising every scalar
  quantity of the FR-4 pilot below to its 2×2-matrix equivalent in the
  family's 2-D working coordinate ψ (`log θ, log(δ − 1)` for BB1;
  `atanh τ, log(ν − 2)` for Student — the same ψ, and the same analytic
  Jacobian to τ/δ/θ or τ/ν/ρ, that `pmcprg.copulas._stderr` already uses for
  its "outside ICE" sandwich on these families, reused rather than
  reimplemented). The complete-data curvature (term 1) needs no extra
  E-step, same as the scalar case; Oakes' correction (term 2) needs one
  E-step per perturbed working coordinate, 4 total for a two-parameter
  family (vs. 2 for one parameter) — the resulting 2×2 "missing
  information" matrix is symmetrised (Oakes' theorem guarantees the *sum*
  with term 1 is symmetric; the antisymmetric part measured in validation
  is 1–2 orders of magnitude below the symmetric part, i.e. noise).
  Two-parameter results are returned as a new `OakesResultMulti`
  (`pair, family, names, estimate, se, se_naive, cov, cov_naive,
  info_complete, info_missing, info_psi, n_eff, at_boundary, boundary`),
  mirroring `pmcprg.copulas._stderr.StandardErrors`'s
  `names`/`estimate`/`se`/`cov` convention rather than diverging in shape;
  the original scalar `OakesResult` (Gauss/Clayton) is untouched, bit-for-bit
  — same functions, same code path, verified by the full pre-existing
  `pmcprg/tests/test_fr4_ice_oakes.py` suite passing unchanged, plus a new
  cross-check that the matrix code, run on a one-parameter family (ψ of size
  1), reproduces the scalar path's numbers to a relative difference of
  ~4·10⁻¹⁶.
  Monte-Carlo validation (K = 2 HMC-DN, N = 600, margins known, R = 150 —
  fewer replicates than the one-parameter pilot's R = 300–400, since a
  two-parameter ICE fit plus 4 extra E-steps per Oakes call costs more per
  replicate; reported with the resulting wider bands):
  BB1 τ = 0.5, δ = 1.5 (n = 150/150) — SE ratio (RMS reported / empirical
  SD) τ: 1.010 Oakes vs 0.782 naive, δ: 0.982 vs 0.681; 95 %/90 % coverage
  τ: 0.907/0.900 Oakes vs 0.860/0.813 naive, δ: 0.960/0.900 vs 0.833/0.760.
  Student τ = 0.5, ν = 6 (n = 139/150, after excluding replicates where ν̂
  hit the fitting box, mostly its upper Gaussian-limit end) — 95 %/90 %
  coverage τ: 0.935/0.885 Oakes vs 0.827/0.734 naive, ν: 0.906/0.892 Oakes
  vs 0.885/0.856 naive (the SE-ratio diagnostic is unreliable for ν at this
  replicate count — its sampling distribution is heavy right-tailed — and
  is not reported for it; coverage, the more robust diagnostic and the one
  FR-4 is actually about, is unaffected). Both studies confirm the FR-4
  hypothesis for the second parameter too: the naive sandwich under-covers
  more than Oakes' matrix SE. See
  `pmcprg/tests/test_fr4_ice_oakes_bb1_student.py`. Still out of scope:
  margins re-estimated inside ICE, gap variants, joint covariance of several
  pairs.

### Added — multiplier bootstrap for the radial symmetry test (FR-10 round 2)

- **`radial_symmetry_test(..., bootstrap="multiplier")`** (`pmcprg.diagnostics.radial_symmetry`),
  alongside the unchanged `bootstrap="parametric"` default from the FR-10
  pilot. Implements the multiplier bootstrap FR-10 actually asks for
  (Kojadinovic & Yan 2011, doi:10.1007/s11222-009-9142-y; Kojadinovic, Yan &
  Holmes 2011, doi:10.5705/ss.2011.037a) instead of the parametric bootstrap's
  per-replicate refit-free resample-and-rerank: under H0 the deterministic
  `u + v − 1` term cancels, so `√n D_n(u,v) = α_n(u,v) − α_n(1−u,1−v) + o_P(1)`
  with `α_n` the empirical copula process; its standard multiplier-CLT
  linearisation (Rémillard & Scaillet 2009 — the same construction underlying
  this codebase's own `Ẇ₁, Ẇ₂` score corrections in `copulas/_stderr.py`)
  replaces `α_n` by `α_n^ξ(u,v) = n^{-1/2} Σᵢ ξᵢ[1{ûᵢ≤u,v̂ᵢ≤v} − Cₙ(u,v) −
  Ċ₁(u,v)(1{ûᵢ≤u}−u) − Ċ₂(u,v)(1{v̂ᵢ≤v}−v)]`, i.i.d. `ξᵢ` (mean 0, variance 1;
  standard normal by default, `multiplier="rademacher"` optionally) resampled
  fresh per replicate, `Ċ₁, Ċ₂` plug-in partial derivatives of `Cₙ` by a
  central finite difference (bandwidth `h = min(0.5, n^{-1/2})`). `Cₙ`, ranks
  and `Ċ₁, Ċ₂` are computed once; every replicate is then only a fresh draw of
  `ξ` and a matrix–vector product, all `B` replicates produced by one
  `(2n×n) @ (n×B)` matrix multiplication — no resampling, no rank
  recomputation, no refit. This is a derivation worked out for this
  statistic's particular reflected-difference form, not a transcription of a
  published formula specific to radial symmetry (see the module docstring's
  "Honesty about fidelity" section) — validated by simulation, not by claimed
  fidelity to a paper, exactly as the pilot's own statistic was.
  **Speed**: N=200, B=150, same seed — 1.95 ms/call (multiplier) vs 40.4
  ms/call (parametric), a **20.7× speed-up**.
  **Size** (N_reps=100, B=150, α=0.05, N=200): Gauss τ=0.3/0.6 → 0.02/0.03;
  Frank τ=0.3/0.6 → 0.06/0.07; Plackett τ=0.3/0.6 → 0.06/0.08 — all at or
  below the pilot's parametric-bootstrap range (0.03–0.07 at nominal 5%),
  i.e. comparable, if anything slightly conservative.
  **Power** (N_reps=100, B=150, α=0.05, same grid as the pilot): Clayton
  τ∈{0.2,0.4,0.6} → 0.11/0.46/0.69 at N=100, 0.49/0.95/1.00 at N=300; Gumbel/GH
  τ∈{0.2,0.4,0.6} → 0.15/0.29/0.37 at N=100, 0.30/0.65/0.69 at N=300; Joe
  τ∈{0.2,0.4,0.6} → 0.43/0.84/0.99 at N=100, 0.83/1.00/1.00 at N=300 — the same
  ordering and comparable magnitude to the parametric bootstrap's own study
  (0.08 to 1.00 over the identical grid), power growing with τ and N in every
  family tested, as before. See `pmcprg/tests/test_radial_symmetry.py`
  (`test_multiplier_bootstrap_*`, `@pytest.mark.slow` for the full-grid
  studies) for the reduced-budget versions of these numbers.

### Added — standard errors of a copula τ fitted inside ICE, pilot (FR-4, "Reste")

- **Oakes' (1999) observed information inside ICE.** `pmcprg.pmc._oakes.ice_oakes_tau_se(model, Y, pairs=None)`
  gives the standard error of a per-pair copula τ fitted as part of ICE's
  M-step, where the naive `pmcprg.copulas._stderr` sandwich applied to the
  final posteriors ξ̂_n(i, j) understates the uncertainty because those
  weights are themselves an estimate over a latent state sequence. Computes
  `I(τ̂) = −∂²Q(θ|θ̂)/∂θ² − ∂²Q(θ|θ')/∂θ∂θ'` (Oakes 1999,
  doi:10.1111/1467-9868.00188) with Q the expected complete-data
  log-likelihood of the pair's copula, both terms central finite differences
  in the working coordinate ψ (2·atanh τ for Gauss, logit τ for Clayton, same
  `H_PSI = 1e-4` convention as `_stderr.py`); the mixed term re-runs one
  E-step (forward-backward + joint posteriors) at τ perturbed by ±H_PSI in ψ,
  everything else held at the ICE fit. Pilot scope only: Gauss and Clayton,
  K = 2, one pair at a time, margins fixed (`fit_margins=False`) — the
  audit's cheapest-first option; Godambe (IFM) and Lystig–Hughes are not
  attempted.
  Monte-Carlo validation (K = 2 HMC-DN, N = 600, R = 400, margins known):
  Gauss τ = 0.6 — SE ratio (RMS reported / empirical SD) 0.974 (Oakes) vs
  0.792 (naive), 95 % coverage 0.938 vs 0.880, 90 % coverage 0.877 vs 0.797;
  Clayton τ = 0.5 — ratio 0.993 vs 0.894, 95 % coverage 0.958 vs 0.930, 90 %
  coverage 0.907 vs 0.863. The naive sandwich under-covers by 5–9 points at
  both levels in both families; Oakes' correction (always negative — it
  *reduces* the naive complete-data information, Louis 1982) brings the ratio
  within a few percent of 1 and the coverage within 1–2 sampling SEs of
  nominal. See `pmcprg/tests/test_fr4_ice_oakes.py`. Not yet: BB1/Student,
  margins re-estimated inside ICE, gap variants, joint covariance of several
  pairs — a fast follow if this pilot is extended.

### Added — radial symmetry screening test (FR-10)

- **Radial symmetry screening test** (`pmcprg.diagnostics.radial_symmetry`,
  audit FR-10): a Cramér–von Mises statistic
  `T_n = n·Σᵢ[Cₙ(ûᵢ,v̂ᵢ) − ûᵢ − v̂ᵢ + 1 − Cₙ(1−ûᵢ,1−v̂ᵢ)]²` comparing the
  empirical copula to its radial reflection (Genest & Nešlehová 2014,
  doi:10.1007/s00362-013-0556-4, adapted — see the module docstring for what
  is a defensible simplification rather than a verified reproduction of the
  paper's exact statistic and multiplier bootstrap), calibrated by a
  parametric bootstrap from a Gaussian copula matched on Kendall's τ.
  Rejecting radial symmetry excludes Gauss, Student, Frank, FGM and Plackett
  at once, pruning the family-selection search before the per-family fits.
  Unweighted only: ξ-weighting has no defensible form for this statistic
  (it is a joint functional of the whole sample, not a U-statistic with a
  known weighted extension), so `radial_symmetry_test` raises
  `NotImplementedError` on a `weights` argument rather than guess.
  Validated by simulation (N_reps=300, B=150 per test): empirical rejection
  rate at nominal 5%/10% stays within 0.03–0.07 / 0.07–0.12 for Gauss, Frank
  and Plackett at τ=0.3 and 0.6, N=200; power against Clayton, Gumbel and Joe
  at τ∈{0.2,0.4,0.6}, N∈{100,300} ranges from 0.08 (Clayton, τ=0.2, N=100 —
  genuinely weak asymmetry) to 1.00 (Joe, τ≥0.4, N=300), rejecting more often
  as τ or N grow, in every family tested.

### Added — Galambos copula, first extreme-value family (FR-9 pilot)

- **`CopulaGalambos`** (`pmcprg.copulas.extreme_value.galambos`, `SHORT_NAME`
  "Galambos"), the simplest of the extreme-value families audited as missing
  (Galambos, Hüsler–Reiss, Tawn, t-EV — FR-9), implemented as a template for
  the other three. Single parameter θ > 0, Pickands function
  A(t) = 1 − (t^{−θ} + (1−t)^{−θ})^{−1/θ}; CDF, pdf and h-function derived
  from scratch (`sympy`) rather than transcribed from a secondary source,
  and checked against an independent 50-digit `mpmath` ground truth: pdf,
  cdf and h agree to relative error ≤ 1.5·10⁻¹³ across (u, v, θ) spanning
  u, v ∈ [10⁻¹², 1 − 10⁻¹²] and τ ∈ [10⁻¹², 0.95], after fixing a
  catastrophic-cancellation bug the ground truth caught during development
  (θ = 19.3, u = 0.3, v = 10⁻¹²: the naive h-bracket returned exactly 0.0
  against a true density ≈ 3.3·10⁻²⁶ — fixed with a `softplus`/`expm1`
  reformulation, see the module docstring).
- **Kendall's τ has no closed form.** The value τ = 1/(θ+2) sometimes quoted
  without derivation is wrong — it decreases with θ, the opposite direction
  from λ_U = 2^{−1/θ}, which must increase with it. τ(θ) is computed by
  adaptive quadrature of the Genest & MacKay (1986) Pickands-function
  identity, verified against the same identity applied to Gumbel's Pickands
  function (reproduces τ = 1 − 1/θ to 1e-12) and against a 200,000-pair
  Monte-Carlo Kendall's τ at θ = 1, 2, 5 (agreement within Monte-Carlo
  standard error). Registered τ range [0 + ε, 1) — extreme-value copulas
  have no negative dependence; reachable up to τ ≈ 1 − 1e-6 (θ capped at
  1e6), beyond which the package stores the τ that θ realises (RB-10
  convention, as Frank/Plackett).
- `inv_h` has no closed form either (Gumbel's monotone-Newton trick does
  not carry over) and uses `CopulaVirt`'s Brent-bracketed default — noted
  as a follow-up for a shared extreme-value/Pickands scaffold once
  Hüsler–Reiss, Tawn or t-EV are added.
- New `pmcprg/tests/test_galambos.py` (37 cases: boundary rejection, τ(θ)
  against the quadrature and a simulated τ, h/h⁻¹ round-trip, sampling's
  marginal uniformity and realised τ, `fit` recovery by both methods) and
  6 new (θ, δ) entries in `test_copula_limits.py`'s high-precision reference
  file (216 grid rows); every pre-existing family's cached reference is
  byte-for-byte unchanged.

### Added — Hüsler–Reiss copula, second extreme-value family (FR-9, round 2)

- **`CopulaHuslerReiss`** (`pmcprg.copulas.extreme_value.husler_reiss`,
  `SHORT_NAME` "HuslerReiss"), the second of the four extreme-value families
  audited as missing (FR-9: Galambos done, Tawn and t-EV still open). Single
  parameter λ > 0, Pickands function
  A(t) = t·Φ(g(t)) + (1−t)·Φ(h(t)), g(t) = 1/λ + (λ/2)ln(t/(1−t)),
  h(t) = 2/λ − g(t), Φ the standard normal CDF (Hüsler & Reiss 1989). CDF,
  pdf and h-function re-derived from scratch from the general EV-copula
  identities (not transcribed), yielding a closed pdf
  c(u,v) = (C/(uv))·[Φ(g)Φ(h) + λφ(g)/(2z)] with a striking simplification
  ℓ_w = Φ(g), ℓ_z = Φ(h) — cross-checked against an independent 40–240-digit
  `mpmath` ground truth (finite differences of C, and of the Pickands
  function A): pdf, cdf and h agree to relative/absolute error ≤ 2·10⁻¹³
  across u, v ∈ [10⁻¹², 1 − 10⁻¹²] and λ spanning 0.3 to 10⁶.
- **The direction of λ was mis-stated in the initial task brief and
  corrected here after independent re-derivation.** A plausible first
  reading of the Pickands formula's `1/λ` term suggests λ → 0 is the
  comonotone copula; direct limit analysis (and cross-checked against the
  textbook tail-dependence formula λ_U = 2Φ(−1/λ)) shows the opposite:
  **λ → 0 is independence, λ → ∞ is comonotone** — the *same* direction as
  Galambos's θ, not the opposite. See the module docstring's "Direction of
  λ" section for the full derivation and the two independent checks
  (limit of A(t) at fixed t; conversion to the alternative η = 1/λ
  parametrisation used in some references).
- **Kendall's τ has no closed form** (confirmed, not assumed — same
  situation as Galambos). τ(λ) is computed by the same adaptive-quadrature
  Genest & MacKay (1986) machinery as Galambos, now factored into a small
  shared module `pmcprg.copulas.extreme_value._pickands`
  (`tau_from_A_terms`, `invert_tau` — just the quadrature-with-breakpoints
  and Brent-inversion pattern; each family keeps its own closed-form A, A',
  A'', pdf, cdf and h, since Hüsler–Reiss's formulas do not share enough
  structure with Galambos's to force a common implementation). Galambos's
  module was refactored to use the same shared helpers — its own 34
  non-slow tests pass bit-identically, unchanged. Hüsler–Reiss's own τ(λ)
  cross-checked by a 20,000-pair Monte-Carlo Kendall's τ at λ = 1, 2, 5:
  quadrature 0.2554/0.5387/0.7914 vs Monte-Carlo 0.2503/0.5434/0.7915
  (agreement within 1–2 Monte-Carlo standard errors, ≈ 0.0055 at n=20,000).
  Registered τ range [0 + ε, 1); reachable up to τ ≈ 1 − 1.13·10⁻⁶ (λ capped
  at 1e6), beyond which the package stores the τ that λ realises (RB-10
  convention, as Frank/Plackett/Galambos).
- `fit` recovers τ within 0.02 of the truth from n=3000 simulated pairs at
  τ = 0.2, 0.5, 0.8, by both `method='tau'` and `method='mle'`.
- `inv_h` has no closed form either and uses `CopulaVirt`'s Brent-bracketed
  default, same scope tradeoff as Galambos.
- New `pmcprg/tests/test_husler_reiss.py` (mirrors `test_galambos.py`'s
  depth: boundary rejection, the λ-direction checks above, τ(λ) against the
  quadrature and a simulated τ, A/A'/A'' against numerical differentiation,
  h/h⁻¹ round-trip, sampling's marginal uniformity and realised τ, `fit`
  recovery by both methods) and 6 new (λ) entries in
  `test_copula_limits.py`'s high-precision reference file (216 new grid
  rows; Hüsler–Reiss's own decimal reference CDF uses `mpmath` for Φ, lazily
  imported so the `min-versions` CI job — which does not install the `dev`
  extra — is unaffected as long as the reference file covers every case, as
  it now does). Every pre-existing family's cached reference is
  byte-for-byte unchanged (verified: 6 entries added, 0 removed, 0 changed).
  `CopulaEnum.available()` count: 19 (18 + Hüsler–Reiss).
- Scoping note for the two remaining FR-9 families: Tawn (asymmetry
  parameters + λ) and t-EV (a Student-style degrees-of-freedom parameter)
  are genuinely two/three-parameter families whose τ does not identify the
  extra parameter(s) — they will need `CopulaVirt`'s joint MLE machinery
  (`pmcprg.copulas._fit._fit_two_parameter_mle`, already used by BB1 and
  Student) rather than the single-parameter Brent search used here and by
  Galambos. Not attempted this round.

## [1.0.0] - 2026-09-15

First public release, under the new name **awesomePMC** (PyPI `awesomepmc`,
import package `pmcprg`). Identifiers such as "audit K-1", "FR-4" or "RB-6" in
this changelog and in code comments refer to internal review reports that are
not part of the public repository.

### Highlights

- **General pairwise Markov chains**: pair-indexed margins f_ij
  (DerrodePieczynski_CSDA2013 Eqs. 12–14) alongside state margins, for the five
  HMC/PMC variants, with 17 copula families computed in log space and checked
  against high-precision references.
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

### Changed — general PMC with pair-indexed margins f_ij (DerrodePieczynski_CSDA2013 Eqs. 12–14)

The package's "PMC" had one margin per state (f_ij = f_i), attributed to
reversibility. By the Proposition of DerrodePieczynski_CSDA2013 §2.1 that is
exactly the case where X is Markov: it was a stationary reversible HMC-DN, not
the general PMC of the paper. On the CSDA-2013 settings this was the main cause
of the gap with the published error rates (smoke run of
`report/reproduce_csda2013.py --margins pair`, 10 replicates: Tables 2 and 4
within about a point of the paper; Table 3(a) and 5(b) gaps remain).

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
- The provenance of every selection criterion is stated: only `huard` is from
  DerrodePieczynski_CSDA2013 (Eq. 20); `mle` is DerrodePieczynski_SP2016's PLM
  rule (Eq. 11), `kolmogorov` DerrodePieczynski_SP2016 Eq. (10);
  AIC/BIC/CvM/xvcic/huard_common/huard_global and the margin rules mle/aic/bic
  are package additions.

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

- `ice()` is documented as the ξ-weighted variant of ICE:
  DerrodePieczynski_CSDA2013 §4.2 (Eqs. 22–24) and DerrodePieczynski_SP2016 §3
  (b)–(c) estimate copulas and margins on **one** posterior draw (L = 1) and
  reserve the conditional expectation for the prior; `sem()` is the hard-draw
  estimator. Said in `ice.py`, the feature ↔ paper map, the README's ICE-vs-SEM
  box and §Exp. 3 of the reproduction report, which no longer claims a faithful
  reproduction of the estimator.
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

- **GICE — automatic margin family selection (paper
  DerrodePieczynski_SP2016 / SP 2016, §3).** Margin blocks now accept
  an optional ``candidates`` list (a set of ``scipy.stats`` family
  names). When non-empty, the ICE M-step fits each candidate and picks
  the winner via a configurable rule: ``mle`` (default),
  ``kolmogorov`` (SP 2016, Example 3.1), ``aic`` or ``bic``. Eight
  families ship with data-aware initialisation heuristics: ``norm,
  gamma, invgamma, betaprime, lognorm, expon, weibull_min, beta``.
  Other ``scipy.stats`` names still work; the M-step falls back to
  ``scipy.stats.<dist>.fit`` for the init point. Public constants
  ``ice.GICE_KNOWN_FAMILIES`` and ``ice.SP2016_DEFAULT_CANDIDATES``
  exposed.
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
- **Documentation: dedicated ``prg/pmc/README.md``** that opens with
  both source-paper citations (DerrodePieczynski_CSDA2013 /
  DerrodePieczynski_SP2016), a feature ↔ paper table, and BibTeX
  entries.
- **Documentation: GitHub Actions workflow** ``.github/workflows/ci.yml``
  mirroring ``.gitlab-ci.yml`` (lint + matrix tests on Python
  3.11 / 3.12 / 3.13, headless Qt via ``QT_QPA_PLATFORM=offscreen``).

### Changed

- ICE selection criteria expanded from MLE-only to ``{mle, aic, bic,
  huard, cvm}`` for copulas (DerrodePieczynski_CSDA2013 Eq. 20
  implemented as ``huard``) and ``{mle, kolmogorov, aic, bic}`` for
  margins (DerrodePieczynski_SP2016 §3).
- Default ``N_default`` lowered from 5000 → 1000 in every shipped
  fixture except ``sp2016_gice_k2.toml`` (which keeps the paper's
  3000), so the auto-loaded default lands on a fast turnaround.
- Top-level README's "References" section rewritten with full
  citations for **both** DerrodePieczynski_CSDA2013 (CSDA 2013) and
  **DerrodePieczynski_SP2016** (SP 2016), including BibTeX. Fixed
  pre-existing author typo "Piecini" → "Pieczynski"; added
  ``docs/SP_2016.pdf`` to the documented directory tree.
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

- **DerrodePieczynski_CSDA2013** — *Unsupervised data classification
  using pairwise Markov chains with automatic copulas selection*,
  Comput. Stat. Data Anal. 63 (2013), 81-98.
  [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
- **DerrodePieczynski_SP2016** — *Unsupervised classification using
  hidden Markov chain with unknown noise copulas and margins*, Signal
  Process. 128 (2016), 8-17.
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
