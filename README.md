# awesomePMC

[![PyPI version](https://img.shields.io/pypi/v/awesomepmc.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/awesomepmc/)
[![Python versions](https://img.shields.io/pypi/pyversions/awesomepmc.svg?logo=python&logoColor=white)](https://pypi.org/project/awesomepmc/)
[![CI status](https://github.com/SDerrode/awesomePMC/actions/workflows/ci.yml/badge.svg)](https://github.com/SDerrode/awesomePMC/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SDerrode/awesomePMC/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)

**Copula-based pairwise and hidden Markov chains for the unsupervised
classification of non-Gaussian time series and images.**

`awesomePMC` implements **pairwise Markov chains** (PMC) — up to the general PMC
of Derrode & Pieczynski (2013), whose observation margins fᵢⱼ are indexed by the
pair of consecutive hidden states — and the **hidden Markov chains** they
contain, with the dependence between consecutive observations modelled by
bivariate **copulas**. A model is a plain TOML file: simulate it, classify a
signal or an image with it, or estimate it from the data alone (ICE or SEM, with
automatic selection of the copula and margin families). Missing observations are
integrated out exactly. Python API, `pmc` command line and PyQt6 GUI.

Reference implementation of:

> S. Derrode, W. Pieczynski. **Unsupervised data classification using pairwise
> Markov chains with automatic copulas selection.** *Computational Statistics &
> Data Analysis* 63 (2013), 81–98.
> [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
>
> S. Derrode, W. Pieczynski. **Unsupervised classification using hidden Markov
> chain with unknown noise copulas and margins.** *Signal Processing* 128
> (2016), 8–17.
> [doi:10.1016/j.sigpro.2016.03.008](https://doi.org/10.1016/j.sigpro.2016.03.008)

---

## Why awesomePMC?

- **The general pairwise Markov chain.** Five variants, from the classical
  hidden Markov chain (HMC-IN) to the PMC of A16 Eqs. 12–14 with *pair margins*
  fᵢⱼ — the law of yₙ given (xₙ, xₙ₊₁) = (i, j), under which X is not a Markov
  chain — or *state margins* fᵢ. Margins are any `scipy.stats` family, or
  multivariate Gaussian (d > 1) for the copula-free variants.
- **17 copula families** — Gaussian, Student-t, Archimedean (Clayton,
  Gumbel–Hougaard, Frank, Joe, A12, A14, AMH), survival, BB1, Plackett, FGM,
  cubic section, independence — all parameterised by Kendall's τ, with
  numerically robust vectorised log-densities: log-space Archimedean kernels
  (Hofert, Mächler & McNeil 2012) and native elliptical log-densities, never
  floored, so strong dependence and the tails keep their likelihood.
- **Supervised classification**: normalised forward–backward, MPM decision,
  posterior marginals and forward-filtering backward-sampling draws.
- **Unsupervised estimation**: ICE and SEM (stochastic EM) with, for every pair
  of states, automatic copula-family selection among candidates (MLE, AIC, BIC,
  Huard's Bayesian evidence of A16 Eq. 20, Cramér–von Mises, cross-validated
  CIC `xvcic`); GICE margin-family selection of A23 (MLE, Kolmogorov, AIC, BIC);
  k-means warm start; multistart with parameter jitter or over copula families.
- **Is a family choice significant?** Analytic standard errors of copula
  parameters, weighted Vuong and Clarke tests with a HAC variance, confidence
  sets of families not significantly worse than the best — also for the pairs
  of states of an ICE run.
- **Missing observations**: exact inference and ICE/SEM estimation with `NaN`
  in the observations, `impute` (posterior mean, sd, quantiles, draws) and `forecast` (state
  probabilities and predictive law h steps ahead), missingness patterns with
  ImputeGAP's geometry and imputation / classification metrics
  (`pmcprg.missing`).
- **Images**: grey-level or multichannel images are linearised along a
  generalised Hilbert curve, then classified or estimated (`classify_image`,
  `ice_image`, `sem_image`, `pmc classify-image`, `pmc estimate-image`).
- **Goodness of fit**: multivariate Kolmogorov–Smirnov test (Naaman 2021),
  parametric-bootstrap calibration, Neyman smooth tests on margins.
- **Three front-ends**: Python API, `pmc` command line, PyQt6 GUI (model editor,
  simulation, classification, estimation and goodness-of-fit in background
  threads).
- **Reproducible**: [`report/`](https://github.com/SDerrode/awesomePMC/tree/main/report)
  reproduces the experiments of CSDA 2013 (§3.2, §3.3, §4.3) with committed
  results and a LaTeX report;
  [`report/missing_benchmark/`](https://github.com/SDerrode/awesomePMC/tree/main/report/missing_benchmark)
  compares exact marginalisation with naive fill-ins on 2500 simulated
  sequences. The test suite runs in CI on Python 3.11–3.13.

| Sub-package | What it does |
|---|---|
| `pmcprg.copulas` | 17 copula families, fitting, model selection, standard errors, bivariate joint laws |
| `pmcprg.pmc` | the 5 HMC/PMC variants: TOML models, simulation, classification, ICE/SEM, missing data, images, CLI, GUI |
| `pmcprg.diagnostics` | multivariate KS test, parametric bootstrap, smooth tests, Vuong/Clarke tests |
| `pmcprg.missing` | missingness patterns and metrics on masked positions |

---

## Table of Contents

- [Installation](#installation)
- [Example notebooks](#example-notebooks)
- [Copula toolkit](#copula-toolkit)
  - [Available families](#available-families)
  - [Quick start](#quick-start)
  - [Fitting and model selection](#fitting-and-model-selection)
  - [Bivariate joint laws](#bivariate-joint-laws)
- [PMC models](#pmc-models)
  - [Model variants](#model-variants)
  - [TOML model files](#toml-model-files)
  - [Python API](#python-api)
  - [Missing observations, imputation and forecasting](#missing-observations-imputation-and-forecasting)
  - [Command-line interface](#command-line-interface)
  - [Graphical interface (PyQt6)](#graphical-interface-pyqt6)
  - [Logging](#logging)
- [Diagnostics — multivariate Kolmogorov–Smirnov test](#diagnostics--multivariate-kolmogorovsmirnov-test)
- [Reproducibility reports](#reproducibility-reports)
- [Folder structure](#folder-structure)
- [Citation and references](#citation-and-references)
- [License](#license)

---

## Installation

### From PyPI (recommended)

```bash
pip install awesomepmc
```

Optional extras:

| Extra | What it adds |
|---|---|
| `[gui]` | PyQt6, for the graphical interface (`pmc gui`) — kept optional so the package installs on headless servers |
| `[ml]` | scikit-learn, for the k-means warm start of ICE/SEM (`init = "kmeans"`) |
| `[image]` | Pillow, to read and write images (`pmc classify-image`, `pmc estimate-image`, image loading in the GUI) |
| `[dev]` | pytest, pytest-cov, ruff and the notebook runners, plus `[image]` and `[ml]` — everything the test suite needs |

```bash
pip install "awesomepmc[gui]"
pip install "awesomepmc[gui,ml,image]"
```

### From source

```bash
git clone https://github.com/SDerrode/awesomePMC.git
cd awesomePMC
pip install .
```

### Development install

```bash
pip install -e ".[dev,gui]"
pytest                      # full suite; add -m "not slow" for a quick run
```

### Requirements

| Package | Min version |
|---------|------------|
| Python | ≥ 3.11 |
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.7 |
| tomli_w | ≥ 1.0 |
| PyQt6 | ≥ 6.4 *(GUI only — extra `[gui]`)* |

The code snippets below run from the root of a clone of the repository (they
load the example models of `pmcprg/pmc/models/`). The models are also shipped in
the installed package:

```python
from importlib.resources import files
from pmcprg.pmc import PMCModel

mdl = PMCModel(files("pmcprg.pmc") / "models" / "pmc_gauss_k2.toml")
```

---

## Example notebooks

[`examples/quickstart.ipynb`](https://github.com/SDerrode/awesomePMC/blob/main/examples/quickstart.ipynb)
walks through the full workflow — copula density, sampling and fitting, Sklar
bivariate law, then PMC simulate → classify → ICE — in less than a minute:

```bash
pip install jupyter
jupyter notebook examples/quickstart.ipynb
```

[`examples/uci_har_smartphone.ipynb`](https://github.com/SDerrode/awesomePMC/blob/main/examples/uci_har_smartphone.ipynb)
runs ICE and SEM on a real 3-D accelerometer recording of the *Human Activity
Recognition Using Smartphones* dataset (UCI ID 240). The dataset is downloaded
on first run and cached in `data/uci_har/` (not versioned); if the download
fails or is declined, the notebook falls back to a synthetic 3-D signal
simulated from the bundled `hmc_in_mvn_k2_d3.toml` model and every later cell
runs unchanged. It exercises the k-means warm start (`init = "kmeans"`), the SEM
estimator (`from pmcprg.pmc import sem`) and the multivariate KS test
(`from pmcprg.diagnostics import mks_2samp`).

Both notebooks are re-run by the test suite (tests marked `slow`).

---

## Copula toolkit

### Available families

All copulas are **τ-parameterised** (Kendall's τ is the single free dependence
parameter). The Student-t copula additionally has a free `df` parameter, fitted
jointly by 2-D MLE.

| ID | SHORT_NAME | Family | τ range | Tail dep. |
|----|-----------|--------|---------|-----------|
| 1 | `Prod` | Independence (Product) | {0} | none |
| 2 | `Gauss` | Gaussian | (−1, 1) | none |
| 3 | `Student` | Student-t (free ν > 2) | (−1, 1) | symmetric |
| 4 | `GH` | Gumbel-Hougaard | (0, 1) | upper |
| 5 | `FGM` | Farlie-Gumbel-Morgenstern | (−2/9, 2/9) | none |
| 6 | `CubSec` | Cubic Section | (0, 0.165) | none |
| 7 | `Clayton` | Clayton | (0, 1) | lower |
| 8 | `A12` | Archimedean 12 | (1/3, 1) | upper |
| 9 | `A14` | Archimedean 14 | (1/3, 1) | upper |
| 10 | `Frank` | Frank | (−1, 1) | none |
| 11 | `Joe` | Joe | (0, 1) | upper |
| 12 | `SClayton` | Survival Clayton | (0, 1) | upper |
| 13 | `SGH` | Survival Gumbel-Hougaard | (0, 1) | lower |
| 14 | `SJoe` | Survival Joe | (0, 1) | lower |
| 15 | `BB1` | BB1 (Joe-Clayton) | (0, 1) | both |
| 16 | `AMH` | Ali-Mikhail-Haq | (≈−0.18, 1/3) | none |
| 17 | `Plackett` | Plackett | (−1, 1) | none |

### Quick start

```python
from pmcprg.copulas import CopulaGaussian, CopulaStudent, CopulaClayton

# Instantiate by Kendall's τ
gauss  = CopulaGaussian(tau_k=0.5)
stud   = CopulaStudent(tau_k=0.5, df=4.0)    # df defaults to 4.0
clay   = CopulaClayton(tau_k=0.6)

# Evaluate
u, v = 0.3, 0.7
print(gauss.pdf([u, v]))          # copula density c(u,v)
print(gauss.cdf([u, v]))          # copula CDF C(u,v)
print(gauss.conditional_cdf(v, u))# h-function h(v|u) = ∂C/∂u
print(gauss.tail_dependence())    # (λ_L, λ_U)

# Sampling  (Rosenblatt inversion)
samples = gauss.sample(n=1000, seed=42)   # shape (1000, 2) on [0,1]²
```

### Fitting and model selection

Data are rank-transformed to pseudo-observations before fitting, so the margins
need not be uniform.

```python
from pmcprg.copulas import CopulaClayton, CopulaFrank, CopulaGaussian, CopulaGH
from pmcprg.copulas._base import CopulaVirt

data = CopulaClayton(tau_k=0.5).sample(n=500, seed=0)

# Single-family fit
result = CopulaGaussian.fit(data, method='tau')   # or method='mle'
print(result.tau_k, result.aic, result.bic)

# 5-fold cross-validation log-likelihood
print(result.cv_loglik(K=5))

# Parametric bootstrap GoF  (Cramér-von Mises)
gof = result.gof_test(B=200)
print(gof.p_value)

# Bootstrap CI on τ
print(result.bootstrap_ci(B=500))

# Analytic standard errors on τ and the native parameters (sandwich for 'mle')
se = CopulaClayton.fit(data, method='mle').standard_errors()
print(se, se.ci())

# Automatic family selection (every family if `families` is omitted)
ranked = CopulaVirt.fit_best(
    data, families=[CopulaGaussian, CopulaClayton, CopulaGH, CopulaFrank], method='mle')
best = ranked[0]                        # sorted by AIC ascending
print(best.copula, best.aic)

# Is the ranking significant?  Vuong tests (HAC variance) and the ties
print(ranked.compare())                 # or ranked.compare(test="clarke")
print(ranked.confidence_set().members)  # families not significantly worse than the best
```

### Bivariate joint laws

```python
from pmcprg.copulas import CopulaGaussian
from pmcprg.copulas.bivariate import BivariateLaw
import scipy.stats as ss

law = BivariateLaw(
    copula=CopulaGaussian(tau_k=0.6),
    left_margin=(ss.norm, -1, 1),    # (scipy_dist, *params)
    right_margin=(ss.norm, +1, 1),
)
print(law.pdf([-0.5, 0.8]))   # joint density f(x,y)
print(law.cdf([ 0.0, 1.0]))   # joint CDF F(x,y)
law.set_seed(0)               # reproducible sampling
samples = law.sample(1000)    # shape (1000, 2)

# IFM two-step fit
fit = BivariateLaw.fit(samples, copula_class=CopulaGaussian,
                       left_family=ss.norm, right_family=ss.norm)
print(fit.copula_fit.tau_k, fit.aic)
```

---

## PMC models

### Model variants

Five variants of the jointly Markovian pair process (X, Y) are supported, from
the simplest to the most general. Every model has one of two **margin
structures** (`[model].margin_structure`):

- **`"state"`** — K densities fᵢ, one per state (fᵢⱼ = fᵢ). By the Proposition
  of A16 §2.1, this is exactly the case where X is a Markov chain: a PMC with
  state margins is a stationary reversible HMC-DN, and a PMC-IN with state
  margins an HMC-IN. Every HMC-* variant has state margins.
- **`"pair"`** — K² densities fᵢⱼ, the law of yₙ given (xₙ, xₙ₊₁) = (i, j): the
  **general PMC** of A16 Eqs. 12–14, where X is *not* Markov. Allowed for PMC
  and PMC-IN only.

| Variant | Prior | Observation density (state margins) | Pair margins | Copula |
|---------|-------|--------------------|------|--------|
| **HMC-IN** | Transition matrix A | fⱼ(yₙ) | — | No |
| **HMC-IN2** | Transition matrix A | fⱼ(yₙ) | — | No |
| **HMC-DN** | Transition matrix A | fⱼ(yₙ) · cᵢⱼ(Fᵢ(yₙ₋₁), Fⱼ(yₙ)) | — | Yes — cᵢⱼ |
| **PMC-IN** | Joint distribution p | fᵢ(yₙ) · fⱼ(yₙ₊₁) | fᵢⱼ(yₙ) · fⱼᵢ(yₙ₊₁) | No |
| **PMC** | Joint distribution p | fᵢ(yₙ) · fⱼ(yₙ₊₁) · cᵢⱼ(Fᵢ(yₙ), Fⱼ(yₙ₊₁)) | fᵢⱼ(yₙ) · fⱼᵢ(yₙ₊₁) · cᵢⱼ(Fᵢⱼ(yₙ), Fⱼᵢ(yₙ₊₁)) | Yes — cᵢⱼ |

Stationarity and reversibility only make the right margin of the pair (i, j)
the left margin of (j, i) — the index inversion fⱼᵢ of A16 Eq. 12; they do
**not** make fᵢⱼ independent of j. Versions 0.5.0–0.8.x collapsed K² margins to
fᵢ on that mistaken ground.

### TOML model files

Models are stored as TOML files. State margins use the **K-format** (one block
per state, indexed by `i` only); pair margins use the **K²-format** (one block
per pair, keys `i` and `j`, see
[`pmc_pair_gauss_k2.toml`](https://github.com/SDerrode/awesomePMC/blob/main/pmcprg/pmc/models/pmc_pair_gauss_k2.toml)):

```toml
[model]
name      = "PMC Gaussien K=2"
variant   = "PMC"        # HMC-IN | HMC-IN2 | HMC-DN | PMC-IN | PMC
K         = 2
N_default = 5000
# margin_structure = "state"   # optional: "state" | "pair"; default inferred
#                                # (K blocks → state, K² blocks → pair)

[prior]
# HMC-* variants: key "A" — K×K row-stochastic transition matrix
# PMC-* variants: key "p" — K×K symmetric joint distribution (sums to 1)
p = [[0.45, 0.05],
     [0.05, 0.45]]

# K margin blocks (one per state) — state margins f_i.
[[margins]]
i      = 0
dist   = "norm"          # any scipy.stats distribution name
params = {loc = -1.0, scale = 1.0}
# candidates = ["norm", "gamma", "invgamma", "betaprime"]   # GICE: families to
#                        # select from when fit_margins = true

[[margins]]
i      = 1
dist   = "norm"
params = {loc = 1.0, scale = 1.0}

[[copulas]]              # only for HMC-DN and PMC — K² blocks (i, j)
i    = 0
j    = 0
name = "Gauss"           # SHORT_NAME from the copula table above
tau  = 0.6
# df = 4.0               # optional extra param for Student copula
# … blocks (0, 1), (1, 0) and (1, 1)

[ice]                    # optional — ICE estimator defaults (also consumed
                         # by SEM for the keys they share)
fit_margins = false
max_iter    = 50
tol         = 1e-4
candidates  = ["Gauss", "Clayton", "GH", "Frank", "Joe"]
selection_criterion = "mle"      # "mle" (default) | "aic" | "bic" | "huard" | "cvm" | "xvcic"
                                 # (+ "huard_common", "huard_global")
margin_selection_rule = "mle"    # GICE: "mle" (default) | "kolmogorov" | "aic" | "bic"
init        = "model"    # "model" (default) | "kmeans"
# kmeans_seed = 0        # RNG seed for sklearn.cluster.KMeans (init="kmeans")
# n_starts = 1           # multistart: best final log-likelihood among n_starts runs
# multistart_families = "none"   # "none" (default) | "random" | "sweep"
# return_best_iterate = false    # true: return the highest-log-likelihood iterate, not the last

# [sem]                  # optional — SEM-specific overrides
# max_iter = 30          # SEM does not converge; defaults to fewer iterations
# sem_seed = 0           # RNG seed for the per-iteration FFBS draw
```

> **ICE vs SEM.** Both estimators share the same M-step and config keys. ICE
> (`algorithm = "ice"`, default) uses *soft* posteriors from forward-backward
> and converges deterministically. SEM (`algorithm = "sem"`) draws a single
> realisation `X̃ ~ P(X | Y)` at every iteration via Forward-Filter
> Backward-Sample, then runs the same M-step on the hard labels — the
> log-likelihood fluctuates around its stationary regime instead of
> converging. CLI: `pmc estimate --algorithm sem --sem-seed 0 …`. GUI: the
> **Estimator** combobox in the ICE-config tab. Note that the papers (A16 §4.2,
> Eqs. 22–24; A23 §3) estimate copulas and margins on *one posterior draw*
> (L = 1) and reserve the expectation for the prior — closer to SEM than to
> this package's ICE; see the docstring of `pmcprg/pmc/ice.py`.
>
> **Selection criteria.** At every M-step, each pair (i, j) gets the copula
> family of `candidates` that is best under `selection_criterion`; its τ is
> always the maximum-likelihood estimate. A margin block that declares
> `candidates` (GICE, A23 §3) gets its family chosen by `margin_selection_rule`
> when `fit_margins = true` — see
> [`sp2016_gice_k2.toml`](https://github.com/SDerrode/awesomePMC/blob/main/pmcprg/pmc/models/sp2016_gice_k2.toml).
>
> **K-means warm-start.** Setting `init = "kmeans"` clusters `Y` with k-means++
> and derives a hard-labelled warm-start model via a single supervised-style
> M-step (prior + copula τ always re-estimated; margins re-estimated only if
> `fit_margins = true`) — useful when the declared initial parameters are far
> from the data. Requires scikit-learn (`pip install "awesomepmc[ml]"`). Works
> for both ICE and SEM.
>
> **Multistart over copula families.** With `n_starts > 1` ICE/SEM keep the
> run with the highest final log-likelihood; the extra starts jitter the
> parameters (`multistart_jitter`) but keep the copula families of the initial
> model. Those families can decide the fixed point: on the CSDA-2013 Exp. 3
> design, ICE started at independence picks Gumbel instead of the true
> Gaussian copula on pair (0, 0) and no amount of jitter changes it.
> `multistart_families = "random"` also redraws the family of every pair
> (i, j) among `candidates` at each extra start (τ uniform on the central 60 %
> of the family's range); `"sweep"` runs every combination of candidate
> families on the diagonal pairs (i, i) at τ = 0.5, after the model itself —
> set `n_starts = 1 + |C|^K` (10 for 3 candidates and K = 2; fewer drops
> combinations with a warning, more adds random starts; beyond 256
> combinations use `"random"`). The winning start is `trace.run_tag`
> (`family-sweep:Gauss/Clayton`, `family-random-4`). Variants without copulas
> ignore the key. GUI: **Start families** in the ICE-config tab.
>
> **Best iterate.** ICE is not monotone, and after `patience` consecutive
> regressions it stops on its last iterate, possibly several nats below its
> best one. `return_best_iterate = true` returns the iterate θ^q with the
> highest `trace.log_liks[q]` (computed with θ^q, before the M-step); a run
> that reaches `max_iter` evaluates the model of its last M-step once more, so
> its trace has `max_iter + 1` entries. Multistart then ranks the starts by the
> log-likelihood of the model each one returns. For SEM it is a heuristic (the
> best point of a noisy chain, not its average). `trace.best_iter` is the
> argmax of `log_liks` and `trace.returned_iter` the iterate returned. Default
> `false` (unchanged results). CLI: `pmc estimate --best-iterate`. GUI:
> **Return best iterate** in the ICE-config tab.
>
> **Degenerate fits.** The likelihood is unbounded: a variance collapsing on
> an atom of discretised data, or a copula driven to τ = ±1, gains nats
> without limit. ICE and SEM check the returned model and log one `WARNING`
> (also printed by `pmc estimate`, which adds a `Degenerate` summary line) when
> a state has a stationary weight below 0.5 %, a margin (state or pair) a
> standard deviation below 1 % of that of the data, or a copula τ within 1e-3
> of ±1. The findings are in `trace.degenerate` (`[]` when none); the fit
> itself is unchanged. Thresholds and their justification:
> `pmcprg.pmc._estim_common.degenerate_states`.

> **K²-format and older files.** Blocks indexed by `(i, j)` are kept as pair
> margins fᵢⱼ on PMC and PMC-IN (an `INFO` line says so once per process), and
> ICE/SEM estimate each fᵢⱼ separately. Older files that relied on the collapse
> to K state margins must add `margin_structure = "state"` under `[model]` (the
> `(i, 0)` block is then kept as fᵢ). HMC-* variants refuse pair margins.
>
> **Estimating pair margins.** ICE and SEM fit fᵢⱼ by weighted maximum
> likelihood on both views of the pair density: yₙ with weight ξₙ(i, j) and
> yₙ₊₁ with weight ξₙ(j, i) (each halved, so the total weight stays N − 1). It
> is an ICE-style estimator, not an exact EM M-step. Copula pseudo-observations
> for cᵢⱼ are (Fᵢⱼ(yₙ), Fⱼᵢ(yₙ₊₁)) with weight ξₙ(i, j).

Nine example models are provided in
[`pmcprg/pmc/models/`](https://github.com/SDerrode/awesomePMC/tree/main/pmcprg/pmc/models):

| File | Variant |
|------|---------|
| `hmc_in_gauss_k2.toml` | HMC-IN (classical HMM) |
| `hmc_in_gauss_k3.toml` | HMC-IN, K = 3 classes |
| `hmc_in_mvn_k2_d3.toml` | HMC-IN, multivariate Gaussian observations (d = 3, RGB) |
| `hmc_in2_gauss_k2.toml` | HMC-IN2 |
| `hmc_dn_gauss_k2.toml` | HMC-DN (copula-dependent HMM) |
| `sp2016_gice_k2.toml` | PMC with non-Gaussian state margins, i.e. an SR HMC-DN (GICE fixture, SP-2016 §5.1) |
| `pmc_in_gauss_k2.toml` | PMC-IN |
| `pmc_gauss_k2.toml` | PMC with state margins (an SR HMC-DN, A16 §2.1 Proposition) |
| `pmc_pair_gauss_k2.toml` | General PMC with pair margins fᵢⱼ (CSDA-2013 Table 1 Gaussian margins, Clayton τ = 0.7) |

### Python API

```python
from pmcprg.pmc import PMCModel, simulate, classify, ice

# ── Load model ────────────────────────────────────────────────────
mdl = PMCModel("pmcprg/pmc/models/pmc_gauss_k2.toml")
print(mdl)                    # PMCModel(name='PMC Gaussien K=2', …)
print(mdl.stationary_pi)      # [0.5, 0.5]
print(mdl.transition_A)       # [[0.9, 0.1], [0.1, 0.9]]
print(mdl.prior_p)            # [[0.45, 0.05], …]

# Margin/copula access
print(mdl.pdf(0, 1, -0.5))    # f_{01}(-0.5)
print(mdl.cdf(1, 0, 0.3))     # F_{10}(0.3)
cop = mdl.copula(0, 0)        # CopulaGaussian(τ=0.6)

# ── Simulate ──────────────────────────────────────────────────────
X, Y = simulate(mdl, N=5000, seed=42)
# X: (5000,) int — latent states 0/1
# Y: (5000,) float — observations

# ── Supervised classification (MPM) ───────────────────────────────
X_hat, gamma, log_lik = classify(mdl, Y)
# gamma: (5000, K) — posterior P(X_n=k | Y)
from pmcprg.pmc import error_rate
print(f"Error rate: {error_rate(X, X_hat):.3f}")

# ── Unsupervised estimation (ICE) ─────────────────────────────────
raw_init = mdl.raw            # deep copy of the TOML dict — safe to mutate
# … perturb parameters …
init_mdl = PMCModel.from_dict(raw_init)
fitted, trace = ice(init_mdl, Y,
    ice_cfg={"max_iter": 30, "candidates": ["Gauss", "Clayton", "GH"]})
print(trace.log_liks[-1])     # final log-likelihood
print(trace.best_iter, trace.returned_iter, trace.degenerate)
fitted.save("fitted_pmc.toml")

# ── Edit and save ─────────────────────────────────────────────────
mdl.save("my_model.toml")
```

`sem` has the same signature as `ice` (`sem_cfg=` instead of `ice_cfg=`), and
`classify_image`, `ice_image` and `sem_image` take a 2-D array `(H, W)` or
`(H, W, d)` instead of `Y`.

### Missing observations, imputation and forecasting

Observations may contain `NaN` (empty, `NaN` or `NA` cells in CSV files). With
a known model, supervised inference integrates the missing values out exactly
(up to quadrature), for every variant and both margin structures:

```python
import numpy as np
from pmcprg.pmc import PMCModel, simulate, classify, impute, forecast
from pmcprg.missing import patterns, metrics

mdl = PMCModel("pmcprg/pmc/models/pmc_pair_gauss_k2.toml")
X, Y = simulate(mdl, N=2000, seed=1)
Y_gap, mask = patterns.mcar(Y, 0.2, block_size=10, seed=2)   # ImputeGAP-like pattern

X_hat, gamma, log_lik = classify(mdl, Y_gap)     # states at every n, missing ones included
imp = impute(mdl, Y_gap, quantiles=(0.05, 0.5, 0.95), n_samples=100, rng=np.random.default_rng(0))
# imp.index, imp.mean, imp.sd, imp.quantile_values, imp.gamma, imp.x_samples, imp.y_samples
fc = forecast(mdl, Y_gap, h=10)                  # fc.state_probs, fc.mean, fc.sd, fc.quantile_values
print(metrics.error_rate_split(X, X_hat, mask))  # error on missing / observed positions
```

* **Exact shortcut** — HMC-IN, HMC-IN2 and PMC-IN with state margins: the
  transition of X does not depend on a missing value, whose density is set to 1.
* **Augmented grid** — HMC-DN, PMC and PMC-IN with pair margins: inside a run of
  missing values the forward–backward runs on (state, value) over `gap_nodes`
  (default 64) Gauss–Legendre nodes in the quantiles of the stationary law of y,
  with an endpoint transform for copula tails. Checked against exact references
  (brute force over state paths, Gaussian AR(1) bridges and forecasts, distinct
  Gaussian regimes, `quad` for Clayton/Gumbel pair models): log-likelihood and
  posteriors to 1e-7 or better at G = 64 except in extreme tails
  ([`test_gaps_references.py`](https://github.com/SDerrode/awesomePMC/blob/main/pmcprg/tests/test_gaps_references.py)).
* A forecast is a trailing gap; `forecast` returns the state probabilities and
  the predictive law of the observation k = 1, …, h steps ahead.
* **Estimation**: `ice`, `sem` (and `pmc estimate`, which prints the number of
  missing values) estimate from the observed data; `trace.log_liks` is
  log p(y_obs). SEM draws the states and the missing values jointly at each
  iteration. ICE has two strategies (config key `missing_strategy`):
  `"available"` (default; exact posteriors given the observed data, margins fitted
  on observed values, copulas on pairs with both ends observed; deterministic)
  and `"impute"` (`missing_draws` completed series per iteration, estimates
  averaged; better when GICE selects margin families). Complete data give
  bit-identical results to the previous code.

  ```python
  fitted, trace = ice(init_mdl, Y_gap, ice_cfg={"missing_strategy": "available"})
  ```
* `pmcprg.missing.patterns` generates missingness patterns with the geometry of
  ImputeGAP's `GenGap` (mcar, aligned, scattered, blackout, disjoint, overlap,
  gaussian, distribution); `pmcprg.missing.metrics` scores imputations (RMSE, MAE,
  MI, Pearson, CRPS, interval coverage) and classification on missing vs
  observed positions.

### Command-line interface

```
pmc COMMAND [options]            # or: python -m pmcprg.pmc COMMAND [options]
```

| Command | Description |
|---------|-------------|
| `simulate` | Generate a synthetic (X, Y) sequence |
| `classify` | Supervised MPM classification of a 1-D signal |
| `classify-image` | Supervised MPM classification of a 2-D image (generalised Hilbert path) |
| `estimate` | Unsupervised parameter estimation — `--algorithm {ice,sem}` (default `ice`) |
| `estimate-image` | Unsupervised estimation on a 2-D image — `--algorithm {ice,sem}` |
| `gui` | Launch the PyQt6 graphical interface |

`pmc COMMAND --help` lists the options of each command.

**Examples:**

```bash
# Simulate 10 000 samples and save to CSV
pmc simulate \
    --model pmcprg/pmc/models/pmc_gauss_k2.toml \
    --N 10000 --seed 42 --out sim.csv

# Classify (reference labels in column X → prints error rate)
pmc classify \
    --model pmcprg/pmc/models/pmc_gauss_k2.toml \
    --data sim.csv --out cls.csv --ref X

# Unsupervised ICE estimation (--algorithm sem --sem-seed 0 for SEM)
pmc estimate \
    --model pmcprg/pmc/models/pmc_gauss_k2.toml \
    --data sim.csv \
    --candidates "Gauss,Clayton,GH" \
    --max-iter 50 \
    --out fitted.toml

# Image: estimate from the image, then segment it (needs the [image] extra)
pmc estimate-image --model init.toml --image photo.png --fit-margins --out fitted.toml
pmc classify-image --model fitted.toml --image photo.png --out segmentation.png

# Launch GUI (with optional startup model)
pmc gui pmcprg/pmc/models/pmc_gauss_k2.toml

# Verbose mode (DEBUG to console)
pmc --verbose simulate --model pmcprg/pmc/models/pmc_gauss_k2.toml --N 500
```

### Graphical interface (PyQt6)

```bash
pmc gui [MODEL.toml]
```

The window is divided into a **model editor** (left) and a **result viewer** (right):

```
┌─────────────────────────────────────┬────────────────────────────────┐
│  Model info  (name, variant, K, N)  │  Matplotlib result panel       │
│  ─────────────────────────────────  │  • Simulate: scatter, histo-   │
│  Tabs                               │    gram, successive-pair plot  │
│    Prior  — editable K×K matrix     │  • Classify: MPM labels,       │
│    Margins — state / pair structure │    posterior curves, error map │
│              (dbl-click a cell)     │  • Estimate: ICE log-lik curve │
│    Copulas — K×K grid (dbl-click)   │                                │
│    ICE Config — estimator, criteria,│                                │
│              init, multistart       │  Log panel (INFO+ records)     │
│  ─────────────────────────────────  │                                │
│  [▶ Simulate]   [◈ Classify]        │                                │
│  [⟳ Estimate]   [✓ GoF test]        │                                │
└─────────────────────────────────────┴────────────────────────────────┘
```

- **File menu**: open / save TOML models; load data (CSV) or an image; save
  data, segmentation, results and plots
- **Analysis menu**: τ confidence intervals, margin adequacy (KS)
- **Double-click** any Margins or Copulas cell to edit the distribution or copula parameters
- Long-running operations (simulate, classify, estimate) run in a background thread — the GUI stays responsive

### Logging

All `pmcprg.*` loggers route to three sinks:

| Sink | Level | Location |
|------|-------|----------|
| **File** | DEBUG (full tracebacks) | `~/.awesomepmc/pmc.log` |
| **Console** | WARNING (or DEBUG with `--verbose`) | stderr |
| **GUI log panel** | INFO+ | bottom-right panel in the GUI |

The file handler captures every `logger.exception(...)` call with its full
traceback, the `logger.debug(...)` records of the forward–backward weights, ICE
iterations and copula fallbacks, and the `logger.warning(...)` records of
numerical near-failures (zero normalisation constants, incompatible
observations).

To configure programmatically:

```python
import logging
from pmcprg.pmc.logging_setup import configure, add_widget_handler

# Point the file to a custom path; set console to DEBUG
configure(level=logging.DEBUG, log_file="run.log")

# In a PyQt6 application, once the QApplication exists, mirror the records
# in a QTextEdit:
# add_widget_handler(text_edit, level=logging.INFO)
```

---

## Diagnostics — multivariate Kolmogorov–Smirnov test

The `pmcprg.diagnostics` sub-package collects standalone goodness-of-fit
utilities that complement the estimation routines of `pmcprg.pmc`: the
multivariate KS test below, `parametric_bootstrap` (calibration of any post-fit
statistic by resampling whole series from the fitted model), Neyman smooth-test
components of a fitted margin (`neyman_components`, `neyman_test`) and the
Vuong / Clarke family comparisons (`vuong_test`, `clarke_test`,
`confidence_set`, `ice_pair_comparisons`).

```python
import numpy as np
from scipy import stats
from pmcprg.diagnostics import mks_1samp, mks_2samp

rng = np.random.default_rng(0)

# 1-sample: is `x` drawn from a 2-D standard normal?
x   = rng.standard_normal(size=(300, 2))
cdf = lambda t: float(stats.norm.cdf(t[0]) * stats.norm.cdf(t[1]))
res = mks_1samp(x, cdf, alpha=0.05)
print(res.statistic, res.critical_value, res.reject)

# 2-sample: do `a` and `b` share the same distribution?
a = rng.standard_normal(size=(300, 2))
b = rng.standard_normal(size=(300, 2)) + 2.0     # shifted mean → reject H0
res = mks_2samp(a, b, alpha=0.05)
assert res.reject
```

The test extends the classical 1-D Kolmogorov–Smirnov statistic to `d > 1`
dimensions via Naaman's construction (*Statistics & Probability Letters* 173,
2021). The default critical value uses the finite-sample union bound (safe
but conservative); pass `asymptotic=True` for the tighter large-`N`
approximation.

---

## Reproducibility reports

- [`report/`](https://github.com/SDerrode/awesomePMC/tree/main/report) reproduces
  the experiments of A16 §3.2 (supervised PMC, impact of the copula shape), §3.3
  (i.i.d. PMM baseline) and §4.3 (unsupervised ICE-based copula selection), with
  the committed CSV results, LaTeX tables and the report
  [`csda2013_reproduction.pdf`](https://github.com/SDerrode/awesomePMC/blob/main/report/csda2013_reproduction.pdf).
  Its [README](https://github.com/SDerrode/awesomePMC/blob/main/report/README.md)
  lists what the reproduction established, the commands and the provenance of
  every result.
- [`report/missing_benchmark/`](https://github.com/SDerrode/awesomePMC/tree/main/report/missing_benchmark)
  compares classification and imputation with missing observations under known
  models: exact marginalisation against plug-in, linear, LOCF and mean fill-ins,
  on 5 models × 5 patterns × 5 missing rates × 20 replicates.

A23 (GICE, §5.1 setting) is exercised by the test suite on the bundled
`sp2016_gice_k2.toml` model.

---

## Folder structure

```text
awesomePMC/
├── pmcprg/                 import package
│   ├── copulas/            17 copula families (archimedean/, elliptical/, explicit/),
│   │                       fitting, standard errors, bivariate joint laws
│   ├── diagnostics/        multivariate KS, parametric bootstrap, smooth tests,
│   │                       Vuong/Clarke tests, pseudo-observations
│   ├── missing/            missingness patterns, metrics, CSV missing cells
│   ├── pmc/                model, simulate, inference, gaps (missing data),
│   │   │                   ice, sem, peano (images), cli
│   │   ├── gui/            PyQt6 interface
│   │   └── models/         example TOML models
│   └── tests/              pytest suite
├── examples/               quickstart.ipynb, uci_har_smartphone.ipynb
├── report/                 CSDA 2013 reproduction: scripts, results/, tables/,
│   │                       figures/, LaTeX report and PDF
│   └── missing_benchmark/  missing-data benchmark
├── scripts/                maintenance scripts
├── data/                   local data folders (UCI HAR cache, not versioned)
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
├── README.md
├── REFERENCES.md
├── pyproject.toml
└── requirements.txt
```

---

## Citation and references

Please cite **both** A16 and A23 if you use this package in published work.
Machine-readable citation metadata is in
[`CITATION.cff`](https://github.com/SDerrode/awesomePMC/blob/main/CITATION.cff)
(GitHub's *Cite this repository* button reads it). Every reference the code and
the reports rely on is listed with its DOI and the module that uses it in
[`REFERENCES.md`](https://github.com/SDerrode/awesomePMC/blob/main/REFERENCES.md).

The `pmcprg.pmc` sub-package implements the unsupervised classification
methods of two papers by **S. Derrode and W. Pieczynski**. The CSDA paper (A16)
underpins the PMC model family and ICE-based copula selection; the Signal
Processing paper (A23) adds GICE — automatic margin family selection — on top
of ICE. [`pmcprg/pmc/README.md`](https://github.com/SDerrode/awesomePMC/blob/main/pmcprg/pmc/README.md)
maps every feature to its paper.

- **A16** — Derrode S., Pieczynski W. *Unsupervised data classification using pairwise Markov chains with automatic copulas selection*. Computational Statistics & Data Analysis 63 (2013), pp. 81–98. [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
- **A23** — Derrode S., Pieczynski W. *Unsupervised classification using hidden Markov chain with unknown noise copulas and margins*. Signal Processing 128 (2016), pp. 8–17. [doi:10.1016/j.sigpro.2016.03.008](https://doi.org/10.1016/j.sigpro.2016.03.008)

```bibtex
@ARTICLE{A16,
  author    = {S. Derrode and W. Pieczynski},
  title     = {Unsupervised data classification using pairwise {M}arkov chains with automatic copulas selection},
  journal   = {Comput. Stat. Data Anal.},
  volume    = {63},
  pages     = {81--98},
  year      = {2013},
  doi       = {10.1016/j.csda.2013.01.027},
}

@ARTICLE{A23,
  author    = {S. Derrode and W. Pieczynski},
  title     = {Unsupervised classification using hidden {M}arkov chain with unknown noise copulas and margins},
  journal   = {Signal Process.},
  volume    = {128},
  pages     = {8--17},
  year      = {2016},
  doi       = {10.1016/j.sigpro.2016.03.008},
}
```

The general PMC is defined by A16 Eqs. 12–14 (pair margins fᵢⱼ):

- **A16, Eq. 12**: f(yₙ, yₙ₊₁ | Xₙ = i, Xₙ₊₁ = j) = fᵢⱼ(yₙ) · fⱼᵢ(yₙ₊₁) · cᵢⱼ(Fᵢⱼ(yₙ), Fⱼᵢ(yₙ₊₁))
- **A16, Eq. 13**: p(Xₙ₊₁ = j | Xₙ = i, Yₙ = y) ∝ p(i, j) · fᵢⱼ(y)
- **A16, Eq. 14**: p(Yₙ₊₁ | Xₙ = i, Xₙ₊₁ = j, Yₙ = y) = fⱼᵢ(yₙ₊₁) · cᵢⱼ(Fᵢⱼ(y), Fⱼᵢ(yₙ₊₁))

With state margins (fᵢⱼ = fᵢ) Eq. 13 no longer depends on y, X is a Markov
chain (A16 §2.1, Proposition) and the model is a stationary reversible HMC-DN —
see [Model variants](#model-variants) above.

Methodological building blocks used by the package:

- Huard D., Évin G., Favre A.-C. *Bayesian copula selection*. Computational Statistics & Data Analysis 51(2) (2006), pp. 809–822 — the `huard` selection criterion (A16, Eq. 20). [doi:10.1016/j.csda.2005.08.010](https://doi.org/10.1016/j.csda.2005.08.010)
- Naaman M. *On the tight constant in the multivariate Dvoretzky–Kiefer–Wolfowitz inequality*. Statistics & Probability Letters 173 (2021), 109088 — the multivariate KS test in `pmcprg.diagnostics`. [doi:10.1016/j.spl.2021.109088](https://doi.org/10.1016/j.spl.2021.109088)
- Anguita D. et al. *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013, pp. 437–442 (the paper has no DOI) — the UCI HAR benchmark used in `examples/`. Dataset DOI (UCI, Reyes-Ortiz & Anguita 2013): [doi:10.24432/C54S4K](https://doi.org/10.24432/C54S4K)

Related work (a selection; the references the code and the reports rely on are
listed in
[`REFERENCES.md`](https://github.com/SDerrode/awesomePMC/blob/main/REFERENCES.md)):

- Gorynin I., Gangloff H., Monfrini E., Pieczynski W. *Assessing the segmentation performance of pairwise and triplet Markov models*. Signal Processing 145 (2018), pp. 183–192 — quantifies the PMM/TMM gain over classical HMMs. [doi:10.1016/j.sigpro.2017.12.006](https://doi.org/10.1016/j.sigpro.2017.12.006)
- Gangloff H., Morales K., Petetin Y. *Deep parameterizations of pairwise and triplet Markov models for unsupervised classification of sequential data*. Computational Statistics & Data Analysis 180 (2023), 107663 — the deep-learning branch of the PMC/TMC lineage. [doi:10.1016/j.csda.2022.107663](https://doi.org/10.1016/j.csda.2022.107663)
- Zimmerman R., Craiu R.V., Leos-Barajas V. *Copula Modelling of Serially Correlated Multivariate Data with Hidden Structures*. JASA 119(548) (2024), pp. 2598–2609 — the closest copula-HMM methodology outside the pairwise family. [doi:10.1080/01621459.2023.2263202](https://doi.org/10.1080/01621459.2023.2263202)
- Nasri B.R., Rémillard B.N., Thioub M.Y. *Goodness-of-fit for regime-switching copula models*. Canadian Journal of Statistics 48(1) (2020), pp. 79–96 — regime-switching copulas and the R package `HMMcopula`. [doi:10.1002/cjs.11534](https://doi.org/10.1002/cjs.11534)
- Grønneberg S., Hjort N.L. *The Copula Information Criteria*. Scandinavian Journal of Statistics 41(2) (2014), pp. 436–459 — why AIC on rank pseudo-observations needs correction (xv-CIC). [doi:10.1111/sjos.12042](https://doi.org/10.1111/sjos.12042)

---

## License

MIT — see [`LICENSE`](https://github.com/SDerrode/awesomePMC/blob/main/LICENSE).
