# copulasformm  ·  v0.4.0

**Copula-based Markov models** for flexible, non-Gaussian transition kernels.

The library provides two complementary layers:

| Layer | Package | What it does |
|---|---|---|
| **Copula toolkit** | `prg.copulas` | 17 copula families, bivariate joint laws, fitting, diagnostics |
| **PMC models** | `prg.pmc` | 5 Hidden/Pairwise Markov Chain variants, simulation, classification, ICE estimation, PyQt6 GUI |

---

## Table of Contents

- [Installation](#installation)
- [Quickstart notebook](#quickstart-notebook)
- [Copula toolkit](#copula-toolkit)
  - [Available families](#available-families)
  - [Quick start](#quick-start)
  - [Fitting and model selection](#fitting-and-model-selection)
  - [Bivariate joint laws](#bivariate-joint-laws)
- [PMC models](#pmc-models)
  - [Model variants](#model-variants)
  - [TOML model files](#toml-model-files)
  - [Python API](#python-api)
  - [Command-line interface](#command-line-interface)
  - [Graphical interface (PyQt6)](#graphical-interface-pyqt6)
  - [Logging](#logging)
- [Folder structure](#folder-structure)
- [References](#references)

---

## Installation

### From source

```bash
git clone https://gitlab.ec-lyon.fr/sderrode/copulasformm.git
cd copulasformm
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

### Optional dependency — GUI

The PyQt6 GUI is not listed as a hard dependency (to keep the package usable in headless/server environments). Install it separately:

```bash
pip install PyQt6
```

### Requirements

| Package | Min version |
|---------|------------|
| Python | ≥ 3.11 |
| numpy | ≥ 1.24 |
| scipy | ≥ 1.10 |
| matplotlib | ≥ 3.7 |
| pandas | ≥ 2.0 |
| rich | ≥ 13.0 |
| PyQt6 | ≥ 6.4 *(GUI only)* |

---

## Quickstart notebook

A runnable Jupyter notebook walks through the full workflow — copula PDF /
sampling / fitting, Sklar bivariate law, then PMC simulate → classify →
ICE — in less than a minute end-to-end:

```bash
pip install jupyter
jupyter notebook examples/quickstart.ipynb
```

The notebook is also re-run as a regression test (`pytest -m slow`).

---

## Copula toolkit

### Available families

All copulas are **τ_K-parameterised** (Kendall's τ as the single free dependence parameter). The Student-t copula additionally exposes a free `df` parameter fitted jointly by 2-D MLE.

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
from prg.copulas import CopulaGaussian, CopulaStudent, CopulaClayton

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

```python
import numpy as np
from prg.copulas import CopulaGaussian, CopulaGH

data = np.random.default_rng(0).normal(size=(500, 2))

# Single-family fit
result = CopulaGaussian.fit(data, method='tau')   # or method='mle'
print(result.tau_k, result.aic, result.bic)

# 5-fold cross-validation log-likelihood
print(result.cv_loglik(K=5))

# Parametric bootstrap GoF  (Cramér-von Mises)
gof = result.gof_test(B=200)
print(gof.p_value)

# Bootstrap CI on τ_K
ci = result.bootstrap_ci(B=500)
print(ci)

# Automatic family selection
from prg.copulas._base import CopulaVirt
ranked = CopulaVirt.fit_best(data)      # sorted by AIC ascending
best   = ranked[0]
print(best.copula, best.aic)
```

### Bivariate joint laws

```python
from prg.copulas import CopulaGaussian
from prg.copulas.bivariate import BivariateLaw
import scipy.stats as ss

law = BivariateLaw(
    copula=CopulaGaussian(tau_k=0.6),
    left_dist=ss.norm(loc=-1, scale=1),
    right_dist=ss.norm(loc=+1, scale=1),
)
print(law.pdf([-0.5, 0.8]))   # joint density f(x,y)
print(law.cdf([ 0.0, 1.0]))   # joint CDF F(x,y)
samples = law.sample(1000, seed=0)

# IFM two-step fit
fit = BivariateLaw.fit(data, copula_class=CopulaGaussian,
                       left_family='norm', right_family='norm')
print(fit.tau_k, fit.aic)
```

---

## PMC models

### Model variants

Five variants of the **(X, Y) jointly Markovian** pair process are supported, ordered from the simplest to the most general. Under the **SR-PMC reversibility contract** (CSDA 2013, Eq. 14) the marginal density of an observation depends only on its current state — there are exactly **K marginal densities** $f_0, \dots, f_{K-1}$ in every variant, regardless of whether the chain is HMC- or PMC-prior:

| Variant | Prior | Observation density | Copula |
|---------|-------|--------------------|--------|
| **HMC-IN** | Transition matrix A | f_j(y_n) | No |
| **HMC-IN2** | Transition matrix A | f_j(y_n) | No |
| **HMC-DN** | Transition matrix A | f_j(y_n) · c_{ij}(F_i(y_{n-1}), F_j(y_n)) | Yes — c_{ij} |
| **PMC-IN** | Joint distribution p | f_i(y_n) · f_j(y_{n+1}) | No |
| **PMC** | Joint distribution p | f_i(y_n) · f_j(y_{n+1}) · c_{ij}(F_i(y_n), F_j(y_{n+1})) | Yes — c_{ij} |

In CSDA-2013 notation, the right margin of the bivariate joint $f_{i,j}$ is "$f_{ji}$"; under SR-PMC reversibility this equals $f_j$ (depends only on its own state).

### TOML model files

Models are stored as TOML files. The canonical schema for margins is **K-format** (one block per state, indexed by `i` only):

```toml
[model]
name      = "PMC Gaussien K=2"
variant   = "PMC"        # HMC-IN | HMC-IN2 | HMC-DN | PMC-IN | PMC
K         = 2
N_default = 5000

[prior]
# HMC-* variants: key "A" — K×K row-stochastic transition matrix
# PMC-* variants: key "p" — K×K symmetric joint distribution (sums to 1)
p = [[0.45, 0.05],
     [0.05, 0.45]]

# K margin blocks (one per state) — SR-PMC contract.
[[margins]]
i      = 0
dist   = "norm"          # any scipy.stats distribution name
params = {loc = -1.0, scale = 1.0}

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

[ice]                    # optional — ICE estimator defaults
fit_margins = false
max_iter    = 50
tol         = 1e-4
candidates  = ["Gauss", "Clayton", "GH", "Frank", "Joe"]
```

> **Legacy K²-format**. Older TOMLs declared K² margin blocks indexed by `(i, j)`. They are still loaded for back-compat: tied entries collapse silently to K state-margins (with an `INFO` log noting the legacy format), and untied entries trigger a `WARNING` listing each conflicting `(i, j)` — the `(i, 0)` anchor is kept as the canonical density. New files should use the K-format above.

Five example models are provided in `prg/pmc/models/`:

| File | Variant |
|------|---------|
| `hmc_in_gauss_k2.toml` | HMC-IN (classical HMM) |
| `hmc_in2_gauss_k2.toml` | HMC-IN2 |
| `hmc_dn_gauss_k2.toml` | HMC-DN (copula-dependent HMM) |
| `pmc_in_gauss_k2.toml` | PMC-IN |
| `pmc_gauss_k2.toml` | PMC (full Pairwise Markov Chain) |

### Python API

```python
from prg.pmc import PMCModel, simulate, classify, ice

# ── Load model ────────────────────────────────────────────────────
mdl = PMCModel("prg/pmc/models/pmc_gauss_k2.toml")
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
from prg.pmc import error_rate
print(f"Error rate: {error_rate(X, X_hat):.3f}")

# ── Unsupervised estimation (ICE) ─────────────────────────────────
import copy
raw_init = copy.deepcopy(mdl._raw)
# … perturb parameters …
init_mdl = PMCModel.from_dict(raw_init)
fitted, log_lik_history = ice(init_mdl, Y,
    ice_cfg={"max_iter": 30, "candidates": ["Gauss", "Clayton", "GH"]})
fitted.save("fitted_pmc.toml")

# ── Edit and save ─────────────────────────────────────────────────
mdl.save("my_model.toml")
```

### Command-line interface

```
python -m prg.pmc COMMAND [options]
```

| Command | Description |
|---------|-------------|
| `simulate` | Generate a synthetic (X, Y) sequence |
| `classify` | Supervised MPM classification |
| `estimate` | Unsupervised ICE parameter estimation |
| `gui` | Launch the PyQt6 graphical interface |

**Examples:**

```bash
# Simulate 10 000 samples and save to CSV
python -m prg.pmc simulate \
    --model prg/pmc/models/pmc_gauss_k2.toml \
    --N 10000 --seed 42 --out sim.csv

# Classify (reference labels in column X → prints error rate)
python -m prg.pmc classify \
    --model prg/pmc/models/pmc_gauss_k2.toml \
    --data sim.csv --out cls.csv --ref X

# Unsupervised ICE estimation
python -m prg.pmc estimate \
    --model prg/pmc/models/pmc_gauss_k2.toml \
    --data sim.csv \
    --candidates "Gauss,Clayton,GH" \
    --max-iter 50 \
    --out fitted.toml

# Launch GUI (with optional startup model)
python -m prg.pmc gui prg/pmc/models/pmc_gauss_k2.toml

# Verbose mode (DEBUG to console)
python -m prg.pmc --verbose simulate --model … --N 500
```

### Graphical interface (PyQt6)

```bash
python -m prg.pmc gui [MODEL.toml]
```

The window is divided into a **model editor** (left) and a **result viewer** (right):

```
┌─────────────────────────────────────┬────────────────────────────────┐
│  Model info  (name, variant, K, N)  │  Matplotlib result panel       │
│  ─────────────────────────────────  │  • Simulate: scatter, histo-   │
│  Tabs                               │    gram, successive-pair plot  │
│    Prior  — editable K×K matrix     │  • Classify: MPM labels,       │
│    Margins — K×K grid (dbl-click)   │    posterior curves, error map │
│    Copulas — K×K grid (dbl-click)   │  • Estimate: ICE log-lik curve │
│    ICE Config — candidates, iter    │                                │
│  ─────────────────────────────────  │  Log panel (INFO+ records)     │
│  [▶ Simulate] [◈ Classify]          │                                │
│         [⟳ Estimate]                │                                │
└─────────────────────────────────────┴────────────────────────────────┘
```

- **File menu**: Open / Save / Save As TOML model; Load data CSV
- **Double-click** any Margins or Copulas cell to edit the distribution or copula parameters
- Long-running operations (simulate, classify, estimate) run in a background thread — the GUI stays responsive

### Logging

All `prg.*` loggers route to two sinks:

| Sink | Level | Location |
|------|-------|----------|
| **File** | DEBUG (full tracebacks) | `~/.copulasformm/pmc.log` |
| **Console** | WARNING (or DEBUG with `--verbose`) | stderr |
| **GUI log panel** | INFO+ | bottom-right panel in the GUI |

The file handler captures:
- Every `logger.exception(...)` call with the full Python traceback
- `logger.debug(...)` from the forward-backward weight computation, ICE iterations, and copula fallbacks
- `logger.warning(...)` for numerical near-failures (zero normalisation constants, incompatible observations)

To configure programmatically:

```python
import logging
from prg.pmc.logging_setup import configure, add_widget_handler

# Point the file to a custom path; set console to DEBUG
configure(level=logging.DEBUG, log_file="run.log")

# (After QApplication exists) wire to a QTextEdit
add_widget_handler(my_widget, level=logging.INFO)
```

---

## Folder structure

```text
copulasformm/
├── prg/
│   ├── copulas/                    Copula toolkit
│   │   ├── _base.py                CopulaEnum, CopulaVirt base class, FitResult
│   │   ├── bivariate.py            BivariateLaw, ConditionalLaw, BivariateFitResult
│   │   ├── archimedean/
│   │   │   ├── a12.py  a14.py  amh.py  bb1.py
│   │   │   ├── clayton.py  frank.py  gumbel.py  joe.py
│   │   │   └── survival.py         SurvivalClayton / SGH / SJoe wrappers
│   │   ├── elliptical/
│   │   │   ├── gaussian.py
│   │   │   └── student.py          2-param MLE (ρ, ν)
│   │   └── explicit/
│   │       ├── cubic_section.py  fgm.py  plackett.py  product.py
│   ├── pmc/                        Pairwise Markov Chain models
│   │   ├── model.py                PMCModel — TOML load/save/validate
│   │   ├── simulate.py             Sequence generator (all 5 variants)
│   │   ├── inference.py            Forward-backward, MPM, smooth
│   │   ├── ice.py                  ICE unsupervised estimator
│   │   ├── logging_setup.py        File + console + QTextEdit handlers
│   │   ├── cli.py  __main__.py     CLI entry point
│   │   ├── gui/
│   │   │   ├── app.py              QApplication entry point
│   │   │   └── main_window.py      QMainWindow — editor + result viewer
│   │   └── models/                 Example TOML files (K=2, Gaussian)
│   │       ├── hmc_in_gauss_k2.toml
│   │       ├── hmc_in2_gauss_k2.toml
│   │       ├── hmc_dn_gauss_k2.toml
│   │       ├── pmc_in_gauss_k2.toml
│   │       └── pmc_gauss_k2.toml
│   ├── settings/
│   │   └── plot_settings.py        Matplotlib rcParams defaults
│   ├── tests/
│   │   ├── test_copulas.py
│   │   ├── test_bivariate.py
│   │   ├── test_conditional.py
│   │   ├── test_scientific.py      PDF/CDF consistency + Kendall τ checks
│   │   └── test_smoke.py           Version check
│   └── tools/
│       └── tools.py                EPS constants, minmaxEPS helper
├── docs/
│   └── CSDA_2013.pdf               Reference paper (Piecini & Derrode)
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## References

- Derrode S., Piecini G. — *Unsupervised classification of radar signals using pairwise Markov chains with copulas*, Computational Statistics & Data Analysis (CSDA), 2013.

The PMC model (SR-PMC variant) is defined by the equations:

- **Eq. 13**: p(X_{n+1} = j | X_n = i, Y_n = y) ∝ p(i, j) · f_i(y)
- **Eq. 14**: p(Y_{n+1} | X_n=i, X_{n+1}=j, Y_n=y) = f_j(y_{n+1}) · c_{ij}(F_i(y_n), F_j(y_{n+1}))

The "reversed-index" notation $f_{ji}$ from CSDA 2013 is the right-margin of the bivariate joint $f_{i,j}$; the **stationarity and reversibility** (SR) constraint on the pair process $(X_n, Y_n)$ collapses it to the per-state density $f_j$. There are therefore exactly K marginal densities in the model, one per state — see *PMC Models — variants and TOML schema* above.
