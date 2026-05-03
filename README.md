# copulasformm

This repository contains programs for **copula-based Markov models** — a framework for modeling temporal dependencies in Markov chains using copulas, enabling flexible and non-Gaussian joint distributions between consecutive states.

---

## Table of Contents

- [copulasformm](#copulasformm)
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Overview](#overview)
    - [Folders structure](#folders-structure)

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

### Requirements

- Python >= 3.10
- numpy, scipy, matplotlib, pandas, rich

---

## Overview

Markov models classically assume a parametric (often Gaussian) transition kernel between states. This project proposes to replace this kernel by a **copula**, decoupling the marginal distributions of consecutive states from their dependence structure. The approach applies to:

- hidden Markov models (HMM)
- Markov chain Monte Carlo (MCMC) sampling
- state-space models with non-Gaussian transitions

The repository currently provides (v0.3.0):

- **Copula families** — Product, Gaussian, Student, Gumbel-Hougaard, Clayton, Frank (disabled), Archimedean A12 / A14, FGM, Cubic Section. All τ_K-parameterised; analytical PDF / CDF / h-function where possible.
- **Bivariate joint laws** — `BivariateLaw` (Sklar's theorem on a copula + two marginals) and `ConditionalLaw` (frozen p(Y|X=x)).
- **Sampling** — Rosenblatt h-inversion (Brent) on every copula; conditional sampling on either dimension via majorant rejection.
- **Parameter estimation** — `fit(data, method='tau'|'mle')` returning a `FitResult` with AIC / BIC / AICc / HQC, K-fold CV log-likelihood, parametric-bootstrap goodness-of-fit (Cramér-von Mises), and bootstrap CI on τ_K. `BivariateLaw.fit(...)` does the same in two-step IFM mode (margin MLE then copula).
- **Model selection** — `CopulaVirt.fit_best(data, families=...)` and `BivariateLaw.fit_best(...)` rank candidates by AIC.
- **Tail dependence** — `tail_dependence() → (λ_L, λ_U)` with analytical formulas for the families that admit them, numerical default otherwise.
- **Visual diagnostics** — `plot_diagnostics(plot_dir)` on both result classes (6-panel: PDF + scatter, PP plot, empirical/fitted copula, residuals, λ_L / λ_U vs empirical estimators).

Planned (not yet implemented):
- Vectorised pdf for n ≥ 10 000
- Vine copulas (C-vine, D-vine) for higher-dimensional dependence
- Filtering, smoothing, and prediction under copula-based transition models

---

## Folders structure

<!-- PROJECT_STRUCTURE_START -->
```text
./
|-- data/
|   |-- datafile/
|   |-- historyTracker/
|   |-- plot/
|   `-- clean_dirs.sh
|-- prg/
|   |-- copulas/
|   |   |-- archimedean/
|   |   |   |-- __init__.py
|   |   |   |-- a12.py
|   |   |   |-- a14.py
|   |   |   |-- clayton.py
|   |   |   |-- frank.py
|   |   |   `-- gumbel.py
|   |   |-- elliptical/
|   |   |   |-- __init__.py
|   |   |   |-- gaussian.py
|   |   |   `-- student.py
|   |   |-- explicit/
|   |   |   |-- __init__.py
|   |   |   |-- cubic_section.py
|   |   |   |-- fgm.py
|   |   |   `-- product.py
|   |   |-- __init__.py
|   |   |-- _base.py
|   |   `-- bivariate.py
|   |-- settings/
|   |   |-- __init__.py
|   |   `-- plot_settings.py
|   |-- tests/
|   |   |-- __init__.py
|   |   |-- conftest.py
|   |   `-- test_smoke.py
|   |-- tools/
|   |   |-- __init__.py
|   |   `-- tools.py
|   `-- __init__.py
|-- .gitignore
|-- .gitlab-ci.yml
|-- CHANGELOG.md
|-- LICENSE
|-- README.md
|-- pyproject.toml
|-- requirements.txt
`-- update_readme_structure.sh

```
<!-- PROJECT_STRUCTURE_END -->
