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

The repository will provide:

- **Copula families** — Gaussian, Student, Clayton, Gumbel, Frank, Joe, and vine copulas
- **Parameter estimation** — maximum likelihood and Bayesian inference
- **Simulation tools** — generation of copula-driven Markov chains
- **Applications** — filtering, smoothing, and prediction under copula-based transition models

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
|   |-- tests/
|   |   |-- __init__.py
|   |   |-- conftest.py
|   |   `-- test_smoke.py
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
