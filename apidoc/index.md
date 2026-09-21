# awesomePMC

`awesomePMC` (distribution `awesomepmc` on PyPI, import package `pmcprg`)
implements **pairwise Markov chains** (PMC) — and the **hidden Markov
chains** they contain — with the dependence between consecutive
observations modelled by bivariate **copulas**: simulation, supervised
classification, unsupervised estimation (ICE, SEM, GICE) with automatic
copula and margin selection, missing-data imputation and forecasting, and a
PyQt6 GUI.

This site is the **API reference**: generated from the package's docstrings,
built locally, not deployed (see [Installation](installation.md)). It does
not repeat the project's narrative documentation — for the feature tour,
the model TOML format, the CLI, the GUI, worked examples and the scientific
references, see the
[README](https://github.com/SDerrode/awesomePMC/blob/main/README.md) and,
for how to contribute,
[CONTRIBUTING.md](https://github.com/SDerrode/awesomePMC/blob/main/CONTRIBUTING.md).

## Reference implementation of

> S. Derrode, W. Pieczynski. **Unsupervised data classification using
> pairwise Markov chains with automatic copulas selection.**
> *Computational Statistics & Data Analysis* 63 (2013), 81-98.
> [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
>
> S. Derrode, W. Pieczynski. **Unsupervised classification using hidden
> Markov chain with unknown noise copulas and margins.**
> *Signal Processing* 128 (2016), 8-17.
> [doi:10.1016/j.sigpro.2016.03.008](https://doi.org/10.1016/j.sigpro.2016.03.008)

## Layout of this site

- **[Installation](installation.md)** — installing the package and its
  optional extras, and building this site locally.
- **API reference** — one page per sub-package:
  [`pmcprg.copulas`](api/copulas.md) (the copula registry and the shared
  `CopulaVirt`/`BivariateLaw` API, fitting and diagnostics helpers),
  [`pmcprg.pmc`](api/pmc.md) (the PMC/HMC model, inference, missing-data
  handling and the ICE/SEM estimators), [`pmcprg.missing`](api/missing.md)
  (missingness patterns and evaluation metrics) and
  [`pmcprg.diagnostics`](api/diagnostics.md) (goodness-of-fit tests).
- **[State labelling](state-labelling.md)** — how the package numbers and
  identifies the hidden states of a fitted model, and what a user should
  expect across estimation runs.
- **[Changelog](changelog.md)** — the project's `CHANGELOG.md`.
