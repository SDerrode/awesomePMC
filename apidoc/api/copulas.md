# pmcprg.copulas

41 bivariate copula families (elliptical, Archimedean, survival,
extreme-value, explicit, and the 90°/270° rotations for negative
dependence) behind a common `CopulaVirt` interface (`pdf`, `cdf`,
`conditional_cdf`, `inv_h`, `sample`, `fit`, plus vectorised `pdf_array` /
`logpdf_array`), a bivariate-law wrapper that combines two margins with a
copula, and standard-error / fit-diagnostic helpers.

Individual family classes (`CopulaGaussian`, `CopulaClayton`, …) are not
listed one by one here — go through the registry (`CopulaEnum`) to look one
up, or the "Available families" tables of the
[README](https://github.com/SDerrode/awesomePMC/blob/main/README.md#available-families).

## Registry

::: pmcprg.copulas.CopulaEnum
    options:
      members:
        - available

## Base classes

`CopulaVirt.fit` and `.fit_best` (`pmcprg/copulas/_base.py`) are omitted
from the rendered members below — their docstrings are mid-edit on another
branch (a public weighted `fit()`, audit FR-12) and currently trip
mkdocstrings' strict-mode docstring parser; see the narrative "Fitting and
model selection" section of the
[README](https://github.com/SDerrode/awesomePMC/blob/main/README.md#fitting-and-model-selection)
in the meantime.

::: pmcprg.copulas.CopulaVirt
    options:
      members:
        - reachable_tau_bounds
        - constrain_params
        - constructible_params
        - pdf
        - cdf
        - pdf_array
        - logpdf_array
        - cdf_array
        - conditional_cdf
        - transposed
        - inv_h
        - inv_h_array
        - tail_dependence
        - tau_range
        - standard_errors
        - sample

::: pmcprg.copulas.FitResult

::: pmcprg.copulas.GoFResult

## Bivariate joint laws

Combine two margins (state or pair densities) with a copula into the joint
law `f_ij(y_1, y_2) = f_ij(y_1) f_ji(y_2) c_ij(F_ij(y_1), F_ji(y_2))`
(DerrodePieczynski_CSDA2013 Eq. 12).

::: pmcprg.copulas.BivariateLaw

::: pmcprg.copulas.ConditionalLaw

::: pmcprg.copulas.BivariateFitResult

::: pmcprg.copulas.BivariateBootstrapCI

## Fitting diagnostics

Standard errors, and the independence / sub-model likelihood-ratio tests
(audit FR-4).

::: pmcprg.copulas.standard_errors

::: pmcprg.copulas.StandardErrors

::: pmcprg.copulas.independence_lr_test

::: pmcprg.copulas.IndependenceLRTest

::: pmcprg.copulas.submodel_lr_test

::: pmcprg.copulas.SubmodelLRTest

::: pmcprg.copulas.mle_tau_discrepancy_test

::: pmcprg.copulas.MleTauDiscrepancyTest

## Robust fitting (density power divergence)

Weighted density-power-divergence estimation (audit FR-7), an alternative
to MLE that trades a small efficiency loss on clean data for a bounded
influence function under contamination.

::: pmcprg.copulas.dpd_fit

::: pmcprg.copulas.DPDFit

::: pmcprg.copulas.select_alpha

::: pmcprg.copulas.DPDAlphaSelection

::: pmcprg.copulas.DPD_ALPHAS

## Nonparametric comparison tool

Not a `CopulaEnum` family (no `fit()`, not selectable by ICE) — a
nonparametric baseline to compare a fitted parametric family against.

::: pmcprg.copulas.EmpiricalBetaCopula
