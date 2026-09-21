# pmcprg.missing

Data layer for missing observations, free of any inference code: artificial
missingness patterns (following the geometry of ImputeGAP's `GenGap`
contamination module) and evaluation metrics restricted to the masked
positions. `pmcprg.pmc` (imputation, forecasting, non-ignorable
missingness) is the inference layer that consumes this one.

## Patterns

::: pmcprg.missing.mask_from_nan

::: pmcprg.missing.mcar

::: pmcprg.missing.aligned

::: pmcprg.missing.scattered

::: pmcprg.missing.blackout

::: pmcprg.missing.disjoint

::: pmcprg.missing.overlap

::: pmcprg.missing.gaussian

::: pmcprg.missing.distribution

::: pmcprg.missing.state_dependent

::: pmcprg.missing.state_markov

## Metrics

::: pmcprg.missing.rmse

::: pmcprg.missing.mae

::: pmcprg.missing.mutual_information

::: pmcprg.missing.pearson

::: pmcprg.missing.crps_from_samples

::: pmcprg.missing.crps_gaussian

::: pmcprg.missing.interval_coverage

::: pmcprg.missing.error_rate_split

::: pmcprg.missing.ErrorRates
