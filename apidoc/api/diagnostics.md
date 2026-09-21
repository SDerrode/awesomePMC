# pmcprg.diagnostics

Goodness-of-fit and inference diagnostics, independent of any specific model
class: each utility takes raw NumPy arrays (and optionally a CDF callable)
and returns a typed result object.

## Multivariate Kolmogorov-Smirnov test

Naaman (2021)'s finite-sample extension.

::: pmcprg.diagnostics.mks_1samp

::: pmcprg.diagnostics.mks_2samp

::: pmcprg.diagnostics.mks_test

::: pmcprg.diagnostics.MKSResult

## Parametric bootstrap

Model-level calibration of any post-fit statistic — resampling whole series
from the fitted model, needed because a tabulated critical value is wrong
on a fitted model (estimated reference, dependent observations, per-state
samples that are assigned rather than given).

::: pmcprg.diagnostics.parametric_bootstrap

::: pmcprg.diagnostics.BootstrapResult

## Smooth tests (Neyman)

Legendre-polynomial components of a fitted margin's goodness of fit — one
component per kind of departure (location, dispersion, asymmetry, tails).

::: pmcprg.diagnostics.neyman_components

::: pmcprg.diagnostics.neyman_test

::: pmcprg.diagnostics.NeymanResult

::: pmcprg.diagnostics.COMPONENT_NAMES

## Copula-family comparison (Vuong / Clarke)

Is a copula family choice significant? Weighted Vuong (1989) and Clarke
(2007) tests with a HAC variance, over all pairs of a candidate set, and
over every pair of states after an ICE run.

::: pmcprg.diagnostics.vuong_test

::: pmcprg.diagnostics.clarke_test

::: pmcprg.diagnostics.comparison_matrix

::: pmcprg.diagnostics.confidence_set

::: pmcprg.diagnostics.ice_pair_comparisons

::: pmcprg.diagnostics.ComparisonMatrix

::: pmcprg.diagnostics.ComparisonResult

::: pmcprg.diagnostics.ConfidenceSet

::: pmcprg.diagnostics.PairComparison

## Radial symmetry and exchangeability screens

Cramér-von Mises statistics on the empirical copula against its radial
reflection / its transpose, calibrated by a parametric (Gaussian-surrogate)
bootstrap. Unweighted only.

::: pmcprg.diagnostics.radial_symmetry_test

::: pmcprg.diagnostics.radial_symmetry_statistic

::: pmcprg.diagnostics.RadialSymmetryResult

::: pmcprg.diagnostics.exchangeability_test

::: pmcprg.diagnostics.exchangeability_statistic

::: pmcprg.diagnostics.ExchangeabilityResult

## Rosenblatt goodness-of-fit

Reduces "does `(x, y)` follow `C_theta`?" to "is `(u, h(v|u))`
independent-uniform on the square?", tested by a Cramér-von Mises statistic
against `Pi(u, v) = u*v`, calibrated by a parametric bootstrap that refits
`theta` on every replicate.

::: pmcprg.diagnostics.rosenblatt_gof_test

::: pmcprg.diagnostics.rosenblatt_transform

::: pmcprg.diagnostics.rosenblatt_statistic

::: pmcprg.diagnostics.RosenblattGoFResult

## Pseudo-observations

Pseudo-observations and margin samples of a fitted chain, for state margins
`f_i` and for the pair margins `f_ij` of a general PMC.

::: pmcprg.diagnostics.copula_pseudo_obs

::: pmcprg.diagnostics.margin_cdfs

::: pmcprg.diagnostics.margin_pit_dual

::: pmcprg.diagnostics.margin_keys

::: pmcprg.diagnostics.margin_of

::: pmcprg.diagnostics.margin_sample

## Dependent-multiplier bootstrap

Serially dependent multipliers for the bootstrap of a copula functional on
a Markov chain (audit FR-5) — an i.i.d. multiplier bootstrap under-covers
because consecutive pairs share an observation.

::: pmcprg.diagnostics.draw_multipliers

::: pmcprg.diagnostics.auto_block_length

::: pmcprg.diagnostics.multiplier_weights

::: pmcprg.diagnostics.multiplier_autocorrelation

::: pmcprg.diagnostics.MULTIPLIER_KERNELS

::: pmcprg.diagnostics.MULTIPLIER_LAWS
