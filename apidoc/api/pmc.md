# pmcprg.pmc

Pairwise Markov Chain models (HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC) loaded
from TOML, with simulation, MPM classification, exact handling of missing
observations, and unsupervised estimation (ICE, SEM, GICE). See the
[README](https://github.com/SDerrode/awesomePMC/blob/main/README.md#pmc-models)
for the TOML model format, the CLI and the GUI.

## Model

::: pmcprg.pmc.PMCModel

::: pmcprg.pmc.Variant

## Simulation

::: pmcprg.pmc.simulate

## Inference

Forward-backward, MPM classification, and Forward-Filter Backward-Sample.

::: pmcprg.pmc.classify

::: pmcprg.pmc.classify_image

::: pmcprg.pmc.forward

::: pmcprg.pmc.backward

::: pmcprg.pmc.smooth

::: pmcprg.pmc.joint_posteriors

::: pmcprg.pmc.mpm

::: pmcprg.pmc.sample_posterior

::: pmcprg.pmc.error_rate

## Missing observations: exact inference, imputation, forecasting

Missing values (`NaN`) are integrated out exactly rather than imputed
before inference.

::: pmcprg.pmc.gap_posterior

::: pmcprg.pmc.GapPosterior

::: pmcprg.pmc.impute

::: pmcprg.pmc.Imputation

::: pmcprg.pmc.forecast

::: pmcprg.pmc.Forecast

::: pmcprg.pmc.NodeCheck

## Non-ignorable missingness mechanisms

The `[missingness]` block of a model's TOML file: the missingness mask
itself carries evidence on the hidden states.

::: pmcprg.pmc.StateMissingness

::: pmcprg.pmc.StateMarkovMissingness

::: pmcprg.pmc.missingness_lr_test

::: pmcprg.pmc.MissingnessLRTest

## Erroneous observations: predictive PIT, flags, flag-and-mask estimation

One-step-ahead predictive PIT of every observed row (exact per variant,
missing rows integrated out), outlier flags with innovation gating and
multiplicity corrections, calibration checks, and a flag-and-mask ICE/SEM
that trims the flagged rows through the missing-data machinery. The
formulas, the multiplicity discussion and the design of the robust loop are
in the docstring of `pmcprg/pmc/outliers.py`; the simulation study is in
`report/erroneous_data`.

::: pmcprg.pmc.predictive_pit

::: pmcprg.pmc.PredictivePIT

::: pmcprg.pmc.flag_outliers

::: pmcprg.pmc.OutlierFlags

::: pmcprg.pmc.pit_checks

::: pmcprg.pmc.PitChecks

::: pmcprg.pmc.robust_estimate

::: pmcprg.pmc.RobustFit

## ICE / SEM unsupervised estimation

Both estimators share the same M-step; ICE uses the soft forward-backward
posteriors and converges deterministically, SEM draws one
`sample_posterior` realisation per iteration. See
["ICE vs SEM"](https://github.com/SDerrode/awesomePMC/blob/main/README.md#pmc-models)
in the README, and [State labelling](../state-labelling.md) for how the
hidden-state indices are numbered and identified across a fit.

::: pmcprg.pmc.ice

::: pmcprg.pmc.ice_image

::: pmcprg.pmc.IceResult

::: pmcprg.pmc.IceTrace

::: pmcprg.pmc.sem

::: pmcprg.pmc.sem_image

::: pmcprg.pmc.SemResult

::: pmcprg.pmc.SemTrace
