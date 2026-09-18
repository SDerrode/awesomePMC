"""
pmcprg.diagnostics — goodness-of-fit and inference diagnostics.

This sub-package collects standalone diagnostic tools that complement the
estimation routines in :mod:`pmcprg.pmc`. They are intentionally independent
of any specific model class: each utility takes raw NumPy arrays (and
optionally a CDF callable) and returns a typed result object.

Public API
----------
mks_1samp, mks_2samp, mks_test : multivariate Kolmogorov-Smirnov tests
    (Naaman 2021 finite-sample extension).
MKSResult                       : typed result of an MKS test.
parametric_bootstrap            : model-level calibration of any post-fit
    statistic. A tabulated critical value is wrong on a fitted model — the
    MKS one rejects a correct PMC 10% of the time at a nominal 5% — because
    the reference is estimated, the observations are dependent, and in a
    latent-state model the per-state sample is assigned rather than given.
    Resampling whole series from the fitted model absorbs all three.
BootstrapResult                 : typed result of that calibration.
neyman_components, neyman_test  : smooth-test components of a fitted margin.
    The transform projected on Legendre polynomials — one component per kind
    of departure (location, dispersion, asymmetry, tails). Measured to reject
    a wrong margin 3x more often than the KS test, and to say *which* way it
    is wrong. `neyman_test` pays the multiplicity: taking the smallest of
    four p-values uncorrected raises the level from 0.05 to 0.175.
NeymanResult                    : typed result of a smooth-test call.
vuong_test, clarke_test,         : is a copula family choice significant?
comparison_matrix, confidence_set,  Weighted Vuong (1989) and Clarke (2007)
ice_pair_comparisons              tests with a HAC variance, all pairs of a
    candidate set, the families not significantly worse than the best (ties
    declared), and the same for every pair of states after an ICE run
    (audit FR-6). Opt-in; see ``pmcprg.diagnostics.model_selection`` for the
    caveats (estimated pseudo-observations, Chen & Fan 2006).
RadialSymmetryResult,           : screening test for radial symmetry (Genest &
radial_symmetry_statistic,        Nešlehová 2014, audit FR-10) — rejecting it
radial_symmetry_test              at once excludes Gauss, Student, Frank, FGM
    and Plackett as candidate families before the per-family fits. A
    Cramér-von Mises statistic on the empirical copula and its radial
    reflection, calibrated by a parametric (Gaussian-surrogate) bootstrap.
    Unweighted only — see the module docstring on why ξ-weighting has no
    defensible form for this statistic.
RosenblattGoFResult,            : goodness-of-fit test for a candidate copula
rosenblatt_transform,             family via its Rosenblatt transform
rosenblatt_statistic,             (Rosenblatt 1952; Genest, Remillard &
rosenblatt_gof_test               Beaudoin 2009, audit FR-10). Reduces "does
    (x, y) follow C_theta?" to "is (u, h(v|u)) independent-uniform on the
    square?", tested by a Cramer-von Mises statistic against Pi(u,v) = u*v,
    calibrated by a parametric bootstrap that refits theta on every
    replicate. Unweighted only, same reason as radial_symmetry_test.
ExchangeabilityResult,          : screening test for exchangeability (Genest,
exchangeability_statistic,        Neslehova & Quessy 2012, audit FR-10,
exchangeability_test              fourth item) — C(u,v) = C(v,u)? A
    Cramer-von Mises statistic on the empirical copula and its transpose,
    calibrated by a parametric (Gaussian-surrogate) bootstrap, the same
    skeleton as radial_symmetry_test with the reflected reference swapped
    for the transposed one. Unweighted only, same reason as
    radial_symmetry_test.
margin_cdfs, copula_pseudo_obs, : pseudo-observations and margin samples of a
margin_pit_dual, margin_keys,     fitted chain, for state margins f_i and for
margin_of, margin_sample          the pair margins f_ij of a general PMC
    (DerrodePieczynski_CSDA2013 Eqs. 12-14): c_ij is tested on (F_ij(y_n),
    F_ji(y_{n+1})) weighted by xi_n(i, j); f_ij on F_ij(y_n) weighted by
    xi_n(i, j) and on F_ij(y_{n+1}) weighted by xi_n(j, i). These need a
    model exposing K, margin_structure and margin(i, j) — duck-typed, the
    model class is not imported.
draw_multipliers, auto_block_length, : serially dependent multipliers for the
multiplier_weights,                    bootstrap of a copula functional on a
multiplier_autocorrelation             Markov chain (audit FR-5). Consecutive
    pairs share an observation, so an i.i.d. multiplier bootstrap is
    calibrated against too narrow a null; smoothing the multipliers with a
    kernel of dependence length l restores the long-run variance. Reached
    through bootstrap="dependent-multiplier" on radial_symmetry_test,
    exchangeability_test and rosenblatt_gof_test; see
    pmcprg.diagnostics.dependent_multiplier for the construction.
"""

from pmcprg.diagnostics.bootstrap import (
    BootstrapResult,
    parametric_bootstrap,
)
from pmcprg.diagnostics.dependent_multiplier import (
    MULTIPLIER_KERNELS,
    MULTIPLIER_LAWS,
    auto_block_length,
    draw_multipliers,
    multiplier_autocorrelation,
    multiplier_weights,
)
from pmcprg.diagnostics.exchangeability import (
    ExchangeabilityResult,
    exchangeability_statistic,
    exchangeability_test,
)
from pmcprg.diagnostics.model_selection import (
    ComparisonMatrix,
    ComparisonResult,
    ConfidenceSet,
    PairComparison,
    clarke_test,
    comparison_matrix,
    confidence_set,
    ice_pair_comparisons,
    vuong_test,
)
from pmcprg.diagnostics.mks import (
    MKSResult,
    mks_1samp,
    mks_2samp,
    mks_test,
)
from pmcprg.diagnostics.pseudo import (
    copula_pseudo_obs,
    margin_cdfs,
    margin_keys,
    margin_of,
    margin_pit_dual,
    margin_sample,
)
from pmcprg.diagnostics.radial_symmetry import (
    RadialSymmetryResult,
    radial_symmetry_statistic,
    radial_symmetry_test,
)
from pmcprg.diagnostics.rosenblatt import (
    RosenblattGoFResult,
    rosenblatt_gof_test,
    rosenblatt_statistic,
    rosenblatt_transform,
)
from pmcprg.diagnostics.smooth import (
    COMPONENT_NAMES,
    NeymanResult,
    neyman_components,
    neyman_test,
)

__all__ = [
    "COMPONENT_NAMES",
    "BootstrapResult",
    "ComparisonMatrix",
    "ComparisonResult",
    "ConfidenceSet",
    "PairComparison",
    "clarke_test",
    "comparison_matrix",
    "confidence_set",
    "ice_pair_comparisons",
    "vuong_test",
    "MKSResult",
    "NeymanResult",
    "copula_pseudo_obs",
    "margin_cdfs",
    "margin_keys",
    "margin_of",
    "margin_pit_dual",
    "margin_sample",
    "neyman_components",
    "neyman_test",
    "parametric_bootstrap",
    "mks_1samp",
    "mks_2samp",
    "mks_test",
    "RadialSymmetryResult",
    "radial_symmetry_statistic",
    "radial_symmetry_test",
    "RosenblattGoFResult",
    "rosenblatt_gof_test",
    "rosenblatt_statistic",
    "rosenblatt_transform",
    "ExchangeabilityResult",
    "exchangeability_statistic",
    "exchangeability_test",
    "MULTIPLIER_KERNELS",
    "MULTIPLIER_LAWS",
    "auto_block_length",
    "draw_multipliers",
    "multiplier_autocorrelation",
    "multiplier_weights",
]
