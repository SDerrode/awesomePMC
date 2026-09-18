"""
pmcprg.pmc — Pairwise Markov Chain models with copula-based transitions.

This package implements the unsupervised classification methods of two
papers by **S. Derrode and W. Pieczynski** (see the DOIs below):

* **DerrodePieczynski_CSDA2013** — *Unsupervised data classification using
  pairwise Markov chains with automatic copulas selection*, Comput. Stat.
  Data Anal., 63 (2013), 81-98. doi:10.1016/j.csda.2013.01.027
  → PMC model family, SR-PMC reversibility, ICE-based copula selection
  (criteria: MLE, AIC, BIC, Huard Eq. 20, CvM).

* **DerrodePieczynski_SP2016** — *Unsupervised classification using hidden
  Markov chain with unknown noise copulas and margins*, Signal Process.,
  128 (2016), 8-17. doi:10.1016/j.sigpro.2016.03.008
  → GICE: automatic margin family selection from a candidate set
  (criteria: MLE, Kolmogorov Ex. 3.1, AIC, BIC).

Please cite both papers if you use this code in published work — see
``pmcprg/pmc/README.md`` for BibTeX entries and a feature ↔ paper map.

Public API
----------
PMCModel  : load/save/validate a PMC/HMC model from a TOML file.
Variant   : enum of the 5 model variants (HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC).
simulate  : generate a synthetic (X, Y) sequence from a PMCModel.
classify  : supervised MPM classification of an observation sequence.
sample_posterior : Forward-Filter Backward-Sample draw  X̃ ~ P(X | Y).
impute    : posterior law of missing observations (NaN) given the observed ones.
forecast  : h-step predictive law of (x_{N+k}, y_{N+k}) given Y.
gap_posterior : α̂, β̂, γ, ξ and log-likelihood of a sequence with missing values.
ice       : unsupervised ICE parameter estimation (DerrodePieczynski_CSDA2013 §4 +
            DerrodePieczynski_SP2016 §3 GICE).
sem       : unsupervised Stochastic-EM parameter estimation (sister of ICE).
"""

from pmcprg.pmc.model         import PMCModel, Variant
from pmcprg.pmc.simulate      import simulate
from pmcprg.pmc.pmm           import simulate_pmm, classify_pmm
from pmcprg.pmc.inference     import (
    classify, classify_image, forward, backward, smooth, joint_posteriors,
    mpm, error_rate, sample_posterior,
)
from pmcprg.pmc.gaps          import (
    Forecast, GapPosterior, Imputation, forecast, gap_posterior, impute,
)
from pmcprg.pmc._estim_common import ice_estim_defaults, sem_estim_defaults
from pmcprg.pmc.ice           import (
    EXTRA_PARAM_BOUNDS,
    GICE_KNOWN_FAMILIES,
    SP2016_DEFAULT_CANDIDATES,
    IceResult,
    IceTrace,
    ice,
    ice_image,
)
from pmcprg.pmc.sem           import SemResult, SemTrace, sem, sem_image
from pmcprg.pmc.peano         import (
    peano_path,
    image_to_signal,
    signal_to_image,
    load_grayscale,
    load_color,
    load_labels,
    save_segmentation,
)
from pmcprg.pmc.logging_setup import configure as configure_logging

__all__ = [
    # model
    "PMCModel",
    "Variant",
    # simulation
    "simulate",
    "simulate_pmm",
    # inference
    "classify",
    "classify_image",
    "classify_pmm",
    "forward",
    "backward",
    "smooth",
    "joint_posteriors",
    "mpm",
    "error_rate",
    "sample_posterior",
    # missing observations (NaN): exact inference, imputation, forecasting
    "impute",
    "forecast",
    "gap_posterior",
    "Imputation",
    "Forecast",
    "GapPosterior",
    # unsupervised estimation
    "ice",
    "ice_image",
    "ice_estim_defaults",
    "sem",
    "sem_image",
    "sem_estim_defaults",
    # estimation constants (single import surface for the GUI — audit A-1)
    "EXTRA_PARAM_BOUNDS",
    "GICE_KNOWN_FAMILIES",
    "SP2016_DEFAULT_CANDIDATES",
    # estimation result/trace types
    "IceTrace",
    "IceResult",
    "SemTrace",
    "SemResult",
    # image ↔ signal transforms (Generalized Hilbert / "gilbert")
    "peano_path",
    "image_to_signal",
    "signal_to_image",
    "load_grayscale",
    "load_color",
    "load_labels",
    "save_segmentation",
    # logging
    "configure_logging",
]
