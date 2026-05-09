"""
prg.pmc — Pairwise Markov Chain models with copula-based transitions.

This package implements the unsupervised classification methods of two
papers by **S. Derrode and W. Pieczynski** (PDFs in ``docs/``):

* **A16** — *Unsupervised data classification using pairwise Markov chains
  with automatic copulas selection*, Comput. Stat. Data Anal., 63 (2013),
  81-98. doi:10.1016/j.csda.2013.01.027
  → PMC model family, SR-PMC reversibility, ICE-based copula selection
  (criteria: MLE, AIC, BIC, Huard Eq. 20, CvM).

* **A23** — *Unsupervised classification using hidden Markov chain with
  unknown noise copulas and margins*, Signal Process., 128 (2016), 8-17.
  doi:10.1016/j.sigpro.2016.03.008
  → GICE: automatic margin family selection from a candidate set
  (criteria: MLE, Kolmogorov Ex. 3.1, AIC, BIC).

Please cite both papers if you use this code in published work — see
``prg/pmc/README.md`` for BibTeX entries and a feature ↔ paper map.

Public API
----------
PMCModel  : load/save/validate a PMC/HMC model from a TOML file.
Variant   : enum of the 5 model variants (HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC).
simulate  : generate a synthetic (X, Y) sequence from a PMCModel.
classify  : supervised MPM classification of an observation sequence.
ice       : unsupervised ICE parameter estimation (A16 §4 + A23 §3 GICE).
"""

from prg.pmc.model         import PMCModel, Variant
from prg.pmc.simulate      import simulate
from prg.pmc.pmm           import simulate_pmm, classify_pmm
from prg.pmc.inference     import classify, classify_image, forward, backward, smooth, mpm, error_rate
from prg.pmc.ice           import ice, ice_image
from prg.pmc.peano         import (
    peano_path,
    image_to_signal,
    signal_to_image,
    load_grayscale,
    load_color,
    load_labels,
    save_segmentation,
)
from prg.pmc.logging_setup import configure as configure_logging

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
    "mpm",
    "error_rate",
    # unsupervised estimation
    "ice",
    "ice_image",
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
