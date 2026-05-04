"""
prg.pmc — Pairwise Markov Chain models with copula-based transitions.

Public API
----------
PMCModel  : load/save/validate a PMC/HMC model from a TOML file.
Variant   : enum of the 5 model variants (HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC).
simulate  : generate a synthetic (X, Y) sequence from a PMCModel.
classify  : supervised MPM classification of an observation sequence.
ice       : unsupervised ICE parameter estimation.
"""

from prg.pmc.model         import PMCModel, Variant
from prg.pmc.simulate      import simulate
from prg.pmc.pmm           import simulate_pmm, classify_pmm
from prg.pmc.inference     import classify, forward, backward, smooth, mpm, error_rate
from prg.pmc.ice           import ice
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
    "classify_pmm",
    "forward",
    "backward",
    "smooth",
    "mpm",
    "error_rate",
    # unsupervised estimation
    "ice",
    # logging
    "configure_logging",
]
