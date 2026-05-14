"""
copulasformm — Copula-based Markov models for non-Gaussian transition kernels.

The package provides two complementary layers:

* :mod:`prg.copulas` — 17 bivariate copula families (Gaussian, Student,
  Archimedean, survival, BB1, …) with a common ``CopulaVirt`` API:
  ``pdf``, ``cdf``, ``conditional_cdf``, ``inv_h``, ``sample``, ``fit``,
  plus vectorised ``pdf_array`` / ``logpdf_array``.

* :mod:`prg.pmc` — Pairwise Markov Chain models (HMC-IN, HMC-IN2, HMC-DN,
  PMC-IN, PMC) loaded from TOML, with simulation, MPM classification,
  ICE unsupervised estimation, a CLI (``pmc``), and a PyQt6 GUI.

Quickstart — copula
-------------------
>>> from prg.copulas import CopulaGaussian
>>> cop = CopulaGaussian(tau_k=0.6)
>>> cop.pdf([0.3, 0.7])                # density at one point
>>> samples = cop.sample(n=1000, seed=0)
>>> result  = CopulaGaussian.fit(samples, method='mle')
>>> result.aic, result.bootstrap_ci(B=200)

Quickstart — PMC
----------------
>>> from prg.pmc import PMCModel, simulate, classify, ice
>>> mdl   = PMCModel("prg/pmc/models/pmc_gauss_k2.toml")
>>> X, Y  = simulate(mdl, N=2000, seed=0)
>>> X_hat, gamma, log_lik = classify(mdl, Y)
>>> fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 20})  # trace.log_liks, trace.tau_history, …

Quickstart — CLI
----------------
After ``pip install -e .``:

    pmc simulate -m model.toml --N 5000 --out sim.csv
    pmc classify -m model.toml -d sim.csv  --out cls.csv
    pmc estimate -m init.toml  -d sim.csv  --out fitted.toml
    pmc gui      [model.toml]      # PyQt6 GUI (extra: pip install copulasformm[gui])
"""

__version__ = "0.8.0"


def configure_logging(*args, **kwargs):
    """Configure the package-wide ``prg`` logger.

    Thin wrapper around :func:`prg.pmc.logging_setup.configure`. Imported lazily
    so that ``import prg`` stays free of optional dependencies.
    """
    from prg.pmc.logging_setup import configure
    return configure(*args, **kwargs)


__all__ = [
    "__version__",
    "configure_logging",
]
