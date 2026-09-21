"""
awesomePMC — Copula-based Markov models for non-Gaussian transition kernels.

The package provides three complementary layers:

* :mod:`pmcprg.copulas` — 41 bivariate copula families (elliptical,
  Archimedean, survival, extreme-value, explicit, and the 90°/270° rotations
  for negative dependence) with a common ``CopulaVirt`` API: ``pdf``, ``cdf``,
  ``conditional_cdf``, ``inv_h``, ``sample``, ``fit``, plus vectorised
  ``pdf_array`` / ``logpdf_array``. ``CopulaEnum`` is the registry.

* :mod:`pmcprg.pmc` — Pairwise Markov Chain models (HMC-IN, HMC-IN2, HMC-DN,
  PMC-IN, PMC) loaded from TOML, with simulation, MPM classification,
  ICE / SEM unsupervised estimation, a CLI (``pmc``), and a PyQt6 GUI.

* :mod:`pmcprg.diagnostics` — model-agnostic goodness-of-fit tools, currently
  the multivariate Kolmogorov-Smirnov test (``mks_1samp`` / ``mks_2samp`` /
  ``mks_test``, Naaman 2021).

Quickstart — copula
-------------------
>>> from pmcprg.copulas import CopulaGaussian
>>> cop = CopulaGaussian(tau_k=0.6)
>>> cop.pdf([0.3, 0.7])                # density at one point
>>> samples = cop.sample(n=1000, seed=0)
>>> result  = CopulaGaussian.fit(samples, method='mle')
>>> result.aic, result.bootstrap_ci(B=200)

Quickstart — PMC
----------------
>>> from pmcprg.pmc import PMCModel, simulate, classify, ice
>>> mdl   = PMCModel("pmcprg/pmc/models/pmc_gauss_k2.toml")
>>> X, Y  = simulate(mdl, N=2000, seed=0)
>>> X_hat, gamma, log_lik = classify(mdl, Y)
>>> fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 20})  # trace.log_liks, trace.tau_history, …

Quickstart — CLI
----------------
After ``pip install -e .``:

    pmc simulate -m model.toml --N 5000 --out sim.csv
    pmc classify -m model.toml -d sim.csv  --out cls.csv
    pmc estimate -m init.toml  -d sim.csv  --out fitted.toml
    pmc gui      [model.toml]      # PyQt6 GUI (extra: pip install awesomepmc[gui])
"""

__version__ = "1.3.1"


def configure_logging(*args, **kwargs):
    """Configure the package-wide ``pmcprg`` logger.

    Thin wrapper around :func:`pmcprg.pmc.logging_setup.configure`. Imported lazily
    so that ``import pmcprg`` stays free of optional dependencies.
    """
    from pmcprg.pmc.logging_setup import configure
    return configure(*args, **kwargs)


__all__ = [
    "__version__",
    "configure_logging",
]
