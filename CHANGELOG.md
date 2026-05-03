# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.3.0] - 2026-05-03

### Added

- **Parameter estimation** — `CopulaVirt.fit(data, method='tau'|'mle')` returns a `FitResult`. Pseudo-observations via ranks; method-of-moments (Kendall's τ) or maximum-likelihood (Brent on τ_K).
- **Model selection** — `CopulaVirt.fit_best(data, families=...)` ranks candidate families by AIC. `BivariateLaw.fit_best(data, left_family, right_family)` does the same with fixed margins.
- **Information criteria** — `aic`, `bic`, `aicc`, `hqc` properties on `FitResult` and `BivariateFitResult`.
- **Cross-validation** — `cv_loglik(K=5)` on both result classes; honest model-selection signal that penalises overfitting.
- **Goodness-of-fit** — `FitResult.gof_test(B=100)` and `BivariateFitResult.gof_test(B=100)`: Cramér-von Mises with parametric bootstrap. Joint variant catches misspecified margins too.
- **Bootstrap CIs** — `FitResult.bootstrap_ci(B=500)` for τ_K. `BivariateFitResult.bootstrap_ci(B=500)` for τ_K + every margin parameter.
- **IFM bivariate fit** — `BivariateLaw.fit(data, copula_class, left_family, right_family)` two-step inference for margins.
- **Tail dependence** — `CopulaVirt.tail_dependence() → (λ_L, λ_U)`. Default numerical limit on the diagonal CDF; analytical overrides on Product, Gaussian, Student, GH, Clayton, FGM, CubSec.
- **Visual diagnostics** — `FitResult.plot_diagnostics(plot_dir)` (6-panel: pseudo-obs+PDF, PP plot, λ_L curve, empirical copula, residuals, λ_U curve). `BivariateFitResult.plot_diagnostics(plot_dir)` (6-panel: data+PDF, two QQ plots, joint PP plot, λ_L/λ_U curves).
- **Result classes** — `FitResult`, `GoFResult`, `BivariateFitResult`, `BivariateBootstrapCI` exported from `prg.copulas`.

### Fixed

- `CopulaA12.pdf` no longer overflows for high τ. The original implementation evaluated `(1/u−1)^θ` directly, which underflows to 0 near u≈1 for large θ; `S^{1/θ−2}` then overflowed. Rewritten in log-space.
- Tail-dependence comments in `a12.py` and `a14.py` `__main__` blocks were incorrect (claimed λ_L = λ_U = 0). They now print the actual analytical values from `tail_dependence()`.
- All copula `__main__` scripts: `sys.path` injection moved above the first `from prg ...` import so `python prg/copulas/.../foo.py` works directly.
- `plot_samples` and `plot_overview` default sample count raised from 500 to 2000 for clearer scatter plots.

---

## [0.2.0] - 2026-05-03

### Added

- `BivariateLaw` and `ConditionalLaw` classes (Sklar joint, conditional pdf/cdf/sample).
- Sample method on every copula (Rosenblatt h-inversion via Brent).
- Plot suite: `plot_pdf`, `plot_cdf`, `plot_h_function`, `plot_samples`, `plot_overview`, `plot_multi_tau`.
- Test suite for conditional copulas, scientific identities, and bivariate construction.

---

## [0.1.0] - 2026-05-03

### Added

- Coherent exception system (`CopulaParameterError`, `CopulaNotAvailableError`, `SamplingConvergenceError`).
- Logging infrastructure throughout the package.
- Initial copula families: Product, Gaussian, Student, GH, Clayton, Frank, A12, A14, FGM, CubSec.

---

## [0.0.0] - 2026-05-03

### Added

- Initial project scaffold: `pyproject.toml`, `README.md`, `CHANGELOG.md`, `.gitignore`, `prg/__init__.py`

[Unreleased]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.0...HEAD
[0.3.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.0.0...v0.1.0
[0.0.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/tags/v0.0.0
