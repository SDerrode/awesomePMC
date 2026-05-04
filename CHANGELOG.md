# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.3.5] - 2026-05-04

### Changed

- **`CopulaStudent`** (`prg/copulas/elliptical/student.py`) — `df` (degrees of freedom ν) promoted from a hardcoded constant (4) to a free parameter.
  - `n_params = 2`; `PARAMETERS_SET_NAME = ["tau_k", "df"]` in `CopulaEnum`.
  - `CopulaStudent(tau_k=τ)` still works — `df` defaults to 4.0 if omitted (backward-compatible).
  - `df > 2` enforced in `_update_params` (raises `CopulaParameterError` otherwise).
  - `df` persisted back to `self.params['df']` so `plot_multi_tau`, bootstrap CI, and GoF bootstraps preserve it across round-trips.
  - **2-parameter MLE** — `CopulaStudent.fit` overrides the base class and jointly optimises (ρ, ν) via L-BFGS-B with bounds `ρ ∈ (−1, 1)`, `ν ∈ (2, ∞)`. `method='tau'` is accepted for API compatibility and warns before falling back.
  - Numerical guard added to `pdf` (errstate + isfinite/positive check → fallback `EPS`).
  - Tail dependence λ = 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) now uses the fitted ν; correctly → 0 as ν → ∞ (recovers Gaussian).

---

## [0.3.4] - 2026-05-04

### Added

- **`CopulaAMH`** (`prg/copulas/archimedean/amh.py`) — Ali-Mikhail-Haq copula. Generator φ(t) = log((1−θ(1−t))/t), θ ∈ [−1, 1). Closed-form CDF C=uv/W, PDF c=[1−θ(2−u−v−uv)+θ²(1−u)(1−v)]/W³, analytical h-function h(v|u)=v(1−θ(1−v))/W². τ_K = 1−2[θ+(1−θ)²log(1−θ)]/(3θ²), τ range ≈ [−0.182, 1/3). λ_L = λ_U = 0. Numerically stable τ→0 limit via Taylor series (avoids 0/0 at θ=0).
- **`CopulaPlackett`** (`prg/copulas/explicit/plackett.py`) — Plackett copula (constant cross-product ratio θ>0). Explicit CDF C=[S−√Δ]/(2(θ−1)) with S=1+(θ−1)(u+v), Δ=S²−4θ(θ−1)uv; θ=1→independence (uv). PDF c=θ[1+(θ−1)(u+v−2uv)]/Δ^{3/2}. Analytical h-function h(v|u)=[√Δ−(S−2θv)]/(2√Δ). τ_K=(θ+1)/(θ−1)−2θlogθ/(θ−1)², full range (−1,1). λ_L = λ_U = 0. Numerically stable τ→0 limit via Taylor at θ=1.
- **`CopulaEnum.AMH`** (ID 16) and **`CopulaEnum.PLACKETT`** (ID 17); auto-imported and picked up by `fit_best`.
- **`_AMH_TAU_MIN`** / **`_AMH_TAU_MAX`** constants in `_base.py` for the AMH τ range (computed at import time from (5−8ln2)/3 and 1/3−ε).

---

## [0.3.3] - 2026-05-04

### Added

- **`CopulaBB1`** (`prg/copulas/archimedean/bb1.py`) — BB1 (Joe-Clayton) copula with **both** lower- and upper-tail dependence. Generator φ(t) = (t^{−θ}−1)^δ, θ > 0, δ ≥ 1. Closed-form τ_K = 1 − 2/(δ(θ+2)), λ_L = 2^{−1/(θδ)}, λ_U = 2 − 2^{1/δ}. Special case δ = 1 reduces to Clayton (verified analytically).
- **2-parameter fitting** — `CopulaBB1.fit` always uses 2-D L-BFGS-B MLE over (θ, δ); `method='tau'` is accepted for API compatibility (falls back to MLE with a warning). `n_params = 2` so AIC/BIC correctly penalise the extra parameter.
- **`CopulaEnum.BB1`** (ID 15) with `PARAMETERS_SET_NAME = ["tau_k", "delta"]`; picked up automatically by `fit_best` and `__init__.py` auto-import.

### Fixed

- **`CopulaVirt.plot_multi_tau`** — replaced `self.__class__(tau_k=float(tau))` with `self.__class__(**{**self.params, 'tau_k': float(tau)})` so extra parameters (e.g., BB1's `delta`) are preserved when sweeping τ.

---

## [0.3.2] - 2026-05-04

### Added

- **`CopulaFrank`** — enabled (`AVAILABLE=True`); τ↔θ inversion now numerically stable for the full range τ ∈ (−1, 1) including high |τ| (tested up to ±0.95). Analytical `conditional_cdf` and `tail_dependence` (λ_L = λ_U = 0). Robustness guards (`minmaxEPS`, `np.errstate`, `isfinite`) on `pdf`/`cdf`.
- **`CopulaJoe`** (`prg/copulas/archimedean/joe.py`) — new Archimedean family with upper-tail dependence only. Generator φ(t) = −log(1−(1−t)^θ), θ ≥ 1. τ↔θ via a fast convergent series (2000 terms, O(1/k³)). Analytical `pdf`, `cdf`, `conditional_cdf`, `tail_dependence` (λ_L = 0, λ_U = 2 − 2^{1/θ}). Full numerical guard suite.
- **`SurvivalCopula`** base class + **`SurvivalClayton`**, **`SurvivalGH`**, **`SurvivalJoe`** (`prg/copulas/archimedean/survival.py`) — 180° rotation wrappers: ĉ(u,v) = c(1−u,1−v), ĥ(v|u) = 1 − h(1−v|1−u), (λ_L, λ_U) swapped. τ_K preserved under rotation.
- **`CopulaEnum`** — 4 new entries: `JOE` (ID 11), `SURVIVAL_CLAYTON` (12), `SURVIVAL_GH` (13), `SURVIVAL_JOE` (14). Auto-import and `fit_best` pick them up automatically via the existing enum-driven mechanism.

### Fixed

- **`kendall_tau_frank`** — replaced the `1 − D₁(θ)` formula with a direct integration of `1 − t/(e^t − 1)`, which vanishes at t = 0 and eliminates catastrophic float cancellation for small |θ| (previously τ(10⁻⁹) returned 1.0 instead of ≈ 0).
- **`find_theta_frank`** — bracket widened from the hardcoded `(1e-5, 50)` (failed for |τ| ≥ 0.93) to dynamic `(±1e-9, ±500)` covering |τ| up to ≈ 0.992; τ = 0 handled as a special case returning θ = 0 directly.

---

## [0.3.1] - 2026-05-04

### Added

- **`CopulaEnum.MODULE`** — champ `MODULE: str` dans `CopulaDataMixin` ; chaque entrée de `CopulaEnum` porte le chemin pointillé de son module (ex. `"prg.copulas.elliptical.gaussian"`).
- **Auto-import piloté par l'enum** — `prg/copulas/__init__.py` boucle sur `CopulaEnum` pour importer et exposer toutes les familles ; `__all__` est généré depuis l'enum. Ajouter une nouvelle copule ne nécessite plus de toucher `__init__.py`.
- **`fit_best` enum-driven** — `CopulaVirt.fit_best` et `BivariateLaw.fit_best` résolvent leurs familles par défaut via `CopulaEnum` + `importlib` (plus d'import en dur).
- **Logging sur les fallbacks numériques** — `a12.py`, `a14.py` et `clayton.py` émettent un `logger.debug(...)` à chaque retour de secours (EPS, 0 ou 1) pour traçabilité.

### Fixed

- **`CopulaA14.pdf`** — `ZeroDivisionError` silencieux quand `u` ou `v` = `ONE_MINUS_EPS` avec θ > 2 (τ > 0,67) : `pow(u, 1/θ) − 1` s'annule en float64. Protégé par `try/except` → retourne `EPS`.
- **`CopulaA14.conditional_cdf`** — même cause (`S^{1/θ−1}` avec S → 0). Protégé par `try/except` → retourne 1 si v > 0,5, sinon 0.
- **`CopulaClayton.pdf / cdf`** — statsmodels retournait `nan` pour `u` = `EPS` et θ grand. Entrées clampées par `minmaxEPS`, calcul sous `np.errstate`, résultat vérifié par `isfinite`.
- **`CopulaClayton.conditional_cdf`** — `inf × 0 = nan` pour `u` → 0⁺. Garde `isfinite` ajoutée → retourne 1 (dépendance de queue inférieure).
- **`set_dir` supprimée** de `prg/tools/tools.py` ; remplacée par `pathlib.Path(...).mkdir(parents=True, exist_ok=True)` dans tous les blocs `__main__`.
- **`list_parameters` supprimée** de `prg/tools/tools.py` ; la clé `"param_names"` du dict interne de `BivariateLaw` n'était jamais lue.

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

[Unreleased]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.5...HEAD
[0.3.5]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.4...v0.3.5
[0.3.4]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.3...v0.3.4
[0.3.3]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.2...v0.3.3
[0.3.2]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.1...v0.3.2
[0.3.1]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.0...v0.3.1
[0.3.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.0.0...v0.1.0
[0.0.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/tags/v0.0.0
