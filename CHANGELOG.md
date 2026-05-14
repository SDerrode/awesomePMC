# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

(no changes yet)

---

## [0.7.0] - 2026-05-14

This release adds **SEM (Stochastic EM)** as a sister estimator to ICE
and introduces a **K-means warm-start** option shared by both
algorithms — both ideas cherry-picked and rewritten from the
`markovchain_todelete` companion project (files `prg/PMC_Estim.py` and
`prg/tools/probaPMC.py::simulRealisationAP`).

### Added

- **SEM — Stochastic EM estimator** in the new `prg/pmc/sem.py` module.
  Public API: `sem(model, Y, sem_cfg=None) -> (PMCModel, SemTrace)` and
  the image wrapper `sem_image(model, img, sem_cfg=None)`. SEM differs
  from ICE by drawing a single posterior realisation
  `X̃ ~ P(X | Y)` at every iteration (Forward-Filter Backward-Sample)
  and running the same supervised-style M-step on these hard labels.
  Shares the M-step (`_m_step`) and the K-means warm-start
  (`_warmstart_from_kmeans`) with ICE, so the two estimators stay
  exactly aligned on the parameter-update logic.
  - `SemTrace` mirrors `IceTrace` (same fields plus
    `sampled_X_history` capturing the per-iteration FFBS draws).
  - `SemResult` mirrors `IceResult` for GUI/diagnostics layers.
  - New config keys: `max_iter` (default 30), `sem_seed` (RNG seed for
    the FFBS draws). Other keys (`fit_margins`, `candidates`,
    `selection_criterion`, `margin_selection_rule`, `init`,
    `kmeans_seed`, `n_starts`, `multistart_seed`, `multistart_jitter`)
    are reused from ICE's config namespace.
  - `[sem]` TOML section recognised (falls back to `[ice]` for shared
    keys).
- **`sample_posterior(model, Y, rng)`** in `prg/pmc/inference.py` —
  Forward-Filter Backward-Sample (FFBS) draw of `X | Y`. Standalone
  utility, also used by SEM internally. Pre-computed `W`/`alpha_hat`
  can be passed in to avoid recomputation.
- **CLI `--algorithm {ice,sem}` flag** on both `estimate` and
  `estimate-image` (default `ice` — backward-compatible). Companion
  flag `--sem-seed` controls SEM's stochastic completion seed. Example:
  `python -m prg.pmc estimate --model M.toml --data Y.csv
  --algorithm sem --max-iter 30 --sem-seed 0 --fit-margins`.
- **GUI `_IceTab`** (ICE config panel) renamed conceptually to the
  "estimator" panel: new top "Algorithm" section with an `Estimator`
  combobox (`ice` / `sem`) and a `SEM seed` spinbox (auto-disabled
  when `ice` is selected). The worker (`_do_estimate`) dispatches to
  `ice()` or `sem()` based on the combobox value; the result-handling
  code is identical because `SemResult` and `IceResult` expose the
  same diagnostic fields.
- **K-means warm-start for ICE & SEM.** New TOML `[ice]` keys `init`
  (one of `"model"` — default, backward-compatible — or `"kmeans"`)
  and `kmeans_seed` (RNG seed forwarded to `sklearn.cluster.KMeans`).
  When `init = "kmeans"`, the estimator clusters `Y` with k-means++ and
  derives a warm-start model from the hard labels via a single
  supervised-style M-step (prior + copula τ always re-estimated;
  margins re-estimated only if `fit_margins` is true). The variant,
  K, margin distribution families and copula candidates are preserved.
  Multistart, when enabled, perturbs the warm-started model.
- **Optional `[ml]` extras** in `pyproject.toml` adding `scikit-learn` —
  required only by the K-means warm-start. Install with
  `pip install 'copulasformm[ml]'`.

### Changed

- **Refactor (no behaviour change)**: the ICE M-step (prior / margins /
  copulas update from posterior weights) is now a stand-alone
  `_m_step()` helper, called by both the ICE iteration loop, the SEM
  iteration loop, and the K-means warm-start path. This eliminates
  the duplication that was about to appear between ICE and SEM.

### Tests

- 16 new tests for the K-means warm-start (`test_ice_kmeans_init.py`).
- 8 new tests for the FFBS posterior sampler (`test_posterior_sampling.py`),
  including a Monte-Carlo check that the empirical marginal of many FFBS
  draws converges to the smoothed posterior γ_n(j).
- 11 new tests for SEM (`test_sem.py`): config plumbing, trace shapes,
  reproducibility under fixed `sem_seed`, stochastic variance across
  seeds, LL improvement from an adversarial init, end-to-end recovery
  with K-means warm-start, multistart, PMC variant smoke test, image
  wrapper smoke test.
- 4 new GUI tests (`_IceTab`): `init` / `kmeans_seed` widgets and the
  new `algorithm` / `sem_seed` widgets.

### Origin

The SEM and K-means warm-start ideas are cherry-picked from the
`markovchain_todelete` companion project (`prg/PMC_Estim.py:65` for
K-means; `prg/tools/probaPMC.py::simulRealisationAP` for FFBS-based
stochastic completion).

---

## [0.6.0] - 2026-05-05

This release adds **GICE** — automatic margin family selection from
Derrode-Pieczynski (Signal Process. 2016) — on top of the existing
ICE machinery, polishes the PyQt6 GUI substantially, and parallelises
the CSDA-2013 reproduction script. No public-API breaking changes.

### Added

- **GICE — automatic margin family selection (paper A23 / SP 2016, §3).**
  Margin blocks now accept an optional ``candidates`` list (a set of
  ``scipy.stats`` family names). When non-empty, the ICE M-step fits
  each candidate and picks the winner via a configurable rule:
  ``mle`` (default), ``kolmogorov`` (SP 2016, Example 3.1),
  ``aic`` or ``bic``. Eight families ship with data-aware
  initialisation heuristics: ``norm, gamma, invgamma, betaprime,
  lognorm, expon, weibull_min, beta``. Other ``scipy.stats`` names
  still work; the M-step falls back to ``scipy.stats.<dist>.fit`` for
  the init point. Public constants ``ice.GICE_KNOWN_FAMILIES`` and
  ``ice.SP2016_DEFAULT_CANDIDATES`` exposed.
- **New shipped fixture** ``models/sp2016_gice_k2.toml`` reproducing
  the SP-2016 §5.1 setup (Gamma + BetaPrime margins).
- **GUI: new ICE view "Parameters: true vs fitted"** (view *M*) — at
  a glance comparison of seed and ICE-fitted parameters: per-state
  margins (true vs fitted family + params), per-pair copulas
  (family + ``τ`` + ``Δτ``, with colour flags for family changes
  and ``|Δτ| > 0.10``), and three K×K joint-prior heatmaps
  (true / fitted / ``fitted − true``).
- **GUI: inline Export button** next to the View combobox, mirroring
  ``File → Export plot…``.
- **GUI: figure-level legend** for the dashboard's family-ribbon
  panel — the ribbon colours now have a decoder, no longer hidden
  inside a stripped-out per-axes legend.
- **GUI: auto-load** the bundled fixture
  ``prg/pmc/models/pmc_gauss_k2.toml`` when the GUI is launched
  without a path argument, so the user lands on a workable state
  instead of a fully-disabled UI.
- **GUI: GICE candidate sets edited as checkboxes**, both at the
  ICE-config level (``_IceTab``) and per-margin (``_MarginDialog``).
  Replaces the previous comma-separated text boxes; quick-pick
  buttons (All / None / SP-2016 §3 / Defaults) cover the common
  selections.
- **GUI: 🎲 random-seed button** for the multistart seed widget,
  drawing from ``secrets.randbelow(2³¹)``. Default seed changed
  ``0 → 42``.
- **GUI: LaTeX-mathtext labels** across every plot (axes, titles,
  legends use ``$Y_n$``, ``$P(X_n = k \mid Y)$``,
  ``$\Vert p^{(t)} - p^{(t-1)} \Vert_F$``, ``$\hat{\tau}$``, etc.)
  rendered with the bundled Computer-Modern font set. Compact font
  sizes: titles 10 pt, axes 9 pt, ticks 8 pt; constrained_layout on
  by default so 3×2 dashboards no longer overlap.
- **GUI: canvas figsize** bumped from (7, 5) to (9.5, 6.5) with
  ``minimumSize=(640, 460)`` — multi-panel figures get room to
  breathe.
- **GUI: Save action** now correctly avoids overwriting the bundled
  fixture when the GUI was auto-loaded; the first ``Ctrl+S`` falls
  through to ``Save As…`` so the user picks a destination.
- **Report: parallel mode (default).** ``reproduce_csda2013.py``
  now dispatches reps and ICE runs across a
  ``ProcessPoolExecutor`` with ``--jobs N`` (default
  ``cpu_count() // 2``). On a 10-core laptop ``--full`` drops from
  ~3 h to ~30-50 min. Results are bit-identical to the sequential
  path for the same seeds (verified end-to-end with a CSV diff).
- **Report: progress bar + ETA** for every experiment row / ICE
  config. Stdlib-only (no tqdm dependency); TTY rewrites in place,
  non-TTY emits one line per ~10 % rate-limited to ≤ 1 / 3 s.
- **Documentation: dedicated ``prg/pmc/README.md``** that opens
  with both source-paper citations (A16 / A23), a feature ↔ paper
  table, and BibTeX entries.
- **Documentation: GitHub Actions workflow** ``.github/workflows/ci.yml``
  mirroring ``.gitlab-ci.yml`` (lint + matrix tests on Python
  3.11 / 3.12 / 3.13, headless Qt via ``QT_QPA_PLATFORM=offscreen``).

### Changed

- ICE selection criteria expanded from MLE-only to
  ``{mle, aic, bic, huard, cvm}`` for copulas (A16 Eq. 20 implemented
  as ``huard``) and ``{mle, kolmogorov, aic, bic}`` for margins
  (A23 §3).
- Default ``N_default`` lowered from 5000 → 1000 in every shipped
  fixture except ``sp2016_gice_k2.toml`` (which keeps the paper's
  3000), so the auto-loaded default lands on a fast turnaround.
- Top-level README's "References" section rewritten with full
  citations for **both** A16 (CSDA 2013) and **A23** (SP 2016),
  including BibTeX. Fixed pre-existing author typo
  "Piecini" → "Pieczynski"; added ``docs/SP_2016.pdf`` to the
  documented directory tree.
- ``[project.urls]`` in ``pyproject.toml`` gains a
  ``Mirror = https://github.com/SDerrode/copulasformm`` entry.

### Fixed

- ``simulate`` no longer crashes with
  ``ValueError: Probabilities do not sum to 1`` when the model's
  cached ``π`` / ``A`` rows have drifted within the validator's
  ``atol=1e-6`` tolerance but past ``rng.choice``'s tighter
  ``√eps ≈ 1.5e-8`` bound. Defensive renormalisation in
  ``simulate.py``.
- Margin & copula ``QTableWidget`` headers now display state indices
  in italic math (``$i=0$`` / ``$j=0$``) instead of plain ASCII.
- Multi-panel figure layouts (ICE dashboard, K×K small multiples)
  no longer stack their suptitle onto the first row's tick labels.
- Param-comparison view: section titles can no longer overlap the
  table headers (each section now lives in its own ``subgridspec``
  with a dedicated header row).
- ``_draw_tau_trajectories`` no longer emits a ``No artists with
  labels found`` UserWarning on HMC variants whose τ-history is all
  NaN.

### Source-paper attribution

Two papers by S. Derrode and W. Pieczynski:

- **A16** — *Unsupervised data classification using pairwise Markov
  chains with automatic copulas selection*, Comput. Stat. Data Anal.
  63 (2013), 81-98.
  [doi:10.1016/j.csda.2013.01.027](https://doi.org/10.1016/j.csda.2013.01.027)
- **A23** — *Unsupervised classification using hidden Markov chain
  with unknown noise copulas and margins*, Signal Process. 128
  (2016), 8-17.
  [doi:10.1016/j.sigpro.2016.03.008](https://doi.org/10.1016/j.sigpro.2016.03.008)

See [`prg/pmc/README.md`](prg/pmc/README.md) for the feature ↔
paper map.

---

## [0.5.0] - 2026-05-04

This is a substantial release covering three audit cycles, a new ICE
diagnostics layer (12 GUI views + animated playback), and a
**breaking-change** clean-up of the SR-PMC marginal-density contract.

Highlights:

- **Breaking** — ``ice()`` now returns ``(PMCModel, IceTrace)`` instead
  of ``(PMCModel, list[float])``. Use ``trace.log_liks`` for the plain
  history.
- **Breaking** — TOML margin schema migrated to K-format (one block per
  state). Legacy K² format is still accepted (with INFO/WARNING).
- **Breaking** — ``PMCModel.margin(i, j)`` now ignores the second
  argument under the SR-PMC contract; ``j`` is optional.

### Breaking — SR-PMC marginals contract enforced

The package now enforces the SR-PMC reversibility identity from CSDA 2013:
the marginal density of an observation depends **only on its current
state**, never on the other state of the bivariate pair. There are
therefore exactly **K marginal densities** in every variant — not K² —
and they are indexed by ``i`` only.

* **TOML schema** — the canonical [[margins]] format now lists K blocks
  (``i``, ``dist``, ``params``); the legacy K² (``i``, ``j``)-keyed
  format is still accepted but:
    * INFO-logged as legacy when entries respect tying;
    * WARN-logged listing each conflicting (i, j) when entries violate
      tying — the (i, 0) anchor is kept as the canonical density.
* **`PMCModel.margin(i, j=None)`** — second argument is now optional and
  ignored under SR-PMC. ``margin(i, 0) is margin(i, 1) is margin(i)``.
* **`PMCModel._state_margins: dict[int, _MarginDist]`** is the new
  internal source of truth (K entries instead of K²); the legacy
  ``_margins[(i, j)]`` mapping is still present for back-compat but all
  entries with the same ``i`` point to the same instance.
* **`PMCModel.margin_blocks()`** always returns the canonical K-format
  (regardless of how the source TOML was written).
* **`PMCModel.weight()` formulas** rewritten in per-state notation
  (``f_i(y_n) · f_j(y_{n+1}) · c_{ij}(F_i(y_n), F_j(y_{n+1}))``).
* **ICE M-step** now fits exactly K marginal densities (one per state)
  weighted by the full ``γ_n(i) = P(X_n=i | Y)``, instead of K² densities
  with partial dual-view weights. Saves K-fold redundant MLE per
  iteration and eliminates the "K replicas drift apart from each other"
  failure mode at finite N.
* **`precompute_weights`** evaluates K margin PDFs/CDFs (vector form)
  and broadcasts to the legacy (N, K, K) tensor shape. Same numerical
  result, K× less work.
* **`simulate._sample_copula_conditional`** uses
  ``model.margin(i).cdf`` and ``model.margin(j).ppf`` (state-only
  indexing) — Rosenblatt inversion is unchanged.
* **TOML model fixtures rewritten** in K-format:
  ``pmc_gauss_k2.toml``, ``pmc_in_gauss_k2.toml``,
  ``hmc_in2_gauss_k2.toml``, ``hmc_dn_gauss_k2.toml``.
  ``hmc_in_gauss_k2.toml`` and ``hmc_in_gauss_k3.toml`` already used the
  K-format.
* **GUI Margins tab** — collapsed from K×K grid to **K cells (one per
  state)**, with explanatory caption. The Copulas tab remains K×K
  (copulas ARE pair-indexed).
* **`Variant.per_class_margin`** is now ``True`` for every variant
  (was: only HMC-IN). Property kept for API stability; it now describes
  a contract, not a per-variant difference.

### Tests added

- ``test_pmcmodel_collapses_legacy_k2_margins`` — tied legacy TOML loads
  with INFO log.
- ``test_pmcmodel_warns_on_legacy_k2_untied_margins`` — untied legacy
  TOML logs a WARNING listing conflicts.
- ``test_pmcmodel_k_format_canonical`` — round-trip and identity of
  ``margin(i, *)`` lookups.
- ``test_ice_preserves_state_margin_tying`` — after ICE,
  ``margin(i, j) is margin(i)`` for every (i, j).

### Refactor (codage audit — third pass)

- **`EXTRA_PARAM_BOUNDS` deduplicated** — single source of truth in
  ``prg.pmc.ice``; the GUI dialog imports it instead of holding a copy.
- **`_BlockGridTab` base class** in ``prg.pmc.gui.tabs`` — Margins and
  Copulas tabs shared 90 % of their code; now each subclass only sets four
  class attributes and overrides ``_format_block`` (~80 LOC saved).
- **`IceResult` dataclass** packages
  ``(initial_model, fitted_model, Y, trace)`` — replaces the worker's
  ad-hoc 4-tuple and shrinks ``PMCMainWindow`` instance state.
- **`_compute_marginal_cdfs(model, Y)`** helper — replaces three
  duplicated ``f_cdf`` K×K loops in ``_do_gof_test``,
  ``_plot_ice_pseudos``, ``_plot_ice_pp``.
- **`_begin_figure()` / `_finalize_plot()` lifecycle helpers** — every
  ``_plot_*`` method now relies on the single canvas-clear and the single
  ``_has_plot``/Export-plot toggle in ``_render_view``. ~30 lines of
  boilerplate eliminated.
- **`_VIEW_DISPATCH` table** replaces the 17-branch ``if/elif`` chain in
  ``_render_view``; adding a view is a one-line entry + a new
  ``_plot_*`` method.
- **`_render_error()` UI** — render failures inside a view now paint a
  friendly message on the canvas and append the exception to the log
  panel (was: silent debug log).
- **Imports hoisted** — ``EPS``, ``ONE_MINUS_EPS``, ``classify``,
  ``error_rate``, ``_empirical_copula``, ``IceTrace`` moved to module
  scope; redundant ``Qt as _Qt`` alias and ``np as _np`` alias removed.
- **Type hints on ICE plot methods** — ``_plot_ice_*`` now declare
  ``trace: IceTrace`` (and ``model: PMCModel`` / ``Y: np.ndarray`` where
  applicable); the diagnostics layer is statically checkable.
- **`IceTrace` type hints tightened** — ``tau_history`` and
  ``p_history`` are now plain ``np.ndarray`` (never ``None``); empty
  traces use ``np.empty((0, 0, 0))``. Docstring gains an ASCII timeline
  diagram explaining snapshot semantics.
- **HMC-IN trace test** verifies ``tau_history`` is all-NaN /
  ``family_history`` all-empty / ``p_history`` sums to 1 for the
  no-copula variant.
- **Top-of-file docstring of ``main_window``** rewritten to describe the
  View-selector architecture and the dispatch flow.

### Added (ICE diagnostics — visual layer)

- **`IceTrace` dataclass** captured per iteration by the ICE driver:
  ``log_liks``, ``tau_history`` ``(T, K, K)``, ``family_history``
  ``(T × K × K SHORT_NAMEs)``, ``p_history`` ``(T, K, K)``,
  ``margin_history`` (per-block params), ``multistart_runs`` (losing
  traces when ``n_starts > 1``), ``run_tag``, ``candidates``.
- **`ice()` now returns ``(PMCModel, IceTrace)``** — backward-incompatible
  signature change. Use ``trace.log_liks`` for the plain LL history.
- **GUI View selector** (combobox above the canvas) — single dispatch
  point for every visualisation; unavailable views are kept in the list
  but disabled with an explanatory tooltip.
- **12 ICE diagnostic views** accessible from the View selector:
  - **A.** τ trajectories per pair (★ markers at family changes)
  - **B.** Family-selection ribbon (one row per pair × T columns,
        candidate-coloured bands)
  - **C.** Pseudo-observations + fitted log-PDF contours, K×K
        small-multiples
  - **D.** Multistart comparison (winner highlighted, losing runs grey)
  - **E.** Joint-prior evolution + ‖Δp‖_F log-y
  - **F.** Margin-parameter evolution (per-block small-multiples)
  - **G.** ICE dashboard (3×2: LL, τ, family, ‖Δp‖, AIC/BIC, summary)
  - **H.** PP plot of fitted copulas (K×K)
  - **I.** Tail dependence λ_L / λ_U heatmap per pair
  - **J.** γ posterior comparison BEFORE vs AFTER ICE
  - **K.** Animated playback — QSlider + Play/Pause button advances
        through iterations, refreshing a 4-panel snapshot view
        (LL / τ / family / p heatmap)
- **CLI / notebook updated** to consume ``IceTrace`` (``trace.log_liks``).

### Added (GUI — second pass)

- **Auto-mirror of SR-PMC `p` cells** — editing `p[i,j]` now mirrors to
  `p[j,i]` live (with an explanatory caption above the table).
- **Modified-indicator (`*`)** in the window title flags unsaved edits;
  reset on Save / Save As / Open / ICE finish.
- **Recent-files submenu** in *File* — last 5 TOML paths persisted via
  `QSettings`; "Clear list" entry to wipe the history.
- **GoF p-value heatmap** — after running GoF, the canvas now shows a
  K×K colour-coded heatmap (red→green, with a 0.05-contour and ‘×’
  markers for errored pairs) in addition to the textual log entry.
- **Per-pair GoF progress** — the determinate progress bar advances one
  step per copula pair (pair tag shown in the status bar).
- **Confirmation before Simulate overwrite** — if `(X, Y)` are already
  loaded, the user is asked before they get overwritten.
- **Auto-scroll log panel** — newly-appended lines are always visible.
- **Pointing-hand cursor** on margin/copula table cells (visual hint
  that double-click opens an editor).
- **Package version in *About*** — reads `prg.__version__`.
- **Export-plot disabled when canvas is empty** — no more accidental
  blank PNG dumps.
- **Symmetrize feedback** — clicking *Symmetrize p* logs
  ``"Symmetrized prior: max asymmetry was 3.4e-04"`` in the log panel
  via the new `_PriorTab.symmetrized(asym, was_renorm)` signal.

### Fixed (GUI — second pass)

- **`closeEvent` waits for the worker** — closing the window during a
  long ICE / GoF run prompts the user (Wait / Terminate / Cancel)
  instead of leaking the `QThread`.
- **`_PriorTab` cell validation** — invalid cells (typos like ``"0.4x"``)
  now raise a typed `PriorTabError` caught by the host, surfaced as a
  friendly QMessageBox; previously they were silently treated as 0.
- **`_Worker.set_progress_cb`** — public setter replaces the previous
  `_kwargs` mutation hack from the host window.
- **Imports cleanup** — `QHBoxLayout/QWidget` now imported at the top of
  `dialogs.py`; `matplotlib.pyplot` imported once at the top of
  `main_window.py` (after `matplotlib.use("QtAgg")`).

### Tests

- 14 new tests in `test_gui_dialogs.py` covering: auto-mirror, prior
  cell validation, symmetrize signal, ICE-tab `changed` signal,
  `_do_gof_test` schema and progress-callback, `_perturb_initial_model`
  non-trivial perturbation, `_read_data_csv` happy path / empty / missing
  Y / non-numeric Y / non-integer X.

### Added (GUI)

- **Multistart controls in the ICE tab** — exposes the four new options
  (`n_starts`, `multistart_seed`, `multistart_jitter`, `patience`).
  Seed/jitter spinboxes auto-disable when `n_starts == 1`.
- **Family-aware copula dialog** — the τ spinbox is reranged dynamically
  to match the selected family's `TAU_MIN_MAX`, and extra-parameter
  widgets (BB1 `δ`, Student `df`) appear only when the chosen family
  declares them. Eliminates the silent τ-clip footgun.
- **Robust margin parameters parser** — accepts both spaces and commas
  as separators (`"loc=0, scale=1"` works as well as `"loc=0 scale=1"`),
  flags unparseable tokens via a friendly `QMessageBox` instead of
  silently dropping them.
- **Symmetrize-p button** in the Prior tab — one-click enforcement of
  `p[i,j] = p[j,i]` for SR-PMC joint priors (visible only for those
  variants).
- **GoF test action** — a new button runs the Cramér-von Mises GoF
  (parametric bootstrap, `B=200`) on every pair-copula and reports
  per-pair p-values in a dialog and the log panel.
- **File ▸ Save data (CSV)** — saves the current `(X, Y)` sequence,
  symmetric with *Load data*.
- **File ▸ Export plot…** — writes the matplotlib canvas to PNG/PDF/SVG.
- **Edit ▸ Reset session (Ctrl+R)** — clears loaded data, canvas, and log.
- **Determinate ICE progress bar** — `ice()` now accepts
  `progress_cb=callable`. The GUI worker connects this to a Qt signal so
  the bar advances iteration-by-iteration, with per-run tagging when
  multistart is on (`ICE [perturbed-1] iter 3/8 …`).

### Fixed (GUI)

- `_load_model` no longer commits `_model_path` until `PMCModel()`
  succeeds — failed loads leave the GUI consistent.
- `_on_load_data` now validates that every `Y` cell parses as a float
  and reports the offending row number; non-integer `X` cells are
  ignored with a warning rather than crashing.
- Removed dead code (`_IceTab._cop_visible` was set and never read;
  unused `mdl` parameter on `_MarginTab._refresh_table`).

### Added

- **ICE multistart** — `ice()` now accepts `n_starts` (default `1`),
  `multistart_seed` and `multistart_jitter` config keys. With
  `n_starts > 1` the driver runs ICE several times from random
  perturbations of the initial model and returns the run with the highest
  final log-likelihood. The first start is always the unperturbed model,
  so the legacy single-start behaviour is preserved by default.
  Hardens fits against multimodal log-likelihoods (Archimedean copula
  selection, HMC variants with similar margins).
- **Data-aware init in `_fit_copula_params`** — weighted Kendall's τ on
  the pseudo-observations is now computed once and used (a) as the
  initial value for the multi-parameter L-BFGS-B (BB1, Student) and
  (b) as the fallback when the 1-D `minimize_scalar` path errors.
  Replaces the previous mid-range / Pearson-based fallbacks.
- **Boundary tests for Archimedean copulas** — `prg/tests/test_copulas.py`
  now exercises every Archimedean family (GH, Clayton, Joe, A12, A14,
  AMH, Frank, BB1, plus the three Survival rotations) at τ_K = τ_min +
  0.95·span and τ_min + 0.05·span. Verifies (i) finite PDF / CDF /
  conditional CDF, (ii) `correct_tau` clips out-of-range τ with a
  WARNING, (iii) `correct_tau` handles NaN, (iv) `logpdf_array` does
  not silently saturate near `-36`, (v) `inv_h_array` returns values
  strictly inside (0, 1).

### Fixed

- **`CopulaA12.conditional_cdf`** now traps `OverflowError` /
  `ZeroDivisionError` and returns the diagonal limit `𝟙{v ≥ u}` (same
  pattern as A14). Previously raised for τ ≳ 0.85 when probed at
  v = EPS, which made `inv_h(brentq)` fail near the boundary. Surfaced
  by the new boundary tests.

### Changed

- **`CopulaVirt.sample` uses Rosenblatt inversion via `inv_h`** instead
  of inlined Brent search. Combined with the new vectorised
  `inv_h_array` (Gaussian, Clayton, Frank), this gives:
    - Gaussian sample(2000): **1.67 s → 0.001 s** (~1500×)
    - Clayton  sample(2000): **0.21 s → 0.001 s** (~200×)
    - Frank    sample(2000): **~50× faster**
- **`BivariateLaw.sample_conditional` uses Rosenblatt** instead of
  acceptance-rejection. Removes the `MAX_AR_ITER` failure mode for
  high-correlation cases and gives O(1) per sample for closed-form-`inv_h`
  copulas. The `MAX_AR_ITER` constant and `SamplingConvergenceError`
  exception are kept exported for backward compatibility.
- **`CopulaVirt.fit(method='mle')` for multi-parameter families** —
  the base-class implementation now does joint L-BFGS-B over (τ, *extras)
  for any family declared with extra parameters in `CopulaEnum`. Student
  and BB1 keep their existing per-class overrides; the base path is the
  fallback for any future 2-parameter family.
- **`CopulaEnum.correct_tau` warns on clipping** — when the input τ is
  outside the family's valid range or non-finite, a `WARNING` is logged
  with the original and clipped values.
- **Bootstrap progress logging** — `gof_test`, `bootstrap_ci`,
  `BivariateFitResult.gof_test` and `bootstrap_ci` now emit a
  `logger.info(...)` at every 10 % of progress for B ≥ 10.

### Added

- **`CopulaVirt.inv_h_array(w, u)`** — vectorised inverse h-function.
  Default fallback is a scalar Python loop; closed-form vectorised
  overrides on Gaussian, Clayton, Frank.
- **Native `logpdf_array` overrides** on six additional copulas:
  AMH, A12, A14, FGM, CubSec, Plackett. Combined with the existing five
  (Gaussian, Clayton, Frank, Joe, BB1) and the three Survival wrappers
  that delegate to their base, **14/17 copulas now have a native
  log-PDF** with extended dynamic range.

- **Native `logpdf_array` overrides** on the five most-used copulas:
  Gaussian, Clayton, Frank, Joe, BB1. These compute log c(u, v) directly
  in the log domain, bypassing the ``log(max(pdf, EPS))`` floor of the
  base implementation. Dynamic range extends below −36 nats (the
  saturation point of the floored path), useful for tails and
  goodness-of-fit on small-density regions.
- **`prg/tests/test_pdf_array.py::test_logpdf_array_native_extends_dynamic_range`**
  — verifies that for the 5 native overrides, (a) the regular range
  matches the floored path to 1e-9 and (b) at least one extreme-tail
  point goes strictly below the floored saturation.
- **SR-PMC dual-view margin update in ICE** — for pair-indexed margins
  (`HMC-IN2`, `HMC-DN`, `PMC-IN`, `PMC`), each `Y[k]` now contributes to
  the weighted MLE of `f_{ij}` from BOTH its "first observation" view
  (weighted by `ξ[k, i, j]`) and its "second observation" view of the
  reversed pair (weighted by `ξ[k-1, j, i]`). Effectively doubles the
  sample size used for each margin, giving ~√2× tighter parameter
  estimates without violating the SR-PMC factorisation.
- **SR-PMC stationary-reversibility diagnostic** — `PMCModel` warns at
  construction if `[prior].p` is not symmetric (max |p[i,j] − p[j,i]| ≥
  1e-6). The package commits to the SR-PMC factorisation; non-symmetric
  `p` matrices give a non-reversible chain and the inference may
  produce statistically dubious results.
- **`error_rate` symmetric warning** — also warns when `X̂` uses labels
  *outside* the true set (typical of a mis-specified K), in addition to
  the existing warning when `X̂` collapses to fewer classes.

### Fixed

- **🔴 ICE `fit_margins=True` no longer overwrites the declared margin family.**
  The previous `_fit_gaussian_margin_weighted` always returned
  `dist="norm"`, silently replacing any non-Gaussian margin (`expon`,
  `lognorm`, `gamma`, …) with a Gaussian. The new `_fit_margin_weighted`:

  - keeps the closed-form Gaussian fast path,
  - dispatches to a generic `scipy.optimize.minimize` (L-BFGS-B) on the
    weighted negative log-likelihood for any other `scipy.stats` family,
  - preserves the `dist` field — only `params` are updated,
  - logs a warning and keeps initial parameters on optimiser failure.

  Verified on `(norm, lognorm)` mixed-margin model: families preserved,
  parameters recovered to within ~10 % on N=2000.

- **🟡 `precompute_weights` (and therefore `forward`, `classify`, `ice`)
  now reject `N < 2`** with a clear error message instead of crashing
  on a cryptic NumPy ``zero-size array reduction`` deep inside the
  call stack.

- **🟡 `CopulaVirt.majorant` is now a guaranteed upper bound** on
  `max_v c(u, v)`. The default has been replaced by a hybrid:
  150-point grid → 1-D scalar refinement → ×1.02 safety multiplier.
  In addition, `CopulaGaussian.majorant` and `CopulaClayton.majorant`
  now use closed-form expressions and return exact bounds.

  Empirical deficit (`true_max / bound − 1`) across 11 families went
  from up to **0.17 %** (Clayton τ=0.5) down to **0**; the AR sampler
  in `BivariateLaw.sample_conditional` is now unconditionally unbiased.

### Added

- **`prg/tests/test_majorant.py`** — 17 tests (one per copula family)
  verifying `cop.majorant(u) ≥ max_v c(u, v)` against a high-precision
  reference computed via `scipy.optimize.minimize_scalar`.
- **`prg/tests/test_pmc.py::test_ice_fit_margins_preserves_family`** —
  regression test for the silent-overwrite fix.

- **`prg/tests/test_forward_backward_invariants.py`** — 36 tests
  (6 invariants × 6 demo models) verifying the probability-conservation
  laws of the forward-backward algorithm:
  - `Σ_j α̂_n(j) = 1` (filtered posterior),
  - `Σ_j β̂_n(j) = 1` (Devijver-normalised backward),
  - `Σ_j γ_n(j) = 1` (smoothed posterior),
  - `Σ_{i,j} ξ_n(i, j) = 1` (joint pair posterior),
  - `Σ_j ξ_n(i, j) = γ_n(i)` (left-marginal consistency),
  - `Σ_i ξ_n(i, j) = γ_{n+1}(j)` (right-marginal consistency).

  Observed deviations are at machine epsilon (~2e-16) on all six demo
  models. These invariants would have caught the PMC kernel bug fixed
  earlier in this release.

- **`CopulaVirt.cdf_array(uv)`** — vectorised CDF on an `(M, 2)` array,
  same fast-path strategy as `pdf_array`. Closed-form overrides for FGM,
  AMH, A12, A14, BB1, CubSec, Plackett, Joe, and the three Survival
  wrappers. Used by the GoF bootstrap.
- **`prg/tests/test_pdf_array.py::test_cdf_array_matches_scalar`** —
  parametrised over all 17 copulas; Student is correctly skipped (no
  closed-form CDF).
- **`logger.warning`** in `prg.pmc.inference.error_rate` when X̂ uses
  fewer classes than X_true (typical symptom of a degenerate ICE solution).
- **Sanity assertion** in `CopulaVirt.inv_h` — verifies that
  `conditional_cdf` is monotone in v before launching Brent. Catches
  buggy custom copulas with a clear message instead of an opaque
  `bracket bracket` error from scipy.

### Changed

- **`prg.copulas._fit._eval_log_likelihood`** uses the vectorised
  `logpdf_array` (≈ **300×** faster on N=5000 — was 0.30 s, now 0.001 s).
- **`prg.copulas._fit._cvm_statistic`** uses `cdf_array` (~5× faster on
  the CDF-evaluation step; the `(n×n)` empirical-copula matrix remains
  the dominant cost for large N).
- **`_class_colors(n_classes)`** indexes the `tab10` palette modulo 10
  rather than dividing the discrete index. K ≤ 10 callers get the
  canonical colours; K > 10 callers cycle (rather than getting
  interpolated intermediates).
- **`BivariateFitResult.cv_loglik`** docstring documents the choice of
  −100 nats as the support-violation floor (vs `log(MIN_POSITIVE) ≈ -708`).

### Fixed

- **🔴 Critical: `prg.pmc.inference.precompute_weights` for PMC and PMC-IN
  variants double-counted the marginal density `f_{ij}(y_n)`** at every
  recursion step. The transition kernel was the *joint* pair density rather
  than the *conditional* kernel. As a consequence:
  - the reported log-likelihood was off by an additive Y-dependent term
    (≈ -2 nats per step on the demo models);
  - smoothed posteriors `γ_n(j)` and joint posteriors `ξ_n(i, j)` were
    biased — by up to 0.6 in extreme cases;
  - MPM error rate was degraded by 1–3 percentage points;
  - ICE optimised a wrong objective and reported wrong AIC/BIC.

  The fix divides the joint kernel by `D[n, i] = Σ_{j'} p[i, j'] · f_{ij'}(y_n)`
  (the marginal `p(X_n=i, Y_n=y_n)`), recovering the correct conditional
  transition kernel. After the fix:
  - all 5 variants match brute-force enumeration to **machine precision**
    (1e-14 on K=2, N=5);
  - PMC-IN now coincides with HMC-IN2 when `p[i,j] = π_i · A[i,j]` (as
    expected mathematically);
  - PMC classification error on `pmc_gauss_k2` drops from **10.7 % to 7.7 %**.

### Added

- **`prg/tests/test_forward_correctness.py`** — 19 tests guarding the
  Devijver kernel against future regressions:
  - 18 parametrised cases (5 demo models + K=3 × 3 seeds) compare
    `forward()` to brute-force enumeration of all `K^(N+1)` state
    sequences (N=5);
  - one test verifies that PMC-IN and HMC-IN2 give identical log-lik
    when `p[i,j] = π_i · A[i,j]` and margins coincide.

- **`examples/quickstart.ipynb`** — end-to-end demo notebook (25 cells, 14
  code) covering: copula PDF / fitting / sampling, vectorised `pdf_array`
  benchmark, Sklar bivariate law, PMC simulation → classification → ICE,
  K=3 example.
- **`prg/tests/test_quickstart_notebook.py`** — `slow`-marked regression
  test that re-runs every cell with a fresh kernel; fails on any cell
  exception.

- **`prg/pmc/models/hmc_in_gauss_k3.toml`** — first K=3 example model (three
  well-separated Gaussian regimes, sticky transitions). All `test_pmc.py`
  parametrised tests pick it up automatically; a dedicated
  `test_classify_hmc_in_k3` verifies error rate < 5 %.
- **`prg/tests/test_logging_setup.py`** — 5 tests covering `configure()`
  (no-file, explicit-path, idempotence, unwriteable-path) and
  `add_widget_handler()` (PyQt6 round-trip).  Coverage of
  `logging_setup.py`: 27 % → **95 %**.
- **`CSVLoadError`** in `prg.exceptions` (replaces the private
  `_DataLoadError` from `cli.py`). Now reusable from library code; keeps
  the same hierarchy (`PMCError, IOError`).

### Changed

- **`MAX_AR_ITER`** in `prg.copulas.bivariate` — renamed from `_NBITERMAX`,
  now public and documented (Sphinx-compatible `#:` comments). Tweak via
  `prg.copulas.bivariate.MAX_AR_ITER = …` if a custom copula has a poor
  majorant.
- **`CopulaEnum` access aligned** — every iteration now uses
  `c.value.X` (CLASS_NAME, MODULE, AVAILABLE, …) instead of mixing
  `c.X` and `c.value.X`. The dataclass mixin still accepts both, but the
  codebase is internally consistent.
- **`_class_colors(K)` → `_class_colors(n_classes)`** — PEP-8 compliant
  parameter name. The `import matplotlib.pyplot as plt` was hoisted out
  of the function body (lazy import was unnecessary in a Qt module).

### Added

- **`Variant` enum properties** — `variant.uses_copula`, `variant.has_markov_prior`,
  `variant.per_class_margin`. The legacy `USES_COPULA` / `MARKOV_PRIOR` /
  `PER_CLASS_MARGIN` sets are kept as backward-compatible aliases.
- **`prg/tests/test_cli.py`** — 7 subprocess-based smoke tests for the
  ``pmc`` console script (``--help``, simulate, classify, estimate;
  reproducibility, missing-data error path).
- **`pytest.mark.slow`** — marker registered in pyproject; CLI estimate
  (the longest single test at ~2 s) is tagged. Run `pytest -m "not slow"`
  for fast iteration.
- **CI updated** (`.gitlab-ci.yml`) — Python 3.11 / 3.12 / 3.13 matrix,
  ruff lint stage, coverage gauge, Cobertura report artefact, headless
  Qt deps installed for `test_gui_dialogs`.

### Changed

- **`prg/__init__.py`** docstring — replaced the obsolete "Public API
  (to be defined)" placeholder with three concrete quickstart examples
  (copula, PMC, CLI).

### Added

- **`CopulaVirt.inv_h(w, u)`** — Rosenblatt inverse step (used by `simulate`).
  Default uses Brent's method; `CopulaGaussian` and `CopulaClayton` override
  with closed-form expressions (~50× faster).
- **`[project.scripts] pmc`** — installs a `pmc` console script
  (`pmc simulate ...`, `pmc classify ...`, `pmc estimate ...`, `pmc gui`).
- **`[project.optional-dependencies] gui`** — `pip install copulasformm[gui]`
  pulls PyQt6.
- **`[tool.coverage]` config** — `pytest --cov` runs out of the box.
- **`prg/copulas/_fit.py`** — extracted from `_base.py`: `FitResult`,
  `GoFResult`, `_eval_log_likelihood`, `_empirical_copula`, `_cvm_statistic`,
  `_empirical_tail_dep`. `_base.py` re-exports for backward compatibility.
- **`prg/pmc/gui/dialogs.py`** — extracted `_MarginDialog` and `_CopulaDialog`
  from `main_window.py`. Now unit-testable in isolation.
- **`prg/pmc/gui/tabs.py`** — extracted the four parameter tabs (`_PriorTab`,
  `_MarginTab`, `_CopulaTab`, `_IceTab`) from `main_window.py`.
- **`prg/pmc/gui/worker.py`** — extracted `_Worker` (background `QThread`).
- **`prg/copulas/_bivariate_fit.py`** — extracted `BivariateFitResult`,
  `BivariateBootstrapCI`, `_empirical_joint_cdf` and `_joint_cvm_statistic`
  from `bivariate.py`. The runtime cycle is broken by lazy imports of
  `BivariateLaw` inside the bootstrap / cv methods.
- **`CopulaFrank.inv_h`** — closed-form override (Aas et al. 2009 derivation):
    v = -1/θ · log(1 + w(e^{-θ}-1) / (e^{-θu} - w(e^{-θu}-1)))
- **`prg/tests/test_gui_dialogs.py`** — 7 unit tests for `_MarginDialog` and
  `_CopulaDialog` (round-trip, defaults, parameter parsing, full copula list).
  Skipped gracefully when PyQt6 is not installed.
- **`prg/tests/test_pdf_array.py`** — 36 new tests verifying that every
  copula's `pdf_array` matches `[pdf([u,v]) for u,v in uv]` and that
  `inv_h` is a true inverse of `conditional_cdf` for Gaussian and Clayton.

### Changed

- **`pyproject.toml`** — multiple correctness fixes:
  - `requires-python = ">=3.11"` (was `>=3.10`; we use `tomllib` from stdlib).
  - Removed `pandas`, `rich` from dependencies (never imported).
  - Added `statsmodels>=0.14` (used by 5 copulas, was undeclared → install
    on a clean env was broken).
  - Classifier list updated to reflect 3.11 / 3.12 / 3.13.
- **`prg.pmc.simulate`** — uses the new `cop.inv_h(w, u)` instead of an
  in-module `brentq` call. The two `simulate(N=10_000)` tests went from
  8.1 s → 1.2 s each; total test suite **22.78 s → 7.25 s**.
- **`prg.pmc.ice._joint_posteriors`** — vectorised via numpy broadcasting;
  removed the `for n in range(N-1)` loop.
- **`BIGGER_SIZE` → `FONT_SIZE`** across the package (the legacy alias was
  misleading; the rename clarifies intent).
- **Logger placement** — moved `logger = logging.getLogger(__name__)` after
  the imports in `prg.pmc.gui.main_window` (was the only file with it before
  imports).
- **`scripts/update_readme_structure.sh`** — moved from the repo root to
  `scripts/` and the in-script usage doc updated.

### Removed

- **`prg.pmc.ice.ice_from_files`** — dead helper (61 LOC), never called from
  anywhere. The CLI uses `cmd_estimate` directly.
- **`prg/tests/__init__.py`** and **`prg/tests/conftest.py`** (empty).
- **`prg.pmc.simulate._inv_h`** — superseded by `CopulaVirt.inv_h`.

### Added (earlier in this cycle, kept here for completeness)

- **`CopulaVirt.pdf_array(uv)` / `logpdf_array(uv)`** — vectorised PDF / log-PDF
  evaluation on an `(M, 2)` array of pseudo-observations. Default fast path
  uses `self._model.pdf` (statsmodels) when available; falls back to a Python
  loop otherwise.
- **`CopulaJoe.pdf_array`** — vectorised closed-form override (Joe has no
  statsmodels backend).
- **`PMCModel` public read API** — `raw`, `path`, `ice_config()`,
  `margin_blocks()`, `copula_blocks()`. All return deep copies — safe to
  mutate. External callers (ICE, GUI) no longer reach into `_raw`.
- **`prg.configure_logging`** re-exported at the top-level package.
- **`prg/tests/test_pmc.py`** — 39 new tests covering model loading,
  prior consistency, weight() formula, save/load round-trip, simulate()
  empirical π, forward/backward invariants, classify() error rate,
  K=3 label permutations, and ICE convergence + τ recovery.

### Changed

- **`prg.pmc.inference.precompute_weights`** — vectorised:
  - HMC-IN/IN2/PMC-IN: full numpy broadcast (no Python loop on N).
  - HMC-DN/PMC: the `(N-1)` Python loop on copula PDFs is replaced by a
    single `cop.pdf_array(uv)` call per `(i, j)` pair.
  - **Speedup**: classify on PMC/HMC-DN, N=5000 → ≈55× (1.2 s → 0.02 s).
- **`prg.pmc.ice._neg_wll` and `_weighted_log_likelihood`** — use
  `cop.logpdf_array(uv)` instead of a Python list comprehension over scalars.
  - **Speedup**: ICE 5 iterations on PMC, N=2000 → ≈130× (27 s → 0.2 s).
- **`prg.pmc.inference.error_rate`** — handles arbitrary K via the
  Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) on the
  confusion matrix; returns the minimum error over all label permutations.
- **`prg.pmc.model.PMCModel.save`** — uses `tomli_w` instead of a custom
  TOML serialiser (~50 LOC removed).
- **`PMCModel.from_dict(raw)`** now deep-copies `raw`, eliminating the
  aliasing pitfall where mutating the source dict silently affected the model.

- **`PMCModel._parse`** — exhaustive `[[margins]]` and `[[copulas]]`
  validation: detects duplicate `(i, j)` keys, missing pairs, and missing
  required fields (`dist`, `i`, `j`). 4 new tests cover these paths.
- **`prg.pmc.ice.ice()`** — distinguishes log-likelihood **regression** from
  convergence: emits a `WARNING` on each decreasing step and stops early
  after `patience` (default 3) consecutive regressions.
- **`prg.copulas._base.CopulaEnum.klass`** — cached lazy property that
  returns the implementation class. Replaces repeated
  `importlib.import_module` calls in `_build_copula` and ICE candidate
  resolution.
- **GUI** — added a **Seed** field to the model header. Empty = random;
  any integer = reproducible simulation. Parsing is tolerant: a non-integer
  value logs a warning and falls back to ``None``.
- **`prg.pmc.gui._MarginDialog` / `_CopulaDialog`** — now inherit directly
  from `QDialog` instead of wrapping a contained `QDialog` in a `QWidget`.
- **CLI** — extracted `_read_observations(path, ref_col)` helper; removed
  duplicated CSV-reading code from `cmd_classify` and `cmd_estimate`.

- **Vectorised `pdf_array` overrides** for every copula without a
  `statsmodels` backend: `BB1`, `AMH`, `A12`, `A14`, `FGM`, `Plackett`,
  `CubSec`, and the three `Survival*` wrappers. Bench against the scalar
  loop on N=2000 random pseudo-observations: speedups from **14× to 100×**;
  numerical match within 1 ulp on every family.
- **`prg/tools/tools.py`** — added `MIN_POSITIVE = sys.float_info.min`
  (smallest strictly-positive normal float, used as a safe non-zero floor
  in normalisation). Previously `1e-300` was hardcoded in five places.
- **`prg.pmc.inference.forward`** — when α₁ has zero density under every
  state (e.g. Y[0] outside every margin's support), the function now
  raises `IncompatibleObservationError` instead of silently flooring
  `C₁` to `1e-300` and producing a meaningless log-likelihood. Test
  added.
- **`prg.pmc.model._stationary_distribution`** — replaces the
  `np.linalg.eig`-based extraction of π with **power iteration**. The
  previous approach could pick the wrong eigenvector for nearly-reducible
  chains (multiple eigenvalues close to 1). Test added on a near-reducible
  3-state chain.
- **`prg.pmc.ice._fit_copula_params`** — multi-parameter copula fitting
  via L-BFGS-B. BB1 (`tau_k`, `delta`) and Student (`tau_k`, `df`) now
  jointly maximise the weighted log-likelihood instead of using the
  hardcoded defaults `delta = 1.5` / `df = 4.0`.

### Removed

- **`prg/logging_config.py`** — superseded by `prg/pmc/logging_setup.py`
  (unused since 0.4.0).

### Dependencies

- Added **`tomli_w>=1.0`** (TOML serialisation).

---

## [0.4.0] - 2026-05-04

### Added

- **`prg/pmc/`** — full Pairwise Markov Chain (PMC) package with 5 model variants:
  - **`HMC-IN`** — Hidden Markov Chain, one marginal per class (classical HMM).
  - **`HMC-IN2`** — HMC with pair-indexed marginals f_{ij}.
  - **`HMC-DN`** — HMC with copula-based temporal dependence.
  - **`PMC-IN`** — Pairwise Markov Chain, independent observations.
  - **`PMC`** — Full PMC with copula-based transitions.

- **`PMCModel`** (`prg/pmc/model.py`) — loads/validates/saves models from TOML.
  - Supports all 5 variants; validates K, prior (A or p), K² margins, K² copulas.
  - `prior_p`, `transition_A`, `stationary_pi` properties.
  - `pdf(i,j,y)`, `cdf(i,j,y)`, `ppf(i,j,q)` — margin access.
  - `copula(i,j)` — returns `CopulaVirt` instance.
  - `weight(i,j,y_n,y_{n+1})` — unified transition weight for all 5 variants.
  - Custom TOML writer (no `tomli_w` dependency).

- **`simulate`** (`prg/pmc/simulate.py`) — sequence generator for all 5 variants.
  - Latent chain sampled from row-stochastic A; observations via Rosenblatt h-inversion for copula variants.
  - `simulate(model, N=None, seed=None) → (X, Y)`.

- **`classify`** / **`forward`** / **`backward`** / **`smooth`** / **`mpm`** (`prg/pmc/inference.py`) — Baum-Welch forward-backward with Devijver normalization.
  - Initialization: α_1(j) = Σ_i p[i,j]·f_{ji}(y_1) — works for all 5 variants.
  - `classify(model, Y) → (X_hat, gamma, log_lik)`.
  - `error_rate(X_true, X_hat)` — handles K=2 label permutation.

- **`ice`** (`prg/pmc/ice.py`) — Iterative Conditional Estimation (ICE) for unsupervised parameter fitting.
  - E-step: forward-backward → marginal γ and joint ξ posteriors.
  - M-step: prior update, copula selection by weighted log-likelihood, τ by weighted MLE.
  - Optional margin re-estimation (Gaussian: weighted mean/std).
  - Convergence by relative log-likelihood change.

- **CLI** (`prg/pmc/cli.py`, `prg/pmc/__main__.py`):
  - `python -m prg.pmc simulate --model M.toml --N 5000 --seed 42 --out seq.csv`
  - `python -m prg.pmc classify --model M.toml --data seq.csv --out cls.csv`
  - `python -m prg.pmc estimate --model INIT.toml --data seq.csv --out fitted.toml`
  - `python -m prg.pmc gui [M.toml]`

- **PyQt6 GUI** (`prg/pmc/gui/`):
  - Single window with three action buttons: **Simulate**, **Classify**, **Estimate**.
  - Tabbed model editor: Prior (K×K editable matrix), Margins (K×K cells), Copulas (K×K cells), ICE Config.
  - Double-click cell dialogs for per-pair margin and copula editing.
  - Matplotlib result panel: scatter + histogram + pair-scatter for simulation; posterior + error map for classification; log-lik convergence for ICE.
  - Background QThread worker keeps GUI responsive during long computations.
  - File menu: Open / Save / Save As TOML; Load data CSV.

- **5 example TOML models** (`prg/pmc/models/`):
  - `hmc_in_gauss_k2.toml`, `hmc_in2_gauss_k2.toml`, `hmc_dn_gauss_k2.toml`
  - `pmc_in_gauss_k2.toml`, `pmc_gauss_k2.toml`

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

[Unreleased]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.4.0...HEAD
[0.4.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.5...v0.4.0
[0.3.5]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.4...v0.3.5
[0.3.4]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.3...v0.3.4
[0.3.3]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.2...v0.3.3
[0.3.2]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.1...v0.3.2
[0.3.1]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.3.0...v0.3.1
[0.3.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.2.0...v0.3.0
[0.2.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.1.0...v0.2.0
[0.1.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/compare/v0.0.0...v0.1.0
[0.0.0]: https://gitlab.ec-lyon.fr/sderrode/copulasformm/-/tags/v0.0.0
