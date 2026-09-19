"""
main_window.py — QMainWindow for the PMC/HMC GUI.

Layout
------
┌─────────────────────────────────────────────────────────────────────┐
│  Menu bar  (File → Open / Recent / Save / Save data / Export plot;  │
│             Edit → Reset; Help)                                     │
├──────────────────────────────┬──────────────────────────────────────┤
│  LEFT PANEL                  │  RIGHT PANEL                         │
│  ┌─────────────────────────┐ │  [View ▾] (selector)                 │
│  │  Model info header      │ │  ┌────────────────────────────────┐  │
│  ├─────────────────────────┤ │  │  matplotlib canvas             │  │
│  │  QTabWidget             │ │  │  (one of the 15 registered     │  │
│  │  ┣ Prior                │ │  │   views: Simulation, Classif., │  │
│  │  ┣ Margins              │ │  │   GoF, ICE-A…ICE-K)            │  │
│  │  ┣ Copulas              │ │  └────────────────────────────────┘  │
│  │  ┗ ICE Config           │ │  [▶ Play] [—slider—] iter X/T  ★    │
│  ├─────────────────────────┤ │  ┌────────────────────────────────┐  │
│  │ [Simulate] [Classify]   │ │  │  Log panel + status bar        │  │
│  │ [Estimate] [GoF test]   │ │  └────────────────────────────────┘  │
│  └─────────────────────────┘ │                                       │
└──────────────────────────────┴──────────────────────────────────────┘

★ The playback row is shown only when the *Animated playback* view is
  active.

Architecture — visualisation
----------------------------
The right-hand canvas is driven by a single :attr:`_VIEW_DISPATCH` table
mapping a view name (``"Simulation"``, ``"ICE — Dashboard"``, …) to an
``(available, render)`` pair of ``self``-taking callables: ``available``
gates the view in the combobox (:meth:`_is_view_available`) and ``render``
repaints it from instance state. The actual drawing lives in
:mod:`pmcprg.pmc.gui.views` as headless functions of a ``Figure``; the render
callable is a thin delegator that clears the canvas, calls the ``views.*``
function, and draws.

Each action handler (``_on_sim_done``, ``_on_cls_done``, ``_on_gof_done``,
``_on_est_done``) stores the data it produced on ``self`` and calls
:meth:`_switch_view`, which:

1. Looks up the ``(available, render)`` pair in :attr:`_VIEW_DISPATCH`.
2. Calls :meth:`_begin_figure` (clears the canvas) inside the delegator.
3. Invokes ``render(self)``.
4. On success, calls :meth:`_finalize_plot` (toggles Export-plot on).
   On exception, calls :meth:`_render_error` which paints a friendly
   error message on the canvas + log panel.

Adding a new view is therefore: declare its name, add its data state
attribute, add one ``(available, render)`` entry to :attr:`_VIEW_DISPATCH`,
and write the ``views.plot_<name>`` renderer — registered in exactly one place.

Thread safety
-------------
Heavy actions (Simulate / Classify / Estimate / GoF) run in a QThread
worker so the GUI stays responsive. ICE and GoF forward a per-iteration
progress signal that drives the determinate progress bar.

Audit fixes
-----------
See ``CHANGELOG.md`` for the full log; recent passes added ICE diagnostic
views (``A``–``K``), a ``View`` selector, animated playback, error UI in
the canvas, dedup of ``EXTRA_PARAM_BOUNDS``, ``_BlockGridTab`` base class
for the Margins/Copulas tabs, ``IceResult`` dataclass packaging for the
worker → GUI hand-off, and centralised plot-lifecycle helpers
(``_begin_figure`` / ``_finalize_plot``).
"""

import csv
import logging
import re
import traceback
from collections.abc import Callable
from pathlib import Path

# Qt setup *before* any pyplot import (incl. the one inside pmcprg.pmc.gui.views).
import matplotlib
matplotlib.use("QtAgg")
import numpy as np                                           # noqa: E402

from PyQt6.QtCore    import QSettings, Qt, QTimer            # noqa: E402
from PyQt6.QtGui     import QCloseEvent                       # noqa: E402
from PyQt6.QtWidgets import (                                # noqa: E402
    QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure                              # noqa: E402

from pmcprg.plot_style      import apply_gui_compact_style, apply_style  # noqa: E402

# The GUI is an application: it restyles the session on purpose. The package
# look first (white facecolor, 150 dpi, 12 pt — ``import pmcprg.copulas`` no
# longer applies it), then the compact, math-friendly rcParams (smaller
# titles/ticks, LaTeX mathtext, constrained_layout) that override it, for
# every Figure created in this module.
apply_style()
apply_gui_compact_style()

from pmcprg                 import __version__ as _PKG_VERSION    # noqa: E402
from pmcprg.missing.cells   import parse_cell                     # noqa: E402
from pmcprg.pmc.gui         import views                          # noqa: E402
from pmcprg.pmc.gui._margins import (                             # noqa: E402
    is_pair, margin_cdfs, state_law, state_law_name,
)
from pmcprg.pmc.gui.tabs    import PriorTabError, _CopulaTab, _IceTab, _MarginTab, _PriorTab  # noqa: E402
from pmcprg.pmc.gui.worker  import _Worker                        # noqa: E402
from pmcprg.pmc             import IceTrace, SemResult            # noqa: E402
from pmcprg.pmc.inference   import error_rate                     # noqa: E402
from pmcprg.pmc.missingness import describe_mechanism as missingness_summary  # noqa: E402
from pmcprg.pmc.model       import PMCModel                       # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# View identifiers — keys for the View QComboBox and ``_switch_view`` dispatch
# ---------------------------------------------------------------------------

_VIEW_NONE             = "—"
# GoF bootstrap replicates (audit S-6). Each one simulates a whole series
# under the fitted model and re-runs forward-backward, so this is the single
# knob that sets the cost of the test.
_GOF_BOOTSTRAP_B       = 200
# GoF statistic. "kendall" is a Cramér-von Mises distance on the **Kendall
# process** — the law of C(U, V) rather than C itself — and it is the shipped
# default because it is the only candidate measured to have power here. Over
# 200 datasets per configuration at N = 1500 (report/gof_power_kendall.py),
# swapping "cvm" for "kendall" takes rejection of a wrong family from 0.115 to
# **0.950** against Clayton and from 0.060 to **0.305** against GH — the latter
# being an alternative that nothing, not even the true-model posterior, could
# see. Level is held (0.070 [0.042, 0.114] on the diagonal, 0.040 on the
# independent pairs, nominal 0.05), and it also fixes the misattribution: under
# GH the diagonal now rejects more than the off-diagonal (0.305 vs 0.135),
# where "cvm" had it backwards (0.060 vs 0.120). "cvm" is kept selectable
# because it is the statistic the ``cvm`` *selection criterion* uses, so the
# two can still be compared on equal terms.
_GOF_STATISTIC         = "kendall"
_GOF_KENDALL_GRID_N    = 99
_GOF_KENDALL_MC        = 20_000
_VIEW_SIMULATION       = "Simulation"
_VIEW_CLASSIFICATION   = "Classification"
_VIEW_GOF_HEATMAP      = "GoF — p-value heatmap"
# G-7: pmcprg.pmc.inference.sample_posterior (the exact FFBS the SEM M-step
# already uses) had no entry point, so nothing on screen distinguished a
# segmentation the posterior is sure of from one it is guessing at.
_VIEW_CLS_POSTERIOR    = "Classification — posterior draws"
_VIEW_TAU_CI           = "ICE — τ with bootstrap CI"
_VIEW_MARGIN_KS        = "Margins — adequacy (KS)"
_MARGIN_KS_B           = 200
# Bootstrap replicates for the τ intervals. B=200 over K² pairs is ~10 s,
# which is why this is a worker action and not a render.
_TAU_CI_B              = 200
_VIEW_IMAGE_RAW        = "Image — input"
_VIEW_IMAGE_SEG        = "Image — segmentation"
# ICE-trace views (only available after a successful estimate)
_ICE_VIEW_LOGLIK       = "ICE — Log-likelihood"
_ICE_VIEW_TAU          = "ICE — τ trajectories"           # A
_ICE_VIEW_FAMILY       = "ICE — Family selection ribbon"  # B
_ICE_VIEW_PSEUDOS      = "ICE — Pseudo-obs & contours"    # C
_ICE_VIEW_MULTISTART   = "ICE — Multistart comparison"    # D
_ICE_VIEW_PRIOR        = "ICE — Joint-prior evolution"    # E
_ICE_VIEW_MARGINS      = "ICE — Margin params evolution"  # F
_ICE_VIEW_DASHBOARD    = "ICE — Dashboard"                # G
_ICE_VIEW_PP           = "ICE — PP/QQ plot of pseudo-obs" # H
_ICE_VIEW_TAIL         = "ICE — Tail dependence (λL, λU)" # I
_ICE_VIEW_GAMMA        = "ICE — γ before vs after"        # J
_ICE_VIEW_PLAYBACK     = "ICE — Animated playback"        # K
_ICE_VIEW_MARGIN_FAMILY = "ICE — Margin family ribbon"    # L (GICE / SP-2016)
_ICE_VIEW_PARAM_COMPARE = "ICE — Parameters: true vs fitted"  # M

# Display order in the combobox (None entries denote separators visually).
_ALL_VIEWS = [
    _VIEW_SIMULATION, _VIEW_CLASSIFICATION, _VIEW_CLS_POSTERIOR,
    _VIEW_TAU_CI, _VIEW_MARGIN_KS,
    _VIEW_IMAGE_RAW, _VIEW_IMAGE_SEG,
    _VIEW_GOF_HEATMAP,
    _ICE_VIEW_DASHBOARD, _ICE_VIEW_PARAM_COMPARE,
    _ICE_VIEW_LOGLIK, _ICE_VIEW_TAU, _ICE_VIEW_FAMILY,
    _ICE_VIEW_MARGIN_FAMILY,
    _ICE_VIEW_PSEUDOS, _ICE_VIEW_MULTISTART, _ICE_VIEW_PRIOR,
    _ICE_VIEW_MARGINS, _ICE_VIEW_PP, _ICE_VIEW_TAIL, _ICE_VIEW_GAMMA,
    _ICE_VIEW_PLAYBACK,
]

# Views that can scrub through iterations via a slider. The "Animated playback"
# view itself shows the slider + play button explicitly; for the others, the
# slider remains hidden (the static plot is rendered).
_PLAYBACK_VIEW = _ICE_VIEW_PLAYBACK


# Persistent settings — used for the Recent-files submenu and any future
# GUI-level preferences. The "organization" + "application" identifiers
# determine where Qt stores the data (per-user, OS-dependent).
_QS_ORG = "awesomePMC"
_QS_APP = "pmc-gui"
_QS_RECENT_KEY = "recent_files"
_RECENT_MAX = 5


# ---------------------------------------------------------------------------
# Matplotlib canvas widget
# ---------------------------------------------------------------------------

class _Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        # Bumped from (7, 5) so multi-panel figures (ICE dashboard,
        # K×K small multiples) have room to breathe. constrained_layout
        # is enabled at the rcParam level by ``apply_gui_compact_style``.
        self.fig = Figure(figsize=(9.5, 6.5))
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Floor on the canvas size so users can't shrink the splitter to
        # the point where titles overlap.
        self.setMinimumSize(640, 460)

    def clear(self):
        self.fig.clear()
        self.draw()


# ---------------------------------------------------------------------------
# PMC Main Window
# ---------------------------------------------------------------------------

class PMCMainWindow(QMainWindow):

    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.setWindowTitle("PMC / HMC Model")
        self.resize(1300, 780)

        self._model: PMCModel | None = None
        self._model_path: Path | None = None
        self._last_X: np.ndarray | None = None
        self._last_Y: np.ndarray | None = None
        self._worker: _Worker | None = None
        self._dirty: bool = False
        self._has_plot: bool = False

        # Per-view state — populated by the corresponding action handlers,
        # consumed by :meth:`_render_view` so the user can re-render any
        # past result by picking it in the View selector.
        self._sim_state: tuple | None     = None      # (X, Y, mdl)
        self._cls_state: tuple | None     = None      # (Y, X_hat, gamma, X_ref, mdl)
        self._gof_state: list | None      = None      # [{...}, ...]
        self._tau_ci_state: list | None   = None      # [{i, j, tau, lo, hi}, …]
        self._margin_ks_state: list | None = None     # [{state, stat, p}, …]
        self._margin_ks_sample: tuple = (None, None)  # (Y, X_draw) redrawn
        # 2D image state — when an image is loaded, ``_last_Y`` holds its
        # linearised gilbert scan and ``_last_image`` keeps the 2D array
        # for the Image-* views. Reference labels (if loaded) are stored
        # 2D in ``_last_image_ref`` AND linearised into ``_last_X`` so the
        # existing ``error_rate`` path works unchanged.
        self._last_image: np.ndarray | None     = None        # (H, W)
        self._last_image_ref: np.ndarray | None = None        # (H, W) int labels
        self._ice_trace                   = None      # :class:`IceTrace`
        self._init_model: PMCModel | None = None      # initial model for ICE
        self._ice_Y:      np.ndarray | None = None    # observation seq fed to ICE
        # Internal flag to detect programmatic combobox changes.
        self._switching_view = False
        self._current_view: str = _VIEW_NONE
        # Animated-playback timer (created on demand; None when idle).
        self._playback_timer: QTimer | None = None

        self._settings = QSettings(_QS_ORG, _QS_APP)

        self._build_menu()
        self._build_ui()

        # Pre-load a model so the user lands on a workable state instead
        # of a fully-disabled UI. Order:
        #   1. The explicit ``model_path`` argument, if any.
        #   2. The fixture shipped with the package (PMC Gauss K=2).
        # The fallback path is silently skipped if the fixture is missing
        # (e.g. user runs from an installed wheel with stripped data).
        if model_path:
            self._load_model(model_path)
        else:
            try:
                from importlib.resources import files
                default = files("pmcprg.pmc") / "models" / "pmc_gauss_k2.toml"
                if default.is_file():
                    self._load_model(str(default), as_template=True)
            except Exception as exc:                          # pragma: no cover
                logger.debug("default-model auto-load skipped: %s", exc)

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        bar  = self.menuBar()
        file = bar.addMenu("&File")

        act_open = file.addAction("&Open model…")
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._on_open)

        # Recent files submenu — populated lazily on aboutToShow.
        self._mnu_recent = QMenu("Recent &files", self)
        self._mnu_recent.aboutToShow.connect(self._refresh_recent_menu)
        file.addMenu(self._mnu_recent)

        file.addSeparator()

        act_save = file.addAction("&Save")
        act_save.setShortcut("Ctrl+S")
        # ``triggered`` emits the action's ``checked`` bool, which would
        # arrive as _on_save's new ``path`` argument.
        act_save.triggered.connect(lambda: self._on_save())

        act_saveas = file.addAction("Save &As…")
        act_saveas.setShortcut("Ctrl+Shift+S")
        act_saveas.triggered.connect(self._on_save_as)

        file.addSeparator()

        act_load_data = file.addAction("Load &data (CSV)…")
        act_load_data.triggered.connect(self._on_load_data)

        act_load_image = file.addAction("Load &image…")
        act_load_image.setToolTip(
            "Load a grayscale image (PNG/JPG/BMP/TIFF). The image is "
            "linearised along the Generalized Hilbert (gilbert) path and "
            "fed to Classify / Estimate as a 1D signal; results can then "
            "be viewed as 2D segmentation maps."
        )
        act_load_image.triggered.connect(self._on_load_image)

        act_load_image_ref = file.addAction("Load image &reference labels…")
        act_load_image_ref.setToolTip(
            "Optional reference label image (paletted PNG or grayscale) — "
            "used to compute the classification error rate."
        )
        act_load_image_ref.triggered.connect(self._on_load_image_ref)

        act_save_data = file.addAction("Sa&ve data (CSV)…")
        act_save_data.setToolTip("Save the last simulated/loaded (X, Y) sequence.")
        act_save_data.triggered.connect(self._on_save_data)
        self._act_save_data = act_save_data

        self._act_save_seg = file.addAction("Save se&gmentation (PNG)…")
        self._act_save_seg.setToolTip(
            "Save the current 2D segmentation map (folded back from the "
            "1D Classify result) as a paletted PNG."
        )
        self._act_save_seg.triggered.connect(self._on_save_segmentation)
        self._act_save_seg.setEnabled(False)

        self._act_export_results = file.addAction("Export &results…")
        self._act_export_results.setToolTip(
            "Save the estimation trace, the classification error rate and "
            "the GoF results — CSV for the per-iteration table, JSON for "
            "everything."
        )
        self._act_export_results.triggered.connect(self._on_export_results)

        self._act_export_plot = file.addAction("Export &plot…")
        self._act_export_plot.setToolTip(
            "Save the current matplotlib canvas as PNG/PDF/SVG."
        )
        self._act_export_plot.triggered.connect(self._on_export_plot)
        self._act_export_plot.setEnabled(False)

        file.addSeparator()
        file.addAction("&Quit").triggered.connect(self.close)

        edit = bar.addMenu("&Edit")
        act_reset = edit.addAction("&Reset session")
        act_reset.setShortcut("Ctrl+R")
        act_reset.setToolTip("Clear loaded data, last results and the plot canvas.")
        act_reset.triggered.connect(self._on_reset)

        # G-7/G-8: there was no analysis menu at all, and several diagnostics
        # in the package had nowhere to be invoked from.
        analysis = bar.addMenu("&Analysis")
        self._act_tau_ci = analysis.addAction("τ confidence intervals…")
        self._act_tau_ci.setToolTip(
            "ξ-weighted bootstrap interval on each pair's τ̂, drawn at the "
            "pair's effective sample size."
        )
        self._act_tau_ci.triggered.connect(self._on_tau_ci)

        self._act_margin_ks = analysis.addAction("Margin adequacy (KS)…")
        self._act_margin_ks.setToolTip(
            "Multivariate KS test of each fitted margin against a posterior "
            "draw of the states, with a bootstrapped critical value."
        )
        self._act_margin_ks.triggered.connect(self._on_margin_ks)
        self._analysis_actions = [self._act_tau_ci, self._act_margin_ks]

        # P6: unlike τ CI / margin KS, this applies to every variant — the
        # mask is evidence on the states whether or not the model uses
        # copulas — so it is enabled separately, not through
        # ``_analysis_actions`` (which requires ``uses_copula``).
        self._act_missingness_lr = analysis.addAction("Missingness LR test…")
        self._act_missingness_lr.setToolTip(
            "Likelihood-ratio test of state-dependent missingness "
            "(pmcprg.pmc.missingness_lr_test): 'state' (π_i per state) "
            "against a common (state-independent) rate. Needs data with "
            "both missing and observed rows."
        )
        self._act_missingness_lr.triggered.connect(self._on_missingness_lr_test)

        hlp = bar.addMenu("&Help")
        hlp.addAction("About").triggered.connect(self._on_about)

        # S-1: every entry point that REPLACES model, data or result state.
        # ``_start_worker`` greyed out the four action buttons but left these
        # live, so loading a CSV mid-run made ``_on_cls_done`` compare the
        # *new* ``_last_X`` against the *old* worker's ``X_hat`` — an
        # exception escaping a Qt slot, i.e. SIGABRT. Toggled by ``_set_busy``.
        self._mutating_actions = [
            act_open, self._mnu_recent.menuAction(), act_load_data,
            act_load_image, act_load_image_ref, act_reset,
        ]

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def _recent_files(self) -> list[str]:
        raw = self._settings.value(_QS_RECENT_KEY, [])
        if isinstance(raw, str):
            return [raw] if raw else []
        return [str(p) for p in (raw or []) if p]

    def _push_recent(self, path: str | Path):
        path  = str(Path(path).resolve())
        files = [p for p in self._recent_files() if p != path]
        files.insert(0, path)
        self._settings.setValue(_QS_RECENT_KEY, files[:_RECENT_MAX])

    def _refresh_recent_menu(self):
        self._mnu_recent.clear()
        files = self._recent_files()
        if not files:
            act = self._mnu_recent.addAction("(no recent files)")
            act.setEnabled(False)
            return
        for path in files:
            act = self._mnu_recent.addAction(path)
            act.triggered.connect(lambda _checked=False, p=path: self._load_model(p))
        self._mnu_recent.addSeparator()
        clear = self._mnu_recent.addAction("Clear list")
        clear.triggered.connect(lambda: self._settings.setValue(_QS_RECENT_KEY, []))

    # ------------------------------------------------------------------
    # Main UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left: model editor ──────────────────────────────────────
        self._left = QWidget()
        left_lay   = QVBoxLayout(self._left)
        left_lay.setSpacing(4)

        # Header info
        info_box = QGroupBox("Model")
        info_lay = QFormLayout(info_box)
        self._lbl_name    = QLineEdit("—")
        self._lbl_variant = QLabel("—")
        self._lbl_K       = QLabel("—")
        self._spn_N       = QSpinBox()
        self._spn_N.setRange(10, 1_000_000)
        self._spn_N.setValue(5000)

        # Seed: blank = non-reproducible (None); any integer = reproducible.
        self._txt_seed = QLineEdit("")
        self._txt_seed.setPlaceholderText("(blank = random)")
        self._txt_seed.setMaximumWidth(120)

        info_lay.addRow("Name:",    self._lbl_name)
        info_lay.addRow("Variant:", self._lbl_variant)
        info_lay.addRow("K:",       self._lbl_K)
        info_lay.addRow("N:",       self._spn_N)
        info_lay.addRow("Seed:",    self._txt_seed)
        left_lay.addWidget(info_box)

        # Connect dirty-tracking to top-level fields.
        self._lbl_name.textChanged.connect(self._mark_dirty)
        self._spn_N.valueChanged.connect(self._mark_dirty)

        # Parameter tabs
        self._tabs = QTabWidget()
        self._tab_prior   = _PriorTab()
        self._tab_margins = _MarginTab()
        self._tab_copulas = _CopulaTab()
        self._tab_ice     = _IceTab()
        self._tabs.addTab(self._tab_prior,   "Prior")
        self._tabs.addTab(self._tab_margins, "Margins")
        self._tabs.addTab(self._tab_copulas, "Copulas")
        self._tabs.addTab(self._tab_ice,     "ICE Config")
        left_lay.addWidget(self._tabs, stretch=1)

        # Wire each tab's "changed" signal to the dirty tracker.
        for tab in (self._tab_prior, self._tab_margins,
                    self._tab_copulas, self._tab_ice):
            tab.changed.connect(self._mark_dirty)
        # Symmetrize feedback → log panel.
        self._tab_prior.symmetrized.connect(self._on_symmetrized)

        # Action buttons (2×2 grid: Simulate, Classify / Estimate, GoF)
        btn_box = QGridLayout()
        self._btn_sim = QPushButton("▶ Simulate")
        self._btn_cls = QPushButton("◈ Classify")
        self._btn_est = QPushButton("⟳ Estimate")
        self._btn_gof = QPushButton("✓ GoF test")
        self._btn_gof.setToolTip(
            "Cramér-von Mises goodness-of-fit on each pair-copula "
            "using the loaded data."
        )
        for btn in (self._btn_sim, self._btn_cls, self._btn_est, self._btn_gof):
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setEnabled(False)
        btn_box.addWidget(self._btn_sim, 0, 0)
        btn_box.addWidget(self._btn_cls, 0, 1)
        btn_box.addWidget(self._btn_est, 1, 0)
        btn_box.addWidget(self._btn_gof, 1, 1)
        left_lay.addLayout(btn_box)

        self._btn_sim.clicked.connect(self._on_simulate)
        self._btn_cls.clicked.connect(self._on_classify)
        self._btn_est.clicked.connect(self._on_estimate)
        self._btn_gof.clicked.connect(self._on_gof_test)

        # Progress bar — switches between determinate (ICE) and
        # indeterminate (other tasks) at runtime.
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)    # indeterminate
        self._progress.setVisible(False)
        left_lay.addWidget(self._progress)

        splitter.addWidget(self._left)

        # ── Right: results ──────────────────────────────────────────
        right    = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setSpacing(4)

        # View selector — a combobox that lets the user re-render any
        # past result (Simulation, Classification, GoF, or any ICE-trace
        # view) without re-running the underlying computation.
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("View:"))
        self._cmb_view = QComboBox()
        self._cmb_view.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._cmb_view.currentTextChanged.connect(self._on_view_changed)
        view_row.addWidget(self._cmb_view, stretch=1)

        # Inline Export button — duplicates ``File → Export plot…`` so
        # exporting the visible canvas is one click away. Enabled state
        # tracks ``_has_plot`` (see :meth:`_set_buttons_enabled` and
        # :meth:`_finalize_plot`).
        self._btn_export = QPushButton("Export…")
        self._btn_export.setToolTip(
            "Save the current plot as PNG / PDF / SVG (same as File → Export plot…)."
        )
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self._on_export_plot)
        view_row.addWidget(self._btn_export)

        right_lay.addLayout(view_row)

        self._canvas = _Canvas()
        right_lay.addWidget(self._canvas, stretch=4)

        # Playback controls — only shown for the "Animated playback" view.
        # The slider scrubs through ICE iterations; the Play button toggles
        # auto-advance via a QTimer.
        self._playback_row = QHBoxLayout()
        self._btn_play = QPushButton("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.toggled.connect(self._on_play_toggled)
        self._sld_iter = QSlider(Qt.Orientation.Horizontal)
        self._sld_iter.setMinimum(0)
        self._sld_iter.setMaximum(0)
        self._sld_iter.valueChanged.connect(self._on_playback_slider)
        self._lbl_iter = QLabel("iter 0/0")
        self._lbl_iter.setMinimumWidth(80)
        self._playback_row.addWidget(self._btn_play)
        self._playback_row.addWidget(self._sld_iter, stretch=1)
        self._playback_row.addWidget(self._lbl_iter)
        # Wrap in a container widget so we can show/hide the whole row.
        self._playback_widget = QWidget()
        self._playback_widget.setLayout(self._playback_row)
        self._playback_widget.setVisible(False)
        right_lay.addWidget(self._playback_widget, stretch=0)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        self._log.setPlaceholderText("Results and messages …")
        right_lay.addWidget(self._log, stretch=0)

        # Initial view list (most options disabled until data is available).
        self._populate_view_selector()

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        # Status bar
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — open a TOML model to begin.")

        # NOTE — there used to be a second Ctrl+R binding here, a global
        # QShortcut added "so the GUI tests can trigger it without a menu
        # bar". No test ever used it, and it did two kinds of damage. It
        # collided with the menu action's own Ctrl+R, so Qt reported an
        # ambiguous overload and the shortcut did *nothing* in normal use;
        # and because a QShortcut is not a QAction it escaped ``_set_busy``,
        # so it fired precisely when Reset was supposed to be locked —
        # wiping the session under a running worker, which then wrote its
        # results back over the reset. Tests call ``_on_reset`` directly.

    # ------------------------------------------------------------------
    # Logging helper — appends to the panel and forces auto-scroll.
    # ------------------------------------------------------------------

    def _log_append(self, text: str):
        """Append a line to the log panel and keep the latest line visible."""
        self._log.append(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # View management
    # ------------------------------------------------------------------

    def _is_view_available(self, name: str) -> bool:
        """Return True if the view's underlying data is currently in memory.

        The per-view availability predicate lives *with* its renderer in
        :attr:`_VIEW_DISPATCH` (audit Q-11), so a view is registered in exactly
        one place — no parallel if/elif ladder to keep in lockstep.
        """
        spec = self._VIEW_DISPATCH.get(name)
        return spec is not None and spec[0](self)

    def _populate_view_selector(self):
        """Refresh the View QComboBox so it names what the canvas holds.

        We always rebuild the full list (small) so the order is stable; items
        whose data is not in memory are inserted but disabled.

        Audit S-4 — the selection used to be restored from the combobox's own
        previous text, or fall back to the first available view, with the
        change signal suppressed and nothing re-rendered. The selector could
        therefore name one view while the canvas showed another, or name a
        view over a blank canvas. The single invariant now is: **the
        combobox names** :attr:`_current_view`. When that is
        :data:`_VIEW_NONE` — nothing has been drawn, or the drawn view's data
        has just been invalidated — a transient, disabled "—" entry says so
        rather than borrowing the name of some other view.
        """
        # Suppress currentTextChanged while we populate.
        self._switching_view = True
        try:
            self._cmb_view.clear()
            for view in _ALL_VIEWS:
                self._cmb_view.addItem(view)
                idx = self._cmb_view.count() - 1
                # Disable items whose data is missing.
                item = self._cmb_view.model().item(idx)
                if item is not None and not self._is_view_available(view):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                    self._cmb_view.setItemData(
                        idx, "(no data — run the corresponding action)",
                        Qt.ItemDataRole.ToolTipRole,
                    )
            if (self._current_view in _ALL_VIEWS
                    and self._is_view_available(self._current_view)):
                self._cmb_view.setCurrentText(self._current_view)
            else:
                self._current_view = _VIEW_NONE
                self._cmb_view.insertItem(0, _VIEW_NONE)
                self._cmb_view.setItemData(
                    0, "Nothing displayed — pick a view, or run an action.",
                    Qt.ItemDataRole.ToolTipRole,
                )
                placeholder = self._cmb_view.model().item(0)
                if placeholder is not None:
                    placeholder.setFlags(
                        placeholder.flags() & ~Qt.ItemFlag.ItemIsEnabled
                    )
                self._cmb_view.setCurrentIndex(0)
        finally:
            self._switching_view = False

    def _on_view_changed(self, name: str):
        if self._switching_view or not name:
            return
        self._switch_view(name)

    def _drop_view_placeholder(self):
        """Retire the transient "nothing displayed" entry, if present."""
        if self._cmb_view.count() and self._cmb_view.itemText(0) == _VIEW_NONE:
            self._switching_view = True
            try:
                self._cmb_view.removeItem(0)
            finally:
                self._switching_view = False

    def _sync_view_combo_text(self):
        """Make the combobox name the view the canvas actually holds."""
        if self._cmb_view.currentText() == self._current_view:
            return
        self._switching_view = True
        try:
            self._cmb_view.setCurrentText(self._current_view)
        finally:
            self._switching_view = False

    def _switch_view(self, name: str):
        """Programmatic view switch: update combobox + render."""
        if not self._is_view_available(name):
            # S-4: bailing out silently left the combobox showing the refused
            # name over a canvas holding something else. Put the text back.
            self._sync_view_combo_text()
            return
        self._drop_view_placeholder()
        self._switching_view = True
        try:
            self._cmb_view.setCurrentText(name)
        finally:
            self._switching_view = False
        self._current_view = name
        # Stop any ongoing playback when switching away from playback view.
        if name != _PLAYBACK_VIEW and self._btn_play.isChecked():
            self._btn_play.setChecked(False)
        self._playback_widget.setVisible(name == _PLAYBACK_VIEW)
        self._render_view(name)

    # ------------------------------------------------------------------
    # Plot lifecycle helpers (called by every _plot_* method)
    # ------------------------------------------------------------------

    def _begin_figure(self) -> Figure:
        """Clear the canvas and return its Figure ready for new artists."""
        fig = self._canvas.fig
        fig.clear()
        return fig

    def _finalize_plot(self):
        """Mark the canvas as having drawn content (enables Export-plot, …).

        Centralised so individual plot methods don't have to repeat the
        ``self._has_plot = True; self._act_export_plot.setEnabled(True)``
        idiom. Called by :meth:`_render_view` after a successful dispatch.
        """
        self._has_plot = True
        self._act_export_plot.setEnabled(True)
        self._btn_export.setEnabled(True)

    def _render_error(self, name: str, exc: Exception):
        """Surface a render failure to the canvas + log panel.

        Replaces the previous behaviour where exceptions inside a render
        method were silently logged at DEBUG level — the user would see a
        blank canvas with no explanation.
        """
        logger.exception("Render failed for view %r", name)
        self._log_append(f"ERROR: render of {name!r} failed — {exc}")
        # PB-3: _has_plot survived from whatever rendered successfully last,
        # so Export-plot stayed enabled and wrote out the red "render failed"
        # card as if it were a figure. There is nothing here to export.
        self._has_plot = False
        self._act_export_plot.setEnabled(False)
        self._btn_export.setEnabled(False)
        fig = self._begin_figure()
        ax  = fig.add_subplot(1, 1, 1)
        ax.text(
            0.5, 0.5,
            f"Render failed for view {name!r}.\nSee log panel for details.",
            transform=ax.transAxes, ha="center", va="center",
            color="darkred", fontsize=11,
        )
        ax.set_axis_off()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Dispatch table — single source of truth for view → plot binding.
    #
    # Lambdas capture ``self`` lazily; the cost is negligible for the 17-entry
    # table built at __init__ time. Compared to a 17-branch if/elif chain the
    # dispatch reads as a registry: adding a view is a one-liner edit here +
    # one new ``_plot_*`` method.
    # ------------------------------------------------------------------

    def _view_provenance(self, name: str) -> str | None:
        """Which model produced the view named ``name``, if any.

        Audit S-14 — every action rebuilds its own model from the widgets and
        stores it in its own state, so two simultaneously available views can
        describe two different models with nothing saying so. The GoF heatmap
        is deliberately excluded: it is a pure function of its own snapshot,
        and naming the *current* model over it would be the misattribution
        this is meant to prevent.
        """
        if name == _VIEW_SIMULATION and self._sim_state is not None:
            return getattr(self._sim_state[2], "name", None)
        if name in (_VIEW_CLASSIFICATION, _VIEW_IMAGE_SEG) \
                and self._cls_state is not None:
            return getattr(self._cls_state[4], "name", None)
        if name.startswith("ICE") and self._model is not None:
            return getattr(self._model, "name", None)
        return None

    def _stamp_provenance(self, name: str):
        """Write the producing model's name in the figure's bottom corner."""
        who = self._view_provenance(name)
        if not who:
            return
        # A plain corner text collides with the axis labels as soon as the
        # canvas narrows; the backing box keeps it readable whatever it lands
        # on, which is the point of a provenance stamp.
        self._canvas.fig.text(
            0.995, 0.005, f"model: {who}", ha="right", va="bottom",
            fontsize=6, alpha=0.75,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.0),
        )

    def _render_view(self, name: str):
        """Dispatch to the matching renderer via the dispatch table."""
        spec = self._VIEW_DISPATCH.get(name)
        if spec is None:
            self._canvas.clear()
            return
        try:
            spec[1](self)              # spec = (available_predicate, render)
            self._stamp_provenance(name)
            self._canvas.draw_idle()
        except Exception as exc:
            self._render_error(name, exc)
            return
        # Playback view manages its own canvas updates via the slider; for
        # all others we mark the plot as drawn so Export-plot is enabled.
        if name != _PLAYBACK_VIEW:
            self._finalize_plot()

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def _setup_playback(self, trace: IceTrace):
        """Configure the slider for the active trace and render frame 0."""
        T = trace.n_iters if trace is not None else 0
        self._sld_iter.setMaximum(max(T - 1, 0))
        self._sld_iter.setValue(0)
        self._lbl_iter.setText(f"iter 0/{T - 1 if T else 0}")
        self._render_playback_frame(0)

    def _on_playback_slider(self, value: int):
        if self._current_view != _PLAYBACK_VIEW:
            return
        T = self._sld_iter.maximum() + 1
        self._lbl_iter.setText(f"iter {value}/{T - 1}")
        self._render_playback_frame(value)

    def _on_play_toggled(self, on: bool):
        if on and self._current_view == _PLAYBACK_VIEW:
            self._btn_play.setText("⏸ Pause")
            if self._playback_timer is None:
                self._playback_timer = QTimer(self)
                self._playback_timer.timeout.connect(self._playback_tick)
            self._playback_timer.start(500)             # 0.5 s per frame
        else:
            self._btn_play.setText("▶ Play")
            if self._playback_timer is not None:
                self._playback_timer.stop()

    def _playback_tick(self):
        v = self._sld_iter.value()
        if v >= self._sld_iter.maximum():
            self._btn_play.setChecked(False)            # auto-stop at end
            return
        self._sld_iter.setValue(v + 1)

    def _render_playback_frame(self, t: int):
        """Snapshot view at iteration ``t``: 2x2 grid (LL, τ, family, p)."""
        trace = self._ice_trace
        if trace is None or trace.n_iters == 0:
            return
        fig = self._begin_figure()
        # Top-left: LL up to t (highlight current point)
        ax1 = fig.add_subplot(2, 2, 1)
        lls = trace.log_liks
        nn  = list(range(len(lls)))
        ax1.plot(nn, lls, "o-", color="steelblue", lw=1.5, markersize=4, alpha=0.4)
        ax1.plot(nn[: t + 1], lls[: t + 1], "o-", color="steelblue",
                 lw=2, markersize=6)
        ax1.axvline(t, color="red", linestyle="--", alpha=0.5)
        ax1.set_title("Log-likelihood")
        ax1.grid(True, alpha=0.3)
        # Top-right: τ trajectories with vertical bar at t
        ax2 = fig.add_subplot(2, 2, 2)
        views.draw_tau_trajectories(ax2, trace, vline=t)
        # Bottom-left: family ribbon up to t
        ax3 = fig.add_subplot(2, 2, 3)
        views.draw_family_ribbon(ax3, trace, t_max=t)
        # Bottom-right: heatmap of p at iteration t
        ax4 = fig.add_subplot(2, 2, 4)
        if trace.p_history is not None and trace.p_history.size:
            P = trace.p_history[t]
            im = ax4.imshow(P, cmap="viridis", vmin=0.0, vmax=max(P.max(), 1e-9))
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    ax4.text(j, i, f"{P[i, j]:.3f}",
                             ha="center", va="center", fontsize=8,
                             color="white" if P[i, j] > 0.5 * P.max() else "black")
            fig.colorbar(im, ax=ax4, fraction=0.04, pad=0.04)
            ax4.set_title(rf"Joint prior $p_{{ij}}$  (iter {t})")
        fig.suptitle(f"ICE iteration {t} / {trace.n_iters - 1}")
        self._canvas.draw()
        # Playback bypasses the central _finalize_plot in _render_view.
        self._finalize_plot()

    # ------------------------------------------------------------------
    # Modified-state tracking
    # ------------------------------------------------------------------

    def _mark_dirty(self, *_):
        """Flag the model as edited; update the title bar."""
        if self._dirty:
            return
        self._dirty = True
        self._refresh_title()

    def _maybe_discard_changes(self, what: str = "continue") -> bool:
        """Ask before throwing unsaved edits away. ``True`` = go ahead.

        Audit S-5 — ``_load_model`` called :meth:`_mark_clean` without ever
        consulting ``_dirty``, and ``closeEvent`` checked only the worker, so
        every edit to the Prior / Margins / Copulas tables was lost without a
        word. *Discard* marks the model clean so the guard fires exactly once
        even when two doors lead to the same load.
        """
        if not self._dirty or self._model is None:
            return True
        ans = QMessageBox.question(
            self, "Unsaved changes",
            f"The model has unsaved changes.\n\nSave them before you {what}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if ans == QMessageBox.StandardButton.Cancel:
            return False
        if ans == QMessageBox.StandardButton.Save:
            self._on_save()
            return not self._dirty            # a failed save must not proceed
        self._mark_clean()
        return True

    def _mark_clean(self):
        self._dirty = False
        self._refresh_title()

    def _refresh_title(self):
        if self._model is None:
            base = "PMC / HMC Model"
        else:
            base = f"PMC — {self._model.name}"
        self.setWindowTitle(f"{base}{' *' if self._dirty else ''}")

    # ------------------------------------------------------------------
    # Model load / sync
    # ------------------------------------------------------------------

    def _load_model(self, path: str | Path, *, as_template: bool = False):
        # B6: only commit ``_model_path`` once the model has been
        # successfully built. Previously we set the path *before* calling
        # PMCModel(), so a parse error left the GUI in an inconsistent
        # state (path set, model None).
        #
        # ``as_template=True`` is used by the no-arg auto-load path: the
        # widgets are populated from the shipped fixture, but
        # ``_model_path`` stays ``None`` so a subsequent Ctrl+S falls
        # through to Save-As — otherwise the user would silently overwrite
        # the in-package fixture.
        if not self._maybe_discard_changes("open another model"):
            return
        try:
            new_path  = Path(path)
            new_model = PMCModel(new_path)
        except Exception as exc:
            logger.exception("Failed to load model from %s", path)
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        # SLOT-1: the try above covers the *parser* only. Everything below
        # — widget population, invalidation, the view selector — used to run
        # bare inside a Qt slot (File → Open, and each Recent-files entry).
        # A file PMCModel accepts but the editor cannot display (a
        # non-numeric [ice] value, a string-valued copula extra, a seed
        # outside int32) therefore aborted the process: exit 134, no dialog,
        # unsaved edits gone. This is the failure mode _worker_done was
        # hardened against; the synchronous load path had been left out.
        prev_model, prev_path = self._model, self._model_path
        try:
            self._model = new_model
            if as_template:
                self._model_path = None
            else:
                self._model_path = new_path
                self._push_recent(new_path)
            # Observations survive an Open, unless the new model cannot read
            # them at all (see :meth:`_data_fits_model`).
            keep_data = self._data_fits_model(new_model)
            if not keep_data:
                d_data = 1 if self._last_Y.ndim == 1 else int(self._last_Y.shape[1])
                self._log_append(
                    f"Loaded data dropped: it is {d_data}-dimensional and "
                    f"{new_model.name} expects d={getattr(new_model, 'd', 1)}."
                )
            self._sync_widgets_from_model()
            # S-2: a result describes the model that produced it. Carrying them
            # across an Open left the header naming model B while the canvas
            # still showed A's plot, with nothing signalling the mismatch. The
            # *data* survives — it is independent of the model — the results do
            # not.
            self._invalidate_results(keep_data=keep_data)
            self._set_buttons_enabled(True)
            self._mark_clean()
            self._populate_view_selector()
            if as_template:
                self._status.showMessage(
                    f"Loaded default fixture (unsaved): {new_path.name}"
                )
            else:
                self._status.showMessage(f"Loaded: {self._model_path}")
            self._log_append(f"Loaded model: {self._model}")
        except Exception as exc:
            logger.exception("Failed to apply model from %s", path)
            self._model, self._model_path = prev_model, prev_path
            try:                       # best effort: put the editor back
                if prev_model is not None:
                    self._sync_widgets_from_model()
            except Exception:                              # pragma: no cover
                logger.exception("Rollback re-sync failed")
            QMessageBox.critical(
                self, "Load Error",
                f"The file parsed as a model, but the editor could not "
                f"display it:\n\n{exc}\n\nThe previous model is unchanged.",
            )
            return

    def _data_fits_model(self, mdl) -> bool:
        """Is the loaded observation sequence usable with ``mdl``?

        The S-2 fix keeps the data across an *Open* on the grounds that
        observations are independent of the model. True of K and of the
        parameter values — false of the observation dimension ``d``: a
        ``(N, 3)`` sequence is meaningless to a scalar model and vice versa.
        Keeping it left all four action buttons enabled on data that could
        only ever produce a raw traceback in a message box.
        """
        if self._last_Y is None:
            return True
        d_data  = 1 if self._last_Y.ndim == 1 else int(self._last_Y.shape[1])
        d_model = int(getattr(mdl, "d", 1) or 1)
        return d_data == d_model

    def _sync_widgets_from_model(self):
        """Populate all editor widgets from the current model."""
        mdl = self._model
        if mdl is None:
            return

        # Block dirty-tracking signals while we programmatically populate.
        # (The tabs each block their own; the top-level fields need this too.)
        was_blocked = self.blockSignals(True)
        try:
            self._lbl_name.setText(mdl.name)
            self._lbl_variant.setText(mdl.variant.value)
            self._lbl_K.setText(str(mdl.K))
            self._spn_N.setValue(mdl.N_default)
            seed = mdl.raw.get("model", {}).get("seed")
            self._txt_seed.setText("" if seed is None else str(seed))
        finally:
            self.blockSignals(was_blocked)

        self._tab_prior.load(mdl)
        self._tab_margins.load(mdl)
        self._tab_copulas.load(mdl)

        # [ice] first so files written before S-11 still restore; the
        # dedicated sections win where present.
        self._tab_ice.load(
            {**mdl.ice_config(), **mdl.sem_config(),
             **mdl.raw.get("gui", {})},
            mdl.variant.uses_copula,
        )

        # Show/hide copula tab
        cop_idx = self._tabs.indexOf(self._tab_copulas)
        self._tabs.setTabVisible(cop_idx, mdl.variant.uses_copula)

        # GoF makes sense only for variants that use copulas.
        self._btn_gof.setEnabled(mdl.variant.uses_copula)

    def _rebuild_model_from_widgets(self) -> PMCModel:
        """Build an updated PMCModel from current widget values.

        Raises :class:`PriorTabError` (a `ValueError` subclass) if the prior
        table has invalid cells — caller is expected to surface it via a
        message box.
        """
        raw = self._model.raw

        # Name / N
        raw.setdefault("model", {})
        raw["model"]["name"]      = self._lbl_name.text()
        raw["model"]["N_default"] = self._spn_N.value()
        # S-10: the Seed field was never persisted, so a reproducible
        # simulation stopped being reproducible after one disk round trip.
        seed = self._parse_seed()
        if seed is None:
            raw["model"].pop("seed", None)
        else:
            raw["model"]["seed"] = int(seed)

        # Prior — may raise PriorTabError
        self._tab_prior.save(raw)
        # Margins
        self._tab_margins.save(raw)
        # Copulas
        if self._model.variant.uses_copula:
            self._tab_copulas.save(raw)
        # ICE — merge, never replace. The tab only knows the keys it has a
        # widget for, so assigning its dict wholesale silently deleted every
        # other key the user's TOML declared (``selection_criterion`` and
        # ``margin_selection_rule`` have no widget: loading a model that set
        # them and saving it back dropped them without a word).
        cfg = self._tab_ice.get_cfg()
        # S-11: two of those keys do not belong to [ice]. ``sem_seed`` is a
        # SEM knob — ``_parse_sem_cfg`` only papered over it with a fallback
        # — and ``algorithm`` is a GUI dispatch key with no API counterpart
        # (``_do_estimate`` strips it before forwarding). Writing both under
        # [ice] left a GUI-saved file non-canonical. PMCModel round-trips
        # sections it does not know, so each can now live where it belongs.
        raw["sem"] = {**raw.get("sem", {}), "sem_seed": cfg["sem_seed"]}
        raw["gui"] = {**raw.get("gui", {}), "algorithm": cfg["algorithm"]}
        raw["ice"] = {
            **{k: v for k, v in raw.get("ice", {}).items()
               if k not in ("sem_seed", "algorithm")},
            **{k: v for k, v in cfg.items()
               if k not in ("sem_seed", "algorithm")},
        }

        return PMCModel.from_dict(raw)

    def _set_buttons_enabled(self, enabled: bool):
        for btn in (self._btn_sim, self._btn_cls, self._btn_est):
            btn.setEnabled(enabled)
        # GoF requires a copula-using variant AND a model.
        if self._model is not None:
            self._btn_gof.setEnabled(enabled and self._model.variant.uses_copula)
        else:
            self._btn_gof.setEnabled(False)
        # Save-data is only meaningful when we have data.
        self._act_save_data.setEnabled(self._last_Y is not None)
        # Export-plot is meaningful only when something has been drawn.
        self._act_export_plot.setEnabled(self._has_plot)
        self._btn_export.setEnabled(self._has_plot)
        # Analysis entries start workers of their own: they follow the buttons.
        for act in getattr(self, "_analysis_actions", ()):
            act.setEnabled(
                enabled and self._model is not None
                and self._model.variant.uses_copula
            )
        # The missingness LR test applies to every variant, copula or not.
        if hasattr(self, "_act_missingness_lr"):
            self._act_missingness_lr.setEnabled(enabled and self._model is not None)

    def _set_busy(self, busy: bool):
        """Lock or release every entry point that mutates state (audit S-1).

        Greying out the action buttons alone was not enough: the File menu
        stayed live during a run, and the load handlers write the very
        attributes the running worker's completion callback reads back.
        """
        self._set_buttons_enabled(not busy)
        for act in self._mutating_actions:
            act.setEnabled(not busy)

    def _invalidate_results(self, *, keep_data: bool = True):
        """Drop every result that describes a *previous* model or dataset.

        Opening a model left ``_sim_state``, ``_cls_state``, ``_gof_state``
        and the ICE trace untouched, so the header announced model B while
        the canvas still showed A's plot — with nothing in the interface
        signalling the mismatch (audit S-2, S-3, S-4).

        ``keep_data`` draws the line: observations are independent of the
        model and survive an *Open*, whereas *results* never do. *Reset
        session* passes ``keep_data=False`` to clear both.
        """
        self._sim_state  = None
        self._cls_state  = None
        self._gof_state  = None
        self._tau_ci_state = None
        self._margin_ks_state = None
        self._margin_ks_sample = (None, None)
        self._ice_trace  = None
        self._init_model = None
        self._ice_Y      = None
        if not keep_data:
            self._last_X         = None
            self._last_Y         = None
            self._last_image     = None
            self._last_image_ref = None
        # Playback indexes the trace we just dropped — stop and hide it.
        if self._btn_play.isChecked():
            self._btn_play.setChecked(False)
        self._playback_widget.setVisible(False)
        self._act_save_seg.setEnabled(False)
        self._act_save_data.setEnabled(self._last_Y is not None)
        self._canvas.clear()
        self._has_plot = False
        self._act_export_plot.setEnabled(False)
        self._btn_export.setEnabled(False)
        # Nothing is rendered any more; say so, or _populate_view_selector
        # would believe the (now blank) canvas still holds the old view.
        self._current_view = _VIEW_NONE

    # ------------------------------------------------------------------
    # Menu actions — file
    # ------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open model", "", "TOML files (*.toml);;All files (*)"
        )
        if path:
            self._load_model(path)

    def _on_save(self, path: str | Path | None = None):
        if self._model is None:
            return
        target = Path(path) if path is not None else self._model_path
        if target is None:
            self._on_save_as()
            return
        try:
            mdl = self._rebuild_model_from_widgets()
            mdl.save(target)
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Failed to save model to %s", target)
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        # S-8: commit the path only once the write has actually landed.
        # ``_on_save_as`` used to assign ``_model_path`` first, so after a
        # failed write a later Ctrl+S silently re-aimed at the bad path
        # instead of falling back to Save-As — the very regression the "B6"
        # note on ``_load_model`` records as fixed for loading.
        self._model_path = target
        self._mark_clean()
        self._status.showMessage(f"Saved: {self._model_path}")
        self._push_recent(self._model_path)

    def _on_save_as(self):
        if self._model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model as", "", "TOML files (*.toml);;All files (*)"
        )
        if path:
            self._on_save(path)

    def _on_load_data(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load data CSV", "", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            Y, X = _read_data_csv(path)
        except Exception as exc:
            logger.exception("Failed to load data from %s", path)
            QMessageBox.critical(self, "Data Error", str(exc))
            return
        # The d check was wired into _load_model only, so this path happily
        # accepted a (N, 3) CSV under a scalar model with every action button
        # live — the same trap L-6 closed on the model side. Refuse here
        # instead of dropping: the file is what the user just chose, so the
        # useful answer names both dimensions.
        d_csv = 1 if Y.ndim == 1 else int(Y.shape[1])
        d_mdl = int(getattr(self._model, "d", 1) or 1) if self._model else d_csv
        if d_csv != d_mdl:
            QMessageBox.warning(
                self, "Dimension mismatch",
                f"The file holds {d_csv}-dimensional observations, but "
                f"{self._model.name} expects d={d_mdl}.\n\n"
                f"Open a matching model first, or pick another file.",
            )
            return
        self._last_Y = Y
        self._last_X = X
        # Loading 1D data invalidates any prior image state …
        self._last_image = None
        self._last_image_ref = None
        # … and every result, each of which describes the sequence we have
        # just replaced (audit S-3). Views stayed selectable and went on
        # describing the *old* data.
        self._invalidate_results()
        self._act_save_data.setEnabled(True)
        # Rows with a missing observation (any component, for d ≥ 2) — named
        # only when there are some, so complete files read exactly as before.
        n_miss = int(np.isnan(Y).any(axis=1).sum() if Y.ndim == 2
                     else np.isnan(Y).sum())
        miss = (f"  missing={n_miss} ({100.0 * n_miss / len(Y):.1f} %)"
                if n_miss else "")
        self._log_append(
            f"Loaded data: N={len(Y)}{miss}  "
            f"{'(X labels present)' if X is not None else '(no X labels)'}"
        )
        self._status.showMessage(f"Data loaded: {path}  N={len(Y)}{miss}")
        self._populate_view_selector()

    def _on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*)",
        )
        if not path:
            return
        from pmcprg.pmc.peano import image_to_signal, load_color, load_grayscale
        # Auto-dispatch on model.d: RGB for multivariate models, grayscale
        # for scalar ones. If no model is loaded, fall back to grayscale.
        d_target = getattr(self._model, "d", 1) if self._model is not None else 1
        try:
            img = load_color(path) if d_target > 1 else load_grayscale(path)
        except Exception as exc:
            logger.exception("Failed to load image %s", path)
            QMessageBox.critical(self, "Image Error", str(exc))
            return
        self._last_image       = img
        self._last_Y           = image_to_signal(img)
        # Loading a new image invalidates any prior reference labels and
        # any prior 1D-only ground truth.
        self._last_image_ref   = None
        self._last_X           = None
        # S-3: this cleared _cls_state and _sim_state but left _gof_state,
        # _ice_trace, _ice_Y and _init_model describing the previous data.
        self._invalidate_results()
        self._act_save_data.setEnabled(True)
        # [:2] — a multivariate model dispatches to load_color above, which
        # returns (H, W, d). Unpacking the bare shape raised a ValueError
        # *outside* the try/except, i.e. straight out of a Qt slot, which
        # aborts the process: loading a colour image with a d>1 model killed
        # the GUI. The reference-label path below already gets this right.
        H, W = img.shape[:2]
        self._log_append(
            f"Loaded image: {path}  ({H}×{W} = {H * W} px)  "
            f"intensity ∈ [{img.min():.3f}, {img.max():.3f}]"
        )
        self._status.showMessage(f"Image loaded: {path}  {H}×{W}")
        self._populate_view_selector()
        self._switch_view(_VIEW_IMAGE_RAW)

    def _on_load_image_ref(self):
        if self._last_image is None:
            QMessageBox.warning(
                self, "No image",
                "Load an input image first — reference labels must match its shape.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference label image",
            "",
            "Images (*.png *.tif *.tiff *.bmp);;All files (*)",
        )
        if not path:
            return
        from pmcprg.pmc.peano import image_to_signal, load_labels
        try:
            ref = load_labels(path)
        except Exception as exc:
            logger.exception("Failed to load reference labels %s", path)
            QMessageBox.critical(self, "Image Error", str(exc))
            return
        H, W = self._last_image.shape[:2]
        if ref.shape != (H, W):
            QMessageBox.warning(
                self, "Shape mismatch",
                f"Reference shape {ref.shape} does not match image spatial "
                f"shape ({H}, {W}).",
            )
            return
        self._last_image_ref = ref
        self._last_X         = image_to_signal(ref)
        self._log_append(
            f"Loaded reference labels: {path}  classes={np.unique(ref).tolist()}"
        )
        self._status.showMessage(f"Reference loaded: {path}")
        # N-3: the segmentation view reads ``_last_image_ref`` directly, so a
        # second reference image repaints the error rate — while the log
        # still showed the figure computed against the previous labels, with
        # nothing recording the change. Restate it against the new labels.
        if self._cls_state is not None:
            X_hat = self._cls_state[1]
            if X_hat.shape == self._last_X.shape:
                er = error_rate(self._last_X, X_hat)
                self._log_append(
                    f"  error vs these labels: {er:.4f} ({er * 100:.1f}%)"
                )
        # The labels are a *direct* input of the segmentation view — they are
        # not carried inside ``_cls_state`` — so whatever is on the canvas is
        # now stale. Every other loader repopulates and re-renders; this one
        # only logged, leaving the segmentation panel silently one panel
        # short, which reads as "the reference image was not accepted".
        self._populate_view_selector()
        if self._is_view_available(_VIEW_IMAGE_SEG):
            self._switch_view(_VIEW_IMAGE_SEG)
        elif self._current_view in _ALL_VIEWS:
            self._render_view(self._current_view)

    def _on_save_segmentation(self):
        if self._cls_state is None or self._last_image is None:
            QMessageBox.information(
                self, "No segmentation",
                "Load an image and run Classify (or Estimate then Classify) first.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save segmentation PNG", "segmentation.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        from pmcprg.pmc.peano import save_segmentation, signal_to_image
        _Y, X_hat, _gamma, _X_ref, mdl = self._cls_state
        try:
            # Spatial shape only: signal_to_image folds a label sequence back
            # into (H, W), and a colour image's (H, W, d) would not unpack.
            X_hat_2d = signal_to_image(X_hat, self._last_image.shape[:2])
            save_segmentation(path, X_hat_2d.astype(np.int32), K=mdl.K)
            self._log_append(f"Saved segmentation: {path}")
            self._status.showMessage(f"Segmentation saved: {path}")
        except Exception as exc:
            logger.exception("Failed to save segmentation to %s", path)
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_save_data(self):
        if self._last_Y is None:
            QMessageBox.information(self, "No data", "Nothing to save yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save data CSV", "data.csv", "CSV files (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            Y = np.asarray(self._last_Y)
            X = self._last_X
            # Mirror the reader: one 'Y' column when scalar, 'Y0'…'Y{d-1}'
            # when multivariate (float(Y[n]) would raise on a d-vector).
            y_cols = (["Y"] if Y.ndim == 1
                      else [f"Y{j}" for j in range(Y.shape[1])])

            def _values(n: int) -> list[float]:
                return [float(Y[n])] if Y.ndim == 1 else [float(v) for v in Y[n]]

            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                if X is not None:
                    writer.writerow(["n", "X", *y_cols])
                    for n in range(len(Y)):
                        writer.writerow([n + 1, int(X[n]), *_values(n)])
                else:
                    writer.writerow(["n", *y_cols])
                    for n in range(len(Y)):
                        writer.writerow([n + 1, *_values(n)])
            self._log_append(f"Saved data: {path}  N={len(Y)}")
            self._status.showMessage(f"Data saved: {path}")
        except Exception as exc:
            logger.exception("Failed to save data to %s", path)
            QMessageBox.critical(self, "Save Error", str(exc))

    def _results_payload(self) -> dict:
        """Everything the session has produced, as plain Python.

        Audit S-7 — the log-likelihood trace, the τ trajectories, the family
        history, the classification error rate and the GoF results existed
        only in memory and in two PNG-able views. Recovering a number meant
        reading a picture, or leaving the GUI and calling ``ice()`` by hand.
        """
        out: dict = {"model": None, "estimation": None,
                     "classification": None, "goodness_of_fit": None}
        if self._model is not None:
            out["model"] = {
                "name": self._model.name,
                "variant": self._model.variant.value,
                "K": int(self._model.K),
                "d": int(getattr(self._model, "d", 1) or 1),
                "margin_structure": getattr(self._model, "margin_structure",
                                            "state"),
                "path": str(self._model_path) if self._model_path else None,
            }
        if self._ice_trace is not None:
            tr = self._ice_trace
            out["estimation"] = {
                "config": self._tab_ice.get_cfg(),
                "n_iterations": len(tr.log_liks),
                "log_liks": list(tr.log_liks),
                "tau_history": np.asarray(tr.tau_history),
                "p_history": np.asarray(tr.p_history),
                "family_history": tr.family_history,
                "margin_history": tr.margin_history,
                "candidates": list(getattr(tr, "candidates", []) or []),
                "best_iter": int(getattr(tr, "best_iter", -1)),
                "returned_iter": int(getattr(tr, "returned_iter", -1)),
                "degenerate": (None if getattr(tr, "degenerate", None) is None
                               else [str(f) for f in tr.degenerate]),
                "multistart_runs": [
                    {"run_tag": r.run_tag, "log_liks": list(r.log_liks)}
                    for r in tr.multistart_runs
                ],
            }
        if self._cls_state is not None:
            _, X_hat, _, X_ref, mdl = self._cls_state
            entry = {"model": getattr(mdl, "name", None),
                     "n": int(np.size(X_hat)), "error_rate": None}
            if X_ref is not None and np.shape(X_ref) == np.shape(X_hat):
                entry["error_rate"] = float(error_rate(X_ref, X_hat))
            out["classification"] = entry
        if self._gof_state is not None:
            out["goodness_of_fit"] = list(self._gof_state)
        return out

    def _on_export_results(self):
        payload = self._results_payload()
        if payload["estimation"] is None and payload["classification"] is None \
                and payload["goodness_of_fit"] is None:
            QMessageBox.information(
                self, "Nothing to export",
                "Run Estimate, Classify or a GoF test first.",
            )
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Export results", "",
            "CSV — per-iteration table (*.csv);;JSON — everything (*.json)",
        )
        if not path:
            return
        target = Path(path)
        as_json = target.suffix.lower() == ".json" or "JSON" in (selected or "")
        if not target.suffix:
            target = target.with_suffix(".json" if as_json else ".csv")
        try:
            if as_json:
                import json
                target.write_text(
                    json.dumps(payload, indent=2, default=_jsonable),
                    encoding="utf-8",
                )
                what = "results (JSON)"
            else:
                # CSV is flat by nature, so it carries the per-iteration
                # table — the part a reader actually plots. Anything nested
                # (GoF rows, the config) belongs to the JSON form; say so
                # rather than dropping it silently.
                if self._ice_trace is None:
                    QMessageBox.information(
                        self, "Nothing to tabulate",
                        "The CSV form holds the per-iteration estimation "
                        "table. Run Estimate first, or export as JSON.",
                    )
                    return
                header, rows = _trace_rows(self._ice_trace)
                for run in self._ice_trace.multistart_runs:
                    _, extra = _trace_rows(run, run.run_tag or "alt")
                    rows += extra
                with open(target, "w", newline="", encoding="utf-8") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(header)
                    wr.writerows(rows)
                what = f"{len(rows)} iteration rows (CSV)"
        except Exception as exc:
            logger.exception("Failed to export results to %s", target)
            QMessageBox.critical(self, "Export Error", str(exc))
            return
        self._log_append(f"Exported {what} → {target}")
        self._status.showMessage(f"Results exported: {target}")

    def _on_export_plot(self):
        if not self._has_plot:
            QMessageBox.information(
                self, "Nothing to export",
                "The plot canvas is empty. Run an action first.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export plot", "plot.png",
            "PNG image (*.png);;PDF (*.pdf);;SVG (*.svg);;All files (*)",
        )
        if not path:
            return
        try:
            self._canvas.fig.savefig(path, dpi=150, bbox_inches="tight")
            self._log_append(f"Exported plot: {path}")
            self._status.showMessage(f"Plot exported: {path}")
        except Exception as exc:
            logger.exception("Failed to export plot to %s", path)
            QMessageBox.critical(self, "Export Error", str(exc))

    def _on_reset(self):
        """Clear loaded data, last results, and the plot canvas."""
        # Belt and braces behind ``_set_busy``: whatever reaches this method
        # while a worker is in flight would clear the state the worker's
        # completion callback is about to write back into.
        if self._worker is not None and self._worker.isRunning():
            self._status.showMessage("Reset ignored — a computation is running.")
            return
        # S-4: this wiped the state but left ``_current_view`` naming the
        # view it had just emptied, so the combobox went on showing e.g.
        # "ICE — Tail dependence" over a blank canvas.
        self._invalidate_results(keep_data=False)
        self._log.clear()
        self._status.showMessage("Session reset.")
        self._set_buttons_enabled(self._model is not None)
        self._populate_view_selector()

    def _on_about(self):
        QMessageBox.information(
            self, "About PMC GUI",
            f"PMC / HMC Copula Model Tool — v{_PKG_VERSION}\n"
            "pmcprg.pmc — Pairwise Markov Chain with copula-based transitions.\n\n"
            "Variants: HMC-IN, HMC-IN2, HMC-DN, PMC-IN, PMC\n"
            "Inference: MPM forward-backward (Devijver normalization)\n"
            "Estimation: ICE (Iterative Conditional Estimation, multistart)"
        )

    # ------------------------------------------------------------------
    # Action: Simulate
    # ------------------------------------------------------------------

    def _on_simulate(self):
        # Confirm before overwriting loaded/simulated data.
        if self._last_Y is not None:
            ans = QMessageBox.question(
                self, "Overwrite data?",
                "This will replace the currently-loaded (X, Y) sequence. "
                "Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before simulation")
            QMessageBox.critical(self, "Model Error", str(exc))
            return

        N    = self._spn_N.value()
        seed = self._parse_seed()
        self._start_worker(
            self._do_simulate, mdl, N, seed,
            on_done=self._on_sim_done,
        )

    def _parse_seed(self) -> int | None:
        """Read the Seed widget; blank → None, otherwise int (warns on parse error)."""
        txt = self._txt_seed.text().strip()
        if not txt:
            return None
        try:
            return int(txt)
        except ValueError:
            logger.warning("Seed %r is not an integer — using None.", txt)
            self._log_append(f"WARN: seed {txt!r} not an integer — using random.")
            return None

    @staticmethod
    def _do_simulate(mdl, N, seed):
        from pmcprg.pmc.simulate import simulate
        X, Y = simulate(mdl, N=N, seed=seed)
        return X, Y, mdl

    def _on_sim_done(self, result):
        X, Y, mdl = result
        self._last_X = X
        self._last_Y = Y
        self._model  = mdl
        # Simulation is 1D — invalidate any previously-loaded image …
        self._last_image = None
        self._last_image_ref = None
        # … and Simulate replaces the observation sequence, so every result
        # describing the *previous* one has to go too. This is the widest
        # case of the S-3 class and the one a user meets most often: one
        # button, and the ICE trace, its observations and the GoF result all
        # went on describing a sequence that no longer existed.
        self._invalidate_results()
        self._sim_state = (X, Y, mdl)
        self._act_save_data.setEnabled(True)
        self._log_append(
            f"Simulated N={len(Y)}  Y∈[{Y.min():.2f}, {Y.max():.2f}]  "
            f"mean={Y.mean():.3f}  std={Y.std():.3f}"
        )
        self._populate_view_selector()
        self._switch_view(_VIEW_SIMULATION)

    # ------------------------------------------------------------------
    # Action: Classify
    # ------------------------------------------------------------------

    def _on_classify(self):
        if self._last_Y is None:
            QMessageBox.warning(self, "No data",
                                "Run Simulate first or load a CSV data file.")
            return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before classification")
            QMessageBox.critical(self, "Model Error", str(exc))
            return

        # S-1(a): the observation and its reference labels travel *with* the
        # job. ``_on_cls_done`` used to re-read ``self._last_X`` on return, so
        # anything replacing the data mid-run produced a shape clash deep
        # inside ``error_rate`` — in a Qt slot, i.e. a process abort.
        self._start_worker(
            self._do_classify, mdl, self._last_Y, self._last_X,
            on_done=self._on_cls_done,
        )

    @staticmethod
    def _do_classify(mdl, Y, X_ref):
        from pmcprg.pmc.inference import classify
        X_hat, gamma, ll = classify(mdl, Y)
        return X_hat, gamma, ll, mdl, Y, X_ref

    def _on_cls_done(self, result):
        X_hat, gamma, ll, mdl, Y, X_ref = result
        msg = f"Classification  log-lik={ll:.2f}"
        if X_ref is not None and X_ref.shape == X_hat.shape:
            er = error_rate(X_ref, X_hat)
            msg += f"  error={er:.4f} ({er*100:.1f}%)"
        self._log_append(msg)
        self._cls_state = (Y, X_hat, gamma, X_ref, mdl)
        # The 2D segmentation only means something if the sequence we just
        # classified is still the one in memory.
        on_image = self._last_image is not None and self._last_Y is Y
        self._act_save_seg.setEnabled(on_image)
        self._populate_view_selector()
        # When the data came from an image, jump to the 2D segmentation view;
        # otherwise keep the existing 1D view.
        self._switch_view(_VIEW_IMAGE_SEG if on_image else _VIEW_CLASSIFICATION)

    # ------------------------------------------------------------------
    # Action: Estimate (ICE) — uses the worker's progress signal
    # ------------------------------------------------------------------

    def _on_estimate(self):
        if self._last_Y is None:
            QMessageBox.warning(self, "No data",
                                "Run Simulate first or load a CSV data file.")
            return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before estimation")
            QMessageBox.critical(self, "Model Error", str(exc))
            return

        Y         = self._last_Y
        est_cfg   = self._tab_ice.get_cfg()
        algorithm = str(est_cfg.get("algorithm", "ice"))
        # Both ICE and SEM use a determinate progress bar — max_iter is
        # known up-front for both. Under a *process*-parallel multistart the
        # per-iteration callback cannot cross the process boundary, so the bar
        # would sit frozen at zero; run it indeterminate instead, which is an
        # honest "working, no ETA" rather than a stalled-looking zero.
        parallel = (int(est_cfg.get("multistart_workers", 1)) > 1
                    and int(est_cfg.get("n_starts", 1)) > 1)
        max_iter = int(est_cfg["max_iter"])   # always emitted by _IceTab.get_cfg
        label    = algorithm.upper()
        if parallel:
            label += f" ×{int(est_cfg['multistart_workers'])}"
        self._start_worker(
            self._do_estimate, mdl, Y, est_cfg,
            on_done=self._on_est_done,
            forward_progress=not parallel,
            progress_max=0 if parallel else max_iter,
            progress_label=label,
        )

    @staticmethod
    def _do_estimate(mdl, Y, est_cfg, *, progress_cb=None):
        """Worker entry: dispatch to ICE or SEM based on ``est_cfg['algorithm']``.

        Returns an :class:`IceResult` (ICE) or :class:`SemResult` (SEM); both
        carry the same fields (``initial_model``, ``fitted_model``, ``Y``,
        ``trace``) so downstream view code can stay agnostic.
        """
        algorithm = str(est_cfg.get("algorithm", "ice"))
        if is_pair(mdl):
            # A failure deep inside the estimator surfaces in the error box as
            # its last traceback line only; on a pair model say which model
            # structure was being fitted, so the message is actionable.
            try:
                return PMCMainWindow._do_estimate_impl(
                    mdl, Y, est_cfg, algorithm, progress_cb=progress_cb)
            except Exception as exc:
                raise RuntimeError(
                    f"{algorithm.upper()} failed on a general PMC with "
                    f"pair-indexed margins f_ij (margin_structure = 'pair', "
                    f"DerrodePieczynski_CSDA2013 Eqs. 12–14): "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return PMCMainWindow._do_estimate_impl(
            mdl, Y, est_cfg, algorithm, progress_cb=progress_cb)

    @staticmethod
    def _do_estimate_impl(mdl, Y, est_cfg, algorithm: str, *, progress_cb=None):
        """Body of :meth:`_do_estimate` — the dispatch to ICE or SEM."""
        # The ICE tab always emits the *union* of both estimators' knobs.
        # Strip the GUI-only dispatch key, then drop each estimator's
        # non-applicable keys so neither receives the other's config
        # (tol / patience are ICE-only; sem_seed is SEM-only) — audit Q-15.
        forward_cfg = {k: v for k, v in est_cfg.items() if k != "algorithm"}
        if algorithm == "sem":
            from pmcprg.pmc import sem
            for k in ("tol", "patience"):          # ICE-only convergence knobs
                forward_cfg.pop(k, None)
            fitted, trace = sem(
                mdl, Y, sem_cfg=forward_cfg, progress_cb=progress_cb,
            )
            return SemResult(
                initial_model=mdl, fitted_model=fitted, Y=Y, trace=trace,
            )
        else:
            from pmcprg.pmc import IceResult, ice
            forward_cfg.pop("sem_seed", None)       # SEM-only knob
            fitted, trace = ice(
                mdl, Y, ice_cfg=forward_cfg, progress_cb=progress_cb,
            )
            return IceResult(
                initial_model=mdl, fitted_model=fitted, Y=Y, trace=trace,
            )

    def _on_est_done(self, result):
        # `result` is an :class:`IceResult` or :class:`SemResult` from the
        # worker — both expose the same diagnostic fields.
        self._model       = result.fitted_model
        self._init_model  = result.initial_model      # for view J
        self._ice_trace   = result.trace              # for views A–K
        self._ice_Y       = result.Y                  # for views C, H, J
        trace = result.trace
        self._sync_widgets_from_model()
        # S-9: this used to call _mark_clean(). The estimator returns a model
        # that differs from the file on disk, so declaring it clean dropped
        # the asterisk and let a later Ctrl+S overwrite the user's source
        # TOML with the fit, without confirmation and without any sign that
        # the two had diverged.
        self._mark_dirty()
        lls = trace.log_liks
        n_runs = 1 + len(trace.multistart_runs)
        # Detect SEM vs ICE from the result type to pick the right label
        # (SemResult subclasses IceResult, so test the subclass).
        algo_label = "SEM" if isinstance(result, SemResult) else "ICE"
        msg = (
            f"{algo_label} done  iters={len(lls)}  "
            f"LL: {lls[0]:.2f} → {lls[-1]:.2f}"
        )
        returned = getattr(trace, "returned_iter", -1)
        if 0 <= returned < len(lls) - 1:     # return_best_iterate kept an earlier one
            msg += f"  (returned best iterate {returned}: LL {lls[returned]:.2f})"
        if n_runs > 1:
            msg += f"  (multistart: {n_runs} runs)"
        self._log_append(msg)
        degenerate = getattr(trace, "degenerate", None)
        if degenerate:
            # The WARNING of the estimator is in the log too; this line ties
            # it to the fit just shown.
            self._log_append(
                f"⚠ {algo_label}: degenerate fitted model — "
                + "; ".join(str(f) for f in degenerate)
            )
        # P6: the fitted [missingness] mechanism, alongside the other
        # estimated parameters (log-lik, degenerate states above) — silent
        # when ignorable (``fitted.missingness is None``, most models).
        mech_summary = missingness_summary(self._model.missingness)
        if mech_summary is not None:
            self._log_append(f"Missingness: {mech_summary}")
        # Refresh the View selector (ICE/SEM views just became available)
        # and auto-switch to the dashboard.
        self._populate_view_selector()
        self._switch_view(_ICE_VIEW_DASHBOARD)

    # ------------------------------------------------------------------
    # Action: GoF test — emits per-pair progress, plots a heatmap.
    # ------------------------------------------------------------------

    def _on_margin_ks(self):
        if self._last_Y is None:
            QMessageBox.warning(self, "No data",
                                "Run Simulate first or load a CSV data file.")
            return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before the margin KS test")
            QMessageBox.critical(self, "Model Error", str(exc))
            return
        self._start_worker(
            self._do_margin_ks, mdl, self._last_Y,
            on_done=self._on_margin_ks_done,
            forward_progress=True,
            progress_max=_MARGIN_KS_B,
            progress_label="Margin KS",
        )

    def _on_margin_ks_done(self, rows: list):
        if not rows:
            self._log_append("Margin KS: nothing to test.")
            return
        B = next((r["B"] for r in rows if "B" in r), _MARGIN_KS_B)
        kind = next((r.get("statistic") for r in rows if r.get("statistic")), "")
        how  = ("Neyman smooth components, Bonferroni-corrected over 4"
                if kind == "neyman" else "multivariate KS")
        self._log_append(
            f"Margin adequacy ({how}, posterior draw, bootstrapped null, "
            f"B={B}):"
        )
        if is_pair(self._model):
            self._log_append(
                "  pair margins: state k is tested against the law of y_n "
                "given x_n = k, g_k = Σ_j A[k,j] f_kj "
                "(DerrodePieczynski_CSDA2013 Eq. 12)"
            )
        for r in rows:
            if "p" not in r:
                self._log_append(
                    f"  state {r['state']} ({r['dist']}): "
                    f"{r.get('error', 'n/a')}"
                )
                continue
            verdict = "✓ accept" if r["p"] >= 0.05 else "✗ reject"
            comps = r.get("components") or []
            detail = (
                f"  {r.get('component', '')}: "
                + " ".join(f"V{j}={v:+.1f}" for j, v in enumerate(comps, 1))
                if comps else
                f"  KS={r.get('statistic_value', float('nan')):.4f}"
            )
            self._log_append(
                f"  state {r['state']}  {r['dist']:<10} p={r['p']:.3f}  "
                f"n={r['n']}  {verdict}{detail}"
            )
        # The view redraws the very sample the statistic was computed on, so
        # it must be the same draw — recomputing would show a different one.
        from pmcprg.pmc.inference import (
            forward, precompute_weights, sample_posterior,
        )
        Y = self._last_Y
        n_series = next((r["n_series"] for r in rows if "n_series" in r),
                        len(Y))
        Y = Y[:n_series]
        W, f_pdf = precompute_weights(self._model, Y)
        alpha, _ = forward(self._model, Y, W=W, f_pdf=f_pdf)
        X_draw = sample_posterior(self._model, Y, np.random.default_rng(0),
                                  W=W, f_pdf=f_pdf, alpha_hat=alpha)
        self._margin_ks_sample = (Y, X_draw)
        self._margin_ks_state = list(rows)
        self._populate_view_selector()
        self._switch_view(_VIEW_MARGIN_KS)

    # ------------------------------------------------------------------
    # Action: Missingness LR test (P6) — text-only result, no dedicated view.
    # ------------------------------------------------------------------

    def _on_missingness_lr_test(self):
        if self._last_Y is None:
            QMessageBox.warning(self, "No data",
                                "Run Simulate first or load a CSV data file.")
            return
        from pmcprg.pmc.gaps import missing_mask
        Y = self._last_Y
        miss = missing_mask(Y)
        if not (0 < int(miss.sum()) < len(miss)):
            QMessageBox.information(
                self, "Not applicable",
                "The missingness LR test needs data with both missing "
                "and observed rows (none here).",
            )
            return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before the missingness LR test")
            QMessageBox.critical(self, "Model Error", str(exc))
            return
        self._start_worker(
            self._do_missingness_lr_test, mdl, Y,
            on_done=self._on_missingness_lr_test_done,
            forward_progress=False,
            progress_label="Missingness LR",
        )

    @staticmethod
    def _do_missingness_lr_test(mdl, Y, *, progress_cb=None):
        """Worker entry: the default test, 'state' (H1) vs a common rate (H0).

        No bootstrap (``n_bootstrap=0``, the asymptotic χ² p-value only) —
        a bootstrap run costs 2×B extra ICE fits, better driven from a script
        (``pmcprg.pmc.missingness_lr_test``) than a blocking menu action.
        """
        from pmcprg.pmc.missingness_lr import missingness_lr_test
        return missingness_lr_test(mdl, Y, progress_cb=progress_cb)

    def _on_missingness_lr_test_done(self, result):
        self._log_append(result.summary())

    def _on_tau_ci(self):
        if self._last_Y is None:
            QMessageBox.warning(self, "No data",
                                "Run Simulate first or load a CSV data file.")
            return
        if self._model is None or not self._model.variant.uses_copula:
            QMessageBox.information(
                self, "Not applicable",
                "τ intervals are only meaningful for variants that use copulas.",
            )
            return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before τ intervals")
            QMessageBox.critical(self, "Model Error", str(exc))
            return
        self._start_worker(
            self._do_tau_ci, mdl, self._last_Y,
            on_done=self._on_tau_ci_done,
            forward_progress=True,
            progress_max=max(len(mdl.copula_blocks()), 1),
            progress_label="τ CI",
        )

    def _on_tau_ci_done(self, rows: list):
        if not rows:
            self._log_append("τ intervals: no copulas to bootstrap.")
            return
        B = next((r["B"] for r in rows if "B" in r), _TAU_CI_B)
        self._log_append(f"τ bootstrap intervals (ξ-weighted, B={B}):")
        for r in rows:
            if "lo" not in r:
                self._log_append(
                    f"  ({r['i']},{r['j']})  {r['name']}: {r.get('error', 'n/a')}"
                )
                continue
            flag = "  (straddles 0)" if r["lo"] <= 0.0 <= r["hi"] else ""
            drift = ("" if abs(r["tau_hat"] - r["tau"]) < 5e-4
                     else f"  model τ={r['tau']:+.3f}")
            self._log_append(
                f"  ({r['i']},{r['j']})  {r['name']:<8} "
                f"τ̂={r['tau_hat']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  "
                f"n_eff={r['n_eff']:.0f}{flag}{drift}"
            )
        self._tau_ci_state = list(rows)
        self._populate_view_selector()
        self._switch_view(_VIEW_TAU_CI)

    def _on_gof_test(self):
        if self._last_Y is None:
            QMessageBox.warning(self, "No data",
                                "Run Simulate first or load a CSV data file.")
            return
        if self._model is None or not self._model.variant.uses_copula:
            QMessageBox.information(
                self, "GoF unavailable",
                "GoF is only meaningful for variants that use copulas.",
            )
            return
        try:
            mdl = self._rebuild_model_from_widgets()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
            return
        except Exception as exc:
            logger.exception("Model rebuild failed before GoF")
            QMessageBox.critical(self, "Model Error", str(exc))
            return

        Y = self._last_Y
        # S-6: the bootstrap now resamples whole series under the fitted
        # model, so progress is measured in replicates, not in pairs — all
        # K² pairs come out of every replicate at once.
        self._start_worker(
            self._do_gof_test, mdl, Y,
            on_done=self._on_gof_done,
            forward_progress=True,
            progress_max=_GOF_BOOTSTRAP_B,
            progress_label="GoF",
        )

    @staticmethod
    def _do_margin_ks(mdl, Y, B: int = _MARGIN_KS_B, *, seed: int = 0,
                      max_len: int = 2000, progress_cb=None):
        """Are the fitted **margins** adequate? — audit G-7, opportunity O7.

        The GoF heatmap tests copulas; nothing tested the margins, although
        ``pmcprg.diagnostics`` was in the package with no entry point. Three
        design points, every one settled by measurement rather than argument.

        **The statistic is a Neyman smooth decomposition, not a KS test.**
        Six statistics were compared over 200 datasets per alternative, all
        calibrated by the same bootstrap so the comparison was purely about
        power. The multivariate KS test the package shipped with sat near the
        nominal level on the hard cases; the smooth components take rejection
        of a wrong margin from 0.098 to **0.338** against Student-t tails and
        from 0.152 to **0.480** against skew-normal asymmetry. And *which*
        component fires is the diagnosis — a p-value says a margin is wrong,
        ``V₃`` against ``V₄`` says whether it is the skewness or the tails.
        MKS remains the fallback for a multivariate margin, where the
        probability integral transform the components rest on does not apply.

        **The multiplicity is paid.** Four components is four tests; reporting
        the smallest p-value uncorrected takes the level from 0.05 to 0.175,
        measured. Bonferroni restores it (0.052) and still beats Neyman's own
        omnibus, because the components are near-orthogonal under H₀.

        **The critical value is bootstrapped, not tabulated.** The tabulated
        MKS one rejects a *correct* model 10% of the time at a nominal 5%
        (1000 datasets): Y is a Markov chain and KS assumes i.i.d. The
        calibration goes through :func:`pmcprg.diagnostics.parametric_bootstrap`,
        which also absorbs the bias of assigning the per-state sample — a
        posterior draw here — since replicates run the identical pipeline.
        As everywhere in this package, the model is not refitted per
        replicate: the null is "the fitted model, taken as given, generated
        Y".
        """
        from pmcprg.diagnostics   import (
            COMPONENT_NAMES, mks_1samp, neyman_components, neyman_test,
            parametric_bootstrap,
        )
        from pmcprg.pmc.inference import (
            forward, precompute_weights, sample_posterior,
        )

        Y = np.asarray(Y)
        if len(Y) > max_len:
            Y = Y[:max_len]                # contiguous — cost is linear in N·B
        K = int(mdl.K)
        n_comp = 4
        # The sample of state k is the law of y_n given x_n = k: f_k on a
        # state model, the mixture g_k = Σ_j A[k, j] f_kj on a pair model
        # (DerrodePieczynski_CSDA2013 Eq. 12 summed over x_{n+1}) — ``margin(k)``
        # raises there.
        laws = [state_law(mdl, k) for k in range(K)]
        # The components need a scalar probability integral transform.
        scalar = getattr(mdl, "d", 1) in (1, None) and Y.ndim == 1

        def _samples(Yv, rng):
            """Per-state slices of one posterior draw."""
            W, f_pdf = precompute_weights(mdl, Yv)
            alpha, _ = forward(mdl, Yv, W=W, f_pdf=f_pdf)
            lab = sample_posterior(mdl, Yv, rng, W=W, f_pdf=f_pdf,
                                   alpha_hat=alpha)
            return {k: np.asarray(Yv)[lab == k] for k in range(K)}

        def _stats(_model, Yv, rng):
            out: dict = {}
            for k, sel in _samples(Yv, rng).items():
                if len(sel) < 20:
                    continue
                if scalar:
                    try:
                        res = neyman_components(
                            sel, cdf=laws[k].cdf_vec, order=n_comp)
                        for j, sq in enumerate(res.squares, start=1):
                            out[(k, j)] = float(sq)
                    except Exception:
                        pass
                else:
                    try:
                        arr = sel.reshape(-1, 1) if sel.ndim == 1 else sel

                        def _cdf(point, _k=k):
                            return float(laws[_k].cdf_vec(
                                np.atleast_1d(np.asarray(point, float)))[0])
                        out[(k, 0)] = float(mks_1samp(arr, cdf=_cdf).statistic)
                    except Exception:
                        pass
            return out

        rng0 = np.random.default_rng(seed)
        obs_samples = _samples(Y, rng0)
        sizes = {k: len(v) for k, v in obs_samples.items()}
        signed: dict[int, list] = {}
        if scalar:
            for k, sel in obs_samples.items():
                if len(sel) >= 20:
                    try:
                        signed[k] = [float(v) for v in neyman_components(
                            sel, cdf=laws[k].cdf_vec,
                            order=n_comp).components]
                    except Exception:
                        pass

        results = parametric_bootstrap(mdl, Y, _stats, B=B, seed=seed,
                                       progress_cb=progress_cb)

        rows = []
        for k in range(K):
            # Looked up by ``i``: ``raw["margins"][k]`` was block number k,
            # i.e. f_0 of the pair (0, 1) on a K²-format file.
            dist = state_law_name(mdl, k)
            n_series = next((r.n_series for r in results.values()), len(Y))
            row = dict(state=k, dist=dist, n=int(sizes.get(k, 0)), B=int(B),
                       n_series=int(n_series),
                       statistic="neyman" if scalar else "mks")
            keys = [(k, j) for j in range(1, n_comp + 1)] if scalar else [(k, 0)]
            got  = [results[key] for key in keys if key in results]
            if not got or all(g.n_valid == 0 for g in got):
                rows.append(dict(**row, error=(
                    f"sample too small (n={sizes.get(k, 0)})"
                    if sizes.get(k, 0) < 20
                    else "no usable bootstrap replicate")))
                continue
            if scalar:
                p_corr, idx = neyman_test([g.p_value for g in got])
                rows.append(dict(
                    **row, p=p_corr,
                    component=COMPONENT_NAMES[idx - 1] if idx else "?",
                    components=signed.get(k, []),
                    p_components=[g.p_value for g in got],
                    n_valid=min(g.n_valid for g in got),
                ))
            else:
                rows.append(dict(**row, statistic_value=got[0].observed,
                                 p=got[0].p_value, n_valid=got[0].n_valid))
        return rows

    @staticmethod
    def _do_tau_ci(mdl, Y, B: int = _TAU_CI_B, *, alpha: float = 0.05,
                   seed: int = 0, progress_cb=None):
        """ξ-weighted bootstrap interval on τ̂ for every pair — audit G-7.

        Every τ in this interface is quoted to three decimals with no
        indication of how firm it is, though ``FitResult.bootstrap_ci`` has
        been in the package all along, unreachable. It resamples uniformly,
        which is the wrong null here: pair (i, j) is fitted on ξ-weighted
        pseudo-observations, so the resample must follow ξ too.

        **The size drawn is the effective one, not the sequence length.**
        Drawing N−1 points with probability ∝ ξ would hand a pair whose
        posterior mass is 45 transitions a bootstrap sample of 1500, and the
        interval would come out far too narrow — confident precisely where
        the data are thinnest. Each replicate therefore draws
        ``round(n_eff)`` points, ``n_eff = Σw`` — the pair's posterior mass,
        i.e. the expected number of transitions it carries, which is the
        package's effective size everywhere (:func:`pmcprg.pmc.ice._effective_n`).
        For w ∈ [0, 1] it never exceeds Kish's (Σw)²/Σw², so the interval
        errs on the wide side. (Audit K-5: this docstring used to claim Kish.)

        Refitting uses the same weighted MLE as the ICE M-step
        (:func:`pmcprg.pmc.ice._fit_copula_params`), so the interval is about the
        estimator the package actually uses.

        Note this deliberately does **not** go through
        :func:`pmcprg.diagnostics.parametric_bootstrap`. That harness resamples
        whole *series* from the fitted model, which is what calibrates a
        goodness-of-fit p-value; here the object is an interval on a
        parameter, obtained by resampling the *pseudo-observations*. Two
        different bootstraps that happen to share a loop shape — unifying
        them would be a mistake.

        The interval reported is the **percentile** bootstrap interval — the
        α/2 and 1 − α/2 quantiles of the B refitted τ̂*.

        References
        ----------
        * Efron, B. (1979). Bootstrap methods: another look at the jackknife.
          *Ann. Statist.* 7(1), 1–26.
        * Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the
          Bootstrap*. Chapman & Hall, ch. 13 (percentile intervals).
        """
        from pmcprg.pmc.ice       import _effective_n, _fit_copula_params
        from pmcprg.pmc.inference import (
            backward, forward, joint_posteriors, precompute_weights,
        )

        Y = np.asarray(Y)
        N = len(Y)
        W, f_pdf = precompute_weights(mdl, Y)
        alpha_hat, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
        xi = joint_posteriors(alpha_hat, W, backward(mdl, Y, W=W))

        # u = F_ij(Y_n), v = F_ji(Y_{n+1}): the left and right margins of the
        # pair (i, j), DerrodePieczynski_CSDA2013 Eq. 12 (F_i and F_j on a
        # state model).
        F = margin_cdfs(mdl, Y, lo=1e-12, hi=1.0 - 1e-12)

        blocks = mdl.copula_blocks()
        rng = np.random.default_rng(seed)
        out: list[dict] = []
        for idx, blk in enumerate(blocks):
            i, j = int(blk["i"]), int(blk["j"])
            cop  = mdl.copula(i, j)
            cls  = type(cop)
            if progress_cb is not None:
                try:
                    progress_cb(idx, len(blocks), 0.0,
                                f"({i},{j}) {cop.copula_enum.value.SHORT_NAME}")
                except Exception:                              # pragma: no cover
                    pass
            w = np.asarray(xi[:, i, j], dtype=float)
            u = F[: N - 1, i, j]
            v = F[1:,      j, i]
            n_eff = float(_effective_n(w))
            row = dict(i=i, j=j, name=cop.copula_enum.value.SHORT_NAME,
                       tau=float(cop.params.get("tau_k", float("nan"))),
                       n_eff=n_eff, B=int(B), alpha=float(alpha))
            m = int(round(n_eff))
            if w.sum() < 1e-12 or m < 8:
                out.append(dict(**row, error=(
                    f"effective sample too small (n_eff={n_eff:.1f})")))
                continue
            # The interval must surround the estimator it is an interval FOR.
            # `cop.params["tau_k"]` is what the *model* declares; after an ICE
            # run the two coincide, but on a hand-edited or freshly-loaded
            # model they need not, and plotting a declared value against a
            # bootstrap of a refitted one invites reading a discrepancy as
            # uncertainty. Refit on the full weighted data and report both.
            try:
                row["tau_hat"] = float(_fit_copula_params(
                    cls, cop.copula_enum, u, v, w,
                )["tau_k"])
            except Exception:                                  # pragma: no cover
                row["tau_hat"] = row["tau"]
            probs = w / w.sum()
            taus  = np.full(B, np.nan)
            ones  = np.ones(m)
            for b in range(B):
                pick = rng.choice(N - 1, size=m, replace=True, p=probs)
                try:
                    taus[b] = _fit_copula_params(
                        cls, cop.copula_enum, u[pick], v[pick], ones,
                    )["tau_k"]
                except Exception:                              # pragma: no cover
                    pass
            good = taus[np.isfinite(taus)]
            if good.size < max(10, B // 10):
                out.append(dict(**row, error="bootstrap did not converge"))
                continue
            lo, hi = np.percentile(good, [100 * alpha / 2,
                                          100 * (1 - alpha / 2)])
            out.append(dict(**row, lo=float(lo), hi=float(hi),
                            n_valid=int(good.size)))
        return out

    @staticmethod
    def _do_gof_test(mdl, Y, B: int = 200, *, progress_cb=None,
                     seed: int = 0, max_points: int = 4000,
                     max_len: int = 2000, statistic: str = _GOF_STATISTIC):
        """ξ-weighted GoF test — genuinely one test per state pair.

        Audit S-6. This used to form ``u = f_cdf[:N-1, i, j]``,
        ``v = f_cdf[1:, j, i]`` and hand them to ``Copula.fit``, which
        **rank-transforms** its columns. The marginal CDFs are strictly
        increasing transforms of the *same* ``Y``, so the ranks came out
        identical for every ``(i, j)``: the heatmap returned the same
        statistic and the same p-value K² times, to sixteen digits. It was
        testing the *unconditional* lag-1 dependence of the whole series,
        K² times over, and the per-pair diagnostic it displayed did not
        exist.

        Statistic
        ---------
        Exactly what the ICE M-step fits, so the test and the estimator
        cannot drift apart: pseudo-observations ``u = F_ij(Y_n)``,
        ``v = F_ji(Y_{n+1})`` — the left and right margins of the pair (i, j),
        DerrodePieczynski_CSDA2013 Eq. 12, which reduce to ``F_i(Y_n)`` and
        ``F_j(Y_{n+1})`` for state margins — weighted by the pair posterior
        ``ξ_n(i, j) = P(X_n=i, X_{n+1}=j | Y)``, fed to the ξ-weighted
        Cramér-von Mises statistic that already backs the ``cvm`` selection
        criterion.

        Null distribution
        -----------------
        The bootstrap resamples at the level of the **model**, not of the
        pair: each replicate simulates a fresh series from the fitted model,
        re-runs forward-backward on it, and recomputes the same weighted
        statistic. A per-pair bootstrap — drawing points straight from
        ``c_ij`` and reusing the observed weights — was tried first and is
        badly mis-calibrated: it rejected the *true* model at p = 0.01 on
        the diagonal pairs. Two reasons, both absent from the pure draw and
        both reproduced by simulating the whole series: the observed
        pseudo-observations are a ξ-weighted *mixture* (every transition
        enters every pair, merely with small weight), and consecutive
        transitions share an observation, so the empirical copula is more
        variable than under an i.i.d. draw.

        Caveat, deliberate: the model is **not refitted** on each replicate,
        so the test does not pay for parameter estimation the way
        Genest-Rémillard's copula-level bootstrap does (Genest, C. &
        Rémillard, B. (2008). Validity of the parametric bootstrap for
        goodness-of-fit testing in semiparametric models. *Ann. Inst. H.
        Poincaré Probab. Statist.* 44(6), 1096–1127) — refitting would mean
        running ICE B times. The null is therefore "the fitted model, taken
        as given, generated Y". Calibration under that null is checked by a
        regression test.

        Reported per pair: ``stat``, ``p``, the family, its τ, and ``n_eff``
        — the effective sample size ``Σw``, the pair's posterior mass
        (:func:`pmcprg.pmc.ice._effective_n`). A pair the chain barely
        visits gets a p-value built on very little, and the heatmap should
        say so rather than let it read like the others.

        Cost
        ----
        Each replicate simulates a whole series and re-runs forward-backward,
        so the bill is linear in ``N·B``: measured at ~126 ms per replicate
        for N = 800 but ~8.6 s for N = 65536 (a 256×256 image), i.e. 29
        minutes at B = 200 — not a thing to put behind a GUI button.
        ``max_len`` therefore bounds the series the test actually runs on. The
        window is **contiguous**: thinning it would destroy the lag-1
        dependence that is the whole object of the test. Replicates are
        simulated at the same length, so calibration is untouched; what is
        lost is sample size, reported as ``n_series``. For image data a
        contiguous stretch of the Hilbert scan is a compact spatial region,
        so the test then speaks about that region, not the whole image.

        ``max_points`` separately caps the O(n²) empirical copula; above it
        transitions are thinned by a uniform stride — never by weight, which
        would bias the very quantity under test — and the count is reported
        in ``n_used``.
        """
        from pmcprg.diagnostics   import parametric_bootstrap
        from pmcprg.pmc.ice       import _effective_n
        from pmcprg.pmc.inference import (backward, forward, joint_posteriors,
                                       precompute_weights)

        Y = np.asarray(Y)
        if len(Y) > max_len:
            Y = Y[:max_len]                # contiguous — see the Cost note
        N = len(Y)
        blocks = mdl.copula_blocks()
        pairs  = [(int(b["i"]), int(b["j"])) for b in blocks]
        cops   = {(i, j): mdl.copula(i, j) for i, j in pairs}

        stride = max(1, int(np.ceil((N - 1) / max_points)))
        sel    = np.arange(0, N - 1, stride)

        # K_θ is a property of the fitted copula, not of the data: one
        # Monte-Carlo per pair, reused across every replicate.
        k_refs: dict[tuple[int, int], np.ndarray] = {}
        if statistic == "kendall":
            for pr in pairs:
                try:
                    k_refs[pr] = _kendall_reference(cops[pr])
                except Exception:                              # pragma: no cover
                    logger.debug("K_theta failed for pair %s", pr)

        def _pair_stats(Yv):
            """Weighted CvM per pair for one series, under the fitted model."""
            W, f_pdf = precompute_weights(mdl, Yv)
            alpha, _ = forward(mdl, Yv, W=W, f_pdf=f_pdf)
            beta     = backward(mdl, Yv, W=W)
            xi       = joint_posteriors(alpha, W, beta)        # (N-1, K, K)
            F        = margin_cdfs(mdl, Yv, lo=1e-12, hi=1.0 - 1e-12)
            out = {}
            for i, j in pairs:
                w  = np.asarray(xi[sel, i, j], dtype=float)
                uv = np.column_stack((F[sel, i, j], F[sel + 1, j, i]))
                try:
                    out[(i, j)] = (
                        _gof_statistic(statistic, uv, w, cops[(i, j)],
                                       k_refs.get((i, j))),
                        float(_effective_n(w)),
                    )
                except Exception:
                    out[(i, j)] = (np.nan, float(_effective_n(w)))
            return out

        # Families without a closed-form CDF (Student) cannot be tested.
        skip: dict[tuple[int, int], str] = {}
        for i, j in pairs:
            try:
                cops[(i, j)].cdf([0.5, 0.5])
            except NotImplementedError as exc:
                skip[(i, j)] = str(exc)
            except Exception as exc:                           # pragma: no cover
                skip[(i, j)] = str(exc)

        # Effective sizes come from the observed pass; the harness carries
        # only the statistics.
        observed = _pair_stats(Y)
        results = parametric_bootstrap(
            mdl, Y,
            lambda m_, Y_, rng_: {pr: s for pr, (s, _n) in _pair_stats(Y_).items()},
            B=B, seed=seed, progress_cb=progress_cb,
        )

        rows: list[dict] = []
        for (i, j), blk in zip(pairs, blocks, strict=False):
            cop   = cops[(i, j)]
            S_n, n_eff = observed[(i, j)]
            base  = dict(i=i, j=j,
                         name=cop.copula_enum.value.SHORT_NAME,
                         tau=float(cop.params.get("tau_k", float("nan"))),
                         n_eff=float(n_eff), n_used=int(len(sel)),
                         n_series=int(N), statistic=statistic)
            if (i, j) in skip:
                rows.append(dict(**base, error=skip[(i, j)]))
                continue
            if n_eff < 4.0:
                # Almost no posterior mass on this pair: a p-value here would
                # be noise dressed as evidence.
                rows.append(dict(
                    **base,
                    error=f"negligible posterior mass (n_eff={n_eff:.2f})",
                ))
                continue
            res = results[(i, j)]
            rows.append(dict(**base, stat=float(S_n), p=res.p_value,
                             n_valid=res.n_valid))
        return rows

    def _on_gof_done(self, results: list[dict]):
        if not results:
            self._log_append("GoF: no copulas to test.")
            return
        # Text summary. n_eff is part of the reading, not decoration: a pair
        # the chain barely visits yields a p-value built on very little.
        # Report the replicates that actually completed, not the requested
        # count: a header claiming B=200 over 25 usable draws would misstate
        # how much the p-values rest on.
        n_valid = max((r.get("n_valid", 0) for r in results), default=0)
        n_series = next((r["n_series"] for r in results if "n_series" in r), None)
        window = (f", window {n_series}" if n_series is not None
                  and self._last_Y is not None
                  and n_series < len(self._last_Y) else "")
        stat_name = next((r["statistic"] for r in results if "statistic" in r),
                         _GOF_STATISTIC)
        label = {"kendall": "ξ-weighted Kendall-process CvM",
                 "cvm": "ξ-weighted Cramér-von Mises"}.get(stat_name, stat_name)
        self._log_append(
            f"GoF ({label}, model-level bootstrap, B={n_valid}{window}):"
        )
        for r in results:
            if "error" in r:
                self._log_append(
                    f"  ({r['i']},{r['j']})  {r['name']}: ERROR — {r['error']}"
                )
            else:
                verdict = "✓ accept" if r["p"] >= 0.05 else "✗ reject"
                self._log_append(
                    f"  ({r['i']},{r['j']})  {r['name']:<8} τ={r['tau']:+.3f}  "
                    f"stat={r['stat']:.3e}  p={r['p']:.3f}  "
                    f"n_eff={r['n_eff']:.1f}  {verdict}"
                )
        # Store state and switch to the heatmap view.
        self._gof_state = list(results)
        self._populate_view_selector()
        self._switch_view(_VIEW_GOF_HEATMAP)

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def _start_worker(
        self, func, *args, on_done,
        forward_progress: bool = False,
        progress_max: int | None = None,
        progress_label: str = "",
    ):
        if self._worker and self._worker.isRunning():
            return
        self._set_busy(True)

        # Configure the progress bar BEFORE showing it.
        self._progress_label = progress_label or "Working"
        if forward_progress and progress_max:
            self._progress.setRange(0, int(progress_max))
            self._progress.setValue(0)
            self._progress.setFormat(f"{self._progress_label} %v / %m")
        else:
            self._progress.setRange(0, 0)                # indeterminate
            self._progress.setFormat("")
        self._progress.setVisible(True)

        self._status.showMessage("Running …")

        self._worker = _Worker(func, *args)
        if forward_progress:
            # P2: use the public setter rather than mutating _kwargs directly.
            self._worker.set_progress_cb()
            self._worker.progress.connect(self._on_progress_update)
        self._worker.done.connect(lambda result: self._worker_done(result, on_done))
        self._worker.error.connect(self._worker_error)
        self._worker.start()

    def _on_progress_update(self, it: int, max_iter: int, value: float, run_tag: str):
        """Update the determinate progress bar.

        Works for both ICE (where ``value`` is the log-likelihood) and GoF
        (where ``value`` is unused / 0 and ``run_tag`` carries the pair
        label). The display is driven by ``self._progress_label`` set at
        worker start.
        """
        if max_iter and self._progress.maximum() != max_iter:
            self._progress.setRange(0, int(max_iter))
        self._progress.setValue(int(it) + 1)
        tag = f" [{run_tag}]" if run_tag else ""
        if value:
            self._status.showMessage(
                f"{self._progress_label}{tag} {it + 1}/{max_iter}  log-lik={value:.2f}"
            )
        else:
            self._status.showMessage(
                f"{self._progress_label}{tag} {it + 1}/{max_iter}"
            )

    def _worker_done(self, result, on_done):
        self._progress.setVisible(False)
        self._set_busy(False)
        self._status.showMessage("Done.")
        # S-1(c): ``on_done`` runs inside a Qt slot, where an escaping
        # exception aborts the process instead of surfacing anywhere. Route
        # it to the ordinary error path. The busy lock is already released
        # above, so the window stays usable whatever happens here.
        try:
            on_done(result)
        except Exception:
            logger.exception("Result handler raised")
            self._worker_error(traceback.format_exc())

    def _worker_error(self, tb: str):
        self._progress.setVisible(False)
        self._set_busy(False)
        self._status.showMessage("Error — see log.")
        # Write to Python logger (picked up by file handler and QTextEdit handler)
        logger.error("Worker thread raised an exception:\n%s", tb)
        # Also directly append to the log panel (fallback if widget handler not attached yet)
        self._log_append(f"ERROR:\n{tb}")
        # S-12: this used to paste ``tb[-1500:]`` — a traceback truncated at
        # the *head*, so the useful line ("marginal density of Y[0]=nan is
        # zero or non-finite …") was buried in stack frames. Lead with the
        # exception itself; the full traceback is already in the log panel.
        lines = tb.strip().splitlines()
        headline = lines[-1] if lines else "Unknown error"
        QMessageBox.critical(
            self, "Error",
            f"{headline}\n\nSee the log panel for the full traceback.",
        )

    # ------------------------------------------------------------------
    # Symmetrize feedback — fired by the prior tab's signal.
    # ------------------------------------------------------------------

    def _on_symmetrized(self, asym: float, was_renorm: bool):
        msg = f"Symmetrized prior: max asymmetry was {asym:.2e}"
        if was_renorm:
            msg += " (renormalised to sum=1)"
        self._log_append(msg)

    # ------------------------------------------------------------------
    # Window close — wait for the worker so we don't leak threads.
    # ------------------------------------------------------------------

    def closeEvent(self, event: QCloseEvent):  # noqa: N802 (Qt API)
        if not self._maybe_discard_changes("quit"):
            event.ignore()
            return
        if self._worker is not None and self._worker.isRunning():
            ans = QMessageBox.question(
                self, "Task running",
                "A computation is still running.\n"
                "  • Yes — wait for it to finish (up to 30 s)\n"
                "  • No  — terminate it and quit",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Yes,
            )
            if ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if ans == QMessageBox.StandardButton.Yes:
                ok = self._worker.wait(30_000)             # ms
                if not ok:
                    self._worker.terminate()
                    self._worker.wait()
            else:
                self._worker.terminate()
                self._worker.wait()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def _plot_simulation(self, X, Y, mdl):
        views.plot_simulation(self._begin_figure(), X, Y, mdl)
        self._canvas.draw()

    def _plot_classification(self, Y, X_hat, gamma, X_ref, mdl):
        views.plot_classification(self._begin_figure(), Y, X_hat, gamma, X_ref, mdl)
        self._canvas.draw()

    def _plot_posterior_draws(self, Y, X_hat, gamma, X_ref, mdl):
        views.plot_posterior_draws(self._begin_figure(), Y, X_hat, gamma, mdl)
        self._canvas.draw()

    def _plot_margin_ks(self, rows: list):
        Y, X_draw = self._margin_ks_sample
        views.plot_margin_ks(self._begin_figure(), rows, self._model,
                             Y, X_draw)
        self._canvas.draw()

    def _plot_tau_ci(self, rows: list):
        views.plot_tau_ci(self._begin_figure(), rows)
        self._canvas.draw()

    def _plot_loglik(self, lls: list):
        views.plot_loglik(self._begin_figure(), lls)
        self._canvas.draw()

    def _plot_image_raw(self, img: np.ndarray):
        views.plot_image_raw(self._begin_figure(), img)
        self._canvas.draw()

    def _plot_image_seg(self, img: np.ndarray, cls_state, ref_2d):
        views.plot_image_seg(self._begin_figure(), img, cls_state, ref_2d)
        self._canvas.draw()

    def _plot_gof_heatmap(self, results: list[dict]):
        views.plot_gof_heatmap(self._begin_figure(), results)
        self._canvas.draw()

    def _plot_ice_tau(self, trace: IceTrace):
        views.plot_ice_tau(self._begin_figure(), trace)
        self._canvas.draw()

    def _plot_ice_family(self, trace: IceTrace):
        views.plot_ice_family(self._begin_figure(), trace)
        self._canvas.draw()

    def _plot_ice_margin_family(self, trace: IceTrace):
        views.plot_ice_margin_family(self._begin_figure(), trace)
        self._canvas.draw()

    def _plot_ice_pseudos(self, trace: IceTrace, model: PMCModel, Y: np.ndarray):
        views.plot_ice_pseudos(self._begin_figure(), trace, model, Y)
        self._canvas.draw()

    def _plot_ice_multistart(self, trace: IceTrace):
        views.plot_ice_multistart(self._begin_figure(), trace)
        self._canvas.draw()

    def _plot_ice_prior(self, trace: IceTrace):
        views.plot_ice_prior(self._begin_figure(), trace)
        self._canvas.draw()

    def _plot_ice_margins(self, trace: IceTrace):
        views.plot_ice_margins(self._begin_figure(), trace)
        self._canvas.draw()

    def _plot_ice_dashboard(self, trace: IceTrace):
        views.plot_ice_dashboard(self._begin_figure(), trace, self._ice_Y)
        self._canvas.draw()

    def _plot_ice_pp(self, model: PMCModel, Y: np.ndarray):
        views.plot_ice_pp(self._begin_figure(), model, Y)
        self._canvas.draw()

    def _plot_ice_tail(self, model: PMCModel):
        views.plot_ice_tail(self._begin_figure(), model)
        self._canvas.draw()

    def _plot_ice_param_compare(self, init_mdl: PMCModel, fitted_mdl: PMCModel):
        views.plot_ice_param_compare(self._begin_figure(), init_mdl, fitted_mdl)
        self._canvas.draw()

    def _plot_ice_gamma_compare(self, init_mdl: PMCModel, fitted_mdl: PMCModel, Y: np.ndarray):
        views.plot_ice_gamma_compare(self._begin_figure(), init_mdl, fitted_mdl, Y)
        self._canvas.draw()

    # Each entry is a ``(available, render)`` pair of ``self``-taking callables
    # (audit Q-11): ``available(self) -> bool`` gates the view in the combobox
    # (:meth:`_is_view_available`) and ``render(self)`` paints it
    # (:meth:`_render_view`). Most ICE views just need a trace, captured once in
    # ``_avail_trace``; the copula-only and compare views carry their own.
    _avail_trace      = staticmethod(lambda self: self._ice_trace is not None)
    _avail_trace_cop  = staticmethod(lambda self: (
        self._ice_trace is not None and self._ice_Y is not None
        and self._model is not None and self._model.variant.uses_copula
    ))

    _VIEW_DISPATCH: dict[str, tuple[Callable[..., bool], Callable[..., None]]] = {
        _VIEW_SIMULATION: (
            lambda self: self._sim_state is not None,
            lambda self: self._plot_simulation(*self._sim_state),
        ),
        _VIEW_CLASSIFICATION: (
            lambda self: self._cls_state is not None,
            lambda self: self._plot_classification(*self._cls_state),
        ),
        _VIEW_MARGIN_KS: (
            lambda self: bool(self._margin_ks_state),
            lambda self: self._plot_margin_ks(self._margin_ks_state),
        ),
        _VIEW_TAU_CI: (
            lambda self: bool(self._tau_ci_state),
            lambda self: self._plot_tau_ci(self._tau_ci_state),
        ),
        _VIEW_CLS_POSTERIOR: (
            lambda self: self._cls_state is not None,
            lambda self: self._plot_posterior_draws(*self._cls_state),
        ),
        _VIEW_GOF_HEATMAP: (
            lambda self: self._gof_state is not None,
            lambda self: self._plot_gof_heatmap(self._gof_state),
        ),
        _VIEW_IMAGE_RAW: (
            lambda self: self._last_image is not None,
            lambda self: self._plot_image_raw(self._last_image),
        ),
        _VIEW_IMAGE_SEG: (
            lambda self: self._last_image is not None and self._cls_state is not None,
            lambda self: self._plot_image_seg(
                self._last_image, self._cls_state, self._last_image_ref,
            ),
        ),
        _ICE_VIEW_LOGLIK: (
            _avail_trace, lambda self: self._plot_loglik(self._ice_trace.log_liks),
        ),
        _ICE_VIEW_TAU: (
            _avail_trace, lambda self: self._plot_ice_tau(self._ice_trace),
        ),
        _ICE_VIEW_FAMILY: (
            _avail_trace, lambda self: self._plot_ice_family(self._ice_trace),
        ),
        _ICE_VIEW_MARGIN_FAMILY: (
            _avail_trace, lambda self: self._plot_ice_margin_family(self._ice_trace),
        ),
        _ICE_VIEW_PSEUDOS: (
            _avail_trace_cop,
            lambda self: self._plot_ice_pseudos(self._ice_trace, self._model, self._ice_Y),
        ),
        _ICE_VIEW_MULTISTART: (
            lambda self: self._ice_trace is not None
                          and len(self._ice_trace.multistart_runs) > 0,
            lambda self: self._plot_ice_multistart(self._ice_trace),
        ),
        _ICE_VIEW_PRIOR: (
            _avail_trace, lambda self: self._plot_ice_prior(self._ice_trace),
        ),
        _ICE_VIEW_MARGINS: (
            _avail_trace, lambda self: self._plot_ice_margins(self._ice_trace),
        ),
        _ICE_VIEW_DASHBOARD: (
            _avail_trace, lambda self: self._plot_ice_dashboard(self._ice_trace),
        ),
        _ICE_VIEW_PP: (
            _avail_trace_cop, lambda self: self._plot_ice_pp(self._model, self._ice_Y),
        ),
        _ICE_VIEW_TAIL: (
            lambda self: self._model is not None and self._model.variant.uses_copula,
            lambda self: self._plot_ice_tail(self._model),
        ),
        _ICE_VIEW_GAMMA: (
            lambda self: self._ice_trace is not None
                          and self._init_model is not None
                          and self._ice_Y is not None,
            lambda self: self._plot_ice_gamma_compare(
                self._init_model, self._model, self._ice_Y,
            ),
        ),
        _ICE_VIEW_PARAM_COMPARE: (
            lambda self: self._init_model is not None and self._model is not None,
            lambda self: self._plot_ice_param_compare(self._init_model, self._model),
        ),
        _ICE_VIEW_PLAYBACK: (
            _avail_trace, lambda self: self._setup_playback(self._ice_trace),
        ),
    }


# ---------------------------------------------------------------------------
# CSV reader for the data-load action — extracted so it can be unit-tested.
# ---------------------------------------------------------------------------

def _kendall_reference(cop, m_mc: int = _GOF_KENDALL_MC,
                       seed: int = 12345) -> np.ndarray:
    """``K_θ(t) = P(C_θ(U, V) ≤ t)`` on the evaluation grid.

    Estimated by Monte-Carlo from the fitted copula: ``m_mc`` = 20 000 draws
    at a fixed seed, so the per-node standard error of ``K_θ(t)`` is
    ``≤ √(0.25 / 20 000) ≈ 0.0035 < 0.005``. That error is *common* to the
    observed statistic and to every bootstrap replicate (the same
    ``k_ref`` is reused), so it shifts the whole null distribution and the
    observed value together; measured, it moves the p-value by a few
    hundredths between seeds. It depends only on the copula, never on the
    data, which is why the caller computes it **once per pair** and reuses
    it for all B replicates — recomputing it per replicate would triple the
    cost of the test for nothing.

    A closed form is not available for every one of the 40 families, but it
    is for two large groups. The Archimedean ones:
    ``K_θ(t) = t − φ(t) / φ'(t)`` with ``φ`` the generator (Genest & Rivest
    1993). And — since FR-9 added them — the extreme-value ones, for which
    ``K_θ(t) = t − (1 − τ) · t · log t`` whatever the Pickands function
    (Ghoudi, Khoudraji & Rivest 1998); checked here against 10⁵ draws for
    Galambos, Hüsler–Reiss, Tawn 1/2/3, t-EV and Gumbel–Hougaard (which is
    both), max deviation ≤ 0.0023 against a Monte-Carlo standard error of
    0.0016, while Clayton, Frank, BB6, survival Joe and Gaussian miss it by
    0.02–0.07. Using either form would remove the Monte-Carlo error for those
    families — a possible refinement, not done.
    """
    grid   = np.linspace(0.01, 0.99, _GOF_KENDALL_GRID_N)
    sample = cop.sample(n=m_mc, seed=seed)
    W      = cop.cdf_array(sample)
    return np.array([float(np.mean(W <= t)) for t in grid])


def _gof_statistic(name: str, uv: np.ndarray, w: np.ndarray, cop,
                   k_ref: np.ndarray | None = None) -> float:
    """One ξ-weighted goodness-of-fit distance.

    ``cvm``     — ``Σ w̃ (C_n − C_θ)²``, the classic Cramér-von Mises distance
                  between the empirical and fitted copula CDFs. Measured to be
                  nearly powerless against a family swap at matched τ.
    ``kendall`` — the same idea one level up, on the *Kendall process*: compare
                  the distribution of ``C(U, V)`` instead of ``C`` pointwise.
                  Two statistics that failed where this one works are on
                  record in ``report/gof_statistic_panel.py`` — variance
                  stabilisation (dividing by ``C_θ(1 − C_θ)`` to give the
                  corners their due) and a direct λ_L/λ_U discrepancy, the
                  latter defeated by the noise in λ̂ at extreme u.

    Which Kendall statistic, precisely (audit K-11)
    ----------------------------------------------
    * The distance is ``∫ (K_n − K_θ)² dw`` with ``w`` Lebesgue measure on a
      uniform grid of ``t`` — the Lebesgue-weighted form of Wang & Wells
      (2000), **not** the ``dK_θ``-weighted ``S_n = ∫ n (K_n − K_θ)² dK_θ``
      of Genest, Quessy & Rémillard (2006). The factor ``n`` is dropped:
      irrelevant under a model-level bootstrap at fixed N.
    * ``W_i = C_n^w(U_i, V_i)`` is the (weighted) empirical copula evaluated
      at the point itself, so the ``j = i`` term is **included**; Genest &
      Rivest (1993) exclude it (``1/(n−1) Σ_{j≠i}``). The difference is an
      O(1/n) bias, absorbed by the bootstrap since observed statistic and
      replicates share it.
    * ``K_θ`` is Monte-Carlo (see :func:`_kendall_reference` for its error
      and the closed-form Archimedean alternative not taken).

    References
    ----------
    * Genest, C. & Rivest, L.-P. (1993). Statistical inference procedures
      for bivariate Archimedean copulas. *JASA* 88(423), 1034–1043.
    * Wang, W. & Wells, M. T. (2000). Model selection and semiparametric
      inference for bivariate failure-time data. *JASA* 95(449), 62–72.
    * Genest, C., Quessy, J.-F. & Rémillard, B. (2006). Goodness-of-fit
      procedures for copula models based on the probability integral
      transformation. *Scand. J. Statist.* 33(2), 337–366.
    * Genest, C., Rémillard, B. & Beaudoin, D. (2009). Goodness-of-fit
      tests for copulas: A review and a power study. *Insurance: Mathematics
      and Economics* 44(2), 199–213 — the ``cvm`` form, ``S_n``, §2.
    """
    from pmcprg.copulas._fit import _empirical_copula, _normalised_weights

    wn = _normalised_weights(w)
    if name == "cvm":
        Cn = _empirical_copula(uv, uv, w)
        return float(np.dot(wn, (Cn - cop.cdf_array(uv)) ** 2))
    if name == "kendall":
        if k_ref is None:
            k_ref = _kendall_reference(cop)
        grid = np.linspace(0.01, 0.99, _GOF_KENDALL_GRID_N)
        Wn   = _empirical_copula(uv, uv, w)
        Kn   = np.array([float(np.sum(wn[Wn <= t])) for t in grid])
        return float(np.mean((Kn - k_ref) ** 2))
    raise ValueError(f"unknown GoF statistic {name!r}")


def _trace_rows(trace, run: str = "best") -> tuple[list[str], list[list]]:
    """Flatten one :class:`IceTrace` into a tidy per-iteration table.

    One row per completed E-step; ``tau_i_j`` / ``p_i_j`` / ``family_i_j``
    columns for every state pair, plus one column per margin parameter —
    ``margin{b}_<param>`` for the state block number b, ``margin_{i}_{j}_<param>``
    for the pair block f_ij. Vector-valued margin parameters (a multivariate
    mean, a covariance) are expanded component-wise rather than dropped.
    """
    T = len(trace.log_liks)
    header = ["run", "iteration", "log_lik"]
    tau = np.asarray(trace.tau_history, dtype=float)
    pj  = np.asarray(trace.p_history,   dtype=float)
    K   = tau.shape[1] if tau.ndim == 3 and tau.size else 0
    for i in range(K):
        for j in range(K):
            header += [f"tau_{i}_{j}", f"p_{i}_{j}", f"family_{i}_{j}"]

    margin_cols: list[tuple[int, str, int]] = []          # (block, param, comp)
    if trace.margin_history:
        for b, blk in enumerate(trace.margin_history[0]):
            for pname, pval in blk.get("params", {}).items():
                try:
                    size = int(np.asarray(pval, dtype=float).size)
                except (TypeError, ValueError):
                    size = 1
                tag = (f"margin_{blk.get('i')}_{blk['j']}" if "j" in blk
                       else f"margin{b}")
                for c in range(size):
                    margin_cols.append((b, pname, c))
                    suffix = f"_{c}" if size > 1 else ""
                    header.append(f"{tag}_{pname}{suffix}")

    rows: list[list] = []
    for it in range(T):
        row: list = [run, it, float(trace.log_liks[it])]
        for i in range(K):
            for j in range(K):
                fam = ""
                if it < len(trace.family_history):
                    try:
                        fam = trace.family_history[it][i][j]
                    except (IndexError, TypeError):
                        fam = ""
                row += [float(tau[it, i, j]) if tau.size else "",
                        float(pj[it, i, j]) if pj.size else "",
                        fam]
        for b, pname, c in margin_cols:
            val = ""
            if it < len(trace.margin_history):
                try:
                    flat = np.ravel(np.asarray(
                        trace.margin_history[it][b]["params"][pname], dtype=float))
                    val = float(flat[c]) if c < flat.size else ""
                except (TypeError, ValueError, KeyError, IndexError):
                    val = ""
            row.append(val)
        rows.append(row)
    return header, rows


def _jsonable(obj):
    """numpy → builtin, so :mod:`json` can serialise a trace verbatim."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _read_data_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a (Y[, X]) CSV. Returns ``(Y, X_or_None)``.

    Raises ``ValueError`` for empty files, missing ``Y`` column, or
    non-numeric ``Y`` cells (with a row-number-aware message).
    Non-integer ``X`` cells trigger a logger warning and ``X = None`` —
    matching the GUI's "labels are optional" semantics.

    Missing observations — an empty cell, ``NaN``/``nan`` or ``NA``
    (:data:`pmcprg.missing.cells.MISSING_TOKENS`) — load as float NaN, with a
    logger warning giving their count and share. A multivariate row with some
    components missing is kept, NaN in those components (inference treats
    the whole row as missing). Still refused: ``±inf``, and a column with no
    observed value at all.
    """
    rows: list[dict] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError("CSV file is empty.")

    # Scalar observations live in a 'Y' column; multivariate ones (d ≥ 2, the
    # multivariate-normal margins) in 'Y0'…'Y{d-1}'. Without the second form
    # there was no way at all to bring d ≥ 2 data into the GUI.
    vec_cols = sorted(
        (c for c in rows[0] if re.fullmatch(r"Y_?\d+", c or "")),
        key=lambda c: int(re.sub(r"\D", "", c)),
    )
    if "Y" not in rows[0] and not vec_cols:
        raise ValueError(
            "CSV must contain a 'Y' column, or 'Y0', 'Y1', … for "
            "multivariate observations."
        )
    cols = ["Y"] if "Y" in rows[0] else vec_cols

    def _column(name: str) -> np.ndarray:
        try:
            return np.array([parse_cell(r[name]) for r in rows])
        except (TypeError, ValueError) as exc:
            for n, r in enumerate(rows):
                try:
                    parse_cell(r[name])
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Non-numeric {name} at row {n + 2} (CSV line "
                        f"{n + 2}): {r[name]!r}"
                    ) from exc
            raise

    Y = (_column("Y") if cols == ["Y"]
         else np.column_stack([_column(c) for c in cols]))
    bad = np.argwhere(np.isinf(Y))
    if bad.size:
        # S-13: non-finite observations used to load silently and only fail
        # at the next click. Name the first offending cell, in the same
        # row-precise style as the non-numeric message above.
        row = int(bad[0][0])
        col = cols[int(bad[0][1])] if Y.ndim == 2 else cols[0]
        raise ValueError(
            f"Non-finite {col} at row {row + 2} (CSV line {row + 2}): "
            f"{float(Y[tuple(bad[0])])}. Observations must be finite or "
            f"missing (empty, NaN, NA)."
        )
    missing = np.isnan(Y)
    if missing.any():
        Y2 = missing if missing.ndim == 2 else missing[:, None]
        for j, col in enumerate(cols):
            if Y2[:, j].all():
                # S-13 again: an all-NaN column is not data. The first cell is
                # named in the same row-precise style.
                raise ValueError(
                    f"Non-finite {col} at row 2 (CSV line 2): nan. Every "
                    f"{col} value is missing ({len(rows)} of {len(rows)} rows "
                    f"empty, NaN or NA); at least one observed value is "
                    f"required."
                )
        n_rows = int(Y2.any(axis=1).sum())
        logger.warning(
            "Load data: %d of %d rows of %s have missing observations "
            "(%.1f %%, %d missing cell(s)) — kept as NaN.",
            n_rows, len(rows), path, 100.0 * n_rows / len(rows),
            int(missing.sum()),
        )
    if Y.ndim == 2 and Y.shape[1] == 1:
        # A lone 'Y0' column describes *scalar* observations written in the
        # vector style. Returning (N, 1) made every downstream scalar path
        # fail — "could not broadcast (30,1) into (30,)" — and the dimension
        # check waved it through, since d = 1 either way.
        Y = Y[:, 0]

    X: np.ndarray | None = None
    if "X" in rows[0]:
        try:
            X = np.array([int(r["X"]) for r in rows])
        except ValueError as exc:
            logger.warning(
                "Load data: non-integer X column in %s — labels ignored (%s)",
                path, exc,
            )
            X = None
    return Y, X


