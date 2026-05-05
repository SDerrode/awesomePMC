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
mapping a view name (``"Simulation"``, ``"ICE — Dashboard"``, …) to a
zero-argument callable that re-renders that view from instance state.
Each action handler (``_on_sim_done``, ``_on_cls_done``, ``_on_gof_done``,
``_on_est_done``) stores the data it produced on ``self`` and calls
:meth:`_switch_view`, which:

1. Looks up the handler in :attr:`_VIEW_DISPATCH`.
2. Calls :meth:`_begin_figure` (clears the canvas).
3. Invokes the handler with ``self`` only.
4. On success, calls :meth:`_finalize_plot` (toggles Export-plot on).
   On exception, calls :meth:`_render_error` which paints a friendly
   error message on the canvas + log panel.

Adding a new view is therefore: declare its name, add its data state
attribute, append the entry to :attr:`_VIEW_DISPATCH`, write
``_plot_<name>`` — no other change required.

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
from pathlib import Path

# Qt setup *before* any pyplot import.
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

from PyQt6.QtCore    import QSettings, Qt, QTimer            # noqa: E402
from PyQt6.QtGui     import QCloseEvent, QKeySequence, QShortcut  # noqa: E402
from PyQt6.QtWidgets import (                                # noqa: E402
    QComboBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QProgressBar,
    QPushButton, QSizePolicy, QSlider, QSpinBox, QSplitter, QStatusBar,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure                              # noqa: E402

from prg                 import __version__ as _PKG_VERSION    # noqa: E402
from prg.copulas._fit    import _empirical_copula              # noqa: E402
from prg.numerics        import EPS, ONE_MINUS_EPS             # noqa: E402
from prg.pmc.gui.tabs    import PriorTabError, _CopulaTab, _IceTab, _MarginTab, _PriorTab  # noqa: E402
from prg.pmc.gui.worker  import _Worker                        # noqa: E402
from prg.pmc.ice         import IceTrace                       # noqa: E402
from prg.pmc.inference   import classify, error_rate           # noqa: E402
from prg.pmc.model       import PMCModel                       # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# View identifiers — keys for the View QComboBox and ``_switch_view`` dispatch
# ---------------------------------------------------------------------------

_VIEW_NONE             = "—"
_VIEW_SIMULATION       = "Simulation"
_VIEW_CLASSIFICATION   = "Classification"
_VIEW_GOF_HEATMAP      = "GoF — p-value heatmap"
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

# Display order in the combobox (None entries denote separators visually).
_ALL_VIEWS = [
    _VIEW_SIMULATION, _VIEW_CLASSIFICATION, _VIEW_GOF_HEATMAP,
    _ICE_VIEW_DASHBOARD, _ICE_VIEW_LOGLIK, _ICE_VIEW_TAU, _ICE_VIEW_FAMILY,
    _ICE_VIEW_MARGIN_FAMILY,
    _ICE_VIEW_PSEUDOS, _ICE_VIEW_MULTISTART, _ICE_VIEW_PRIOR,
    _ICE_VIEW_MARGINS, _ICE_VIEW_PP, _ICE_VIEW_TAIL, _ICE_VIEW_GAMMA,
    _ICE_VIEW_PLAYBACK,
]

# ICE-trace views need an :class:`IceTrace`; everything else needs different
# state. The :meth:`PMCMainWindow._is_view_available` method checks per-view.
_ICE_VIEWS = {
    _ICE_VIEW_LOGLIK, _ICE_VIEW_TAU, _ICE_VIEW_FAMILY, _ICE_VIEW_PSEUDOS,
    _ICE_VIEW_MULTISTART, _ICE_VIEW_PRIOR, _ICE_VIEW_MARGINS,
    _ICE_VIEW_DASHBOARD, _ICE_VIEW_PP, _ICE_VIEW_TAIL, _ICE_VIEW_GAMMA,
    _ICE_VIEW_PLAYBACK, _ICE_VIEW_MARGIN_FAMILY,
}

# Views that can scrub through iterations via a slider. The "Animated playback"
# view itself shows the slider + play button explicitly; for the others, the
# slider remains hidden (the static plot is rendered).
_PLAYBACK_VIEW = _ICE_VIEW_PLAYBACK


# Persistent settings — used for the Recent-files submenu and any future
# GUI-level preferences. The "organization" + "application" identifiers
# determine where Qt stores the data (per-user, OS-dependent).
_QS_ORG = "copulasformm"
_QS_APP = "pmc-gui"
_QS_RECENT_KEY = "recent_files"
_RECENT_MAX = 5


# ---------------------------------------------------------------------------
# Matplotlib canvas widget
# ---------------------------------------------------------------------------

class _Canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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

        if model_path:
            self._load_model(model_path)

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
        act_save.triggered.connect(self._on_save)

        act_saveas = file.addAction("Save &As…")
        act_saveas.setShortcut("Ctrl+Shift+S")
        act_saveas.triggered.connect(self._on_save_as)

        file.addSeparator()

        act_load_data = file.addAction("Load &data (CSV)…")
        act_load_data.triggered.connect(self._on_load_data)

        act_save_data = file.addAction("Sa&ve data (CSV)…")
        act_save_data.setToolTip("Save the last simulated/loaded (X, Y) sequence.")
        act_save_data.triggered.connect(self._on_save_data)
        self._act_save_data = act_save_data

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

        hlp = bar.addMenu("&Help")
        hlp.addAction("About").triggered.connect(self._on_about)

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

        # Reset shortcut as a global QShortcut as well (in addition to the
        # menu action) so the GUI tests can trigger it without a menu bar.
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._on_reset)

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
        """Return True if the view's underlying data is currently in memory."""
        if name == _VIEW_SIMULATION:
            return self._sim_state is not None
        if name == _VIEW_CLASSIFICATION:
            return self._cls_state is not None
        if name == _VIEW_GOF_HEATMAP:
            return self._gof_state is not None
        if name == _ICE_VIEW_MULTISTART:
            return (self._ice_trace is not None
                    and len(self._ice_trace.multistart_runs) > 0)
        if name == _ICE_VIEW_GAMMA:
            return (self._ice_trace is not None
                    and self._init_model is not None
                    and self._ice_Y is not None)
        if name in (_ICE_VIEW_PSEUDOS, _ICE_VIEW_PP):
            return (self._ice_trace is not None
                    and self._ice_Y is not None
                    and self._model is not None
                    and self._model.variant.uses_copula)
        if name == _ICE_VIEW_TAIL:
            return (self._model is not None
                    and self._model.variant.uses_copula)
        if name in _ICE_VIEWS:
            return self._ice_trace is not None
        return False

    def _populate_view_selector(self):
        """Refresh the View QComboBox, disabling unavailable items.

        We always rebuild the full list (small) so the order is stable;
        items whose data is not in memory are inserted but disabled.
        """
        # Suppress currentTextChanged while we populate.
        self._switching_view = True
        try:
            current = self._cmb_view.currentText() if self._cmb_view.count() else ""
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
            # Restore selection if still valid; otherwise pick the first
            # available view.
            if current in _ALL_VIEWS and self._is_view_available(current):
                self._cmb_view.setCurrentText(current)
            else:
                for view in _ALL_VIEWS:
                    if self._is_view_available(view):
                        self._cmb_view.setCurrentText(view)
                        break
        finally:
            self._switching_view = False

    def _on_view_changed(self, name: str):
        if self._switching_view or not name:
            return
        self._switch_view(name)

    def _switch_view(self, name: str):
        """Programmatic view switch: update combobox + render."""
        if not self._is_view_available(name):
            return
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

    def _render_error(self, name: str, exc: Exception):
        """Surface a render failure to the canvas + log panel.

        Replaces the previous behaviour where exceptions inside a render
        method were silently logged at DEBUG level — the user would see a
        blank canvas with no explanation.
        """
        logger.exception("Render failed for view %r", name)
        self._log_append(f"ERROR: render of {name!r} failed — {exc}")
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

    def _render_view(self, name: str):
        """Dispatch to the matching ``_plot_*`` method via the dispatch table."""
        handler = self._VIEW_DISPATCH.get(name)
        if handler is None:
            self._canvas.clear()
            return
        try:
            handler(self)
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
        self._draw_tau_trajectories(ax2, trace, vline=t)
        # Bottom-left: family ribbon up to t
        ax3 = fig.add_subplot(2, 2, 3)
        self._draw_family_ribbon(ax3, trace, t_max=t)
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
            ax4.set_title(f"Joint prior p (iter {t})")
        fig.suptitle(f"ICE iteration {t} / {trace.n_iters - 1}", y=0.995)
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

    def _load_model(self, path: str | Path):
        # B6: only commit ``_model_path`` once the model has been
        # successfully built. Previously we set the path *before* calling
        # PMCModel(), so a parse error left the GUI in an inconsistent
        # state (path set, model None).
        try:
            new_path  = Path(path)
            new_model = PMCModel(new_path)
        except Exception as exc:
            logger.exception("Failed to load model from %s", path)
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._model_path = new_path
        self._model      = new_model
        self._sync_widgets_from_model()
        self._set_buttons_enabled(True)
        self._mark_clean()
        self._push_recent(new_path)
        self._status.showMessage(f"Loaded: {self._model_path}")
        self._log_append(f"Loaded model: {self._model}")

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
        finally:
            self.blockSignals(was_blocked)

        self._tab_prior.load(mdl)
        self._tab_margins.load(mdl)
        self._tab_copulas.load(mdl)

        self._tab_ice.load(mdl.ice_config(), mdl.variant.uses_copula)

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

        # Prior — may raise PriorTabError
        self._tab_prior.save(raw)
        # Margins
        self._tab_margins.save(raw)
        # Copulas
        if self._model.variant.uses_copula:
            self._tab_copulas.save(raw)
        # ICE
        raw["ice"] = self._tab_ice.get_cfg()

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

    # ------------------------------------------------------------------
    # Menu actions — file
    # ------------------------------------------------------------------

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open model", "", "TOML files (*.toml);;All files (*)"
        )
        if path:
            self._load_model(path)

    def _on_save(self):
        if self._model is None:
            return
        if self._model_path is None:
            self._on_save_as()
            return
        try:
            mdl = self._rebuild_model_from_widgets()
            mdl.save(self._model_path)
            self._mark_clean()
            self._status.showMessage(f"Saved: {self._model_path}")
            self._push_recent(self._model_path)
        except PriorTabError as exc:
            QMessageBox.warning(self, "Invalid prior", str(exc))
        except Exception as exc:
            logger.exception("Failed to save model to %s", self._model_path)
            QMessageBox.critical(self, "Save Error", str(exc))

    def _on_save_as(self):
        if self._model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model as", "", "TOML files (*.toml);;All files (*)"
        )
        if path:
            self._model_path = Path(path)
            self._on_save()

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
        self._last_Y = Y
        self._last_X = X
        self._act_save_data.setEnabled(True)
        self._log_append(
            f"Loaded data: N={len(Y)}  "
            f"{'(X labels present)' if X is not None else '(no X labels)'}"
        )
        self._status.showMessage(f"Data loaded: {path}  N={len(Y)}")

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
            Y = self._last_Y
            X = self._last_X
            with open(path, "w", newline="") as fh:
                writer = csv.writer(fh)
                if X is not None:
                    writer.writerow(["n", "X", "Y"])
                    for n in range(len(Y)):
                        writer.writerow([n + 1, int(X[n]), float(Y[n])])
                else:
                    writer.writerow(["n", "Y"])
                    for n in range(len(Y)):
                        writer.writerow([n + 1, float(Y[n])])
            self._log_append(f"Saved data: {path}  N={len(Y)}")
            self._status.showMessage(f"Data saved: {path}")
        except Exception as exc:
            logger.exception("Failed to save data to %s", path)
            QMessageBox.critical(self, "Save Error", str(exc))

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
        self._last_X = None
        self._last_Y = None
        self._canvas.clear()
        self._has_plot = False
        self._log.clear()
        self._status.showMessage("Session reset.")
        self._set_buttons_enabled(self._model is not None)

    def _on_about(self):
        QMessageBox.information(
            self, "About PMC GUI",
            f"PMC / HMC Copula Model Tool — v{_PKG_VERSION}\n"
            "prg.pmc — Pairwise Markov Chain with copula-based transitions.\n\n"
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
        from prg.pmc.simulate import simulate
        X, Y = simulate(mdl, N=N, seed=seed)
        return X, Y, mdl

    def _on_sim_done(self, result):
        X, Y, mdl = result
        self._last_X = X
        self._last_Y = Y
        self._model  = mdl
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

        Y = self._last_Y
        self._start_worker(
            self._do_classify, mdl, Y,
            on_done=self._on_cls_done,
        )

    @staticmethod
    def _do_classify(mdl, Y):
        from prg.pmc.inference import classify
        X_hat, gamma, ll = classify(mdl, Y)
        return X_hat, gamma, ll, mdl

    def _on_cls_done(self, result):
        X_hat, gamma, ll, mdl = result
        msg = f"Classification  log-lik={ll:.2f}"
        if self._last_X is not None:
            er = error_rate(self._last_X, X_hat)
            msg += f"  error={er:.4f} ({er*100:.1f}%)"
        self._log_append(msg)
        self._cls_state = (self._last_Y, X_hat, gamma, self._last_X, mdl)
        self._populate_view_selector()
        self._switch_view(_VIEW_CLASSIFICATION)

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

        Y       = self._last_Y
        ice_cfg = self._tab_ice.get_cfg()
        # ICE uses a determinate progress bar (we know max_iter ahead of time).
        max_iter = int(ice_cfg.get("max_iter", 50))
        self._start_worker(
            self._do_estimate, mdl, Y, ice_cfg,
            on_done=self._on_est_done,
            forward_progress=True,
            progress_max=max_iter,
            progress_label="ICE",
        )

    @staticmethod
    def _do_estimate(mdl, Y, ice_cfg, *, progress_cb=None):
        from prg.pmc.ice import IceResult, ice
        fitted, trace = ice(mdl, Y, ice_cfg=ice_cfg, progress_cb=progress_cb)
        # Pack everything the GUI needs into a single typed bundle: the
        # initial model is needed for view J (γ before/after), and Y for
        # views C, H, J.
        return IceResult(
            initial_model=mdl, fitted_model=fitted, Y=Y, trace=trace,
        )

    def _on_est_done(self, result):
        # `result` is an :class:`IceResult` from the worker.
        self._model       = result.fitted_model
        self._init_model  = result.initial_model      # for view J
        self._ice_trace   = result.trace              # for views A–K
        self._ice_Y       = result.Y                  # for views C, H, J
        trace = result.trace
        self._sync_widgets_from_model()
        # ICE returned a fresh model — that *is* a state change, but we
        # treat it as the new clean baseline (the user will Save explicitly).
        self._mark_clean()
        lls = trace.log_liks
        n_runs = 1 + len(trace.multistart_runs)
        msg = (
            f"ICE done  iters={len(lls)}  "
            f"LL: {lls[0]:.2f} → {lls[-1]:.2f}"
        )
        if n_runs > 1:
            msg += f"  (multistart: {n_runs} runs)"
        self._log_append(msg)
        # Refresh the View selector (ICE views just became available) and
        # auto-switch to the dashboard for an at-a-glance summary.
        self._populate_view_selector()
        self._switch_view(_ICE_VIEW_DASHBOARD)

    # ------------------------------------------------------------------
    # Action: GoF test — emits per-pair progress, plots a heatmap.
    # ------------------------------------------------------------------

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
        n_pairs = len(mdl.copula_blocks())
        self._start_worker(
            self._do_gof_test, mdl, Y,
            on_done=self._on_gof_done,
            forward_progress=True,
            progress_max=max(n_pairs, 1),
            progress_label="GoF",
        )

    @staticmethod
    def _do_gof_test(mdl, Y, B: int = 200, *, progress_cb=None):
        """Run a Cramér-von Mises GoF test on each pair-copula.

        The optional ``progress_cb(idx, n_pairs, 0.0, "(i,j)")`` is called
        before each pair starts, so the GUI can advance a determinate bar
        across pairs. Each pair internally runs ``B`` bootstrap replicates;
        we don't surface that finer granularity (it would dwarf the per-pair
        signal and make the bar misleading).
        """
        results: list[dict] = []
        N = len(Y)
        # Pre-compute marginal CDFs once (expensive for large N).
        f_cdf = _compute_marginal_cdfs(mdl, Y)

        blocks = mdl.copula_blocks()
        n_pairs = len(blocks)
        for idx, blk in enumerate(blocks):
            i = int(blk["i"]); j = int(blk["j"])
            cop = mdl.copula(i, j)
            short = cop.copula_enum.value.SHORT_NAME
            if progress_cb is not None:
                try:
                    progress_cb(idx, n_pairs, 0.0, f"({i},{j}) {short}")
                except Exception:                              # pragma: no cover
                    pass
            u  = f_cdf[: N - 1, i, j]
            v  = f_cdf[1:,      j, i]
            uv = np.column_stack((u, v))
            try:
                # Refit on the pseudos and run the parametric-bootstrap GoF.
                cls    = type(cop)
                fit_r  = cls.fit(uv, method="mle")
                gof_r  = fit_r.gof_test(B=B, seed=0)
                results.append(dict(
                    i=i, j=j, name=short,
                    tau=float(fit_r.tau_k),
                    stat=float(gof_r.statistic),
                    p=float(gof_r.p_value),
                    n_valid=int(gof_r.n_valid_bootstrap),
                ))
            except NotImplementedError as exc:
                # Family lacks a closed-form CDF (e.g. Student) — skip cleanly.
                results.append(dict(i=i, j=j, name=short, error=str(exc)))
            except Exception as exc:                       # pragma: no cover
                results.append(dict(i=i, j=j, name=short, error=str(exc)))
        return results

    def _on_gof_done(self, results: list[dict]):
        if not results:
            self._log_append("GoF: no copulas to test.")
            return
        # Text summary
        self._log_append("GoF (Cramér-von Mises, B=200):")
        for r in results:
            if "error" in r:
                self._log_append(
                    f"  ({r['i']},{r['j']})  {r['name']}: ERROR — {r['error']}"
                )
            else:
                verdict = "✓ accept" if r["p"] >= 0.05 else "✗ reject"
                self._log_append(
                    f"  ({r['i']},{r['j']})  {r['name']:<8} τ={r['tau']:+.3f}  "
                    f"stat={r['stat']:.4f}  p={r['p']:.3f}  {verdict}"
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
        self._set_buttons_enabled(False)

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
        self._set_buttons_enabled(True)
        self._status.showMessage("Done.")
        on_done(result)

    def _worker_error(self, tb: str):
        self._progress.setVisible(False)
        self._set_buttons_enabled(True)
        self._status.showMessage("Error — see log.")
        # Write to Python logger (picked up by file handler and QTextEdit handler)
        logger.error("Worker thread raised an exception:\n%s", tb)
        # Also directly append to the log panel (fallback if widget handler not attached yet)
        self._log_append(f"ERROR:\n{tb}")
        QMessageBox.critical(self, "Error", tb[-1500:])

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
        fig = self._begin_figure()
        K = mdl.K
        colors = _class_colors(K)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 1, 2)

        # Scatter: index vs Y, coloured by X
        nn = np.arange(len(Y))
        for k in range(K):
            mask = X == k
            ax1.scatter(nn[mask], Y[mask], s=4, alpha=0.5, color=colors[k],
                        label=f"X={k}")
        ax1.set_xlabel("n"); ax1.set_ylabel("Y"); ax1.set_title("Simulated sequence")
        ax1.legend(markerscale=3, fontsize=8)

        # Histogram per class
        for k in range(K):
            vals = Y[X == k]
            if len(vals):
                ax2.hist(vals, bins=40, alpha=0.5, color=colors[k],
                         label=f"X={k}", density=True)
        ax2.set_xlabel("Y"); ax2.set_title("Marginal histograms")
        ax2.legend(fontsize=8)

        # Successive pairs (Y_n, Y_{n+1})
        for k in range(K):
            mask = (X[:-1] == k)
            ax3.scatter(Y[:-1][mask], Y[1:][mask], s=4, alpha=0.4,
                        color=colors[k], label=f"X_n={k}")
        ax3.set_xlabel("Y_n"); ax3.set_ylabel("Y_{n+1}")
        ax3.set_title("Successive-pair scatter (Y_n, Y_{n+1})")
        ax3.legend(markerscale=3, fontsize=8)

        self._canvas.draw()

    def _plot_classification(self, Y, X_hat, gamma, X_ref, mdl):
        fig = self._begin_figure()
        K = mdl.K
        colors = _class_colors(K)

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 1, 2)
        nn  = np.arange(len(Y))

        # Sequence coloured by X_hat
        for k in range(K):
            mask = X_hat == k
            ax1.scatter(nn[mask], Y[mask], s=4, alpha=0.5, color=colors[k],
                        label=f"X̂={k}")
        ax1.set_xlabel("n"); ax1.set_ylabel("Y")
        ax1.set_title("MPM classification")
        ax1.legend(markerscale=3, fontsize=8)

        # Posterior marginals γ
        for k in range(K):
            ax2.plot(nn, gamma[:, k], lw=0.7, alpha=0.8, color=colors[k],
                     label=f"γ(X={k}|Y)")
        ax2.set_xlabel("n"); ax2.set_ylabel("P(X_n=k | Y)")
        ax2.set_title("Posterior marginals")
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=8)

        # Errors vs reference (if available)
        if X_ref is not None:
            errors = (X_hat != X_ref).astype(float)
            ax3.fill_between(nn, errors, alpha=0.3, color="red", label="errors")
            ax3.plot(nn, errors, lw=0.5, color="red")
            ax3.set_xlabel("n"); ax3.set_ylabel("error")
            ax3.set_title(f"Classification errors  (rate={errors.mean():.4f})")
            ax3.set_ylim(-0.1, 1.1)
            ax3.legend(fontsize=8)
        else:
            # Show histogram of max posterior
            max_gamma = gamma.max(axis=1)
            ax3.hist(max_gamma, bins=50, color="steelblue", edgecolor="white")
            ax3.set_xlabel("max γ_n"); ax3.set_ylabel("count")
            ax3.set_title("Confidence histogram (max posterior)")

        self._canvas.draw()

    def _plot_loglik(self, lls: list):
        fig = self._begin_figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(lls, "o-", color="steelblue", lw=2, markersize=6)
        ax.set_xlabel("ICE iteration"); ax.set_ylabel("Log-likelihood")
        ax.set_title("ICE convergence")
        ax.grid(True, alpha=0.3)
        self._canvas.draw()

    def _plot_gof_heatmap(self, results: list[dict]):
        """K×K p-value heatmap for the GoF test results.

        Errored pairs are shown as grey ‘×’ markers; valid pairs are
        coloured from red (p≈0, reject) through yellow (p≈0.05, edge) to
        green (p≈1, accept). The 0.05 contour is highlighted.
        """
        if not results:
            return
        # Recover K from the indices.
        K = max(max(r["i"], r["j"]) for r in results) + 1
        P = np.full((K, K), np.nan)
        labels = np.full((K, K), "", dtype=object)
        errored: list[tuple[int, int]] = []
        for r in results:
            i, j = int(r["i"]), int(r["j"])
            if "error" in r:
                errored.append((i, j))
                labels[i, j] = f"{r['name']}\nERR"
            else:
                P[i, j] = float(r["p"])
                labels[i, j] = f"{r['name']}\np={r['p']:.2f}"

        fig = self._begin_figure()
        ax = fig.add_subplot(1, 1, 1)

        # Colour map: red → yellow → green via "RdYlGn"; clip vmin/vmax for
        # numerical stability.
        cmap = plt.get_cmap("RdYlGn")
        im = ax.imshow(P, cmap=cmap, vmin=0.0, vmax=1.0,
                       aspect="equal", origin="upper")
        # The 0.05 acceptance threshold as a contour line.
        if np.isfinite(P).any():
            ax.contour(P, levels=[0.05], colors="black", linewidths=1.0)

        for i in range(K):
            for j in range(K):
                ax.text(j, i, labels[i, j],
                        ha="center", va="center", fontsize=8,
                        color="black")
        for (ie, je) in errored:
            ax.plot(je, ie, marker="x", markersize=22, color="dimgrey",
                    markeredgewidth=2)

        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([f"j={k}" for k in range(K)])
        ax.set_yticklabels([f"i={k}" for k in range(K)])
        ax.set_title("Goodness-of-fit p-values  (CvM, parametric bootstrap)")

        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        cbar.set_label("p-value")
        # Mark the 0.05 reject/accept threshold on the colour bar.
        cbar.ax.axhline(0.05, color="black", linewidth=1.0)

        self._canvas.draw()


    # ==================================================================
    # ICE-trace visualisations (views A–K)
    # ==================================================================

    # ----- internal drawing helpers (shared with the dashboard / playback) -----

    def _draw_tau_trajectories(self, ax, trace: IceTrace, vline: int | None = None):
        """Draw τ_{ij}(iter) curves with family-change markers on ``ax``."""
        if trace.tau_history is None or trace.tau_history.size == 0:
            ax.text(0.5, 0.5, "no copulas in this variant",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            return
        T, K, _ = trace.tau_history.shape
        nn = np.arange(T)
        for i in range(K):
            for j in range(K):
                series = trace.tau_history[:, i, j]
                if not np.any(np.isfinite(series)):
                    continue
                ax.plot(nn, series, "-",
                        label=f"({i},{j}) {trace.family_history[-1][i][j]}")
                # ★ markers where family changed
                fam_series = [trace.family_history[t][i][j] for t in range(T)]
                for t in range(1, T):
                    if fam_series[t] and fam_series[t] != fam_series[t - 1]:
                        ax.plot(t, series[t], "*", markersize=10,
                                color="black", markerfacecolor="yellow")
        if vline is not None:
            ax.axvline(vline, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("ICE iteration")
        ax.set_ylabel("τ_K")
        ax.set_title("τ trajectories per pair  (★ = family change)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    def _draw_family_ribbon(self, ax, trace: IceTrace, t_max: int | None = None):
        """Draw the family-selection ribbon on ``ax``.

        ``t_max`` truncates the ribbon to that iteration (used by the
        playback view); pass None to show the full history.
        """
        if not trace.family_history:
            ax.text(0.5, 0.5, "no copulas", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
            return
        T = len(trace.family_history)
        K = len(trace.family_history[0])
        # Build colour map from the *candidate* list (stable order across runs).
        candidates = trace.candidates or []
        # Plus any family that ended up selected even if not in candidates
        # (shouldn't happen, but defensive).
        seen = sorted({fam for row in trace.family_history
                       for r in row for fam in r if fam})
        all_fams = candidates + [f for f in seen if f not in candidates]
        cmap = plt.get_cmap("tab10")
        fam_color = {f: cmap(k % 10) for k, f in enumerate(all_fams)}

        # Map each (i, j) to a display row (skip pairs that never have a copula).
        row_idx = []
        for i in range(K):
            for j in range(K):
                if any(trace.family_history[t][i][j]
                       for t in range(T)):
                    row_idx.append((i, j))
        if not row_idx:
            ax.text(0.5, 0.5, "no copulas", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_axis_off()
            return

        upper = T if t_max is None else min(t_max + 1, T)
        for r, (i, j) in enumerate(row_idx):
            for t in range(upper):
                fam = trace.family_history[t][i][j]
                if not fam:
                    continue
                ax.broken_barh([(t - 0.5, 1.0)], (r - 0.4, 0.8),
                               facecolors=fam_color.get(fam, "lightgrey"),
                               edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(row_idx)))
        ax.set_yticklabels([f"({i},{j})" for i, j in row_idx])
        ax.set_xlabel("ICE iteration")
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, len(row_idx) - 0.5)
        ax.set_title("Family selection ribbon")
        # Build a legend from the families actually present.
        handles = [
            plt.matplotlib.patches.Patch(color=fam_color[f], label=f)
            for f in all_fams if any(
                trace.family_history[t][i][j] == f
                for t in range(upper) for i, j in row_idx
            )
        ]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper right",
                      ncol=min(len(handles), 4))

    # ----- view A: τ trajectories per pair ---------------------------------

    def _plot_ice_tau(self, trace: IceTrace):
        fig = self._begin_figure()
        ax = fig.add_subplot(1, 1, 1)
        self._draw_tau_trajectories(ax, trace)
        self._canvas.draw()

    # ----- view B: family selection ribbon ---------------------------------

    def _plot_ice_family(self, trace: IceTrace):
        fig = self._begin_figure()
        ax = fig.add_subplot(1, 1, 1)
        self._draw_family_ribbon(ax, trace)
        self._canvas.draw()

    # ----- view L: GICE margin family ribbon -------------------------------

    def _plot_ice_margin_family(self, trace: IceTrace):
        """Per-state ribbon of the margin family selected at each ICE iter.

        This is the SP-2016 GICE counterpart of view~B (which tracks
        copula families). One row per state ``i``; each row is a horizontal
        ribbon coloured by the ``dist`` field of ``trace.margin_history[t][block_idx]``
        across iterations. Useful when the user enables ``candidates`` per
        margin block to visualise family identification convergence.
        """
        fig = self._begin_figure()
        ax  = fig.add_subplot(1, 1, 1)

        history = trace.margin_history
        if not history:
            ax.text(0.5, 0.5, "no margin history",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw()
            return

        T  = len(history)
        n_blocks = len(history[0])
        # Collect every distinct family that appeared across the trace.
        seen = sorted({
            history[t][k]["dist"] for t in range(T) for k in range(n_blocks)
        })
        cmap = plt.get_cmap("tab10")
        fam_color = {f: cmap(k % 10) for k, f in enumerate(seen)}

        # Try to label rows with the state index ``i``; fall back to
        # block index when blocks are not ``i``-keyed.
        labels = []
        for k in range(n_blocks):
            blk0 = history[0][k]
            if "i" in blk0:
                labels.append(f"i={blk0['i']}")
            else:
                labels.append(f"#{k}")

        for r in range(n_blocks):
            for t in range(T):
                fam = history[t][r]["dist"]
                ax.broken_barh([(t - 0.5, 1.0)], (r - 0.4, 0.8),
                               facecolors=fam_color[fam],
                               edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(n_blocks))
        ax.set_yticklabels(labels)
        ax.set_xlabel("ICE iteration")
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, n_blocks - 0.5)
        ax.set_title("Margin family ribbon (GICE — SP-2016 §3)")
        handles = [
            plt.matplotlib.patches.Patch(color=fam_color[f], label=f)
            for f in seen
        ]
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="upper right",
                      ncol=min(len(handles), 4))
        self._canvas.draw()

    # ----- view C: pseudo-observations + fitted PDF contours ---------------

    def _plot_ice_pseudos(self, trace: IceTrace, model: PMCModel, Y: np.ndarray):
        """K×K small-multiples: scatter (u, v) + fitted copula PDF contours."""
        fig = self._begin_figure()
        K = model.K
        N = len(Y)
        f_cdf = _compute_marginal_cdfs(model, Y)

        # Grid for contours
        ug = np.linspace(0.05, 0.95, 40)
        UU, VV = np.meshgrid(ug, ug)

        for i in range(K):
            for j in range(K):
                ax = fig.add_subplot(K, K, i * K + j + 1)
                cop = model.copula(i, j)
                u = f_cdf[: N - 1, i, j]
                v = f_cdf[1:,      j, i]
                ax.scatter(u, v, s=2, alpha=0.3, color="black")
                # PDF contours using vectorised evaluation.
                try:
                    grid_uv = np.column_stack((UU.ravel(), VV.ravel()))
                    Z = cop.pdf_array(grid_uv).reshape(UU.shape)
                    Z = np.clip(Z, 1e-6, None)
                    ax.contour(UU, VV, np.log10(Z), levels=8,
                               cmap="viridis", linewidths=1.0)
                except Exception:                              # pragma: no cover
                    pass
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_aspect("equal")
                ax.set_title(
                    f"({i},{j})  {cop.copula_enum.value.SHORT_NAME}"
                    f"  τ̂={cop.params['tau_k']:.3f}",
                    fontsize=9,
                )
                if i == K - 1:
                    ax.set_xlabel("u")
                if j == 0:
                    ax.set_ylabel("v")
        fig.suptitle("Pseudo-observations + fitted copula log-PDF contours")
        self._canvas.draw()

    # ----- view D: multistart comparison -----------------------------------

    def _plot_ice_multistart(self, trace: IceTrace):
        fig = self._begin_figure()
        ax = fig.add_subplot(1, 1, 1)
        # Best run (the trace itself) in steel-blue, others in grey.
        for losing in trace.multistart_runs:
            ax.plot(losing.log_liks, "-", color="grey", alpha=0.6, lw=1.0,
                    label=f"{losing.run_tag}  final={losing.log_liks[-1]:.2f}")
        ax.plot(trace.log_liks, "o-", color="steelblue", lw=2.0, markersize=4,
                label=f"WINNER  ({trace.run_tag or 'unperturbed'})  "
                      f"final={trace.log_liks[-1]:.2f}")
        ax.set_xlabel("ICE iteration")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"Multistart comparison "
                     f"({1 + len(trace.multistart_runs)} runs)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        self._canvas.draw()

    # ----- view E: joint prior p[i,j] evolution ----------------------------

    def _plot_ice_prior(self, trace: IceTrace):
        fig = self._begin_figure()
        if trace.p_history is None or trace.p_history.size == 0:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "no prior history",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            self._canvas.draw()
            return
        T, K, _ = trace.p_history.shape
        nn = np.arange(T)

        ax1 = fig.add_subplot(2, 1, 1)
        for i in range(K):
            for j in range(K):
                # Under SR-PMC, only show upper-triangle entries.
                if j < i:
                    continue
                ax1.plot(nn, trace.p_history[:, i, j], "-",
                         label=f"p[{i},{j}]")
        ax1.set_ylabel("p[i,j]")
        ax1.set_title("Joint-prior evolution  (upper triangle under SR-PMC)")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=7, ncol=K)

        ax2 = fig.add_subplot(2, 1, 2)
        if T >= 2:
            diffs = np.linalg.norm(
                trace.p_history[1:] - trace.p_history[:-1], axis=(1, 2),
            )
            ax2.semilogy(nn[1:], diffs, "o-", color="firebrick")
        ax2.set_xlabel("ICE iteration")
        ax2.set_ylabel("‖p^(t) − p^(t−1)‖_F")
        ax2.set_title("Convergence of the joint prior")
        ax2.grid(True, alpha=0.3, which="both")
        self._canvas.draw()

    # ----- view F: margin parameter evolution ------------------------------

    def _plot_ice_margins(self, trace: IceTrace):
        fig = self._begin_figure()
        if not trace.margin_history:
            self._canvas.clear()
            return
        # margin_history[t] is a list of dicts (in declaration order).
        T = len(trace.margin_history)
        n_blocks = len(trace.margin_history[0]) if T else 0
        if n_blocks == 0:
            self._canvas.clear()
            return
        # Layout: one subplot per block, in a square-ish grid.
        ncol = int(np.ceil(np.sqrt(n_blocks)))
        nrow = int(np.ceil(n_blocks / ncol))
        for k in range(n_blocks):
            ax = fig.add_subplot(nrow, ncol, k + 1)
            blk0 = trace.margin_history[0][k]
            i, j = blk0.get("i", "?"), blk0.get("j", "")
            dist = blk0.get("dist", "?")
            param_names = list(blk0.get("params", {}).keys())
            for pname in param_names:
                series = [
                    float(trace.margin_history[t][k]["params"].get(pname, np.nan))
                    for t in range(T)
                ]
                ax.plot(series, "-o", markersize=3, label=pname)
            ax.set_title(f"({i},{j}) {dist}", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Margin-parameter evolution")
        self._canvas.draw()

    # ----- view G: ICE dashboard (multi-panel summary) ---------------------

    def _plot_ice_dashboard(self, trace: IceTrace):
        fig = self._begin_figure()
        # 3×2 grid: LL, τ, family, ‖Δp‖, AIC/BIC, status
        ax_ll  = fig.add_subplot(3, 2, 1)
        ax_tau = fig.add_subplot(3, 2, 2)
        ax_fam = fig.add_subplot(3, 2, 3)
        ax_dp  = fig.add_subplot(3, 2, 4)
        ax_ic  = fig.add_subplot(3, 2, 5)
        ax_st  = fig.add_subplot(3, 2, 6); ax_st.set_axis_off()

        lls = trace.log_liks
        ax_ll.plot(lls, "o-", color="steelblue")
        ax_ll.set_title("Log-likelihood"); ax_ll.grid(True, alpha=0.3)

        self._draw_tau_trajectories(ax_tau, trace)
        self._draw_family_ribbon(ax_fam, trace)

        if trace.p_history is not None and trace.p_history.shape[0] >= 2:
            diffs = np.linalg.norm(
                trace.p_history[1:] - trace.p_history[:-1], axis=(1, 2),
            )
            ax_dp.semilogy(diffs, "o-", color="firebrick")
            ax_dp.set_title("‖Δp‖_F (joint-prior change)")
            ax_dp.grid(True, alpha=0.3, which="both")

        # Approximate AIC/BIC: requires a parameter count.
        # We use a coarse upper bound: K² × (#copula params + #margin params).
        if self._ice_Y is not None:
            n = len(self._ice_Y)
            # Heuristic: τ + extras per block + margin params per block
            n_params = 0
            if trace.tau_history is not None:
                n_params += int(np.isfinite(trace.tau_history[-1]).sum())
            if trace.margin_history:
                n_params += sum(
                    len(blk.get("params", {})) for blk in trace.margin_history[-1]
                )
            aic = 2 * n_params - 2 * np.array(lls)
            bic = n_params * np.log(max(n, 1)) - 2 * np.array(lls)
            ax_ic.plot(aic, "o-", label="AIC", color="darkorange")
            ax_ic.plot(bic, "s-", label="BIC", color="purple")
            ax_ic.set_title(f"AIC / BIC  (k={n_params})")
            ax_ic.legend(fontsize=8)
            ax_ic.grid(True, alpha=0.3)
        else:
            ax_ic.text(0.5, 0.5, "AIC/BIC unavailable\n(no observation sequence)",
                       transform=ax_ic.transAxes, ha="center", va="center")
            ax_ic.set_axis_off()

        # Summary box
        n_iters = trace.n_iters
        ll_init = lls[0] if lls else float("nan")
        ll_fin  = lls[-1] if lls else float("nan")
        msg = (
            f"Iterations: {n_iters}\n"
            f"LL initial: {ll_init:.3f}\n"
            f"LL final:   {ll_fin:.3f}\n"
            f"ΔLL:        {ll_fin - ll_init:+.3f}\n"
            f"Multistart: {1 + len(trace.multistart_runs)} run(s)\n"
            f"Candidates: {', '.join(trace.candidates) or '—'}"
        )
        ax_st.text(0.05, 0.95, msg, transform=ax_st.transAxes,
                   ha="left", va="top", family="monospace", fontsize=10)
        ax_st.set_title("Summary", loc="left")

        fig.suptitle("ICE Dashboard", fontsize=12, y=0.995)
        self._canvas.draw()

    # ----- view H: PP/QQ plot of fitted copulas ----------------------------

    def _plot_ice_pp(self, model: PMCModel, Y: np.ndarray):
        fig = self._begin_figure()
        K = model.K
        N = len(Y)
        f_cdf = _compute_marginal_cdfs(model, Y)

        for i in range(K):
            for j in range(K):
                ax = fig.add_subplot(K, K, i * K + j + 1)
                cop = model.copula(i, j)
                u   = f_cdf[: N - 1, i, j]
                v   = f_cdf[1:,      j, i]
                uv  = np.column_stack((u, v))
                try:
                    Cn = _empirical_copula(uv, uv)
                    Ct = np.array([cop.cdf(list(row)) for row in uv])
                    ax.scatter(Cn, Ct, s=4, alpha=0.4)
                    ax.plot([0, 1], [0, 1], "r--", lw=1)
                    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
                    ax.set_title(
                        f"({i},{j}) {cop.copula_enum.value.SHORT_NAME}",
                        fontsize=9,
                    )
                except NotImplementedError:
                    ax.text(0.5, 0.5, "no CDF", transform=ax.transAxes,
                            ha="center", va="center")
                    ax.set_axis_off()
                if i == K - 1:
                    ax.set_xlabel("C_n  (empirical)")
                if j == 0:
                    ax.set_ylabel("C_θ  (fitted)")
        fig.suptitle("PP plots — empirical vs. fitted copula CDF")
        self._canvas.draw()

    # ----- view I: tail dependence per pair --------------------------------

    def _plot_ice_tail(self, model: PMCModel):
        fig = self._begin_figure()
        K = model.K
        lam_L = np.zeros((K, K))
        lam_U = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                try:
                    lL, lU = model.copula(i, j).tail_dependence()
                except NotImplementedError:
                    lL, lU = np.nan, np.nan
                lam_L[i, j] = lL
                lam_U[i, j] = lU
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for ax, mat, label in (
            (ax1, lam_L, "λ_L (lower tail)"),
            (ax2, lam_U, "λ_U (upper tail)"),
        ):
            im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=1.0)
            for i in range(K):
                for j in range(K):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}",
                                ha="center", va="center", fontsize=10,
                                color="white" if mat[i, j] < 0.6 else "black")
            ax.set_title(label)
            ax.set_xticks(range(K)); ax.set_yticks(range(K))
            ax.set_xticklabels([f"j={k}" for k in range(K)])
            ax.set_yticklabels([f"i={k}" for k in range(K)])
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
        fig.suptitle("Tail dependence per pair (final ICE model)")
        self._canvas.draw()

    # ----- view J: γ before vs. after ICE ----------------------------------

    def _plot_ice_gamma_compare(self, init_mdl: PMCModel, fitted_mdl: PMCModel, Y: np.ndarray):
        fig = self._begin_figure()
        K  = init_mdl.K
        nn = np.arange(len(Y))

        ax1 = fig.add_subplot(2, 1, 1)
        _, gamma_i, ll_i = classify(init_mdl, Y)
        for k in range(K):
            ax1.plot(nn, gamma_i[:, k], lw=0.7, alpha=0.8, label=f"γ(X={k}|Y)")
        ax1.set_title(f"BEFORE ICE  (LL={ll_i:.2f})")
        ax1.set_ylabel("P(X_n=k | Y)"); ax1.set_ylim(-0.05, 1.05)
        ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(2, 1, 2)
        _, gamma_f, ll_f = classify(fitted_mdl, Y)
        for k in range(K):
            ax2.plot(nn, gamma_f[:, k], lw=0.7, alpha=0.8, label=f"γ(X={k}|Y)")
        ax2.set_title(f"AFTER ICE  (LL={ll_f:.2f}, ΔLL={ll_f - ll_i:+.2f})")
        ax2.set_xlabel("n"); ax2.set_ylabel("P(X_n=k | Y)"); ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        fig.suptitle("Posterior marginals γ — before vs. after ICE")
        self._canvas.draw()

    # ==================================================================
    # View → handler dispatch table
    # ==================================================================
    #
    # Each entry takes a single ``self`` argument and renders into the
    # canvas. ``_render_view`` looks up this dict and calls the handler;
    # plot-finalisation (Export-plot toggle, ``_has_plot`` flag) is done
    # centrally afterwards. Adding a new view is one entry here + one
    # ``_plot_*`` method.
    _VIEW_DISPATCH: dict[str, "callable"] = {
        _VIEW_SIMULATION:     lambda self: self._plot_simulation(*self._sim_state),
        _VIEW_CLASSIFICATION: lambda self: self._plot_classification(*self._cls_state),
        _VIEW_GOF_HEATMAP:    lambda self: self._plot_gof_heatmap(self._gof_state),
        _ICE_VIEW_LOGLIK:     lambda self: self._plot_loglik(self._ice_trace.log_liks),
        _ICE_VIEW_TAU:        lambda self: self._plot_ice_tau(self._ice_trace),
        _ICE_VIEW_FAMILY:     lambda self: self._plot_ice_family(self._ice_trace),
        _ICE_VIEW_MARGIN_FAMILY:
            lambda self: self._plot_ice_margin_family(self._ice_trace),
        _ICE_VIEW_PSEUDOS:    lambda self: self._plot_ice_pseudos(
            self._ice_trace, self._model, self._ice_Y,
        ),
        _ICE_VIEW_MULTISTART: lambda self: self._plot_ice_multistart(self._ice_trace),
        _ICE_VIEW_PRIOR:      lambda self: self._plot_ice_prior(self._ice_trace),
        _ICE_VIEW_MARGINS:    lambda self: self._plot_ice_margins(self._ice_trace),
        _ICE_VIEW_DASHBOARD:  lambda self: self._plot_ice_dashboard(self._ice_trace),
        _ICE_VIEW_PP:         lambda self: self._plot_ice_pp(self._model, self._ice_Y),
        _ICE_VIEW_TAIL:       lambda self: self._plot_ice_tail(self._model),
        _ICE_VIEW_GAMMA:      lambda self: self._plot_ice_gamma_compare(
            self._init_model, self._model, self._ice_Y,
        ),
        _ICE_VIEW_PLAYBACK:   lambda self: self._setup_playback(self._ice_trace),
    }


# ---------------------------------------------------------------------------
# CSV reader for the data-load action — extracted so it can be unit-tested.
# ---------------------------------------------------------------------------

def _read_data_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a (Y[, X]) CSV. Returns ``(Y, X_or_None)``.

    Raises ``ValueError`` for empty files, missing ``Y`` column, or
    non-numeric ``Y`` cells (with a row-number-aware message).
    Non-integer ``X`` cells trigger a logger warning and ``X = None`` —
    matching the GUI's "labels are optional" semantics.
    """
    rows: list[dict] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise ValueError("CSV file is empty.")
    if "Y" not in rows[0]:
        raise ValueError("CSV must contain a 'Y' column.")

    try:
        Y = np.array([float(r["Y"]) for r in rows])
    except ValueError as exc:
        # Find first offending row for a useful error message.
        for n, r in enumerate(rows):
            try:
                float(r["Y"])
            except ValueError:
                raise ValueError(
                    f"Non-numeric Y at row {n + 2} (CSV line {n + 2}): "
                    f"{r['Y']!r}"
                ) from exc
        raise

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


# ---------------------------------------------------------------------------
# Shared helper: marginal CDFs evaluated on Y, clipped to (EPS, 1-EPS).
# Used by the GoF action and the ICE Pseudo-obs / PP views.
# ---------------------------------------------------------------------------

def _compute_marginal_cdfs(model: PMCModel, Y: np.ndarray) -> np.ndarray:
    """Return ``f_cdf[n, i, j] = F_{ij}(Y[n])`` clipped to ``(EPS, 1-EPS)``.

    This is the single hottest reusable computation in the diagnostic
    layer — extracting it eliminates a triple-duplicated loop across
    :meth:`PMCMainWindow._do_gof_test`, :meth:`._plot_ice_pseudos`,
    :meth:`._plot_ice_pp`.
    """
    K = model.K
    N = len(Y)
    f_cdf = np.zeros((N, K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            f_cdf[:, i, j] = model.margin(i, j).cdf_vec(Y)
    np.clip(f_cdf, EPS, ONE_MINUS_EPS, out=f_cdf)
    return f_cdf


# ---------------------------------------------------------------------------
# Colour palette for K classes
# ---------------------------------------------------------------------------

def _class_colors(n_classes: int) -> list:
    """Return a list of ``n_classes`` distinct RGBA colours from tab10.

    ``tab10`` is a discrete 10-colour palette — we index it modulo 10 so
    callers always get the canonical colours rather than interpolated
    intermediates (which would happen with a non-integer index).
    """
    cmap = plt.get_cmap("tab10")
    return [cmap(k % 10) for k in range(n_classes)]
