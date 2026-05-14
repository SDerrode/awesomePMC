"""
prg.pmc.gui.tabs — the four tabs of the PMC main window.

* :class:`_PriorTab`  — A or p matrix editor, with a "Symmetrize" button
  for SR-PMC (joint-prior) variants and live auto-mirroring of off-diagonal
  cells under those variants.
* :class:`_MarginTab` — K×K grid of margin distributions (double-click to edit).
* :class:`_CopulaTab` — K×K grid of copula families + τ (double-click to edit).
* :class:`_IceTab`    — ICE configuration including multistart options.

Each tab exposes:

* ``load(model_or_cfg, ...)`` to populate widgets from a :class:`PMCModel`
  (or a config dict for ICE).
* ``save(raw_dict)`` to write the current widget values back into the raw
  TOML dict that ``PMCModel.from_dict`` will rebuild.
* ``changed`` (Qt signal, where applicable) — fires on any user edit so the
  host window can flag the model as "modified".

These were originally inside ``main_window.py``; extracted to keep the
QMainWindow file readable.
"""

from __future__ import annotations

import copy

import numpy as np

from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from prg.copulas._base   import CopulaEnum
from prg.pmc.gui.dialogs import _CopulaDialog, _MarginDialog
from prg.pmc.model       import PMCModel, Variant


class PriorTabError(ValueError):
    """Raised by :meth:`_PriorTab.save` on invalid cell content.

    The host window catches it to show a QMessageBox pointing at the
    offending cell — far better UX than a silent zero-fill.
    """


class _PriorTab(QWidget):
    """Editor for the prior distribution (transition matrix A or joint p).

    For SR-PMC variants (joint prior ``p``) a "Symmetrize" button enforces
    ``p[i,j] = p[j,i]`` in one click, and live auto-mirroring keeps the
    off-diagonal cells in sync as the user edits.

    Signals
    -------
    changed     : pyqtSignal — fires whenever the user edits a cell.
    symmetrized : pyqtSignal(float, bool) — fires after the Symmetrize
                  button finishes; arguments are (max |p − pᵀ| before
                  symmetrising, was renormalisation needed).
    """

    changed     = pyqtSignal()
    symmetrized = pyqtSignal(float, bool)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self._label = QLabel("—")
        lay.addWidget(self._label)

        self._mirror_hint = QLabel(
            "<i>SR-PMC: off-diagonal cells are auto-mirrored as you edit.</i>"
        )
        self._mirror_hint.setVisible(False)
        lay.addWidget(self._mirror_hint)

        self._table = QTableWidget()
        # Connect *once*; an internal flag suppresses recursive triggers
        # when the auto-mirror writes the symmetric cell.
        self._mirror_busy = False
        self._table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self._table)

        # "Symmetrize" button — visible only for joint-prior variants.
        btn_row = QHBoxLayout()
        self._btn_sym = QPushButton("Symmetrize p (SR-PMC)")
        self._btn_sym.setToolTip(
            "Replace p with (p + pᵀ)/2 and renormalise. Required for SR-PMC."
        )
        self._btn_sym.clicked.connect(self._on_symmetrize)
        self._btn_sym.setVisible(False)
        btn_row.addWidget(self._btn_sym)
        btn_row.addStretch(1)
        lay.addLayout(btn_row)

        self._K       = 0
        self._variant: Variant | None = None
        # Latest symmetrize report — tuple (max_asym_before, was_renorm).
        # Read by main_window for the log message.
        self._last_sym_report: tuple[float, bool] | None = None

    # ------------------------------------------------------------------
    # Cell helpers
    # ------------------------------------------------------------------

    def _read_cell(self, i: int, j: int) -> float:
        """Return the float value at ``(i, j)`` or raise :class:`PriorTabError`.

        Invalid (non-parseable) cells trigger a host-level message via the
        custom exception type, avoiding silent corruption.
        """
        item = self._table.item(i, j)
        text = item.text() if item is not None else ""
        try:
            return float(text)
        except (TypeError, ValueError) as exc:
            raise PriorTabError(
                f"Prior cell ({i}, {j}) contains an invalid number: "
                f"{text!r}."
            ) from exc

    def _read_matrix(self) -> np.ndarray:
        K = self._K
        out = np.zeros((K, K), dtype=float)
        for i in range(K):
            for j in range(K):
                out[i, j] = self._read_cell(i, j)
        return out

    def _write_cell(self, i: int, j: int, value: float):
        """Set a cell text without re-triggering the mirror callback."""
        item = self._table.item(i, j)
        text = f"{value:.6f}"
        was_busy = self._mirror_busy
        self._mirror_busy = True
        try:
            if item is None:
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(i, j, item)
            else:
                item.setText(text)
        finally:
            self._mirror_busy = was_busy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, mdl: PMCModel):
        self._K       = mdl.K
        self._variant = mdl.variant

        if mdl.variant.has_markov_prior:
            data  = mdl.transition_A
            label = "A[i,j] = P(X_{n+1}=j | X_n=i)  (rows sum to 1)"
            self._btn_sym.setVisible(False)
            self._mirror_hint.setVisible(False)
        else:
            data  = mdl.prior_p
            label = "p[i,j] = P(X_n=i, X_{n+1}=j)  (sum to 1; SR-PMC: p[i,j]=p[j,i])"
            self._btn_sym.setVisible(True)
            self._mirror_hint.setVisible(True)

        self._label.setText(label)
        K = mdl.K
        # Suppress the changed/mirror signals while we prefill the table.
        self._mirror_busy = True
        try:
            self._table.setRowCount(K)
            self._table.setColumnCount(K)
            hdrs = [str(k) for k in range(K)]
            self._table.setHorizontalHeaderLabels(hdrs)
            self._table.setVerticalHeaderLabels(hdrs)
            for i in range(K):
                for j in range(K):
                    item = QTableWidgetItem(f"{data[i, j]:.6f}")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table.setItem(i, j, item)
            self._table.resizeColumnsToContents()
        finally:
            self._mirror_busy = False

    def save(self, raw: dict):
        if self._K == 0:
            return
        try:
            mat = self._read_matrix().tolist()
        except PriorTabError:
            raise   # surfaced by the host (main_window) as a QMessageBox.
        raw.setdefault("prior", {})
        if self._variant.has_markov_prior:
            raw["prior"]["A"] = mat
        else:
            raw["prior"]["p"] = mat

    # ------------------------------------------------------------------
    # Auto-mirror & change tracking
    # ------------------------------------------------------------------

    def _on_item_changed(self, item: QTableWidgetItem):
        """Mirror off-diagonal edits under SR-PMC and emit ``changed``."""
        if self._mirror_busy:
            return
        # Always notify the host that the user edited something.
        self.changed.emit()

        # SR-PMC auto-mirror: only for joint-prior variants and only on
        # off-diagonal cells.
        if self._variant is None or self._variant.has_markov_prior:
            return
        i, j = item.row(), item.column()
        if i == j:
            return
        try:
            val = float(item.text())
        except ValueError:
            return                                 # leave the typo for the user to see
        self._write_cell(j, i, val)

    # ------------------------------------------------------------------
    # Symmetrize action — joint-prior variants only
    # ------------------------------------------------------------------

    def _on_symmetrize(self):
        """Read the K×K table, apply (p + pᵀ)/2 + renormalisation, write back.

        The maximum |p − pᵀ| before symmetrising is recorded in
        :attr:`_last_sym_report` so the host window can log a meaningful
        message ("max asymmetry was 3.4e-04").
        """
        K = self._K
        if K == 0:
            return
        try:
            p = self._read_matrix()
        except PriorTabError as exc:
            QMessageBox.warning(self, "Symmetrize failed", str(exc))
            return

        asym = float(np.abs(p - p.T).max()) if K else 0.0
        p = 0.5 * (p + p.T)
        p = np.clip(p, 0.0, None)
        s = float(p.sum())
        was_renorm = abs(s - 1.0) > 1e-12
        if s > 0:
            p /= s
        # Write back without triggering the auto-mirror callback.
        self._mirror_busy = True
        try:
            for i in range(K):
                for j in range(K):
                    self._table.item(i, j).setText(f"{p[i, j]:.6f}")
        finally:
            self._mirror_busy = False
        self._last_sym_report = (asym, was_renorm)
        self.symmetrized.emit(asym, was_renorm)
        self.changed.emit()


class _BlockGridTab(QWidget):
    """Shared base class for the Margins and Copulas tabs.

    Each tab shows a read-only grid whose cells open a modal editor on
    double-click. Subclasses configure:

    * :attr:`_INFO_LABEL`  — the label above the grid.
    * :attr:`_RAW_KEY`     — TOML key under which the blocks live.
    * :attr:`_DIALOG`      — the dialog class spawned by double-click.
    * :attr:`_BLOCKS_ATTR` — the :class:`PMCModel` method that returns
                              the list of blocks.
    * :attr:`_PAIR_INDEXED` — if False (default for the Margins tab under
                              SR-PMC), the grid is 1×K (one cell per
                              state ``i``); otherwise it is K×K.

    Subclasses must also implement :meth:`_format_block`.
    """

    _INFO_LABEL:   str   = "(override _INFO_LABEL)"
    _RAW_KEY:      str   = "(override _RAW_KEY)"
    _DIALOG:       type  = type(None)
    _BLOCKS_ATTR:  str   = "(override _BLOCKS_ATTR)"   # method name on PMCModel
    # Margins are state-indexed under SR-PMC (1×K grid); copulas remain
    # pair-indexed (K×K grid).
    _PAIR_INDEXED: bool  = True

    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self._info = QLabel(self._INFO_LABEL)
        lay.addWidget(self._info)
        self._table = QTableWidget()
        self._table.cellDoubleClicked.connect(self._on_cell_dbl)
        self._table.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self._table)
        self._blocks: list[dict] = []
        self._K = 0

    # ------------------------------------------------------------------
    # Shared lifecycle
    # ------------------------------------------------------------------

    def load(self, mdl: PMCModel):
        self._K      = mdl.K
        self._blocks = list(getattr(mdl, self._BLOCKS_ATTR)())
        hdrs = [str(k) for k in range(mdl.K)]
        if self._PAIR_INDEXED:
            self._table.setRowCount(mdl.K)
            self._table.setColumnCount(mdl.K)
            self._table.setHorizontalHeaderLabels(hdrs)
            self._table.setVerticalHeaderLabels(hdrs)
        else:
            # State-indexed (one row per state) — single-column grid.
            self._table.setRowCount(mdl.K)
            self._table.setColumnCount(1)
            self._table.setHorizontalHeaderLabels(["density"])
            self._table.setVerticalHeaderLabels([f"i={k}" for k in range(mdl.K)])
        self._refresh_table()

    def _refresh_table(self):
        K = self._K
        if self._PAIR_INDEXED:
            for i in range(K):
                for j in range(K):
                    self._set_cell(i, j, self._get_block(i, j))
        else:
            for i in range(K):
                self._set_cell(i, 0, self._get_block_state(i))
        self._table.resizeColumnsToContents()

    def _set_cell(self, row: int, col: int, blk: dict | None):
        text = self._format_block(blk) if blk else "—"
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        self._table.setItem(row, col, item)

    def _get_block(self, i: int, j: int) -> dict | None:
        """Pair-indexed lookup (used by Copulas tab and any K²-format margins)."""
        for blk in self._blocks:
            bi = int(blk.get("i", -1))
            bj = int(blk.get("j", -1))
            if bi == i and bj == j:
                return blk
            # State-indexed margins — only ``i`` key present.
            if bi == i and "j" not in blk:
                return blk
        return None

    def _get_block_state(self, i: int) -> dict | None:
        """State-indexed lookup (used by Margins tab under SR-PMC)."""
        for blk in self._blocks:
            if int(blk.get("i", -1)) == i:
                return blk
        return None

    def _on_cell_dbl(self, row: int, col: int):
        if self._PAIR_INDEXED:
            blk = self._get_block(row, col)
        else:
            blk = self._get_block_state(row)
        if blk is None:
            return
        dlg = self._DIALOG(blk, self)
        if dlg.exec():
            blk.update(dlg.get_block())
            self._refresh_table()
            self.changed.emit()

    def save(self, raw: dict):
        raw[self._RAW_KEY] = copy.deepcopy(self._blocks)

    # ------------------------------------------------------------------
    # Subclasses customise only this method.
    # ------------------------------------------------------------------

    def _format_block(self, blk: dict) -> str:                 # pragma: no cover
        raise NotImplementedError


class _MarginTab(_BlockGridTab):
    """K-cell grid (one per state) for the marginal densities f_i.

    Under the SR-PMC contract there are exactly K state-indexed marginal
    densities — the legacy K×K layout was abandoned. Cells correspond to
    states 0…K-1; double-click opens the margin editor.
    """

    _INFO_LABEL   = "Margins f_i (one per state, SR-PMC) — double-click to edit."
    _RAW_KEY      = "margins"
    _DIALOG       = _MarginDialog
    _PAIR_INDEXED = False                   # ← state-indexed, 1-column grid
    _BLOCKS_ATTR = "margin_blocks"

    def _format_block(self, blk: dict) -> str:
        params = ", ".join(f"{k}={v}" for k, v in blk.get("params", {}).items())
        return f"{blk['dist']}({params})"


class _CopulaTab(_BlockGridTab):
    """K×K grid of cells for copula selection and τ."""

    _INFO_LABEL  = "Copulas c_{ij}  — double-click a cell to edit."
    _RAW_KEY     = "copulas"
    _DIALOG      = _CopulaDialog
    _BLOCKS_ATTR = "copula_blocks"

    def _format_block(self, blk: dict) -> str:
        extras = ", ".join(
            f"{k}={v:.3g}" for k, v in blk.items()
            if k not in ("name", "tau", "i", "j")
        )
        text = f"{blk.get('name','?')}  τ={blk.get('tau', 0):.3f}"
        if extras:
            text += f"  ({extras})"
        return text


class _IceTab(QWidget):
    """ICE / SEM estimator configuration panel.

    Exposes every key parsed by :func:`prg.pmc.ice._parse_ice_cfg`
    (``max_iter``, ``tol``, ``patience``, ``fit_margins``, ``candidates``,
    ``init``, ``kmeans_seed``, ``n_starts``, ``multistart_seed``,
    ``multistart_jitter``) and adds the algorithm switch ``algorithm`` plus
    its companion ``sem_seed`` for the SEM stochastic completion (see
    :func:`prg.pmc.sem._parse_sem_cfg`).

    The multistart-specific widgets (seed, jitter) are disabled when
    ``n_starts == 1``; the K-means seed widget is disabled when
    ``init != "kmeans"`` — so it is visually clear they have no effect.
    """

    changed = pyqtSignal()

    # Default candidate set for new tabs (a sensible mix that's eligible
    # at any τ value).
    _DEFAULT_CANDIDATES_CHECKED = ("Gauss", "Clayton", "GH", "Frank", "Joe")

    def __init__(self):
        super().__init__()
        lay = QFormLayout(self)

        # ── Algorithm switch ──────────────────────────────────────────
        self._combo_algorithm = QComboBox()
        self._combo_algorithm.addItems(["ice", "sem"])
        self._combo_algorithm.setToolTip(
            "Estimation algorithm. 'ice' (default, deterministic) iterates "
            "soft-posterior M-steps until convergence; 'sem' draws X̃ ~ "
            "P(X|Y) at every iteration and runs a supervised-style M-step "
            "on it (stochastic — log-lik fluctuates around the stationary "
            "regime)."
        )
        self._combo_algorithm.currentTextChanged.connect(self._on_algorithm_changed)

        self._spn_sem_seed = QSpinBox()
        self._spn_sem_seed.setRange(0, 2_147_483_647)
        self._spn_sem_seed.setValue(0)
        self._spn_sem_seed.setToolTip(
            "RNG seed for SEM's per-iteration FFBS draw."
        )

        # ── Convergence controls ──────────────────────────────────────
        self._spn_maxiter = QSpinBox()
        self._spn_maxiter.setRange(1, 1000)
        self._spn_maxiter.setValue(50)

        self._spn_tol = QDoubleSpinBox()
        self._spn_tol.setRange(1e-10, 1.0)
        self._spn_tol.setDecimals(8)
        self._spn_tol.setValue(1e-4)
        self._spn_tol.setSingleStep(1e-5)

        self._spn_patience = QSpinBox()
        self._spn_patience.setRange(1, 50)
        self._spn_patience.setValue(3)
        self._spn_patience.setToolTip(
            "Stop early after this many consecutive log-likelihood regressions."
        )

        # ── Margin/copula candidates ─────────────────────────────────
        self._chk_margins = QCheckBox()
        # Multi-select grid: one QCheckBox per available copula family.
        # Clearer + less error-prone than the previous comma-separated
        # text field, and the user immediately sees the full menu.
        self._cand_box   = QGroupBox("Candidate copulas")
        self._cand_box.setToolTip(
            "Tick the families ICE may select among at every M-step. "
            "Family-aware τ ranges are enforced at fit time, so you can "
            "leave all families ticked even if τ is far from their range."
        )
        cand_layout = QGridLayout(self._cand_box)
        cand_layout.setContentsMargins(6, 6, 6, 6)
        cand_layout.setHorizontalSpacing(10)
        cand_layout.setVerticalSpacing(2)
        self._cand_checks: dict[str, QCheckBox] = {}
        all_short = [c.value.SHORT_NAME for c in CopulaEnum.available()]
        n_cols = 4
        for k, short in enumerate(all_short):
            cb = QCheckBox(short)
            cb.setChecked(short in self._DEFAULT_CANDIDATES_CHECKED)
            cb.toggled.connect(self.changed)
            cand_layout.addWidget(cb, k // n_cols, k % n_cols)
            self._cand_checks[short] = cb
        # "All"/"None" quick-pick buttons.
        btn_row = QHBoxLayout()
        btn_all  = QPushButton("All")
        btn_none = QPushButton("None")
        btn_def  = QPushButton("Defaults")
        btn_all.setToolTip("Tick every copula family.")
        btn_none.setToolTip("Untick every family (don't fit copulas at all).")
        btn_def.setToolTip("Reset to the default 5-family set.")
        btn_all.clicked.connect(lambda:  self._set_candidates(all_short))
        btn_none.clicked.connect(lambda: self._set_candidates([]))
        btn_def.clicked.connect(lambda:  self._set_candidates(
            list(self._DEFAULT_CANDIDATES_CHECKED)
        ))
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_def)
        btn_row.addStretch(1)
        # Place the buttons on a new row at the bottom of the grid.
        cand_layout.addLayout(btn_row,
                              (len(all_short) + n_cols - 1) // n_cols, 0,
                              1, n_cols)

        # ── Initialisation strategy ───────────────────────────────────
        self._combo_init = QComboBox()
        self._combo_init.addItems(["model", "kmeans"])
        self._combo_init.setToolTip(
            "Initial ICE parameters. 'model' (default) uses the prior, "
            "margins and copulas declared in the loaded model. 'kmeans' "
            "first clusters Y with k-means, then derives a warm-start "
            "model from the hard labels via a single supervised-style "
            "M-step. Requires scikit-learn (pip install copulasformm[ml])."
        )
        self._combo_init.currentTextChanged.connect(self._on_init_changed)

        self._spn_kmeans_seed = QSpinBox()
        self._spn_kmeans_seed.setRange(0, 2_147_483_647)
        self._spn_kmeans_seed.setValue(0)
        self._spn_kmeans_seed.setToolTip(
            "RNG seed forwarded to sklearn.cluster.KMeans."
        )

        # ── Multistart ────────────────────────────────────────────────
        self._spn_n_starts = QSpinBox()
        self._spn_n_starts.setRange(1, 20)
        self._spn_n_starts.setValue(1)
        self._spn_n_starts.setToolTip(
            "Number of independent ICE runs from random perturbations of "
            "the initial model. The best run (highest final log-likelihood) "
            "is returned."
        )
        self._spn_n_starts.valueChanged.connect(self._on_n_starts_changed)

        self._spn_ms_seed = QSpinBox()
        self._spn_ms_seed.setRange(0, 2_147_483_647)
        self._spn_ms_seed.setValue(42)               # default seed
        self._spn_ms_seed.setToolTip("RNG seed controlling the multistart perturbations.")
        # 🎲 button to draw a random seed.
        self._btn_seed_random = QPushButton("🎲")
        self._btn_seed_random.setMaximumWidth(34)
        self._btn_seed_random.setToolTip(
            "Draw a fresh random seed (uniform in [0, 2³¹−1])."
        )
        self._btn_seed_random.clicked.connect(self._on_random_seed_clicked)
        seed_box = QWidget()
        seed_lay = QHBoxLayout(seed_box)
        seed_lay.setContentsMargins(0, 0, 0, 0)
        seed_lay.addWidget(self._spn_ms_seed, stretch=1)
        seed_lay.addWidget(self._btn_seed_random, stretch=0)

        self._spn_ms_jitter = QDoubleSpinBox()
        self._spn_ms_jitter.setRange(0.01, 0.50)
        self._spn_ms_jitter.setDecimals(2)
        self._spn_ms_jitter.setSingleStep(0.05)
        self._spn_ms_jitter.setValue(0.10)
        self._spn_ms_jitter.setToolTip(
            "Relative perturbation amplitude (fraction of each parameter's "
            "natural scale). 0.10 = 10 %."
        )

        # ── Layout ────────────────────────────────────────────────────
        lay.addRow(QLabel("<b>Algorithm</b>"))
        lay.addRow("Estimator:",              self._combo_algorithm)
        lay.addRow("SEM seed:",               self._spn_sem_seed)
        lay.addRow("Max iterations:",         self._spn_maxiter)
        lay.addRow("Convergence tol:",        self._spn_tol)
        lay.addRow("Patience (regressions):", self._spn_patience)
        lay.addRow("Fit margins:",            self._chk_margins)
        lay.addRow(self._cand_box)
        lay.addRow(QLabel("<b>Initialisation</b>"))
        lay.addRow("Init strategy:",          self._combo_init)
        lay.addRow("K-means seed:",           self._spn_kmeans_seed)
        lay.addRow(QLabel("<b>Multistart</b>"))
        lay.addRow("n_starts:",               self._spn_n_starts)
        lay.addRow("Seed:",                   seed_box)
        lay.addRow("Jitter:",                 self._spn_ms_jitter)

        # Disable multistart widgets initially (n_starts=1) and the K-means
        # seed (init=model). The SEM seed defaults to disabled too — only
        # SEM uses it.
        self._on_n_starts_changed(1)
        self._on_init_changed(self._combo_init.currentText())
        self._on_algorithm_changed(self._combo_algorithm.currentText())

        # Wire change-tracking — every editable widget signals ``changed``.
        for w in (
            self._spn_maxiter, self._spn_tol, self._spn_patience,
            self._spn_n_starts, self._spn_ms_seed, self._spn_ms_jitter,
            self._spn_kmeans_seed, self._spn_sem_seed,
        ):
            w.valueChanged.connect(self.changed)
        self._chk_margins.toggled.connect(self.changed)
        self._combo_init.currentTextChanged.connect(self.changed)
        self._combo_algorithm.currentTextChanged.connect(self.changed)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_n_starts_changed(self, value: int):
        active = value > 1
        self._spn_ms_seed.setEnabled(active)
        self._btn_seed_random.setEnabled(active)
        self._spn_ms_jitter.setEnabled(active)

    def _on_init_changed(self, value: str):
        """Enable the K-means seed widget only when init=='kmeans'."""
        self._spn_kmeans_seed.setEnabled(value == "kmeans")

    def _on_algorithm_changed(self, value: str):
        """Enable the SEM seed widget (and the ``patience`` spinbox) according
        to the selected estimator. SEM has no early-stop, so ``patience`` is
        not consumed there — we leave the spinbox enabled for clarity but
        annotate it via tooltip."""
        is_sem = (value == "sem")
        self._spn_sem_seed.setEnabled(is_sem)
        # SEM has no ``tol`` early-stop. Keep the widget enabled (the value
        # is harmless for SEM) but make the tooltip explicit.
        self._spn_tol.setToolTip(
            "Relative log-likelihood convergence threshold. "
            "Used by ICE only — SEM does not converge deterministically."
            if is_sem else
            "Relative log-likelihood convergence threshold."
        )

    def _on_random_seed_clicked(self):
        """Draw a fresh random seed and write it into the spin box."""
        import secrets
        seed = secrets.randbelow(self._spn_ms_seed.maximum() + 1)
        self._spn_ms_seed.setValue(int(seed))

    def _set_candidates(self, names):
        """Tick exactly ``names`` (and untick everything else)."""
        names_set = set(names)
        for short, cb in self._cand_checks.items():
            cb.setChecked(short in names_set)

    def _selected_candidates(self) -> list[str]:
        """Return the SHORT_NAMEs currently ticked, in CopulaEnum order."""
        return [s for s, cb in self._cand_checks.items() if cb.isChecked()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, cfg: dict, show_copula_cfg: bool = True):
        # Hide the candidate-copulas group for variants that do not use
        # copulas (HMC-IN). The whole QGroupBox row hides together.
        self._cand_box.setVisible(show_copula_cfg)

        # Block the changed signal while we programmatically populate widgets.
        self.blockSignals(True)
        try:
            self._chk_margins.setChecked(bool(cfg.get("fit_margins", False)))
            self._spn_maxiter.setValue(int(cfg.get("max_iter", 50)))
            self._spn_tol.setValue(float(cfg.get("tol", 1e-4)))
            self._spn_patience.setValue(int(cfg.get("patience", 3)))
            self._set_candidates(cfg.get(
                "candidates", list(self._DEFAULT_CANDIDATES_CHECKED),
            ))

            init_value = str(cfg.get("init", "model"))
            if init_value not in ("model", "kmeans"):
                init_value = "model"
            self._combo_init.setCurrentText(init_value)
            self._spn_kmeans_seed.setValue(int(cfg.get("kmeans_seed", 0)))
            self._on_init_changed(self._combo_init.currentText())

            algo_value = str(cfg.get("algorithm", "ice"))
            if algo_value not in ("ice", "sem"):
                algo_value = "ice"
            self._combo_algorithm.setCurrentText(algo_value)
            self._spn_sem_seed.setValue(int(cfg.get("sem_seed", 0)))
            self._on_algorithm_changed(self._combo_algorithm.currentText())

            self._spn_n_starts.setValue(int(cfg.get("n_starts", 1)))
            self._spn_ms_seed.setValue(int(cfg.get("multistart_seed", 42)))
            self._spn_ms_jitter.setValue(float(cfg.get("multistart_jitter", 0.10)))
            self._on_n_starts_changed(self._spn_n_starts.value())
        finally:
            self.blockSignals(False)

    def get_cfg(self) -> dict:
        return {
            "algorithm":         self._combo_algorithm.currentText(),
            "sem_seed":          self._spn_sem_seed.value(),
            "fit_margins":       self._chk_margins.isChecked(),
            "max_iter":          self._spn_maxiter.value(),
            "tol":               self._spn_tol.value(),
            "patience":          self._spn_patience.value(),
            "candidates":        self._selected_candidates(),
            "init":              self._combo_init.currentText(),
            "kmeans_seed":       self._spn_kmeans_seed.value(),
            "n_starts":          self._spn_n_starts.value(),
            "multistart_seed":   self._spn_ms_seed.value(),
            "multistart_jitter": self._spn_ms_jitter.value(),
        }
