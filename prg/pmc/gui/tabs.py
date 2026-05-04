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
    QCheckBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

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
    """ICE configuration panel.

    Exposes every key parsed by :func:`prg.pmc.ice._parse_ice_cfg`:
    ``max_iter``, ``tol``, ``patience``, ``fit_margins``, ``candidates``,
    ``n_starts``, ``multistart_seed``, ``multistart_jitter``.

    The multistart-specific widgets (seed, jitter) are disabled when
    ``n_starts == 1`` so it is visually clear they have no effect.
    """

    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QFormLayout(self)

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
        self._lst_cands   = QLineEdit("Gauss,Clayton,GH,Frank,Joe")

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
        self._spn_ms_seed.setValue(0)
        self._spn_ms_seed.setToolTip("RNG seed controlling the multistart perturbations.")

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
        lay.addRow("Max iterations:",                 self._spn_maxiter)
        lay.addRow("Convergence tol:",                self._spn_tol)
        lay.addRow("Patience (regressions):",         self._spn_patience)
        lay.addRow("Fit margins:",                    self._chk_margins)
        lay.addRow("Candidates (SHORT_NAMEs, comma-sep):", self._lst_cands)
        lay.addRow(QLabel("<b>Multistart</b>"))
        lay.addRow("n_starts:",                       self._spn_n_starts)
        lay.addRow("Seed:",                           self._spn_ms_seed)
        lay.addRow("Jitter:",                         self._spn_ms_jitter)

        # Disable multistart widgets initially (n_starts=1).
        self._on_n_starts_changed(1)

        # Wire change-tracking — every editable widget signals ``changed``.
        for w in (
            self._spn_maxiter, self._spn_tol, self._spn_patience,
            self._spn_n_starts, self._spn_ms_seed, self._spn_ms_jitter,
        ):
            w.valueChanged.connect(self.changed)
        self._chk_margins.toggled.connect(self.changed)
        self._lst_cands.textChanged.connect(self.changed)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_n_starts_changed(self, value: int):
        active = value > 1
        self._spn_ms_seed.setEnabled(active)
        self._spn_ms_jitter.setEnabled(active)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, cfg: dict, show_copula_cfg: bool = True):
        # Hide the candidates field for variants that do not use copulas
        # (HMC-IN). The label-only form-row pair is hidden together.
        self._lst_cands.setVisible(show_copula_cfg)
        # Find the QLabel sibling (FormLayout pairs label↔widget).
        form = self.layout()
        for k in range(form.rowCount()):
            field = form.itemAt(k, form.ItemRole.FieldRole)
            if field is not None and field.widget() is self._lst_cands:
                lbl = form.itemAt(k, form.ItemRole.LabelRole)
                if lbl is not None and lbl.widget() is not None:
                    lbl.widget().setVisible(show_copula_cfg)
                break

        # Block the changed signal while we programmatically populate widgets.
        self.blockSignals(True)
        try:
            self._chk_margins.setChecked(bool(cfg.get("fit_margins", False)))
            self._spn_maxiter.setValue(int(cfg.get("max_iter", 50)))
            self._spn_tol.setValue(float(cfg.get("tol", 1e-4)))
            self._spn_patience.setValue(int(cfg.get("patience", 3)))
            cands = cfg.get("candidates", ["Gauss", "Clayton", "GH"])
            self._lst_cands.setText(",".join(cands))

            self._spn_n_starts.setValue(int(cfg.get("n_starts", 1)))
            self._spn_ms_seed.setValue(int(cfg.get("multistart_seed", 0)))
            self._spn_ms_jitter.setValue(float(cfg.get("multistart_jitter", 0.10)))
            self._on_n_starts_changed(self._spn_n_starts.value())
        finally:
            self.blockSignals(False)

    def get_cfg(self) -> dict:
        return {
            "fit_margins":       self._chk_margins.isChecked(),
            "max_iter":          self._spn_maxiter.value(),
            "tol":               self._spn_tol.value(),
            "patience":          self._spn_patience.value(),
            "candidates":        [s.strip() for s in self._lst_cands.text().split(",") if s.strip()],
            "n_starts":          self._spn_n_starts.value(),
            "multistart_seed":   self._spn_ms_seed.value(),
            "multistart_jitter": self._spn_ms_jitter.value(),
        }
