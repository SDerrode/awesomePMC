"""
pmcprg.pmc.gui.tabs — the four tabs of the PMC main window.

* :class:`_PriorTab`  — A or p matrix editor, with a "Symmetrize" button
  for SR-PMC (joint-prior) variants and live auto-mirroring of off-diagonal
  cells under those variants.
* :class:`_MarginTab` — marginal densities, double-click to edit: K cells f_i
  (state margins) or K×K cells f_ij, the law of y_n given (x_n = i, x_{n+1} = j)
  (pair margins, general PMC — DerrodePieczynski_CSDA2013 Eqs. 12–14), with a
  selector switching between the two structures.
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
import os

import numpy as np

from PyQt6.QtCore    import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from pmcprg.copulas._base   import CopulaEnum
from pmcprg.pmc             import ice_estim_defaults, sem_estim_defaults
from pmcprg.pmc._estim_common import (
    MAX_SWEEP_COMBINATIONS, MULTISTART_FAMILY_MODES, ice_missing_defaults,
)
from pmcprg.pmc.ice         import MARGIN_SELECTION_RULES, MISSING_STRATEGIES, SELECTION_CRITERIA
from pmcprg.pmc.gui.dialogs import _CopulaDialog, _MarginDialog
from pmcprg.pmc.model       import PMCModel, Variant

# Single-source estimator defaults (audit Q-9): the spinbox initial values
# and the ``load()`` fallbacks below must match what ``_parse_ice_cfg`` /
# ``_parse_sem_cfg`` resolve, or the GUI silently drifts from the API.
_ICE_DEFAULTS = ice_estim_defaults()
_SEM_DEFAULTS = sem_estim_defaults()
_MISSING_DEFAULTS = ice_missing_defaults()


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
    * :meth:`_pair_indexed` — True (default, copulas) for a K×K grid, one
                              cell per pair ``(i, j)``; False for a K×1 grid,
                              one cell per state ``i``. The Margins tab
                              answers from the model's margin structure.

    Subclasses must also implement :meth:`_format_block`.
    """

    _INFO_LABEL:   str   = "(override _INFO_LABEL)"
    _RAW_KEY:      str   = "(override _RAW_KEY)"
    _DIALOG:       type  = type(None)
    _BLOCKS_ATTR:  str   = "(override _BLOCKS_ATTR)"   # method name on PMCModel
    # Copulas are pair-indexed (K×K grid); the Margins tab overrides
    # :meth:`_pair_indexed` with the model's margin structure.
    _PAIR_INDEXED: bool  = True

    changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self._info = QLabel(self._INFO_LABEL)
        lay.addWidget(self._info)
        self._add_header_widgets(lay)
        self._table = QTableWidget()
        self._table.cellDoubleClicked.connect(self._on_cell_dbl)
        self._table.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self._table)
        self._blocks: list[dict] = []
        self._K = 0

    def _add_header_widgets(self, lay: QVBoxLayout) -> None:
        """Hook: widgets placed between the label and the grid (none here)."""

    def _pair_indexed(self) -> bool:
        return self._PAIR_INDEXED

    def _pair_headers(self, K: int) -> tuple[list[str], list[str]]:
        """``(column labels, row labels)`` of the K×K grid."""
        hdrs = [str(k) for k in range(K)]
        return hdrs, hdrs

    # ------------------------------------------------------------------
    # Shared lifecycle
    # ------------------------------------------------------------------

    def load(self, mdl: PMCModel):
        self._K      = mdl.K
        self._blocks = list(getattr(mdl, self._BLOCKS_ATTR)())
        self._layout_grid()
        self._refresh_table()

    def _layout_grid(self):
        K = self._K
        if self._pair_indexed():
            cols, rows = self._pair_headers(K)
            self._table.setRowCount(K)
            self._table.setColumnCount(K)
            self._table.setHorizontalHeaderLabels(cols)
            self._table.setVerticalHeaderLabels(rows)
        else:
            # State-indexed (one row per state) — single-column grid.
            self._table.setRowCount(K)
            self._table.setColumnCount(1)
            self._table.setHorizontalHeaderLabels(["density"])
            self._table.setVerticalHeaderLabels([f"i={k}" for k in range(K)])

    def _refresh_table(self):
        K = self._K
        if self._pair_indexed():
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
        """Pair-indexed lookup (used by Copulas tab and pair margins)."""
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
        """State-indexed lookup (used by Margins tab for state margins)."""
        for blk in self._blocks:
            if int(blk.get("i", -1)) == i:
                return blk
        return None

    def _block_at(self, row: int, col: int) -> dict | None:
        """The block displayed in cell ``(row, col)``."""
        if self._pair_indexed():
            return self._get_block(row, col)
        return self._get_block_state(row)

    def _on_cell_dbl(self, row: int, col: int):
        blk = self._block_at(row, col)
        if blk is None:
            return
        dlg = self._DIALOG(blk, self)
        if dlg.exec():
            self._apply_edit(blk, dlg.get_block())
            self._refresh_table()
            self.changed.emit()

    def _apply_edit(self, blk: dict, new: dict) -> None:
        """Write the dialog's result ``new`` into the block ``blk`` in place."""
        blk.update(new)

    def save(self, raw: dict):
        raw[self._RAW_KEY] = copy.deepcopy(self._blocks)

    # ------------------------------------------------------------------
    # Subclasses customise only this method.
    # ------------------------------------------------------------------

    def _format_block(self, blk: dict) -> str:                 # pragma: no cover
        raise NotImplementedError


class _MarginTab(_BlockGridTab):
    """Marginal densities — one per state (f_i) or one per pair (f_ij).

    The grid follows :attr:`PMCModel.margin_structure`:

    * ``"state"`` — K cells, row ``i``: f_i, the law of y_n given x_n = i.
      With such margins the hidden chain X is Markov (Proposition of
      DerrodePieczynski_CSDA2013 §2.1) and a PMC reduces to an SR HMC-DN.
    * ``"pair"`` — K×K cells, row ``i``, column ``j``: f_ij, the law of y_n
      given (x_n = i, x_{n+1} = j) — the general PMC of DerrodePieczynski_CSDA2013
      Eqs. 12–14. The right observation of the pair (i, j) follows f_ji. PMC and
      PMC-IN only.

    The *Margin structure* selector switches between the two, for the variants
    that accept pair margins:

    * state → pair copies f_i into every f_ij — the same model, whose K²
      densities can then be edited apart;
    * pair → state keeps f_i0 for each state, the anchor also kept by
      ``[model].margin_structure = "state"``, after a confirmation when some
      f_ij differ across j (that j-dependence is discarded).

    :meth:`save` writes ``[[margins]]`` in the structure's own format (K
    blocks keyed by ``i``, or K² blocks keyed by ``i`` and ``j``). An explicit
    ``[model].margin_structure`` is kept in step with the structure shown; an
    absent one stays absent, the format alone determining the structure.

    Double-click a cell to edit.

    Reference: Derrode, S. & Pieczynski, W. (2013). *Computational Statistics
    & Data Analysis* 63, 81–98 (DerrodePieczynski_CSDA2013).
    """

    _INFO_STATE = ("Margins f_i: law of y_n given x_n = i (one per state) "
                   "— double-click to edit.")
    _INFO_PAIR  = ("Margins f_ij: law of y_n given (x_n = i, x_{n+1} = j) — "
                   "row i, column j (general PMC, "
                   "DerrodePieczynski_CSDA2013 Eqs. 12–14) — "
                   "double-click to edit.")
    _INFO_LABEL   = _INFO_STATE
    _RAW_KEY      = "margins"
    _DIALOG       = _MarginDialog
    _BLOCKS_ATTR  = "margin_blocks"

    #: ``(combobox text, margin_structure)`` in display order.
    _STRUCTURES = (
        ("state — f_i, one margin per state", "state"),
        ("pair — f_ij, one margin per pair (i, j)", "pair"),
    )

    # Until :meth:`load` reads them from a model.
    _structure:    str  = "state"
    _pair_allowed: bool = False

    def __init__(self):
        super().__init__()
        self._info.setWordWrap(True)

    def _add_header_widgets(self, lay: QVBoxLayout) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel("Margin structure:"))
        self._combo_structure = QComboBox()
        for text, value in self._STRUCTURES:
            self._combo_structure.addItem(text, value)
        self._combo_structure.currentIndexChanged.connect(
            self._on_structure_changed)
        row.addWidget(self._combo_structure, stretch=1)
        lay.addLayout(row)

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @property
    def structure(self) -> str:
        """The margin structure currently displayed: ``"state"`` or ``"pair"``."""
        return self._structure

    def _pair_indexed(self) -> bool:
        return self._structure == "pair"

    def _pair_headers(self, K: int) -> tuple[list[str], list[str]]:
        return [f"j={k}" for k in range(K)], [f"i={k}" for k in range(K)]

    def _sync_structure_widgets(self):
        self._info.setText(
            self._INFO_PAIR if self._structure == "pair" else self._INFO_STATE)
        was = self._combo_structure.blockSignals(True)
        try:
            idx = self._combo_structure.findData(self._structure)
            self._combo_structure.setCurrentIndex(max(idx, 0))
        finally:
            self._combo_structure.blockSignals(was)
        self._combo_structure.setEnabled(self._pair_allowed)
        self._combo_structure.setToolTip(
            "State margins f_i make the hidden chain Markov "
            "(DerrodePieczynski_CSDA2013 §2.1 Proposition); pair margins f_ij "
            "give the general PMC (DerrodePieczynski_CSDA2013 Eqs. 12–14)."
            if self._pair_allowed else
            "This variant takes state margins only: pair margins f_ij are "
            "defined for PMC and PMC-IN "
            "(DerrodePieczynski_CSDA2013 §2.1 Proposition)."
        )

    def load(self, mdl: PMCModel):
        self._structure = mdl.margin_structure
        self._pair_allowed = bool(mdl.variant.allows_pair_margins)
        super().load(mdl)
        self._sync_structure_widgets()

    def _on_structure_changed(self, idx: int):
        new = self._combo_structure.itemData(idx)
        if new == self._structure or self._K == 0:
            return
        if new == "pair" and not self._pair_allowed:
            self._sync_structure_widgets()              # HMC-*: state only
            return
        if new == "pair":
            blocks = self._state_to_pair(self._blocks, self._K)
        else:
            untied = self._untied_states(self._blocks, self._K)
            if untied:
                ans = QMessageBox.question(
                    self, "Collapse pair margins?",
                    "Switching to state margins keeps f_i0 for each state i "
                    "and discards the other f_ij.\n\n"
                    f"f_ij depends on j for state(s) {untied}: that "
                    "dependence will be lost.\n\nContinue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if ans != QMessageBox.StandardButton.Yes:
                    self._sync_structure_widgets()          # revert the combo
                    return
            blocks = self._pair_to_state(self._blocks, self._K)
        self._blocks = blocks
        self._structure = new
        self._layout_grid()
        self._refresh_table()
        self._sync_structure_widgets()
        self.changed.emit()

    @staticmethod
    def _state_to_pair(blocks: list[dict], K: int) -> list[dict]:
        """K state blocks → K² pair blocks with f_ij = f_i (sorted by (i, j))."""
        by_i = {int(b["i"]): b for b in blocks}
        out: list[dict] = []
        for i in range(K):
            src = by_i[i]
            for j in range(K):
                blk = {"i": i, "j": j, "dist": src["dist"],
                       "params": copy.deepcopy(src.get("params", {}))}
                if "candidates" in src:
                    blk["candidates"] = list(src["candidates"])
                out.append(blk)
        return out

    @staticmethod
    def _pair_to_state(blocks: list[dict], K: int) -> list[dict]:
        """K² pair blocks → K state blocks keeping the anchor f_i0."""
        by_ij = {(int(b["i"]), int(b["j"])): b for b in blocks}
        out: list[dict] = []
        for i in range(K):
            src = by_ij[(i, 0)]
            blk = {"i": i, "dist": src["dist"],
                   "params": copy.deepcopy(src.get("params", {}))}
            if "candidates" in src:
                blk["candidates"] = list(src["candidates"])
            out.append(blk)
        return out

    @staticmethod
    def _untied_states(blocks: list[dict], K: int) -> list[int]:
        """States i whose f_ij are not all equal to f_i0 (dist, params, candidates)."""
        by_ij = {(int(b["i"]), int(b["j"])): b for b in blocks}

        def _key(b):
            return (b.get("dist"), b.get("params", {}), b.get("candidates"))

        return [i for i in range(K)
                if any(_key(by_ij[(i, j)]) != _key(by_ij[(i, 0)])
                       for j in range(1, K))]

    # ------------------------------------------------------------------
    # Editing and persistence
    # ------------------------------------------------------------------

    def _apply_edit(self, blk: dict, new: dict) -> None:
        # The dialog omits ``candidates`` when none is ticked: that means "no
        # GICE for this margin", so the key must go, not linger from before.
        blk.update(new)
        if "candidates" not in new:
            blk.pop("candidates", None)

    def save(self, raw: dict):
        super().save(raw)
        model = raw.get("model")
        if isinstance(model, dict) and "margin_structure" in model:
            model["margin_structure"] = self._structure

    def _format_block(self, blk: dict) -> str:
        params = ", ".join(f"{k}={v}" for k, v in blk.get("params", {}).items())
        return f"{blk['dist']}({params})"


class _CopulaTab(_BlockGridTab):
    """K×K grid of cells for copula selection and τ."""

    _INFO_LABEL  = "Copulas c_{ij}  — double-click a cell to edit."
    _RAW_KEY     = "copulas"
    _DIALOG      = _CopulaDialog
    _BLOCKS_ATTR = "copula_blocks"

    @staticmethod
    def _fmt_value(v, spec: str) -> str:
        """Format ``v`` numerically, falling back to ``str`` when it is not.

        The block dict comes straight from the TOML and PMCModel keeps keys
        it does not know, so an extra may be a comment string or a list. The
        bare ``f"{v:.3g}"`` raised on those — while painting the grid, on the
        unguarded stretch of ``_load_model``, i.e. a process abort.
        """
        try:
            return format(float(v), spec)
        except (TypeError, ValueError):
            return str(v)

    def _format_block(self, blk: dict) -> str:
        extras = ", ".join(
            f"{k}={self._fmt_value(v, '.3g')}" for k, v in blk.items()
            if k not in ("name", "tau", "i", "j")
        )
        text = (f"{blk.get('name','?')}  "
                f"τ={self._fmt_value(blk.get('tau', 0), '.3f')}")
        if extras:
            text += f"  ({extras})"
        return text


class _IceTab(QWidget):
    """ICE / SEM estimator configuration panel.

    Exposes every key parsed by :func:`pmcprg.pmc.ice._parse_ice_cfg`
    (``max_iter``, ``tol``, ``patience``, ``return_best_iterate``,
    ``fit_margins``, ``candidates``,
    ``init``, ``kmeans_seed``, ``n_starts``, ``multistart_seed``,
    ``multistart_jitter``, ``multistart_workers``, ``multistart_families``,
    ``selection_criterion``, ``margin_selection_rule``,
    ``missing_strategy``, ``missing_draws``, ``missing_seed``) and adds the
    algorithm switch ``algorithm`` plus
    its companion ``sem_seed`` for the SEM stochastic completion (see
    :func:`pmcprg.pmc.sem._parse_sem_cfg`).

    The multistart-specific widgets (seed, jitter, workers, families) are
    disabled when
    ``n_starts == 1``; the K-means seed widget is disabled when
    ``init != "kmeans"``; the imputation-only widgets (``missing_draws``,
    ``missing_seed``) are disabled when ``missing_strategy != "impute"`` —
    so it is visually clear they have no effect.
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
        self._spn_sem_seed.setValue(_SEM_DEFAULTS["sem_seed"])
        self._spn_sem_seed.setToolTip(
            "RNG seed for SEM's per-iteration FFBS draw."
        )

        # ── Convergence controls ──────────────────────────────────────
        self._spn_maxiter = QSpinBox()
        self._spn_maxiter.setRange(1, 1000)
        self._spn_maxiter.setValue(_ICE_DEFAULTS["max_iter"])

        self._spn_tol = QDoubleSpinBox()
        self._spn_tol.setRange(1e-10, 1.0)
        self._spn_tol.setDecimals(8)
        self._spn_tol.setValue(_ICE_DEFAULTS["tol"])
        self._spn_tol.setSingleStep(1e-5)

        self._spn_patience = QSpinBox()
        self._spn_patience.setRange(1, 50)
        self._spn_patience.setValue(_ICE_DEFAULTS["patience"])
        self._spn_patience.setToolTip(
            "Stop early after this many consecutive log-likelihood regressions."
        )

        self._chk_best_iterate = QCheckBox()
        self._chk_best_iterate.setChecked(bool(_ICE_DEFAULTS["return_best_iterate"]))
        self._chk_best_iterate.setToolTip(
            "Return the iterate with the highest log-likelihood instead of the\n"
            "last one. ICE is not monotone: after 'Patience' regressions it stops\n"
            "below its best iterate. A run that reaches 'Max iterations' then\n"
            "evaluates its last model once more. For SEM, a heuristic: the best\n"
            "point of a noisy chain, not its average. Multistart ranks the\n"
            "starts by the log-likelihood of the model each one returns."
        )

        # ── Selection criteria ────────────────────────────────────────
        # Driven by the registries so a criterion added to the package cannot
        # go missing from the GUI (they all used to be unreachable here).
        self._combo_criterion = QComboBox()
        self._combo_criterion.addItems(list(SELECTION_CRITERIA))
        self._combo_criterion.setCurrentText(_ICE_DEFAULTS["selection_criterion"])
        self._combo_criterion.setToolTip(
            "Rule for picking the copula family at every M-step.\n"
            "  mle          — highest weighted log-likelihood (no penalty)\n"
            "  aic / bic    — penalise the parameter count; bic uses the\n"
            "                 block's effective sample size Σξ\n"
            "  huard        — Bayesian evidence, uniform prior on each\n"
            "                 family's own τ-range (CSDA-2013 Eq. 20)\n"
            "  huard_common — same, integrated over the shared τ-support\n"
            "  huard_global — same, prior of common mass (diagnostic: these\n"
            "                 two isolate how much of huard's edge is the\n"
            "                 declared τ-range rather than the averaging)\n"
            "  cvm          — ξ-weighted Cramér–von Mises distance\n"
            "  xvcic        — cross-validated out-of-fold log-likelihood\n"
            "                 (better founded, ~5.6× the cost)"
        )

        self._combo_margin_rule = QComboBox()
        self._combo_margin_rule.addItems(sorted(MARGIN_SELECTION_RULES))
        self._combo_margin_rule.setCurrentText(_ICE_DEFAULTS["margin_selection_rule"])
        self._combo_margin_rule.setToolTip(
            "Rule for picking each margin's family when 'Fit margins' is on\n"
            "(GICE, SP-2016). Only used when margins are re-estimated."
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
            "M-step. Requires scikit-learn (pip install awesomepmc[ml])."
        )
        self._combo_init.currentTextChanged.connect(self._on_init_changed)

        self._spn_kmeans_seed = QSpinBox()
        self._spn_kmeans_seed.setRange(0, 2_147_483_647)
        self._spn_kmeans_seed.setValue(_ICE_DEFAULTS["kmeans_seed"])
        self._spn_kmeans_seed.setToolTip(
            "RNG seed forwarded to sklearn.cluster.KMeans."
        )

        # ── Multistart ────────────────────────────────────────────────
        self._spn_n_starts = QSpinBox()
        # Up to a full family sweep (1 + |C|^K ≤ 1 + MAX_SWEEP_COMBINATIONS).
        self._spn_n_starts.setRange(1, 1 + MAX_SWEEP_COMBINATIONS)
        self._spn_n_starts.setValue(_ICE_DEFAULTS["n_starts"])
        self._spn_n_starts.setToolTip(
            "Number of independent ICE runs from random perturbations of "
            "the initial model. The best run (highest final log-likelihood) "
            "is returned."
        )
        self._spn_n_starts.valueChanged.connect(self._on_n_starts_changed)

        self._spn_ms_seed = QSpinBox()
        self._spn_ms_seed.setRange(0, 2_147_483_647)
        # Deliberate GUI-side deviation from the API default (0): 42 is more
        # recognisable, and the 🎲 button re-rolls it anyway (tested).
        self._spn_ms_seed.setValue(42)
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

        self._spn_ms_workers = QSpinBox()
        self._spn_ms_workers.setRange(1, max(1, (os.cpu_count() or 2)))
        self._spn_ms_workers.setValue(_ICE_DEFAULTS["multistart_workers"])
        self._spn_ms_workers.setToolTip(
            "Run the multistart fits in this many worker processes.\n"
            "Results are identical to the sequential run (the starting points\n"
            "are pre-drawn), only wall-clock changes. Above 1, per-iteration\n"
            "progress cannot cross process boundaries, so the progress bar\n"
            "becomes indeterminate until the runs finish."
        )

        self._spn_ms_jitter = QDoubleSpinBox()
        self._spn_ms_jitter.setRange(0.01, 0.50)
        self._spn_ms_jitter.setDecimals(2)
        self._spn_ms_jitter.setSingleStep(0.05)
        self._spn_ms_jitter.setValue(_ICE_DEFAULTS["multistart_jitter"])
        self._spn_ms_jitter.setToolTip(
            "Relative perturbation amplitude (fraction of each parameter's "
            "natural scale). 0.10 = 10 %."
        )

        self._combo_ms_families = QComboBox()
        self._combo_ms_families.addItems(list(MULTISTART_FAMILY_MODES))
        self._combo_ms_families.setCurrentText(_ICE_DEFAULTS["multistart_families"])
        self._combo_ms_families.setToolTip(
            "Copula families of the starts after the first (variants with copulas).\n"
            "  none   — parameter jitter only; every start keeps the model's\n"
            "           families, so it cannot leave a wrong-family basin\n"
            "  random — each start also redraws the family of every pair (i, j)\n"
            "           among the candidates, τ in the central band of its range\n"
            "  sweep  — every combination of candidates on the diagonal pairs\n"
            "           (i, i) at τ = 0.5: needs n_starts = 1 + |C|^K (e.g. 10\n"
            "           for 3 candidates, K = 2); fewer drops combinations,\n"
            "           more adds random starts"
        )

        # ── Missing observations (NaN rows of Y) ───────────────────────
        # Driven by MISSING_STRATEGIES so a strategy added to the package
        # cannot go missing from the GUI, same rationale as the criterion
        # combos above.
        self._combo_missing_strategy = QComboBox()
        self._combo_missing_strategy.addItems(list(MISSING_STRATEGIES))
        self._combo_missing_strategy.setCurrentText(_MISSING_DEFAULTS["missing_strategy"])
        self._combo_missing_strategy.setToolTip(
            "Strategy for the missing observations (NaN rows of Y):\n"
            "  available — exact posteriors from the observed data only;\n"
            "              deterministic and 3-4x (grid variants) to 8-19x\n"
            "              (GICE) faster than impute with 5 draws (default)\n"
            "  impute    — 'Imputation draws' completed series per iteration,\n"
            "              estimates averaged; on the GICE fixture it recovers\n"
            "              the true margin families in 57% of fits vs 39% for\n"
            "              available (61% on complete data). Prefer it when\n"
            "              margin families are selected and the extra cost is\n"
            "              acceptable."
        )
        self._combo_missing_strategy.currentTextChanged.connect(self._on_missing_strategy_changed)

        self._spn_missing_draws = QSpinBox()
        self._spn_missing_draws.setRange(1, 100)
        self._spn_missing_draws.setValue(_MISSING_DEFAULTS["missing_draws"])
        self._spn_missing_draws.setToolTip(
            "Number of completed series drawn per ICE iteration when "
            "missing_strategy is 'impute' (no effect otherwise)."
        )

        self._spn_missing_seed = QSpinBox()
        self._spn_missing_seed.setRange(0, 2_147_483_647)
        self._spn_missing_seed.setValue(_MISSING_DEFAULTS["missing_seed"])
        self._spn_missing_seed.setToolTip(
            "RNG seed of the imputation draws when missing_strategy is\n"
            "'impute' (no effect otherwise). A multistart run's start s uses\n"
            "missing_seed + s."
        )

        # ── Layout ────────────────────────────────────────────────────
        lay.addRow(QLabel("<b>Algorithm</b>"))
        lay.addRow("Estimator:",              self._combo_algorithm)
        lay.addRow("SEM seed:",               self._spn_sem_seed)
        lay.addRow("Max iterations:",         self._spn_maxiter)
        lay.addRow("Convergence tol:",        self._spn_tol)
        lay.addRow("Patience (regressions):", self._spn_patience)
        lay.addRow("Return best iterate:",    self._chk_best_iterate)
        lay.addRow(QLabel("<b>Selection</b>"))
        lay.addRow("Copula criterion:",       self._combo_criterion)
        lay.addRow("Margin rule:",            self._combo_margin_rule)
        lay.addRow("Fit margins:",            self._chk_margins)
        lay.addRow(self._cand_box)
        lay.addRow(QLabel("<b>Initialisation</b>"))
        lay.addRow("Init strategy:",          self._combo_init)
        lay.addRow("K-means seed:",           self._spn_kmeans_seed)
        lay.addRow(QLabel("<b>Multistart</b>"))
        lay.addRow("n_starts:",               self._spn_n_starts)
        lay.addRow("Seed:",                   seed_box)
        lay.addRow("Jitter:",                 self._spn_ms_jitter)
        lay.addRow("Start families:",         self._combo_ms_families)
        lay.addRow("Worker processes:",       self._spn_ms_workers)
        lay.addRow(QLabel("<b>Missing observations</b>"))
        lay.addRow("Strategy:",               self._combo_missing_strategy)
        lay.addRow("Imputation draws:",       self._spn_missing_draws)
        lay.addRow("Imputation seed:",        self._spn_missing_seed)

        # Disable multistart widgets initially (n_starts=1) and the K-means
        # seed (init=model). The SEM seed defaults to disabled too — only
        # SEM uses it. Same for the imputation-only widgets (strategy is
        # "available" by default).
        self._on_n_starts_changed(1)
        self._on_init_changed(self._combo_init.currentText())
        self._on_algorithm_changed(self._combo_algorithm.currentText())
        self._on_missing_strategy_changed(self._combo_missing_strategy.currentText())

        # Wire change-tracking — every editable widget signals ``changed``.
        for w in (
            self._spn_maxiter, self._spn_tol, self._spn_patience,
            self._spn_n_starts, self._spn_ms_seed, self._spn_ms_jitter,
            self._spn_kmeans_seed, self._spn_sem_seed,
            self._spn_missing_draws, self._spn_missing_seed,
        ):
            w.valueChanged.connect(self.changed)
        self._chk_margins.toggled.connect(self.changed)
        self._chk_best_iterate.toggled.connect(self.changed)
        self._combo_init.currentTextChanged.connect(self.changed)
        self._combo_algorithm.currentTextChanged.connect(self.changed)
        self._combo_ms_families.currentTextChanged.connect(self.changed)
        self._combo_missing_strategy.currentTextChanged.connect(self.changed)

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_n_starts_changed(self, value: int):
        active = value > 1
        self._spn_ms_seed.setEnabled(active)
        self._btn_seed_random.setEnabled(active)
        self._spn_ms_jitter.setEnabled(active)
        self._spn_ms_workers.setEnabled(active)
        self._combo_ms_families.setEnabled(active)

    def _on_init_changed(self, value: str):
        """Enable the K-means seed widget only when init=='kmeans'."""
        self._spn_kmeans_seed.setEnabled(value == "kmeans")

    def _on_missing_strategy_changed(self, value: str):
        """Enable the imputation-only widgets only when strategy=='impute'."""
        active = value == "impute"
        self._spn_missing_draws.setEnabled(active)
        self._spn_missing_seed.setEnabled(active)

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

    @staticmethod
    def _spin_value(spin, value, default, *, cast=int):
        """Coerce a TOML value into something ``spin`` can actually hold.

        ``mdl.ice_config()`` hands the ``[ice]`` table over verbatim —
        PMCModel neither type-checks nor range-checks it — and the bare
        ``int(...)`` / ``setValue(...)`` below raised on a non-numeric value
        (``max_iter = "many"``) or one outside int32 (a 20-digit seed). That
        happens inside ``_load_model``'s unguarded stretch, so it took the
        whole GUI down. The two *string* keys (``init``, ``algorithm``)
        already fell back to a default; the numeric ones did not.
        """
        try:
            out = cast(value)
        except (TypeError, ValueError):
            out = cast(default)
        return max(spin.minimum(), min(spin.maximum(), out))

    def load(self, cfg: dict, show_copula_cfg: bool = True):
        # Hide the candidate-copulas group for variants that do not use
        # copulas (HMC-IN). The whole QGroupBox row hides together.
        self._cand_box.setVisible(show_copula_cfg)

        # Block the changed signal while we programmatically populate widgets.
        self.blockSignals(True)
        try:
            self._chk_margins.setChecked(
                bool(cfg.get("fit_margins", _ICE_DEFAULTS["fit_margins"])))
            self._spn_maxiter.setValue(self._spin_value(
                self._spn_maxiter, cfg.get("max_iter", _ICE_DEFAULTS["max_iter"]),
                _ICE_DEFAULTS["max_iter"], cast=int))
            self._spn_tol.setValue(self._spin_value(
                self._spn_tol, cfg.get("tol", _ICE_DEFAULTS["tol"]),
                _ICE_DEFAULTS["tol"], cast=float))
            self._spn_patience.setValue(self._spin_value(
                self._spn_patience, cfg.get("patience", _ICE_DEFAULTS["patience"]),
                _ICE_DEFAULTS["patience"], cast=int))
            best = cfg.get("return_best_iterate", _ICE_DEFAULTS["return_best_iterate"])
            # Only a real boolean (or 0 / 1) — bool("false") is True.
            if not (isinstance(best, (bool, int)) and best in (0, 1)):
                best = _ICE_DEFAULTS["return_best_iterate"]
            self._chk_best_iterate.setChecked(bool(best))
            self._set_candidates(cfg.get(
                "candidates", list(self._DEFAULT_CANDIDATES_CHECKED),
            ))

            init_value = str(cfg.get("init", _ICE_DEFAULTS["init"]))
            if init_value not in ("model", "kmeans"):
                init_value = "model"
            self._combo_init.setCurrentText(init_value)
            self._spn_kmeans_seed.setValue(self._spin_value(
                self._spn_kmeans_seed, cfg.get("kmeans_seed", _ICE_DEFAULTS["kmeans_seed"]),
                _ICE_DEFAULTS["kmeans_seed"], cast=int))
            self._on_init_changed(self._combo_init.currentText())

            algo_value = str(cfg.get("algorithm", "ice"))
            if algo_value not in ("ice", "sem"):
                algo_value = "ice"
            self._combo_algorithm.setCurrentText(algo_value)
            self._spn_sem_seed.setValue(self._spin_value(
                self._spn_sem_seed, cfg.get("sem_seed", _SEM_DEFAULTS["sem_seed"]),
                _SEM_DEFAULTS["sem_seed"], cast=int))
            self._on_algorithm_changed(self._combo_algorithm.currentText())

            self._spn_n_starts.setValue(self._spin_value(
                self._spn_n_starts, cfg.get("n_starts", _ICE_DEFAULTS["n_starts"]),
                _ICE_DEFAULTS["n_starts"], cast=int))
            self._spn_ms_seed.setValue(self._spin_value(
                self._spn_ms_seed, cfg.get("multistart_seed", 42), 42))
            self._spn_ms_jitter.setValue(self._spin_value(
                self._spn_ms_jitter, cfg.get("multistart_jitter", _ICE_DEFAULTS["multistart_jitter"]),
                _ICE_DEFAULTS["multistart_jitter"], cast=float))
            self._spn_ms_workers.setValue(self._spin_value(
                self._spn_ms_workers, cfg.get("multistart_workers", _ICE_DEFAULTS["multistart_workers"]),
                _ICE_DEFAULTS["multistart_workers"], cast=int))
            families = str(cfg.get("multistart_families",
                                   _ICE_DEFAULTS["multistart_families"]))
            if families not in MULTISTART_FAMILY_MODES:
                families = _ICE_DEFAULTS["multistart_families"]
            self._combo_ms_families.setCurrentText(families)
            self._combo_criterion.setCurrentText(str(cfg.get(
                "selection_criterion", _ICE_DEFAULTS["selection_criterion"])))
            self._combo_margin_rule.setCurrentText(str(cfg.get(
                "margin_selection_rule",
                _ICE_DEFAULTS["margin_selection_rule"])))
            self._on_n_starts_changed(self._spn_n_starts.value())

            strategy = str(cfg.get("missing_strategy", _MISSING_DEFAULTS["missing_strategy"]))
            if strategy not in MISSING_STRATEGIES:
                strategy = _MISSING_DEFAULTS["missing_strategy"]
            self._combo_missing_strategy.setCurrentText(strategy)
            self._spn_missing_draws.setValue(self._spin_value(
                self._spn_missing_draws, cfg.get("missing_draws", _MISSING_DEFAULTS["missing_draws"]),
                _MISSING_DEFAULTS["missing_draws"], cast=int))
            self._spn_missing_seed.setValue(self._spin_value(
                self._spn_missing_seed, cfg.get("missing_seed", _MISSING_DEFAULTS["missing_seed"]),
                _MISSING_DEFAULTS["missing_seed"], cast=int))
            self._on_missing_strategy_changed(self._combo_missing_strategy.currentText())
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
            "return_best_iterate": self._chk_best_iterate.isChecked(),
            "candidates":        self._selected_candidates(),
            "init":              self._combo_init.currentText(),
            "kmeans_seed":       self._spn_kmeans_seed.value(),
            "n_starts":          self._spn_n_starts.value(),
            "multistart_seed":   self._spn_ms_seed.value(),
            "multistart_jitter": self._spn_ms_jitter.value(),
            "multistart_workers": self._spn_ms_workers.value(),
            "multistart_families": self._combo_ms_families.currentText(),
            "selection_criterion":   self._combo_criterion.currentText(),
            "margin_selection_rule": self._combo_margin_rule.currentText(),
            "missing_strategy":  self._combo_missing_strategy.currentText(),
            "missing_draws":     self._spn_missing_draws.value(),
            "missing_seed":      self._spn_missing_seed.value(),
        }
