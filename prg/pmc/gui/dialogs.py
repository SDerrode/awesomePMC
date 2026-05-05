"""
prg.pmc.gui.dialogs — modal dialogs for editing margin and copula blocks.

These were originally part of ``main_window.py``; extracted to keep the main
window focused on layout. Both classes are unit-testable in isolation
(no PMCMainWindow dependency).

Audit fixes (2026-05):

* ``_CopulaDialog`` now reranges the τ spinbox dynamically as the family
  selection changes, and shows extra-parameter widgets for multi-parameter
  copulas (``δ`` for BB1, ``df`` for Student). Previously τ was hard-bounded
  to ``[-1, 1]`` regardless of family, which let users enter values that
  ``CopulaEnum.correct_tau`` would silently clip at rebuild.
* ``_MarginDialog`` accepts both ``"k=v k=v"`` and ``"k=v, k=v"`` separators
  (previously the comma form silently lost values), and validates that each
  value parses as a float (otherwise the dialog refuses to ``Accept``).
"""

from __future__ import annotations

import re

from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QWidget,
)

from prg.copulas._base import CopulaEnum
from prg.pmc.ice       import EXTRA_PARAM_BOUNDS as _EXTRA_PARAM_DEFAULTS


_COPULA_NAMES = [e.value.SHORT_NAME for e in CopulaEnum if e.value.AVAILABLE]


def _entry_for(short_name: str) -> CopulaEnum | None:
    """Return the CopulaEnum entry for a SHORT_NAME (or ``None``)."""
    for c in CopulaEnum:
        if c.value.AVAILABLE and c.value.SHORT_NAME == short_name:
            return c
    return None


class _MarginDialog(QDialog):
    """Inline dialog to edit a margin block dict (dist + params).

    The "Parameters" line accepts ``k=v`` tokens separated by either spaces
    or commas (``"loc=0 scale=1"``, ``"loc=0, scale=1"``, or even
    ``"loc=0,scale=1"`` all work). Invalid float values raise a friendly
    QMessageBox instead of an unhandled ValueError.
    """

    # Tokens like ``key=value`` separated by whitespace and/or commas.
    _TOKEN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^\s,]+)")

    def __init__(self, blk: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Edit margin ({blk.get('i','?')},{blk.get('j','?')})"
        )
        lay = QFormLayout(self)

        self._dist = QLineEdit(blk.get("dist", "norm"))
        params = blk.get("params", {})
        # Use comma+space as separator so the placeholder is unambiguous.
        self._params_edit = QLineEdit(
            ", ".join(f"{k}={v}" for k, v in params.items())
        )
        self._params_edit.setPlaceholderText("loc=0, scale=1")

        # GICE candidate set (Derrode-Pieczynski SP 2016 §3) — comma-
        # separated scipy.stats family names. When non-empty, ICE will
        # auto-select the family from this list at every M-step.
        cands = blk.get("candidates", [])
        self._candidates_edit = QLineEdit(", ".join(cands))
        self._candidates_edit.setPlaceholderText(
            "norm, gamma, invgamma, betaprime  (leave blank to disable GICE)"
        )

        lay.addRow("Distribution:",        self._dist)
        lay.addRow("Parameters (k=v, …):", self._params_edit)
        lay.addRow("GICE candidates:",     self._candidates_edit)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)

    # ------------------------------------------------------------------
    # Parsing / validation
    # ------------------------------------------------------------------

    def _parse_params(self) -> tuple[dict, str | None]:
        """Parse the params line. Returns ``(params, error)``.

        ``error`` is ``None`` on success, otherwise a human-readable message
        suitable for a QMessageBox.
        """
        text   = self._params_edit.text().strip()
        params: dict = {}
        if not text:
            return params, None

        # Detect tokens that are *not* of the ``key=value`` form so we can
        # complain about them rather than silently dropping them.
        consumed = []
        for m in self._TOKEN_RE.finditer(text):
            consumed.append((m.start(), m.end()))
            key, val = m.group(1), m.group(2)
            try:
                params[key] = float(val)
            except ValueError:
                return params, f"Cannot parse value {val!r} for parameter {key!r} as a number."

        # Anything outside the consumed spans (and not just commas/whitespace)
        # is junk — flag it.
        residue = list(text)
        for s, e in consumed:
            for k in range(s, e):
                residue[k] = " "
        leftover = "".join(residue).replace(",", " ").strip()
        if leftover:
            return params, f"Unparseable token(s): {leftover!r}. Use ‘k=v, k=v’ syntax."

        return params, None

    def _on_accept(self):
        params, err = self._parse_params()
        if err is not None:
            QMessageBox.warning(self, "Invalid parameters", err)
            return
        # Cache for get_block (we already have it; no need to re-parse).
        self._parsed_params = params
        self.accept()

    # ------------------------------------------------------------------
    # Public API used by the host tab
    # ------------------------------------------------------------------

    def exec(self) -> bool:
        return super().exec() == QDialog.DialogCode.Accepted

    def get_block(self) -> dict:
        # If the dialog was shown and accepted, _parsed_params is set.
        # Otherwise (e.g. test code that does not call exec()), re-parse
        # leniently and just drop unparseable tokens for back-compat.
        params = getattr(self, "_parsed_params", None)
        if params is None:
            params, _ = self._parse_params()
        out: dict = {"dist": self._dist.text().strip(), "params": params}
        # GICE candidate list — only present in the output when the user
        # actually filled it in. Empty means "no GICE for this margin".
        cand_text = self._candidates_edit.text().strip()
        if cand_text:
            cands = [c.strip() for c in cand_text.split(",") if c.strip()]
            if cands:
                out["candidates"] = cands
        return out


class _CopulaDialog(QDialog):
    """Inline dialog to edit a copula block dict (family + τ + extras).

    The τ spinbox is reranged whenever the family selection changes, and
    extra-parameter widgets (``δ`` for BB1, ``df`` for Student) appear
    only when the selected family declares them.
    """

    def __init__(self, blk: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"Edit copula ({blk.get('i','?')},{blk.get('j','?')})"
        )
        lay = QFormLayout(self)

        self._name = QComboBox()
        self._name.addItems(_COPULA_NAMES)
        cur = blk.get("name", "Gauss")
        idx = self._name.findText(cur)
        if idx >= 0:
            self._name.setCurrentIndex(idx)

        # τ spinbox — range will be set by _refresh_for_family.
        # Using 6 decimals so non-trivial bounds like ±2/9 (FGM) and 1/3
        # (A12, A14) are represented faithfully rather than truncated to
        # a value just outside the family's actual range.
        self._tau = QDoubleSpinBox()
        self._tau.setDecimals(6)
        self._tau.setSingleStep(0.01)

        # Range hint label (shown next to the spinbox so users know what
        # values are valid for the current family).
        self._tau_range_lbl = QLabel("")

        # Extra-parameter widgets (lazy-instantiated; one per param name).
        # We keep all rows present in the layout but show/hide them by
        # family. Storing widget + row label so we can toggle visibility.
        self._extra_widgets: dict[str, tuple[QLabel, QDoubleSpinBox]] = {}
        # Pre-build widgets for every known extra param across all families.
        all_extras: dict[str, tuple[float, float, float]] = {}
        for fam_extras in _EXTRA_PARAM_DEFAULTS.values():
            all_extras.update(fam_extras)

        lay.addRow("Family (SHORT_NAME):", self._name)

        # τ row with range hint
        tau_box = QWidget()
        tau_lay = QHBoxLayout(tau_box)
        tau_lay.setContentsMargins(0, 0, 0, 0)
        tau_lay.addWidget(self._tau, stretch=1)
        tau_lay.addWidget(self._tau_range_lbl, stretch=0)
        lay.addRow("τ (Kendall):", tau_box)

        for ekey, (lo, hi, default) in all_extras.items():
            lbl  = QLabel(f"{ekey}:")
            spin = QDoubleSpinBox()
            spin.setRange(lo, hi)
            spin.setDecimals(3)
            spin.setSingleStep(0.1)
            spin.setValue(blk.get(ekey, default))
            lay.addRow(lbl, spin)
            self._extra_widgets[ekey] = (lbl, spin)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addRow(btns)

        # Wire family-change signal AFTER all widgets exist.
        self._name.currentTextChanged.connect(self._refresh_for_family)

        # Initial state: configure τ range and extras for the current family.
        # Set tau value AFTER reranging so it is clamped to the valid range.
        self._refresh_for_family(self._name.currentText())
        self._tau.setValue(self._clamp_tau(blk.get("tau", 0.0)))

    # ------------------------------------------------------------------
    # Family-aware widget refresh
    # ------------------------------------------------------------------

    def _clamp_tau(self, tau: float) -> float:
        lo, hi = self._tau.minimum(), self._tau.maximum()
        return max(lo, min(hi, float(tau)))

    def _refresh_for_family(self, name: str):
        """Update τ range and show/hide extra-parameter widgets."""
        entry = _entry_for(name)
        if entry is None:
            return

        tau_min, tau_max = entry.value.TAU_MIN_MAX
        # Preserve the current value when feasible (clamping if needed).
        cur = self._tau.value() if self._tau.maximum() != self._tau.minimum() else None
        self._tau.setRange(float(tau_min), float(tau_max))
        self._tau_range_lbl.setText(
            f"  ∈ [{tau_min:.4f}, {tau_max:.4f}]"
        )
        if cur is not None:
            self._tau.setValue(self._clamp_tau(cur))

        # Show only the extra-param rows that belong to this family.
        cls_name      = entry.value.CLASS_NAME
        active_extras = set(_EXTRA_PARAM_DEFAULTS.get(cls_name, {}).keys())
        for ekey, (lbl, spin) in self._extra_widgets.items():
            visible = ekey in active_extras
            lbl.setVisible(visible)
            spin.setVisible(visible)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exec(self) -> bool:
        return super().exec() == QDialog.DialogCode.Accepted

    def get_block(self) -> dict:
        out: dict = {
            "name": self._name.currentText(),
            "tau":  self._tau.value(),
        }
        # Include any visible extra-param spinbox values.
        entry = _entry_for(out["name"])
        if entry is not None:
            cls_name = entry.value.CLASS_NAME
            for ekey in _EXTRA_PARAM_DEFAULTS.get(cls_name, {}):
                _, spin = self._extra_widgets[ekey]
                out[ekey] = spin.value()
        return out
