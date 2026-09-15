"""
pmcprg.pmc.gui.dialogs — modal dialogs for editing margin and copula blocks.

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
* ``_MarginDialog`` exposes the GICE candidate set as a checkbox grid
  (one box per family in ``ice.GICE_KNOWN_FAMILIES``) plus three quick-pick
  buttons (SP-2016 §3 / All / None) and an "Other" line for scipy.stats
  families outside the known set. Replaces the previous comma-separated
  ``QLineEdit`` whose typo-tolerance was poor.
"""

from __future__ import annotations

import ast
import math
import re

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from pmcprg.copulas._base import CopulaEnum
from pmcprg.pmc           import (
    EXTRA_PARAM_BOUNDS as _EXTRA_PARAM_DEFAULTS,
    GICE_KNOWN_FAMILIES,
    SP2016_DEFAULT_CANDIDATES,
)


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
        # A block keyed by ``i`` alone is a state margin f_i (law of y_n given
        # x_n = i); one keyed by ``i`` and ``j`` is a pair margin f_ij (law of
        # y_n given x_n = i, x_{n+1} = j — A16 Eq. 12).
        if "j" in blk:
            title = (f"Edit margin f_ij  (i={blk.get('i', '?')}, "
                     f"j={blk['j']})")
        else:
            title = f"Edit margin f_i  (i={blk.get('i', '?')})"
        self.setWindowTitle(title)
        lay = QFormLayout(self)

        self._dist = QLineEdit(blk.get("dist", "norm"))
        params = blk.get("params", {})
        # Use comma+space as separator so the placeholder is unambiguous.
        self._params_edit = QLineEdit(
            ", ".join(f"{k}={v}" for k, v in params.items())
        )
        self._params_edit.setPlaceholderText("loc=0, scale=1")

        lay.addRow("Distribution:",        self._dist)
        lay.addRow("Parameters (k=v, …):", self._params_edit)

        # ----- GICE candidate set (SP 2016 §3) ---------------------------
        # Replaces the old comma-separated QLineEdit. The 8 families in
        # ``GICE_KNOWN_FAMILIES`` ship with data-aware init heuristics in
        # ``ice._INIT_PARAM_HEURISTICS``; the "Other" line lets power users
        # add any extra scipy.stats name (it will fall back to scipy's
        # default ``fit`` for an init point).
        cands_in = list(blk.get("candidates", []))
        self._cand_checks: dict[str, QCheckBox] = {}
        cand_box = QGroupBox(
            "GICE candidates (auto-select family at every ICE M-step)"
        )
        cand_v   = QVBoxLayout(cand_box)

        # Hint label so users know what the empty state means.
        cand_v.addWidget(QLabel(
            "Tick the scipy.stats families ICE may try. "
            "Leave everything unchecked to disable GICE for this margin."
        ))

        # 8 known families in a 4-column grid.
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        for k, name in enumerate(GICE_KNOWN_FAMILIES):
            cb = QCheckBox(name)
            cb.setChecked(name in cands_in)
            self._cand_checks[name] = cb
            grid.addWidget(cb, k // 4, k % 4)
        cand_v.addLayout(grid)

        # Quick-pick buttons.
        btn_row = QHBoxLayout()
        b_sp    = QPushButton("SP-2016 §3")
        b_sp.setToolTip(
            "Tick {norm, gamma, invgamma, betaprime} — the four families "
            "used in Derrode-Pieczynski SP 2016, Example 3.1."
        )
        b_all   = QPushButton("All")
        b_none  = QPushButton("None")
        b_sp.clicked.connect(
            lambda: self._set_candidates(SP2016_DEFAULT_CANDIDATES)
        )
        b_all.clicked.connect(
            lambda: self._set_candidates(GICE_KNOWN_FAMILIES)
        )
        b_none.clicked.connect(lambda: self._set_candidates(()))
        for b in (b_sp, b_all, b_none):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        cand_v.addLayout(btn_row)

        # "Other" line for scipy.stats families outside the known set.
        # Anything entered here is appended to the checked names at
        # ``get_block`` time. Stored verbatim (no canonicalisation).
        known = set(GICE_KNOWN_FAMILIES)
        extras = [c for c in cands_in if c not in known]
        self._cand_extras = QLineEdit(", ".join(extras))
        self._cand_extras.setPlaceholderText(
            "Other scipy.stats families (comma-separated, e.g. pareto, t, chi2)"
        )
        cand_v.addWidget(self._cand_extras)

        lay.addRow(cand_box)

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

        # Vector / matrix values first: a multivariate margin reads
        # ``mean=[-1.0, -0.8], cov=[[1.0, 0.3], [0.3, 1.0]]``, which the
        # scalar token scanner below chops at the first comma (it used to
        # report « Cannot parse value '[-1.0' … as a number » and return an
        # EMPTY dict, silently dropping every parameter of the block).
        # Parsed as a call's keywords and evaluated with ``literal_eval``:
        # structure-aware, and no code is executed.
        if "[" in text or "(" in text:
            try:
                node = ast.parse(f"_f({text})", mode="eval").body
                if node.args:
                    raise ValueError("positional values are not supported")
                return {kw.arg: ast.literal_eval(kw.value)
                        for kw in node.keywords}, None
            except (SyntaxError, ValueError) as exc:
                return {}, (
                    f"Cannot parse parameters: {exc}. Use ‘k=v’ with numbers, "
                    "lists (‘mean=[1, 2]’) or matrices (‘cov=[[1, 0], [0, 1]]’)."
                )

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
    # GICE-candidates helpers
    # ------------------------------------------------------------------

    def _set_candidates(self, names) -> None:
        """Tick exactly the families in ``names``; clear everything else.

        Names not present as a checkbox are appended to the "Other" line
        (verbatim) so quick-picks remain idempotent if the caller passes
        an unknown family. ``names`` may be any iterable of strings.
        """
        wanted = list(names)
        wanted_set = set(wanted)
        for short, cb in self._cand_checks.items():
            cb.setChecked(short in wanted_set)
        # Anything not represented as a checkbox goes to the extras line.
        known   = set(self._cand_checks)
        extras  = [w for w in wanted if w not in known]
        self._cand_extras.setText(", ".join(extras))

    def _selected_candidates(self) -> list[str]:
        """Merge ticked checkboxes with the Other-line entries (in order)."""
        ticked = [s for s, cb in self._cand_checks.items() if cb.isChecked()]
        extra_text = self._cand_extras.text().strip()
        extras = (
            [c.strip() for c in extra_text.split(",") if c.strip()]
            if extra_text else []
        )
        # Preserve order, drop dupes (first wins).
        seen: set[str] = set()
        out: list[str] = []
        for c in ticked + extras:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out

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
        # GICE candidate list — only present in the output when at least
        # one box is ticked or the "Other" line is non-empty. An empty
        # list means "no GICE for this margin".
        cands = self._selected_candidates()
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

        # Registered range, cut to the τ the family reaches (Frank |τ| ≤ 0.994299,
        # Plackett −0.99352457 ≤ τ ≤ 0.99352457).
        reg_min, reg_max = entry.value.TAU_MIN_MAX
        tau_min, tau_max = entry.reachable_tau(reg_min), entry.reachable_tau(reg_max)
        # Preserve the current value when feasible (clamping if needed).
        cur = self._tau.value() if self._tau.maximum() != self._tau.minimum() else None
        self._tau.setRange(float(tau_min), float(tau_max))
        # The spin box rounds its range to its decimals. A cut bound rounded
        # outwards (Plackett 0.99352457 → 0.993525) would offer a τ the copula
        # clamps: step it back inside (G2). Registered bounds are left as Qt
        # rounds them.
        scale = 10.0 ** self._tau.decimals()
        if tau_max != reg_max and self._tau.maximum() > tau_max:
            self._tau.setMaximum(math.floor(tau_max * scale) / scale)
        if tau_min != reg_min and self._tau.minimum() < tau_min:
            self._tau.setMinimum(math.ceil(tau_min * scale) / scale)
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
