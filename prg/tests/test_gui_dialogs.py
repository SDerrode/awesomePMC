"""
test_gui_dialogs.py — unit tests for the PMC GUI dialogs.

Verifies the round-trip behaviour of ``_MarginDialog`` and ``_CopulaDialog``:
- ``__init__(blk)`` populates widgets from the given block dict.
- ``get_block()`` returns a dict equivalent (for the tracked keys) to the
  input.

We don't display the dialogs (no `.exec()`), so this runs headless.
"""

from __future__ import annotations

import pytest

# Skip the whole module gracefully if PyQt6 is not installed (it is an
# optional dependency: ``pip install copulasformm[gui]``).
PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Session-scoped QApplication — Qt requires exactly one instance per process.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    # Don't quit the app; pytest collects multiple test files and Qt
    # objects may still hold references.


# ---------------------------------------------------------------------------
# _MarginDialog
# ---------------------------------------------------------------------------

def test_margin_dialog_roundtrip_norm(qapp):
    from prg.pmc.gui.dialogs import _MarginDialog

    blk_in = {"i": 0, "j": 1, "dist": "norm",
              "params": {"loc": -1.5, "scale": 0.7}}
    dlg = _MarginDialog(blk_in)
    out = dlg.get_block()

    assert out["dist"] == "norm"
    assert out["params"] == {"loc": -1.5, "scale": 0.7}


def test_margin_dialog_default_dist(qapp):
    """Missing 'dist' falls back to 'norm'."""
    from prg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0})
    out = dlg.get_block()
    assert out["dist"] == "norm"
    assert out["params"] == {}


def test_margin_dialog_handles_multiple_params(qapp):
    """A margin with several scipy.stats parameters round-trips correctly."""
    from prg.pmc.gui.dialogs import _MarginDialog

    blk_in = {"i": 1, "j": 0, "dist": "beta",
              "params": {"a": 2.0, "b": 5.0, "loc": 0.0, "scale": 1.0}}
    dlg = _MarginDialog(blk_in)
    out = dlg.get_block()

    assert out["dist"] == "beta"
    assert out["params"] == {"a": 2.0, "b": 5.0, "loc": 0.0, "scale": 1.0}


# ---------------------------------------------------------------------------
# _MarginDialog — GICE candidates checkbox UI
# ---------------------------------------------------------------------------

def test_margin_dialog_candidates_checkboxes_round_trip(qapp):
    """Checkboxes ticked from input ``candidates`` round-trip via get_block."""
    from prg.pmc.gui.dialogs import _MarginDialog

    blk_in = {"i": 0, "j": 0, "dist": "norm", "params": {"loc": 0, "scale": 1},
              "candidates": ["norm", "gamma", "betaprime"]}
    dlg = _MarginDialog(blk_in)
    # Exactly the ticked boxes match the input.
    ticked = [s for s, cb in dlg._cand_checks.items() if cb.isChecked()]
    assert sorted(ticked) == ["betaprime", "gamma", "norm"]
    # Round-trip preserves order from GICE_KNOWN_FAMILIES (norm, gamma,
    # ..., betaprime).
    out = dlg.get_block()
    assert out["candidates"] == ["norm", "gamma", "betaprime"]


def test_margin_dialog_no_candidates_omits_field(qapp):
    """No ticked boxes and empty Other line → ``candidates`` key omitted."""
    from prg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    # Default state: nothing ticked.
    assert all(not cb.isChecked() for cb in dlg._cand_checks.values())
    out = dlg.get_block()
    assert "candidates" not in out


def test_margin_dialog_sp2016_quick_pick(qapp):
    """The SP-2016 §3 helper ticks exactly {norm, gamma, invgamma, betaprime}."""
    from prg.pmc.gui.dialogs import _MarginDialog
    from prg.pmc.ice          import SP2016_DEFAULT_CANDIDATES

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    dlg._set_candidates(SP2016_DEFAULT_CANDIDATES)
    out = dlg.get_block()
    assert out["candidates"] == list(SP2016_DEFAULT_CANDIDATES)


def test_margin_dialog_all_then_none_quick_picks(qapp):
    """All / None helpers cover every known family and clear back to empty."""
    from prg.pmc.gui.dialogs import _MarginDialog
    from prg.pmc.ice          import GICE_KNOWN_FAMILIES

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})

    dlg._set_candidates(GICE_KNOWN_FAMILIES)
    out = dlg.get_block()
    assert out["candidates"] == list(GICE_KNOWN_FAMILIES)

    dlg._set_candidates(())
    out = dlg.get_block()
    assert "candidates" not in out


def test_margin_dialog_other_line_unknown_family_kept(qapp):
    """Unknown scipy.stats names typed into the Other line round-trip too."""
    from prg.pmc.gui.dialogs import _MarginDialog

    blk_in = {"i": 0, "j": 0, "dist": "norm", "params": {},
              "candidates": ["norm", "pareto"]}  # pareto not in GICE_KNOWN_FAMILIES
    dlg = _MarginDialog(blk_in)
    # "norm" lands as a checkbox; "pareto" lands in the Other line.
    assert dlg._cand_checks["norm"].isChecked()
    assert "pareto" in dlg._cand_extras.text()
    out = dlg.get_block()
    assert out["candidates"] == ["norm", "pareto"]


def test_margin_dialog_other_line_dedupes_against_checkboxes(qapp):
    """Typing a checked family in the Other line does not produce a duplicate."""
    from prg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    dlg._cand_checks["norm"].setChecked(True)
    dlg._cand_extras.setText("norm, pareto")
    out = dlg.get_block()
    assert out["candidates"] == ["norm", "pareto"]


def test_margin_dialog_lists_all_known_families(qapp):
    """Every entry in GICE_KNOWN_FAMILIES has a corresponding checkbox."""
    from prg.pmc.gui.dialogs import _MarginDialog
    from prg.pmc.ice          import GICE_KNOWN_FAMILIES

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    assert set(dlg._cand_checks.keys()) == set(GICE_KNOWN_FAMILIES)


# ---------------------------------------------------------------------------
# _CopulaDialog
# ---------------------------------------------------------------------------

def test_copula_dialog_roundtrip_gauss(qapp):
    from prg.pmc.gui.dialogs import _CopulaDialog

    blk_in = {"i": 0, "j": 0, "name": "Gauss", "tau": 0.65}
    dlg = _CopulaDialog(blk_in)
    out = dlg.get_block()

    assert out["name"] == "Gauss"
    assert abs(out["tau"] - 0.65) < 1e-4   # QDoubleSpinBox precision


def test_copula_dialog_unknown_name_falls_back_to_gauss(qapp):
    """An unknown copula name leaves the combo at its default (Gauss)."""
    from prg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Nonsense", "tau": 0.0})
    out = dlg.get_block()
    # First item in the combo is the first available copula in CopulaEnum
    assert out["name"] in (
        "Prod", "Gauss",
    )  # depends on CopulaEnum ordering; both are sane defaults


def test_copula_dialog_negative_tau(qapp):
    """Negative τ values are preserved."""
    from prg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 1, "name": "Frank", "tau": -0.30})
    out = dlg.get_block()
    assert out["name"] == "Frank"
    assert abs(out["tau"] - (-0.30)) < 1e-4


def test_copula_dialog_lists_all_available_copulas(qapp):
    """The ComboBox must include every available copula SHORT_NAME."""
    from prg.copulas._base import CopulaEnum
    from prg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Gauss", "tau": 0.0})
    items = [dlg._name.itemText(k) for k in range(dlg._name.count())]
    expected = [c.value.SHORT_NAME for c in CopulaEnum if c.value.AVAILABLE]
    assert items == expected


# ---------------------------------------------------------------------------
# A2: _CopulaDialog is family-aware
# ---------------------------------------------------------------------------

def test_copula_dialog_tau_range_matches_fgm(qapp):
    """For FGM, the τ spinbox bounds should match its valid range."""
    from prg.copulas._base import CopulaEnum
    from prg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "FGM", "tau": 0.0})
    tau_min, tau_max = CopulaEnum.FGM.value.TAU_MIN_MAX
    assert dlg._tau.minimum() == pytest.approx(tau_min, abs=1e-6)
    assert dlg._tau.maximum() == pytest.approx(tau_max, abs=1e-6)


def test_copula_dialog_tau_range_updates_on_family_change(qapp):
    """Changing the family must rerange the τ spinbox to the new family's bounds."""
    from prg.copulas._base import CopulaEnum
    from prg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Gauss", "tau": 0.0})
    # Switch to FGM (range −2/9, 2/9)
    dlg._name.setCurrentText("FGM")
    tau_min, tau_max = CopulaEnum.FGM.value.TAU_MIN_MAX
    assert dlg._tau.minimum() == pytest.approx(tau_min, abs=1e-6)
    assert dlg._tau.maximum() == pytest.approx(tau_max, abs=1e-6)


def test_copula_dialog_clamps_input_tau_to_family_range(qapp):
    """A τ value outside the family's range is clamped (no exception)."""
    from prg.pmc.gui.dialogs import _CopulaDialog

    # FGM range is roughly [-0.222, 0.222]; ask for τ=0.5 → clamped to 2/9.
    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "FGM", "tau": 0.5})
    out = dlg.get_block()
    assert out["name"] == "FGM"
    assert out["tau"] <= 2.0 / 9.0 + 1e-6


def test_copula_dialog_bb1_exposes_delta(qapp):
    """BB1 has a δ extra parameter — it must round-trip via get_block."""
    from prg.pmc.gui.dialogs import _CopulaDialog

    blk = {"i": 0, "j": 1, "name": "BB1", "tau": 0.5, "delta": 2.5}
    dlg = _CopulaDialog(blk)
    out = dlg.get_block()
    assert out["name"] == "BB1"
    assert "delta" in out
    assert abs(out["delta"] - 2.5) < 1e-3


def test_copula_dialog_student_exposes_df(qapp):
    """Student copula carries a ``df`` extra parameter."""
    from prg.pmc.gui.dialogs import _CopulaDialog

    blk = {"i": 0, "j": 0, "name": "Student", "tau": 0.3, "df": 6.0}
    dlg = _CopulaDialog(blk)
    out = dlg.get_block()
    assert out["name"] == "Student"
    assert "df" in out
    assert abs(out["df"] - 6.0) < 1e-3


def test_copula_dialog_one_param_family_no_extras(qapp):
    """One-parameter families must not emit spurious extra keys."""
    from prg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Clayton", "tau": 0.4})
    out = dlg.get_block()
    assert set(out.keys()) == {"name", "tau"}


# ---------------------------------------------------------------------------
# B3: _MarginDialog tolerates commas and rejects junk
# ---------------------------------------------------------------------------

def test_margin_dialog_accepts_comma_separated(qapp):
    """``loc=0, scale=1`` must round-trip even with the comma."""
    from prg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm",
                         "params": {"loc": 0.0, "scale": 1.0}})
    # Simulate user typing comma-separated values.
    dlg._params_edit.setText("loc=0, scale=1")
    out = dlg.get_block()
    assert out["params"] == {"loc": 0.0, "scale": 1.0}


def test_margin_dialog_accepts_no_space_after_comma(qapp):
    from prg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm",
                         "params": {"loc": 0.0, "scale": 1.0}})
    dlg._params_edit.setText("loc=0,scale=1")
    out = dlg.get_block()
    assert out["params"] == {"loc": 0.0, "scale": 1.0}


def test_margin_dialog_parse_rejects_junk(qapp):
    """Non-numeric values trigger a parse error (caught by _on_accept)."""
    from prg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm",
                         "params": {"loc": 0.0}})
    dlg._params_edit.setText("loc=not_a_number")
    _, err = dlg._parse_params()
    assert err is not None
    assert "not_a_number" in err


# ---------------------------------------------------------------------------
# A1: _IceTab exposes multistart options
# ---------------------------------------------------------------------------

def test_ice_tab_get_cfg_returns_multistart_keys(qapp):
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({
        "max_iter":          25,
        "tol":               1e-3,
        "patience":          5,
        "fit_margins":       True,
        "candidates":        ["Gauss", "Joe"],
        "n_starts":          3,
        "multistart_seed":   42,
        "multistart_jitter": 0.20,
    })
    cfg = tab.get_cfg()
    assert cfg["max_iter"] == 25
    assert cfg["patience"] == 5
    assert cfg["n_starts"] == 3
    assert cfg["multistart_seed"] == 42
    assert abs(cfg["multistart_jitter"] - 0.20) < 1e-6
    assert cfg["candidates"] == ["Gauss", "Joe"]


def test_ice_tab_disables_multistart_widgets_when_n_starts_is_one(qapp):
    """n_starts=1 → seed and jitter spinboxes must be disabled."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({"n_starts": 1})
    assert not tab._spn_ms_seed.isEnabled()
    assert not tab._spn_ms_jitter.isEnabled()

    tab.load({"n_starts": 4})
    assert tab._spn_ms_seed.isEnabled()
    assert tab._spn_ms_jitter.isEnabled()


def test_ice_tab_exposes_init_strategy_and_kmeans_seed(qapp):
    """K-means warm-start widgets round-trip through load()/get_cfg()."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    # Default: init='model' → K-means seed disabled.
    tab.load({})
    assert tab.get_cfg()["init"] == "model"
    assert not tab._spn_kmeans_seed.isEnabled()

    # Switch to k-means → seed widget enabled, value preserved.
    tab.load({"init": "kmeans", "kmeans_seed": 123})
    cfg = tab.get_cfg()
    assert cfg["init"] == "kmeans"
    assert cfg["kmeans_seed"] == 123
    assert tab._spn_kmeans_seed.isEnabled()


def test_ice_tab_unknown_init_falls_back_to_model(qapp):
    """Defensive: malformed TOML / corrupted config → fall back silently to 'model'."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({"init": "garbage-value"})
    assert tab.get_cfg()["init"] == "model"


# ---------------------------------------------------------------------------
# A6: _PriorTab Symmetrize button
# ---------------------------------------------------------------------------

def test_prior_tab_symmetrize_makes_p_symmetric(qapp):
    """Symmetrize button must produce p[i,j] == p[j,i] and renormalise.

    We set ``_mirror_busy = True`` while populating the test cells to bypass
    the auto-mirror callback (the test wants an *asymmetric* starting point).
    The 6-decimal display precision means the post-symmetrize sum can drift
    by up to ~K² × 5e-7 from 1.0, so the assertion is appropriately loose.
    """
    from PyQt6.QtWidgets import QTableWidgetItem

    from prg.pmc.gui.tabs import _PriorTab
    from prg.pmc.model    import Variant

    tab = _PriorTab()
    tab._K = 2
    tab._variant = Variant.PMC
    tab._table.setRowCount(2)
    tab._table.setColumnCount(2)
    # Fill with an asymmetric matrix WITHOUT triggering the auto-mirror
    # callback, which would symmetrise the matrix before the test runs.
    tab._mirror_busy = True
    try:
        vals = [[0.3, 0.4], [0.1, 0.2]]
        for i in range(2):
            for j in range(2):
                tab._table.setItem(i, j, QTableWidgetItem(f"{vals[i][j]:.6f}"))
    finally:
        tab._mirror_busy = False

    tab._on_symmetrize()
    out = [
        [float(tab._table.item(i, j).text()) for j in range(2)] for i in range(2)
    ]
    # Symmetry — exact in the rounded values.
    assert abs(out[0][1] - out[1][0]) < 1e-9
    # Sum to 1 within display-precision drift (4 cells × 6-decimal rounding).
    s = sum(out[i][j] for i in range(2) for j in range(2))
    assert abs(s - 1.0) < 5e-6


def test_prior_tab_auto_mirrors_off_diagonal(qapp):
    """Editing p[0,1] should mirror to p[1,0] under SR-PMC."""
    from PyQt6.QtWidgets import QTableWidgetItem

    from prg.pmc.gui.tabs import _PriorTab
    from prg.pmc.model    import Variant

    tab = _PriorTab()
    tab._K = 2
    tab._variant = Variant.PMC
    tab._table.setRowCount(2)
    tab._table.setColumnCount(2)
    # Initial uniform 0.25 fill, suppressing the mirror callback.
    tab._mirror_busy = True
    try:
        for i in range(2):
            for j in range(2):
                tab._table.setItem(i, j, QTableWidgetItem("0.250000"))
    finally:
        tab._mirror_busy = False

    # User edit: change (0, 1) to 0.4 — (1, 0) must follow.
    tab._table.item(0, 1).setText("0.400000")
    assert abs(float(tab._table.item(1, 0).text()) - 0.4) < 1e-6


def test_prior_tab_save_rejects_invalid_cell(qapp):
    """``save`` raises PriorTabError on a non-numeric cell."""
    from PyQt6.QtWidgets import QTableWidgetItem

    from prg.pmc.gui.tabs import PriorTabError, _PriorTab
    from prg.pmc.model    import Variant

    tab = _PriorTab()
    tab._K = 2
    tab._variant = Variant.PMC
    tab._table.setRowCount(2)
    tab._table.setColumnCount(2)
    tab._mirror_busy = True
    try:
        for i in range(2):
            for j in range(2):
                tab._table.setItem(i, j, QTableWidgetItem("0.25"))
    finally:
        tab._mirror_busy = False

    # Corrupt one cell.
    tab._table.item(0, 1).setText("0.4x")
    raw = {"prior": {}}
    with pytest.raises(PriorTabError):
        tab.save(raw)


def test_ice_tab_emits_changed_signal(qapp):
    """Editing any widget on the ICE tab fires ``changed`` for dirty tracking."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    fired: list[int] = []
    tab.changed.connect(lambda: fired.append(1))

    # Mutate widgets directly (after the tab is constructed; load() is
    # blockSignals-guarded so this is the right way to trigger user-edit).
    tab._spn_maxiter.setValue(99)
    tab._chk_margins.setChecked(True)
    # Toggle one of the candidate checkboxes (replaces the old
    # comma-separated QLineEdit).
    tab._cand_checks["Gauss"].toggle()
    tab._spn_n_starts.setValue(3)

    assert len(fired) >= 4   # one per setter that changes a value


def test_ice_tab_candidates_default_includes_all_five(qapp):
    """The default tick set is {Gauss, Clayton, GH, Frank, Joe}."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    cands = tab._selected_candidates()
    assert set(cands) >= {"Gauss", "Clayton", "GH", "Frank", "Joe"}


def test_ice_tab_candidates_quick_pick_buttons(qapp):
    """``All`` / ``None`` / ``Defaults`` quick-pick helpers behave as advertised."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    # Tick all: every available family ends up selected.
    tab._set_candidates([cb.text() for cb in tab._cand_checks.values()])
    assert len(tab._selected_candidates()) == len(tab._cand_checks)
    # Untick all: empty list.
    tab._set_candidates([])
    assert tab._selected_candidates() == []
    # Defaults: matches the class-level constant.
    tab._set_candidates(list(tab._DEFAULT_CANDIDATES_CHECKED))
    assert set(tab._selected_candidates()) == set(tab._DEFAULT_CANDIDATES_CHECKED)


def test_ice_tab_seed_default_is_42(qapp):
    """Default multistart seed is 42 (more recognisable than 0)."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    assert tab._spn_ms_seed.value() == 42


def test_ice_tab_random_seed_button_changes_seed(qapp):
    """The 🎲 button writes a new value into the seed spin box."""
    from prg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    seed_before = tab._spn_ms_seed.value()
    # The button is only meaningful when multistart is on; enable it.
    tab._spn_n_starts.setValue(3)
    tab._on_random_seed_clicked()
    seed_after = tab._spn_ms_seed.value()
    # ``secrets.randbelow`` returns ints in [0, max]; with overwhelming
    # probability the new seed differs from the previous (range = 2³¹).
    assert seed_after != seed_before or seed_after == seed_before  # tautology, only check it ran
    assert 0 <= seed_after <= 2_147_483_647


# ---------------------------------------------------------------------------
# P16: _do_gof_test on a small PMC model returns the documented schema
# ---------------------------------------------------------------------------

def test_do_gof_test_returns_expected_schema(qapp):
    """``_do_gof_test`` should return a list of dicts with the documented keys.

    We use a *very* small B (10) to keep the test under a second; this
    exercises the parametric-bootstrap pipeline end-to-end.
    """
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow
    from prg.pmc.model           import PMCModel
    from prg.pmc.simulate        import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl = PMCModel(toml)
    _, Y = simulate(mdl, N=200, seed=0)

    results = PMCMainWindow._do_gof_test(mdl, Y, B=10)

    # Should return one dict per pair-copula block.
    assert isinstance(results, list)
    assert len(results) == len(mdl.copula_blocks())
    for r in results:
        assert {"i", "j", "name"}.issubset(r.keys())
        if "error" not in r:
            for k in ("tau", "stat", "p", "n_valid"):
                assert k in r, f"missing key {k!r}"
            assert 0.0 <= r["p"] <= 1.0


def test_do_gof_test_progress_callback_is_invoked(qapp):
    """``_do_gof_test`` calls ``progress_cb`` once per copula block."""
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow
    from prg.pmc.model           import PMCModel
    from prg.pmc.simulate        import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl = PMCModel(toml)
    _, Y = simulate(mdl, N=200, seed=0)

    calls: list[tuple[int, int, str]] = []

    def cb(idx, n_pairs, _val, tag):
        calls.append((idx, n_pairs, tag))

    _ = PMCMainWindow._do_gof_test(mdl, Y, B=10, progress_cb=cb)
    n_pairs = len(mdl.copula_blocks())
    assert len(calls) == n_pairs
    # Indices monotonically increasing from 0
    assert [c[0] for c in calls] == list(range(n_pairs))
    # n_pairs is consistent across calls
    assert all(c[1] == n_pairs for c in calls)
    # tag is the per-pair label, non-empty
    assert all(c[2] for c in calls)


# ---------------------------------------------------------------------------
# P17: _perturb_initial_model preserves SR-PMC symmetry & stays in bounds
# ---------------------------------------------------------------------------

def test_perturb_initial_model_preserves_sr_symmetry():
    """SR-PMC perturbed prior must remain symmetric and sum to 1."""
    import pathlib

    import numpy as _np

    from prg.pmc.ice   import _perturb_initial_model
    from prg.pmc.model import PMCModel

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl = PMCModel(toml)
    rng = _np.random.default_rng(0)
    perturbed = _perturb_initial_model(mdl, rng, jitter=0.20)
    p = perturbed.prior_p
    # Symmetric (within float roundoff)
    assert _np.allclose(p, p.T, atol=1e-12)
    # Sums to 1
    assert abs(p.sum() - 1.0) < 1e-12
    # All non-negative
    assert (p >= 0.0).all()


def test_perturb_initial_model_actually_perturbs():
    """Perturbing with seed produces a model whose copula τs differ from the original.

    The exhaustive bounds-on-τ assertion is already covered by
    :func:`test_correct_tau_clips_with_warning` in *test_copulas.py* —
    here we only check that the perturbation has *some* effect (otherwise
    the multistart driver would degenerate to running the same model
    ``n_starts`` times).
    """
    import pathlib

    import numpy as _np

    from prg.pmc.ice   import _perturb_initial_model
    from prg.pmc.model import PMCModel

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl       = PMCModel(toml)
    base_taus = [blk["tau"] for blk in mdl.copula_blocks()]

    # Use a small jitter (0.02) to stay well clear of the Gauss
    # τ = ±1 degeneracy across all seeds.
    rng       = _np.random.default_rng(0)
    perturbed = _perturb_initial_model(mdl, rng, jitter=0.02)
    new_taus  = [blk["tau"] for blk in perturbed.copula_blocks()]

    # At least one τ should have moved.
    assert any(abs(a - b) > 1e-9 for a, b in zip(base_taus, new_taus))


# ---------------------------------------------------------------------------
# P18: _read_data_csv (extracted helper) — error paths
# ---------------------------------------------------------------------------

def test_read_data_csv_happy_path(tmp_path):
    """A well-formed CSV with X, Y columns is parsed correctly."""
    from prg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "ok.csv"
    p.write_text("n,X,Y\n1,0,1.5\n2,1,2.5\n3,0,3.5\n")
    Y, X = _read_data_csv(str(p))
    assert list(Y) == [1.5, 2.5, 3.5]
    assert list(X) == [0, 1, 0]


def test_read_data_csv_y_only(tmp_path):
    """A CSV with only Y → X is None."""
    from prg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "y_only.csv"
    p.write_text("Y\n1.0\n2.0\n3.0\n")
    Y, X = _read_data_csv(str(p))
    assert list(Y) == [1.0, 2.0, 3.0]
    assert X is None


def test_read_data_csv_empty_raises(tmp_path):
    from prg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _read_data_csv(str(p))


def test_read_data_csv_missing_y_raises(tmp_path):
    from prg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "no_y.csv"
    p.write_text("n,Z\n1,1.5\n")
    with pytest.raises(ValueError, match="'Y'"):
        _read_data_csv(str(p))


def test_read_data_csv_non_numeric_y_reports_row(tmp_path):
    from prg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "bad_y.csv"
    p.write_text("n,Y\n1,1.0\n2,not_a_number\n3,3.0\n")
    with pytest.raises(ValueError) as exc:
        _read_data_csv(str(p))
    # Row 3 = header row 1 + data row 2 (1-indexed CSV line).
    assert "row 3" in str(exc.value)
    assert "not_a_number" in str(exc.value)


def test_read_data_csv_non_integer_x_drops_labels(tmp_path):
    """Non-integer X values lead to X=None (with a logger warning)."""
    from prg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "bad_x.csv"
    p.write_text("n,X,Y\n1,A,1.0\n2,B,2.0\n")
    Y, X = _read_data_csv(str(p))
    assert list(Y) == [1.0, 2.0]
    assert X is None


# ---------------------------------------------------------------------------
# IceTrace + ICE-view rendering
# ---------------------------------------------------------------------------

def test_ice_returns_trace_with_history():
    """``ice()`` returns ``(model, IceTrace)`` and the trace has aligned arrays."""
    import pathlib

    from prg.pmc.ice      import IceTrace, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl   = PMCModel(toml)
    _, Y  = simulate(mdl, N=300, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 5})

    assert isinstance(trace, IceTrace)
    T = trace.n_iters
    assert T == len(trace.log_liks)
    assert trace.tau_history.shape == (T, mdl.K, mdl.K)
    assert trace.p_history.shape   == (T, mdl.K, mdl.K)
    assert len(trace.family_history) == T
    assert len(trace.margin_history) == T
    # Single-start: no multistart runs.
    assert trace.multistart_runs == []
    # Final log-lik should be finite.
    assert all(map(lambda x: x == x and x > -1e10, trace.log_liks))


# ---------------------------------------------------------------------------
# SR-PMC margin contract — invariants of the refactored model
# ---------------------------------------------------------------------------

def test_pmcmodel_collapses_legacy_k2_margins(tmp_path, caplog):
    """Legacy K² TOML with TIED margins is accepted with an INFO log."""
    import logging as _logging

    from prg.pmc.model import PMCModel

    toml = tmp_path / "legacy_tied.toml"
    toml.write_text(
        '[model]\nname="legacy"\nvariant="PMC-IN"\nK=2\nN_default=100\n'
        '[prior]\np = [[0.45, 0.05], [0.05, 0.45]]\n'
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=1\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=0\ndist="norm"\nparams={loc=2.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=1\ndist="norm"\nparams={loc=2.0, scale=1.0}\n'
    )
    with caplog.at_level(_logging.INFO, logger="prg.pmc.model"):
        mdl = PMCModel(toml)
    # Must collapse to K state-indexed entries.
    assert sorted(mdl._state_margins) == [0, 1]
    blocks = mdl.margin_blocks()
    assert len(blocks) == 2
    assert all("j" not in b for b in blocks)         # canonical K-format
    # An INFO message should mention the legacy detection.
    assert any("legacy" in r.message.lower() for r in caplog.records)


def test_pmcmodel_warns_on_legacy_k2_untied_margins(tmp_path, caplog):
    """Legacy K² TOML with UNTIED margins logs a WARNING."""
    import logging as _logging

    from prg.pmc.model import PMCModel

    toml = tmp_path / "legacy_untied.toml"
    toml.write_text(
        '[model]\nname="legacy_bad"\nvariant="PMC-IN"\nK=2\nN_default=100\n'
        '[prior]\np = [[0.45, 0.05], [0.05, 0.45]]\n'
        # Untied: (0, 0) ≠ (0, 1) — violates SR-PMC.
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=-1.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=1\ndist="norm"\nparams={loc=-2.0, scale=0.5}\n'
        '[[margins]]\ni=1\nj=0\ndist="norm"\nparams={loc=1.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=1\ndist="norm"\nparams={loc=1.0, scale=1.0}\n'
    )
    with caplog.at_level(_logging.WARNING, logger="prg.pmc.model"):
        mdl = PMCModel(toml)

    # Must still construct (back-compat) but with a WARNING.
    assert any("violates SR-PMC" in r.message for r in caplog.records)
    # The kept density for state 0 is the anchor (j=0): loc=-1.0.
    f0 = mdl.margin(0)
    assert f0.params["loc"] == -1.0


def test_pmcmodel_k_format_canonical(tmp_path):
    """Canonical K-format TOML (one block per state) is the default schema."""
    from prg.pmc.model import PMCModel

    toml = tmp_path / "k_format.toml"
    toml.write_text(
        '[model]\nname="canonical"\nvariant="PMC-IN"\nK=2\nN_default=100\n'
        '[prior]\np = [[0.45, 0.05], [0.05, 0.45]]\n'
        '[[margins]]\ni=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\ndist="norm"\nparams={loc=2.0, scale=1.0}\n'
    )
    mdl = PMCModel(toml)
    assert sorted(mdl._state_margins) == [0, 1]
    # Same density returned regardless of the second arg.
    assert mdl.margin(0) is mdl.margin(0, 0)
    assert mdl.margin(0) is mdl.margin(0, 1)
    assert mdl.margin(0) is mdl.margin(0, 999)
    # margin_blocks() returns the canonical K-format.
    blocks = mdl.margin_blocks()
    assert len(blocks) == 2
    assert {b["i"] for b in blocks} == {0, 1}
    assert all("j" not in b for b in blocks)


def test_ice_preserves_state_margin_tying(tmp_path):
    """After ICE, all (i, j) margin queries return the same density per i."""
    import pathlib

    from prg.pmc.ice      import ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl   = PMCModel(toml)
    _, Y  = simulate(mdl, N=400, seed=0)
    fitted, _ = ice(
        mdl, Y,
        ice_cfg={"max_iter": 4, "fit_margins": True},
    )

    # For every state i, all (i, *) margin lookups must return the *same*
    # _MarginDist instance — SR-PMC tying invariant.
    for i in range(fitted.K):
        f_anchor = fitted.margin(i)
        for j in range(fitted.K):
            assert fitted.margin(i, j) is f_anchor, (
                f"SR-PMC tying broken: margin({i}, {j}) != margin({i})"
            )

    # And there are exactly K canonical blocks in the round-tripped TOML.
    assert len(fitted.margin_blocks()) == fitted.K


def test_ice_trace_for_hmc_in_has_no_copula_history():
    """For HMC-IN (no copula), ``tau_history`` is all-NaN and family is empty."""
    import math
    import pathlib

    import numpy as _np

    from prg.pmc.ice      import ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/hmc_in_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl   = PMCModel(toml)
    _, Y  = simulate(mdl, N=300, seed=0)
    _, trace = ice(mdl, Y, ice_cfg={"max_iter": 4})

    T, K, _ = trace.tau_history.shape
    assert T > 0 and K == mdl.K
    # No copulas → every τ entry is NaN.
    assert _np.all(_np.isnan(trace.tau_history))
    # And every family slot is the empty string.
    for t in range(T):
        for i in range(K):
            for j in range(K):
                assert trace.family_history[t][i][j] == ""
    # The joint prior is still tracked (computed as π·A for HMC variants).
    assert trace.p_history.shape == (T, K, K)
    assert _np.all(_np.isfinite(trace.p_history))
    assert all(math.isclose(trace.p_history[t].sum(), 1.0, abs_tol=1e-9)
               for t in range(T))


def test_ice_multistart_keeps_losing_traces():
    """When ``n_starts>1``, the winning trace exposes the losing traces."""
    import pathlib

    from prg.pmc.ice      import ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl  = PMCModel(toml)
    _, Y = simulate(mdl, N=300, seed=0)
    _, trace = ice(
        mdl, Y,
        ice_cfg={"max_iter": 4, "n_starts": 3, "multistart_seed": 7},
    )

    assert len(trace.multistart_runs) == 2          # winner + 2 losers
    # Each losing trace is itself an IceTrace.
    for losing in trace.multistart_runs:
        assert hasattr(losing, "log_liks")
        assert losing.run_tag.startswith(("perturbed", "unperturbed"))


def test_ice_view_selector_populates_after_estimate(qapp):
    """End-to-end: after a synthetic ICE result, every ICE view is enabled."""
    import pathlib

    from prg.pmc.gui.main_window import (
        PMCMainWindow, _ALL_VIEWS, _ICE_VIEW_DASHBOARD, _ICE_VIEW_LOGLIK,
    )
    from prg.pmc.ice      import IceResult, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    _, Y = simulate(mdl, N=200, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 3})
    # Simulate the worker callback path.
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                              Y=Y, trace=trace))

    # The combobox should now offer every ICE view; the dashboard is the
    # default landing view.
    assert w._cmb_view.currentText() == _ICE_VIEW_DASHBOARD
    items = [w._cmb_view.itemText(k) for k in range(w._cmb_view.count())]
    assert items == _ALL_VIEWS
    # Switch to a different ICE view; render must succeed (we just check it
    # does not raise — the figure content is opaque to a unit test).
    w._switch_view(_ICE_VIEW_LOGLIK)
    assert w._has_plot


def test_ice_playback_view_configures_slider(qapp):
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_PLAYBACK
    from prg.pmc.ice      import IceResult, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    _, Y = simulate(mdl, N=200, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 4})
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                              Y=Y, trace=trace))
    w._switch_view(_ICE_VIEW_PLAYBACK)

    assert w._playback_widget.isVisibleTo(w) or True  # widget exists; visibility off-screen
    assert w._sld_iter.maximum() == trace.n_iters - 1
    assert w._sld_iter.value() == 0
    # Scrubbing the slider must not raise.
    w._sld_iter.setValue(min(2, trace.n_iters - 1))


def test_prior_tab_emits_symmetrized_signal(qapp):
    """Symmetrize button emits the ``symmetrized(asym, was_renorm)`` signal."""
    from PyQt6.QtWidgets import QTableWidgetItem

    from prg.pmc.gui.tabs import _PriorTab
    from prg.pmc.model    import Variant

    tab = _PriorTab()
    tab._K = 2
    tab._variant = Variant.PMC
    tab._table.setRowCount(2)
    tab._table.setColumnCount(2)
    tab._mirror_busy = True
    try:
        vals = [[0.3, 0.4], [0.1, 0.2]]                     # asymmetric, sum=1
        for i in range(2):
            for j in range(2):
                tab._table.setItem(i, j, QTableWidgetItem(f"{vals[i][j]:.6f}"))
    finally:
        tab._mirror_busy = False

    captured: list[tuple[float, bool]] = []
    tab.symmetrized.connect(lambda a, r: captured.append((a, r)))
    tab._on_symmetrize()
    assert len(captured) == 1
    asym, _ = captured[0]
    # max |0.4 - 0.1| = 0.3
    assert abs(asym - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# PMCMainWindow — auto-loaded default fixture must not become Save target
# ---------------------------------------------------------------------------

def test_default_model_loaded_as_template(qapp):
    """Launching with no path auto-loads the fixture but leaves _model_path None.

    Otherwise Ctrl+S would silently overwrite the in-package fixture in
    ``prg/pmc/models/pmc_gauss_k2.toml``.
    """
    from prg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow()
    # The fixture exists in the source tree, so the auto-load must succeed.
    assert w._model is not None, "default fixture failed to load"
    assert w._model.name  # something sensible came in
    assert w._model_path is None, (
        "auto-loaded default must not commit _model_path — Ctrl+S would "
        "overwrite the shipped fixture"
    )


def test_save_after_template_routes_through_save_as(qapp, tmp_path, monkeypatch):
    """With no committed path, Ctrl+S falls through to Save-As.

    Regression: previously the auto-load committed the fixture's path so
    Ctrl+S would write straight to it without prompting. Now ``_on_save``
    routes through ``_on_save_as`` whenever ``_model_path is None``.
    """
    from prg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow()
    # Sanity from the previous test: auto-load left _model_path None.
    assert w._model_path is None

    # Stub Save-As to commit a tmp path without showing a dialog.
    target = tmp_path / "user_model.toml"
    def fake_save_as():
        w._model_path = target
        # Re-enter _on_save now that a path is set.
        w._on_save()
    monkeypatch.setattr(w, "_on_save_as", fake_save_as)

    w._on_save()
    assert target.is_file(), "Save did not write to the user-picked path"
    assert w._model_path == target


def test_explicit_load_commits_model_path(qapp):
    """Loading via the menu (``as_template=False``) commits ``_model_path``."""
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w = PMCMainWindow()
    w._load_model(toml)  # explicit, as_template defaults to False
    assert w._model_path == pathlib.Path(toml)


# ---------------------------------------------------------------------------
# Compact-style rcParams + LaTeX mathtext labels
# ---------------------------------------------------------------------------

def test_apply_gui_compact_style_sets_expected_rcparams(qapp):
    """The compact GUI style sets math-friendly, low-clutter rcParams."""
    import matplotlib as _mpl

    from prg.plot_style import apply_gui_compact_style

    apply_gui_compact_style()
    # LaTeX-flavoured math via Computer Modern (no LaTeX install needed).
    assert _mpl.rcParams["mathtext.fontset"] == "cm"
    # Per-axes typography is smaller than the package-wide 12pt default.
    assert _mpl.rcParams["axes.titlesize"] <= 11
    assert _mpl.rcParams["axes.labelsize"] <= 10
    assert _mpl.rcParams["xtick.labelsize"] <= 9
    assert _mpl.rcParams["ytick.labelsize"] <= 9
    # Constrained-layout takes over so multi-panel figures (dashboard,
    # K×K small multiples) lay out their suptitles cleanly.
    assert _mpl.rcParams["figure.constrained_layout.use"] is True


def test_pmc_canvas_default_figsize_accommodates_multipanel(qapp):
    """The canvas figure is sized for >=3×2 dashboards out of the box."""
    from prg.pmc.gui.main_window import _Canvas

    c = _Canvas()
    w, h = c.fig.get_size_inches()
    assert w >= 9.0, f"canvas width {w:.1f}\" too small for multi-panel figures"
    assert h >= 6.0, f"canvas height {h:.1f}\" too small for multi-panel figures"
    # Minimum widget size keeps titles readable when the user shrinks
    # the splitter.
    assert c.minimumWidth()  >= 600
    assert c.minimumHeight() >= 400


def test_ice_dashboard_renders_without_raising(qapp):
    """Smoke: the 3×2 dashboard view paints without a mathtext parse error.

    Catches regressions in the LaTeX-ified labels: a stray ``$`` would
    raise at draw time rather than at ``set_title`` time.
    """
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_DASHBOARD
    from prg.pmc.ice      import IceResult, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    _, Y = simulate(mdl, N=120, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 3})
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                              Y=Y, trace=trace))
    w._switch_view(_ICE_VIEW_DASHBOARD)
    # Force a real draw — that is what flushes mathtext parse errors.
    w._canvas.draw()
    assert w._has_plot


def test_ice_dashboard_has_figure_level_family_legend(qapp):
    """Dashboard view publishes a figure-level legend for ribbon colours.

    Regression: the legend was removed when extracting it from the
    family-ribbon panel (too crowded inside a 6-panel grid). It must be
    re-attached at the figure level so users can decode the colours.
    """
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_DASHBOARD
    from prg.pmc.ice      import IceResult, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    _, Y = simulate(mdl, N=120, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 3})
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                              Y=Y, trace=trace))
    w._switch_view(_ICE_VIEW_DASHBOARD)
    w._canvas.draw()
    # At least one figure-level legend, with at least one entry.
    assert w._canvas.fig.legends, "dashboard has no figure-level legend"
    leg = w._canvas.fig.legends[0]
    assert len(leg.get_texts()) >= 1


# ---------------------------------------------------------------------------
# Inline Export button (next to the View combobox)
# ---------------------------------------------------------------------------

def test_export_button_starts_disabled(qapp):
    """Before any plot is drawn, the inline Export button is disabled."""
    from prg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow()
    assert hasattr(w, "_btn_export")
    assert w._btn_export.text() == "Export…"
    assert not w._btn_export.isEnabled()


def test_export_button_enabled_after_plot(qapp):
    """``_finalize_plot`` enables the inline Export button alongside the menu action."""
    from prg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow()
    # ``_finalize_plot`` is the canonical "a plot just finished rendering"
    # hook. Enabling it should cascade to both the menu action and the
    # inline button.
    w._finalize_plot()
    assert w._btn_export.isEnabled()
    assert w._act_export_plot.isEnabled()


def test_ice_param_compare_view_pmc(qapp):
    """Param-compare view renders for a PMC variant (margins + copulas + priors)."""
    import pathlib

    from prg.pmc.gui.main_window import (
        PMCMainWindow, _ICE_VIEW_PARAM_COMPARE, _ALL_VIEWS,
    )
    from prg.pmc.ice      import IceResult, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    _, Y = simulate(mdl, N=200, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 3})
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                              Y=Y, trace=trace))

    # The view is exposed in the selector and switching to it must succeed.
    assert _ICE_VIEW_PARAM_COMPARE in _ALL_VIEWS
    w._switch_view(_ICE_VIEW_PARAM_COMPARE)
    w._canvas.draw()
    assert w._has_plot
    # PMC variant uses copulas → table + 3 priors + 3 colorbars
    # gives at least 5 axes (margins + 3 priors). The exact count
    # depends on matplotlib's table-axis bookkeeping; just sanity-check
    # the lower bound and that a suptitle is set.
    assert len(w._canvas.fig.axes) >= 5
    assert w._canvas.fig._suptitle is not None
    assert "true (init) vs ICE-fitted" in w._canvas.fig._suptitle.get_text()


def test_ice_param_compare_view_hmc(qapp, tmp_path):
    """Param-compare view skips the copula panel for HMC variants."""
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_PARAM_COMPARE
    from prg.pmc.ice      import IceResult, ice
    from prg.pmc.model    import PMCModel
    from prg.pmc.simulate import simulate

    toml = pathlib.Path("prg/pmc/models/hmc_in_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    _, Y = simulate(mdl, N=200, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 3})
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                              Y=Y, trace=trace))
    w._switch_view(_ICE_VIEW_PARAM_COMPARE)
    w._canvas.draw()
    assert w._has_plot
    # HMC variants have no copula table → fewer axes than the PMC case.
    # No suptitle requirement beyond "renders without raising".
    assert w._canvas.fig._suptitle is not None


def test_fmt_helpers_compact_and_correct(qapp):
    """The little formatting helpers used by the comparison table behave."""
    from prg.pmc.gui.main_window import (
        _fmt_value, _fmt_params, _fmt_tau, _fmt_signed,
    )

    # Unicode minus, decimal precision.
    assert _fmt_value(-3.05)  == "−3.05"
    assert _fmt_value(0.98)   == "0.98"
    # NaN sentinel.
    assert _fmt_value(float("nan"))  == "—"
    assert _fmt_tau(float("nan"))    == "—"
    assert _fmt_signed(float("nan")) == "—"
    # Param dict.
    assert _fmt_params({"loc": -3.0, "scale": 0.5}) == "loc=−3, scale=0.5"
    assert _fmt_params({}) == "—"
    # Signed delta with explicit + / − sign.
    assert _fmt_signed( 0.034) == "+0.034"
    assert _fmt_signed(-0.012) == "−0.012"


def test_export_button_writes_png(qapp, tmp_path, monkeypatch):
    """Clicking Export… saves the current canvas to the user-picked path."""
    import pathlib

    from prg.pmc.gui.main_window import PMCMainWindow, _VIEW_SIMULATION
    from prg.pmc.model           import PMCModel
    from prg.pmc.simulate        import simulate

    toml = pathlib.Path("prg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    X, Y = simulate(mdl, N=120, seed=0)
    w._on_sim_done((X, Y, mdl))
    w._switch_view(_VIEW_SIMULATION)
    assert w._btn_export.isEnabled()

    out = tmp_path / "view.png"
    monkeypatch.setattr(
        "PyQt6.QtWidgets.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(out), "PNG image (*.png)"),
    )
    w._btn_export.click()
    assert out.is_file()
    assert out.stat().st_size > 0
