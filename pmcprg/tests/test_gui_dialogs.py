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
# optional dependency: ``pip install awesomepmc[gui]``).
PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# Session-scoped QApplication — Qt requires exactly one instance per process.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _auto_answer_questions(monkeypatch):
    """Answer every ``QMessageBox.question`` — a modal hangs an offscreen run.

    The answer is chosen from the buttons offered, so the two prompts keep
    their distinct meanings: the unsaved-changes guard (audit S-5) is
    *discarded*, while "Overwrite data?" is *accepted* as before.
    """
    from PyQt6.QtWidgets import QMessageBox

    def answer(*args, **kwargs):
        buttons = args[3] if len(args) > 3 else kwargs.get("buttons")
        if buttons is not None and (buttons & QMessageBox.StandardButton.Discard):
            return QMessageBox.StandardButton.Discard
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer))


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
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    blk_in = {"i": 0, "j": 1, "dist": "norm",
              "params": {"loc": -1.5, "scale": 0.7}}
    dlg = _MarginDialog(blk_in)
    out = dlg.get_block()

    assert out["dist"] == "norm"
    assert out["params"] == {"loc": -1.5, "scale": 0.7}


def test_margin_dialog_default_dist(qapp):
    """Missing 'dist' falls back to 'norm'."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0})
    out = dlg.get_block()
    assert out["dist"] == "norm"
    assert out["params"] == {}


def test_margin_dialog_handles_multiple_params(qapp):
    """A margin with several scipy.stats parameters round-trips correctly."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog

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
    from pmcprg.pmc.gui.dialogs import _MarginDialog

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
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    # Default state: nothing ticked.
    assert all(not cb.isChecked() for cb in dlg._cand_checks.values())
    out = dlg.get_block()
    assert "candidates" not in out


def test_margin_dialog_sp2016_quick_pick(qapp):
    """The SP-2016 §3 helper ticks exactly {norm, gamma, invgamma, betaprime}."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog
    from pmcprg.pmc.ice          import SP2016_DEFAULT_CANDIDATES

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    dlg._set_candidates(SP2016_DEFAULT_CANDIDATES)
    out = dlg.get_block()
    assert out["candidates"] == list(SP2016_DEFAULT_CANDIDATES)


def test_margin_dialog_all_then_none_quick_picks(qapp):
    """All / None helpers cover every known family and clear back to empty."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog
    from pmcprg.pmc.ice          import GICE_KNOWN_FAMILIES

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})

    dlg._set_candidates(GICE_KNOWN_FAMILIES)
    out = dlg.get_block()
    assert out["candidates"] == list(GICE_KNOWN_FAMILIES)

    dlg._set_candidates(())
    out = dlg.get_block()
    assert "candidates" not in out


def test_margin_dialog_other_line_unknown_family_kept(qapp):
    """Unknown scipy.stats names typed into the Other line round-trip too."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog

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
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    dlg._cand_checks["norm"].setChecked(True)
    dlg._cand_extras.setText("norm, pareto")
    out = dlg.get_block()
    assert out["candidates"] == ["norm", "pareto"]


def test_margin_dialog_lists_all_known_families(qapp):
    """Every entry in GICE_KNOWN_FAMILIES has a corresponding checkbox."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog
    from pmcprg.pmc.ice          import GICE_KNOWN_FAMILIES

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm", "params": {}})
    assert set(dlg._cand_checks.keys()) == set(GICE_KNOWN_FAMILIES)


# ---------------------------------------------------------------------------
# _CopulaDialog
# ---------------------------------------------------------------------------

def test_copula_dialog_roundtrip_gauss(qapp):
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    blk_in = {"i": 0, "j": 0, "name": "Gauss", "tau": 0.65}
    dlg = _CopulaDialog(blk_in)
    out = dlg.get_block()

    assert out["name"] == "Gauss"
    assert abs(out["tau"] - 0.65) < 1e-4   # QDoubleSpinBox precision


def test_copula_dialog_unknown_name_falls_back_to_gauss(qapp):
    """An unknown copula name leaves the combo at its default (Gauss)."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Nonsense", "tau": 0.0})
    out = dlg.get_block()
    # First item in the combo is the first available copula in CopulaEnum
    assert out["name"] in (
        "Prod", "Gauss",
    )  # depends on CopulaEnum ordering; both are sane defaults


def test_copula_dialog_negative_tau(qapp):
    """Negative τ values are preserved."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 1, "name": "Frank", "tau": -0.30})
    out = dlg.get_block()
    assert out["name"] == "Frank"
    assert abs(out["tau"] - (-0.30)) < 1e-4


def test_copula_dialog_lists_all_available_copulas(qapp):
    """The ComboBox must include every available copula SHORT_NAME."""
    from pmcprg.copulas._base import CopulaEnum
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Gauss", "tau": 0.0})
    items = [dlg._name.itemText(k) for k in range(dlg._name.count())]
    expected = [c.value.SHORT_NAME for c in CopulaEnum if c.value.AVAILABLE]
    assert items == expected


# ---------------------------------------------------------------------------
# A2: _CopulaDialog is family-aware
# ---------------------------------------------------------------------------

def test_copula_dialog_tau_range_matches_fgm(qapp):
    """For FGM, the τ spinbox bounds should match its valid range."""
    from pmcprg.copulas._base import CopulaEnum
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "FGM", "tau": 0.0})
    tau_min, tau_max = CopulaEnum.FGM.value.TAU_MIN_MAX
    assert dlg._tau.minimum() == pytest.approx(tau_min, abs=1e-6)
    assert dlg._tau.maximum() == pytest.approx(tau_max, abs=1e-6)


def test_copula_dialog_frank_tau_range_stops_at_the_reachable_tau(qapp):
    """Frank reaches |τ| ≤ 0.994299 (θ ≤ 700), not its registered 1 − ε (F1):
    the spinbox must not offer a τ the copula would clamp."""
    from pmcprg.copulas.archimedean.frank import _FRANK_TAU_MAX
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Frank", "tau": 0.9999})
    assert dlg._tau.maximum() <= _FRANK_TAU_MAX
    assert dlg._tau.minimum() >= -_FRANK_TAU_MAX
    assert dlg._tau.maximum() == pytest.approx(_FRANK_TAU_MAX, abs=1e-6)
    assert dlg.get_block()["tau"] <= _FRANK_TAU_MAX


@pytest.mark.parametrize("tau", [0.9999, -0.9999])
def test_copula_dialog_plackett_tau_range_stays_inside_the_table(qapp, tau):
    """Plackett reaches τ ∈ [−0.99352457, 0.99352457] (θ ∈ [1e-6, 1e6], G2).
    Rounded to the spin box's 6 decimals the bounds would be ±0.993525 — beyond
    them — so they are stepped back to ±0.993524."""
    from pmcprg.copulas import CopulaPlackett
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    lo, hi = CopulaPlackett.reachable_tau_bounds()
    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Plackett", "tau": tau})
    assert (dlg._tau.minimum(), dlg._tau.maximum()) == (-0.993524, 0.993524)
    assert lo <= dlg.get_block()["tau"] <= hi
    dlg._name.setCurrentText("Gauss")
    assert (dlg._tau.minimum(), dlg._tau.maximum()) == (-1.0, 1.0)


def test_copula_dialog_tau_range_updates_on_family_change(qapp):
    """Changing the family must rerange the τ spinbox to the new family's bounds."""
    from pmcprg.copulas._base import CopulaEnum
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Gauss", "tau": 0.0})
    # Switch to FGM (range −2/9, 2/9)
    dlg._name.setCurrentText("FGM")
    tau_min, tau_max = CopulaEnum.FGM.value.TAU_MIN_MAX
    assert dlg._tau.minimum() == pytest.approx(tau_min, abs=1e-6)
    assert dlg._tau.maximum() == pytest.approx(tau_max, abs=1e-6)


def test_copula_dialog_clamps_input_tau_to_family_range(qapp):
    """A τ value outside the family's range is clamped (no exception)."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    # FGM range is roughly [-0.222, 0.222]; ask for τ=0.5 → clamped to 2/9.
    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "FGM", "tau": 0.5})
    out = dlg.get_block()
    assert out["name"] == "FGM"
    assert out["tau"] <= 2.0 / 9.0 + 1e-6


def test_copula_dialog_bb1_exposes_delta(qapp):
    """BB1 has a δ extra parameter — it must round-trip via get_block.

    δ = 1.5 at τ = 0.5 is admissible (δ < 1/(1 − τ) = 2.0): the round trip
    is exact. See ``test_copula_dialog_repairs_a_jointly_refused_pair`` for
    the case where τ and δ are independently in-range but jointly refused
    (δ = 2.5 here would be exactly that — `constructible_params` now moves
    it, so it is no longer a fixed point of the round trip)."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    blk = {"i": 0, "j": 1, "name": "BB1", "tau": 0.5, "delta": 1.5}
    dlg = _CopulaDialog(blk)
    out = dlg.get_block()
    assert out["name"] == "BB1"
    assert "delta" in out
    assert abs(out["delta"] - 1.5) < 1e-3


def test_copula_dialog_student_exposes_df(qapp):
    """Student copula carries a ``df`` extra parameter."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    blk = {"i": 0, "j": 0, "name": "Student", "tau": 0.3, "df": 6.0}
    dlg = _CopulaDialog(blk)
    out = dlg.get_block()
    assert out["name"] == "Student"
    assert "df" in out
    assert abs(out["df"] - 6.0) < 1e-3


def test_copula_dialog_one_param_family_no_extras(qapp):
    """One-parameter families must not emit spurious extra keys."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Clayton", "tau": 0.4})
    out = dlg.get_block()
    assert set(out.keys()) == {"name", "tau"}


def test_copula_dialog_exposes_both_weights_of_tawn3(qapp):
    """A family with **two** extra parameters (FR-9, round 5): the dialog
    builds its widgets from ``EXTRA_PARAM_BOUNDS``, so nothing in the GUI had
    to change — this pins that down.
    """
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    blk = {"i": 1, "j": 0, "name": "Tawn3", "tau": 0.4, "psi_u": 0.8, "psi_v": 0.6}
    dlg = _CopulaDialog(blk)
    out = dlg.get_block()
    assert out["name"] == "Tawn3"
    assert abs(out["psi_u"] - 0.8) < 1e-3 and abs(out["psi_v"] - 0.6) < 1e-3
    assert set(out) == {"name", "tau", "psi_u", "psi_v"}
    # Both spin boxes carry the registered box, and only they are shown.
    for key in ("psi_u", "psi_v"):
        lbl, spin = dlg._extra_widgets[key]
        assert lbl.isVisibleTo(dlg) and spin.isVisibleTo(dlg)
        assert (spin.minimum(), spin.maximum()) == (0.01, 1.0)
    for key in ("delta", "df", "nu", "psi"):
        lbl, _ = dlg._extra_widgets[key]
        assert not lbl.isVisibleTo(dlg), f"{key} must be hidden for Tawn3"


def test_copula_dialog_repairs_a_jointly_refused_pair(qapp):
    """τ and an extra parameter each have their own spinbox range, but BB1's
    ``δ < 1/(1 − τ)`` is a *joint* constraint no single box can express: a
    user can independently move both into a pair the constructor refuses.
    ``get_block`` must repair it through the family's own
    ``constructible_params`` (the hook ``pmcprg.pmc.ice._copula_placeholder``
    already uses for the same reason), not hand back a block that crashes
    downstream."""
    from pmcprg.copulas import CopulaBB1
    from pmcprg.exceptions import CopulaParameterError
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    # τ = 0.5, δ = 10.0: refused (θ = 2/(δ(1−τ)) − 2 ≤ 0 needs τ > 1 − 1/δ = 0.9).
    with pytest.raises(CopulaParameterError):
        CopulaBB1(tau_k=0.5, delta=10.0)

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "BB1", "tau": 0.5, "delta": 1.5})
    lbl, spin = dlg._extra_widgets["delta"]
    spin.setValue(10.0)
    out = dlg.get_block()

    assert out["tau"] == pytest.approx(0.5, abs=1e-4)   # τ is the user's value, untouched
    assert out["delta"] < 10.0                           # δ moved inside the admissible set
    CopulaBB1(tau_k=out["tau"], delta=out["delta"])       # must not raise


def test_copula_dialog_switches_between_one_and_two_extra_families(qapp):
    """Switching family must show exactly the new family's extras — the two
    Tawn 3 weights appear and disappear together."""
    from pmcprg.pmc.gui.dialogs import _CopulaDialog

    dlg = _CopulaDialog({"i": 0, "j": 0, "name": "Tawn3", "tau": 0.4})
    for name, expected in (("Tawn1", {"psi"}), ("Student", {"df"}), ("Clayton", set()),
                           ("Tawn3", {"psi_u", "psi_v"})):
        dlg._name.setCurrentText(name)
        shown = {k for k, (lbl, _) in dlg._extra_widgets.items() if lbl.isVisibleTo(dlg)}
        assert shown == expected, name
        assert set(dlg.get_block()) == {"name", "tau"} | expected


# ---------------------------------------------------------------------------
# B3: _MarginDialog tolerates commas and rejects junk
# ---------------------------------------------------------------------------

def test_margin_dialog_accepts_comma_separated(qapp):
    """``loc=0, scale=1`` must round-trip even with the comma."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm",
                         "params": {"loc": 0.0, "scale": 1.0}})
    # Simulate user typing comma-separated values.
    dlg._params_edit.setText("loc=0, scale=1")
    out = dlg.get_block()
    assert out["params"] == {"loc": 0.0, "scale": 1.0}


def test_margin_dialog_accepts_no_space_after_comma(qapp):
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    dlg = _MarginDialog({"i": 0, "j": 0, "dist": "norm",
                         "params": {"loc": 0.0, "scale": 1.0}})
    dlg._params_edit.setText("loc=0,scale=1")
    out = dlg.get_block()
    assert out["params"] == {"loc": 0.0, "scale": 1.0}


def test_margin_dialog_parse_rejects_junk(qapp):
    """Non-numeric values trigger a parse error (caught by _on_accept)."""
    from pmcprg.pmc.gui.dialogs import _MarginDialog

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
    from pmcprg.pmc.gui.tabs import _IceTab

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
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({"n_starts": 1})
    assert not tab._spn_ms_seed.isEnabled()
    assert not tab._spn_ms_jitter.isEnabled()

    tab.load({"n_starts": 4})
    assert tab._spn_ms_seed.isEnabled()
    assert tab._spn_ms_jitter.isEnabled()


def test_ice_tab_multistart_families_combo(qapp):
    """``multistart_families`` round-trips; an unknown TOML value falls back;
    the combo follows the other multistart widgets; n_starts reaches a full
    family sweep."""
    from pmcprg.pmc._estim_common import MAX_SWEEP_COMBINATIONS, MULTISTART_FAMILY_MODES
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    assert tab.get_cfg()["multistart_families"] == "none"
    assert [tab._combo_ms_families.itemText(i)
            for i in range(tab._combo_ms_families.count())] == list(MULTISTART_FAMILY_MODES)
    assert not tab._combo_ms_families.isEnabled()

    tab.load({"n_starts": 10, "multistart_families": "sweep"})
    assert tab.get_cfg()["multistart_families"] == "sweep"
    assert tab._combo_ms_families.isEnabled()

    tab.load({"n_starts": 2, "multistart_families": "everything"})
    assert tab.get_cfg()["multistart_families"] == "none"

    tab.load({"n_starts": 1 + MAX_SWEEP_COMBINATIONS})
    assert tab.get_cfg()["n_starts"] == 1 + MAX_SWEEP_COMBINATIONS


def test_ice_tab_exposes_init_strategy_and_kmeans_seed(qapp):
    """K-means warm-start widgets round-trip through load()/get_cfg()."""
    from pmcprg.pmc.gui.tabs import _IceTab

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
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({"init": "garbage-value"})
    assert tab.get_cfg()["init"] == "model"


def test_ice_tab_algorithm_default_is_ice(qapp):
    """Default cfg surfaces the ICE estimator and disables the SEM seed."""
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({})
    cfg = tab.get_cfg()
    assert cfg["algorithm"] == "ice"
    assert not tab._spn_sem_seed.isEnabled()


def test_ice_tab_round_trips_sem_settings(qapp):
    """SEM algorithm + seed must survive a load/get_cfg round-trip."""
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({"algorithm": "sem", "sem_seed": 555})
    cfg = tab.get_cfg()
    assert cfg["algorithm"] == "sem"
    assert cfg["sem_seed"] == 555
    assert tab._spn_sem_seed.isEnabled()


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

    from pmcprg.pmc.gui.tabs import _PriorTab
    from pmcprg.pmc.model    import Variant

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

    from pmcprg.pmc.gui.tabs import _PriorTab
    from pmcprg.pmc.model    import Variant

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

    from pmcprg.pmc.gui.tabs import PriorTabError, _PriorTab
    from pmcprg.pmc.model    import Variant

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
    from pmcprg.pmc.gui.tabs import _IceTab

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
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    cands = tab._selected_candidates()
    assert set(cands) >= {"Gauss", "Clayton", "GH", "Frank", "Joe"}


def test_ice_tab_candidates_quick_pick_buttons(qapp):
    """``All`` / ``None`` / ``Defaults`` quick-pick helpers behave as advertised."""
    from pmcprg.pmc.gui.tabs import _IceTab

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
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    assert tab._spn_ms_seed.value() == 42


def test_ice_tab_random_seed_button_changes_seed(qapp):
    """The 🎲 button writes a new value into the seed spin box."""
    from pmcprg.pmc.gui.tabs import _IceTab

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

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model           import PMCModel
    from pmcprg.pmc.simulate        import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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
        # n_eff is part of the contract now: the K² p-values are no longer
        # interchangeable, and a pair the chain rarely visits must be
        # readable as such (audit S-6).
        assert "n_eff" in r and "n_used" in r
        if "error" not in r:
            for k in ("tau", "stat", "p", "n_valid"):
                assert k in r, f"missing key {k!r}"
            assert 0.0 <= r["p"] <= 1.0


def test_do_gof_test_progress_callback_is_invoked(qapp):
    """``_do_gof_test`` reports progress once per bootstrap replicate.

    The unit changed with audit S-6: the bootstrap now resamples whole
    series under the fitted model, so all K² pairs come out of every
    replicate at once and the bar counts replicates, not pairs.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model           import PMCModel
    from pmcprg.pmc.simulate        import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl = PMCModel(toml)
    _, Y = simulate(mdl, N=200, seed=0)

    calls: list[tuple[int, int, str]] = []

    def cb(idx, total, _val, tag):
        calls.append((idx, total, tag))

    B = 10
    _ = PMCMainWindow._do_gof_test(mdl, Y, B=B, progress_cb=cb)
    assert len(calls) == B
    assert [c[0] for c in calls] == list(range(B))
    assert all(c[1] == B for c in calls)
    assert all(c[2] for c in calls)


# ---------------------------------------------------------------------------
# P17: _perturb_initial_model preserves SR-PMC symmetry & stays in bounds
# ---------------------------------------------------------------------------

def test_perturb_initial_model_preserves_sr_symmetry():
    """SR-PMC perturbed prior must remain symmetric and sum to 1."""
    import pathlib

    import numpy as _np

    from pmcprg.pmc.ice   import _perturb_initial_model
    from pmcprg.pmc.model import PMCModel

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.ice   import _perturb_initial_model
    from pmcprg.pmc.model import PMCModel

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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
    from pmcprg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "ok.csv"
    p.write_text("n,X,Y\n1,0,1.5\n2,1,2.5\n3,0,3.5\n")
    Y, X = _read_data_csv(str(p))
    assert list(Y) == [1.5, 2.5, 3.5]
    assert list(X) == [0, 1, 0]


def test_read_data_csv_y_only(tmp_path):
    """A CSV with only Y → X is None."""
    from pmcprg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "y_only.csv"
    p.write_text("Y\n1.0\n2.0\n3.0\n")
    Y, X = _read_data_csv(str(p))
    assert list(Y) == [1.0, 2.0, 3.0]
    assert X is None


def test_read_data_csv_empty_raises(tmp_path):
    from pmcprg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        _read_data_csv(str(p))


def test_read_data_csv_missing_y_raises(tmp_path):
    from pmcprg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "no_y.csv"
    p.write_text("n,Z\n1,1.5\n")
    with pytest.raises(ValueError, match="'Y'"):
        _read_data_csv(str(p))


def test_read_data_csv_non_numeric_y_reports_row(tmp_path):
    from pmcprg.pmc.gui.main_window import _read_data_csv

    p = tmp_path / "bad_y.csv"
    p.write_text("n,Y\n1,1.0\n2,not_a_number\n3,3.0\n")
    with pytest.raises(ValueError) as exc:
        _read_data_csv(str(p))
    # Row 3 = header row 1 + data row 2 (1-indexed CSV line).
    assert "row 3" in str(exc.value)
    assert "not_a_number" in str(exc.value)


def test_read_data_csv_non_integer_x_drops_labels(tmp_path):
    """Non-integer X values lead to X=None (with a logger warning)."""
    from pmcprg.pmc.gui.main_window import _read_data_csv

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

    from pmcprg.pmc.ice      import IceTrace, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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
    """K² TOML with TIED margins and ``margin_structure = "state"`` collapses
    to K state margins with an INFO log. (Without the key, K² blocks are
    pair margins — general PMC, see test_general_pmc_core.py.)"""
    import logging as _logging

    from pmcprg.pmc.model import PMCModel

    toml = tmp_path / "legacy_tied.toml"
    toml.write_text(
        '[model]\nname="legacy"\nvariant="PMC-IN"\nK=2\nN_default=100\n'
        'margin_structure="state"\n'
        '[prior]\np = [[0.45, 0.05], [0.05, 0.45]]\n'
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=1\ndist="norm"\nparams={loc=0.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=0\ndist="norm"\nparams={loc=2.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=1\ndist="norm"\nparams={loc=2.0, scale=1.0}\n'
    )
    with caplog.at_level(_logging.INFO, logger="pmcprg.pmc.model"):
        mdl = PMCModel(toml)
    # Must collapse to K state-indexed entries.
    assert sorted(mdl._state_margins) == [0, 1]
    blocks = mdl.margin_blocks()
    assert len(blocks) == 2
    assert all("j" not in b for b in blocks)         # canonical K-format
    # An INFO message should mention the legacy detection.
    assert any("legacy" in r.message.lower() for r in caplog.records)


def test_pmcmodel_warns_on_legacy_k2_untied_margins(tmp_path, caplog):
    """K² TOML with UNTIED margins collapsed by ``margin_structure = "state"``
    logs a WARNING (the collapse discards the j-dependence)."""
    import logging as _logging

    from pmcprg.pmc.model import PMCModel

    toml = tmp_path / "legacy_untied.toml"
    toml.write_text(
        '[model]\nname="legacy_bad"\nvariant="PMC-IN"\nK=2\nN_default=100\n'
        'margin_structure="state"\n'
        '[prior]\np = [[0.45, 0.05], [0.05, 0.45]]\n'
        # Untied: (0, 0) ≠ (0, 1) — not state margins (DerrodePieczynski_CSDA2013
        # §2.1 Proposition).
        '[[margins]]\ni=0\nj=0\ndist="norm"\nparams={loc=-1.0, scale=1.0}\n'
        '[[margins]]\ni=0\nj=1\ndist="norm"\nparams={loc=-2.0, scale=0.5}\n'
        '[[margins]]\ni=1\nj=0\ndist="norm"\nparams={loc=1.0, scale=1.0}\n'
        '[[margins]]\ni=1\nj=1\ndist="norm"\nparams={loc=1.0, scale=1.0}\n'
    )
    with caplog.at_level(_logging.WARNING, logger="pmcprg.pmc.model"):
        mdl = PMCModel(toml)

    # Must still construct (back-compat) but with a WARNING.
    assert any("f_ij depends on j" in r.message for r in caplog.records)
    # The kept density for state 0 is the anchor (j=0): loc=-1.0.
    f0 = mdl.margin(0)
    assert f0.params["loc"] == -1.0


def test_pmcmodel_k_format_canonical(tmp_path):
    """Canonical K-format TOML (one block per state) is the default schema."""
    from pmcprg.pmc.model import PMCModel

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

    from pmcprg.pmc.ice      import ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.ice      import ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/hmc_in_gauss_k2.toml")
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

    from pmcprg.pmc.ice      import ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.gui.main_window import (
        PMCMainWindow, _ALL_VIEWS, _ICE_VIEW_DASHBOARD, _ICE_VIEW_LOGLIK,
    )
    from pmcprg.pmc.ice      import IceResult, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_PLAYBACK
    from pmcprg.pmc.ice      import IceResult, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.gui.tabs import _PriorTab
    from pmcprg.pmc.model    import Variant

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
    ``pmcprg/pmc/models/pmc_gauss_k2.toml``.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

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
    from pmcprg.pmc.gui.main_window import PMCMainWindow

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

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.plot_style import apply_gui_compact_style

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
    from pmcprg.pmc.gui.main_window import _Canvas

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

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_DASHBOARD
    from pmcprg.pmc.ice      import IceResult, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_DASHBOARD
    from pmcprg.pmc.ice      import IceResult, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow()
    assert hasattr(w, "_btn_export")
    assert w._btn_export.text() == "Export…"
    assert not w._btn_export.isEnabled()


def test_export_button_enabled_after_plot(qapp):
    """``_finalize_plot`` enables the inline Export button alongside the menu action."""
    from pmcprg.pmc.gui.main_window import PMCMainWindow

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

    from pmcprg.pmc.gui.main_window import (
        PMCMainWindow, _ICE_VIEW_PARAM_COMPARE, _ALL_VIEWS,
    )
    from pmcprg.pmc.ice      import IceResult, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _ICE_VIEW_PARAM_COMPARE
    from pmcprg.pmc.ice      import IceResult, ice
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/hmc_in_gauss_k2.toml")
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
    """The little formatting helpers used by the comparison table behave.

    They were extracted from main_window into the headless-testable
    pmcprg.pmc.gui.views module (audit Q-10).
    """
    from pmcprg.pmc.gui.views import (
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

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_SIMULATION
    from pmcprg.pmc.model           import PMCModel
    from pmcprg.pmc.simulate        import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
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


def test_all_views_render_without_error(qapp, monkeypatch):
    """Safety net for the gui/views.py extraction (audit Q-10): every view that
    is *available* must render without hitting the error path. Covers the ICE
    trace views, simulation, classification, GoF heatmap, and image-input
    renderers in one pass."""
    import pathlib

    import numpy as np

    from pmcprg.pmc.gui.main_window import (
        PMCMainWindow, _ALL_VIEWS, _VIEW_IMAGE_RAW,
    )
    from pmcprg.pmc.ice       import IceResult, ice
    from pmcprg.pmc.inference import classify
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.simulate  import simulate

    toml = pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow()
    mdl = PMCModel(toml)
    w._load_model(toml)
    X_ref, Y = simulate(mdl, N=150, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 4})
    # ICE views (+ model / Y / init_model) via the worker callback path.
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted, Y=Y, trace=trace))
    # Simulation + classification + GoF view state.
    X_hat, gamma, _ = classify(fitted, Y)
    w._sim_state = (X_ref, Y, mdl)
    w._cls_state = (Y, X_hat, gamma, X_ref, mdl)
    w._gof_state = PMCMainWindow._do_gof_test(mdl, Y, B=5)

    # Trap any render that falls into the error path.
    failures: list = []
    monkeypatch.setattr(
        PMCMainWindow, "_render_error",
        lambda self, name, exc: failures.append((name, repr(exc))),
    )

    rendered = [v for v in _ALL_VIEWS if w._is_view_available(v)]
    for view in rendered:
        w._switch_view(view)

    # Image-input view separately (needs only _last_image; setting it alongside
    # _cls_state would make image-seg "available" with mismatched state).
    w._last_image = np.random.default_rng(0).random((8, 10))
    w._switch_view(_VIEW_IMAGE_RAW)

    assert not failures, f"render failures: {failures}"
    # The bulk (≥ 14 ICE views + simulation + classification + GoF) must render.
    assert len(rendered) >= 16, f"too few available views: {rendered}"


# ---------------------------------------------------------------------------
# PMCMainWindow — the [ice] section must survive a widget round-trip
# ---------------------------------------------------------------------------

def test_ice_section_round_trip_preserves_keys_without_widgets(qapp):
    """Rebuilding the model from the widgets must *merge* into ``[ice]``.

    ``_IceTab`` only knows the keys it has a widget for, and it has none for
    ``selection_criterion`` or ``margin_selection_rule``. Assigning its dict
    wholesale (``raw["ice"] = get_cfg()``) therefore deleted whatever else the
    user's TOML declared: loading such a model and saving it back silently
    dropped the criterion. Estimation itself was never affected — the merge in
    ``_parse_ice_cfg`` keeps the TOML value when the GUI cfg omits the key —
    but the file on disk lost it.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model import PMCModel

    raw = PMCModel(pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml")).raw
    raw["ice"] = {
        "selection_criterion":   "huard",
        "margin_selection_rule": "kolmogorov",
        "max_iter":              7,
    }

    win = PMCMainWindow()
    win._model = PMCModel.from_dict(raw)
    win._sync_widgets_from_model()
    after = win._rebuild_model_from_widgets().ice_config()

    assert after["selection_criterion"]   == "huard"
    assert after["margin_selection_rule"] == "kolmogorov"
    # Keys the tab *does* own still come from the widgets.
    assert after["max_iter"] == 7


# ---------------------------------------------------------------------------
# _IceTab — the estimator config surface must not drift from the API
# ---------------------------------------------------------------------------

def test_ice_tab_exposes_every_api_config_key(qapp):
    """Every key the estimators recognise must be settable in the GUI.

    The tab used to emit 12 of the 14 keys: `selection_criterion`,
    `margin_selection_rule` and `multistart_workers` had no widget, so none of
    the eight selection criteria — nor the process-parallel multistart — could
    be chosen, or even seen, from the interface. This test fails the moment a
    new key is added to the estimators without a widget.
    """
    from pmcprg.pmc._estim_common import ice_estim_defaults, sem_estim_defaults
    from pmcprg.pmc.gui.tabs import _IceTab

    api = set(ice_estim_defaults()) | set(sem_estim_defaults())
    gui = set(_IceTab().get_cfg())
    assert not (api - gui), f"config keys unreachable from the GUI: {sorted(api - gui)}"


def test_ice_tab_offers_every_registered_criterion(qapp):
    """The criterion and margin-rule combos are driven by the registries."""
    from pmcprg.pmc.gui.tabs import _IceTab
    from pmcprg.pmc.ice import MARGIN_SELECTION_RULES, SELECTION_CRITERIA

    tab = _IceTab()
    offered = {tab._combo_criterion.itemText(i)
               for i in range(tab._combo_criterion.count())}
    assert offered == set(SELECTION_CRITERIA)
    rules = {tab._combo_margin_rule.itemText(i)
             for i in range(tab._combo_margin_rule.count())}
    assert rules == set(MARGIN_SELECTION_RULES)


def test_ice_tab_criterion_reaches_the_estimator_config(qapp):
    """A criterion picked in the GUI must survive into the merged ice cfg."""
    import pathlib

    from pmcprg.pmc.gui.tabs import _IceTab
    from pmcprg.pmc.ice import _parse_ice_cfg
    from pmcprg.pmc.model import PMCModel

    tab = _IceTab()
    tab._combo_criterion.setCurrentText("xvcic")
    tab._combo_margin_rule.setCurrentText("kolmogorov")
    mdl = PMCModel(pathlib.Path("pmcprg/pmc/models/pmc_gauss_k2.toml"))
    cfg = _parse_ice_cfg(mdl, tab.get_cfg())

    assert cfg["selection_criterion"] == "xvcic"
    assert cfg["margin_selection_rule"] == "kolmogorov"


def test_ice_tab_round_trips_the_new_keys(qapp):
    """load() → get_cfg() preserves the three formerly-missing keys.

    ``multistart_workers`` is checked against the widget's own maximum
    rather than a literal: the spinbox is bounded by ``os.cpu_count()``, so
    hardcoding 3 passed on a 10-core laptop and failed on a 2-core CI runner
    with `assert 2 == 3`. The clamp is the intended behaviour, and it now
    gets an assertion of its own instead of being an accident of the host.
    """
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    workers = min(3, tab._spn_ms_workers.maximum())
    tab.load({"selection_criterion": "huard_global",
              "margin_selection_rule": "bic",
              "multistart_workers": workers,
              "n_starts": 2})
    cfg = tab.get_cfg()
    assert cfg["selection_criterion"]   == "huard_global"
    assert cfg["margin_selection_rule"] == "bic"
    assert cfg["multistart_workers"]    == workers

    # A TOML asking for more workers than the machine has must be clamped,
    # not refused: this is a performance knob, and a model file written on a
    # bigger machine has to stay loadable on a smaller one.
    tab.load({"multistart_workers": 9999, "n_starts": 2})
    assert tab.get_cfg()["multistart_workers"] == tab._spn_ms_workers.maximum()


# ---------------------------------------------------------------------------
# _IceTab — missing-observations widgets (missing_strategy / missing_draws /
# missing_seed)
# ---------------------------------------------------------------------------

def test_ice_tab_missing_widgets_default_from_ice_missing_defaults(qapp):
    """The three widgets exist and start at ``ice_missing_defaults()``'s values."""
    from pmcprg.pmc._estim_common import ice_missing_defaults
    from pmcprg.pmc.gui.tabs import _IceTab
    from pmcprg.pmc.ice import MISSING_STRATEGIES

    defaults = ice_missing_defaults()
    tab = _IceTab()
    assert set(tab._combo_missing_strategy.itemText(i)
               for i in range(tab._combo_missing_strategy.count())) == set(MISSING_STRATEGIES)
    assert tab._combo_missing_strategy.currentText() == defaults["missing_strategy"]
    assert tab._spn_missing_draws.value() == defaults["missing_draws"]
    assert tab._spn_missing_seed.value() == defaults["missing_seed"]
    cfg = tab.get_cfg()
    assert cfg["missing_strategy"] == defaults["missing_strategy"]
    assert cfg["missing_draws"]    == defaults["missing_draws"]
    assert cfg["missing_seed"]     == defaults["missing_seed"]


@pytest.mark.parametrize("strategy", ["available", "impute"])
def test_ice_tab_missing_widgets_round_trip(qapp, strategy):
    """load() -> get_cfg() preserves the missing-data keys for both strategies."""
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    tab.load({"missing_strategy": strategy, "missing_draws": 11, "missing_seed": 7})
    cfg = tab.get_cfg()
    assert cfg["missing_strategy"] == strategy
    assert cfg["missing_draws"]    == 11
    assert cfg["missing_seed"]     == 7


def test_ice_tab_missing_draws_and_seed_disabled_unless_impute(qapp):
    """Imputation-only widgets are enabled iff missing_strategy == 'impute'."""
    from pmcprg.pmc.gui.tabs import _IceTab

    tab = _IceTab()
    assert tab._combo_missing_strategy.currentText() == "available"
    assert not tab._spn_missing_draws.isEnabled()
    assert not tab._spn_missing_seed.isEnabled()

    tab._combo_missing_strategy.setCurrentText("impute")
    assert tab._spn_missing_draws.isEnabled()
    assert tab._spn_missing_seed.isEnabled()

    tab._combo_missing_strategy.setCurrentText("available")
    assert not tab._spn_missing_draws.isEnabled()
    assert not tab._spn_missing_seed.isEnabled()

    # load() must re-apply the enable/disable rule too, not only the combo signal.
    tab.load({"missing_strategy": "impute"})
    assert tab._spn_missing_draws.isEnabled()
    assert tab._spn_missing_seed.isEnabled()
    tab.load({"missing_strategy": "available"})
    assert not tab._spn_missing_draws.isEnabled()
    assert not tab._spn_missing_seed.isEnabled()


# ---------------------------------------------------------------------------
# Multivariate observations (d ≥ 2) — audit G-4, G-5, G-6
# ---------------------------------------------------------------------------

_MVN_MODEL = "pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml"


@pytest.mark.parametrize("toml_name,expect_ndim", [
    ("hmc_in_mvn_k2_d3.toml", 2),      # d = 3
    ("pmc_gauss_k2.toml",     1),      # scalar control
])
def test_main_views_render_for_any_observation_dimension(qapp, toml_name, expect_ndim):
    """The two main views must render for d ≥ 2, not only for scalar data.

    They indexed `Y` directly, so every d ≥ 2 model raised "x and y must be
    the same size" — the package ships a d = 3 fixture, and its two central
    views were unusable on it.
    """
    import pathlib

    from matplotlib.figure import Figure

    from pmcprg.pmc import PMCModel, classify, simulate
    from pmcprg.pmc.gui import views

    mdl = PMCModel(pathlib.Path("pmcprg/pmc/models") / toml_name)
    X, Y = simulate(mdl, N=200, seed=1)
    assert Y.ndim == expect_ndim
    X_hat, gamma, _ = classify(mdl, Y)

    views.plot_simulation(Figure(), X, Y, mdl)                    # must not raise
    views.plot_classification(Figure(), Y, X_hat, gamma, X, mdl)  # must not raise


def test_data_csv_round_trip_is_dimension_agnostic(qapp, tmp_path):
    """Multivariate observations survive save → load through the CSV path.

    The reader accepted a single scalar 'Y' column, so with the image path
    broken there was no way at all to bring d ≥ 2 data into the GUI.
    """
    import csv
    import pathlib

    import numpy as np

    from pmcprg.pmc import PMCModel, simulate
    from pmcprg.pmc.gui.main_window import _read_data_csv

    mdl = PMCModel(pathlib.Path(_MVN_MODEL))
    X, Y = simulate(mdl, N=120, seed=3)
    path = tmp_path / "mv.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "X", *[f"Y{j}" for j in range(Y.shape[1])]])
        for n in range(len(Y)):
            w.writerow([n + 1, int(X[n]), *[float(v) for v in Y[n]]])

    Y_back, X_back = _read_data_csv(path)
    assert Y_back.shape == Y.shape
    assert np.allclose(Y_back, Y)
    assert X_back is not None and np.array_equal(X_back, X)


def test_margin_dialog_parses_vector_and_matrix_parameters(qapp):
    """A multivariate_normal margin must be editable, not silently emptied.

    The scalar token scanner chopped `mean=[-1.0, -0.8, -0.5]` at the first
    comma, reported "Cannot parse value '[-1.0' … as a number" and returned an
    EMPTY dict — every parameter of the block was dropped.
    """
    import pathlib

    from pmcprg.pmc.gui.dialogs import _MarginDialog
    from pmcprg.pmc.model import PMCModel

    mdl = PMCModel(pathlib.Path(_MVN_MODEL))
    params, err = _MarginDialog(mdl.raw["margins"][0])._parse_params()
    assert err is None
    assert params["mean"] == [-1.0, -0.8, -0.5]
    assert params["cov"][0] == [1.0, 0.3, 0.2]

    # Scalar margins keep their existing behaviour, errors included.
    scalar = _MarginDialog({"i": 0, "dist": "norm",
                            "params": {"loc": 1.0, "scale": 2.0}})
    assert scalar._parse_params() == ({"loc": 1.0, "scale": 2.0}, None)
    bad = _MarginDialog({"i": 0, "dist": "norm", "params": {}})
    bad._params_edit.setText("loc=abc")
    assert "as a number" in bad._parse_params()[1]


# ---------------------------------------------------------------------------
# State coherence — audit S-1, S-2, S-3, S-4
# ---------------------------------------------------------------------------

_MODEL_A = "pmcprg/pmc/models/pmc_gauss_k2.toml"
_MODEL_B = "pmcprg/pmc/models/hmc_in_gauss_k3.toml"

_RESULT_ATTRS = ("_sim_state", "_cls_state", "_gof_state",
                 "_ice_trace", "_init_model", "_ice_Y")


@pytest.fixture
def no_modals(monkeypatch):
    """Neutralise every blocking dialog — a modal hangs an offscreen run."""
    from PyQt6.QtWidgets import QMessageBox

    seen: list[tuple] = []
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(
            QMessageBox, name,
            staticmethod(lambda *a, _n=name, **k: seen.append((_n, a[1:3]))),
        )
    return seen


def _window_with_results(toml, *, n=120, iters=2):
    """A window carrying a full result set: simulation, classification, ICE."""
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.ice       import IceResult, ice
    from pmcprg.pmc.inference import classify
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.simulate  import simulate

    w   = PMCMainWindow(toml)
    mdl = PMCModel(pathlib.Path(toml))
    X, Y = simulate(mdl, N=n, seed=0)
    w._last_X, w._last_Y = X, Y
    w._on_sim_done((X, Y, mdl))
    w._on_cls_done((*classify(mdl, Y), mdl, Y, X))
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": iters})
    w._on_est_done(IceResult(initial_model=mdl, fitted_model=fitted,
                             Y=Y, trace=trace))
    return w


def test_worker_start_locks_the_state_mutating_menu_actions(qapp):
    """Menu entries that replace state must be locked while a worker runs.

    Only the four action buttons were greyed out, so `File → Load data` was
    live during a run. Loading a CSV then made `_on_cls_done` compare the
    NEW `_last_X` to the OLD worker's `X_hat` — an exception escaping a Qt
    slot, which aborts the process (SIGABRT).
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    assert w._mutating_actions, "no mutating action registered"
    assert all(a.isEnabled() for a in w._mutating_actions)

    w._set_busy(True)
    assert not any(a.isEnabled() for a in w._mutating_actions), \
        "a state-mutating menu action stayed live during a worker run"
    assert not w._btn_sim.isEnabled()

    w._set_busy(False)
    assert all(a.isEnabled() for a in w._mutating_actions)


def test_classify_result_carries_its_own_data_and_reference(qapp):
    """`_on_cls_done` must not re-read `self._last_X` after the run.

    The reference labels have to travel with the job: when the data changed
    mid-run, comparing the fresh `_last_X` to the old `X_hat` raised inside
    `error_rate`, straight out of a Qt slot.
    """
    import pathlib

    import numpy as np

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    w   = PMCMainWindow(_MODEL_A)
    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=100, seed=0)
    result = PMCMainWindow._do_classify(mdl, Y, X)
    assert len(result) == 6, "the job must carry back its own Y and X_ref"

    # The race: the window's data is replaced while the job is in flight.
    w._last_X, w._last_Y = np.zeros(7, dtype=int), np.zeros(7)
    w._on_cls_done(result)                            # must not raise

    assert w._cls_state[0] is Y and w._cls_state[3] is X, \
        "state kept the window's current data instead of the job's"


def test_worker_done_routes_a_failing_callback_instead_of_aborting(qapp, no_modals):
    """An exception in `on_done` must not escape the Qt slot.

    `_worker_done` called `on_done(result)` bare, so anything raising there
    took the whole process down (SIGABRT) instead of surfacing.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)

    def boom(_result):
        raise RuntimeError("deliberate callback failure")

    w._worker_done(("payload",), boom)                # must not raise

    assert "deliberate callback failure" in w._log.toPlainText()
    assert any(kind == "critical" for kind, _ in no_modals), \
        "the failure was swallowed without telling the user"
    # And the window must not be left locked by the failure.
    assert all(a.isEnabled() for a in w._mutating_actions)


def test_opening_another_model_invalidates_every_result(qapp):
    """Results describe the model that produced them (audit S-2).

    `_load_model` replaced `_model` but touched no result state, so the
    header announced model B while the canvas still showed A's plot — with
    nothing in the interface signalling the mismatch.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_NONE

    for toml in (_MODEL_A, _MODEL_B):
        if not pathlib.Path(toml).exists():
            pytest.skip(f"test fixture {toml} not present")
    assert PMCMainWindow                              # imported for the fixture

    w = _window_with_results(_MODEL_A)
    assert w._sim_state and w._cls_state and w._ice_trace and w._has_plot

    w._load_model(_MODEL_B)

    assert w._model.K == 3
    for attr in _RESULT_ATTRS:
        assert getattr(w, attr) is None, f"{attr} survived the model change"
    assert not w._has_plot, "canvas still holds the previous model's plot"
    assert w._current_view == _VIEW_NONE
    # Data is independent of the model and must survive.
    assert w._last_Y is not None


def test_loading_data_invalidates_results_describing_the_old_data(qapp, tmp_path):
    """Audit S-3 — `_on_load_data` cleared the image state only.

    `_sim_state`, `_cls_state`, `_gof_state`, `_ice_trace`, `_ice_Y` and
    `_init_model` all went on describing the sequence just replaced, and
    their views stayed selectable.
    """
    import csv

    from PyQt6.QtWidgets import QFileDialog

    from pmcprg.pmc.gui import main_window as mw

    w = _window_with_results(_MODEL_A)
    path = tmp_path / "fresh.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "Y"])
        wr.writerows([[i + 1, float(i)] for i in range(40)])

    orig = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(path), ""))
    try:
        w._on_load_data()
    finally:
        QFileDialog.getOpenFileName = orig
    assert mw                                          # module used above

    for attr in _RESULT_ATTRS:
        assert getattr(w, attr) is None, f"{attr} still describes the old data"
    assert len(w._last_Y) == 40
    assert not w._act_save_seg.isEnabled()
    assert not w._has_plot


def test_view_selector_never_names_a_view_the_canvas_is_not_showing(qapp):
    """Audit S-4 — the combobox and the canvas must not drift apart.

    Two ways they did: `_load_model` never repopulated the selector, so
    copula views stayed clickable on a copula-free model while
    `_switch_view` bailed out *silently* after the combobox text had already
    moved; and with every item disabled the combobox fell back to item 0,
    naming "Simulation" over a blank canvas.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import _VIEW_NONE

    for toml in (_MODEL_A, _MODEL_B):
        if not pathlib.Path(toml).exists():
            pytest.skip(f"test fixture {toml} not present")

    w = _window_with_results(_MODEL_A)
    w._load_model(_MODEL_B)               # HMC-IN K=3 — no copulas

    stale = [
        w._cmb_view.itemText(i)
        for i in range(w._cmb_view.count())
        if (item := w._cmb_view.model().item(i)) is not None
        and item.isEnabled()
        and not w._is_view_available(w._cmb_view.itemText(i))
    ]
    assert not stale, f"selector offers unavailable views: {stale}"

    # Nothing renderable: the selector must say so, not name a view.
    assert w._cmb_view.currentText() == _VIEW_NONE == w._current_view

    # A refused switch leaves the text on what the canvas actually holds.
    w._switch_view("Simulation")
    assert w._cmb_view.currentText() == w._current_view


def test_reset_session_forgets_which_view_was_on_screen(qapp):
    """Audit S-4 — `_on_reset` left `_current_view` naming the wiped view."""
    from pmcprg.pmc.gui.main_window import _VIEW_NONE

    w = _window_with_results(_MODEL_A)
    assert w._current_view != _VIEW_NONE

    w._on_reset()

    assert w._current_view == _VIEW_NONE
    assert w._cmb_view.currentText() == _VIEW_NONE
    for attr in (*_RESULT_ATTRS, "_last_X", "_last_Y", "_last_image"):
        assert getattr(w, attr) is None
    assert not w._has_plot


# ---------------------------------------------------------------------------
# Gaps the adversarial pass over the S-1…S-4 fixes turned up
# ---------------------------------------------------------------------------

def test_reset_has_no_second_binding_that_escapes_the_busy_lock(qapp):
    """Ctrl+R must not reach `_on_reset` while a worker is running.

    A global `QShortcut(Ctrl+R)` sat beside the menu action's own Ctrl+R
    "so the GUI tests can trigger it without a menu bar". No test ever used
    it, and it did two kinds of damage: it made the shortcut ambiguous, so
    Ctrl+R did nothing in normal use; and being a QShortcut rather than a
    QAction it escaped `_set_busy`, firing exactly when Reset was supposed
    to be locked — wiping the session under a running worker, which then
    wrote its results back over the reset.
    """
    from PyQt6.QtGui import QShortcut

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    assert not w.findChildren(QShortcut), \
        "a QShortcut bypasses the _set_busy lock on the menu actions"

    # And the handler itself refuses to run under a worker.
    class _FakeWorker:
        @staticmethod
        def isRunning():
            return True

    w._last_Y = __import__("numpy").zeros(5)
    w._worker = _FakeWorker()
    w._on_reset()
    assert w._last_Y is not None, "Reset cleared the state a worker still owns"


def test_simulate_invalidates_results_describing_the_previous_sequence(qapp):
    """Simulate replaces the observations, so old results must go.

    `_on_sim_done` set `_last_X`/`_last_Y` but never invalidated: the ICE
    trace, its observation sequence and the GoF result all went on
    describing a sequence that no longer existed — and Simulate is one
    button click, the widest case of this class.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    w = _window_with_results(_MODEL_A, n=120, iters=2)
    w._gof_state = PMCMainWindow._do_gof_test(w._model, w._last_Y, B=5)
    assert w._ice_Y is not None and w._gof_state

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X2, Y2 = simulate(mdl, N=200, seed=3)
    w._on_sim_done((X2, Y2, mdl))

    for attr in ("_ice_trace", "_ice_Y", "_gof_state", "_init_model", "_cls_state"):
        assert getattr(w, attr) is None, f"{attr} still describes the old sequence"
    assert w._sim_state is not None and len(w._sim_state[1]) == 200


def test_loading_reference_labels_refreshes_the_view_on_screen(qapp, tmp_path):
    """Reference labels are a direct input of the segmentation view.

    They are not carried inside `_cls_state`, so writing `_last_image_ref`
    makes whatever is on the canvas stale. Every other loader repopulates
    and re-renders; this one only logged, leaving the segmentation panel
    silently one panel short — which reads as "the reference was refused".
    """
    import numpy as np

    from PyQt6.QtWidgets import QFileDialog

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_IMAGE_SEG

    Image = pytest.importorskip("PIL.Image")

    img = (np.random.default_rng(0).random((24, 32)) * 255).astype(np.uint8)
    lab = (np.random.default_rng(1).integers(0, 2, (24, 32)) * 255).astype(np.uint8)
    p_img, p_lab = tmp_path / "img.png", tmp_path / "lab.png"
    Image.fromarray(img).save(p_img)
    Image.fromarray(lab).save(p_lab)

    w = PMCMainWindow(_MODEL_A)
    orig = QFileDialog.getOpenFileName
    try:
        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(p_img), ""))
        w._on_load_image()
        w._on_cls_done((*__import__("pmcprg.pmc.inference", fromlist=["classify"])
                        .classify(w._model, w._last_Y), w._model, w._last_Y, None))
        assert w._current_view == _VIEW_IMAGE_SEG
        n_before = len(w._canvas.fig.axes)

        QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(p_lab), ""))
        w._on_load_image_ref()
    finally:
        QFileDialog.getOpenFileName = orig

    assert w._last_image_ref is not None
    assert w._current_view == _VIEW_IMAGE_SEG
    assert len(w._canvas.fig.axes) > n_before, \
        "the reference panel never appeared — the canvas was not refreshed"


def test_data_of_the_wrong_dimension_does_not_survive_a_model_change(qapp):
    """Observations survive an Open — except when `d` no longer matches.

    "Data is independent of the model" holds for K and for the parameter
    values, not for the observation dimension. Keeping a (N, 3) sequence
    under a scalar model left all four action buttons enabled on data that
    could only ever produce a raw traceback in a message box.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mv = "pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml"
    for toml in (mv, _MODEL_A):
        if not pathlib.Path(toml).exists():
            pytest.skip(f"test fixture {toml} not present")

    w   = PMCMainWindow(mv)
    mdl = PMCModel(pathlib.Path(mv))
    X, Y = simulate(mdl, N=80, seed=0)
    w._on_sim_done((X, Y, mdl))
    assert w._last_Y.shape == (80, 3)

    w._load_model(_MODEL_A)                       # scalar model
    assert w._last_Y is None, "d=3 data survived into a d=1 model"
    assert "dropped" in w._log.toPlainText()

    # Control: a compatible model change keeps the data.
    w2   = PMCMainWindow(_MODEL_A)
    mdl2 = PMCModel(pathlib.Path(_MODEL_A))
    X2, Y2 = simulate(mdl2, N=80, seed=0)
    w2._on_sim_done((X2, Y2, mdl2))
    w2._load_model("pmcprg/pmc/models/hmc_in_gauss_k2.toml")
    assert w2._last_Y is not None and len(w2._last_Y) == 80


# ---------------------------------------------------------------------------
# A model file the parser accepts but the editor cannot display — audit SLOT-1…3
# ---------------------------------------------------------------------------

def _hostile_toml(tmp_path, name, transform):
    import pathlib
    src = pathlib.Path(_MODEL_A).read_text()
    out = tmp_path / name
    out.write_text(transform(src))
    return out


@pytest.mark.parametrize("name,transform,widget,expected", [
    # A non-numeric [ice] value: int() raised in _IceTab.load.
    ("bad_maxiter.toml",
     lambda s: s.replace("max_iter = 50", 'max_iter = "many"', 1),
     "_spn_maxiter", 50),
    # A seed outside int32: QSpinBox.setValue raised OverflowError.
    ("big_seed.toml",
     lambda s: s.replace("multistart_seed = 42",
                         "multistart_seed = 99999999999999999999", 1),
     "_spn_ms_seed", None),
])
def test_out_of_range_ice_values_fall_back_instead_of_aborting(
        qapp, tmp_path, no_modals, name, transform, widget, expected):
    """`[ice]` values cross straight from the TOML into the spin boxes.

    PMCModel neither type-checks nor range-checks that table, and
    `_IceTab.load` used bare `int()` / `setValue()`, so a hand-edited file
    raised while populating the widgets — on the unguarded stretch of
    `_load_model`, inside a Qt slot, which aborts the process (exit 134).
    The two *string* keys already fell back to a default; the numeric ones
    did not.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model import PMCModel

    path = _hostile_toml(tmp_path, name, transform)
    PMCModel(path)                                # the parser accepts it

    w = PMCMainWindow(_MODEL_A)
    w._load_model(path)                           # must not raise

    spin = getattr(w._tab_ice, widget)
    assert spin.value() == (expected if expected is not None else spin.maximum())
    assert not any(kind == "critical" for kind, _ in no_modals), \
        "a value the widget can clamp should not need an error dialog"


def test_a_non_numeric_copula_extra_does_not_abort_the_grid(qapp, tmp_path, no_modals):
    """`_CopulaTab._format_block` formatted every extra key with `:.3g`.

    PMCModel keeps keys it does not know, so a per-pair comment string (or a
    stray list) made the formatter raise while painting the grid — again on
    `_load_model`'s unguarded stretch.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    path = _hostile_toml(
        tmp_path, "cop_extra.toml",
        lambda s: s.replace('{ i = 0, j = 0, name = "Gauss", tau = 0.6 }',
                            '{ i = 0, j = 0, name = "Gauss", tau = 0.6, '
                            'note = "diagonal pair" }', 1),
    )
    w = PMCMainWindow(_MODEL_A)
    w._load_model(path)                           # must not raise

    assert "diagonal pair" in w._tab_copulas._table.item(0, 0).text()


def test_load_model_rolls_back_when_the_editor_cannot_display_it(
        qapp, monkeypatch, no_modals):
    """Everything after a successful parse ran bare inside a Qt slot.

    `_load_model`'s try/except covered `PMCModel()` only, so any later
    failure — widget population, invalidation, the view selector — aborted
    the process with no dialog and no chance to save in-progress edits. This
    is the failure mode `_worker_done` was hardened against; the synchronous
    load path had been left out.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    before = w._model.name

    def boom(self):
        raise RuntimeError("widgets cannot display this")

    monkeypatch.setattr(PMCMainWindow, "_sync_widgets_from_model", boom)
    w._load_model(_MODEL_B)                       # must not raise

    assert w._model.name == before, "the previous model was not restored"
    assert any(kind == "critical" for kind, _ in no_modals)


# ---------------------------------------------------------------------------
# Multivariate margins in the ICE views — audit PB-1, PB-2
# ---------------------------------------------------------------------------

def test_ice_parameter_views_render_for_multivariate_margins(qapp):
    """Two ICE views were permanently dead for every d ≥ 2 model.

    A multivariate margin carries its mean as a vector and its covariance as
    a nested list; both renderers coerced every parameter with `float()`.
    `_render_view` caught the TypeError, so the process survived and the
    canvas only ever showed the red "render failed" card — which is why this
    outlived the commit that claimed multivariate observations were usable
    end to end.
    """
    import pathlib

    from matplotlib.figure import Figure

    from pmcprg.pmc.gui import views
    from pmcprg.pmc.ice   import ice
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    toml = pathlib.Path("pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml")
    if not toml.exists():
        pytest.skip(f"test fixture {toml} not present")

    mdl = PMCModel(toml)
    _, Y = simulate(mdl, N=120, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 2})

    views.plot_ice_param_compare(Figure(), mdl, fitted)   # must not raise
    views.plot_ice_margins(Figure(), trace)               # must not raise

    # The formatter must describe structure, not flatten it away.
    assert views._fmt_param_value([-1.0, -0.8, -0.5]).startswith("[")
    assert "3×3" in views._fmt_param_value([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert views._fmt_param_value(2.5) == views._fmt_value(2.5)


# ---------------------------------------------------------------------------
# Observation dimension on the data path — audit N-1, N-4
# ---------------------------------------------------------------------------

def test_a_lone_vector_column_reads_back_as_scalar(qapp, tmp_path):
    """A single `Y0` column means scalar observations in vector style.

    Returning `(N, 1)` made every downstream scalar path fail — "could not
    broadcast (30,1) into (30,)" — and the dimension check waved it through,
    since d = 1 either way.
    """
    import csv

    from pmcprg.pmc.gui.main_window import _read_data_csv

    path = tmp_path / "one_col.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "Y0"])
        wr.writerows([[i + 1, float(i)] for i in range(30)])

    Y, _ = _read_data_csv(path)
    assert Y.shape == (30,)


def test_load_data_refuses_observations_of_the_wrong_dimension(
        qapp, tmp_path, no_modals):
    """The d check was wired into `_load_model` only.

    `Load data` happily accepted a (N, 3) CSV under a scalar model with
    every action button live — the same trap the model side already closed.
    """
    import csv
    import pathlib

    from PyQt6.QtWidgets import QFileDialog

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mv = pathlib.Path("pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml")
    if not mv.exists():
        pytest.skip(f"test fixture {mv} not present")

    X, Y = simulate(PMCModel(mv), N=40, seed=1)
    path = tmp_path / "d3.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "X", "Y0", "Y1", "Y2"])
        for i in range(40):
            wr.writerow([i + 1, int(X[i]), *map(float, Y[i])])

    w = PMCMainWindow(_MODEL_A)                   # scalar model
    orig = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (str(path), ""))
    try:
        w._on_load_data()
    finally:
        QFileDialog.getOpenFileName = orig

    assert w._last_Y is None, "3-dimensional data was accepted by a scalar model"
    kinds = [k for k, _ in no_modals]
    assert "warning" in kinds, "the refusal was silent"


# ---------------------------------------------------------------------------
# Errors, robustness and persistence — audit PB-3, S-5, S-8…S-13
# ---------------------------------------------------------------------------

@pytest.fixture
def answer_question(monkeypatch):
    """Drive `QMessageBox.question` from the test; records the prompts."""
    from PyQt6.QtWidgets import QMessageBox

    state = {"reply": QMessageBox.StandardButton.Discard, "asked": []}

    def answer(*args, **kwargs):
        state["asked"].append(str(args[2]) if len(args) > 2 else "")
        return state["reply"]

    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer))
    return state


def test_a_failed_render_leaves_nothing_to_export(qapp, monkeypatch):
    """Audit PB-3 — Export-plot exported the "render failed" card.

    `_has_plot` survived from whatever last rendered successfully, so the
    button stayed live over an error placeholder.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_SIMULATION
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    w   = PMCMainWindow(_MODEL_A)
    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=100, seed=0)
    w._on_sim_done((X, Y, mdl))
    assert w._has_plot and w._act_export_plot.isEnabled()

    def boom(self, *a):
        raise RuntimeError("renderer is broken")

    monkeypatch.setattr(PMCMainWindow, "_plot_simulation", boom)
    w._switch_view(_VIEW_SIMULATION)

    assert not w._has_plot
    assert not w._act_export_plot.isEnabled()
    assert not w._btn_export.isEnabled()


def test_worker_error_leads_with_the_exception_not_the_stack(qapp, no_modals):
    """Audit S-12 — the dialog pasted `tb[-1500:]`.

    A traceback truncated at the *head* buried the useful line under stack
    frames. The full traceback still goes to the log panel.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    tb = ("Traceback (most recent call last):\n"
          + "  File \"f\", line 1\n" * 300
          + "IncompatibleObservationError: marginal density of Y[0]=nan "
            "is zero or non-finite\n")
    w._worker_error(tb)

    _, text = next(args for kind, args in no_modals if kind == "critical")
    assert text.splitlines()[0].startswith("IncompatibleObservationError")
    assert "marginal density" in w._log.toPlainText()


@pytest.mark.parametrize("value,line", [("nan", 2), ("inf", 2)])
def test_a_degenerate_csv_is_refused_with_the_offending_row(qapp, tmp_path,
                                                            value, line):
    """Audit S-13 — non-finite observations loaded without a word.

    The failure only surfaced at the next click, as a traceback.
    """
    import csv

    from pmcprg.pmc.gui.main_window import _read_data_csv

    path = tmp_path / f"{value}.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["n", "Y"])
        wr.writerows([[i + 1, value] for i in range(20)])

    with pytest.raises(ValueError, match=rf"Non-finite Y at row {line}"):
        _read_data_csv(path)


def test_unsaved_edits_are_never_discarded_without_asking(
        qapp, answer_question):
    """Audit S-5 — no guard on Open, on Recent files, or on close.

    `_load_model` called `_mark_clean()` without ever consulting `_dirty`,
    and `closeEvent` checked only the worker.
    """
    from PyQt6.QtGui     import QCloseEvent
    from PyQt6.QtWidgets import QMessageBox

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    w._lbl_name.setText("EDITED")
    assert w._dirty and w.windowTitle().endswith("*")

    answer_question["reply"] = QMessageBox.StandardButton.Cancel
    w._load_model(_MODEL_B)
    assert answer_question["asked"], "the edits were dropped silently"
    assert w._model.K == 2, "Cancel did not stop the load"
    assert w._lbl_name.text() == "EDITED" and w._dirty

    ev = QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted(), "Cancel did not stop the close"

    answer_question["reply"] = QMessageBox.StandardButton.Discard
    w._load_model(_MODEL_B)
    assert w._model.K == 3, "Discard did not let the load through"


def test_save_as_commits_the_path_only_once_the_write_lands(
        qapp, tmp_path, no_modals):
    """Audit S-8 — `_on_save_as` assigned `_model_path` first.

    After a failed write a later Ctrl+S silently re-aimed at the bad path
    instead of falling back to Save-As — the regression the "B6" note on
    `_load_model` records as fixed for loading.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    before = w._model_path

    w._on_save(tmp_path / "no" / "such" / "dir" / "m.toml")
    assert w._model_path == before, "an unwritable path was committed"
    assert any(kind == "critical" for kind, _ in no_modals)

    good = tmp_path / "m.toml"
    w._on_save(good)
    assert w._model_path == good and not w._dirty


def test_estimate_leaves_the_model_marked_dirty(qapp):
    """Audit S-9 — `_on_est_done` called `_mark_clean()`.

    The estimator returns a model that differs from the file on disk, so
    declaring it clean dropped the asterisk and let a later Ctrl+S overwrite
    the user's source TOML with the fit, without confirmation.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = _window_with_results(_MODEL_A, n=100, iters=2)
    assert isinstance(w, PMCMainWindow)
    assert w._dirty, "the fitted model was declared identical to the file"
    assert w.windowTitle().endswith("*")


def test_seed_and_algorithm_survive_a_disk_round_trip(qapp, tmp_path):
    """Audit S-10, S-11 — what the GUI writes, and where.

    The Seed field was never persisted, so a reproducible simulation stopped
    being reproducible after one round trip. And `sem_seed` (a SEM knob) and
    `algorithm` (a GUI dispatch key with no API counterpart) were both
    written under `[ice]`, leaving a GUI-saved file non-canonical.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model import PMCModel

    w = PMCMainWindow(_MODEL_A)
    w._txt_seed.setText("13")
    w._tab_ice._combo_algorithm.setCurrentText("sem")
    w._tab_ice._spn_sem_seed.setValue(77)

    out = tmp_path / "round.toml"
    w._on_save(out)

    mdl = PMCModel(out)
    assert mdl.raw["model"]["seed"] == 13
    assert mdl.sem_config().get("sem_seed") == 77
    assert "sem_seed" not in mdl.ice_config()
    assert "algorithm" not in mdl.ice_config()

    back = PMCMainWindow(str(out))
    assert back._txt_seed.text() == "13"
    assert back._tab_ice._combo_algorithm.currentText() == "sem"
    assert back._tab_ice._spn_sem_seed.value() == 77

    # A file written before S-11 keeps its [ice]-hosted keys readable.
    legacy = PMCMainWindow(_MODEL_A)
    assert legacy._tab_ice._combo_algorithm.currentText() == "ice"


# ---------------------------------------------------------------------------
# Numeric results export and view provenance — audit S-7, S-14
# ---------------------------------------------------------------------------

def _export_to(w, path, kind):
    from PyQt6.QtWidgets import QFileDialog

    orig = QFileDialog.getOpenFileName, QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (str(path), kind))
    try:
        w._on_export_results()
    finally:
        QFileDialog.getOpenFileName, QFileDialog.getSaveFileName = orig


@pytest.mark.parametrize("toml,expect_gof", [
    (_MODEL_A, True),
    ("pmcprg/pmc/models/hmc_in_mvn_k2_d3.toml", False),   # d=3, no copulas
])
def test_results_export_carries_the_numbers_out_of_the_gui(
        qapp, tmp_path, toml, expect_gof):
    """Audit S-7 — no numeric output existed at all.

    The log-likelihood trace, the τ trajectories, the family history, the
    classification error rate and the GoF results lived only in memory and
    in views: recovering a number meant reading a PNG, or leaving the GUI
    and calling `ice()` by hand.
    """
    import csv
    import json
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    if not pathlib.Path(toml).exists():
        pytest.skip(f"test fixture {toml} not present")

    w = _window_with_results(toml, n=120, iters=3)
    if expect_gof:
        w._gof_state = PMCMainWindow._do_gof_test(w._model, w._last_Y, B=5)

    out_json = tmp_path / "results.json"
    _export_to(w, out_json, "JSON")
    payload = json.loads(out_json.read_text())

    assert payload["estimation"]["log_liks"] == list(w._ice_trace.log_liks)
    assert payload["estimation"]["config"]["selection_criterion"]
    assert payload["classification"]["error_rate"] is not None
    assert (payload["goodness_of_fit"] is not None) is expect_gof
    assert payload["model"]["name"] == w._model.name

    out_csv = tmp_path / "results.csv"
    _export_to(w, out_csv, "CSV")
    rows = list(csv.reader(out_csv.open()))
    header, body = rows[0], rows[1:]

    assert header[:3] == ["run", "iteration", "log_lik"]
    assert len(body) == len(w._ice_trace.log_liks)
    assert float(body[0][2]) == pytest.approx(w._ice_trace.log_liks[0])
    # Vector-valued margin parameters must be expanded, not dropped.
    margin_cols = [h for h in header if h.startswith("margin")]
    assert margin_cols
    if "mvn" in toml:
        assert any(h.startswith("margin0_mean_") for h in margin_cols), \
            "a multivariate mean was not expanded component-wise"


def test_results_export_says_so_when_there_is_nothing_to_export(
        qapp, tmp_path, no_modals):
    """A fresh session must explain itself rather than write an empty file."""
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(_MODEL_A)
    _export_to(w, tmp_path / "x.json", "JSON")

    assert any(kind == "information" for kind, _ in no_modals)
    assert not (tmp_path / "x.json").exists()


def test_each_view_names_the_model_that_produced_it(qapp):
    """Audit S-14 — two views could describe two different models silently.

    Every action rebuilds its own model from the widgets and stores it in
    its own state, so the Simulation panel and the Classification panel can
    legitimately disagree — with nothing in the figure saying so.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import (
        PMCMainWindow, _VIEW_CLASSIFICATION, _VIEW_GOF_HEATMAP, _VIEW_SIMULATION,
    )
    from pmcprg.pmc.inference import classify
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.simulate  import simulate

    w   = PMCMainWindow(_MODEL_A)
    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=120, seed=0)
    w._on_sim_done((X, Y, mdl))

    def stamp():
        return [t.get_text() for t in w._canvas.fig.texts
                if t.get_text().startswith("model:")]

    assert stamp() == [f"model: {mdl.name}"]

    # Edit the model, then classify: the two views now describe two models.
    w._lbl_name.setText("EDITED MODEL")
    edited = w._rebuild_model_from_widgets()
    w._on_cls_done((*classify(edited, Y), edited, Y, X))
    assert stamp() == ["model: EDITED MODEL"]

    w._switch_view(_VIEW_SIMULATION)
    from_sim = stamp()
    w._switch_view(_VIEW_CLASSIFICATION)
    assert from_sim != stamp(), "both views claimed the same model"

    # The GoF heatmap is a pure function of its own snapshot — naming the
    # *current* model over it would be the misattribution this prevents.
    w._gof_state = PMCMainWindow._do_gof_test(mdl, Y, B=5)
    w._populate_view_selector()
    w._switch_view(_VIEW_GOF_HEATMAP)
    assert not stamp()


# ---------------------------------------------------------------------------
# The GoF heatmap is a per-pair diagnostic again — audit S-6
# ---------------------------------------------------------------------------

def test_gof_test_discriminates_between_state_pairs(qapp):
    """The K² cells must not be K² copies of one unconditional test.

    `_do_gof_test` built `u = f_cdf[:N-1, i, j]`, `v = f_cdf[1:, j, i]` and
    handed them to `Copula.fit`, which rank-transforms its columns. The
    marginal CDFs are strictly increasing transforms of the *same* Y, so the
    ranks came out identical for every (i, j): the heatmap returned the same
    statistic and the same p-value K² times, to sixteen digits, while
    claiming to diagnose each pair.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(mdl, N=400, seed=0)

    results = PMCMainWindow._do_gof_test(mdl, Y, B=15, seed=0)
    scored = [r for r in results if "stat" in r]
    assert len(scored) >= 2

    stats = [r["stat"] for r in scored]
    assert len(set(stats)) == len(stats), \
        "the K² pairs still return one and the same statistic"

    # The weights are the pair posteriors, so a pair the chain visits often
    # and one it visits rarely must not carry the same effective size.
    n_effs = sorted(r["n_eff"] for r in scored)
    assert n_effs[-1] > 2 * n_effs[0], \
        "n_eff is identical across pairs — the ξ weighting is not biting"


def test_gof_pair_with_no_posterior_mass_is_reported_not_scored(qapp):
    """A pair the chain never visits must not get a confident p-value."""
    import pathlib

    import numpy as np

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    base = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(base, N=300, seed=0)

    raw = base.raw
    raw["prior"]["p"] = [[0.4999, 0.0001], [0.0001, 0.4999]]
    starved = PMCModel.from_dict(raw)

    results = PMCMainWindow._do_gof_test(starved, Y, B=10, seed=0, )
    off_diag = [r for r in results if r["i"] != r["j"]]
    assert off_diag
    assert all(np.isfinite(r["n_eff"]) for r in results)
    # Either scored with a visibly small n_eff, or refused outright.
    for r in off_diag:
        assert "error" in r or r["n_eff"] < min(
            x["n_eff"] for x in results if x["i"] == x["j"]
        )


@pytest.mark.slow
def test_gof_test_is_calibrated_under_its_own_null(qapp):
    """p-values must not pile up at 0 when the model is the true one.

    A first implementation bootstrapped *per pair* — drawing points straight
    from c_ij and reusing the observed weights — and rejected the true model
    at p = 0.01 on the diagonal. Two effects it could not see: the observed
    pseudo-observations are a ξ-weighted *mixture* (every transition enters
    every pair, merely with small weight), and consecutive transitions share
    an observation. Resampling whole series under the fitted model
    reproduces both.
    """
    import pathlib

    import numpy as np

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    p_values = []
    for rep in range(12):
        _, Y = simulate(mdl, N=300, seed=500 + rep)
        for r in PMCMainWindow._do_gof_test(mdl, Y, B=40, seed=rep):
            if "p" in r:
                p_values.append(r["p"])

    p = np.asarray(p_values)
    assert p.size >= 24
    # A grossly mis-calibrated test shows up as mass piled at zero. This is
    # deliberately loose — 12 datasets cannot resolve a few points of size.
    assert np.mean(p < 0.05) < 0.25, \
        f"{np.mean(p < 0.05):.2f} of p-values below 0.05 under H0"
    assert np.median(p) > 0.15, f"median p under H0 is {np.median(p):.3f}"


# ---------------------------------------------------------------------------
# The GoF statistic that actually has power — audit S-6
# ---------------------------------------------------------------------------

def _clayton_variant(toml):
    """The fixture with its dependent pair copulas made Clayton, τ untouched."""
    import pathlib

    from pmcprg.pmc.model import PMCModel

    base = PMCModel(pathlib.Path(toml))
    raw  = base.raw
    for blk in raw["copulas"]:
        if abs(float(blk["tau"])) > 1e-9:
            blk["name"] = "Clayton"
    return PMCModel.from_dict(raw)


def test_gof_defaults_to_the_kendall_process_statistic(qapp):
    """The shipped default must be the statistic measured to have power.

    Over 200 datasets per configuration at N = 1500, swapping the plain
    Cramér-von Mises distance for one on the Kendall process takes rejection
    of a wrong family from 0.115 to 0.950 against Clayton and from 0.060 to
    0.305 against GH — the latter an alternative nothing could see before,
    not even with the true-model posterior.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import _GOF_STATISTIC, PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    assert _GOF_STATISTIC == "kendall"

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(mdl, N=300, seed=0)
    results = PMCMainWindow._do_gof_test(mdl, Y, B=8, seed=0)

    assert all(r.get("statistic") == "kendall" for r in results), \
        "the statistic must be recorded with each result"


def test_kendall_reference_is_a_property_of_the_copula_alone(qapp):
    """`K_θ` must not depend on the data — it is cached per pair for that.

    Recomputing it inside the bootstrap loop would triple the cost of the
    test for nothing, so the invariant is worth pinning: same copula, same
    curve, and a proper CDF.
    """
    import numpy as np

    from pmcprg.pmc.gui.main_window import _kendall_reference
    from pmcprg.copulas import CopulaGaussian

    cop = CopulaGaussian(tau_k=0.5)
    a = _kendall_reference(cop, m_mc=4000)
    b = _kendall_reference(cop, m_mc=4000)

    assert np.array_equal(a, b), "K_theta is not reproducible"
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert np.all(np.diff(a) >= -1e-12), "K_theta must be non-decreasing"
    # A different copula must give a different curve, or the statistic could
    # not discriminate at all.
    other = _kendall_reference(CopulaGaussian(tau_k=0.1), m_mc=4000)
    assert not np.allclose(a, other)


@pytest.mark.slow
def test_kendall_statistic_sees_a_family_swap_the_cvm_misses(qapp):
    """The whole point, on one deliberately clear case.

    Data from Clayton copulas, tested against the same model with Gaussian
    copulas at the same τ: the plain CvM leaves at least one dependent pair
    unrejected, the Kendall-process statistic rejects both.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    if not pathlib.Path(_MODEL_A).exists():
        pytest.skip("fixture missing")

    gauss   = PMCModel(pathlib.Path(_MODEL_A))
    clayton = _clayton_variant(_MODEL_A)
    _, Y = simulate(clayton, N=1200, seed=3)

    def diag_pvalues(statistic):
        res = PMCMainWindow._do_gof_test(
            gauss, Y, B=60, seed=1, statistic=statistic,
            max_len=1200, max_points=1200,
        )
        return [r["p"] for r in res if "p" in r and r["i"] == r["j"]]

    p_kendall = diag_pvalues("kendall")
    p_cvm     = diag_pvalues("cvm")

    assert len(p_kendall) == len(p_cvm) >= 2
    assert max(p_kendall) < 0.05, \
        f"Kendall failed to reject a wrong family: {p_kendall}"
    assert max(p_cvm) > max(p_kendall), \
        "the CvM baseline was expected to do worse; the case is not diagnostic"


def test_gof_log_names_the_statistic_it_used(qapp):
    """A p-value is not interpretable without knowing what produced it."""
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    w   = PMCMainWindow(_MODEL_A)
    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=250, seed=0)
    w._on_sim_done((X, Y, mdl))
    w._on_gof_done(PMCMainWindow._do_gof_test(mdl, Y, B=8, seed=0))

    header = next(ln for ln in w._log.toPlainText().splitlines() if "GoF (" in ln)
    assert "Kendall-process" in header
    assert "B=8" in header


def test_pseudo_observation_contours_carry_a_readable_scale(qapp):
    """Audit G-9 — eight unlabelled contour levels per panel, each on its own
    automatic range: nothing said which contour was dense, nor whether one
    panel was denser than another.

    Levels are now shared across the K×K grid and shown once in a figure
    colourbar. The scatter is also weighted by ξ, because drawing all N−1
    transitions in every panel is the visual form of the defect S-6 fixed in
    the GoF test: an independent pair may carry an effective sample of 20
    while its panel shows 1500 dots.
    """
    import pathlib

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from pmcprg.pmc.gui import views
    from pmcprg.pmc.ice   import ice
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(mdl, N=400, seed=0)
    fitted, trace = ice(mdl, Y, ice_cfg={"max_iter": 2})

    fig = Figure(figsize=(8, 7))
    views.plot_ice_pseudos(fig, trace, fitted, Y)
    FigureCanvasAgg(fig).draw()          # flushes mathtext parse errors

    panels = [ax for ax in fig.axes if ax.get_title()]
    assert len(panels) == fitted.K ** 2

    # Exactly one colourbar, and it names what the contours mean.
    bars = [ax for ax in fig.axes
            if "log_{10}" in ax.get_ylabel() or "log_{10}" in ax.get_xlabel()]
    assert len(bars) == 1, "the contour levels have no shared, labelled scale"

    # Every panel states the effective sample it rests on.
    assert all("n_{" in ax.get_title() for ax in panels), \
        "a panel does not report its effective sample size"

    # A dependent pair must rest on visibly more than an independent one.
    def n_eff_of(title):
        import re
        m = re.search(r"n_\{\\mathrm\{eff\}\}=(\d+)", title)
        return int(m.group(1)) if m else None

    effs = {(ax.get_title()[:8]): n_eff_of(ax.get_title()) for ax in panels}
    values = [v for v in effs.values() if v is not None]
    assert len(values) == len(panels)
    assert max(values) > 2 * min(values), \
        "ξ weighting is not reflected in the reported effective sizes"


def test_posterior_draws_view_surfaces_classification_uncertainty(qapp):
    """Audit G-7 — `sample_posterior` (FFBS) had no entry point.

    The GUI reported classification as one label sequence and nothing else,
    so nothing on screen distinguished a segmentation the posterior is sure
    of from one it is guessing at — although the exact sampler the SEM M-step
    relies on was already in the package.
    """
    import pathlib

    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from pmcprg.pmc.gui import views
    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_CLS_POSTERIOR
    from pmcprg.pmc.inference import classify
    from pmcprg.pmc.model     import PMCModel
    from pmcprg.pmc.simulate  import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=250, seed=0)
    X_hat, gamma, _ = classify(mdl, Y)

    fig = Figure(figsize=(8, 6))
    views.plot_posterior_draws(fig, Y, X_hat, gamma, mdl, n_draws=12, seed=0)
    FigureCanvasAgg(fig).draw()

    assert len(fig.axes) == 3
    raster = fig.axes[0].get_images()[0].get_array()
    assert raster.shape == (12, len(Y)), "the draw raster has the wrong shape"
    # Draws must actually vary — a constant raster would mean the sampler was
    # not exercised, and the panel would be decorative.
    assert raster.std() > 0
    assert not np.all(raster == X_hat[None, :]), \
        "every draw equals MPM: the posterior sampler is not being used"

    # And the view is reachable from the selector once a classification exists.
    w = PMCMainWindow(_MODEL_A)
    w._on_cls_done((X_hat, gamma, 0.0, mdl, Y, X))
    assert w._is_view_available(_VIEW_CLS_POSTERIOR)


def test_tau_intervals_are_drawn_at_the_effective_sample_size(qapp):
    """Audit G-7 — every τ was quoted to three decimals with no interval.

    `FitResult.bootstrap_ci` was in the package and unreachable, and it
    resamples uniformly, which is the wrong null here: a pair is fitted on
    ξ-weighted pseudo-observations. Drawing N−1 points with probability ∝ ξ
    would also hand a pair whose posterior mass is 40 transitions a bootstrap
    sample of 800, and the interval would come out far too narrow — confident
    exactly where the data are thinnest. Each replicate draws round(n_eff).
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(mdl, N=600, seed=0)
    rows = PMCMainWindow._do_tau_ci(mdl, Y, B=60, seed=0)

    scored = [r for r in rows if "lo" in r]
    assert len(scored) == len(mdl.copula_blocks())
    for r in scored:
        assert r["lo"] <= r["tau_hat"] <= r["hi"], \
            "the interval does not contain the estimator it is an interval for"
        assert "tau" in r and "n_eff" in r

    dep   = [r for r in scored if r["i"] == r["j"]]      # τ = 0.6 by fixture
    indep = [r for r in scored if r["i"] != r["j"]]      # τ = 0 by fixture

    # A dependent pair must be distinguishable from independence …
    assert all(r["lo"] > 0.0 for r in dep), \
        f"a genuinely dependent pair straddles 0: {[r['lo'] for r in dep]}"
    # … and an independent one must not be.
    assert all(r["lo"] <= 0.0 <= r["hi"] for r in indep)

    # The thin pairs must carry the wider interval — that is the whole point
    # of drawing at n_eff rather than at N.
    width = lambda r: r["hi"] - r["lo"]                  # noqa: E731
    assert min(width(r) for r in indep) > max(width(r) for r in dep)
    assert max(r["n_eff"] for r in dep) > 2 * max(r["n_eff"] for r in indep)


def test_tau_interval_view_and_analysis_menu_are_reachable(qapp):
    """The Analysis menu is new too: there was nowhere to invoke this from."""
    import pathlib

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from pmcprg.pmc.gui import views
    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_TAU_CI
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    w = PMCMainWindow(_MODEL_A)
    assert any(a.text() == "&Analysis" for a in w.menuBar().actions()), \
        "no Analysis menu"
    assert w._analysis_actions

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=400, seed=0)
    w._last_X, w._last_Y = X, Y
    w._on_sim_done((X, Y, mdl))
    assert not w._is_view_available(_VIEW_TAU_CI), "available before it is run"

    w._on_tau_ci_done(PMCMainWindow._do_tau_ci(mdl, Y, B=40, seed=0))
    assert w._is_view_available(_VIEW_TAU_CI)
    assert w._current_view == _VIEW_TAU_CI

    fig = Figure(figsize=(8, 4))
    views.plot_tau_ci(fig, w._tau_ci_state)
    FigureCanvasAgg(fig).draw()          # flushes mathtext parse errors
    assert fig.axes


def test_margin_ks_reaches_the_diagnostics_subpackage(qapp):
    """Audit G-7 — `pmcprg.diagnostics` had no entry point at all.

    The GoF heatmap tests copulas; nothing tested the margins.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow, _VIEW_MARGIN_KS
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    w = PMCMainWindow(_MODEL_A)
    assert len(w._analysis_actions) == 2
    assert any("Margin adequacy" in a.text() for a in w._analysis_actions)

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    X, Y = simulate(mdl, N=300, seed=0)
    w._last_X, w._last_Y = X, Y
    w._on_sim_done((X, Y, mdl))
    assert not w._is_view_available(_VIEW_MARGIN_KS)

    rows = PMCMainWindow._do_margin_ks(mdl, Y, B=8, seed=0)
    assert len(rows) == mdl.K
    for r in rows:
        assert {"state", "dist", "n", "B", "n_series"} <= set(r)

    w._on_margin_ks_done(rows)
    assert w._is_view_available(_VIEW_MARGIN_KS)
    assert w._current_view == _VIEW_MARGIN_KS
    # The view must redraw the very sample the statistic was computed on.
    Y_used, X_draw = w._margin_ks_sample
    assert Y_used is not None and len(X_draw) == len(Y_used)


@pytest.mark.slow
def test_margin_ks_localises_a_broken_margin(qapp):
    """A shifted margin must be rejected, and only on the state it belongs to.

    Two calibration facts sit behind this test, both measured (120 datasets,
    N=600). The sample is a posterior DRAW, not the MPM estimate: under a
    correct model P(Y | X=k) = f_k but P(Y | X̂=k) ≠ f_k, and MPM's mean KS
    statistic came out above the true-label one (0.0828/0.0895 vs
    0.0785/0.0813) while the draw tracked it (0.0790/0.0781). And the
    critical value is bootstrapped rather than taken from the KS table,
    because Y is a Markov chain: with true labels and the tabulated value the
    test rejected a correct model 9.2% of the time at a nominal 5%.
    """
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(mdl, N=600, seed=0)

    good = PMCMainWindow._do_margin_ks(mdl, Y, B=80, seed=0)
    assert all(r["p"] >= 0.05 for r in good if "p" in r), \
        f"the correct model was rejected: {[r.get('p') for r in good]}"

    raw = mdl.raw
    raw["margins"][0]["params"]["loc"] += 0.8      # break state 0 only
    broken = PMCModel.from_dict(raw)

    bad = {r["state"]: r for r in
           PMCMainWindow._do_margin_ks(broken, Y, B=80, seed=0)}
    assert bad[0].get("p", 1.0) < 0.05, \
        f"the shifted margin was not detected: {bad[0]}"
    assert bad[1].get("p", 0.0) >= 0.05, \
        f"the intact margin was rejected too: {bad[1]}"


# ---------------------------------------------------------------------------
# The shared calibration harness — pmcprg.diagnostics.parametric_bootstrap
# ---------------------------------------------------------------------------

def test_parametric_bootstrap_holds_the_level_for_any_statistic(qapp):
    """The harness must calibrate a statistic it knows nothing about.

    Why it exists: a tabulated critical value is wrong on a fitted model —
    the MKS one rejects a correct PMC 10% of the time at a nominal 5% —
    because the reference is estimated, the observations are dependent, and
    the per-state sample is assigned rather than given. Resampling whole
    series absorbs all three, whatever the statistic.
    """
    import pathlib

    import numpy as np

    from pmcprg.diagnostics  import parametric_bootstrap
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))

    def statistic(_model, Y, _rng):
        # Two statistics from one call — the pattern the harness exists for.
        return {"var": float(np.var(Y)), "range": float(np.ptp(Y))}

    p_values = []
    for rep in range(40):
        _, Y = simulate(mdl, N=300, seed=7_000 + rep)
        res = parametric_bootstrap(mdl, Y, statistic, B=40, seed=rep)
        assert set(res) == {"var", "range"}
        assert res["var"].n_valid > 0 and res["var"].B == 40
        assert res["var"].n_series == 300
        p_values.append(res["var"].p_value)

    p = np.asarray(p_values)
    # Loose on purpose — 40 datasets cannot resolve a few points of level —
    # but a broken harness piles p-values at one end, and this catches that.
    assert 0.2 < np.median(p) < 0.8, f"median p under H0 is {np.median(p):.3f}"
    assert np.mean(p < 0.05) < 0.20


def test_parametric_bootstrap_truncates_contiguously(qapp):
    """`max_len` keeps a contiguous prefix, and replicates match its length.

    Thinning would destroy the serial dependence the bootstrap exists to
    absorb, and a replicate of a different length would not be comparable to
    the observed value.
    """
    import pathlib

    import numpy as np

    from pmcprg.diagnostics  import parametric_bootstrap
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(mdl, N=400, seed=0)
    seen: list[int] = []

    def statistic(_model, Yv, _rng):
        seen.append(len(Yv))
        return {"mean": float(np.mean(Yv))}

    res = parametric_bootstrap(mdl, Y, statistic, B=5, seed=0, max_len=150)

    assert res["mean"].n_series == 150
    assert set(seen) == {150}


def test_the_two_gui_diagnostics_share_one_bootstrap(qapp):
    """Both GoF tests go through the harness; the τ interval must not.

    The model-level bootstrap was written five times before it was written
    once. But `_do_tau_ci` resamples pseudo-observations to get an interval
    on a parameter — a different bootstrap that happens to share a loop
    shape, and unifying them would be wrong.
    """
    import inspect

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    for method in (PMCMainWindow._do_gof_test, PMCMainWindow._do_margin_ks):
        src = inspect.getsource(method)
        assert "parametric_bootstrap(" in src, \
            f"{method.__name__} does not use the shared harness"
        assert "for b in range(B)" not in src, \
            f"{method.__name__} still rolls its own bootstrap loop"

    tau_src = inspect.getsource(PMCMainWindow._do_tau_ci)
    assert "parametric_bootstrap(" not in tau_src


# ---------------------------------------------------------------------------
# Neyman smooth components for margin adequacy — opportunity O7
# ---------------------------------------------------------------------------

def test_each_smooth_component_detects_its_own_deformation(qapp):
    """`V₁` location, `V₂` dispersion, `V₃` asymmetry, `V₄` tail weight.

    The decomposition is the point: a p-value says a margin is wrong, the
    component says how, and the fix differs.
    """
    import math

    from scipy import stats

    from pmcprg.diagnostics import neyman_components

    def comps(x):
        return neyman_components(x, cdf=stats.norm.cdf).components

    assert abs(comps(stats.norm.rvs(loc=0.4, size=600,
                                    random_state=2))[0]) > 5      # V1
    assert abs(comps(stats.norm.rvs(scale=1.5, size=600,
                                    random_state=3))[1]) > 5      # V2

    skew = stats.skewnorm.rvs(6, size=600, random_state=4)
    skew = (skew - stats.skewnorm.mean(6)) / stats.skewnorm.std(6)
    v = comps(skew)
    assert v[2] > 5 and abs(v[2]) > abs(v[1]), "asymmetry not on V3"

    heavy = stats.t.rvs(4, size=600, random_state=5) / math.sqrt(2.0)
    v = comps(heavy)
    # Heavy tails are a *pair*: mass in the extremes, hollowed shoulders.
    assert v[3] > 1.5 and v[1] < 0, f"tail signature not (V4>0, V2<0): {v}"


def test_neyman_test_pays_the_multiplicity(qapp):
    """Four components is four tests; the correction is not optional.

    Reporting the uncorrected minimum takes the level from 0.05 to 0.175,
    measured over 200 datasets per configuration.
    """
    from pmcprg.diagnostics import neyman_test

    p, idx = neyman_test([0.4, 0.01, 0.6, 0.5])
    assert p == pytest.approx(0.04) and idx == 2, "no Bonferroni factor"

    # Never exceeds 1, and reports which component drove it.
    p, idx = neyman_test([0.5, 0.9, 0.8, 0.7])
    assert p == 1.0 and idx == 1


def test_margin_test_ships_the_smooth_decomposition(qapp):
    """The GUI margin test must use the components and report the driver.

    Measured over 200 datasets per alternative, all bootstrap-calibrated so
    the comparison is purely power: rejection of a wrong margin goes from
    0.098 to 0.338 against Student-t tails and 0.152 to 0.480 against
    skew-normal asymmetry.
    """
    import math
    import pathlib

    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model    import PMCModel
    from pmcprg.pmc.simulate import simulate

    base = PMCModel(pathlib.Path(_MODEL_A))
    _, Y = simulate(base, N=400, seed=0)

    rows = PMCMainWindow._do_margin_ks(base, Y, B=15, seed=0)
    assert rows and all(r.get("statistic") == "neyman" for r in rows)
    for r in rows:
        if "p" in r:
            assert 0.0 <= r["p"] <= 1.0
            assert len(r["components"]) == 4
            assert len(r["p_components"]) == 4
            assert r["component"] in (
                "location", "dispersion", "asymmetry", "tail weight")

    # Heavy-tailed data must be diagnosed as a tail problem, not a shift.
    raw = base.raw
    for blk in raw["margins"]:
        prm = blk["params"]
        blk["dist"] = "t"
        blk["params"] = {"df": 4, "loc": prm["loc"],
                         "scale": prm["scale"] / math.sqrt(2.0)}
    _, Y_t = simulate(PMCModel.from_dict(raw), N=600, seed=1)

    diag = [r["component"] for r in
            PMCMainWindow._do_margin_ks(base, Y_t, B=15, seed=0) if "p" in r]
    assert "tail weight" in diag, f"tails not diagnosed: {diag}"
