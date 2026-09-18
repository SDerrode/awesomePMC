"""
test_gui_pair_margins.py — the GUI on a general PMC with pair-indexed margins.

A model with ``margin_structure == "pair"`` carries K² margins f_ij, the law of
y_n given (x_n = i, x_{n+1} = j) (Derrode & Pieczynski 2013, CSDA 63:81–98 —
"DerrodePieczynski_CSDA2013" — Eqs. 12–14). ``PMCModel.margin(i)`` raises on
such a model, and every panel that used it, or assumed K margin blocks, broke.
These tests cover:

* the Margins tab — K×K cells labelled f_ij, editing, the state ↔ pair switch,
  and the unchanged K-cell grid of state models;
* a load → save through the GUI reproducing the TOML tables exactly;
* the diagnostics (margin adequacy, τ intervals, GoF) and every view, which
  use f_ij / f_ji for the pseudo-observations of a pair and, where they speak
  of the margin of state i, the mixture g_i = Σ_j (p_ij / p_i) f_ij.

Estimation on pair margins is being fixed separately (wave 2A, ``ice.py`` /
``sem.py``); the one test that needs a real ICE run skips while it fails.
The views are otherwise exercised on a stand-in trace built from two models.

Runs headless (``QT_QPA_PLATFORM=offscreen``).
"""

from __future__ import annotations

import copy
import pathlib
import tomllib

import numpy as np
import pytest

PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

PAIR_TOML  = "pmcprg/pmc/models/pmc_pair_gauss_k2.toml"
STATE_TOML = "pmcprg/pmc/models/pmc_gauss_k2.toml"
HMC_TOML   = "pmcprg/pmc/models/hmc_in_gauss_k2.toml"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _answer_questions(monkeypatch):
    """Answer every ``QMessageBox.question`` with Yes/Discard — a modal hangs."""
    def answer(*args, **kwargs):
        buttons = args[3] if len(args) > 3 else kwargs.get("buttons")
        if buttons is not None and (buttons & QMessageBox.StandardButton.Discard):
            return QMessageBox.StandardButton.Discard
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer))


@pytest.fixture
def no_modals(monkeypatch):
    seen: list[tuple] = []
    for name in ("critical", "warning", "information"):
        monkeypatch.setattr(
            QMessageBox, name,
            staticmethod(lambda *a, _n=name, **k: seen.append((_n, a[1:3]))),
        )
    return seen


def _grid(tab):
    """``(column headers, row headers, cell texts)`` of a block-grid tab."""
    t = tab._table
    cols = [t.horizontalHeaderItem(c).text() for c in range(t.columnCount())]
    rows = [t.verticalHeaderItem(r).text() for r in range(t.rowCount())]
    cells = [[t.item(r, c).text() for c in range(t.columnCount())]
             for r in range(t.rowCount())]
    return cols, rows, cells


def _structure_index(tab, value: str) -> int:
    return tab._combo_structure.findData(value)


class _FakeDialog:
    """Stands in for ``_MarginDialog``: accepts at once with a preset block."""

    result: dict = {}
    seen: list = []

    def __init__(self, blk, parent=None):
        _FakeDialog.seen.append(copy.deepcopy(blk))

    def exec(self):
        return True

    def get_block(self):
        return copy.deepcopy(_FakeDialog.result)


# ---------------------------------------------------------------------------
# Margins tab — layout
# ---------------------------------------------------------------------------

def test_pair_model_margins_tab_shows_k2_cells_labelled_f_ij(qapp):
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w   = PMCMainWindow(PAIR_TOML)
    tab = w._tab_margins
    cols, rows, cells = _grid(tab)

    assert tab.structure == "pair"
    assert (len(rows), len(cols)) == (2, 2)
    assert cols == ["j=0", "j=1"] and rows == ["i=0", "i=1"]
    # Row i, column j is f_ij — the fixture's four distinct densities.
    assert cells == [["norm(loc=0.0, scale=1.0)", "norm(loc=0.3, scale=1.6)"],
                     ["norm(loc=1.1, scale=1.4)", "norm(loc=1.5, scale=1.0)"]]
    info = tab._info.text()
    assert "f_ij" in info and "x_{n+1} = j" in info and "DerrodePieczynski_CSDA2013" in info
    assert "SR-PMC" not in info
    assert tab._combo_structure.currentData() == "pair"
    assert tab._combo_structure.isEnabled()


def test_state_model_margins_tab_is_unchanged(qapp):
    """K cells, one per state, same headers and texts as before wave 2B."""
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w   = PMCMainWindow(STATE_TOML)
    tab = w._tab_margins
    cols, rows, cells = _grid(tab)

    assert tab.structure == "state"
    assert cols == ["density"] and rows == ["i=0", "i=1"]
    assert cells == [["norm(loc=-1.0, scale=1.0)"], ["norm(loc=1.0, scale=1.0)"]]
    assert "f_i" in tab._info.text() and "SR-PMC" not in tab._info.text()
    # A PMC may switch to pair margins; an HMC may not (DerrodePieczynski_CSDA2013
    # §2.1 Proposition).
    assert tab._combo_structure.isEnabled()
    w_hmc = PMCMainWindow(HMC_TOML)
    assert not w_hmc._tab_margins._combo_structure.isEnabled()
    # The rebuilt state model keeps its K-format blocks.
    mdl = w._rebuild_model_from_widgets()
    assert mdl.margin_structure == "state"
    assert mdl.margin_blocks() == w._model.margin_blocks()


def test_hmc_margins_tab_refuses_pair_structure_even_programmatically(qapp):
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w   = PMCMainWindow(HMC_TOML)
    tab = w._tab_margins
    tab._combo_structure.setCurrentIndex(_structure_index(tab, "pair"))
    assert tab.structure == "state"
    assert tab._combo_structure.currentData() == "state"
    assert w._rebuild_model_from_widgets().margin_structure == "state"


# ---------------------------------------------------------------------------
# Margins tab — editing
# ---------------------------------------------------------------------------

def test_editing_a_pair_cell_updates_that_f_ij_only(qapp, monkeypatch):
    from pmcprg.pmc.gui import tabs as tabs_mod
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w   = PMCMainWindow(PAIR_TOML)
    tab = w._tab_margins
    monkeypatch.setattr(tabs_mod._MarginTab, "_DIALOG", _FakeDialog)
    _FakeDialog.seen = []
    _FakeDialog.result = {"dist": "norm", "params": {"loc": -2.5, "scale": 0.7}}

    before = w._model
    tab._on_cell_dbl(1, 0)                                   # row i=1, column j=0

    assert _FakeDialog.seen[0]["i"] == 1 and _FakeDialog.seen[0]["j"] == 0
    assert _grid(tab)[2][1][0] == "norm(loc=-2.5, scale=0.7)"
    assert w._dirty
    mdl = w._rebuild_model_from_widgets()
    assert mdl.margin_structure == "pair"
    assert mdl.margin(1, 0).params == {"loc": -2.5, "scale": 0.7}
    for (i, j) in [(0, 0), (0, 1), (1, 1)]:
        assert mdl.margin(i, j).params == before.margin(i, j).params


def test_clearing_gice_candidates_in_the_dialog_removes_them(qapp, monkeypatch):
    """The dialog omits ``candidates`` when none is ticked — the key must go."""
    from pmcprg.pmc.gui import tabs as tabs_mod
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w   = PMCMainWindow(PAIR_TOML)
    tab = w._tab_margins
    tab._blocks[1]["candidates"] = ["norm", "gamma"]          # block (0, 1)
    monkeypatch.setattr(tabs_mod._MarginTab, "_DIALOG", _FakeDialog)
    _FakeDialog.result = {"dist": "norm", "params": {"loc": 0.3, "scale": 1.6}}
    tab._on_cell_dbl(0, 1)
    assert "candidates" not in tab._blocks[1]


def test_margin_dialog_title_names_the_margin(qapp):
    from pmcprg.pmc.gui.dialogs import _MarginDialog

    pair  = _MarginDialog({"i": 0, "j": 1, "dist": "norm", "params": {}})
    state = _MarginDialog({"i": 1, "dist": "norm", "params": {}})
    assert "f_ij" in pair.windowTitle() and "i=0, j=1" in pair.windowTitle()
    assert "f_i " in state.windowTitle() and "i=1" in state.windowTitle()


# ---------------------------------------------------------------------------
# Margins tab — switching the structure
# ---------------------------------------------------------------------------

def test_switch_state_to_pair_ties_f_ij_to_f_i_and_keeps_the_model(
        qapp, monkeypatch):
    """state → pair copies f_i into every f_ij: the same law of (X, Y)."""
    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.inference import classify
    from pmcprg.pmc.simulate import simulate

    w   = PMCMainWindow(STATE_TOML)
    tab = w._tab_margins
    state_mdl = w._rebuild_model_from_widgets()

    tab._combo_structure.setCurrentIndex(_structure_index(tab, "pair"))
    assert tab.structure == "pair" and w._dirty
    cols, rows, cells = _grid(tab)
    assert cols == ["j=0", "j=1"] and rows == ["i=0", "i=1"]
    assert cells[0] == ["norm(loc=-1.0, scale=1.0)"] * 2
    assert cells[1] == ["norm(loc=1.0, scale=1.0)"] * 2

    pair_mdl = w._rebuild_model_from_widgets()
    assert pair_mdl.margin_structure == "pair"
    assert len(pair_mdl.margin_blocks()) == 4
    assert "margin_structure" not in pair_mdl.raw["model"]   # format suffices
    _, Y = simulate(state_mdl, N=150, seed=0)
    _, g_state, ll_state = classify(state_mdl, Y)
    _, g_pair,  ll_pair  = classify(pair_mdl, Y)
    assert ll_pair == pytest.approx(ll_state, rel=1e-10, abs=1e-8)
    assert np.allclose(g_pair, g_state, atol=1e-10)

    # … and back: tied blocks collapse without a question.
    asked: list = []
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: asked.append(a)
                     or QMessageBox.StandardButton.Yes),
    )
    tab._combo_structure.setCurrentIndex(_structure_index(tab, "state"))
    assert not asked
    assert tab.structure == "state"
    back = w._rebuild_model_from_widgets()
    assert back.margin_structure == "state"
    assert back.margin_blocks() == state_mdl.margin_blocks()


def test_switch_pair_to_state_asks_before_discarding_the_j_dependence(
        qapp, monkeypatch):
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w   = PMCMainWindow(PAIR_TOML)
    tab = w._tab_margins
    answers: list = []

    def refuse(*args, **kwargs):
        answers.append(args[2])
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(refuse))
    tab._combo_structure.setCurrentIndex(_structure_index(tab, "state"))
    assert answers and "[0, 1]" in answers[0]            # both states untied
    assert tab.structure == "pair"
    assert tab._combo_structure.currentData() == "pair"   # combo reverted
    assert not w._dirty

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    tab._combo_structure.setCurrentIndex(_structure_index(tab, "state"))
    assert tab.structure == "state"
    mdl = w._rebuild_model_from_widgets()
    assert mdl.margin_structure == "state"
    # The anchor f_i0 is kept — the same rule as [model].margin_structure="state".
    assert mdl.margin(0).params == {"loc": 0.0, "scale": 1.0}
    assert mdl.margin(1).params == {"loc": 1.1, "scale": 1.4}


def test_an_explicit_margin_structure_key_follows_the_switch(qapp, tmp_path):
    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model import PMCModel

    src = tmp_path / "explicit.toml"
    src.write_text(
        pathlib.Path(STATE_TOML).read_text().replace(
            'variant = "PMC"', 'variant = "PMC"\nmargin_structure = "state"')
    )
    assert PMCModel(src).raw["model"]["margin_structure"] == "state"
    w   = PMCMainWindow(str(src))
    tab = w._tab_margins
    tab._combo_structure.setCurrentIndex(_structure_index(tab, "pair"))
    out = tmp_path / "out.toml"
    w._on_save(out)
    saved = PMCModel(out)
    assert saved.raw["model"]["margin_structure"] == "pair"
    assert saved.margin_structure == "pair"


# ---------------------------------------------------------------------------
# Load / save round trip
# ---------------------------------------------------------------------------

def test_pair_model_gui_save_reproduces_the_toml_tables(qapp, tmp_path):
    """Load → Save leaves [model], [prior], [[margins]], [[copulas]] as they were.

    The GUI adds its own [ice]/[sem]/[gui] sections on save (tested
    elsewhere); everything the model is made of comes back identical, and a
    second generation is byte-identical to the first.
    """
    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model import PMCModel

    src = tomllib.loads(pathlib.Path(PAIR_TOML).read_text())
    out1 = tmp_path / "gen1.toml"
    PMCMainWindow(PAIR_TOML)._on_save(out1)
    gen1 = tomllib.loads(out1.read_text())

    for key in ("model", "prior", "margins", "copulas"):
        assert gen1[key] == src[key], key
    assert "margin_structure" not in gen1["model"]
    mdl = PMCModel(out1)
    assert mdl.margin_structure == "pair"
    assert mdl.margin_blocks() == PMCModel(PAIR_TOML).margin_blocks()

    out2 = tmp_path / "gen2.toml"
    PMCMainWindow(str(out1))._on_save(out2)
    assert out2.read_bytes() == out1.read_bytes()


def test_edited_pair_model_survives_a_disk_round_trip(qapp, tmp_path, monkeypatch):
    from pmcprg.pmc.gui import tabs as tabs_mod
    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow(PAIR_TOML)
    monkeypatch.setattr(tabs_mod._MarginTab, "_DIALOG", _FakeDialog)
    _FakeDialog.result = {"dist": "gamma",
                          "params": {"a": 2.0, "loc": 0.0, "scale": 0.5}}
    w._tab_margins._on_cell_dbl(0, 1)
    out = tmp_path / "edited.toml"
    w._on_save(out)

    back = PMCMainWindow(str(out))
    assert back._tab_margins.structure == "pair"
    assert _grid(back._tab_margins)[2][0][1] == "gamma(a=2.0, loc=0.0, scale=0.5)"
    assert back._model.margin(0, 1).dist_name == "gamma"


# ---------------------------------------------------------------------------
# Diagnostics on a pair model
# ---------------------------------------------------------------------------

def test_state_law_is_the_mixture_over_the_next_state(qapp):
    """g_i(y) = Σ_j (p_ij / p_i) f_ij(y) — DerrodePieczynski_CSDA2013 Eq. 12
    summed over x_{n+1}."""
    from pmcprg.pmc.gui._margins import state_law, state_law_name
    from pmcprg.pmc.model import PMCModel

    mdl = PMCModel(PAIR_TOML)
    p   = mdl.prior_p
    y   = np.linspace(-3.0, 4.0, 41)
    for i in range(2):
        want_cdf = sum(p[i, j] / p[i].sum() * mdl.margin(i, j).cdf_vec(y)
                       for j in range(2))
        want_pdf = sum(p[i, j] / p[i].sum() * mdl.margin(i, j).pdf_vec(y)
                       for j in range(2))
        law = state_law(mdl, i)
        assert np.allclose(law.cdf_vec(y), want_cdf, atol=1e-14)
        assert np.allclose(law.pdf_vec(y), want_pdf, atol=1e-14)
        assert state_law_name(mdl, i) == "norm mix"

    state = PMCModel(STATE_TOML)
    assert state_law(state, 0) is state.margin(0)
    assert state_law_name(state, 1) == "norm"


def test_pair_pseudo_observations_use_f_ij_left_and_f_ji_right(qapp):
    from pmcprg.pmc.gui._margins import margin_cdfs
    from pmcprg.pmc.gui.views import compute_marginal_cdfs
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(PAIR_TOML)
    _, Y = simulate(mdl, N=60, seed=1)
    F = margin_cdfs(mdl, Y, lo=1e-12, hi=1 - 1e-12)
    for i in range(2):
        for j in range(2):
            assert np.array_equal(
                F[:, i, j], np.clip(mdl.margin(i, j).cdf_vec(Y), 1e-12, 1 - 1e-12))
    assert not np.allclose(F[:, 0, 1], F[:, 0, 0])
    assert compute_marginal_cdfs(mdl, Y).shape == (60, 2, 2)

    state = PMCModel(STATE_TOML)
    Fs = margin_cdfs(state, Y, lo=1e-12, hi=1 - 1e-12)
    for i in range(2):
        for j in range(2):
            assert np.array_equal(
                Fs[:, i, j], np.clip(state.margin(i).cdf_vec(Y), 1e-12, 1 - 1e-12))


@pytest.mark.parametrize("toml", [PAIR_TOML, STATE_TOML])
def test_gui_margin_helpers_agree_with_the_diagnostics_helpers(qapp, toml):
    """The GUI's margin helpers are those of pmcprg.diagnostics.pseudo, bit for bit."""
    from pmcprg.diagnostics import pseudo
    from pmcprg.numerics import EPS
    from pmcprg.pmc.gui import _margins
    from pmcprg.pmc.gui.views import compute_marginal_cdfs
    from pmcprg.pmc.inference import (
        backward, forward, joint_posteriors, precompute_weights,
    )
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(toml)
    _, Y = simulate(mdl, N=80, seed=2)
    for name in ("StateMixture", "is_pair", "mixture_label", "state_law",
                 "state_law_name", "state_weights"):
        assert getattr(_margins, name) is getattr(pseudo, name)

    F_gui = _margins.margin_cdfs(mdl, Y, lo=1e-12, hi=1.0 - 1e-12)
    F_dia = pseudo.margin_cdfs(mdl, Y)
    assert F_gui.shape == F_dia.shape == (80, 2, 2)
    assert np.ascontiguousarray(F_gui).tobytes() == F_dia.tobytes()
    # A state model keeps the GUI's read-only broadcast view; the diagnostics
    # (and every pair model) get a writable array.
    assert F_gui.flags.writeable == _margins.is_pair(mdl)
    assert F_dia.flags.writeable
    assert compute_marginal_cdfs(mdl, Y).tobytes() == pseudo.margin_cdfs(mdl, Y, clip=EPS).tobytes()

    W, f_pdf = precompute_weights(mdl, Y)
    alpha, _ = forward(mdl, Y, W=W, f_pdf=f_pdf)
    xi = joint_posteriors(alpha, W, backward(mdl, Y, W=W))
    for i in range(2):
        for j in range(2):
            uv_gui, w_gui = pseudo.copula_pseudo_obs(F_gui, xi, i, j)
            uv_dia, w_dia = pseudo.copula_pseudo_obs(F_dia, xi, i, j)
            assert uv_gui.tobytes() == uv_dia.tobytes()
            assert w_gui.tobytes() == w_dia.tobytes()
    y = np.linspace(-3.0, 4.0, 29)
    for i in range(2):
        law = _margins.state_law(mdl, i)
        want = sum(w * mdl.margin(i, j).cdf_vec(y)
                   for j, w in enumerate(pseudo.state_weights(mdl, i)))
        assert np.allclose(law.cdf_vec(y), want, rtol=0, atol=1e-14)


def test_diagnostics_run_on_a_pair_model(qapp, no_modals):
    from pmcprg.pmc.gui.main_window import (
        PMCMainWindow, _VIEW_GOF_HEATMAP, _VIEW_MARGIN_KS, _VIEW_TAU_CI,
    )
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(PAIR_TOML)
    X, Y = simulate(mdl, N=300, seed=0)

    rows = PMCMainWindow._do_margin_ks(mdl, Y, B=6, seed=0)
    assert [r["state"] for r in rows] == [0, 1]
    assert all(r["dist"] == "norm mix" and "p" in r for r in rows)

    ci = PMCMainWindow._do_tau_ci(mdl, Y, B=15, seed=0)
    assert [(r["i"], r["j"]) for r in ci] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all("lo" in r and r["lo"] <= r["hi"] for r in ci)

    gof = PMCMainWindow._do_gof_test(mdl, Y, B=5, seed=0)
    assert len(gof) == 4 and all(np.isfinite(r["stat"]) for r in gof)

    w = PMCMainWindow(PAIR_TOML)
    w._on_sim_done((X, Y, mdl))
    w._on_margin_ks_done(rows)
    assert w._current_view == _VIEW_MARGIN_KS and w._has_plot
    w._on_tau_ci_done(ci)
    assert w._current_view == _VIEW_TAU_CI and w._has_plot
    w._on_gof_done(gof)
    assert w._current_view == _VIEW_GOF_HEATMAP and w._has_plot
    assert not [s for s in no_modals if s[0] == "critical"]


def _stand_in_estimation(mdl, Y):
    """An ``IceResult`` for ``mdl`` without running ICE (wave 2A pending).

    The "fitted" model moves every f_ij and τ a little; the trace holds three
    snapshots going from the initial model to it, in the format of
    :class:`pmcprg.pmc.ice.IceTrace`.
    """
    from pmcprg.pmc.ice import IceResult, IceTrace
    from pmcprg.pmc.model import PMCModel

    raw = mdl.raw
    for blk in raw["margins"]:
        blk["params"]["loc"] = float(blk["params"]["loc"]) + 0.1
    raw["margins"][1]["dist"] = "t"
    raw["margins"][1]["params"] = {"df": 5.0, "loc": 0.4, "scale": 1.5}
    for blk in raw["copulas"]:
        blk["tau"] = 0.6
    raw["prior"]["p"] = [[0.49, 0.06], [0.06, 0.39]]
    fitted = PMCModel.from_dict(raw)
    K = mdl.K

    def snap(m):
        tau = np.array([[m.copula(i, j).params["tau_k"] for j in range(K)]
                        for i in range(K)])
        fam = [[m.copula(i, j).copula_enum.value.SHORT_NAME for j in range(K)]
               for i in range(K)]
        return tau, fam, m.prior_p, m.margin_blocks()

    snaps = [snap(mdl), snap(mdl), snap(fitted)]
    trace = IceTrace(
        log_liks=[-500.0, -480.0, -475.0],
        tau_history=np.stack([s[0] for s in snaps]),
        family_history=[s[1] for s in snaps],
        p_history=np.stack([s[2] for s in snaps]),
        margin_history=[s[3] for s in snaps],
        candidates=["Clayton", "Gauss"],
    )
    return IceResult(initial_model=mdl, fitted_model=fitted, Y=Y, trace=trace)


def test_every_view_renders_on_a_pair_model(qapp, monkeypatch, no_modals):
    """Every available view, on a pair model, with a stand-in ICE result."""
    from pmcprg.pmc.gui.main_window import (
        PMCMainWindow, _ALL_VIEWS, _ICE_VIEW_MARGIN_FAMILY,
        _ICE_VIEW_PARAM_COMPARE,
    )
    from pmcprg.pmc.inference import classify
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(PAIR_TOML)
    X, Y = simulate(mdl, N=150, seed=0)
    w = PMCMainWindow(PAIR_TOML)
    w._on_sim_done((X, Y, mdl))
    result = _stand_in_estimation(mdl, Y)
    w._on_est_done(result)
    # The editor now shows the "fitted" pair model.
    assert w._tab_margins.structure == "pair"
    assert _grid(w._tab_margins)[2][0][1].startswith("t(")

    X_hat, gamma, _ = classify(result.fitted_model, Y)
    w._sim_state = (X, Y, mdl)
    w._cls_state = (Y, X_hat, gamma, X, mdl)
    w._gof_state = PMCMainWindow._do_gof_test(mdl, Y, B=4)
    w._tau_ci_state = PMCMainWindow._do_tau_ci(mdl, Y, B=12)
    rows = PMCMainWindow._do_margin_ks(mdl, Y, B=4)
    w._on_margin_ks_done(rows)

    failures: list = []
    monkeypatch.setattr(
        PMCMainWindow, "_render_error",
        lambda self, name, exc: failures.append((name, repr(exc))),
    )
    rendered = [v for v in _ALL_VIEWS if w._is_view_available(v)]
    for view in rendered:
        w._switch_view(view)
    assert not failures, f"render failures: {failures}"
    assert len(rendered) >= 18, rendered

    # Parameter comparison: one margin row per pair, labelled (i,j).
    w._switch_view(_ICE_VIEW_PARAM_COMPARE)
    texts = {c.get_text().get_text()
             for ax in w._canvas.fig.axes for tbl in ax.tables
             for c in tbl.get_celld().values()}
    assert {"(i,j)", "(0,0)", "(0,1)", "(1,0)", "(1,1)"} <= texts
    assert "t" in texts                                   # the family change

    # Margin family ribbon: one row per f_ij.
    w._switch_view(_ICE_VIEW_MARGIN_FAMILY)
    labels = [t.get_text() for t in w._canvas.fig.axes[0].get_yticklabels()]
    assert labels == ["$f_{0,0}$", "$f_{0,1}$", "$f_{1,0}$", "$f_{1,1}$"]


def test_results_export_names_the_pair_blocks(qapp):
    from pmcprg.pmc.gui.main_window import PMCMainWindow, _trace_rows
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(PAIR_TOML)
    _, Y = simulate(mdl, N=80, seed=0)
    result = _stand_in_estimation(mdl, Y)
    header, rows = _trace_rows(result.trace)
    # Columns follow the first snapshot's blocks, named by their pair.
    assert {"margin_0_0_loc", "margin_0_1_scale", "margin_1_1_scale"} <= set(header)
    assert not any(h.startswith("margin1_") for h in header)
    assert len(rows) == 3

    w = PMCMainWindow(PAIR_TOML)
    w._on_est_done(result)
    assert w._results_payload()["model"]["margin_structure"] == "pair"


def test_estimate_on_a_pair_model(qapp):
    """A real ICE run through the GUI entry point — skipped until wave 2A."""
    from pmcprg.pmc.gui.main_window import PMCMainWindow
    from pmcprg.pmc.model import PMCModel
    from pmcprg.pmc.simulate import simulate

    mdl = PMCModel(PAIR_TOML)
    _, Y = simulate(mdl, N=200, seed=0)
    w = PMCMainWindow(PAIR_TOML)
    cfg = w._tab_ice.get_cfg()
    cfg.update(max_iter=2, algorithm="ice")
    try:
        result = PMCMainWindow._do_estimate(mdl, Y, cfg)
    except RuntimeError as exc:
        # The failure names the model structure (clear message, no bare
        # "margin(0) without j" from deep inside the estimator).
        assert "pair-indexed margins f_ij" in str(exc)
        pytest.skip(f"ICE on pair margins pending wave 2A (ice.py/sem.py): "
                    f"{exc.__cause__!r}")
    assert result.fitted_model.margin_structure == "pair"
    w._on_est_done(result)
    assert w._tab_margins.structure == "pair"
