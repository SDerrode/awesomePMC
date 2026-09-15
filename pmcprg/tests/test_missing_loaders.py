"""CSV loaders with missing observations — CLI ``_read_observations`` and GUI
``_read_data_csv`` — plus the gap shading of the classification view.

Missing cells (empty, NaN/nan, NA) load as NaN with a warning; non-numeric
text, ±inf (GUI) and all-missing columns are still refused; complete files
load exactly as before.
"""

from __future__ import annotations

import csv
import logging

import numpy as np
import pytest

from pmcprg.exceptions import CSVLoadError
from pmcprg.missing.cells import parse_cell
from pmcprg.pmc.cli import _read_observations


def _write(path, header, rows):
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    return path


def test_parse_cell_tokens():
    for tok in ("", "  ", "NaN", "nan", "NAN", "NA", "na", " NA "):
        assert np.isnan(parse_cell(tok)), tok
    assert parse_cell(" 1.5 ") == 1.5
    assert parse_cell("-2e3") == -2000.0
    with pytest.raises(ValueError):
        parse_cell("abc")
    with pytest.raises(TypeError):
        parse_cell(None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_reads_missing_cells_as_nan(tmp_path, caplog):
    ys = ["0.5", "", "NaN", "1.25", "nan", "NA", "-3"]
    p = _write(tmp_path / "gaps.csv", ["n", "X", "Y"],
               [[i + 1, i % 2, y] for i, y in enumerate(ys)])
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.cli"):
        Y, X = _read_observations(str(p), "X")
    np.testing.assert_array_equal(np.isnan(Y), [0, 1, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(Y[[0, 3, 6]], [0.5, 1.25, -3.0])
    np.testing.assert_array_equal(X, [0, 1, 0, 1, 0, 1, 0])
    assert "4 of 7 Y values missing (57.1 %)" in caplog.text


def test_cli_complete_file_is_unchanged(tmp_path, caplog):
    rows = [[i + 1, i % 3, f"{0.1 * i - 2:.6f}"] for i in range(50)]
    p = _write(tmp_path / "full.csv", ["n", "X", "Y"], rows)
    with caplog.at_level(logging.WARNING):
        Y, X = _read_observations(str(p), "X")
    np.testing.assert_array_equal(Y, np.array([float(r[2]) for r in rows]))
    assert Y.dtype == np.float64 and X.dtype == np.array([1]).dtype
    assert "missing" not in caplog.text


def test_cli_refuses_garbage_with_the_row(tmp_path):
    p = _write(tmp_path / "bad.csv", ["n", "Y"], [[1, "1.0"], [2, "oops"], [3, "2"]])
    with pytest.raises(CSVLoadError, match=r"non-numeric Y at row 3 \(CSV line 3\).*'oops'"):
        _read_observations(str(p))


def test_cli_refuses_an_all_missing_column(tmp_path):
    p = _write(tmp_path / "void.csv", ["n", "Y"], [[1, ""], [2, "NA"], [3, "nan"]])
    with pytest.raises(CSVLoadError, match=r"every Y value is missing \(3 of 3"):
        _read_observations(str(p))


def test_cli_classify_reports_a_bad_cell_cleanly(tmp_path):
    """The command prints ERROR and exits 1 — no traceback."""
    import argparse

    from pmcprg.pmc.cli import cmd_classify

    p = _write(tmp_path / "bad.csv", ["n", "Y"], [[1, "1.0"], [2, "x"]])
    args = argparse.Namespace(model="pmcprg/pmc/models/pmc_gauss_k2.toml",
                              data=str(p), ref=None, out=str(tmp_path / "o.csv"))
    assert cmd_classify(args) == 1


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


@pytest.fixture()
def read_data_csv():
    pytest.importorskip("PyQt6")
    from pmcprg.pmc.gui.main_window import _read_data_csv
    return _read_data_csv


def test_gui_y_only_with_gaps(tmp_path, caplog, read_data_csv):
    p = _write(tmp_path / "y.csv", ["Y"], [["1.0"], [""], ["NA"], ["4.0"]])
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gui.main_window"):
        Y, X = read_data_csv(p)
    assert Y.shape == (4,) and X is None
    np.testing.assert_array_equal(np.isnan(Y), [0, 1, 1, 0])
    assert "2 of 4 rows" in caplog.text and "50.0 %" in caplog.text


def test_gui_y_and_x_with_gaps(tmp_path, read_data_csv):
    p = _write(tmp_path / "xy.csv", ["n", "X", "Y"],
               [[1, 0, "0.1"], [2, 1, "nan"], [3, 1, "0.3"]])
    Y, X = read_data_csv(p)
    np.testing.assert_array_equal(X, [0, 1, 1])
    assert np.isnan(Y[1]) and Y[2] == 0.3


def test_gui_multivariate_rows_keep_their_observed_components(tmp_path, caplog,
                                                              read_data_csv):
    p = _write(tmp_path / "mv.csv", ["n", "Y0", "Y1", "Y2"],
               [[1, "1", "2", "3"], [2, "", "5", "6"], [3, "NA", "NaN", ""],
                [4, "7", "8", "9"]])
    with caplog.at_level(logging.WARNING, logger="pmcprg.pmc.gui.main_window"):
        Y, _ = read_data_csv(p)
    assert Y.shape == (4, 3)
    np.testing.assert_array_equal(Y[1], [np.nan, 5.0, 6.0])
    assert np.isnan(Y[2]).all()
    np.testing.assert_array_equal(Y[[0, 3]], [[1, 2, 3], [7, 8, 9]])
    assert "2 of 4 rows" in caplog.text and "4 missing cell(s)" in caplog.text


def test_gui_refusals(tmp_path, read_data_csv):
    p = _write(tmp_path / "g.csv", ["Y"], [["1"], ["two"]])
    with pytest.raises(ValueError, match=r"Non-numeric Y at row 3 \(CSV line 3\): 'two'"):
        read_data_csv(p)
    p = _write(tmp_path / "inf.csv", ["Y"], [["1"], [""], ["-inf"]])
    with pytest.raises(ValueError, match="Non-finite Y at row 4"):
        read_data_csv(p)
    p = _write(tmp_path / "col.csv", ["Y0", "Y1"], [["1", ""], ["2", "NA"]])
    with pytest.raises(ValueError, match="Every Y1 value is missing"):
        read_data_csv(p)


def test_gui_complete_file_is_unchanged(tmp_path, caplog, read_data_csv):
    rows = [[i + 1, f"{np.sin(i):.5f}", f"{np.cos(i):.5f}"] for i in range(30)]
    p = _write(tmp_path / "full.csv", ["n", "Y0", "Y1"], rows)
    with caplog.at_level(logging.WARNING):
        Y, X = read_data_csv(p)
    np.testing.assert_array_equal(
        Y, np.column_stack([[float(r[1]) for r in rows], [float(r[2]) for r in rows]]))
    assert X is None and "missing" not in caplog.text


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app                                          # keep a reference alive


def test_gui_load_names_the_missing_count(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog

    from pmcprg.pmc.gui.main_window import PMCMainWindow

    w = PMCMainWindow("pmcprg/pmc/models/pmc_gauss_k2.toml")
    p = _write(tmp_path / "gaps.csv", ["n", "Y"],
               [[i + 1, "" if i in (3, 4, 9) else f"{0.1 * i}"] for i in range(20)])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(p), "")))
    w._on_load_data()
    assert w._status.currentMessage().endswith("N=20  missing=3 (15.0 %)")
    assert "missing=3 (15.0 %)" in w._log.toPlainText()

    q = _write(tmp_path / "full.csv", ["n", "Y"], [[i + 1, f"{0.1 * i}"] for i in range(20)])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(q), "")))
    w._on_load_data()
    assert w._status.currentMessage() == f"Data loaded: {q}  N=20"


@pytest.mark.parametrize("d,with_x", [(1, False), (1, True), (3, True)])
def test_gui_save_then_load_round_trips_a_masked_sequence(qapp, tmp_path,
                                                          monkeypatch, d, with_x):
    """A pattern's output survives File → Save data → Load data: NaN are
    written as 'nan' and read back as missing, at the same positions."""
    from PyQt6.QtWidgets import QFileDialog

    from pmcprg.missing import mcar
    from pmcprg.pmc.gui.main_window import PMCMainWindow, _read_data_csv

    rng = np.random.default_rng(d)
    Y = rng.normal(size=(80, d)) if d > 1 else rng.normal(size=80)
    X = rng.integers(0, 2, size=80) if with_x else None
    Ym, mask = mcar(Y, 0.25, block_size=4, seed=0)

    w = PMCMainWindow("pmcprg/pmc/models/pmc_gauss_k2.toml")
    w._last_Y, w._last_X = Ym, X
    out = tmp_path / "masked.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    w._on_save_data()

    Y_back, X_back = _read_data_csv(out)
    assert Y_back.shape == Y.shape
    np.testing.assert_array_equal(np.isnan(Y_back), mask)
    np.testing.assert_array_equal(Y_back[~mask], Y[~mask])
    if with_x:
        np.testing.assert_array_equal(X_back, X)
    else:
        assert X_back is None
    Y_cli, _ = _read_observations(str(out)) if d == 1 else (Y_back, None)
    np.testing.assert_array_equal(np.isnan(Y_cli), mask)


# ---------------------------------------------------------------------------
# Classification view — gaps drawn
# ---------------------------------------------------------------------------


def test_classification_view_shades_the_gaps():
    pytest.importorskip("PyQt6")
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure

    from pmcprg.pmc import PMCModel, classify, simulate
    from pmcprg.pmc.gui import views

    mdl = PMCModel("pmcprg/pmc/models/pmc_gauss_k2.toml")
    X, Y = simulate(mdl, N=120, seed=1)
    X_hat, gamma, _ = classify(mdl, Y)                 # complete data

    fig = Figure()
    views.plot_classification(fig, Y, X_hat, gamma, X, mdl)
    ax1 = fig.axes[0]
    assert "missing" not in ax1.get_title()
    assert not any(isinstance(c, PolyCollection) and "missing" in str(c.get_label())
                   for c in ax1.collections)

    Yg = Y.copy()
    Yg[[10, 11, 12, 50, 90, 91]] = np.nan             # three runs, six rows
    fig = Figure()
    views.plot_classification(fig, Yg + 100.0, X_hat, gamma, X, mdl)
    ax1 = fig.axes[0]
    assert "6 missing (5.0%)" in ax1.get_title()
    (gaps,) = [c for c in ax1.collections if str(c.get_label()).startswith("missing")]
    assert len(gaps.get_paths()) == 3
    # Bands span the axes height without dragging the y-limits towards [0, 1].
    assert ax1.get_ylim()[0] > 50
    assert views._draw_gaps(ax1, np.column_stack([Yg, Y])) == 6   # any component
