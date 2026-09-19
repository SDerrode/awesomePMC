"""A library must not change the global matplotlib settings of its caller.

Before 1.2.0, ``import pmcprg.copulas`` (hence ``import pmcprg.pmc`` and
``import pmcprg.diagnostics``, which import it) ran ``plot_style.apply_style()``
and rewrote ten ``rcParams`` — dpi, facecolor, every font size — for every
figure the user drew afterwards. The package look is now opt-in: its own plot
methods apply it inside :func:`pmcprg.plot_style.style_context`, and the
applications (GUI) restyle the session explicitly.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from pmcprg.copulas import BivariateLaw, CopulaClayton, CopulaGaussian
from pmcprg.plot_style import (
    DEFAULT_DPI,
    DEFAULT_FONT_SIZE,
    apply_style,
    style_context,
    with_package_style,
)

matplotlib.use("Agg")


def _run(script: str) -> str:
    """Run ``script`` in a fresh interpreter and return its stdout — with the
    child's stderr in the failure message (``check=True`` hides it, which is
    how a Linux-only failure once had to be diagnosed blind from CI)."""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, (
        f"the child interpreter exited with {done.returncode}\n"
        f"--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}")
    return done.stdout


def _snapshot() -> dict:
    """``rcParams`` without the backend, whose lazy resolution is not a setting."""
    return {k: v for k, v in matplotlib.rcParams.items() if k != "backend"}


# ---------------------------------------------------------------------------
# Importing the package
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [
    "pmcprg", "pmcprg.copulas", "pmcprg.diagnostics", "pmcprg.pmc", "pmcprg.missing",
])
def test_importing_the_package_leaves_rcparams_alone(module):
    """In a fresh interpreter: this process has imported everything already."""
    script = textwrap.dedent(f"""
        import importlib, warnings
        warnings.filterwarnings("ignore")
        import matplotlib
        snap = lambda: {{k: v for k, v in matplotlib.rcParams.items() if k != "backend"}}
        before = snap()
        importlib.import_module({module!r})
        after = snap()
        print(sorted(k for k in before if before[k] != after[k]))
    """)
    out = _run(script).strip().splitlines()[-1]
    assert out == "[]", f"import {module} changed rcParams: {out}"


# ---------------------------------------------------------------------------
# The package's own plots keep their look and leave no trace
# ---------------------------------------------------------------------------

def _dpi_of(png) -> float:
    return float(Image.open(png).info["dpi"][0])


def test_a_plot_method_draws_in_the_package_look_without_leaking_it(tmp_path):
    users = {"figure.dpi": 77, "savefig.dpi": 77, "font.size": 9}
    with matplotlib.rc_context(users):
        before = _snapshot()
        CopulaClayton(tau_k=0.5).plot_pdf(str(tmp_path), "t_")
        after = _snapshot()
    assert after == before, "the plot method left the package style in rcParams"

    (png,) = list(tmp_path.glob("t_PDF_*.png"))
    assert _dpi_of(png) == pytest.approx(DEFAULT_DPI, abs=1), (
        "the file was not saved at the package dpi: the style is not applied "
        "inside the plot method")


@pytest.mark.parametrize("draw", [
    lambda d: CopulaClayton(tau_k=0.5).plot_cdf(d),
    lambda d: CopulaClayton(tau_k=0.5).plot_h_function(d),
    lambda d: CopulaClayton(tau_k=0.5).plot_samples(d, n=200),
    lambda d: CopulaClayton(tau_k=0.5).plot_overview(d),
    lambda d: CopulaClayton(tau_k=0.5).plot_multi_tau(d, tau_values=[0.2, 0.6]),
    lambda d: BivariateLaw(copula=CopulaGaussian(tau_k=0.4), left_margin=_norm(),
                           right_margin=_norm()).plot_pdf(d),
    lambda d: BivariateLaw(copula=CopulaGaussian(tau_k=0.4), left_margin=_norm(),
                           right_margin=_norm()).plot_pdf_with_margins(d),
    lambda d: CopulaClayton.fit(CopulaClayton(tau_k=0.5).sample(n=200, seed=0),
                                method="mle").plot_diagnostics(d),
], ids=["cdf", "h", "samples", "overview", "multi_tau", "law_pdf", "law_margins", "diagnostics"])
def test_every_plot_method_is_styled_and_side_effect_free(tmp_path, draw):
    with matplotlib.rc_context({"figure.dpi": 77, "savefig.dpi": 77}):
        before = _snapshot()
        draw(str(tmp_path))
        assert _snapshot() == before
    pngs = list(tmp_path.glob("*.png"))
    assert pngs, "the plot method wrote nothing"
    assert all(_dpi_of(p) == pytest.approx(DEFAULT_DPI, abs=1) for p in pngs)
    plt.close("all")


def _norm():
    from scipy import stats
    return (stats.norm, 0.0, 1.0)


# ---------------------------------------------------------------------------
# The helpers themselves
# ---------------------------------------------------------------------------

def test_style_context_scopes_the_look_and_restores_on_error():
    before = _snapshot()
    with style_context(dpi=200, font_size=15):
        assert matplotlib.rcParams["figure.dpi"] == 200
        assert matplotlib.rcParams["font.size"] == 15
    assert _snapshot() == before

    with pytest.raises(RuntimeError):
        with style_context():
            raise RuntimeError("boom")
    assert _snapshot() == before


def test_plot_methods_do_not_depend_on_a_global_constrained_layout(tmp_path):
    """The GUI enables ``figure.constrained_layout.use`` globally; the package's
    methods use ``tight_layout``, which matplotlib refuses to mix with a
    colorbar under it. Before the fix ``plot_diagnostics`` raised in any
    process that had imported the GUI."""
    res = CopulaClayton.fit(CopulaClayton(tau_k=0.5).sample(n=200, seed=0), method="mle")
    with matplotlib.rc_context({"figure.constrained_layout.use": True}):
        before = _snapshot()
        res.plot_diagnostics(str(tmp_path))
        CopulaClayton(tau_k=0.5).plot_pdf(str(tmp_path))
        assert _snapshot() == before            # the caller's setting is back
        assert matplotlib.rcParams["figure.constrained_layout.use"] is True
    assert len(list(tmp_path.glob("*.png"))) == 2
    plt.close("all")


def test_apply_style_is_an_explicit_session_wide_opt_in():
    with matplotlib.rc_context():                       # restores on exit
        matplotlib.rcParams["figure.dpi"] = 60
        apply_style(dpi=123, font_size=11)
        assert matplotlib.rcParams["figure.dpi"] == 123
        assert matplotlib.rcParams["font.size"] == 11
        assert matplotlib.rcParams["savefig.facecolor"] == "white"


def test_with_package_style_preserves_the_function_identity():
    def draw(a, b=2):
        """docstring"""
        return matplotlib.rcParams["font.size"], a, b

    wrapped = with_package_style(draw)
    assert wrapped.__name__ == "draw" and wrapped.__doc__ == "docstring"
    assert inspect.signature(wrapped) == inspect.signature(draw)
    with matplotlib.rc_context({"font.size": 8}):
        assert wrapped(1) == (DEFAULT_FONT_SIZE, 1, 2)
        assert matplotlib.rcParams["font.size"] == 8


def test_decorated_plot_methods_keep_their_public_signature():
    sig = inspect.signature(CopulaClayton(tau_k=0.5).plot_samples)
    assert list(sig.parameters) == ["plot_dir", "prefix", "n", "seed"]


# ---------------------------------------------------------------------------
# Applications restyle on purpose: the GUI
# ---------------------------------------------------------------------------

def test_the_gui_restyles_the_session_explicitly():
    pytest.importorskip("PyQt6")
    script = textwrap.dedent("""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        import warnings; warnings.filterwarnings("ignore")
        import matplotlib
        # main_window switches matplotlib to the QtAgg backend, which matplotlib
        # refuses on a headless Linux (no DISPLAY) unless a QApplication is
        # already running — as it always is when the GUI starts. macOS never
        # reports "headless", which is how this passed locally and failed in CI.
        from PyQt6.QtWidgets import QApplication
        _app = QApplication([])
        import pmcprg.pmc.gui.main_window          # noqa: F401
        rc = matplotlib.rcParams
        # package look first (apply_style) ...
        print(rc["figure.dpi"], rc["savefig.facecolor"], rc["font.size"])
        # ... then the compact overrides win (apply_gui_compact_style)
        print(rc["mathtext.fontset"], rc["axes.titlesize"], rc["figure.constrained_layout.use"])
    """)
    out = _run(script).strip().splitlines()[-2:]
    dpi, face, size = out[0].split()
    assert (float(dpi), face, float(size)) == (DEFAULT_DPI, "white", DEFAULT_FONT_SIZE)
    fontset, title, constrained = out[1].split()
    assert (fontset, float(title), constrained) == ("cm", 10.0, "True")


def test_a_user_figure_after_importing_the_package_is_untouched():
    """The point of it all: the user's own plot after ``import pmcprg``."""
    script = textwrap.dedent("""
        import warnings; warnings.filterwarnings("ignore")
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pmcprg.copulas                       # noqa: F401
        fig = plt.figure()
        print(fig.dpi, matplotlib.rcParams["font.size"])
    """)
    dpi, size = _run(script).split()[-2:]
    assert float(dpi) == matplotlib.rcParamsDefault["figure.dpi"]
    assert float(size) == matplotlib.rcParamsDefault["font.size"]
    assert np.isfinite(float(dpi))
