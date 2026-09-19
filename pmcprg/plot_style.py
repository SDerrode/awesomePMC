"""
pmcprg.plot_style — package-wide matplotlib defaults, **opt-in**.

The package's plots share one look (white facecolor, 150 dpi, 12 pt text).
Importing ``pmcprg`` or ``pmcprg.copulas`` does **not** touch
``matplotlib.rcParams``: a library has no business changing the global
settings of the program that imports it — before 1.2.0 ``import
pmcprg.copulas`` did, silently restyling every figure the user drew
afterwards (their dpi, their font sizes, their facecolor).

Three ways to get the package look, from least to most invasive:

* the package's own plot methods (``plot_pdf``, ``plot_overview``,
  ``plot_diagnostics``…) apply it *inside* a :func:`style_context` and leave
  ``rcParams`` exactly as they found it — nothing to do;
* wrap your own plotting code: ``with style_context(): ...``;
* restyle the whole session on purpose: ``apply_style()`` (or with your own
  ``facecolor``/``dpi``/``font_size``) — the CLI, the GUI and the report
  scripts are applications and do this explicitly.
"""

import contextlib as _contextlib
import functools as _functools

import matplotlib as _mpl


# Default values — mirror the legacy ``pmcprg.settings.plot_settings`` constants.
DEFAULT_FACECOLOR = "white"
DEFAULT_DPI       = 150
DEFAULT_FONT_SIZE = 12


def _style_rc(
    facecolor: str = DEFAULT_FACECOLOR,
    dpi:       int = DEFAULT_DPI,
    font_size: int = DEFAULT_FONT_SIZE,
) -> dict:
    """The ``rcParams`` of the package look, as a plain dict."""
    return {
        # Background
        "figure.facecolor":  facecolor,
        "savefig.facecolor": facecolor,
        "axes.facecolor":    facecolor,
        # Resolution
        "figure.dpi":        dpi,
        "savefig.dpi":       dpi,
        # Typography — uniform default; explicit overrides handle "bigger" titles
        "font.size":         font_size,
        "axes.titlesize":    font_size,
        "axes.labelsize":    font_size,
        "xtick.labelsize":   font_size,
        "ytick.labelsize":   font_size,
        "legend.fontsize":   font_size,
        "figure.titlesize":  font_size,
    }


def apply_style(
    facecolor: str = DEFAULT_FACECOLOR,
    dpi:       int = DEFAULT_DPI,
    font_size: int = DEFAULT_FONT_SIZE,
) -> None:
    """Restyle the **whole session**: update ``matplotlib.rcParams`` globally.

    Explicit and opt-in — nothing in the package calls it on import. Use
    :func:`style_context` to scope the same settings to a block instead.

    Parameters
    ----------
    facecolor : str — figure & savefig facecolor (default 'white').
    dpi       : int — figure & savefig DPI (default 150).
    font_size : int — base font size used for body text, axes labels AND
                      titles (default 12). Code that wants larger emphasis
                      passes ``fontsize=DEFAULT_FONT_SIZE + 2`` explicitly.
    """
    _mpl.rcParams.update(_style_rc(facecolor, dpi, font_size))


@_contextlib.contextmanager
def style_context(
    facecolor: str = DEFAULT_FACECOLOR,
    dpi:       int = DEFAULT_DPI,
    font_size: int = DEFAULT_FONT_SIZE,
):
    """The package look for the duration of a ``with`` block only.

    ``matplotlib.rc_context`` under the hood: ``rcParams`` is restored on exit,
    normally or by an exception, so the caller's own settings are untouched.

    It also pins ``figure.constrained_layout.use`` **off**. The package's plot
    methods lay figures out with ``tight_layout``, which matplotlib refuses to
    combine with a colorbar once constrained layout is on — and the GUI turns
    it on globally (:func:`apply_gui_compact_style`) for its own figures, so in
    a process that had imported the GUI ``plot_diagnostics`` raised
    ``RuntimeError: Colorbar layout of new layout engine not compatible``.
    Inside the context a figure comes out the same whatever the caller's
    ``rcParams``.
    """
    rc = _style_rc(facecolor, dpi, font_size)
    rc["figure.constrained_layout.use"] = False
    with _mpl.rc_context(rc):
        yield


def with_package_style(func):
    """Decorator: run ``func`` inside :func:`style_context`.

    Applied to the package's plot methods so that they draw and save in the
    package look without leaving it behind in the caller's ``rcParams``.
    """
    @_functools.wraps(func)
    def wrapper(*args, **kwargs):
        with style_context():
            return func(*args, **kwargs)
    return wrapper


def apply_gui_compact_style() -> None:
    """Compact, math-friendly rcParams for the PyQt6 GUI canvas.

    Targeted at multi-panel figures (ICE dashboard, K×K small multiples,
    pseudo-observation grids) where the per-axes labels would clobber
    each other under the default sizes. Run *after* :func:`apply_style`
    so the GUI overrides win.

    Differences vs. :func:`apply_style`:

    * Math labels rendered with ``mathtext.fontset = "cm"`` so ``$\\tau$``,
      ``$Y_n$``, ``$\\hat{X}$`` etc. look LaTeX-like — no LaTeX install
      required (matplotlib ships the Computer-Modern fonts).
    * Smaller per-axes titles (10 pt), labels (9 pt), and ticks (8 pt).
    * Slightly tighter ``axes.titlepad`` / ``axes.labelpad`` for a
      denser layout.
    * Constrained-layout enabled by default so suptitles never collide
      with subplot titles.
    """
    _mpl.rcParams.update({
        # LaTeX-flavoured math via matplotlib's bundled Computer Modern.
        "mathtext.fontset":          "cm",
        # Compact typography for multi-panel figures.
        "axes.titlesize":            10,
        "axes.titlepad":             4.0,
        "axes.labelsize":            9,
        "axes.labelpad":             2.0,
        "xtick.labelsize":           8,
        "ytick.labelsize":           8,
        "legend.fontsize":           8,
        "legend.title_fontsize":     8,
        "figure.titlesize":          11,
        "figure.titleweight":        "regular",
        # Constrained layout handles suptitles better than tight_layout
        # for grids with many subplots (e.g. the ICE dashboard).
        "figure.constrained_layout.use":     True,
        "figure.constrained_layout.h_pad":   0.04,
        "figure.constrained_layout.w_pad":   0.04,
        "figure.constrained_layout.hspace":  0.04,
        "figure.constrained_layout.wspace":  0.04,
    })
